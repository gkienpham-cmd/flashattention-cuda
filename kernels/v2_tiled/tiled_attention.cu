// v2 tiled attention — the operand-reuse win, with the bandwidth wall STILL standing.
//
// v1's wall was redundant HBM traffic: qk_kernel re-read a full Q row and K row from global
// memory for EVERY one of the N^2 score entries, and pv_kernel re-read V the same way. That is
// O(N^2 * d) traffic with no reuse (AI ~0.2 FLOP/byte) — pinned to the 320 GB/s HBM roof.
//
// v2 fixes exactly that, and nothing else. Each block stages its Q/K (and later P/V) tiles into
// shared memory ONCE, then every thread in the block reuses them from on-chip smem. A Q row now
// hits HBM once per j-tile (N_k/TN times) instead of once per key (N_k times): a ~TN x cut. The
// roofline model predicts AI climbs 0.2 -> ~6-8 (a ~30x traffic cut) but DOES NOT cross the T4's
// FP32 ridge of 25.3 — so v2 is predicted to stay HBM-bound. The reason is deliberate: we KEEP
// the three-pass structure and KEEP materializing the full S matrix in HBM, so the S round-trip
// survives. Only online softmax (v3) removes it. This isolates the operand-reuse lesson from the
// S-elimination lesson — one variable per step.
//
// Precision: FP32 in / FP32 accumulate / FP32 out, identical to v1 (atol/rtol 1e-4 vs SDPA). v2
// changes ONLY the memory schedule, never the math, so it remains an exact-ish correctness anchor.
//
// Tile sizes adapt to the head dim d so the QK-pass smem (sQ + sK = (TM+TN)*d floats) fits the
// T4's 48 KB static-smem budget:
//   d=64  -> 64x64 tile  (sQ 64x64 + sK 64x64 = 32 KB)   -> predicted tiled AI 8.0
//   d=128 -> 32x32 tile  (sQ 32x128 + sK 32x128 = 32 KB) -> predicted tiled AI 6.4 (bigger d eats
//            the smem budget that buys reuse, so the tile -- and the reuse factor -- must shrink)
//
// Layout: q,k,v are [B, H, N, d] row-major (project convention). Causal masking, as in v1, is
// applied in the softmax pass (mask key j > query i); tiling does not touch it.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>   // at::cuda::getCurrentCUDAStream
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>   // FLT_MAX

namespace {

constexpr int kBlock = 256;  // threads per block for all three passes; a plain, occupancy-friendly size

// Round-up integer division, for sizing grids/chunks to cover `n` elements.
inline int ceil_div(int64_t n, int b) { return static_cast<int>((n + b - 1) / b); }

// ---------------------------------------------------------------------------------------
// Pass 1: S = scale * (Q @ K^T), tiled.
// One block computes a TM x TN output tile of S for one (batch,head). It loads the tile's TM
// query rows and TN key rows (each of length d) into shared memory ONCE, then all kBlock threads
// compute the TM*TN dot products reusing those rows from smem. d (<=128) is the contraction and
// fits whole in smem, so there is no inner K-loop here. THIS is the reuse v1 lacked: a Q row is
// fetched from HBM once per block (N_k/TN blocks) instead of once per key (N_k times).
// Template params are compile-time so the smem arrays and loop bounds size exactly to the tile.
// ---------------------------------------------------------------------------------------
template <int TM, int TN, int D>
__global__ void qk_tiled_kernel(const float* __restrict__ Q,
                                const float* __restrict__ K,
                                float* __restrict__ S,
                                int N_q, int N_k, float scale) {
    __shared__ float sQ[TM * D];   // this tile's query rows  [TM][D]
    __shared__ float sK[TN * D];   // this tile's key rows    [TN][D]

    const int i0 = blockIdx.y * TM;            // first query row of this tile
    const int j0 = blockIdx.x * TN;            // first key   row of this tile
    const int64_t bh = blockIdx.z;             // (batch*head) slice
    const int tid = threadIdx.x;

    // Cooperative load: Q rows [i0, i0+TM) -> sQ. Out-of-range rows (last partial tile) load 0;
    // they are masked out on store, so the padding never reaches S.
    const float* Qbh = Q + bh * (int64_t)N_q * D;
    for (int idx = tid; idx < TM * D; idx += kBlock) {
        int r = idx / D, t = idx % D, gi = i0 + r;
        sQ[idx] = (gi < N_q) ? Qbh[(int64_t)gi * D + t] : 0.f;
    }
    const float* Kbh = K + bh * (int64_t)N_k * D;
    for (int idx = tid; idx < TN * D; idx += kBlock) {
        int r = idx / D, t = idx % D, gj = j0 + r;
        sK[idx] = (gj < N_k) ? Kbh[(int64_t)gj * D + t] : 0.f;
    }
    __syncthreads();

    // Compute: grid-stride over the TM*TN output elements of this tile. Every dot product reads
    // its Q row and K row straight from smem — the bytes v1 re-fetched from HBM are now reused.
    float* Sbh = S + bh * (int64_t)N_q * N_k;
    for (int e = tid; e < TM * TN; e += kBlock) {
        int r = e / TN, c = e % TN, gi = i0 + r, gj = j0 + c;
        if (gi >= N_q || gj >= N_k) continue;          // skip padding of the last partial tile
        float acc = 0.f;
        for (int t = 0; t < D; ++t) acc += sQ[r * D + t] * sK[c * D + t];  // FP32 accumulate
        Sbh[(int64_t)gi * N_k + gj] = acc * scale;     // S still round-trips HBM (v3 removes this)
    }
}

// ---------------------------------------------------------------------------------------
// Pass 2: row-wise softmax over the key axis, in place in S (P overwrites S).
// Identical to v1: one thread per row, three sweeps (max / exp-sum / normalize), causal cutoff,
// masked tail zeroed. A row reduction has no operand reuse to win, so tiling leaves it untouched;
// the v2 win lives entirely in the two GEMM passes around it.
// ---------------------------------------------------------------------------------------
__global__ void softmax_kernel(float* __restrict__ S,
                               int N_q, int N_k, bool causal,
                               int64_t n_rows) {
    int64_t row = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    int i = row % N_q;                 // query position, for the causal cutoff
    float* s = S + row * (int64_t)N_k; // this row's N_k scores
    int j_max = causal ? min(i, N_k - 1) : (N_k - 1);

    float m = -FLT_MAX;
    for (int j = 0; j <= j_max; ++j) m = fmaxf(m, s[j]);

    float denom = 0.f;
    for (int j = 0; j <= j_max; ++j) {
        float e = __expf(s[j] - m);
        s[j] = e;
        denom += e;
    }

    float inv = 1.f / denom;
    for (int j = 0; j <= j_max; ++j) s[j] *= inv;
    for (int j = j_max + 1; j < N_k; ++j) s[j] = 0.f;  // masked tail ignored by pass 3
}

// ---------------------------------------------------------------------------------------
// Pass 3: O = P @ V, tiled.
// One block computes a TM x D output tile (TM query rows, all D head-dim columns) for one
// (batch,head). The contraction is over N_k keys (large), so it is tiled into TK-wide chunks:
// for each chunk, stage P[TM][TK] and V[TK][D] into smem, then accumulate into per-output
// registers carried across chunks. Like pass 1, V and P are fetched from HBM once per chunk and
// reused across the tile, instead of v1's per-output-element re-read of a full V column.
// ---------------------------------------------------------------------------------------
template <int TM, int TK, int D>
__global__ void pv_tiled_kernel(const float* __restrict__ P,   // = softmaxed S
                                const float* __restrict__ V,
                                float* __restrict__ O,
                                int N_q, int N_k) {
    __shared__ float sP[TM * TK];   // this tile's prob rows   [TM][TK]
    __shared__ float sV[TK * D];    // this chunk's V rows     [TK][D]

    const int i0 = blockIdx.x * TM;            // first query row of this tile
    const int64_t bh = blockIdx.y;             // (batch*head) slice
    const int tid = threadIdx.x;

    // Each thread owns a fixed, grid-strided set of the TM*D outputs and carries their partial
    // sums in registers across the chunk loop (N_ACC of them; both shipped configs give 16).
    constexpr int N_ACC = (TM * D + kBlock - 1) / kBlock;
    float acc[N_ACC];
    #pragma unroll
    for (int o = 0; o < N_ACC; ++o) acc[o] = 0.f;

    const float* Pbh = P + bh * (int64_t)N_q * N_k;
    const float* Vbh = V + bh * (int64_t)N_k * D;

    for (int j0 = 0; j0 < N_k; j0 += TK) {
        // Stage P[i0.., j0..] -> sP and V[j0.., :] -> sV. Out-of-range entries load 0 so they
        // contribute nothing to the dot product (padding-safe for partial tiles/chunks).
        for (int idx = tid; idx < TM * TK; idx += kBlock) {
            int r = idx / TK, c = idx % TK, gi = i0 + r, gj = j0 + c;
            sP[idx] = (gi < N_q && gj < N_k) ? Pbh[(int64_t)gi * N_k + gj] : 0.f;
        }
        for (int idx = tid; idx < TK * D; idx += kBlock) {
            int r = idx / D, t = idx % D, gj = j0 + r;
            sV[idx] = (gj < N_k) ? Vbh[(int64_t)gj * D + t] : 0.f;
        }
        __syncthreads();

        // Accumulate this chunk's contribution to each owned output (row r, head-dim col t).
        for (int o = 0, e = tid; e < TM * D; ++o, e += kBlock) {
            int r = e / D, t = e % D;
            float a = acc[o];
            for (int c = 0; c < TK; ++c) a += sP[r * TK + c] * sV[c * D + t];
            acc[o] = a;
        }
        __syncthreads();   // protect sP/sV before the next chunk overwrites them
    }

    // Write the completed output tile. Mask the partial-tile row overhang.
    float* Obh = O + bh * (int64_t)N_q * D;
    for (int o = 0, e = tid; e < TM * D; ++o, e += kBlock) {
        int r = e / D, t = e % D, gi = i0 + r;
        if (gi < N_q) Obh[(int64_t)gi * D + t] = acc[o];
    }
}

// Launch the tiled QK and PV passes for a fixed compile-time tile config (chosen by head dim).
template <int TM, int TN, int D>
void launch_passes(const float* q, const float* k, const float* p_v, float* S, float* O,
                   int B, int H, int N_q, int N_k, float scale, bool causal,
                   cudaStream_t stream) {
    const int64_t BH = (int64_t)B * H;

    // Pass 1: QK^T over a 2D grid of TM x TN output tiles, one z-slice per (batch,head).
    dim3 g1(ceil_div(N_k, TN), ceil_div(N_q, TM), (unsigned)BH);
    qk_tiled_kernel<TM, TN, D><<<g1, kBlock, 0, stream>>>(q, k, S, N_q, N_k, scale);

    // Pass 2: softmax, one thread per row (TK is the contraction tile, reused for PV chunks).
    const int64_t n_rows = BH * N_q;
    softmax_kernel<<<ceil_div(n_rows, kBlock), kBlock, 0, stream>>>(S, N_q, N_k, causal, n_rows);

    // Pass 3: P@V over TM-row tiles (full D columns), contraction chunked by TK (= TN).
    dim3 g3(ceil_div(N_q, TM), (unsigned)BH);
    pv_tiled_kernel<TM, TN, D><<<g3, kBlock, 0, stream>>>(S, p_v, O, N_q, N_k);
}

}  // anonymous namespace

// Host entry point: orchestrates the three tiled passes. Returns O = attention(Q, K, V).
// Exposed to Python by binding.cpp as `forward`. Mirrors v1's contract exactly.
torch::Tensor tiled_attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                      double scale, bool causal) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q,k,v must be CUDA tensors");
    TORCH_CHECK(q.dtype() == torch::kFloat32, "v2 tiled is FP32-only (the clean baseline)");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "expected [B,H,N,d] tensors");
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous();

    const int B = q.size(0), H = q.size(1), N_q = q.size(2), d = q.size(3);
    const int N_k = k.size(2);
    TORCH_CHECK(k.size(3) == d && v.size(3) == d, "head_dim mismatch across q,k,v");
    TORCH_CHECK(v.size(2) == N_k, "K and V must share sequence length");

    auto S = torch::empty({B, H, N_q, N_k}, q.options());  // still materialized in HBM (v3 removes)
    auto O = torch::empty_like(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Tile config picked so (TM+TN)*d floats of smem fit the T4's 48 KB static budget; these are
    // exactly the tiles the Step 2 roofline table predicts (AI 8.0 at d=64, 6.4 at d=128).
    if (d == 64) {
        launch_passes<64, 64, 64>(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                                  S.data_ptr<float>(), O.data_ptr<float>(),
                                  B, H, N_q, N_k, (float)scale, causal, stream);
    } else if (d == 128) {
        launch_passes<32, 32, 128>(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                                   S.data_ptr<float>(), O.data_ptr<float>(),
                                   B, H, N_q, N_k, (float)scale, causal, stream);
    } else {
        TORCH_CHECK(false, "v2 tiled supports head_dim 64 or 128 (got ", d,
                    "); the tile config is specialized per head dim to fit the 48 KB smem budget");
    }

    return O;
}

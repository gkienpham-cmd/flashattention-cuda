// v4 fused FlashAttention-1 — the schedule fix. Keep v3's S-elimination, restore v2's parallelism.
//
// Step 3 (v3) proved the ALGORITHMIC win: online softmax keeps the N_q x N_k score matrix S off
// HBM entirely (125x less peak memory than v2). But v3 ran 3-7x SLOWER than v2. The Step 2 ncu
// read had already shown nothing was bandwidth-bound on the T4 (the 4 MB L2 owns Q/K/V at every
// N), so deleting ~99% of DRAM traffic bought ~0 wall-clock. v3's measured limiter was instead
// OCCUPANCY/LATENCY: it maps ONE THREAD to a query row, so 31/32 lanes of every warp sit idle, and
// each row-thread reads K/V from global one row at a time (uncoalesced, unstaged). Measured 151x
// above the 17 ms MMA floor at 8192x64, pass2-dominated.
//
// v4 attacks exactly that schedule while keeping S on-chip:
//   - ONE WARP PER QUERY ROW (32 lanes cooperate on that row), not one thread. The block's 8 warps
//     process 8 query rows at once -> the lanes that v3 wasted are now busy.
//   - STAGED K/V: each key-block is loaded into shared memory ONCE by the whole block and reused by
//     all 8 warps (coalesced global reads, smem broadcast), vs v3's per-row global re-reads.
//   - REGISTER-RESIDENT O: the output row lives in registers, sliced D/32 elements per lane. Never
//     spills to smem (v3 used smem sO) or HBM.
//   - SINGLE PASS with the O-RESCALE: v3 went two-pass precisely to DODGE the O-rescale (pass 1 got
//     final (m,l), so pass 2's p was already normalized and O needed no correction). v4 fuses to
//     one pass, which means O is accumulated while the running max is still moving, so every time
//     the max climbs we must retroactively rescale the partial O by exp(m_old - m_new) BEFORE
//     adding the new contribution. That O-rescale is the whole point of this step.
//
// THE ONLINE UPDATE, per key (running max m, running denom l, running unnormalized output O):
//   s     = scale * dot(q_row, k_j)                 // warp-reduced across the 32 lanes
//   m_new = max(m, s)
//   alpha = exp(m_old - m_new)                       // <= 1, the rescale factor (never overflows)
//   p     = exp(s - m_new)
//   l     = alpha*l + p
//   O     = alpha*O + p * v_j                        // O-rescale BEFORE the add (the fiddly piece)
//   m     = m_new
// After the whole key axis: O *= 1/l (normalize once), then store. Empty-accumulator init is
// m=-FLT_MAX, l=0, O=0: the first score gives alpha=exp(-FLT_MAX - s)=0 (wipes the empty state),
// p=exp(0)=1, so l=1 and O=v_j — same trick v3 uses on the scalar l.
//
// Precision: FP32 in / FP32 accumulate / FP32 out, identical to v1/v2/v3 (atol/rtol 1e-4 vs SDPA).
// Tensor cores are Step 5. Layout: q,k,v are [B,H,N,d] row-major. Causal excludes keys j > query i.
//
// Smem budget: only the staged K/V tiles (no sQ/sO/S) = 2*TN*D floats:
//   d=64  -> TN=64  (sK 64x64 + sV 64x64 = 32 KB)
//   d=128 -> TN=32  (sK 32x128 + sV 32x128 = 32 KB)
// well under the T4's 48 KB. Q and O live in registers (D/32 per lane).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>   // at::cuda::getCurrentCUDAStream
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>   // FLT_MAX

namespace {

constexpr int kBlock = 256;        // threads per block
constexpr int kWarp  = 32;         // lanes per warp
constexpr int kWarps = kBlock / kWarp;  // 8 warps per block -> 8 query rows per block

// Round-up integer division, for sizing grids to cover `n` elements.
inline int ceil_div(int64_t n, int b) { return static_cast<int>((n + b - 1) / b); }

// Butterfly all-reduce: every lane ends up with the full 32-lane sum (so all lanes can use s).
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = kWarp / 2; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

// ---------------------------------------------------------------------------------------
// Fused single-pass kernel. One block owns kWarps consecutive query rows of one (batch,head);
// warp `w` owns query row i0+w and carries that row's running (m, l) and register-resident O
// across the WHOLE key axis. The block streams the key axis in TN-wide blocks, staging each K/V
// block into smem ONCE (all warps reuse it). Each warp computes its row's score for every staged
// key via a 32-lane dot product, folds it into (m, l, O) with the online-softmax O-rescale, and
// finally normalizes and stores O. S is never formed in HBM; O never leaves registers until store.
// ---------------------------------------------------------------------------------------
template <int TN, int D>
__global__ void fused_attention_kernel(const float* __restrict__ Q,
                                       const float* __restrict__ K,
                                       const float* __restrict__ V,
                                       float* __restrict__ O,
                                       int N_q, int N_k, float scale, bool causal) {
    static_assert(D % kWarp == 0, "head_dim must be a multiple of the warp size (32)");
    constexpr int EPT = D / kWarp;   // elements of the row each lane owns (d=64->2, d=128->4)

    __shared__ float sK[TN * D];     // current key-block  [TN][D], reloaded per block
    __shared__ float sV[TN * D];     // current value-block[TN][D], reloaded per block

    const int warp = threadIdx.x >> 5;          // 0..kWarps-1 : which query row in this tile
    const int lane = threadIdx.x & (kWarp - 1); // 0..31       : which slice of the row
    const int i0 = blockIdx.x * kWarps;         // first query row of this tile
    const int gi = i0 + warp;                   // global query row this warp owns
    const int64_t bh = blockIdx.y;              // (batch*head) slice
    const bool active = (gi < N_q);             // last tile may have idle warps

    const float* Qbh = Q + bh * (int64_t)N_q * D;
    const float* Kbh = K + bh * (int64_t)N_k * D;
    const float* Vbh = V + bh * (int64_t)N_k * D;

    // Load this warp's query-row slice into registers; lane owns elements {lane, lane+32, ...}.
    // O accumulator starts at 0 (unnormalized). Both are register-resident for the whole kernel.
    float q_reg[EPT], o_reg[EPT];
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        int t = lane + kWarp * e;
        q_reg[e] = active ? Qbh[(int64_t)gi * D + t] : 0.f;
        o_reg[e] = 0.f;
    }

    // Running online-softmax stats for this row. Empty state wiped by the first score (see header).
    float m_cur = -FLT_MAX, l_cur = 0.f;

    for (int j0 = 0; j0 < N_k; j0 += TN) {
        // Cooperative load of this key/value block by the WHOLE block (all warps participate so the
        // __syncthreads below is reached uniformly, even by idle warps). Coalesced in t across lanes.
        for (int idx = threadIdx.x; idx < TN * D; idx += kBlock) {
            int r = idx / D, t = idx % D, gj = j0 + r;
            bool ok = gj < N_k;
            sK[idx] = ok ? Kbh[(int64_t)gj * D + t] : 0.f;
            sV[idx] = ok ? Vbh[(int64_t)gj * D + t] : 0.f;
        }
        __syncthreads();   // sK, sV visible to every warp before any compute

        if (active) {
            for (int c = 0; c < TN; ++c) {
                int gj = j0 + c;
                if (gj >= N_k) break;            // padding tail of the last key-block
                if (causal && gj > gi) break;    // keys ascend, so all further c are masked too

                // 32-lane dot product q_row . k_c: each lane sums its EPT owned products, then the
                // butterfly reduce broadcasts the full score s to all lanes (warp-uniform).
                float partial = 0.f;
#pragma unroll
                for (int e = 0; e < EPT; ++e)
                    partial += q_reg[e] * sK[c * D + lane + kWarp * e];
                float s = warp_reduce_sum(partial) * scale;

                // Online update with the O-rescale: shift the baseline to s when it is a new max,
                // correcting BOTH the running l and the partial O by exp(m_old - m_new) in (0,1].
                float m_new = fmaxf(m_cur, s);
                float alpha = __expf(m_cur - m_new);
                float p     = __expf(s - m_new);
                l_cur = l_cur * alpha + p;
#pragma unroll
                for (int e = 0; e < EPT; ++e)
                    o_reg[e] = o_reg[e] * alpha + p * sV[c * D + lane + kWarp * e];
                m_cur = m_new;
            }
        }
        __syncthreads();   // protect sK/sV before the next block overwrites them
    }

    // Normalize once (divide the unnormalized O by the final denom l) and store, coalesced in t.
    if (active) {
        float inv = 1.f / l_cur;
        float* Obh = O + bh * (int64_t)N_q * D;
#pragma unroll
        for (int e = 0; e < EPT; ++e) {
            int t = lane + kWarp * e;
            Obh[(int64_t)gi * D + t] = o_reg[e] * inv;
        }
    }
}

// Launch the fused kernel for a fixed compile-time tile config (chosen by head dim).
template <int TN, int D>
void launch_fused(const float* q, const float* k, const float* v, float* O,
                  int B, int H, int N_q, int N_k, float scale, bool causal,
                  cudaStream_t stream) {
    const int64_t BH = (int64_t)B * H;
    dim3 grid(ceil_div(N_q, kWarps), (unsigned)BH);   // one block per (query-row-tile, batch-head)
    fused_attention_kernel<TN, D><<<grid, kBlock, 0, stream>>>(q, k, v, O, N_q, N_k, scale, causal);
}

}  // anonymous namespace

// Host entry point: a single fused pass with online softmax + O-rescale. S never materialized, and
// (unlike v3) there is NO HBM scratch at all — not even per-row (m, l) — since the fused loop keeps
// them in registers. Exposed to Python by binding.cpp as `forward`. Same contract as v1/v2/v3.
torch::Tensor fused_attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                      double scale, bool causal) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q,k,v must be CUDA tensors");
    TORCH_CHECK(q.dtype() == torch::kFloat32, "v4 fused is FP32-only (tensor cores are Step 5)");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "expected [B,H,N,d] tensors");
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous();

    const int B = q.size(0), H = q.size(1), N_q = q.size(2), d = q.size(3);
    const int N_k = k.size(2);
    TORCH_CHECK(k.size(3) == d && v.size(3) == d, "head_dim mismatch across q,k,v");
    TORCH_CHECK(v.size(2) == N_k, "K and V must share sequence length");

    auto O = torch::empty_like(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Tile config picked so the staged K/V tiles (2*TN*D floats) sit well under the 48 KB smem
    // budget; mirrors v2/v3's per-head-dim choice (64x64 @ d=64, 32x32 @ d=128).
    if (d == 64) {
        launch_fused<64, 64>(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                             O.data_ptr<float>(), B, H, N_q, N_k, (float)scale, causal, stream);
    } else if (d == 128) {
        launch_fused<32, 128>(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                              O.data_ptr<float>(), B, H, N_q, N_k, (float)scale, causal, stream);
    } else {
        TORCH_CHECK(false, "v4 fused supports head_dim 64 or 128 (got ", d,
                    "); the tile config is specialized per head dim to fit the 48 KB smem budget");
    }

    return O;
}

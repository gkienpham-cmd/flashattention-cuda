// v3 online-softmax attention — the S-elimination win. S never touches HBM.
//
// The Step 2 ncu read settled the bottleneck story: the N_q x N_k score matrix S was ~99% of
// measured DRAM traffic (softmax re-reads it ~12x), while v2's tiling cut ~0% DRAM because the
// T4's 4 MB L2 already owns the Q/K/V operands at every N. So the one big DRAM consumer left is
// S itself. v3 deletes it.
//
// HOW: online (streaming) softmax. Softmax needs two full-row reductions — the max m and the sum
// l = Σ exp(s - m). The naive way materializes all of S to get them. The streaming way keeps a
// running (m, l) per query row and folds in one key-block at a time, so each score is computed
// on-chip, consumed, and discarded — never written out. The trick is the RUNNING-MAX RESCALE:
// when a new score s pushes the running max up to m_new, every already-accumulated term in l was
// measured against the old baseline, so it picks up the exact factor exp(m_old - m_new) (the
// exp(a+b)=exp(a)exp(b) identity — algebra, not approximation), and that factor is in (0,1] so it
// can never overflow.
//
// TWO PASSES (the deliberate learning-stepping-stone, simpler than canonical single-pass FA):
//   Pass 1 (stats): stream K-blocks per query row -> final (m, l). The rescale lives on the
//                   scalar l only; no output accumulator is involved.
//   Pass 2 (output): stream K/V again, recompute each score on-chip, form p = exp(s - m)/l with
//                   the now-FINAL m, l, accumulate O += p·V. Because m, l are final, O needs NO
//                   rescaling — every p is already correctly normalized. (Single-pass FA fuses
//                   these and pays for it by ALSO rescaling the running O accumulator on every max
//                   update; that O-rescale is deferred to a future v4.)
// Trade: extra compute (QK done twice; scores recomputed in pass 2) for a rescale-free, trivially
// correct O loop. Both passes keep S on-chip, so S never hits HBM either way.
//
// ONE VARIABLE PER STEP: v2 isolated the operand-reuse lesson (smem tiling) while keeping S in
// HBM. v3 isolates the S-elimination lesson and KEEPS operand handling simple: pass 2 reads K and
// V straight from global memory rather than re-staging them in smem. Step 2 measured that L2 owns
// those operands at every N, so this costs ~0 extra DRAM — exactly the property we exploit. Re-
// introducing operand staging on top of online softmax is a later optimization, not this step.
//
// Precision: FP32 in / FP32 accumulate / FP32 out, identical to v1/v2 (atol/rtol 1e-4 vs SDPA).
// Layout: q,k,v are [B, H, N, d] row-major. Causal masking excludes keys j > query i.
//
// Smem budget (both passes stage two TM*D / TN*D float tiles = 32 KB, well under the T4's 48 KB):
//   d=64  -> TM=TN=64  (sQ 64x64 + sK/sO 64x64 = 32 KB)
//   d=128 -> TM=TN=32  (sQ 32x128 + sK/sO 32x128 = 32 KB)

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>   // at::cuda::getCurrentCUDAStream
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>   // FLT_MAX

namespace {

constexpr int kBlock = 256;  // threads per block for both passes; occupancy-friendly, plain

// Round-up integer division, for sizing grids to cover `n` elements.
inline int ceil_div(int64_t n, int b) { return static_cast<int>((n + b - 1) / b); }

// ---------------------------------------------------------------------------------------
// Pass 1: running (m, l) per query row, streaming over key-blocks. NO S written to HBM.
//
// One block owns TM consecutive query rows of one (batch,head); thread tid<TM owns row tid and
// carries that row's running (m_cur, l_cur) in registers across the whole key axis. The block
// streams the key axis in TN-wide blocks, staging each K-block into smem ONCE so all TM row-
// threads reuse it. For each key it computes the score on-chip and folds it into (m, l) with the
// running-max rescale — the score is never stored.
// ---------------------------------------------------------------------------------------
template <int TM, int TN, int D>
__global__ void pass1_stats_kernel(const float* __restrict__ Q,
                                   const float* __restrict__ K,
                                   float* __restrict__ M_out,
                                   float* __restrict__ L_out,
                                   int N_q, int N_k, float scale, bool causal) {
    __shared__ float sQ[TM * D];   // this tile's query rows  [TM][D], loaded once
    __shared__ float sK[TN * D];   // current key-block       [TN][D], reloaded per block

    const int i0 = blockIdx.x * TM;            // first query row of this tile
    const int64_t bh = blockIdx.y;             // (batch*head) slice
    const int tid = threadIdx.x;

    // Stage the TM query rows once. Out-of-range rows (last partial tile) load 0; they are never
    // stored (active guard below), so the padding is inert.
    const float* Qbh = Q + bh * (int64_t)N_q * D;
    for (int idx = tid; idx < TM * D; idx += kBlock) {
        int r = idx / D, t = idx % D, gi = i0 + r;
        sQ[idx] = (gi < N_q) ? Qbh[(int64_t)gi * D + t] : 0.f;
    }

    // Each thread tid<TM owns query row gi = i0+tid and its running stats. -inf / 0 is the empty
    // accumulator: the first valid score s sets m_cur=s (the rescale factor exp(-inf - s)=0 wipes
    // the empty l) and l_cur becomes exp(0)=1.
    const int gi = i0 + tid;
    const bool active = (tid < TM) && (gi < N_q);
    float m_cur = -FLT_MAX, l_cur = 0.f;

    const float* Kbh = K + bh * (int64_t)N_k * D;
    for (int j0 = 0; j0 < N_k; j0 += TN) {
        // Cooperative load of this key-block (all threads participate, regardless of `active`).
        for (int idx = tid; idx < TN * D; idx += kBlock) {
            int r = idx / D, t = idx % D, gj = j0 + r;
            sK[idx] = (gj < N_k) ? Kbh[(int64_t)gj * D + t] : 0.f;
        }
        __syncthreads();   // sK (and sQ, written before the loop) visible before any compute

        if (active) {
            for (int c = 0; c < TN; ++c) {
                int gj = j0 + c;
                if (gj >= N_k) break;            // padding tail of the last key-block
                if (causal && gj > gi) break;    // keys ascend, so all further c are masked too
                float dot = 0.f;
                for (int t = 0; t < D; ++t) dot += sQ[tid * D + t] * sK[c * D + t];
                float s = dot * scale;
                // Running-max rescale: shift the baseline to s if it is a new max, correcting the
                // accumulated l by the exact factor exp(m_cur - s) in (0,1]; then add this term.
                if (s > m_cur) { l_cur *= __expf(m_cur - s); m_cur = s; }
                l_cur += __expf(s - m_cur);
            }
        }
        __syncthreads();   // protect sK before the next block overwrites it
    }

    if (active) {
        M_out[bh * (int64_t)N_q + gi] = m_cur;
        L_out[bh * (int64_t)N_q + gi] = l_cur;
    }
}

// ---------------------------------------------------------------------------------------
// Pass 2: O = (softmax(S)) @ V, using the FINAL (m, l) from pass 1. Still no S in HBM.
//
// Same tiling: one block owns TM query rows, thread tid<TM owns row tid. Its output row O[gi][:]
// is accumulated in shared memory (sO) so there is no large per-thread register array and no risk
// of a spill polluting the DRAM measurement. The row-thread re-streams the key axis, recomputing
// each score on-chip from sQ and K, forming p = exp(s - m)/l with the final stats (no rescale
// needed — m, l are final), and accumulating p·V into its sO row.
//
// K and V are read straight from global memory (NOT re-staged in smem): Step 2 proved L2 owns
// them at every N, so the redundant reads across row-threads are L2 broadcasts that cost ~0 DRAM.
// This is the deliberate "one variable per step" choice — v3 changes only the S-elimination.
// ---------------------------------------------------------------------------------------
template <int TM, int D>
__global__ void pass2_output_kernel(const float* __restrict__ Q,
                                    const float* __restrict__ K,
                                    const float* __restrict__ V,
                                    const float* __restrict__ M_in,
                                    const float* __restrict__ L_in,
                                    float* __restrict__ O,
                                    int N_q, int N_k, float scale, bool causal) {
    __shared__ float sQ[TM * D];   // this tile's query rows  [TM][D]
    __shared__ float sO[TM * D];   // this tile's output rows [TM][D], accumulated in smem

    const int i0 = blockIdx.x * TM;
    const int64_t bh = blockIdx.y;
    const int tid = threadIdx.x;

    const float* Qbh = Q + bh * (int64_t)N_q * D;
    for (int idx = tid; idx < TM * D; idx += kBlock) {
        int r = idx / D, t = idx % D, gi = i0 + r;
        sQ[idx] = (gi < N_q) ? Qbh[(int64_t)gi * D + t] : 0.f;
        sO[idx] = 0.f;   // zero the output accumulator
    }
    __syncthreads();

    const int gi = i0 + tid;
    const bool active = (tid < TM) && (gi < N_q);
    float m_r = 0.f, inv = 0.f;
    if (active) {
        m_r = M_in[bh * (int64_t)N_q + gi];
        inv = 1.f / L_in[bh * (int64_t)N_q + gi];   // 1/l, applied to every p in this row
    }

    const float* Kbh = K + bh * (int64_t)N_k * D;   // L2-resident operands (read, not staged)
    const float* Vbh = V + bh * (int64_t)N_k * D;
    if (active) {
        for (int gj = 0; gj < N_k; ++gj) {
            if (causal && gj > gi) break;            // masked keys contribute nothing
            float dot = 0.f;
            for (int t = 0; t < D; ++t) dot += sQ[tid * D + t] * Kbh[(int64_t)gj * D + t];
            float p = __expf(dot * scale - m_r) * inv;   // final m,l -> p is already normalized
            for (int t = 0; t < D; ++t) sO[tid * D + t] += p * Vbh[(int64_t)gj * D + t];
        }
    }
    __syncthreads();   // sO fully accumulated before the cooperative store reads it

    float* Obh = O + bh * (int64_t)N_q * D;
    for (int idx = tid; idx < TM * D; idx += kBlock) {
        int r = idx / D, t = idx % D, gi2 = i0 + r;
        if (gi2 < N_q) Obh[(int64_t)gi2 * D + t] = sO[idx];   // O written exactly once
    }
}

// Launch the two online-softmax passes for a fixed compile-time tile config (chosen by head dim).
template <int TM, int TN, int D>
void launch_online(const float* q, const float* k, const float* v,
                   float* M, float* L, float* O,
                   int B, int H, int N_q, int N_k, float scale, bool causal,
                   cudaStream_t stream) {
    const int64_t BH = (int64_t)B * H;
    dim3 grid(ceil_div(N_q, TM), (unsigned)BH);   // one block per (query-tile, batch-head)

    pass1_stats_kernel<TM, TN, D><<<grid, kBlock, 0, stream>>>(q, k, M, L, N_q, N_k, scale, causal);
    pass2_output_kernel<TM, D><<<grid, kBlock, 0, stream>>>(q, k, v, M, L, O, N_q, N_k, scale, causal);
}

}  // anonymous namespace

// Host entry point: streams online softmax in two passes, with S never materialized in HBM.
// The only scratch in HBM is the per-row stats (m, l) — 2 floats per query row, negligible.
// Exposed to Python by binding.cpp as `forward`. Same contract as v1/v2.
torch::Tensor online_attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                       double scale, bool causal) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q,k,v must be CUDA tensors");
    TORCH_CHECK(q.dtype() == torch::kFloat32, "v3 online is FP32-only (the clean baseline)");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "expected [B,H,N,d] tensors");
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous();

    const int B = q.size(0), H = q.size(1), N_q = q.size(2), d = q.size(3);
    const int N_k = k.size(2);
    TORCH_CHECK(k.size(3) == d && v.size(3) == d, "head_dim mismatch across q,k,v");
    TORCH_CHECK(v.size(2) == N_k, "K and V must share sequence length");

    // Per-row running stats; the ONLY HBM scratch (2 floats/row). S is never allocated.
    auto ml_opts = q.options();
    auto M = torch::empty({B, H, N_q}, ml_opts);
    auto L = torch::empty({B, H, N_q}, ml_opts);
    auto O = torch::empty_like(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Tile config picked so both staged tiles (sQ + sK / sO = 2*TM*D floats) sit well under the
    // 48 KB smem budget; mirrors v2's per-head-dim choice (64x64 @ d=64, 32x32 @ d=128).
    if (d == 64) {
        launch_online<64, 64, 64>(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                                  M.data_ptr<float>(), L.data_ptr<float>(), O.data_ptr<float>(),
                                  B, H, N_q, N_k, (float)scale, causal, stream);
    } else if (d == 128) {
        launch_online<32, 32, 128>(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                                   M.data_ptr<float>(), L.data_ptr<float>(), O.data_ptr<float>(),
                                   B, H, N_q, N_k, (float)scale, causal, stream);
    } else {
        TORCH_CHECK(false, "v3 online supports head_dim 64 or 128 (got ", d,
                    "); the tile config is specialized per head dim to fit the 48 KB smem budget");
    }

    return O;
}

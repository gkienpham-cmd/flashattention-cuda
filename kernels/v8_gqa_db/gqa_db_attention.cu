// v8.5 — GQA M-packing with a DOUBLE-BUFFERED KV pipeline (CUDA-core, sm_75/T4-portable). v8 Cut 1
// measured the decode kernel at only ~10-11% HBM: per-CTA-bound, not bandwidth-bound. Cut 1's hot loop
// stalls on every KV tile — `load tile -> __syncthreads -> compute -> __syncthreads -> load` — so the
// global-load latency is EXPOSED (nothing overlaps it). v8.5 attacks exactly that with a textbook
// software pipeline: prefetch tile N+1 while computing on tile N, so the load latency hides behind the
// warp-shuffle compute. The goal is to push %HBM up toward the floor (the "schedule before bytes" step
// v8 proved is still needed before v9 FP8).
//
// This is the PORTABLE double-buffer (ordinary `ld.global`, issued early) — NOT `cp.async`, which is
// Ampere-only (sm_80); the A100 `cp.async` version is a follow-on. To keep two KV buffers within the
// 48 KB static-smem budget WITHOUT dropping below Cut 1's 2 blocks/SM, KV is staged as __half (not
// Cut 1's FP32): 2 half-buffers (sK[2],sV[2]) = 2*2*TN*D*2 B = exactly Cut 1's single FP32 sK+sV
// (2*TN*D*4 B), so occupancy is unchanged and the ONLY new variable is the load/compute overlap. The
// FP16->FP32 conversion just moves from store-time (Cut 1) to read-time here; the math is identical
// (same FP16 values, FP32 accumulate), so correctness holds at the same tol 2e-2.
//
// Everything else is Cut 1 VERBATIM: the packed-row index math (m_row -> g_local,i_q,h_q; gather head
// h_kv; causal on i_q), the online-softmax + O-rescale, the unnormalized (O,m,l) split partials + LSE
// merge, choose_splits, the [B,H_q,N_q,S,*] query-head-shaped workspace.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cfloat>

namespace {

constexpr int kBlock = 256;
constexpr int kWarp  = 32;
constexpr int kWarps = kBlock / kWarp;

inline int ceil_div(int64_t n, int b) { return static_cast<int>((n + b - 1) / b); }

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = kWarp / 2; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

// Cooperative paged gather of one key/value tile [j0, j0+TN) into the given HALF smem buffers. Whole
// block participates (uniform for the following __syncthreads). Keys past j_end are zero-padded; the
// `gj < j_end` guard also bounds the block-table access, so an empty/over-range tile loads zeros with
// no OOB (safe to call for the preload even when the split has 0 tiles).
template <int TN, int D>
__device__ __forceinline__ void load_kv_tile(__half* __restrict__ sKbuf, __half* __restrict__ sVbuf,
                                             const __half* __restrict__ K_pool,
                                             const __half* __restrict__ V_pool,
                                             const int* __restrict__ bt_b,
                                             int j0, int j_end, int page_size, int H_kv, int h_kv) {
    for (int idx = threadIdx.x; idx < TN * D; idx += kBlock) {
        int r = idx / D, t = idx % D, gj = j0 + r;
        if (gj < j_end) {
            int pb  = bt_b[gj / page_size];
            int off = gj % page_size;
            int64_t src = ((int64_t)(pb * page_size + off) * H_kv + h_kv) * D + t;
            sKbuf[idx] = K_pool[src];
            sVbuf[idx] = V_pool[src];
        } else {
            sKbuf[idx] = __float2half(0.f);
            sVbuf[idx] = __float2half(0.f);
        }
    }
}

// ---------------------------------------------------------------------------------------
// PARTIAL kernel. Identical grid/work to v8 Cut 1 (grid = (ceil_div(M,kWarps), num_splits, B*H_kv)),
// but the KV stream is double-buffered: preload tile 0, then each iteration prefetches tile t+1 into
// the OTHER buffer before computing tile t, so the prefetch's global loads overlap the compute.
// ---------------------------------------------------------------------------------------
template <int TN, int D>
__global__ void gqa_db_partial_kernel(const __half* __restrict__ Q,
                                      const __half* __restrict__ K_pool,
                                      const __half* __restrict__ V_pool,
                                      const int* __restrict__ block_table,
                                      float* __restrict__ O_partial,
                                      float* __restrict__ m_partial,
                                      float* __restrict__ l_partial,
                                      int H_q, int H_kv, int G, int N_q, int N_k,
                                      int num_splits, int chunk,
                                      int page_size, int n_logical, int q_offset,
                                      float scale, bool causal) {
    static_assert(D % kWarp == 0, "head_dim must be a multiple of the warp size (32)");
    constexpr int EPT = D / kWarp;

    // Two HALF KV buffers (double-buffer). 2*2*TN*D*2 B == Cut 1's single FP32 sK+sV (2*TN*D*4 B),
    // so per-CTA smem and thus the 2-blocks/SM residency are UNCHANGED — the only new variable is the
    // prefetch overlap.
    __shared__ __half sK[2][TN * D];
    __shared__ __half sV[2][TN * D];

    const int warp  = threadIdx.x >> 5;
    const int lane  = threadIdx.x & (kWarp - 1);
    const int m_row = blockIdx.x * kWarps + warp;
    const int split = blockIdx.y;
    const int64_t bh_kv = blockIdx.z;
    const int b     = (int)(bh_kv / H_kv);
    const int h_kv  = (int)(bh_kv % H_kv);
    const int M     = G * N_q;
    const bool active = (m_row < M);

    const int g_local = active ? (m_row / N_q) : 0;
    const int i_q     = active ? (m_row % N_q) : 0;
    const int h_q     = h_kv * G + g_local;

    const int j_start = split * chunk;
    const int j_end   = (j_start + chunk < N_k) ? (j_start + chunk) : N_k;

    const __half* Qrow = Q + (((int64_t)b * H_q + h_q) * N_q + i_q) * D;
    const int* bt_b    = block_table + (int64_t)b * n_logical;

    float q_reg[EPT], o_reg[EPT];
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        int t = lane + kWarp * e;
        q_reg[e] = active ? __half2float(Qrow[t]) : 0.f;
        o_reg[e] = 0.f;
    }

    float m_cur = -FLT_MAX, l_cur = 0.f;

    // tiles covering [j_start, j_end) in TN-wide steps. Computed inline (NOT host ceil_div — this is
    // device code; ceil_div is __host__-only and calling it here is what failed the first build).
    const int span    = j_end - j_start;
    const int n_tiles = (span > 0) ? ((span + TN - 1) / TN) : 0;

    // Preload tile 0 into buffer 0 (safe even if n_tiles==0: loads zeros, no OOB).
    load_kv_tile<TN, D>(sK[0], sV[0], K_pool, V_pool, bt_b, j_start, j_end, page_size, H_kv, h_kv);
    __syncthreads();

    for (int t = 0; t < n_tiles; ++t) {
        const int cur = t & 1;
        const int j0  = j_start + t * TN;

        // Prefetch the NEXT tile into the other buffer BEFORE computing — its global loads overlap the
        // compute below (the whole point). Different buffer than `cur`, so no read/write conflict.
        if (t + 1 < n_tiles)
            load_kv_tile<TN, D>(sK[(t + 1) & 1], sV[(t + 1) & 1], K_pool, V_pool, bt_b,
                                j_start + (t + 1) * TN, j_end, page_size, H_kv, h_kv);

        if (active) {
            const __half* sKc = sK[cur];
            const __half* sVc = sV[cur];
            for (int c = 0; c < TN; ++c) {
                int gj = j0 + c;
                if (gj >= j_end) break;
                if (causal && gj > i_q + q_offset) break;

                float partial = 0.f;
#pragma unroll
                for (int e = 0; e < EPT; ++e)
                    partial += q_reg[e] * __half2float(sKc[c * D + lane + kWarp * e]);
                float s = warp_reduce_sum(partial) * scale;

                float m_new = fmaxf(m_cur, s);
                float alpha = __expf(m_cur - m_new);
                float p     = __expf(s - m_new);
                l_cur = l_cur * alpha + p;
#pragma unroll
                for (int e = 0; e < EPT; ++e)
                    o_reg[e] = o_reg[e] * alpha + p * __half2float(sVc[c * D + lane + kWarp * e]);
                m_cur = m_new;
            }
        }
        __syncthreads();   // prefetch stores to the next buffer visible + cur compute done before swap
    }

    if (active) {
        int64_t bh_q = (int64_t)b * H_q + h_q;
        int64_t ml = ((bh_q * N_q) + i_q) * num_splits + split;
        int64_t ob = ml * D;
#pragma unroll
        for (int e = 0; e < EPT; ++e)
            O_partial[ob + lane + kWarp * e] = o_reg[e];
        if (lane == 0) {
            m_partial[ml] = m_cur;
            l_partial[ml] = l_cur;
        }
    }
}

// MERGE kernel — identical to v8 Cut 1 / v6 / v7.
__global__ void gqa_db_merge_kernel(const float* __restrict__ O_partial,
                                    const float* __restrict__ m_partial,
                                    const float* __restrict__ l_partial,
                                    float* __restrict__ O,
                                    int N_q, int num_splits, int D) {
    const int i   = blockIdx.x;
    const int64_t bh = blockIdx.y;
    const int t   = threadIdx.x;
    const int64_t ml0 = (((int64_t)bh * N_q) + i) * num_splits;

    float m = -FLT_MAX;
    for (int s = 0; s < num_splits; ++s) m = fmaxf(m, m_partial[ml0 + s]);
    float l = 0.f;
    for (int s = 0; s < num_splits; ++s) l += __expf(m_partial[ml0 + s] - m) * l_partial[ml0 + s];
    const float inv = (l > 0.f) ? (1.f / l) : 0.f;

    if (t < D) {
        float acc = 0.f;
        for (int s = 0; s < num_splits; ++s)
            acc += __expf(m_partial[ml0 + s] - m) * O_partial[(ml0 + s) * D + t];
        O[(((int64_t)bh * N_q) + i) * D + t] = acc * inv;
    }
}

template <int TN, int D>
void launch_partial(const __half* q, const __half* k_pool, const __half* v_pool,
                    const int* block_table,
                    float* O_partial, float* m_partial, float* l_partial,
                    int64_t BH_kv, int H_q, int H_kv, int G, int N_q, int N_k,
                    int num_splits, int chunk, int page_size, int n_logical, int q_offset,
                    float scale, bool causal, cudaStream_t stream) {
    const int M = G * N_q;
    dim3 grid(ceil_div(M, kWarps), (unsigned)num_splits, (unsigned)BH_kv);
    gqa_db_partial_kernel<TN, D><<<grid, kBlock, 0, stream>>>(
        q, k_pool, v_pool, block_table, O_partial, m_partial, l_partial,
        H_q, H_kv, G, N_q, N_k, num_splits, chunk, page_size, n_logical, q_offset, scale, causal);
}

// choose_splits — identical to v8 Cut 1.
int choose_splits(int64_t B, int H_kv, int G, int N_q, int N_k, int num_sm = 40) {
    if (N_k <= 0) return 1;
    const int   S_CAP        = 32;
    const int   MIN_CHUNK    = 256;
    const int   target_blocks = 2 * num_sm;
    const int   row_tiles    = ceil_div((int64_t)G * N_q, kWarps);
    const int64_t base_blocks = (int64_t)B * H_kv * (int64_t)row_tiles;
    int by_occ  = ceil_div(target_blocks, std::max<int64_t>(base_blocks, 1));
    int by_size = ceil_div(N_k, MIN_CHUNK);
    int s = std::min(by_occ, by_size);
    return std::min(std::max(s, 1), S_CAP);
}

}  // anonymous namespace

// Host entry: identical signature/contract to v8 Cut 1's gqa_attention_forward (G derived from shapes).
torch::Tensor gqa_db_attention_forward(torch::Tensor q, torch::Tensor k_pool, torch::Tensor v_pool,
                                       torch::Tensor block_table, int64_t page_size, int64_t n_k,
                                       double scale, bool causal, int64_t q_offset) {
    TORCH_CHECK(q.is_cuda() && k_pool.is_cuda() && v_pool.is_cuda() && block_table.is_cuda(),
                "q, k_pool, v_pool, block_table must be CUDA tensors");
    TORCH_CHECK(q.dim() == 4, "q must be [B,H_q,N_q,d]");
    TORCH_CHECK(k_pool.dim() == 4 && v_pool.dim() == 4,
                "k_pool, v_pool must be [num_blocks, page_size, H_kv, d]");
    TORCH_CHECK(block_table.dim() == 2, "block_table must be [B, n_logical]");
    TORCH_CHECK(block_table.scalar_type() == torch::kInt32, "block_table must be int32");
    q = q.contiguous(); k_pool = k_pool.contiguous(); v_pool = v_pool.contiguous();
    block_table = block_table.contiguous();

    const int B = q.size(0), H_q = q.size(1), N_q = q.size(2), d = q.size(3);
    const int H_kv = k_pool.size(2);
    const int N_k = (int)n_k;
    const int n_logical = (int)block_table.size(1);
    TORCH_CHECK(k_pool.size(2) == v_pool.size(2), "k_pool/v_pool must share H_kv");
    TORCH_CHECK(H_kv >= 1 && H_q % H_kv == 0,
                "H_q must be a positive multiple of H_kv (GQA group factor G = H_q/H_kv); got H_q=",
                H_q, " H_kv=", H_kv);
    const int G = H_q / H_kv;
    TORCH_CHECK(k_pool.size(1) == page_size && v_pool.size(1) == page_size,
                "k_pool/v_pool page_size dim must equal page_size argument");
    TORCH_CHECK(k_pool.size(3) == d && v_pool.size(3) == d, "pool head_dim must equal q's d");
    TORCH_CHECK(block_table.size(0) == B, "block_table must have B rows");
    TORCH_CHECK((int64_t)n_logical * page_size >= N_k,
                "block_table n_logical*page_size must cover N_k logical positions");

    auto qh = q.to(torch::kHalf);
    auto kh = k_pool.to(torch::kHalf);
    auto vh = v_pool.to(torch::kHalf);
    auto O  = torch::empty({B, H_q, N_q, d}, q.options().dtype(torch::kFloat32));

    const int64_t BH_kv = (int64_t)B * H_kv;
    const int S = choose_splits(B, H_kv, G, N_q, N_k);
    const int chunk = ceil_div(N_k, S);

    auto opts_f = q.options().dtype(torch::kFloat32);
    auto O_partial = torch::empty({B, H_q, N_q, S, d}, opts_f);
    auto m_partial = torch::empty({B, H_q, N_q, S},    opts_f);
    auto l_partial = torch::empty({B, H_q, N_q, S},    opts_f);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const __half* qp = reinterpret_cast<const __half*>(qh.data_ptr<at::Half>());
    const __half* kp = reinterpret_cast<const __half*>(kh.data_ptr<at::Half>());
    const __half* vp = reinterpret_cast<const __half*>(vh.data_ptr<at::Half>());
    const int* btp = block_table.data_ptr<int>();
    float* Op = O_partial.data_ptr<float>();
    float* Mp = m_partial.data_ptr<float>();
    float* Lp = l_partial.data_ptr<float>();

    // Tile config per head dim (same TN as Cut 1; half double-buffer = same smem as Cut 1's FP32 single).
    if (d == 64) {
        launch_partial<64, 64>(qp, kp, vp, btp, Op, Mp, Lp, BH_kv, H_q, H_kv, G, N_q, N_k, S, chunk,
                               (int)page_size, (int)n_logical, (int)q_offset, (float)scale, causal, stream);
    } else if (d == 128) {
        launch_partial<32, 128>(qp, kp, vp, btp, Op, Mp, Lp, BH_kv, H_q, H_kv, G, N_q, N_k, S, chunk,
                                (int)page_size, (int)n_logical, (int)q_offset, (float)scale, causal, stream);
    } else {
        TORCH_CHECK(false, "v8.5 gqa_db supports head_dim 64 or 128 (got ", d, ")");
    }

    dim3 mgrid((unsigned)N_q, (unsigned)(B * H_q));
    gqa_db_merge_kernel<<<mgrid, d, 0, stream>>>(Op, Mp, Lp, O.data_ptr<float>(), N_q, S, d);

    return O;
}

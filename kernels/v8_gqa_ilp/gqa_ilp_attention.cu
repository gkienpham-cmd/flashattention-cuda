// v8.6 Arm 2 (KEY-ILP) — GQA M-packing with an UNROLLED key loop (CUDA-core, sm_75/T4). Forked from
// v8 Cut 1 (gqa_attention.cu); the decode algorithm (GQA-packed split-KV online-softmax partial + LSE
// merge, FP16-in/FP32-accum, paged gather, packed-row index math, causal-on-i_q, FP32-staged smem) is
// carried BYTE-IDENTICAL. This arm changes exactly ONE thing — the SHAPE of the inner key loop — to
// attack the limiter v8 + v8.5 MEASURED (decode is compute-latency-bound at ~10% HBM; the wall is the
// per-key warp-shuffle reduction + serial online-softmax recurrence, NOT the KV-load latency).
//
//   KEY-ILP LEVER. Cut 1 processes one key per iteration: EPT FMAs -> a 5-deep __shfl_xor butterfly
//   (warp_reduce_sum, ~30-40 cyc EXPOSED) -> a softmax update that depends on the previous key. The
//   reduction latency is exposed because there is nothing independent to overlap it with. This arm
//   unrolls the key loop by KU=4: it computes KU INDEPENDENT dot-product partials, then issues their
//   KU reductions back-to-back so the independent shfl dependency chains PIPELINE (the scheduler hides
//   each reduction's latency under the next), THEN applies the KU softmax updates sequentially. The
//   online-softmax recurrence (m_cur/l_cur/o_reg) stays serial on purpose — that is the variable's
//   limit, and the prediction is that ILP helps only the reduction sub-part while the serial
//   recurrence stays exposed (so this arm is predicted WEAKER than Arm 1's occupancy lever).
//
//   smem stays FP32 (Cut 1 VERBATIM) -> 2 blocks/SM unchanged, so ILP is the ONLY variable vs Cut 1
//   (Arm 1 isolates occupancy separately; mixing the two here would confound the ablation). NOTE on
//   vectorized loads: the lane-strided access sK[c*D + lane + 32*e] is NOT contiguous per lane
//   (consecutive e are 32 floats apart), so float2/half2 vectorization does not apply to this layout —
//   the ILP unroll, not a wider load, is the lever.
//
// The monotone causal/j_end cutoff is precomputed per tile as c_lim (a key c is valid iff
// gj=j0+c < j_end AND, if causal, gj <= i_q+q_offset — both monotone in c), so the unrolled inner loop
// runs c in [0, c_lim) with no per-key break. c_lim is warp-uniform (i_q, j0, j_end are), so the
// guarded warp_reduce_sum calls stay warp-uniform (no divergent __shfl). A fully-future tile gives
// c_lim=0 -> no work -> writes (m=-inf, l=0, O=0); the merge weights it out (same as Cut 1).

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
constexpr int KU     = 4;     // key-unroll factor: # of independent reductions kept in flight

inline int ceil_div(int64_t n, int b) { return static_cast<int>((n + b - 1) / b); }

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = kWarp / 2; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

// ---------------------------------------------------------------------------------------
// PARTIAL kernel. Grid/work IDENTICAL to v8 Cut 1 (grid = (ceil_div(M,kWarps), num_splits, B*H_kv),
// M = G*N_q), FP32 smem unchanged. The ONLY change is the inner key loop is KU-unrolled (compute KU
// independent partials + reductions, then KU sequential softmax updates).
// ---------------------------------------------------------------------------------------
template <int TN, int D>
__global__ void gqa_ilp_partial_kernel(const __half* __restrict__ Q,        // [B,H_q,N_q,D] dense
                                       const __half* __restrict__ K_pool,    // [num_blocks,page_size,H_kv,D]
                                       const __half* __restrict__ V_pool,    // [num_blocks,page_size,H_kv,D]
                                       const int* __restrict__ block_table,  // [B, n_logical]
                                       float* __restrict__ O_partial,        // [B,H_q,N_q,S,D]
                                       float* __restrict__ m_partial,        // [B,H_q,N_q,S]
                                       float* __restrict__ l_partial,        // [B,H_q,N_q,S]
                                       int H_q, int H_kv, int G, int N_q, int N_k,
                                       int num_splits, int chunk,
                                       int page_size, int n_logical, int q_offset,
                                       float scale, bool causal) {
    static_assert(D % kWarp == 0, "head_dim must be a multiple of the warp size (32)");
    constexpr int EPT = D / kWarp;

    __shared__ float sK[TN * D];   // FP32 staged tiles (Cut 1 verbatim) -> 2 blocks/SM unchanged
    __shared__ float sV[TN * D];

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

    for (int j0 = j_start; j0 < j_end; j0 += TN) {
        // Cooperative paged gather into FP32 smem — byte-identical to Cut 1.
        for (int idx = threadIdx.x; idx < TN * D; idx += kBlock) {
            int r = idx / D, t = idx % D, gj = j0 + r;
            bool ok = gj < j_end;
            if (ok) {
                int pb  = bt_b[gj / page_size];
                int off = gj % page_size;
                int64_t src = ((int64_t)(pb * page_size + off) * H_kv + h_kv) * D + t;
                sK[idx] = __half2float(K_pool[src]);
                sV[idx] = __half2float(V_pool[src]);
            } else {
                sK[idx] = 0.f;
                sV[idx] = 0.f;
            }
        }
        __syncthreads();

        if (active) {
            // Monotone valid-key cutoff for this tile (replaces Cut 1's per-key break). A key c is
            // valid iff gj=j0+c < j_end AND (!causal OR gj <= i_q+q_offset); both monotone in c.
            int c_lim = j_end - j0;                       // gj < j_end
            if (c_lim > TN) c_lim = TN;
            if (causal) {
                int cc = i_q + q_offset - j0 + 1;         // gj <= i_q+q_offset
                if (cc < c_lim) c_lim = cc;
            }
            if (c_lim < 0) c_lim = 0;

            for (int c0 = 0; c0 < c_lim; c0 += KU) {
                const int n = (c_lim - c0 < KU) ? (c_lim - c0) : KU;   // valid keys in this group

                // Stage 1: KU INDEPENDENT dot products + reductions. Separate statements so the
                // scheduler pipelines the KU shfl chains (the whole point). n is warp-uniform, so the
                // guarded warp_reduce_sum stays uniform across the warp.
                float s_arr[KU];
#pragma unroll
                for (int u = 0; u < KU; ++u) {
                    if (u < n) {
                        const int c = c0 + u;
                        float partial = 0.f;
#pragma unroll
                        for (int e = 0; e < EPT; ++e)
                            partial += q_reg[e] * sK[c * D + lane + kWarp * e];
                        s_arr[u] = warp_reduce_sum(partial) * scale;
                    }
                }

                // Stage 2: KU SEQUENTIAL online-softmax updates (the recurrence stays serial — the
                // ILP only overlaps Stage 1's reductions, by design).
                for (int u = 0; u < n; ++u) {
                    const int c = c0 + u;
                    const float s = s_arr[u];
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
        }
        __syncthreads();   // protect sK/sV before the next tile overwrites them
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
__global__ void gqa_ilp_merge_kernel(const float* __restrict__ O_partial,
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
    gqa_ilp_partial_kernel<TN, D><<<grid, kBlock, 0, stream>>>(
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
torch::Tensor gqa_ilp_attention_forward(torch::Tensor q, torch::Tensor k_pool, torch::Tensor v_pool,
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

    // Tile config per head dim (FP32 smem, same as Cut 1).
    if (d == 64) {
        launch_partial<64, 64>(qp, kp, vp, btp, Op, Mp, Lp, BH_kv, H_q, H_kv, G, N_q, N_k, S, chunk,
                               (int)page_size, (int)n_logical, (int)q_offset, (float)scale, causal, stream);
    } else if (d == 128) {
        launch_partial<32, 128>(qp, kp, vp, btp, Op, Mp, Lp, BH_kv, H_q, H_kv, G, N_q, N_k, S, chunk,
                                (int)page_size, (int)n_logical, (int)q_offset, (float)scale, causal, stream);
    } else {
        TORCH_CHECK(false, "v8.6 gqa_ilp supports head_dim 64 or 128 (got ", d, ")");
    }

    dim3 mgrid((unsigned)N_q, (unsigned)(B * H_q));
    gqa_ilp_merge_kernel<<<mgrid, d, 0, stream>>>(Op, Mp, Lp, O.data_ptr<float>(), N_q, S, d);

    return O;
}

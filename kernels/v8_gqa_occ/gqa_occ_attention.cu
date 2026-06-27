// v8.6 Arm 1 (OCCUPANCY) — GQA M-packing with HALF-resident smem (CUDA-core, sm_75/T4). Forked from
// v8 Cut 1 (gqa_attention.cu); the decode algorithm (GQA-packed split-KV online-softmax partial + LSE
// merge, FP16-in/FP32-accum, paged gather, packed-row index math, causal-on-i_q) is carried
// BYTE-IDENTICAL. This arm changes exactly ONE thing — the staged KV tile's dtype — to attack the
// limiter v8 + v8.5 MEASURED (decode is per-CTA / compute-latency-bound at ~10% HBM; the wall is the
// per-key warp-shuffle reduction + serial online-softmax recurrence, NOT the KV-load latency).
//
//   OCCUPANCY LEVER. Cut 1 stages sK+sV as FP32 = 2*TN*D*4 B = 32 KB/block, which caps residency at
//   2 blocks/SM (T4 has 64 KB smem/SM; registers allow ~8, so smem is the binding limit). Storing the
//   tile as __half halves it to 16 KB/block -> 4 blocks/SM = 2x the resident warps. At decode G=8 that
//   lifts SM warp occupancy from ~50% to ~100%, so thread-level parallelism (more warps in flight) can
//   hide the reduction + softmax latency that one warp's serial inner loop exposes. This is the
//   single-variable test of "is the latency hideable by occupancy?".
//
// DISTINCTION vs v8.5 (gqa_db_attention.cu): v8.5 ALSO stored the tile as half, but spent the freed
// 16 KB on a SECOND buffer (double-buffering) and so stayed at 2 blocks/SM — it tested load/compute
// overlap (null result). This arm keeps a SINGLE buffer and uses the freed smem to RAISE occupancy:
// same half tile, opposite use of the savings. The FP16->FP32 conversion moves from store-time (Cut 1)
// to read-time (here, like v8.5); the math is identical (same FP16 values, FP32 accumulate), so
// correctness holds at the same tol 2e-2.
//
// Everything else is Cut 1 VERBATIM: cooperative paged gather, online-softmax + O-rescale, the
// unnormalized (O,m,l) split partials + cross-split LSE merge, choose_splits, the [B,H_q,N_q,S,*]
// query-head-shaped workspace, and the 64x64 @ d=64 / 32x32 @ d=128 tiles.

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

// ---------------------------------------------------------------------------------------
// PARTIAL kernel. Grid/work IDENTICAL to v8 Cut 1 (grid = (ceil_div(M,kWarps), num_splits, B*H_kv),
// M = G*N_q). The ONLY change is sK/sV are __half (16 KB total, vs Cut 1's 32 KB FP32) -> 4 blocks/SM.
// Read-time FP16->FP32 conversion in the dot-product and the p*V accumulate (math-identical to Cut 1).
// ---------------------------------------------------------------------------------------
template <int TN, int D>
__global__ void gqa_occ_partial_kernel(const __half* __restrict__ Q,        // [B,H_q,N_q,D] dense
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
    constexpr int EPT = D / kWarp;   // elements of the row each lane owns (d=64->2, d=128->4)

    // HALF staged tiles: 2*TN*D*2 B = 16 KB (half of Cut 1's 32 KB FP32) -> 4 blocks/SM. This is the
    // single isolated variable of Arm 1 (occupancy); the conversion to FP32 happens at READ time below.
    __shared__ __half sK[TN * D];
    __shared__ __half sV[TN * D];

    const int warp  = threadIdx.x >> 5;
    const int lane  = threadIdx.x & (kWarp - 1);
    const int m_row = blockIdx.x * kWarps + warp;  // global PACKED-M row
    const int split = blockIdx.y;
    const int64_t bh_kv = blockIdx.z;              // (batch * KV head)
    const int b     = (int)(bh_kv / H_kv);
    const int h_kv  = (int)(bh_kv % H_kv);
    const int M     = G * N_q;
    const bool active = (m_row < M);

    const int g_local = active ? (m_row / N_q) : 0;
    const int i_q     = active ? (m_row % N_q) : 0;
    const int h_q     = h_kv * G + g_local;        // GLOBAL query head this warp owns

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
        // Cooperative paged gather of this key/value tile into HALF smem (store half directly — no
        // FP32 conversion here, unlike Cut 1). Whole block participates so the __syncthreads is
        // reached uniformly. Keys past j_end zero-padded; the gj<j_end guard also bounds N_k.
        for (int idx = threadIdx.x; idx < TN * D; idx += kBlock) {
            int r = idx / D, t = idx % D, gj = j0 + r;
            bool ok = gj < j_end;
            if (ok) {
                int pb  = bt_b[gj / page_size];
                int off = gj % page_size;
                int64_t src = ((int64_t)(pb * page_size + off) * H_kv + h_kv) * D + t;
                sK[idx] = K_pool[src];   // half -> half (read-time conversion to FP32 below)
                sV[idx] = V_pool[src];
            } else {
                sK[idx] = __float2half(0.f);
                sV[idx] = __float2half(0.f);
            }
        }
        __syncthreads();

        if (active) {
            for (int c = 0; c < TN; ++c) {
                int gj = j0 + c;
                if (gj >= j_end) break;
                if (causal && gj > i_q + q_offset) break;   // mask uses i_q, NOT m_row

                float partial = 0.f;
#pragma unroll
                for (int e = 0; e < EPT; ++e)
                    partial += q_reg[e] * __half2float(sK[c * D + lane + kWarp * e]);
                float s = warp_reduce_sum(partial) * scale;

                float m_new = fmaxf(m_cur, s);
                float alpha = __expf(m_cur - m_new);
                float p     = __expf(s - m_new);
                l_cur = l_cur * alpha + p;
#pragma unroll
                for (int e = 0; e < EPT; ++e)
                    o_reg[e] = o_reg[e] * alpha + p * __half2float(sV[c * D + lane + kWarp * e]);
                m_cur = m_new;
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

// MERGE kernel — identical to v8 Cut 1 / v6 / v7 (sees only the [B,H_q,N_q,S,*] workspace).
__global__ void gqa_occ_merge_kernel(const float* __restrict__ O_partial,
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
    gqa_occ_partial_kernel<TN, D><<<grid, kBlock, 0, stream>>>(
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
torch::Tensor gqa_occ_attention_forward(torch::Tensor q, torch::Tensor k_pool, torch::Tensor v_pool,
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

    // Tile config per head dim. HALF smem = 2*TN*D*2 B (16 KB), comfortably under the 48 KB budget and
    // half of Cut 1's FP32 footprint -> 4 blocks/SM.
    if (d == 64) {
        launch_partial<64, 64>(qp, kp, vp, btp, Op, Mp, Lp, BH_kv, H_q, H_kv, G, N_q, N_k, S, chunk,
                               (int)page_size, (int)n_logical, (int)q_offset, (float)scale, causal, stream);
    } else if (d == 128) {
        launch_partial<32, 128>(qp, kp, vp, btp, Op, Mp, Lp, BH_kv, H_q, H_kv, G, N_q, N_k, S, chunk,
                                (int)page_size, (int)n_logical, (int)q_offset, (float)scale, causal, stream);
    } else {
        TORCH_CHECK(false, "v8.6 gqa_occ supports head_dim 64 or 128 (got ", d, ")");
    }

    dim3 mgrid((unsigned)N_q, (unsigned)(B * H_q));
    gqa_occ_merge_kernel<<<mgrid, d, 0, stream>>>(Op, Mp, Lp, O.data_ptr<float>(), N_q, S, d);

    return O;
}

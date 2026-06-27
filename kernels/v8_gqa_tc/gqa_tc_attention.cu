// v8 GQA M-packing on TENSOR CORES — Cut 2a (Turing WMMA, sm_75/T4-compatible). This is the
// GEMV->GEMM step ON TOP of Cut 1's M-packing: keep the paged GQA split-KV decode skeleton
// (kernels/v8_gqa/gqa_attention.cu) but run the two matmuls on 16x16x16 WMMA tensor cores (v5's
// fix, kernels/v5_wmma/wmma_attention.cu) instead of the warp-shuffle dot product.
//
// WHY a Turing-WMMA cut before the A100 mma.m16n8k16 + cp.async version (Cut 2b): the *question* of
// Cut 2 — does turning the M=G score GEMV into a real GEMM help decode, and which M>=16 strategy
// wins — is answerable on the free T4. WMMA's only tensor-core shape is 16x16x16, so M=G<16 is
// PADDED to 16 (rows G..15 carry a zero query -> zero score -> dropped, never written). That wastes
// half the tensor-core rows at G=8 (the efficiency cost the A100 mma.m16n8k16 path later removes),
// but it measures the GEMV->GEMM transition cheaply. cp.async (async smem staging) is likewise a
// latency refinement deferred to Cut 2b.
//
// THE M>=16 ABLATION (this file = variant 1; 2 and 3 are follow-ups once this is green on T4):
//   1. PAD M=G->16 + mask    (here): one WMMA M-tile, rows >=G zeroed. Half-utilized at G=8.
//   2. multi-group-pack          : pack 2 KV-groups so M=16 is full (a sibling kernel/flag).
//   3. CUDA-core-QK + WMMA-PV     : Cut 1's warp-shuffle QK, tensor-core PV only (a sibling).
//
// Carried UNCHANGED from Cut 1: the 3D grid (batch, KV head, split), the packed-row index math
// (m_row -> g_local,i_q,h_q; gather head h_kv; causal mask on i_q NOT m_row), the paged block-table
// gather, the unnormalized (O,m,l) split partials + the LSE merge, choose_splits, and the
// query-head-shaped [B,H_q,N_q,S,*] workspace. Carried from v5: the opaque-fragment softmax-through-
// smem dance (WMMA accumulators are un-indexable, so S goes QK->smem->row-softmax->P-as-half->PV) and
// the O-rescale folded into the PV accumulator (load rescaled oRun into the accumulator, mma P@V on
// top). Precision: FP16-in / FP32-accum / FP32-out, tol 2e-2 like v5/v6/v7/v8.
//
// Layout: Q dense [B,H_q,N_q,d]. K_pool/V_pool [num_blocks,page_size,H_kv,d]. block_table int32
// [B,n_logical], per-sequence. Causal excludes keys j > (query position i_q + q_offset).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <algorithm>
#include <cfloat>

namespace {
using namespace nvcuda;

constexpr int kWarp  = 32;
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;   // the only Turing half/float WMMA shape
constexpr int BM     = WMMA_M;                          // one padded M-tile = up to 16 packed rows

inline int ceil_div(int64_t n, int b) { return static_cast<int>((n + b - 1) / b); }

// ---------------------------------------------------------------------------------------
// PARTIAL kernel. Grid = (ceil_div(M,16), num_splits, B*H_kv), M = G*N_q. ONE warp per block owns the
// 16-row packed M-tile [m_tile*16, m_tile*16+16) of one (batch, KV head) for ONE key split, padded to
// 16 (rows whose packed index >= M carry a zero query). The block streams this split's keys in BN-wide
// blocks (paged gather -> smem as half), does WMMA QK -> smem softmax (mask pad/causal/split) -> WMMA
// PV on top of the rescaled running oRun, and writes the UNNORMALIZED (oRun, m, l) per active row to
// the [B,H_q,N_q,S,*] workspace. The merge kernel (identical to Cut 1) combines across splits.
// ---------------------------------------------------------------------------------------
template <int BN, int D>
__global__ void gqa_tc_partial_kernel(const __half* __restrict__ Q,        // [B,H_q,N_q,D]
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
    static_assert(BN % WMMA_N == 0, "BN must be a multiple of 16");
    static_assert(D  % WMMA_K == 0, "head_dim must be a multiple of 16");
    constexpr int DK = D  / WMMA_K;   // QK contraction subtiles (over d)
    constexpr int NN = BN / WMMA_N;   // key subtiles (QK output / PV contraction)
    constexpr int DN = D  / WMMA_N;   // PV output subtiles (over d)
    constexpr int kBlock = kWarp;     // one warp per block

    __shared__ __half sQ[BM * D];     // packed query tile (zero-padded rows >= M)   [16][D]
    __shared__ __half sK[BN * D];     // current key block (paged gather, half)      [BN][D]
    __shared__ __half sV[BN * D];     // current value block                          [BN][D]
    __shared__ float  sS[BM * BN];    // QK scores scratch                            [16][BN]
    __shared__ __half sP[BM * BN];    // softmax probs, reloaded for PV               [16][BN]
    __shared__ float  oRun[BM * D];   // running UNNORMALIZED output (FP32 accum)     [16][D]

    const int lane  = threadIdx.x & (kWarp - 1);   // 0..31; lanes <16 own softmax rows
    const int m_tile = blockIdx.x;                 // which 16-row packed M-tile
    const int split = blockIdx.y;
    const int64_t bh_kv = blockIdx.z;
    const int b     = (int)(bh_kv / H_kv);
    const int h_kv  = (int)(bh_kv % H_kv);
    const int M     = G * N_q;                      // packed-M extent
    const int i0    = m_tile * BM;                  // first packed row of this tile

    const int j_start = split * chunk;
    const int j_end   = (j_start + chunk < N_k) ? (j_start + chunk) : N_k;
    const int* bt_b   = block_table + (int64_t)b * n_logical;

    // Stage this tile's 16 packed query rows into smem as half (zero-padded past M), zero oRun. Row r
    // decodes to packed row m_row=i0+r -> (g_local,i_q) -> global query head h_q.
    for (int idx = threadIdx.x; idx < BM * D; idx += kBlock) {
        int r = idx / D, t = idx % D, m_row = i0 + r;
        if (m_row < M) {
            int g_local = m_row / N_q, i_q = m_row % N_q, h_q = h_kv * G + g_local;
            sQ[idx] = Q[(((int64_t)b * H_q + h_q) * N_q + i_q) * D + t];
        } else {
            sQ[idx] = __float2half(0.f);
        }
        oRun[idx] = 0.f;
    }
    __syncthreads();

    // Q resident in fragments for the whole split (one warp -> the single 16-row M-tile).
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> qFrag[DK];
#pragma unroll
    for (int kk = 0; kk < DK; ++kk)
        wmma::load_matrix_sync(qFrag[kk], &sQ[kk * WMMA_K], D);

    // Per-row running softmax stats: lane l<16 owns packed row i0+l.
    float m_run = -FLT_MAX, l_run = 0.f;

    for (int j0 = j_start; j0 < j_end; j0 += BN) {
        // Cooperative paged gather of this key/value block into smem (half). Each logical key gj is
        // gathered through the block table (lb -> physical page pb -> pool row, head slice h_kv);
        // zero-padded past j_end so partial tiles are safe for WMMA.
        for (int idx = threadIdx.x; idx < BN * D; idx += kBlock) {
            int r = idx / D, t = idx % D, gj = j0 + r;
            if (gj < j_end) {
                int pb  = bt_b[gj / page_size];
                int off = gj % page_size;
                int64_t src = ((int64_t)(pb * page_size + off) * H_kv + h_kv) * D + t;
                sK[idx] = K_pool[src];
                sV[idx] = V_pool[src];
            } else {
                sK[idx] = __float2half(0.f);
                sV[idx] = __float2half(0.f);
            }
        }
        __syncthreads();

        // ---- QK: S[16][BN] = Q @ K^T (K^T via col_major B-fragment, no transpose). ----
#pragma unroll
        for (int ns = 0; ns < NN; ++ns) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
            wmma::fill_fragment(acc, 0.f);
#pragma unroll
            for (int kk = 0; kk < DK; ++kk) {
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> kFrag;
                wmma::load_matrix_sync(kFrag, &sK[(ns * WMMA_N) * D + kk * WMMA_K], D);
                wmma::mma_sync(acc, qFrag[kk], kFrag, acc);
            }
            wmma::store_matrix_sync(&sS[ns * WMMA_N], acc, BN, wmma::mem_row_major);
        }
        __syncthreads();

        // ---- Online softmax over this tile's rows (lane l<16 owns packed row i0+l from smem). ----
        if (lane < WMMA_M) {
            const int r     = lane;                 // local row in [0,16)
            const int m_row = i0 + r;               // packed row
            const bool act  = (m_row < M);
            const int i_q   = act ? (m_row % N_q) : 0;   // query position (for causal); pad -> 0
            float*  sSr = &sS[r * BN];
            __half* sPr = &sP[r * BN];

            // pass 1: scale + mask in place (pad rows, padded keys, future keys), find this block's max
            float rowmax = -FLT_MAX;
            for (int c = 0; c < BN; ++c) {
                int gj = j0 + c;
                float s = sSr[c] * scale;
                if (!act || gj >= j_end || (causal && gj > i_q + q_offset)) s = -FLT_MAX;
                sSr[c] = s;
                rowmax = fmaxf(rowmax, s);
            }
            float m_new = fmaxf(m_run, rowmax);
            // ALL-MASKED BLOCK GUARD (the split+WMMA edge case v5/Cut1 never hit): if this whole tile
            // row is masked AND nothing was seen yet, m_new is still -FLT_MAX. Then exp(sSr[c]-m_new) =
            // exp(-FLT_MAX - -FLT_MAX) = exp(0) = 1 would wrongly inject l += BN. Force a no-op update
            // (alpha=1, p=0) so a fully-future split (prefill causal) / pad row writes l=0 -> dropped in
            // merge. When m_new is finite, exp(-FLT_MAX - finite)=0 already handles per-entry masking.
            const bool empty = (m_new == -FLT_MAX);
            float alpha = empty ? 1.f : __expf(m_run - m_new);   // exp(m_old-m_new) wipes empty state
            float psum  = 0.f;
            // pass 2: probabilities as half for the PV GEMM (masked -> exp(-inf)=0)
            for (int c = 0; c < BN; ++c) {
                float p = empty ? 0.f : __expf(sSr[c] - m_new);
                sPr[c]  = __float2half(p);
                psum   += p;
            }
            l_run = l_run * alpha + psum;
            m_run = m_new;
            // rescale running O row by alpha BEFORE PV adds this block (the O-rescale)
            float* oR = &oRun[r * D];
#pragma unroll
            for (int cc = 0; cc < D; ++cc) oR[cc] *= alpha;
        }
        __syncthreads();

        // ---- PV: oRun += P @ V. Load rescaled oRun into the accumulator, mma P@V on top, store back. ----
#pragma unroll
        for (int nd = 0; nd < DN; ++nd) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
            wmma::load_matrix_sync(acc, &oRun[nd * WMMA_N], D, wmma::mem_row_major);
#pragma unroll
            for (int kk = 0; kk < NN; ++kk) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> pFrag;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> vFrag;
                wmma::load_matrix_sync(pFrag, &sP[kk * WMMA_K], BN);
                wmma::load_matrix_sync(vFrag, &sV[(kk * WMMA_K) * D + nd * WMMA_N], D);
                wmma::mma_sync(acc, pFrag, vFrag, acc);
            }
            wmma::store_matrix_sync(&oRun[nd * WMMA_N], acc, D, wmma::mem_row_major);
        }
        __syncthreads();
    }

    // Write this split's UNNORMALIZED partial (oRun, m_run, l_run) for each ACTIVE packed row, indexed
    // by the query head. lane l<16 owns row i0+l. The merge kernel normalizes + combines across splits.
    if (lane < WMMA_M) {
        const int r = lane, m_row = i0 + r;
        if (m_row < M) {
            int g_local = m_row / N_q, i_q = m_row % N_q, h_q = h_kv * G + g_local;
            int64_t bh_q = (int64_t)b * H_q + h_q;
            int64_t ml = ((bh_q * N_q) + i_q) * num_splits + split;
            int64_t ob = ml * D;
            float* oR = &oRun[r * D];
            for (int cc = 0; cc < D; ++cc) O_partial[ob + cc] = oR[cc];
            m_partial[ml] = m_run;
            l_partial[ml] = l_run;
        }
    }
}

// MERGE kernel — IDENTICAL to Cut 1 (kernels/v8_gqa/gqa_attention.cu) / v6 / v7. The merge sees only
// the [B,H_q,N_q,S,*] workspace, so neither GQA nor tensor cores reach it.
__global__ void gqa_tc_merge_kernel(const float* __restrict__ O_partial,
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

template <int BN, int D>
void launch_partial(const __half* q, const __half* k_pool, const __half* v_pool,
                    const int* block_table,
                    float* O_partial, float* m_partial, float* l_partial,
                    int64_t BH_kv, int H_q, int H_kv, int G, int N_q, int N_k,
                    int num_splits, int chunk, int page_size, int n_logical, int q_offset,
                    float scale, bool causal, cudaStream_t stream) {
    const int M = G * N_q;
    dim3 grid(ceil_div(M, BM), (unsigned)num_splits, (unsigned)BH_kv);
    gqa_tc_partial_kernel<BN, D><<<grid, kWarp, 0, stream>>>(
        q, k_pool, v_pool, block_table, O_partial, m_partial, l_partial,
        H_q, H_kv, G, N_q, N_k, num_splits, chunk, page_size, n_logical, q_offset, scale, causal);
}

// choose_splits — identical policy to Cut 1, but row_tiles covers M in 16-row WMMA tiles.
int choose_splits(int64_t B, int H_kv, int G, int N_q, int N_k, int num_sm = 40) {
    if (N_k <= 0) return 1;
    const int   S_CAP        = 32;
    const int   MIN_CHUNK    = 256;
    const int   target_blocks = 2 * num_sm;
    const int   row_tiles    = ceil_div((int64_t)G * N_q, BM);
    const int64_t base_blocks = (int64_t)B * H_kv * (int64_t)row_tiles;
    int by_occ  = ceil_div(target_blocks, std::max<int64_t>(base_blocks, 1));
    int by_size = ceil_div(N_k, MIN_CHUNK);
    int s = std::min(by_occ, by_size);
    return std::min(std::max(s, 1), S_CAP);
}

}  // anonymous namespace

// Host entry: identical signature/contract to Cut 1's gqa_attention_forward (G derived from shapes),
// but the matmuls run on Turing WMMA tensor cores. FP16-in / FP32-accum / FP32-out.
torch::Tensor gqa_tc_attention_forward(torch::Tensor q, torch::Tensor k_pool, torch::Tensor v_pool,
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

    // BN per head dim (WMMA multiples of 16; mirrors Cut 1's TN and v5's smem budget).
    if (d == 64) {
        launch_partial<64, 64>(qp, kp, vp, btp, Op, Mp, Lp, BH_kv, H_q, H_kv, G, N_q, N_k, S, chunk,
                               (int)page_size, (int)n_logical, (int)q_offset, (float)scale, causal, stream);
    } else if (d == 128) {
        launch_partial<32, 128>(qp, kp, vp, btp, Op, Mp, Lp, BH_kv, H_q, H_kv, G, N_q, N_k, S, chunk,
                                (int)page_size, (int)n_logical, (int)q_offset, (float)scale, causal, stream);
    } else {
        TORCH_CHECK(false, "v8 gqa_tc supports head_dim 64 or 128 (got ", d, ")");
    }

    dim3 mgrid((unsigned)N_q, (unsigned)(B * H_q));
    gqa_tc_merge_kernel<<<mgrid, d, 0, stream>>>(Op, Mp, Lp, O.data_ptr<float>(), N_q, S, d);

    return O;
}

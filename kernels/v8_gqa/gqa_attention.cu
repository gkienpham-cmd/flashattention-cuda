// v8 GQA M-packing — Cut 1 (CUDA-core, sm_75/T4-compatible). Forked from v7 paged-KV decode; the
// decode algorithm (split-KV online-softmax partial + LSE merge, FP16-in/FP32-accum, paged gather)
// is carried BYTE-IDENTICAL. v8 changes exactly ONE thing — which work each warp owns — to attack
// the limiter v7 MEASURED (decode is per-CTA-bound: at N_q=1 only 1 of 8 warps computes, and
// sK+sV=32 KB caps residency at 2 blocks/SM, so batch adds waves, not per-SM parallelism).
//
//   GQA M-PACKING. A GQA model has H_q query heads but only H_kv = H_q/G key/value heads; the G
//   query heads of a group all attend the SAME KV head. v7 ran one (batch,query-head) per z-block,
//   so the G heads of a group each re-read that KV head and each used only warp 0. v8 packs the G
//   query heads of one group into the score GEMM's M dimension:
//       grid.z iterates (batch, KV head) — B*H_kv blocks, G× fewer than v7's B*H_q.
//       within a block, packed row m_row = blockIdx.x*kWarps + warp owns query head h_kv*G + g_local
//       (g_local = m_row / N_q), all warps SHARE the one staged KV tile (sK/sV).
//   Effects, all aimed at the per-CTA wall: KV read ONCE per group (not G×), G compute-warps light
//   up (not 1), and decode AI = 2/b -> 2G/b. The staged KV tile (32 KB) and the per-warp register
//   accumulators are UNCHANGED, so residency stays 2 blocks/SM — but each resident block now does G×
//   the useful work per staged KV byte. This is a per-CTA-EFFICIENCY win, NOT an occupancy one
//   (v7 proved filling the grid doesn't move %HBM). Cut 2 (sm_80, separate) turns the M=G GEMV into
//   a tensor-core GEMM; this CUDA-core cut isolates the M-packing variable on T4 first.
//
// What is carried from v7 unchanged: the cooperative paged gather (block table -> physical page),
// the online-softmax core with the O-rescale, the cross-split LSE merge, the [B,H_q,N_q,S,*]
// workspace, choose_splits, and the 64x64 @ d=64 / 32x32 @ d=128 tiles. The causal query-offset is
// carried too (decode places the single query at logical N_k-1 so it attends the whole cache).
//
// Layout: Q dense [B,H_q,N_q,d] row-major. K_pool/V_pool [num_blocks, page_size, H_kv, d] row-major.
// block_table int32 [B, n_logical], per-sequence (shared across the G query heads of a group).
// Causal excludes keys j > (query position i_q + q_offset) — NOTE i_q, not the packed row index.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>   // at::cuda::getCurrentCUDAStream
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>   // std::min/std::max on the host
#include <cfloat>      // FLT_MAX

namespace {

constexpr int kBlock = 256;             // threads per block (partial kernel)
constexpr int kWarp  = 32;              // lanes per warp
constexpr int kWarps = kBlock / kWarp;  // 8 warps per block -> up to 8 packed query rows per block

// Round-up integer division, for sizing grids/chunks to cover `n` elements.
inline int ceil_div(int64_t n, int b) { return static_cast<int>((n + b - 1) / b); }

// Butterfly all-reduce: every lane ends up with the full 32-lane sum (so the score is warp-uniform).
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = kWarp / 2; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

// ---------------------------------------------------------------------------------------
// PARTIAL kernel. Grid = (ceil_div(M, kWarps), num_splits, B*H_kv), M = G*N_q. One block owns kWarps
// consecutive PACKED rows of one (batch, KV head) for ONE key split; warp `w` owns packed row
// m_row = blockIdx.x*kWarps + w, which decodes to (query head h_kv*G + g_local, query position i_q).
// All warps share the staged KV tile of head h_kv (read once), each carries its own row's running
// (m,l) + register-resident O over the keys [split*chunk, min((split+1)*chunk, N_k)). Same
// single-pass online-softmax + O-rescale as v4/v6/v7; the KV reads gather through the block table.
// A split entirely in the causal future (j_start > i_q + q_offset) does no work and writes
// (m=-inf, l=0, O=0); merge weights it out.
//
// DIFF vs v7's paged_partial_kernel (paged_attention.cu:67-175): only the prologue/epilogue index
// math + the gather head change. The cooperative load and the score/softmax loop are byte-identical;
// the two token-level changes are marked [v8] below (gather uses H_kv,h_kv; causal uses i_q).
// ---------------------------------------------------------------------------------------
template <int TN, int D>
__global__ void gqa_partial_kernel(const __half* __restrict__ Q,        // [B,H_q,N_q,D] dense
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

    __shared__ float sK[TN * D];     // current key-block   [TN][D], reloaded per block (FP32 in smem)
    __shared__ float sV[TN * D];     // current value-block [TN][D], reloaded per block

    const int warp  = threadIdx.x >> 5;            // 0..kWarps-1 : which packed row in this tile
    const int lane  = threadIdx.x & (kWarp - 1);   // 0..31       : which slice of the row
    const int m_row = blockIdx.x * kWarps + warp;  // [v8] global PACKED-M row (was query row gi)
    const int split = blockIdx.y;                  // which KV split this block covers
    const int64_t bh_kv = blockIdx.z;              // [v8] (batch * KV head) slice (was batch*head)
    const int b     = (int)(bh_kv / H_kv);         // batch index (block table is per-sequence)
    const int h_kv  = (int)(bh_kv % H_kv);         // [v8] KV head — selects the head slice in a page
    const int M     = G * N_q;                     // [v8] packed-M extent
    const bool active = (m_row < M);               // surplus warps idle (G need not divide kWarps)

    // [v8] Decode the packed row into (query head within group, query position) and the GLOBAL query
    // head. Row-major M = [G, N_q] (head-major): g_local outer so each warp's Q load is one row and
    // the workspace write lands at the natural query-head index. Decode (N_q=1): g_local=m_row, i_q=0.
    const int g_local = active ? (m_row / N_q) : 0;
    const int i_q     = active ? (m_row % N_q) : 0;
    const int h_q     = h_kv * G + g_local;        // [v8] GLOBAL query head this warp owns

    // This split's key range [j_start, j_end). j_end clamps the last split to N_k.
    const int j_start = split * chunk;
    const int j_end   = (j_start + chunk < N_k) ? (j_start + chunk) : N_k;

    // [v8] Q is dense [B,H_q,N_q,D]; each warp reads ITS OWN query-head row (the base pointer is
    // per-warp now, since the G warps own different heads — in v7 all warps shared one head's base).
    const __half* Qrow = Q + (((int64_t)b * H_q + h_q) * N_q + i_q) * D;
    const int* bt_b    = block_table + (int64_t)b * n_logical;   // this sequence's block table row

    // Load this warp's query-row slice into registers (FP16 -> FP32); lane owns elems {lane, lane+32,..}.
    float q_reg[EPT], o_reg[EPT];
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        int t = lane + kWarp * e;
        q_reg[e] = active ? __half2float(Qrow[t]) : 0.f;
        o_reg[e] = 0.f;
    }

    // Running online-softmax stats for this row (within this split). Empty state wiped by first score.
    float m_cur = -FLT_MAX, l_cur = 0.f;

    for (int j0 = j_start; j0 < j_end; j0 += TN) {
        // Cooperative load of this key/value block by the WHOLE block (all warps participate so the
        // __syncthreads below is reached uniformly). Each logical key gj is GATHERED through the block
        // table: lb -> physical page pb -> pool row (pb*page_size + off), head slice h_kv. FP16 global
        // read -> FP32 smem. [v8] the only change here vs v7 is the pool head count/index (H_kv, h_kv).
        for (int idx = threadIdx.x; idx < TN * D; idx += kBlock) {
            int r = idx / D, t = idx % D, gj = j0 + r;
            bool ok = gj < j_end;                  // j_end <= N_k, so this also bounds N_k
            if (ok) {
                int pb  = bt_b[gj / page_size];    // physical block for this logical position
                int off = gj % page_size;          // offset within the page
                int64_t src = ((int64_t)(pb * page_size + off) * H_kv + h_kv) * D + t;  // [v8] H_kv,h_kv
                sK[idx] = __half2float(K_pool[src]);
                sV[idx] = __half2float(V_pool[src]);
            } else {
                sK[idx] = 0.f;
                sV[idx] = 0.f;
            }
        }
        __syncthreads();   // sK, sV visible to every warp before any compute

        if (active) {
            for (int c = 0; c < TN; ++c) {
                int gj = j0 + c;
                if (gj >= j_end) break;                    // padding tail of the last key-block
                if (causal && gj > i_q + q_offset) break;  // [v8] mask uses i_q (the query position),
                                                           // NOT m_row — packed rows share a position.

                // 32-lane dot product q_row . k_c: each lane sums its EPT owned products, then the
                // butterfly reduce broadcasts the full score s to all lanes (warp-uniform).
                float partial = 0.f;
#pragma unroll
                for (int e = 0; e < EPT; ++e)
                    partial += q_reg[e] * sK[c * D + lane + kWarp * e];
                float s = warp_reduce_sum(partial) * scale;

                // Online update with the O-rescale (identical to v4/v6/v7): shift baseline to s on a
                // new max, correcting BOTH the running l and partial O by exp(m_old - m_new) in (0,1].
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

    // Write this split's UNNORMALIZED partial (no /l) to the workspace, indexed by the QUERY head.
    // The G active warps write distinct h_q slices -> no collisions. The merge kernel combines.
    if (active) {
        int64_t bh_q = (int64_t)b * H_q + h_q;                          // [v8] query-head flat index
        int64_t ml = ((bh_q * N_q) + i_q) * num_splits + split;         // index into [B,H_q,N_q,S]
        int64_t ob = ml * D;                                            // index into [B,H_q,N_q,S,D]
#pragma unroll
        for (int e = 0; e < EPT; ++e)
            O_partial[ob + lane + kWarp * e] = o_reg[e];
        if (lane == 0) {   // m_cur, l_cur are warp-uniform (s is); one lane writes the scalars
            m_partial[ml] = m_cur;
            l_partial[ml] = l_cur;
        }
    }
}

// ---------------------------------------------------------------------------------------
// MERGE kernel. Grid = (N_q, B*H_q), blockDim = D. Thread t owns output dim t of one (bh_q, query-row).
// Recombines the num_splits partials with the LSE combine. IDENTICAL to v7/v6 (the merge sees only
// the [B,H_q,N_q,S,*] workspace, so GQA never reaches it — by here every packed row is already an
// independent query-head output). Masked/empty splits carry m=-inf, l=0 -> exp(m_s - m)=0.
// ---------------------------------------------------------------------------------------
__global__ void gqa_merge_kernel(const float* __restrict__ O_partial,
                                 const float* __restrict__ m_partial,
                                 const float* __restrict__ l_partial,
                                 float* __restrict__ O,
                                 int N_q, int num_splits, int D) {
    const int i   = blockIdx.x;              // query row
    const int64_t bh = blockIdx.y;           // (batch * query head)
    const int t   = threadIdx.x;             // output dim (< D)
    const int64_t ml0 = (((int64_t)bh * N_q) + i) * num_splits;   // base index into [B,H_q,N_q,S]

    // Global running max across splits, then the LSE denom.
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

// Launch the GQA partial kernel for a fixed compile-time tile config (chosen by head dim).
template <int TN, int D>
void launch_partial(const __half* q, const __half* k_pool, const __half* v_pool,
                    const int* block_table,
                    float* O_partial, float* m_partial, float* l_partial,
                    int64_t BH_kv, int H_q, int H_kv, int G, int N_q, int N_k,
                    int num_splits, int chunk,
                    int page_size, int n_logical, int q_offset,
                    float scale, bool causal, cudaStream_t stream) {
    const int M = G * N_q;                                  // packed rows; tiled over grid.x by kWarps
    dim3 grid(ceil_div(M, kWarps), (unsigned)num_splits, (unsigned)BH_kv);
    gqa_partial_kernel<TN, D><<<grid, kBlock, 0, stream>>>(
        q, k_pool, v_pool, block_table, O_partial, m_partial, l_partial,
        H_q, H_kv, G, N_q, N_k, num_splits, chunk, page_size, n_logical, q_offset, scale, causal);
}

// Pick num_splits to fill the grid without shrinking chunks below a floor. Carried from v7
// (paged_attention.cu:233-244) with one change: base_blocks uses B*H_kv (the z-extent is now KV
// heads, G× fewer) and row_tiles covers M=G*N_q. PREFILL (large N_q) -> base_blocks >> target -> 1
// split -> reduces to plain attention (square-shape tests). DECODE (N_q=1) -> M=G small -> add splits.
// NOTE (v7-measured): this self-disabling is grid-sizing only; it does NOT make decode bandwidth-bound
// (v7's --batch sweep stayed flat at ~10% HBM, BH=8->512). The v8 win is the G compute-warps + KV-read-
// once per CTA, not occupancy. With G query heads now packed per block, split-KV self-disables at G×
// larger batch than v7 — fine, because batch never moved %HBM anyway.
int choose_splits(int64_t B, int H_kv, int G, int N_q, int N_k, int num_sm = 40) {
    if (N_k <= 0) return 1;
    const int   S_CAP        = 32;     // also the merge kernel's "small num_splits" assumption
    const int   MIN_CHUNK    = 256;    // don't split KV into chunks smaller than this
    const int   target_blocks = 2 * num_sm;
    const int   row_tiles    = ceil_div((int64_t)G * N_q, kWarps);   // packed-M row tiles
    const int64_t base_blocks = (int64_t)B * H_kv * (int64_t)row_tiles;
    int by_occ  = ceil_div(target_blocks, std::max<int64_t>(base_blocks, 1));  // splits to fill SMs
    int by_size = ceil_div(N_k, MIN_CHUNK);                                    // splits the KV allows
    int s = std::min(by_occ, by_size);
    return std::min(std::max(s, 1), S_CAP);
}

}  // anonymous namespace

// Host entry: paged GQA M-packed split-KV decode in two passes (partial + merge) behind one `forward`.
// FP16-in / FP32-accum / FP32-out; the public contract stays FP32-in (tests/bench pass fp32), so we
// cast q, k_pool, v_pool to half here, matching v5/v6/v7. K/V are PHYSICAL pools
// [num_blocks, page_size, H_kv, d]; block_table is int32 [B, n_logical]. The GQA group factor
// G = H_q / H_kv is DERIVED from the shapes (no extra argument). q_offset places the decode query in
// the logical sequence for causal masking. Exposed to Python by binding.cpp.
torch::Tensor gqa_attention_forward(torch::Tensor q, torch::Tensor k_pool, torch::Tensor v_pool,
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
    // True logical KV length is passed explicitly (NOT n_logical*page_size) so a non-multiple N_k
    // never scans the last page's padding tail: the kernel loops gj < N_k.
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

    // FP16-in: cast (no-op if already half). Accumulation + output are FP32.
    auto qh = q.to(torch::kHalf);
    auto kh = k_pool.to(torch::kHalf);
    auto vh = v_pool.to(torch::kHalf);
    auto O  = torch::empty({B, H_q, N_q, d}, q.options().dtype(torch::kFloat32));  // query-head shaped

    const int64_t BH_kv = (int64_t)B * H_kv;
    const int S = choose_splits(B, H_kv, G, N_q, N_k);
    const int chunk = ceil_div(N_k, S);

    // FP32 workspace: per-(query-head, query-row, split) unnormalized partial output + its (m, l).
    // Laid out [B,H_q,N_q,S,*] (QUERY-head shaped — each packed row is an independent query head).
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

    // Tile config per head dim so the staged K/V tiles (2*TN*D floats) sit under the 48 KB smem budget
    // (same split as v2/v3/v4/v6/v7: 64x64 @ d=64, 32x32 @ d=128).
    if (d == 64) {
        launch_partial<64, 64>(qp, kp, vp, btp, Op, Mp, Lp, BH_kv, H_q, H_kv, G, N_q, N_k, S, chunk,
                               (int)page_size, (int)n_logical, (int)q_offset, (float)scale, causal, stream);
    } else if (d == 128) {
        launch_partial<32, 128>(qp, kp, vp, btp, Op, Mp, Lp, BH_kv, H_q, H_kv, G, N_q, N_k, S, chunk,
                                (int)page_size, (int)n_logical, (int)q_offset, (float)scale, causal, stream);
    } else {
        TORCH_CHECK(false, "v8 gqa supports head_dim 64 or 128 (got ", d,
                    "); the tile config is specialized per head dim to fit the 48 KB smem budget");
    }

    // Merge: one block per (query-row, batch-query-head); d threads each own one output dim.
    dim3 mgrid((unsigned)N_q, (unsigned)(B * H_q));
    gqa_merge_kernel<<<mgrid, d, 0, stream>>>(Op, Mp, Lp, O.data_ptr<float>(), N_q, S, d);

    return O;
}

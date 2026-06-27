// v7 paged-KV decode — v6 split-KV (Flash-Decoding) with a BLOCK-TABLE gather, the plumbing a
// from-scratch mini-vLLM needs. The decode algorithm is byte-identical to v6 (split-KV partial +
// LSE merge, FP16-in/FP32-accum, no tensor cores at N_q=1); v7 changes exactly ONE thing in the hot
// loop and adds one harness-facing knob:
//
//   (a) PAGED KV. v6 read a CONTIGUOUS per-(batch,head) KV slice: Kbh = K + bh*N_k*D, then
//       Kbh[gj*D + t]. A real KV cache is not contiguous — it's a pool of fixed-size pages plus a
//       per-sequence block table mapping logical token positions to physical pages (vLLM's layout).
//       v7 stores K/V as physical pools [num_blocks, page_size, H, d] and gathers each logical key j
//       through block_table[b][j / page_size]:
//           lb = j / page_size;  pb = block_table[b*n_logical + lb];  off = j % page_size;
//           src = ((pb*page_size + off)*H + h)*D + t;        // index the pool
//       The block table is per-SEQUENCE (per batch b), shared across all H heads of that sequence —
//       so the kernel derives b = bh/H, h = bh%H. Splits still chunk LOGICAL positions [j_start,
//       j_end); a chunk may straddle pages — fine, every key is gathered independently. Roofline is
//       UNCHANGED: AI = 2/b = 1.0 (FP16). The gather adds O(N_k/page_size) int32 index reads (~0.1%
//       of the 2*N_k*d*b KV bytes) — byte-neutral and occupancy-neutral by construction. v7 is
//       deliberately NOT an attack on the limiter; it sets up v8 (GQA M-packing, the occupancy lever).
//
//   (b) CAUSAL QUERY-OFFSET. v6 masked keys j > i with the query row i; at decode i=0, so causal
//       attended only key 0 (degenerate — SDPA short-circuits it, so the v6 causal-decode bench rows
//       were meaningless). v7 adds q_offset so the mask is j > i + q_offset: placing the single
//       decode query at logical position N_k-1 (q_offset = N_k - N_q) lets it attend the WHOLE cache,
//       so causal decode == the non-causal full-cache scan and the timing becomes meaningful.
//       q_offset = 0 reproduces v6 exactly.
//
// Everything else — choose_splits, the online-softmax core, the cross-split LSE merge, the
// [B,H,N_q,S,*] workspace, the 64x64 @ d=64 / 32x32 @ d=128 tiles — is carried from v6 unchanged.
//
// Layout: Q is dense [B,H,N_q,d] row-major. K_pool/V_pool are [num_blocks, page_size, H, d]
// row-major. block_table is int32 [B, n_logical]. Causal excludes keys j > (query i + q_offset).

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
constexpr int kWarps = kBlock / kWarp;  // 8 warps per block -> 8 query rows per block

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
// PARTIAL kernel. Grid = (ceil_div(N_q, kWarps), num_splits, B*H). One block owns kWarps consecutive
// query rows of one (batch,head) for ONE key split; warp `w` owns query row i0+w and carries that
// row's running (m, l) + register-resident O across only the keys [split*chunk, min((split+1)*chunk,
// N_k)). Same single-pass online-softmax + O-rescale as v4/v6, but the KV reads now GATHER through the
// block table (paged pool) instead of a contiguous per-(b,h) slice, and the causal mask uses a query
// offset. A split entirely in the causal future (j_start > gi + q_offset) does no work and writes
// (m=-inf, l=0, O=0); merge weights it out.
// ---------------------------------------------------------------------------------------
template <int TN, int D>
__global__ void paged_partial_kernel(const __half* __restrict__ Q,
                                     const __half* __restrict__ K_pool,   // [num_blocks,page_size,H,D]
                                     const __half* __restrict__ V_pool,   // [num_blocks,page_size,H,D]
                                     const int* __restrict__ block_table, // [B, n_logical]
                                     float* __restrict__ O_partial,   // [B,H,N_q,S,D]
                                     float* __restrict__ m_partial,   // [B,H,N_q,S]
                                     float* __restrict__ l_partial,   // [B,H,N_q,S]
                                     int H, int N_q, int N_k, int num_splits, int chunk,
                                     int page_size, int n_logical, int q_offset,
                                     float scale, bool causal) {
    static_assert(D % kWarp == 0, "head_dim must be a multiple of the warp size (32)");
    constexpr int EPT = D / kWarp;   // elements of the row each lane owns (d=64->2, d=128->4)

    __shared__ float sK[TN * D];     // current key-block   [TN][D], reloaded per block (FP32 in smem)
    __shared__ float sV[TN * D];     // current value-block [TN][D], reloaded per block

    const int warp  = threadIdx.x >> 5;            // 0..kWarps-1 : which query row in this tile
    const int lane  = threadIdx.x & (kWarp - 1);   // 0..31       : which slice of the row
    const int i0    = blockIdx.x * kWarps;         // first query row of this tile
    const int gi    = i0 + warp;                   // global query row this warp owns
    const int split = blockIdx.y;                  // which KV split this block covers
    const int64_t bh = blockIdx.z;                 // (batch*head) slice
    const int bdx   = (int)(bh / H);               // batch index  (block table is per-sequence)
    const int hdx   = (int)(bh % H);               // head index   (selects the head slice in a page)
    const bool active = (gi < N_q);                // last row-tile may have idle warps

    // This split's key range [j_start, j_end). j_end clamps the last split to N_k.
    const int j_start = split * chunk;
    const int j_end   = (j_start + chunk < N_k) ? (j_start + chunk) : N_k;

    const __half* Qbh = Q + bh * (int64_t)N_q * D;       // Q stays dense [B,H,N_q,D]
    const int* bt_b   = block_table + (int64_t)bdx * n_logical;   // this sequence's block table row

    // Load this warp's query-row slice into registers (FP16 -> FP32); lane owns elems {lane, lane+32,..}.
    float q_reg[EPT], o_reg[EPT];
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        int t = lane + kWarp * e;
        q_reg[e] = active ? __half2float(Qbh[(int64_t)gi * D + t]) : 0.f;
        o_reg[e] = 0.f;
    }

    // Running online-softmax stats for this row (within this split). Empty state wiped by first score.
    float m_cur = -FLT_MAX, l_cur = 0.f;

    for (int j0 = j_start; j0 < j_end; j0 += TN) {
        // Cooperative load of this key/value block by the WHOLE block (all warps participate so the
        // __syncthreads below is reached uniformly). Each logical key gj is GATHERED through the block
        // table: lb -> physical page pb -> pool row (pb*page_size + off), head slice hdx. FP16 global
        // read -> FP32 smem.
        for (int idx = threadIdx.x; idx < TN * D; idx += kBlock) {
            int r = idx / D, t = idx % D, gj = j0 + r;
            bool ok = gj < j_end;                  // j_end <= N_k, so this also bounds N_k
            if (ok) {
                int pb  = bt_b[gj / page_size];    // physical block for this logical position
                int off = gj % page_size;          // offset within the page
                int64_t src = ((int64_t)(pb * page_size + off) * H + hdx) * D + t;
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
                if (causal && gj > gi + q_offset) break;   // keys ascend, so all further c are masked

                // 32-lane dot product q_row . k_c: each lane sums its EPT owned products, then the
                // butterfly reduce broadcasts the full score s to all lanes (warp-uniform).
                float partial = 0.f;
#pragma unroll
                for (int e = 0; e < EPT; ++e)
                    partial += q_reg[e] * sK[c * D + lane + kWarp * e];
                float s = warp_reduce_sum(partial) * scale;

                // Online update with the O-rescale (identical to v4/v6): shift baseline to s on a new
                // max, correcting BOTH the running l and partial O by exp(m_old - m_new) in (0,1].
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

    // Write this split's UNNORMALIZED partial (no /l) to the workspace. The merge kernel combines.
    if (active) {
        int64_t ml = (((int64_t)bh * N_q) + gi) * num_splits + split;   // index into [B,H,N_q,S]
        int64_t ob = ml * D;                                            // index into [B,H,N_q,S,D]
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
// MERGE kernel. Grid = (N_q, B*H), blockDim = D. Thread t owns output dim t of one (bh, query-row).
// Recombines the num_splits partials with the LSE combine. Identical to v6 (the merge sees only the
// workspace, so paging never reaches it). Masked/empty splits carry m=-inf, l=0 -> exp(m_s - m)=0.
// ---------------------------------------------------------------------------------------
__global__ void paged_merge_kernel(const float* __restrict__ O_partial,
                                   const float* __restrict__ m_partial,
                                   const float* __restrict__ l_partial,
                                   float* __restrict__ O,
                                   int N_q, int num_splits, int D) {
    const int i   = blockIdx.x;              // query row
    const int64_t bh = blockIdx.y;           // (batch*head)
    const int t   = threadIdx.x;             // output dim (< D)
    const int64_t ml0 = (((int64_t)bh * N_q) + i) * num_splits;   // base index into [B,H,N_q,S]

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

// Launch the paged partial kernel for a fixed compile-time tile config (chosen by head dim).
template <int TN, int D>
void launch_partial(const __half* q, const __half* k_pool, const __half* v_pool,
                    const int* block_table,
                    float* O_partial, float* m_partial, float* l_partial,
                    int64_t BH, int H, int N_q, int N_k, int num_splits, int chunk,
                    int page_size, int n_logical, int q_offset,
                    float scale, bool causal, cudaStream_t stream) {
    dim3 grid(ceil_div(N_q, kWarps), (unsigned)num_splits, (unsigned)BH);
    paged_partial_kernel<TN, D><<<grid, kBlock, 0, stream>>>(
        q, k_pool, v_pool, block_table, O_partial, m_partial, l_partial,
        H, N_q, N_k, num_splits, chunk, page_size, n_logical, q_offset, scale, causal);
}

// Pick num_splits to fill the grid without shrinking chunks below a floor.
// IDENTICAL to v6: at num_splits=1 the grid is (row_tiles, 1, BH) = base_blocks; we raise splits only
// until the total block count reaches ~2x the SM count. PREFILL (large N_q) already has base_blocks >>
// target -> 1 split, so v7 reduces to plain attention there too (square-shape tests). At DECODE
// (N_q=1, small BH) base_blocks is tiny, so we add splits. This self-disabling at BH >= 2*num_sm is
// still the correct grid-sizing logic -- but the v7 --batch sweep (BH=8->512, N_k=8192, T4) REFUTED the
// occupancy->bandwidth crossover it was meant to enable: %HBM stayed FLAT at 9.4-12.4% across the whole
// range (no climb even at 12.8 blocks/SM). Filling the grid does NOT make the kernel bandwidth-bound --
// it stays per-CTA-bound at EVERY batch size: sK+sV=32 KB/block caps residency at 2 blocks/SM (T4
// 64 KB/SM) regardless of grid, and at N_q=1 only 1 of 8 warps computes. So the "occupancy before bytes"
// reorder (GQA M-packing -> v8) now rests on per-CTA efficiency (GEMV->GEMM lights up G warps), NOT on
// "filling the SMs." B300 retune = num_sm 40 -> 160.
int choose_splits(int64_t BH, int N_q, int N_k, int num_sm = 40) {
    if (N_k <= 0) return 1;
    const int   S_CAP        = 32;     // also the merge kernel's "small num_splits" assumption
    const int   MIN_CHUNK    = 256;    // don't split KV into chunks smaller than this
    const int   target_blocks = 2 * num_sm;
    const int   row_tiles    = ceil_div(N_q, kWarps);
    const int64_t base_blocks = BH * (int64_t)row_tiles;
    int by_occ  = ceil_div(target_blocks, std::max<int64_t>(base_blocks, 1));  // splits to fill SMs
    int by_size = ceil_div(N_k, MIN_CHUNK);                                    // splits the KV allows
    int s = std::min(by_occ, by_size);
    return std::min(std::max(s, 1), S_CAP);
}

}  // anonymous namespace

// Host entry: paged split-KV decode in two passes (partial + merge) behind one `forward`. FP16-in /
// FP32-accum / FP32-out; the public contract stays FP32-in (tests/bench pass fp32), so we cast q,
// k_pool, v_pool to half here, matching v5/v6. K/V are PHYSICAL pools [num_blocks, page_size, H, d];
// block_table is int32 [B, n_logical] mapping logical blocks to physical ones. q_offset places the
// decode query in the logical sequence for causal masking. Exposed to Python by binding.cpp.
torch::Tensor paged_attention_forward(torch::Tensor q, torch::Tensor k_pool, torch::Tensor v_pool,
                                      torch::Tensor block_table, int64_t page_size, int64_t n_k,
                                      double scale, bool causal, int64_t q_offset) {
    TORCH_CHECK(q.is_cuda() && k_pool.is_cuda() && v_pool.is_cuda() && block_table.is_cuda(),
                "q, k_pool, v_pool, block_table must be CUDA tensors");
    TORCH_CHECK(q.dim() == 4, "q must be [B,H,N_q,d]");
    TORCH_CHECK(k_pool.dim() == 4 && v_pool.dim() == 4,
                "k_pool, v_pool must be [num_blocks, page_size, H, d]");
    TORCH_CHECK(block_table.dim() == 2, "block_table must be [B, n_logical]");
    TORCH_CHECK(block_table.scalar_type() == torch::kInt32, "block_table must be int32");
    q = q.contiguous(); k_pool = k_pool.contiguous(); v_pool = v_pool.contiguous();
    block_table = block_table.contiguous();

    const int B = q.size(0), H = q.size(1), N_q = q.size(2), d = q.size(3);
    // True logical KV length is passed explicitly (NOT n_logical*page_size) so a non-multiple N_k
    // never scans the last page's padding tail: the kernel loops gj < N_k.
    const int N_k = (int)n_k;
    const int n_logical = (int)block_table.size(1);
    TORCH_CHECK(k_pool.size(1) == page_size && v_pool.size(1) == page_size,
                "k_pool/v_pool page_size dim must equal page_size argument");
    TORCH_CHECK(k_pool.size(2) == H && v_pool.size(2) == H, "pool head count must equal q's H");
    TORCH_CHECK(k_pool.size(3) == d && v_pool.size(3) == d, "pool head_dim must equal q's d");
    TORCH_CHECK(block_table.size(0) == B, "block_table must have B rows");
    TORCH_CHECK((int64_t)n_logical * page_size >= N_k,
                "block_table n_logical*page_size must cover N_k logical positions");

    // FP16-in: cast (no-op if already half). Accumulation + output are FP32.
    auto qh = q.to(torch::kHalf);
    auto kh = k_pool.to(torch::kHalf);
    auto vh = v_pool.to(torch::kHalf);
    auto O  = torch::empty({B, H, N_q, d}, q.options().dtype(torch::kFloat32));

    const int64_t BH = (int64_t)B * H;
    const int S = choose_splits(BH, N_q, N_k);
    const int chunk = ceil_div(N_k, S);

    // FP32 workspace: per-(bh, query-row, split) unnormalized partial output + its (m, l). Laid out
    // [B,H,N_q,S,*] exactly as v6 (the workspace was already paging-ready). Freed at function return.
    auto opts_f = q.options().dtype(torch::kFloat32);
    auto O_partial = torch::empty({B, H, N_q, S, d}, opts_f);
    auto m_partial = torch::empty({B, H, N_q, S},    opts_f);
    auto l_partial = torch::empty({B, H, N_q, S},    opts_f);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const __half* qp = reinterpret_cast<const __half*>(qh.data_ptr<at::Half>());
    const __half* kp = reinterpret_cast<const __half*>(kh.data_ptr<at::Half>());
    const __half* vp = reinterpret_cast<const __half*>(vh.data_ptr<at::Half>());
    const int* btp = block_table.data_ptr<int>();
    float* Op = O_partial.data_ptr<float>();
    float* Mp = m_partial.data_ptr<float>();
    float* Lp = l_partial.data_ptr<float>();

    // Tile config per head dim so the staged K/V tiles (2*TN*D floats) sit under the 48 KB smem budget
    // (same split as v2/v3/v4/v6: 64x64 @ d=64, 32x32 @ d=128).
    if (d == 64) {
        launch_partial<64, 64>(qp, kp, vp, btp, Op, Mp, Lp, BH, H, N_q, N_k, S, chunk,
                               (int)page_size, (int)n_logical, (int)q_offset, (float)scale, causal, stream);
    } else if (d == 128) {
        launch_partial<32, 128>(qp, kp, vp, btp, Op, Mp, Lp, BH, H, N_q, N_k, S, chunk,
                                (int)page_size, (int)n_logical, (int)q_offset, (float)scale, causal, stream);
    } else {
        TORCH_CHECK(false, "v7 paged supports head_dim 64 or 128 (got ", d,
                    "); the tile config is specialized per head dim to fit the 48 KB smem budget");
    }

    // Merge: one block per (query-row, batch-head); D threads each own one output dim.
    dim3 mgrid((unsigned)N_q, (unsigned)BH);
    paged_merge_kernel<<<mgrid, d, 0, stream>>>(Op, Mp, Lp, O.data_ptr<float>(), N_q, S, d);

    return O;
}

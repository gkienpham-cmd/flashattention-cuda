// v6 split-KV decode (Flash-Decoding) — the OCCUPANCY fix for the decode regime (N_q ~ 1).
//
// The v1->v5 curve is a PREFILL story: every kernel parallelizes over query rows, so the grid is
// (ceil_div(N_q, rows), B*H). At DECODE the query length collapses to N_q = 1, so that grid becomes
// (1, B*H) blocks — a handful of CTAs that leave the T4's 40 SMs (B300's 160) almost idle while a
// single block streams the whole KV cache. The kernels are shape-legal at N_q=1 (v4/v5 take separate
// N_q/N_k), but they are OCCUPANCY-STARVED, not bandwidth/compute bound. This is exactly the decode
// trap FA4 never closes (docs/b300-decode-research.md, blind-spot #2: grid (1,heads,batch) -> SM
// starvation), and the keystone of the decode arc v6->v11 in that research (split-KV is "v6
// foundation", rank #1: high payoff, low risk, no B300 needed).
//
// THE FIX — split the KV axis across blocks (Flash-Decoding). Partition the N_k keys into `num_splits`
// contiguous chunks; one block owns (query-row tile, split, batch-head). Each block runs v4's
// single-pass online softmax over ONLY its key chunk and writes an UNNORMALIZED partial:
//     O_partial[bh, i, split, :] = sum_{j in chunk} p_j * V_j   (no /l yet)
//     m_partial[bh, i, split]    = local row-max over the chunk
//     l_partial[bh, i, split]    = local denom  sum_{j in chunk} p_j
// A second MERGE kernel then recombines the `num_splits` partials per (bh, i) with the log-sum-exp
// (LSE) combine — the same algebra online softmax already uses, applied across splits instead of
// across keys:
//     m = max_s m_s ;  l = sum_s exp(m_s - m) * l_s ;  O = sum_s exp(m_s - m) * O_partial_s ;  O /= l
// At N_q=1 this turns a 1-block-per-head loop into num_splits blocks per head, filling the SMs. On T4
// the merge is a SEPARATE kernel; the research's on-chip 2-CTA-cluster + DSMEM merge is a Blackwell
// feature, deferred. Contiguous KV only here — paged/block-table gather is v7.
//
// PRECISION: FP16-in / FP32-accum (research §8 tags v6 "FP16"), tol ~2e-2 like v5. The public API
// stays FP32-in: the host entry casts q,k,v to half, so the kernel reads 2-byte KV from HBM (the decode
// bandwidth that matters) but stages to smem as float and accumulates the long online reduction in
// FP32. This pins the decode roofline at b=2 bytes -> AI = 2/b = 1.0 (research §4); FP8/NVFP4 KV (b=1,
// b~0.56) are the v8/v9 levers that cut it further. NO TENSOR CORES: at N_q=1 the matmuls are M=1
// (GEMV-shaped), so v6 uses v4's per-key warp-shuffle dot on the CUDA cores, not WMMA — tensor cores
// would idle on a 1-row tile.
//
// Smem budget: only the staged K/V tiles (float) = 2*TN*D floats, same as v4:
//   d=64  -> TN=64  (32 KB) ; d=128 -> TN=32 (32 KB)  — well under the T4's 48 KB. Q/O in registers.
//
// Layout: q,k,v are [B,H,N,d] row-major. Causal excludes keys j > query i.

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
// N_k)). Same single-pass online-softmax + O-rescale as v4, but (a) the key loop is bounded to the
// split, and (b) at the end it writes the UNNORMALIZED (O, m, l) to the workspace instead of
// normalizing — the merge kernel does the cross-split LSE combine. A split entirely in the causal
// future (split*chunk > gi) does no work and writes (m=-inf, l=0, O=0); merge weights it out.
// ---------------------------------------------------------------------------------------
template <int TN, int D>
__global__ void splitkv_partial_kernel(const __half* __restrict__ Q,
                                       const __half* __restrict__ K,
                                       const __half* __restrict__ V,
                                       float* __restrict__ O_partial,   // [B,H,N_q,S,D]
                                       float* __restrict__ m_partial,   // [B,H,N_q,S]
                                       float* __restrict__ l_partial,   // [B,H,N_q,S]
                                       int N_q, int N_k, int num_splits, int chunk,
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
    const bool active = (gi < N_q);                // last row-tile may have idle warps

    // This split's key range [j_start, j_end). j_end clamps the last split to N_k.
    const int j_start = split * chunk;
    const int j_end   = (j_start + chunk < N_k) ? (j_start + chunk) : N_k;

    const __half* Qbh = Q + bh * (int64_t)N_q * D;
    const __half* Kbh = K + bh * (int64_t)N_k * D;
    const __half* Vbh = V + bh * (int64_t)N_k * D;

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
        // __syncthreads below is reached uniformly). FP16 global read -> FP32 smem. Coalesced in t.
        for (int idx = threadIdx.x; idx < TN * D; idx += kBlock) {
            int r = idx / D, t = idx % D, gj = j0 + r;
            bool ok = gj < j_end;                  // j_end <= N_k, so this also bounds N_k
            sK[idx] = ok ? __half2float(Kbh[(int64_t)gj * D + t]) : 0.f;
            sV[idx] = ok ? __half2float(Vbh[(int64_t)gj * D + t]) : 0.f;
        }
        __syncthreads();   // sK, sV visible to every warp before any compute

        if (active) {
            for (int c = 0; c < TN; ++c) {
                int gj = j0 + c;
                if (gj >= j_end) break;          // padding tail of the last key-block in this split
                if (causal && gj > gi) break;    // keys ascend, so all further c are masked too

                // 32-lane dot product q_row . k_c: each lane sums its EPT owned products, then the
                // butterfly reduce broadcasts the full score s to all lanes (warp-uniform).
                float partial = 0.f;
#pragma unroll
                for (int e = 0; e < EPT; ++e)
                    partial += q_reg[e] * sK[c * D + lane + kWarp * e];
                float s = warp_reduce_sum(partial) * scale;

                // Online update with the O-rescale (identical to v4): shift baseline to s on a new max,
                // correcting BOTH the running l and partial O by exp(m_old - m_new) in (0,1].
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
// Recombines the num_splits partials with the LSE combine. m and l are scalars shared by all D
// threads; with num_splits <= S_CAP (32) each thread just recomputes them (cheap) rather than paying a
// block reduction. Masked/empty splits carry m=-inf, l=0 -> exp(m_s - m)=0, so they drop out.
// ---------------------------------------------------------------------------------------
__global__ void splitkv_merge_kernel(const float* __restrict__ O_partial,
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

// Launch the split-KV partial kernel for a fixed compile-time tile config (chosen by head dim).
template <int TN, int D>
void launch_partial(const __half* q, const __half* k, const __half* v,
                    float* O_partial, float* m_partial, float* l_partial,
                    int64_t BH, int N_q, int N_k, int num_splits, int chunk,
                    float scale, bool causal, cudaStream_t stream) {
    dim3 grid(ceil_div(N_q, kWarps), (unsigned)num_splits, (unsigned)BH);
    splitkv_partial_kernel<TN, D><<<grid, kBlock, 0, stream>>>(
        q, k, v, O_partial, m_partial, l_partial, N_q, N_k, num_splits, chunk, scale, causal);
}

// Pick num_splits to fill the SMs (research blind-spot #2) without shrinking chunks below a floor.
// At num_splits=1 the grid is (row_tiles, 1, BH) = base_blocks; we raise splits only until the total
// block count reaches ~2x the SM count. PREFILL (large N_q) already has base_blocks >> target -> 1
// split, so v6 reduces to plain attention there (this is what makes the square-shape tests pass). At
// DECODE (N_q=1, small BH) base_blocks is tiny, so we add splits. B300 retune = num_sm 40 -> 160.
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

// Host entry: split-KV decode in two passes (partial + merge) behind one `forward`. FP16-in /
// FP32-accum / FP32-out; the public contract stays FP32-in (every test/bench passes fp32), so we cast
// q,k,v to half here, matching v5. The (O_partial, m, l) workspace is FP32 scratch, freed on return.
// Exposed to Python by binding.cpp as `forward`.
torch::Tensor splitkv_attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                        double scale, bool causal) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q,k,v must be CUDA tensors");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "expected [B,H,N,d] tensors");
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous();

    const int B = q.size(0), H = q.size(1), N_q = q.size(2), d = q.size(3);
    const int N_k = k.size(2);
    TORCH_CHECK(k.size(3) == d && v.size(3) == d, "head_dim mismatch across q,k,v");
    TORCH_CHECK(v.size(2) == N_k, "K and V must share sequence length");

    // FP16-in: cast (no-op if already half). Accumulation + output are FP32.
    auto qh = q.to(torch::kHalf);
    auto kh = k.to(torch::kHalf);
    auto vh = v.to(torch::kHalf);
    auto O  = torch::empty({B, H, N_q, d}, q.options().dtype(torch::kFloat32));

    const int64_t BH = (int64_t)B * H;
    const int S = choose_splits(BH, N_q, N_k);
    const int chunk = ceil_div(N_k, S);

    // FP32 workspace: per-(bh, query-row, split) unnormalized partial output + its (m, l). Freed when
    // these tensors drop at function return. Laid out [B,H,N_q,S,*] so a future paged variant slots in.
    auto opts_f = q.options().dtype(torch::kFloat32);
    auto O_partial = torch::empty({B, H, N_q, S, d}, opts_f);
    auto m_partial = torch::empty({B, H, N_q, S},    opts_f);
    auto l_partial = torch::empty({B, H, N_q, S},    opts_f);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const __half* qp = reinterpret_cast<const __half*>(qh.data_ptr<at::Half>());
    const __half* kp = reinterpret_cast<const __half*>(kh.data_ptr<at::Half>());
    const __half* vp = reinterpret_cast<const __half*>(vh.data_ptr<at::Half>());
    float* Op = O_partial.data_ptr<float>();
    float* Mp = m_partial.data_ptr<float>();
    float* Lp = l_partial.data_ptr<float>();

    // Tile config per head dim so the staged K/V tiles (2*TN*D floats) sit under the 48 KB smem budget
    // (same split as v2/v3/v4: 64x64 @ d=64, 32x32 @ d=128).
    if (d == 64) {
        launch_partial<64, 64>(qp, kp, vp, Op, Mp, Lp, BH, N_q, N_k, S, chunk, (float)scale, causal, stream);
    } else if (d == 128) {
        launch_partial<32, 128>(qp, kp, vp, Op, Mp, Lp, BH, N_q, N_k, S, chunk, (float)scale, causal, stream);
    } else {
        TORCH_CHECK(false, "v6 split-KV supports head_dim 64 or 128 (got ", d,
                    "); the tile config is specialized per head dim to fit the 48 KB smem budget");
    }

    // Merge: one block per (query-row, batch-head); D threads each own one output dim.
    dim3 mgrid((unsigned)N_q, (unsigned)BH);
    splitkv_merge_kernel<<<mgrid, d, 0, stream>>>(Op, Mp, Lp, O.data_ptr<float>(), N_q, S, d);

    return O;
}

// v11 — MLA (Multi-head Latent Attention) latent-KV decode. Forked from v10 (nvfp4_attention.cu); the
// score-stationary inner loop, split-KV, paged gather, LSE merge, [B,H_q,N_q,S,*] workspace,
// choose_splits, and host are carried with one change — the attention SHAPE. v10 is GQA-over-H_kv-heads
// (G query heads share one KV head, M=G); v11 is MQA-over-ONE-shared-latent (ALL h_q query heads share
// the single latent, M=h_q). This is the SHAPE change the decode arc has been building toward.
//
//   WHY (docs/v11-kickoff.md, the v10 close-out). v10 proved confound-free across T4/B200/B300 to 2M
//   tokens that decode is PER-CTA / low-MLP latency-bound (1 active warp at N_q=1), NOT bandwidth-bound
//   — cutting KV bytes buys nothing for decode latency. The ONLY lever left is to RAISE M above 1.
//   MLA shares ONE low-rank latent across all h_q heads, so decode packs M=h_q (h_q warps active across
//   the grid, not 1) and lifts AI from 2G/b (<=8-16 GQA) to ~3.78*h_q/b (~235 at h_q=128 fp16) — the
//   first decode shape in the v1->v11 arc the PURE roofline puts near/at the FP16-TC ridge. PREDICTION
//   (results.md Step 11): pure roofline = compute-bound (a FLIP); the per-CTA-corrected real prediction
//   = the limiter RENAMES to smem-capacity (staging the latent tile caps occupancy at ~1 block/SM —
//   realized concretely here: ~46 KB smem at TN=32 vs the T4's 48 KB static limit); counter-prediction
//   = if measured %HBM stays at the ~0.5%/~40 GB/s per-CTA floor, MLA did not leave per-CTA-bound and
//   the speculative q_len>1 shape-lever is the fallback. Either sign is publishable.
//
//   THE SHAPE CHANGE (the single variable). MLA stores ONE shared latent per token of width
//   DQK = kv_lora_rank + rope_dim (DeepSeek-V3: 512 content + 64 decoupled-RoPE = 576). The absorbed
//   up-projections W^UK/W^UV fold into the OFFLINE Q/O projections (NOT this kernel), so the kernel
//   receives q_absorbed [B,h_q,N_q,DQK] already in the latent basis and computes:
//       score_h = q_absorbed_h . latent          (a DQK-wide dot — MQA over the latent)
//       O_latent_h = sum_j p_hj . latent_j[:DV]   (PV over the first DV=kv_lora_rank dims; RoPE carries
//                                                  no value, so PV ignores the last rope_dim dims)
//   The latent serves as BOTH K (all DQK) and V (first DV), so it is read ONCE for all heads and there
//   is no separate V pool (v10's two pools collapse to one — the real ~93% KV-cache reduction). W^UV is
//   applied to O_latent OFFLINE, so this kernel outputs O_latent [B,h_q,N_q,DV] in the latent basis.
//
//   THE SINGLE TRANSPOSED smem BUFFER (the load-bearing trick). v10 staged sK transposed [d][key] (for
//   the conflict-free lane=key QK read) AND sV natural [key][d] (for PV). Here ONE transposed buffer
//   sK_T[DQK][key+pad] serves BOTH: QK reads sK_T[d*(TN+1)+c] (lane=key c, stride-1, conflict-free), and
//   PV reads V[kc][d] = sK_T[d*(TN+1)+kc] for the lane's owned dims d=lane+32*e (e<DV/32; consecutive
//   lanes differ by TN+1=33 words -> distinct banks -> conflict-free). No sV. At TN=32, DQK=576:
//   sK_T = 576*33*2 = 38016 B + sQ = 8*576*2 = 9216 B ~= 46.1 KB < the T4 48 KB static smem limit.
//
//   CUDA-core / dequant-to-FP16 (default). M=h_q packs h_q warps across grid.x blocks (8 warps/block,
//   warp-per-head-row as v10) — it does NOT form one M=h_q tensor-core GEMM. The native FP4 tcgen05
//   compute arm (M>=128 one GEMM) is deferred to ONLY-IF the dev rung proves M=128 is one GEMM and the
//   limiter flips (kickoff §9 Q1/Q2). NVFP4 latent storage is carried byte-identical from v10 so the
//   v11-vs-v10 A/B isolates the SHAPE (same storage + split-KV; only GQA->MQA-over-latent changes).
//
// Layout: q_absorbed dense [B,h_q,N_q,DQK] (FP16). L_pool [num_blocks,page_size,1,DQK/2] (packed E2M1
// nibbles, uint8). L_scale [num_blocks,page_size,1,DQK/16] (E4M3 micro-scales, uint8). block_table
// int32 [B,n_logical].

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cfloat>
#include <cstdint>

namespace {

constexpr int kBlock = 256;
constexpr int kWarp  = 32;
constexpr int kWarps = kBlock / kWarp;

inline int ceil_div(int64_t n, int b) { return static_cast<int>((n + b - 1) / b); }

// The 8 E2M1 magnitudes (4-bit element: 1 sign + 3-bit magnitude index). Must match _E2M1_LEVELS in
// fa_kernels/paged.py exactly. (Carried byte-identical from v10.)
__device__ const float kE2M1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

// NVFP4 nibble -> signed E2M1 value. `hi` selects the high nibble (odd dim index) vs the low (even).
__device__ __forceinline__ float dequant_nvfp4(uint8_t byte, int hi) {
    uint8_t nib = hi ? (byte >> 4) : (byte & 0xF);
    float v = kE2M1[nib & 0x7];
    return (nib & 0x8) ? -v : v;
}

// E4M3 byte -> float (the per-16 micro-scale). Software-emulated on sm_75/T4 via cuda_fp8.h.
__device__ __forceinline__ float dequant_e4m3(uint8_t b) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)));
}

// Butterfly all-reduce SUM and MAX. Both leave the full 32-lane result in every lane; both must be
// called by ALL 32 lanes (masked keys feed the identity). Carried byte-identical from v10.
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = kWarp / 2; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}
__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int off = kWarp / 2; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

// ---------------------------------------------------------------------------------------
// PARTIAL kernel. Grid = (ceil_div(M,kWarps), num_splits, B*H_kv) with M = G*N_q and H_kv=1, G=h_q for
// MLA — so M=h_q, grid.z=B, and every query head is a packed-M row sharing the single staged latent.
// DQK = latent/score width (kv_lora_rank + rope_dim); DV = kv_lora_rank (the PV/output width). The
// score-stationary inner loop is byte-identical to v10 except: ONE latent pool (no V pool), PV reads
// the transposed sK_T (no sV), and the QK dot spans DQK while PV/output span DV.
// ---------------------------------------------------------------------------------------
template <int TN, int DQK, int DV>
__global__ void mla_partial_kernel(const __half*  __restrict__ Q,          // [B,H_q,N_q,DQK] dense FP16
                                   const uint8_t* __restrict__ L_pool,      // [.,page_size,H_kv,DQK/2] nibbles
                                   const uint8_t* __restrict__ L_scale,     // [.,page_size,H_kv,DQK/16] E4M3
                                   const int* __restrict__ block_table,     // [B, n_logical]
                                   float* __restrict__ O_partial,           // [B,H_q,N_q,S,DV]
                                   float* __restrict__ m_partial,           // [B,H_q,N_q,S]
                                   float* __restrict__ l_partial,           // [B,H_q,N_q,S]
                                   int H_q, int H_kv, int G, int N_q, int N_k,
                                   int num_splits, int chunk,
                                   int page_size, int n_logical, int q_offset,
                                   float scale, bool causal,
                                   float scale_l) {                         // per-tensor FP32 latent scale
    static_assert(DQK % kWarp == 0, "DQK (latent width) must be a multiple of the warp size (32)");
    static_assert(DV  % kWarp == 0, "DV (kv_lora_rank) must be a multiple of the warp size (32)");
    static_assert(TN  % kWarp == 0, "tile width must be a multiple of the warp size (32)");
    static_assert(DQK % 16 == 0, "DQK must be a multiple of the NVFP4 block (16)");
    static_assert(DV <= DQK, "DV (PV width) must not exceed DQK (score width)");
    constexpr int EPT_Q = DQK / kWarp;   // query/latent dims each lane stages for QK
    constexpr int EPT_O = DV  / kWarp;   // output dims each lane owns in PV (DV=512 -> 16)
    constexpr int NG    = TN  / kWarp;   // 32-key groups per staged tile (TN=32 -> 1)
    constexpr int DP    = DQK / 2;       // packed-nibble feature width (bytes per token)
    constexpr int DM    = DQK / 16;      // micro-scale feature width (E4M3 per token)

    // ONE transposed FP16 latent tile sK_T[d][key]+1-pad: serves BOTH the conflict-free lane=key QK dot
    // (over DQK) AND the conflict-free PV read (over the first DV dims). sQ holds each warp's q_absorbed.
    __shared__ __half sQ[kWarps * DQK];
    __shared__ __half sK_T[DQK * (TN + 1)];

    const int warp  = threadIdx.x >> 5;
    const int lane  = threadIdx.x & (kWarp - 1);
    const int m_row = blockIdx.x * kWarps + warp;   // global PACKED-M row (= a query head at N_q=1)
    const int split = blockIdx.y;
    const int64_t bh_kv = blockIdx.z;
    const int b     = (int)(bh_kv / H_kv);
    const int h_kv  = (int)(bh_kv % H_kv);           // = 0 for MLA (H_kv=1)
    const int M     = G * N_q;
    const bool active = (m_row < M);

    const int g_local = active ? (m_row / N_q) : 0;
    const int i_q     = active ? (m_row % N_q) : 0;
    const int h_q     = h_kv * G + g_local;          // GLOBAL query head this warp owns (= g_local, H_kv=1)

    const int j_start = split * chunk;
    const int j_end   = (j_start + chunk < N_k) ? (j_start + chunk) : N_k;

    const __half* Qrow = Q + (((int64_t)b * H_q + h_q) * N_q + i_q) * DQK;
    const int* bt_b    = block_table + (int64_t)b * n_logical;

    // Stage this warp's q_absorbed row into sQ[warp*DQK + .] (the KV-load __syncthreads below makes it
    // visible to all lanes). Idle warps skip — they never read sQ.
    if (active) {
#pragma unroll
        for (int e = 0; e < EPT_Q; ++e)
            sQ[warp * DQK + lane + kWarp * e] = Qrow[lane + kWarp * e];
    }

    // Running online-softmax stats for this row (across the split). Register-resident O accumulator.
    float m_run = -FLT_MAX, l_run = 0.f, o_reg[EPT_O];
#pragma unroll
    for (int e = 0; e < EPT_O; ++e) o_reg[e] = 0.f;

    for (int j0 = j_start; j0 < j_end; j0 += TN) {
        // Cooperative paged gather of this latent tile. Whole block participates (uniform for the
        // __syncthreads). Each NVFP4 element is dequantized to FP16 here — FUSED per-tile, never a
        // full-cache prepass. For latent dim t: packed byte at d-index t/2 (nibble t&1), micro-scale at
        // d-index t/16. Written TRANSPOSED (sK_T[t*(TN+1)+r]) — one buffer, used for both QK and PV.
        for (int idx = threadIdx.x; idx < TN * DQK; idx += kBlock) {
            int r = idx / DQK, t = idx % DQK, gj = j0 + r;   // r = key within tile, t = latent dim
            if (gj < j_end) {
                int pb  = bt_b[gj / page_size];
                int off = gj % page_size;
                int64_t tok  = (int64_t)(pb * page_size + off) * H_kv + h_kv;   // token-(latent-head) index
                int64_t psrc = tok * DP + (t >> 1);          // packed byte holding dim t
                int64_t msrc = tok * DM + (t >> 4);          // E4M3 micro-scale for dim t's 16-block
                float lv = dequant_nvfp4(L_pool[psrc], t & 1) * dequant_e4m3(L_scale[msrc]) * scale_l;
                sK_T[t * (TN + 1) + r] = __float2half(lv);   // [d][key]+pad
            } else {
                sK_T[t * (TN + 1) + r] = __float2half(0.f);
            }
        }
        __syncthreads();   // sK_T, sQ visible to every lane before any compute

        if (active) {
            // Monotone valid-key cutoff for this tile: key c valid iff gj=j0+c < j_end AND
            // (!causal OR gj <= i_q+q_offset); both monotone in c. Warp-uniform.
            int c_lim = j_end - j0;
            if (c_lim > TN) c_lim = TN;
            if (causal) {
                int cc = i_q + q_offset - j0 + 1;
                if (cc < c_lim) c_lim = cc;
            }
            if (c_lim < 0) c_lim = 0;

#pragma unroll
            for (int g = 0; g < NG; ++g) {
                const int c = g * kWarp + lane;            // this lane's key within the tile
                const bool valid = (c < c_lim);

                // Full dot product q_absorbed . latent_c over DQK, entirely in this lane's registers (no
                // cross-lane reduction). sQ read is a warp-broadcast (d uniform); sK_T read is stride-1
                // in c -> conflict-free.
                float s = -INFINITY;
                if (valid) {
                    float acc = 0.f;
                    // Bounded unroll (DQK is up to 576 — a full unroll bloats SASS/compile; 8 keeps ILP).
#pragma unroll 8
                    for (int d = 0; d < DQK; ++d)
                        acc += __half2float(sQ[warp * DQK + d]) * __half2float(sK_T[d * (TN + 1) + c]);
                    s = acc * scale;
                }

                // Per-GROUP softmax: one max + one sum over the 32 lane-scores (all lanes participate;
                // invalid lanes fed the identities -inf / 0). Recurrence advances once per 32 keys.
                const float m_tile = warp_reduce_max(s);
                const float m_new  = fmaxf(m_run, m_tile);
                const float p      = valid ? __expf(s - m_new) : 0.f;
                const float l_tile = warp_reduce_sum(p);
                const float alpha  = __expf(m_run - m_new);  // (0,1]; wipes prior on first real score
                l_run = l_run * alpha + l_tile;
#pragma unroll
                for (int e = 0; e < EPT_O; ++e) o_reg[e] *= alpha;

                // PV transpose: O_latent += sum_c p_c * latent_c[:DV]. Broadcast p from each lane (single
                // hop, pipelines); invalid key's p_cc=0 zeroes its term. Lane owns output dims {lane,
                // lane+32, ...} (< DV). Read V[kc][d] = sK_T[d*(TN+1)+kc] from the same transposed tile.
#pragma unroll
                for (int cc = 0; cc < kWarp; ++cc) {
                    const float p_cc = __shfl_sync(0xffffffffu, p, cc);
                    const int kc = g * kWarp + cc;          // key index within the tile
#pragma unroll
                    for (int e = 0; e < EPT_O; ++e) {
                        const int d = lane + kWarp * e;     // owned output/content dim (< DV)
                        o_reg[e] += p_cc * __half2float(sK_T[d * (TN + 1) + kc]);
                    }
                }
                m_run = m_new;
            }
        }
        __syncthreads();   // protect sK_T before the next tile overwrites it
    }

    // Write this split's UNNORMALIZED partial (no /l), indexed by the QUERY head. Width DV.
    if (active) {
        int64_t bh_q = (int64_t)b * H_q + h_q;
        int64_t ml = ((bh_q * N_q) + i_q) * num_splits + split;
        int64_t ob = ml * DV;
#pragma unroll
        for (int e = 0; e < EPT_O; ++e)
            O_partial[ob + lane + kWarp * e] = o_reg[e];
        if (lane == 0) {
            m_partial[ml] = m_run;
            l_partial[ml] = l_run;
        }
    }
}

// MERGE kernel — identical to v10 (sees only the [B,H_q,N_q,S,*] FP32 workspace; the MLA shape change
// never reaches it). D here = DV (the latent-output width).
__global__ void mla_merge_kernel(const float* __restrict__ O_partial,
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

template <int TN, int DQK, int DV>
void launch_partial(const __half* q, const uint8_t* l_pool, const uint8_t* l_scale,
                    const int* block_table, float* O_partial, float* m_partial, float* l_partial,
                    int64_t BH_kv, int H_q, int H_kv, int G, int N_q, int N_k,
                    int num_splits, int chunk, int page_size, int n_logical, int q_offset,
                    float scale, bool causal, float scale_l, cudaStream_t stream) {
    const int M = G * N_q;
    dim3 grid(ceil_div(M, kWarps), (unsigned)num_splits, (unsigned)BH_kv);
    mla_partial_kernel<TN, DQK, DV><<<grid, kBlock, 0, stream>>>(
        q, l_pool, l_scale, block_table, O_partial, m_partial, l_partial,
        H_q, H_kv, G, N_q, N_k, num_splits, chunk, page_size, n_logical, q_offset, scale, causal,
        scale_l);
}

// choose_splits — identical to v10/v8.7 (with H_kv=1, G=h_q for MLA).
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

// Host entry: MLA latent-KV decode. ONE latent pool (packed NVFP4 nibbles [.,.,1,DQK/2] uint8 + E4M3
// micro-scales [.,.,1,DQK/16] uint8) shared by all h_q query heads; q_absorbed [B,h_q,N_q,DQK] FP16.
// kv_lora_rank = DV (the PV/output width); DQK = q.size(3) (the score width, = kv_lora_rank + rope_dim).
// G = H_q (every query head packs into M); H_kv = 1. Output O_latent [B,h_q,N_q,DV] FP32.
torch::Tensor mla_attention_forward(torch::Tensor q, torch::Tensor l_pool, torch::Tensor l_scale,
                                    torch::Tensor block_table, int64_t page_size, int64_t n_k,
                                    int64_t kv_lora_rank, double scale_l,
                                    double scale, bool causal, int64_t q_offset) {
    TORCH_CHECK(q.is_cuda() && l_pool.is_cuda() && l_scale.is_cuda() && block_table.is_cuda(),
                "q, l_pool, l_scale, block_table must be CUDA tensors");
    TORCH_CHECK(q.dim() == 4, "q (q_absorbed) must be [B,H_q,N_q,DQK]");
    TORCH_CHECK(l_pool.dim() == 4, "l_pool must be [num_blocks, page_size, H_kv=1, DQK/2]");
    TORCH_CHECK(l_scale.dim() == 4, "l_scale must be [num_blocks, page_size, H_kv=1, DQK/16]");
    TORCH_CHECK(l_pool.scalar_type() == torch::kUInt8,
                "l_pool must be uint8 (packed NVFP4 nibbles); build with build_paged_kv_mla");
    TORCH_CHECK(l_scale.scalar_type() == torch::kUInt8, "l_scale must be uint8 (E4M3 micro-scale bytes)");
    TORCH_CHECK(block_table.dim() == 2, "block_table must be [B, n_logical]");
    TORCH_CHECK(block_table.scalar_type() == torch::kInt32, "block_table must be int32");
    q = q.contiguous(); l_pool = l_pool.contiguous(); l_scale = l_scale.contiguous();
    block_table = block_table.contiguous();

    const int B = q.size(0), H_q = q.size(1), N_q = q.size(2), DQK = q.size(3);
    const int H_kv = l_pool.size(2);
    const int DV = (int)kv_lora_rank;
    const int N_k = (int)n_k;
    const int n_logical = (int)block_table.size(1);
    TORCH_CHECK(H_kv == 1, "MLA latent pool must have exactly one (latent) head; got H_kv=", H_kv);
    TORCH_CHECK(DQK % 16 == 0, "DQK (latent width) must be a multiple of the NVFP4 block (16); got ", DQK);
    TORCH_CHECK(DV >= 1 && DV <= DQK && DV % 32 == 0,
                "kv_lora_rank (DV) must be in [1,DQK], a multiple of 32; got ", DV, " DQK=", DQK);
    TORCH_CHECK(l_pool.size(3) == DQK / 2,
                "packed latent pool width must equal DQK/2 (two NVFP4 nibbles per byte)");
    TORCH_CHECK(l_scale.size(3) == DQK / 16,
                "micro-scale pool width must equal DQK/16 (one E4M3 scale per 16-element block)");
    TORCH_CHECK(l_scale.size(1) == page_size && l_pool.size(1) == page_size &&
                l_scale.size(2) == H_kv, "latent pools must share [num_blocks, page_size, 1, *]");
    TORCH_CHECK(block_table.size(0) == B, "block_table must have B rows");
    TORCH_CHECK((int64_t)n_logical * page_size >= N_k,
                "block_table n_logical*page_size must cover N_k logical positions");

    const int G = H_q;   // MLA: every query head packs into M (one shared latent head)

    auto qh = q.to(torch::kHalf);
    auto O  = torch::empty({B, H_q, N_q, DV}, q.options().dtype(torch::kFloat32));

    const int64_t BH_kv = (int64_t)B * H_kv;   // = B
    const int S = choose_splits(B, H_kv, G, N_q, N_k);
    const int chunk = ceil_div(N_k, S);

    auto opts_f = q.options().dtype(torch::kFloat32);
    auto O_partial = torch::empty({B, H_q, N_q, S, DV}, opts_f);
    auto m_partial = torch::empty({B, H_q, N_q, S},     opts_f);
    auto l_partial = torch::empty({B, H_q, N_q, S},     opts_f);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const __half* qp = reinterpret_cast<const __half*>(qh.data_ptr<at::Half>());
    const uint8_t* lp = l_pool.data_ptr<uint8_t>();
    const uint8_t* ls = l_scale.data_ptr<uint8_t>();
    const int* btp = block_table.data_ptr<int>();
    float* Op = O_partial.data_ptr<float>();
    float* Mp = m_partial.data_ptr<float>();
    float* Lp = l_partial.data_ptr<float>();

    // Dispatch on (DQK, DV). TN=32 keeps smem under the T4 48 KB static limit even at the real
    // (576, 512) DeepSeek-V3 shape (sK_T 38 KB + sQ 9 KB ~= 46 KB). The small configs are for the
    // correctness sweep (tiny smem, fast compile). Add a (DQK, DV) tuple + a TORCH_CHECK branch here
    // to support a new latent shape.
    #define MLA_LAUNCH(TN_, DQK_, DV_)                                                            \
        launch_partial<TN_, DQK_, DV_>(qp, lp, ls, btp, Op, Mp, Lp, BH_kv, H_q, H_kv, G, N_q,     \
                                       N_k, S, chunk, (int)page_size, (int)n_logical,             \
                                       (int)q_offset, (float)scale, causal, (float)scale_l, stream)
    if (DQK == 96 && DV == 64) {
        MLA_LAUNCH(32, 96, 64);
    } else if (DQK == 160 && DV == 128) {
        MLA_LAUNCH(32, 160, 128);
    } else if (DQK == 576 && DV == 512) {
        MLA_LAUNCH(32, 576, 512);
    } else {
        TORCH_CHECK(false, "v11 mla: unsupported (DQK,DV)=(", DQK, ",", DV, "); supported: (96,64) "
                    "(160,128) (576,512). Add a template instantiation in mla_attention.cu.");
    }
    #undef MLA_LAUNCH

    dim3 mgrid((unsigned)N_q, (unsigned)(B * H_q));
    mla_merge_kernel<<<mgrid, DV, 0, stream>>>(Op, Mp, Lp, O.data_ptr<float>(), N_q, S, DV);

    return O;
}

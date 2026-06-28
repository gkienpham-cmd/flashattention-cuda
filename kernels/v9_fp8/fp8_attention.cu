// v9 — FP8 E4M3 KV-cache decode. Forked from v8.7 (gqa_ss_attention.cu); the score-stationary inner
// loop, M-packing grid, split-KV, paged gather structure, LSE merge, [B,H_q,N_q,S,*] workspace,
// choose_splits, and host are carried BYTE-IDENTICAL. This kernel changes exactly ONE thing — the KV
// STORAGE PRECISION: the paged K/V pool holds FP8 E4M3 (1 byte) instead of FP16 (2 bytes), and a
// per-tensor FP32 scale is applied at the smem-gather step (fused per-tile dequant, NOT a prepass).
//
//   WHY (docs/v9-kickoff.md). v8.7 closed the decode-SCHEDULE arc: score-stationary + M-packing beat
//   SDPA 8-16x on CUDA cores, but %HBM plateaued ~10-12% (per-CTA-bound OR L2-resident — confounded).
//   FP8 attacks the BYTES: decode AI = 2G/b rises 8.0 -> 16.0 (b: 2 -> 1), the HBM floor HALVES. The
//   roofline is BLIND to dequant latency and to the L2 confound. PREDICTION: on the L2-resident
//   micro-bench FP8 is likely capacity-only (no latency win); its byte win only shows past L2 / large
//   batch IFF bandwidth-bound there. FP8's durable value is KV-cache CAPACITY + accuracy (v10 NVFP4/B300).
//
//   THE FUSED DEQUANT (the single variable). At the cooperative gather (below), each FP8 byte is read,
//   converted (cuda_fp8.h, software-emulated on sm_75/T4), multiplied by the per-tensor scale, and
//   stored as FP16 into the existing sK(transposed)/sV(natural) smem — exactly where v8.7 wrote its
//   FP16 K/V. The score-stationary inner loop reads sK/sV identically, so this is a clean byte-only
//   ablation: smem layout, recurrence, PV fan, merge — all untouched.
//
// Layout: Q dense [B,H_q,N_q,d] (FP16). K_pool/V_pool [num_blocks,page_size,H_kv,d] (FP8 E4M3 bytes,
// passed as uint8). block_table int32 [B,n_logical].

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

// FP8 E4M3 byte -> float. Software-emulated on sm_75/T4 via cuda_fp8.h (CUDA >= 11.8) — verify it
// compiles on Colab. FALLBACK if ptxas rejects it on sm_75: store int8-symmetric and dequant with
// `return (float)(int8_t)b;` (same 1-byte HBM win, isolates the byte variable identically; the
// E4M3-vs-int8 accuracy gap is itself a v9 deliverable). The Python quantizer must match whichever
// path is active here (build_paged_kv_fp8 -> torch.float8_e4m3fn for E4M3).
__device__ __forceinline__ float dequant_e4m3(uint8_t b) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)));
}

// Butterfly all-reduce SUM (reused from Cut 1) and MAX (the per-tile softmax). Both leave the full
// 32-lane result in every lane; both must be called by ALL 32 lanes (masked keys feed the identity).
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
// PARTIAL kernel. Grid/work IDENTICAL to v8.7 (grid = (ceil_div(M,kWarps), num_splits, B*H_kv),
// M=G*N_q). Only the KV pool dtype (uint8 FP8) + the per-tensor dequant scales differ; the staged
// smem is still FP16, so the score-stationary inner loop is byte-identical to v8.7.
// ---------------------------------------------------------------------------------------
template <int TN, int D>
__global__ void fp8_partial_kernel(const __half*  __restrict__ Q,          // [B,H_q,N_q,D] dense FP16
                                   const uint8_t* __restrict__ K_pool,      // [num_blocks,page_size,H_kv,D] FP8
                                   const uint8_t* __restrict__ V_pool,      // [num_blocks,page_size,H_kv,D] FP8
                                   const int* __restrict__ block_table,     // [B, n_logical]
                                   float* __restrict__ O_partial,           // [B,H_q,N_q,S,D]
                                   float* __restrict__ m_partial,           // [B,H_q,N_q,S]
                                   float* __restrict__ l_partial,           // [B,H_q,N_q,S]
                                   int H_q, int H_kv, int G, int N_q, int N_k,
                                   int num_splits, int chunk,
                                   int page_size, int n_logical, int q_offset,
                                   float scale, bool causal,
                                   float scale_k, float scale_v) {         // per-tensor FP8 dequant scales
    static_assert(D % kWarp == 0, "head_dim must be a multiple of the warp size (32)");
    static_assert(TN % kWarp == 0, "tile width must be a multiple of the warp size (32)");
    constexpr int EPT = D / kWarp;    // output dims each lane owns in PV (d=64->2, d=128->4)
    constexpr int NG  = TN / kWarp;   // 32-key groups per staged tile (TN=64->2, TN=32->1)

    // FP16 staged tiles (the FP8 bytes are dequantized into here at load time). sK TRANSPOSED [d][key]
    // + 1-pad (bank-conflict-free read AND write); sV natural [key][d]; sQ holds each warp's query row.
    __shared__ __half sQ[kWarps * D];
    __shared__ __half sK[D * (TN + 1)];
    __shared__ __half sV[TN * D];

    const int warp  = threadIdx.x >> 5;
    const int lane  = threadIdx.x & (kWarp - 1);
    const int m_row = blockIdx.x * kWarps + warp;   // global PACKED-M row
    const int split = blockIdx.y;
    const int64_t bh_kv = blockIdx.z;
    const int b     = (int)(bh_kv / H_kv);
    const int h_kv  = (int)(bh_kv % H_kv);
    const int M     = G * N_q;
    const bool active = (m_row < M);

    const int g_local = active ? (m_row / N_q) : 0;
    const int i_q     = active ? (m_row % N_q) : 0;
    const int h_q     = h_kv * G + g_local;         // GLOBAL query head this warp owns

    const int j_start = split * chunk;
    const int j_end   = (j_start + chunk < N_k) ? (j_start + chunk) : N_k;

    const __half* Qrow = Q + (((int64_t)b * H_q + h_q) * N_q + i_q) * D;
    const int* bt_b    = block_table + (int64_t)b * n_logical;

    // Stage this warp's query row into sQ[warp*D + .] (intra-warp; the KV-load __syncthreads below makes
    // it visible to all lanes of the warp before the first read). Idle warps skip — they never read sQ.
    if (active) {
#pragma unroll
        for (int e = 0; e < EPT; ++e)
            sQ[warp * D + lane + kWarp * e] = Qrow[lane + kWarp * e];
    }

    // Running online-softmax stats for this row (across the split). Register-resident O accumulator.
    float m_run = -FLT_MAX, l_run = 0.f, o_reg[EPT];
#pragma unroll
    for (int e = 0; e < EPT; ++e) o_reg[e] = 0.f;

    for (int j0 = j_start; j0 < j_end; j0 += TN) {
        // Cooperative paged gather of this key/value tile. Whole block participates (uniform for the
        // __syncthreads). Each FP8 byte is dequantized (x scale) to FP16 here — FUSED per-tile, never a
        // full-cache prepass (a prepass re-reads the cache and eats the byte savings — QServe). K is
        // written TRANSPOSED (sK[t*(TN+1)+r]); V natural (sV[r*D+t]).
        for (int idx = threadIdx.x; idx < TN * D; idx += kBlock) {
            int r = idx / D, t = idx % D, gj = j0 + r;     // r = key within tile, t = head-dim elem
            if (gj < j_end) {
                int pb  = bt_b[gj / page_size];
                int off = gj % page_size;
                int64_t src = ((int64_t)(pb * page_size + off) * H_kv + h_kv) * D + t;
                sK[t * (TN + 1) + r] = __float2half(dequant_e4m3(K_pool[src]) * scale_k);  // [d][key]+pad
                sV[r * D + t]        = __float2half(dequant_e4m3(V_pool[src]) * scale_v);  // [key][d]
            } else {
                sK[t * (TN + 1) + r] = __float2half(0.f);
                sV[r * D + t]        = __float2half(0.f);
            }
        }
        __syncthreads();   // sK, sV, sQ visible to every lane before any compute

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

                // Full dot product q . k_c, entirely in this lane's registers (no cross-lane reduction).
                // sQ read is a warp-broadcast (d uniform); sK read is stride-1 in c -> conflict-free.
                float s = -INFINITY;
                if (valid) {
                    float acc = 0.f;
#pragma unroll
                    for (int d = 0; d < D; ++d)
                        acc += __half2float(sQ[warp * D + d]) * __half2float(sK[d * (TN + 1) + c]);
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
                for (int e = 0; e < EPT; ++e) o_reg[e] *= alpha;

                // PV transpose: O += sum_c p_c * V[c]. Broadcast p from each lane (single hop, pipelines);
                // invalid key's p_cc=0 zeroes its V term. Lane owns output dims {lane, lane+32, ...}.
#pragma unroll
                for (int cc = 0; cc < kWarp; ++cc) {
                    const float p_cc = __shfl_sync(0xffffffffu, p, cc);
                    const int kc = g * kWarp + cc;          // key index within the tile
#pragma unroll
                    for (int e = 0; e < EPT; ++e)
                        o_reg[e] += p_cc * __half2float(sV[kc * D + lane + kWarp * e]);
                }
                m_run = m_new;
            }
        }
        __syncthreads();   // protect sK/sV before the next tile overwrites them
    }

    // Write this split's UNNORMALIZED partial (no /l), indexed by the QUERY head — identical to v8.7.
    if (active) {
        int64_t bh_q = (int64_t)b * H_q + h_q;
        int64_t ml = ((bh_q * N_q) + i_q) * num_splits + split;
        int64_t ob = ml * D;
#pragma unroll
        for (int e = 0; e < EPT; ++e)
            O_partial[ob + lane + kWarp * e] = o_reg[e];
        if (lane == 0) {
            m_partial[ml] = m_run;
            l_partial[ml] = l_run;
        }
    }
}

// MERGE kernel — identical to v8.7 / v6 / v7 (sees only the [B,H_q,N_q,S,*] FP32 workspace; the FP8
// storage change never reaches it).
__global__ void fp8_merge_kernel(const float* __restrict__ O_partial,
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
void launch_partial(const __half* q, const uint8_t* k_pool, const uint8_t* v_pool,
                    const int* block_table,
                    float* O_partial, float* m_partial, float* l_partial,
                    int64_t BH_kv, int H_q, int H_kv, int G, int N_q, int N_k,
                    int num_splits, int chunk, int page_size, int n_logical, int q_offset,
                    float scale, bool causal, float scale_k, float scale_v, cudaStream_t stream) {
    const int M = G * N_q;
    dim3 grid(ceil_div(M, kWarps), (unsigned)num_splits, (unsigned)BH_kv);
    fp8_partial_kernel<TN, D><<<grid, kBlock, 0, stream>>>(
        q, k_pool, v_pool, block_table, O_partial, m_partial, l_partial,
        H_q, H_kv, G, N_q, N_k, num_splits, chunk, page_size, n_logical, q_offset, scale, causal,
        scale_k, scale_v);
}

// choose_splits — identical to v8.7.
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

// Host entry: v8.7's contract + two new per-tensor dequant scales (scale_k, scale_v). KV pools are FP8
// E4M3 bytes passed as uint8; Q stays FP16. G derived from shapes (H_q/H_kv).
torch::Tensor fp8_attention_forward(torch::Tensor q, torch::Tensor k_pool, torch::Tensor v_pool,
                                    torch::Tensor block_table, int64_t page_size, int64_t n_k,
                                    double scale_k, double scale_v,
                                    double scale, bool causal, int64_t q_offset) {
    TORCH_CHECK(q.is_cuda() && k_pool.is_cuda() && v_pool.is_cuda() && block_table.is_cuda(),
                "q, k_pool, v_pool, block_table must be CUDA tensors");
    TORCH_CHECK(q.dim() == 4, "q must be [B,H_q,N_q,d]");
    TORCH_CHECK(k_pool.dim() == 4 && v_pool.dim() == 4,
                "k_pool, v_pool must be [num_blocks, page_size, H_kv, d]");
    TORCH_CHECK(k_pool.scalar_type() == torch::kUInt8 && v_pool.scalar_type() == torch::kUInt8,
                "k_pool, v_pool must be uint8 (FP8 E4M3 bytes); quantize with build_paged_kv_fp8");
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
    const uint8_t* kp = k_pool.data_ptr<uint8_t>();
    const uint8_t* vp = v_pool.data_ptr<uint8_t>();
    const int* btp = block_table.data_ptr<int>();
    float* Op = O_partial.data_ptr<float>();
    float* Mp = m_partial.data_ptr<float>();
    float* Lp = l_partial.data_ptr<float>();

    // Tile config per head dim (same TN as v8.7).
    if (d == 64) {
        launch_partial<64, 64>(qp, kp, vp, btp, Op, Mp, Lp, BH_kv, H_q, H_kv, G, N_q, N_k, S, chunk,
                               (int)page_size, (int)n_logical, (int)q_offset, (float)scale, causal,
                               (float)scale_k, (float)scale_v, stream);
    } else if (d == 128) {
        launch_partial<32, 128>(qp, kp, vp, btp, Op, Mp, Lp, BH_kv, H_q, H_kv, G, N_q, N_k, S, chunk,
                                (int)page_size, (int)n_logical, (int)q_offset, (float)scale, causal,
                                (float)scale_k, (float)scale_v, stream);
    } else {
        TORCH_CHECK(false, "v9 fp8 supports head_dim 64 or 128 (got ", d, ")");
    }

    dim3 mgrid((unsigned)N_q, (unsigned)(B * H_q));
    fp8_merge_kernel<<<mgrid, d, 0, stream>>>(Op, Mp, Lp, O.data_ptr<float>(), N_q, S, d);

    return O;
}

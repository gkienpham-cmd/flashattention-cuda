// v5 WMMA FlashAttention-1 — the GEMV->GEMM fix. Keep v4's single-pass fused schedule, but do both
// matmuls on Turing's tensor cores (FP16-in / FP32-accum) instead of warp-shuffle dot products.
//
// Step 4 (v4) landed the thesis — S off HBM is also a wall-clock win — but measured ~6x slower than
// SDPA, ~18x off the FP32 MMA floor. Diagnosed limiter: FMA under-utilization. v4 maps one WARP to a
// query row and computes each score with a 32-lane __shfl tree reduction; that reduction plumbing
// (~5 shuffle steps per single output scalar) is GEMV-shaped and leaves the FMA units mostly idle —
// it never approaches even the 8.1 TFLOPS CUDA-core peak.
//
// v5 restructures both matmuls into 16x16x16 GEMM tiles the tensor cores execute natively:
//   - QK:  S[BM][BN] = Q @ K^T  via wmma::mma_sync (the reduction over d lives INSIDE the tensor
//          core; no __shfl).  K is realized as a col_major B-fragment, so K^T needs no transpose.
//   - PV:  O[BM][D]  += P @ V  via wmma::mma_sync (reduction over the key block).
// This is the "GEMV -> GEMM" transition: one mma_sync produces a 16x16 tile (256 depth-16 dot
// products) per instruction. The FP16 inputs are the price of admission to the 65 TFLOPS path; the
// accumulator stays FP32 so the long online-softmax reduction (l, O across up to N=16384 keys) keeps
// v3/v4's rescale stability.
//
// THE OPAQUE-FRAGMENT TAX: a WMMA accumulator's 256 results are scattered across the warp's lanes in
// an undocumented layout you cannot index. So softmax cannot stay in the accumulator (v4 kept O in
// named registers). Instead we force S through shared memory:
//     QK -> store S to smem  ->  row-softmax in smem (each lane owns a whole row, NO __shfl — the
//     GEMM already reduced)  ->  write P back to smem as half  ->  reload P as fragments for PV.
// The O-rescale (O = alpha*O + P@V, alpha = exp(m_old - m_new)) also leaves the accumulator: we keep
// the running O as FP32 in smem (oRun), rescale it per-row by alpha with ordinary threads, then for
// PV we LOAD oRun back into the accumulator fragment and mma_sync P@V on top — so the add is the
// tensor core's own C += A*B, with no separate output scratch.
//
// Tiling — one warp per 16-row query M-tile; warp w owns rows [16w, 16w+16), so softmax never
// crosses a warp boundary (lane l<16 owns the full row 16w+l in smem). Chosen to fit Turing's 48 KB
// static-smem budget (oRun is the heavy FP32 [BM][D] buffer):
//   d=64  -> BM=64, BN=32, 4 warps  (smem ~44 KB)
//   d=128 -> BM=32, BN=32, 2 warps  (smem ~46 KB)
//
// Precision: FP16 in / FP32 accumulate / FP32 out. First version NOT bit-comparable to FP32 SDPA —
// correctness uses a looser tolerance (see tests). Layout: q,k,v are [B,H,N,d] row-major; the host
// entry casts the fp32 inputs to half so the public attention() contract stays fp32-in. Causal
// excludes keys j > query i.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>   // at::cuda::getCurrentCUDAStream
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cfloat>   // FLT_MAX

namespace {
using namespace nvcuda;

constexpr int kWarp  = 32;
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;   // the only Turing half/float WMMA shape we use

// Round-up integer division, for sizing grids to cover `n` elements.
inline int ceil_div(int64_t n, int b) { return static_cast<int>((n + b - 1) / b); }

// ---------------------------------------------------------------------------------------
// Fused single-pass WMMA kernel. One block owns BM consecutive query rows of one (batch,head);
// warp `w` owns the 16-row M-tile [16w, 16w+16) and carries that tile's running (m, l) (per owning
// lane) while the FP32 running output O lives in shared memory (oRun). The block streams the key
// axis in BN-wide blocks, staging each K/V block into smem ONCE. Per block: WMMA QK -> store S to
// smem -> online softmax (mask + rescale oRun by alpha + write P as half) -> WMMA PV accumulated on
// top of oRun. After the key axis: normalize by l and store. S is never formed in HBM.
// ---------------------------------------------------------------------------------------
template <int BM, int BN, int D, int WARPS>
__global__ void wmma_attention_kernel(const __half* __restrict__ Q,
                                      const __half* __restrict__ K,
                                      const __half* __restrict__ V,
                                      float* __restrict__ O,
                                      int N_q, int N_k, float scale, bool causal) {
    static_assert(BM % WMMA_M == 0, "BM must be a multiple of 16");
    static_assert(BN % WMMA_N == 0, "BN must be a multiple of 16");
    static_assert(D  % WMMA_K == 0, "head_dim must be a multiple of 16");
    static_assert(WARPS == BM / WMMA_M, "exactly one warp per 16-row query M-tile");
    constexpr int kBlock = WARPS * kWarp;   // threads per block
    constexpr int DK = D  / WMMA_K;         // # k-subtiles along d   (QK contraction)
    constexpr int NN = BN / WMMA_N;         // # subtiles along the key block (QK output / PV contraction)
    constexpr int DN = D  / WMMA_N;         // # subtiles along d     (PV output)
    // Static smem budget (half: sQ,sK,sV,sP @ 2B; float: sS,oRun @ 4B). Turing static-smem cap is
    // 48 KB; assert here so a future tile-config edit fails at compile, not with a launch error.
    static_assert((BM*D + BN*D + BN*D + BM*BN) * 2 + (BM*BN + BM*D) * 4 <= 49152,
                  "v5 smem tiles exceed the 48 KB static budget; shrink BM/BN for this head dim");

    __shared__ __half sQ[BM * D];   // this block's query band, zero-padded past N_q   [BM][D] row-major
    __shared__ __half sK[BN * D];   // current key block,   zero-padded past N_k       [BN][D] row-major
    __shared__ __half sV[BN * D];   // current value block, zero-padded past N_k       [BN][D] row-major
    __shared__ float  sS[BM * BN];  // QK scores scratch (consumed each block)         [BM][BN] row-major
    __shared__ __half sP[BM * BN];  // softmax probabilities, reloaded for PV           [BM][BN] row-major
    __shared__ float  oRun[BM * D]; // running UNNORMALIZED output (FP32 accumulator)   [BM][D] row-major

    const int warp = threadIdx.x >> 5;            // 0..WARPS-1 : which 16-row M-tile
    const int lane = threadIdx.x & (kWarp - 1);   // 0..31
    const int i0   = blockIdx.x * BM;             // first query row of this block's M-band
    const int64_t bh = blockIdx.y;                // (batch*head) slice

    const __half* Qbh = Q + bh * (int64_t)N_q * D;
    const __half* Kbh = K + bh * (int64_t)N_k * D;
    const __half* Vbh = V + bh * (int64_t)N_k * D;

    // Stage this block's Q band into smem (zero-padded for rows >= N_q so partial M-tiles are safe),
    // and zero the running output. Both done cooperatively by the whole block.
    for (int idx = threadIdx.x; idx < BM * D; idx += kBlock) {
        int r = idx / D, t = idx % D, gi = i0 + r;
        sQ[idx]   = (gi < N_q) ? Qbh[(int64_t)gi * D + t] : __float2half(0.f);
        oRun[idx] = 0.f;
    }
    __syncthreads();

    // Q stays resident in fragments for the whole key loop (loaded once from the padded smem band).
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> qFrag[DK];
#pragma unroll
    for (int kk = 0; kk < DK; ++kk)
        wmma::load_matrix_sync(qFrag[kk], &sQ[(warp * WMMA_M) * D + kk * WMMA_K], D);

    // Per-row running softmax stats. lane l<16 owns local row (warp*16 + l); these registers persist
    // across the key axis. Empty state (m=-FLT_MAX, l=0, oRun=0) is wiped by the first block's alpha.
    float m_run = -FLT_MAX, l_run = 0.f;

    for (int j0 = 0; j0 < N_k; j0 += BN) {
        // Cooperative load of this key/value block (coalesced; zero-padded past N_k). Whole block
        // participates so the __syncthreads is reached uniformly.
        for (int idx = threadIdx.x; idx < BN * D; idx += kBlock) {
            int r = idx / D, t = idx % D, gj = j0 + r;
            bool ok = gj < N_k;
            sK[idx] = ok ? Kbh[(int64_t)gj * D + t] : __float2half(0.f);
            sV[idx] = ok ? Vbh[(int64_t)gj * D + t] : __float2half(0.f);
        }
        __syncthreads();   // sK, sV visible before any WMMA

        // ---- QK: S[BM][BN] = Q @ K^T.  K^T via a col_major B-fragment (no explicit transpose). ----
#pragma unroll
        for (int ns = 0; ns < NN; ++ns) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
            wmma::fill_fragment(acc, 0.f);
#pragma unroll
            for (int kk = 0; kk < DK; ++kk) {
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> kFrag;
                // col_major B at (key n0=ns*16, dim k0=kk*16), ldm=D, realizes B[k][n]=K[n][k].
                wmma::load_matrix_sync(kFrag, &sK[(ns * WMMA_N) * D + kk * WMMA_K], D);
                wmma::mma_sync(acc, qFrag[kk], kFrag, acc);
            }
            // store this warp's 16x16 score tile into sS[16w..][ns*16..]
            wmma::store_matrix_sync(&sS[(warp * WMMA_M) * BN + ns * WMMA_N], acc, BN, wmma::mem_row_major);
        }
        __syncthreads();   // sS complete before softmax reads it

        // ---- Online softmax over this warp's rows (lane l<16 owns the whole row from smem). ----
        if (lane < WMMA_M) {
            const int r  = warp * WMMA_M + lane;   // local row in [0,BM)
            const int gi = i0 + r;                 // global query row (may be >= N_q on the last tile)
            float*  sSr = &sS[r * BN];
            __half* sPr = &sP[r * BN];

            // pass 1: scale + mask in place, find this block's row max
            float rowmax = -FLT_MAX;
            for (int c = 0; c < BN; ++c) {
                int gj = j0 + c;
                float s = sSr[c] * scale;
                if (gj >= N_k || (causal && gj > gi)) s = -FLT_MAX;   // padded / future keys excluded
                sSr[c] = s;
                rowmax = fmaxf(rowmax, s);
            }
            // online update: shift baseline to the running max, rescale by alpha=exp(m_old-m_new)
            float m_new = fmaxf(m_run, rowmax);
            float alpha = __expf(m_run - m_new);   // in (0,1]; exp(-FLT_MAX - finite)=0 wipes empty state
            float psum  = 0.f;
            // pass 2: probabilities (masked entries -> exp(-inf)=0), written as half for the PV GEMM
            for (int c = 0; c < BN; ++c) {
                float p = __expf(sSr[c] - m_new);
                sPr[c]  = __float2half(p);
                psum   += p;
            }
            l_run = l_run * alpha + psum;
            m_run = m_new;
            // rescale the running O row by alpha BEFORE PV adds this block's contribution
            float* oR = &oRun[r * D];
#pragma unroll
            for (int cc = 0; cc < D; ++cc) oR[cc] *= alpha;
        }
        __syncthreads();   // sP written + oRun rescaled before PV

        // ---- PV: oRun += P @ V.  Load the (already rescaled) oRun tile into the accumulator, add
        //         P@V on top via mma_sync, store back — the O-rescale add IS the tensor core's C+=AB.
#pragma unroll
        for (int nd = 0; nd < DN; ++nd) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
            wmma::load_matrix_sync(acc, &oRun[(warp * WMMA_M) * D + nd * WMMA_N], D, wmma::mem_row_major);
#pragma unroll
            for (int kk = 0; kk < NN; ++kk) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> pFrag;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> vFrag;
                wmma::load_matrix_sync(pFrag, &sP[(warp * WMMA_M) * BN + kk * WMMA_K], BN);
                wmma::load_matrix_sync(vFrag, &sV[(kk * WMMA_K) * D + nd * WMMA_N], D);
                wmma::mma_sync(acc, pFrag, vFrag, acc);
            }
            wmma::store_matrix_sync(&oRun[(warp * WMMA_M) * D + nd * WMMA_N], acc, D, wmma::mem_row_major);
        }
        __syncthreads();   // protect sK/sV/sS/oRun before the next key block overwrites them
    }

    // ---- Normalize by the final denom and store (lane l<16 owns a row; coalesced enough at d<=128).
    if (lane < WMMA_M) {
        const int r  = warp * WMMA_M + lane;
        const int gi = i0 + r;
        if (gi < N_q) {
            float inv = (l_run > 0.f) ? (1.f / l_run) : 0.f;
            float* Obh = O + bh * (int64_t)N_q * D;
            float* oR  = &oRun[r * D];
            for (int cc = 0; cc < D; ++cc) Obh[(int64_t)gi * D + cc] = oR[cc] * inv;
        }
    }
}

// Launch the WMMA kernel for a fixed compile-time tile config (chosen by head dim).
template <int BM, int BN, int D, int WARPS>
void launch_wmma(const __half* q, const __half* k, const __half* v, float* O,
                 int B, int H, int N_q, int N_k, float scale, bool causal, cudaStream_t stream) {
    const int64_t BH = (int64_t)B * H;
    dim3 grid(ceil_div(N_q, BM), (unsigned)BH);   // one block per (query-M-band, batch-head)
    wmma_attention_kernel<BM, BN, D, WARPS><<<grid, WARPS * kWarp, 0, stream>>>(
        q, k, v, O, N_q, N_k, scale, causal);
}

}  // anonymous namespace

// Host entry: single fused WMMA pass, FP16-in / FP32-accum / FP32-out. The public attention()
// contract stays fp32-in (every test/bench passes fp32), so we cast q,k,v to half HERE — the cast is
// intentionally inside the timed region. Exposed to Python by binding.cpp as `forward`.
torch::Tensor wmma_attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
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
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const __half* qp = reinterpret_cast<const __half*>(qh.data_ptr<at::Half>());
    const __half* kp = reinterpret_cast<const __half*>(kh.data_ptr<at::Half>());
    const __half* vp = reinterpret_cast<const __half*>(vh.data_ptr<at::Half>());
    float* op = O.data_ptr<float>();

    // Tile config picked per head dim so the FP32 oRun band + staged tiles fit the 48 KB smem budget.
    if (d == 64) {
        launch_wmma<64, 32, 64, 4>(qp, kp, vp, op, B, H, N_q, N_k, (float)scale, causal, stream);
    } else if (d == 128) {
        launch_wmma<32, 32, 128, 2>(qp, kp, vp, op, B, H, N_q, N_k, (float)scale, causal, stream);
    } else {
        TORCH_CHECK(false, "v5 wmma supports head_dim 64 or 128 (got ", d,
                    "); tile config is specialized per head dim to fit the 48 KB smem budget");
    }

    return O;
}

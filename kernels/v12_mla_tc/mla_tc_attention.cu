// v12 — native tcgen05 TENSOR-CORE MLA latent-KV decode. Forks v11 (mla_attention.cu) changing ONE
// variable — the compute ENGINE: the M=h_q QK/PV matmuls move from v11's warp-per-head CUDA-core GEMV
// onto Blackwell 5th-gen tensor cores (tcgen05), first in FP8 (Arm 1), then native NVFP4 (Arm 2).
// Everything else — the paged NVFP4 latent pool/layout, split-KV, the LSE merge, choose_splits, the
// [B,H_q,N_q,S,*] workspace, scores≥FP16 — is carried byte-identical from v11 so the v12-vs-v11 A/B
// isolates the engine (decisions.md / results.md Step 12).
//
//   WHY (results.md Step 11 close-out, the measured 4× gap). v11 proved the MLA decode SHAPE is right
//   (M=128 by construction) but measured it PER-CTA-bound at 0.75 TFLOP/s on B300 — ~4× SLOWER than
//   torch dense-MQA — because torch routes the M=128 head-pack into a cuBLAS TENSOR-CORE GEMM while v11
//   runs warp-per-head CUDA-core GEMV. v11 crossed the last lever off: the only way to close that 4×
//   self-gap is the tensor-core path. v12 IS that path.
//
//   PRE-REGISTERED PREDICTION (results.md Step 12; the paper-grade result is the SHORTFALL).
//     - Pure roofline, engine-correct ridge: Arm 1 (FP8 engine, ridge 625) is compute-bound by 1.34×;
//       Arm 2 (NVFP4 engine, ridge 1875) falls BACK to HBM-bound (AI 835 < 1875 — the 15 PF peak
//       overshoots) with softmax-exp (measured 0.5×) as the #2 term.
//     - Per-CTA-corrected (the real prediction): single-token decode carries only 1–4 MMAs in flight;
//       tcgen05 wants 256–1024 to saturate → v12 is SMEM-BW / MMA-pipeline-depth-bound, realized
//       throughput well below the FP8/FP4 peak, and will NOT beat FlashMLA's ~410 TFLOP/s. The measured
//       shortfall — why even the right engine can't fill a decode CTA — is the contribution.
//     - Counter-prediction (the prize): if achieved TFLOP/s climbs >~10× v11's 0.75 with high %SMEM-BW,
//       the limiter finally LEFT per-CTA (first in the v1→v12 arc).
//
// ============================================================================================
//  *** GPU-SIDE WORK — KERNEL BODY NOT YET IMPLEMENTED ON THE AUTHOR MACHINE ***
//  The author machine has no CUDA toolchain and no CUTLASS; the tcgen05 kernel is built + measured on a
//  rented root/bare-metal B300/sm_103a (CUDA 12.9+, CUTLASS 4.x). This file is the SCAFFOLD: the host
//  entry, input contract (carried from v11), the engine gate, and the CUTLASS ex77 integration PLAN.
//  The `forward` below validates inputs then TORCH_CHECK(false, …) with a precise "implement me" pointer,
//  so (a) the wiring/build is testable on the box and (b) correctness tests fail loudly until the body
//  lands. Drop the real kernel into the marked region; do NOT silently return wrong numbers.
// ============================================================================================
//
//  THE BUILD (fork CUTLASS example 77, do NOT hand-roll tcgen05/tmem PTX):
//   1. Base = CUTLASS `examples/77_blackwell_fmha` — the weight-absorbed latent-512/rope-64 MLA *DECODE*
//      kernel (NOT the FA4 prefill kernel; it has no decode path). Use 2-SM `cta_group::2` for the
//      512-wide latent accumulator (ex77's reason for 2-SM — the latent doesn't fit one SM's tmem alone).
//      [verify the ex77 variant + CUTLASS version on the target box]
//   2. Arm 1 (engine=0, FP8) lands first: keep the v11 paged NVFP4 latent on HBM; dequant NVFP4→FP8 at
//      the SMEM stage (the dequant_nvfp4/dequant_e4m3 helpers below are the v11-faithful math), then feed
//      the tcgen05 FP8 MMA (gate M≥64). This is the FlashMLA-proven path. Scores/softmax stay ≥ FP16.
//   3. Arm 2 (engine=1, NVFP4) gated on Arm 1: `kind::mxf4nvf4`, M=128 / K=256 / TN-only + per-16 E4M3
//      microscales. HIGH-RISK (TRT-LLM #4412: FP4Gemm can be slower than FP8Gemm at decode M-sizes) —
//      keep the gate; it is sm_103a-specific (no mma.sync FP4 on datacenter Blackwell).
//
//  §9 Q1 (resolve BEFORE banking Arm 2): does M=128 pack as ONE tcgen05 GEMM, or fragment? CUTLASS ex77's
//  realized M-blocking `num_groups` is reportedly capped at 32, not 128 — verify the head-count→M=128
//  tile mapping on the target CUTLASS version. The absorbed-QK is Q'[128×576]·C[N_k×576]ᵀ (M=128); the
//  decoupled-RoPE q_R[128×64]·k_R[N_k×64]ᵀ accumulates into the SAME score → it splits the K-dimension
//  (an extra small-K=64 GEMM), NOT the M-tile. Confirm this structurally in the ex77 source.
//
// Layout (carried byte-identical from v11): q_absorbed dense [B,h_q,N_q,DQK] (FP16). L_pool
// [num_blocks,page_size,1,DQK/2] (packed E2M1 nibbles, uint8). L_scale [num_blocks,page_size,1,DQK/16]
// (E4M3 micro-scales, uint8). block_table int32 [B,n_logical]. Output O_latent [B,h_q,N_q,DV] (FP32).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace {

// The 8 E2M1 magnitudes — must match _E2M1_LEVELS in fa_kernels/paged.py (carried byte-identical from
// v11). The real kernel dequants NVFP4→FP8/bf16 at the SMEM stage using these before the tcgen05 MMA.
__device__ const float kE2M1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

__device__ __forceinline__ float dequant_nvfp4(uint8_t byte, int hi) {
    uint8_t nib = hi ? (byte >> 4) : (byte & 0xF);
    float v = kE2M1[nib & 0x7];
    return (nib & 0x8) ? -v : v;
}
__device__ __forceinline__ float dequant_e4m3(uint8_t b) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)));
}

// ----------------------------------------------------------------------------------------------------
//  TODO[GPU]: the tcgen05 partial + LSE-merge kernels (fork of CUTLASS ex77). Engine gate:
//    engine == 0 -> FP8-dense MMA (Arm 1, M≥64)
//    engine == 1 -> native NVFP4 block-scaled MMA (Arm 2, M≥128/K=256/TN; gated on sm_103a + Arm 1)
//  Keep split-KV / choose_splits / the [B,H_q,N_q,S,*] workspace / the LSE merge byte-identical to v11
//  (kernels/v11_mla/mla_attention.cu: mla_partial_kernel, mla_merge_kernel, choose_splits) so the only
//  changed variable is the inner matmul (CUDA-core GEMV -> tcgen05 MMA). Scores/softmax stay ≥ FP16.
// ----------------------------------------------------------------------------------------------------

}  // anonymous namespace

// Host entry: native tensor-core MLA latent-KV decode. Same I/O contract as v11's
// mla_attention_forward, plus `engine` (0 = FP8 Arm 1, 1 = NVFP4 Arm 2). kv_lora_rank = DV (PV/output
// width); DQK = q.size(3) (score width = kv_lora_rank + rope_dim). Output O_latent [B,h_q,N_q,DV] FP32.
torch::Tensor mla_tc_attention_forward(torch::Tensor q, torch::Tensor l_pool, torch::Tensor l_scale,
                                       torch::Tensor block_table, int64_t page_size, int64_t n_k,
                                       int64_t kv_lora_rank, double scale_l,
                                       double scale, bool causal, int64_t q_offset,
                                       int64_t engine) {
    TORCH_CHECK(q.is_cuda() && l_pool.is_cuda() && l_scale.is_cuda() && block_table.is_cuda(),
                "q, l_pool, l_scale, block_table must be CUDA tensors");
    TORCH_CHECK(q.dim() == 4, "q (q_absorbed) must be [B,H_q,N_q,DQK]");
    TORCH_CHECK(l_pool.dim() == 4, "l_pool must be [num_blocks, page_size, H_kv=1, DQK/2]");
    TORCH_CHECK(l_scale.dim() == 4, "l_scale must be [num_blocks, page_size, H_kv=1, DQK/16]");
    TORCH_CHECK(l_pool.scalar_type() == torch::kUInt8,
                "l_pool must be uint8 (packed NVFP4 nibbles); build with build_paged_kv_mla");
    TORCH_CHECK(l_scale.scalar_type() == torch::kUInt8, "l_scale must be uint8 (E4M3 micro-scale bytes)");
    TORCH_CHECK(block_table.dim() == 2 && block_table.scalar_type() == torch::kInt32,
                "block_table must be int32 [B, n_logical]");
    TORCH_CHECK(engine == 0 || engine == 1,
                "engine must be 0 (FP8 Arm 1) or 1 (NVFP4 Arm 2); got ", engine);

    const int B = q.size(0), H_q = q.size(1), N_q = q.size(2), DQK = q.size(3);
    const int H_kv = l_pool.size(2);
    const int DV = (int)kv_lora_rank;
    TORCH_CHECK(H_kv == 1, "MLA latent pool must have exactly one (latent) head; got H_kv=", H_kv);
    TORCH_CHECK(DQK % 16 == 0, "DQK (latent width) must be a multiple of the NVFP4 block (16); got ", DQK);
    TORCH_CHECK(DV >= 1 && DV <= DQK && DV % 32 == 0,
                "kv_lora_rank (DV) must be in [1,DQK], a multiple of 32; got ", DV, " DQK=", DQK);
    // Arm 2 (native NVFP4) gates M=H_q>=128 / K=256 / TN; Arm 1 (FP8) gates M>=64. The full gate +
    // sm_103a check lands with the kernel body. Flagged here so the contract is explicit.
    if (engine == 1) {
        TORCH_CHECK(H_q >= 128, "v12 Arm 2 (native NVFP4 MMA) needs M=H_q>=128 (the block-scaled "
                    "tcgen05 gate); got H_q=", H_q, " — use engine=0 (FP8) below 128");
    } else {
        TORCH_CHECK(H_q >= 16, "v12 Arm 1 (FP8 MMA) packs M=H_q; got H_q=", H_q);
    }

    // ============================================================================================
    //  *** IMPLEMENT THE tcgen05 KERNEL HERE (fork of CUTLASS ex77; see file header §"THE BUILD") ***
    //  Until then, fail loudly rather than return wrong numbers — the wiring + build are still testable.
    // ============================================================================================
    (void)page_size; (void)n_k; (void)scale_l; (void)scale; (void)causal; (void)q_offset; (void)DV;
    TORCH_CHECK(false,
        "v12_mla_tc kernel body is not yet implemented — this is the author-machine SCAFFOLD. Build + "
        "fill the tcgen05 MLA-decode kernel on a rented B300/sm_103a (fork CUTLASS example 77; Arm 1 = "
        "FP8 MMA, Arm 2 = native NVFP4). See kernels/v12_mla_tc/mla_tc_attention.cu header and "
        "docs/v12-kickoff.md.");
    return torch::Tensor();   // unreachable; silences the non-void return warning
}

"""Map a backend name to a compiled kernel, with hardware-capability gating.

A future inference engine calls `attention(q, k, v, backend="v5")` and this module decides
which compiled kernel actually runs, refusing backends the current GPU can't support (e.g.
asking for an fp8 backend on a Turing T4). For Phase 1 there is exactly one backend: v1_naive.
"""

from __future__ import annotations

import torch

from bindings.load import build_kernel

# Minimum compute capability (major, minor) each backend needs. The gate is intentionally
# explicit so an unsupported request fails loudly instead of producing garbage or a cryptic
# nvcc error. Capabilities grow as we add tensor-core / async / fp8 kernels.
_MIN_CAPABILITY = {
    "v1_naive": (7, 0),   # any CUDA GPU; pure FP32 CUDA-core math
    "v2_tiled": (7, 0),   # same FP32 math; shared-memory tiling needs no extra capability
    "v3_online": (7, 0),  # same FP32 math; online softmax needs no extra capability
    "v4_fused": (7, 0),   # same FP32 math; warp-per-row + shuffle reductions need no extra capability
    "v5_wmma": (7, 5),    # first cap bump: FP16-in/FP32-accum on Turing WMMA tensor cores (sm_75+)
    "v6_splitkv": (7, 0), # FP16-in/FP32-accum on CUDA cores (no WMMA at N_q=1); split-KV + LSE merge
    "v7_paged": (7, 0),   # v6 + block-table gather (paged KV) + causal query-offset; occupancy-neutral
    "v8_gqa": (7, 0),     # Cut 1: GQA M-packing on CUDA cores (sm_75/T4).
    "v8_gqa_tc": (7, 5),  # Cut 2a: GQA M-packing on Turing WMMA tensor cores (sm_75/T4, pad-G->16).
                          # Cut 2b's mma.m16n8k16 + cp.async peak version bumps a sibling to (8, 0).
    "v8_gqa_db": (7, 0),  # v8.5: GQA M-packing + portable double-buffered KV pipeline (sm_75/T4; ordinary
                          # ld.global prefetch, NOT cp.async). The schedule pass toward bandwidth-bound.
    "v8_gqa_occ": (7, 0), # v8.6 Arm 1: GQA M-packing + HALF-resident smem (16 KB -> 4 blocks/SM). The
                          # occupancy lever to hide the reduction latency (single-variable vs Cut 1).
    "v8_gqa_ilp": (7, 0), # v8.6 Arm 2: GQA M-packing + KU=4-unrolled key loop (pipeline independent shfl
                          # reductions). The ILP lever (FP32 smem, 2 blocks/SM unchanged) vs Cut 1.
    "v8_gqa_ss": (7, 0),  # v8.7: GQA M-packing + SCORE-STATIONARY inner loop (lane=key full dot, no per-key
                          # reduction; per-32-key-group softmax; PV transpose). FP16 transposed smem. The
                          # lever that REMOVES (not hides) the v8.6-measured reduction wall.
    "v9_fp8": (7, 0),     # v9: forks v8.7 + FP8 E4M3 KV storage (1 byte) with fused per-tile dequant. E4M3
                          # conversion is software-emulated on sm_75/T4 (cuda_fp8.h); decode uses no tensor
                          # cores, so (7, 0) holds. The byte lever (decode AI = 2G/b -> doubled).
    "v10_nvfp4": (7, 0),  # v10: forks v9 + NVFP4 KV storage (0.5625 byte: packed E2M1 nibble + per-16 E4M3
                          # micro-scale) with fused per-tile dequant-to-FP16. RECORD target is sm_103 (B300);
                          # (7, 0) is the T4-EMULATED fallback (store 4-bit, unpack in-kernel) for no-rental
                          # correctness + capacity + accuracy ONLY — emulated software unpack is MORE ALU than
                          # v9's E4M3, so T4 latency is NOT valid. Decode uses no FP4 MMA (M=G<64), so (7, 0)
                          # compiles. The byte lever (decode AI = 2G/b -> ~3.55x vs FP16).
    "v11_mla": (7, 0),    # v11: forks v10 changing the SHAPE — GQA-over-H_kv-heads -> MQA-over-ONE-shared-
                          # latent (M=G -> M=h_q). All h_q query heads share one latent (read once), so >1
                          # warp is active at N_q=1 (the per-CTA lever v10 proved is the decode wall) and AI
                          # rises 2G/b -> ~3.78*h_q/b. Default = CUDA-core / dequant-to-FP16 (NVFP4 latent
                          # storage carried byte-identical from v10), so (7, 0) builds on T4/Colab for the
                          # correctness + capacity + accuracy gate. The native FP4 tcgen05 compute arm
                          # (M>=128 ONE GEMM) would gate sm_103a and is deferred (kickoff §9 Q1/Q2).
}


def _current_capability() -> tuple[int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device visible; kernels run on the GPU (Colab T4 / rented box)")
    return torch.cuda.get_device_capability()


def get_backend(name: str):
    """Return the compiled module for `name`, or raise if this GPU can't run it."""
    if name not in _MIN_CAPABILITY:
        raise KeyError(f"unknown backend {name!r}; known: {sorted(_MIN_CAPABILITY)}")
    need = _MIN_CAPABILITY[name]
    have = _current_capability()
    if have < need:
        raise RuntimeError(
            f"backend {name!r} needs compute capability >= {need[0]}.{need[1]}, "
            f"but this GPU is {have[0]}.{have[1]}"
        )
    return build_kernel(name)

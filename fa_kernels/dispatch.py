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

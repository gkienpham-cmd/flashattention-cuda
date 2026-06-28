"""JIT compilation of versioned CUDA kernels via torch.utils.cpp_extension.

On Colab we don't ship a prebuilt .so — we compile the .cu/.cpp sources in-session the first
time a kernel is requested, then cache the loaded module so repeated calls are free. A later
phase (SSH/persistent box) can add a setup.py prebuilt path; the public API here stays the same.
"""

from __future__ import annotations

import os
from functools import lru_cache

# Each versioned kernel lives in kernels/<name>/ with a fixed file pair:
#   <name>.cu   — the CUDA kernels
#   binding.cpp — the pybind/torch glue exposing the forward entry point
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_KERNELS_DIR = os.path.join(_REPO_ROOT, "kernels")

# Per-version source manifest. Add a line here when a new kernel version lands.
_SOURCES = {
    "v1_naive": ["naive_attention.cu", "binding.cpp"],
    "v2_tiled": ["tiled_attention.cu", "binding.cpp"],
    "v3_online": ["online_attention.cu", "binding.cpp"],
    "v4_fused": ["fused_attention.cu", "binding.cpp"],
    "v5_wmma": ["wmma_attention.cu", "binding.cpp"],
    "v6_splitkv": ["splitkv_attention.cu", "binding.cpp"],
    "v7_paged": ["paged_attention.cu", "binding.cpp"],
    "v8_gqa": ["gqa_attention.cu", "binding.cpp"],
    "v8_gqa_tc": ["gqa_tc_attention.cu", "binding.cpp"],
    "v8_gqa_db": ["gqa_db_attention.cu", "binding.cpp"],
    "v8_gqa_occ": ["gqa_occ_attention.cu", "binding.cpp"],
    "v8_gqa_ilp": ["gqa_ilp_attention.cu", "binding.cpp"],
    "v8_gqa_ss": ["gqa_ss_attention.cu", "binding.cpp"],
    "v9_fp8": ["fp8_attention.cu", "binding.cpp"],
}


# Per-kernel gencode. The DEFAULT targets BOTH the Colab T4 (sm_75) and a rented A100 (sm_80) so any
# kernel runs on either box (a sm_75-only binary will NOT load on an A100 — different SASS, no JIT PTX).
# A kernel that uses arch-specific features overrides with its own list. The roofline arch constant and
# this gencode must agree (see roofline/archs.py: T4=sm_75, A100=sm_80).
_DEFAULT_ARCH = [
    "-gencode=arch=compute_75,code=sm_75",   # Tesla T4 (Turing) — the free Colab box
    "-gencode=arch=compute_80,code=sm_80",   # A100 (Ampere) — the v8 Cut 2b rental
]
_ARCH = {
    # Cut 2b's cp.async + mma.m16n8k16 path (when written) is Ampere-only -> sm_80 alone:
    # "v8_gqa_tc_sm80": ["-gencode=arch=compute_80,code=sm_80"],
}


@lru_cache(maxsize=None)
def build_kernel(name: str):
    """Compile (once) and return the loaded extension module for kernel version `name`.

    Cached: the expensive nvcc compile happens on the first call per session; later calls
    return the already-loaded module. Raises if torch/CUDA isn't available (i.e. not on a GPU).
    """
    # Imported lazily so that `import fa_kernels` works on a CPU-only authoring machine where
    # torch may be absent — only *building* a kernel requires the CUDA toolchain.
    from torch.utils.cpp_extension import load

    if name not in _SOURCES:
        raise KeyError(f"unknown kernel version {name!r}; known: {sorted(_SOURCES)}")

    src_dir = os.path.join(_KERNELS_DIR, name)
    sources = [os.path.join(src_dir, f) for f in _SOURCES[name]]

    return load(
        name=f"fa_{name}",
        sources=sources,
        extra_cuda_cflags=["-O3", *_ARCH.get(name, _DEFAULT_ARCH)],
        verbose=True,
    )

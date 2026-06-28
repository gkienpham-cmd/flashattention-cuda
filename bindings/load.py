"""JIT compilation of versioned CUDA kernels via torch.utils.cpp_extension.

On Colab we don't ship a prebuilt .so — we compile the .cu/.cpp sources in-session the first
time a kernel is requested, then cache the loaded module so repeated calls are free. A later
phase (SSH/persistent box) can add a setup.py prebuilt path; the public API here stays the same.
"""

from __future__ import annotations

import glob
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
    "v10_nvfp4": ["nvfp4_attention.cu", "binding.cpp"],
}


# Per-kernel gencode. Fallback DEFAULT targets BOTH the Colab T4 (sm_75) and a rented A100 (sm_80) so a
# binary built without a queryable GPU runs on either box. A kernel that uses arch-specific features
# overrides with its own list. The roofline arch constant and this gencode must agree (see
# roofline/archs.py: T4=sm_75, A100=sm_80, B200=sm_100, B300=sm_103).
_DEFAULT_ARCH = [
    "-gencode=arch=compute_75,code=sm_75",   # Tesla T4 (Turing) — the free Colab box
    "-gencode=arch=compute_80,code=sm_80",   # A100 (Ampere) — the v8 Cut 2b rental
]
_ARCH = {
    # Cut 2b's cp.async + mma.m16n8k16 path (when written) is Ampere-only -> sm_80 alone:
    # "v8_gqa_tc_sm80": ["-gencode=arch=compute_80,code=sm_80"],
}


def _detect_arch_flags():
    """Gencode for the CURRENT GPU. We build for the device actually present rather than a fixed
    multi-arch list because the Blackwell targets are CUDA-version-gated: sm_100 (B200) needs CUDA
    >=12.8 and sm_103 (B300) needs CUDA >=12.9, so putting compute_103 in a global default would BREAK
    the Colab-T4 build (its older nvcc rejects the unknown arch). Querying the device gives each box
    exactly the arch its toolchain supports: T4->compute_75, A100->compute_80, B200->compute_100,
    B300->compute_103. A trailing PTX (code=compute_XX) keeps it JIT-forward on a newer driver. Honors
    FA_CUDA_ARCH (e.g. '103' or '10.3') to force a target; falls back to _DEFAULT_ARCH if no GPU is
    queryable. Our kernels use no arch-specific (sm_103a) features, so plain sm_103 SASS compiles."""
    forced = os.environ.get("FA_CUDA_ARCH")
    cap = None
    if forced:
        cap = forced.replace(".", "")
    else:
        try:
            import torch
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability()
                cap = f"{major}{minor}"
        except Exception:
            cap = None
    if not cap:
        return _DEFAULT_ARCH
    return [
        f"-gencode=arch=compute_{cap},code=sm_{cap}",       # native SASS for this device
        f"-gencode=arch=compute_{cap},code=compute_{cap}",  # PTX fallback (JIT-forward safety)
    ]


def _cuda_lib_includes():
    """Include dirs for the CUDA-library headers torch's includes pull in (cusparse.h, cublas_v2.h,
    cusolverDn.h, ...). On a SLIM CUDA image whose /usr/local/cuda ships nvcc but NOT the dev headers
    (some vast.ai / cloud templates), the build dies with 'cusparse.h: No such file or directory'. The
    pip `nvidia-*-cu12` wheels that torch depends on DO ship those headers under
    site-packages/nvidia/<lib>/include, so add them to the include path. Returns [] (harmless) on a full
    `-devel` toolkit where /usr/local/cuda already has them, or when torch isn't importable."""
    try:
        import torch
        site = os.path.dirname(os.path.dirname(os.path.abspath(torch.__file__)))   # .../site-packages
    except Exception:
        return []
    return sorted(glob.glob(os.path.join(site, "nvidia", "*", "include")))


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

    arch_flags = _ARCH.get(name) or _detect_arch_flags()
    return load(
        name=f"fa_{name}",
        sources=sources,
        extra_cuda_cflags=["-O3", *arch_flags],
        extra_include_paths=_cuda_lib_includes(),
        verbose=True,
    )

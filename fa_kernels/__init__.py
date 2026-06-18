"""fa_kernels — FlashAttention from scratch, one measured speedup at a time.

Public API (stable across the journey; the kernel *behind* it changes per version):

    from fa_kernels import attention
    out = attention(q, k, v, causal=False, backend="v1_naive")

`q, k, v` are [B, H, N, d] tensors on a CUDA device. `backend` selects which versioned kernel
runs; as the roadmap progresses the default will advance to the current best version. A future
inference engine imports exactly this function.
"""

from __future__ import annotations

from .config import AttnConfig

__all__ = ["attention", "AttnConfig"]

# The default backend advances as the journey progresses. Today: the naive baseline.
_DEFAULT_BACKEND = "v1_naive"


def attention(q, k, v, *, causal: bool = False, scale: float | None = None,
              backend: str = _DEFAULT_BACKEND):
    """Scaled dot-product attention via a chosen hand-written kernel.

    q, k, v : [B, H, N, d] CUDA tensors (contiguous, same dtype the backend expects).
    causal  : apply a causal mask (no query attends to a future key).
    scale   : softmax scale; None -> 1/sqrt(head_dim).
    backend : kernel version, e.g. "v1_naive".
    """
    # Imported here (not at module top) so `import fa_kernels` succeeds on a CPU-only box;
    # dispatch pulls in torch + the CUDA toolchain only when you actually run a kernel.
    from .dispatch import get_backend

    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    module = get_backend(backend)
    return module.forward(q, k, v, scale, causal)

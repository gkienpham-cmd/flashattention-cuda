"""Correctness anchors.

Two references, used at different phases:
  - `sdpa_reference`: torch's scaled_dot_product_attention. The everyday correctness oracle
    for the FP16 fundamentals (Phases 1-2). Same precision class as our kernels.
  - `fp64_reference`: an attention computed entirely in float64. The *ground truth* for the
    quantized phase (Phase 3), where we report RMSE of a low-precision kernel against it,
    the way FlashAttention-3 validates its FP8 path. Overkill for FP16 work, essential for INT8.

Both take [B, H, N, d] tensors and return [B, H, N, d], matching our layout convention.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sdpa_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   *, causal: bool = False, scale: float | None = None) -> torch.Tensor:
    """Reference output via torch SDPA. This is what every Phase 1-2 test compares against."""
    # torch>=2.1 accepts an explicit `scale`; passing None lets it default to 1/sqrt(d),
    # which matches AttnConfig.effective_scale(), so the two agree by construction.
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)


def fp64_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   *, causal: bool = False, scale: float | None = None) -> torch.Tensor:
    """Ground-truth attention in float64. Used to measure quantization RMSE in Phase 3.

    Computed the long way on purpose: upcast everything, do an explicit, numerically careful
    softmax (subtract the row max before exp), and keep float64 the whole way through.
    """
    qd, kd, vd = q.double(), k.double(), v.double()
    d = qd.shape[-1]
    s = scale if scale is not None else 1.0 / (d ** 0.5)
    scores = torch.matmul(qd, kd.transpose(-1, -2)) * s          # [B,H,N,N] in fp64
    if causal:
        n_q, n_k = scores.shape[-2], scores.shape[-1]
        # mask[i,j] = True where j > i (a query may not attend to future keys)
        mask = torch.triu(torch.ones(n_q, n_k, dtype=torch.bool, device=scores.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    scores = scores - scores.amax(dim=-1, keepdim=True)          # stabilize before exp
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, vd)                               # [B,H,N,d] in fp64

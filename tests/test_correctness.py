"""Correctness of v1 naive attention vs torch SDPA.

Tolerance (documented per the per-step loop): v1 is FP32 in / FP32 accumulate / FP32 out, so
it should match SDPA's FP32 path tightly. We use atol=1e-4, rtol=1e-4 — looser than machine
epsilon only because the two implementations sum the reduction in a different order, which
perturbs the last few bits. If this ever needs loosening, that's a finding, not a knob to turn.

These tests require a GPU; on a CPU-only authoring box they skip.
"""

from __future__ import annotations

import pytest
import torch

from conftest import requires_cuda
from fa_kernels import attention
from fa_kernels.reference import sdpa_reference

# (B, H, N, d) shapes spanning the sweep we care about. Small + the head dims 64/128.
SHAPES = [
    (1, 4, 128, 64),
    (2, 8, 512, 64),
    (1, 8, 512, 128),
    (1, 2, 2048, 64),
]

ATOL, RTOL = 1e-4, 1e-4


@requires_cuda()
@pytest.mark.parametrize("B,H,N,d", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_v1_matches_sdpa(B, H, N, d, causal):
    torch.manual_seed(0)  # deterministic inputs so a failure is reproducible
    q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)

    out = attention(q, k, v, causal=causal, backend="v1_naive")
    ref = sdpa_reference(q, k, v, causal=causal)  # scale=None -> 1/sqrt(d), matches our default

    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


@requires_cuda()
def test_v1_explicit_scale():
    """A non-default scale must flow through identically to the reference."""
    torch.manual_seed(1)
    B, H, N, d = 1, 4, 256, 64
    q = torch.randn(B, H, N, d, device="cuda")
    k = torch.randn(B, H, N, d, device="cuda")
    v = torch.randn(B, H, N, d, device="cuda")
    scale = 0.1

    out = attention(q, k, v, scale=scale, backend="v1_naive")
    ref = sdpa_reference(q, k, v, scale=scale)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)

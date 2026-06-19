"""Correctness of the hand-written attention kernels vs torch SDPA.

Tolerance (documented per the per-step loop): v1 and v2 are both FP32 in / FP32 accumulate /
FP32 out, so they should match SDPA's FP32 path tightly. We use atol=1e-4, rtol=1e-4 — looser
than machine epsilon only because the two implementations sum the reduction in a different
order, which perturbs the last few bits. v2 (tiling) changes only the memory schedule, never
the math, so it holds the same tolerance as v1. If this ever needs loosening, that's a finding,
not a knob to turn.

These tests require a GPU; on a CPU-only authoring box they skip.
"""

from __future__ import annotations

import pytest
import torch

from conftest import requires_cuda
from fa_kernels import attention
from fa_kernels.reference import sdpa_reference

# (B, H, N, d) shapes spanning the sweep we care about. Small + the head dims 64/128. The last
# two have N that is NOT a multiple of v2's tile (64 at d=64, 32 at d=128) to exercise the
# partial-tile boundary guard in the tiled kernel; v1 (no tiling) treats them no differently.
SHAPES = [
    (1, 4, 128, 64),
    (2, 8, 512, 64),
    (1, 8, 512, 128),
    (1, 2, 2048, 64),
    (1, 2, 130, 64),    # 130 % 64 != 0  -> partial QK/PV tile at d=64
    (1, 2, 100, 128),   # 100 % 32 != 0  -> partial QK/PV tile at d=128
]

BACKENDS = ["v1_naive", "v2_tiled", "v3_online"]

ATOL, RTOL = 1e-4, 1e-4


@requires_cuda()
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("B,H,N,d", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_matches_sdpa(backend, B, H, N, d, causal):
    torch.manual_seed(0)  # deterministic inputs so a failure is reproducible
    q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)

    out = attention(q, k, v, causal=causal, backend=backend)
    ref = sdpa_reference(q, k, v, causal=causal)  # scale=None -> 1/sqrt(d), matches our default

    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


@requires_cuda()
@pytest.mark.parametrize("backend", BACKENDS)
def test_explicit_scale(backend):
    """A non-default scale must flow through identically to the reference."""
    torch.manual_seed(1)
    B, H, N, d = 1, 4, 256, 64
    q = torch.randn(B, H, N, d, device="cuda")
    k = torch.randn(B, H, N, d, device="cuda")
    v = torch.randn(B, H, N, d, device="cuda")
    scale = 0.1

    out = attention(q, k, v, scale=scale, backend=backend)
    ref = sdpa_reference(q, k, v, scale=scale)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


# v3-specific: the running-max rescale accumulates across the WHOLE key axis, so a long N is the
# stress case for online-softmax numerical stability — many more rescales than the short shapes
# above, and the only place a drifting (m, l) would show up. Restricted to v3_online: v1/v2
# materialize S and softmax it in one sweep (no streaming rescale to stress), and running the
# naive O(N^2 d) v1 at N=16384 would be punishingly slow for no extra coverage.
@requires_cuda()
@pytest.mark.parametrize("causal", [False, True])
def test_online_long_n_stability(causal):
    torch.manual_seed(2)
    B, H, N, d = 1, 2, 16384, 64
    q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)

    out = attention(q, k, v, causal=causal, backend="v3_online")
    ref = sdpa_reference(q, k, v, causal=causal)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)

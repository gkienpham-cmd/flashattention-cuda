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

BACKENDS = ["v1_naive", "v2_tiled", "v3_online", "v4_fused", "v5_wmma"]

# Default FP32 tolerance (v1-v4: FP32 in / FP32 accumulate / FP32 out).
ATOL, RTOL = 1e-4, 1e-4

# Per-backend tolerance. v5_wmma is the first version that is NOT bit-comparable to the FP32 SDPA
# reference: it casts the inputs to FP16 (so ~2^-11 relative error enters before the math), runs the
# matmuls on tensor cores, and only accumulates in FP32. The tight 1e-4 cannot hold — 2e-2 is the
# usual FP16-attention band. Per the per-step loop, a miss is a finding to investigate, not a knob to
# keep widening. v1-v4 keep the strict FP32 tolerance.
_TOL = {
    "v1_naive": (ATOL, RTOL),
    "v2_tiled": (ATOL, RTOL),
    "v3_online": (ATOL, RTOL),
    "v4_fused": (ATOL, RTOL),
    "v5_wmma": (2e-2, 2e-2),
}


def tol_for(backend: str):
    """(atol, rtol) for a backend — loosened only for the FP16-in v5 path."""
    return _TOL[backend]


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

    atol, rtol = tol_for(backend)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


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
    atol, rtol = tol_for(backend)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


# v3-specific: the running-max rescale accumulates across the WHOLE key axis, so a long N is the
# stress case for online-softmax numerical stability — many more rescales than the short shapes
# above, and the only place a drifting (m, l) would show up. Restricted to v3_online: v1/v2
# materialize S and softmax it in one sweep (no streaming rescale to stress), and running the
# naive O(N^2 d) v1 at N=16384 would be punishingly slow for no extra coverage.
@requires_cuda()
@pytest.mark.parametrize("causal", [False, True])
def test_v3_online_long_n_stability(causal):
    torch.manual_seed(2)
    B, H, N, d = 1, 2, 16384, 64
    q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)

    out = attention(q, k, v, causal=causal, backend="v3_online")
    ref = sdpa_reference(q, k, v, causal=causal)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


# v4-specific stress case: single-pass online softmax accumulates the O-RESCALE across the whole
# key axis (v3 dodged this by going two-pass — its pass 2 used the final (m, l), so O needed no
# correction). If v4 applies alpha=exp(m_old-m_new) to the running l but not to the partial O, or
# applies it AFTER the p*V add instead of before, the output drifts — and at N=16384 (the most
# rescales) is exactly where that drift becomes visible. d=128 also exercises the 4-elems-per-lane
# register slice. Restricted to v4_fused for the same reasons as the v3 case above.
@requires_cuda()
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_v4_fused_long_n_stability(causal, d):
    torch.manual_seed(3)
    B, H, N = 1, 2, 16384
    q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)

    out = attention(q, k, v, causal=causal, backend="v4_fused")
    ref = sdpa_reference(q, k, v, causal=causal)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


# v5-specific stress case: same single-pass O-rescale as v4, but now the running max/denom and the
# FP32 oRun accumulator are driven by FP16 tensor-core matmuls. N=16384 is the most rescales and the
# longest FP32 accumulation — exactly where an O-rescale ordering bug (alpha applied to l but not the
# smem O, or after the P@V add) or FP16 drift would show. d=128 also exercises the BM=32/2-warp tile
# config (vs d=64's BM=64/4-warp). Looser FP16 tolerance via tol_for("v5_wmma").
@requires_cuda()
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_v5_wmma_fp16_stability(causal, d):
    torch.manual_seed(4)
    B, H, N = 1, 2, 16384
    q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)

    out = attention(q, k, v, causal=causal, backend="v5_wmma")
    ref = sdpa_reference(q, k, v, causal=causal)
    atol, rtol = tol_for("v5_wmma")
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)

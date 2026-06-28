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
from fa_kernels import attention, gqa_attention, paged_attention
from fa_kernels.paged import build_paged_kv
from fa_kernels.reference import sdpa_reference, sdpa_reference_gqa

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

BACKENDS = ["v1_naive", "v2_tiled", "v3_online", "v4_fused", "v5_wmma", "v6_splitkv"]

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
    # v6_splitkv is also FP16-in/FP32-accum (research §8 tags v6 "FP16"), so it shares v5's band. The
    # extra approximation vs v5 is the cross-split LSE merge, but that combine is exact in FP32 — the
    # only new FP16 error is reading KV as half, identical to v5. Same 2e-2 holds.
    "v6_splitkv": (2e-2, 2e-2),
    # v8_gqa is the same FP16-in/FP32-accum precision class as v6/v7 (it carries their split-KV + LSE
    # merge byte-for-byte; M-packing changes only which warp owns which row, not the math). Same 2e-2.
    "v8_gqa": (2e-2, 2e-2),
    # v8_gqa_tc (Cut 2a) runs the same GQA M-packing on Turing WMMA tensor cores (FP16-in/FP32-accum,
    # pad-G->16). Same precision class as v5/v8 -> same 2e-2 band.
    "v8_gqa_tc": (2e-2, 2e-2),
    # v8_gqa_db (v8.5) is Cut 1 + a double-buffered KV pipeline (half-resident smem). Identical math
    # (same FP16 values, FP32 accumulate; conversion just moves to read-time) -> same 2e-2 band.
    "v8_gqa_db": (2e-2, 2e-2),
    # v8.6 Arm 1 (occupancy: half-resident smem -> 4 blocks/SM) and Arm 2 (key-ILP: KU=4-unrolled key
    # loop). Both carry Cut 1's GQA M-packing + split-KV + LSE merge unchanged; same FP16-in/FP32-accum
    # math (Arm 1 converts FP16->FP32 at read time like v8.5; Arm 2 keeps FP32 smem) -> same 2e-2 band.
    "v8_gqa_occ": (2e-2, 2e-2),
    "v8_gqa_ilp": (2e-2, 2e-2),
    # v8.7 score-stationary (lane=key full dot, per-32-key-group softmax, PV transpose). Same
    # FP16-in/FP32-accum precision class; the per-tile online softmax is FP32-rounding-equal to the
    # per-key form (max/sum associativity), not bit-identical -> same 2e-2 band.
    "v8_gqa_ss": (2e-2, 2e-2),
}

# GQA backends exercised by the v8 cases below: Cut 1 (CUDA-core), Cut 2a (Turing WMMA), and v8.5
# (double-buffered). All go through gqa_attention(..., backend=...) with an identical contract, so one
# test body covers all three.
GQA_BACKENDS = ["v8_gqa", "v8_gqa_tc", "v8_gqa_db", "v8_gqa_occ", "v8_gqa_ilp", "v8_gqa_ss"]


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


# v6-specific: the DECODE regime (N_q=1, N_k large) is exactly what split-KV exists for and what the
# square SHAPES above never exercise — there choose_splits picks num_splits=1 (base_blocks already
# fill the SMs), so v6 reduces to plain attention and the merge is trivial. At N_q=1 with a small head
# count choose_splits raises num_splits well above 1, so this is the only test that drives the real
# cross-split LSE merge: each block owns a KV chunk, emits an unnormalized (O, m, l), and the merge
# recombines. An N_k that is NOT a multiple of the chunk (8190) exercises the clamped last split and
# the empty-/short-split merge weights; causal also makes whole splits future-only (l=0 -> dropped in
# merge). FP16-in tolerance via tol_for("v6_splitkv").
@requires_cuda()
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("N_k", [4096, 8192, 8190])
@pytest.mark.parametrize("causal", [False, True])
def test_v6_splitkv_decode(causal, N_k, d):
    torch.manual_seed(5)
    B, H = 1, 8
    q = torch.randn(B, H, 1,   d, device="cuda", dtype=torch.float32)   # decode: one query token
    k = torch.randn(B, H, N_k, d, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, N_k, d, device="cuda", dtype=torch.float32)

    out = attention(q, k, v, causal=causal, backend="v6_splitkv")
    ref = sdpa_reference(q, k, v, causal=causal)
    atol, rtol = tol_for("v6_splitkv")
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


# v7-specific: PAGED KV. v7 carries v6's split-KV + LSE merge unchanged and adds exactly one thing in
# the hot loop — the KV reads GATHER through a per-sequence block table instead of a contiguous slice
# (plus a causal query-offset). The oracle: scatter a dense KV into SHUFFLED physical pages + the
# matching block table, run v7 on the paged layout, and compare to SDPA on the ORIGINAL dense KV. If
# the gather indexing is wrong the shuffle guarantees a mismatch (a contiguous read would grab the
# wrong tokens). FP16-in tolerance via tol_for("v6_splitkv") (v7 is the same precision class).
@requires_cuda()
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("N_k", [4096, 8192, 8190])          # 8190 -> partial last page (non-multiple)
@pytest.mark.parametrize("page_size", [128, 256])
@pytest.mark.parametrize("causal", [False, True])
def test_v7_paged_decode(causal, page_size, N_k, d):
    torch.manual_seed(6)
    B, H = 1, 8
    q = torch.randn(B, H, 1,   d, device="cuda", dtype=torch.float32)   # decode: one query token
    k = torch.randn(B, H, N_k, d, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, N_k, d, device="cuda", dtype=torch.float32)

    k_pool, v_pool, block_table, n_k = build_paged_kv(k, v, page_size, seed=6)
    # Causal decode: place the single query at logical N_k-1 (q_offset = n_k - N_q) so it attends the
    # WHOLE cache -> the result equals the non-causal full scan. (q_offset=0 with N_q=1 would attend
    # only key 0 -- the degenerate case the harness fix exists to kill.) Either way the reference is
    # SDPA over all keys.
    q_offset = (n_k - 1) if causal else 0
    out = paged_attention(q, k_pool, v_pool, block_table, page_size, n_k,
                          causal=causal, q_offset=q_offset)
    ref = sdpa_reference(q, k, v, causal=False)
    atol, rtol = tol_for("v6_splitkv")
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


# v7 regression: the square (prefill) shape must still reduce to num_splits=1 through the paged path,
# exactly as v6 does for the contiguous case (base_blocks already fill the SMs / N_k < MIN_CHUNK -> 1
# split, trivial merge). Tests both causal directions: at N_q=N_k with q_offset=0 the mask gj > gi is
# the standard upper-left causal SDPA uses, so it compares directly to SDPA causal.
@requires_cuda()
@pytest.mark.parametrize("causal", [False, True])
def test_v7_paged_square_reduces(causal):
    torch.manual_seed(7)
    B, H, N, d = 2, 8, 512, 64
    page_size = 128
    q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)

    k_pool, v_pool, block_table, n_k = build_paged_kv(k, v, page_size, seed=7)
    out = paged_attention(q, k_pool, v_pool, block_table, page_size, n_k, causal=causal, q_offset=0)
    ref = sdpa_reference(q, k, v, causal=causal)
    atol, rtol = tol_for("v6_splitkv")
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


# v8-specific: GQA M-PACKING. v8 carries v7's paged split-KV + LSE merge unchanged and changes exactly
# which work each warp owns — it packs the G = H_q/H_kv query heads that SHARE one KV head into the
# score GEMM's M dimension, so a CTA reads that KV head once and runs G query rows against it (G warps
# active, not 1). The oracle must broadcast-expand KV by G with repeat_interleave (sdpa_reference_gqa),
# matching the kernel's h_q = h_kv*G + g_local mapping; a plain `repeat` would be a silent-wrong oracle.
# The KV pool here has H_kv heads (fewer than q's H_q). N_k=8190 is non-multiple (partial last page +
# clamped last split under packing). FP16-in tolerance via tol_for("v8_gqa").
@requires_cuda()
@pytest.mark.parametrize("backend", GQA_BACKENDS)
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("N_k", [4096, 8190])               # 8190 -> non-multiple last page/split
@pytest.mark.parametrize("G", [1, 2, 4, 8])                 # group factor; G<8 pads/idles past M
@pytest.mark.parametrize("causal", [False, True])
def test_v8_gqa_decode(causal, G, N_k, d, backend):
    torch.manual_seed(8)
    B, H_kv = 1, 2
    H_q = G * H_kv
    page_size = 128
    q = torch.randn(B, H_q,  1,   d, device="cuda", dtype=torch.float32)   # decode: one query token
    k = torch.randn(B, H_kv, N_k, d, device="cuda", dtype=torch.float32)   # GQA: only H_kv KV heads
    v = torch.randn(B, H_kv, N_k, d, device="cuda", dtype=torch.float32)

    k_pool, v_pool, block_table, n_k = build_paged_kv(k, v, page_size, seed=8)
    # Causal decode: place the single query at logical N_k-1 (q_offset = n_k - N_q) so it attends the
    # WHOLE cache -> equals the non-causal full scan. Oracle = SDPA over KV expanded by G.
    q_offset = (n_k - 1) if causal else 0
    out = gqa_attention(q, k_pool, v_pool, block_table, page_size, n_k,
                        causal=causal, q_offset=q_offset, backend=backend)
    ref = sdpa_reference_gqa(q, k, v, causal=False)
    atol, rtol = tol_for(backend)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


# v8 idle-warp + multi-row-tile coverage: G=3 is non-power-of-2 and leaves 5 idle warps (M=3 < 8) —
# the `active = m_row < M` guard must drop them, NOT a `G % 8 == 0` assumption. G=16 forces M=16 > 8,
# so grid.x = ceil_div(16, 8) = 2 row-tiles per (batch, KV head): the block re-stages the KV head once
# per tile (still 8x fewer reads than v7's per-query-head read). Both must match the GQA oracle.
@requires_cuda()
@pytest.mark.parametrize("backend", GQA_BACKENDS)
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("G", [3, 16])
def test_v8_gqa_idle_warps_and_multiblock(G, d, backend):
    torch.manual_seed(80)
    B, H_kv, N_k = 1, 2, 8192
    H_q = G * H_kv
    page_size = 256
    q = torch.randn(B, H_q,  1,   d, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H_kv, N_k, d, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H_kv, N_k, d, device="cuda", dtype=torch.float32)

    k_pool, v_pool, block_table, n_k = build_paged_kv(k, v, page_size, seed=80)
    out = gqa_attention(q, k_pool, v_pool, block_table, page_size, n_k, causal=False, q_offset=0,
                        backend=backend)
    ref = sdpa_reference_gqa(q, k, v, causal=False)
    atol, rtol = tol_for(backend)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


# v8 regression: the square (prefill) shape must still reduce to num_splits=1 through the GQA path,
# exactly as v6/v7 do (base_blocks already fill the SMs -> 1 split, trivial merge). With M = G*N_q now
# large, choose_splits returns 1 and each packed query row does the full attention. q_offset=0 with
# N_q=N_k makes the mask gj > i_q the standard upper-left causal SDPA uses, so it compares directly to
# SDPA causal over the G-expanded KV.
@requires_cuda()
@pytest.mark.parametrize("backend", GQA_BACKENDS)
@pytest.mark.parametrize("causal", [False, True])
def test_v8_gqa_square_reduces(causal, backend):
    torch.manual_seed(81)
    B, H_kv, N, d = 2, 2, 512, 64
    G = 4
    H_q = G * H_kv
    page_size = 128
    q = torch.randn(B, H_q,  N, d, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H_kv, N, d, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H_kv, N, d, device="cuda", dtype=torch.float32)

    k_pool, v_pool, block_table, n_k = build_paged_kv(k, v, page_size, seed=81)
    out = gqa_attention(q, k_pool, v_pool, block_table, page_size, n_k, causal=causal, q_offset=0,
                        backend=backend)
    ref = sdpa_reference_gqa(q, k, v, causal=causal)
    atol, rtol = tol_for(backend)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)

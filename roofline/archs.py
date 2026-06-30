"""Per-architecture hardware constants for the roofline model.

These are the numbers the whole project argues against. Keep them honest and cite the source
in a comment; a wrong constant here makes every prediction wrong. We only fill in an arch when
we actually build on it (T4 now; A100/H100 when rented).

Units:
  hbm_bw_gbps        : GB/s  (10^9 bytes/s) of HBM bandwidth
  fp16_tc_flops      : FP16 tensor-core FLOP/s with FP32 accumulate
  fp32_cuda_flops    : FP32 CUDA-core FLOP/s (non-tensor)
  int8_tc_ops        : INT8 tensor-core OP/s
  smem_bw_gbps       : on-chip shared-memory bandwidth, aggregate across SMs (approx)
  mufu_ratio         : MUFU special-function throughput as a fraction of FP32 issue rate
                       (exp2/rcp run slower than plain FP ops; this is why softmax's exp matters)
  num_sm, smem_per_sm_kb, hbm_gb, boost_clock_mhz : descriptive, for bench logging
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Arch:
    name: str
    sm: str
    num_sm: int
    boost_clock_mhz: int
    hbm_gb: int
    hbm_bw_gbps: float
    smem_per_sm_kb: int
    smem_bw_gbps: float
    fp16_tc_flops: float
    fp32_cuda_flops: float
    int8_tc_ops: float
    mufu_ratio: float
    # Optional byte-lever peaks, only for arches where we measure them (B300 for the v9 FP8 / v10
    # NVFP4 steps). Default None so older entries (T4, A100) need not invent numbers. exp_per_s is
    # an absolute MUFU exp throughput where a vendor states it directly (B300); when None we fall
    # back to mufu_ratio * (fp32 FMA/s). l2_mb left None where there's no primary source.
    fp8_tc_flops: float | None = None
    fp4_tc_flops: float | None = None
    exp_per_s: float | None = None
    l2_mb: float | None = None

    def ridge_fp16_tc(self) -> float:
        """FLOP/byte where FP16 tensor-core compute stops being the bottleneck and HBM does."""
        return self.fp16_tc_flops / (self.hbm_bw_gbps * 1e9)

    def ridge_fp32_cuda(self) -> float:
        return self.fp32_cuda_flops / (self.hbm_bw_gbps * 1e9)


# Tesla T4, Turing sm_75. Source: NVIDIA T4 datasheet + CUDA programming guide (sm_75 row).
# - 65 FP16 TFLOPS (TC), 8.1 FP32 TFLOPS, 130 INT8 TOPS, 320 GB/s GDDR6, 40 SMs, 64KB smem/SM.
# - smem_bw is an estimate: ~32 banks * 4B * 1.59GHz * 40 SM ~ 8 TB/s aggregate; we use a
#   conservative 8000 GB/s and refine it against ncu later.
# - mufu_ratio ~0.25: Turing MUFU issues exp2/rcp at roughly 1/4 the FP32 rate per partition.
T4 = Arch(
    name="Tesla T4",
    sm="sm_75",
    num_sm=40,
    boost_clock_mhz=1590,
    hbm_gb=16,
    hbm_bw_gbps=320.0,
    smem_per_sm_kb=64,
    smem_bw_gbps=8000.0,
    fp16_tc_flops=65.0e12,
    fp32_cuda_flops=8.1e12,
    int8_tc_ops=130.0e12,
    mufu_ratio=0.25,
    l2_mb=4.0,                   # 4 MB L2 — the v9 regime-characterization hinges on this (decode KV
                                 # at N_k=8192/d=128/H_kv=1 ~= 4.2 MB ~= L2, the L2-residency confound)
)

# A100-SXM4-80GB, Ampere sm_80. Source: NVIDIA A100 datasheet + Ampere whitepaper (sm_80 row).
# - 312 FP16 TFLOPS (TC, FP32 accumulate, dense), 19.5 FP32 TFLOPS, 624 INT8 TOPS.
# - 2039 GB/s HBM2e (the 80GB SXM part; the 40GB part is 1555 GB/s — pick the rented SKU here).
# - 108 SMs, 164 KB max opt-in smem/SM (192 KB unified L1/smem).
# - smem_bw estimate: ~108 SM * 128 B/clk * 1.41 GHz ~ 19.5 TB/s aggregate; refine vs ncu.
# - mufu_ratio ~0.25: Ampere MUFU issues exp2/rcp at ~1/4 the FP32 rate per partition (as Turing).
A100 = Arch(
    name="NVIDIA A100-SXM4-80GB",
    sm="sm_80",
    num_sm=108,
    boost_clock_mhz=1410,
    hbm_gb=80,
    hbm_bw_gbps=2039.0,
    smem_per_sm_kb=164,
    smem_bw_gbps=19500.0,
    fp16_tc_flops=312.0e12,
    fp32_cuda_flops=19.5e12,
    int8_tc_ops=624.0e12,
    mufu_ratio=0.25,
    l2_mb=40.0,
)

# B200 (Blackwell), sm_100. An OPTIONAL cheaper dev rung for the v10 NVFP4 path (~$3.44/hr; most
# tcgen05/TMEM code ports sm_100 -> sm_103), NOT the destination — the record/paper runs on B300
# (sm_103), the project's final goal, because no published FA paper has characterized a B300 (FA4
# stops at B200). NOT yet measured by us. The one number that matters for the decode roofline is firm:
# HBM 8 TB/s (same flat 8 TB/s as B300). Compute peaks are
# vendor "dense" figures and SPECULATIVE placeholders — fix vs a primary spec sheet before quoting any
# sm_100 compute prediction. Decode is memory-bound regardless (AI << ridge), so the 8 TB/s drives the
# floor; FP8/FP4 dense are ~2/3 of B300's (Ultra is ~1.5x B200). num_sm / clock / L2 / smem now MEASURED
# on a vast.ai B200 (v10 B300-regime run, 2026-06-29) — see docs/results.md Step 10 B200 result.
B200 = Arch(
    name="NVIDIA B200 (Blackwell)",
    sm="sm_100",
    num_sm=148,                  # MEASURED (torch device props, 2026-06-29)
    boost_clock_mhz=1965,        # MEASURED max SM clock (nvidia-smi)
    hbm_gb=192,                  # MEASURED 191.5 GB
    hbm_bw_gbps=8000.0,          # firm: 8 TB/s (flat, same as B300)
    smem_per_sm_kb=228,          # MEASURED (CC 10.x = Hopper config)
    smem_bw_gbps=33000.0,        # SPECULATIVE estimate
    fp16_tc_flops=2.25e15,       # SPECULATIVE (~FP8 dense / 2); confirm vs spec sheet
    fp32_cuda_flops=60.0e12,     # SPECULATIVE placeholder
    int8_tc_ops=4.5e15,          # ~= FP8 dense
    mufu_ratio=0.25,             # SPECULATIVE placeholder
    fp8_tc_flops=4.5e15,         # SPECULATIVE: ~B300 FP8 (5 PF) scaled down; confirm
    fp4_tc_flops=9.0e15,         # SPECULATIVE: ~B300 NVFP4 (15 PF) scaled down; the v10 headline lever
    l2_mb=132.6,                 # MEASURED 132.6 MB (refutes the 192 MB aggregator claim; B300 same die)
)

# B300 (Blackwell Ultra), sm_103. THE FINAL GOAL — the v10/v11 record target and the paper's novelty
# (first open roofline-documented FA decode study on sm_103; FA4 stops at B200). Device constants now
# MEASURED on a vast.ai B300/sm_103 (2026-06-29): num_sm/clock/L2 below — see results.md Step 10 sm_103 record.
# B300-only levers the paper exploits: 2x exp/SFU throughput (exp_per_s below), 288 GB capacity, NVFP4
# 15 PF dense. Sources:
# NVIDIA Blackwell Ultra / GB300 briefings + docs/v7-deep-research.md. HBM bandwidth is the one
# number that matters for the decode roofline and it is firm: 8 TB/s, FLAT vs B200 (only capacity
# grew 192->288 GB). Constants refreshed by the v10 B300-research pass (2026-06-29, NVIDIA primary
# sources — Inside Blackwell Ultra blog, Blackwell Tuning Guide, GB300 NVL72 page); each field tagged
# FACT / LIKELY / UNCONFIRMED. The UNCONFIRMED ones (L2, clock, smem_bw, fp32) are MEASURED on the
# rental (notebooks/v10_b300_regime.ipynb §4 arch-measure). NOTE the INT8 gutting (~95% vs B200) — it
# strengthens the NVFP4-not-INT8 KV choice. Decode kernel compiles to plain sm_103 (no tcgen05); v11's
# native FP4 compute would need sm_103a. See docs/b300-decode-research.md + docs/v10-b300-runbook.md.
B300 = Arch(
    name="NVIDIA B300 (Blackwell Ultra, GB300)",
    sm="sm_103",
    num_sm=148,                  # MEASURED on a vast.ai B300/sm_103 (torch device props, 2026-06-29) —
                                 # refutes the 160 spec; same 148-SM die as the measured B200
    boost_clock_mhz=2032,        # MEASURED max SM clock (nvidia-smi, B300 sm_103) — below the 2600 estimate
    hbm_gb=288,                  # FACT (measured 287.4 GB usable)
    hbm_bw_gbps=8000.0,          # FACT: 8 TB/s, flat vs B200
    smem_per_sm_kb=228,          # MEASURED (CC 10.x = Hopper config; combined L1+smem 256 KB, TMEM 256 KB)
    smem_bw_gbps=36800.0,        # LIKELY (~230 GB/s/SM x 148, microbench-derived) — MEASURE
    fp16_tc_flops=2.5e15,        # LIKELY (BF16/FP16 dense; rack 180 dense / 72. NOT 2.25=B200, NOT 3.5=rumor)
    fp32_cuda_flops=105.0e12,    # UNCONFIRMED (20480 cores x 2 x ~2.6 GHz) — MEASURE with an FMA microkernel
    int8_tc_ops=0.15e15,         # LIKELY: GUTTED ~95% vs B200 (~150 TOPS) to fund the NVFP4 uplift
    mufu_ratio=0.25,             # placeholder; exp_per_s below is the firm softmax figure
    fp8_tc_flops=5.0e15,         # FACT: FP8 5 PF dense
    fp4_tc_flops=15.0e15,        # FACT: NVFP4 15 PF dense (the v10 headline lever). tcgen05 BLOCK-SCALED
                                 # NVFP4 MMA gate is M>=128 (K=256, TN-only) — NOT M>=64 (that is the
                                 # FP16/FP8/dense-non-scaled-FP4 floor). v11's MLA M=128 meets it; corrected
                                 # 2026-06-30 (v11 close-out C04). Cosmetic: MLA roofline uses fp16_tc_flops.
    exp_per_s=10.7e12,           # FACT (vendor peak). MEASURED achievable on a dependent-EX2 microkernel
                                 # = 5.33 TExp/s (0.50x of peak, 2026-06-29) — but mufu share of decode is
                                 # <3% so the 2x-exp lever barely moves M=1 decode regardless.
    l2_mb=132.6,                 # MEASURED 132.6 MB (B300 sm_103) — refutes the 192 MB aggregator; B200 same die
)

# Registry so other modules / bench logs can look an arch up by its sm string.
ARCHS = {a.sm: a for a in (T4, A100, B200, B300)}


def get_arch(sm: str) -> Arch:
    if sm not in ARCHS:
        raise KeyError(f"no roofline constants for {sm!r} yet (filled in when we build on it); "
                       f"have: {sorted(ARCHS)}")
    return ARCHS[sm]

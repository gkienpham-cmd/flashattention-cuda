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

# B300 (Blackwell Ultra), sm_103. The v9/v10 byte-lever target — NOT yet measured by us. Sources:
# NVIDIA Blackwell Ultra / GB300 briefings + docs/v7-deep-research.md. HBM bandwidth is the one
# number that matters for the decode roofline and it is firm: 8 TB/s, FLAT vs B200 (only capacity
# grew 192->288 GB). The dense-compute peaks are vendor "dense" figures; FP16/FP32 here are
# SPECULATIVE placeholders (derived ~FP8/2 and a rough CUDA-core estimate) and tagged as such —
# fix them against a primary spec sheet before quoting any sm_103 compute prediction. boost clock
# and smem/SM are likewise unconfirmed placeholders. exp_per_s is the stated 2x-exp throughput.
B300 = Arch(
    name="NVIDIA B300 (Blackwell Ultra, GB300)",
    sm="sm_103",
    num_sm=160,
    boost_clock_mhz=1800,        # SPECULATIVE placeholder — no confirmed boost clock
    hbm_gb=288,
    hbm_bw_gbps=8000.0,          # firm: 8 TB/s, flat vs B200 (deep-research verified)
    smem_per_sm_kb=228,          # SPECULATIVE placeholder (Hopper-class); confirm for Blackwell
    smem_bw_gbps=33000.0,        # SPECULATIVE estimate
    fp16_tc_flops=2.5e15,        # SPECULATIVE (~FP8 dense / 2); confirm vs spec sheet
    fp32_cuda_flops=80.0e12,     # SPECULATIVE placeholder
    int8_tc_ops=5.0e15,          # ~= FP8 dense
    mufu_ratio=0.25,             # SPECULATIVE placeholder; exp_per_s below is the firm figure
    fp8_tc_flops=5.0e15,         # deep-research: FP8 5 PF dense
    fp4_tc_flops=15.0e15,        # deep-research: NVFP4 15 PF dense (the v10 headline lever)
    exp_per_s=10.7e12,           # deep-research: 2x exp throughput (10.7 TeraExp/s)
    # l2_mb left None: no primary source.
)

# Registry so other modules / bench logs can look an arch up by its sm string.
ARCHS = {a.sm: a for a in (T4, A100, B300)}


def get_arch(sm: str) -> Arch:
    if sm not in ARCHS:
        raise KeyError(f"no roofline constants for {sm!r} yet (filled in when we build on it); "
                       f"have: {sorted(ARCHS)}")
    return ARCHS[sm]

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

# Registry so other modules / bench logs can look an arch up by its sm string.
ARCHS = {T4.sm: T4}


def get_arch(sm: str) -> Arch:
    if sm not in ARCHS:
        raise KeyError(f"no roofline constants for {sm!r} yet (filled in when we build on it); "
                       f"have: {sorted(ARCHS)}")
    return ARCHS[sm]

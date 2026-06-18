"""The roofline model for attention.

Given a problem shape and precision, estimate the time each of the three resources would take
*if it were the only bottleneck*, then the real lower bound is the max of the three and the
limiter is whichever resource owns that max. This is the first-principles prediction we record
BEFORE writing/optimizing a kernel, and check against ncu AFTER.

The three resources (see docs and roofline/archs.py):
  1. MMA compute   — the two matmuls QK^T and PV.
  2. HBM traffic   — bytes moved to/from global memory (this includes the naive S round-trip).
  3. MUFU exp      — the softmax exponentials on the special-function unit.

We model whole-problem attention (summed over B,H). `materialize_s=True` models the non-fused
versions (v1 naive, v2 tiled): the N_q x N_k score matrix round-trips HBM AND the matmul
operands are re-read from HBM `M*N*K / tile` times each, so `tile_m`/`tile_n` capture how much
shared-memory tiling cuts that redundant operand traffic (naive = 1x1 = each row re-read per
output element). Fused kernels set `materialize_s=False`: S stays on-chip and operands are read
~once. NOTE: the tile_m/tile_n operand-reuse term is why a naive kernel's true arithmetic
intensity is ~1/nbytes (deeply memory-bound), not the read-once ideal an earlier cut assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

from .archs import Arch

_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "fp8": 1, "int4": 0.5}


@dataclass
class RooflineEstimate:
    limiter: str            # "mma" | "hbm" | "mufu"
    seconds: float          # predicted lower-bound runtime = max of the three
    t_mma: float
    t_hbm: float
    t_mufu: float
    arithmetic_intensity: float   # total FLOPs / total HBM bytes
    ridge: float                  # arch ridge point for this precision (FLOP/byte)

    def utilization(self) -> dict[str, float]:
        """Fraction of the bound each resource would hit; the limiter is 1.0."""
        return {
            "mma": self.t_mma / self.seconds,
            "hbm": self.t_hbm / self.seconds,
            "mufu": self.t_mufu / self.seconds,
        }


def estimate(arch: Arch, *, B: int, H: int, N_q: int, N_k: int, d: int,
             precision: str = "fp16", materialize_s: bool = False,
             use_tensor_core: bool = True, tile_m: int = 1, tile_n: int = 1) -> RooflineEstimate:
    bh = B * H
    nbytes = _BYTES[precision]

    # --- compute: two matmuls, each 2*M*N*K FLOPs ---
    mma_flops = 2.0 * bh * N_q * N_k * d   # QK^T
    mma_flops += 2.0 * bh * N_q * N_k * d  # PV
    peak = arch.fp16_tc_flops if use_tensor_core else arch.fp32_cuda_flops
    # v1 is FP32 CUDA-core; later FP16 versions use the tensor-core peak.
    if precision == "fp32":
        peak = arch.fp32_cuda_flops
    t_mma = mma_flops / peak

    # --- HBM traffic ---
    # The output O is written exactly once in every version.
    o_write = bh * N_q * d * nbytes

    if materialize_s:
        # Non-fused versions (v1 naive, v2 tiled): the score matrix S round-trips HBM AND the
        # matmul operands get re-read from HBM, with the re-read count set by tiling. For a tiled
        # GEMM C[M,N] = A[M,K]·B[K,N] with output tile (tile_m x tile_n), each operand is read
        # M*N*K / tile times (A reused across tile_n output cols, B across tile_m output rows).
        # Naive = tile 1x1: each Q/K row re-read once per output element -> the real O(N^2 * d)
        # cost the first model wrongly ignored (it assumed read-once). Bigger tile = more on-chip
        # reuse = fewer HBM reads. Two matmuls, four operands: QK^T (contraction K=d) and PV
        # (contraction K=N_k); the per-operand counts are symmetric in tile_m/tile_n for both.
        reuse = (1.0 / tile_m + 1.0 / tile_n)
        qk_operand_reads = bh * N_q * N_k * d * reuse    # Q and K
        pv_operand_reads = bh * N_q * d * N_k * reuse    # P and V
        operand_bytes = (qk_operand_reads + pv_operand_reads) * nbytes
        # The S round-trip: pass1 writes S, pass2 reads+writes S, pass3 reads S (~4 sweeps over
        # the N_q x N_k matrix). Tiling does NOT remove this — only online softmax (v3) does.
        s_bytes = 4.0 * bh * N_q * N_k * nbytes
        hbm_bytes = operand_bytes + s_bytes + o_write
    else:
        # Fused versions (v3+ online softmax): S never touches HBM and streaming means the
        # operands are read ~once each. This is the read-once ideal the journey climbs toward.
        hbm_bytes = 3.0 * bh * N_k * d * nbytes + o_write
    t_hbm = hbm_bytes / (arch.hbm_bw_gbps * 1e9)

    # --- MUFU exp: one exp per score entry ---
    exp_count = float(bh) * N_q * N_k
    # MUFU op/s ~= (FP32 FMA/s) * ratio. fp32_cuda_flops counts 2 FLOPs per FMA, so /2.
    mufu_rate = (arch.fp32_cuda_flops / 2.0) * arch.mufu_ratio
    t_mufu = exp_count / mufu_rate

    times = {"mma": t_mma, "hbm": t_hbm, "mufu": t_mufu}
    limiter = max(times, key=times.get)
    ai = mma_flops / hbm_bytes
    ridge = arch.ridge_fp32_cuda() if precision == "fp32" else arch.ridge_fp16_tc()

    return RooflineEstimate(limiter=limiter, seconds=times[limiter],
                            t_mma=t_mma, t_hbm=t_hbm, t_mufu=t_mufu,
                            arithmetic_intensity=ai, ridge=ridge)

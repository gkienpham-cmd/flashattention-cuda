"""The single shared configuration object.

Every layer of the project reads from `AttnConfig` so nothing drifts: the kernels read
tile shapes from it, the tests build inputs from it, the bench harness sweeps over it, and
the roofline tool predicts the bottleneck from it. If a constant matters in more than one
place, it lives here.

Convention (fixed for the whole project, see docs/decisions.md):
  - Tensor layout is [B, H, N, d] row-major. B=batch, H=heads, N=seq len, d=head dim.
  - The FP16 fundamentals use low-precision inputs with FP32 accumulation. v1 is FP32 in/out.
  - scale defaults to 1/sqrt(d), the standard scaled-dot-product factor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class AttnConfig:
    # --- problem shape ---
    batch: int = 1
    n_heads: int = 8
    seq_len_q: int = 512
    seq_len_k: int = 512
    head_dim: int = 64

    # --- numerics ---
    # dtype is the *storage/compute input* dtype; accum_dtype is the accumulator. The whole
    # point of the FP16 phase is that these differ (fp16 in, fp32 accumulate). v1 uses fp32/fp32.
    dtype: str = "fp32"
    accum_dtype: str = "fp32"
    causal: bool = False
    # scale=None means "use 1/sqrt(head_dim)", computed lazily so it always matches head_dim.
    scale: Optional[float] = None

    # --- tiling (set per kernel version; the roofline tool reads these) ---
    # v1 naive has no tiling — it materializes the full S matrix — so tiles are unset (None).
    tile_m: Optional[int] = None
    tile_n: Optional[int] = None

    # --- hardware target ---
    # arch gates which backends are legal (e.g. v12_mla_tc needs sm_100+; v9_fp8 DOES run on sm_75 —
    # E4M3 is software-emulated there, see dispatch.py "v9_fp8"). See dispatch.py.
    arch: str = "sm_75"

    # fixed convention; kept as a field so it travels with the config in logs/results.
    layout: str = "BHND"

    def effective_scale(self) -> float:
        """The softmax scale actually applied: explicit value, else 1/sqrt(d)."""
        return self.scale if self.scale is not None else 1.0 / math.sqrt(self.head_dim)

    def shape(self) -> tuple[int, int, int, int]:
        """The query tensor shape [B, H, N_q, d]."""
        return (self.batch, self.n_heads, self.seq_len_q, self.head_dim)

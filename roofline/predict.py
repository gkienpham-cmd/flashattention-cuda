"""CLI front-end for the roofline model.

    python -m roofline.predict --arch sm_75 --shape 1x8x2048x64 --precision fp32 --materialize-s

prints the predicted limiter, the per-resource times, and the arithmetic intensity vs the
arch ridge point. Run this BEFORE coding a step to record the prediction; the per-step loop
then checks the measured ncu reading against it.
"""

from __future__ import annotations

import argparse

from .archs import get_arch
from .model import estimate


def predict_bottleneck(sm: str, B: int, H: int, N_q: int, N_k: int, d: int,
                       precision: str, materialize_s: bool, use_tensor_core: bool,
                       tile_m: int = 1, tile_n: int = 1, G: int = 1,
                       mla: bool = False, h_q: int = 128,
                       kv_lora_rank: int = 512, rope_dim: int = 64,
                       mma_engine: str = "fp16"):
    arch = get_arch(sm)
    return arch, estimate(arch, B=B, H=H, N_q=N_q, N_k=N_k, d=d,
                          precision=precision, materialize_s=materialize_s,
                          use_tensor_core=use_tensor_core, tile_m=tile_m, tile_n=tile_n, G=G,
                          mla=mla, h_q=h_q, kv_lora_rank=kv_lora_rank, rope_dim=rope_dim,
                          mma_engine=mma_engine)


def _parse_shape(s: str) -> tuple[int, int, int, int]:
    # "BxHxNxd", e.g. "1x8x2048x64"
    b, h, n, d = (int(x) for x in s.lower().split("x"))
    return b, h, n, d


def main() -> None:
    p = argparse.ArgumentParser(description="Predict the attention roofline bottleneck.")
    p.add_argument("--arch", default="sm_75", help="compute capability, e.g. sm_75")
    p.add_argument("--shape", default="1x8x2048x64", help="BxHxNxd (N used for both q and k)")
    p.add_argument("--precision", default="fp32",
                   choices=["fp32", "fp16", "bf16", "int8", "fp8", "nvfp4"])
    p.add_argument("--materialize-s", action="store_true",
                   help="model the non-fused S round-trip + tiled operand re-reads (v1/v2)")
    p.add_argument("--tile", default="1x1",
                   help="shared-memory tile MxN for operand reuse; 1x1 = naive (re-read per "
                        "output element), e.g. 64x64. Only affects the --materialize-s traffic.")
    p.add_argument("--no-tensor-core", action="store_true",
                   help="use CUDA-core peak instead of tensor-core peak for the MMA bound")
    p.add_argument("--kv-len", type=int, default=None, dest="kv_len",
                   help="decode: N_k (KV-cache length) separate from the shape's N (=N_q). Omit for "
                        "the square N_q=N_k case. Use e.g. --shape 8x8x1x128 --kv-len 8192 for decode.")
    p.add_argument("--gqa-group", type=int, default=1, dest="gqa_group",
                   help="GQA group factor G = H_q/H_kv (G query heads share one KV head). Decode AI "
                        "= 2G/b; G=1 is plain MHA. The v8 lever.")
    p.add_argument("--mla", action="store_true",
                   help="MLA (v11) latent-KV decode: all h_q query heads share ONE low-rank latent "
                        "read -> M=h_q, decode AI = 2*h_q*(2L+R)/((L+R)*b) ~= 3.78*h_q/b. Ignores "
                        "H/G/--tile; uses --h-q/--kv-lora-rank/--rope-dim. The v11 shape change.")
    p.add_argument("--h-q", type=int, default=128, dest="h_q",
                   help="MLA: number of query heads packed into M (default 128 = DeepSeek-V3).")
    p.add_argument("--kv-lora-rank", type=int, default=512, dest="kv_lora_rank",
                   help="MLA: latent content rank L (default 512 = DeepSeek-V2/V3).")
    p.add_argument("--rope-dim", type=int, default=64, dest="rope_dim",
                   help="MLA: decoupled-RoPE key dims R carried alongside the latent (default 64).")
    p.add_argument("--mma-engine", default="fp16", dest="mma_engine",
                   choices=["fp16", "fp8", "nvfp4"],
                   help="MLA only: the COMPUTE engine for the M=128 matmul, independent of --precision "
                        "(storage). fp16 = v11 dequant-to-fp16 (default). fp8/nvfp4 = v12 native "
                        "tcgen05 (Arm 1 / Arm 2): sets the ridge to the FP8-TC / NVFP4-TC peak.")
    a = p.parse_args()

    B, H, N, d = _parse_shape(a.shape)
    N_q = N
    N_k = a.kv_len if a.kv_len is not None else N
    tile_m, tile_n = (int(x) for x in a.tile.lower().split("x"))
    arch, est = predict_bottleneck(a.arch, B, H, N_q, N_k, d, a.precision,
                                   a.materialize_s, not a.no_tensor_core,
                                   tile_m=tile_m, tile_n=tile_n, G=a.gqa_group,
                                   mla=a.mla, h_q=a.h_q, kv_lora_rank=a.kv_lora_rank,
                                   rope_dim=a.rope_dim, mma_engine=a.mma_engine)

    util = est.utilization()
    print(f"arch        : {arch.name} ({arch.sm})")
    if a.mla:
        print(f"shape       : B={B} N_q={N_q} N_k={N_k}  MLA latent (h_q={a.h_q} "
              f"kv_lora_rank={a.kv_lora_rank} rope_dim={a.rope_dim})  storage={a.precision}  "
              f"engine={a.mma_engine}")
    else:
        print(f"shape       : B={B} H={H} N_q={N_q} N_k={N_k} d={d}  precision={a.precision}  "
              f"materialize_S={a.materialize_s}  tile={tile_m}x{tile_n}  G={a.gqa_group}")
    print(f"LIMITER     : {est.limiter.upper()}   (predicted lower bound {est.seconds*1e3:.3f} ms)")
    print(f"  t_mma     : {est.t_mma*1e3:8.3f} ms   util {util['mma']*100:5.1f}%")
    print(f"  t_hbm     : {est.t_hbm*1e3:8.3f} ms   util {util['hbm']*100:5.1f}%")
    print(f"  t_mufu    : {est.t_mufu*1e3:8.3f} ms   util {util['mufu']*100:5.1f}%")
    print(f"intensity   : {est.arithmetic_intensity:.1f} FLOP/byte   "
          f"(arch ridge {est.ridge:.1f}; {'BELOW -> memory-bound' if est.arithmetic_intensity < est.ridge else 'ABOVE -> compute-bound'})")


if __name__ == "__main__":
    main()

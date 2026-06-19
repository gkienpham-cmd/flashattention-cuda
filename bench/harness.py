"""Benchmark harness: our kernel vs torch SDPA across the seq x head-dim sweep.

Reports, per shape: p50 and max latency (ms), tokens/s, and the speedup vs SDPA. It also asks
the roofline tool for the predicted limiter so each printed row carries prediction-vs-reality
side by side (the honesty check from the per-step loop). Records GPU/arch/clocks so a results.md
row is reproducible; the free-tier T4 thermally throttles, so we note clocks at run time.

    python -m bench.harness --backend v1_naive --precision fp32

Comparisons against FlashAttention-2 and cuDNN are added in bench/compare.py when those steps
arrive; SDPA is the baseline that's always available.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess

import torch

from fa_kernels import attention
from fa_kernels.reference import sdpa_reference
from roofline.archs import get_arch
from roofline.model import estimate

SEQ_LENS = [512, 2048, 8192]
HEAD_DIMS = [64, 128]


def _time_ms(fn, *, warmup: int = 10, iters: int = 50) -> tuple[float, float]:
    """Return (p50, max) latency in ms using CUDA events. Synchronizes around each timed call.

    The tail number is the max (worst) of `iters` samples, not a true p99: at the default iters=50
    a 99th-percentile index rounds to the last element anyway, so we report it honestly as the max.
    Bump iters into the hundreds if a real percentile is ever wanted.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))  # ms
    samples.sort()
    p50 = statistics.median(samples)
    p_max = samples[-1]
    return p50, p_max


def _dtype(precision: str):
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]


def _sm_clock_mhz() -> tuple[int, int]:
    """Current and max SM clock (MHz) from nvidia-smi.

    torch.cuda.clock_rate() returns 0 on the Colab torch build, which silently dropped the
    throttling signal (it logged clock~0MHz). nvidia-smi reports it reliably, so we shell out.
    The current-vs-max gap is the whole point: on the free-tier T4 it idles far below the
    1590 MHz max, so every bench row should carry both numbers to be reproducible.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.current.sm,clocks.max.sm",
             "--format=csv,noheader,nounits"], text=True).splitlines()[0]
        cur, mx = (int(x) for x in out.split(","))
        return cur, mx
    except Exception:
        return -1, -1


def _roofline_tile(backend: str, d: int) -> tuple[int, int]:
    """The (tile_m, tile_n) the roofline model should use for this backend+head dim.

    v1 is naive: 1x1, i.e. each operand re-read per output element. v2 launches the smem tile
    chosen per head dim in tiled_attention.cu (64x64 at d=64, 32x32 at d=128); feeding the same
    tile here makes the printed roofline the tiled prediction (AI ~6-8), not the naive 0.2.
    v3 fuses softmax (materialize_s=False below), so the tile is irrelevant to its HBM traffic and
    this returns the 1x1 default — operand reuse is no longer a traffic factor once S is gone.
    """
    if backend == "v2_tiled":
        return (64, 64) if d == 64 else (32, 32)
    return (1, 1)


def run(backend: str, precision: str, B: int, H: int, causal: bool,
        seq_lens: list[int] | None = None, head_dims: list[int] | None = None) -> None:
    # Default to the full sweep; --seq/--dim narrow it so ncu can profile one shape's 3 passes.
    seq_lens = seq_lens or SEQ_LENS
    head_dims = head_dims or HEAD_DIMS
    dev = torch.device("cuda")
    cap = torch.cuda.get_device_capability()
    sm = f"sm_{cap[0]}{cap[1]}"
    name = torch.cuda.get_device_name()
    # Current/max SM clock (MHz) — captures throttling on the free tier at the moment of the run.
    clk_cur, clk_max = _sm_clock_mhz()
    print(f"# device: {name} ({sm})  clock~{clk_cur}/{clk_max}MHz  backend={backend}  precision={precision}  causal={causal}")
    print(f"# {'shape':>16} | {'ours p50/max ms':>18} | {'sdpa p50/max ms':>18} | "
          f"{'speedup':>8} | {'tok/s(ours)':>12} | roofline")

    arch = get_arch(sm) if sm in {"sm_75"} else None
    dt = _dtype(precision)

    for N in seq_lens:
        for d in head_dims:
            q = torch.randn(B, H, N, d, device=dev, dtype=dt)
            k = torch.randn(B, H, N, d, device=dev, dtype=dt)
            v = torch.randn(B, H, N, d, device=dev, dtype=dt)

            ours = lambda: attention(q, k, v, causal=causal, backend=backend)
            base = lambda: sdpa_reference(q, k, v, causal=causal)

            o_p50, o_max = _time_ms(ours)
            s_p50, s_max = _time_ms(base)
            speedup = s_p50 / o_p50
            tokens = B * H * N
            toks_s = tokens / (o_p50 / 1e3)

            roof = ""
            if arch is not None:
                # v1 and v2 both materialize S in HBM; they differ only in operand reuse, which
                # the model reads from the tile. v1 = naive 1x1 (re-read per output element);
                # v2 = the smem tile it actually launches (64x64 @ d=64, 32x32 @ d=128), so the
                # predicted AI reflects the ~30x traffic cut the row should show off. v3 fuses
                # softmax (materialize_s=False): S leaves HBM entirely, so the model drops to the
                # read-once ideal (AI ~1000+). NOTE the model counts ONE exp per score, but v3's
                # two-pass recomputes scores -> it does ~2x the exp work, so the printed MUFU bound
                # is optimistic; the ncu read is what settles MMA-vs-MUFU (the Step-2 discipline).
                tile_m, tile_n = _roofline_tile(backend, d)
                est = estimate(arch, B=B, H=H, N_q=N, N_k=N, d=d, precision=precision,
                               materialize_s=backend in ("v1_naive", "v2_tiled"),
                               tile_m=tile_m, tile_n=tile_n)
                roof = f"{est.limiter.upper()} (~{est.seconds*1e3:.2f}ms)"

            print(f"  {f'{B}x{H}x{N}x{d}':>16} | {o_p50:7.3f}/{o_max:7.3f} | "
                  f"{s_p50:7.3f}/{s_max:7.3f} | {speedup:7.2f}x | {toks_s:12.3e} | {roof}")


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark a kernel backend vs SDPA.")
    p.add_argument("--backend", default="v1_naive")
    p.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--seq", type=int, nargs="+", default=None,
                   help="override seq lengths to run (e.g. --seq 8192); default is the full sweep. "
                        "Use to profile one shape's qk/softmax/pv passes under ncu.")
    p.add_argument("--dim", type=int, nargs="+", default=None,
                   help="override head dims to run (e.g. --dim 64); default is the full sweep.")
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; run this on the GPU (Colab T4 / rented box)")
    run(a.backend, a.precision, a.batch, a.heads, a.causal, seq_lens=a.seq, head_dims=a.dim)


if __name__ == "__main__":
    main()

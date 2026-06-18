"""Benchmark harness: our kernel vs torch SDPA across the seq x head-dim sweep.

Reports, per shape: p50 and p99 latency (ms), tokens/s, and the speedup vs SDPA. It also asks
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

import torch

from fa_kernels import attention
from fa_kernels.reference import sdpa_reference
from roofline.archs import get_arch
from roofline.model import estimate

SEQ_LENS = [512, 2048, 8192]
HEAD_DIMS = [64, 128]


def _time_ms(fn, *, warmup: int = 10, iters: int = 50) -> tuple[float, float]:
    """Return (p50, p99) latency in ms using CUDA events. Synchronizes around each timed call."""
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
    p99 = samples[min(len(samples) - 1, int(0.99 * len(samples)))]
    return p50, p99


def _dtype(precision: str):
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]


def run(backend: str, precision: str, B: int, H: int, causal: bool) -> None:
    dev = torch.device("cuda")
    cap = torch.cuda.get_device_capability()
    sm = f"sm_{cap[0]}{cap[1]}"
    name = torch.cuda.get_device_name()
    # Current SM clock (MHz) — captures throttling on the free tier at the moment of the run.
    try:
        clock = torch.cuda.clock_rate() // 1000  # kHz -> MHz (torch>=2.3); best-effort
    except Exception:
        clock = -1
    print(f"# device: {name} ({sm})  clock~{clock}MHz  backend={backend}  precision={precision}  causal={causal}")
    print(f"# {'shape':>16} | {'ours p50/p99 ms':>18} | {'sdpa p50/p99 ms':>18} | "
          f"{'speedup':>8} | {'tok/s(ours)':>12} | roofline")

    arch = get_arch(sm) if sm in {"sm_75"} else None
    dt = _dtype(precision)

    for N in SEQ_LENS:
        for d in HEAD_DIMS:
            q = torch.randn(B, H, N, d, device=dev, dtype=dt)
            k = torch.randn(B, H, N, d, device=dev, dtype=dt)
            v = torch.randn(B, H, N, d, device=dev, dtype=dt)

            ours = lambda: attention(q, k, v, causal=causal, backend=backend)
            base = lambda: sdpa_reference(q, k, v, causal=causal)

            o_p50, o_p99 = _time_ms(ours)
            s_p50, s_p99 = _time_ms(base)
            speedup = s_p50 / o_p50
            tokens = B * H * N
            toks_s = tokens / (o_p50 / 1e3)

            roof = ""
            if arch is not None:
                est = estimate(arch, B=B, H=H, N_q=N, N_k=N, d=d, precision=precision,
                               materialize_s=(backend == "v1_naive"))
                roof = f"{est.limiter.upper()} (~{est.seconds*1e3:.2f}ms)"

            print(f"  {f'{B}x{H}x{N}x{d}':>16} | {o_p50:7.3f}/{o_p99:7.3f} | "
                  f"{s_p50:7.3f}/{s_p99:7.3f} | {speedup:7.2f}x | {toks_s:12.3e} | {roof}")


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark a kernel backend vs SDPA.")
    p.add_argument("--backend", default="v1_naive")
    p.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--causal", action="store_true")
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; run this on the GPU (Colab T4 / rented box)")
    run(a.backend, a.precision, a.batch, a.heads, a.causal)


if __name__ == "__main__":
    main()

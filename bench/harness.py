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

from fa_kernels import attention, paged_attention
from fa_kernels.paged import build_paged_kv
from fa_kernels.reference import sdpa_reference
from roofline.archs import get_arch
from roofline.model import estimate

SEQ_LENS = [512, 2048, 8192]
HEAD_DIMS = [64, 128]
DECODE_KV_LENS = [2048, 8192, 16384]   # KV-cache lengths swept in --decode mode (N_q collapses to 1)


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


def run(backend: str, precision: str, batches: list[int], H: int, causal: bool,
        seq_lens: list[int] | None = None, head_dims: list[int] | None = None,
        decode: bool = False, q_len: int | None = None, page_size: int = 256) -> None:
    # Default to the full sweep; --seq/--dim narrow it so ncu can profile one shape's 3 passes.
    # In --decode mode the swept N is the KV length and the query length collapses to q_len (1) — the
    # regime v6/v7 decode targets, where the metric shifts to us/token + % HBM bandwidth (research §8).
    # `batches` sweeps B: at decode, BH = B*H is the occupancy knob, so a multi-B sweep MEASURES the
    # occupancy->bandwidth crossover (%HBM climbing as BH passes ~2*SM) that the re-plan only predicted.
    paged = backend == "v7_paged"
    seq_lens = seq_lens or (DECODE_KV_LENS if decode else SEQ_LENS)
    head_dims = head_dims or HEAD_DIMS
    nq = (q_len if q_len is not None else (1 if decode else None))   # None -> square (N_q = N_k)
    dev = torch.device("cuda")
    cap = torch.cuda.get_device_capability()
    sm = f"sm_{cap[0]}{cap[1]}"
    name = torch.cuda.get_device_name()
    # Current/max SM clock (MHz) — captures throttling on the free tier at the moment of the run.
    clk_cur, clk_max = _sm_clock_mhz()
    print(f"# device: {name} ({sm})  clock~{clk_cur}/{clk_max}MHz  backend={backend}  precision={precision}  causal={causal}  decode={decode}")
    if decode:
        print(f"# {'shape(q x kv)':>16} | {'ours p50/max ms':>18} | {'us/tok':>8} | "
              f"{'%HBM':>6} | {'vs sdpa':>8} | {'vs naive':>8} | roofline")
    else:
        print(f"# {'shape':>16} | {'ours p50/max ms':>18} | {'sdpa p50/max ms':>18} | "
              f"{'speedup':>8} | {'tok/s(ours)':>12} | roofline")

    arch = get_arch(sm) if sm in {"sm_75"} else None
    dt = _dtype(precision)

    for B in batches:
      for N in seq_lens:
        for d in head_dims:
            qn = nq if nq is not None else N
            q = torch.randn(B, H, qn, d, device=dev, dtype=dt)
            k = torch.randn(B, H, N, d, device=dev, dtype=dt)
            v = torch.randn(B, H, N, d, device=dev, dtype=dt)

            base = lambda: sdpa_reference(q, k, v, causal=causal)
            if paged:
                # v7 reads a PAGED KV: scatter the dense k,v into shuffled physical pages + a block
                # table, then time the gather kernel on that layout. Decode causal places the single
                # query at logical N_k-1 (q_offset = N_k - qn) so it attends the WHOLE cache instead of
                # the degenerate 1-key short-circuit -> the causal-decode rows become meaningful.
                k_pool, v_pool, block_table, n_k = build_paged_kv(k, v, page_size)
                q_off = (N - qn) if causal else 0
                ours = lambda: paged_attention(q, k_pool, v_pool, block_table, page_size, n_k,
                                               causal=causal, q_offset=q_off)
            else:
                ours = lambda: attention(q, k, v, causal=causal, backend=backend)

            o_p50, o_max = _time_ms(ours)
            s_p50, s_max = _time_ms(base)
            speedup = s_p50 / o_p50
            tokens = B * H * qn
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
                # v6 (and v5) read FP16 KV regardless of the input dtype (they cast in the kernel),
                # so the roofline byte count uses fp16 -> at N_q=1 this is the AI=2/b=1.0 decode bound.
                tile_m, tile_n = _roofline_tile(backend, d)
                rp = "fp16" if backend in ("v5_wmma", "v6_splitkv", "v7_paged") else precision
                est = estimate(arch, B=B, H=H, N_q=qn, N_k=N, d=d, precision=rp,
                               materialize_s=backend in ("v1_naive", "v2_tiled"),
                               tile_m=tile_m, tile_n=tile_n)
                roof = f"{est.limiter.upper()} (~{est.seconds*1e3:.2f}ms)"

            if decode:
                # Decode headline metrics (research §8): us/token, % of peak HBM bandwidth achieved on
                # the K+V read (2*B*H*N*d*b bytes, b=2 for fp16 KV), and the speedup over a NAIVE
                # seqlen_q=1 loop (v5 run at N_q=1, the 1xBH single-block-per-head schedule v6 must beat).
                us_tok = o_p50 * 1e3 / tokens
                if arch is not None:
                    kv_bytes = 2.0 * B * H * N * d * 2  # K and V, fp16
                    hbm_pct = (kv_bytes / (o_p50 / 1e3)) / (arch.hbm_bw_gbps * 1e9) * 100.0
                else:
                    hbm_pct = float("nan")
                try:
                    naive = lambda: attention(q, k, v, causal=causal, backend="v5_wmma")
                    n_p50, _ = _time_ms(naive)
                    vs_naive = n_p50 / o_p50
                except Exception:
                    vs_naive = float("nan")
                print(f"  {f'{B}x{H}x{qn}x{d}/{N}':>16} | {o_p50:7.3f}/{o_max:7.3f} | {us_tok:8.2f} | "
                      f"{hbm_pct:5.1f}% | {speedup:7.2f}x | {vs_naive:7.2f}x | {roof}")
            else:
                print(f"  {f'{B}x{H}x{N}x{d}':>16} | {o_p50:7.3f}/{o_max:7.3f} | "
                      f"{s_p50:7.3f}/{s_max:7.3f} | {speedup:7.2f}x | {toks_s:12.3e} | {roof}")


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark a kernel backend vs SDPA.")
    p.add_argument("--backend", default="v1_naive")
    p.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--batch-sweep", type=int, nargs="+", default=None, dest="batch_sweep",
                   help="decode only: sweep B over these values (e.g. --batch-sweep 1 8 16 32 64) to "
                        "MEASURE the occupancy->bandwidth crossover (%%HBM vs BH=B*H). Overrides "
                        "--batch; pair with --seq 8192 to hold N_k fixed. The crown-jewel v7 deliverable.")
    p.add_argument("--page-size", type=int, default=256, dest="page_size",
                   help="page size (tokens per block) for the v7_paged backend's block table.")
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--seq", type=int, nargs="+", default=None,
                   help="override seq lengths to run (e.g. --seq 8192); default is the full sweep. "
                        "Use to profile one shape's qk/softmax/pv passes under ncu.")
    p.add_argument("--dim", type=int, nargs="+", default=None,
                   help="override head dims to run (e.g. --dim 64); default is the full sweep.")
    p.add_argument("--decode", action="store_true",
                   help="decode regime: query length collapses to --q-len (default 1) and the swept "
                        "--seq is the KV length. Reports us/token, %%HBM bandwidth, and the speedup "
                        "over a naive seqlen_q=1 loop (v5_wmma). This is v6 split-KV's home turf.")
    p.add_argument("--q-len", type=int, default=None, dest="q_len",
                   help="query length in --decode mode (default 1); ignored without --decode.")
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; run this on the GPU (Colab T4 / rented box)")
    batches = a.batch_sweep if a.batch_sweep else [a.batch]
    run(a.backend, a.precision, batches, a.heads, a.causal, seq_lens=a.seq, head_dims=a.dim,
        decode=a.decode, q_len=a.q_len, page_size=a.page_size)


if __name__ == "__main__":
    main()

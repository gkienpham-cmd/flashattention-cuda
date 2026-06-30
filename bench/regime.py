"""v9 Task 1 — regime characterization: is decode HBM-bandwidth-bound, per-CTA-bound, or just L2-resident?

NOT a new kernel — a measurement. Since v6 every decode step read "~10% HBM, per-CTA-bound," but the v8
deep-research close-out (C12) showed that verdict is CONFOUNDED: at the bench sizes the KV cache fits in
the T4's 4 MB L2, so the DRAM counter reads low even if the kernel is memory-bound (bound by the *wrong*
memory), and free Colab never locked clocks. This module earns the verdict the honest way:

  - LOCK CLOCKS (so wall-times are comparable across runs; needs root).
  - FLUSH L2 between timed iters (so each launch's KV reads actually miss L2; the jan.ai technique).
  - SWEEP N_k 1K..128K so the KV working set crosses the 4 MB L2 (isolation B=1,H_kv=1,d=128 -> ~8K).
  - COUNTER-FREE L2 TEST: effective_bw = kv_bytes / time; if it EXCEEDS HBM peak the data physically
    could not have come from HBM -> it was L2-served, so %HBM is meaningless as a boundedness metric.

The decisive read is HBM% vs N_k: climbs toward the achievable ceiling as the working set passes L2 ->
genuinely memory-bound (FP8's byte win vindicated); stays ~10% with l2_served=False past L2 -> per-CTA-
bound, confound-free at last. Either is a first-class result. Spec: docs/v9-kickoff.md Task 1.

Run:  python -m bench.regime --backend v8_gqa_ss --lock-clocks
      python -m bench.regime --backend v9_fp8   --lock-clocks
      python -m bench.regime --profile 1,1,16384,128 --backend v8_gqa_ss   # one shape, for ncu

NOTE: torch + fa_kernels are imported INSIDE the functions (not at module top) so `--help` and a syntax
smoke-test work on a CUDA-less author box. Only the actual sweep needs a GPU.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess


# --------------------------------------------------------------------------------------------------
# Clock locking (root only). nvidia-smi is the same tool harness._sm_clock_mhz() already shells out to;
# we reimplement the tiny query here so importing this module never pulls bench.harness (-> torch).
# --------------------------------------------------------------------------------------------------
def _sm_clock_mhz() -> tuple[int, int]:
    """(current, max) SM clock in MHz from nvidia-smi; (-1,-1) if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.current.sm,clocks.max.sm",
             "--format=csv,noheader,nounits"], text=True).splitlines()[0]
        cur, mx = (int(x) for x in out.split(","))
        return cur, mx
    except Exception:
        return -1, -1


def _throttle_reasons() -> str:
    """Active clock-throttle reasons (HW slowdown / thermal / power); '' if none or unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-q", "-d", "PERFORMANCE"], text=True)
        active = [ln.strip() for ln in out.splitlines()
                  if ": Active" in ln and "Idle" not in ln]
        return "; ".join(active)
    except Exception:
        return ""


def lock_clocks(boost: bool = True) -> tuple[bool, int, int, str]:
    """Pin SM + memory clocks to max so wall-times are comparable across runs. Needs root.

    Returns (ok, sm_mhz, mem_mhz, throttle). On a non-root box the nvidia-smi calls fail with a
    permission error — we print a LOUD warning and return ok=False so the caller still runs the
    counter-free sweep (the %HBM/eff_bw signal survives; only cross-run wall-time comparison doesn't).
    """
    _, sm_max = _sm_clock_mhz()
    try:
        # max memory clock (last value reported by --query-supported-clocks=mem)
        mem_clocks = subprocess.check_output(
            ["nvidia-smi", "--query-supported-clocks=mem", "--format=csv,noheader,nounits"],
            text=True).split()
        mem_max = max(int(x) for x in mem_clocks) if mem_clocks else 0
        subprocess.run(["nvidia-smi", "-pm", "1"], check=True, capture_output=True)
        subprocess.run(["nvidia-smi", f"--lock-gpu-clocks={sm_max}"], check=True, capture_output=True)
        if mem_max:
            subprocess.run(["nvidia-smi", f"--lock-memory-clocks={mem_max}"], check=True,
                           capture_output=True)
        cur, _ = _sm_clock_mhz()
        thr = _throttle_reasons()
        print(f"# clocks LOCKED: sm={cur}MHz (target {sm_max}) mem={mem_max}MHz"
              + (f"  THROTTLE: {thr}" if thr else "  (no throttle)"))
        return True, cur, mem_max, thr
    except Exception as e:
        print("# !!! CLOCKS NOT LOCKED (need root / bare-metal) — cross-run wall-times are CONFOUNDED.")
        print(f"#     ({type(e).__name__}: run on a root T4 to lock; the counter-free sweep still runs.)")
        cur, _ = _sm_clock_mhz()
        return False, cur, 0, _throttle_reasons()


def reset_clocks() -> None:
    """Undo lock_clocks() — reset GPU + memory clocks and disable persistence mode. Best-effort."""
    for args in (["-rgc"], ["-rmc"], ["-pm", "0"]):
        try:
            subprocess.run(["nvidia-smi", *args], check=True, capture_output=True)
        except Exception:
            pass
    print("# clocks reset.")


# --------------------------------------------------------------------------------------------------
# L2-flushing timer. Mirrors harness._time_ms but scrubs L2 BEFORE each timed launch (outside the
# CUDA-event window) so the kernel's KV reads actually miss L2 and have to hit HBM.
# --------------------------------------------------------------------------------------------------
def _l2_flush_buf():
    """A >=2x-L2 scratch buffer; zeroing it evicts the kernel's KV from L2 (the jan.ai technique)."""
    import torch

    n = torch.cuda.get_device_properties(0).L2_cache_size
    return torch.empty(2 * int(n), dtype=torch.uint8, device="cuda")


def time_ms_l2flush(fn, flush_buf=None, *, warmup: int = 10, iters: int = 50) -> tuple[float, float]:
    """(p50, max) ms via CUDA events. If `flush_buf` is given, zero it before each iter — stream
    ordering runs the eviction memset before the kernel, but it sits OUTSIDE the [start,end] window,
    so the timing measures the kernel against a cold L2."""
    import torch

    for _ in range(warmup):
        if flush_buf is not None:
            flush_buf.zero_()
        fn()
    torch.cuda.synchronize()
    samples = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        if flush_buf is not None:
            flush_buf.zero_()          # enqueued before start.record() -> evicts L2, not timed
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return statistics.median(samples), samples[-1]


# --------------------------------------------------------------------------------------------------
# The sweep.
# --------------------------------------------------------------------------------------------------
# v11 MLA: --dim is the latent score-width DQK; this maps it to the content/output width DV
# (kv_lora_rank). The trailing DQK-DV dims are the decoupled RoPE. Real DeepSeek-V3 = 576/512.
_MLA_DV = {96: 64, 160: 128, 576: 512}


def _build_ours(backend, q, k, v, page_size, q_off, engine="fp8"):
    """Construct the timed decode-kernel lambda for `backend`, plus a human note. FP8 pools are uint8;
    NVFP4 pools are packed nibbles + per-16 E4M3 micro-scales (both uint8). For v11_mla/v12_mla_tc, `k`
    carries the single latent [B,1,N,DQK] (`v` unused) and the DV is read from _MLA_DV[DQK=k.shape[-1]].
    `engine` selects the v12 tcgen05 arm (fp8 / nvfp4); ignored by every other backend."""
    from fa_kernels import fp8_attention, gqa_attention, mla_attention, mla_tc_attention, nvfp4_attention
    from fa_kernels.paged import (build_paged_kv, build_paged_kv_fp8, build_paged_kv_mla,
                                  build_paged_kv_nvfp4)

    if backend == "v9_fp8":
        kp, vp, bt, nk, sk, sv = build_paged_kv_fp8(k, v, page_size)
        return lambda: fp8_attention(q, kp, vp, bt, page_size, nk, sk, sv,
                                     causal=False, q_offset=q_off, backend=backend)
    if backend == "v10_nvfp4":
        kp, km, vp, vm, bt, nk, sk, sv = build_paged_kv_nvfp4(k, v, page_size)
        return lambda: nvfp4_attention(q, kp, km, vp, vm, bt, page_size, nk, sk, sv,
                                       causal=False, q_offset=q_off, backend=backend)
    if backend == "v11_mla":
        latent = k                                       # [B,1,N,DQK] — the shared latent (one head)
        DV = _MLA_DV[latent.shape[-1]]
        lp, lm, bt, nk, sl = build_paged_kv_mla(latent, page_size)
        return lambda: mla_attention(q, lp, lm, bt, page_size, nk, DV, sl,
                                     causal=False, q_offset=q_off, backend=backend)
    if backend == "v12_mla_tc":
        latent = k                                       # SAME latent bytes as v11 (engine-only A/B)
        DV = _MLA_DV[latent.shape[-1]]
        lp, lm, bt, nk, sl = build_paged_kv_mla(latent, page_size)
        return lambda: mla_tc_attention(q, lp, lm, bt, page_size, nk, DV, sl, engine=engine,
                                        causal=False, q_offset=q_off, backend=backend)
    kp, vp, bt, nk = build_paged_kv(k, v, page_size)
    return lambda: gqa_attention(q, kp, vp, bt, page_size, nk,
                                 causal=False, q_offset=q_off, backend=backend)


def sweep(backend: str, kv_lens, batches, head_dims, h_kvs, *, gqa_group: int = 1,
          page_size: int = 256, max_ws_gb: float = 8.0, l2_flush: bool = True,
          warmup: int = 10, iters: int = 50, engine: str = "fp8") -> list[dict]:
    """Clock-locked (if the caller locked), L2-flushed KV-length sweep. Returns structured rows so a
    notebook can plot HBM%/eff_bw vs N_k in-process. KV is FP16 (b=2) for v8_gqa_ss, FP8 (b=1) for
    v9_fp8 — the L2 crossing for FP8 sits at ~2x the N_k (a built-in cross-check)."""
    import torch

    from roofline.archs import get_arch

    is_fp8 = backend == "v9_fp8"
    is_nvfp4 = backend == "v10_nvfp4"
    is_mla = backend in ("v11_mla", "v12_mla_tc")   # MLA: ONE shared latent (read once), --dim = DQK; NVFP4-stored
    # KV bytes/elem: FP16=2, FP8=1, NVFP4=0.5625 (4-bit nibble + per-16 E4M3 micro-scale — count the
    # scale, or the working-set/eff_bw/%HBM numbers and the L2-crossing point are wrong).
    b_bytes = 0.5625 if (is_nvfp4 or is_mla) else (1 if is_fp8 else 2)
    dev = torch.device("cuda")
    cap = torch.cuda.get_device_capability()
    sm = f"sm_{cap[0]}{cap[1]}"
    try:
        arch = get_arch(sm)
        hbm_peak = arch.hbm_bw_gbps * 1e9
        l2_mb = arch.l2_mb
    except KeyError:
        arch, hbm_peak, l2_mb = None, float("nan"), None

    clk_cur, clk_max = _sm_clock_mhz()
    flush = _l2_flush_buf() if l2_flush else None
    name = torch.cuda.get_device_name()
    kv_kind = "nvfp4-latent" if is_mla else ("nvfp4" if is_nvfp4 else ("fp8" if is_fp8 else "fp16"))
    print(f"# device: {name} ({sm})  clock~{clk_cur}/{clk_max}MHz  backend={backend}  "
          f"KV={kv_kind}  l2_flush={l2_flush}  gqa_group={gqa_group}")
    print(f"# {'shape(BxHq x1x d /Nk) Hkv':>27} | {'us/tok':>8} | {'eff_bw':>9} | {'%HBM':>6} | "
          f"{'WS(MB)':>8} | {'L2res?':>6} | {'L2served':>8}")

    rows: list[dict] = []
    for N in kv_lens:
        for B in batches:
            for d in head_dims:
                for H_kv in h_kvs:
                    H_q = gqa_group * H_kv
                    if is_mla and d not in _MLA_DV:
                        print(f"  # skip MLA d{d}: supported DQK in {sorted(_MLA_DV)} (576 real, 96 smoke)")
                        continue
                    # MLA reads ONE shared latent of width DQK=d, ONCE for all H_q heads (no 2x K/V, no
                    # H_kv factor); every other backend reads K+V over H_kv heads.
                    ws = (1.0 * B * N * d * b_bytes) if is_mla else (2.0 * B * H_kv * N * d * b_bytes)
                    if ws > max_ws_gb * 1e9:
                        print(f"  # skip B{B} H_kv{H_kv} d{d} N{N}: working set "
                              f"{ws/1e9:.1f} GB > --max-ws-gb {max_ws_gb}")
                        continue
                    try:
                        q = torch.randn(B, H_q,  1, d, device=dev, dtype=torch.float16)
                        if is_mla:
                            k = torch.randn(B, 1, N, d, device=dev, dtype=torch.float16)  # the latent
                            v = k                                                          # unused by MLA
                        else:
                            k = torch.randn(B, H_kv, N, d, device=dev, dtype=torch.float16)
                            v = torch.randn(B, H_kv, N, d, device=dev, dtype=torch.float16)
                        ours = _build_ours(backend, q, k, v, page_size, q_off=0, engine=engine)
                        p50, _ = time_ms_l2flush(ours, flush, warmup=warmup, iters=iters)
                    except RuntimeError as e:                       # OOM or build failure
                        print(f"  # skip B{B} H_kv{H_kv} d{d} N{N}: {type(e).__name__} {str(e)[:60]}")
                        torch.cuda.empty_cache()
                        continue

                    tokens = B * H_q
                    kv_bytes = (1.0 * B * N * d * b_bytes) if is_mla else (2.0 * B * H_kv * N * d * b_bytes)
                    eff_bw = kv_bytes / (p50 / 1e3)                 # bytes/s achieved on the KV read
                    us_tok = p50 * 1e3 / tokens
                    hbm_pct = (eff_bw / hbm_peak * 100.0) if arch else float("nan")
                    l2_served = bool(eff_bw > hbm_peak) if arch else False     # counter-free L2 test
                    l2_res_pred = (ws <= l2_mb * 1e6) if l2_mb else None       # working set <= L2 cap
                    row = dict(backend=backend, B=B, H_q=H_q, H_kv=H_kv, d=d, N_k=N,
                               us_tok=us_tok, eff_bw_gbps=eff_bw / 1e9, hbm_pct=hbm_pct,
                               ws_mb=ws / 1e6, l2_resident_pred=l2_res_pred, l2_served=l2_served,
                               clk_cur=clk_cur, clk_max=clk_max, p50_ms=p50)
                    rows.append(row)
                    res = "yes" if l2_res_pred else ("no" if l2_res_pred is not None else "?")
                    served = "L2!" if l2_served else "-"
                    print(f"  {f'{B}x{H_q}x1x{d}/{N} Hkv{H_kv}':>27} | {us_tok:8.2f} | "
                          f"{eff_bw/1e9:7.1f}GB | {hbm_pct:5.1f}% | {ws/1e6:8.2f} | {res:>6} | {served:>8}")
                    del q, k, v, ours
                    torch.cuda.empty_cache()
    return rows


def profile_one(backend: str, B: int, H_kv: int, N: int, d: int, *, gqa_group: int = 1,
                page_size: int = 256, iters: int = 100, engine: str = "fp8") -> None:
    """Build ONE shape and loop the kernel `iters` times (no timing) so an external `ncu` can attach to
    exactly this config and read L2 hit-rate / DRAM%. Mirrors how profiling/capture.sh pins one shape."""
    import torch

    dev = torch.device("cuda")
    H_q = gqa_group * H_kv
    q = torch.randn(B, H_q,  1, d, device=dev, dtype=torch.float16)
    if backend in ("v11_mla", "v12_mla_tc"):
        k = torch.randn(B, 1, N, d, device=dev, dtype=torch.float16)   # the shared latent (one head)
        v = k                                                          # unused by MLA
    else:
        k = torch.randn(B, H_kv, N, d, device=dev, dtype=torch.float16)
        v = torch.randn(B, H_kv, N, d, device=dev, dtype=torch.float16)
    ours = _build_ours(backend, q, k, v, page_size, q_off=0, engine=engine)
    for _ in range(iters):
        ours()
    torch.cuda.synchronize()
    print(f"# profiled {iters} launches of {backend} @ B{B} H_kv{H_kv} N{N} d{d} (for ncu)")


def main() -> None:
    p = argparse.ArgumentParser(description="v9 Task 1 — clock-locked, L2-flushed decode regime sweep.")
    p.add_argument("--backend", default="v8_gqa_ss",
                   help="decode kernel to characterize (v8_gqa_ss = FP16, v9_fp8 = FP8 KV, "
                        "v10_nvfp4 = NVFP4 KV, v11_mla = MLA latent-KV, v12_mla_tc = tensor-core MLA "
                        "(set --engine) — use --dim 576 for the real DeepSeek-V3 latent; the T1 "
                        "crossover deliverable).")
    p.add_argument("--engine", default="fp8", choices=["fp8", "nvfp4"],
                   help="v12_mla_tc only: the tcgen05 arm — fp8 (Arm 1) or nvfp4 (Arm 2). Ignored otherwise.")
    p.add_argument("--kv-lens", type=int, nargs="+", dest="kv_lens",
                   default=[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
                   help="KV lengths to sweep (crosses the 4 MB L2 around N_k=8192 for B1/H_kv1/d128).")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--batch-sweep", type=int, nargs="+", default=None, dest="batch_sweep",
                   help="sweep B over these values (hold one N_k via --kv-lens) to test whether %%HBM "
                        "climbs as BH passes ~2*SM (the large-batch occupancy question).")
    p.add_argument("--dim", type=int, nargs="+", default=[64, 128], dest="head_dims")
    p.add_argument("--h-kv", type=int, nargs="+", default=[1, 8], dest="h_kvs",
                   help="KV-head counts. H_kv=1 is the cleanest L2-crossing isolation; H_kv=8 a realistic case.")
    p.add_argument("--gqa-group", type=int, default=1, dest="gqa_group",
                   help="G = H_q/H_kv (default 1 = pure isolation, KV bytes = 2*B*H_kv*N*d*b).")
    p.add_argument("--page-size", type=int, default=256, dest="page_size")
    p.add_argument("--max-ws-gb", type=float, default=8.0, dest="max_ws_gb",
                   help="skip configs whose K+V working set exceeds this (OOM guard at large N_k x batch).")
    p.add_argument("--lock-clocks", action="store_true", dest="lock",
                   help="pin SM+memory clocks to max for the run (needs root); reset afterwards.")
    p.add_argument("--no-l2-flush", action="store_true", dest="no_l2_flush",
                   help="disable the between-iter L2 flush (to SHOW the L2-resident confound directly).")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--profile", default=None,
                   help="profile ONE shape 'B,H_kv,N_k,d' (loop the kernel for ncu, no timing).")
    a = p.parse_args()

    import torch  # noqa: local import keeps --help runnable without CUDA
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; run this on the GPU (root/bare-metal T4 for clock-lock + ncu)")

    if a.profile:
        B, H_kv, N, d = (int(x) for x in a.profile.split(","))
        profile_one(a.backend, B, H_kv, N, d, gqa_group=a.gqa_group, page_size=a.page_size,
                    engine=a.engine)
        return

    locked = False
    if a.lock:
        locked, *_ = lock_clocks()
    try:
        batches = a.batch_sweep if a.batch_sweep else [a.batch]
        sweep(a.backend, a.kv_lens, batches, a.head_dims, a.h_kvs, gqa_group=a.gqa_group,
              page_size=a.page_size, max_ws_gb=a.max_ws_gb, l2_flush=not a.no_l2_flush,
              warmup=a.warmup, iters=a.iters, engine=a.engine)
    finally:
        if locked:
            reset_clocks()


if __name__ == "__main__":
    main()

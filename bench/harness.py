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

from fa_kernels import (attention, fp8_attention, gqa_attention, mla_attention, mla_tc_attention,
                        nvfp4_attention, paged_attention)
from fa_kernels.paged import (build_paged_kv, build_paged_kv_fp8, build_paged_kv_mla,
                             build_paged_kv_nvfp4, dequantize_fp8_e4m3, dequantize_nvfp4,
                             quantize_fp8_e4m3, quantize_nvfp4)
from fa_kernels.reference import sdpa_reference, sdpa_reference_gqa
from roofline.archs import get_arch
from roofline.model import estimate

SEQ_LENS = [512, 2048, 8192]
HEAD_DIMS = [64, 128]
DECODE_KV_LENS = [2048, 8192, 16384]   # KV-cache lengths swept in --decode mode (N_q collapses to 1)

# v11 MLA: --dim selects the latent score-width DQK; this maps DQK -> the content/output width DV
# (kv_lora_rank); the trailing DQK-DV dims are the decoupled RoPE. The kernel templates the same three
# (DQK,DV) pairs. The real DeepSeek-V3 latent is 576/512 (--dim 576); 96/64 is the cheap smoke config.
_MLA_DV = {96: 64, 160: 128, 576: 512}


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
        decode: bool = False, q_len: int | None = None, page_size: int = 256,
        gqa_groups: list[int] | None = None, mla_engine: str = "fp8") -> None:
    # Default to the full sweep; --seq/--dim narrow it so ncu can profile one shape's 3 passes.
    # In --decode mode the swept N is the KV length and the query length collapses to q_len (1) — the
    # regime v6/v7/v8 decode targets, where the metric shifts to us/token + % HBM bandwidth (research §8).
    # `batches` sweeps B: at decode, BH = B*H is the occupancy knob, so a multi-B sweep MEASURES the
    # occupancy->bandwidth crossover (%HBM climbing as BH passes ~2*SM) that the re-plan only predicted.
    # `gqa_groups` (v8 only) sweeps the GQA group factor G = H_q/H_kv: H stays the query-head count and
    # H_kv = H//G shrinks as G grows, so KV bytes drop by G and decode AI = 2/b -> 2G/b. The v8 analogue
    # of the --batch sweep: it MEASURES the GEMV->GEMM (per-CTA efficiency) win as G activates G warps.
    paged = backend in ("v7_paged", "v8_gqa", "v8_gqa_tc", "v8_gqa_db", "v8_gqa_occ", "v8_gqa_ilp", "v8_gqa_ss", "v9_fp8", "v10_nvfp4")
    is_gqa = backend in ("v8_gqa", "v8_gqa_tc", "v8_gqa_db", "v8_gqa_occ", "v8_gqa_ilp", "v8_gqa_ss", "v9_fp8", "v10_nvfp4")
    is_fp8 = backend == "v9_fp8"   # v9: GQA-class shapes/sweep, but FP8 pools + scales + a quantizing oracle
    is_nvfp4 = backend == "v10_nvfp4"   # v10: GQA-class, but NVFP4 packed pools + micro-scales + scales
    is_mla = backend in ("v11_mla", "v12_mla_tc")  # MLA path: ONE shared latent (M=h_q), --dim = DQK
    is_mla_tc = backend == "v12_mla_tc"  # v12: the ENGINE change — tcgen05 MMA over the SAME latent bytes
    # v12 engine (Arm 1 fp8 / Arm 2 nvfp4) is the `mla_engine` arg; it sets the roofline ridge (the
    # engine-correct ridge: FP8 625 / NVFP4 1875 on B300 — results.md Step 12). v11 stays fp16 (the
    # dequant-to-fp16 path). `precision` keeps its dtype meaning (q tensor dtype), decoupled from the engine.
    engine = mla_engine if is_mla_tc else "fp16"
    groups = gqa_groups or [1]
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
        print(f"# {'shape(q x kv)':>19} | {'ours p50/max ms':>18} | {'us/tok':>8} | "
              f"{'%HBM':>6} | {'vs sdpa':>8} | {'vs naive':>8} | roofline")
    else:
        print(f"# {'shape':>19} | {'ours p50/max ms':>18} | {'sdpa p50/max ms':>18} | "
              f"{'speedup':>8} | {'tok/s(ours)':>12} | roofline")

    # Roofline row prints for any arch we have constants for (T4/A100/B300); unknown sm -> skip it.
    try:
        arch = get_arch(sm)
    except KeyError:
        arch = None
    dt = _dtype(precision)

    for B in batches:
      for N in seq_lens:
        for d in head_dims:
          for G in groups:
            qn = nq if nq is not None else N
            # GQA (v8): H is the query-head count (held fixed); H_kv = H//G KV heads. As G grows, KV
            # bytes drop by G (shared across the group). Skip G that doesn't divide H. Non-GQA: G=1,
            # H_kv=H (every prior backend), so the body is unchanged for them.
            if is_gqa and H % G != 0:
                print(f"  # skip G={G}: H={H} not divisible by G")
                continue
            H_kv = (H // G) if is_gqa else H
            q = torch.randn(B, H,    qn, d, device=dev, dtype=dt)
            # MLA builds its own single shared latent inside the is_mla branch; skip the (unused, and at
            # h_q=128/large-N potentially OOM-sized) per-head K/V allocation here.
            if is_mla:
                k = v = None
            else:
                k = torch.randn(B, H_kv, N,  d, device=dev, dtype=dt)
                v = torch.randn(B, H_kv, N,  d, device=dev, dtype=dt)

            q_off = (N - qn) if causal else 0
            if is_fp8:
                # v9 reads an FP8 E4M3 GQA KV pool (uint8) + per-tensor dequant scales. Oracle is
                # apples-to-apples: SDPA on the SAME dequantized E4M3 bytes. CRITICAL: dequantize ONCE
                # here, OUTSIDE the timed `base` lambda. The earlier `sdpa_reference_gqa_fp8(q,k,v,...)`
                # form re-quantized K,V on every call (an amax reduction + full passes over the cache),
                # inflating the SDPA baseline ~4x and making `vs sdpa` meaningless — measured 2026-06-28.
                kf, vf, bt_f, nkf, sk, sv = build_paged_kv_fp8(k, v, page_size)
                kb, _ = quantize_fp8_e4m3(k)
                vb, _ = quantize_fp8_e4m3(v)
                k_hat = dequantize_fp8_e4m3(kb, sk).to(k.dtype)   # the exact bytes the kernel reads
                v_hat = dequantize_fp8_e4m3(vb, sv).to(v.dtype)
                base = lambda: sdpa_reference_gqa(q, k_hat, v_hat, causal=causal)
                ours = lambda: fp8_attention(q, kf, vf, bt_f, page_size, nkf, sk, sv,
                                             causal=causal, q_offset=q_off, backend=backend)
            elif is_nvfp4:
                # v10 reads NVFP4 packed pools (nibbles + per-16 E4M3 micro-scales) + per-tensor scales.
                # Oracle = SDPA on the SAME dequantized NVFP4 bytes, dequantized ONCE outside the timed
                # `base` lambda (the v9 lesson: a re-quantizing oracle inflates the baseline).
                kp, km, vp, vm, bt_n, nkn, sk, sv = build_paged_kv_nvfp4(k, v, page_size)
                kpb, kmb, skb = quantize_nvfp4(k)
                vpb, vmb, svb = quantize_nvfp4(v)
                k_hat = dequantize_nvfp4(kpb, kmb, skb).to(k.dtype)   # exact bytes the kernel reads
                v_hat = dequantize_nvfp4(vpb, vmb, svb).to(v.dtype)
                base = lambda: sdpa_reference_gqa(q, k_hat, v_hat, causal=causal)
                ours = lambda: nvfp4_attention(q, kp, km, vp, vm, bt_n, page_size, nkn, sk, sv,
                                               causal=causal, q_offset=q_off, backend=backend)
            elif is_mla:
                # v11 MLA: ONE shared latent (read once by all h_q=H query heads). --dim = DQK (the
                # score width = kv_lora_rank + rope); DV = the content/output width. The generic q above
                # IS q_absorbed (width d=DQK, H query heads). Oracle = MQA over the SAME dequantized NVFP4
                # latent, dequantized ONCE outside the timed lambda (the v9 lesson). NVFP4 latent storage
                # is carried from v10, so vs the kernel this isolates the SHAPE (GQA->MQA-over-latent).
                DV = _MLA_DV.get(d)
                if DV is None:
                    print(f"  # skip MLA --dim {d}: supported DQK in {sorted(_MLA_DV)} (e.g. 576 real, 96 smoke)")
                    continue
                latent = torch.randn(B, 1, N, d, device=dev, dtype=dt)        # [B,1,N_k,DQK] one latent head
                lp, lm, bt_m, nkm, sl = build_paged_kv_mla(latent, page_size)
                lpb, lmb, slb = quantize_nvfp4(latent)
                lat_hat = dequantize_nvfp4(lpb, lmb, slb).to(latent.dtype)     # exact bytes the kernel reads
                mla_s = 1.0 / (d ** 0.5)
                def _mla_base(qq=q, lh=lat_hat, dv=DV, ss=mla_s):             # MQA-over-latent reference
                    sc = torch.matmul(qq, lh.transpose(-1, -2)) * ss
                    p = torch.softmax(sc - sc.amax(dim=-1, keepdim=True), dim=-1)
                    return torch.matmul(p, lh[..., :dv])
                base = _mla_base
                if is_mla_tc:
                    # v12: SAME latent bytes/oracle as v11; only the inner matmul changes (CUDA-core GEMV
                    # -> tcgen05 MMA). `engine` picks the arm (fp8 Arm 1 / nvfp4 Arm 2).
                    ours = lambda eng=engine: mla_tc_attention(q, lp, lm, bt_m, page_size, nkm, DV, sl,
                                                  engine=eng, causal=causal, q_offset=q_off, backend=backend)
                else:
                    ours = lambda: mla_attention(q, lp, lm, bt_m, page_size, nkm, DV, sl,
                                                 causal=causal, q_offset=q_off, backend=backend)
            elif is_gqa:
                # v8 reads a PAGED GQA KV (H_kv heads); the oracle expands KV by G (repeat_interleave).
                base = lambda: sdpa_reference_gqa(q, k, v, causal=causal)
                k_pool, v_pool, block_table, n_k = build_paged_kv(k, v, page_size)
                ours = lambda: gqa_attention(q, k_pool, v_pool, block_table, page_size, n_k,
                                             causal=causal, q_offset=q_off, backend=backend)
            elif paged:
                # v7 reads a PAGED KV: scatter the dense k,v into shuffled physical pages + a block
                # table, then time the gather kernel on that layout. Decode causal places the single
                # query at logical N_k-1 (q_offset = N_k - qn) so it attends the WHOLE cache instead of
                # the degenerate 1-key short-circuit -> the causal-decode rows become meaningful.
                base = lambda: sdpa_reference(q, k, v, causal=causal)
                k_pool, v_pool, block_table, n_k = build_paged_kv(k, v, page_size)
                ours = lambda: paged_attention(q, k_pool, v_pool, block_table, page_size, n_k,
                                               causal=causal, q_offset=q_off)
            else:
                base = lambda: sdpa_reference(q, k, v, causal=causal)
                ours = lambda: attention(q, k, v, causal=causal, backend=backend)

            o_p50, o_max = _time_ms(ours)
            s_p50, s_max = _time_ms(base)
            speedup = s_p50 / o_p50
            tokens = B * H * qn            # work is per QUERY head; H is H_q (unchanged by G)
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
                # v5/v6/v7/v8 read FP16 KV regardless of the input dtype (they cast in the kernel),
                # so the roofline byte count uses fp16. For v8, G>1 -> the decode bound is AI=2G/b.
                tile_m, tile_n = _roofline_tile(backend, d)
                # v9 stores KV as FP8 (1 byte) -> the decode HBM floor uses precision="fp8" (AI=2G/b
                # doubles vs fp16). Other FP16-KV backends use fp16 regardless of the input dtype.
                if is_fp8:
                    rp = "fp8"
                elif is_nvfp4 or is_mla:
                    rp = "nvfp4"   # KV/latent stored as 0.5625 B/elem
                elif backend in ("v5_wmma", "v6_splitkv", "v7_paged", "v8_gqa", "v8_gqa_tc", "v8_gqa_db", "v8_gqa_occ", "v8_gqa_ilp", "v8_gqa_ss"):
                    rp = "fp16"
                else:
                    rp = precision
                if is_mla:
                    # MLA: ONE shared latent over all h_q=H heads -> AI = 2*h_q*(2L+R)/((L+R)*b). d=DQK,
                    # DV=kv_lora_rank, R=DQK-DV. The model uses the TC peak (M=h_q>=16 engages it).
                    dv = _MLA_DV[d]
                    # v12: the engine sets the ridge (FP8/NVFP4 TC peak); v11 uses fp16 (dequant-to-fp16).
                    est = estimate(arch, B=B, H=H, N_q=qn, N_k=N, d=d, precision=rp,
                                   mla=True, h_q=H, kv_lora_rank=dv, rope_dim=d - dv,
                                   mma_engine=engine)
                else:
                    est = estimate(arch, B=B, H=H, N_q=qn, N_k=N, d=d, precision=rp,
                                   materialize_s=backend in ("v1_naive", "v2_tiled"),
                                   tile_m=tile_m, tile_n=tile_n, G=(G if is_gqa else 1))
                roof = f"{est.limiter.upper()} (~{est.seconds*1e3:.2f}ms)"

            if decode:
                # Decode headline metrics (research §8): us/token, % of peak HBM bandwidth achieved on
                # the K+V read (2*B*H_kv*N*d*b bytes, b=2 for fp16 KV — H_kv, NOT H_q, since GQA shares
                # KV across the group), and the speedup over the NO-M-PACKING floor. For v8 that floor is
                # v7 run on the SAME attention with KV broadcast-expanded to H_q heads (each query head
                # re-reads its KV head, G times the bytes) — the cleanest same-session isolation of
                # M-packing's one variable. For v6/v7 it stays the naive seqlen_q=1 loop (v5 @ N_q=1).
                us_tok = o_p50 * 1e3 / tokens
                # KV bytes/elem: FP16=2, FP8=1, NVFP4=0.5625 (4-bit nibble + per-16 E4M3 micro-scale —
                # count the scale, or the %HBM is overstated).
                kv_b = 0.5625 if (is_nvfp4 or is_mla) else (1 if is_fp8 else 2)
                if arch is not None:
                    if is_mla:
                        # MLA reads ONE shared latent of width DQK=d, ONCE for all heads (no 2x K/V, no
                        # H_kv factor) — the per-token byte count that makes AI = ~3.78*h_q/b.
                        kv_bytes = 1.0 * B * N * d * kv_b
                    else:
                        kv_bytes = 2.0 * B * H_kv * N * d * kv_b   # K and V; H_kv KV heads
                    eff_bw = kv_bytes / (o_p50 / 1e3)          # achieved bytes/s on the KV read
                    hbm_pct = eff_bw / (arch.hbm_bw_gbps * 1e9) * 100.0
                else:
                    hbm_pct = float("nan")
                # No clean same-shape "no-packing" baseline for MLA (the meaningful matched-work A/B vs
                # v10-GQA is a modeling choice made explicitly in the gate notebook). The honest in-harness
                # signal is the ABSOLUTE us/tok + %HBM + roofline columns: does MLA leave the ~0.5%-HBM
                # per-CTA floor GQA decode never left (the T1 deliverable)? So vs_naive is N/A for MLA.
                try:
                    if is_mla_tc:
                        # v12 baseline = v11 (CUDA-core MLA) on the SAME latent bytes/pools, so the A/B
                        # isolates the ONE variable: the compute ENGINE (tcgen05 MMA vs CUDA-core GEMV).
                        # This is the trustworthy same-session, clock-matched "vs v11" engine number — it
                        # prints under the "vs naive" column for MLA-TC rows (label still B x H ...).
                        naive = lambda: mla_attention(q, lp, lm, bt_m, page_size, nkm, DV, sl,
                                                      causal=causal, q_offset=q_off, backend="v11_mla")
                        n_p50, _ = _time_ms(naive)
                        vs_naive = n_p50 / o_p50
                    elif is_mla:
                        vs_naive = float("nan")
                    elif is_fp8 or is_nvfp4:
                        # v9/v10 baseline = v8.7 (score-stationary, FP16 KV) on the SAME GQA shape, so the
                        # A/B isolates the ONE variable: KV storage precision (FP8/NVFP4 vs FP16). Same
                        # packing. (The gate notebook also runs a v10-vs-v9 byte-only A/B for the FP4 step.)
                        kp8, vp8, bt8, nk8 = build_paged_kv(k, v, page_size)
                        naive = lambda: gqa_attention(q, kp8, vp8, bt8, page_size, nk8,
                                                      causal=causal, q_offset=q_off, backend="v8_gqa_ss")
                    elif is_gqa:
                        k_exp = k.repeat_interleave(G, dim=1)   # [B, H_q, N, d] — no-packing baseline
                        v_exp = v.repeat_interleave(G, dim=1)
                        kp7, vp7, bt7, nk7 = build_paged_kv(k_exp, v_exp, page_size)
                        naive = lambda: paged_attention(q, kp7, vp7, bt7, page_size, nk7,
                                                        causal=causal, q_offset=q_off)
                    else:
                        naive = lambda: attention(q, k, v, causal=causal, backend="v5_wmma")
                    if not is_mla:
                        n_p50, _ = _time_ms(naive)
                        vs_naive = n_p50 / o_p50
                except Exception:
                    vs_naive = float("nan")
                # Counter-free L2 test (no ncu): if the KV working set streamed from L2 (~4x HBM on T4),
                # %HBM exceeds 100 -> "L2!" -> the %HBM number is NOT a bandwidth-boundedness signal (the
                # v9 Task-1 confound). vs_naive for v9 is the FP8-vs-FP16 (v8.7) byte-isolation ratio.
                l2flag = " L2!" if (is_fp8 and hbm_pct == hbm_pct and hbm_pct > 100.0) else ""
                label = f"{B}x{H}x{qn}x{d}/{N}" + (f" G{G}" if is_gqa else "")
                print(f"  {label:>19} | {o_p50:7.3f}/{o_max:7.3f} | {us_tok:8.2f} | "
                      f"{hbm_pct:5.1f}%{l2flag} | {speedup:7.2f}x | {vs_naive:7.2f}x | {roof}")
            else:
                label = f"{B}x{H}x{N}x{d}" + (f" G{G}" if is_gqa else "")
                print(f"  {label:>19} | {o_p50:7.3f}/{o_max:7.3f} | "
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
    p.add_argument("--mla-engine", default="fp8", choices=["fp8", "nvfp4"], dest="mla_engine",
                   help="v12_mla_tc only: the tcgen05 arm — fp8 (Arm 1) or nvfp4 (Arm 2). Sets the "
                        "engine-correct roofline ridge; the 'vs naive' column is vs v11 (engine A/B).")
    p.add_argument("--gqa-group", type=int, nargs="+", default=None, dest="gqa_groups",
                   help="v8 only: sweep the GQA group factor G = H_q/H_kv over these values "
                        "(e.g. --gqa-group 1 2 4 8 16 32). --heads is H_q (held fixed; must be "
                        "divisible by each G); H_kv = H_q//G shrinks as G grows, so KV bytes drop by G "
                        "and decode AI = 2/b -> 2G/b. The v8 analogue of --batch-sweep: it MEASURES the "
                        "GEMV->GEMM per-CTA-efficiency win (G compute-warps + KV read once). Pair with "
                        "--decode --seq 8192 to hold N_k fixed; use --heads 32 to allow G up to 32.")
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; run this on the GPU (Colab T4 / rented box)")
    batches = a.batch_sweep if a.batch_sweep else [a.batch]
    run(a.backend, a.precision, batches, a.heads, a.causal, seq_lens=a.seq, head_dims=a.dim,
        decode=a.decode, q_len=a.q_len, page_size=a.page_size, gqa_groups=a.gqa_groups,
        mla_engine=a.mla_engine)


if __name__ == "__main__":
    main()

# v9 kickoff — FP8 KV cache + the regime-characterization (earn the bandwidth verdict)

> Paste-in starter for a fresh session. Follows the per-step loop in `ROADMAP.md`
> (roofline-first → explain → write → correctness → bench → results/decisions → quiz) and mirrors
> `docs/v8-kickoff.md`. Born from the 2026-06-28 v8 deep-research close-out (see `docs/results.md`
> "Step 8 — threats to validity", `docs/decode-replan.md` §5 v9, `docs/decisions.md` Step 8 close-out).

## Why v9 exists (read first)
Steps v6→v8.7 all concluded "decode is per-CTA-bound, NOT bandwidth-bound (~10% HBM)." The deep-research
close-out found that verdict is **confounded**: at the bench sizes the KV cache fits in the T4's 4 MB L2
(reclaim G=8/B=1 → H_kv=1 → ~4.2 MB ≈ L2), clocks were never locked (free Colab), and N_k never exceeded
~16K. So bandwidth physically couldn't appear. v9 has two tasks: **Task 1 (gating science)** — earn the
verdict confound-free. **Task 2 (the kernel)** — FP8 KV, measured where bytes actually matter.

The honest counterweight (keep this in mind): there IS partial evidence the per-CTA verdict survives —
the v8 G-sweep's low-G configs (G=2 → H_kv=16 → ~67 MB KV ≫ L2) still showed only ~11% HBM. But L2 was
never measured and clocks weren't locked, so it's unproven either way. Task 1 settles it.

## Task 1 — Regime characterization (NO new kernel; existing v8.7 `v8_gqa_ss`)
Goal: definitively answer "is this decode kernel bandwidth-bound, per-CTA-bound, or L2-resident?"
- **Hardware:** rent a ROOT/bare-metal T4 (vast.ai ~$0.10–0.20/hr) so clocks lock and ncu counters
  work (free Colab can't lock; containerized rentals give ERR_NVGPUCTRPERM). T4's small 4 MB L2 makes
  L2 spilling tractable — do NOT do this on B200/B300 (126–192 MB L2 needs enormous N_k to spill).
- **Pin clocks every run:** `nvidia-smi -pm 1`; `nvidia-smi --lock-gpu-clocks=<boost>`;
  `nvidia-smi --lock-memory-clocks=<mem>`; verify it held (`clocks.sm`, throttle reasons); reset after.
  Never compare wall-times across different clocks.
- **Flush L2 between timed iters:** allocate a buffer of `torch.cuda.get_device_properties().L2_cache_size`
  and `.zero_()` it before each launch (the jan.ai technique).
- **Sweep:** N_k ∈ {1K,2K,4K,8K,16K,32K,64K,128K} (crosses 4 MB at several points) × batch ∈
  {1,8,32,64,128} × d ∈ {64,128} × H_kv ∈ {1 (isolation), 8 (GQA)}.
- **Measure per row:** HBM achieved BW + % of *achievable* (~65–75% on T4, not datasheet 320 GB/s);
  **L2 hit-rate + L2 BW%** (`lts__t_sector_hit_rate`, `lts__throughput`); and the **counter-free L2
  test:** effective BW = KV_bytes / time — if it exceeds ~320 GB/s, the data came from L2.
- **The decisive plot:** HBM% and L2-hit-rate vs N_k. Where HBM% plateaus near the achievable ceiling
  as hit-rate→0 = genuinely HBM-bound. If even at N_k=128K / large B with hit-rate≈0 the HBM% *still*
  sits ~10%, then "per-CTA/launch-bound" is finally confound-free (and FP8-as-capacity is vindicated).
- **Deliverable:** a one-paragraph verdict + the plot in `docs/results.md` Step 9 Task 1. This either
  overturns or earns the 6-step claim — both are first-class results.

## Task 2 — v9 FP8 KV kernel (`kernels/v9_fp8/`)
**Roofline FIRST (record before coding).** Decode AI = 2G/b. v8 FP16 (b=2,G=8) → AI=8.0; v9 FP8
(b=1,G=8) → **AI=16.0**. T4 fp16 ridge≈203, so still HBM-bound (FP8 doubles AI, doesn't flip the
limiter). HBM floor **halves**. **Prediction:** on the L2-resident micro-bench, FP8 makes KV *more*
L2-resident → likely **no latency win (capacity-only)**; in Task 1's past-L2 regime, FP8 → ~2× less
HBM traffic → up to ~2× latency **iff** bandwidth-bound there AND dequant is fused (not a prepass).
**Counter-prediction:** if still per-CTA-bound even past L2, FP8 is capacity-only on this kernel and
the residual ceiling is launch/per-CTA (→ persistent-kernel / megakernel territory, a future lever).

**The single variable = KV storage precision.** Fork `kernels/v8_gqa_ss/` → `kernels/v9_fp8/`
(`fp8_attention.cu` + `binding.cpp`). Change ONLY the KV path:
- Paged pool stores K/V as **FP8 E4M3** (1 byte) instead of FP16 (2 bytes) → KV HBM traffic halves.
- **Fused per-tile dequant** at smem-load time (where ss already converts to FP16): load FP8 byte →
  `× scale` → FP16 into the existing `sK`(transposed)/`sV` smem. The score-stationary inner loop is
  untouched → clean byte-only ablation.
- **Scaling:** start **per-tensor** FP32 scale (research: near-lossless at 8-bit). Per-token V /
  per-channel K only if the accuracy delta demands it (that comparison is a deliverable).
- **Build gotcha:** `__nv_fp8_e4m3` (cuda_fp8.h) conversion is software on sm_75 — verify it compiles
  on Colab's CUDA. Fallback: store `uint8` + manual E4M3 unpack, or int8-symmetric (same 1-byte HBM
  win; isolates bytes identically; the E4M3-vs-int8 accuracy gap is itself worth measuring).
- Keep **FP32 accumulation** (you already do — dodges the FA-3 FP8-accum cliff).
- Register in `bindings/load.py` `_SOURCES` and `dispatch.py` `_MIN_CAPABILITY=(7,0)` (T4-valid).

## Correctness (Gate 1)
- Oracle: SDPA with FP16 KV. FP8 quantization means no bit-match — measure **RMSE / max-rel-error vs
  the FP16-KV reference** and document it (this IS the accuracy deliverable, not just a pass/fail).
  Loosen tol beyond the FP16-in 2e-2 band (expect ~5e-2; report the actual distribution).
- New `fp8_attention()` API + an `sdpa_reference` path that quantizes KV the same way for an
  apples-to-apples accuracy number. Sweep G∈{1,2,4,8} × N_k(incl. non-multiple) × d × causal both ways.

## Benchmark (Gate 2 inputs) + deliverables
1. **Capacity:** KV footprint FP8 vs FP16 (trivially 2×; state max-context-at-fixed-memory).
2. **Accuracy:** RMSE / perplexity-proxy vs FP16 KV (per-tensor; optionally per-token/channel).
3. **Latency, two regimes:** (a) L2-resident micro-bench (expect ~null — capacity-only, confirms the
   thesis); (b) **Task-1's past-L2 / large-batch regime** (the real test — does halving bytes give the
   ~2× the roofline predicts? prediction-vs-measured, finally in a regime where the model applies).
4. Same-session vs-v8.7 and vs-SDPA ratios at **locked clock**.

Then update `docs/results.md` + `docs/decisions.md` (Step 9), add interview-prep C13, and QUIZ.

## Hardware
v9 runs entirely on **T4** (FP8 storage + dequant + the regime sweep). No B-series needed until v10
(NVFP4 native throughput — see `docs/decode-replan.md` §5 v10).

## North star (why this arc exists)
The project's research goal is a **paper**: *the first open, roofline-documented, prediction-vs-measured
FlashAttention decode study on **B300 / GB300 (sm_103)**, with an asymmetric-precision FP4 KV recipe,
benchmarked vs FlashInfer/FlashMLA.* FA4 stops at B200/sm_100, so sm_103 is the publishable novelty.
(Honest scope: production libs already run GB300 decode — the contribution is the *open paper-grade
roofline + FP4 recipe + honest methodology* on sm_103, not "first to run.") **v9's job is to build the
honest measurement methodology (clock-lock, L2-aware, counter-free BW test) and the FP8 dequant
machinery that the B300 record runs (v10/v11) carry forward.** B300 is the final goal; T4 (v9) is where
the method is forged cheaply. See `docs/decode-replan.md` §5 v10/v11.

## Traps (carried from v8 + new)
- Causal mask uses `i_q`, not `m_row` (the v8 trap). GQA oracle uses `repeat_interleave(G)`, not
  `repeat`. Idle-warp barrier participation. Fully-future split → (m=-inf,l=0,O=0).
- NEW: dequant must be **fused per-tile**, never a full-cache prepass (a prepass re-reads the cache and
  eats the byte savings — QServe). NEW: don't compare wall-times across different clocks — lock them.

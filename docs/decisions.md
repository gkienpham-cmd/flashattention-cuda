# Decision log

One entry per step: the bottleneck identified, options considered, what we chose, and what
would change on a different architecture. This is the "defend it in an interview" record.

---

## Step 0 — Foundation & hardware

**Hardware:** Colab **T4, Turing sm_75** (40 SMs, 320 GB/s HBM, 64 KB smem/SM, 2nd-gen tensor
cores ~65 FP16 TFLOPS / ~130 INT8 TOPS, MUFU for `exp2`). Build loop: Colab notebook +
`cpp_extension.load()` JIT. A100/H100 rented per-step.

**Why a versioned `kernels/vN_*` library instead of scripts:** the end goal is a mini-vLLM
inference engine that does `from fa_kernels import attention`. A stable public API
(`fa_kernels/__init__.py`) over swappable versioned kernels lets the engine pin a backend while
the journey keeps adding faster ones. The single `AttnConfig` keeps tile shapes / dtypes /
scale from drifting between kernel, test, bench, and roofline tool.

**What changes on another arch:** the build's `-gencode` and the roofline arch constants
(`roofline/archs.py`) are the only places hardware leaks in; everything above the dispatch
layer is arch-agnostic.

**Build/profile environment:** dev runs on **free Colab T4** for compile + correctness + timing
(`notebooks/colab_bootstrap.ipynb`: clone → clean stale JIT cache → `pip install ninja` → build →
pytest → bench). ncu **did run** on the current Colab runtime (it connected and wrote a
`.ncu-rep`), so Colab profiling is not universally blocked — but it's runtime-dependent and not
guaranteed, and `profiling/capture.sh` currently profiles the wrong kernels (it caught the
`torch.randn` RNG kernel, not qk/softmax/pv — needs a `--kernel-name` filter / launch skip). So
for a *reliable* ncu reading a **dedicated rented T4 (vast.ai ~$0.15/hr)** is still the safer bet;
free Colab covers build/test/bench. This is also why the plan rents A100/H100 per-step (Phase
2/3) rather than buying Colab Pro. Repo: `github.com/gkienpham-cmd/flashattention-cuda` (public,
so Colab `git clone` needs no token).

**Roofline tooling fix (2026-06-20).** `roofline/model.py` was selecting the FP16 tensor-core peak
for *every* non-FP32 precision, so `--precision int8` predicted against 65 TFLOPS instead of the
T4's 130 INT8 TOPS — a 2× too-slow MMA bound, with `arch.int8_tc_ops` dead and the ridge stuck at
203 instead of 406. The MMA peak and the ridge are now both precision-selected (T4: fp32 25.3 /
fp16 203 / int8 406 FLOP·byte⁻¹), matching the ridge points already cited in `interview-prep.md`.
No measured row changes (v1–v4 are FP32) — this just unblocks an honest INT8 prediction for the
Phase-3 precision work (ROADMAP #9). Same `p99`→`max` relabel landed in `bench/harness.py`/results.

---

## Step 1 — Naive attention (the bandwidth wall)

**Bottleneck identified (predicted):** HBM bandwidth, at *every* head dim. Two sources of HBM
traffic: (1) the full `S = QK^T` matrix round-trips global memory ~4× (write in QK, read+write in
softmax, read in PV) — O(N²) bytes; and (2) — the dominant one — the matmuls **re-read their
operands from HBM with no reuse**: `qk_kernel` re-reads a full Q row and K row for every one of
the N² score entries (`naive_attention.cu:50-54`), `pv_kernel` likewise (`:116`). That is
**O(N²·d)** traffic, a factor of *d* larger than the S round-trip. Counting it,
`python -m roofline.predict --shape 1x8x2048x64 --precision fp32 --materialize-s --tile 1x1`
gives arithmetic intensity **~0.2 FLOP/byte** vs the FP32 CUDA-core ridge of **25.3** → deeply
memory-bound, predicted limiter **HBM** (~100% util, MMA ~1%, MUFU ~0%; lower-bound ~109 ms at
d=64, ~216 ms at d=128). Intentionally awful — this is the "before."

**Correction to an earlier number (kept honest, not hidden):** an earlier cut of the roofline
model assumed operands were read *once* from HBM, which gave "~15 FLOP/byte" at d=64 and a
spurious "d=128 flips to MMA-bound at ~30." Both were artifacts of ignoring the redundant operand
reads — the read-once assumption is actually closer to a *tiled* kernel than to the naive one.
With the operand-reuse term added to `roofline/model.py`, true-naive v1 is HBM-bound at AI ~0.2
for **all** d. (See [Step 2](#step-2--shared-memory-tiling-the-operand-reuse-win) for the model
fix and the tiled prediction.)

**v1's two real weaknesses, correctly placed:** (a) HBM traffic — redundant operand reads (big)
+ the S round-trip (smaller); HBM-bound now, attacked by Steps 2–4. (b) no tensor cores — this
bites *later*: only once a fused kernel reaches AI ~512 does the limiter become MMA, and then on
the weak 8 TFLOPS FP32 **CUDA cores** (fused d=64 ≈ 1.06 ms, d=128 ≈ 2.12 ms, both MMA-bound).
Step 6 (tensor cores) raises that ceiling. So (a) is a v1 problem; (b) is a post-fusion problem.

**Options considered:**
- **A1 three-pass (chosen)** vs A2 single fused-naive kernel — three separate launches make the
  S round-trip unmissable in `ncu` (each pass profiles independently), maximizing the teaching
  value and the drama of the later fused speedup.
- **B1 FP32 everywhere (chosen)** vs B2 FP16-in/FP32-accum — FP32 is a clean, exact-ish
  correctness anchor (atol/rtol 1e-4 vs SDPA) before precision enters as its own variable at v2.

**What we chose:** A1 + B1. One thread per output element, no tiling, no reuse — deliberately
naive so every later optimization has a measured "before."

**What changes on another arch:** an A100 (1.5 TB/s, 2 TB/s) raises the HBM roof ~5–6× but the
*shape* of the wall is identical — naive attention is memory-bound on every GPU; that
architecture-independence is the whole reason FlashAttention exists.

**MEASURED (Colab T4, 2026-06-18) — and the L2 plot twist:** 9/9 correctness tests pass vs SDPA
(atol/rtol 1e-4). Bench is ~0.03–0.06× SDPA (≈20–30× slower) across the sweep — the intended
"before." But the honesty check surfaced something: measured p50 lands *below* the cache-free
roofline lower bound at mid-N (2048×64: 85.6 ms measured vs 109 ms predicted = 0.78×; 2048×128:
0.77×), then *converges* to it at N=8192 (64: 1.01×; 128: 0.90×). You can't beat a true HBM
floor — so the floor's assumption is wrong: the model counts every redundant operand read as HBM
traffic, but the T4's **4 MB L2 absorbs most of them** while the working set fits. At N=8192 a
single head's K (2 MB) + head-interleaving overflows L2, the hit rate collapses, and measured
meets the cache-free floor. Takeaway: the AI≈0.25 cache-free roofline is the **worst case**,
realized exactly when L2 can't help (large N). This sharpens Step 2's thesis — explicit
shared-memory tiling makes reuse *guaranteed and N-independent*, so **its biggest win is predicted
at N=8192**, where L2 currently fails. (Model refinement worth considering later: an L2 hit-rate
term so the predicted floor tracks the measured mid-N speedup; logged, not yet built.)

---

## Step 2 — Shared-memory tiling (the operand-reuse win)

**Roofline model fix (done before predicting):** Step 1's prediction exposed that
`roofline/model.py` assumed each operand was read from HBM once — so it could not see the
redundant O(N²·d) operand reads that *are* the naive wall, and could not distinguish naive from
tiled. We added tile-aware operand traffic: for a tiled GEMM with output tile `tile_m × tile_n`,
each operand is read `M*N*K / tile` times (naive = 1×1 = re-read per output element). The flag
`--tile MxN` drives it; `--materialize-s` off remains the fused read-once ideal.

**Bottleneck identified (predicted):** still **HBM**, but far less badly. Tiling raises on-chip
reuse, cutting operand traffic ~`tile×`. Predictions at the tiles the v2 kernel will use
(FP32, N=2048):

| regime                         | d=64            | d=128           |
|--------------------------------|-----------------|-----------------|
| naive (`--tile 1x1`)           | AI 0.2, HBM     | AI 0.2, HBM     |
| **tiled (Step 2)**             | **AI 8.0, HBM** (`64x64`) | **AI 6.4, HBM** (`32x32`) |
| fused ideal (`no -materialize-s`) | AI 512, MMA  | AI 512, MMA     |

**The key prediction:** tiling is a ~30× traffic cut (AI 0.2 → ~6–8) but **does NOT cross the
25.3 ridge** — v2 stays HBM-bound at *both* head dims, because the S round-trip survives (only
online softmax, Step 3, removes it). So we predict a large measured speedup over v1 with the
limiter *unchanged*. That "still HBM-bound" is the motivation handed to Step 3.

**Nuance — bigger d means less reuse:** d=128 must drop to a 32×32 tile (a 64×64 FP32 Q+K tile
is 64 KB, leaving no room), so its reuse factor is smaller and its tiled AI (6.4) is *below*
d=64's (8.0). Head dim eats the shared-memory budget that buys reuse.

**Options / what we chose:** keep the three-pass structure (S still materialized) so Step 2
isolates the operand-reuse lesson from the S-elimination lesson (Step 3). Tile sizes adapt to d
to fit the 64 KB smem. Online softmax and fusion are deliberately deferred.

**Gate (cleared 2026-06-19):** Step 1 went green + benched + quiz-passed, so v2 shipped:
`kernels/v2_tiled/tiled_attention.cu`, FP32, three-pass, S still materialized. Tiles are picked
per head dim (64x64 @ d=64, 32x32 @ d=128) to fit the T4's 48 KB static smem; correctness is
**26/26 vs SDPA** at atol/rtol 1e-4, incl. non-tile-multiple shapes for the boundary guard.

**MEASURED (Colab T4, 2026-06-19) — bench:** v2 is **1.3–3.2× faster than v1** (clock-matched at
SM ~300 MHz), below the ridge (not compute-bound), as predicted. Still 0.04–0.11× SDPA (fused
FlashAttention), because the S round-trip survives; that's v3's job.

**MEASURED (Colab T4, 2026-06-19) — ncu `dram__bytes_read`, two shapes — the *traffic* prediction
was wrong, and instructively so.** We predicted tiling cuts DRAM traffic ~30× and stays HBM-bound.
Measured total DRAM reads: N=512 → v1 71.9 MB / v2 66.5 MB (1.08×); N=8192 → v1 30.80 GB / v2
30.22 GB (1.02×). **Tiling cut essentially zero DRAM traffic at either shape, and nothing saturates
HBM** (DRAM throughput ≤35%). Two facts the traffic model couldn't see:
- **L2 owns the operands at *every* N.** v1's qk pass reads 74 MB at N=8192 — the 4 MB L2 serves the
  redundant Q/K reads the cache-free model charged hundreds of GB for. Tiling had nothing left to
  cut; on qk, v2 reads *19× more* than v1 (worse L2 reuse from the tiled pattern + halved occupancy).
- **S is ~99% of DRAM, not a floor under an operand mountain.** The softmax pass re-reads the
  materialized S ~12× (26.3 GB at N=8192); operands are <1%; and S is byte-identical v1↔v2. The
  model named the right villain (S round-trip) but mis-sized it by ~100×: it *is* the traffic.

So v2's speedup is a **compute/scheduling win, not a bandwidth win** — DRAM is flat. The headline
lesson stands and hardens: *a traffic-bound roofline is blind to the cache that serves the traffic
and to the kernel inefficiency that spends the time* — here it got the magnitude, the location,
**and** the limiter wrong, all because it can't model the T4's L2. The one thing it nailed
structurally — only online softmax removes S — is now the empirically dominant ~99% of traffic.

But the Step 1 corollary — *"biggest win at N=8192, where L2 fails"* — was **wrong**. The measured
v2/v1 win *peaks at mid-N* (2048×128 = 3.20×) and *shrinks* at N=8192 (2.02–2.84×). The honest
post-mortem, separating two effects:
- **Magnitude (why ~3×, not the predicted ~30×):** the roofline says traffic drops ~30×, but
  *neither kernel sits on its roofline*. v1 runs faster than its cache-free floor (L2 help), and
  v2 runs 6–21× *above* its own floor (scalar loads, low occupancy from the 32 KB tile, PV-pass
  `__syncthreads` overhead). Those compress 30× → ~3×. The S round-trip caps the *roofline* win
  (AI 8, ~32×), not the realized one — so S is not why the win is small.
- **Shape (why it peaks at mid-N, not 8192):** operand and S traffic both scale as N², so the
  composition is N-independent and cannot drive a trend. The trend is purely off-roofline — v2's
  distance-from-floor is worst at the extremes (N=512 = 21× above, overhead-bound; N=8192 = 15×,
  load/sync-bound) and best at mid-N. The Step 1 corollary tracked only v1's L2 cliff and missed
  that **v2's own efficiency curve dominates where the win lands.**
- **The one predictable part:** d=128 beats d=64 at every N because the operand term tiling cuts
  is a larger traffic share there (operand:S ≈ 4:1 vs 1:1 at d=64).

**Lesson (bench view; revised by the ncu read below):** the roofline bounds *traffic*, not a
kernel's distance from that bound, and realized speedup is the ratio of two such distances — so it
missed the magnitude and location of the win. From the bench alone it *looked* like the limiter
prediction (HBM) survived; the `dram__bytes_read` read then showed even that was wrong (≤35% DRAM
throughput — never bandwidth-bound). Full measured-DRAM table in
[results.md](results.md#step-2--shared-memory-tiling-fp32-three-pass-s-still-materialized).

**Next (Step 3 — online softmax):** the S round-trip is no longer a "named target" — the ncu read
proved it's **~99% of measured DRAM traffic**, so removing it is the entire bandwidth story.
Replace the three-pass materialize-S structure with running max/sum so S never touches HBM — the
model predicts AI jumps to ~512 and the limiter crosses to MMA. *Caveat carried forward from this
step:* the model also predicted v1/v2 were HBM-bound and they weren't (L2), so we predict Step 3's
crossing-to-MMA but verify it against ncu before believing it. **Step 2 gates: both cleared
(2026-06-19)** — quiz passed, and the v1-vs-v2 `dram__bytes_read` read is done (above).

**What changes on another arch:** more shared memory (A100 164 KB, H100 228 KB vs T4 64 KB)
allows larger tiles → higher reuse → higher tiled AI, and async copy (Phase 2) hides the load
latency. The *ridge* also moves, so where "tiled" lands relative to it is arch-specific — but on
every arch, tiling-without-fusion stays bandwidth-bound because S still round-trips.

---

## Step 3 — Online softmax (the S-elimination is real; the speedup isn't, yet)

**Bottleneck going in:** the Step-2 ncu read proved S is ~99% of DRAM traffic. So the named target
was: delete S from HBM via online (streaming) softmax — running `(m, l)` per query row, scores
computed on-chip and discarded.

**Options considered:**
1. **Single-pass FlashAttention-1** (the canonical fused kernel): one sweep maintaining `m, l` *and*
   a running O accumulator that must be rescaled by `exp(m_old−m_new)` on every max update.
2. **Two-pass online softmax** (chosen): pass 1 streams K → final `(m, l)`; pass 2 re-streams K/V,
   recomputes scores, forms `O = (exp(s−m)/l)·V` with no O-rescale (m,l are final).

**Choice — two-pass, and why:** it isolates *one variable*. The O-rescale (coupling the softmax
denominator into the matmul accumulation) is the single fiddliest, most bug-prone part of FA;
deferring it to v4 lets v3 prove the S-elimination mechanism in isolation, exactly as v2 isolated
operand-reuse while keeping S materialized. Cost: QK computed twice + scores recomputed in pass 2.
Same reasoning extended to operand handling: v3 reads K/V unstaged from global (L2-resident per
Step 2) rather than re-staging them — so the *only* thing v3 changes vs v2 is removing S.

**Measured (T4 sm_75, 2026-06-19) — the prediction missed a third time, for a third reason:**
- **The mechanism works:** 15/15 correctness (incl. N=16384 rescale stability); peak memory at
  8192×64 is **+17 MB (v3) vs +2164 MB
  (v2)** — S provably never materializes (125× less, no profiler needed).
- **The speedup is negative:** v3 is **3–7× slower than v2** and ~50–100× slower than SDPA. Deleting
  ~99% of DRAM traffic bought nothing, because Step 2 already proved nothing was bandwidth-bound. We
  optimized a resource that was never the constraint.
- **Real limiter = occupancy/latency, not MMA or MUFU.** Measured 2556 ms at 8192×64 is **151× above
  the 17 ms MMA floor**, so it's nowhere near compute-saturated. The torch profiler (CUPTI trace, no
  counters) shows **pass2 = 88.6%** of time: one-thread-per-row leaves 192/256 threads idle, and
  pass 2's unstaged per-row K/V global reads dominate. The roofline mispredicts *again* because it
  assumes an efficient schedule — the inefficiency is the term it can't model (the Step-2 L2 lesson
  in a new disguise).

**What changes on another arch:** nothing rescues v3's *scheduling* — one-thread-per-row is
arch-independently bad. On A100/H100 the larger register file + smem would let v4 stage K/V and keep
a register-resident O accumulator with the single-pass rescale, and async copy hides the load
latency — that's where the S-off-HBM property finally converts to wall-clock. The S-elimination
itself is arch-independent and permanent.

**ncu deferred:** all containerized rentals block hardware counters (`ERR_NVGPUCTRPERM`); the
MMA-vs-MUFU pipe-util read waits for a bare-metal/dedicated box. The counter-free evidence
(memory + CUPTI trace + roofline distance) already names the limiter as occupancy/latency.

**Next (Step 4 — fused FA-1):** single pass, O-rescale, one warp per query row, staged K/V,
register-resident accumulator. Keeps S off HBM (v3's win) *and* schedules like v2's GEMMs — the
first version where "S never touches HBM" should actually beat v2 in wall-clock.

## Step 4 — Fused FlashAttention-1 (the thesis lands: S-off-HBM finally beats tiling)

**Bottleneck going in:** v3 proved S-elimination but was occupancy/latency-bound — 151× off the MMA
floor, 88.6% of time in pass 2, one-thread-per-row leaving 192/256 lanes idle. Named target: keep S
off HBM but schedule like v2's GEMMs.

**Options considered:**
1. **Single-pass FA-1, warp-per-row, register-O, staged K/V (chosen).**
2. **Two-pass but better-scheduled** — rejected: doesn't repair the fundamental one-thread-per-row
   waste and proves nothing new.
3. **Block-per-row with register-tiled score *tiles* (GEMM-shaped scoring)** — deferred: closer to
   canonical FA and likely faster, but it couples the O-rescale into a tiled matmul in the same step.
   We took warp-per-row first to land the O-rescale correctly with one isolated structural change.

**Choice — warp-per-row single-pass, and why:** the smallest step from v3 that fixes the schedule.
One warp (32 lanes) per row restores parallelism (v3 wasted 31/32), staged K/V gives coalesced reuse,
register-O keeps the accumulator off-chip, and the O-rescale finally gets implemented (v3's deferred
piece). One isolated variable: thread→warp granularity.

**Measured (T4 sm_75, 2026-06-20) — thesis confirmed, new limiter exposed:**
- **Correct:** 17/17, incl. N=16384 O-rescale stability at d=64 *and* d=128, causal both ways.
- **v4 beats v2 (1.7–2.6×) and v3 (7.5–15×).** S-off-HBM is now also a wall-clock win — the goal
  since Step 3. The v2 win is **two things at once**: fusion (v2's 2 GB S DRAM round-trip, re-read
  ~12×, eliminated; 3 kernels → 1) *and* a tight single-kernel schedule. Three-way contrast proves
  you need both — v3 had S-off-HBM without the schedule and was *slower* than v2; v2 has the schedule
  without S-off-HBM and is slower than v4. Clean split needs ncu (deferred).
- **S still gone:** +16.8 MB at 8192×64 (< v3's +17.3; no per-row HBM scratch).
- **Schedule fixed:** CUPTI shows a single fused kernel at 100% of CUDA time — the pass2/occupancy
  wall is structurally gone. Distance to floor 151× → ~18×.
- **New limiter = FMA under-utilization (GEMV-shaped scoring).** ~18× off the FP32 floor because each
  per-key score is a 5-step `__shfl` reduction + 2 FMAs — reduction overhead dominates, never near
  the 8.1 TFLOPS FMA peak. Roofline missed the *magnitude* (4th miss), same flops/bytes blind spot.

**What changes on another arch:** the warp-shuffle GEMV is arch-independently FMA-inefficient — a
bigger register file won't change the *shape* of the computation. The real fix is GEMM-shaped MMA
(tensor cores, or register-blocked score tiles). On A100/H100 the absolute numbers improve but the
~18×-off-floor shape persists until the scoring becomes a real matmul.

**ncu deferred:** containerized rentals block counters (`ERR_NVGPUCTRPERM`); the memory + CUPTI +
roofline-distance evidence already names the limiter.

**Next (Step 5 — tensor cores):** FP16-in/FP32-accum WMMA. Raises the ceiling 8→65 TFLOPS *and*
forces GEMM-shaped MMA tiles — directly attacking v4's FMA-efficiency gap. Expect the limiter to
finally approach a real (tensor-core) MMA bound instead of a reduction-overhead wall.

---

## Step 5 — Tensor-core WMMA (the GEMV→GEMM fix)

> **Backfilled 2026-06-27, PARTIAL.** Kernel + wiring landed (`ad021b7`) and the **correctness gate is
> green** (tol 2e-2, rebuilt + passed during the Step-6 run). **The prefill bench was never captured**
> — `notebooks/step5_run_of_record.ipynb` is unexecuted (all cells `execution_count=None`, no saved
> outputs). So the *decision* below is recorded from design + roofline; the measured speedup that would
> normally close it is OUTSTANDING (run the run-of-record). Logged as a gap, not papered over.

**Bottleneck going in:** v4 was compute-bound but ~18× off the FP32 MMA floor and ~6× slower than
SDPA. Diagnosed cause = **FMA under-utilization**: scoring one key per warp via a 5-step `__shfl`
reduction + 2 FMAs is GEMV-shaped — ~70% reduction plumbing, never near the 8.1 TFLOPS FP32 peak.
Named target: make the math GEMM-shaped *and* raise the ceiling.

**Options considered:**
1. **WMMA tensor cores, FP16-in/FP32-accum, keep v4's fused single-pass schedule (chosen).**
2. **Register-blocked FP32 score *tiles* on CUDA cores** — rejected: turns scoring into a GEMM
   (fixes the shape) but stays at the 8.1 TFLOPS ceiling, so it can't reach the 8× headroom tensor
   cores expose; also more register pressure for less upside.
3. **`mma.sync` PTX / CUTLASS-style pipelined tiles** — deferred: closer to production FA but a much
   bigger step; WMMA is the minimal, didactic on-ramp to tensor cores and isolates one variable
   (CUDA-core dot product → tensor-core MMA) on top of v4's already-correct schedule.

**Choice — WMMA, and why:** the smallest step from v4 that attacks the diagnosed limiter on *both*
axes simultaneously — GEMV→GEMM (the reduction moves inside the MMA, no shuffle) and 8→65 TFLOPS
ceiling. One isolated variable vs v4: the two matmuls become `wmma::mma_sync`. Everything else
(online softmax, O-rescale, S-off-HBM) is carried unchanged. The price of admission is FP16 inputs
(hence the first loosened tolerance, 2e-2) and **the opaque-fragment tax**: WMMA accumulator results
are scattered across lanes in an un-indexable layout, so softmax can no longer live in registers —
S is forced through smem (store → row-softmax → reload P as half), and `oRun` (the FP32 O) lives in
smem so the O-rescale can be folded into the PV accumulator. That smem S round-trip is the structural
risk this step introduces — it may become the *new* limiter even as the FMA wall falls.

**Measured (T4 sm_75) — correctness only; speedup OUTSTANDING:**
- **Correct:** green at tol 2e-2 — 17 cases (6 shapes × causal, explicit-scale, + the
  `test_v5_wmma_fp16_stability` N=16384 O-rescale/FP16-drift case at d=64 *and* d=128). The d=128 case
  also exercises the BM=32/2-warp tile config.
- **Speedup vs v4 / SDPA / floor:** NOT captured (run-of-record unexecuted). The roofline predicts the
  floor drops 8× (16.97 → 2.114 ms at 8192×64, fp16); whether v5 reaches it is the open question.

**What changes on another arch:** tensor-core shapes/throughput are arch-specific — Turing WMMA is
16×16×16 at ~65 TFLOPS FP16; Ampere/Hopper add `mma.sync`/`wgmma`, bf16/tf32/fp8 and far higher peaks.
The *structural* lesson (GEMV→GEMM, accumulate in FP32) is arch-independent; the absolute floor and the
best MMA instruction are not. On A100/H100 the smem S round-trip matters less (more smem, async copy)
so the limiter may shift again.

**Outstanding (to fully close Step 5):** execute `notebooks/step5_run_of_record.ipynb` and backfill
the speedup table in `results.md` + this section + the status line in `CLAUDE.md`; optionally the
peak-memory proof and CUPTI single-kernel trace (expected ≈ v4's).

---

## Step 6 — Split-KV decode (v6)

*Measured 2026-06-27 (vast.ai Tesla T4 sm_75, torch 2.6.0+cu124). 25/25 correctness. This opens the
decode arc v6→v11 (`docs/b300-decode-research.md`): the long-term goal is a B300 low-precision decode
kernel that beats FA4 (a BF16 prefill/training kernel) in the regime it never targets.*

**Bottleneck (predicted): decode occupancy.** At `N_q = 1` the prefill grid
`(ceil_div(N_q, rows), B·H)` collapses to `(1, B·H)` — a handful of blocks that starve the 40 SMs
(research blind-spot #2). Decode is HBM-bound (`AI = 2/b`, §4); the issue is the kernel can't *reach*
that bound with one block per head — there is no work to spread.

**Options:**
- **A — run v4/v5 unchanged at `N_q = 1`:** correct but `1×BH` blocks, SM-starved. This is the *naive
  baseline* v6 must beat (the bench `vs naive` column).
- **B (chosen) — split-KV / Flash-Decoding:** partition KV across blocks; each emits an unnormalized
  partial `(O, m, ℓ)`; a merge kernel does the log-sum-exp combine across splits. Fills the SMs without
  changing the result. Research §5 lever (a), §7 rank #1 ("v6 foundation"): high payoff, low risk, no
  B300 needed.

**Choice:** FP16-in/FP32-accum (research §8 tags v6 "FP16"), two kernels behind one `forward`,
contiguous KV (paged gather is v7). `choose_splits` raises `num_splits` until the block count hits ~2×
the SM count, capped at 32 and floored at a 256-key chunk; **prefill (large `N_q`) → `num_splits = 1` →
v6 reduces to plain attention** (which is what makes the square-shape correctness tests pass). No
tensor cores: at `N_q = 1` the matmuls are M=1 (GEMV), so WMMA would idle on a 1-row tile.

**What changes on B300 (research §3; SM count now MEASURED in v10):** **148 SMs** (MEASURED on the vast.ai
B300 — the 160 figure is the spec/marketing number, refuted by the device readout) ⇒ more splits (retune
`choose_splits` `num_sm` 40→148);
**HBM bandwidth is FLAT vs B200** ⇒ the real decode wins are **precision** (NVFP4 KV ≈ 3.5× fewer bytes,
v10), **occupancy**, and **2× hardware exp** — *not* GB/s. The separate merge kernel becomes an on-chip
2-CTA-cluster + DSMEM merge (Blackwell-only). FP8/NVFP4 KV + asymmetric precision arrive at v9/v10 (GQA
M-packing is now v8 — reordered ahead of the byte cuts; see "What this measurement reorders" below).

**Measured (decode B=1 H=8 N_q=1, non-causal = the real decode workload):**
- **Correct:** 25/25 vs SDPA (square SHAPES via the `num_splits→1` reduction + decode `N_q=1` shapes,
  causal both ways, non-multiple `N_k`).
- **v6 beats the naive `N_q=1` loop (v5 @ N_q=1) 5.7–8.2×**, growing with `N_k` (6.0× @2048 → 8.2×
  @16384, d=64). The split-KV thesis lands: 8–10 splits × BH=8 = 64–80 blocks fill the SMs the
  single-block loop starved.
- **v6 beats torch SDPA 1.5–3.3× (non-causal)** — SDPA isn't tuned for `N_q=1` on Turing. Bigger edge at
  d=64 (GEMV scoring cheaper) than d=128.
- **But %HBM is only ~9–15% (≈7–9× above the HBM floor).** Roofline got the *location* (HBM) right but
  not the magnitude — same flops/bytes blind spot, 5th miss. The actual limiter is still
  **occupancy/launch + reduction latency**: BH=8 gives only ~2 blocks/SM, plus a two-kernel launch and
  an under-occupied merge. Split-KV moved decode from "can't fill the SMs at all" to "fills them
  partially" — it did **not** reach the bandwidth wall.
- **Causal rows are a degenerate-shape artifact:** at `N_q=1` with `q` at row 0, causal masks every key
  but key 0 → 1 effective key. SDPA short-circuits it; v6 doesn't, so causal `vs sdpa` (0.03–0.28×) and
  `%HBM` are meaningless (correctness still matches — it's in the 25/25). Realistic causal decode needs
  the query at position `N_k−1`; the non-causal rows already measure that. Filed as a harness fix.

**What changes on B300 (research §3; SM count now MEASURED in v10):** **148 SMs** (MEASURED on the vast.ai
B300 — the 160 figure is the spec/marketing number, refuted by the device readout) ⇒ more splits (retune
`choose_splits` `num_sm` 40→148);
**HBM bandwidth is FLAT vs B200** ⇒ the real decode wins are **precision** (NVFP4 KV ≈ 3.5× fewer bytes,
v10), **occupancy**, and **2× hardware exp** — *not* GB/s. The separate merge kernel becomes an on-chip
2-CTA-cluster + DSMEM merge (Blackwell-only). FP8/NVFP4 KV + asymmetric precision arrive at v9/v10 (GQA
M-packing is now v8 — reordered ahead of the byte cuts; see "What this measurement reorders" below).

**What this measurement reorders:** because v6 reaches only ~12% of HBM, the next lever is *more
occupancy* (GQA M-packing to grow the grid + raise intensity — research §4's `AI = 2/b → 2G/b`), not
fewer bytes. Low-precision KV only pays once the kernel is actually bandwidth-bound; chasing it before
fixing occupancy would optimize a wall we're 8× away from. So the **reordered arc** (see
[`decode-replan.md §5`](decode-replan.md)): **v7 paged KV** + harness fixes → **v8 GQA M-packing** (the
occupancy lever, promoted) → **v9 FP8 KV** → **v10 NVFP4 + asymmetric precision** (headline) → v11 MLA.

**Correction — "occupancy before bytes" is batch-conditional (deep-research, 2026-06-27):** the 12% is
partly a `BH=8` micro-bench artifact. `choose_splits` self-disables (`num_splits→1`) once
`base_blocks = BH ≥ 2·num_sm`, so at production batch (T4 `BH≥80`, i.e. `B≥10`; B300 `BH≥320`, `B≥40`)
batch *alone* fills the SMs and the kernel reaches its HBM ceiling — there *bytes-first* is correct.
v6 measured only `B=1`, the worst-case corner, so this large-batch end-state is **predicted (code-trace),
not measured** → **v7 must add a `--batch` sweep** to pin the crossover empirically. GQA M-packing is the
hedge that leads regardless: it wins in *both* the small-batch (occupancy) and large-batch
(intensity+reuse) regimes. Honest framing for the paper: occupancy-first holds for `BH ≲ 2·SM`.

**Quiz:** passed 2026-06-27 (Gate 2). **Deep-research close-out 2026-06-27:** 13-agent verify+research
pass → [`decode-replan.md`](decode-replan.md) (paper-grade synthesis, 5 new diagrams). Verified: B300 HBM
BW flat at 8 TB/s; decode `AI=2/b` memory-bound; v6-12%-is-small-batch-artifact. Reframed: FA4 is
acquiring a decode path and FlashInfer/FlashMLA *are* B300-proven, so the contribution is "an open,
roofline-documented FP4 decode kernel vs FlashInfer/FlashMLA," not "we beat FA4." **Next (v7):** paged KV
gather (non-contiguous KV correctness) + the `--batch` sweep + the `--decode` causal query-offset fix.

## Step 7 — Paged KV gather + decode-harness fixes (v7)

*DONE 2026-06-27 (vast.ai Tesla T4 sm_75, torch 2.6.0+cu124), both gates: **51/51 correctness** +
**quiz passed**. **The headline is a refutation:** the `--batch` sweep shows NO occupancy→bandwidth
crossover — decode here is per-CTA-bound (smem cap + single-warp GEMV), not grid-occupancy-bound, at
every batch size. This corrects `decode-replan §2.1`.*

**Bottleneck (predicted): NONE new — v7 is occupancy-neutral.** This is the deliberate design point.
v6 named the limiter (occupancy/launch at small batch); v7 does not touch it. It isolates one variable
— **non-contiguous (paged) KV via a per-sequence block table** — so any measured change is attributable
to the gather alone, and adds two harness fixes that *measure* what v6 could only predict.
`AI = 2/b = 1.0` is unchanged: the gather adds `O(N_k/page_size)` index reads (~0.1% of KV bytes).

**Options:**
- **A — keep contiguous KV (v6), defer paging:** simplest, but never exercises the layout a real
  KV-cache engine uses; the crossover stays predicted-only.
- **B (chosen) — paged pool + block table, v6 kernels otherwise unchanged:** replace the contiguous
  `gj→offset` at the cooperative smem load with a block-table lookup; carry split-KV + LSE merge
  verbatim. Foundational plumbing every later decode kernel (v8–v11) rides on.

**Choices (ratified with the user):**
- **Per-sequence (per-b) block table, vLLM-faithful.** K/V physical pools `[num_blocks, page_size, H,
  d]`; `block_table[b][logical_block] → physical_block`, shared across all H heads of a sequence; the
  kernel derives `b = bh/H, h = bh%H`. (Alternative per-`(b,h)` table was simpler but diverges from
  vLLM's per-sequence convention the mini-vLLM will use.)
- **New public API `paged_attention(...)`** rather than overloading `attention(q,k,v)` — v7's signature
  (pool + block table + page_size + `n_k` + `q_offset`) can't ride the fixed dense API. `attention()`
  and v1–v6 are untouched; v7 is **not** in the dense-API `BACKENDS` list (it gets its own paged tests).
- **True logical `N_k` passed explicitly** (not `n_logical·page_size`) so a non-multiple length never
  scans the last page's padding tail (kernel loops `gj < N_k`).
- **Causal query-offset** (`mask: gj > gi + q_offset`, default 0 = v6): at decode, `q_offset = N_k − N_q`
  puts the query at the cache end so causal == the full scan. Correctness anchors against non-causal
  SDPA (the equivalence), so no bottom-right-mask reference is needed.
- **Deferred:** the merge-kernel `(m,ℓ)`-recompute cleanup (every d-thread recomputes the scalars) —
  kept out to keep v7 a single isolated change. If the `--batch` sweep shows the tiny `(N_q,BH)` merge
  dominating wall-clock at large BH, a single-/persistent-kernel merge becomes **v8.5** (`decode-replan §7.5`).

**What changes on another arch:** identical to v6 — `choose_splits` `num_sm` 40→160 on B300; the
crossover threshold `BH ≥ 2·num_sm` scales with it (T4 `B≥10`, B300 `B≥40` @ H=8). Paging itself is
arch-independent (pure indexing); the gather's L1/L2 residency assumption holds across arches.

**Measured (2026-06-27):**
- **Correctness 51/51** vs SDPA (paged shuffled-pool gather + v6 regression), tol 2e-2. The gather
  indexing is correct; v6 is untouched.
- **The `--batch` crossover sweep REFUTES the prediction.** At fixed `N_k=8192`, `%HBM` is **flat at
  9.4–12.4% from BH=8 to BH=512** (per-token cost ~constant, even at B=64 where `num_splits→1` gives
  12.8 blocks/SM — far past full occupancy). The predicted climb to 60–80%+ by BH≈80–160 **does not
  happen.** Decode here is **per-CTA-bound, not grid-occupancy-bound**, and batch replicates the
  inefficiency rather than curing it.
- **Why (code-verified):** (1) `sK+sV = 32 KB`/block caps residency at **2 blocks/SM** (T4 64 KB/SM),
  independent of batch/split count — more blocks = more waves at the same per-SM occupancy = linear
  time = flat %HBM; (2) at `N_q=1` only **1 of 8 warps computes** (warp `w` owns query row `w`), so ~2
  active compute-warps/SM can't hide the per-key shuffle-reduction latency (v4's GEMV wall).
- **Causal query-offset fix works:** causal µs/tok ≈ non-causal per row (66.79 vs 66.65 @64/8192) — the
  query at `N_k−1` does the full-cache scan. (Bench `vs sdpa` for causal is now meaningless in the other
  direction: SDPA's baseline still uses the degenerate upper-left mask. `vs naive` stays valid.)
- **Apparent paging overhead:** v7 ~15–25% slower than v6's recorded run at B=1 (dependent block-table
  load + div/mod per key). Cross-session clocks unverified (`clock~-1/-1`) → indicative, not proven.

**What this measurement reorders (sharpened, and a correction):** the "occupancy before bytes" reorder
**still holds**, but the data corrects *why* and kills the batch-conditional hedge:
- **The reason is per-CTA efficiency, not grid occupancy.** GQA M-packing (v8) leads not to "fill the
  SMs" (batch already over-fills the grid — and it doesn't help) but because `M = G > 1` activates **G
  compute-warps/block** *and* turns the GEMV into a tensor-core GEMM with KV read once. It attacks the
  smem-cap + single-warp-GEMV limiter directly.
- **Bytes-first is premature at *all* batch sizes, not just small batch.** v7 never gets within ~8× of
  the bandwidth wall at any BH (flat ~10–12%), so FP8/NVFP4 (v9/v10) multiply a term the kernel can't
  reach. This **refutes `decode-replan §2.1`'s batch-conditional claim** ("at production batch the kernel
  reaches its HBM ceiling, so bytes-first is right there"). It isn't — not until per-CTA efficiency is
  fixed. The reordered arc (v7 → **v8 GQA M-packing** → v9 FP8 → v10 NVFP4 → v11 MLA) is unchanged; the
  justification is now "GEMV→GEMM before bytes," measured at all batch sizes.
- **Possible v8.5 contingency:** the merge kernel + two-kernel launch were *not* isolated as the limiter
  (the smem cap + single-warp GEMV dominate), so a persistent/fused merge is lower priority than first
  thought — but worth a pipe-util read on bare metal to confirm the smem-residency story directly.
- **Adversarial close-out (2026-06-27, 35-agent pass: 6 forensics + 7 research + 7 claims through a 2-of-3 gate):** the per-CTA-bound headline
  **survives 0/3 refute**; the "GQA before bytes" reorder **survives 0/3**. Two premises corrected (both were
  "sound conclusion + stale premise"): (a) **%HBM is fp16-correct, not 2× understated** — the kernel casts K/V
  to half (`paged_attention.cu:274–276`), so `precision=fp32` is a cosmetic header label; the "is %HBM
  understated?" worry closes **NO**; (b) the apparent ~15–25% v6→v7 paging overhead stays **UNPROVEN**
  (separate vast.ai sessions, `clock~-1/-1`; same-shape delta 19–27% but inseparable from clock drift) →
  resolve with a **same-session v6/v7 A/B** in v8's harness. New finding: **SDPA overtakes v7 by B=8** (v7
  SM-saturated → flat; SDPA amortizes launch ~4.5×) → v8 gains a **"reclaim SDPA at batch"** deliverable.
  **Carried-forward cleanups (not blockers):** the bottom-right causal-mask reference, and stale comments
  asserting the refuted "split-KV fills the SMs → HBM ceiling" in `paged_attention.cu:221–226` +
  `roofline/model.py:96–99`.
- **Competitive framing (research B4/B7):** name **FlashInfer** (`use_tensor_cores=True`) + **TRT-LLM XQA** as
  v8's M-packing comparators and **vLLM PagedAttention v2** (split-KV + paging, *no* M-packing — the cleanest
  isolation of v8's one variable) as the CUDA-core baseline floor. **Do NOT claim "beat FA4":** the
  `fa4-no-decode` claim **died 3/3** — FA4's decode path is now *upstreamed* (Modal split-KV/single-query/
  paged/FP8/`pack_GQA` PRs; pack_GQA = 2.92×), so it already carries the split-KV+GQA+FP8 levers v6–v9 build.
  Frame as "open, roofline-documented, measured vs FlashInfer/FlashMLA." B300 confirmed: HBM **flat 8 TB/s**,
  288 GB, NVFP4 15 PF dense, exp **2× (10.7 TeraExp/s)**, `sm_103`. *(Full synthesis:
  [`v7-deep-research.md`](v7-deep-research.md).)*

**Quiz:** passed 2026-06-27 (Gate 2). Kien nailed the refutation (flat %HBM, crossover false) and the
per-CTA mechanism: smem caps residency at 2 blocks/SM, so batch replicates an inefficient block without
making it efficient. **Next (v8 — GQA M-packing):** pack `G` query heads into `M` (GEMV→`M=G` GEMM,
tensor cores re-engage, KV read once, `AI = 2/b → 2G/b`, **G active warps/block**). v7's data says v8
must lead at *all* batch sizes; `decode-replan §5`.

## Step 8 — GQA M-packing (v8)

*Cut 1 MEASURED 2026-06-28 (Colab T4 sm_75): **Gate 1 ✅ 64/64**; G-sweep + reclaim-at-batch captured.
**Quiz (Gate 2) pending.** Cut 2 (sm_80 tensor cores + M≥16 ablation) still `[RENT A100]`. The build is
STAGED (user decision): Cut 1 = CUDA-core M-pack on sm_75/T4 (cheap correctness, M-packing isolated as one
variable); Cut 2 = sm_80 tensor-core variant + the M≥16 ablation.*

**Bottleneck (predicted/attacked): the per-CTA wall v7 MEASURED.** v7 proved decode is per-CTA-bound at
every batch size (flat ~10–12% HBM, BH=8→512): at `N_q=1` only **1 of 8 warps computes** and `sK+sV=32 KB`
caps residency at 2 blocks/SM. v8 attacks exactly this: packing the `G = H_q/H_kv` query heads of a group
into `M` lights up **G compute-warps** against a KV head **read once**, raising `AI = 2/b → 2G/b`. The win
is **per-CTA efficiency, not occupancy** (v7 proved filling the grid doesn't move %HBM).

**Options:**
- **A — bytes first (FP8/NVFP4 KV):** refuted by v7 — the kernel never gets within ~8× of the bandwidth
  wall at any batch, so cutting bytes multiplies a term it can't reach. Deferred to v9/v10.
- **B — persistent/fused merge (v8.5):** the two-kernel launch + merge were *not* isolated as the limiter
  (smem cap + single-warp GEMV dominate), so this is lower priority. Deferred.
- **C (chosen) — GQA M-packing:** the one lever that directly converts the 1-of-8-warps GEMV into a
  G-row GEMM with KV read once. Leads the reordered arc at *all* batch sizes.

**Choices (ratified with the user):**
- **Staged arch: Cut 1 CUDA-core (sm_75/T4) before Cut 2 tensor-core (sm_80).** The kickoff doc targeted
  sm_80 directly, but a Turing T4 can't run `mma.m16n8k16`/`cp.async`, so a tensor-core-from-the-start
  kernel couldn't use the cheap T4 correctness loop. Cut 1 keeps both matmuls on CUDA cores — it still
  gets G-warps-active + KV-read-once + `AI=2G/b`, just without the GEMV→GEMM tensor-core uplift — so
  M-packing is validated cheaply and isolated as one variable. `_MIN_CAPABILITY["v8_gqa"] = (7,0)` for
  Cut 1; bumps to `(8,0)` when Cut 2 lands.
- **Fork v7 verbatim; change only the index math.** The hot loop (cooperative paged gather + online
  softmax + O-rescale), the LSE merge, and `choose_splits` are carried byte-for-byte. The only changes:
  the warp→work mapping (`m_row → (g_local, i_q, h_q)`), the gather head (`H→H_kv`), and `base_blocks =
  B·H_kv·row_tiles`. Two correctness traps encoded: the causal mask uses the **query position `i_q`**
  (not the packed row `m_row` — packed rows share a position), and the workspace/merge stay **query-head
  shaped** `[B,H_q,N_q,S,*]` while the pool gather is KV-head shaped.
- **`G` derived from shapes, not an argument.** `H_kv = k_pool.size(2)`, `G = H_q/H_kv`; new
  `gqa_attention(...)` API mirrors `paged_attention` (pool now has `H_kv` heads). v8 is **not** in the
  dense `BACKENDS` list — it gets its own GQA tests (like v7).
- **GQA reference uses `repeat_interleave(G)`, not `repeat`.** KV head `h_kv` must serve query heads
  `[h_kv·G, h_kv·G+G)` to match the kernel's `h_q = h_kv·G + g_local`; `repeat`/tile would be a
  silent-wrong oracle.
- **Cut 2 ablation (full, user decision): pad-M=8→16+mask vs multi-group-pack vs CUDA-core-QK.** G=8 < 16
  doesn't fill a tensor-core tile, and the three approaches disagree on magnitude — the G-sweep across the
  M<16→M≥16 threshold is the headline experiment.

**What changes on another arch:** Cut 1 is arch-independent CUDA-core math (runs sm_70+). Cut 2 needs
sm_80 (`mma.m16n8k16` + `cp.async`); the tile/`BLOCK_M` retune per arch. `choose_splits` `num_sm` 40→108
(A100)→160 (B300). The split-KV self-disables at `G×` larger batch than v7 (z-extent is now `B·H_kv`),
which is fine — batch never moved %HBM anyway.

**Roofline prediction (recorded, A100 sm_80, decode `N_q=1`, `N_k=8192`, FP16):** `AI = 2G/b` rises
exactly `G×` (1.0→8.0 at G=8) and the HBM floor drops `G×` (0.132→0.016 ms), but the limiter **stays
HBM** — A100's FP16 ridge is 153, so even G=8 (and G=32, AI≈32) is far below. **No limiter flip in the
realistic GQA range.** So the headline is *not* "v8 becomes compute-bound"; it's that the kernel should
move much closer to the now-`G×`-lower floor (the roofline has no schedule term — magnitude-wrong 5
straight steps — so the µs/tok drop + reclaim-SDPA-at-batch are the deliverables, not the floor).

**Measured (Cut 1, 2026-06-28, Colab T4 sm_75, `notebooks/v8_gqa_gate_output.ipynb`):**
- **Gate 1 ✅ 64/64 correctness** (38 v8 + 26 v7 regression, tol 2e-2, 81.8 s). The idle-warp `G=3` and
  multi-tile `G=16` cases pass → `G` need not divide 8 and `M>8` tiles over `grid.x` correctly; the
  `repeat_interleave(G)` oracle confirms the head mapping.
- **The G-sweep CONFIRMS the per-CTA mechanism (the headline):** `vs no-pack` (= v8 ÷ v7 on the same
  workload, KV broadcast to `H_q`) tracks **~`G×` up to G=8 — 8.59×/8.71× (d64/d128) at G=8** — then
  sub-linear (12.7× at G=32, where `M>8` re-stages KV over `⌈G/8⌉` blocks). µs/tok falls ~15×/12.7× from
  G=1→32. This is `AI=2G/b` realized as wall-clock (`G` warps + KV read once), **on CUDA cores, no tensor
  cores** — Cut 1's thesis lands.
- **Reclaim-SDPA-at-batch ✅:** at G=8, v8 beats torch SDPA **6.1–9.9× across B=1→64** — where v7 *lost*
  (0.34–0.56× at B≥8). The serving-batch regime is reclaimed, the v7-data deliverable delivered.
- **Prediction-vs-measured — a partial roofline WIN (first in 6 steps):** the `AI=2G/b` model got the
  *speedup magnitude* right (~`G×`), but `%HBM` stays **≤11%** at every G → still **per-CTA-bound, NOT
  bandwidth-bound** (the kernel never reaches the floor the model draws; M-packing closes most of the gap,
  ~9× headroom remains for Cut 2's tensor cores). So the model's relative scaling was right; its absolute
  floor stays unreached — as flagged (no schedule term).
- **Caveats (not failures):** Colab T4 clock-throttled (`~360-390/1590 MHz`) → absolute µs/tok slow but
  same-session ratios robust; causal `vs sdpa` is the carried top-left-mask artifact (correctness uses the
  honest `causal=False` oracle); `B=64,d128` `vs no-pack`=nan is an OOM in the auxiliary G-expanded
  baseline, not a v8 failure.
- **Still pending:** Cut 2 `[RENT A100]` — the 3-way M≥16 ablation vs FlashInfer / XQA / vLLM
  PagedAttention v2; the quiz (Gate 2).

**Measured (Cut 2a — Turing WMMA tensor cores, 2026-06-28, Colab T4, `notebooks/v8_gqa_tc_gate_cut2a_output.ipynb`):**
- **Correctness ✅ 38/38** (`pytest -k v8_gqa_tc`, 2.7 s) — the WMMA GQA M-pack kernel is correct, incl.
  the **all-masked-block guard** I added in review (the split+WMMA edge case where a fully causally-masked
  split row would inject `l+=BN`; v5/Cut1 dodge it via full-attn-block-0 / `break`). Blind-written kernel +
  fix landed first try.
- **Perf prediction REFUTED — WMMA is SLOWER than Cut 1's CUDA-core GEMV at every G** (1.2–3.3× raw; ~1.4–1.6×
  clock-normalized at G=8; `%HBM` 2–3× lower). It loses even at **G=16/32 (full WMMA tile)**, so it is NOT
  the pad-to-16 waste. **Mechanism:** v5's GEMV→GEMM win was for *prefill (large M)*; decode `M=G≤16` is too
  small for the 16×16×16 GEMM to amortize the opaque-fragment tax (QK→smem→softmax→P-as-half→reload→PV +
  extra syncs). The lean register-resident CUDA-core GEMV wins. **This re-frames the whole Cut-2 premise:
  Cut 1's 8.6× was G-warps + KV-read-once, NOT tensor cores — tensor cores are a prefill tool the decode
  thesis didn't need.**
- **Honesty caveats:** (a) the A/B's two runs read different throttle clocks (tc~360 vs cuda~555 MHz), so
  ~1.5× of the raw gap is clock — but the direction is uniform (12/12 rows) and `%HBM` (a clock-robust ratio)
  confirms it; (b) Cut 2a is a correctness-first **1-warp/block** schedule, so some slowdown is its under-fed
  KV load, not purely the tensor cores. A clean same-clock re-run would tighten the magnitude (not the sign).
- **Decision:** ablation arms 2 (multi-group→full M=16) and 3 (CUDA-core-QK + WMMA-PV) are **documented but
  not pursued** — G=16 is already a full tile and still lost 2.1×, so they won't rescue WMMA. Cut 2b (A100
  `mma.m16n8k16` + cp.async) becomes an **open question, not a foregone port**: worth a rental only to test
  whether load-overlap + finer fragments + occupancy overturn the Turing result.

**Measured (Cut 2b PROBE — A100-SXM4-80GB sm_80, 2026-06-28, `notebooks/v8_cut2b_a100_probe_output.ipynb`):
the verdict did NOT flip — Cut 2 is CLOSED.** Rather than blind-write the hard `mma`+cp.async kernel on a
weak hypothesis, we probed the *existing* kernels on a rented A100 (per-kernel gencode now sm_75+sm_80 so
both build/run on Ampere; 38/38 correctness on A100).
- **WMMA loses HARDER on A100 — 1.8–4.6× slower than the CUDA-core GEMV** (vs 1.2–3.3× on T4). The smoking
  gun: WMMA **barely moved T4→A100** (G8/d128 42→39 µs/tok) despite ~5× tensor-core throughput + ~6× HBM BW,
  while the CUDA-core path nearly halved (16.9→9.7). So WMMA is **neither compute- nor bandwidth-bound** —
  it's pinned by per-CTA scheduling overhead (opaque-fragment smem-softmax round-trip + 1-warp load) that
  doesn't scale with the GPU. Reclaim-at-batch: Cut 1 beats SDPA **2.5–8.6× all batch on A100**; WMMA
  **loses to SDPA at B≥8**. (Caveat: A100 start-clocks read 330–1410 MHz, but warmup boosts before timing and
  the T4→A100 non-scaling is clock-independent.)
- **Decision: do NOT build the cp.async/`mma.m16n8k16` kernel.** cp.async only fixes the load-starvation
  half; the opaque-fragment tax is fundamental and the gap *worsened* on the faster GPU. **The GEMV→GEMM
  fix that won for v5 prefill is the wrong tool for decode — a clean two-architecture negative result.**
  v8's deliverable is **Cut 1 (CUDA-core GQA M-packing)**. Tensor-core decode would need SOTA scheduling
  (FlashMLA/FlashInfer), out of scope and irrelevant to the v8 thesis (occupancy, not compute).
- **What this means for the arc:** v8 is a clean win (Cut 1) + a clean negative (tensor cores). The decode
  thesis stands: the lever is per-CTA efficiency (G warps + KV-read-once), and the next real lever is
  **bytes (v9 FP8 / v10 NVFP4)** — but only once a kernel is actually bandwidth-bound, which neither v7
  (per-CTA) nor v8 (still ≤11% HBM) is. The honest open question carried forward: closing the remaining
  per-CTA gap on CUDA cores (more warps, better load) before bytes.

**Claim discipline (carried from v7):** frame as "an open, roofline-documented decode kernel measured vs
FlashInfer/FlashMLA"; op-level ~2.9× single-token decode, NOT end-to-end, NOT "beat FA4" (FA4's decode
path is upstreamed). See `decode-replan §5`, `v8-kickoff.md`.

## Step 8.6 — attack the reduction wall: occupancy vs key-ILP (v8_gqa_occ, v8_gqa_ilp)

**Bottleneck (measured, not predicted):** after v8.5's null result (double-buffering the KV load moved
nothing), the decode floor is pinned to the **per-key warp-shuffle reduction + serial online-softmax
recurrence** — the kernel is *compute-latency-bound at ~10% HBM*, not memory- or compute-throughput-bound.
The latency is exposed because one warp's inner loop is a serial chain (`__shfl_xor` butterfly → dependent
softmax update) with too few independent warps (~2 blocks/SM) to hide it.

**Options to hide that latency (the user-scoped lever set — *hide*, not *remove*):**
1. **Occupancy** — halve smem (FP16 stage, single buffer: 16 KB → 4 blocks/SM) to double resident warps
   (TLP hides the whole chain). → **Arm 1, `v8_gqa_occ`.**
2. **Key-ILP** — unroll the key loop (KU=4) so independent reduction chains pipeline. → **Arm 2,
   `v8_gqa_ilp`** (FP32 smem kept, 2 blocks/SM, so ILP is the lone variable).
3. **Vectorized loads** — rejected: the lane-strided dot-product layout (`lane + 32·e`) is non-contiguous,
   so float2/half2 don't apply; and v8.5 already showed loads aren't the wall.
4. **Score-stationary rewrite** (one key per lane → no per-key cross-lane reduction) — *removes* the wall
   but is a full redesign; **deferred** to a future v8.7, gated on whether 1/2 move the needle first.

**Choice:** a **2-arm single-variable ablation** (Arms 1 & 2), CUDA-core / T4-cheap, both forking Cut 1.
Matches the arc's discipline (Cut 2 and v8.5 were each clean single-variable runs) and *attributes* any
gain to occupancy vs ILP rather than confounding them in one kernel.

**Prediction (recorded before measuring):** the byte-roofline is **blind** (AI=2G/b, floor, limiter
identical to Cut 1 for both arms — confirmed on the T4 arch). Mechanistically **occupancy > ILP** (TLP
hides the entire serial chain; ILP overlaps only the reductions, leaving the softmax recurrence exposed).
**If both null → the floor is the serial recurrence itself** → score-stationary redesign is the real fix
and v9 FP8 stays premature.

**What changes on another arch:** more warps/SM and larger register files (A100/H100) raise the occupancy
lever's ceiling (Arm 1 should help more there); the ILP lever is arch-insensitive (it's instruction-level).
On a genuinely bandwidth-bound regime (N_k past L2, or large batch saturating HBM) the whole question flips
and bytes (v9) finally pay — neither holds on this T4 micro-bench.

**Measured (2026-06-28, Colab T4): correctness ✅ 190 passed; BOTH levers NULL on the clock-robust `%HBM`
(flat ~8–10% at every G/batch — clocks varied 360–1590 MHz so µs/tok is unreliable). Occupancy never even
engaged at small batch (split-KV already fills 2 blocks/SM → no spare block for the 4-block ceiling); ILP
tracked Cut 1 exactly. The counter-prediction landed: the floor is the per-row serial online-softmax
recurrence — unhideable by TLP or ILP. Fourth consecutive negative → mandate for v8.7 (remove, don't hide).
See `results.md` Step 8.6 + `decisions.md`/`results.md` Step 8.7.**

## Step 8.7 — score-stationary decode inner loop (v8_gqa_ss)

**Bottleneck (now four-times measured):** the per-row serial online-softmax recurrence + the per-key
warp-shuffle reduction in Cut 1's output-stationary GEMV inner loop. Unhideable (Cut 2 / v8.5 / v8.6 occ /
v8.6 ilp all null). The only move left is to **remove** it.

**Options:**
1. **Score-stationary relayout (lane=key)** — full per-lane dot product (no per-key reduction), per-32-key-group
   softmax (recurrence 32× shorter), PV transpose via pipelined `__shfl` broadcasts. The textbook
   FlashDecoding inner loop. → **chosen (v8.7).**
2. **Thread-group (>1 lane/key, tunable group size)** — vLLM-style; balances per-lane dot length vs a tiny
   group reduction. More tunable but more complex → **deferred to v8.8** if pure 1-lane/key is smem-BW-bound at d=128.
3. **Bytes (v9 FP8)** — premature: the kernel is per-CTA-bound at ~10% HBM, not bandwidth-bound, so cutting
   bytes can't help latency (v8.5/v8.6 proved it). Gated on whether v8.7 makes it load/bandwidth-bound.

**Choice + the smem sub-decision:** pure score-stationary, single-variable vs Cut 1 (inner-loop layout only).
**smem staged FP16 to HOLD occupancy** — the layout forces extra smem (`sQ` + a bank-conflict pad on the
transposed K); in FP32 that would drop the T4 from 2→1 block/SM and confound a null with lost residency. v8.6
already measured FP16 smem is correctness-safe + perf-neutral, so FP16 (~18 KB → ~3 blocks/SM) holds
occupancy and *also* makes `v8_gqa_ss` differ from `v8_gqa_occ` in **only** the inner-loop layout → a clean
layout-isolated A/B (headline stays vs Cut 1).

**Prediction (before measuring):** roofline blind (AI=2G/b unchanged). µs/tok should drop, best at d=64;
d=128 at risk of flipping to smem-read-BW-bound (full-D `sK` reads per key). Counter: if µs/tok drops but
%HBM stays ~10%, the floor is per-CTA load latency → v9 FP8 (capacity) becomes the right next lever.

**What changes on another arch:** A100/H100 have far more smem-read bandwidth + bigger register files, so the
d=128 smem-BW risk (R2) shrinks and the score-stationary win should be cleaner there; the layout itself is
arch-independent (it's the standard decode primitive FlashInfer/FlashMLA use).

**Measured (2026-06-28, Colab T4): WIN — ✅ 228 passed; ss beats Cut 1 1.1–1.6× at matched clock across G
and batch (best at d=64; d=128 wins less — 1.16–1.18× — confirming the R2 smem-read-BW drag but NOT a null),
beats SDPA 8–16×, and is the FIRST lever since Cut 1 to move the clock-robust `%HBM` (up ~2–3 pts toward
~10–12%). ss vs occ (both FP16 smem) confirms it's the LAYOUT (ss faster at a lower clock). Two truths: the
reduction/recurrence WAS a real component of the floor (remove-not-hide vindicated, v8.6 correctly said hiding
wouldn't work), BUT %HBM plateaued ~10–12% (not the floor) → a residual per-CTA ceiling remains; v8.7 did NOT
make the kernel bandwidth-bound, so v9 FP8's value stays capacity+accuracy, not micro-bench latency. Closes
the decode-schedule arc: M-packing + score-stationary are the two real decode levers. See `results.md` Step 8.7.**

---

## Step 8 close-out — deep-research audit + v9 decision (2026-06-28)

**Context:** before opening v9, a 6-investigation close-out (3 forensic on code/results/notebooks + 3
web-research) audited the v8 family and the planned roadmap, with a mandate to question every conclusion.

**What it confirmed (keep):** M-packing's 8.6×-vs-v7 / 6–10×-vs-SDPA headline (same-session ratios, clock-
robust) and the WMMA negative (T4→A100 non-scaling = tensor cores wrong for an M≤16 decode GEMV).

**Bottleneck (the correction):** the limiter diagnosis recurring since v6 — "per-CTA-bound, NOT bandwidth-
bound (~10% HBM)" — is **confounded and not yet earned**. (1) **L2 residency:** the bench KV fits in the
T4's 4 MB L2 (reclaim G=8/B=1 → H_kv=1 → ~4.2 MB ≈ L2), so the DRAM counter reads low even if the kernel is
memory-bound — bound by the wrong memory. (2) **Unlockable clock:** free Colab can't pin clocks (CUR swung
360–1590 MHz), so absolutes/cross-run comparisons carry a ± band. The bench never measured L2 traffic, locked
clocks, or pushed N_k past ~16K. *Partial counterweight:* low-G configs (KV ~67 MB ≫ L2) still showed ~11%
HBM. Net: probably per-CTA-bound, unproven. See `results.md` "Step 8 — threats to validity".

**Options for v9:** (a) FP8 KV + fold the regime-fix in as Task 1; (b) a separate pure-measurement step
first; (c) pivot (mini-vLLM integration / prefill). **Choice: (a)** — `v9 = FP8 KV + regime-characterization`.
Task 1 (gating, existing `v8_gqa_ss` on a root T4): lock clocks, flush L2, sweep N_k 1K…128K × batch × d ×
H_kv measuring HBM%/L2-hit-rate/L2-BW% + the counter-free L2 test (effective BW > HBM peak ⇒ L2-served). Task 2:
FP8 E4M3 KV (fork of ss) with **fused per-tile dequant** (a prepass eats the savings — QServe), FP32 accum.
*Why fold-in, not a separate step:* the same FP8 byte-cut is what creates the memory-bound regime, and after
six steps the user wanted forward progress, not another detour. The decode-byte arc (v9→v10→v11) is preserved
and now rests on an earned verdict.

**FP8 framing (web-research, regime-scoped):** "FP8 = capacity-only, not latency" is right for the L2-resident
micro-bench but too strong generally — FP8 KV becomes a real latency win past ~4–7k tokens / under load (vLLM:
per-token cost → 54% of BF16). So v9's latency deliverable is two-regime: L2-resident null vs past-L2 win.

**Hardware (UPDATED — B300 is the final goal, the paper's novelty):** v9 = **T4, no rental** (a decode
GEMV uses no tensor cores; FP8's win is storage bytes, decoupled from sm_89+ FP8-MMA). v10/v11 (NVFP4,
MLA/spec) = **B300 / GB300 (sm_103)** — the project's research north star, because **no published FA
*paper* has characterized a B300** (FA4 stops at B200/sm_100). Honest scope (survives a reviewer):
production libs (FlashInfer/FlashMLA) already run measured GB300 decode, so the contribution is *the
first open, roofline-documented, prediction-vs-measured FA decode study on sm_103 + the
asymmetric-precision FP4 recipe + the honest methodology*, vs those libs — **not** "first to run on
GB300." B300-specific levers the paper exploits: 2× exp/SFU throughput (10.7 TeraExp/s, hits softmax's
MUFU term), 288 GB (holds the long-context KV that reaches the bandwidth-bound regime), NVFP4 15 PF
dense. **B200 (sm_100, ~$3.44/hr) is an OPTIONAL cheaper dev rung** (most tcgen05/TMEM code ports to
sm_103); the *record* runs on **B300 (~$5.44/hr, spot ~$2.45)**. RTX 5090 (sm_120, different PTX) /
T4-emulated FP4 are budget fallbacks only. *(Supersedes the earlier "B200 over B300" note — that was
right for a pure engineering micro-bench, wrong for a publishable contribution where sm_103 IS the
differentiator.)* Regime-method dev stays on T4 (4 MB L2 spills at tractable N_k); the B300 long-context
roofline (huge L2) is itself a paper figure.

**What changes on another arch:** the L2 confound is arch-relative — bigger L2 (A100 40 MB, H100 50 MB,
B200/B300 **132.6 MB MEASURED** in v10 — refutes the 126–192 MB estimates) pushes the L2-spill N_k far higher, so the "bandwidth-bound regime" needs proportionally
longer context there. The decode roofline math (AI=2/b, memory-bound at every reachable precision) is
arch-independent; only *where the kernel reaches the floor* moves.

See [`v9-kickoff.md`](v9-kickoff.md), `decode-replan.md` §5 v9/v10 + §2.1 L2 confound, `interview-prep.md` C12.

## Step 9 — FP8 E4M3 KV cache (v9_fp8) — Task 2 (the kernel)

**Bottleneck (v8.7's residual ceiling):** `%HBM` plateaued ~10–12% — a per-CTA limit (load latency /
small-CTA launch) **OR** an L2-residency artifact (the bench KV fits in the T4's 4 MB L2). Confounded.
v9 splits into **Task 1** (earn the bandwidth verdict on a root T4 — locked clock, L2-flush, N_k sweep)
and **Task 2** (the FP8 kernel). This step is Task 2; Task 1 is deferred (scope decision this session).

**Options (KV byte-cut mechanism):**
1. **FP8 E4M3 KV with fused per-tile dequant** — dequant at the smem gather (where v8.7 already converts
   to FP16), per-tensor FP32 scale, FP32 accumulation. → **chosen (v9).**
2. **Post-load / per-key dequant in the inner loop** — dequant per lane per key; redundant work on the
   shared tile → **rejected** (throughput-worse, and pollutes the byte-only ablation).
3. **Full-cache dequant prepass** → **rejected** (re-reads the whole cache, eats the byte savings — QServe).
4. **Native FP8 tensor-core math** → **deferred to v10** (needs sm_89+ FP8 MMA; decode is a GEMV, no
   tensor cores, and the paper's FP4 throughput lever is a B300/sm_103 story).
5. **int8-symmetric instead of E4M3** → **fallback only** (same 1-byte HBM win; kept as a 1-line swap if
   E4M3 fails ptxas on sm_75; the E4M3-vs-int8 accuracy gap is itself a deliverable if we run both).

**Choice + sub-decisions:** fork `v8_gqa_ss` → `v9_fp8` changing ONLY the KV path. KV pool = uint8 (FP8
E4M3 bytes); two new runtime args `scale_k/scale_v` (per-tensor, = amax/448); Q stays FP16; merge kernel
untouched (FP32 partials). **Per-tensor scaling first** (research: near-lossless at 8-bit); per-token V /
per-channel K only if RMSE demands it (that comparison is a deliverable, not a default). The correctness
oracle is **apples-to-apples** (SDPA on the same dequantized E4M3 bytes, tol 5e-2) so it isolates the
kernel's math from the quantization error; the **quantization RMSE vs fp16 KV** is reported separately as
the accuracy deliverable.

**Prediction (before measuring):** roofline blind to dequant latency; AI doubles 8.0→16.0, HBM floor
halves 13.12→6.56 µs, limiter stays HBM. **Mechanistic:** L2-resident micro-bench → likely capacity-only
(no µs/tok win). **Counter:** if the bench prints "L2!" (`effective_bw > HBM peak`), `%HBM` isn't a
boundedness signal here and FP8's latency win lives past L2 / under load — which Task 1 creates.

**What changes on another arch:** FP8's **capacity** win (2× KV cache at fixed memory) is arch-independent
and is the durable deliverable. The **latency** win is arch- and regime-dependent: it appears only where
the kernel is genuinely bandwidth-bound, which a bigger L2 (A100 40 MB … B300 132.6 MB MEASURED in v10) pushes to far
longer context. On B300 (sm_103, the paper) the same dequant machinery carries forward to NVFP4 (v10),
where native FP4 throughput (15 PF dense) adds a *compute* lever this T4 GEMV doesn't have.

**Measured (2026-06-28, Colab T4): correctness ✅ 76 passed; E4M3 built on sm_75 with NO fallback; the
"capacity-only" prediction is REFUTED — FP8 buys a real ~1.3× decode latency win, and it's a LOAD-bandwidth
win (not capacity-only, not HBM-bandwidth).** `vs naive` (FP8÷FP16 v8.7, same packing, same session,
clock-matched) = **0.96–1.52×, median ~1.3×**, FLAT across batch B=1→64. Airtight as a load-bytes effect:
the only kernel change is the 1-byte-vs-2 global load (smem stays FP16 → identical occupancy/inner loop),
and FP8 is faster *despite* adding software E4M3→half ALU. **Smoking gun:** the win shrinks as G grows
(d=64 1.37 G2 → 0.96 G32) because M-packing amortizes the KV load across G heads → the bottleneck shifts
load→compute, so fewer bytes matter less; d=128 wins more (2× bytes/key). `%HBM` *dropped* vs FP16 (no
`L2!` flag), so FP8 did NOT flip the limiter — still per-CTA/L2-load-bound ~10% HBM, ~14× above the halved
floor. Accuracy: per-tensor E4M3 quant RMSE **~6–7e-4** (oracle ~6e-6 → kernel math exact). **Decision
update:** the "bytes won't help latency until past-L2" claim (recurring since v6) was too strong — the
residual post-v8.7 ceiling is *partly* L2→SM load bandwidth, which FP8 relieves. "L2-resident" ≠ "memory
is free." Task 1 (locked-clock past-L2 sweep) + v10 NVFP4 (a *compute* lever) remain the bandwidth story.
**Harness fix (post-run):** v9's `vs sdpa` was inflated (the FP8 oracle re-quantized inside the timed
baseline); now dequantized once outside — trust `vs naive`. **Gate-2 quiz PASSED 2026-06-28 → Step 9 DONE.**

See [`v9-kickoff.md`](v9-kickoff.md), `results.md` Step 9, `interview-prep.md` C13.

## Step 9 — Task 1: regime characterization — VERDICT (per-CTA-bound, confound-free)

**Measured (2026-06-28, ROOT T4, clocks LOCKED 1590 MHz no-throttle, L2 flushed, ncu live): the decode
kernel is GENUINELY PER-CTA-BOUND, not bandwidth-bound — the six-step verdict is now EARNED.** Three
confounds removed at once: clocks pinned (was the unlockable-clock confound), L2 flushed + N_k pushed to
128K so the working set hit **537 MB ≫ 4 MB L2** (was the L2-residency confound), and — for the first time
in the project — **ncu counters worked** (every prior step had `ERR_NVGPUCTRPERM`; the root T4 unlocked
them).

**The evidence:** `v8_gqa_ss` achieved-%HBM plateaus at **~11–14% (low occupancy, H_kv=1)** rising to a
hard **~28–29% ceiling (H_kv=8 or batch≥8)** — and stays there to a 1 GB working set, never near the ~70%
achievable ceiling. ncu confirms past L2 the **L2 hit-rate is 1.1%** (data genuinely from HBM) while **DRAM
is only 12.85%** → HBM-served yet ≈13% busy = per-CTA-bound. The `%HBM`-vs-N_k curve rises (launch-overhead
amortization) then plateaus, with **no bandwidth knee at the L2 crossing** — the cleanest possible proof a
kernel that isn't bytes-bound doesn't care where the bytes live.

**Decision impact (locks several open threads):**
- **The limiter is named, confound-free:** per-CTA / occupancy (at N_q=1 each CTA runs ~1 compute warp →
  low memory-level parallelism). "~10% HBM" was an *occupancy* artifact, not the floor; the real cap is
  ~28% of peak. This retires the "per-CTA OR L2-resident, unproven" hedge that has run since v6.
- **Bytes are NOT the decode wall** on this kernel → **FP8/NVFP4 are confirmed a capacity+accuracy play,
  not a decode-latency/bandwidth play** (corroborates Step 9 Task 2's L2-load-bandwidth framing). v10
  NVFP4's contribution is capacity + accuracy + the B300 long-context regime + native FP4 *compute*, NOT a
  micro-bench bandwidth-latency claim.
- **The counter-free %HBM method is validated** against ncu DRAM% (13.8% vs 12.85% at the same shape) — so
  every prior counter-free decode reading is retroactively trustworthy *as a throughput proxy*; what was
  missing was the L2-vs-HBM discriminator, which only ncu (or the now-known one-sided `L2!` test) provides.
- **Reopener — RESOLVED (2026-06-29, `results.md` Step 8.5/8.6 past-L2 re-test):** re-ran v8.5 (double-buffer)
  + v8.6 (occ/ILP) through `bench.regime` **past L2** (clock-robust speedup-vs-Cut-1). **db & ILP are dead even
  at N_k=131072** (the L2-artifact hypothesis is REFUTED — the floor is the serial recurrence, only the v8.7
  *relayout* removes it; "decode-schedule CLOSED" is confirmed confound-free at B=1). **One amendment:** the
  **occupancy arm (`occ`) revives to ~1.4× at B≥32 past L2** (4 blocks/SM only pays once the grid fills it —
  invisible at the B=1 micro-bench every prior measurement used). So occupancy is a **live serving-regime lever**
  the B=1-only v8.6 run missed → **candidate v8.8 (confirm + ship), or fold the 4-blocks/SM residency into v10's
  kernel.** ncu bonus (this run had counters): db moves byte-identical traffic to Cut-1 (DRAM ~7.6%, L2-hit ~1%)
  → confirms it does nothing. Not a v10 blocker; v10 NVFP4 (capacity+accuracy) proceeds.

**What changes on another arch:** the per-CTA cap is a *schedule* limit (1 active warp at N_q=1), so it's
arch-independent in shape; bigger-L2 archs (A100 40 MB … B300 132.6 MB MEASURED in v10) just push the L2-spill N_k far higher,
so the confound-free measurement there needs proportionally longer context (why Task 1 was done on the T4's
small 4 MB L2). On B300 the same kernel would still be per-CTA-bound at short context; the bandwidth-bound
regime only appears at very long context where even ~28% of 8 TB/s is large in absolute terms.

See [`v9-kickoff.md`](v9-kickoff.md) Task 1, `results.md` Step 9 Task 1, `interview-prep.md` C12 (the
ncu-validation story), `notebooks/v9_task1_regime_output.ipynb`.

## Step 9 — deep-research close-out + Step 10 (v10 NVFP4) reframe

**A 7-agent verify-and-research pass (2026-06-28) confirmed v9 ran and is code-honest, then sharpened the
wording and — importantly — corrected three v10 design assumptions the data/research now refute.** Full
record in `results.md` Step 9 close-out; decision-relevant deltas:

**Verified (no decision change):** code audit confirmed fused dequant / byte-identical A/B / in-session
`vs naive` / the `vs sdpa` fix; the not-bandwidth-bound verdict is ncu-validated and stands.

**Decision-relevant refinements (locked into the record):**
- **Limiter name:** "per-CTA-bound" → **"per-CTA / low-MLP latency-bound, occupancy-lifted to a hard ~28%
  cap."** ncu proves *not bandwidth*; the batch-sweep decline at B≥64 (29.3→25.4%) is a latency/MLP tell,
  not occupancy starvation. **Consequence:** the *next* schedule lever, if one is wanted, is latency-hiding
  (deeper pipelining), not "fill the SMs" — which reopens v8.5/v8.6 **past L2** as the cheap experiment that
  settles it (kernels exist, one notebook). "Decode-schedule CLOSED" is **CLOSED for the L2-resident
  regime** only.
- **FP8 latency claim:** the ~1.3× is a **regime-specific, bytes-sensitive load-latency** win (NOT
  L2-bandwidth — ncu L2 throughput <3%), and it **flips negative under L2-flush** (software-dequant ALU tax
  on sm_75). This *strengthens* the operative decision "FP8/NVFP4 = capacity + accuracy; latency is
  conditional," and it means **byte-cuts are NOT a decode-latency lever on a per-CTA-bound kernel.**

**Step 10 (v10 = NVFP4 + asymmetric-precision KV, B300/sm_103) — decision recorded BEFORE coding.** Full
plan: [`docs/v10-kickoff.md`](v10-kickoff.md). The reframe vs the old "bytes → ~3.5× faster" framing:

- **Bottleneck the step attacks:** capacity (KV cache size at fixed HBM) + accuracy (FP4 KV recipe), **not**
  decode latency — because v9 Task 1 proved decode is per-CTA-bound, so cutting bytes cannot buy latency at
  reachable occupancy. Latency relevance is **conditional** on a B300 long-context bandwidth-bound regime
  that v10 must *measure* (carry the Task-1 method to B300's ~126 MB L2 → sweep N_k 256K–1M).
- **Options + choices:**
  1. **NVFP4 storage + fused per-tile dequant to FP16, CUDA-core score-stationary loop** (byte-identical to
     v9 except storage format) → **chosen for v10.** Clean byte-only A/B vs v9/v8.7.
  2. **Native FP4 tensor-core compute** → **deferred to v11.** ⚠️ Correction: the Blackwell `tcgen05.mma`
     gate is **M ≥ 64** (M=128 = 100% datapath), not M≥16 (that's the legacy `mma.sync` path). N_q=1 decode
     packs only M=G<64, so the 5th-gen FP4 cores stay dark — engaging them needs query-token packing
     (speculative/multi-token decode) = v11. This *generalizes* v8's measured "tensor cores are the wrong
     decode tool" onto native-FP4 silicon.
  3. **Asymmetric precision (the recipe):** **storage** = V→NVFP4 per-token (convex-combo-safe), K→NVFP4
     per-channel with the **score reconstructed at ≥FP16** (K drives score fragility — KVTuner). The
     project's old "P·V is the cheap one to FP4" intuition is **refuted for *compute*** (quantizing
     post-softmax P piles cvt onto the softmax bottleneck and *slows* the kernel — Attn-QAT keeps P·V in
     BF16) — but that only bites when FP4 *matmul* engages (v11), so v10 (FP4-storage, no FP4 matmul) is
     unaffected. Q stays FP16; softmax/O/merge FP32 (+ hardware exp2 on B300).
  4. **The sm_103 novelty hook = the softmax/exp term** (B300's 2× exp, 5→10.7 TeraExp/s) — center the
     roofline there and ablate it; it's the cleanest sm_103-vs-sm_100 delta and may flip the "keep P·V
     BF16" decision on B300 (a v11 lead).
- **Prediction before measuring:** roofline says NVFP4 doubles AI vs FP8 (G8: 16→28.4), HBM floor ~3.55×
  below FP16, limiter stays HBM (AI 28–57 ≪ FP4 ridge 1875), **but the model is blind to the per-CTA wall**
  → expect capacity (certain) + accuracy (headline) + only a fragile/conditional latency effect. Counter:
  a bandwidth knee past B300's ~126 MB L2 would overturn it (first-class either way).
- **What changes on another arch:** capacity is arch-independent; the per-CTA cap is a *schedule* limit so
  it's arch-independent in shape (bigger L2 just pushes the spill N_k out); native FP4 *compute* needs
  Blackwell + M≥64; the 2×-exp lever is sm_103-specific.
- **Novelty (adversarially checked):** "first to run / first FP4-KV decode / first Blackwell roofline" are
  **dead** (FlashInfer `trtllm-gen` ships NVFP4 KV decode now; vLLM published GB300 NVFP4 decode Feb 2026).
  Survives: **the open, roofline-documented, prediction-vs-measured sm_103 decode + asymmetric FP4 recipe +
  per-CTA-honest methodology — complementing, not beating, FlashInfer/FlashMLA, timestamped June 2026.**

**B300 facts that constrain v10 (web-research, confidence-tagged in
`docs/b300-decode-research.md`):** HBM **8 TB/s flat** vs B200 (only capacity grew 192→**288 GB**); NVFP4
**15 PF dense** / FP8 **5 PF** / exp **10.7 TExp/s** (all FACT); B300 **trades away INT8 + most FP64** for
the NVFP4 uplift; **L2 size UNCONFIRMED** (~126 MB B200, likely same die → measure on first rent); toolchain
**CUDA 12.9 / PTX 8.8**.

See [`v10-kickoff.md`](v10-kickoff.md), `results.md` Step 9 close-out, `interview-prep.md` C14,
`b300-decode-research.md` (v10 banner).

## Step 10 — NVFP4 KV cache (v10_nvfp4) — kickoff (prediction recorded before coding, 2026-06-29)

**The single variable: KV storage format.** v10 forks v9 (`fp8_attention.cu`) changing ONLY the paged
K/V pool format — FP8 E4M3 (1 B/elem) → **NVFP4: a packed 4-bit E2M1 nibble + one E4M3 micro-scale per
16 elems + a per-tensor FP32 scale = 0.5625 B/elem.** The score-stationary inner loop, M-packing grid,
split-KV partial, LSE merge, `[B,H_q,N_q,S,*]` workspace, `choose_splits`, and host launch are carried
byte-identical. The fused per-tile dequant (`dequant_nvfp4`: nibble → `kE2M1[8]` magnitude → sign →
× E4M3 micro × per-tensor scale → FP16) writes the same FP16 sK/sV — so this is a clean byte-only
ablation vs v9 *and* v8.7. Q·Kᵀ is reconstructed at FP16 from the dequant (sK is FP16), never a raw FP4
dot (which would amplify into the softmax). **Decode uses NO FP4 tensor cores** — M=G<64 at N_q=1 is
below Blackwell's tcgen05 M≥64 gate, so native FP4 compute is **v11** (multi-token); v10 is CUDA-core /
dequant-to-FP16, byte-identical to v9 except storage.

**Why NVFP4 now (the ranked thesis, governed by v9 Task 1's per-CTA verdict).**
- **T1 (primary, limiter-independent): capacity + accuracy.** ~3.55× more resident context/batch vs
  FP16 (arch-independent arithmetic); the asymmetric-precision recipe is the research content. This is
  the durable headline regardless of whether bytes are ever the decode wall.
- **T2 (the sm_103 hook): the 2×-exp/SFU softmax delta** (B300 5→10.7 TExp/s) — a measurable
  prediction-vs-measured number, the cleanest sm_103-vs-sm_100 wedge.
- **T3 (conditional, the ONE bandwidth claim — only if earned): does a bandwidth-bound decode regime
  exist on B300's ~126 MB L2 at all?** T4 stayed per-CTA-bound even past its 4 MB L2; B300's L2 pushes
  the spill to N_k in the 100Ks–millions. Find the knee → NVFP4 wins there. No knee → the stronger,
  more surprising result (per-CTA-bound to 1M tokens on sm_103). Either sign publishable.
- **T4 (DEFERRED to v11): native FP4 tensor-core compute** — gated at M≥64, needs query-axis packing
  (speculative/multi-token), which generalizes v8's "tensor cores are the wrong tool for decode."

**The recorded prediction (before coding).** Decode AI = 2G/b: NVFP4 G=8 = 28.4 (vs FP8 16.0, FP16
8.0), **HBM-bound, ~3.55× lower floor than FP16, far below every ridge.** Per-CTA-corrected: capacity
certain, accuracy = the headline, latency win only fragile-L2-resident (shrinks with G, likely flips
negative under flush) OR in a B300 long-context regime IFF one exists (T3). Counter: %HBM climbs past
the ~126 MB L2 → bandwidth-bound on sm_103 → byte cut converts.

**The asymmetric-precision recipe (the deliverable; kickoff implements the standard rung, ablation
follows Gate 1).** Standard NVFP4 = per-tensor FP32 + per-16 E4M3 micro-scale (the "native 1×16" rung),
kept clean for the single-variable A/B. The accuracy ladder layered in after Gate 1:

| Tensor | v10 decision (FP4 *storage* + dequant; no FP4 matmul) | Why |
|---|---|---|
| **V (cache)** | NVFP4, **per-token** scale *(granularity ablation)* | post-softmax convex combo → FP4 error averages out; biggest byte-saver |
| **K (cache)** | NVFP4 storage, **per-channel** scale, **score ≥ FP16** | score fragility is K-driven (KVTuner); reconstruct Q·Kᵀ at ≥FP16 |
| **Q** | FP16 | tiny O(d); never degrade the query that gates softmax |
| **scores / softmax / O / merge** | FP32 (hardware exp2 on B300) | dodges the FA-3 FP8-accum cliff |

⚠️ The *compute* asymmetry (Attn-QAT keeps P·V in BF16, only Q·Kᵀ in FP4, because quantizing P piles
`cvt` onto the softmax bottleneck) is **moot for v10** (no FP4 matmul) and a **v11** concern — recorded
now so v11 inherits the framing. The FP4-everything ablation (show the softmax collapse) justifies the
asymmetry as a paper figure.

**Hardware staging (Kien's call: no-rental correctness first).** Gate 1 on **T4-emulated** (store 4-bit,
unpack in-kernel — free Colab; correctness + capacity + accuracy ONLY, latency NOT valid — emulated
software unpack is more ALU than v9's E4M3). The latency/regime/sm_103 deliverables run on a **root
B300** (`_MIN_CAPABILITY="v10_nvfp4":(7,0)` is the emulated fallback; the record target is sm_103).

**Novelty (adversarially narrowed, timestamped June 2026).** FlashInfer `trtllm-gen` already ships
NVFP4 KV decode; vLLM published reproducible GB300 NVFP4 decode; FA4 roofs B200 prefill. What survives:
**no open, roofline-documented, prediction-vs-measured DECODE study on sm_103 with an asymmetric FP4 KV
recipe + a confound-free per-CTA-vs-bandwidth methodology.** Frame: **complementing, not beating**
FlashInfer/FlashMLA. The per-CTA-bound finding (v9 Task 1) is the contribution.

**Status: kickoff landed + Gate 1 MEASURED GREEN (Colab T4, 2026-06-29).** Correctness ✅ 146 passed;
**capacity MEASURED 3.56× vs FP16 / 1.78× vs FP8** (closes v9's asserted-not-measured gap); **accuracy =
the finding — standard NVFP4 costs ~4× more RMSE than FP8** (~2.4e-3 vs ~6e-4; expected from E2M1's
1-bit mantissa, NOT a bug → the asymmetric per-token-V/per-channel-K recipe is the lever to recover it);
**latency prediction landed — no win, NVFP4 ≤ FP8, %HBM falls to 3% (per-CTA-bound reconfirmed a 7th
time, not bandwidth-bound).** Prediction-vs-measured 4/4. **Still pending: Gate 2 quiz; the T4
asymmetric-recipe ablation; the root-B300 measured core** (T3 knee-hunt + sm_103 exp + FlashInfer/
FlashMLA). See [`v10-kickoff.md`](v10-kickoff.md), `results.md` Step 10, `interview-prep.md` C15,
`notebooks/v10_nvfp4_gate_output.ipynb`.

**Asymmetric-recipe ablation — Stage A (synthetic) DONE 2026-06-29: hypothesis REFUTED, but
informatively.** On i.i.d. Gaussian KV the asymmetric K=channel/V=token recipe is *worse* than standard
block16 (2.98e-3 vs 2.43e-3), and no NVFP4 granularity reaches the FP8 floor (~3.5× gap). **This is a
substrate problem, not a recipe defeat:** per-channel-K / per-token-V exploit outlier *structure*
(KIVI/KVQuant) that Gaussian noise lacks, so block16 (the finest granularity) wins by default. **Decision:
the recipe must be tested on REAL model KV** (`notebooks/v10_realkv_ablation.ipynb` — GPT-2 capture +
outlier diagnostic + the same matrix); **Stage B (kernel per-channel/per-token scales) is deferred until
real KV earns it.** Clean positive: the FP4-everything demo confirmed score-quantization collapses softmax
(4.6–6.6× RMSE blow-up) → **score ≥ FP16 justified** (already in the kernel). Tooling:
`fa_kernels/nvfp4_recipes.py` (model-agnostic fake-quant). See `results.md` Step 10 asymmetric subsection.

**Stage A′ (REAL GPT-2 KV) DONE 2026-06-29: conditionally vindicated, block16 stays the default →
Stage B (kernel) NOT justified.** Captured real K,V from GPT-2 + an outlier diagnostic (K channel
max/mean 11.6 at layer 2 vs Gaussian 5.9, gone by layer 11; V token outliers mild 1.5–2.2). The matrix:
**per-channel-K beats block16 exactly at the layer with strong channel outliers (layer 2, 16% better)
and loses where they vanish (layer 11) — the KIVI/KVQuant mechanism CONFIRMED, but layer-conditional.**
block16 (standard NVFP4) is best/tied at 2 of 3 layers; V=token never clearly wins (gpt2-small sinks too
weak); no NVFP4 beats per-tensor FP8 but the gap narrows to 1.49× at mid layers (vs 3.5× synthetic).
**Decision: ship block16; a fixed per-channel/per-token kernel is not worth the complexity for a decode
micro-study.** The accuracy chapter is honest + complete; an optional stronger-model rung (larger /
known-sink) would firm the V lever. See `results.md` Step 10 Stage-A′ subsection,
`notebooks/v10_realkv_ablation_output.ipynb`.

**B300-measured-core — B200 dev-rung MEASURED 2026-06-29 (the T3 verdict, confound-free on the L2 axis;
sm_103 record + ncu still owed).** Ran the regime knee-hunt end-to-end on a vast.ai **B200** (after fixing
the cusparse-header build, `bindings/load.py`). **Decision-relevant findings:** (1) **No bandwidth knee** —
%HBM dead flat ~0.5% from N_k 8K→2M, *including* an 8× L2 overflow (1 GB WS ≫ measured 132.6 MB L2) →
**decode is per-CTA-bound on Blackwell at all reachable context**, now confound-free on the L2 axis
*without ncu* (the 1 GB WS kills the L2-residency confound by construction). (2) **Architecture-independent
latency ceiling:** ~40 GB/s achieved at B=1 on BOTH T4 and B200 (11% of 320 GB/s vs 0.5% of 8 TB/s) — the
strongest latency-bound evidence in the project. (3) **NVFP4 is latency-NEGATIVE past L2** (12–30% slower
than FP16; FP8 ≈ FP16) → **NVFP4 = capacity + accuracy, NOT latency, settled for every regime measured**
(to 2M tokens / B=128 on T4, B200 *and* sm_103) — the ncu L2-hit-rate cross-check on Blackwell and a
clock-locked rerun remain owed as belt-and-suspenders, so "settled" here means "across the measured
regimes," matching this record's own "corroboration, not discovery" framing, not "proven for all regimes."
(4) Occupancy
(batch) is the only throughput lever (4× from batching) but caps at 2.3% HBM. (5) Arch constants measured
(L2 **132.6 MB** refutes the 192 MB aggregator claim; 148 SMs; 1965 MHz suggests B300's 2600 estimate is
high). **Net: the B300/sm_103 run is now corroboration, not discovery** — reproduce the flat %HBM + add the
ncu L2-hit-rate (needs a **privileged/bare-metal** box; vast.ai containers block counters at the host
kernel-module level) + the 2×-exp delta. The science answer won't change. See `results.md` Step 10 B200
result, `notebooks/v10_b300_regime_output.ipynb`, `docs/v10-b300-runbook.md` §0.

**B200 nsys schedule corroboration (MEASURED 2026-06-29).** Added the *schedule* read to complement the
*bandwidth* read: `nsys` (CUPTI trace) runs on the unprivileged AIO container (no `ERR_NVGPUCTRPERM`,
unlike ncu), so it's the free way to confirm which kernels run. **Decision-relevant:** the
`nvfp4_partial_kernel` owns **99.3% of GPU time**, the `nvfp4_merge_kernel` is **0.04%** (~2550× smaller),
and the torch quant/paging kernels are one-time (≤0.1% each) — i.e. decode is **one under-occupied
dominant kernel with no hidden second phase**, exactly the per-CTA-bound schedule the %HBM verdict
predicts. So both the bandwidth axis (flat %HBM past a 1 GB WS) and the schedule axis (single dominant
kernel) now agree; **the only owed cross-check is ncu's L2-hit-rate** (privileged box). Tooling decision:
the AIO image's bare `apt-get install nsight-systems` ships an *incomplete* nsys (missing `QdstrmImporter`
→ `--stats` fails); install `cuda-nsight-systems-12-9` (pulls a complete 2025.x) from the
already-configured NVIDIA CUDA repo, and the regime notebook's §10b now auto-finds a complete nsys and
runs `nsys stats` as a separate step. Data of record: `notebooks/v10_nsys_kernsum.txt`. See `results.md`
Step 10 "B200 nsys schedule corroboration", `interview-prep.md` C15.

**B300 / sm_103 RECORD — Pass 1 MEASURED 2026-06-29 (the paper's headline arch; counter-free + nsys, ncu
owed).** Reproduced the whole dev-rung on a real **B300 (sm_103)**, unprivileged vast.ai container.
**Decision-relevant:** every B200 finding **transfers to sm_103** — (1) **no bandwidth knee**, %HBM flat
~0.5%(FP16)/0.1%(NVFP4) from N_k 8K→2M incl. an 8× L2 overflow (1.07 GB WS ≫ measured **132.6 MB L2**) →
per-CTA-bound on Blackwell Ultra at all reachable context, confound-free without ncu; (2) the **~40 GB/s
B=1 latency ceiling now holds on THREE arches** (B300/B200/T4 — same absolute BW across a 25× bandwidth
span = the cleanest latency-bound proof); (3) **NVFP4 latency-negative** (2M: FP16 24,798 vs NVFP4 27,579
µs/tok, ~11% slower) → capacity+accuracy not latency; (4) occupancy caps at 2.3% HBM; (5) **nsys schedule
99.4%/0.04%** matches B200 (needed **nsys 2025.3.2** — the image's 2025.1.3 CUPTI records an empty trace on
sm_103). **Arch constants MEASURED** → `roofline/archs.py` B300: 148 SMs (not 160 spec), L2 132.6 MB (not
192 aggregator), clock 2032 MHz (not 2600 est). **Two new measured items:** (a) **2×-exp delta is a MISS** —
EX2 microkernel hits 5.33 TExp/s = **0.50× the 10.7 claim**, and the mufu share of decode is <3% so even a
real 2× wouldn't matter (the sm_103 exp lever is negligible for M=1 decode — honest, recorded); (b)
**FlashInfer trtllm-gen NVFP4 is ~3× faster** (1484.51 vs our 4491.43 µs/step, FP8-Q vs our FP16-Q) →
frame "complementing, not beating": ~3× off a tuned closed kernel, same per-CTA wall, tuning buys the
constant not a regime change. **Net: the sm_103 record is in hand; only ncu (Pass 2, bare-metal) + the
Gate-2 quiz remain before v10 DONE.** Data of record: `notebooks/v10_b300_*_output*.ipynb`,
`v10_b300_nsys_kernsum.txt`, `docs/diagrams/v10-b300-regime.svg`. See `results.md` Step 10 sm_103 record,
`interview-prep.md` C15.

## Step 10 deep-research close-out + v11 decision (2026-06-30, 18-agent verify+adversarial+research pass)

**An 18-agent pass (7 data-forensics re-extracting every notebook number, 5 adversarial refute-first verifiers
on the load-bearing interpretations, 5 B300/v11 web-research, 1 synthesis) verified v10 and decided v11.**
Full record in `results.md` Step 10 close-out. Decision-relevant deltas:

**Verified (no science change).** Every load-bearing number reconciles to the notebooks (146-passed; capacity
268.44→134.22→75.50 MB = 3.56×/1.78×; accuracy NVFP4 ~2.1–2.7e-3 vs FP8 ~6–7e-4; %HBM monotone fall
10.4→6.3→3.0%; NVFP4 latency-negative 24798 vs 27579 µs/tok @ 2M; arch 148 SMs / 132.6 MB L2 / 2032 MHz;
FlashInfer 1484.51 vs 4491.43; EX2 5.33 vs 10.7). **Prediction-vs-measured is genuinely 4/4 + 1 honest miss.**
All 5 adversarial verdicts: **HOLDS-WITH-CAVEAT (high confidence)** — the per-CTA-bound verdict and "NVFP4 =
capacity+accuracy not latency" stand on the strongest evidence (the arch-independent ~40 GB/s ceiling + flat
%HBM past an 8× L2 overflow); the caveats are honesty hedges now folded into the docs.

**Honesty/provenance corrections applied (this pass).** (1) **"settled definitively" → "settled for every regime
measured"** (ncu on Blackwell + a clock-locked rerun still owed). (2) **nsys provenance flagged** — the 99.4%/99.3%
tables are hand-saved `.txt`; the committed notebook's nsys cell ran 2025.1.3 → *empty* trace, the 2025.3.2 run was
off-notebook (cross-checks the sweep within 0.46%/2.3%; regenerate in-notebook before the paper). (3) **"~40 GB/s"
scoped to FP16** (FP8 ~20, NVFP4 ~10 at B=1 — same latency restated; the cross-*arch* constancy is the real claim;
do not attach it to the B=64 FlashInfer comparator, where the kernel runs ~478 GB/s). (4) **NVFP4 latency penalty
stated as the full 12–30% range**, not the best-case 2M "~11%". (5) **ncu on the B300 box was not even installed**
(never reached the `RmProfilingAdminOnly` gate) — Pass-2 needs an install *and* privilege. (6) Stale **160 SMs /
192 MB** forward-looking notes corrected to the measured **148 / 132.6 MB**. Cosmetic-only and deferred: a `nanx`
self-ratio in the gate's run-of-record, a "160 SMs" label string in a B300 notebook cell.

**Decision: v11 = MLA (Multi-head Latent Attention) latent-KV decode — the SHAPE change.** Full plan:
[`docs/v11-kickoff.md`](v11-kickoff.md). The reasoning, against the three required criteria:
- **Justified by the per-CTA verdict.** v10 proved across T4/B200/B300 (confound-free, to 2M tokens) that decode
  is per-CTA/low-MLP latency-bound and cutting bytes does *nothing* for latency. The only lever left is to change
  the SHAPE so >1 warp is active and AI rises off the floor — not more bytes (v9/v10 exhausted that), not more
  occupancy alone (caps at ~2.3% HBM), not tensor cores at M=1 (v8 measured them null; v10's FP4 cores stay dark).
- **Why MLA over the alternatives.** MLA shares **one** latent across all `h_q` query heads → it packs **M=128 BY
  CONSTRUCTION at N_q=1** (meets the tcgen05 NVFP4 `M≥128` gate with **no speculative draft**), and raises decode
  AI from `2G/b` (≤8–16) to **`2·h_q` ≈ 256** (near the B300 FP16 ridge ~312) — the first decode shape in the arc
  plausibly compute-bound. The **speculative/multi-token** path (EAGLE-3 τ≈6.2, MTP ~1.8×) needs `q_len≥8–16` just
  to clear M=64 and adds acceptance-rate risk → kept as the **fallback** shape-lever. The **occupancy v8.8**
  finding (~1.4× at B≥32) is **not** a shape change → folded into v11's kernel for the serving regime, not its own step.
- **Most defensible contribution.** FlashMLA (Blackwell-native, tcgen05+TMEM) and FlashInfer are concrete SOTA
  comparators, yet **no published kernel-level roofline (compute-vs-memory-vs-per-CTA) for MLA decode on sm_103
  exists** — the exact empty cell, and a *stronger* novelty than continuing the GQA+FP4 line FlashInfer already
  beats us ~3× on. It keeps the project's signature question alive: **does the limiter FLIP to compute, or just
  MOVE to smem-capacity** (the absorbed `W^UK`/`W^UV` weight must stay on-chip)? Either sign is publishable.
- **ONLY-IF the data warrants:** commit to the full tcgen05/TMEM warp-specialized **native-FP4-compute** rewrite
  only after the B200/T4 dev-rung confirms M=128 packs as **one** GEMM (not fragmented by the decoupled-RoPE /
  absorbed-matrix layout) and that the limiter actually flips. Default v11 = CUDA-core/dequant-to-FP16 MLA, a clean
  single-variable A/B vs v10 (same storage + split-KV, only the shape changes). **What changes on another arch:**
  the M≥128 FP4-compute gate is Blackwell-`sm_103a`-specific (no `mma.sync` FP4 fallback); the AI lift and the
  capacity win are arch-independent. Field is moving to **sparse** (DeepSeek V3.2 DSA / V4 CSA) and **GLA** (Tri Dao,
  beat FlashMLA ~20%) → flagged **v12** (dense MLA is the roofline-clean pedagogical rung first). See
  [`v11-kickoff.md`](v11-kickoff.md), `results.md` Step 10 close-out, `interview-prep.md` C16, `decode-replan.md` §5.

## Step 11 — MLA latent-KV decode (v11_mla) — kickoff (prediction recorded before coding, 2026-06-30)

**Roofline FIRST, per the per-step loop.** Extended `roofline/model.py` with an MLA branch (`mla=True`,
`h_q`/`kv_lora_rank`/`rope_dim`) + a `--mla` knob in `roofline/predict.py`, and recorded the prediction
in `results.md` Step 11 **before** any kernel. Non-MLA paths are byte-identical (GQA-8 nvfp4 still 28.4,
v1 naive still 0.2).

- **The model.** MLA shares ONE latent across all `h_q` query heads (latent serves as both K and V →
  the `2×` K/V read collapses to `1×`; `bh/G → B`), so decode **AI = 2·h_q·(2L+R)/((L+R)·b) ≈ 3.78·h_q/b**
  with `L=512`, `R=64`. The mma bound uses the **tensor-core peak** because M=h_q≥16 engages the cores.
- **Recorded numbers (B300/sm_103, h_q=128, tool incl. Q/O bytes):** fp16 **AI 234.8** (0.75× the 312.5
  FP16-TC ridge → HBM, knife-edge); fp8 **469.7** (1.50× → MMA, **flips compute-bound**); nvfp4 **835.0**
  (2.67× → MMA, deep). ~30× the GQA-8 fp16 AI of 8. T4 cross-check: fp16 234.8 > 203.1 ridge → compute
  even on the dev rung.
- **The choice this records:** the mma peak is the **TC peak** (not the M=1 GQA "cores dark" peak) — the
  defensible default because M=h_q≥16 genuinely engages tensor cores, which is the entire premise of the
  shape change. The byte-lever ridge crossing (fp8/nvfp4 push AI past the FP16-TC ridge) is recorded as a
  **pure-roofline flip prediction**; the **per-CTA-corrected** real prediction is a **rename to
  smem/TMEM-capacity** (`q_absorbed` staging ≈ 147 KB ≈ most of 228 KB/SM → ~1 block/SM). **What changes
  on another arch:** the AI lift + capacity win are arch-independent; whether the flip *realizes* depends
  on smem/SM (228 KB Blackwell) and the M≥128 tcgen05 gate (`sm_103a`-specific).
- **§9 roofline answers (resolve by dev-rung measurement):** Q1 M=128 is ONE GEMM in M (RoPE splits the
  K-dim, not the M-tile) → T2 holds structurally; Q2 predict **rename to capacity**, not clean flip
  (correction: `W^UK`/`W^UV` fold into the offline Q/O proj — the kernel stages `q_absorbed`, not the
  weights); Q3 the compute-vs-memory crossover IS the deliverable. See `results.md` Step 11,
  `v11-kickoff.md` §2/§9, `interview-prep.md` C16.

## Step 11 deep-research close-out + v12 decision (2026-06-30, verify+adversarial+research pass)

**v11 VERIFIED; all 16 adversarial claims HOLD or HOLD-WITH-CAVEAT (none refuted); every load-bearing
number reconciles to the raw regime/nsys files + both gate notebooks.** Full record in `results.md` Step 11
close-out. Decision-relevant deltas:

**Verified (no science change).** Per-CTA-bound to 2M past a 5.1× L2 overflow (%HBM 0.0%, eff_bw 0.9 GB/s
logical, WS 679.48 MB ≫ 132.6 MB L2); 4× cuBLAS gap (0.23–0.29× real, 0.49–0.83× smoke, back-out exact);
capacity 202×/12.6×/99.5% (56.9× shape × 3.56× bytes); accuracy ~3.5× FP8 (cross-arch 3.54/3.44); nsys
partial 99.9% @ 387 ms/call cross-checks the N_k=1M regime row to 0.03%; 0.75 TFLOP/s = 0.7%/0.03% of the
FP32-CUDA/FP16-TC peaks. **Prediction-vs-measured: the two-layer model called it** (pure = compute-flip,
per-CTA-corrected = stays per-CTA on CUDA cores; the REAL layer landed).

**Honesty/provenance corrections applied this pass (do not restate the killed framings as fact).** (1) The
4× **cause** → inference, not measurement (the torch baseline is a **FP32 batched-GEMV M=1**, no torch-side
profile). (2) "§9-Q1 validated" → the **shape** admits a TC GEMM (indirect); our absorbed-QK-as-one-tcgen05
-GEMM is **untested**. (3) %HBM≈0% scoped to "not bandwidth-bound" (high-AI); the per-CTA proof is the
TFLOP/s + 4× gap + ceiling. (4) "147 KB staging → ~1 block/SM" → the kernel stages **~46 KB**; ~1 block/SM
is **T4-specific** (B300 = ~4 blocks/SM, the GEMV shape is the limiter). (5) merge "0.04%" → **0.0%
(0.0009%)**. (6) `archs.py:153` "tcgen05 gate M≥64" → **M≥128 for block-scaled NVFP4** (M≥64 is the
FP16/FP8/dense-FP4 floor; the v10-era M=G<64 statements stay correct in their FP8/GQA context). (7) "FA4
stops at B200" → "FA4 **targets/benchmarks** B200; its kernel is **deployed but never characterized** on
sm_103." (8) Logged cleanup: `choose_splits` hardcodes `num_sm=40` on the 148-SM B300 (under-splits to 80
blocks; doesn't change the B=1 verdict). (9) accuracy ~3.5× is **our substrate-specific** number, single-seed.

**Decision: v12 = native tcgen05 tensor-core MLA decode (FP8-arm → NVFP4-arm).** Full plan:
`docs/v12-kickoff.md`. Against the three required criteria:

- **Justified by the measured 4× gap (the data warrants the roadmap change).** v11 proved the M=128 shape is
  a real TC GEMM (cuBLAS beats our CUDA-core GEMV ~4× on it) and that the CUDA-core default is the wrong
  tool — the **only lever left to close v11's own self-gap** is the tensor-core path. This is NOT v8's "TC
  null at M=1" (that was M=1; v11 raised M to 128). So the previously-deferred native-FP4 arm graduates from
  "ONLY-IF" to **the data-motivated primary v12**.
- **Why native-FP4-TC over the alternatives (weighed impartially).** It realizes the GEMM v11 localized; M=128
  meets the tcgen05 gate (FP8 M≥64, **NVFP4 M≥128/K=256/TN-only**) by construction. **Speculative q_len>1** is
  the **fallback** (only if the dev-rung shows M=128 fragments — but MLA already raised M, so this is
  redundant unless the TC arm fails). **Occupancy-v8.8** (~1.4× @ B≥32 past L2, measured) is **not a shape
  change** → fold into v12's residency tuning, not its own step. **GLA/sparse (DSA/CSA)** → v13 (dense MLA is
  the roofline-clean rung first). **Harden-v11-measurement (ncu/locked-clock/torch-profile)** → do
  *alongside* v12, it's a measurement task not a competing kernel.
- **Most defensible build + the honest ceiling.** Fork **CUTLASS example 77** (`77_blackwell_fmha`, the
  weight-absorbed latent-512/rope-64 MLA *decode* kernel, **2-SM `cta_group::2`** for the 512-wide
  accumulator) — **NOT** the FA4 prefill kernel (no decode path). **Arm 1 = FP8-dense MMA** (M≥64; dequant
  NVFP4-KV→FP8 at SMEM, the FlashMLA-proven path), **Arm 2 = native NVFP4** (`kind::mxf4nvf4`, gated on Arm
  1), scores ≥ FP16. **Pre-registered prediction (the paper-grade result): even with tensor cores,
  single-token decode is SMEM-BW / MMA-pipeline-depth-bound → the realized speedup likely will NOT approach
  the FP4 FLOP peak, and v12 will NOT beat FlashMLA's ~410 TFLOP/s decode (~3 orders above v11) on
  wall-clock.** Frame as **complementing, not beating** FlashInfer/FlashMLA — the contribution is the open
  methodology + the measured SMEM-BW shortfall, not SOTA latency.
- **Build risk to flag:** CUTLASS ex77's realized M-blocking num-groups is reportedly capped at 32 (not
  128) — verify the head-count→M=128 mapping on the target CUTLASS version **before** banking §9-Q1 (echoes
  v11-kickoff's "verify M=128 is one GEMM before banking T2"). **What changes on another arch:** the M≥128
  NVFP4-*compute* gate is `sm_103a`-specific (no `mma.sync` FP4 on datacenter Blackwell); the SMEM-BW wall is
  arch-general. `[RENT root B300/sm_103a; CUDA 12.9+, CUTLASS 4.x]`.

**Paper positioning (honest, characterization-grade).** Defensible novelty = *the first open, kernel-level,
prediction-vs-measured roofline characterization of attention decode on sm_103 (MHA→GQA→MLA,
FP16→FP8→NVFP4)* — **not** "first to run" (serving stacks did, Feb 2026), **not** "beat
FlashInfer/FlashMLA" (faster by construction). The methodology itself is **not** novel (Microbenchmark
-Driven Modeling arXiv 2605.04178 et al. own prediction-vs-measured); novelty lives in the
{open + sm_103 + decode + the specific measured findings} intersection. Closest prior = Tri Dao's GLA
(arXiv 2505.21487, open roofline-documented MLA decode on H100) → our delta is exactly **sm_103 + KV-quant**.
Venue path: **PMBS@SC26 / ES-FoMo / IISWC 2027** (characterization venues; top-tier systems would
desk-reject as a SKU-bump). See `docs/v12-kickoff.md`, `results.md` Step 11 close-out, `interview-prep.md` C17.

## Step 12 — native tcgen05 tensor-core MLA decode (v12_mla_tc) — kickoff (prediction recorded before coding, 2026-06-30)

**The one variable: the compute ENGINE.** v11 proved the MLA decode SHAPE is right (M=128) but measured it
per-CTA-bound at 0.75 TFLOP/s — ~4× slower than torch dense-MQA, which routes M=128 into a cuBLAS
tensor-core GEMM while v11 runs warp-per-head CUDA-core GEMV. v12 moves the M=128 QK/PV matmuls onto
Blackwell **tcgen05 tensor cores** — Arm 1 FP8-dense MMA (M≥64) → Arm 2 native NVFP4 (M≥128/K=256/TN, gated
on Arm 1). Everything else (split-KV, LSE merge, score-stationary structure, NVFP4 latent pool/layout,
`choose_splits`) is byte-identical to v11 so the A/B isolates the engine. Build = fork **CUTLASS example 77**
(`77_blackwell_fmha`, 2-SM `cta_group::2`), NOT the FA4 prefill kernel. Scores/softmax stay ≥ FP16.

**Bottleneck (predicted).** Same NVFP4 storage → AI unchanged (835 @ h_q=128). The engine sets the *ridge*:
the roofline model gained an `mma_engine` selector (storage `precision` and engine decoupled). Under the
**engine-correct** ridge on B300 — FP8 625 / NVFP4 1875 — **Arm 1 (FP8) is pure-roofline compute-bound by
1.34×; Arm 2 (NVFP4) falls BACK to HBM-bound** (AI 835 < 1875; the 15 PF peak overshoots), with softmax-exp
(measured 0.5×) the #2 term. This **corrects kickoff §1** (which judged both arms against the fixed fp16
ridge 312.5 → "both MMA"). **Two-layer prediction:** pure roofline as above; **per-CTA-corrected (the real
one): SMEM-BW / MMA-pipeline-depth-bound either arm** — 1–4 MMAs in flight vs tcgen05's 256–1024, so
realized TFLOP/s << peak and v12 will NOT beat FlashMLA's ~410 TFLOP/s. **Counter (the prize):** achieved
TFLOP/s >~10× v11's 0.75 with high %SMEM-BW ⇒ the limiter LEFT per-CTA (first in the v1→v12 arc).

**Options weighed.** (1) **native-FP4-TC engine [CHOSEN]** — the only lever left to close v11's measured 4×
self-gap; M=128 meets the tcgen05 gate by construction. (2) **speculative q_len>1** — the *fallback*, only
if M=128 fragments on tcgen05 (redundant since MLA already raised M). (3) **occupancy-v8.8** (~1.4× @ B≥32
past L2) — not a shape change → fold into v12 residency tuning. (4) **GLA/sparse (DSA/CSA)** → v13. (5)
**harden-v11-measurement** (ncu/clock-lock/torch-profile) → done *alongside* v12, not a competing kernel.

**Why native-FP4-TC.** It realizes the GEMM v11 localized; the engine is the single unmeasured variable in
the 4× gap. Risk flagged: CUTLASS ex77 `num_groups` reportedly caps at 32 (not 128) — verify the M=128
mapping before banking Arm 2 (§9 Q1); Arm 2 may lose to Arm 1 even on TC (TRT-LLM #4412). Honest ceiling:
v12 **complements, not beats** — the contribution is the open methodology + the measured SMEM-BW shortfall,
not SOTA latency.

**What changes on another arch.** The M≥128 NVFP4-*compute* gate is `sm_103a`-specific (no `mma.sync` FP4 on
datacenter Blackwell); the SMEM-BW/pipeline-depth wall is arch-general (a property of single-token decode),
which is what makes the result transferable. Dev rung B200/sm_100 (FP8 Arm 1); record on B300/sm_103a.

**Status.** Roofline ✅ (model `mma_engine` selector + `--mma-engine` CLI; `results.md` Step 12). Author
scaffold ✅ — `kernels/v12_mla_tc/` (documented CUTLASS-ex77 fork plan + binding), wiring (`load.py`,
`dispatch.py` `(10,0)`, `mla_tc_attention()` API, `__init__.py`, harness `--mla-engine` + `vs-v11` A/B,
regime `--engine`), tests (vs the v11 oracle + absorption identity, tol 5e-2, Blackwell-gated), gate
notebook `notebooks/v12_mla_tc_gate.ipynb`. GPU work (rented root B300/sm_103a): implement the tcgen05
body, Gate 1 correctness, the engine A/B + regime sweep, the four honesty debts (ncu/clock-lock/torch-fp16
profile/2×-exp@M=128), then the quiz. See `docs/v12-kickoff.md`, `results.md` Step 12, `interview-prep.md` C18.

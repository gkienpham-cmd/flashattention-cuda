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

**What changes on B300 (research §3):** 160 SMs ⇒ more splits (retune `choose_splits` `num_sm` 40→160);
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

**What changes on B300 (research §3):** 160 SMs ⇒ more splits (retune `choose_splits` `num_sm` 40→160);
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

*Status: code complete + locally validated (2026-06-27); **GPU gate + quiz PENDING** (run
`notebooks/v7_paged_gate.ipynb`). The decision + prediction are recorded; the measured block fills in
after the run.*

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

**Measured (PENDING — paste after the gate run):**
- Correctness: *N/N* vs SDPA (paged shuffled-pool gather + v6 regression), tol 2e-2.
- `--batch` crossover at N_k=8192: *did %HBM climb 12% → saturation past BH≈80–160?* — **the answer
  that confirms or refutes "occupancy before bytes."** This is the load-bearing measurement of the
  whole reorder.
- Paging overhead: v7 µs/tok vs v6 at B=1 (expect within noise — byte-neutral).
- Causal decode now ≈ non-causal per row (the query-offset fix).

**Quiz:** PENDING (Gate 2). **Next (v8 — GQA M-packing):** the occupancy lever — pack `G` query heads
into `M` (GEMV→`M=G` GEMM, tensor cores re-engage, KV read once, `AI = 2/b → 2G/b`). v7's `--batch`
data decides whether v8 leads at all batch sizes or only small-batch; `decode-replan §5`.

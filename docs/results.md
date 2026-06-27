# Results

The growing speedup + roofline curve. One block per step. Each row pairs the **predicted**
limiter (from `roofline/`) with the **measured** reality (bench + Nsight Compute), because the
honesty of the prediction is itself a deliverable.

Every row records GPU / arch / clock so it's reproducible; the free-tier T4 throttles, so clocks
are captured at run time.

> **Tail-latency column.** The second number in each `p50/max ms` pair is the **max** (worst of 50
> timed iters), not a percentile. It was labeled `p99` through Step 4, but at iters=50 the p99 index
> rounds to the last (max) sample — same number, now named honestly (`bench/harness.py`).

---

## Step 1 — Naive attention (FP32, three-pass)

**Roofline prediction (T4 sm_75, B=1 H=8, precision fp32, materialize_S, naive tile 1x1):**

| shape (BxHxNxd) | intensity (FLOP/B) | ridge | predicted limiter | predicted lower bound |
|---|---|---|---|---|
| 1x8x512x64   | 0.25 | 25.3 | **HBM** | 6.82 ms |
| 1x8x2048x64  | 0.25 | 25.3 | **HBM** | 109.1 ms |
| 1x8x8192x64  | 0.25 | 25.3 | **HBM** | 1744.9 ms |
| 1x8x2048x128 | 0.25 | 25.3 | **HBM** | 216.5 ms |

> These replace an earlier table (intensities ~14–30) that assumed operands were read once from
> HBM — wrong for the naive kernel, which re-reads each Q/K row per output element. Counting that
> O(N²·d) traffic (`roofline/model.py`, `--tile 1x1`), true-naive AI is ~0.25 FLOP/byte at every
> shape: deeply HBM-bound. The old "d=128 → MMA-bound" was the same artifact. See
> [decisions.md](decisions.md#step-1--naive-attention-the-bandwidth-wall) and the Step 2 entry.

**Measured (Colab T4, sm_75, torch 2.11/cu128, FP32, causal=False — 2026-06-18):**
Correctness: **9/9 pass vs SDPA** (atol/rtol 1e-4).

| shape | ours p50/max ms | SDPA p50/max ms | speedup vs SDPA | tok/s (ours) | measured ÷ cache-free LB |
|---|---|---|---|---|---|
| 1x8x512x64    | 6.77 / 14.17    | 0.20 / 5.10  | 0.03× | 6.06e5 | 0.99× (at floor) |
| 1x8x512x128   | 10.08 / 10.18   | 0.33 / 0.37  | 0.03× | 4.06e5 | 0.74× (L2 helps) |
| 1x8x2048x64   | 85.56 / 89.47   | 3.03 / 3.22  | 0.04× | 1.92e5 | **0.78× (L2 helps)** |
| 1x8x2048x128  | 166.59 / 171.40 | 5.56 / 6.26  | 0.03× | 9.84e4 | 0.77× (L2 helps) |
| 1x8x8192x64   | 1763.7 / 1816.8 | 48.00 / 48.96 | 0.03× | 3.72e4 | **1.01× (L2 overflow → at floor)** |
| 1x8x8192x128  | 3112.7 / 3134.2 | 96.44 / 97.05 | 0.03× | 2.10e4 | 0.90× |

**Reading it:** ~20–30× slower than SDPA everywhere — the intended "before." The honesty payoff
is the last column: measured beats the *cache-free* roofline floor at mid-N (0.77–0.78×) because
the T4's **4 MB L2 catches the redundant operand reads** the model charges to HBM; at N=8192 the
working set overflows L2 and measured **converges to the floor** (1.01×). So AI≈0.25 is the worst
case, hit exactly when caching can't help. Step 2 (explicit shared-mem tiling) makes that reuse
guaranteed and N-independent → **biggest predicted win at N=8192.** Full reasoning in
[decisions.md](decisions.md#step-1--naive-attention-the-bandwidth-wall).

**ncu:** ran on this Colab runtime but `profiling/capture.sh` profiled the `torch.randn` RNG
kernel, not qk/softmax/pv — needed a `--kernel-name` filter. Harness nit: bench logged
`clock~0MHz` (`torch.cuda.clock_rate()` returns 0 on this build) though the GPU reported
300/1590 MHz. **Both fixed in Step 2** — capture.sh now filters `regex:(qk|softmax|pv)` and the
header reads `clock~CUR/MAXMHz`.

---

## Step 2 — Shared-memory tiling (FP32, three-pass, S still materialized)

**Roofline prediction (T4 sm_75, B=1 H=8, fp32, materialize_S, tile = 64x64 @ d=64 / 32x32 @ d=128):**
tiling cuts operand re-reads, raising AI 0.25 → **8.0 (d=64)** / **6.4 (d=128)** — a ~30× traffic
cut — but **stays below the 25.3 ridge**, so the predicted limiter is **still HBM** at both head
dims. The thesis: large measured speedup over v1 with the limiter *unchanged* (only v3 online
softmax removes the surviving S round-trip).

**Measured (Colab T4, sm_75, torch 2.11/cu128, FP32, causal=False — 2026-06-19):**
Correctness: **26/26 pass vs SDPA** (atol/rtol 1e-4; v1+v2 × 6 shapes × {causal,non-causal} +
scale), including N=130@d64 and N=100@d128 to exercise the partial-tile boundary guard. v2/v1
below is **clock-matched** — both the v1 and v2 columns are from runs at SM clock **~300 MHz**.

| shape | v2 p50/max ms | v2 vs SDPA | **v2/v1 (@300 MHz)** | v2 measured ÷ tiled LB |
|---|---|---|---|---|
| 1x8x512x64   | 4.52 / 4.56     | 0.04× | 1.33× | 21.5× (overhead-bound, tiny work) |
| 1x8x512x128  | 3.49 / 3.56     | 0.09× | 2.95× | 6.6× |
| 1x8x2048x64  | 31.49 / 32.55   | 0.10× | 2.82× | 9.3× |
| 1x8x2048x128 | 54.05 / 56.20   | 0.11× | **3.20× (peak)** | 6.4× |
| 1x8x8192x64  | 808.1 / 915.8   | 0.06× | 2.02× | 15.0× |
| 1x8x8192x128 | 1087.0 / 1134.5 | 0.10× | 2.84× | 8.1× |

**Reading it — one confirmation, one corrected prediction, two honest misses** (the bench view;
the ncu block below revises the limiter further):
- ⚠️ **Limiter: predicted HBM, but ncu says *not bandwidth-saturated*** (see the measured block
  below). AI is below the 25.3 ridge, so not compute-bound (✅) — but DRAM throughput never exceeds
  ~35%, so the kernels aren't on the HBM roof either. "Below the ridge" held; "bandwidth-bound" did
  not. The real per-pass limiters are L2 bandwidth/latency (qk), the multi-pass S re-reads
  (softmax), and compute (pv, ~80% SM).
- ✅ **Still far slower than SDPA** (0.04–0.11×), because SDPA is fused FlashAttention and we still
  round-trip S through HBM. This is the gap v3 attacks.
- ❌ **The "biggest win at N=8192" sub-prediction did NOT hold.** The win *peaks at mid-N*
  (2048×128 = 3.20×) and *shrinks* at 8192 (2.02–2.84×). Two separate things are going on:
  - **Why only ~3×, not the predicted ~30×:** the roofline says tiled traffic drops ~30× (AI
    0.25→8), but *neither kernel sits on its roofline*. v1 runs **faster** than its cache-free
    floor (the L2 help from Step 1), and v2 runs **6–21× above its own floor** (scalar/un-vectorized
    loads, low occupancy from the 32 KB tile, and PV-pass `__syncthreads` overhead). Those two
    off-roofline effects pull in opposite directions and compress 30× → ~3×. The S round-trip is
    *not* the reason the win is small — S caps the *roofline* win (at AI 8, ~32×), not the realized one.
  - **Why the N-trend (and not monotone in N):** operand traffic and the S round-trip *both* scale
    as N², so the traffic *composition* is N-independent — S cannot explain a trend. The trend is
    entirely off-roofline: v2's distance-from-floor is worst at the extremes (N=512 = 21× above,
    overhead-dominated tiny work; N=8192 = 15× above, load/sync-bound) and best at mid-N (6.4× at
    2048×128), so the ratio peaks at mid-N. The Step 1 corollary tracked only v1's L2 cliff and
    ignored v2's own efficiency curve — which turned out to dominate where the win lands.
  - **Why d=128 beats d=64 at every N (this part was predictable):** the operand term that tiling
    cuts is a bigger share of traffic at d=128 (operand:S ≈ 16:4 = 4:1, vs ≈ 4:4 = 1:1 at d=64),
    so tiling bites harder. Every d=128 row out-speeds its d=64 sibling.

So the *headline* Step 1→2 lesson holds (guaranteed reuse → 2–3.2× with the limiter unchanged),
but the L2-driven corollary about *where* the win lands was wrong — a clean reminder that **the
roofline bounds traffic, not a kernel's distance from that bound**, and realized speedup is the
ratio of two such distances.

**ncu — measured `dram__bytes_read.sum` per pass (Colab T4 sm_75, 2026-06-19), two shapes:**

| shape · ver | qk | softmax (S) | pv | **total reads** | v1/v2 |
|---|---|---|---|---|---|
| N=512 · v1  | 3.68 MB | 49.83 MB | 18.36 MB | **71.9 MB** | — |
| N=512 · v2  | 3.11 MB | 49.91 MB | 13.49 MB | **66.5 MB** | 1.08× |
| N=8192 · v1 | 0.074 GB | 26.33 GB | 4.39 GB | **30.80 GB** | — |
| N=8192 · v2 | 1.433 GB | 25.53 GB | 3.26 GB | **30.22 GB** | 1.02× |

SoL context: DRAM throughput peaks at ~35% (softmax @ N=8192), ≤15% at N=512 — never bandwidth-
saturated. SM throughput: pv ~80% (compute-bound), qk ~6% (latency-bound). v2 occupancy halves on
qk/pv (97→49%) — the 32 KB tile cost, measured.

**The ncu read overturns the predicted traffic story — the deepest honest miss yet:**
- **Tiling did NOT cut DRAM traffic — at either shape (1.08× / 1.02×).** The predicted ~30× operand
  cut never appears: the T4's 4 MB L2 *already* serves the redundant Q/K reads, so v1's qk pass
  reads just **74 MB at N=8192** (not the ~hundreds of GB the cache-free model charges). L2 had
  already done tiling's job, leaving v2 nothing to cut — and on qk v2 reads **19× *more*** than v1
  (1.43 vs 0.07 GB), because its tiled access + halved occupancy gets *worse* L2 reuse than v1's
  plain streaming.
- **~99% of all DRAM reads are the S matrix.** At N=8192 the softmax pass alone re-reads S ~12× =
  26.3 GB, pv reads it again (~4 GB), and Q/K/V operands are <1% (S-share ≈ 80% at N=512 → 99% at
  N=8192). S is byte-identical in v1 and v2 (26.33 vs 25.53 GB) — tiling the matmul operands does
  nothing to the score matrix. The model had it backwards: **S isn't a floor *under* the operand
  mountain — S *is* the mountain; the operands are the rounding error.**
- **Nothing is HBM-bandwidth-bound** (≤35% DRAM throughput): bound by L2 bandwidth/latency (qk),
  the multi-pass S re-reads (softmax), and compute (pv). The roofline's "HBM limiter" was a
  prediction the hardware never honored.
- **So v2's 1.3–3.2× speedup is a compute/scheduling win, not a bandwidth win** — DRAM traffic is
  flat; the tiled matmuls just execute more efficiently. (`dram__bytes_read` is reads only; the S
  *writes* aren't counted, so S's true dominance is even larger.)

**This sharpens Step 3 instead of weakening it:** online softmax deletes the materialized S — i.e.
~99% of measured DRAM traffic. The ncu read promotes "kill the S round-trip" from a footnote to the
whole game. Reports: `profiling/raw/{v1_naive,v2_tiled,v1_n8192_d64,v2_n8192_d64}.ncu-rep`.

---

## Step 3 — Online softmax (FP32, two-pass, S never materialized)

`kernels/v3_online/` — two passes that stream the key axis keeping a running `(m, l)` per query
row, so the score matrix S is computed on-chip and discarded. Pass 1 produces the final `(m, l)`
(running-max rescale on the scalar `l`); pass 2 recomputes scores and forms `O = (exp(s−m)/l)·V`
with no O-rescale. The only HBM scratch is `(m, l)` — 2 floats/query row. Deliberately keeps v2's
operand handling *out* of scope (pass 2 reads K/V from global, L2-resident per the Step-2 read) so
the one isolated variable is S-elimination.

**Roofline prediction (T4 sm_75, B=1 H=8, fp32, `materialize_S=False`, read-once, tile 1×1):**

| shape (BxHxNxd) | intensity (FLOP/B) | ridge | predicted limiter | predicted lower bound |
|---|---|---|---|---|
| 1x8x512x64   | 128.0  | 25.3 | **MMA** | 0.066 ms |
| 1x8x512x128  | 128.0  | 25.3 | **MMA** | 0.133 ms |
| 1x8x2048x64  | 512.0  | 25.3 | **MMA** | 1.060 ms |
| 1x8x2048x128 | 512.0  | 25.3 | **MMA** | 2.121 ms |
| 1x8x8192x64  | 2048.0 | 25.3 | **MMA** | 16.968 ms |
| 1x8x8192x128 | 2048.0 | 25.3 | **MMA** | 33.936 ms |

> AI = N/4 (HBM bytes collapse to Q/K/V-once + O-write, ~16·B·H·N·d), so it climbs with N and sits
> 5–80× above the FP32 ridge → model says **MMA-bound at every shape**, HBM util ~5%, MUFU ~3%.
>
> **Do NOT trust this crossing yet (the Step-2 discipline).** The model is blind to L2 and **under-
> counts this kernel two ways**: (1) two-pass computes scores *twice*, so real MMA is ~1.5× the
> modeled QK+PV (QK twice + PV once vs once+once); (2) it counts *one* `exp` per score, but two-pass
> does ~2× the `exp` work — so the modeled MUFU 3% is optimistic and `exp2`/MUFU is the **dark-horse
> limiter**. Both pushes are off-roofline; ncu (gate 6) is what settles MMA-vs-MUFU. The number the
> prediction is most confident about — that S-driven DRAM is *gone* — is the one Step 2 makes us
> trust, because S genuinely leaves HBM here rather than merely being L2-cached.

**Measured (vast.ai rented T4 sm_75, driver 560.35.03, torch 2.6.0+cu124, FP32 — 2026-06-19):**
Correctness: **15/15 pass vs SDPA** (atol/rtol 1e-4) — all shapes × causal × the partial-tile
boundaries, plus the N=16384 rescale-stability case (causal both ways).

| shape | v3 p50/max ms | v2 p50 ms | SDPA p50 ms | v3÷SDPA | v3÷v2 | roofline |
|---|---|---|---|---|---|---|
| 1x8x512x64    | 12.51/21.30     | 5.45   | 0.198  | 0.02× | **0.44×** | MMA (~0.07ms) |
| 1x8x512x128   | 27.36/27.80     | 3.74   | 0.327  | 0.01× | 0.14× | MMA (~0.13ms) |
| 1x8x2048x64   | 172.7/173.1     | 32.21  | 3.05   | 0.02× | 0.19× | MMA (~1.06ms) |
| 1x8x2048x128  | 379.7/381.0     | 55.51  | 5.60   | 0.01× | 0.15× | MMA (~2.12ms) |
| 1x8x8192x64   | 2556.9/2561.6   | 780.7  | 51.56  | 0.02× | 0.31× | MMA (~16.97ms) |
| 1x8x8192x128  | 5661.7/5673.2   | 1049.3 | 102.66 | 0.02× | 0.19× | MMA (~33.94ms) |

(`v3÷v2 < 1` means v3 is **slower** than v2. Causal roughly halves v3's time, as expected.)

**The headline miss — v3 is 3–7× SLOWER than v2, and the roofline floor is off by 150×:**
- **S is gone — proven without a profiler.** Peak CUDA memory at 8192×64: v2 allocates **+2164 MB**
  (the 2147 MB S matrix + O), v3 allocates **+17 MB** (just O + the `m,l` stats). v3 uses **125×
  less** memory — a direct, counter-free measurement that S never materializes. Gate-6's "is S gone"
  is settled structurally and numerically.
- **But deleting S bought nothing, because nothing was DRAM-bound.** Step 2 already proved the T4 is
  never HBM-bandwidth-limited (≤35% DRAM throughput; L2 owns the operands). v3 removed ~99% of *DRAM
  traffic* that *wasn't the bottleneck* — so wall-clock didn't improve; it got **worse**, because
  v3's deliberately-simple scheduling is far less efficient than v2's tiled GEMMs.
- **Where the time goes (torch profiler, CUPTI trace — no counters needed):** at 8192×64,
  **pass2_output = 88.6%** (2273 ms/call) vs **pass1_stats = 11.4%** (292 ms/call), a 7.8× split.
  Pass 2 carries the PV accumulation *and* re-reads K and V unstaged from global per key, with only
  64 of 256 threads active (one-thread-per-row). The 2565 ms/iter the profiler sums matches the
  2556 ms bench — internally consistent.
- **The roofline was wrong about the limiter — again, and for a new reason.** Measured 2556 ms at
  8192×64 is **151× above** the 17 ms MMA floor, so the kernel is nowhere near compute-saturated.
  It isn't HBM-bound (memory proof) and isn't MMA-bound (151× off) and isn't MUFU-bound — it's
  **occupancy/latency-bound**: low active-thread count + uncoalesced per-row global K/V reads in
  pass 2. The roofline keeps mispredicting because it assumes an *efficient* kernel; v3's scheduling
  is the dominant term it can't see (the Step-2 L2 lesson, in a new disguise).

**Lesson:** v3 *isolates and proves the S-elimination mechanism* (correct output + 125× less memory),
which is the algorithmic foundation FlashAttention is built on — but on a machine that was never
bandwidth-bound, removing DRAM traffic while regressing the schedule is a **net wall-clock loss**.
The speedup is deferred to v4 (single-pass fusion with O-rescale + proper parallelization: one warp
per query row, staged K/V, register-resident O accumulator), which keeps S off HBM *and* schedules
like v2's GEMMs.

**ncu (MMA-vs-MUFU pipe-util read):** *deferred to a bare-metal box.* Every containerized rental
(vast.ai, RunPod, Lambda containers) blocks GPU hardware-counter access (`ERR_NVGPUCTRPERM`) for
tenant isolation — confirmed across multiple hosts and `--cap-add=SYS_ADMIN` attempts. The
counter-free path (CUDA-event timing + memory measurement + execution-path decomposition via CUPTI
trace + roofline) already settles the story; the formal pipe-util read is a one-cell follow-up on a
dedicated/bare-metal GPU (Qubrid/Lambda dedicated/CoreWeave), exactly as Step 1 deferred its ncu.

## Step 4 — Fused FlashAttention-1 (FP32, single-pass, S off HBM, warp-per-row)

`kernels/v4_fused/` — one fused kernel. One warp owns a query row; the block stages each K/V block
into smem once (reused by all 8 warps); each score is a 32-lane `__shfl` butterfly dot product; the
online update keeps a running `(m, l)` and a **register-resident O accumulator**, rescaling O by
`exp(m_old−m_new)` on every max climb — the O-rescale v3 deferred. S never materializes and there is
no per-row HBM scratch at all. One isolated change vs v3: thread→warp granularity (+ the rescale).

**Roofline prediction (T4 sm_75, B=1 H=8, fp32, `materialize_S=False`, tile 1×1)** — identical floor
to v3; the deliverable is the *distance*, not a new prediction:

| shape (BxHxNxd) | intensity (FLOP/B) | ridge | predicted limiter | predicted lower bound |
|---|---|---|---|---|
| 1x8x512x64   | 128.0  | 25.3 | **MMA** | 0.066 ms |
| 1x8x2048x64  | 512.0  | 25.3 | **MMA** | 1.060 ms |
| 1x8x8192x64  | 2048.0 | 25.3 | **MMA** | 16.968 ms |
| 1x8x8192x128 | 2048.0 | 25.3 | **MMA** | 33.936 ms |

> Single-pass computes QK *once* (v3's two-pass did it twice), so the modeled MMA is now exactly
> right rather than under-counted. The model still says MMA-bound everywhere; Steps 2–3 already
> taught us it's blind to the schedule. The question is *how far* v4 lands from the floor now that
> the schedule is fixed.

**Measured (vast.ai rented T4 sm_75, torch 2.6.0+cu124, FP32 — 2026-06-20):**
Correctness: **17/17 pass vs SDPA** (atol/rtol 1e-4) — all shapes × causal × partial-tile boundaries,
the explicit-scale case, plus the N=16384 O-rescale stability at d=64 *and* d=128, causal both ways.

| shape | v4 p50/max ms | v2 p50 ms | v3 p50 ms | SDPA p50 ms | v4÷SDPA | v4÷v2 | v4÷v3 | v4 ÷ floor |
|---|---|---|---|---|---|---|---|---|
| 1x8x512x64   | 1.68/1.69     | 3.55   | 12.56  | 0.317 | 0.19× | **2.12×** | 7.49×  | 25.4× |
| 1x8x512x128  | 2.19/2.20     | 3.72   | 27.59  | 0.392 | 0.18× | 1.70× | 12.59× | 16.5× |
| 1x8x2048x64  | 18.57/18.91   | 34.20  | 172.6  | 3.03  | 0.16× | **1.84×** | 9.30×  | 17.5× |
| 1x8x2048x128 | 25.45/26.08   | 59.45  | 382.9  | 5.65  | 0.22× | 2.34× | 15.05× | 12.0× |
| 1x8x8192x64  | 299.6/303.9   | 652.8  | 2568   | 47.01 | 0.16× | **2.18×** | 8.57×  | 17.7× |
| 1x8x8192x128 | 432.5/439.7   | 1118   | 5924   | 99.03 | 0.23× | 2.59× | 13.70× | 12.7× |

(`v4÷v2 > 1` means v4 is **faster** than v2. Causal runs ≈ 0.6–0.8× the non-causal time — the
early-out mask path — e.g. 8192×64 drops 299.6 → 204.2 ms.)

**Reading it — the thesis is confirmed, with an honest ceiling:**
- ✅ **v4 beats v2 (1.7–2.6×) and crushes v3 (7.5–15×).** This is the first version where "S never
  touches HBM" is *also* a wall-clock win — the goal set since Step 3. **The v2 win is two things at
  once:** (a) *fusion* — v4 eliminates v2's **2 GB** S DRAM round-trip (v2 writes S then re-reads it
  ~12× across the softmax/PV sweeps; S is far too big for the 4 MB L2) and collapses three kernels
  into one; (b) a *tight single-kernel schedule*. The clean three-way contrast proves you need
  **both**: v3 had (a) without (b) — S gone but one-thread-per-row — and was *slower* than v2; v2 has
  (b) without (a) — good tiled GEMMs but the S round-trip — and is slower than v4. So S-elimination
  is **necessary but not sufficient**; splitting the 2× cleanly between saved-S and schedule needs
  ncu (deferred).
- ✅ **S still gone — +16.8 MB at 8192×64** (vs v2's +2164 MB), even less than v3's +17.3 MB because
  the fused loop keeps `(m, l)` in registers — no HBM scratch at all.
- ✅ **Schedule fixed — one kernel, no pass2 wall.** CUPTI trace: a single `fused_attention_kernel`
  at **100% of CUDA time**, 292 ms/call (matches the 299.6 ms p50). v3's 88.6%-pass2 occupancy wall
  is structurally gone — there are no passes. Distance to floor went **151× (v3) → ~18× (v4)**, so
  thread→warp closed ~8.5× of the gap.
- ⚠️ **But still ~6× slower than SDPA and ~18× above the FP32 MMA floor.** v4 doesn't reach the floor
  because the warp-per-row scoring is **GEMV-shaped**: each score is a 5-step `__shfl` reduction + 2
  FMAs, so reduction overhead — not FMA throughput — dominates. It never approaches the 8.1 TFLOPS
  FP32 peak the floor assumes. Fixing *occupancy* (v3→v4) and fixing *FMA utilization* (the next step)
  are different axes.
- **Roofline mispredicted a 4th time — in magnitude (17 ms predicted, 292 ms measured).** The limiter
  has moved every step (HBM → MMA → occupancy → FMA-efficiency) but the model's blind spot is
  constant: flops/bytes can't see reduction overhead or FMA-pipe utilization, so it always assumes an
  efficient kernel.

**Lesson:** v4 delivers the Step-4 thesis — keep S off HBM *and* schedule like a GEMM, and you finally
beat tiling (2× over v2). "Compute-bound on the roofline" turned out **not** to mean "near peak
FLOPs": a kernel can be compute-bound *and* 18× slow if the compute is shaped wrong (GEMV reductions
instead of GEMM FMAs). That remaining gap is the explicit target of Step 5 (tensor cores), which both
raises the ceiling 8→65 TFLOPS *and* forces GEMM-shaped MMA tiles.

**ncu:** still deferred — containerized rentals block hardware counters (`ERR_NVGPUCTRPERM`). The
counter-free evidence (memory proof + CUPTI single-kernel trace + roofline distance) already names
the limiter as FMA under-utilization; the pipe-util read is a bare-metal follow-up.

---

## Step 6 — Split-KV decode (v6)

DONE (measured 2026-06-27, vast.ai Tesla T4 sm_75, torch 2.6.0+cu124). FP16-in/FP32-accum, two kernels
(partial + merge) behind one `forward`. **25/25 correctness** vs SDPA (square SHAPES where
`choose_splits → 1` reduces v6 to plain attention + `test_v6_splitkv_decode`: `N_q=1`, `N_k ∈ {4096,
8192, 8190}`, d 64/128, causal both ways). The decode arc (v6→v11) is in `docs/b300-decode-research.md`.

**Why this step:** v1–v5 parallelize over query rows, so at decode (`N_q = 1`) the grid collapses to
`(1, B·H)` and the T4's 40 SMs sit idle (research blind-spot #2). v6 splits the KV axis across blocks
(Flash-Decoding) to fill the SMs without changing the math.

**Roofline prediction (T4 sm_75, decode `N_q = 1`, FP16 KV `b = 2`):** decode arithmetic intensity
`AI = 2/b = 1.0` FLOP/byte, **independent of `N_k`** (research §4) — far below the T4 fp16 ridge (~203),
so the predicted limiter is **HBM** and the floor is `time ≈ (2·B·H·N_k·d·2 bytes) / 320 GB/s`. The
split-KV schedule does not change these bytes; it only lets the kernel *reach* this bound.

**Measured — decode (B=1, H=8, N_q=1), the real decode workload (non-causal over the full KV cache):**

| q×kv shape | ours p50/max ms | µs/tok | %HBM | vs SDPA | vs naive (v5 @ N_q=1) | HBM floor |
|---|---|---|---|---|---|---|
| 1x8x1x64/2048   | 0.152/0.182 |  18.94 |  8.6% | 2.44× | 6.03× | ~0.013ms |
| 1x8x1x128/2048  | 0.219/0.233 |  27.38 | 12.0% | 1.50× | 5.73× | ~0.026ms |
| 1x8x1x64/8192   | 0.449/0.467 |  56.06 | 11.7% | 3.16× | 7.86× | ~0.052ms |
| 1x8x1x128/8192  | 0.696/0.709 |  87.00 | 15.1% | 1.80× | 6.90× | ~0.105ms |
| 1x8x1x64/16384  | 0.848/0.868 | 105.99 | 12.4% | 3.32× | 8.23× | ~0.105ms |
| 1x8x1x128/16384 | 1.358/1.371 | 169.73 | 15.4% | 1.83× | 7.04× | ~0.210ms |

(`%HBM` = the K+V read `2·B·H·N_k·d·2 B` as a fraction of the 320 GB/s peak. `vs naive` = v5_wmma run at
`N_q=1` — the `1×BH` single-block-per-head loop v6 must beat. Clock read was unavailable on this box —
`clock~-1/-1`; timing is CUDA-event based, unaffected.)

**Reading it — the split-KV thesis lands, but the bandwidth bound is NOT reached:**
- ✅ **v6 beats the naive `N_q=1` loop 5.7–8.2×, growing with `N_k`** (6.0× @2048 → 8.2× @16384, d=64).
  This is the deliverable: split-KV spreads the KV scan across `choose_splits` blocks per head (BH=8 →
  8 splits @2048, 10 @8192/16384 → 64–80 blocks), filling the SMs that the single-block loop starved.
  The win grows with `N_k` because there's more KV to parallelize and the fixed launch/merge cost
  amortizes.
- ✅ **v6 beats torch SDPA 1.5–3.3× (non-causal).** At `N_q=1` SDPA falls back to a path not tuned for a
  single query on Turing; the hand-written split-KV is faster. The edge is bigger at d=64 (2.4–3.3×)
  than d=128 (1.5–1.8×) — d=128 has more work per key, where v6's GEMV-shaped per-key scoring is less
  efficient.
- ⚠️ **But %HBM is only ~9–15% — v6 does NOT saturate HBM.** Measured time is **~7–9× above the HBM
  floor** (0.449 ms vs 0.052 ms floor @64/8192). The roofline got the *location* right (HBM-bound in
  principle) but the kernel never reaches it: at BH=8 the grid is only 64–80 blocks on 40 SMs (~2
  blocks/SM, ~16 of 32 warps), and there's a **two-kernel launch + a tiny under-occupied merge** on top.
  The real limiter is still **occupancy / launch + reduction latency**, not bandwidth — the same finding
  as v3/v4, now at decode. The roofline mispredicted *magnitude* a 5th time, same flops/bytes blind spot
  (it can't see the schedule). %HBM rising with `N_k` (8.6→12.4%) is the fixed overhead amortizing, not
  the kernel getting closer to the bandwidth wall structurally.

**The causal rows are a degenerate-shape artifact — disregard for perf:** with `q` at row index 0 and
the causal rule `j > i`, the single query (`i=0`) can attend to **only key 0** — one key, not the cache.
SDPA short-circuits that (v6 still launches all splits over the full `N_k`), so causal `vs sdpa` reads a
meaningless 0.03–0.28× and `%HBM` is inflated (the formula counts the full `2·N·d` bytes the kernel
never needed). Correctness still holds there (v6 matches SDPA on the 1-key answer — it's in the 25/25).
**Realistic causal decode would place the query at position `N_k−1` (attending to all keys), which is
exactly what the non-causal rows already measure.** Recorded as a harness limitation: `--decode` should
offset the query position for a meaningful causal-decode timing (a v7+ improvement, not a v6 bug).

For reference, the causal (degenerate) rows as run: 64/2048 0.100 ms (vs naive 9.13×), 128/2048 0.170
(7.38×), 64/8192 0.303 (11.62×), 128/8192 0.560 (8.57×), 64/16384 0.563 (12.38×), 128/16384 1.085
(8.80×). The `vs naive` numbers are still valid (both v5 and v6 see the same 1-key shape); only `vs sdpa`
and `%HBM` are meaningless here.

**ncu:** deferred (counter-free norm — the bench's CUDA-event timing + the roofline-distance read above
already name the limiter as occupancy/launch, not bandwidth; a pipe-util read is a bare-metal follow-up).

**Next (v7 — paged KV gather):** block-table indirection, correctness with non-contiguous KV; and the
harness causal-decode query offset above. The %HBM gap says the lever after that is *occupancy* (bigger
batch/GQA packing to grow the grid) before *bytes* (FP8/NVFP4 KV at v8/v9).

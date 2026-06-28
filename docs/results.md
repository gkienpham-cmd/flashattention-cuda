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

## Step 5 — Tensor-core FlashAttention-1 (v5 WMMA, FP16-in/FP32-accum, GEMV→GEMM)

> **Status — PARTIAL / backfilled 2026-06-27.** The kernel + wiring landed in commit `ad021b7`
> (`kernels/v5_wmma/`, registered in `bindings/load.py` and gated to cap (7,5) in `dispatch.py`) and
> the **correctness gate is green** (rebuilt + passed during the Step-6 vast.ai run; see below). **The
> prefill bench headline (v5÷v4, v5÷SDPA) was never captured:** `notebooks/step5_run_of_record.ipynb`
> was authored but never executed — every cell shows `execution_count=None` with zero saved outputs,
> and no v5 *prefill* bench appears in any other notebook. The only *measured* v5 numbers that exist
> are v5 run `@N_q=1` as the "naive" decode baseline in Steps 6/7 (those tables). What is recorded
> below is verified from source + the (CPU-reproducible) roofline + the Step-6 correctness rebuild;
> the speedup table is left as an **OUTSTANDING measurement** — run the run-of-record to fill it. This
> is logged as a gap rather than papered over (project ethos).

`kernels/v5_wmma/` — the **GEMV→GEMM fix** for v4's diagnosed limiter. Keep v4's single-pass fused
schedule (online softmax, O-rescale, S never on HBM) but move *both* matmuls onto Turing's tensor
cores (WMMA, 16×16×16, FP16-in / FP32-accum) instead of v4's 32-lane `__shfl` dot products:
- **QK:** `S = Q @ Kᵀ` via `wmma::mma_sync` — the reduction over `d` lives *inside* the tensor core
  (no shuffle). K is loaded as a `col_major` B-fragment so Kᵀ needs no transpose.
- **PV:** `O += P @ V` via `wmma::mma_sync` — the running FP32 O (`oRun`) is loaded *into* the
  accumulator fragment so the O-rescale add is the tensor core's own `C += A·B`.
- **The opaque-fragment tax:** a WMMA accumulator's 256 results are scattered across lanes in an
  undocumented layout you cannot index, so softmax *cannot* stay in registers (v4 kept O in named
  regs). v5 forces S through smem: `QK → store S to smem → row-softmax in smem (each lane owns a full
  row, no shuffle — the GEMM already reduced) → write P back as half → reload P as fragments for PV`.
- **Tiling** — one warp per 16-row query M-tile, sized to Turing's 48 KB static-smem budget:
  `d=64 → BM=64, BN=32, 4 warps (~44 KB smem)`; `d=128 → BM=32, BN=32, 2 warps (~46 KB)`.
- **Precision:** first version **not** bit-comparable to FP32 SDPA — FP16 in, FP32 accumulate, FP32
  out; the public `attention()` contract stays FP32-in (host casts to half).

**Roofline prediction (T4 sm_75, B=1 H=8, fp16, `materialize_S=False`):** the FP16 tensor-core peak
(~65 TFLOPS) is ~8× the FP32 CUDA-core peak (8.1), so **the MMA floor drops ~8× vs v4** and the ridge
moves up 8× (25.3 → **203.1** FLOP/byte) — still compute-bound at every shape. The deliverable is the
*distance to this lowered floor*:

| shape (BxHxNxd) | intensity (FLOP/B) | ridge | predicted limiter | v5 fp16 floor | v4 fp32 floor |
|---|---|---|---|---|---|
| 1x8x512x64   | 256.0  | 203.1 | **MMA** | 0.008 ms | 0.066 ms |
| 1x8x512x128  | 256.0  | 203.1 | **MMA** | 0.017 ms | 0.132 ms |
| 1x8x2048x64  | 1024.0 | 203.1 | **MMA** | 0.132 ms | 1.060 ms |
| 1x8x2048x128 | 1024.0 | 203.1 | **MMA** | 0.264 ms | 2.120 ms |
| 1x8x8192x64  | 4096.0 | 203.1 | **MMA** | 2.114 ms | 16.968 ms |
| 1x8x8192x128 | 4096.0 | 203.1 | **MMA** | 4.229 ms | 33.936 ms |

> The floor is exactly ⅛ of v4's at every shape (the 65/8.1 TFLOPS ratio). v5 attacks v4's gap on
> *both* axes at once: it raises the ceiling 8→65 TFLOPS **and** forces GEMM-shaped MMA tiles instead
> of v4's GEMV-shaped shuffle reductions. Whether v5 lands *near* this far-lower floor — or whether a
> new limiter (the smem S round-trip the opaque-fragment tax forces, or the small 16×16 tiles
> under-filling the warp) appears — is exactly what the un-captured prefill bench must answer.

**Measured (vast.ai rented T4 sm_75, torch 2.6.0+cu124, FP16-in):**
- ✅ **Correctness — green.** v5 is in the `BACKENDS` sweep at tol atol/rtol **2e-2** (the first
  loosened band, FP16-in; v6 later shares it). Coverage is the same 17 cases as v4: 6 shapes × causal
  both ways (12) + the explicit-scale case (1) + `test_v5_wmma_fp16_stability` — d 64/128 × causal,
  N=16384 (4), which stresses the O-rescale ordering and FP16 drift across the longest accumulation
  *and* the d=128 BM=32/2-warp tile. Rebuilt and passed during the Step-6 vast.ai run (per the Step-6
  rebuild; the standalone `step5_run_of_record.ipynb` was never executed/committed with outputs).
- ✅ **S-elimination — by construction.** v5 keeps v4's design: running `(m,l)` per owning lane,
  FP32 `oRun` in smem, S only transiently in smem, never in HBM. No HBM scratch (the smem S round-trip
  is the new cost, not a DRAM one). Peak-memory proof not separately captured, but structurally
  identical to v4's +16.8 MB.
- ⏳ **Prefill speedup (v5÷v4, v5÷SDPA, distance-to-floor) — OUTSTANDING.** Not measured; run
  `notebooks/step5_run_of_record.ipynb` (cells 19–20 bench v5 fp16 then v4 fp32 for the apples-to-FP32
  comparison; cell 22 is the causal sweep) to fill this in.

| shape | v5 p50/max ms | v4 p50 ms | SDPA p50 ms | v5÷SDPA | v5÷v4 | v5 ÷ fp16 floor |
|---|---|---|---|---|---|---|
| 1x8x512x64   | — | 1.68   | 0.317 | — | — | — |
| 1x8x512x128  | — | 2.19   | 0.392 | — | — | — |
| 1x8x2048x64  | — | 18.57  | 3.03  | — | — | — |
| 1x8x2048x128 | — | 25.45  | 5.65  | — | — | — |
| 1x8x8192x64  | — | 299.6  | 47.01 | — | — | — |
| 1x8x8192x128 | — | 432.5  | 99.03 | — | — | — |

*(v4/SDPA columns carried from the Step-4 table for the eventual side-by-side; v5 columns pending the
run-of-record.)*

**Lesson (partial, pending the measurement):** v5 is the structural answer to v4's "compute-bound but
18× slow" finding — fix the math-*shape* (GEMV→GEMM via tensor cores), not just the schedule. The
prediction says the floor drops 8×; the open question the unexecuted bench leaves is whether v5
*reaches* it or trades the FMA-utilization wall for a new one (smem S round-trip / small-tile
under-fill). Until the run-of-record runs, Step 5's headline is a prediction, not a result.

**ncu:** deferred, same `ERR_NVGPUCTRPERM` containerized-rental block as Steps 3–4.

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
- ⚠️ **But %HBM is only ~9–15% — v6 does NOT saturate HBM.** Measured time is **6.5–11.6× above the HBM
  floor** (0.449 ms vs 0.052 ms floor @64/8192 = 8.6×; spread re-derived from the table, tighter than the
  earlier "~7–9×"). The roofline got the *location* right (HBM-bound in
  principle) but the kernel never reaches it: at BH=8 the grid is only 64–80 blocks on 40 SMs (~2
  blocks/SM, ~16 of 32 warps), and there's a **two-kernel launch + a tiny under-occupied merge** on top.
  The real limiter is still **occupancy / launch + reduction latency**, not bandwidth — the same finding
  as v3/v4, now at decode. The roofline mispredicted *magnitude* a 5th time, same flops/bytes blind spot
  (it can't see the schedule). %HBM rising with `N_k` (8.6→12.4%) is the fixed overhead amortizing, not
  the kernel getting closer to the bandwidth wall structurally.
- ⚠️ **This 12% is a `BH=8` (single-stream) result — and it is batch-conditional (deep-research, 2026-06-27).**
  `choose_splits` self-disables (`num_splits→1`) once `base_blocks = BH ≥ 2·num_sm`, so at production batch
  (T4 `BH≥80`, `B≥10`; B300 `BH≥320`, `B≥40`) batch *alone* fills the SMs and the kernel should reach its HBM
  ceiling — there it *is* bandwidth-bound. **The bench only ran `B=1`**, the worst-case corner, so the
  large-batch end-state is **predicted (code-trace), not measured** → **v7 adds a `--batch` sweep** to turn
  the occupancy→bandwidth crossover into a measured curve (see `docs/decode-replan.md §2.1` +
  `diagrams/decode-roofline-crossover.svg`).

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

**Next (v7 — paged KV gather):** block-table indirection, correctness with non-contiguous KV; the
harness causal-decode query offset above; **and a `--batch` sweep** to measure the crossover. The %HBM gap
says the lever after that is *occupancy* — **GQA M-packing (v8, `AI=2/b→2G/b`, GEMV→GEMM)** — before
*bytes* (FP8 v9 / NVFP4 v10 on B300). Full reordered plan + math + diagrams:
[`decode-replan.md`](decode-replan.md).

## Step 7 — Paged KV gather + decode-harness fixes (v7)

*DONE 2026-06-27 (vast.ai Tesla T4 sm_75, torch 2.6.0+cu124, `clock~-1/-1`), both gates: **51/51
correctness** (`-k "v7_paged or v6_splitkv"`: paged shuffled-pool gather + v6 regression) + **quiz
passed**. **The `--batch` sweep refutes the predicted occupancy→bandwidth crossover** — see below; this
is the load-bearing finding of the whole reorder.*

**Why this step:** v6's KV was contiguous; a real KV cache is a pool of fixed-size pages plus a
per-sequence block table (vLLM's layout — the import surface a from-scratch mini-vLLM consumes). v7
adds that **block-table gather** and two harness fixes that turn v6's *predicted* occupancy→bandwidth
crossover into a *measured* one. **v7 deliberately does NOT attack the limiter** — it carries v6's
split-KV + LSE merge byte-for-byte and isolates one new variable; it sets up v8 (GQA M-packing).

**Roofline prediction (T4 sm_75, decode `N_q=1`, FP16 KV `b=2`) — UNCHANGED from v6:**
`AI = 4·N_k·d / (2·N_k·d·b) = 2/b = 1.0` FLOP/byte, `N_k`-independent. The gather adds only
`O(N_k/page_size)` int32 index reads — at page_size=16, N_k=8192, d=64 that's ~2 KB of indices vs ~2 MB
of KV (~0.1%), so v7 is **byte-neutral and occupancy-neutral by construction**. Predicted limiter:
**HBM** (the floor is v6's); the wall-clock *gap* to that floor is the occupancy story the `--batch`
sweep measures. (Caveat, recorded per `decode-replan §7`: `roofline/model.py` has no schedule term —
it has been magnitude-wrong 5 straight steps — so treat its %HBM as an upper bound of 100% and read
the 100%→measured gap as the deliverable, not the model's number.)

**THE prediction v7 tested (the crossover):** `choose_splits` collapses `num_splits→1` once
`base_blocks = BH ≥ 2·num_sm` (= 80 on the T4), so split-KV self-disables and batch *alone* was
predicted to fill the SMs and drive `%HBM` from ~12% toward 60–80%+ by BH≈80–160. **This prediction is
measured FALSE (see the sweep below).**

**Measured — correctness (Gate 1 ✅): 51/51** (`pytest -k "v7_paged or v6_splitkv"`, 46 s). v7's paged
cases (shuffled-pool gather; `N_q=1`, `N_k ∈ {4096,8192,8190}`, page_size ∈ {128,256}, d 64/128, causal
both ways via the query-offset; the `num_splits→1` square regression = 24 + 2) **and** all of v6's cases
(12 + 1 + 12, untouched) pass at tol 2e-2. The shuffle in `build_paged_kv` is the teeth: a kernel that
read contiguously instead of through the block table would grab the wrong tokens and fail.

**Measured — decode bench (B=1, H=8, non-causal):**

| q×kv | µs/tok | %HBM | vs SDPA | vs naive (v5 @ N_q=1) |
|---|---|---|---|---|
| 1x8x1x64/2048   |  24.05 |  6.8% | 1.92× | 4.76× |
| 1x8x1x128/2048  |  34.04 |  9.6% | 1.22× | 4.57× |
| 1x8x1x64/8192   |  66.65 |  9.8% | 2.66× | 6.59× |
| 1x8x1x128/8192  | 107.38 | 12.2% | 1.46× | 5.59× |
| 1x8x1x64/16384  | 128.01 | 10.2% | 2.75× | 6.81× |
| 1x8x1x128/16384 | 210.88 | 12.4% | 1.48× | 5.66× |

v7 reproduces v6's shape (HBM-limiter, beats SDPA 1.2–2.8×, beats v5-naive 4.6–6.8×). It runs **~15–25%
slower than v6's recorded run** (64/8192: 66.7 vs v6 56.1 µs/tok) — *apparent* gather overhead (the
per-key block-table lookup is a dependent global load + a div/mod the contiguous read didn't have). The
roofline says byte-neutral and is right on bytes; the cost is a *schedule* term (dependent-load latency)
it can't see. **Caveat:** v6 and v7 were separate vast.ai sessions with `clock~-1/-1` (clocks
unreadable), so the cross-run delta is indicative, not proven — the within-run sweep below is the robust
signal.

**Measured — causal decode now meaningful (the query-offset fix worked):** causal µs/tok now ≈
non-causal per row — 64/8192 **66.79** (vs 66.65 non-causal), 128/16384 **210.18** (vs 210.88). The
single query at logical `N_k−1` attends the whole cache, so causal == the full scan, exactly as
designed. (`vs sdpa` reads 0.02–0.15× now because the bench's SDPA baseline still uses the *degenerate*
upper-left causal mask — SDPA computes 1 key, v7 computes all N_k. The column is meaningless here in the
opposite direction; `vs naive` 4.6–6.8× stays valid. A fully honest causal `vs sdpa` would need the
reference built with a bottom-right mask — a follow-up.)

**Measured — the `--batch` crossover sweep (the crown jewel), N_k=8192:**

| B | BH | µs/tok (d=64) | %HBM (d=64) | µs/tok (d=128) | %HBM (d=128) | vs SDPA | vs naive |
|---|---|---|---|---|---|---|---|
| 1  | 8   |  66.83 |  9.8% | 107.78 | 12.2% | 2.65/1.46× | 6.59/5.58× |
| 8  | 64  |  70.02 |  9.4% | 114.93 | 11.4% | 0.56/0.34× | 2.92/2.12× |
| 16 | 128 |  69.62 |  9.4% | 114.55 | 11.4% | 0.56/0.35× | 2.93/2.12× |
| 32 | 256 |  68.93 |  9.5% | 113.26 | 11.6% | 0.51/0.32× | 2.63/1.93× |
| 64 | 512 |  63.66 | 10.3% | 106.03 | 12.4% | 0.50/0.32× | 2.68/1.96× |

**The answer: NO — the crossover does not happen. `%HBM` is FLAT at 9.4–12.4% from BH=8 to BH=512.**
Per-token cost is essentially constant (even *slightly better* at B=64, where `num_splits→1` gives 512
blocks = 12.8/SM, far past the "4 blocks/SM full occupancy" mark). Batch over-fills the grid and `%HBM`
does not budge. **The predicted occupancy→bandwidth crossover is measured false** — this refutes
`decode-replan §2.1`'s batch-conditional claim that batch alone would reach the HBM ceiling.

**Why (code-verified) — the limiter is *per-CTA*, not grid occupancy, so batch replicates it:**
1. **smem caps residency at 2 blocks/SM:** `sK + sV = 2·TN·D·4B = 32 KB`/block (64×64 and 32×128 both),
   T4 has 64 KB/SM → a hard cap of **2 resident blocks/SM, independent of batch or split count**. More
   blocks just means more *waves* at the same per-SM occupancy → linear time → flat %HBM.
2. **At `N_q=1` only 1 of 8 warps computes** (warp `w` owns query row `w`; only `w=0` is active in the
   score loop — the other 7 do the cooperative load then idle). So ~2 active compute-warps/SM — far too
   few to hide the per-key warp-shuffle reduction latency (the same GEMV wall v4 hit).

So decode here is bound by **smem-limited residency + single-warp GEMV latency** — a per-block
structural limit that **batch cannot cure** (it was never grid-starved past split-KV). `vs sdpa` flips
from 2.65× (B=1, where SDPA's `N_q=1` path is untuned) to 0.3–0.56× at batch (a well-tuned batched
attention pulls ahead), which independently confirms v7's per-CTA inefficiency.

**What this means for the roadmap (the reorder holds, the *reason* sharpens):** GQA M-packing (v8) leads
**not** because it "fills the SMs" (batch already over-fills the grid and it doesn't help) but because
`M = G > 1` activates **G compute-warps/block** *and* turns the GEMV into a tensor-core GEMM with KV read
once — it attacks the actual binding constraint (per-CTA efficiency). And **bytes-first (FP8/FP4) is
premature at *all* batch sizes, not just small batch**: v7 never gets within 8× of the bandwidth wall at
any BH, so cutting bytes multiplies a term it can't reach. `diagrams/decode-roofline-crossover.svg` needs
the "predicted crossover" arc replaced with the measured-flat line. See `decisions.md` Step 7.

**Deep-research close-out (2026-06-27) — adversarial verify pass (35 agents: 6 forensics + 7 research + 7 claims, 2-of-3 refute gate); the headline survives 0/3, and three facts get layered on:**
1. **%HBM is fp16-correct — the printed ~7–12.5% is right, NOT 2× understated.** A skeptic flagged the
   `precision=fp32` bench header; resolved decisively: the kernel *casts* K/V to half
   (`paged_attention.cu:274–276`) and every HBM KV load is `__half` = 2 B/elem (lines 125–126), so the
   `kv_bytes·2`-byte denominator matches bytes-on-the-wire. The header is a stale *input-contract* label (the
   "true %HBM is 2× higher" claim died 3/3 refute). Peak achieved HBM = 12.5% (1×8×1×128/16384) — still ~8×
   below the wall, so per-CTA-bound stands unchanged. *(Cosmetic follow-up: relabel header
   `in=fp32 / kv-read=fp16`; the `.to(kHalf)` cast is a real throwaway fp16 re-allocation **inside** the timed
   region a true fp16-native vLLM cache wouldn't pay — worth a one-line note for the mini-vLLM framing.)*
2. **SDPA overtakes v7 by B=8 (non-causal) — sharper mechanism than "untuned SDPA."** v7 wins *only* at B=1
   (2.65×/1.46×); for B≥8 it **loses** (0.56→0.50× d=64, 0.34→0.32× d=128). Cause: v7 is already SM-saturated
   at B=1 (10 splits → 80 blocks = the 2-blocks/SM cap) so its per-token cost is **flat** (~67 µs/tok d=64),
   while SDPA's per-token cost **collapses ~4.5×** as batch fills *its* grid. At B=1 v7 wins *because* SDPA
   under-fills the grid even worse (~3.7% HBM vs v7's 9.8%). Crossover bracketed in (1, 8] (no B=2/4 sample).
   **This makes "reclaim SDPA at batch" a first-class v8 deliverable** — serving runs continuous batching B≥8.
3. **Causal `vs_sdpa` (0.15→0.02×) is a baseline artifact, not a regression** — confirmed: v7 scans all N_k
   keys while the SDPA reference (`is_causal=True`, top-left-aligned `[1,N_k]` mask) attends ~1 key, so the
   ratio tracks ~1/N_k. The correctness test (`test_correctness.py:209`, `causal=False`) is already the right
   full-cache oracle and passes 51/51; only the bench column misleads. Fix (v8-harness, ~6 lines):
   bottom-right-aligned causal mask in the reference. *(Full cited synthesis with the GQA/B300 research:
   [`v7-deep-research.md`](v7-deep-research.md).)*

## Step 8 — GQA M-packing (v8)

*Cut 1 MEASURED 2026-06-28 (Colab Tesla T4 sm_75, torch 2.11.0+cu128, throttled `clock~360-390/1590MHz`);
**Gate 1 ✅ 64/64 correctness**, decode bench + G-sweep + reclaim-at-batch captured
(`notebooks/v8_gqa_gate_output.ipynb`). **Quiz (Gate 2) pending.** Cut 2 = sm_80 tensor-core variant +
the M≥16 ablation is still `[RENT A100]`. **The thesis lands on Cut 1: G-packing buys ~`G×` wall-clock
(8.6× at G=8) over the no-packing baseline and reclaims SDPA at all batch sizes — without tensor cores.***

**Why this step:** v7 measured decode is **per-CTA-bound at every batch size** — flat ~10–12% HBM from
BH=8→512, refuting the occupancy→bandwidth crossover. The binding constraint is *inside the CTA*: at
`N_q=1` only **1 of 8 warps computes** (GEMV shape) and `sK+sV=32 KB` caps residency at 2 blocks/SM.
v8's one new variable — **GQA M-packing** — packs the `G = H_q/H_kv` query heads that share one KV head
into the score GEMM's `M` dimension, so a CTA reads that KV head **once** and computes `G` query rows
against it: **G warps light up instead of 1**, KV bytes drop by `G`, and decode `AI = 2/b → 2G/b`. It
attacks the *measured* limiter (per-CTA efficiency), which is why it's promoted ahead of the byte levers
(v9 FP8 / v10 NVFP4) — cutting bytes is premature while the kernel sits ≥8× below the bandwidth wall.

**Roofline prediction (Task 1 — `roofline/model.py` now takes `G`; decode `AI = 2G/b`).** Recorded on
**A100 sm_80** (the Cut-2 perf target; the headline runs there), decode `N_q=1`, `N_k=8192`, FP16 KV
`b=2`, B=8 H=8:

| G (group) | model | AI (FLOP/byte) | limiter | t_hbm floor | note |
|---|---|---|---|---|---|
| 1 (MHA)        | `2/b`  | **1.0** | HBM | 0.132 ms | reproduces the v6/v7 decode bound exactly |
| 4 (Llama-3-8B) | `2G/b` | **4.0** | HBM | 0.033 ms | KV read once per 4 heads → 4× fewer bytes |
| 8 (L3-70B)     | `2G/b` | **8.0** | HBM | 0.016 ms | 8× lower floor; still HBM-bound (8 < ridge) |

**The prediction, stated honestly (so the measurement can refute it):**
1. **AI rises exactly `G×` (1→4→8) and the HBM *floor* drops `G×`** (0.132→0.016 ms at G=8) — KV shared
   across the group is the whole mechanism.
2. **The limiter does NOT flip to compute.** A100's FP16 tensor-core ridge is **153 FLOP/byte**; even at
   G=8, AI=8.0 ≪ 153, so the model stays **HBM-bound**. A flip would need `G ≈ 153`, which GQA never
   reaches (real `G ≤ 8–16`). So the headline is *not* "v8 becomes compute-bound."
3. **The real, model-invisible win is per-CTA efficiency.** v7 measured the kernel at only ~10% of its
   HBM floor; v8 should (a) move the kernel much closer to the now-`G×`-lower floor by lighting up `G`
   warps and reading KV once, and (b) on Cut 2, re-engage tensor cores via the `M=G` GEMM. The roofline
   has been magnitude-wrong 5 straight steps because it has no schedule term — so the deliverable is the
   **measured µs/tok drop and the reclaim-SDPA-at-batch**, not the model's floor.

**Predicted G-sweep shape (to be measured):** µs/tok should fall as `G` rises (more warps active, KV read
once), with the **biggest jump on Cut 2 at `G` crossing the M<16→M≥16 tensor-core threshold** (G=8 pads
to 16; G=16 fills a tile). On Cut 1 (CUDA-core) the win is bounded by "G warps + KV-once" with no
tensor-core uplift — that gap is exactly what Cut 2's ablation measures.

**Measured — correctness (Gate 1 ✅): 64/64** (`pytest -k "v8_gqa or v7_paged"`, 81.8 s on the Colab T4).
38 v8 cases (decode `G∈{1,2,4,8}` × non-multiple `N_k∈{4096,8190}` × d 64/128 × causal both ways = 32;
idle-warp `G=3` + multi-tile `G=16` = 4; square-reduction = 2) **and** 26 v7 regression cases pass at tol
2e-2. The `repeat_interleave(G)` oracle (`sdpa_reference_gqa`) confirms the `h_q = h_kv·G + g_local`
mapping; the idle-warp/multi-tile cases confirm `G` need not divide 8 and `M>8` tiles over `grid.x`.

**Measured — the G-sweep (THE deliverable), N_k=8192, H_q=32 fixed, H_kv=32/G, non-causal:**

| G | µs/tok (d64) | %HBM (d64) | **vs no-pack (d64)** | µs/tok (d128) | %HBM (d128) | **vs no-pack (d128)** |
|---|---|---|---|---|---|---|
| 1  | 102.52 | 6.4% | 0.85× | 142.02 |  9.2% | 1.00× |
| 2  |  35.07 | 9.3% | 2.52× |  58.74 | 11.2% | 2.43× |
| 4  |  18.31 | 8.9% | 4.79× |  30.40 | 10.8% | 4.69× |
| 8  |  10.21 | 8.0% | **8.59×** |  16.32 | 10.0% | **8.71×** |
| 16 |   8.44 | 4.9% | 10.42× |  13.49 |  6.1% | 10.59× |
| 32 |   6.92 | 3.0% | 12.74× |  11.20 |  3.7% | 12.73× |

**The mechanism is confirmed.** `vs no-pack` (= v8 ÷ v7 run on the same workload with KV broadcast to
`H_q` heads — the clean same-session isolation of M-packing) tracks **almost exactly `G×` up to G=8**
(8.6×/8.7×), then goes sub-linear (12.7× at G=32, where `M>8` re-stages the KV head over `⌈G/8⌉` blocks).
This is the per-CTA-efficiency win the roofline's `AI=2G/b` implied — `G` warps active + KV read once →
~`G×` wall-clock — realized **on CUDA cores, no tensor cores.** Within v8, µs/tok itself falls ~15× (d64)
/ ~12.7× (d128) from G=1→G=32.

**Prediction-vs-measured (a partial roofline WIN, the first in 6 steps):** the model predicted AI rises
`G×`, the HBM floor drops `G×`, and the limiter **stays HBM** (no compute-flip). Measured: the µs/tok
speedup tracks `G×` (right!), and `%HBM` stays **≤11%** at every G — still **per-CTA-bound, NOT
bandwidth-bound** (the kernel never approaches the floor; M-packing closes most of the per-CTA gap but
leaves headroom for Cut 2's tensor cores). So the model got the *speedup magnitude* right via `AI=2G/b`,
while its *absolute floor* remains unreached — exactly as flagged (the model has no schedule term).
**Caveat:** the Colab T4 was clock-throttled (`~360-390` vs 1590 MHz), so absolute µs/tok is slow, but the
`vs no-pack` / `vs sdpa` ratios are same-session/same-clock and robust.

**Measured — reclaim-SDPA-at-batch (✅ the headline), G=8, H_q=8, N_k=8192, non-causal:**

| B | BH | µs/tok (d64) | **vs SDPA (d64)** | µs/tok (d128) | **vs SDPA (d128)** |
|---|---|---|---|---|---|
| 1  | 8   | 46.22 | 9.93× | 33.52 | 7.81× |
| 8  | 64  |  9.89 | **8.40×** | 16.34 | **7.62×** |
| 16 | 128 | 10.08 | 8.21× | 16.86 | 7.35× |
| 32 | 256 | 12.36 | 6.38× | 19.84 | 6.09× |
| 64 | 512 | 10.91 | 7.32× | 18.32 | 6.63× |

**v7 LOST to SDPA at B≥8 (0.34–0.56×); v8 BEATS it 6.1–9.9× at every batch size** — the serving regime is
reclaimed emphatically. (The `vs no-pack` column hits `nan` at `B=64, d128`: the auxiliary v7-no-packing
baseline — which broadcasts KV to `H_q` heads — errored at the largest shape, likely OOM building the
G-expanded paged pool; v8 itself ran fine, 18.32 µs/tok. Not a v8 failure, a baseline-construction limit.)

**Measured — G=8 canonical decode (H_q=8, H_kv=1):** non-causal `vs no-pack` 1.26–4.25× (grows with
`N_k`: more KV to share → bigger KV-read-once win). Causal `vs no-pack` 1.10–3.95× (valid). Causal
`vs sdpa` (0.39–2.50×) is the **known top-left-mask artifact** carried from v7 (the SDPA reference attends
~1 key while v8 scans the whole cache via `q_offset`) — the correctness test already uses the honest
`causal=False` oracle; a bottom-right-mask reference is the deferred harness cleanup.

**Cut 2 — tensor cores (the GEMV→GEMM step on top of M-packing). Re-staged into 2a/2b:**

*Roofline prediction (recorded BEFORE the kernel runs — the gate).* Tensor cores **do not change the
bytes**, so the model is unchanged: `AI = 2G/b`, limiter still **HBM**, same `G×`-lower floor. The
prediction is therefore *not* a new roofline number but a **schedule** claim the model can't express:
Cut 1 left `%HBM ≤ 11%` (≈9× of headroom) because the M=G score matmul was still a warp-shuffle
GEMV with the opaque-fragment softmax done by hand; replacing it with a 16×16×16 tensor-core GEMM
should **close part of that per-CTA gap → lower µs/tok and higher %HBM at a given G**, with the
*caveat* that at G=8 the WMMA tile is **half-empty** (8 real rows padded to 16), so Cut 2a captures the
GEMV→GEMM direction but not the peak — that's what Cut 2b's `mma.m16n8k16` removes. Predicted ordering:
`Cut 2a > Cut 1` on µs/tok at G≥4 (where the GEMM amortizes the smem-softmax overhead), with the gap
widening toward G=16 (full tile, no padding waste).

**Cut 2a MEASURED (Turing WMMA, sm_75/T4 — `kernels/v8_gqa_tc/`, 2026-06-28, Colab T4): correctness ✅
38/38, but the perf prediction is REFUTED — WMMA tensor cores are SLOWER than Cut 1's CUDA-core GEMV.**
Variant 1 of the M≥16 ablation (pad `M=G→16` + mask), grafting v5's WMMA QK/PV + smem-softmax onto Cut
1's paged split-KV/merge skeleton, one warp per 16-row M-tile. The same-session G-sweep A/B
(N_k=8192, H_q=32, non-causal — note the two runs read different throttle clocks, `tc~360` vs
`cuda~555` MHz, so ~1.5× of the raw gap is clock, flagged below):

| G | tc µs/tok (d128) | Cut 1 µs/tok (d128) | tc÷cuda | tc %HBM | Cut 1 %HBM |
|---|---|---|---|---|---|
| 1  | 468.4 | 142.4 | 3.3× slower | 2.8% |  9.2% |
| 4  |  80.2 |  30.9 | 2.6× slower | 4.1% | 10.6% |
| 8  |  42.1 |  16.9 | 2.5× slower | 3.9% |  9.7% |
| 16 |  27.5 |  13.2 | 2.1× slower | 3.0% |  6.2% |
| 32 |  25.4 |  11.3 | 2.2× slower | 1.6% |  3.6% |

**Prediction-vs-measured — a MISS (recorded honestly):** I predicted `Cut 2a > Cut 1` (the GEMM closes
the per-CTA gap). Measured the **opposite at every G**: µs/tok rose 1.2–3.3× and `%HBM` *fell* (tc
1.6–4.2% vs Cut 1's 7.9–11.1%). It loses even at **G=16/32 where the WMMA tile is full**, so it is NOT
the pad-to-16 waste. Why: v5's GEMV→GEMM win was for **prefill (large M)**; decode's `M=G≤16` is too
small for the 16×16×16 GEMM to amortize the **opaque-fragment tax** (QK→smem→row-softmax→P-as-half→
reload→PV, with extra `__syncthreads`) — the lean register-resident CUDA-core GEMV wins. **Cut 1's
8.6× was about G warps + KV-read-once, NOT tensor cores.** Two caveats keep this honest: (a) the
clock confound inflates the raw gap ~1.5×, but clock-normalized tc is still ~1.4–1.6× slower at G=8
*and* `%HBM` (a clock-robust ratio) is 2–3× lower; (b) Cut 2a is a **correctness-first 1-warp/block**
schedule, so part of the slowdown is its under-fed KV load (32 threads vs Cut 1's 256), not purely the
tensor cores. Reclaim-at-batch (G=8): tc still beats SDPA **2.1–4.8× across B=1→64**, but by less than
Cut 1's 6.1–9.9× — consistent (tc is worse than Cut 1, still > SDPA). The burden of proof flips:
tensor cores must now *earn* a decode role — exactly Cut 2b's job.

**Cut 2b PROBE MEASURED (A100-SXM4-80GB sm_80, 2026-06-28, `notebooks/v8_cut2b_a100_probe_output.ipynb`):
the verdict did NOT flip — WMMA loses HARDER on Ampere, so Cut 2 is CLOSED (CUDA-core GEMV is v8's
decode primitive).** Before writing the hard `mma`+cp.async kernel we probed the *existing* kernels on a
rented A100 (build now targets sm_75+sm_80; both compiled + 38/38 correctness on Ampere). A/B µs/tok
(N_k=8192, H_q=32, non-causal):

| G | tc µs/tok (d128) | Cut 1 µs/tok (d128) | tc÷cuda | T4 was |
|---|---|---|---|---|
| 1  | 263.8 | 57.8 | 4.6× slower | 3.3× |
| 8  |  38.7 |  9.7 | 4.0× slower | 2.5× |
| 16 |  18.9 |  7.7 | 2.5× slower | 2.1× |
| 32 |  18.8 |  7.7 | 2.5× slower | 2.2× |

**The smoking gun (why this is definitive, not a clock fluke):** WMMA **barely moved T4→A100** — G8/d128
42→39 µs/tok — despite A100's ~5× tensor-core throughput and ~6× HBM BW, **while the CUDA-core path nearly
halved** (16.9→9.7). So WMMA is **neither tensor-core- nor bandwidth-bound**; it's pinned by per-CTA
scheduling overhead (the opaque-fragment smem-softmax round-trip + the 1-warp/block load) that doesn't
scale with the GPU. A bigger tensor core can't speed up a kernel that isn't compute-bound. Reclaim-at-batch
seals it: on A100 **Cut 1 beats SDPA 2.5–8.6× across all batch**, while **WMMA loses to SDPA at B≥8
(0.6–0.8×)** — the CUDA-core kernel reclaims the serving regime, the tensor-core one doesn't. (Caveat: the
A100 start-clock readings varied 330–1410 MHz, but the 10-iter warmup boosts before timing and the
*mechanistic* T4→A100 non-scaling is clock-independent — the direction is robust.)

**Decision: Cut 2 is closed; the cp.async/`mma.m16n8k16` kernel and ablation arms 2/3 are NOT pursued.**
cp.async would only fix the load-starvation half of the overhead; the opaque-fragment softmax tax is
fundamental to WMMA attention and the gap *worsened* on the faster GPU. The from-scratch GEMV→GEMM idea
that won for v5 *prefill* is the wrong tool for *decode* — and that's the real, two-architecture result.
**v8's deliverable is Cut 1 (CUDA-core GQA M-packing): 8.6× over no-packing, beats SDPA at all batch, on
both T4 and A100.** (Making tensor cores win decode needs SOTA-level scheduling — FlashMLA/FlashInfer
territory — out of scope for a from-scratch single-kernel and irrelevant to the v8 thesis.)

## Step 8.5 — double-buffered KV pipeline (v8_gqa_db)

*Roofline-first recorded; T4 gate pending (`notebooks/v8_5_gqa_db_gate.ipynb`). The "schedule before
bytes" step v8 proved is still needed — v8 sits at ≤11% HBM (per-CTA-bound), and FP8 (v9) only pays once
bandwidth-bound.*

**Why:** v8 Cut 1's hot loop stalls on every KV tile (`load → __syncthreads → compute → __syncthreads`),
so the global-load latency is **exposed** — the reason it never gets past ~11% HBM. v8.5 software-pipelines
it: prefetch tile `N+1` while computing tile `N`, so the load latency hides behind the warp-shuffle
compute. **Portable** double-buffer (ordinary `ld.global` issued early — NOT `cp.async`, which is
Ampere-only sm_80; the `cp.async` version is an A100 follow-on). To keep two KV buffers within budget
*without* dropping below Cut 1's 2 blocks/SM, KV is staged as **half** (2 half-buffers = Cut 1's one FP32
buffer), so occupancy is unchanged and the **only new variable is the load/compute overlap** (the FP16→FP32
convert just moves from store-time to read-time; same math, same 2e-2).

**Roofline prediction (the gate — schedule claim the model can't express):** `AI=2G/b`, the HBM floor, and
the limiter are **identical to v8** (same bytes). Predicted: `%HBM` **climbs from ~11% toward the floor**
and µs/tok drops — *if* the load latency is the dominant stall (the per-CTA hypothesis). **Counter-prediction
(equally a finding):** if the real stall is the per-key warp-shuffle reduction (the v4 GEMV wall), not the
load, double-buffering won't move `%HBM` — which would point the next lever at the reduction, not the load.

**Measured (2026-06-28, Colab T4, `notebooks/v8_5_gqa_db_gate_output.ipynb`): correctness ✅ 38/38, but the
prediction is REFUTED — double-buffering did NOTHING. The null outcome IS the finding.** Same-session A/B
(G-sweep, N_k=8192, H_q=32, non-causal): `v8_gqa_db` ≈ `v8_gqa` at **every** G — `%HBM` flat within
~0.1–0.8% and µs/tok within noise (and the db run was at a *lower* throttle clock, 360 vs 450 MHz, so the
tiny apparent edge is clock, not the prefetch):

| G (d=128) | db µs/tok | db %HBM | Cut 1 µs/tok | Cut 1 %HBM |
|---|---|---|---|---|
| 2  | 59.0 | 11.1% | 60.3 | 10.9% |
| 8  | 17.0 |  9.6% | 18.5 |  8.8% |
| 32 | 11.3 |  3.6% | 11.5 |  3.5% |

**`%HBM` did not move toward the floor → the decode stall is NOT the exposed KV-load latency. It's the
per-key warp-shuffle reduction (the v4 GEMV wall) + online-softmax dependency chain.** Prefetching the
load can't help because the warps aren't waiting on memory — they're serialized in the reduction, and with
~2 blocks/SM there aren't enough independent warps to hide that *compute* latency. So the kernel is
**compute-latency-bound at ~10% HBM; memory was never the wall.** This closes the decode investigation:
neither tensor cores (Cut 2, opaque-fragment tax) NOR load-overlap (v8.5) move it, because the floor is the
reduction, not bandwidth. **Implication for v9:** FP8 (byte-cutting) won't show a latency win on this
micro-bench either — bytes aren't the bottleneck. To make memory the wall you'd need a faster reduction
(vectorize/ILP/more occupancy — a "v8.6") or a genuinely memory-bound regime (much larger `N_k` past L2);
FP8/NVFP4's real value is then KV-cache *capacity* + accuracy (RMSE vs FP64), not micro-bench speedup.
(Reclaim-at-batch unchanged: db beats SDPA 6–12× across B=1→64, same as Cut 1 — double-buffering breaks
nothing, just doesn't help.)

## Step 8.6 — attack the reduction wall: occupancy vs key-ILP (v8_gqa_occ, v8_gqa_ilp)

*Roofline-first recorded; T4 gate pending (`notebooks/v8_6_reduction_gate.ipynb`). v8.5 isolated the wall
to the **per-key warp-shuffle reduction + serial online-softmax recurrence** (not the KV-load latency).
v8.6 tries to **hide** that compute latency — the only lever left before bytes (v9) can ever pay.*

**Why:** v8.5 proved the decode kernel is **compute-latency-bound at ~10% HBM** — one warp's inner loop
runs a 5-deep `__shfl_xor` butterfly per key (~30–40 cyc exposed) chained into a serial softmax update,
with nothing to overlap it. v8.6 is a **2-arm single-variable ablation** (both CUDA-core / T4, both fork
Cut 1, each changing exactly one thing):
- **Arm 1 — occupancy (`v8_gqa_occ`):** stage KV as **FP16 smem** (16 KB single buffer) → **4 blocks/SM**
  (Cut 1 is 32 KB FP32 → 2). 2× resident warps hide the latency via thread-level parallelism. *(Distinct
  from v8.5, which also used half smem but spent it on a 2nd buffer and stayed at 2 blocks/SM.)*
- **Arm 2 — key-ILP (`v8_gqa_ilp`):** **KU=4-unrolled key loop** — compute 4 independent dot-product
  partials, issue their 4 reductions back-to-back so the independent shfl chains pipeline, then 4
  sequential softmax updates. FP32 smem kept → 2 blocks/SM unchanged (ILP is the only variable).

**Roofline prediction (the gate — the model is BLIND here):** both arms keep identical bytes + FLOPs, so
`AI=2G/b`, the HBM floor, and the limiter are **identical to Cut 1** (verified on the T4 arch: G=8,
N_k=8192, d=128 → **AI=8.0, HBM-bound, floor 0.105 ms**, far below ridge 203). The byte-roofline cannot
distinguish Cut 1 from either arm — so v8.6's prediction is a **scheduling** claim the model can't express,
and the measured µs/tok + %HBM movement *is* the gap between the byte-model and the real warp schedule.

**Mechanistic prediction (the real hypothesis under test):** **Arm 1 (occupancy) is the stronger lever** —
doubling resident warps hides the *entire* serial chain (reduction + exp + softmax update) with TLP,
independent of the recurrence. **Arm 2 (ILP) is weaker** — it overlaps only the independent *reduction*
chains, leaving the serial softmax recurrence (`m_cur/l_cur/o_reg`) exposed. **Counter-prediction (equally a
finding): if BOTH are null → the floor is the serial online-softmax recurrence itself**, which neither
occupancy nor ILP can hide → the real fix is a parallel-across-keys (score-stationary) redesign that
*removes* the per-key reduction (a future "v8.7"), and v9 FP8 stays premature until then.

**Measured (2026-06-28, Colab T4, `notebooks/v8_6_reduction_gate_output.ipynb`): correctness ✅ 190 passed
(both arms + Cut 1 regression), but BOTH levers are NULL — neither moved `%HBM` off ~10%. The
counter-prediction landed: the floor is the serial online-softmax recurrence.**

⚠️ The six bench runs read wildly different SM clocks (`360/435/540` MHz in the G-sweep, `480/1590/1590` in
reclaim), so **µs/tok is not comparable across backends**; the clock-robust read is `%HBM`. G-sweep d=128:

| G (d=128) | Cut 1 %HBM (360MHz) | occ %HBM (435MHz) | ilp %HBM (540MHz) |
|---|---|---|---|
| 2  | 11.2% | 10.8% | 10.9% |
| 8  |  9.6% |  9.7% |  9.9% |
| 32 |  3.6% |  3.1% |  3.4% |

`%HBM` is flat within ~0.5% at every G — and the same in reclaim (~8–10% across all three at every batch).
Two things make it airtight rather than "no signal": (1) **occupancy never engaged at small batch** — at
BH=32/G=8 split-KV already emits ~80 partial blocks = exactly 2 blocks/SM on 40 SMs, so Arm 1's 4-block
*ceiling* has no 3rd/4th block to schedule; the one place spare blocks exist (reclaim B=64, d=64) shows occ
**8.2% vs Cut 1's 7.3%** — a real but ~1% nudge, nowhere near the floor. (2) **ILP tracks Cut 1's %HBM
almost exactly** (reclaim B=32/64 d=64: ilp 6.5%/7.4% vs Cut 1 6.5%/7.3%) — pipelining the reductions bought
nothing. **Both null → the floor is the per-row serial recurrence** (`m_cur/l_cur/o_reg`, dependent on the
previous key): neither more warps (TLP) nor more in-flight reductions (ILP) can hide a chain that is
inherently sequential within a row. This is the **fourth** consecutive negative (Cut 2 / v8.5 / v8.6 occ /
v8.6 ilp), all converging: the decode kernel is pinned at ~10% HBM by the serial inner loop, not by memory,
compute throughput, load latency, or reduction latency. **Mandate for v8.7:** *remove* the per-key reduction
+ shorten the recurrence with a score-stationary relayout — the only decode-side lever left before bytes.

## Step 8.7 — score-stationary decode inner loop (v8_gqa_ss)

*Roofline-first recorded; T4 gate pending (`notebooks/v8_7_score_stationary_gate.ipynb`). v8.6's fourth
negative mandated this: stop hiding the per-key reduction wall and REMOVE it.*

**Why:** every prior decode cut kept Cut 1's output-stationary GEMV inner loop — one warp owns one query
row, 32 lanes split head-dim, and per key it pays a 5-deep `__shfl_xor` butterfly + a serial
`(m,l,O)` update. v8.6 proved that latency is unhideable. v8.7 changes the **inner-loop layout** (the single
variable): **lane = key** — lane `l` computes the *full* dot product q·k_c, so the score lives in its own
register with **no per-key cross-lane reduction**; softmax runs **once per 32-key group** (one
`warp_reduce_max` + one `warp_reduce_sum`), shortening the serial recurrence **32×**; and PV becomes a
transpose `O[d]=Σ_c p_c·V[c][d]` done with single-hop `__shfl` broadcasts of `p_c` that pipeline (vs Cut 1's
serial chain). The layout inverts Cut 1: QK is now reduction-free, the cross-lane traffic moves to the
2-per-group softmax reductions + the PV broadcast fan.

**smem held FP16 (the locked decision):** Cut 1's FP32 smem + the new `sQ` + a bank-conflict pad would drop
the T4 to 1 block/SM and confound the layout variable; staging K/V/Q as FP16 (~18 KB → ~3 blocks/SM) holds
occupancy (v8.6 measured FP16 smem is correctness-safe + perf-neutral). K is staged **transposed `[d][key]`
+ 1-pad** so the lane=key read is bank-conflict-free; V natural `[key][d]`; Q per-warp in `sQ`. This makes
the inner-loop **layout** the only difference vs `v8_gqa_occ` (also FP16-smem GEMV) → a clean layout-only A/B.

**Roofline prediction (the gate — still BLIND):** `AI=2G/b`, the HBM floor, and the limiter are **identical
to Cut 1** (the byte-model can't see the inner-loop layout; G=8 → AI=8.0, HBM-bound, 0.105 ms). The
deliverable is the **measured µs/tok**, not a roofline shift.

**Mechanistic prediction (recorded before the run):** v8.7 should finally **drop µs/tok** — strongest at
**d=64**. **d=128 is at risk** (R2): QK now reads the full D from `sK` per key (more smem-read traffic than
Cut 1's EPT), so d=128 could flip to **smem-bandwidth-bound** and show a smaller win or a null. **Counter-
prediction (also a finding): if µs/tok drops but %HBM still sits at ~10%**, the new floor is per-CTA **load**
latency, not compute — which finally makes **v9 FP8 (capacity + fewer bytes)** the right next lever. Either
outcome closes the decode-schedule investigation.

**Measured:** *pending T4 run-of-record (`notebooks/v8_7_score_stationary_gate.ipynb`).* Fill the 3-way A/B
(`v8_gqa` Cut 1, `v8_gqa_occ` layout-isolated, `v8_gqa_ss`) µs/tok + %HBM per G + reclaim-at-batch, then
state whether removing the reduction moved the floor (and at which d), and whether %HBM finally climbed.

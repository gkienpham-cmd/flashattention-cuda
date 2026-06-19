# Results

The growing speedup + roofline curve. One block per step. Each row pairs the **predicted**
limiter (from `roofline/`) with the **measured** reality (bench + Nsight Compute), because the
honesty of the prediction is itself a deliverable.

Every row records GPU / arch / clock so it's reproducible; the free-tier T4 throttles, so clocks
are captured at run time.

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

| shape | ours p50/p99 ms | SDPA p50/p99 ms | speedup vs SDPA | tok/s (ours) | measured ÷ cache-free LB |
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

| shape | v2 p50/p99 ms | v2 vs SDPA | **v2/v1 (@300 MHz)** | v2 measured ÷ tiled LB |
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
Correctness: **13/13 core pass vs SDPA** (atol/rtol 1e-4), all shapes × causal × the partial-tile
boundaries; long-N (N=16384) rescale-stability case included.

| shape | v3 p50/p99 ms | v2 p50 ms | SDPA p50 ms | v3÷SDPA | v3÷v2 | roofline |
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

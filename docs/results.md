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

**Reading it — two confirmations and one honest miss:**
- ✅ **Limiter stays HBM** at every shape, exactly as predicted — tiling does not cross the ridge.
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

**ncu:** `profiling/capture.sh v2_tiled` now correctly profiles `qk_tiled_kernel<64,64,64>` /
`softmax_kernel` / `pv_tiled_kernel<64,64,64>` (confirmed in the run log), wrote
`profiling/raw/v2_tiled.ncu-rep`. **TODO (UI read):** open both reports and record `dram__bytes_read`
for v1 vs v2 — the operand-reuse win should show as a large drop in the QK/PV passes' DRAM reads,
and the S-round-trip cap should show as a *floor* on total DRAM traffic that tiling can't push below.

How to fill the ncu row: open `profiling/raw/{v1_naive,v2_tiled}.ncu-rep` in Nsight Compute,
read `dram__bytes_read.sum` per pass per `GUIDE.md`, and note whether the measured traffic-ratio
matches the predicted ~30× operand cut net of the S floor.

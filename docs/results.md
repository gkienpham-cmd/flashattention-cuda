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
kernel, not qk/softmax/pv — needs a `--kernel-name` filter (TODO before the per-step ncu row is
real). Harness nit: bench logs `clock~0MHz` (nvidia-smi clock parse) though the GPU reported
300/1590 MHz — fix the parse so throttling is recorded.

How to fill this in: run `notebooks/colab_bootstrap.ipynb` (bench + ncu cells), paste the
numbers, and note whether prediction and measurement agree.

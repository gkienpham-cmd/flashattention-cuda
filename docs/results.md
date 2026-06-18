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

**Measured (fill in from Colab):**

| shape | ours p50/p99 ms | SDPA p50/p99 ms | speedup vs SDPA | tok/s (ours) | ncu limiter | occupancy | mem tput | MMA util | MUFU util |
|---|---|---|---|---|---|---|---|---|---|
| 1x8x512x64   | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 1x8x2048x64  | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 1x8x8192x64  | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 1x8x2048x128 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

> Expectation for Step 1: we will be **much slower than SDPA** (SDPA already uses a fused
> FlashAttention/efficient kernel). That's the point — this is the "before." We expect the ncu
> reading to confirm **DRAM/memory throughput near roofline at both d=64 and d=128** with **low
> MMA + low MUFU utilization**, matching the AI~0.25 HBM-bound prediction. The "no tensor cores"
> weakness does *not* show here — it only surfaces once a fused kernel (Step 3+) reaches AI~512
> and becomes MMA-bound on the FP32 cores; see decisions.md.

How to fill this in: run `notebooks/colab_bootstrap.ipynb` (bench + ncu cells), paste the
numbers, and note whether prediction and measurement agree.

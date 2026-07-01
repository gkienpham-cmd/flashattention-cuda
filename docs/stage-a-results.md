# Stage A result — native FP4 compute does NOT help MLA decode (KILL)

**Measured 2026-07-02, vast.ai B300 SXM6 (sm_103a), CUDA 12.9, CUTLASS v4.5.2.** Data of record:
[`notebooks/stage_a_results.csv`](../notebooks/stage_a_results.csv) (30/30 rows) + the executed
[`notebooks/stage_a_gemm_profiler.ipynb`](../notebooks/stage_a_gemm_profiler.ipynb). This is the
kill-or-proceed micro-test for paper **§5.3**.

## Question & pre-registered prediction

Does native FP4 tensor-core compute beat FP8 at the MLA-decode QK-GEMM shape (M=128, all h_q query
heads packed; K∈{256,512,576}; N = context length)? **Pre-registered: NO** — decode is HBM/work-
starved, not compute-bound, so FP4's 3× compute peak can't convert. (PV can't be FP4 — softmax
collapses it, v10 — so only QK is testable.)

## Verdict: KILL — prediction confirmed

**Neither precision approaches its compute peak.** Across the whole sweep FP4 tops out at **5.8% of
15 PF** and FP8 at **21.7% of 5 PF**; every shape is HBM-bound (AI 100–176 ≪ ridges 1875/625). So
native FP4 *compute* is never the operative limit → it cannot accelerate MLA decode. The negative
goes into §5.3. Consistent with the whole-kernel v6→v12 finding (per-CTA / work-starvation).

### The FP4/FP8 raw ratio flips with N — and neither direction is a compute win

| K | N | FP4 TFLOP/s (72a) | FP8 TFLOP/s (cuBLAS) | FP4/FP8 | winner |
|---|---|---|---|---|---|
| 256 | 1024 | 9.1 | 3.3 | 2.75 | FP4 |
| 256 | 8192 | 71.5 | 33.2 | 2.15 | FP4 |
| 256 | 32768 | 184.9 | 134.1 | 1.38 | FP4 |
| 256 | 131072 | 381.4 | 549.1 | 0.69 | **FP8** |
| 256 | 524288 | 485.9 | 741.2 | 0.66 | **FP8** |
| 512 | 131072 | 710.9 | 982.7 | 0.72 | **FP8** |
| 512 | 524288 | 871.6 | 1055.6 | 0.83 | **FP8** |
| 576 | 131072 | 738.7 | 1086.0 | 0.68 | **FP8** |
| 576 | 524288 | 792.3 | 1041.7 | 0.76 | **FP8** |

(Full 15-shape table in the CSV.) Two regimes:
- **Small N (≤32K):** FP4 leads 1.4–2.7×, but both are tiny (<200 TFLOP/s) — latency/launch-bound
  small GEMMs, not a compute statement.
- **Large N (≥131K, the realistic long-context decode regime): FP8 wins** (up to ~1.5×). cuBLAS FP8
  sits right at its HBM roofline (~AI·8 TB/s ≈ 1080 TFLOP/s @ AI≈135); the FP4 example kernel runs
  *below* its own (higher) HBM ceiling.

**Why "not a compute win" either way:** FP4's higher arithmetic intensity (fewer operand bytes,
0.5625 vs 1.0 B/elem) gives it a *higher* HBM ceiling than FP8, so a fully-tuned FP4 kernel could
in principle exceed FP8 at large N — but that would be a **bandwidth / operand-size** advantage
(the storage lever, already the v9/v10 story), **not** the tensor-core compute peak. The compute
peak (15 PF) is never reached, so the Stage-A question is settled negative.

## Honest caveats (so the claim isn't overstated)

1. **Cross-harness comparison.** FP4 = CUTLASS **example** 72a (block-scaled NVFP4→bf16, an example
   kernel, not a tuned production GEMM); FP8 = **cuBLAS** dense E4M3 via `torch._scaled_mm` (tuned).
   So the head-to-head ratio is *not* a clean kernel-vs-kernel result. The KILL rests on the
   **peak-fraction + roofline** argument (both kernel-quality-independent: AI ≪ ridge caps FP4 far
   below compute peak regardless of tuning), not on the ratio. Fixing the profiler build (it OOM/
   compile-failed on the whole multi-arch library) or building a matched MXFP8 example would make
   the head-to-head clean — optional hardening, does not change the verdict.
2. **Format asymmetry.** FP8 here is dense per-tensor E4M3; FP4 is block-scaled NVFP4. Fine for the
   peak-compute question (each at its native peak), but not a like-for-like block-scaled comparison.
3. **Single seed, clocks not locked.** The verdict is a peak-fraction, far from any clock-sensitivity
   threshold, so this doesn't move it; noted for completeness.

## Paper wording (§5.3)

> At the M=128 MLA-decode QK-GEMM shape, both NVFP4 and FP8 GEMMs are HBM-bound across all context
> lengths (arithmetic intensity 100–176 FLOP/byte, far below the sm_103 ridges of 1875 / 625),
> reaching at most 5.8% / 21.7% of their respective tensor-core peaks. Native FP4 *compute* is
> therefore never the bottleneck and does not accelerate decode; the M=128 packing advantage is a
> red herring for the decode regime. Any FP4-vs-FP8 throughput gap we observe is a bandwidth /
> operand-size effect (the KV-capacity/storage lever of §5.1–5.2), not a compute win.

**Decision:** Stage B/C (accuracy) is **not** unlocked by a compute win — there is none. FP4/NVFP4
remains a capacity + accuracy lever (v9/v10), not a decode-compute lever.

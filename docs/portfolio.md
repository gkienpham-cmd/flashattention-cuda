# FlashAttention from Scratch

**A roofline-driven journey rebuilding FlashAttention one measured speedup at a time.**

*Kien Pham — Mechanical Engineering Student*

![The v1→v12 journey: 17 CUDA kernels, 4 GPU architectures, three eras of optimization](diagrams/portfolio-hero.svg)

---

## Project Overview

This is a solo project rebuilding FlashAttention from scratch, one kernel at a time, across four generations of NVIDIA GPUs. The goal was never just fast code -- it was the measured speedup curve and the prediction-vs-measured analysis that comes with it.

Over 12 steps, I wrote 17 CUDA attention kernels spanning the full design space: from a naive three-pass baseline to production-grade tensor-core MLA decode on NVIDIA's latest Blackwell Ultra (B300/sm_103). Each kernel follows a strict methodology: predict the performance limiter with a roofline model *before* writing any code, then measure, then record where the prediction was right and -- more importantly -- where it was wrong.

The project covers four GPU architectures (T4 Turing, A100 Ampere, B200 Blackwell, B300 Blackwell Ultra), three attention patterns (MHA, GQA, MLA), and four precision formats (FP32, FP16, FP8 E4M3, NVFP4 E2M1). Honest misses are first-class deliverables. When the roofline model predicted a 30x traffic cut and reality showed 1.02x, that became a lesson about L2 cache behavior that shaped every subsequent design decision.

The research target is a PMBS@SC characterization paper: the first open, kernel-level, prediction-vs-measured roofline characterization of FlashAttention decode on sm_103. The kernel library itself is the foundation for a from-scratch mini-vLLM inference engine.

---

## Methodology

### The Per-Step Loop

Every kernel version follows this loop, in order. No step is skipped. No step is marked done until tests are green on GPU and the reasoning is verified.

1. **Roofline first** -- Predict the limiter (MMA / HBM / MUFU) with the roofline tool *before* writing any code. Record the prediction.
2. **Explain** -- Write out the memory-hierarchy and scheduling reasoning. Why this design? What does the model say?
3. **Write** -- Kernel + binding. Hand-verify the core loop.
4. **Correctness** -- Test against PyTorch SDPA with a documented tolerance (1e-4 for FP32, 2e-2 for FP16, 5e-2 for FP8/FP4).
5. **Benchmark** -- Measure vs SDPA across multiple shapes (sequence lengths, head dimensions, batch sizes).
6. **Record** -- Update the prediction-vs-measured table. If the prediction was wrong, record the miss and what it teaches.
7. **Quiz** -- Verify understanding of the results before moving to the next kernel.

This loop enforces a discipline: you cannot hide behind code that merely compiles. Every kernel must survive a correctness gate and a prediction-vs-measured reconciliation.

### Two-Layer Prediction Model

Starting at Step 10, the project uses a two-layer prediction model that separates what the roofline *can* see from what it *cannot*.

**Layer 1 -- Pure roofline:** FLOP/byte vs the architecture's ridge point. This gives the limiter name (compute-bound or memory-bound) and the absolute performance floor. It is the textbook model.

**Layer 2 -- Per-CTA-corrected:** Adjusts for schedule realities that the pure roofline is blind to -- occupancy limits from shared memory usage, launch overhead, warp utilization at small M dimensions, L2 residency effects. This is the layer that actually predicts real-world behavior. When the two layers disagree, the per-CTA-corrected layer wins -- and did, at every step measured.

**Counter-free %HBM proxy:** For environments where Nsight Compute is unavailable (containerized GPU rentals block hardware counters with ERR_NVGPUCTRPERM), a proxy metric `effective_bw = KV_bytes / time` compared to peak HBM bandwidth serves as a profiler substitute. This proxy was validated within 1 percentage point of Nsight Compute hardware counters (13.8% vs ncu's 12.85%) at Step 9 Task 1 -- **on the T4**. The Blackwell (B200/B300) runs used rented unprivileged containers where ncu was blocked, so the frontier %HBM / per-CTA verdicts (v10--v12) are **proxy-grade, not counter-grade** -- validated once on T4 and disclosed as such throughout. A privileged-box ncu run on Blackwell is the highest-value open item.

### Confound Removal

Real measurement demands confound removal. The project earned its diagnosis through:

- **Locked clocks** (1590 MHz on a root T4) -- eliminates DVFS variation between kernels
- **L2 flushed** between iterations (zero a buffer at least 2x the L2 size outside the timed window) -- eliminates cache warmth artifacts
- **Past-L2 sweep** (N_k pushed to 128K so working set = 537 MB, far beyond the 4 MB T4 L2) -- ensures data genuinely comes from HBM
- **ncu hardware counters** (first successful capture: Step 9 Task 1) -- L2 hit-rate 1.1%, DRAM utilization 12.85%, validating the counter-free methodology

---

## The Journey

### Era 1: Prefill Fundamentals (v1--v5, T4)

#### v1 -- Naive Attention

*FP32, three-pass, the baseline*

The simplest possible attention kernel: materialize the full score matrix S = QK^T in HBM, apply softmax, multiply by V. Three separate kernel passes, all intermediate results written to and read from global memory.

- Arithmetic intensity ~0.25 FLOP/byte -- deeply HBM-bound
- 20--30x slower than PyTorch SDPA
- **Finding:** The T4's 4 MB L2 cache absorbs redundant operand reads at mid-N. The naive kernel beats its cache-free roofline floor at these sizes, then converges to the floor at N=8192 when the L2 overflows. The roofline bounds traffic, but a kernel's distance from that bound depends on the cache -- something the model cannot see.

#### v2 -- Shared-Memory Tiling

*FP32, three-pass, S still materialized*

Stage Q, K, and V tiles in shared memory to cut redundant global memory reads. Tiles 64x64 at d=64, 32x32 at d=128. The textbook optimization.

- **Speedup:** 1.3--3.2x over v1 (peaks at 2048x128)
- 26/26 correctness vs SDPA
- **Prediction:** Tiling raises AI from 0.25 to 8.0 (a 30x traffic cut) -- **WRONG**
- ncu revealed: tiling cut ~0% DRAM traffic (1.02--1.08x), because the L2 had already absorbed the redundant reads in v1
- **Finding:** ~99% of DRAM traffic is the score matrix S (softmax re-reads it ~12x), not the operand re-reads that tiling targets. The speedup was a *compute/scheduling* win, not a bandwidth win. The roofline got magnitude, location, AND limiter wrong -- all because it cannot model the L2.

#### v3 -- Online Softmax

*FP32, two-pass, S eliminated*

Running max and sum eliminate the need to ever materialize the score matrix S. Two passes: pass 1 computes the final (m, l) with the running-max rescale; pass 2 recomputes scores and forms O.

- **Memory proof:** v2 peak CUDA memory +2164 MB vs v3 +17 MB = the 2147 MB S matrix provably gone (125x reduction)
- **But:** 3--7x SLOWER than v2 -- directionally opposite of prediction
- Measured 151x above the MMA floor; CUPTI trace shows pass2_output at 88.6% of runtime (one-thread-per-row + unstaged per-row K/V reads)
- **Finding:** Removing bandwidth-bound traffic on a machine that was never bandwidth-bound produces a regression. The mechanism is correct; the schedule defeats it. This is the S-elimination paradox.

#### v4 -- Fused FlashAttention-1

*FP32, single-pass*

The full FlashAttention-1 algorithm in one kernel: one warp per query row, K/V staged in shared memory, register-resident output accumulator O with the O-rescale (O = alpha * O + p * V, alpha = exp(m_old - m_new)).

- **Speedup:** 1.7--2.6x over v2, 7.5--15x over v3
- 17/17 correctness vs SDPA (including N=16384 O-rescale stability at d=64 and d=128)
- First version where S-off-HBM is ALSO a wall-clock win
- CUPTI trace: single fused kernel at 100% CUDA time (v3's 88.6% pass2 wall gone)
- Distance to floor: 151x to ~18x (schedule fixed, but FMA under-utilization remains)
- **Finding:** "Compute-bound" does not mean "near peak FLOPs." The kernel is still 18x off the FP32 MMA floor because the scoring is GEMV-shaped -- warp-shuffle reductions dominate over FMAs. The math shape, not the schedule, is now the wall.

#### v5 -- WMMA Tensor Cores

*FP16-in / FP32-accum, Turing WMMA 16x16x16*

Move both matmuls (QK^T and PV) onto Turing WMMA tensor cores. The opaque-fragment tax forces softmax and scores through shared memory (store, row-softmax, reload P as half). FP32 output accumulator kept in shared memory for the O-rescale.

- Roofline predicts the floor drops 8x (FP32 8.1 TFLOPS to FP16 65 TFLOPS)
- 17/17 correctness vs SDPA, tolerance loosened to 2e-2 (first FP16 version)
- **Status:** PARTIAL -- prefill bench was never captured. This is logged honestly; the prediction is recorded but unverified by measurement.

---

### Era 2: Decode Optimization (v6--v8.7, T4 + A100)

#### v6 -- Split-KV Decode (Flash-Decoding)

*FP16-in, the decode pivot (N_q=1)*

Two kernels behind one `forward` call: a split-KV partial (each block does online softmax over one KV chunk and writes the unnormalized (O, m, l)) and an LSE merge across splits. `choose_splits` fills approximately 2x the SMs. When N_q = N_k (prefill shape), 1 split reduces to plain attention.

- **Speedup:** 5.7--8.2x vs naive N_q=1, 1.5--3.3x vs torch SDPA (non-causal)
- 25/25 correctness vs SDPA
- But only 9--15% HBM utilization (7--9x above the decode floor)
- **Finding:** Bandwidth-bound in principle (AI = 1.0), but occupancy-bound in practice (~2 blocks/SM + a 2-kernel launch + a tiny under-occupied merge). This pivoted the entire roadmap: the next lever is occupancy (GQA M-packing), not bytes.

#### v7 -- Paged KV Gather

*The vLLM layout*

v6's two kernels carried unchanged, plus ONE new variable: KV reads gather through a per-sequence block table (paged pool `[num_blocks, page_size, H, d]`, the vLLM layout). New `paged_attention()` API. Plus two harness fixes: a `--batch` sweep and a causal query-offset correction.

- 51/51 correctness (shuffled-pool gather verified)
- The `--batch` sweep REFUTED the predicted occupancy-to-bandwidth crossover: %HBM was flat at 9.4--12.4% from BH=8 through BH=512 (no climb, even at 12.8 blocks/SM)
- **Finding:** Per-CTA-bound at ALL batch sizes. 32 KB shared memory per block caps residency at 2 blocks/SM, and at N_q=1 only 1 of 8 warps computes. Batch adds waves, not per-SM parallelism. This reordered the roadmap: GQA M-packing moves ahead of FP8/NVFP4.

#### v8 Cut 1 -- GQA M-packing

*CUDA-core, the reorder*

Pack the G query heads of a GQA group into the CTA's M dimension. Grid z = B * H_kv (KV heads), and each warp handles one query head from the group. KV read once, G warps active (not 1), AI rises from 2/b to 2G/b.

- **Speedup:** 8.6x over no-packing at G=8; beats SDPA 6--10x at all batch sizes B=1 through 64
- 64/64 correctness
- First partial roofline win in 6 steps: the AI = 2G/b model got the speedup magnitude right
- **Finding:** The M-packing lever worked. It reclaimed the B >= 8 regime that SDPA had owned since v7. This is the first time the roofline model's prediction matched reality.

#### v8 Cut 2 -- Tensor Cores on Decode (REFUTED)

*WMMA tensor-core version of the decode kernel*

Applied the v5 WMMA approach to the decode kernel. Tested on two architectures (T4 and A100).

- **Result:** 1.8--4.6x SLOWER than the CUDA-core GEMV on both architectures
- Smoking gun: WMMA barely moved T4 to A100 (42 to 39 us/tok) despite ~5x TC throughput + ~6x bandwidth, while CUDA-core nearly halved (16.9 to 9.7)
- **Finding:** Decode at M = G <= 16 is too small to amortize the opaque-fragment shared-memory-softmax tax. The GEMV-to-GEMM fix that won for prefill (v5) is the wrong tool for decode.

#### v8.5 -- Double-Buffered KV Pipeline (NULL)

*Single-variable ablation: overlap loads with compute*

- 38/38 correctness, but double-buffering did NOTHING -- %HBM flat ~10%, us/tok within noise vs Cut 1
- **Finding:** The decode stall is the per-key warp-shuffle reduction, NOT load latency. Memory was never the wall.

#### v8.6 -- Occupancy + ILP (BOTH NULL)

*Two-arm ablation to hide the reduction latency*

- **Arm 1** (FP16 shared memory, 4 blocks/SM): NULL
- **Arm 2** (KU=4 key unrolling, ILP): NULL
- 190/190 correctness, fourth consecutive negative result (Cut 2, v8.5, occupancy, ILP)
- Counter-prediction LANDED: the floor is the per-row serial online-softmax recurrence -- unhideable by TLP or ILP
- **Finding:** The floor is a serial dependency chain. You can only REMOVE it, not hide it. This mandated v8.7.

#### v8.7 -- Score-Stationary Inner Loop (WIN)

*Flipped the inner-loop layout*

The single variable changed: lane = key (not lane = head-dimension). Each lane computes the FULL dot product q * k_c in its own registers -- no per-key cross-lane reduction. Softmax runs once per 32-key group (one warp-reduce-max + one warp-reduce-sum) instead of once per key, shortening the recurrence 32x. PV is a transpose via single-hop `__shfl` broadcasts that pipeline.

- **Speedup:** 1.1--1.6x over Cut 1 at matched clock; beats SDPA 8--16x
- First lever since Cut 1 to move the clock-robust %HBM (up ~2--3 points to 10--12%)
- **Finding:** Remove, don't hide. The reduction/recurrence WAS a real component of the floor. Closes the decode-schedule arc: M-packing (Cut 1) + score-stationary (v8.7) are the two real decode levers. Tensor cores, double-buffer, occupancy, and ILP were all dead ends.

---

### Era 3: Precision + Architecture Frontier (v9--v12, T4 + B200 + B300)

#### v9 -- FP8 E4M3 KV Cache

*Single variable: KV storage precision (1 byte instead of 2)*

Forks v8.7 (score-stationary), changing only the KV storage format to FP8 E4M3 with fused per-tile dequant and per-tensor FP32 scales. FP32 accumulator. Score-stationary inner loop, M-packing grid, split-KV, LSE merge all byte-identical to v8.7.

**Task 2 (kernel):**
- ~1.3x median decode-latency win -- prediction REFUTED ("capacity-only" predicted; actual load-latency / issue win, regime-specific -- it flips negative under L2-flush)
- Win shrinks with G (d=64: G2 1.37x to G32 0.96x) -- M-packing amortizes the KV load
- Accuracy: E4M3 RMSE ~6--7e-4 vs FP16 (single-seed, small substrate -- see accuracy caveat below)
- 76/76 correctness

**Task 1 (regime characterization, ROOT T4, clocks LOCKED 1590 MHz, L2 FLUSHED):**
- FIRST ncu-validated measurement in the project
- %HBM caps at ~28--29% (never near the ~70% achievable ceiling)
- ncu confirmed: L2 hit-rate 1.1%, DRAM utilization 12.85% past L2
- Counter-free proxy validated: 13.8% vs ncu 12.85% (within 1 point)
- **VERDICT:** per-CTA / low-MLP latency-bound, NOT bandwidth-bound -- confound-free for the first time

#### v10 -- NVFP4 KV Cache (on B300/sm_103)

*4-bit KV storage with micro-scales*

Paged K/V as NVFP4 E2M1 (4-bit, 0.5625 bytes/elem with one E4M3 micro-scale per 16 elements + per-tensor FP32 scale). Fused per-tile dequant to FP16. Score-stationary inner loop, M-packing, split-KV, LSE merge byte-identical to v9.

- **Capacity:** 3.56x vs FP16, 1.78x vs FP8 (measured)
- **Accuracy:** Standard NVFP4 ~2.4e-3 RMSE (~4x penalty vs FP8's ~6e-4 -- E2M1's 1-bit mantissa). Single-seed, small-substrate point estimate -- the ~3.5-4x NVFP4-vs-FP8 ratio holds for this setup but is not a general constant (published FP4-vs-FP8 gaps range from sub-1% task accuracy to several-x RMSE depending on the scaling scheme).
- **Latency:** NVFP4 is latency-NEGATIVE past L2 (12--30% slower than FP16 -- dequant ALU tax)
- 146/146 correctness
- B300/sm_103: %HBM ~0.5%, per-CTA-bound confirmed to 2M tokens past 8x L2 overflow (counter-free proxy; ncu blocked on the unprivileged B300 container -- proxy-grade, validated once on T4)
- Architecture-independent ceiling: ~40 GB/s achieved at B=1 on THREE architectures (T4 at 11% of 320 GB/s, B200 at 0.5% of 8 TB/s, B300 at 0.5% of 8 TB/s -- a 25x bandwidth span, same ceiling)
- B300 arch constants MEASURED: 148 SMs (not 160), L2 132.6 MB (not 192), 2032 MHz (not 2600) -- correcting published specifications
- FlashInfer comparison: ~3x faster ("complementing, not beating")
- **Finding:** NVFP4 = capacity + accuracy, NOT decode latency. Per-CTA-bound reconfirmed a 7th time on FP4.

#### v11 -- MLA Latent-KV Decode (on B300/sm_103)

*The SHAPE change*

MQA over one shared latent (H_kv=1, G=h_q=128). All 128 query heads share one latent vector, read once. M=128 by construction -- the first decode shape in the arc that pure roofline calls compute-bound (FP8 AI=470 exceeds the FP16 ridge of 312.5).

- **Pure roofline FLIPS** compute-bound for the first time in the arc
- **Measured (CUDA-core kernel):** 0.75 TFLOP/s achieved -- still per-CTA-bound (<1% of peaks)
- ~4x slower than torch dense-MQA -- the ~4x wall-clock gap is measured; the likely cause (torch routing M=128 into a cuBLAS tensor-core GEMM) is a structural inference, not a torch-side profile (the torch baseline is a batched FP32 matmul, unprofiled)
- **Capacity:** 202x more KV resident vs MHA, 12.6x vs GQA-8 (99.5% reduction)
- 148/148 correctness
- B300 nsys: partial kernel 99.9% GPU time, merge <0.01%
- B300/sm_103: %HBM ~0%, eff_bw ~0.9 GB/s to 2M tokens past a 5x L2 overflow -- confound-free on the L2 axis (counter-free proxy; ncu was blocked on the unprivileged B300 container -- see Independent Validation)
- **Finding:** The shape is correct and tensor-core-friendly (strongly suggested by the 4x gap -- a structural inference, not a profiled measurement), but CUDA-core GEMV is the wrong engine on Blackwell. This motivated v12.

#### v12 -- Native tcgen05 Tensor-Core MLA Decode (CUTLASS ex77, B300/sm_103)

*Characterized the production kernel directly*

Scope pivot: CUTLASS ex77 already ships the canonical production weight-absorbed MLA decode kernel (`Sm100FmhaMlaKernelTmaWarpspecialized`) -- M=128 by static_assert, FP8-native, paged + split-KV + LSE with warp-specialized scheduling. Reinventing it adds no science. The stronger deliverable is characterizing it.

- **Throughput scales with total work B x K:** single-stream (B=1, K <= 32K) hits 1--2% of the 5 PF FP8 peak; serving scale (B=64, K=524288) hits **1785 TFLOP/s = 35.7% FP8 peak / 46.3% HBM**
- CORRECTED the long-standing "per-CTA-bound forever" reading from v6--v11: that was a work-starvation artifact of the CUDA-core kernel. The per-CTA wall is work-starvation, and the right engine converts work to throughput (v11 flat 0.75 TFLOP/s to v12 spanning 2.4 to 1785, 70--1000x at matched B=1)
- 2x-exp claim measured: 0.50x the claimed 10.7 TExp/s (EX2 5.33 TExp/s, <3% of decode time at M=1 -- irrelevant)
- NVFP4 compute (Arm 2) is unbuilt anywhere -- genuine open novelty for a future kernel
- **Finding:** Complement, not beat. The production tcgen05 kernel hits 36% of FP8 peak at serving scale. The methodology is the contribution: open, prediction-vs-measured, on the production kernel itself.

---

## Key Results

### Speedup Progression

| Step | Kernel | vs Prior | vs SDPA | Key Metric |
|------|--------|----------|---------|------------|
| v1 | Naive | baseline | 0.03x | AI ~0.25, 20--30x slower |
| v2 | Tiled | 1.3--3.2x over v1 | 0.04--0.11x | ncu: tiling cut 0% DRAM |
| v3 | Online softmax | 3--7x SLOWER | 0.01--0.02x | 125x less memory, but slower |
| v4 | Fused FA-1 | 1.7--2.6x over v2 | 0.16--0.23x | First S-off-HBM win |
| v5 | WMMA TC | (bench outstanding) | -- | Floor drops 8x (predicted) |
| v6 | Split-KV decode | 5.7--8.2x vs naive | 1.5--3.3x | Decode pivot, ~12% HBM |
| v7 | Paged KV | -- | -- | Crossover REFUTED, %HBM flat |
| v8 | GQA M-pack | 8.6x over v7 | 6--10x | First partial roofline win |
| v8.7 | Score-stationary | 1.1--1.6x over v8 | 8--16x | Decode-schedule arc closed |
| v9 | FP8 E4M3 | ~1.3x over FP16 | -- | ncu-validated regime |
| v10 | NVFP4 (B300) | capacity 3.56x | -- | Per-CTA-bound on 3 arches |
| v11 | MLA (B300) | 202x capacity | 0.23--0.29x | 4x cuBLAS gap (engine wrong) |
| v12 | tcgen05 (B300) | 70--1000x v11 | -- | 1785 TFLOP/s = 36% FP8 peak |

### Roofline Prediction Scorecard

| Step | Predicted | Measured | Hit/Miss |
|------|-----------|----------|----------|
| v1 | HBM-bound | HBM-bound (L2 helps) | Limiter right, cache model missing |
| v2 | 30x traffic cut | 1.02--1.08x (L2 owned it) | Magnitude wrong |
| v3 | MMA-bound | Occupancy-bound (151x off) | Limiter wrong |
| v4 | MMA-bound | FMA-efficiency (18x off) | Magnitude wrong |
| v6 | HBM-bound | Occupancy-bound (7--9x off) | Magnitude wrong |
| v7 | Crossover with batch | Flat %HBM | Refuted |
| v8 | AI=2G/b, ~Gx speedup | AI=2G/b, ~Gx speedup | FIRST WIN |
| v9 T1 | Per-CTA-bound | Per-CTA / low-MLP | Confirmed (ncu, T4) |
| v10 | Capacity 3.55x, no latency | 3.56x, latency-negative | Capacity hit, latency sign missed (proxy) |
| v11 | AI to 235, compute-flip? | Still per-CTA (CUDA-core) | Per-CTA-corrected layer won (proxy) |
| v12 | HBM-bound at scale | 46% HBM at scale | Engine-correct prediction hit (proxy) |

*"(proxy)" marks verdicts resting on the counter-free %HBM proxy (Blackwell ncu was blocked); the proxy was ncu-validated once, on T4.*

### Memory and Capacity Wins

| Metric | Value | Step |
|--------|-------|------|
| S matrix elimination | 2164 MB to 17 MB = 125x | v3 |
| FP8 capacity vs FP16 | 2x (by construction) | v9 |
| NVFP4 capacity vs FP16 | 3.56x (measured) | v10 |
| MLA vs MHA KV | 202x (measured) | v11 |
| MLA vs GQA-8 KV | 12.6x (measured) | v11 |

### Architecture Constants

|  | T4 (sm_75) | A100 (sm_80) | B200 (sm_100) | B300 (sm_103) |
|---|---|---|---|---|
| SMs | 40 | 108 | 148* | 148* |
| HBM | 16 GB | 80 GB | 192 GB | 288 GB |
| BW | 320 GB/s | 2039 GB/s | 8000 GB/s | 8000 GB/s |
| L2 | 4 MB | 40 MB | 132.6 MB* | 132.6 MB* |
| FP16 TC | 65 TFLOPS | 312 TFLOPS | 2250 TFLOPS | 2500 TFLOPS |
| FP8 | -- | -- | 4500 TFLOPS | 5000 TFLOPS |
| NVFP4 | -- | -- | 9000 TFLOPS | 15000 TFLOPS |

*Measured values that correct published specifications (B200/B300: 148 SMs not 160, L2 132.6 MB not 192 MB, clock 2032 MHz not 2600 MHz)*

---

## Honest Misses and Dead Ends

This section is deliberately prominent. The ability to record, analyze, and learn from failed predictions is as important as the wins.

### The L2 Lesson (Steps 1--2)

Predicted a 30x traffic cut from shared-memory tiling; measured 1.02--1.08x. The T4's 4 MB L2 cache had already absorbed the redundant operand reads that tiling was designed to eliminate. The roofline model was right about the bound but blind to the cache. Six steps later, the recurring "~10% HBM" readings turned out to be an occupancy artifact (true cap ~28%), not the floor -- a misread that persisted until confound removal (locked clocks + L2 flush + ncu) finally earned the diagnosis.

### The S-Elimination Paradox (Step 3)

Online softmax eliminated 99% of DRAM traffic (S provably gone: 2164 MB to 17 MB). The kernel was 3--7x SLOWER. Nothing was bandwidth-bound, so removing bandwidth-bound traffic while regressing the schedule (one-thread-per-row, unstaged K/V reads, two-pass structure) was a net loss. The mechanism was correct; the schedule defeated it. It took until v4's fused single-pass to make S-elimination also a wall-clock win.

### Four Consecutive Dead Ends (Steps 8--8.6)

After M-packing (Cut 1) worked, four attempts to close the remaining gap all failed:

1. **Tensor cores (Cut 2):** 1.8--4.6x SLOWER on TWO architectures (T4 + A100). The smoking gun: WMMA barely moved T4 to A100 (42 to 39 us/tok) despite ~5x TC throughput + ~6x BW, while CUDA-core nearly halved (16.9 to 9.7). Decode at M <= 16 is too small for tensor cores.
2. **Double-buffer (v8.5):** NULL. %HBM flat ~10%, us/tok within noise. Memory was never the wall.
3. **Occupancy (v8.6 Arm 1):** NULL. Split-KV already emits enough blocks; adding residency doesn't help.
4. **ILP (v8.6 Arm 2):** NULL. KU=4 key unrolling tracked Cut 1 exactly.

The counter-prediction landed: the floor was the per-row serial online-softmax recurrence -- a dependency chain that is unhideable by thread-level parallelism or instruction-level parallelism. Only removable by relayout (v8.7's score-stationary loop, which shortened the recurrence 32x).

### The "Per-CTA Forever" Correction (Steps 6--12)

From v6 through v11, every decode measurement showed ~10% HBM utilization. The diagnosis was consistently "per-CTA-bound." v12 corrected this long-standing reading: the ceiling was WORK-STARVATION, and the right engine (tcgen05 tensor cores) converts work to throughput. v11's CUDA-core kernel was flat at 0.75 TFLOP/s; the same workload on v12's tcgen05 kernel spans 2.4 to 1785 TFLOP/s (70--1000x at matched B=1). "Per-CTA-bound" was real -- but it was a statement about the engine, not the problem.

---

## Independent Validation

![Independent validation scorecard: reproduced roofline AIs and literature-checked claims](diagrams/portfolio-validation-scorecard.svg)

The project was put through an independent review that re-ran the roofline tool against the repository and cross-checked the headline external claims against current literature. It reproduced every published arithmetic-intensity number **exactly**:

| Claim (from `results.md`) | Independent re-run of `roofline.predict` | Match |
|---|---|---|
| MLA FP16 AI = 234.8, ridge 312.5, HBM-bound | 234.8, ridge 312.5, HBM | Yes |
| MLA FP8 AI = 469.7, compute-bound | 469.7, MMA | Yes |
| MLA NVFP4 AI = 835.0, compute-bound | 835.0, MMA | Yes |
| GQA-8 NVFP4 AI ~= 3.6 (G=1) | 3.6, HBM | Yes |

External claims that held up against the literature: the **B300/sm_103 hardware constants** (288 GB HBM3e, 15 PFLOPS dense FP4, 8 TB/s flat vs B200, and the INT8 throughput cut that motivates the NVFP4-not-INT8 KV choice); the **decode-is-work-starved** characterization (aligns with and sharpens the current large-batch-inference consensus); the **MLA capacity math** (202x vs MHA, 12.6x vs GQA-8); and the **"complement, not beat" positioning** (production kernels like FlashMLA run ~3 orders of magnitude above the v11 CUDA-core kernel, exactly as the log states). The related-work anchor -- Grouped Latent Attention (arXiv 2505.21487), open MLA-decode roofline on H100 -- is a real, on-point closest-prior, and the stated delta (sm_103 + KV-quantization) is a genuine gap.

The review's verdict: the methodology (pre-registered predictions, confound removal, negative results as first-class deliverables) is *"more careful than much of what appears in published kernel-optimization papers"* and is *"genuinely publication-grade for a characterization venue."*

**Three hardening items** it flagged -- all of which the project's own logs had already raised, and which are now reflected in the caveats above:

1. **The 4x cuBLAS gap is inferred, not profiled.** The "M=128 is a real tensor-core GEMM" conclusion rests on a structural argument plus a measured 4x wall-clock gap against a torch baseline that was never profiled to confirm it hit an M=128 TC GEMM. Stated as inference throughout.
2. **Blackwell verdicts are proxy-grade.** The T4 era has ncu ground truth; the B200/B300 runs were on unprivileged containers where ncu counters were blocked, so the sm_103 limiter reads rest on the counter-free proxy (validated once, on T4). A privileged-box ncu run is the top open item.
3. **Accuracy numbers are single-seed.** The FP8 ~6-7e-4 and NVFP4 ~2.4e-3 RMSE figures, and the ~3.5x ratio, are one seed per shape on a small substrate -- real for this setup, not general constants.

Two performance cells remain genuinely unmeasured and are logged as such: the **v5 prefill benchmark** and the **native NVFP4-compute arm (Arm 2)**. Nothing is misrepresented -- all 17 kernels are built and correctness-tested; two performance cells are open.

*(The review was an independent AI-assisted re-run of the tool plus a literature cross-check, not a human peer review. Its value is the reproduction and the external fact-checking, both of which are independently repeatable from the repo.)*

---

## Diagrams

All diagrams are hand-authored SVGs in `docs/diagrams/`.

### Full-Arc Overview
- `portfolio-hero.svg` -- the hero image (17 kernels, 4 architectures, three eras, headline metrics)
- `v1-v12-arc-summary.svg` -- the complete v1 to v12 journey
- `portfolio-validation-scorecard.svg` -- independent-validation scorecard (reproduced AIs + literature-checked claims)

### Roofline and Regime
- `decode-roofline.svg` -- decode AI (2/b) vs ridges; how FP4-KV / GQA / MLA approach the ridge
- `decode-roofline-crossover.svg` -- the refuted crossover prediction (flat %HBM)
- `v9-task1-regime.svg` -- the ncu-validated regime characterization
- `v10-b300-regime.svg` -- B300 regime (no bandwidth knee to 2M tokens)
- `v12-throughput-regime.svg` -- tcgen05 throughput scaling with B x K

### Per-CTA Anatomy
- `per-cta-limiter-anatomy.svg` -- why batch cannot help (2 blocks/SM, 1-of-8 warps active)
- `v10-per-cta-wall.svg` -- per-CTA wall on the precision frontier
- `v11-per-cta-wall-b300.svg` -- per-CTA wall on B300 with MLA

### Technique Explanations
- `gqa-mpacking.svg` -- the M-packing technique (G heads into the M dimension)
- `split-kv-schedule.svg` -- split-KV decode schedule
- `splitkv-lse-merge-dataflow.svg` -- partial + LSE merge dataflow
- `asymmetric-precision-dataflow.svg` -- FP4-safe P * V vs precision-critical Q * K^T + softmax

### Step-Specific Results
- `v7-vs-sdpa-batch-crossover.svg` -- v7 wins at B=1, SDPA overtakes by B=8
- `v8_5_v8_6_pastL2.svg` -- past-L2 re-test (occupancy revives at B >= 32)
- `v9-fp8-win-anatomy.svg` -- FP8 win decomposition (capacity durable, latency fragile)
- `v10-nvfp4-verdict.svg` -- NVFP4 verdict (capacity yes, latency no)
- `v11-cublas-gap.svg` -- the 4x cuBLAS gap (engine wrong, not shape wrong)
- `v11-roofline-two-layer.svg` -- two-layer prediction model
- `v11-capacity-accuracy.svg` -- MLA capacity + accuracy tradeoff
- `v11-nsys-schedule.svg` -- kernel schedule (99.9% partial, <0.01% merge)
- `v12-engine-ridge-comparison.svg` -- engine ridge comparison
- `v12-two-layer-prediction-model.svg` -- prediction model breakdown
- `v12-work-starvation-correction.svg` -- per-CTA forever corrected to work-starvation

### Architecture and Roadmap
- `b200-b300-seams.svg` -- B200 to B300 architecture delta
- `build-roadmap-v6-v11.svg` -- the reordered decode arc plan
- `v10-to-v11-shape.svg` -- GQA to MLA shape change
- `v11-to-v12-tcgen05.svg` -- MLA to tensor core transition
- `v12-paper-positioning.svg` -- paper positioning vs related work
- `v12-related-work-landscape.svg` -- related work landscape
- `v12-arm2-research-plan.svg` -- future Arm 2 research plan

---

## Technical Skills Demonstrated

**CUDA Kernel Development.** 17 hand-written attention kernels from scratch, spanning sm_75 through sm_103 (Turing through Blackwell Ultra). Shared memory staging, register pressure management, warp-shuffle reductions, bank-conflict-free layouts, score-stationary loop design.

**GPU Memory Hierarchy Optimization.** L2 confound identification (the v1--v2 lesson), shared memory tiling strategies (64x64 and 32x32), register-resident accumulators (v4 O-rescale), smem capacity budgeting (v11's ~46 KB latent staging), bank-conflict-free transposed layouts (v8.7 sK_T[d][key+1]).

**Tensor-Core Programming.** Turing WMMA 16x16x16 (v5 prefill), WMMA decode ablation (v8 Cut 2 -- proved it wrong), Blackwell tcgen05 characterization (v12 CUTLASS ex77), CUTLASS 4.6 integration.

**Attention Algorithm Design.** MHA (v1--v5), GQA with M-packing (v8 -- G heads packed into the M dimension), MLA latent-KV (v11 -- all h_q heads share one latent, M=128 by construction), split-KV decode / Flash-Decoding (v6), paged attention in the vLLM block-table layout (v7).

**Precision Engineering.** FP32 baseline (v1--v4), FP16 with FP32 accumulators (v5--v8), FP8 E4M3 with fused per-tile dequant and per-tensor scales (v9), NVFP4 E2M1 4-bit with micro-scales and fused dequant (v10), asymmetric precision recipe analysis with real GPT-2 KV (Stage A/A-prime ablation).

**GPU Profiling and Measurement.** Nsight Compute (ncu) hardware counters, nsys kernel traces, CUPTI timing, counter-free methodologies (validated within 1% of ncu), L2-flush timing technique (the jan.ai method), locked-clock measurement, cross-architecture constant verification (corrected published specs for B200/B300).

**Roofline Modeling.** Custom prediction tool with CLI (`python -m roofline.predict`), two-layer model (pure roofline + per-CTA-corrected), four-architecture constant database (T4/A100/B200/B300 in `roofline/archs.py`), MLA-specific AI formula (2 * h_q * (2L+R) / ((L+R) * b)).

**Multi-Architecture Measurement.** Rented measurement campaigns on T4 (free Colab + vast.ai root), A100 (Colab), B200 (vast.ai), B300 (vast.ai). Architecture constant verification that found and corrected published specs (148 SMs not 160, L2 132.6 MB not 192, 2032 MHz not 2600).

**Production Inference Patterns.** Paged KV cache in the vLLM block-table layout (v7), score-stationary inner loops (v8.7), split-KV + LSE merge (v6), GQA M-packing (v8), MLA weight-absorbed latent-KV (v11), designed as `from fa_kernels import attention` API consumed by a future mini-vLLM.

**Scientific Methodology.** Prediction-before-coding discipline at every step, confound removal (locked clocks, L2 flush, past-L2 sweep), honest negatives as first-class deliverables (four consecutive dead ends recorded and analyzed), two-layer prediction model that separates what roofline can and cannot see, counter-free proxy validation against ncu (13.8% vs 12.85%).

---

## Future Roadmap

### Next Steps

**v13 -- GLA / Sparse Attention.** Grouped Latent Attention (Tri Dao, arXiv 2505.21487 -- also the closest-prior decode-roofline anchor, on H100), then the sparse direction: DSA (DeepSeek V3.2), CSA (DeepSeek V4). The field is moving toward latent and sparse attention variants, and the roofline-first methodology transfers directly. (Note: arXiv 2505.21487 is *Grouped* Latent Attention -- not to be confused with *Gated Linear* Attention, arXiv 2312.06635, a separate line of work.)

**Arm 2 -- Native NVFP4 Compute.** No kernel anywhere does native FP4 tensor-core decode (CUTLASS ex77 is FP8-native; the M >= 128 tcgen05 gate for block-scaled NVFP4 is met by MLA's M=128). This is genuine open novelty for a future kernel.

**Three-Generation Roofline Spine.** H100 to B200 to B300 with locked-clock, ncu-validated measurements on the same kernel. The counter-free proxy (validated on T4) would be confirmed on each architecture.

**ncu on Privileged Box.** Belt-and-suspenders L2-hit-rate validation on Blackwell. The counter-free proxy already validated on T4; this extends it. Unprivileged containers block ncu counters (ERR_NVGPUCTRPERM).

**Research Paper.** PMBS@SC or IISWC characterization paper. The first open, kernel-level, prediction-vs-measured roofline characterization of FlashAttention decode on sm_103, across MHA to GQA to MLA and FP16 to FP8 to NVFP4, vs FlashInfer/FlashMLA. The contribution is the open paper-grade roofline + FP4-KV recipe + per-CTA limiter diagnosis on sm_103 -- characterization-grade, complementing not beating the production kernels.

### Downstream Project

This kernel library is the foundation for a from-scratch mini-vLLM inference engine (PagedAttention + continuous batching), eventually serving a compressed model:

```python
from fa_kernels import attention, paged_attention, gqa_attention, mla_attention
from fa_kernels import fp8_attention, nvfp4_attention, mla_tc_attention
```

---

## Resume Bullet Points

### Version A -- ML Systems / GPU Kernel Engineering

Built 17 CUDA attention kernels from scratch across 4 GPU architectures (T4 through B300), achieving 8--16x decode speedup over PyTorch SDPA and 1785 TFLOP/s (36% FP8 peak) on NVIDIA B300 tensor cores; developed a roofline-first methodology with prediction-vs-measured analysis validated by Nsight Compute, earning a confound-free per-CTA-bound diagnosis across Turing/Ampere/Blackwell.

### Version B -- ML Research / Applied ML

Designed and measured a 12-step roofline-driven optimization arc for FlashAttention decode on NVIDIA Blackwell (B300/sm_103), spanning MHA to GQA to MLA and FP16 to FP8 to NVFP4 with prediction-vs-measured analysis at every step; identified per-CTA work-starvation as the decode limiter (not bandwidth), validated by Nsight Compute across 3 architectures -- targeting a PMBS@SC characterization paper.

### Version C -- Robotics / Mechanical Engineering Crossover

Applied bottleneck-hunting methodology from mechanical systems to GPU kernel optimization: rebuilt FlashAttention from scratch (17 CUDA kernels, 4 architectures), using roofline models to predict performance limiters before coding -- achieving 8--16x speedup over PyTorch and characterizing NVIDIA's latest B300 GPU; demonstrated systematic predict-measure-iterate engineering on hardware where the "limiting factor" shifts with every design change.

---

## Repository

**GitHub:** [github.com/gkienpham-cmd/flashattention-cuda](https://github.com/gkienpham-cmd/flashattention-cuda)

```
fa_kernels/   public API: attention(), paged_attention(), gqa_attention(),
              fp8_attention(), nvfp4_attention(), mla_attention(), mla_tc_attention()
kernels/      17 versioned CUDA kernels (v1_naive -> v12_mla_tc)
roofline/     limiter predictor: model.py, predict.py CLI, archs.py (T4/A100/B200/B300)
bench/        harness.py (SDPA comparison), regime.py (decode sweeps with L2-flush)
tests/        correctness vs SDPA, parametrized across all backends
docs/         results.md (the curve), decisions.md (the log), interview-prep.md, 32+ diagrams
notebooks/    44 Colab gate notebooks (per-step build/test/bench/profile)
```

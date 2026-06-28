# ROADMAP

The journey, one measured speedup at a time. Each step identifies the **current bottleneck**
on the target hardware, then co-designs the algorithm around it — the FlashAttention-2/3/4
principle that every hardware generation has a different limiting factor.

**Hardware legend**
- **[BUILD]** — runs on the Colab **T4 (Turing sm_75)** I have now.
- **[RENT]** — build on a rented **A100 (Ampere sm_80)** or **H100 (Hopper sm_90)**.
- **[STUDY]** — Blackwell-only or no-hardware; written roofline analysis + CuTe-DSL prototype, not a build target.

Status: `[ ]` not started · `[~]` in progress · `[x]` done (tests green + quiz passed).

---

## ★ ACTIVE TRACK (2026-06-27): the decode arc v6 → v11

The kernel versions diverged from the linear curriculum below: v1–v4 built the prefill fundamentals,
**v5** jumped to tensor cores (WMMA, = old #6), and **v6** pivoted to the **decode regime** (`N_q=1`,
KV-cache) — the bridge to the from-scratch mini-vLLM. The live plan is now the decode arc, **reordered
after v6 measured occupancy (not bytes) as the limiter** (full math + per-step deliverable + 5 paper
diagrams in [`docs/decode-replan.md`](docs/decode-replan.md); thesis in
[`docs/b300-decode-research.md`](docs/b300-decode-research.md)):

- `[x]` **v6 — split-KV decode (Flash-Decoding)** — fill the SMs at `N_q=1`; partial + LSE merge. **[BUILD]** · *DONE 2026-06-27: beats naive `N_q=1` 5.7–8.2×, SDPA 1.5–3.3×, but only ~12% HBM → occupancy-bound, not bandwidth-bound (a `BH=8` micro-bench artifact; batch-conditional).*
- `[x]` **v7 — paged KV gather + harness fixes** — block-table indirection; **`--batch` sweep** + **`--decode` causal query-offset**. **[BUILD/RENT]** · *DONE 2026-06-27 (vast.ai T4, both gates: 51/51 correct + quiz passed). **The `--batch` sweep REFUTED the predicted crossover** — %HBM flat 9.4–12.4% from BH=8→512; decode is per-CTA-bound (32 KB smem → 2 blocks/SM; 1-of-8 warps active), NOT grid-occupancy-bound, at every batch. Corrects the "batch-conditional" claim: GEMV→GEMM before bytes, at all batch sizes. Causal query-offset fix works. **Deep-research close-out (2026-06-27):** per-CTA-bound + GQA-before-bytes both survive 0/3 adversarial refute; %HBM confirmed fp16-correct (not 2×); SDPA overtakes v7 by B=8 → v8 gains a "reclaim-batch" deliverable. See [`docs/v7-deep-research.md`](docs/v7-deep-research.md).*
- `[x]` **v8 — GQA M-packing** ★ *the reorder* — pack `G` query heads into `M` (one block per KV-group, G warps active, KV read once, `AI = 2/b → 2G/b`). **DONE 2026-06-28 (T4 + A100, both gates: correctness green + quiz passed).** Deliverable = **Cut 1 (CUDA-core M-packing): 8.6× over no-packing, beats SDPA 2.5–10× at all batch** (reclaims the B≥8 regime v7 lost). **The tensor-core sub-thesis was REFUTED on TWO architectures:** Cut 2a (Turing WMMA) + a cheap Cut 2b A100 probe both measured tensor cores **1.8–4.6× SLOWER** than the CUDA-core GEMV — *worse on the faster GPU*, because decode `M=G≤16` is too small to amortize the opaque-fragment smem-softmax tax → **per-CTA-schedule-bound, not compute-bound** (WMMA didn't scale T4→A100 while CUDA-core nearly halved). The GEMV→GEMM fix that won for v5 *prefill* is the wrong tool for *decode*. Still **≤11% HBM** → per-CTA-bound, so v9 bytes remain premature. See `results.md`/`decisions.md` Step 8, `interview-prep.md` C11.
- `[x]` **v8.5 — double-buffered KV pipeline** (portable, not cp.async) — **MEASURED 2026-06-28 (Colab T4): correctness 38/38, but double-buffering did NOTHING** — `%HBM` flat ~10%, µs/tok within noise vs Cut 1. **The decode stall is the per-key warp-shuffle reduction (v4 GEMV wall), NOT the KV-load latency** (prefetch can't help warps that are compute-serialized, not memory-waiting; ~2 blocks/SM can't hide the reduction). So the kernel is **compute-latency-bound at ~10% HBM — memory was never the wall.** Closes the decode investigation: neither tensor cores (Cut 2) nor load-overlap (v8.5) move it. **Implication: v9 FP8 won't show a latency win on this micro-bench either** — bytes aren't the bottleneck. A faster reduction (vectorize/ILP/occupancy = "v8.6") or a genuinely memory-bound regime (N_k past L2) is needed first; FP8/NVFP4's real value is KV-cache *capacity* + accuracy. See `results.md` Step 8.5.
- `[x]` **v8.6 — hide the reduction latency** (CUDA-core, T4) — **MEASURED 2026-06-28 (Colab T4): 190/190 correct, but BOTH arms NULL.** **Arm 1 `v8_gqa_occ`** (FP16 smem → 4 blocks/SM) and **Arm 2 `v8_gqa_ilp`** (KU=4-unrolled key loop): on the clock-robust `%HBM` (clocks swung 360–1590 MHz, so µs/tok unusable cross-backend) both stayed **flat ~8–10% at every G/batch**. Occupancy never even engaged at the micro-bench batch (split-KV already emits ~80 blocks = 2/SM → no spare block for the 4-block ceiling; only a ~1% nudge at large batch); ILP tracked Cut 1 exactly. **Counter-prediction landed: the floor is the per-row serial online-softmax recurrence — unhideable by TLP or ILP.** Fourth consecutive negative (Cut 2 / v8.5 / occ / ilp) → mandate for v8.7. See `results.md`/`decisions.md` Step 8.6, `interview-prep.md` C11.5.
- `[x]` **v8.7 — score-stationary inner loop** (CUDA-core, T4) ★ *remove, don't hide* — **MEASURED 2026-06-28 (Colab T4): WIN, ✅ 228 passed.** Flipped the inner-loop layout to **lane = key** (full dot product q·k_c, no per-key cross-lane reduction; softmax once per 32-key group → recurrence 32× shorter; PV transpose via pipelined `__shfl` broadcasts), FP16 transposed smem (~3 blocks/SM, occupancy held). **ss beats Cut 1 1.1–1.6× at matched clock across G and batch** (best at d=64; d=128 wins less, 1.16–1.18× — the R2 smem-read-BW drag, but not a null), **beats SDPA 8–16×**, and is the **FIRST lever since Cut 1 to move the clock-robust `%HBM`** (up ~2–3 pts → ~10–12%). ss-vs-occ (both FP16 smem) confirms it's the LAYOUT. **Two truths:** the reduction/recurrence WAS a real component of the floor (remove-not-hide vindicated), but %HBM plateaued ~10–12% — a residual per-CTA ceiling remains, so the kernel is still NOT bandwidth-bound → v9 FP8's value stays capacity+accuracy, not micro-bench latency. **Closes the decode-schedule arc:** M-packing (Cut 1) + score-stationary (v8.7) are the two real decode levers. See `results.md`/`decisions.md` Step 8.7, `interview-prep.md` C11.6. **Combined v8.6+v8.7 Gate-2 quiz PASSED 2026-06-28 → both steps DONE.**
- `[ ]` **v9 — FP8 KV cache** — first byte cut (`b:2→1`, `AI ×2`); accuracy delta vs FP16 documented. **Bytes only pay once bandwidth-bound** — v8 is still ≤11% HBM (per-CTA-bound), so **the schedule levers (v8.5/v8.6) come first.** **[B300]** (H100 ok for correctness).
- `[ ]` **v10 — NVFP4 + asymmetric precision** — *the headline:* `b≈0.56` (~3.5× fewer bytes); FP4 `P·V` + MXFP8 scores + hardware `exp2`. **[B300]**.
- `[ ]` **v11 — MLA / speculative decode (stretch)** — latent-KV (`AI → 2H/b`) / draft `M>1`; push intensity toward the ridge. **[B300]**.

This **supersedes the ordering** of Phases 3–5 below for the active work (FP8/FP4 now follow GQA, not
precede it); the precision/exp/masking techniques in those phases remain the technique menu the decode
arc draws from. The phased curriculum below is kept as the bottleneck-by-bottleneck teaching context.

---

## Phase 1 — FP16 fundamentals · bottleneck: **HBM bandwidth**
- `[x]` **1. Naive** — materialize `S=QK^T`, softmax, `PV`. Baseline; exposes the bandwidth wall. **[BUILD]**
- `[x]` **2. Tiling w/ shared memory** — stage Q/K/V tiles, cut global traffic. **[BUILD]** · *Measured: tiling cut ~0% DRAM (L2 already owned the operands); S = ~99% of traffic. Quiz + ncu gates cleared.*
- `[x]` **3. Online softmax** — running max/sum, never materialize S. **[BUILD]** · *Both gates cleared (2026-06-19): quiz passed + counter-free measurement. 15/15 correct, S eliminated (peak mem +17 MB vs v2's +2164 MB = 125×), but **3–7× slower than v2** — deleting DRAM traffic bought nothing on a machine Step 2 proved was never bandwidth-bound; real limiter is occupancy/latency (151× above MMA floor, pass2=88.6%). ncu pipe-util deferred to bare-metal (counters blocked on cloud).*
- `[x]` **4. Fused FlashAttention-1** (= **v4**) — one kernel, O-rescale. **[BUILD]** · *DONE 2026-06-20: beats v2 1.7–2.6×, v3 7.5–15×; new limiter = FMA under-util (GEMV-shaped scoring), ~18× off floor.*
- `[~]` **5. FlashAttention-2 partitioning** — folded into v5/v6 (v5 added the tensor-core path; v6's split-KV is the decode-side seq-len parallelization). Standalone prefill-partitioning pass deferred.
- `[x]` **6. Tensor cores** (= **v5 WMMA**) — FP16-in/FP32-accum, sm_75. **[BUILD]** · *kernel + wiring landed, tests green; prose close-out is the outstanding Step-5 backfill.*

## Phase 2 — asynchrony & warp-specialization (FA-3 class) · bottleneck: **latency / instruction issue**
- `[ ]` **7. Producer/consumer split** — `cp.async` needs Ampere **[RENT A100]**; portable double-buffer prototype **[BUILD]**.
- `[ ]` **8. Pingpong + 2-stage GEMM-softmax pipelining** — portable idea **[BUILD]**; full WGMMA/TMA warpgroup **[RENT H100]**.

## Phase 3 — low-precision frontier (the Sconce twist) · bottleneck: **matmul throughput + precision/accuracy**
- `[ ]` **9. INT8 attention** — block quantization + incoherent (Hadamard) processing; RMSE-vs-FP64 study with simulated outliers; accuracy/latency Pareto. Bench vs **SageAttention**. **[BUILD]** (Turing INT8 TCs)
- `[ ]` **10. FP8 / INT4** — FP8 **[RENT H100]**, INT4 **[BUILD/STUDY]**; vs **SageAttention2** (INT4/FP8), **SageAttention3** (FP4 **[STUDY]**).

## Phase 4 — FA-4 frontier (roofline co-design) · bottleneck: **shared-mem traffic + exponential unit**
- `[ ]` **11. Conditional softmax rescaling** — skip rescale when `m_j − m_{j−1} ≤ τ` (τ = log2(256) = 8). Portable. **[BUILD]**
- `[ ]` **12. Software `exp2` emulation** — Cody-Waite range reduction + degree-3 Horner on FMA units, emulate 10–25% of entries. Portable. **[BUILD]**
- `[ ]` **13. CuTe-DSL port** — FA-4's framework; TMEM / 2-CTA MMA / larger tiles as written roofline analysis. **[STUDY]** (+ prototype where possible)

## Phase 5 — inference specialization (bridge to mini-vLLM)
- `[ ]` **14. Causal + variable-length masking; LPT scheduling** for load balance. **[BUILD]**
- `[ ]` **15. KV-cache + paged-attention decode path** — the import surface my from-scratch engine consumes. **[BUILD]**

---

### The per-step loop (every step, in order)
1. **Roofline first** — predict the limiter (MMA / HBM / MUFU) on the T4 + this tile, before coding.
2. **Explain** the memory-hierarchy / scheduling reasoning.
3. **Write** kernel + binding (hand-verify the core loop line by line).
4. **Correctness** vs SDPA with a documented tolerance (FP64 anchor in Phase 3).
5. **Benchmark** vs SDPA (+ FA-2, + cuDNN) across seq 512/2K/8K × d 64/128.
6. Update `docs/results.md` (speedup + roofline prediction-vs-measured + ncu reading).
7. Update `docs/decisions.md` (bottleneck, options, choice, what changes on another arch).
8. **Quiz** on why it worked and what it traded against — before moving on.

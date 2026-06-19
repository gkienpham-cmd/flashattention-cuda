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

## Phase 1 — FP16 fundamentals · bottleneck: **HBM bandwidth**
- `[x]` **1. Naive** — materialize `S=QK^T`, softmax, `PV`. Baseline; exposes the bandwidth wall. **[BUILD]**
- `[x]` **2. Tiling w/ shared memory** — stage Q/K/V tiles, cut global traffic. **[BUILD]** · *Measured: tiling cut ~0% DRAM (L2 already owned the operands); S = ~99% of traffic. Quiz + ncu gates cleared.*
- `[x]` **3. Online softmax** — running max/sum, never materialize S. **[BUILD]** · *Both gates cleared (2026-06-19): quiz passed + counter-free measurement. 15/15 correct, S eliminated (peak mem +17 MB vs v2's +2164 MB = 125×), but **3–7× slower than v2** — deleting DRAM traffic bought nothing on a machine Step 2 proved was never bandwidth-bound; real limiter is occupancy/latency (151× above MMA floor, pass2=88.6%). ncu pipe-util deferred to bare-metal (counters blocked on cloud).*
- `[ ]` **4. Fused FlashAttention-1** — one kernel, correct output rescaling. **[BUILD]**
- `[ ]` **5. FlashAttention-2 partitioning** — parallelize over seq-len, better warp work split. **[BUILD]**
- `[ ]` **6. Tensor cores** — WMMA / `mma.sync` `m16n8k8` FP16, tuned for sm_75. **[BUILD]** (Turing has TCs)

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

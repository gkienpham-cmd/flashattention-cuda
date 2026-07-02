# Paper outline — PMBS@SC 2026 (8-page short paper)

> Target: arXiv by Jul 28, PMBS full-paper deadline Aug 5 (AoE), late-breaking Aug 26.
> Word budget: ~6000 words + figures. 8 pages, two-column IEEE format.

> **STATUS (2026-07-02): SUBMISSION-READY — `flash-paper/` (IEEEtran, two-column).**
> All 7 sections drafted, figures redesigned (Claude Design) and committed as vector PDFs,
> all 17 `refs.bib` entries author-verified, compiled locally with tectonic
> (`flash-paper/rgd-paper.pdf`, 9 pages), arXiv bundle `rgd-arxiv.tar.gz` built.
> **Final external review passed:** see [`paper-final-check-brief.md`](paper-final-check-brief.md)
> (Fable-5 check: no hallucinations, citations clean; its 3 consistency items + hedge audit
> were resolved same day — resolution section appended there). A first-person ("I/my") copy
> lives in `flash-paper/first-person/`. Remaining before PMBS (not arXiv): possible trim from
> 9 to 8 pages if the workshop limit includes references.

---

## Title

**Roofline-Guided Characterization of Attention Decode on Blackwell:
From Per-CTA-Bound to Work-Starvation Across MHA, GQA, and MLA**

## Abstract (~200 words)

- **Problem:** Attention decode is the latency-critical phase of LLM serving, yet no published
  work provides prediction-vs-measured roofline characterization on NVIDIA's latest datacenter
  GPU (B300/sm_103) across attention types and precisions.
- **Method:** We build 12 attention kernels of increasing sophistication (naive → tiled → fused →
  split-KV → GQA-packed → score-stationary → FP8 → NVFP4 → MLA → tcgen05), each with a
  pre-registered two-layer roofline prediction (pure roofline + per-CTA-corrected) evaluated
  against measured results on T4, A100, B200, and B300.
- **Key finding:** Decode's apparent per-CTA ceiling is work-starvation — the right engine
  (tcgen05 tensor cores) converts work to throughput (0.75 → 1785 TFLOP/s), but only in the
  high-throughput serving regime (large batch × long context). At single-stream decode, even
  NVIDIA's production kernel runs at 1–2% of FP8 peak.
- **Contribution:** The first open, roofline-grade characterization of attention decode on sm_103
  across MHA→GQA→MLA and FP16→FP8→NVFP4. The two-layer prediction model called every decode
  result correctly (7/7) while pure roofline was wrong 5/7 times.
- **Stage A (§5.3), DONE:** a pre-registered boundary test on B300 confirms native NVFP4 *compute*
  does not help MLA decode (both FP4 and FP8 GEMMs HBM-bound, ≤5.8%/21.7% of peak). Negative result.

---

## 1. Introduction (~600 words, ~0.75 pages)

**Hook:** LLM serving latency is dominated by attention decode — one token at a time, memory-bound
by the KV cache. Production kernels (FlashInfer, FlashMLA, cuDNN) are deployed on Blackwell but
no open paper characterizes their performance with prediction-vs-measured roofline analysis.

**The empty cell:** Table 1 (= `v12-related-work-landscape.svg`). Map the landscape:
- FA4 → B200 prefill, no decode
- GLA (Tri Dao) → H100 MLA decode roofline (closest prior; our delta: sm_103 + KV-quant)
- SnapMLA → Hopper FP8 MLA decode, no Blackwell/roofline
- Attn-QAT / SageAttention3 → FP4 attention, prefill only, consumer SM120
- FlashInfer/FlashMLA → B300 decode production, no roofline published
- arXiv 2605.04178 → roofline methodology (GEMM, B200; owns the method — we cite + apply)

**Our contribution (3 bullets):**
1. First open, prediction-vs-measured roofline characterization of attention decode on B300/sm_103
   across MHA→GQA→MLA and FP16→FP8→NVFP4.
2. A two-layer prediction model (pure roofline + per-CTA-corrected) with pre-registered predictions:
   Layer 2 correct 7/7, Layer 1 wrong 5/7.
3. Central finding: decode's per-CTA wall is work-starvation, not a fixed architectural limit.
   The right engine converts work → throughput (v11 CUDA-core flat 0.75 TFLOP/s → v12 tcgen05
   2.4–1785 TFLOP/s scaling with B×K).

**Honest scope:** We complement, not beat production kernels. The methodology is established
(cite arXiv 2605.04178); the application to this workload × architecture × precision space is novel.

---

## 2. Background (~600 words, ~0.75 pages)

### 2.1 Attention variants
- **MHA** (standard multi-head): H_q = H_kv, KV per head → O(B·H·N·d) cache.
- **GQA** (grouped-query): H_kv < H_q, G = H_q/H_kv heads share one KV → G× cache reduction.
  Decode packs G queries per CTA → AI rises from 2/b to 2G/b.
- **MLA** (multi-head latent, DeepSeek): one shared latent (dim 576) replaces all KV heads →
  H_kv=1, all h_q=128 heads share it → M=128 by construction at decode. AI ≈ 3.78·h_q/b.
  The only decode shape where M=128 natively meets the NVFP4 M≥128 gate.

### 2.2 The roofline model
- Classical: AI = FLOPs / bytes, limiter = max(t_hbm, t_mma, t_mufu).
- **Why it fails for decode** (cite arXiv 2605.04178): datasheet peaks overstate sustained
  throughput; ignores L2 cache, per-CTA scheduling, pipeline fill.
- Our extension: the per-CTA-corrected layer adds on-chip constraints (smem capacity → occupancy,
  warp packing GEMV vs GEMM, pipeline depth vs available MMA tiles).

### 2.3 Blackwell architecture (B300/sm_103)
- 148 SMs (measured, not datasheet 160), 8 TB/s HBM, 132.6 MB L2 (measured, not 192 MB).
- tcgen05 tensor cores: FP16 2.5 PF, FP8 5 PF, NVFP4 15 PF.
- Measured EX2 throughput: 5.33 TExp/s (0.50× the 10.7 claim).
- Clock: 2032 MHz (measured, not 2600 MHz marketing).

---

## 3. Methodology (~800 words, ~1 page)

### 3.1 The two-layer prediction model
- **Layer 1 (pure roofline):** AI from operand traffic + FLOPs; limiter from peak throughputs.
  Captures the WHAT (how much work and data) but blind to the HOW (scheduling).
- **Layer 2 (per-CTA-corrected):** Same AI but adds:
  (a) smem/TMEM staging → occupancy cap (how many CTAs fit per SM);
  (b) warp-level packing (M=1 GEMV vs M=128 GEMM);
  (c) TC pipeline depth vs available MMA tiles;
  (d) counter-prediction (what would kill the thesis).
- Figure: `v12-two-layer-prediction-model.svg`.

### 3.2 Pre-registered predictions
- Every step records BOTH predictions BEFORE coding/measuring. The gap between layers IS the finding.
- Counter-predictions give the result value regardless of sign (negative = publishable boundary).

### 3.3 Kernel progression (the 12-step journey)
- Table: v1→v12, one row per kernel, columns: what changed (single variable), predicted limiter,
  measured limiter, prediction hit/miss.
- Figure: `v1-v12-arc-summary.svg` (four phases: prefill schedule → decode shape → precision →
  engine).

### 3.4 Measurement methodology
- Counter-free proxy: %HBM = effective_bw / HBM_peak. Validated vs ncu on root T4
  (13.8% vs 12.85%, within ~1 pt). All decode readings use this.
- L2 flush technique (jan.ai zero-buffer), clock locking (root T4), CUDA-event timing.
- Cross-architecture: T4 (sm_75), A100 (sm_80), B200 (sm_100), B300 (sm_103).

---

## 4. Results (~1500 words, ~2 pages)

### 4.1 The prefill arc (v1→v5, T4) — brief
- Table: v1 naive → v2 tiled → v3 online → v4 fused → v5 WMMA. Speedup chain.
- Key finding: L2 hides the tiling win (v2 DRAM traffic ≈ v1); S-elimination is a memory win but
  not a latency win until the schedule is fixed (v4); GEMV→GEMM (v5 tensor cores) only helps
  prefill, not decode.
- Prediction misses are first-class (v2 L2 blind spot, v3 150× off floor).

### 4.2 The decode arc (v6→v12) — the main contribution

#### 4.2.1 Split-KV + paging (v6–v7)
- Decode kernel 5.7–8.2× over naive loop, 1.5–3.3× over SDPA.
- **But only 9–15% HBM** — per-CTA-bound (occupancy), not bandwidth-bound.
- v7 batch sweep REFUTES the predicted occupancy→bandwidth crossover: %HBM flat at all batch sizes.

#### 4.2.2 GQA M-packing + score-stationary (v8, v8.7)
- Cut 1 (CUDA-core M-packing): G× speedup, beats SDPA 6–10× at all batch.
- Dead ends: tensor cores (Cut 2, 1.8–4.6× SLOWER), double-buffer (v8.5 null), ILP (v8.6 null).
- v8.7 score-stationary: the ONLY inner-loop lever that moved the clock (1.1–1.6× over Cut 1).
  Removed the per-key serial reduction — the remove-not-hide principle.

#### 4.2.3 Cross-precision (v9–v10)
- FP8 E4M3 KV: ~1.3× latency win (L2→SM load bandwidth, not HBM). Flips negative under L2 flush.
- NVFP4 KV: capacity 3.56× vs FP16 / 1.78× vs FP8. Latency-negative past L2 (12–30% slower).
- **v9 Task 1 (root T4, ncu):** the confound-free verdict. %HBM caps at ~28%, ncu L2 hit-rate 1.1%
  past L2. Per-CTA-bound, not bandwidth-bound. Counter-free proxy validated.

#### 4.2.4 MLA decode (v11–v12) — the shape change
- v11 (CUDA-core MLA): M=128 by construction, AI jumps to 235. **But stays per-CTA-bound**
  (0.75 TFLOP/s, %HBM ~0% to 2M past 5× L2 overflow). ~4× SLOWER than torch dense-MQA.
- **v12 (tcgen05, the headline):** CUTLASS ex77 production kernel. Figure: `v12-throughput-regime.svg`.
  - B=1, K≤32K → 1–2% FP8 peak (per-CTA/pipeline-depth floor, CONFIRMED).
  - B=64, K=524288 → 1785 TFLOP/s = 35.7% peak / 46.3% HBM (work converts, CONFIRMED).
  - **The correction:** "per-CTA-bound forever" was work-starvation. The engine is the lever.
  - Figure: `v12-work-starvation-correction.svg`.

### 4.3 Cross-architecture: the ~40 GB/s per-CTA ceiling
- T4, B200, B300 all show ~40 GB/s achieved bandwidth at B=1 decode (11% of 320, 0.5% of 8000).
- Architecture-independent latency ceiling: the per-CTA wall is a SHAPE property (M=1 or small M
  with CUDA cores), not a hardware limit.

---

## 5. Discussion (~800 words, ~1 page)

### 5.1 The engine-ridge framework
- Figure: `v12-engine-ridge-comparison.svg`.
- Each engine (FP16/FP8/NVFP4) has its own ridge. NVFP4's 15 PF peak overshoots into HBM-bound
  (ridge 1875 > AI 835 → the FP4 peak is doubly unreachable for decode).

### 5.2 Implications for serving systems
- Single-stream moderate-context decode (the common case): even the production kernel is at 1–2%
  peak. The lever is batch × context (pipeline fill), not precision or algorithm.
- Long-context / large-batch serving: the roofs ARE reachable (~46% HBM at B=64/K=524K).
- FP8/NVFP4 KV value: capacity + accuracy (halving/quartering cache size), NOT decode latency.
  The latency win is regime-specific (L2-resident only for FP8; negative for NVFP4 past L2).

### 5.3 Does native FP4 compute help MLA decode? (Stage A result) — DONE, NEGATIVE (2026-07-02)
- **Measured on B300/sm_103a** (CUDA 12.9, CUTLASS v4.5.2). M=128 FP4-vs-FP8 GEMM sweep,
  K∈{256,512,576} × N∈{1K…524K}. FP4 = CUTLASS example 72a (block-scaled NVFP4); FP8 = cuBLAS
  E4M3 (`torch._scaled_mm`), matched bf16 output. **Data:** `notebooks/stage_a_results.csv`;
  **full write-up + caveats:** [`docs/stage-a-results.md`](stage-a-results.md).
- **KILL / negative confirmed:** both precisions HBM-bound at every N (AI 100–176 ≪ ridges 1875/625),
  reaching at most **5.8%** (FP4 of 15 PF) / **21.7%** (FP8 of 5 PF) of peak → native FP4 *compute*
  is never the bottleneck. Raw FP4/FP8 ratio flips with N (FP4 leads ≤32K where both are tiny;
  **FP8 wins ≥131K**, the realistic regime) — but neither is a compute win; the gap is
  bandwidth/operand-size + kernel-tuning. The M=128-by-construction packing advantage is a red
  herring for decode; FP4/NVFP4 stays a capacity+accuracy lever (§5.1–5.2), not a compute lever.
- **Caveat (recorded):** cross-harness (CUTLASS example FP4 vs tuned cuBLAS FP8) — the verdict rests
  on the kernel-quality-independent peak-fraction + roofline, not the head-to-head ratio.

### 5.4 Threats to validity
- Counter-free proxy (validated on T4, not yet B300 ncu).
- Clock variability (Colab throttling; locked-clock runs on root T4 and vast.ai B300).
- Single-seed accuracy for FP8/NVFP4 (gpt2-small, not a published constant).
- v5 prefill bench never executed (prediction, not measurement).
- nsys provenance: v10 B300 nsys data from off-notebook 2025.3.2 run (committed cell ran 2025.1.3 → empty).

---

## 6. Related Work (~500 words, ~0.5 pages)

Structured as the landscape table (`v12-related-work-landscape.svg`), with prose for the 3 closest:
1. **GLA** (Tri Dao, arXiv 2505.21487) — closest: MLA decode roofline on H100. Delta: sm_103 + KV-quant.
2. **Attn-QAT** (arXiv 2603.00040) — FP4 QK on Blackwell prefill. Delta: decode + MLA + roofline.
3. **arXiv 2605.04178** — roofline methodology (1.31% MAE on B200 GEMM). We cite as prior art for
   the method; our contribution is the APPLICATION to attention decode.
4. Brief mentions: SnapMLA, SageAttention3, FA4, FlashInfer/FlashMLA, arXiv 2506.02523, arXiv 2512.02189.

---

## 7. Conclusion (~200 words)

- Restate the finding: per-CTA wall = work-starvation; engine + work converts.
- The two-layer model's track record (7/7 vs 5/7).
- The empty cell filled: first open decode characterization on sm_103 across types and precisions.
- Future: native FP4 compute (Arm 2), GLA/sparse attention on Blackwell, the cross-gen roofline spine.

---

## Figures (8 + tables)

| # | Figure | SVG source | Section |
|---|---|---|---|
| 1 | Related work landscape (the empty cell) | `v12-related-work-landscape.svg` | 1 (Intro) |
| 2 | v1→v12 arc summary (4-phase journey) | `v1-v12-arc-summary.svg` | 3.3 |
| 3 | Two-layer prediction model | `v12-two-layer-prediction-model.svg` | 3.1 |
| 4 | v12 throughput regime (TFLOP/s vs B×K) | `v12-throughput-regime.svg` | 4.2.4 |
| 5 | Work-starvation correction (v6→v11 vs v12) | `v12-work-starvation-correction.svg` | 4.2.4 |
| 6 | Engine-ridge comparison | `v12-engine-ridge-comparison.svg` | 5.1 |
| 7 | Paper positioning (what we claim / don't) | `v12-paper-positioning.svg` | 1 or 6 |
| 8 | Arm 2 research plan (Stage A decision tree) | `v12-arm2-research-plan.svg` | 5.3 |

Tables: prediction-vs-measured per step (§3.3), v12 throughput data (§4.2.4), cross-arch ceiling (§4.3).

---

## Appendix / supplementary (if space allows)

- Full prediction-vs-measured table for all 12 steps (may be too large for 8 pages → supplementary).
- The ncu-validated counter-free proxy data (v9 Task 1).
- Kernel source availability (GitHub link).

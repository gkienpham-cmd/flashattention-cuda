# Research strategy — from v12 to a published paper

> Written 2026-06-30 based on the complete v1→v12 arc. Web-research findings incorporated
> 2026-06-30 (venue deadlines verified, novelty claims checked). Honest about scope:
> characterization-grade, not top-tier systems.

---

## 1. What we have (the measured assets)

The v1→v12 arc produced a unique dataset that no other published work contains:

| Asset | What it proves | Where it lives |
|---|---|---|
| 12-step prediction-vs-measured roofline | Two-layer model (pure + per-CTA-corrected) called every decode result; Layer 1 wrong 5/7, Layer 2 correct 7/7 | `results.md`, `decisions.md` |
| B300/sm_103 regime curve | CUTLASS ex77 tcgen05 MLA decode: 2.4→1785 TFLOP/s scaling with B×K; "per-CTA forever" corrected to work-starvation | `results.md` Step 12 |
| Cross-precision decode characterization | FP16→FP8→NVFP4 on the same kernel, same shape; FP8≈FP16 throughput = not compute-bound | Steps 9–12 |
| Cross-attention-type characterization | MHA (v1–v5) → GQA (v8) → MLA (v11–v12) on sm_103 | Full arc |
| Cross-architecture data points | T4 (sm_75), A100 (sm_80, partial), B200 (sm_100), B300 (sm_103) — ~40 GB/s per-CTA ceiling on all three | v9 Task 1, v10, v11 |
| ncu-validated counter-free proxy | %HBM ≈ ncu DRAM within ~1 pt (13.8% vs 12.85%) — validates all profiler-free readings | v9 Task 1 |
| 7 SVG paper figures (new) | v12 regime, engine-ridge, work-starvation, arc summary, Arm 2 plan, two-layer model, paper positioning | `docs/diagrams/v12-*` |

---

## 2. The three publication paths (ranked by effort and impact)

### Path A: Pure characterization paper (HIGHEST CONFIDENCE, ~2–3 weeks to draft)

**Title candidate:** "Roofline-guided characterization of attention decode on Blackwell: from
per-CTA-bound to work-starvation across MHA, GQA, and MLA"

**Thesis:** The first open, prediction-vs-measured roofline characterization of attention decode
on sm_103 across three attention types (MHA→GQA→MLA) and three precisions (FP16→FP8→NVFP4).
Central finding: decode's apparent per-CTA ceiling is work-starvation, not a fixed architectural
limit — the right engine (tcgen05) converts work to throughput (0.75→1785 TFLOP/s), but only in
the high-throughput serving regime.

**Why it's publishable:**
- The empty cell exists: no published paper has roofline-grade (prediction-vs-measured, pre-registered)
  characterization of attention decode on B300/sm_103. FA4 benchmarks B200 prefill; FlashInfer/FlashMLA
  ship B300 decode but publish no roofline analysis.
- Closest prior = Tri Dao's GLA (arXiv 2505.21487, MLA decode roofline on H100). Our delta: sm_103 + KV-quant.
- The two-layer prediction model is an *applied* contribution (pure roofline vs per-CTA-corrected);
  the methodology itself is established (arXiv 2605.04178 owns the "naive roofline fails" claim —
  must cite; our contribution is the APPLICATION to attention decode across types and precisions).
- The "per-CTA forever" → "work-starvation" correction is a falsifiable, counterintuitive finding.
- The prediction misses (v2 L2 blind spot, v3 150× off) are first-class results, not embarrassments.

**Venue targets (VERIFIED deadlines, 2026-06-30):**
- **PMBS@SC 2026** (Performance Modeling, Benchmarking and Simulation of HPC Systems, co-located with
  SC26, Chicago, Nov 15 2026). **Full paper deadline: Aug 5, 2026 (AoE). Late-breaking/short paper:
  Aug 26 (AoE).** Notification Sep 2 / Sep 9. Camera-ready Sep 25. 8-page short papers. PERFECT fit —
  this is exactly what PMBS publishes. Workshop tier, but cited and indexed. **PRIMARY TARGET.**
- **IISWC 2026** (IEEE International Symposium on Workload Characterization, Boulder CO, Sep 27–29).
  **DEADLINE PASSED: paper submission was May 21, 2026 (no extensions). Rebuttal Jul 6–10. Camera-ready
  Aug 28.** We missed this one — the exact-fit venue. **IISWC 2027 is the fallback** (deadline likely
  ~May 2027).
- **NeurIPS 2026 workshop** (Dec 11–12, Sydney / Dec 12–13, Paris + Atlanta). Workshop acceptance
  notification Jul 11; **suggested workshop paper submission ~Aug 29; accept/reject by Sep 29.**
  ES-FoMo ran at ICML 2023/2024/2025 — no NeurIPS edition announced yet. Watch for an efficient-
  systems workshop in the accepted list after Jul 11. If one appears, deadline will be ~Aug–Sep.
- **MLSys 2027** (May 17–22, 2027; deadline likely ~Oct 2026 based on prior years). Full venue,
  higher bar. The characterization angle needs a "so what" punchline (the Arm 2 negative provides it).
- **arXiv first** — post immediately for timestamp priority; submit to venue in parallel. The
  dataset itself has value regardless of venue acceptance.

**Effort:** ~2–3 weeks to write (data already exists; figures already created). No new GPU time needed.
This is the lowest-risk, highest-confidence path.

### Path B: Arm 2 NVFP4 compute paper (MEDIUM CONFIDENCE, ~2–6 weeks)

**Title candidate:** "Does native FP4 compute help MLA decode? A pre-registered boundary result
on Blackwell"

**Thesis:** MLA decode is the only attention shape where M=128 by construction, meeting the NVFP4
block-scaled MMA M≥128 gate with zero padding waste. We test whether this advantage converts to a
throughput win. Pre-registered prediction: it won't (decode is HBM/work-starved, not compute-bound).

**The staged plan (kill-early, every stage publishes):**

| Stage | What | Effort | Kill condition |
|---|---|---|---|
| A — GEMM crossover | FP4 vs FP8 block-scaled GEMM at M=128/K=256 via CUTLASS profiler | 3 days | FP4 ≤ FP8 → publish the negative |
| B — Accuracy | QK-in-NVFP4 quality on GPT-2 KV (reuse v10 recipes) | 3 days | Accuracy unrecoverable |
| C — Fused kernel | Fork ex77, plumb block-scale SF into QK mainloop | 3–6 weeks | Only if A+B pass |

**Why even a negative is publishable:**
- Nobody has tested FP4 compute on M=128 MLA decode (confirmed: CUTLASS ex77 is fp8/fp16 only;
  FlashInfer/vLLM do NVFP4 storage not compute).
- The negative resolves TRT-LLM #4412's M-padding observation for the MLA case specifically.
- "The padding penalty was a red herring for decode" is a falsifiable, useful finding.
- Combines with Path A for a stronger paper (the Arm 2 result gives the characterization a "so what").

**Effort:** Stage A = 3 days on the B300 you already rent. If it's a negative (LIKELY), publish
within 2 weeks. If positive, commit to Stage C (3–6 weeks).

### Path C: Alternative novel directions (EXPLORATORY, higher risk/reward)

These are genuinely unexplored but require more new work:

**C1. Cross-generation roofline spine (T4→A100→B200→B300)**
- The "~40 GB/s per-CTA ceiling on three architectures" finding is already in the data.
- A paper showing the per-CTA ceiling is architecture-INDEPENDENT (same ~40 GB/s despite 25×
  HBM bandwidth jump) would be novel. Need to formalize + add the A100 data point cleanly.
- Effort: ~1 week (data exists, need clean A100 measurement + writeup).

**C2. GLA / linear attention decode characterization on sm_103**
- Tri Dao's GLA paper (arXiv 2505.21487) did the roofline on H100. Nobody has done it on B300.
- The tcgen05 pipeline-fill story may differ for GLA (different compute pattern).
- Effort: ~2–3 weeks (need to build/port a GLA kernel or benchmark FlashInfer's).
- Risk: may be too derivative of Tri Dao's own work.

**C3. Sparse attention decode (DSA/CSA) on sm_103**
- DeepSeek V3.2 (DSA = Dynamic Sparse Attention) and V4 (CSA = Context-Sparse Attention) are
  the field's direction. No roofline characterization published.
- Effort: HIGH (sparse patterns change the roofline model fundamentally).
- Reward: HIGH (if you can characterize sparse decode on Blackwell, that's genuinely novel).

**C4. The "prediction methodology" paper**
- Frame the two-layer model itself as the contribution: "A practitioner's guide to predicting
  attention kernel performance on modern GPUs."
- Systematic comparison of pure roofline vs per-CTA-corrected across 12 kernels.
- Risk: methodology papers are hard to publish without a top-tier systems venue.

---

## 3. Recommended plan (6 weeks, 4–5 hr/day)

**Week 1–2: Execute Paths A + B-Stage-A in parallel**

| Day | Path A (characterization paper) | Path B (Arm 2 GEMM test) |
|---|---|---|
| 1–2 | Outline + abstract + figure selection | Set up CUTLASS profiler on B300 |
| 3–4 | Write Sections 1–3 (intro, background, methodology) | Run M=128/K=256 FP4-vs-FP8 sweep |
| 5–7 | Write Sections 4–5 (results, analysis) | Analyze → decide kill/proceed |
| 8–10 | Write Section 6 (discussion) + polish | If negative: write up the finding |
| 11–14 | Internal review, figures, submission prep | Fold into Path A or standalone |

**Week 3–4: Depending on Stage A result**

- If Stage A **negative** (LIKELY): Path A paper is the primary deliverable. Polish, submit to
  arXiv + PMBS@SC. The Arm 2 negative becomes Section 5.3 ("Does native FP4 compute help?").
- If Stage A **positive**: Run Stage B (accuracy, 3 days). If accuracy holds, commit to Stage C.

**Week 5–6: Submission + hardening** (target: **arXiv by ~Jul 28, PMBS by Aug 5**)

- Submit to arXiv (timestamp priority — before PMBS deadline).
- Submit to **PMBS@SC 2026** (full paper Aug 5; late-breaking Aug 26 as fallback).
  IISWC 2026 is MISSED (May 21 deadline passed). IISWC 2027 (~May 2027) is the next chance.
- Watch NeurIPS 2026 workshop acceptances (Jul 11) for an efficient-systems workshop; if one
  appears, submit by ~Aug 29.
- MLSys 2027 deadline (~Oct 2026) as the stretch goal if the paper + Arm 2 negative are strong.
- Optional: run the clean A100 data point for the cross-generation spine (Path C1).
- Optional: ncu on a privileged B300 (belt-and-suspenders, not gating).

---

## 4. The honest bottom line

**The safest high-value path is Path A (pure characterization) + Path B Stage A (the quick GEMM
crossover test).** Together they produce a paper in ~3 weeks with novel content (the empty cell +
the negative result). The characterization paper stands alone; the Arm 2 result strengthens it.

**Don't inflate the scope.** This is a characterization/workshop paper, not a top-tier systems
contribution. The methodology isn't novel (arXiv 2605.04178 owns it — cite them); the application
to this specific empty cell (MLA decode on sm_103 across precisions) is. Frame it as
"complementing, not beating" production kernels.

**Watch Attn-QAT (arXiv 2603.00040).** This is the closest prior art for Arm 2 — they do FP4 QK
on Blackwell. But they benchmark prefill only, no decode, no M-padding analysis. If they publish a
decode extension before your submission, the Arm 2 angle narrows. Move fast on Stage A.

**The strongest claim you can make:** "We show that attention decode's apparent per-CTA ceiling
is work-starvation, not a fixed architectural limit, through a 12-step prediction-vs-measured
roofline study on Blackwell — the first such open characterization on sm_103."

---

## 5. Figures for the paper (already created)

| Figure | File | Purpose |
|---|---|---|
| 1. v12 throughput regime | `diagrams/v12-throughput-regime.svg` | Central result: TFLOP/s vs B×K |
| 2. Engine-ridge comparison | `diagrams/v12-engine-ridge-comparison.svg` | Why NVFP4 overshoots into HBM-bound |
| 3. Work-starvation correction | `diagrams/v12-work-starvation-correction.svg` | The v6→v12 flat→scaling comparison |
| 4. v1→v12 arc summary | `diagrams/v1-v12-arc-summary.svg` | The 12-step journey overview |
| 5. Two-layer prediction model | `diagrams/v12-two-layer-prediction-model.svg` | Methodological contribution |
| 6. Paper positioning | `diagrams/v12-paper-positioning.svg` | Landscape + empty cell |
| 7. Arm 2 research plan | `diagrams/v12-arm2-research-plan.svg` | Staged kill-early decision tree |
| 8. Related work landscape | `diagrams/v12-related-work-landscape.svg` | Prior art table + empty cell delta |

Plus 24 existing diagrams from v6→v11 (see `docs/diagrams/`).

---

## 6. Web-verified novelty assessment (2026-06-30)

### Claim 1: "No published paper has roofline-grade characterization of attention decode on B300/sm_103"

**CONFIRMED.** Web search found no paper combining "roofline" + "attention decode" + "B300" or
"sm_103" + "prediction-vs-measured." The closest work:
- **arXiv 2512.01644** ("A Systematic Characterization of LLM Inference on GPUs", Dec 2025) — does
  roofline analysis of LLM inference phases including decode, but on older GPUs (A6000, not Blackwell).
- **RooflineBench** (arXiv 2602.11506, 2026) — roofline framework for on-device LLMs, mobile/edge focus.
- **arXiv 2605.04178** — the roofline methodology paper we already cite as prior art.
- **FA4** (arXiv 2603.05451) — roofline analysis of Blackwell *prefill*, not decode. Identifies SMEM
  traffic + exp as dominant (25–60% of execution). Benchmarks B200, never characterizes sm_103 decode.
- **FlashInfer / FlashMLA** — ship B300 decode kernels but publish no roofline analysis or prediction-
  vs-measured data.

**The empty cell exists.** Our delta vs Tri Dao's GLA (arXiv 2505.21487, H100 MLA decode roofline):
sm_103 + KV-quant (FP8/NVFP4) + cross-attention-type (MHA→GQA→MLA).

### Claim 2: "No existing kernel does native FP4 *compute* for attention decode (Arm 2)"

**CONFIRMED with nuance.** CUTLASS example 77 supports fp16, bf16, and fp8 for MLA decode — **no
FP4 compute path.** CUTLASS does ship block-scaled NVFP4 GEMMs (for MoE/linear layers, SM120), but
these are not fused into the attention kernel. FlashInfer supports NVFP4 *KV storage* (dequant to
FP16/FP8 for compute) via vLLM's `--kv-cache-dtype nvfp4`, but the QK matmul itself runs in FP8 or
FP16. The vultr/deepseek-v4-nvfp4-kernel repo (hand-written CuTeDSL) targets MoE, not attention.

**Arm 2 (native FP4 *compute* in the QK matmul of MLA decode) is genuinely unbuilt.**

### Claim 3: "MLA decode is the only attention shape with M=128 by construction (zero FP4 padding)"

**CONFIRMED.** CUTLASS ex77 source has `static_assert(TileShapeH==128)` — the MLA kernel requires
M=128 (all h_q heads packed). Normal decode is M=1 → FP4 block-scaled MMA needs padding to M=128
(128× wasted work per TRT-LLM #4412). No other attention shape provides M=128 at decode time
without padding.

### Claim 4: "FlashInfer/FlashMLA are faster — we complement, don't beat"

**CONFIRMED.** FlashInfer's trtllm-gen NVFP4 decode was ~3× faster than our v10 (measured in the
v10 B300 pass). vLLM blog (2026-02-13) documents DeepSeek-V3.2 on GB300 using FlashInfer with
native FP4 MoE + TMA attention. We measured NVIDIA's canonical ex77 kernel, not a SOTA claim.

### Related work landscape update (deep-research workflow, 22 sources, 104 claims)

**Closest prior art for Arm 2 (FP4 compute in attention):**
- **Attn-QAT** (arXiv 2603.00040, Hao AI Lab @ UCSD): THE closest work. Uses native FP4 tensor
  core MMA for the QK matmul on RTX 5090 (consumer Blackwell). **BUT:** (a) only QK is FP4 — PV
  stays BF16 (confirmed by Hao AI Lab blog: "we choose to run block-scaled NVFP4 QK and BF16 PV
  on a B200"); (b) benchmarks prefill/throughput ONLY — **no decode-specific (M=1 or M=128)
  characterization**; (c) vLLM integration is "fake quantization" for evaluation, not production
  FP4 compute. **Our Arm 2 delta: MLA-shaped M=128 decode specifically, which Attn-QAT doesn't
  test. The prefill-only gap and M-padding issue remain unaddressed.**

**MLA decode — closest prior:**
- **SnapMLA** (arXiv 2602.10718): "First open-source FP8 decoding framework tailored to MLA
  decoding." Uses FP8 tensor cores for actual QK/PV matmuls (not just storage). **Hopper ONLY
  (not Blackwell/sm_103).** No FP4/NVFP4 path. No roofline analysis. **Our delta: sm_103 +
  NVFP4 + roofline.**
- **arXiv 2506.02523** (MLA hardware acceleration analysis): Claims "first hardware-acceleration-
  systems analysis of MLA." **Purely analytical** (Stream DSE framework, Edge TPU / Apple A17
  reference points) — NO GPU kernel implementation, NO real measurements, NO comparison to
  FlashInfer/FlashMLA/CUTLASS. Different abstraction level entirely. **Does NOT fill our cell.**

**Roofline methodology — who owns it:**
- **arXiv 2605.04178** (microbenchmark-driven GPU perf model, May 2026): Achieves 1.31% MAE on
  B200 vs naive roofline >95% error. **Covers B200 (sm_100), NOT B300 (sm_103).** Covers GEMM
  workloads, NOT attention kernels. **They own the methodology claim** (3 reasons naive roofline
  fails). Our paper CANNOT claim methodological novelty — must cite them and frame our two-layer
  model as an *application* of established methodology to a new workload + architecture.
- **arXiv 2512.02189** (Blackwell MMA microbenchmarks): FP4 tensor cores achieve 7700.2 TFLOPS
  at 96.2% peak on B200 (m64n8k16 tile). **GEMM-level only**, not attention. Useful as a
  reference for Arm 2 Stage A expectations.

**Production state confirmed:**
- **TensorRT-LLM** (NVIDIA blog, DeepSeek V3.2 on Blackwell): Uses "Per-tensor FP8 Math" for
  Sparse MLA attention, "NVFP4" for output projection weights. Confirmed: NVFP4 is storage for
  weights, FP8 is compute for attention. **No NVFP4 compute-path in attention.**
- **vLLM FP8 KV-cache** (blog post): FP8 E4M3 for BOTH storage AND compute in attention (QK +
  ScoreV matmuls). No NVFP4 compute-path mentioned.
- **FlashInfer FP4**: Benchmarks on MoE forward passes only. No FP4 compute-path attention.

**FP4 attention (prefill-only, not decode):**
- **SageAttention3** (arXiv 2505.11594, NeurIPS 2025 Spotlight): FP4 attention using microscaling
  on RTX 5090 (SM120, consumer Blackwell). Achieves 1038 TOPS = 5× FA on RTX 5090. **Prefill-only
  (video generation inference), NO decode, NO MLA, NO datacenter Blackwell (B200/B300/sm_103), NO
  roofline analysis.** Our delta: decode + MLA + sm_103 + roofline.
- **Florian Mattana blog** (2026-03-17, open-source): FP4 fused attention on SM120 using
  `mma.sync.aligned.kind::mxf8f6f4.block_scale` PTX. Prefill (processes 64 tokens/tile). Achieves
  2–3 TFLOPS vs 474 theoretical (quantization overhead is the bottleneck, per NCU). **SM120 only**
  (explicitly notes SM100 datacenter uses different instructions `tcgen05.mma`). **NO decode, NO
  MLA, NO sm_103, NO prediction-vs-measured roofline.**

**Other (orthogonal):**
- **MixFP4** (arXiv 2605.31035): adaptive FP4/INT4 block representations for weight quant.
- **RaZeR** (arXiv 2501.04052): redundant zero remapping for NVFP4 weight quant.
- **NVIDIA blog** ("NVFP4 KV Cache"): NVFP4 KV *storage* optimization, not compute.

**Bottom line (updated 2026-07-01 with SageAttention3 + Mattana):** the Arm 2 direction (native FP4
tensor-core compute in the QK matmul of MLA decode) remains novel. Three groups have built FP4
*attention* kernels (Attn-QAT, SageAttention3, Mattana) — ALL are prefill-only, standard attention,
consumer Blackwell. Nobody has done FP4 decode, FP4 MLA, or FP4 attention on datacenter
sm_100/sm_103. The likely negative result is itself publishable.

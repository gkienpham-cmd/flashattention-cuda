# v12 kickoff — native tcgen05 tensor-core MLA decode on B300/sm_103a (close the 4× gap v11 measured)

> Paste-in starter for a fresh session. Follows the per-step loop in `ROADMAP.md`
> (roofline → explain → write → correctness → bench → results → decisions → quiz).
> Supersedes the now-complete `v11-kickoff.md`. Read `CLAUDE.md` Step 11 + its close-out,
> `results.md`/`decisions.md` Step 11 close-out, and `interview-prep.md` C17 first.

---

## 0. The one-line goal

**v12 changes ONE variable from v11: the compute ENGINE.** v11 proved the MLA decode *shape* is correct
(M=128 by construction, AI lifted ~30×, capacity 202×, correctness incl. the absorption identity) but
measured it **per-CTA-bound at 0.75 TFLOP/s — ~4× slower than torch dense-MQA** because our kernel runs a
**warp-per-head CUDA-core GEMV** while torch routes the M=128 head-pack into a **cuBLAS tensor-core GEMM**.
v12 puts the M=128 QK and PV matmuls onto **Blackwell 5th-gen tensor cores (tcgen05)** — first in FP8, then
native NVFP4 — to close that self-gap. Everything else (split-KV, LSE merge, score-stationary outer
structure, NVFP4 latent storage, `choose_splits`, the paged pool layout) stays **byte-identical** to v11 so
the A/B isolates the engine.

**This is the data-motivated arm the v11 kickoff deferred as "ONLY-IF."** v11's measured 4× gap promoted it
from conditional to primary (see `decisions.md` Step 11 close-out, criterion 1).

---

## 1. Roofline first (record the prediction BEFORE coding — two layers + a counter)

Reuse `roofline/model.py` MLA branch. At the real DeepSeek shape (L=512, R=64, DQK=576, h_q=128) on B300
(`python -m roofline.predict --arch sm_103 --mla --h-q 128`):

| precision | decode AI (≈3.78·h_q/b) | vs B300 FP16-TC ridge 312.5 | pure-roofline limiter |
|---|---|---|---|
| fp16 | 234.8 | 0.75× | HBM (knife-edge) |
| fp8  | 469.7 | 1.50× | **MMA (compute)** |
| nvfp4 | 835.0 | 2.67× | **MMA (compute)** |

- **PURE ROOFLINE (layer 1):** v11 already showed fp8/nvfp4 AI sit *past* the ridge → tensor-core compute
  is the predicted limiter. v11's CUDA-core kernel could not realize it; **v12's tcgen05 kernel can attempt
  it.** So layer 1 says: with real tensor cores, v12 should move toward the compute roof.
- **PER-CTA-CORRECTED (layer 2 — the prediction I actually believe):** single-token decode carries only
  **1–4 MMAs in flight** per CTA; the tcgen05 pipeline wants **256–1024** to saturate (web:tcgen05-cutlass).
  So even on tensor cores, v12 is predicted **SMEM-bandwidth / MMA-pipeline-depth-bound**, NOT FP4-FLOP-bound
  — the realized throughput will land **well below the 5 PF (FP8) / 15 PF (NVFP4) peak**, and v12 will
  **NOT** beat FlashMLA's ~410 TFLOP/s decode (~3 orders above v11's 0.75) on wall-clock.
- **COUNTER-PREDICTION (the prize, either sign publishable):** if v12 *does* approach a compute- or
  SMEM-BW-bound regime (achieved TFLOP/s climbs > ~10× v11, %SMEM-BW high), the limiter finally **left
  per-CTA** — the first time in the arc. If it stays at the per-CTA floor (~0.75 TFLOP/s, MMAs starved),
  then **even the right engine can't fill a single-token decode CTA**, which is itself the paper-grade
  prediction-vs-measured result and motivates speculative q_len>1 (the fallback shape-lever).

**Record all three in `results.md` Step 12 before writing the kernel** (the loop's Step 1).

---

## 2. The build (single variable = the engine; fork CUTLASS, don't hand-roll tcgen05)

**Do NOT hand-write tcgen05/tmem PTX.** Fork the proven kernel:

1. **Base: CUTLASS example 77 (`77_blackwell_fmha`)** — the weight-absorbed latent-512 / rope-64 MLA
   **decode** kernel. **NOT** the FA4 prefill kernel (it has no decode path). Use **2-SM `cta_group::2`** for
   the 512-wide latent accumulator (ex77's reason for 2-SM — the latent doesn't fit one SM's tmem alone).
   `[verify the ex77 variant + CUTLASS version on the target box]`
2. **Arm 1 (lands first) = FP8-dense MMA.** Keep v11's paged NVFP4 latent pool on HBM; **dequant
   NVFP4→FP8/bf16 at the SMEM stage**, then feed the tcgen05 **FP8 MMA (gate M≥64)**. This is the
   FlashMLA-proven path (FlashMLA dequants its KV to FP8 before the MMA; nobody runs native-FP4 *compute* on
   KV today — they run FP8 compute on dequantized-from-FP4 KV). Scores/softmax stay **≥ FP16** (FP4 scores
   collapse softmax — proven in v10).
3. **Arm 2 (gated on Arm 1 working) = native NVFP4 MMA.** `kind::mxf4nvf4`, **M=128 / K=256 / TN-only +
   per-16 E4M3 microscales** — the headline-novel but high-risk arm. TRT-LLM #4412 reports FP4Gemm *slower*
   than FP8Gemm at decode M-sizes (128-token M-padding waste), so **Arm 2 may lose to Arm 1 even on tensor
   cores** — keep the gate, set expectations accordingly. The open lane is NVFP4 *KV-cache* compute for MLA
   decode (production ships NVFP4 *weights* + FP8 *KV*); that's the genuine novelty, also the riskiest bet.

**The A/B that isolates the engine:** same NVFP4 latent bytes, same split-KV/merge/score-stationary
structure as v11; only the inner matmul changes (CUDA-core GEMV → tcgen05 MMA). Same-session, clock-matched
`vs v11` is the trustworthy number.

---

## 3. §9 questions to resolve on the dev rung (gate the NVFP4 arm on these)

- **Q1 — does M=128 pack as ONE tcgen05 GEMM?** v11 validated this *indirectly* (cuBLAS beats us 4× on the
  shape) but never directly (no tcgen05 path ran). **Build risk:** CUTLASS ex77's realized M-blocking
  `num_groups` is reportedly capped at **32**, not 128 — verify the head-count→M=128 tile mapping on the
  target CUTLASS version **before** banking the native-FP4 arm. If M is split into 16/group, RoPE must split
  the K-dim not the M-tile (the §9-Q1 structural argument — re-confirm in the ex77 source).
- **Q2 — does the limiter flip, or rename to SMEM-BW?** This is the deliverable. Measure achieved TFLOP/s +
  the SMEM-load-cycles vs math-cycles ratio (ncu `smsp__inst_executed_pipe_tensor` + `l1tex` SMEM
  throughput). Predicted: **renames to SMEM-BW / pipeline-depth**, not a clean compute flip.
- **Q3 — sm_103 2×-exp at M=128 (the carried-over re-ablation).** v10/v11 measured EX2 = 5.33 TExp/s =
  0.50× the 10.7 claim, but at **M=1** (a dependent chain, <3% of decode) → irrelevant. At M=128 the exp
  share rises and the chain caveat lifts — **re-ablate** (this is where FA4's prefill 2×-exp concern would
  actually bite). A clean "still 0.5× / now Nx" answer completes the 2×-exp story.

---

## 4. Hardware, gates, and the honesty debts to PAY this time

- **Hardware:** `[RENT root / bare-metal B300 / sm_103a]` (CUDA 12.9+, CUTLASS 4.x). The dev rung can be
  B200/sm_100 (cheaper, same tcgen05 family) to de-risk the build; the **record runs on B300**.
- **Gate 1 (dev rung):** build clean on sm_103a (or sm_100), correctness vs the v11 oracle (tol 5e-2), the
  absorption identity, and the FP8-arm A/B vs v11. **Pay v11's debts here:** **install + run ncu** (the
  unprivileged container blocked it — needs a privileged/bare-metal box) and **lock clocks** (not
  idle-pinned).
- **Gate 2 (the measured core + quiz):** (1) the achieved-TFLOP/s + SMEM-BW regime sweep vs N_k past L2,
  clock-locked, **ncu-validated** (turns the counter-free proxy into a profiled result — the single
  highest-value hardening item, closes the deepest reviewer wound); (2) same-session clock-matched `vs v11`
  (the engine A/B) and `vs FlashMLA / FlashInfer trtllm-gen` at matched precision; (3) the Q3 2×-exp
  re-ablation; (4) nsys 2025.3.2+ in-notebook (keep the provenance paid); (5) the quiz.

**The four honesty debts to clear (carried from v11):** ncu on a privileged B300; clock-locked regime run;
a **torch-side profile** of the dense-MQA baseline (to convert the "cuBLAS-TC-GEMM 4× cause" from inference
to measurement — and re-run the torch baseline at fp16/bf16, since v11's was FP32 batched-GEMV); the M=128
2×-exp re-ablation.

---

## 5. The honest prediction-vs-measured framing (this IS the contribution)

**v12 is NOT a "beat FlashMLA" attempt — say so up front.** The pre-registered result is the *shortfall*:
even with the right engine, single-token decode is SMEM-BW/pipeline-depth-bound, so v12 closes v11's own 4×
self-gap toward the realizable TC ceiling but **does not reach the FP4 FLOP peak and does not beat the
production kernels' wall-clock**. That measured shortfall — *why even tensor cores can't fill a decode CTA*
— is exactly the open cell no production kernel publishes. Frame as **complementing, not beating**
FlashInfer / FlashMLA / cuDNN.

Comparators to benchmark (clock-locked, matched precision): **FlashMLA** (~410 TFLOP/s SM100 decode, FP8 KV,
the tensor-core SOTA), **FlashInfer trtllm-gen** (the default sm_103 attention backend, FP8 KV; NVFP4-*KV*
is "being developed," not shipped — our NVFP4-KV-compute arm is the genuine open lane).

---

## 6. Paper positioning (characterization-grade, verified honest) + the experiments that exceed FA4

**Defensible novelty (survives a hostile reviewer):** *the first open, kernel-level, prediction-vs-measured
roofline characterization of attention **decode** on sm_103, across MHA→GQA→MLA and FP16→FP8→NVFP4.* NOT
"first to run" (vLLM/SGLang/FA4's kernel do, Feb 2026), NOT "beat" anyone. The *methodology* is not novel
(arXiv 2605.04178 et al.); novelty = the {open + sm_103 + decode + measured-findings} intersection. **Closest
prior = Tri Dao's GLA (arXiv 2505.21487)** — an open roofline-documented MLA-decode study on H100; our delta
is exactly **sm_103 + KV-quant**. A reviewer *will* cite GLA — name it first.

**The four reviewer-resistant deltas to lead with** (chip-specific measured findings, not techniques):
1. per-CTA-bound decode on sm_103, confound-free past a 5.1× L2 overflow (contradicts the "decode is
   HBM-bound" wisdom; BLASST arXiv 2512.12087);
2. the **2×-exp claim measured at 0.50×** and irrelevant at M=1 (engages FA4's own footnote — the most
   reviewer-resistant result; re-ablate at M=128 in v12);
3. **NVFP4 KV latency-negative past L2** — bytes are not the decode lever;
4. measured B300 constants (**L2 = 132.6 MB**, 148 SMs enabled, ~2032 MHz) — sm_103 is academically
   uncharacterized.

**The FA4 thesis bridge (one sentence for the paper):** *FA4 redesigned the attention pipeline around
Blackwell's asymmetric scaling but only for prefill/training, where the score matrix is square (M=N=128) and
the limiter is SMEM+exp; decode (M=1) is per-CTA-bound and untouched — and MLA latent-KV decode raises M to
h_q=128, reproducing exactly the tile shape FA4 designed for, which makes its exp-emulation,
correction-warpgroup overlap, and tcgen05 MMA portable to decode for the first time, while NVFP4 latent KV
and the measured 2×-exp / per-CTA-limiter characterization on sm_103 fill cells no FlashAttention paper has.*

**Experiments that lift "nice study" → "groundbreaking-for-a-characterization-paper"** (ranked by
leverage/effort, from the deep-research synthesis):
1. **Three-generation decode roofline spine: H100 → B200/sm_100 → B300/sm_103, prediction-vs-measured** (the
   strongest single move — dilutes "just a new SKU"; B200+T4 data already exist, add H100). 
2. **One ncu-validated B300 counter** (upgrades every counter-free reading to profiler-validated).
3. **v12 itself** (the tcgen05 kernel measured vs FlashMLA/FlashInfer — the SMEM-BW shortfall is the result).
4. **2×-exp re-ablation at M=128** (completes the FA4-footnote engagement).
5. **Asymmetric recipe on a known-attention-sink model** (firms the conditional V-token lever; keep it a
   negative/conditional result — the weakest axis).
6. **Port FA4's software-emulated exp + conditional-rescale into the MLA inner loop, re-tuned for sm_103**
   (the emulation fraction is a B200 number; B300's 2×-exp changes the optimum — a knob no one has reported).

**Venue path (re-dated — IISWC 2026 deadline passed):** **PMBS@SC26** (the prediction-vs-measured
performance-modeling workshop — perfect topical fit, likely live) + **ES-FoMo/ENLSP** (fast feedback,
negative-results-friendly) near-term; **IISWC 2027 / MLSys 2027** as the strengthened archival targets after
experiments 1+2+3. Avoid top-tier systems (MLSys/ASPLOS/ISCA) as-is — the steelman desk-rejects a SKU-bump
characterization; only fits if pivoted to a Blackwell-Ultra µarch paper (lead the 2×-exp refutation +
per-CTA + TMEM).

**Working title:** *Per-CTA, Not Bandwidth-Bound: An Open Prediction-vs-Measured Roofline Characterization
of Attention Decode on NVIDIA Blackwell Ultra (sm_103).*

---

## 7. What changes on another arch / what's `sm_103a`-specific

- The **M≥128 NVFP4-compute gate** is `sm_103a`-specific (no `mma.sync` FP4 fallback on datacenter
  Blackwell; the warp-level m16n8k64 FP4 path is SM120+ only, low-throughput).
- The **SMEM-BW / pipeline-depth wall** (the predicted v12 limiter) is **arch-general** — it's a property of
  single-token decode, not of sm_103. That generality is what makes the result transferable (and the
  three-generation spine the strongest experiment).
- The **AI lift + capacity win** are arch-independent; whether the flip *realizes* depends on tmem/SMEM per
  SM (228 KB Blackwell) and the tcgen05 gate.

---

## 8. Fold-ins and the fallback (don't make these their own steps)

- **Occupancy-v8.8** (4 blocks/SM, measured ~1.4× at B≥32 past L2 — the serving regime) → fold into v12's
  residency tuning, not a separate step.
- **Speculative / multi-token (q_len>1)** → the fallback shape-lever, only if the tcgen05 dev-rung shows
  M=128 fragments into too-small tiles. MLA already raised M, so this is redundant unless Arm 1+2 both fail.
- **GLA / sparse (DSA / CSA)** → v13 (a different shape, larger scope; dense MLA is the roofline-clean rung
  first). The field's direction (Tri Dao GLA beats FlashMLA ~20% @ q_len=1; DeepSeek V3.2 DSA / V4 CSA) makes
  this the likely v13.

---

> **Author-machine constraint:** the Mac has no CUDA toolchain — build + correctness + bench + profile + quiz
> all run on the rented B300 (dev rung B200 optional). Author here: the roofline prediction (recorded first),
> the kernel + binding + wiring, the tests, and the gate notebook; measure there.

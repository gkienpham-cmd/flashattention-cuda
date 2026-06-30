# Arm 2 — native NVFP4 KV-cache *compute* for MLA decode: research + journey plan

> The one genuinely-novel path identified in the v12 close-out. This doc grounds it in the current
> (June 2026) state of the art, states the honest thesis + the honest risk, and lays out a **staged,
> kill-early** journey where every stage is a publishable checkpoint — so you don't gamble months of
> CUTLASS work on a single all-or-nothing kernel. Follows the project's per-step loop (roofline →
> build → measure → record).

---

## 1. The thesis (why this is novel, and the one structural insight that makes it plausible)

**What ships today (researched, not assumed):**
- Production NVFP4 inference = **FP4 weights + FP8 KV** (DeepSeek-V3/V4 NVFP4 builds, TRT-LLM, the Azure/MS
  DeepSeek-NVFP4 writeup). KV is FP8, *not* FP4, for compute.
- **NVFP4 KV *storage*** is now landing (vLLM `--kv-cache-dtype nvfp4`, issue #43562) — but it's
  **store-FP4, dequant-to-higher-precision-to-compute**, exactly what v10/v11 did. Buggy on consumer SM120.
- **NVFP4 KV *compute*** (feed FP4 KV straight into the tensor-core MMA, fused in the attention kernel) is
  **shipped by nobody.** CUTLASS ex77's MLA decode kernel is instantiated only for `float_e4m3_t`/`half_t`
  (we confirmed this in the source) — no FP4 path.

**Why FP4 *compute* loses at decode everywhere else (TRT-LLM #4412, confirmed):** the block-scaled FP4
tensor-core MMA gates at **M ≥ 128**. Normal decode is **M = 1**, so the quantizer **pads M 1→128** — doing
128× the work for one real row. That padding waste is *why* "FP4Gemm is slower than FP8Gemm during decode."

**The structural insight (the novel claim):** **MLA decode has M = h_q = 128 BY CONSTRUCTION** — the exact
property v11 established (all query heads share one latent → they pack the M-tile). So **MLA decode is the
single attention shape where the FP4 `M ≥ 128` gate is met with ZERO padding waste.** Every "FP4 decode is
slow" result in the literature is about M=1-padded attention; none of them tested the one shape that
naturally fills M=128. *That gap is the contribution* — and nobody has looked because MLA + native-FP4-KV-
compute is a niche intersection.

---

## 2. The honest prediction (brutal — this is likely a NEGATIVE/boundary result, and that's still a paper)

v12 measured the decisive fact: **MLA decode is not compute-bound** — FP8 ≈ FP16 throughput, and at
single-stream it's at 1–2% of peak (work-starved), trending to **HBM-bound (~46%)** only at large B×K.
NVFP4 *compute* only helps a kernel that is **compute-bound**. So:

- **Pre-registered prediction:** native FP4 compute will **NOT** beat FP8 compute on MLA-decode *latency*,
  because the padding waste was never the real decode bottleneck — **work-starvation / HBM is.** Eliminating
  the M=128 padding (the #4412 problem) removes a penalty that doesn't bind here. Most likely outcome:
  FP4-compute ≈ FP8-compute at small batch, and at large batch both become HBM-bound where **the win that
  matters is the FP4 *byte* read (half of FP8), not the FP4 *math*** — which you already get from FP4
  *storage* without FP4 compute.
- **The counter (the prize, low probability but real):** there may be a **compute-leaning band** — very
  large batch × moderate context, where the kernel sits near the compute roof — in which FP4's 3× MMA
  throughput over FP8 finally converts. Finding that band (or proving it's empty for MLA decode) is the
  result.
- **Either way it's publishable** *because the artifact is novel*: "the first native-NVFP4-KV-compute MLA
  decode kernel, and a characterization of the FP4-vs-FP8 compute crossover" — a clean positive *or* a
  clean negative both fill an empty cell. A negative result with a working novel kernel + a sharp "here's
  why decode doesn't benefit from FP4 math, M=128 notwithstanding" is a legitimate workshop paper.

**Do not oversell.** This is a narrow contribution at a characterization/workshop tier (PMBS@SC / ES-FoMo /
IISWC), not a top-tier systems paper. The realistic win is "real novel artifact + honest boundary result,"
not "we beat FlashMLA."

---

## 3. The build constraints (researched — what the kernel must satisfy)

From the CUTLASS block-scaled docs + Colfax tutorials:
- **MMA gate:** `mxf4nvf4`, **bM ≥ 128** (1-CTA) / **≥ 256** (2-CTA pair-UMMA), **bK = 256** (4 UMMA atoms
  × 64 B), **TN-only** (K-major operands), **N ≥ 128** for the SFB tiling (sub-128 needs workaround logic).
- **Scale factors:** per-16 elements, **E4M3** (`.scale_vec::4X`), loaded via `tcgen05.cp` `.multicast =
  warpx4` (duplicated to all 32 lanes). Layout is fiddly (SF for adjacent UMMA atoms 16 columns apart).
- **The PV GEMM cannot be FP4.** Post-softmax probs need **≥ FP16** (FP4 scores collapse softmax — v10
  proved it; corroborated by the literature). So Arm 2 is **QK-in-NVFP4 only**: `q_absorbed`(FP4) ·
  `latent`(FP4, block-scaled) → scores; softmax ≥ FP16; **PV stays dequant-to-FP8/FP16**. This also means
  **q_absorbed must be quantized to FP4** (both MMA operands FP4) — a new query-quant error to measure.
- **MLA shape fit:** QK contraction is over `L+R = 576` → tiles of K=256 fit (576 = 256+256+64; the rope-64
  tail is the awkward bit — likely a separate small accumulate, as v11 noted RoPE splits the K-dim). M=128
  ✓ by construction. N = KV-tile ≥ 128 ✓. So the QK GEMM *structurally fits* the FP4 gate — the first
  attention GEMM in the arc that does.

**Hardware:** an **unprivileged B300/sm_103a is fine** (GEMM/throughput benchmarks need no ncu/privilege —
the thing you can't get). A **consumer SM120 Blackwell (RTX 50-series / RTX Pro)** is a *much cheaper* dev
rung for the FP4 GEMM micro-tests (it has block-scaled FP4 via `mma.sync.block_scale`, not tcgen05 — note
the SM120 path differs and has known bugs, FlashInfer #2723 / vLLM #43562, so port carefully). Record on B300.

---

## 4. The staged journey (kill-early; every stage is a checkpoint that publishes)

### Stage A — GEMM-level decisive test (CHEAP, no kernel surgery, answers the core question first)
**Question:** at the *MLA-decode QK shape* (M=128 natural, K=256, sweep N = KV-tile, sweep batch), does the
**FP4 block-scaled GEMM beat the FP8 GEMM** — i.e. does M=128 eliminate the #4412 padding penalty?
**How:** use CUTLASS's *existing* block-scaled NVFP4 GEMM example/profiler (`cutlass_profiler` + the SM100
block-scaled GEMM examples — no attention code). Compare FP4 vs FP8 GEMM TFLOP/s across (M=128, K=256,
N∈{128…8192}, batch∈{1…256}).
**Roofline first:** record the prediction (FP4 MMA peak 3× FP8 → FP4 wins *iff* compute-bound at that
shape; at decode-ish N it's likely launch/latency-bound → ~tie).
**Decision gate:**
- FP4 GEMM **never beats** FP8 at these shapes → **Arm 2 is dead for latency; STOP and publish the
  GEMM-level negative** ("M=128 removes the padding penalty but FP4 decode still doesn't win — because the
  shape is latency/HBM-bound, not compute-bound; the #4412 padding was a red herring for decode"). Novel,
  clean, ~2 weeks of work.
- FP4 GEMM **wins above some batch/N** → you've found the compute-leaning band → proceed to B/C with a
  concrete target regime.
**Cost:** days. **This is the highest-ROI experiment in the whole plan — do it before anything else.**

### Stage B — accuracy of QK-in-NVFP4 for MLA (CHEAP, mostly CPU/any-GPU)
**Question:** does quantizing **both** `q_absorbed` and the latent to NVFP4 for the QK score (softmax ≥ FP16,
PV ≥ FP8) preserve model quality? Reuse `fa_kernels/nvfp4_recipes.py` + the v10 real-KV harness (GPT-2 / a
small DeepSeek-MLA proxy). Measure score-RMSE and end-task vs FP8-KV and FP16. Test the asymmetric recipe
(per-channel-K / per-token; KIVI/KVQuant structure) on the latent.
**Decision gate:** if QK-FP4 blows up accuracy beyond FP8 in a way no recipe recovers → Arm 2 is an
accuracy non-starter → publish the accuracy boundary (also a result). If acceptable → proceed.
**Cost:** days. Can run on any GPU (or T4-emulated, like v10).

### Stage C — the fused kernel (HARD; only if A and B both pass)
**Build:** fork ex77's `Sm100FmhaMlaKernelTmaWarpspecialized`, swap the QK `CollectiveMmaQK` from the plain
builder to the **block-scaled NVFP4 builder** (`make_blockscaled_trivial_tiled_mma` / `MmaMXF4NVF4Op`), plumb
the per-16 E4M3 scale-factor tensors (SFA/SFB) + the `tcgen05.cp` scale loads through the mainloop; keep PV
FP8, softmax ≥ FP16, split-KV/merge/scheduler untouched. **This is the months-of-CUTLASS step** — the SF
plumbing in a warp-specialized kernel is the genuinely hard part.
**Roofline first:** record the AI (FP4 storage 0.5625 B → AI ~835 at h_q=128) vs the FP4 ridge 1875 →
pure-roofline HBM-bound (the v12 reconciliation already showed this); the per-CTA-corrected prediction =
the Stage-A regime answer.
**Measure:** fused FP4-QK MLA decode vs v12's FP8 (`77_blackwell_mla_2sm_fp8`), same B×K sweep; correctness
vs the v11 oracle (tol 5e-2). Wire as the real `kernels/v12_mla_tc/` Arm-2 (the scaffold + dispatch + tests
already exist for it).
**Decision gate / result:** the FP4-vs-FP8 fused crossover curve **is** the paper, positive or negative.

---

## 5. Feasibility, cost, and kill-criteria (honest)

| Stage | Effort | GPU | Kills the project if… | Publishable on its own? |
|---|---|---|---|---|
| A (GEMM) | days | unpriv B300 or SM120 | FP4 GEMM never beats FP8 at M=128/K=256 | **Yes** — the cleanest negative |
| B (accuracy) | days | any/T4 | QK-FP4 accuracy unrecoverable | Yes — accuracy boundary |
| C (fused kernel) | **weeks–months** | unpriv B300 | SF plumbing infeasible / loses to FP8 | Yes — the artifact + crossover |

**Brutal kill-criterion:** if Stage A shows FP4 GEMM ties/loses FP8 at the MLA-decode shape (the *likely*
outcome, given v12 proved decode isn't compute-bound), **do NOT build Stage C.** Write the Stage-A+B
negative as a short characterization paper and call it. Stage C is only worth months if A finds a real
compute-leaning band.

**What this does NOT need:** a privileged B300 / ncu (you can't get it, and GEMM/throughput crossovers
don't require it — counter-free wall-clock TFLOP/s is the metric, validated once on T4 already).

---

## 6. Where this lands you (publishability, honest)

- **Best realistic outcome:** a workshop/characterization paper (PMBS@SC / ES-FoMo / IISWC) — *"Native
  NVFP4 KV-compute for MLA decode: the one attention shape that meets the FP4 M≥128 gate, and why
  [it does / doesn't] convert on Blackwell."* Novel artifact + sharp boundary result. ~50/50 at a workshop.
- **Strengtheners (cheap, no privilege):** pair with the **three-generation spine** (the FP4-GEMM crossover
  measured on B200 + B300 + consumer SM120) and the **honest "padding was a red herring for decode" framing**
  — that lifts it from "one kernel" to "a characterization of FP4 decode compute across the Blackwell family."
- **It is still not a top-tier systems paper** — the contribution is narrow. But it is a *real* contribution
  (novel artifact + open question), which is the bar you asked about, and Stage A tells you in days whether
  it's worth pursuing.

**Recommended next action:** do **Stage A** on the (unprivileged) Blackwell box you already know how to rent.
It's days of work, needs no new infrastructure, and decisively tells you whether Arm 2 is a paper or a
negative result — before you spend a single hour on the hard kernel.

Sources: TRT-LLM #4412 (FP4 decode padding), CUTLASS block-scaled / Colfax NVFP4 tutorials (M≥128/K=256/TN,
per-16 E4M3 SF), vLLM `--kv-cache-dtype nvfp4` #43562 (FP4 KV storage shipping, compute not), FlashInfer
#2723 (SM120 FP4 caveats), DeepSeek-NVFP4 builds (FP4 weights + FP8 KV is the production norm).

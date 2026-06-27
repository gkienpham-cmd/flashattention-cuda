# Decode re-plan: what v6 measured, and the reordered v7→v11 path to a B300 FP4 decode kernel

**Compiled 2026-06-27.** Post-mortem of the v6 split-KV decode result + a verified, math-rooted
re-plan of the decode arc. Built from a 13-agent deep-research pass: 3 internal verification agents
(roofline math re-derived from scratch, kernel correctness audit, occupancy-vs-bandwidth analysis),
5 web-research agents (B300 specs, FA4 regime, decode SOTA, FP4-KV accuracy, GQA occupancy), 4
adversarial verifiers on the load-bearing claims, and one synthesis pass. Every quantitative claim
below is marked **[verified]** (primary-source or first-principles re-derived), **[predicted]**
(code-traced / roofline, not yet measured), or **[spec]** (third-party, needs on-chip confirmation).

The companion docs are [`b300-decode-research.md`](b300-decode-research.md) (the original decode-arc
thesis — now partly superseded on *ordering*, see §6) and [`fa-next-research.md`](fa-next-research.md)
(the FA-3/FA-4 technique menu). Diagrams in [`diagrams/`](diagrams/).

---

## 0. TL;DR — focus, don't pivot; but reorder

**The decode-on-B300 direction is sound and the external research strengthens it on every axis. The
one thing v6's measurement changes is the build *order*: occupancy (GQA M-packing) must come *before*
bytes (FP8/NVFP4 KV).**

v6 works — it beats a naive `N_q=1` loop **5.7–8.2×** and torch SDPA **1.5–3.3×** — but it reaches
**only 9–15% of HBM bandwidth** (6.5–11.6× above the bandwidth floor). It is **occupancy/launch-bound,
not bandwidth-bound.** Cutting KV bytes (the FP4 headline) multiplies a term the kernel is 8× away
from, so it is provably premature. The lever that *is* binding — fill the SMs and raise arithmetic
intensity — is **GQA M-packing** (`AI = 2/b → 2G/b`), which also turns the decode GEMV back into a
small GEMM (tensor cores re-engage) and reads each KV tile once instead of `G` times. It wins in
**both** the small-batch (latency) and large-batch (throughput) regimes, so it leads regardless.

| | OLD order (pre-v6) | **NEW order (post-v6)** | why |
|---|---|---|---|
| v7 | paged KV gather | **paged KV gather** (+ `--batch` & causal-offset bench) | unchanged — plumbing the mini-vLLM needs |
| v8 | FP8 KV | **GQA M-packing** (`AI=2G/b`) | v6 is occupancy-bound; this is the binding lever |
| v9 | NVFP4 + asym precision | **FP8 KV** (bytes, step 1) | bytes only pay once near the wall |
| v10 | GQA M-packing | **NVFP4 + asymmetric precision** (headline) | the differentiating result, on a now-bandwidth-bound kernel |
| v11 | MLA / speculative | **MLA / speculative** | push `AI` toward the ridge |

---

## 1. What v6 measured

`kernels/v6_splitkv/` — FP16-in/FP32-accum, two kernels behind one `forward`: a **split-KV partial**
(each block runs v4's online softmax over one KV chunk → writes the *unnormalized* `(O, m, ℓ)`) and an
**LSE merge** across splits. `choose_splits` raises `num_splits` until the block count hits ~2× the SM
count; **prefill `N_q=N_k` → 1 split → reduces to plain attention** (why the square-shape tests pass).
**25/25 correctness** vs SDPA. Measured on a vast.ai T4 (sm_75), B=1, H=8, non-causal full-cache scan:

| q×kv | µs/tok | %HBM | vs SDPA | vs naive | × above floor |
|---|---|---|---|---|---|
| 1×8×1×64 / 8192   | 56.1  | 11.7% | 3.16× | 7.86× | 8.6× |
| 1×8×1×128 / 16384 | 169.7 | 15.4% | 1.83× | 7.04× | 6.5× |

(Full six-row table in [`results.md`](results.md#step-6--split-kv-decode-v6). `%HBM` = `floor ÷
measured`; the K+V read is `2·B·H·N_k·d·2 B`.)

**The kernel is correct and the design is clean** (independent audit, cites line numbers):
- The partial kernel's online-softmax core (`splitkv_attention.cu:143-150`) is **byte-identical to v4**;
  the only deltas are the chunk-bounded key loop and the unnormalized write — both intentional. **[verified]**
- The LSE merge is the **mathematically exact** generalization of online softmax from across-keys to
  across-splits; empty/masked splits carry `(m=-FLT_MAX, ℓ=0)` and drop out via `exp(-∞-m)→0`, with an
  explicit `inv = (ℓ>0)?1/ℓ:0` guard against the all-empty `1/0→NaN`. **[verified]**
- `choose_splits` auto-degrades prefill to a zero-overhead single-block path, and the workspace layout
  `[B,H,N_q,S,*]` is already paging-ready for v7. **[verified]**

---

## 2. The surprise, stated precisely

The roofline predicted decode is **HBM-bound** (`AI = 2/b`, far below the ridge). The kernel **is**
HBM-shaped — but it reaches only **~12% of HBM bandwidth.** This is the **5th consecutive step** where
the roofline got the limiter *location* and the byte count right but missed wall-clock by an order of
magnitude, **for the same reason every time:** `roofline/model.py` is purely `max(t_mma, t_hbm,
t_mufu)` from FLOPs/bytes/exp — it has **no term for the schedule.** At decode that omission is the
whole story:

- **Grid occupancy.** At `BH=8` the partial grid is 64–80 blocks on 40 SMs ≈ **2 blocks/SM** (~16 of 32
  warps) — about half the ~4 blocks/SM Turing needs for full occupancy. **[verified]**
- **Two-kernel launch** (partial + merge) — fixed latency, worst at small `N_k`. **[verified]**
- **Under-occupied merge.** The merge grid is `(N_q, BH) = (1, 8)` = **8 blocks total**, regardless of
  split count, and every one of its `d` threads redundantly recomputes the scalar `(m, ℓ)`. **[verified]**
- **Per-key warp-shuffle GEMV scoring** — a 5-step butterfly reduction per key; reduction latency, not
  FMA throughput, dominates (the same wall v4 hit). **[verified]**

### 2.1 The crucial qualifier the docs were missing: it's batch-conditional

The 12%-HBM figure is **partly an artifact of the `BH=8` micro-benchmark**, and "occupancy before
bytes" is a **small-batch** truth, not a universal one. The mechanism is in `choose_splits` itself:

```
at decode  N_q = 1  →  row_tiles = 1  →  base_blocks = BH
num_splits = min( ceil(2·num_sm / base_blocks),  ceil(N_k / 256),  32 )
⇒ num_splits collapses to 1 exactly when  base_blocks ≥ 2·num_sm
```

So the **same kernel** self-disables splitting and becomes a plain one-block-per-`(b,h)` attention once
the batch alone fills the machine. The crossover **[verified by code-trace]**:

| | split-KV stops (`BH ≥ 2·SM`) | full warp occupancy (`BH ≥ 4·SM`) |
|---|---|---|
| **T4** (40 SM) | BH ≥ 80 → **B ≥ 10** @ H=8 | BH ≥ 160 → B ≥ 20 |
| **B300** (160 SM) | BH ≥ 320 → **B ≥ 40** @ H=8 | BH ≥ 640 → B ≥ 80 |

**The bench only ever ran `B=1` (`bench/harness.py --batch` defaults to 1).** So `BH=8` is the
*single worst-case occupancy corner* — exactly where `%HBM` is minimal. At `BH=512` (B=64) the grid is
512 blocks = 12.8/SM, well past saturation, and the kernel *was predicted* to reach its HBM ceiling.
**That end-state was [predicted]; v7 measured it — and the prediction was WRONG.**

> **⛔ MEASURED CORRECTION (v7, 2026-06-27 — this subsection's central claim is REFUTED).** The v7
> `--batch` sweep (B=1→64, BH=8→512, fixed N_k=8192, T4) shows **`%HBM` FLAT at 9.4–12.4% across the
> whole range** — *no climb*, even at BH=512 (`num_splits→1`, 12.8 blocks/SM). The occupancy→bandwidth
> crossover **does not exist for this kernel**, so "occupancy-first is small-batch-only / bytes-first holds
> at large batch" is **false**: v7 is per-CTA-bound at *every* batch size and never gets within ~8× of the
> bandwidth wall. **Root cause (code-verified), the term this section missed:** `sK+sV = 32 KB`/block caps
> residency at **2 blocks/SM** (T4 64 KB/SM) regardless of grid size, and at `N_q=1` only **1 of 8 warps
> computes** — so batch adds *waves*, not per-SM parallelism. The reorder still stands but the reason
> changes: **GQA M-packing leads because it fixes per-CTA efficiency (G warps + GEMV→GEMM), not because it
> "fills the SMs."** Bytes-first (v9/v10) is premature at all batch sizes, not just small. See `results.md`
> / `decisions.md` Step 7; `diagrams/decode-roofline-crossover.svg` (predicted arc → measured-flat line).

> **~~Honest framing for the paper:~~ [SUPERSEDED by the correction above]** ~~*occupancy-first* holds for
> the **latency / small-batch** decode regime (`BH ≲ 2·SM`) that v6 measured; *bytes-first* holds for
> **throughput / large-batch**.~~ Measured: there is no large-batch bytes-first regime for this kernel —
> it's per-CTA-bound at all batch. The accurate framing is **GEMV→GEMM (per-CTA efficiency) before bytes,
> at all batch sizes**. GQA M-packing is still the right lead.

---

## 3. The decode roofline (the math the paper rests on)

One decode step, `N` KV tokens, head dim `d`, `b` bytes per KV element, single head:

```
Work    = QKᵀ (2Nd)  +  P·V (2Nd)            = 4Nd  FLOPs
Traffic = read K (Nd) + read V (Nd)          = 2Nd·b  bytes      (Q, O are O(d), negligible)

           4Nd        2
  AI  =  ────────  =  ───   FLOP/byte         — N AND d cancel identically ⇒ N-independent
          2Nd·b        b
```

**[verified]** by first-principles re-derivation (bit-identical across the full `N×d` sweep) and
corroborated independently by FlashInfer ("decode operational intensity is `O(1)`, irrelevant to batch
size; always underneath the bandwidth ceiling ⇒ IO-bound"), arXiv 2503.08311 (attention-kernel AI
stays ~constant as batch grows — *no data reuse in batching*, the opposite of weight matmuls), and the
Scaling Book (generation MHA `AI ≈ 1`, "almost always memory bound").

| KV precision | bytes `b` | decode `AI = 2/b` | with GQA-8 `AI = 2G/b` |
|---|---|---|---|
| FP16/BF16 | 2 | 1.0 | 8.0 |
| FP8 | 1 | 2.0 | 16 |
| **NVFP4** (16×E2M1 + E4M3 scale ⇒ 72b/16 = **4.5 b/elem = 0.5625 B**) | 0.5625 | **3.56** | **28.6** |

**B300 ridge points** `AI* = peak / (8 TB/s)`: `AI*_FP4 = 15e15/8e12 = 1875` **[verified]**;
`AI*_FP8 = 5e15/8e12 = 625`; `AI*_BF16 = 3.5e15/8e12 ≈ 437.5` **[spec — corrects the `≈310`
in `b300-decode-research.md §4`, which back-solved to an unsourced 2.48 PF; confirm on-chip]**. Even
GQA-8 + NVFP4 (`AI ≈ 28.6`) sits **~65× below** the FP4 ridge — **decode is memory-bound at every
reachable precision.** FP4 is therefore *always* a bandwidth lever and *never* a compute lever; tensor
cores re-engage only for the *shape* (`M = G > 1`), not because we approach the compute roof.

### 3.1 Two orthogonal axes — the key conceptual fix

v6 conflated two things the re-plan separates:

1. **Occupancy axis (how close to the roof you get).** Batch / split-KV fills the grid so the kernel
   *reaches* its fixed HBM ceiling. Batch leaves `AI` **unchanged**; it only lifts the achieved
   fraction of peak. This is the axis v6 is stuck on at 12%.
2. **Intensity axis (where the roof is).** GQA/MLA share KV across heads, *raising* `AI = 2/b → 2G/b →
   2H/b`. This moves you rightward toward the ridge and cuts redundant KV bytes.

**GQA M-packing is the unique lever that pushes on both at once** — `+G×` useful work per CTA
(occupancy) *and* `AI = 2G/b` with KV read once (intensity + bytes). That is why it leads.
See [`diagrams/gqa-mpacking.svg`](diagrams/gqa-mpacking.svg).

---

## 4. Verified hardware & competitive landscape

### 4.1 B200 → B300 seams (what actually changed)

[`diagrams/b200-b300-seams.svg`](diagrams/b200-b300-seams.svg). The strategic surprise is at the top.

| Spec | B200 | B300 / Blackwell Ultra | decode consequence | status |
|---|---|---|---|---|
| HBM capacity | 192 GB | **288 GB** (+50%) | longer context / bigger batch resident | [verified] |
| **HBM bandwidth** | ~8 TB/s | **8 TB/s — FLAT** | **the surprise: no GB/s headroom on B300** | **[verified]** |
| NVFP4 dense | ~10 PF | **15 PFLOPS** (20 sparse) | FP4 KV reads ~3.5× fewer bytes (the v10 lever) | [verified] |
| FP8 dense | ~4.5 PF | **5 PFLOPS** | FP8 KV halves bytes (the v9 lever) | [verified] |
| BF16 dense | 2.25 PF | ~3.5 PF | (ridge 437.5; reconcile vs §4's 310) | [spec] |
| Exp / SFU (MUFU.EX2) | 5 TExp/s | **~10.7 TExp/s (~2×)** | delete FA4 exp-emulation; run hardware `exp2` | [verified] |
| SMs | 148 | **160** | more split-KV parallelism **but worsens `M=1` SM-starvation** | [verified] |
| Compute capability | sm_100 | **sm_103** (CUDA 12.9; `sm_103a` for exclusives) | the build target | [verified] |
| L2 cache | ~128 MB | ~192 MB | more KV resident | [spec] |

**Why bandwidth is flat (mechanism, not just a number):** B300 goes from 8-Hi to 12-Hi HBM3e stacks —
that adds DRAM *layers* (capacity), not pins or pin-speed; the 16×512-bit (8192-bit) controller layout
is unchanged, so bandwidth is unchanged *by construction*. One aggregator's "+25% BW" is a verified
**error** (the same page then says "unchanged at 8 TB/s"). **[verified — NVIDIA eng. blog + SemiAnalysis
teardown, 6 corroborating sources].**

**Consequence, sharpened:** on B300 a memory-bound decode kernel's *only* levers are **fewer bytes
(FP4), more occupancy (GQA), faster exp (2× hardware EX2).** That is exactly the v8→v11 set. The "2×
exp" payoff is real hardware (MUFU.EX2 ~4943 → ~10024 Gop/s FP32) but its *decode* benefit is
workload-dependent (M=1 matmuls are tiny) — **measure it, don't assume it.**

### 4.2 FA4 is the wrong-regime kernel — but frame the claim carefully

**[verified]** FA4 (arXiv 2603.05451, 2026-03) is **BF16, compute-bound, large-tile** (128×128 /
256×128, 2-CTA), **forward *and* backward**, benchmarked on **square** `seqlen_q=seqlen_kv` at **71% of
B200's 2.25 PFLOPS** (~1605–1613 TFLOPS). Its signature moves — async MMA, software `exp2`-on-FMA
emulation, 2-CTA/TMEM, `τ=8` conditional rescale — all assume the query tile `M ≥ 128`. **Decode is
`M = 1`:** both matmuls collapse to GEMV (`AI = 2/b`, <1% MMA util), and FA4 even pads queries to 256.
The published paper has **no decode-specialized compute kernel** — only decode-*aware scheduling* (LPT
batch-sorting). FA3 explicitly deferred inference to future work; **cuDNN 9.19 already matches FA4 on
BF16 prefill** — that advantage is commoditized.

**⚠️ Two corrections to the original framing (claim discipline):**

1. **FA4 is *acquiring* a decode path.** By 2026-06-11 Modal upstreamed a `q_stage=1` single-query-tile
   + split-KV + GQA-packing path into the Dao-AILab kernel (single-split decode **1.79 → 5.47** (4
   splits) **→ 20.7 TFLOP/s** (GQA packing)). "Beating FA4 in decode is nearly free" is true of the
   **paper**, not the **repo HEAD**. **[verified]**
2. **"None of FlashInfer/FlashMLA is B300-proven" is FALSE.** FlashInfer shipped measured GB300 decode
   throughput in vLLM (DeepSeek-V3.2, Feb 2026); FlashMLA served DeepSeek-V4 **day-0 on GB300** via
   SGLang (Apr 2026); the FlashInfer sm103 FMHA deadlock (#2939) was **fixed**. **Only SageAttention**
   remains consumer-only (RTX 5090 / sm120). **[verified — refutes `b300-decode-research.md §6`'s
   "immature/deadlock" hedge, now partly stale].**

> **The defensible contribution, stated honestly:** *not* "we beat FA4." Rather — **an open,
> roofline-documented, asymmetric-precision FP4 split-KV decode kernel on B300, measured against the
> real bar (FlashInfer TRTLLM-Gen / FlashMLA-class decode).** The wedge is *openness + the
> prediction-vs-measured methodology*, not a headline speedup over a kernel that was never a decode
> kernel.

### 4.3 The real bar, and FP4 accuracy

- **Bar to beat:** FlashInfer's **TRTLLM-Gen attention** (the vLLM/SGLang default), **FlashMLA**
  (latent-KV, ~93% KV reduction), with **Flash-Decoding** as the shared split-KV primitive v6 already
  implements. **SageAttention3** is the FP4 *accuracy recipe* (NVFP4, two-level `P` scaling, 1038 TOPS,
  5× FA2) but on consumer silicon. **[verified]**
- **FP4-KV accuracy is the genuine risk.** The asymmetric-precision principle is sound and supported:
  `P·V` tolerates FP4 because post-softmax `P ∈ [0,1]` is a normalized convex combination (bounded,
  sums to 1, error averages out); `Q·Kᵀ` does **not** (it feeds `exp`, so error amplifies `δ → e^δ`,
  and logits carry outliers) → keep scores at **MXFP8 + outlier residual.** Production stacks often keep
  KV at **FP8** for accuracy, so the `2/0.56` FP4 win is an **upper bound**, not a guaranteed operating
  point. **Mandate: measure perplexity / logit-error vs an FP16 reference, not MSE** — and sequence
  **FP8 (v9) before NVFP4 (v10)** to de-risk. **[verified]** See
  [`diagrams/asymmetric-precision-dataflow.svg`](diagrams/asymmetric-precision-dataflow.svg).

---

## 5. The reordered build plan (v7 → v11), roofline-first

Loop each step: predict limiter → explain → build + hand-verify → correctness vs reference → bench →
roofline read (prediction vs measured) → quiz. Decode metric: **µs/token, %HBM, accuracy vs FP16/FP32.**
See [`diagrams/build-roadmap-v6-v11.svg`](diagrams/build-roadmap-v6-v11.svg).

### v7 — Paged KV-cache gather + decode-harness fixes · **[BUILD/RENT]** T4/A100 · occupancy-neutral
- **Why first:** the import surface a from-scratch mini-vLLM consumes; foundational plumbing every later
  kernel rides on. Carries v6's split-KV + LSE merge unchanged.
- **Build:** block-table indirection — replace the contiguous `gj→offset` at `splitkv_attention.cu:119-124`
  with a page-table lookup in the cooperative smem load. Add a non-contiguous-KV correctness case.
- **Harness (turns inferences into measurements):** (a) `--batch` sweep `B∈{16,32,64}` → measure `%HBM`
  climbing to saturation, **pinning the crossover empirically** (closes §2.1's [predicted]); (b)
  `--decode` causal **query-offset** (place `q` at `N_k−1`) so causal decode attends the whole cache,
  killing the degenerate 1-key artifact.
- **Math:** `AI = 2/b` unchanged; gather adds `O(N_k/page_size)` index reads, negligible vs KV bytes.
  Deliberately does **not** attack the limiter — it sets up the measurement that justifies v8.

### v8 — GQA M-packing (GEMV → small GEMM, `AI = 2G/b`) · **[RENT]** A100/H100 · **the occupancy lever**
- **Why promoted ahead of precision:** v6 proved the limiter is occupancy/intensity, not bytes. Packing
  the `G` query heads of one GQA group into the CTA's `M` dimension fixes **three things in one change**:
  KV read **once** (not `G×`), occupancy rises `G×` at fixed batch (crossover `BH` drops by `G`), and
  `M = G > 1` lets **WMMA/MMA tensor cores re-engage** (recovering the v5 path in the decode regime).
- **Deliverable:** measured speedup vs v6/v7 as `G: 1→8` at fixed `N_k`; the prediction-vs-measured
  curve for `AI = 2G/b`; `%HBM` should climb well above 12%; record whether tensor cores engage at `M=G≥8`.
- **Math:** `AI = 2G/b` (GQA-8 BF16 = 8.0, FP4 ≈ 28.6). Shared-KV read drops bytes
  `2·BH·N_k·d·b → 2·(BH/G)·N_k·d·b`. The documented escape from the GEMV trap.

### v9 — FP8 KV cache (bytes lever, step 1) · **[B300]** (or H100 for correctness)
- **Why here:** bytes-reduction only pays once v8 has pushed the kernel toward the wall. FP8 (`b:2→1`)
  doubles `AI`, lower accuracy risk than FP4, and establishes the in-loop dequant machinery NVFP4 reuses.
- **Deliverable:** decode speedup from halving KV bytes **and** an accuracy delta vs FP16 KV. The
  honest test: the prediction should *finally track measured* **iff** v8 made the kernel bandwidth-bound.
  If it doesn't track, that is itself the deliverable (still occupancy-bound → more M-packing).
- **Math:** `AI = 2/b = 2.0` (or `2G/b = 16`); B300 ridge `AI*_FP8 = 625` ⇒ still memory-bound ⇒ byte
  cut maps ~directly to `µs/tok = KV_bytes / 8 TB/s`.

### v10 — NVFP4 + asymmetric precision (the headline contribution) · **[B300]** required
- **Why last among precision steps:** lands on a kernel already occupancy-filled (v8) and byte-disciplined
  (v9), where cutting bytes actually converts to wall-clock. Asymmetric precision: FP4 `P·V`
  (convex-combination-safe) + MXFP8 `Q·Kᵀ` + outlier residual + FP32 hardware `exp2`; NVFP4 KV (`b≈0.56`).
- **Deliverable:** the headline — `µs/tok` vs FP16 and FP8 KV on B300, the **FP4 accuracy delta**
  (perplexity / logit error vs FP16, documented), and the prediction-vs-measured roofline at
  `AI = 2/0.56 ≈ 3.57` (or `2G/0.56 ≈ 28.6`).
- **Math:** `AI ≈ 3.57` (NVFP4) / `≈ 28.6` (GQA-8); ridge 1875 ⇒ ~50–500× below ⇒ firmly memory-bound,
  so ~3.5× fewer bytes → ~3.5× `µs/tok` **iff** bandwidth-bound (the v8/v9 precondition).

### v11 — MLA / latent-KV decode + speculative (stretch) · **[B300]**
- **Why last:** most architecturally invasive, highest `AI`. MLA shares **one** latent KV across all `H`
  heads (`AI → 2H/b`), pushing furthest toward the ridge — the only decode regime where tensor cores
  stop idling. Speculative decoding restores `M = k` (draft length) → small GEMM along the query-time
  axis, the same `M>1` win as GQA but temporal.
- **Deliverable:** `µs/tok` for MLA-shaped KV vs the GQA path; the `AI`-toward-ridge curve; closes the arc.

---

## 6. What this supersedes (and what it keeps)

- **Supersedes** the *ordering* in [`b300-decode-research.md §8`](b300-decode-research.md) (FP8/FP4 before
  GQA) and the OLD `build-roadmap-v6-v11.svg`. GQA M-packing moves **v10 → v8**; FP8/FP4 shift to **v9/v10**.
- **Supersedes** the flat "occupancy before bytes / low-precision only pays once bandwidth-bound"
  statements in `results.md` and `decisions.md` Step 6 — they are correct **for `BH ≲ 2·SM`** but were
  written without the batch qualifier. Now qualified.
- **Corrects** `b300-decode-research.md §4` `AI*_BF16 ≈ 310 → ≈ 437.5`, and §6's "FlashInfer/FlashMLA
  immature on sm103" (both now ship measured GB300 decode).
- **Keeps** the entire thesis: decode-on-B300 is the open lane; FA4 is a prefill/training kernel; the
  asymmetric-precision FP4 recipe is the differentiating research; counter-free profiling stays the norm.

---

## 7. Open questions / what to measure next

1. **[v7] Pin the occupancy→bandwidth crossover empirically.** ✅ **DONE (2026-06-27) — and it
   REFUTED the prediction.** Swept `B=1→64` (BH=8→512, N_k=8192, T4): `%HBM` stayed **flat at 9.4–12.4%**,
   no climb even at 12.8 blocks/SM. There is **no crossover** — decode is per-CTA-bound (32 KB smem →
   2 blocks/SM cap; 1-of-8 warps active at `N_q=1`), not grid-occupancy-bound, at every batch size. The
   reorder survives but the reason is **per-CTA efficiency (GEMV→GEMM), not "fill the SMs," and it is NOT
   batch-conditional.** §2.1 corrected. Follow-up: a bare-metal pipe-util read to confirm the
   smem-residency story directly (counters blocked on cloud).
2. **[v8] Does GQA M-packing engage tensor cores at `M=G=8`,** and does measured track `AI=2G/b`?
3. **[v9/v10] FP4 accuracy:** perplexity/logit-error vs FP16 on real shapes, not MSE. Is FP8 enough?
4. **[roofline honesty]** Either add an explicit occupancy/launch term to `model.py` so the
   prediction-vs-measured *gap itself* is predicted, **or** keep it an honest pure lower bound and keep
   recording the schedule gap as a first-class deliverable. Do not let v8–v10 quietly claim the roofline
   "predicted" a speedup it structurally cannot model.
5. **[merge]** Measure the merge kernel's wall-clock share at small `BH`; if the 2-kernel launch + tiny
   `(1,8)` merge dominate, a single-/persistent-kernel or Blackwell 2-CTA-cluster + DSMEM fusion is a
   v8.5 contingency. GQA raises `M`, not the merge grid.

---

## 8. Diagrams (paper-ready, in [`diagrams/`](diagrams/))

| File | Shows |
|---|---|
| `decode-roofline-crossover.svg` | **NEW headline.** v6 sits below *both* the HBM roof and the compute ridge; the binding constraint switches occupancy→bandwidth as `BH` grows past `2·SM`. The "occupancy before bytes" argument in one figure. |
| `gqa-mpacking.svg` | **NEW.** `M=1` GEMV (KV read `G×`) → `M=G` GEMM (KV read once), `AI = 2/b → 2G/b`, tensor cores re-engage. |
| `splitkv-lse-merge-dataflow.svg` | **NEW.** The two-kernel schedule the roofline can't see — where the launch + under-occupied-merge overhead lives. |
| `b200-b300-seams.svg` | **NEW.** Verified B200→B300 deltas; flat-bandwidth surprise front and center. |
| `build-roadmap-v6-v11.svg` | **REVISED** to the reordered sequence (GQA before bytes), with the struck-through old order. |
| `decode-roofline.svg`, `asymmetric-precision-dataflow.svg`, `split-kv-schedule.svg` | unchanged (still accurate). |

---

## 9. Sources & method

13-agent workflow (704K tokens, 204 tool-calls). Internal: `roofline/model.py`, `archs.py`,
`kernels/v6_splitkv/`, `bench/harness.py`, the Step-6 tables. External (primary, adversarially
verified at 1-skeptic-per-claim, refute-biased):

- B300 specs — NVIDIA "Inside Blackwell Ultra" + "Making Softmax More Efficient with Blackwell Ultra"
  (2× MUFU.EX2); SemiAnalysis GB300 teardown (flat 8 TB/s mechanism); vendor spec tables.
- FA4 — arXiv 2603.05451; Colfax / Together / Tri Dao blogs; **Modal** "adding a decode path to FA4"
  (2026-06-11, `q_stage=1`+split-KV+GQA-packing); Dao-AILab `flash-attention` #2596 (exp2_emu off for
  decode MLA on sm100); cuDNN 9.19 parity.
- Decode SOTA — Flash-Decoding (Stanford CRFM); FlashInfer (arXiv 2501.01005; vLLM GB300, Feb 2026;
  #2939 fixed); FlashMLA (DeepSeek; SGLang GB300 DeepSeek-V4 day-0, Apr 2026); GLA; LeanAttention.
- FP4 accuracy — SageAttention3 (arXiv 2505.11594, NVFP4 two-level P, RTX 5090); KVQuant / KIVI /
  QServe-Atom KV-quant literature; NVFP4/MXFP4 microscaling (E2M1 + E4M3, 4.5 b/elem).
- AI=2/b corroboration — FlashInfer (IO-bound); arXiv 2503.08311 (AI batch-independent, no reuse);
  arXiv 2505.21487 (Hardware-Efficient Attention; `AI ∝ G`); the Scaling Book.

**Verification verdicts (refute-biased):** B300-bandwidth-flat → **supported/high**; decode `AI=2/b`
memory-bound → **supported/high**; v6-12%-is-small-batch-artifact → **supported/high**; FA4-prefill +
"beat-it-free" → **mixed/high** (true of the paper; FA4 acquiring a decode path; FlashInfer/FlashMLA
*are* B300-proven — reframe the contribution accordingly).

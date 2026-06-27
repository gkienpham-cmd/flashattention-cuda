# v7 paged-KV decode — deep-research close-out (cited synthesis)

*Tesla T4 (sm_75), torch 2.6.0+cu124, vast.ai, measured 2026-06-27. This is the full close-out of the
v7 gate: a 35-agent pass — 6 internal data-forensics agents over [`notebooks/v7_paged_gate_output.ipynb`](../notebooks/v7_paged_gate_output.ipynb)
+ source, 7 external web-research agents (GQA M-packing + B300), and 7 load-bearing claims each run
through a 3-vote adversarial gate (2-of-3 refutes kill). It mirrors what [`decode-replan.md`](decode-replan.md)
was for Step 6: the narrative docs ([`results.md`](results.md)/[`decisions.md`](decisions.md)/
[`interview-prep.md`](interview-prep.md)) carry the surgical deltas; this file carries the evidence,
citations, and the contradictions processed impartially. The v8 starter is [`v8-kickoff.md`](v8-kickoff.md).*

**Headline:** v7 ran (Gate 1 ✅ 51/51; Gate 2 quiz pending). The per-CTA-bound thesis and the
"GQA-M-packing-before-bytes" reorder **both survive 0/3 adversarial refute**. Two doc premises were
corrected (not the conclusions — the *reasons*), one genuinely new finding was added, and the roadmap
is **unchanged in order** (v8 GQA M-packing stays first) with two additions.

---

## 1. Proofcheck — did v7 actually run?

**Yes. Gate 1 cleared; Gate 2 (quiz) is the only thing outstanding — so v7 is NOT done.**

- **Gate 1 — correctness: PASS, 51/51** (`pytest -k "v7_paged or v6_splitkv"`, 46 s, 0 failed / 75
  deselected). Covers paged decode (`N_q=1`, `N_k ∈ {4096, 8192, 8190}`, page_size ∈ {128, 256}, d ∈
  {64, 128}, causal both ways via query-offset), the `num_splits→1` square-shape regression, and all of
  v6's untouched cases. The teeth: `build_paged_kv` **shuffles** the physical pages, so a kernel reading
  contiguously instead of through the block table would gather wrong tokens and fail. `N_k=8190`
  exercises the partial-last-page tail.
- **The 7 technical claims behind the existing analysis all reproduce from source** (this session):
  `sK+sV = 32 KB`/block → 2 blocks/SM (exact for d=64 *and* d=128); only warp 0 active at `N_q=1`
  (`if (active)` guard, `paged_attention.cu:92`); `choose_splits` self-disables at `BH≥2·SM=80`
  (`:227`); gather overhead ~0.006% (docs' ~0.1% is conservative); `%HBM = achieved/peak` with T4 peak
  320 GB/s; causal reference is non-causal SDPA matched to the q_offset.
- **Caveats (all honest, already hedged in-repo):** (1) the ~15–25% v6→v7 paging overhead is **unproven**
  (separate sessions, `clock~-1/-1`; same-shape delta measured 19–27% but inseparable from clock drift);
  (2) the causal `vs_sdpa` column is a measurement artifact (§3c); (3) no profiler counters
  (`ERR_NVGPUCTRPERM`) — the per-CTA story is code-traced + timing-inferred, not ncu-measured.

---

## 2. The data story — per-CTA-bound, decisively *(survives 0/3 refute)*

**The v7 decode kernel is per-CTA-bound at *every* batch size, NOT grid-occupancy-bound. There is no
occupancy→bandwidth crossover.** All 10 batch-sweep `%HBM` values reproduce to <0.1 pp from
`harness.py:174-175`.

| metric | value |
|---|---|
| %HBM, batch sweep N_k=8192, BH=8→512, **d=64** | **9.4–10.3%** (range 0.9 pp, non-monotonic) |
| %HBM, same, **d=128** | **11.4–12.4%** (range 1.0 pp) |
| µs/tok, d=64, BH=8→512 | 66.83 → 70.02 → 69.62 → 68.93 → 63.66 (flat; B=64 *fastest*) |
| Distance to HBM saturation (= 100/%HBM) | **8.1–10.6×** at every B and d |

**One statistic, not two.** At `N_q=1` with constant bytes/token, `µs/tok` and `%HBM` are algebraically
locked (`100/%HBM` = the multiple above the bandwidth wall) — the "~10% HBM" and "~8–10× above floor"
framings are the *same* measurement inverted, not independent corroborations.

**The proof it's per-CTA, not grid (code-verified, `paged_attention.cu`):**
1. **smem caps residency at 2 blocks/SM:** `sK+sV = 32 KB`/block (identical for d=64 TN=64 and d=128
   TN=32), T4 has 64 KB/SM → hard cap of **2 resident blocks/SM, batch- and N_k-independent** (`:81-82`).
2. **At `N_q=1` only 1 of 8 warps computes:** the `active = (gi < N_q)` guard (`:92`) leaves warp 0 alone
   in the score/PV loop while the other 7 do the cooperative load then idle (`:134-159`).
3. **`choose_splits` self-disables at BH≥80** (`:227`): B≥16 runs `num_splits=1`, so B=16/32/64 launch
   128/256/**512** blocks (12.8/SM at B=64) — heavily over-subscribed — yet %HBM doesn't budge. **The
   cleanest possible proof:** the grid is already saturated and bandwidth still doesn't move. Batch adds
   *waves*, not per-SM parallelism.

**The d=128 corroboration:** d=128 costs **1.42× (N_k=2048) → 1.65× (N_k=16384)** of d=64, NOT 2×. KV
bytes scale 2× with d, but only the per-key FMAs scale; the 5-`__shfl` reduction + 2 `__expf` + launch/
merge are d-independent. An HBM-bound kernel would show 2×; sub-2× is the fingerprint of compute/latency-
bound. **The B=64 dip** (63.66 < 66.83 µs/tok, 4.7%) is fixed-overhead (2-kernel launch + under-occupied
merge) amortized over more waves — inferred, within clock noise, and rules out thermal throttling
(throttling would slow, not speed, the largest batch).

**Diagram:** [`diagrams/decode-roofline-crossover.svg`](diagrams/decode-roofline-crossover.svg) (refreshed
predicted-arc → measured-flat); [`diagrams/per-cta-limiter-anatomy.svg`](diagrams/per-cta-limiter-anatomy.svg).

---

## 3. Three new findings (each a settled fact)

### (a) vs-SDPA batch crossover — SDPA overtakes v7 by B=8 *(survives 0/2; new)*

v7 beats torch SDPA **only at B=1** (2.65× d=64 / 1.46× d=128). For all sampled B≥8 it **loses**:
0.56→0.50× (d=64), 0.34→0.32× (d=128). Crossover **bracketed in (1, 8]** (no B=2/4 sample).

**Mechanism (corrected from the first guess):** it is **NOT** "v7 becomes plain split=1 at batch" — at
the crossover (B=8, BH=64) `choose_splits` is **still active (=2)**; split-disabling happens later at
BH≥80. The real cause: v7 is **already SM-saturated at B=1** (10 splits → 80 blocks = the 2-blocks/SM
cap), so its per-token cost is **flat** (~67 µs/tok d=64), while SDPA's per-token cost **collapses ~4.5×**
(reconstructed ~177→32 µs/tok) as batch fills *its* grid. Both are launch/occupancy-bound at B=1 (v7 at
9.8% HBM, reconstructed SDPA at only ~3.7%) — **v7 wins at B=1 precisely because SDPA under-fills the grid
even worse there.** The small-batch win vanishes under serving batch. **→ "reclaim SDPA at batch" is a
first-class v8 deliverable.** Diagram: [`diagrams/v7-vs-sdpa-batch-crossover.svg`](diagrams/v7-vs-sdpa-batch-crossover.svg).

### (b) %HBM dtype — RESOLVED: KV reads are **fp16**; the printed %HBM is **correct, not 2× understated**

A skeptic flagged that the bench header prints `precision=fp32` while `%HBM` counts fp16 bytes. **Resolved
decisively in favour of fp16:** the bench allocates fp32 K/V (`harness.py:120,126-128`) and `build_paged_kv`
preserves fp32, **but the v7 kernel never reads those tensors.** The host entry casts to half —
`k_pool.to(torch::kHalf)`, `v_pool.to(torch::kHalf)` (`paged_attention.cu:274-276`), binds device pointers
to the half copies (`:293-294`), declares params `const __half*` (`:69-70`), and every in-kernel HBM load
is `__half2float(K_pool[src])` = **2 bytes/elem** (`:125-126`). The `%HBM` denominator `kv_bytes = 2·B·H·N·d·2`
uses 2 B/elem — **it matches the dtype the kernel actually reads.**

→ **No correction. The printed ~7–12.5% is authoritative.** Independently re-derived: `1×8×1×128/16384`
total time **1.687 ms** (not 0.211 ms — `µs/tok = total·1000/(B·H·qn=8)`), `kv_bytes` 67.1 MB → 39.8 GB/s →
12.4% = 8.04× above the 0.21 ms floor. The `hbm-fp16-bytes` "2× undercount" claim **died 3/3**. *(Cosmetic:
relabel header `in=fp32 / kv-read=fp16`. Note: the `.to(kHalf)` cast allocates a throwaway fp16 copy of the
whole cache **inside** the timed region — a cost a true fp16-native vLLM cache wouldn't pay; invisible to
%HBM but worth flagging for the mini-vLLM framing.)*

### (c) Causal `vs_sdpa` collapse (0.15→0.02×) — a baseline artifact, NOT a regression

On causal rows v7 places its query at `q_offset = N_k−1` and scans the whole cache (all N_k keys), while
the bench's SDPA reference is `is_causal=True` on a `[1, N_k]` mask, which PyTorch **top-left-aligns** →
query 0 attends **exactly 1 key**. The reference does ~1/N_k the work, so `vs_sdpa` tracks ~1/N_k. **Proof
v7 is correct:** causal µs/tok ≈ non-causal per row (66.79 vs 66.65 @ 64/8192) — same keys, same time. The
correctness test (`test_correctness.py:209`, `causal=False`, the right full-cache oracle) already passes
51/51; **only the bench column misleads.** Fix (v8-harness, ~6 lines): bottom-right-aligned causal mask in
the reference (`torch.nn.attention.bias.causal_lower_right` if it keeps the fused path on 2.6.0). Scope all
competitive claims to **non-causal** until then.

---

## 4. External research synthesis

### v8-enabling: GQA M-packing

**The recipe (HIGH; unanimous across 5 production kernels).** Map the `G = H_qo/H_kv` query heads sharing
one KV head into the GEMM's **M (row) dimension** so one CTA computes all G heads against KV staged in smem
**once**. This (a) raises AI `2/b → 2G/b`, (b) turns the width-1 GEMV into a GEMM, (c) reads KV once not G×,
(d) lights up **G compute-warps** — the exact fix for v7's "1-of-8 warps" wall. Grid is universally **3D:
(batch, kv_head, kv_split)**; keep split-KV for occupancy when `batch·kv_heads < #SMs`, then an LSE merge.
Tiling constraint: **`M_block % G == 0`**.
- Sources: FlashInfer paper [arXiv 2501.01005](https://arxiv.org/html/2501.01005v1); Character.AI/Colfax
  (16 heads × 4 tokens → 64-wide WGMMA); Modal FA4 `pack_GQA`; FlashMLA (`BLOCK_SIZE_M=64`); TRT-LLM XQA.

**Tensor-cores-at-M=8 — VERDICT: they engage, via padding M=8→16; claim survives 0/3 (HIGH).** The hard
gate is **tensor-core minimum M = 16 rows** (FlashInfer, verbatim: *"For query tile size 1, we use CUDA
Cores… tensor core instruction m (minimum rows) is 16, and use Tensor Cores for other query tile sizes"*).
On A100 the MMA atom is `mma.m16n8k16`. **G=8 < 16, so a single Llama-3-70B group does not fill a tile** —
the three options: (a) **pad M=8→16 + mask** (half row-util), (b) pack 2 KV-groups to M=16, (c) CUDA-core
QK + tensor-core PV. **This trio is the key v8 ablation and the main place sources disagree.** The win is
real either way (KV read once, AI→O(G)); the *magnitude* is what the padding choice decides.

**Target G (HIGH).** Bench **G=8, d=128** (Llama-2/3-70B: 32 q / 4–8 kv — FlashInfer's exact benchmark) and
**G=4, d=128** (Llama-3-8B: 32 q / 8 kv — [Llama 3 Herd, arXiv 2407.21783](https://arxiv.org/pdf/2407.21783)).
**Sweep G ∈ {1,2,4,8,16,32}** — G is v8's batch-sweep analogue, the variable that crosses the CUDA→tensor-
core threshold and sets AI=2G/b. (DeepSeek-V2/V3 use MLA, not plain GQA — a v11 concern.)

**Landscape comparators (HIGH).** **FlashInfer** `BatchDecodeWithPagedKVCache(use_tensor_cores=True)` +
**TRT-LLM XQA** = M-packing SOTA; **vLLM PagedAttention v2** = the CUDA-core baseline floor v8 must clear
(split-KV + paging but **no M-packing** — the cleanest isolation of v8's one variable; vLLM's own
[RFC #15351](https://github.com/vllm-project/vllm/issues/15351) proposes adding exactly this). Realistic
envelope: **~2.9× single-token decode at the op level** (Modal `pack_GQA`: 7.1→20.7 TFLOP/s; FlashInfer
2–3× vs vLLM at batch=64), but only **~30% end-to-end decode** (Colfax). **Claim the op-level number as
op-level, not end-to-end.**

### B300 horizon

| claim | verdict | conf | key source |
|---|---|---|---|
| **HBM bandwidth FLAT 8 TB/s** B200→B300 | **CONFIRMED** (survives 0/3) — *"8 TB/s per GPU (same as base Blackwell)"*; only capacity grew | HIGH | [NVIDIA Blackwell Ultra blog](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/), [SemiAnalysis](https://newsletter.semianalysis.com/p/nvidias-christmas-present-gb300-b300-reasoning-inference-amazon-memory-supply-chain) |
| **Capacity 192→288 GB** (12-Hi vs 8-Hi), +50% with **zero** BW gain | CONFIRMED | HIGH | same |
| **FP4: 15 PF dense / 20 PF sparse** NVFP4 (1.5× B200), native 5th-gen TC two-level scaling | CONFIRMED — use **15 PF dense** as the v10 ceiling (dense decode won't hit 2:4 sparse) | HIGH | NVIDIA blog |
| **SFU/exp DOUBLED → 10.7 TeraExp/s** (~2× faster softmax) | CONFIRMED — the MUFU/exp term in `model.py` is ~half the B200 cost on B300 | HIGH | NVIDIA blog |
| **Compute capability = sm_103** (CC 10.3), NOT sm_100 (=B100/B200) | CONFIRMED | HIGH | [Blackwell compat guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/) |
| **B300 L2 size** (192 MB?) | **UNVERIFIED** — secondary-wiki number; no primary disclosure. Do NOT hard-code in `archs.py`. | LOW | — |

**Per-GPU hygiene (load-bearing):** always quote **per-GPU** (8 TB/s, 15 PF, 288 GB, 160 SMs) in roofline
math; NVL72 figures (576 TB/s, 1.1 EF, 20.7 TB) are **72×** and will silently wreck any AI calc. Sanity:
576 TB/s ÷ 72 = 8 TB/s.

**FA4 framing — CORRECTED (see §5, claim 6).** FA4 is BF16-prefill/training (query-dim parallelism is
"kryptonite" for decode), and cuDNN ~9.1x already matches it on BF16 prefill — but its decode path is now
**LANDED, not "acquiring"**: Modal upstreamed split-KV (#1940), single-query (#1993), paged (#1999/#2104),
FP8 (#2109), **GQA-packing #2186 (2.92×)** into Dao-AILab/flash-attention. Frame v8–v11 as *"an open,
roofline-documented, asymmetric-precision FP4 decode kernel measured vs FlashInfer/FlashMLA"* — **not** "we
beat FA4" (it's a moving target with the very levers v6–v9 build).

**v10 asymmetric precision (the headline) — MEDIUM, sharpened.** v10 means **compute-stage asymmetry**: V
in NVFP4 for the error-tolerant P·V; QK scores in FP8/BF16; exp in FP32. Precedent: SageAttention2 (QK INT4
/ PV FP8 + FP32 accum), KV-AdaQuant (**K4V2 = 75.2% vs K2V4 = 54.7% GSM8K**, a ~20 pt swing favouring more
bits on K). **Caution:** NVIDIA's shipped NVFP4-KV is **symmetric** (K=V=NVFP4, <1% loss, 3× TTFT), and the
asymmetry payoff shrinks at 4-bit (it's large at 2-bit). **So make asymmetric-vs-symmetric NVFP4 the v10
headline ablation, and pre-register that "symmetric is enough" is itself a clean finding.** Exp-in-FP32 is
table stakes, not a contribution.

---

## 5. Adversarial results — 5 of 7 survived

| # | claim | verdict | why |
|---|---|---|---|
| 1 | **per-cta-all-batch** | ✅ SURVIVES 0/3 | all 10 sweep values reproduce <0.1 pp; over-subscribed grid (12.8 blocks/SM at B=64) yet %HBM flat = cleanest proof the wall is in-CTA. |
| 2 | **v8-gqa-right-lever** (GQA before bytes) | ✅ SURVIVES 0/3 | at ~10% HBM, FP8/FP4 move a floor off the critical path; production literature confirms M-packing is *the* lever. Honest caveats: v8 payoff is **[predicted]**; stale comments in `paged_attention.cu:221-226` + `roofline/model.py:96-99` still assert the refuted "split-KV fills SMs" story (cleanup). |
| 3 | **sdpa-overtakes** (SDPA wins by B≥8) | ✅ SURVIVES 0/2 | every number reproduces; mechanism (v7 SM-saturated → flat, SDPA amortizes ~4.5×) verified. Crossover bracketed (1,8], not pinned. |
| 4 | **wmma-pays-at-m8** | ✅ SURVIVES 0/3 | M≥16 threshold is a hardware fact (FlashInfer verbatim); FlashInfer reports 2–3× on A100 *and* H100. Caveats: prediction for unbuilt v8; M=8-pad-16 = half row-util; the 2.92× is B200 (Modal), but FlashInfer corroborates A100/H100. |
| 5 | **b300-flat-bw** | ✅ SURVIVES 0/3 | primary NVIDIA + SemiAnalysis unanimous; one aggregator outlier (7.7 TB/s = actually H200) contradicted by dominant sources. |
| 6 | **fa4-no-decode** | ❌ DIES 3/3 | **conclusion right, premise false.** "FA4 has NO shipping decode path" is outdated — split-KV #1940 (merged 2025-11-04) … GQA-packing #2186 (merged 2026-03-20), FP8 #2109 (2026-04-17). The "vs FlashInfer/FlashMLA, not beat FA4" framing survives **for the opposite reason**: FA4 now *has* the very levers v6–v9 build → moving target. |
| 7 | **hbm-fp16-bytes** ("true %HBM is 2× higher") | ❌ DIES 3/3 | premise true (bench allocates fp32), **conclusion false**: the kernel never reads those tensors — it casts to half (`cu:274-276`); every HBM KV load is `__half` (2 B/elem). The fp16 denominator **matches** bytes-on-the-wire → printed ~10–12.5% is correct, **no 2× correction.** |

**Both deaths are "sound conclusion + stale/false premise" bundles** — the actionable recommendations (frame
vs FlashInfer/FlashMLA; %HBM is correct) stand; only the stated *reasons* were wrong and are corrected above.

---

## 6. Contradictions left unresolved (honest)

1. **Tensor cores at small G (4–8) — genuine open design choice.** FlashInfer falls back to CUDA cores at
   query-tile-1 (M<16); Character.AI/FA4/XQA describe packing as *the route to* tensor cores. Reconciliation
   (sources never state it together): clean TC engagement needs G≥16 (MLA/MQA) or pad/multi-group at G=8.
   **v8 must measure pad-16 vs multi-group vs CUDA-core-QK — unresolved until benched.**
2. **Does asymmetric NVFP4 beat symmetric NVFP4?** KV-AdaQuant's 20 pt swing is at **2-bit**; NVIDIA's
   symmetric NVFP4-KV already hits <1% at **4-bit**. The asymmetric win may be small → it's the v10 headline
   ablation; "symmetric is enough" is a legitimate possible outcome.
3. **The ~23% paging overhead (v6 vs v7).** Inseparable from clock drift (`clock~-1/-1`, separate sessions).
   v6's Step-6-RoR %HBM (8.6–15.4%) is *higher* than v7's at matched shapes — consistent with a slower v7
   clock OR real overhead; indistinguishable. **Unresolved until same-session A/B.**
4. **Exact integer SDPA crossover B** — only bracketed (1,8]; one cheap `--batch-sweep 1 2 4 8` pins it.
5. **B300 L2 size** — no primary source; do not hard-code.
6. **FA4-decode now exists** — a *moving target*; maintain the "don't claim to beat FA4" discipline as
   upstream evolves.

---

## 7. What this changes — doc deltas + roadmap verdict

**Surgical deltas applied** (this pass, append-only — existing prose preserved): [`results.md`](results.md)
Step 7 (the three new facts), [`decisions.md`](decisions.md) Step 7 (adversarial close-out + competitive
framing), [`interview-prep.md`](interview-prep.md) C10 (close-out addenda), [`decode-replan.md`](decode-replan.md)
§2.1 (adversarially-confirmed stamp) + §4.2 (FA4-decode-now-merged), [`ROADMAP.md`](../ROADMAP.md) (v7 +
v8 lines), [`CLAUDE.md`](../CLAUDE.md) (status + next-steps). Diagrams: crossover refreshed +
`v7-vs-sdpa-batch-crossover.svg` + `per-cta-limiter-anatomy.svg`.

**Roadmap verdict: NO reorder — v8 GQA M-packing stays first and correct.** Three grounds: (1) the per-CTA
limiter is exactly what M-packing fixes (G warps + GEMV→GEMM + KV-read-once); (2) at ~10% HBM, FP8/FP4
multiply a floor the kernel can't reach, so v9/v10 stay *after* v8; (3) the data **kills the batch-conditional
hedge** — there is no large-batch bytes-first regime. **Additions only:** v8 gains a "reclaim SDPA at batch"
deliverable + the M≥16 tensor-core ablation; capability target pinned to **sm_80 (A100)**; two carried-forward
cleanups (bottom-right causal mask; stale "split-KV fills SMs" comments).

**Confidence honesty:** v8's payoff is **[predicted]**, and this very project just had a [predicted] crossover
refuted by measurement — so "GQA M-packing IS the win" remains a forward hypothesis (well-grounded in measured
Part-A data + production literature, unproven on this codebase until v8 runs). That is the roofline-first,
prediction-vs-measured discipline the project is built on.

---

## 8. v8 kickoff → [`v8-kickoff.md`](v8-kickoff.md)

The copy-paste starter for a fresh session: thesis, the sm_80 capability decision, file touches, the
`AI=2G/b` roofline extension as task 1, deliverables (incl. reclaim-SDPA-at-batch and the G-sweep), the M≥16
ablation, and the vast.ai host notes.

---

## 9. Sources & method

35 agents, ~2.18 M tokens. Internal forensics read [`bench/harness.py`](../bench/harness.py),
[`kernels/v7_paged/paged_attention.cu`](../kernels/v7_paged/paged_attention.cu),
[`kernels/v6_splitkv/`](../kernels/v6_splitkv/), [`roofline/model.py`](../roofline/model.py),
[`tests/test_correctness.py`](../tests/test_correctness.py). External web research (primary-source-first):
NVIDIA Blackwell Ultra blog + compatibility guide, SemiAnalysis GB300 teardown, FlashInfer (arXiv 2501.01005),
Llama 3 Herd (arXiv 2407.21783), Dao-AILab/flash-attention PRs (#1940/#1993/#1999/#2104/#2109/#2186), Modal
FA4 blog, vLLM RFC #15351, SageAttention2 / KV-AdaQuant / KIVI. Every load-bearing claim passed a 3-vote
adversarial gate (2-of-3 refutes kill); 5 of 7 survived, the 2 deaths were stale-premise bundles whose
conclusions were re-derived correctly.

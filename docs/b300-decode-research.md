# FlashAttention-Next: beating FA4 on B300 in the decode regime

Deep-research synthesis + build plan. Compiled 2026-06-27 from: a full read of the FA4 paper
(arXiv 2603.05451) and FA3 paper; four red-team agents on the PDFs; two web-research workflows
(adversarially fact-checked); and a secondary Gemini report (red-teamed, not taken at face value).

Markings: **[fact]** = primary-source verified; **[infer]** = reasoned from the evidence;
**[spec]** = speculative / needs on-hardware confirmation.

> **▶ UPDATE 2026-06-27 (post-v6 measurement + deep-research) — read [`decode-replan.md`](decode-replan.md) first.**
> v6's measured result (only ~12% of HBM ⇒ **occupancy-bound, not bandwidth-bound**) and a 13-agent
> verify+research pass **reorder and correct** this doc:
> - **Build order (§8) is revised:** the **GQA M-packing** lever (occupancy, `AI=2/b→2G/b`) moves *ahead
>   of* the FP8/FP4 byte levers → **v7 paged · v8 GQA M-packing · v9 FP8 · v10 NVFP4 · v11 MLA.** Bytes
>   only pay once the kernel is actually near the bandwidth wall.
> - **"Decode is memory-bound" is correct but batch-conditional:** the 12% is a `BH=8` artifact;
>   split-KV self-disables past `BH≈2·SM` so batch alone fills the SMs (predicted, not yet measured — v7
>   adds a `--batch` sweep).
> - **§4 correction:** `AI*_BF16 ≈ 310` → **≈ 437.5** (3.5 PF / 8 TB/s); the 310 back-solved to an
>   unsourced 2.48 PF.
> - **§6 correction / claim discipline:** FlashInfer (vLLM GB300, Feb 2026) and FlashMLA (SGLang GB300
>   DeepSeek-V4 day-0, Apr 2026) **are** B300-proven (the #2939 deadlock was fixed); FA4 is *acquiring* a
>   decode path (Modal upstreamed split-KV + GQA-packing). So the contribution is "an **open,
>   roofline-documented FP4 decode kernel** vs FlashInfer/FlashMLA," not "we beat FA4."
> - **Verified [fact]:** B300 HBM bandwidth is **flat 8 TB/s** vs B200 (only capacity grew 192→288 GB);
>   decode `AI = 2/b` is N-independent and memory-bound; the 2× MUFU.EX2 exp doubling is real.

---

## 0. Thesis

FA4 is a **BF16, compute-bound, large-tile kernel built for B200 prefill/training.** Decode on B300
is a **low-precision, memory-bound, GEMV-shaped** problem on **different silicon.** So the way to beat
FA4 is not a better GEMM — it is to **read far fewer KV bytes (FP4-microscaled KV cache) and fill all
160 SMs (split-KV)**, while deleting the machinery FA4 built to dodge a bottleneck B300 halves.

This is not a fight on FA4's home turf: FA3 explicitly deferred "optimizing for LLM inference" as
future work **[fact]**; FA4 never closed it; and cuDNN 9.19 already matches FA4 on BF16 prefill
**[fact]**. Decode + low-precision on B300 is the open lane.

---

## 1. What FA4 is, and the regime trap

FA4's premise is **asymmetric hardware scaling** **[fact]**: H100→B200 doubled BF16 tensor-core
throughput (1→2.25 PFLOPS) while the exponential/SFU unit and shared-memory bandwidth stayed flat, so
softmax + smem traffic became the bottleneck. Its three signature moves all attack that:

1. Async MMA + larger 128×128 / 256×128 tiles to overlap softmax under matmul.
2. Software-emulated `exp2` — a degree-3 Horner polynomial on FMA units (coeffs `p0=1.0, p1≈0.6951,
   p2≈0.2276, p3≈0.0771`, Sollya-fit) **[fact]** — plus conditional softmax rescaling (skip unless the
   running max jumps by `τ=log2 256=8`).
3. TMEM accumulators + 2-CTA MMA mode to cut smem traffic and global atomics in the **backward** pass.

Every move assumes the query tile `M ≥ 128`. The benchmarks confirm it: square attention
(`seqlen_q = seqlen_kv`), 1K–32K, BF16, forward **and** backward, on B200, hitting 71% of 2.25 PFLOPS
**[fact]**. **Decode is `M = 1`** — `Q·Kᵀ` and `P·V` collapse to GEMV and the premise inverts.

---

## 2. Consolidated blind spots (four independent red-teams agreed)

| # | FA4 choice | Why it's prefill-shaped | Decode-on-B300 consequence |
|---|---|---|---|
| 1 | 128×128 / 256×128 MMA tiles | Feed the doubled tensor cores | `M=1` → **≤0.8% MMA utilization**; B300 doubling TC worsens it |
| 2 | Grid `(mblocks, heads, batch)` | Enough worktiles in prefill | Decode → `(1, heads, batch)`; few CTAs vs 160 SMs → SM starvation |
| 3 | Ping-pong two-Q-tile softmax overlap | Hide softmax under a fat MMA | No second tile, no fat MMA → the latency-hiding mechanism idles |
| 4 | FMA polynomial `exp2` emulation | Dodge B200's 16-ops/clk MUFU | B300 doubles MUFU to 32 ops/clk; FA4 **already disables it on sm103** |
| 5 | Dense contiguous BF16 KV | Reuse KV across query tiles | Decode needs paged + quantized KV; FA4 has neither |
| 6 | BF16 throughout | "FP4 is consumer-GPU work" (p.2) | Leaves B300's 15-PFLOP NVFP4 unused — the biggest decode lever |
| 7 | `τ=8` rescale, row-per-thread softmax | Amortized over 128-row tiles | Degenerate at `M=1`; never validated for decode |
| 8 | 2-CTA pairing, correction warpgroup, TMEM budget | Relieve B200 backward pressure | Inference is forward-only — pure overhead + launch rigidity |

Tell: FA4 is already weakest right next to decode — at 1K–2K sequences it is at parity or **behind**
cuDNN/Gluon (its own Figs 4–5) **[fact]**; the gap should widen as `seqlen_q → 1`.

Precision note: FA4 is not literally decode-blind — it has decode-**aware scheduling** (LPT
batch-sorting for continuous batching) **[fact]**. What it lacks is a decode-specialized **compute
kernel**.

Empirical confirmation of #4: GitHub `Dao-AILab/flash-attention` issue **#2596** **[fact]** —
"ex2_emu breaks decode/prefill bitwise consistency on SM100 (MLA 192,128)." Decode kernels
(`seqlen_q=1`, paged) have a small fragment count → already use **hardware** exp2; prefill uses the
polynomial. So FA4 effectively already drops the emulation in decode, and the maintainer gates it off
entirely under `is_sm103` **[fact]**.

---

## 3. B200 → B300: verified seams

| Spec | B200 | B300 / Blackwell Ultra | Source |
|---|---|---|---|
| HBM3e capacity | 192 GB | **288 GB** (+50%) | NVIDIA **[fact]** |
| HBM bandwidth | ~8 TB/s | **8 TB/s (FLAT)** | NVIDIA **[fact]** |
| NVFP4 dense | ~10 PF | **15 PFLOPS** (20 sparse) | NVIDIA **[fact]** |
| FP8 dense | ~4.5 PF | **5 PFLOPS** (10 sparse) | NVIDIA table **[fact]** |
| BF16 dense | 2.25 PF | not in NVIDIA table (~3.5 per third-party) | **[spec]** |
| Exp / SFU | 5 TExp/s | **10.7 TExp/s** (~2×) | NVIDIA **[fact]** |
| SMs | 148 | **160** | NVIDIA **[fact]** |
| L2 cache | ~128 MB | **~192 MB** | third-party **[spec]** |
| TMEM / SM | 256 KB | 256 KB | **[fact]** |
| Compute capability | sm_100 (CC 10.0) | **sm_103 (CC 10.3)**, CUDA 12.9; family-portable `sm_100f`, but B300-exclusive features need `sm_103a` | NVIDIA **[fact]** |

**Strategic surprise: B300 HBM bandwidth is FLAT vs B200.** Only capacity grew. Since decode is
HBM-bandwidth-bound, the B300 wins are exactly three, and they map onto the design:
- **+50% FP4 + the format existing** → FP4 KV cache reads **~3.5× fewer bytes** (the real "bandwidth" win).
- **2× exponential** → delete FA4's exp emulation; run hardware `exp2`.
- **+50% capacity, bigger L2, +12 SMs** → longer context / bigger batch, more KV resident, more split-KV parallelism.

---

## 4. The decode roofline (the math)

One decode step, `N` KV tokens, head dim `d`, `b` bytes per KV element:
- Work: `Q·Kᵀ = 2Nd` FLOPs, `P·V = 2Nd` FLOPs → **4Nd FLOPs**.
- Traffic: read K and V → **2Nd·b bytes** (Q, O are O(d), negligible).

**Arithmetic intensity `AI = 4Nd / (2Nd·b) = 2/b` FLOP/byte — independent of N.** Pure memory-bound.

| KV precision | bytes b | AI |
|---|---|---|
| BF16 | 2 | 1.0 |
| FP8 | 1 | 2.0 |
| NVFP4 (1×16 + E4M3 scale) | ~0.56 | ~3.5 |

B300 ridge points: `AI*_FP4 = 15e15 / 8e12 ≈ 1875`; `AI*_BF16 ≈ 437.5` *(= 3.5 PF / 8 TB/s; corrected
from `≈310`, which back-solved to an unsourced 2.48 PF — see `decode-replan.md §3`)*. Decode (AI 1–4) is **~300–500×
below the ridge** — the tensor cores are nearly idle and `time/token ≈ KV_bytes / 8 TB/s`.

Escape: **share KV across query heads.** GQA (`G` heads per KV head) → `AI = 2G/b` (GQA-8, FP4 ≈ 30).
MLA (one latent KV shared across all heads) pushes AI toward the ridge — why FlashMLA exists and why
MLA is the decode-friendly architecture. This is the "repack GEMV→GEMM" lever, quantified.

---

## 5. The smart trick — ASP-Decode (asymmetric-precision split-KV FP4 decode)

The two matmuls have **opposite precision tolerance**:
- After softmax, `P ∈ [0,1]` is a normalized distribution (bounded, sums to 1, no outliers) → `P·V` is
  a convex combination where FP4 error **averages out**. **Safe in FP4.**
- `Q·Kᵀ` feeds an exponential → error is amplified (`δ → e^δ`), and logits carry outliers. **Unsafe —
  keep in MXFP8 + outlier residual.**

| Lever | What | Why it beats FA4 |
|---|---|---|
| (a) Split-KV | Partition KV across SMs; partial `(O,m,ℓ)`; LSE merge (reuse 2-CTA cluster + DSMEM for on-chip merge) | FA4 has no KV-axis parallelism |
| (b) NVFP4 KV cache | 1×16 blocks, E4M3 scales | ~3.5× fewer bytes — the decode bottleneck |
| (c) Asymmetric precision | scores MXFP8 + outliers; softmax FP32 on B300 hw exp2; `P·V` NVFP4 + two-level `P` scaling (`sP1 = rowmax/(448·6)`) | captures FP4 throughput without breaking softmax |
| (d) Delete | exp emulation, 2-CTA pairing, correction warpgroup, `τ` heuristic | overhead for the wrong regime/chip |
| (e) Deep TMA paged-KV prefetch | ≥3-stage ring over block-table KV | KV is read once in decode (no L2 reuse) → hide HBM latency |

Risk: the score side is the knife-edge. FP4 *scores* without outlier handling collapse the softmax —
hence MXFP8 + outlier residual on scores, FP4 only on KV/`P·V`. Must be validated end-to-end vs FP32
(perplexity, not MSE).

---

## 6. Competitive landscape (measure against / borrow from)

"Beat FA4 in decode" is nearly free (FA4 is not a decode kernel). The real bar is the best decode
kernel. **None of the leaders below is B300-proven** — that is the wedge.

| Kernel | Decode result (verified) | Role |
|---|---|---|
| Flash-Decoding | batch=1 classic FA uses <1% of an A100; split-KV ~50× on the attention op, ~8× end-to-end long-context | v6 foundation |
| FlashMLA (DeepSeek) | MLA latent-KV; ~93% KV-cache reduction; ships an SM100 backend | MLA-decode baseline |
| GLA (Grouped Latent Attention) | ~20% faster than FlashMLA @ QL=1, >2× @ QL=2 (H100); halves per-device KV at TP≥2 | MLA-decode SOTA to study |
| LeanAttention | Stream-K decode reduction: 2.6× over FA2, up to 8.33× @ 512k | merge-stage technique |
| FlashInfer / TRT-LLM (sm103) | immature — Blackwell-Ultra FMHA deadlock on GB300 (#2939, reverted #2956) | the practical opening |
| SageAttention3 | FP4 attention on Blackwell (NVFP4, two-level P); 1038 TOPS, 5× over FA2 — but **consumer GPU (RTX 5090)** | the FP4 accuracy recipe |

Out of scope but noted: Gemini's "FA5" (Attn-QAT auxiliary caching + 4-CTA backward) is a **training**
kernel — there is no backward pass at inference. Attn-QAT (arXiv 2603.00040) is explicitly QAT
training on RTX 5090 **[fact]**. 4-CTA MMA could not be verified as a real hardware primitive (Blackwell's
documented cooperative MMA is 2-CTA) **[spec]** and is irrelevant to M=1 decode regardless.

---

## 7. Ranked tricks (feasibility × payoff)

| Rank | Trick | Payoff | Feasibility | Risk |
|---|---|---|---|---|
| 1 | Split-KV / flash-decoding | ★★★★★ | ★★★★★ | low |
| 2 | NVFP4 KV cache (~3.5× bytes) | ★★★★★ | ★★★ | accuracy |
| 3 | Asymmetric precision (MXFP8 scores / FP4 P·V) | ★★★★ | ★★★ | accuracy |
| 4 | Delete exp emulation + B300 retune | ★★★ | ★★★★★ | low |
| 5 | GQA M-packing (GEMV→GEMM) | ★★★ | ★★★ | needs batch/GQA |
| 6 | MLA latent-KV decode | ★★★★★ | ★★ | scope |

---

## 8. Build plan (v6 → v11), roofline-first

> **⚠ REORDERED post-v6 (see the UPDATE banner + [`decode-replan.md §5`](decode-replan.md)):** the order
> below was written bytes-first. v6 measured occupancy as the limiter, so the live order is **v7 paged ·
> v8 GQA M-packing (occupancy) · v9 FP8 · v10 NVFP4 (headline) · v11 MLA** — GQA moves ahead of the byte
> cuts. The per-step entries below are still accurate descriptions; only their *sequence* changed.

Loop each step: predict limiter → explain → build + hand-verify → correctness vs reference → bench →
roofline read (prediction vs measured) → quiz. Metric shifts to **µs/token, % HBM bandwidth, accuracy
vs FP32**.

- **v6 — split-KV decode (FP16)** · T4/rented · *deliverable:* decode roofline + LSE merge; beats a naive `seqlen_q=1` loop; % HBM BW reported. **Keystone, no B300 needed.**
- **v7 — paged KV gather** · T4/rented · block-table indirection; correctness with non-contiguous KV.
- **v8 — FP8 KV cache** · rent B300 #1 · ~2× bytes cut; accuracy vs FP32 documented.
- **v9 — NVFP4 + asymmetric precision** · B300 · the headline; ~3.5× bytes; beats FA4 µs/token; end-to-end accuracy table.
- **v10 — GQA M-packing + B300 retune** · B300 · GEMV→GEMM; 160 SMs; drop ex2_emu; % of FP4 peak.
- **v11 — MLA / speculative decode (stretch)** · B300 · raise AI toward the ridge; FlashMLA-class comparison.

Rent B300 in concentrated bursts for the precision/MLA steps (v9/v10/v11 in the reordered plan — see the
UPDATE banner); develop v6–v8 cheap (T4 / rented A100/H100).

---

## 9. Honest caveats

- Claim discipline: "beats FA4 in decode" is nearly free; the real result is "competitive with /
  beats FlashInfer/SageAttention3/GLA-class decode on B300 GQA models." State it precisely.
- FP4 accuracy is the genuine risk — scores are fragile; asymmetry + outlier residual + end-to-end
  eval are mandatory.
- B300 bandwidth is flat vs B200 — the win is precision + occupancy + exp, not GB/s. The roofline
  must predict that, and you must measure it.
- Counter-free profiling stays the norm (ncu `ERR_NVGPUCTRPERM`): `max_memory_allocated` for bytes,
  CUPTI / torch.profiler for µs/token.
- Confirm B300 dense FP8/BF16 peaks and L2 size on-chip — NVIDIA's table lists FP4 15 / FP8 5 dense
  and no BF16 row; third-party 7.0/3.5 figures are unverified.

---

## Sources (primary)

- FA4 paper, arXiv 2603.05451 (and FA3 paper) — local PDFs.
- NVIDIA, "Inside NVIDIA Blackwell Ultra" — B300 specs.
- NVIDIA, "Making Softmax More Efficient with NVIDIA Blackwell Ultra" — 2× SFU microbenchmarks.
- NVIDIA, "Family-specific architecture features" + CUDA GPUs table — sm_103 / CC 10.3 / CUDA 12.9.
- Colfax, "FlashAttention-4 … co-design" — exp-emulation coefficients, TMEM/2-CTA.
- GitHub Dao-AILab/flash-attention #2596, #2595, #2358; flashinfer-ai/flashinfer #2939/#2956.
- SageAttention3, arXiv 2505.11594 — NVFP4 two-level P, 1038 TOPS.
- Attn-QAT, arXiv 2603.00040 — FP4 QAT (training; RTX 5090).
- Flash-Decoding (Stanford CRFM / PyTorch), FlashInfer (arXiv 2501.01005), FlashMLA (DeepSeek), LeanAttention.

Diagrams (roofline, asymmetric-precision dataflow, split-KV schedule, v6→v11 roadmap) were generated
in the research session.

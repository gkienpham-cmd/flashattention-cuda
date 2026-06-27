# FlashAttention-next: research menu (FA-3 / FA-4 → what to build on)

Source-of-record for the "where does attention go after FA-4" question. Built from a direct read of
the FA-3 (arXiv 2407.08608) and FA-4 (arXiv 2603.05451) papers plus a fan-out deep-research pass
(23 sources, 105 claims extracted, 25 adversarially verified at 3-vote / 2-of-3-to-kill; 24
confirmed, 1 killed). Every quantitative claim below carries a verification status. Dated 2026-06-27.

This is a **research log, not a decision record** — no kernel committed to yet. The per-step loop in
`CLAUDE.md` still gates any actual `vN` work. The point here is to have the cited menu on hand when
we scope the next step, and to mark which levers are reachable on our **T4 (Turing sm_75)** vs which
need a rented Hopper/Blackwell box.

---

## 0. The thesis everything hangs on: asymmetric hardware scaling

FA-1/2/3 played one game: **feed the tensor cores** (FA-2 was only ~35% util on H100; FA-3 got to
75%). FA-4's premise is that the game changed because the hardware scaled *unevenly*:

- **H100 → B200: BF16 tensor-core throughput went 1 → 2.25 PFLOPS, but the SFU/exponential unit and
  shared-memory bandwidth did NOT move.** [✅ 3-0, Tri Dao blog + Colfax + Together + arXiv]
- Per-SM B200 budget: **tensor core 8192 ops/cyc, exponential (MUFU) 16 ops/cyc, SMEM 128 B/cyc.**
  [✅ 3-0]
- Result: FA-4 reaches **~1605–1613 TFLOPS = 71% of B200 peak**, and the remaining ~29% is *not*
  tensor-core-bound — it is **(a) the SFU/exp unit in the forward pass, (b) shared-memory traffic in
  the backward pass.** [✅ 3-0]

Why this matters for us: it is the **same lesson our T4 curve keeps hitting** — Steps 2/3/4 each found
the limiter was scheduling / occupancy / non-matmul overhead, never HBM bandwidth. FA-4 is that
finding promoted to the top of the hardware stack. The roofline reframe: model the **exp unit and
SMEM** as first-class limiters, not just MMA and HBM. (Our `roofline/model.py` already has a MUFU
term; this says make it central.)

---

## 1. Menu of build-on-able techniques

Grouped by which bottleneck they attack. Each carries payoff + regime + verification.

### A. Attack the exponential unit (forward-pass limiter)

- **Polynomial exp on FMA units** (FA-4's own headline). Emulate `2^x` with a **degree-3 Cody-Waite
  polynomial** on the FMA units, running in parallel with the MUFU, applied to only ~10–25% of
  entries. Coefficients (Horner form): `p0=1.0, p1≈0.6951, p2≈0.2276, p3≈0.0771`, chosen via Sollya.
  Degree-3 matches hardware MUFU within ~1 ULP *after BF16 rounding*, so no accuracy cost at BF16.
  [✅ 3-0; coefficients independently reverse-engineered from the FA-4 SASS by Modal]
- **Conditional softmax rescaling** (FA-4). Skip the online-softmax rescale when `m_j − m_{j-1} ≤ τ`;
  decide at **warp granularity** to avoid divergence; the final normalization corrects the skipped
  steps so the output is exact. Removes most rescale vector-mults (a non-matmul cost). [✅ 3-0]
- **Hardware partly bails you out:** **B300/GB300 (Blackwell Ultra) doubles SFU/exp throughput
  16 → 32 ops/cyc/SM** (~2× measured, e.g. BF16 ~4938 → ~9738 Gop/s). So the *software* exp-emulation
  payoff shrinks on GB300, but still matters on B200/Hopper/Turing. [✅ 3-0]

### B. Attack shared-memory traffic (backward-pass limiter)

- **TMEM-resident accumulators.** Blackwell **tensor memory: 256 KB/SM** (512 cols × 128 lanes × 32b);
  MMA writes outputs directly to TMEM asynchronously instead of into registers. Kills the
  register-pressure wall that forced FA-3's serialized compute graph; frees registers for larger
  tiles. Foundational on Blackwell, not optional. [✅ 3-0]
- **2-CTA / tcgen05 "pair-UMMA".** Two CTAs in a cluster jointly compute one MMA tile (M=128 or 256),
  each staging **half of operand B** in its own SMEM and sharing via **DSMEM**. **Halves SMEM traffic
  for operand B**; in the backward dQ step it also **halves global atomic adds**. [✅ 3-0]
- **Design rule — small-N tiles are SMEM-bound:** on SM100, SS-mode MMA is *entirely*
  SMEM-bandwidth-bound for **N < 128** (FP16 M128/N64/K16: 48 SMEM cyc vs 32 math cyc); only at N≥128
  does it go math-limited. ⇒ **keep attention scoring tiles ≥128 in N** or the tensor cores idle on
  SMEM. [✅ 3-0]

### C. Drop precision (biggest raw multiplier — but trades exactness)

These are **approximate** attention unless noted. Verified speedups:

| Method | Precision | Measured | Status |
|---|---|---|---|
| SageAttention | INT8 QK | ~2.1× over FA-2 (~2.7× over xformers) | ✅ 3-0 |
| SageAttention2 | INT4 QK (per-thread) + FP8 PV | ~3× the OPS of FA-2 | ✅ 3-0 |
| SageAttention3 | **NVFP4** (1×16 microscale, E2M1 data + E4M3 scale) on Blackwell FP4 TC | **1038 TOPS on RTX 5090 = 5× fastest FlashAttention on that GPU** | ✅ 3-0 |
| FA-3 FP8 | FP8 + block-quant + incoherent (Hadamard) | ~1.2 PFLOPS; 2.6× lower error than naive per-tensor FP8 | ✅ 3-0 |

⚠️ **KILLED claim (3-0 refute):** "SageAttention2 matches FP8 FA-3 speed on Hopper while being more
accurate." Sources do **not** support that head-to-head — do not repeat it. Honest framing:
low-precision buys 2–5× but trades exactness; accuracy is *preserved* by **block / microscaling
quantization + incoherent (Hadamard) processing**, which is the transferable technique.

### D. Algorithmic reformulations (keep exactness, change the math)

- **PASA (online pseudo-average shifting attention).** Mathematically *equivalent* to FlashAttention's
  online softmax, but reformulates the running shift so the whole pass runs in **FP16 without
  overflow**. The one genuinely **exact + low-precision-enabling** algorithmic idea found. Directly
  de-risks our planned FP32→FP16 transition. [✅ 3-0]
- **Softpick / softmax-free variants.** Rectified, *not-sum-to-one* replacements that eliminate
  attention sinks — but they **change the operator**, so not exact attention. Filed as "different
  model," not a drop-in. [✅ verified approximate]

### E. Implementation substrate & inference-specific kernels

- **CuTe-DSL (Python).** FA-4 is written entirely in it: **20–30× faster compiles** than C++ templates.
  Productivity lever, not a perf lever.
- **A portable DSL can match hand-tuned CUDA:** a pure-**Triton** paged-attention kernel was tuned from
  19.7% → **98.6–105.9% of FA-3** on H100. [✅ 3-0] Counters the "must be CUDA C++" assumption.
- **ThunderKittens** matches FA-3 forward and **beats it ~10–40% on the backward pass** (tile
  abstraction). [✅ 3-0]
- **Inference ≠ training kernel.** **Flash-Decoding** adds a KV-split parallelism dim FA lacks at
  decode (query length 1). **FlashInfer** (now the vLLM/SGLang backend) gets **29–69% inter-token
  latency** reduction. **NVFP4 KV-cache** cuts KV memory **~50% vs FP8** and **~3× better TTFT**. Prefill
  and decode want different designs. [✅ 3-0] — relevant to the long-term mini-vLLM goal.

---

## 2. What a "FlashAttention-next" design actually combines

Not one trick — a co-design picking one lever per bottleneck:

1. **Forward:** polynomial-exp offload + conditional rescaling → starve the SFU less (on GB300, lean
   on the 2× hardware exp instead).
2. **Backward:** 2-CTA UMMA + TMEM accumulators + DSMEM operand-sharing → halve SMEM traffic and dQ
   atomics.
3. **Precision:** if the app tolerates it, **NVFP4 microscaling + incoherent processing** is the
   single biggest multiplier (3–5×) — but approximate; gate behind an accuracy budget.
4. **Tiling:** N≥128 or the tensor cores idle on SMEM.
5. **Substrate:** prototype in Triton / CuTe-DSL; ~100% of CUDA SoTA reachable without C++ templates.

**The open research seam:** the **exact-attention × low-precision intersection** — PASA-style
reformulations that allow FP8/FP4 *without* the quantization-error tax. That's the lane where "more
optimized AND still exact" lives.

---

## 3. Reachability from our T4 (Turing sm_75)

Most FA-4 headline levers — TMEM, 2-CTA UMMA, FP8/FP4/NVFP4 tensor cores, WGMMA — **do not exist on
Turing.** Splitting the menu:

**Portable to the T4 curve (no rental, stays apples-to-apples with v1–v5 baselines):**
- **Conditional softmax rescaling** — pure algorithm, any GPU; clean candidate for a Step-6 increment.
- **Polynomial-exp-on-FMA** — Turing has the same MUFU/`exp2` bottleneck; lets us measure the exact
  SFU-vs-FMA tradeoff the roofline tool loves.
- **PASA-style FP16 online softmax** — directly de-risks the FP32→FP16 jump (v1–v4 were FP32).
- **N≥128 tiling rule + asymmetric-scaling roofline framing** — sharpens `roofline/model.py`.

**Requires renting Hopper/Blackwell:** TMEM, 2-CTA/UMMA, FP8/FP4/NVFP4 — i.e. the whole FA-3/FA-4
headline act. Per `CLAUDE.md`, the v1→v5 curve stays on T4 for apples-to-apples; a bigger-GPU re-run
is a deliberate later pass.

---

## 4. Sources (quality-tagged)

Primary unless noted. FA-4 figures cross-confirmed across the author blog, Together.ai, Colfax
(co-authors), and arXiv.

- FA-4: tridao.me/blog/2026/flash4 · together.ai/blog/flashattention-4 ·
  research.colfax-intl.com/flashattention-4-... · arxiv.org/abs/2603.05451 (html: 2603.05451v1)
- FA-3: arxiv.org/abs/2407.08608
- Low-precision: SageAttention3 arxiv.org/abs/2505.11594 · SageAttention2 arxiv.org/abs/2411.10958 ·
  SageAttention arxiv.org/abs/2410.02367
- Algorithmic: PASA arxiv.org/abs/2503.01873 · Softpick arxiv.org/abs/2504.20966 ·
  (TurboAttention/FlashQ arxiv.org/abs/2412.08585 — approximate)
- Blackwell HW: jianyuh.github.io/cuda/2026/04/12/blackwell-sm100.html (blog) ·
  Colfax TMEM + thread-block-cluster CUTLASS tutorials ·
  developer.nvidia.com/blog/making-softmax-more-efficient-with-nvidia-blackwell-ultra (GB300 exp 2×)
- Implementations / inference: FlashInfer arxiv.org/abs/2501.01005 ·
  Flash-Decoding crfm.stanford.edu/2023/10/12/flashdecoding.html ·
  ThunderKittens (HazyResearch) · Triton paged-attention openreview.net/pdf/f4b2b2d3...aadc5.pdf ·
  NVFP4 KV-cache developer.nvidia.com/blog/optimizing-inference-...-nvfp4-kv-cache ·
  developer.nvidia.com/blog/next-generation-of-flashattention (cuDNN absorbed FA techniques)

**Method note:** one synthesis-stage agent returned a placeholder; report reconstructed from the
verified-claim set (this is why claims carry explicit vote tallies). Killed claim recorded in §1C.

# v8 kickoff — GQA M-packing (copy-paste into a fresh Claude session)

> **How to use this file:** paste the block below into a new session as the first message. It is
> self-contained — it assumes no memory of the v7 deep-research conversation. Everything it references
> (file paths, the measured limiter, the capability decision, the deliverables) is spelled out. Then
> follow the per-step loop in [`ROADMAP.md`](../ROADMAP.md) / [`CLAUDE.md`](../CLAUDE.md): **roofline
> first, explain, write, correctness, benchmark, update `docs/`, quiz.**

---

## ✂️ PASTE-ME — v8 starter prompt

We're starting **Step 8 (v8 — GQA M-packing)** of the FlashAttention-from-scratch project (read
`CLAUDE.md` and `ROADMAP.md` first). v7 (paged-KV decode) is closed: Gate 1 ✅ 51/51, quiz passed. The
v7 deep-research close-out is in [`docs/v7-deep-research.md`](v7-deep-research.md) — read it; it is the
justification for everything below.

**Where we are (measured, not assumed).** v7 decode is **per-CTA-bound at every batch size** — the
`--batch` sweep measured `%HBM` FLAT at 9.4–12.4% from BH=8→512 (no occupancy→bandwidth crossover; that
prediction was refuted 0/3 in adversarial review). The wall is *inside* the CTA, code-verified:
`sK+sV = 32 KB`/block caps residency at **2 blocks/SM** (T4 64 KB/SM), and at `N_q=1` only **1 of 8 warps
computes**. Batch adds waves, not per-SM parallelism. Also measured: **torch SDPA overtakes v7 by B=8**
(v7 wins only at B=1), so v8 must **reclaim the serving-batch regime**.

**The v8 thesis.** Pack the `G = H_qo / H_kv` query heads that share one KV head into the score GEMM's
**M dimension**: `S = Q_pack[G×d] @ Kᵀ[d×N_tile]` becomes an **M=G GEMM** instead of G width-1 GEMVs.
Effects: **AI = 2/b → 2G/b**, KV read **once** per CTA (not G×), and **G compute-warps active** — this
directly fixes v7's measured "1-of-8 warps + GEMV-latency" limiter. This is the per-CTA-efficiency lever,
promoted ahead of low-precision (FP8 v9 / NVFP4 v10) because cutting bytes is premature while the kernel
sits ≥8× below the bandwidth wall at every batch size.

**Capability decision: target sm_80 (A100), `[RENT]` — NOT sm_75 (T4 WMMA).** Reason: tensor cores
engage only at **M ≥ 16** (FlashInfer: *"tensor core instruction m minimum rows is 16"*). The clean path
is the Ampere `mma.m16n8k16` atom + `cp.async` KV staging (Turing lacks `cp.async` and its 16×16×16 WMMA
fragment can't express a padded M=8 tile efficiently). Add a B300/sm_103 arch entry later for v9/v10 byte
work.

**Roofline first (do this BEFORE coding — the per-step gate).** Extend `roofline/model.py` `estimate()`
to take a GQA group factor `G` (default 1) and compute the decode `AI = 2G/b` (share KV across the G
heads: `hbm_bytes = 2·(bh/G)·N_k·d·b + …`). Predict: at G=8, AI = 8.0 (FP16) — does the limiter flip
HBM→compute? Where's the new ridge? Record the prediction, then measure it (the project's whole point is
prediction-vs-measured, and v7 just had a prediction refuted — stay honest).

**The ablation that IS v8 (where the sources disagree — measure all three).** G=8 < 16, so a single
Llama-3-70B group doesn't fill a tensor-core tile:
1. **pad M=8→16 + mask** (runs the MMA at half row-utilization), vs
2. **pack 2 KV-groups** so M is a multiple of 16, vs
3. **CUDA-core FMA for QK scores, tensor cores only for P·V.**
The win (KV read once, AI→O(G), G warps) is real either way; the *magnitude* is what this choice decides.

**Deliverables (the step isn't done until all are green + quiz passed).**
1. **(TASK 1) `AI=2G/b` roofline extension** in `roofline/model.py` — the roofline-first prediction.
2. **The G-sweep** — `G ∈ {1,2,4,8,16,32}` at fixed N_k (v8's analogue of v7's batch sweep): the
   prediction-vs-measured curve as G crosses the M<16→M≥16 tensor-core threshold.
3. **RECLAIM SDPA AT BATCH** — show v8 beats torch SDPA at B≥8 (where v7 lost ~0.5×). The headline v8
   deliverable the v7 data created.
4. **Correctness 0-fail** vs SDPA, tol 2e-2 (FP16-in/FP32-accum), across: non-multiple N_k; G not
   dividing the warp tile; causal + decode query-offset; prefill square-shape (N_q=N_k) reduction.
   *(flash-attention's `pack_gqa` had an illegal-memory-access bug under varlen — test these explicitly.)*
5. Bench vs **v6/v7/SDPA/FlashInfer** at the canonical **G=8, d=128** (Llama-2/3-70B) and **G=4, d=128**
   (Llama-3-8B) configs.

**File touches (v8 is greenfield — no GQA scaffolding exists yet; fork v7 as the template).**
- `kernels/v8_gqa/{gqa_attention.cu, binding.cpp}` — new pair. 3D grid `(batch, kv_head, kv_split)`; pack
  G into M; `cp.async` KV→smem once; `mma.m16n8k16` QK + PV with FP32 accum; carry the LSE merge from
  `kernels/v6_splitkv/` / `kernels/v7_paged/`.
- `bindings/load.py` — add `"v8_gqa": ["gqa_attention.cu", "binding.cpp"]` to `_SOURCES`.
- `fa_kernels/dispatch.py` — `_MIN_CAPABILITY["v8_gqa"] = (8, 0)`.
- `fa_kernels/reference.py` — add a GQA-aware reference (accept `n_kv_heads`, broadcast-expand KV by
  `G = n_q_heads / n_kv_heads`) so correctness has an oracle.
- `roofline/model.py` — the `G` / `AI=2G/b` extension (**task 1**). `roofline/archs.py` — add A100 (sm_80)
  and a B300 (sm_103: HBM 8e12 B/s, 288 GB, NVFP4 15e15, FP8 5e15, 160 SMs, exp 10.7e12/s — keep L2
  unset, no primary source).
- `bench/harness.py` — add a **`--gqa-group G`** sweep; a **same-session v6/v7/v8 A/B** (resolves the
  unproven ~15–25% paging overhead); a **bottom-right causal mask** in the SDPA reference (fixes the
  causal `vs_sdpa` artifact); **reclaim-SDPA-at-batch** rows (B≥8).
- `tests/test_correctness.py` — add `"v8_gqa"` to `BACKENDS` (and `"v7_paged"` if isolating paged tests);
  add the four cases above.
- `notebooks/v8_gqa_gate.ipynb` — standalone Run-All gate (fork `notebooks/v7_paged_gate.ipynb`).

**Comparators & claim discipline.** Bench against **FlashInfer** (`BatchDecodeWithPagedKVCache(use_tensor_cores=True)`)
and **TRT-LLM XQA** as M-packing SOTA, and **vLLM PagedAttention v2** as the no-M-packing CUDA-core floor
(the cleanest isolation of v8's one variable). Realistic envelope: ~2.9× single-token decode at the
**op level** (claim it as op-level, not end-to-end; end-to-end is ~30%). **Do NOT frame as "beat FA4"** —
FA4's decode path is now upstreamed (`pack_GQA` 2.92×), so it's a moving target. Frame: "an open,
roofline-documented decode kernel measured vs FlashInfer/FlashMLA."

**Open design questions to resolve while building.**
- The M≥16 ablation (pad-16 / multi-group / CUDA-core-QK) — the main unknown.
- `BLOCK_M ∈ {16,32,64}` with `M_block % G == 0` (FlashMLA uses 64).
- Keep split-KV (orthogonal — grid = batch×kv_head×split); confirm the occupancy crossover `BH` drops by G.
- Reuse lever: design the M=G GEMM so **one kernel serves prefill (N_q=N_k) and packed-decode (N_q=1,
  M=G)** — how FlashInfer does it (and it feeds the `from fa_kernels import attention` API goal).

**Carried-forward cleanups (not blockers, fold into v8's harness pass):** the bottom-right causal-mask
reference; the stale comments asserting the refuted "split-KV fills the SMs → HBM ceiling" in
`kernels/v7_paged/paged_attention.cu:221-226` and `roofline/model.py:96-99`.

**GPU host notes (vast.ai T4 for correctness; rent A100 for the perf deliverable).** torch installs into a
venv (`/venv/main`) — install with `%pip` / `sys.executable -m pip` and prepend the venv `bin/` to PATH so
`!python -m …` resolves; add `numpy`. Counter-free profiling is the norm
(`torch.cuda.max_memory_allocated()` + CUDA-event timing); `ncu` counters are blocked
(`ERR_NVGPUCTRPERM`) on containerized rentals — a bare-metal box is needed only for the optional pipe-util
confirmation.

---

*Full evidence + citations: [`docs/v7-deep-research.md`](v7-deep-research.md). Decode-arc math + diagrams:
[`docs/decode-replan.md`](decode-replan.md) §5. Diagrams: `diagrams/gqa-mpacking.svg`,
`diagrams/per-cta-limiter-anatomy.svg`, `diagrams/decode-roofline-crossover.svg`.*

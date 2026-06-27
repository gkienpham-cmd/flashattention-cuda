# CLAUDE.md — FlashAttention from scratch

A roofline-driven journey rebuilding FlashAttention one measured speedup at a time. The deliverable
is the **measured speedup + roofline curve** (prediction vs reality), not just fast code. Long-term
goal: a stable `from fa_kernels import attention` API that a future from-scratch mini-vLLM consumes.

## How this project works (read before changing anything)

**The per-step loop** (in `ROADMAP.md`; every kernel version follows it, in order):
1. **Roofline first** — predict the limiter (MMA / HBM / MUFU) with `roofline/` *before* coding.
2. **Explain** the memory-hierarchy / scheduling reasoning.
3. **Write** kernel + binding; hand-verify the core loop.
4. **Correctness** vs SDPA with a documented tolerance.
5. **Benchmark** vs SDPA across seq 512/2K/8K × d 64/128.
6. Update `docs/results.md` (speedup + prediction-vs-measured + ncu reading).
7. Update `docs/decisions.md` (bottleneck, options, choice, what changes on another arch).
8. **Quiz** the user before moving on.

**A step is not done until tests are green on Colab AND the user passes the quiz.** Honor this gate
— do not start the next kernel before both. The user (Kien) is *learning*: teach roofline-first,
quiz him, and correct him when he's wrong. Prediction-vs-measured *misses* are first-class
deliverables — record them honestly (see Step 2 in `docs/results.md`), never paper over them.

## Hardware & build constraints

- **Author machine is a Mac with no CUDA toolchain.** You cannot compile `.cu`, run kernels, run
  pytest (needs GPU), or run `nvidia-smi`/`ncu` here. IDE diagnostics on `.cu` files
  (`torch/extension.h not found`, `__global__ unknown`) are clangd noise — **not real errors.**
- **All compile/test/bench/profile happens on Colab T4 (Turing sm_75)** via `cpp_extension.load()`
  JIT, driven by `notebooks/colab_bootstrap.ipynb`. A100/H100 are rented per-step (Phase 2+).
- What you *can* validate locally: Python syntax, the `roofline/` tool (pure CPU), source wiring,
  and `bash -n` on scripts.

## Colab workflow (notebooks/colab_bootstrap.ipynb)

- Cell ~5 does `git clone` then `os.chdir('flashattention-cuda')`, so **all later cells run from
  the repo root** — shell commands use `!python -m ...` with **no `cd` prefix**. (`!cd x && ...`
  only lasts one cell; use `%cd` if you must persist a dir.)
- Cell 5 only clones when the folder is absent — it never pulls. Add `!git pull origin main` as a
  cell to pick up new commits.
- Repo is **public** (`github.com/gkienpham-cmd/flashattention-cuda`), so clone needs no token.
- Colab needs `pip install ninja` and a clean-stale-JIT-cache step (both already in the notebook).
- `ncu` *can* run on Colab but it's runtime-dependent/unreliable; a rented T4 (vast.ai ~$0.15/hr)
  is the safer bet for the profiling deliverable. Free Colab covers build/test/bench.
- The user runs the notebook on Colab, commits the outputs, and pushes. You `git pull` and read the
  outputs from the `.ipynb` JSON (parse `cells[].outputs`).

## Repo layout & conventions

- `fa_kernels/` — public API (`attention`), `AttnConfig` (the single shared config), `dispatch.py`
  (backend → compiled kernel, capability-gated), `reference.py` (SDPA baseline).
- `kernels/vN_name/` — versioned CUDA, **always a `{kernel}.cu` + `binding.cpp` pair**. Register a
  new version in `bindings/load.py` `_SOURCES` and `dispatch.py` `_MIN_CAPABILITY`.
- `roofline/` — limiter predictor (`model.py` = the math, `predict.py` = CLI, `archs.py` = arch
  constants). Runs on CPU. `model.py` models v1/v2 with `materialize_s=True` + `tile_m/tile_n`.
- `bench/harness.py` — bench vs SDPA; prints prediction-vs-reality per row; records `clock~CUR/MAX`.
- `tests/test_correctness.py` — vs SDPA, parametrized over backends; tol atol/rtol 1e-4 (FP32).
- `profiling/capture.sh` — Nsight capture; filters `--kernel-name 'regex:(qk|softmax|pv)'`.
- `docs/` — `decisions.md` (what we chose + why + measured), `results.md` (the curve),
  `interview-prep.md` (C-chains: reasoning to recite; add a C-entry per step), `writeup.md` (essay).
- **Conventions:** tensors are `[B, H, N, d]` row-major. FP16 phases use FP16-in/FP32-accum; v1 and
  v2 are FP32 throughout. All four kernels take an optional `causal` flag (lower-triangular mask —
  keys `j > i` excluded; tested both ways vs SDPA). Hardware leaks in only at the build `-gencode`
  and `roofline/archs.py`.

## Status (2026-06-27)

- **Step 1 (v1 naive)** — DONE: tests green, benched, quiz passed. Key finding: naive beats its
  cache-free roofline floor at mid-N (L2 absorbs redundant operand reads), converges to it at
  N=8192 (L2 overflow). AI ≈ 0.25, HBM-bound at all d.
- **Step 2 (v2 tiled, `kernels/v2_tiled/`)** — DONE (2026-06-19): both gates cleared (quiz passed +
  ncu read). FP32, three-pass, S still materialized; tiles 64×64 @ d=64 / 32×32 @ d=128. **26/26
  correctness** vs SDPA; benched **1.3–3.2× over v1**. **The ncu read flipped the story:** tiling
  cut ~0% DRAM traffic (1.08× @ N=512, 1.02× @ N=8192) because the T4's 4 MB L2 already owns the
  operands at every N (v1's qk reads 74 MB at N=8192, not the predicted ~hundreds of GB). **S is
  ~99% of DRAM** (softmax re-reads it ~12×), byte-identical v1↔v2, and **nothing saturates HBM**
  (≤35% DRAM throughput). So v2's speedup is a *compute/scheduling* win, not bandwidth. The roofline
  got magnitude, location, AND limiter wrong — all because it can't model L2. See `docs/results.md`
  Step 2 (measured DRAM table), `decisions.md` Step 2, `interview-prep.md` C5.
- **Step 3 (v3 online, `kernels/v3_online/`)** — DONE (2026-06-19): both gates cleared (quiz passed +
  counter-free measurement). FP32, **two-pass** online softmax (pass 1 → final `(m,l)` with the
  running-max rescale; pass 2 recomputes scores + forms O, no O-rescale); S never materialized.
  **15/15 correctness** vs SDPA (incl. N=16384 rescale-stability). **S-elimination proven without a
  profiler:** peak CUDA mem at 8192×64 is **+17 MB (v3) vs +2164 MB (v2)** = the 2147 MB S matrix
  gone (125× less). **But v3 is 3–7× SLOWER than v2** — deleting ~99% of DRAM traffic bought nothing
  because Step 2 proved nothing was bandwidth-bound. Real limiter = **occupancy/latency** (measured
  151× above the 17 ms MMA floor; torch CUPTI trace puts pass2_output at 88.6% — one-thread-per-row +
  unstaged per-row K/V reads). Roofline mispredicted a 3rd time (HBM→MMA→neither), same root cause:
  blind to the schedule. **ncu pipe-util read deferred to bare-metal** (`ERR_NVGPUCTRPERM` blocks
  counters on all containerized rentals — vast.ai/RunPod/Lambda; confirmed across hosts +
  `--cap-add` attempts). See `docs/results.md`/`decisions.md` Step 3, `interview-prep.md` C6.
- **Step 4 (v4 fused, `kernels/v4_fused/`)** — DONE (2026-06-20): both gates cleared (quiz passed +
  counter-free measurement). FP32, **single-pass** FA-1 — **one warp per query row**, staged K/V in
  smem, **register-resident O** with the **O-rescale** (`O = α·O + p·V`, α=exp(m_old−m_new) applied
  before the add — the piece v3 deferred). **17/17 correctness** vs SDPA (incl. N=16384 O-rescale
  stability at d=64 *and* d=128). **The thesis landed: v4 beats v2 1.7–2.6× and v3 7.5–15×** — the
  first version where S-off-HBM is *also* a wall-clock win. **S still gone:** +16.8 MB @ 8192×64 (<
  v3's +17.3, no HBM scratch). **Schedule fixed:** CUPTI shows a single fused kernel at 100% CUDA time
  (v3's 88.6%-pass2 wall gone); distance to floor **151×→~18×**. **But still ~6× slower than SDPA /
  18× off the FP32 MMA floor** — new limiter = **FMA under-utilization**: warp-per-row scoring is
  GEMV-shaped (per-key `__shfl` reduction ≫ FMAs), never near the 8.1 TFLOPS peak. Roofline missed
  *magnitude* a 4th time, same blind spot. ncu still deferred. See `docs/results.md`/`decisions.md`
  Step 4, `interview-prep.md` C7.
- **Step 5 (v5 WMMA, `kernels/v5_wmma/`)** — **PARTIAL** (kernel + wiring landed `ad021b7`; correctness
  gate green; prose close-out backfilled 2026-06-27). The **GEMV→GEMM fix** for v4's FMA-utilization
  wall: keep v4's fused single-pass schedule but move *both* matmuls onto Turing WMMA tensor cores
  (FP16-in/FP32-accum, 16×16×16). **The opaque-fragment tax** (WMMA accumulators are un-indexable)
  forces softmax/S through smem (store→row-softmax→reload P as half) and keeps FP32 `oRun` in smem so
  the O-rescale folds into the PV accumulator. Tiles `d=64→BM=64/4-warp`, `d=128→BM=32/2-warp`.
  **Correctness green** (in `BACKENDS`, tol **2e-2** FP16-in — the first loosened band; 17 cases incl.
  the N=16384 O-rescale/FP16-drift stability at d=64 *and* d=128; rebuilt + passed during the Step-6
  run). **Roofline predicts the floor drops 8×** (65/8.1 TFLOPS → 16.97→2.114 ms @ 8192×64; ridge
  25.3→203.1, still compute-bound). **⚠️ OUTSTANDING MEASUREMENT: the prefill bench (v5÷v4, v5÷SDPA,
  distance-to-floor) was NEVER captured** — `notebooks/step5_run_of_record.ipynb` is unexecuted (all
  cells `execution_count=None`, no saved outputs); the only measured v5 numbers are v5@N_q=1 as the
  decode "naive" baseline in Steps 6/7. So Step 5's headline is a *prediction, not a result*; run the
  run-of-record to fill the speedup table in `results.md` + `decisions.md` + this line. See
  `docs/results.md`/`decisions.md` Step 5, `interview-prep.md` C7.5.
- **Step 6 (v6 split-KV decode, `kernels/v6_splitkv/`)** — DONE (2026-06-27): both gates cleared (quiz
  passed + counter-free decode bench, vast.ai T4). FP16-in/FP32-accum, **two kernels** behind one
  `forward` — a split-KV partial (each block does v4-style online softmax over one KV chunk → writes
  the *unnormalized* `(O,m,l)`) and an LSE merge across splits; `choose_splits` fills ~2× the SMs, and
  **prefill `N_q=N_k` → 1 split → reduces to plain attention** (why the square-shape tests pass). No
  tensor cores (decode matmuls are M=1 GEMV). **25/25 correctness** vs SDPA (square SHAPES + decode
  `N_q=1`, causal both ways, non-multiple `N_k`). **The split-KV thesis lands: v6 beats the naive
  `N_q=1` loop (v5@N_q=1) 5.7–8.2× and torch SDPA 1.5–3.3× (non-causal), the win growing with `N_k`.**
  **But only ~9–15% of HBM BW (≈7–9× above the decode floor):** roofline got the *location* right
  (HBM, `AI=2/b=1.0`, N-independent) but **magnitude wrong a 5th time** — real limiter is still
  **occupancy/launch** (BH=8 → ~2 blocks/SM + a 2-kernel launch + a tiny under-occupied merge), not
  bandwidth. So the measurement **reorders the roadmap: next lever is occupancy (GQA M-packing → v8,
  `AI=2G/b`), not bytes** — FP8/NVFP4 KV (now v9/v10 on B300) only pays once actually bandwidth-bound.
  Causal-decode rows are a degenerate-shape artifact (`q` at row 0 → 1 key; SDPA short-circuits, v6
  doesn't) — a `--decode` query-offset is a v7 harness fix. Opens the decode arc v6→v11.
- **Step 6 deep-research close-out (2026-06-27)** — DONE: a 13-agent verify-and-research pass →
  `docs/decode-replan.md` (paper-grade synthesis + 5 new diagrams in `docs/diagrams/`). Headline:
  **focus + REORDER, don't pivot.** The decode/B300/FP4 thesis is sound and the external research
  strengthens it (B300 HBM BW is **flat 8 TB/s** vs B200 — verified, only capacity grew 192→288 GB; FA4
  is BF16 prefill/training; cuDNN 9.19 already matches it on BF16 prefill). What v6's 12%-HBM result
  changes is the build **order**: the occupancy lever **GQA M-packing (now v8: `AI=2/b→2G/b`,
  GEMV→GEMM, KV read once, tensor cores re-engage)** moves *ahead of* the byte levers **FP8 (v9) /
  NVFP4 (v10)**. Two corrections: (a) "occupancy before bytes" is **batch-conditional** — the 12% is a
  `BH=8` micro-bench artifact; split-KV self-disables past `BH≈2·SM` (80 on T4, 320 on B300) so batch
  alone fills the SMs; that large-batch end-state is *predicted, not measured* → **v7 adds a `--batch`
  sweep** to pin the crossover. (b) Claim discipline: FA4 is *acquiring* a decode path (Modal upstreamed
  split-KV+GQA-packing) and FlashInfer/FlashMLA **are** B300-proven, so frame the contribution as "an
  **open, roofline-documented asymmetric-precision FP4 decode kernel** measured vs FlashInfer/FlashMLA,"
  **not** "we beat FA4." See `docs/decode-replan.md`, `results.md`/`decisions.md` Step 6,
  `interview-prep.md` C8–C9.
- **Step 7 (v7 split-KV paged, `kernels/v7_paged/`)** — DONE 2026-06-27 (vast.ai T4, **both gates:
  51/51 correctness + quiz passed**). v6's two kernels carried unchanged + ONE new
  variable: KV reads **gather through a per-sequence block table** (paged pool `[num_blocks,page_size,
  H,d]`, the vLLM layout a mini-vLLM consumes) via a new `paged_attention()` API; plus two harness
  fixes (`--batch` sweep, causal query-offset). **The `--batch` sweep REFUTED the predicted
  occupancy→bandwidth crossover:** at N_k=8192, `%HBM` is **FLAT at 9.4–12.4% from BH=8 to BH=512** (no
  climb, even at 12.8 blocks/SM) — decode here is **per-CTA-bound, not grid-occupancy-bound, at EVERY
  batch size.** Code-verified cause: `sK+sV=32 KB`/block caps residency at **2 blocks/SM** (T4 64 KB/SM)
  and at `N_q=1` only **1 of 8 warps computes** — batch adds waves, not per-SM parallelism. This
  **corrects the "occupancy before bytes is batch-conditional" claim** (decode-replan §2.1): there is no
  large-batch bytes-first regime for this kernel; the accurate framing is **GEMV→GEMM (per-CTA
  efficiency) before bytes, at all batch sizes.** The reorder (GQA M-packing → v8) survives and is
  *strengthened* — GQA leads because `M=G` lights up G warps + GEMV→GEMM, not because it "fills the SMs."
  Causal query-offset fix works (causal µs/tok ≈ non-causal). Apparent ~15–25% paging overhead vs v6 at
  B=1 (dependent block-table load), cross-session clocks unverified. See `results.md`/`decisions.md`
  Step 7, `interview-prep.md` C10, `decode-replan.md` §2.1/§7 (corrected).
- **Step 7 deep-research close-out (2026-06-27)** — DONE: 35-agent pass (6 data-forensics + 7 web-research
  + 7 adversarial claims, 2-of-3 gate) → [`docs/v7-deep-research.md`](docs/v7-deep-research.md) +
  [`docs/v8-kickoff.md`](docs/v8-kickoff.md) + 2 new diagrams. **Per-CTA-bound + GQA-before-bytes both
  survive 0/3 refute.** Three settled: (1) **%HBM is fp16-correct, NOT 2× understated** — the kernel casts
  K/V to half (`paged_attention.cu:274-276`); `precision=fp32` is a cosmetic header label (the "2×
  undercount" claim died 3/3); (2) **SDPA overtakes v7 by B=8** (v7 SM-saturated/flat, SDPA amortizes launch
  ~4.5×) → v8 gains a **"reclaim-batch"** deliverable; (3) causal `vs_sdpa` is a top-left-mask artifact (v7
  scans all N_k, ref ~1 key), correctness test already honest. v8 **tensor-core gate: M≥16** → G=8 needs
  pad-to-16 / multi-group / CUDA-core-QK (**the ablation**); target **sm_80 (A100)**. Frame vs
  **FlashInfer/FlashMLA, NOT "beat FA4"** — FA4's decode path is now *upstreamed* (Modal PRs, `pack_GQA`
  2.92×). B300 confirmed: HBM flat **8 TB/s**, 288 GB, NVFP4 15 PF dense, exp 2× (10.7 TeraExp/s), `sm_103`.
  Cleanup TODOs: bottom-right causal-mask ref; stale "split-KV fills SMs" comments in
  `paged_attention.cu:221-226` + `roofline/model.py:96-99` (both now **corrected**).
- **Step 8 (v8 GQA M-packing, `kernels/v8_gqa/`)** — **Cut 1 MEASURED (Colab T4, 2026-06-28): Gate 1 ✅
  64/64; G-sweep + reclaim-at-batch captured. Cut 2a (Turing WMMA tensor cores, `kernels/v8_gqa_tc/`)
  MEASURED (Colab T4, 2026-06-28): correctness ✅ 38/38, but the perf prediction is REFUTED — WMMA is
  1.4–1.6× SLOWER than Cut 1's CUDA-core GEMV (even at full G=16/32 tiles), because decode `M=G≤16` is too
  small to amortize the opaque-fragment tax. So Cut 1's win was G-warps + KV-read-once, NOT tensor cores
  (a prefill tool). Arms 2/3 likely DOA; Cut 2b (A100 mma+cp.async) is now an open question, not a
  foregone port. Quiz (Gate 2) deferred until Cut 2 closed (Kien's call).** Cut 2 was re-staged: **2a =
  Turing WMMA on T4** (answered the GEMV→GEMM question cheaply — it loses) before **2b = A100
  `mma.m16n8k16`+cp.async `[RENT]`** (only if rescuing tensor cores is worth it). Cut 1 headline: G-packing
  buys **~`G×` wall-clock over no-packing (8.6× at G=8)** and **beats
  SDPA 6–10× at every batch B=1→64** (v7 lost at B≥8) — on CUDA cores, no tensor cores. `%HBM` stays ≤11%
  (still per-CTA-bound, ~9× headroom for Cut 2). The `AI=2G/b` model got the *speedup magnitude* right (a
  partial roofline win, first in 6 steps) while its absolute floor stays unreached. Built STAGED (Kien's
  call): **Cut 1 = CUDA-core M-pack on sm_75/T4**
  (`_MIN_CAPABILITY=(7,0)`; isolates M-packing cheaply, no tensor cores) → **Cut 2 = sm_80 tensor-core +
  the full 3-way M≥16 ablation** (pad-16 / multi-group-pack / CUDA-core-QK) `[RENT A100]`. Cut 1 **forks
  v7 verbatim** → `gqa_attention.cu`, changing ONLY the index math: grid z = `B·H_kv` (KV heads), packed
  row `m_row=blockIdx.x·8+warp` → `(g_local=m_row/N_q, i_q=m_row%N_q, h_q=h_kv·G+g_local)`; gather uses
  `H_kv,h_kv`; **causal mask uses `i_q` NOT `m_row`** (trap); workspace/merge stay query-head-shaped
  `[B,H_q,N_q,S,*]`. So the G query heads of a group share one staged KV tile → **G warps active (not 1),
  KV read once, `AI = 2/b → 2G/b`** — attacks v7's MEASURED per-CTA wall. New `gqa_attention()` API +
  `sdpa_reference_gqa` oracle (KV via **`repeat_interleave(G)`**, not `repeat` — trap). **Roofline (Task 1)
  extended** (`estimate(...,G)`; A100/B300 archs added): **prediction recorded BEFORE coding** — at G=8 the
  AI rises 8× (1.0→8.0) and the HBM floor drops 8× (0.132→0.016 ms), but the **limiter STAYS HBM** (A100
  fp16 ridge=153, so even G=32 is far below — NO compute-flip in the realistic GQA range; the win is
  per-CTA efficiency the model can't see → deliverable is the µs/tok drop + **reclaim-SDPA-at-B≥8**).
  Tests (3 GQA fns: decode G∈{1,2,4,8}×non-multiple `N_k`×causal, idle-warp G=3 + multi-tile G=16, square
  reduction), harness `--gqa-group` sweep + same-session vs-v7-no-packing A/B, gate notebook
  `notebooks/v8_gqa_gate.ipynb` all landed. **Author machine can't compile `.cu`** → build + correctness
  (Gate 1) + bench + quiz (Gate 2) are the outstanding GPU work. See `results.md`/`decisions.md` Step 8,
  `interview-prep.md` C11, `docs/v8-kickoff.md`.

## Next steps

**The reordered decode arc is `docs/decode-replan.md` §5 (math + per-step deliverable); summary:**
v7 paged KV → **v8 GQA M-packing (the reorder — occupancy)** → v9 FP8 KV → v10 NVFP4 + asymmetric
precision (headline) → v11 MLA/speculative.

1. **Step 7 — paged KV gather (`kernels/v7_paged/`) — DONE (both gates, 2026-06-27).** Deep-research
   close-out applied (`docs/v7-deep-research.md` + `docs/v8-kickoff.md`); `diagrams/decode-roofline-crossover.svg`
   **refreshed** to the measured-flat line (+ new `v7-vs-sdpa-batch-crossover.svg`, `per-cta-limiter-anatomy.svg`),
   and the stale "split-KV fills the SMs" comments were corrected in `paged_attention.cu` + `roofline/model.py`.
   Carry-forward cleanups (not gating, → v8's harness): a fully-honest causal `vs sdpa` needs the reference
   built with a bottom-right mask; bare-metal pipe-util to confirm the smem-residency (2 blocks/SM) story directly.
2. **Step 8 — GQA M-packing (the reorder, NOW the per-CTA-efficiency lever) — Cut 1 code complete; GPU
   gate (build + correctness + bench + quiz) pending, then Cut 2 `[RENT A100]` tensor cores + ablation.**
   Pack the `G` query heads
   of a GQA group into the CTA's `M` dim → GEMV becomes an `M=G` GEMM (tensor cores re-engage), KV read
   once, `AI = 2/b → 2G/b`, **and G compute-warps/block** (the fix v7 proved is needed). Promoted ahead
   of low-precision because v7 measured the kernel is per-CTA-bound at **all** batch sizes (flat ~10–12%
   HBM, BH=8→512), so cutting bytes is premature regardless of batch. `[RENT]` A100/H100. Then bytes: v9
   FP8 KV, v10 NVFP4 + asymmetric precision `[B300]`. (Diagrams: `gqa-mpacking.svg`,
   `decode-roofline-crossover.svg`, `per-cta-limiter-anatomy.svg`.) **Refinements (deep-research, 2026-06-27;
   full starter in [`docs/v8-kickoff.md`](docs/v8-kickoff.md)):** tensor cores engage only at **M≥16** → G=8
   needs pad-to-16 / multi-group-pack / CUDA-core-QK (**the v8 ablation**); sweep `G∈{1,2,4,8,16,32}`; target
   **sm_80 (A100)** (m16n8k16 + cp.async, not Turing WMMA); **task 1 = the `AI=2G/b` roofline extension**;
   comparators FlashInfer + XQA (M-packing SOTA) vs vLLM PagedAttention v2 (CUDA-core floor); new deliverable
   **reclaim SDPA at batch B≥8**.
3. **Finish Step 5** — prose close-out is backfilled (results/decisions/status/C7.5, 2026-06-27); the
   one remaining item is the **measurement**: execute `notebooks/step5_run_of_record.ipynb` (it's
   unexecuted — no saved outputs) and fill the prefill speedup table (v5÷v4, v5÷SDPA, distance-to-floor)
   in `results.md` Step 5 + `decisions.md` + the status line. Until then Step 5 is PARTIAL (correct +
   predicted, not measured).
4. **GPU host:** **vast.ai** T4 (~$0.10–0.20/hr). Image gotchas confirmed this run: torch installs into
   a **venv** (`/venv/main`) — install with `%pip`/`sys.executable -m pip` and prepend the venv `bin/`
   to `PATH` so `!python -m …` cells resolve (do NOT symlink system python3 over the venv); add
   `numpy`. `notebooks/v6_decode_gate.ipynb` is the standalone Run-All gate. **Counter-free profiling
   is the norm:** `torch.cuda.max_memory_allocated()` + CUDA-event/CUPTI timing; ncu counters blocked
   (`ERR_NVGPUCTRPERM`) on containerized rentals — pipe-util only on a bare-metal box.

## Git

- Solo learning repo; `main` is the working branch and the Colab pull target — commit directly to
  `main` (do not branch; the notebook pulls `main`). Commit/push only when the user asks.
- Do **not** commit `.vscode/` or other IDE cruft.
- **ALWAYS sync Kien's local `main` after pushing to `main` (learned 2026-06-27).** Work usually
  happens in a harness-created git **worktree** (`.claude/worktrees/<name>/`) — a *separate checkout*
  from Kien's main project folder (`/Users/kienpham/Documents/flashattention-cuda`, branch `main`). A
  commit + `git push origin HEAD:main` from the worktree updates `origin/main` but leaves Kien's main
  checkout BEHIND, so he doesn't see the new files locally and has to ask "where did they go?". So
  **every time you commit + push to `main`, immediately fast-forward his main checkout in the same
  turn:**
  `git -C /Users/kienpham/Documents/flashattention-cuda pull --ff-only origin main`
  (it's a clean fast-forward since he never commits on local `main` himself; the untracked
  `build-roadmap-*.svg` won't conflict). Then confirm the new files are present in the main folder.

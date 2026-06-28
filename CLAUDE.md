# CLAUDE.md — FlashAttention from scratch

A roofline-driven journey rebuilding FlashAttention one measured speedup at a time. The deliverable
is the **measured speedup + roofline curve** (prediction vs reality), not just fast code. Long-term
goal: a stable `from fa_kernels import attention` API that a future from-scratch mini-vLLM consumes.

**Research north star (the paper):** the decode arc (v6→v11) targets **B300 / GB300 (sm_103)** as its
final goal — *the first open, roofline-documented, prediction-vs-measured FlashAttention decode study on
sm_103, with an asymmetric-precision FP4 KV recipe, vs FlashInfer/FlashMLA.* FA4 stops at B200/sm_100, so
sm_103 is the publishable novelty. Honest scope: production libs already run GB300 decode — the
contribution is the *open paper-grade roofline + FP4 recipe + honest methodology*, not "first to run."

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

## Status (2026-06-28)

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
  `AI=2G/b`), not bytes** — FP8/NVFP4 KV (now v9 on T4 / v10 on B300) only pays once actually bandwidth-bound.
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
  MEASURED (T4 + A100, 2026-06-28): correctness ✅ (38/38 both archs), but the perf prediction is REFUTED
  on BOTH — WMMA is 1.8–4.6× SLOWER than Cut 1's CUDA-core GEMV (worse on the faster A100). Smoking gun:
  WMMA barely moved T4→A100 (42→39 µs/tok) despite ~5× TC throughput + ~6× BW, while CUDA-core nearly
  halved (16.9→9.7) → WMMA is pinned by per-CTA overhead (opaque-fragment smem-softmax + 1-warp load), not
  compute/BW. So decode's GEMV→GEMM is the WRONG tool (v5's prefill win didn't transfer). **Cut 2 CLOSED
  via a cheap A100 PROBE (ran the existing kernels on Ampere, build now sm_75+sm_80) — the hard
  cp.async/`mma` kernel + arms 2/3 are NOT pursued.** v8's deliverable is **Cut 1 (CUDA-core M-packing)**.
  **Both gates cleared (quiz passed 2026-06-28) → Step 8 DONE.** **v8.5 (double-buffered KV pipeline,
  `kernels/v8_gqa_db/`) MEASURED 2026-06-28 (T4): 38/38 correct but double-buffering did NOTHING — %HBM
  flat ~10%, µs/tok within noise vs Cut 1.** So the decode stall is the **per-key warp-shuffle reduction
  (v4 GEMV wall), NOT the load latency** — the kernel is compute-latency-bound at ~10% HBM, memory was
  never the wall. Neither tensor cores (Cut 2) nor load-overlap (v8.5) move it. **So v9 FP8 won't show a
  latency win on this micro-bench either** (bytes aren't the bottleneck); needs a faster reduction
  (vectorize/ILP/occupancy = "v8.6") or a memory-bound regime (N_k past L2) first — FP8/NVFP4's real value
  is KV-cache capacity + accuracy. Cut 1 headline: G-packing buys **~`G×` wall-clock over no-packing
  (8.6× at G=8)** and **beats
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
- **Step 8.6 (v8.6 hide the reduction latency, `kernels/v8_gqa_occ/` + `kernels/v8_gqa_ilp/`)** — **MEASURED
  2026-06-28 (Colab T4): 190/190 correct, BOTH arms NULL on the clock-robust `%HBM` (flat ~8–10% at every
  G/batch; clocks swung 360–1590 MHz so µs/tok is unusable cross-backend). Occupancy never engaged at the
  micro-bench batch (split-KV already emits ~80 blocks = 2/SM → no spare block for the 4-block ceiling; only a
  ~1% nudge at large batch); ILP tracked Cut 1 exactly. COUNTER-PREDICTION LANDED: the floor is the per-row
  serial online-softmax recurrence — unhideable by TLP or ILP. Fourth consecutive negative → mandate for
  v8.7.** v8.5's null pinned the decode floor to the **per-key warp-shuffle
  reduction + serial online-softmax recurrence** (compute-latency-bound at ~10% HBM, NOT load/bytes). v8.6 is
  a **2-arm single-variable ablation** to *hide* that latency, both CUDA-core/T4, both fork Cut 1 changing ONE
  thing: **Arm 1 `v8_gqa_occ`** = FP16 smem single-buffer (16 KB → **4 blocks/SM**, 2× resident warps; the
  distinction from v8.5 is it spends the freed smem on occupancy, not a 2nd buffer) and **Arm 2 `v8_gqa_ilp`**
  = **KU=4-unrolled key loop** (compute 4 independent partials → pipeline 4 shfl reductions → 4 serial softmax
  updates; FP32 smem kept → 2 blocks/SM, ILP the lone variable; a monotone per-tile `c_lim` replaces Cut 1's
  per-key break, keeping the all-masked edge correct). float2/half2 vectorization REJECTED (lane-strided layout
  non-contiguous; loads already shown irrelevant by v8.5). **Roofline is BLIND** (AI=2G/b, floor, limiter
  identical to Cut 1 — confirmed on T4 arch G=8: AI=8.0, HBM-bound, 0.105 ms; a *schedule* claim the model
  can't express). **Prediction recorded BEFORE the run: occupancy (Arm 1) > ILP (Arm 2)** — TLP hides the whole
  serial chain, ILP only the reduction sub-part; **counter-prediction (the prize): if BOTH null → the floor is
  the serial recurrence itself → score-stationary redesign (a future v8.7) is the real fix, and v9 FP8 stays
  premature.** Wired (load/dispatch/harness/tests, both `(7,0)`, tol 2e-2, added to `GQA_BACKENDS`); gate
  notebook `notebooks/v8_6_reduction_gate.ipynb` (fork of v8.5's, builds both arms, `-k "v8_gqa_occ or
  v8_gqa_ilp or v8_gqa"`, 3-way A/B G-sweep + reclaim-at-batch). See `results.md`/`decisions.md` Step 8.6,
  `interview-prep.md` C11.5. **Gate-2 quiz PASSED 2026-06-28 (combined with v8.7) → Step 8.6 DONE.**
- **Step 8.7 (v8.7 score-stationary inner loop, `kernels/v8_gqa_ss/`)** — **MEASURED 2026-06-28 (Colab T4):
  WIN, ✅ 228 passed. ss beats Cut 1 1.1–1.6× at MATCHED CLOCK (ss ran 375/465 MHz vs Cut 1 360/480 — clock-
  fair; occ ran hot 585/1590) across G and batch (best at d=64; d=128 wins less 1.16–1.18× = the R2
  smem-read-BW drag, NOT a null), beats SDPA 8–16×, and is the FIRST lever since Cut 1 to move the
  clock-robust %HBM (up ~2–3 pts → ~10–12%). ss-vs-occ (both FP16 smem) confirms the win is the LAYOUT. TWO
  TRUTHS: (1) removing the per-key reduction + 32×-shortening the recurrence was a REAL win → the
  reduction/recurrence WAS a genuine component of the floor (remove-not-hide vindicated; v8.6 correctly said
  hiding wouldn't work); (2) %HBM plateaued ~10–12%, NOT the floor → a residual per-CTA ceiling remains (load
  latency / small-CTA launch), the kernel looks NOT bandwidth-bound → v9 FP8's value stays capacity+accuracy,
  not micro-bench latency. CLOSES the decode-schedule arc: M-packing (Cut 1) + score-stationary (v8.7) are the
  two real decode levers; tensor cores / double-buffer / occupancy / ILP were dead ends. **⚠️ CAVEAT (deep-
  research 2026-06-28): that "~10% HBM / not bandwidth-bound" read is CONFOUNDED — the bench KV fits in the T4's
  4 MB L2 + clocks were never locked, so bandwidth couldn't appear. "per-CTA-bound OR L2-resident, unproven";
  v9 Task 1 earns it. See "Next steps" + `results.md` threats-section + C12.** **Combined v8.6+v8.7
  Gate-2 quiz PASSED 2026-06-28 → Step 8.7 DONE.** v8.6's fourth negative mandated REMOVING (not hiding) the wall. Single variable = the
  inner-loop LAYOUT: flip from Cut 1's output-stationary GEMV (lane splits head-dim → per-key `__shfl_xor`
  butterfly + serial recurrence) to **score-stationary: lane = key** — lane `l` computes the FULL dot product
  q·k_c in its own registers (**no per-key cross-lane reduction**); softmax runs **once per 32-key group** (one
  `warp_reduce_max` + one `warp_reduce_sum` → recurrence **32× shorter**); PV is a transpose `O[d]=Σ_c p_c·V[c][d]`
  via single-hop `__shfl` broadcasts of `p_c` that **pipeline** (vs Cut 1's serial chain). The layout INVERTS
  Cut 1 (QK reduction-free, cross-lane traffic moves to the 2-per-group softmax + PV fan). **smem staged FP16**
  (sQ per-warp + K **transposed `[d][key]`+1-pad** for bank-conflict-free lane=key reads + V natural; ~18 KB →
  ~3 blocks/SM) to HOLD occupancy — FP32 + the new sQ/pad would drop T4 to 1 block/SM and confound the layout
  variable (v8.6 measured FP16 smem is perf-neutral); bonus: `v8_gqa_ss` vs `v8_gqa_occ` differ in ONLY the
  layout → clean isolated A/B. Prologue/epilogue/host/merge/choose_splits **byte-identical** to Cut 1; added
  `warp_reduce_max`; reused the ilp fork's monotone `c_lim`; traps handled (warp-uniform reductions on masked
  lanes via s=-inf/p=0, idle-warp barrier participation, fully-future split → (m=-inf,l=0,O=0), three distinct
  smem strides sQ/`(TN+1)`/D). **Roofline BLIND** (AI=2G/b unchanged). **Prediction recorded BEFORE the run:
  µs/tok drops, best at d=64; d=128 at RISK of flipping to smem-read-BW-bound (full-D sK reads/key);
  counter-prediction: if µs/tok drops but %HBM stays ~10% → floor is per-CTA LOAD latency → v9 FP8 (capacity)
  is the right next lever.** Wired (load/dispatch/harness 3 tuples/tests, `(7,0)`, tol 2e-2, `GQA_BACKENDS`);
  gate notebook `notebooks/v8_7_score_stationary_gate_output.ipynb` (run-of-record; build ss, 3-way A/B
  Cut1/occ/ss G-sweep + reclaim). See `results.md`/`decisions.md` Step 8.7, `interview-prep.md` C11.6.
- **Step 9 (v9 FP8 E4M3 KV, `kernels/v9_fp8/`)** — **Task 2 DONE (Colab T4, 2026-06-28): both gates
  cleared (Gate 1 ✅ 76 passed + Gate-2 quiz PASSED 2026-06-28). E4M3 BUILT on sm_75 with NO int8 fallback. The "capacity-only" prediction is REFUTED — FP8
  buys a real same-session clock-matched ~1.3× decode-latency win (`vs naive` FP8÷FP16 v8.7 = 0.96–1.52×,
  median ~1.3×, FLAT across batch B=1→64), and it's a LOAD-bandwidth (L2→SM) win: smoking gun = the win
  SHRINKS as G grows (d=64 1.37 G2→0.96 G32) because M-packing amortizes the KV load (load→compute shift);
  d=128 wins more (2× bytes/key). `%HBM` DROPPED vs FP16, no `L2!` flag → FP8 did NOT flip the limiter
  (still per-CTA/L2-load-bound ~10% HBM, ~14× above the halved floor). Per-tensor E4M3 quant RMSE ~6–7e-4
  (oracle ~6e-6 = kernel math exact). So "bytes won't help latency until past-L2" (recurring since v6) was
  too strong: the residual post-v8.7 ceiling is PARTLY L2→SM load (bytes relieve it) + partly per-CTA
  compute (they don't). "L2-resident ≠ memory free." Harness fix post-run: v9 `vs sdpa` was inflated (FP8
  oracle re-quantized inside the timed baseline) → now dequant once outside; trust `vs naive`. Task 1
  (locked-clock past-L2 sweep) + v10 NVFP4 (a compute lever) remain the bandwidth story.** Forks
  `v8_gqa_ss` changing the SINGLE variable **KV storage precision**: the paged K/V pool holds **FP8 E4M3
  (1 byte, uint8)** instead of FP16, dequantized **fused per-tile** at the smem gather (`__nv_cvt_fp8_to_halfraw`,
  software-emulated on sm_75; int8-symmetric is a 1-line fallback if ptxas rejects E4M3) with **per-tensor
  FP32 scales** (`scale_k/scale_v`, = amax/448), FP32 accum. Score-stationary inner loop / M-packing grid /
  split-KV / LSE merge **byte-identical** → clean byte-only A/B vs v8.7. New `fp8_attention()` API +
  `build_paged_kv_fp8` (returns uint8 pools + scales) + `sdpa_reference_gqa_fp8` apples-to-apples oracle
  (dequant the SAME bytes; tol **5e-2**); 3 dedicated v9 tests (decode G∈{1,2,4,8}, idle-warp G=3/multi-tile
  G=16, square) + an RMSE-vs-fp16 accuracy assertion. Harness `v9_fp8` branch (FP8 pool, `precision=fp8`
  roofline, baseline = v8.7 on the fp16 pool to isolate bytes, **counter-free "L2!" flag** when
  effective_bw > HBM peak). Wired (`load.py`, `dispatch.py` `(7,0)`, `__all__`). **Roofline recorded BEFORE
  coding (confirmed via `roofline.model.estimate`):** FP8 DOUBLES AI (G8: 8.0→16.0) and HALVES the HBM floor
  (13.12→6.56 µs), limiter STAYS HBM (16 ≪ T4 fp16 ridge 203) — **blind to dequant latency + the L2
  confound. Prediction: capacity-only (no µs/tok win) on the L2-resident micro-bench; latency win only
  past-L2/under-load (v9 Task 1 territory).** See `results.md`/`decisions.md` Step 9, `interview-prep.md`
  C13, `docs/v9-kickoff.md`.
- **Step 9 Task 1 (regime characterization) — MEASURED 2026-06-28 (ROOT T4, clocks LOCKED 1590 MHz
  no-throttle, L2 flushed, ncu LIVE): VERDICT = PER-CTA-BOUND, CONFOUND-FREE. The six-step "per-CTA, not
  bandwidth-bound" read is now EARNED, and this is the FIRST ncu-validated measurement in the project (all
  prior steps had ncu blocked by `ERR_NVGPUCTRPERM`).** Removed all three confounds at once (locked clocks;
  L2 flushed + N_k→128K so WS hit 537 MB ≫ 4 MB L2; ncu counters worked). `v8_gqa_ss` achieved-%HBM
  plateaus ~11–14% (low occupancy H_kv=1) → hard **~28–29% ceiling** (H_kv=8 or batch≥8), never near the
  ~70% achievable ceiling, to a 1 GB working set. ncu past L2: **L2 hit-rate 1.1%** (data genuinely from
  HBM) yet **DRAM 12.85%** → HBM-served + ~13% busy = per-CTA-bound. `%HBM`-vs-N_k RISES (launch-overhead
  amortization) then PLATEAUS with NO bandwidth knee at the L2 crossing. **Counter-free method VALIDATED:**
  sweep `eff_bw=KV_bytes/time` = 13.8% vs ncu DRAM 12.85% (same shape, within ~1 pt). So **"~10% HBM" was
  an OCCUPANCY artifact, not the floor (true cap ~28%)**, the limiter is per-CTA (1 active warp at N_q=1 →
  low MLP), and **FP8/NVFP4 are confirmed a capacity+accuracy play, NOT a decode-latency/bandwidth play**
  (corroborates Task 2). Caveat: `v9_fp8` clock sagged to 1350 MHz despite the lock (FP8 dequant ALU
  power-caps sm_75) → cross-kernel `us/tok` clock-confounded this run; within-kernel %HBM trends robust.
  Reopener (logged): v8.5/v8.6 nulls were at L2-resident sizes; past-L2 headroom (29%→70%) means
  latency-hiding might bite there. Verdict + ncu-validation story in `results.md`/`decisions.md` Step 9
  Task 1 + `interview-prep.md` C12; data of record `notebooks/v9_task1_regime_output.ipynb`. Tooling:
  `bench/regime.py` (`python -m bench.regime`):
  `lock_clocks()`/`reset_clocks()` (root; loud warn + continue if not), an **L2-flushing CUDA-event timer**
  (zero a ≥2×L2 buffer outside the timed window — the jan.ai technique), a `sweep()` returning structured
  rows with the **counter-free L2 test** (`eff_bw = kv_bytes/time > 320 GB/s ⇒ L2-served`, flagged `L2!`)
  + OOM guard, and a `--profile B,H_kv,N_k,d` mode so `ncu` can attach to one shape. Gate notebook
  `notebooks/v9_task1_regime.ipynb` (clock-lock → roofline recap → build → **isolation sweep** N_k 1K→128K
  L2-flushed, both kernels → **the decisive matplotlib plot** %HBM-vs-N_k → `diagrams/v9-task1-regime.svg`
  → large-batch confirmation → optional ncu → reset → verdict). **Roofline prediction recorded:** decode
  AI=2/b HBM-bound but BLIND to L2 → %HBM should climb past the 4 MB L2 crossing (N_k≈8192 d128 fp16; ~2×
  for fp8) IF memory-bound; counter (the C12 survivor): flat ~10% past L2 with `L2served=False` → per-CTA-
  bound confound-free. Degrades gracefully (counter-free sweep runs on Colab; clock-lock+ncu need root T4 —
  vast.ai ~$0.10–0.20/hr). Decisive read + criteria in `results.md` Step 9 Task 1. See `docs/v9-kickoff.md`
  Task 1, `interview-prep.md` C12.
- **Step 9 deep-research close-out (2026-06-28)** — DONE: a 7-agent verify+research pass (2 notebook
  forensics + code audit + adversarial red-team + 2 web-research B300/NVFP4 + planning). **v9 VERIFIED**
  (code audit: fused dequant, byte-identical A/B, in-session `vs naive`, `vs sdpa` fix all confirmed; both
  notebooks clean). Headlines survive; wording sharpened + three v10 assumptions corrected. **Refinements
  (in `results.md`/`decisions.md` Step 9 close-out):** (1) "per-CTA-bound" → **"per-CTA / low-MLP
  latency-bound, ~28% cap"** (the batch-sweep *decline* at B≥64, 29.3→25.4%, is a latency/MLP fingerprint,
  not occupancy); (2) the FP8 win is a **bytes-sensitive load-LATENCY** effect (ncu L2 thrpt <3%, NOT
  L2-bandwidth) that is **regime-specific — flips NEGATIVE under L2-flush** (sm_75 dequant ALU tax), so
  byte-cuts are NOT a decode-latency lever; (3) the G-sweep **confounds G with H_kv/occupancy** (lead with
  d=128, the clean monotone case); (4) "decode-schedule CLOSED" → **"CLOSED for the L2-resident regime"**;
  (5) median is **1.205** (G-sweep) not 1.3, capacity 2× is **by construction** (unmeasured), accuracy is
  single-seed, and there is **no trustworthy v9-vs-SDPA number** (the gate's `vs sdpa` is the pre-fix
  inflated oracle). **Highest-value cheap follow-up (one notebook, settles 1+2+4):** re-run **v8.5/v8.6
  past L2** (N_k≥32K, locked) — Task 1 shows 29%→70% headroom where latency-hiding could finally bite.
  **v10 course-corrections (data/research-driven):** NVFP4 reframed to **capacity + accuracy + the sm_103
  2×-exp softmax delta**, bandwidth-latency **conditional** on a B300 long-context regime to be *measured*;
  ⚠️ Blackwell `tcgen05` gate is **M≥64** (M=128=100%) **not M≥16** → native FP4 *compute* slips to **v11**
  (multi-token); asymmetric recipe is **storage** (V→FP4 per-token, K→FP4 per-channel + score≥FP16), the
  "P·V-cheap-to-FP4" intuition **refuted for compute** (v11 concern); novelty narrows (FlashInfer ships
  NVFP4 KV decode now) → **"open roofline-documented prediction-vs-measured sm_103 decode + asymmetric FP4
  recipe, complementing not beating FlashInfer/FlashMLA."** New v9 figures `diagrams/v9-task1-regime.svg`
  (now in-repo, hand-authored — the notebook's matplotlib version was Colab-host-only) +
  `diagrams/v9-fp8-win-anatomy.svg`. Plan to paste: [`docs/v10-kickoff.md`](docs/v10-kickoff.md).
  See `results.md`/`decisions.md` Step 9 close-out, `interview-prep.md` C14.

## Next steps

**The reordered decode arc is `docs/decode-replan.md` §5 (math + per-step deliverable); summary:**
v7 paged KV → **v8 GQA M-packing (the reorder — occupancy)** → **v9 FP8 KV + regime-fix (T4) — DONE** →
**v10 NVFP4 + asymmetric precision (headline, B300/sm_103 — the paper)** → **v11 MLA/speculative + native
FP4 compute (B300)**. **v10 plan to paste into a fresh session: [`docs/v10-kickoff.md`](docs/v10-kickoff.md)**
(supersedes the now-complete `v9-kickoff.md`).

**Immediate cheap experiment before v10 (from the v9 close-out — one T4 notebook, settles three open
threads):** re-run **v8.5 (double-buffer) + v8.6 (occupancy/ILP) through `bench/regime.py` PAST L2**
(N_k≥32K, clocks locked, L2-flushed). Their nulls were measured only at L2-resident sizes; Task 1 shows
real 29%→70% headroom past L2 where latency-hiding could finally bite. If double-buffer lifts %HBM there,
"decode-schedule CLOSED" reopens and the residual limiter is *latency* (not occupancy) — which retargets
the next schedule lever. Run it, record it, then start v10 (or fold it into v10's opening cell).

**Decode-SCHEDULE chapter CLOSED (2026-06-28, Steps 8/8.6/8.7 all DONE).** The two real decode levers are
**M-packing (Cut 1, per-CTA/occupancy)** + **score-stationary (v8.7, inner-loop)** — together they beat
SDPA 8–16× and v7-no-packing many×, on CUDA cores. Measured dead ends: tensor cores (Cut 2), double-buffer
(v8.5), occupancy & key-ILP (v8.6) — the decode floor was a serial dependency chain you relayout, not a knob
you tune.

**✅ The limiter diagnosis is now RESOLVED, confound-free (v9 Task 1, 2026-06-28).** The "per-CTA-bound,
NOT bandwidth-bound" verdict (recurring + hedged since **v6**) was confounded because the bench KV fit in
the T4's 4 MB L2, clocks were never locked, and N_k never passed ~16K. v9 Task 1 removed all three on a
**root T4** (clocks LOCKED 1590 MHz, L2 flushed, N_k→128K so WS=537 MB ≫ L2) **with ncu finally working**
(first time in the project): achieved %HBM caps at **~28–29%** (occupancy-lifted from ~11%), never near the
~70% ceiling, and **ncu confirms L2 hit-rate 1.1% / DRAM 12.85% past L2** = HBM-served yet per-CTA-bound.
**The counter-free %HBM proxy was validated against ncu (13.8% vs 12.85%)**, so every prior profiler-free
decode reading holds as a throughput proxy. "~10% HBM" was an *occupancy* artifact (true cap ~28%); the
limiter is per-CTA (1 active warp at N_q=1 → low MLP). **Net: bytes are NOT the decode wall → FP8/NVFP4 are
a capacity+accuracy play, not a decode-latency/bandwidth play.** The M-packing/SDPA *ratio* headlines were
always robust; now the limiter *name* is earned too. See `results.md`/`decisions.md` Step 9 Task 1, C12,
`notebooks/v9_task1_regime_output.ipynb`.

**Reframed v9 (decision recorded):** `v9 = FP8 KV + regime-characterization`. **Task 1 (gating)** earns the
verdict — on a *root T4* (clocks lock, ncu works), pin clocks, flush L2, sweep N_k {1K…128K} × batch × d ×
H_kv measuring HBM%/L2-hit-rate/L2-BW% + the counter-free L2 test (effective BW = KV_bytes/time > HBM peak ⇒
L2-served). **Task 2** = FP8 E4M3 KV (fork `v8_gqa_ss`) with *fused per-tile* dequant (a prepass eats the
savings), FP32 accum. **FP8 is capacity+accuracy on the L2-resident micro-bench, AND a latency lever in the
past-L2 / large-batch regime** (vLLM: per-token cost → 54% of BF16 past ~4–7k tokens) — Task 1 creates that
regime and Task 2 measures it. **Hardware: v9 on T4 (no rental — decode GEMV uses no tensor cores; FP8's win
is storage bytes); v10/v11 NVFP4 + MLA on B300 / GB300 (sm_103) — the FINAL goal and the paper's novelty**
(no published FA paper has characterized a B300; FA4 stops at B200). B200 (sm_100) is an optional cheaper dev
rung; the record runs on B300. B300-only levers the paper uses: 2× exp/SFU throughput (softmax MUFU term),
288 GB (long-context KV → the regime where bandwidth *could* matter — **to be measured; T4 stayed
per-CTA-bound even past L2**, so this is the v10 T3 question, not an assumption), NVFP4 15 PF.
**[v9 DONE — the close-out above CORRECTS this prediction's optimism: FP8's measured latency win is
L2-resident-only and flips negative under L2-flush; decode stayed per-CTA-bound past L2 on T4. v10's
bandwidth-latency claim is conditional on a B300 long-context regime that must be measured.]**

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
   FP8 KV `[T4]` (+ regime-fix), v10 NVFP4 + asymmetric precision `[B300]` (the paper). (Diagrams: `gqa-mpacking.svg`,
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

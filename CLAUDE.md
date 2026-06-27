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
- **Step 5 (v5 WMMA, `kernels/v5_wmma/`)** — kernel + wiring landed (commit `ad021b7`) and tests are
  green (in `BACKENDS`, tol 2e-2 FP16-in; rebuilt + passed during the Step-6 run); run-of-record in
  `notebooks/step5_run_of_record.ipynb`. **OUTSTANDING BACKFILL:** the prose close-out (`results.md`/
  `decisions.md` Step 5 + this status line + `interview-prep.md`) was deferred — pull the measured
  speedups from the run-of-record notebook before calling Step 5 fully documented.
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
  bandwidth. So the measurement **reorders the roadmap: next lever is occupancy (batch / GQA M-pack,
  `AI=2G/b`), not bytes** — FP8/NVFP4 KV (v8/v9 on B300) only pays once actually bandwidth-bound.
  Causal-decode rows are a degenerate-shape artifact (`q` at row 0 → 1 key; SDPA short-circuits, v6
  doesn't) — a `--decode` query-offset is a v7 harness fix. Opens the decode arc v6→v11
  (`docs/b300-decode-research.md`). See `docs/results.md`/`decisions.md` Step 6, `interview-prep.md` C8.

## Next steps

1. **Step 7 — paged KV-cache gather (`kernels/v7_*`):** block-table indirection so the KV cache need
   not be contiguous (the import surface a from-scratch mini-vLLM consumes). Carry v6's split-KV +
   LSE merge; add a non-contiguous-KV correctness case. Also fold in the **`--decode` causal
   query-offset** fix (place the query at `N_k−1` so causal decode attends the whole cache, not 1
   key). Per the Step-6 finding, the occupancy lever (bigger batch / GQA M-packing → `AI=2G/b`) comes
   before the low-precision-KV lever — v6 reaches only ~12% of HBM, so cutting bytes is premature.
2. **Backfill Step 5 docs** from `notebooks/step5_run_of_record.ipynb` (results/decisions/status/C-chain).
3. **GPU host:** **vast.ai** T4 (~$0.10–0.20/hr). Image gotchas confirmed this run: torch installs into
   a **venv** (`/venv/main`) — install with `%pip`/`sys.executable -m pip` and prepend the venv `bin/`
   to `PATH` so `!python -m …` cells resolve (do NOT symlink system python3 over the venv); add
   `numpy`. `notebooks/v6_decode_gate.ipynb` is the standalone Run-All gate. **Counter-free profiling
   is the norm:** `torch.cuda.max_memory_allocated()` + CUDA-event/CUPTI timing; ncu counters blocked
   (`ERR_NVGPUCTRPERM`) on containerized rentals — pipe-util only on a bare-metal box.

## Git

- Solo learning repo; `main` is the working branch and the Colab pull target — commit directly to
  `main` (do not branch; the notebook pulls `main`). Commit/push only when the user asks.
- Do **not** commit `.vscode/` or other IDE cruft.

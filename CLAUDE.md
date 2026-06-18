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
  v2 are FP32 throughout. Hardware leaks in only at the build `-gencode` and `roofline/archs.py`.

## Status (2026-06-19)

- **Step 1 (v1 naive)** — DONE: tests green, benched, quiz passed. Key finding: naive beats its
  cache-free roofline floor at mid-N (L2 absorbs redundant operand reads), converges to it at
  N=8192 (L2 overflow). AI ≈ 0.25, HBM-bound at all d.
- **Step 2 (v2 tiled, `kernels/v2_tiled/`)** — IN PROGRESS: FP32, three-pass, S still materialized;
  tiles 64×64 @ d=64 / 32×32 @ d=128. **26/26 correctness** vs SDPA incl. boundary shapes; benched
  at **1.3–3.2× over v1**, limiter still HBM (as predicted). Honest miss: predicted ~30× / biggest
  at N=8192; measured ~3× / peaks at mid-N — because realized speedup is the ratio of two
  *off-roofline* runtimes (v1 helped by L2, v2 above its own floor). See `docs/results.md` Step 2
  and `interview-prep.md` C4.
- **Step 2 remaining gates:** read v1-vs-v2 `dram__bytes_read` in Nsight (UI; `.ncu-rep` is binary),
  then the Step 2 quiz. Only then is Step 2 done.

## Next steps

1. Open `profiling/raw/{v1_naive,v2_tiled}.ncu-rep` in Nsight; record `dram__bytes_read.sum` per
   pass — expect a large QK/PV operand-read drop, with total DRAM floored by the S round-trip.
2. Step 2 quiz (close the gate; mark `ROADMAP.md` Step 2 `[x]`).
3. **Step 3 — online softmax (`kernels/v3_*`):** running max/sum so S never touches HBM. Predicted
   AI ~512, limiter finally crosses to MMA. This is the step that removes the surviving S
   round-trip Step 2 left standing.

## Git

- Solo learning repo; `main` is the working branch and the Colab pull target — commit directly to
  `main` (do not branch; the notebook pulls `main`). Commit/push only when the user asks.
- Do **not** commit `.vscode/` or other IDE cruft.

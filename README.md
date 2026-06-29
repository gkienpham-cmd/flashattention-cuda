# flashattention-cuda

> **Draft — written by my mentor for me to rewrite in my own words.** The README is mine to own;
> this is scaffolding prose, not the final voice.

**Bringing a kart racer's bottleneck-hunting to the GPU.** Rebuilding FlashAttention from scratch
as a roofline-driven optimization journey — where the journey itself, one *measured* speedup at a
time, is the deliverable. The guiding principle, from the FlashAttention-2/3/4 line: every
hardware generation has a different bottleneck, so before each optimization I identify the current
limiter and co-design the algorithm around it.

On a race track you find the limiting factor and trim time against a measurement. Same here: each
kernel version names its bottleneck (memory bandwidth, instruction issue, matmul throughput, the
exponential unit), predicts it with a roofline model *before* coding, then checks the prediction
against Nsight Compute *after*.

## The arc

| Phase | Era | Bottleneck | What answers it |
|---|---|---|---|
| 1 | FP16 fundamentals (T4) | HBM bandwidth | tiling, online softmax, fusion, tensor cores |
| 2 | FA-3 class (Ampere/Hopper) | latency / instruction issue | async copy, warp-specialization |
| 3 | low-precision (my Sconce twist) | matmul throughput + precision | INT8/FP8/INT4, block quant, incoherent processing |
| 4 | FA-4 frontier | shared-mem traffic + exponential unit | conditional rescaling, software `exp2` |
| 5 | inference | serving | causal/var-len masking, KV-cache, paged attention |

Full step list and hardware gating: [ROADMAP.md](ROADMAP.md). Per-step bottleneck/tradeoff log:
[docs/decisions.md](docs/decisions.md). The growing speedup curve: [docs/results.md](docs/results.md).

## Why it exists

This kernel library is the foundation for a from-scratch LLM inference engine I'll build next
(mini-vLLM style: PagedAttention + continuous batching), eventually serving a Sconce-compressed
model. That engine imports exactly one thing:

```python
from fa_kernels import attention
out = attention(q, k, v, causal=True, backend="v1_naive")  # backend advances as the journey does
```

## Repo layout

```
fa_kernels/   importable package: public API, config, dispatch, references
kernels/      versioned CUDA sources (vN_*) — the journey
bindings/     load.py — JIT build + arch-gencode auto-detect (T4/A100/B200/B300)
roofline/     the analysis tool: predict the limiter before each step
bench/        SDPA / FA-2 / cuDNN comparison harness (+ regime.py for the decode sweeps)
tests/        correctness vs SDPA (vs FP64 in the quantized phase)
profiling/    Nsight Compute capture + reading guide
docs/         results.md (the curve), decisions.md (the log), interview-prep.md (the concept chains),
              writeup.md (the essay), diagrams/ (figures), v*-kickoff.md + *-research.md (per-step plans)
notebooks/    build/test/bench/profile — see notebooks/README.md for the per-step index
```

## Hardware

Built and benchmarked on a **Colab T4 (Turing sm_75)**; Ampere/Hopper steps run on rented
A100/H100; Blackwell-only techniques are study-and-prototype. Every benchmark records GPU, arch,
and clocks (the free-tier T4 thermally throttles).

## Run it (on a GPU)

```bash
python -m roofline.predict --shape 1x8x2048x64 --precision fp32 --materialize-s  # predict
pytest tests/                                                                     # correctness vs SDPA
python -m bench.harness --backend v1_naive --precision fp32                       # measure
```

## Status

Phase 1 Step 1 (naive baseline) in progress. See [ROADMAP.md](ROADMAP.md).

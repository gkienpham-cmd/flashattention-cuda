# FlashAttention from Scratch

Rebuilt FlashAttention from the ground up in 17 CUDA kernels across 4 GPU architectures (T4, A100, B200, B300), following a roofline-first methodology where every optimization is predicted before coding and measured after. The deliverable is the measured speedup curve and the prediction-vs-reality scorecard, not just fast code. When the roofline model is wrong -- and it was wrong in 5 of 12 steps -- that miss is recorded, analyzed, and learned from. Four consecutive dead ends (tensor cores, double-buffering, occupancy, ILP) led to finding the score-stationary layout that actually moved the needle. Honest negatives are first-class results.

## The Journey

| Step | Kernel | Technique | Key Finding | Headline |
|------|--------|-----------|-------------|----------|
| 1 | v1_naive | FP32 three-pass baseline | L2 absorbs redundant reads; converges to floor at N=8192 | 20-30x slower than SDPA |
| 2 | v2_tiled | Shared-memory tiling | ncu: tiling cut ~0% DRAM -- L2 already owned operands; S = 99% of traffic | 1.3-3.2x over v1 |
| 3 | v3_online | Online softmax (S eliminated) | 125x less memory but 3-7x SLOWER -- nothing was bandwidth-bound | S proven gone (counter-free) |
| 4 | v4_fused | Fused FA-1 (O-rescale) | First S-off-HBM wall-clock win; GEMV-shaped FMA under-utilization | 1.7-2.6x over v2, 7.5-15x over v3 |
| 5 | v5_wmma | WMMA tensor cores (FP16) | Floor drops 8x; prefill bench outstanding | Correctness green (17/17) |
| 6 | v6_splitkv | Split-KV decode | Decode pivot: only 9-15% HBM (occupancy-bound, not bandwidth) | 5.7-8.2x vs naive, 1.5-3.3x vs SDPA |
| 7 | v7_paged | Paged KV (vLLM layout) | Crossover REFUTED: %HBM flat 9.4-12.4% at all batch sizes | Per-CTA-bound everywhere |
| 8 | v8_gqa | GQA M-packing | AI=2G/b model got magnitude right (first roofline win in 6 steps) | 8.6x over v7, beats SDPA 6-10x |
| 8 | v8_gqa_tc | Tensor cores on decode | **DEAD END:** REFUTED on T4 + A100 (1.8-4.6x slower -- wrong tool for decode) | |
| 8.5 | v8_gqa_db | Double-buffer KV | **DEAD END:** NULL -- memory was never the wall | |
| 8.6 | v8_gqa_occ/ilp | Occupancy + ILP | **DEAD END:** BOTH NULL -- floor is the serial recurrence (unhideable) | 4th consecutive negative |
| 8.7 | v8_gqa_ss | Score-stationary layout | Remove, don't hide -- recurrence 32x shorter; closes decode-schedule arc | 1.1-1.6x over v8, 8-16x vs SDPA |
| 9 | v9_fp8 | FP8 E4M3 KV cache | First ncu-validated measurement; %HBM caps ~28%; per-CTA confirmed | ~1.3x latency win |
| 10 | v10_nvfp4 | NVFP4 KV (B300/sm_103) | Per-CTA-bound to 2M tokens on 3 architectures; ~40 GB/s ceiling | Capacity 3.56x, latency negative |
| 11 | v11_mla | MLA latent-KV (B300) | Shape correct but engine wrong (4x cuBLAS gap -- need tensor cores) | 202x capacity, 148/148 correct |
| 12 | v12_mla_tc | tcgen05 TC (B300, CUTLASS) | Work-starvation, not per-CTA forever -- engine converts work to throughput | 1785 TFLOP/s = 36% FP8 peak |

The dead ends are a feature, not a bug. Each one narrowed the search space and built the evidence that led to the score-stationary layout (the real fix) and the work-starvation diagnosis (the real framing).

## Headline Results

- **Decode speedup:** 8-16x over PyTorch SDPA at all batch sizes (v8.7 score-stationary + GQA M-packing)
- **Tensor-core throughput:** 1785 TFLOP/s = 35.7% of B300 FP8 peak at serving scale (v12 CUTLASS tcgen05)
- **Per-CTA diagnosis:** earned across 7 steps, validated by Nsight Compute (counter-free proxy 13.8% vs ncu 12.85%)
- **Memory wins:** 125x (S elimination), 3.56x (NVFP4 capacity), 202x (MLA vs MHA)
- **Honest negatives:** 4 consecutive dead ends (tensor cores, double-buffer, occupancy, ILP) led to finding the score-stationary layout
- **Multi-architecture:** same ~40 GB/s per-CTA ceiling measured on T4, B200, B300

## Methodology

- **Roofline-first:** predict the limiter BEFORE coding, measure AFTER -- the prediction scorecard is itself a deliverable
- **Two-layer prediction model:** pure roofline (FLOP/byte) + per-CTA-corrected (schedule-aware) -- the second layer is the one that works
- **Counter-free %HBM proxy:** validated within 1 percentage point of Nsight Compute hardware counters
- **Honest misses:** when the model is wrong, that's recorded, analyzed, and learned from -- 5 of 12 steps had load-bearing magnitude misses
- **Research north star:** PMBS@SC characterization paper -- the first open, kernel-level, prediction-vs-measured roofline characterization of FlashAttention decode on B300/sm_103, across MHA/GQA/MLA and FP16/FP8/NVFP4, complementing (not beating) production kernels like FlashInfer and FlashMLA

## Hardware

| GPU | Arch | SMs | HBM | BW | FP16 TC | L2 |
|-----|------|-----|-----|----|---------|----|
| T4 | sm_75 (Turing) | 40 | 16 GB | 320 GB/s | 65 TFLOPS | 4 MB |
| A100 | sm_80 (Ampere) | 108 | 80 GB | 2039 GB/s | 312 TFLOPS | 40 MB |
| B200 | sm_100 (Blackwell) | 148* | 192 GB | 8 TB/s | 2250 TFLOPS | 132.6 MB* |
| B300 | sm_103 (Blackwell Ultra) | 148* | 288 GB | 8 TB/s | 2500 TFLOPS | 132.6 MB* |

\*Measured values that correct published specs (148 SMs not 160, L2 132.6 MB not 192 MB)

Built and measured on Colab T4 (free), rented T4/A100/B200/B300 via vast.ai (~$0.10-$0.50/hr). B300 is the paper's final target.

## Repo Layout

```
fa_kernels/   public API: attention(), paged_attention(), gqa/fp8/nvfp4/mla_attention()
kernels/      17 versioned CUDA kernels (v1_naive -> v12_mla_tc)
roofline/     limiter predictor: model.py, predict.py CLI, archs.py (T4/A100/B200/B300)
bench/        harness.py (SDPA comparison), regime.py (decode sweeps with L2-flush timer)
tests/        correctness vs SDPA, parametrized across all backends
docs/         results.md (the curve), decisions.md (the log), 32+ hand-authored SVG diagrams
notebooks/    44 Colab gate notebooks (per-step build/test/bench/profile)
```

## Run It (on a GPU)

```bash
# Predict the limiter before coding
python -m roofline.predict --shape 1x8x2048x64 --precision fp16

# Correctness vs SDPA
pytest tests/

# Prefill benchmark
python -m bench.harness --backend v4_fused --precision fp32

# Decode benchmark (GQA, score-stationary)
python -m bench.harness --backend v8_gqa_ss --gqa-group 8 --precision fp16

# Regime sweep (locked clocks, L2-flushed)
python -m bench.regime --backend v8_gqa_ss --sweep
```

## Status

Steps 1-12: DONE (June 2026). All correctness gates green, all quizzes passed. One outstanding item: v5 prefill bench (logged honestly -- the measurement was never captured).

**Research target:** PMBS@SC characterization paper -- the first open, kernel-level, prediction-vs-measured roofline characterization of FlashAttention decode on NVIDIA B300 (sm_103), across MHA/GQA/MLA and FP16/FP8/NVFP4, complementing (not beating) production kernels like FlashInfer and FlashMLA.

## What's Next

- **v13:** GLA / sparse attention (DSA, CSA -- the field's direction)
- **Arm 2:** Native NVFP4 tensor-core compute (unbuilt anywhere -- open novelty)
- **Three-generation spine:** H100/B200/B300 locked-clock measurements
- **Mini-vLLM:** The from-scratch inference engine this library feeds

## Links

- Full results: [docs/results.md](docs/results.md)
- Decision log: [docs/decisions.md](docs/decisions.md)
- Interview prep (concept chains): [docs/interview-prep.md](docs/interview-prep.md)
- Diagrams (32+ SVGs): [docs/diagrams/](docs/diagrams/)
- Decode arc replan: [docs/decode-replan.md](docs/decode-replan.md)

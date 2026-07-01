# Related work papers

Reference PDFs for the PMBS@SC 2026 paper. See `docs/research-strategy.md` Section 6 for
the full novelty assessment.

| File | arXiv | Title | Relevance |
|---|---|---|---|
| `attn-qat_2603.00040.pdf` | 2603.00040 | Attn-QAT: 4-Bit Attention With QAT | FP4 QK on Blackwell (prefill only, no decode/MLA) |
| `sageattention3_2505.11594.pdf` | 2505.11594 | SageAttention3: Microscaling FP4 Attention | FP4 attention on RTX 5090 (prefill only, no decode/MLA) |
| `snapmla_2602.10718.pdf` | 2602.10718 | SnapMLA: FP8 MLA Decoding | FP8 MLA decode on Hopper (no Blackwell/FP4/roofline) |
| `gla-tri-dao_2505.21487.pdf` | 2505.21487 | GLA (Tri Dao) | Closest prior: MLA decode roofline on H100 |
| `flashattention4_2603.05451.pdf` | 2603.05451 | FlashAttention-4 | BF16 prefill on B200 (no decode/sm_103) |
| `gpu-perf-model_2605.04178.pdf` | 2605.04178 | Microbenchmark-Driven GPU Perf Modeling | Owns roofline methodology (1.31% MAE, B200 GEMM) |
| `blackwell-microbench_2512.02189.pdf` | 2512.02189 | Microbenchmarking Blackwell Architecture | FP4 MMA at 96% peak on B200 (GEMM only) |
| `mla-hw-analysis_2506.02523.pdf` | 2506.02523 | Hardware-Centric Analysis of MLA | Analytical MLA study (no GPU kernel/measurements) |

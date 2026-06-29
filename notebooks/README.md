# notebooks/ — index

All build/test/bench/profile happens here (the author machine has no CUDA). Naming convention:

- **`*_gate.ipynb`** / **`*_run_of_record.ipynb`** — the clean, runnable source notebook.
- **`*_output.ipynb`** — the **executed** notebook with saved outputs = the **data of record** for that step.
- **`*_kernsum.txt`** — a saved `nsys` kernel-summary table (schedule corroboration).

Run on a rented GPU (vast.ai) or Colab T4; each notebook's §1 cell clones the public repo and `chdir`s to
the root, so later cells use `python -m ...` with no path prefix. See `../CLAUDE.md` for the per-step status
and `../docs/results.md` for the measured curve.

## Driver / setup

| notebook | purpose |
|---|---|
| `colab_bootstrap.ipynb` | Colab T4 build/test/bench driver (the Phase-1 workhorse) |
| `flashattention_original.ipynb` | the original monolithic scratch notebook (historical starting point) |

## Phase 1 — FP16 prefill fundamentals (T4 / A100)

| step | source | output (data of record) |
|---|---|---|
| v4 fused | `step4_run_of_record.ipynb` | `step4_outputs.ipynb` |
| v5 WMMA | `step5_run_of_record.ipynb` | *(unexecuted — Step 5 is PARTIAL)* |

## Decode arc — v6 → v10 (the B300/sm_103 paper)

| step | source gate | output (data of record) |
|---|---|---|
| v6 split-KV decode | `v6_decode_gate.ipynb` | *(in `step6_run_of_record.ipynb`)* |
| v7 paged KV | `v7_paged_gate.ipynb` | `v7_paged_gate_output.ipynb` |
| v8 GQA M-pack (Cut 1) | `v8_gqa_gate.ipynb` | `v8_gqa_gate_output.ipynb` |
| v8 Cut 2a (Turing WMMA) | `v8_gqa_tc_gate.ipynb` | `v8_gqa_tc_gate_cut2a_output.ipynb` |
| v8 Cut 2b (A100 probe) | `v8_cut2b_a100_probe.ipynb` | `v8_cut2b_a100_probe_output.ipynb` |
| v8.5 double-buffer | `v8_5_gqa_db_gate.ipynb` | `v8_5_gqa_db_gate_output.ipynb` |
| v8.5/8.6 past-L2 re-test | `v8_5_v8_6_pastL2_regime.ipynb` | `v8_5_v8_6_pastL2_regime_output.ipynb` |
| v8.6 reduction (occ/ILP) | — | `v8_6_reduction_gate_output.ipynb` |
| v8.7 score-stationary | `v8_7_score_stationary_gate.ipynb` | `v8_7_score_stationary_gate_output.ipynb` |
| v9 FP8 KV (Task 2) | `v9_fp8_gate.ipynb` | `v9_fp8_gate_output.ipynb` |
| v9 Task 1 regime char | `v9_task1_regime.ipynb` | `v9_task1_regime_output.ipynb` |
| v10 NVFP4 Gate 1 | `v10_nvfp4_gate.ipynb` | `v10_nvfp4_gate_output.ipynb` |
| v10 asymmetric recipe (synthetic) | `v10_asymmetric_ablation.ipynb` | `v10_asymmetric_ablation_output.ipynb` |
| v10 asymmetric recipe (real GPT-2 KV) | `v10_realkv_ablation.ipynb` | `v10_realkv_ablation_output.ipynb` |

## v10 B300/sm_103 record (the paper's measured core)

| deliverable | source | output (data of record) |
|---|---|---|
| regime knee-hunt | `v10_b300_regime.ipynb` | `v10_b300_regime_output.ipynb` (B200 dev-rung), `v10_b300_regime_outputb300.ipynb` (**B300/sm_103 record**) |
| 2×-exp softmax delta | `v10_b300_exp_ablation.ipynb` | `v10_b300_exp_ablation_output.ipynb` |
| FlashInfer comparator | `v10_b300_comparators.ipynb` | `v10_b300_comparators_output.ipynb` |
| nsys schedule | — | `v10_nsys_kernsum.txt` (B200), `v10_b300_nsys_kernsum.txt` (B300) |

# B300 decode-kernel research diagrams

Standalone SVGs accompanying [`../b300-decode-research.md`](../b300-decode-research.md) and the post-v6
re-plan [`../decode-replan.md`](../decode-replan.md). Each is self-contained (embedded colors, white
background) — open in a browser, macOS Preview/Quick Look, or VS Code and it renders correctly on its own.

**Post-v6 figures (the new measured result + the reorder):**

| File | What it shows |
|---|---|
| `decode-roofline-crossover.svg` | **The headline — REFRESHED to MEASURED (v7, 2026-06-27).** Achieved %HBM vs effective batch `BH`. The v7 `--batch` sweep measured it **FLAT at 9.4–12.4% from BH=8→512** (d=64 and d=128 lines); the predicted rising crossover is now **ghosted and labelled REFUTED**. The wall is per-CTA (32 KB smem → 2 blocks/SM; 1-of-8 warps at N_q=1), so batch only adds waves. The next lever is GQA M-packing (per-CTA efficiency), not fewer bytes. |
| `gqa-mpacking.svg` | The occupancy lever: packing `G` query heads into `M` turns the `M=1` GEMV (KV read `G×`) into an `M=G` GEMM (KV read once), raising `AI = 2/b → 2G/b` and re-engaging tensor cores. |
| `splitkv-lse-merge-dataflow.svg` | The two-kernel schedule the roofline can't see — split-KV partial + LSE merge, and where the launch / under-occupied-merge / warp-shuffle overhead lives. |
| `b200-b300-seams.svg` | Verified B200→B300 deltas with the **flat-bandwidth** surprise highlighted; the decode consequence of each. |
| `build-roadmap-v6-v11.svg` | **Reordered** plan: v6 split-KV → v7 paged KV → **v8 GQA M-packing (the reorder)** → v9 FP8 KV → v10 NVFP4 + asymmetric precision (headline) → v11 MLA / speculative. Old FP8/FP4-before-GQA order shown struck through. |

**v7 deep-research close-out figures (2026-06-27 — accompany [`../v7-deep-research.md`](../v7-deep-research.md)):**

| File | What it shows |
|---|---|
| `v7-vs-sdpa-batch-crossover.svg` | **The v8 motivation.** v7's speedup over torch SDPA vs decode batch B (log). v7 wins **only at B=1** (2.65×/1.46×); SDPA overtakes by **B=8** (v7 drops to ~0.5×/0.32×) and stays ahead through B=64. v7 is SM-saturated at B=1 → per-token cost flat, while SDPA's collapses ~4.5× with batch. The serving regime (B≥8) is what **v8 must reclaim**. |
| `per-cta-limiter-anatomy.svg` | **Why batch can't help.** One T4 SM holds only **2 resident blocks** (32 KB smem each / 64 KB); inside each block, all 8 warps load KV but at `N_q=1` only **warp 0 computes**. Batch adds serial *waves* at the same 2-blocks/SM cap → flat %HBM. Right panel: v8 M-packing lights up G warps + GEMV→GEMM (M≥16 tensor-core gate). |

**Original decode-arc figures (still accurate):**

| File | What it shows |
|---|---|
| `decode-roofline.svg` | Decode arithmetic intensity (≈ 2/b FLOP/byte) vs the B300 FP4 ridge — why decode is ~300–500× memory-bound, and how FP4 KV / GQA / MLA walk it toward the ridge. |
| `asymmetric-precision-dataflow.svg` | The "smart trick": FP4-safe `P·V` (convex combination) vs precision-critical `Q·Kᵀ` + softmax (exp-amplified), with the KV cache in NVFP4. |
| `split-kv-schedule.svg` | FA4's `(1, heads, batch)` grid leaving ~147 SMs idle vs split-KV across all 160 SMs + a log-sum-exp merge. |

> Note: GitHub's inline Markdown preview sanitizes some SVG; if a diagram looks off in the web UI,
> open the raw file or view it locally — it renders fully in any standalone SVG viewer.

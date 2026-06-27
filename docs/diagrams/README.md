# B300 decode-kernel research diagrams

Standalone SVGs accompanying [`../b300-decode-research.md`](../b300-decode-research.md) and the post-v6
re-plan [`../decode-replan.md`](../decode-replan.md). Each is self-contained (embedded colors, white
background) — open in a browser, macOS Preview/Quick Look, or VS Code and it renders correctly on its own.

**Post-v6 figures (the new measured result + the reorder):**

| File | What it shows |
|---|---|
| `decode-roofline-crossover.svg` | **The headline.** Achieved %HBM vs effective batch `BH`. v6 was measured only at `BH=8` (12% HBM, occupancy-bound); split-KV self-disables and batch fills the SMs past the crossover `BH = 2·SM` (80 on T4, 320 on B300). Left of it = occupancy-bound (FP4 doesn't help); right = bandwidth-bound (FP4 pays). |
| `gqa-mpacking.svg` | The occupancy lever: packing `G` query heads into `M` turns the `M=1` GEMV (KV read `G×`) into an `M=G` GEMM (KV read once), raising `AI = 2/b → 2G/b` and re-engaging tensor cores. |
| `splitkv-lse-merge-dataflow.svg` | The two-kernel schedule the roofline can't see — split-KV partial + LSE merge, and where the launch / under-occupied-merge / warp-shuffle overhead lives. |
| `b200-b300-seams.svg` | Verified B200→B300 deltas with the **flat-bandwidth** surprise highlighted; the decode consequence of each. |
| `build-roadmap-v6-v11.svg` | **Reordered** plan: v6 split-KV → v7 paged KV → **v8 GQA M-packing (the reorder)** → v9 FP8 KV → v10 NVFP4 + asymmetric precision (headline) → v11 MLA / speculative. Old FP8/FP4-before-GQA order shown struck through. |

**Original decode-arc figures (still accurate):**

| File | What it shows |
|---|---|
| `decode-roofline.svg` | Decode arithmetic intensity (≈ 2/b FLOP/byte) vs the B300 FP4 ridge — why decode is ~300–500× memory-bound, and how FP4 KV / GQA / MLA walk it toward the ridge. |
| `asymmetric-precision-dataflow.svg` | The "smart trick": FP4-safe `P·V` (convex combination) vs precision-critical `Q·Kᵀ` + softmax (exp-amplified), with the KV cache in NVFP4. |
| `split-kv-schedule.svg` | FA4's `(1, heads, batch)` grid leaving ~147 SMs idle vs split-KV across all 160 SMs + a log-sum-exp merge. |

> Note: GitHub's inline Markdown preview sanitizes some SVG; if a diagram looks off in the web UI,
> open the raw file or view it locally — it renders fully in any standalone SVG viewer.

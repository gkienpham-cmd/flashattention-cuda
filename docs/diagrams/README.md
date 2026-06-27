# B300 decode-kernel research diagrams

Standalone SVGs accompanying [`../b300-decode-research.md`](../b300-decode-research.md). Each is
self-contained (embedded colors, white background) — open in a browser, macOS Preview/Quick Look, or
VS Code and it renders correctly on its own.

| File | What it shows |
|---|---|
| `decode-roofline.svg` | Decode arithmetic intensity (≈ 2/b FLOP/byte) vs the B300 FP4 ridge — why decode is ~300–500× memory-bound, and how FP4 KV / GQA / MLA walk it toward the ridge. |
| `asymmetric-precision-dataflow.svg` | The "smart trick": FP4-safe `P·V` (convex combination) vs precision-critical `Q·Kᵀ` + softmax (exp-amplified), with the KV cache in NVFP4. |
| `split-kv-schedule.svg` | FA4's `(1, heads, batch)` grid leaving ~147 SMs idle vs split-KV across all 160 SMs + a log-sum-exp merge. |
| `build-roadmap-v6-v11.svg` | Staged build plan: split-KV (v6) → paged KV (v7) → FP8 KV (v8) → NVFP4 + asymmetric precision (v9, headline) → GQA M-packing + B300 retune (v10) → MLA / speculative decode (v11). |

> Note: GitHub's inline Markdown preview sanitizes some SVG; if a diagram looks off in the web UI,
> open the raw file or view it locally — it renders fully in any standalone SVG viewer.

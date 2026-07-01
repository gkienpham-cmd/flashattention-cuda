# Rendering the paper figures on Claude Design

The 8 figures in `figures/` are hand-authored SVGs. Six of them are **conceptual /
schematic** diagrams that Claude Design (claude.ai) can make look considerably more
polished. Two are **quantitative data charts** whose bar heights encode exact
measured numbers — keep those accurate (see the warning below).

## Workflow

1. Open Claude Design and start a new artifact.
2. Paste the **House style** block below, then **one figure block** under it.
3. Iterate in the chat until it looks right.
4. **Export as SVG** (Design → export / "download SVG"). Ask for *text as real
   `<text>` elements, not outlined paths*, so it stays crisp and editable.
5. Save the exported file **over the same path** `figures/<name>.svg` — the basename
   is unchanged, so `main.tex` needs no edit. Re-compile on Overleaf and eyeball
   legibility at column width.

> `/design-sync` / the DesignSync tool is a **different** mechanism — it syncs a
> whole *component library* with a Claude Design design-system project and needs
> your claude.ai design authorization. It is overkill (and the wrong shape) for
> one-off paper figures. The export-and-replace flow above is the right path. If
> you ever do want to manage these as a Design System project, say so and I can set
> up the sync.

## Which figures to render

| Fig | File (basename) | Render on Claude Design? | Why |
|----|------------------|--------------------------|-----|
| 1 | `v12-related-work-landscape` | **Yes (high value)** | comparison matrix — polishes well |
| 2 | `v1-v12-arc-summary` | **Yes (high value)** | 4-phase journey / timeline |
| 3 | `v12-two-layer-prediction-model` | **Yes** | two-panel concept diagram |
| 7 | `v12-paper-positioning` | **Yes** | claim / don't-claim panels |
| 8 | `v12-arm2-research-plan` | **Yes** | decision-flow + result |
| 6 | `v12-engine-ridge-comparison` | Optional | semi-quantitative (ridge values must stay exact) |
| 4 | `v12-throughput-regime` | **Keep as data chart** | bar heights = exact measured % |
| 5 | `v12-work-starvation-correction` | **Keep as data chart** | bar heights = exact measured TFLOP/s |

> **Accuracy warning for Figs 4 & 5:** these encode real measured numbers as bar
> lengths. A generative tool will happily redraw "prettier" bars with the *wrong*
> proportions. If you render them, lock the exact numbers in the prompt and verify
> every bar against the table before using it — or leave them as the current
> accurate SVGs (or re-plot with matplotlib for a clean data-viz look).

---

## House style (paste this first, every time)

```
You are producing a figure for an IEEE two-column academic paper (PMBS @ SC26) on
GPU attention-decode performance. Constraints:
- Output a single clean, standalone SVG. Text must be real <text> elements (NOT
  converted to outlined paths), so it stays crisp and editable.
- The figure will be shrunk to ~3.5 inch column width (or ~7.2 inch for full-width
  ones), so favor few words and large type. Minimum effective text ~14px on a
  ~600px-wide canvas; titles ~22px. No text smaller than 13px.
- Flat, print-friendly, professional. A restrained palette: blue #3B82F6, green
  #22C55E, amber #F59E0B, red #EF4444, neutral grays #374151/#6B7280, near-black
  #2C2C2A text, white background. Light tints for box fills (e.g. #EFF6FF, #F0FDF4,
  #FEF3E4, #FEF2F2). No heavy drop-shadows, no gradients, no 3D — they muddy at
  print size.
- Do NOT invent numbers or labels. Use exactly the text/values I give you.
- Sans-serif (Helvetica/Arial/Inter). Generous whitespace. Clean 1–1.5px strokes.
```

---

## Fig 1 — related-work landscape (full width, target ~760×300)

```
Make a clean comparison MATRIX titled "The empty cell in decode characterization".
6 rows x 5 data columns. Columns: Work | Arch | Attention | Precision | Decode |
Roofline. Use a green check for yes and a red cross for no in the Decode and
Roofline columns. Rows (verbatim):
- FA4        | B200      | Standard   | BF16        | cross | "partial" (amber)
- GLA        | H100      | MLA        | BF16        | check | check   ; small blue tag "closest prior"
- SnapMLA    | H100      | MLA        | FP8         | check | cross
- SageAttn3  | RTX 5090  | Standard   | FP4         | cross | cross
- FlashInfer / MLA | B300 | GQA / MLA | FP8 / NVFP4 | check | cross
- Ours       | T4–B300   | MHA→MLA    | FP16→NVFP4  | check | check
Highlight the "Ours" row with a light green fill and green border; make it read as
the punchline (the only row with decode + roofline across all types/precisions).
Header row bold with a rule beneath it.
```

## Fig 2 — v1→v12 arc summary (full width, target ~760×360)

```
Make a 4-column "phased journey" diagram titled "The v1→v12 arc: four phases, one
variable each". Four labeled phase columns, each a header + a vertical stack of
small rounded "kernel" chips. Each chip has a bold version label, a 3–5 word
takeaway, and a small status dot on the right. Colors per phase (header + chip
tint): Phase 1 purple, Phase 2 blue, Phase 3 amber, Phase 4 green.

Phase 1 "Prefill schedule": v1 naive — "HBM-bound; L2 masks it" (amber dot);
v2 tiled — "3× faster; 0% DRAM cut" (red); v3 online — "S gone; 3–7× slower" (red);
v4 fused — "1.7–2.6×; GEMV wall" (red); v5 WMMA — "predicted, not measured" (gray).
Phase 2 "Decode schedule": v6 split-KV — "5.7–8.2×; per-CTA" (red); v7 paged —
"flat %HBM, all batch" (red); v8 GQA pack — "8.6× at G=8" (amber); v8.7 score-st. —
"1.1–1.6×; recurrence out" (green).
Phase 3 "Precision": v9 FP8 KV — "1.3×; capacity+accuracy" (red); v10 NVFP4 —
"3.56× capacity; no latency" (amber).
Phase 4 "Shape + engine": v11 MLA — "M=128; still per-CTA" (red); v12 tcgen05 —
"2.4→1785; work-starvation" (green).
Bottom legend for the dots: green "roofline landed", amber "partial", red "missed
(per-CTA layer corrected it)".
```

## Fig 3 — two-layer prediction model (single column, target ~600×320)

```
Two side-by-side panels titled "Two-layer prediction model".
Left panel (neutral gray tint) "Layer 1: pure roofline": subtitle "AI = FLOPs /
bytes; peak roofs". A "Blind to:" list with red crosses: "L2 cache (hides HBM
traffic)", "per-CTA schedule / occupancy", "pipeline fill (work-starvation)".
Footer line (gray): "Wrong regime at every single-stream decode step."
Right panel (light blue tint) "Layer 2: per-CTA-corrected": subtitle "same AI + the
schedule". An "Adds:" list with green checks: "smem/TMEM → occupancy", "GEMV vs GEMM
(M=1 / 128)", "pipeline depth; counter-prediction". Footer (bold blue): "Matched the
limiter at all 7 decode steps."
Centered caption line under both: "Both predictions are pre-registered before coding
— the gap between the layers is the finding."
```

## Fig 6 — engine-ridge comparison (single column, target ~600×300)

```
A roofline-style schematic titled "Each engine sets its own ridge", subtitle
"MLA decode AI = 835 FLOP/byte; HBM 8 TB/s". A single horizontal axis labeled
"ridge = peak / bandwidth (FLOP/byte)" spanning 0 to ~2000. Draw a bold dashed
vertical reference line at AI = 835 (label it "AI = 835").
Three horizontal bars from the left, one per engine, length proportional to that
engine's ridge value (keep these EXACT): FP16 ridge 312 (green, ends left of the
835 line) labeled "FP16 · ridge 312 → compute-bound"; FP8 ridge 625 (green, still
left of 835) "FP8 · ridge 625 → compute (knife-edge)"; NVFP4 ridge 1875 (red,
extends well past 835) "NVFP4 · ridge 1875 → HBM-bound (peak unreachable)".
Small side annotations: left of the line "ridge < AI: compute-bound" (green), right
"ridge > AI: HBM-bound" (red). The visual point: two short green bars stop before
the line; the long red bar overshoots it.
```

## Fig 7 — paper positioning (single column, target ~600×300)

```
Two side-by-side panels titled "Positioning: complement, not beat".
Left panel (light green, green border) "We claim", with green checks:
"First open prediction-vs-measured decode roofline on sm_103"; "MHA → GQA → MLA ×
FP16 → FP8 → NVFP4"; "Per-CTA wall = work-starvation"; "Two-layer model, predictions
pre-registered before coding".
Right panel (light red, red border) "We do NOT claim", with red crosses:
"Faster than FlashInfer / FlashMLA / cuDNN"; "A novel roofline method"; "First to run
on sm_103 (FA4 already deployed)"; "A new SOTA kernel — v12 measured NVIDIA's own
kernel".
Balanced, symmetric two-column layout.
```

## Fig 8 — Arm 2 boundary result (single column, target ~600×300)

```
A small decision-flow titled "Does native FP4 compute help decode?" that ends in a
negative result.
Left: a prominent box (light red, red border) "Stage A (measured, B300)" with:
"FP4 vs FP8 GEMM at M=128 (MLA decode)"; "Both HBM-bound: AI 100–176"; "FP4 ≤ 5.8%
of 15 PF peak"; "FP8 ≤ 21.7% of 5 PF peak"; and a bold verdict line "KILL: FP4
compute never the limiter".
An arrow (gray) to a dashed grayed-out box "Stage B (accuracy) / Stage C (kernel) —
not triggered".
A full-width bottom banner (light blue, blue border, bold): "FP4/NVFP4 is a
capacity + accuracy lever, not a decode-compute lever."
```

---

## Optional — Figs 4 & 5 (only if you want them restyled; verify the numbers!)

### Fig 4 — throughput regime (single column ~600×400)
```
A grouped VERTICAL bar chart titled "Throughput scales with work B×K", y-axis
"% of FP8 peak" (0–40). Two series across four context lengths K = 8K, 32K, 131K,
524K. Series B=1 (blue) values: 1.0, 2.2, 5.9, 15.8. Series B=64 (red) values:
18.9, 29.5, 34.3, 35.7. Label the B=64 bar tops with their numbers. Legend: blue
"B=1 (single-stream)", red "B=64 (serving batch)". A small callout box:
"1–2% → 46% HBM as B×K grows. v11: flat 0.75 TF." Keep the bar heights EXACTLY
proportional to the values above — do not adjust for looks.
```

### Fig 5 — work-starvation correction (single column ~600×340)
```
Two panels titled "The per-CTA wall was work-starvation".
Left (amber) "CUDA-core (v6–v11)": six identical-length horizontal bars (the point
is they're all the SAME) labeled v6 split-KV, v7 paged, v8 GQA, v8.7 score-st.,
v9 FP8, v11 MLA; caption "every lever → flat 0.75 TF"; small note "dead ends: TC,
double-buffer, ILP, occupancy, bytes".
An arrow "swap engine" to the right.
Right (blue) "tcgen05 (v12)": five GROWING horizontal bars with lengths proportional
to 51, 295, 945, 1717, 1785 labeled "B1 K8K: 51", "B1 K131K: 295", "B64 K8K: 945",
"B64 K131K: 1717", "B64 K524K: 1785"; caption "2.4 → 1785 TF with work" and
"36% FP8 peak / 46% HBM at serving scale". Keep bar lengths proportional to the
numbers — verify before use.
```

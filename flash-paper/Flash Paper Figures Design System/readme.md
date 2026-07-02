# flash-paper — Figure Design System

House style + figure set for **"Roofline-Guided Characterization of Attention Decode on Blackwell: From Per-CTA-Bound to Work-Starvation Across MHA, GQA, and MLA"** — a PMBS @ SC26 submission (Kien Pham, Tufts University). This is not a product brand: the "product" is one IEEE two-column academic paper, and the design system exists to keep its 8 figures consistent, print-legible, and honest.

## Sources

- **Codebase**: mounted local folder `flash-paper/` — LaTeX draft (`main.tex`, `sections/*.tex`, `refs.bib`), original `figures/*.svg`, and `claude-design-prompts.md` (the house-style + per-figure prompt spec this system encodes).
- **Uploads**: the 8 original hand-authored figure SVGs (`uploads/*.svg`), preserved verbatim in `assets/originals/`.
- **Paper artifact repo** (referenced in the paper): https://github.com/gkienpham-cmd/flashattention-cuda

## The figure set

| Fig | File | Status |
|----|------|--------|
| 1 | `figures/v12-related-work-landscape.svg` (760×300) | restyled |
| 2 | `figures/v1-v12-arc-summary.svg` (760×360) | restyled |
| 3 | `figures/v12-two-layer-prediction-model.svg` (600×320) | restyled |
| 4 | `figures/v12-throughput-regime.svg` | **original kept** — bar heights = exact measured % of FP8 peak |
| 5 | `figures/v12-work-starvation-correction.svg` | **original kept** — bar lengths = exact measured TFLOP/s |
| 6 | `figures/v12-engine-ridge-comparison.svg` (600×300) | restyled — ridge bars exactly 312/625/1875 on a 0–2000 axis |
| 7 | `figures/v12-paper-positioning.svg` (600×300) | restyled |
| 8 | `figures/v12-arm2-research-plan.svg` (600×300) | restyled |

All restyled figures use real `<text>` elements (never outlined paths) so they stay crisp and editable, and keep the source's copy **verbatim** — numbers and labels are never invented or reworded. Basenames match the paper repo, so a file can be dropped over `flash-paper/figures/<name>.svg` with no `main.tex` edit.

**Accuracy rule (non-negotiable):** Figs 4–5 (and the bar lengths in Fig 6) encode measured data as geometry. Never redraw them "prettier" without locking every proportion to the numbers; when in doubt, keep the original.

## CONTENT FUNDAMENTALS

- **Voice**: first-person plural academic — "We build twelve attention kernels…", "We complement, not beat". Confident but scrupulously bounded: claims are always paired with explicit non-claims ("We do NOT claim").
- **Casing**: sentence case everywhere — titles ("The empty cell in decode characterization"), headings, list items. Never title case, never all-caps except verdict words ("KILL", "NOT").
- **Density**: telegraphic figure copy — 3–5 word takeaways ("HBM-bound; L2 masks it"), semicolons over sentences, symbols over words (→, ×, ≤, =, %). Full sentences only in footers/captions.
- **Terminology is exact and never simplified**: kernel version tags (`v6 split-KV`, `v8.7 score-st.`, `v12 tcgen05`), arch names (`sm_103`, B300, tcgen05, TMEM), precision chains (`FP16 → FP8 → NVFP4`), units (TFLOP/s, FLOP/byte, TB/s).
- **Numbers are sacred**: every figure number comes from a measurement or a pre-registered prediction. Do not invent, round, or "clean up" values.
- **No emoji, ever.** Verdicts use unicode ✓ / ✗ / the word "partial".
- **Rhetorical signature**: negative results are stated as prominently as positive ones ("KILL: FP4 compute never the limiter"); the punchline row/panel is visually highlighted (green tint) and reads as the resolution of the figure.

## VISUAL FOUNDATIONS

- **Background**: always flat white. No gradients, no textures, no images, no drop-shadows, no 3D — everything must survive grayscale print at ~50% shrink.
- **Color**: five flat accents — blue #3B82F6, green #22C55E, amber #F59E0B, red #EF4444, purple #8B5CF6 — each with a dark text tone and a light tint fill (see `tokens/colors.css`). Neutrals: ink #2C2C2A for titles, #374151 body, #6B7280 captions, #9CA3AF muted. Color is **semantic**, never decorative: green = claim/win/landed, red = non-claim/miss/kill, amber = partial/caution, blue = our model/takeaway, purple = phase-1 only, gray = blind/inactive/not-triggered.
- **Type**: sans-serif system stack (Helvetica/Arial). Scale on a ~600px canvas: 22px/700 titles, 16px/700 panel headings, 14px/700 labels, 13px body. **Hard floor 13px** — the figure prints at roughly half size.
- **Panels**: tint fill + 1.5px solid border in the matching accent + 10px radius. Dashed border = inactive/"not triggered". Chips: 8px radius, 1px soft border. Tags: 4px radius, tint fill, no border.
- **Strokes**: 1–1.5px, 2px max for highlight rows. Dashed 5,4 for reference lines (e.g. the AI = 835 line).
- **Layout**: title top-left, optional gray subtitle under it; generous whitespace; two-panel comparisons are symmetric; the "punchline" element (Ours row, verdict banner) is the only emphasized element per figure.
- **Charts**: geometry always exactly proportional to values; label bars with their numbers; axes light gray with 13px tick labels.
- **Motion / interaction**: none. These are static print figures — no animation, hover, or transitions anywhere in this system.
- **Canvas sizes**: single column 600px wide (~3.5in printed), full width 760px (~7.2in). Heights 300–400px typical.

## ICONOGRAPHY

- **No icon font, no icon set, no logos.** The paper has no brand mark; where an identity is needed, render "flash-paper" or the paper title in plain type. (No logo was provided, so none was created.)
- The only glyphs are **unicode text characters** used as semantic marks: ✓ (green, bold), ✗ (red, bold), status dots (plain circles), arrows → and ×/≤/= inside copy. The word "partial" in amber substitutes for a half-check.
- Directional arrows in flow diagrams are simple 1.5–2px gray lines with small triangular markers.
- Never introduce emoji, illustrations, clip art, or pictographic icons.

## Fonts

No font binaries ship with this system by design: the house style specifies the ubiquitous **Helvetica/Arial** system stack so SVGs render identically in browsers, Inkscape (Overleaf's `svg` package), and print without embedding. If you'd rather standardize on Inter, provide the files and `tokens/typography.css` gets an `@font-face`.

## Index

- `styles.css` — global entry; imports everything in `tokens/`
- `tokens/colors.css`, `tokens/typography.css`, `tokens/figure.css` — the token layer
- `figures/` — the 8 deliverable SVGs + `figures/index.html` gallery + `figures/cards/` (Design System tab cards)
- `assets/originals/` — the 8 untouched source SVGs
- `guidelines/` — specimen cards: accent palette, neutrals, panel tints, type scale, verdict marks, canvas sizes
- `components/figure/` — React figure primitives
- `templates/paper-figure/` — blank house-style figure template
- `SKILL.md` — agent skill entry point

## Components

- `FigureFrame` — white canvas + 22px title + gray subtitle (column/full width)
- `TintPanel` — tinted concept panel (6 tones, dashed variant)
- `Verdict` — ✓ / ✗ / "partial" list row
- `StatusDot` — landed/partial/missed/unmeasured dot + label
- `KernelChip` — version chip with takeaway + status dot

**Intentional additions**: all five components are extracted from recurring patterns in the 8 source figures (panels, verdict lists, kernel chips, status dots) — the LaTeX codebase defines no component inventory, so this small set exists solely to compose new figures in the established style.

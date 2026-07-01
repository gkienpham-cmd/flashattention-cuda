# flash-paper — PMBS@SC 2026 draft

First LaTeX draft of *"Roofline-Guided Characterization of Attention Decode on
Blackwell: From Per-CTA-Bound to Work-Starvation Across MHA, GQA, and MLA."*

## Layout

```
main.tex              # IEEEtran, conference, two-column; \input's the 7 sections
sections/*.tex        # intro, background, method, results, discussion, related, conclusion
refs.bib              # bibliography (IEEEtran style)
figures/*.svg         # the 8 main figures (copied from ../docs/diagrams/)
Makefile              # optional local build (needs a LaTeX toolchain + rsvg/inkscape)
```

## Building on Overleaf (recommended)

1. Upload the whole `flash-paper/` folder (or import from the repo).
2. **Menu → Settings → Compiler: pdfLaTeX.** Overleaf enables `--shell-escape` and
   ships Inkscape by default, which is what the `svg` package needs to render the
   `figures/*.svg` at compile time. No manual figure conversion required.
3. Compile. BibTeX runs automatically for `\bibliography{refs}`.

If a figure fails to render on Overleaf, the usual fix is to confirm shell-escape
is on (it is by default) — or fall back to PDF figures (see Makefile).

## Building locally

This machine has **no LaTeX toolchain and no SVG→PDF converter**, so the draft was
**not compiled here** — it is structured for Overleaf. To build locally you need
`pdflatex`/`latexmk` plus `rsvg-convert` or `inkscape`. The Makefile has:

- `make pdf`   — latexmk with `-shell-escape` (renders SVGs via the `svg` package).
- `make figures` — convert `figures/*.svg` → `figures/*.pdf` with `rsvg-convert`
  (fallback if you prefer `\includegraphics{...pdf}` over `\includesvg`).

## Format caveat

The exact PMBS'26 document class and page limit could **not** be auto-confirmed
from the CfP (only the deadlines were: full paper **Aug 5 2026 AoE**, late-breaking
Aug 26, workshop Nov 15). This draft defaults to `IEEEtran` (conference option),
two-column, targeting ≤8 pages. Verify the required class/limit against the PMBS
submission page (`pmbs-workshop.github.io`) before submitting.

## Status

First draft. See the honesty punch-list in the parent report / `docs/paper-outline.md`.
Open items include `\citeverify`-flagged author fields for a few real-but-not-fully-
attributed references in `refs.bib` (marked "Anonymous").

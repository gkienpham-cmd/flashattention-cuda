# flash-paper — RGD (PMBS@SC 2026 / arXiv)

LaTeX source of *"Roofline-Guided Characterization (RGD) of Attention Decode on
Blackwell: From Per-CTA-Bound to Work-Starvation Across MHA, GQA, and MLA."*

## Layout

```
main.tex               IEEEtran, conference, two-column; \input's the 7 sections
sections/*.tex         intro, background, method, results, discussion, related, conclusion
refs.bib               bibliography (all entries author-verified 2026-07-02)
figures/*.pdf          the 8 figures, arXiv-ready (vector PDF, embedded fonts)
figures/*.svg          editable SVG sources for the PDFs
arxiv-abstract.txt     plain-text title/abstract for the arXiv metadata form
rgd-arxiv.tar.gz       the upload bundle (main.tex + sections + refs.bib + figures/*.pdf)
Makefile               local build + figure regeneration targets
```

## arXiv submission

1. Upload `rgd-arxiv.tar.gz` at the "Prepare Files" step (it contains only
   `main.tex`, `sections/`, `refs.bib`, and `figures/*.pdf`; all file names use
   arXiv-safe characters, no spaces).
2. Compiler: **PDFLaTeX** (auto-detected; all figures are PDF, as arXiv requires —
   no shell-escape, no `svg` package).
3. Top-level TeX file: `main.tex`.
4. Paste the title and abstract from `arxiv-abstract.txt` into the metadata form
   (it is macro-free plain text).
5. Preview the PDF, check the 8 figures and the bibliography render, then submit.

## Building on Overleaf or locally

- Overleaf: upload the folder, compiler pdfLaTeX. No special settings needed
  (the `svg` package dependency was removed; figures are plain
  `\includegraphics{...pdf}`).
- Local: `make pdf` (needs latexmk/pdflatex; no shell-escape required).
- To regenerate a figure PDF after editing its SVG: `make figures` uses headless
  Google Chrome (present on this machine) to print each SVG to a tightly-cropped
  vector PDF; `rsvg-convert`/`inkscape` also work if installed.

## Provenance

Figures were refined in Claude Design (`Flash Paper Figures Design System/`,
committed for provenance) from the hand-authored SVGs, then audited: exact bar
geometry against the measured record, min 13px fonts, real text elements. Every
number in the paper reconciles to `../docs/results.md` / `../docs/decisions.md` /
`../docs/stage-a-results.md`; every bibliography entry was web-verified
(title + authors + arXiv id).

# Optional local build. Overleaf is the recommended path (see README.md).
# Requires a LaTeX toolchain (latexmk/pdflatex) AND an SVG converter
# (rsvg-convert or inkscape). Neither is assumed present on the author machine.

MAIN    := main
FIGDIR  := figures
SVGS    := $(wildcard $(FIGDIR)/*.svg)
PDFS    := $(SVGS:.svg=.pdf)

.PHONY: pdf figures clean

# Primary path: the `svg` package renders SVGs at compile time (needs shell-escape).
pdf:
	latexmk -pdf -shell-escape -interaction=nonstopmode $(MAIN).tex

# Fallback path: pre-convert SVG -> PDF, then switch \includesvg to \includegraphics.
figures: $(PDFS)

$(FIGDIR)/%.pdf: $(FIGDIR)/%.svg
	@command -v rsvg-convert >/dev/null 2>&1 && rsvg-convert -f pdf -o $@ $< || \
	 command -v inkscape >/dev/null 2>&1 && inkscape $< --export-type=pdf --export-filename=$@ || \
	 (echo "ERROR: install rsvg-convert or inkscape to convert $<"; exit 1)

clean:
	latexmk -C $(MAIN).tex 2>/dev/null || true
	rm -f $(MAIN).bbl $(MAIN).run.xml *.auxlock
	rm -f $(FIGDIR)/*.pdf

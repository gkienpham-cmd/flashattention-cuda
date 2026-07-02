# Local build + figure regeneration. Figures are already committed as PDF, so a
# plain pdfLaTeX toolchain suffices (no shell-escape, no svg package).

MAIN    := main
FIGDIR  := figures
SVGS    := $(wildcard $(FIGDIR)/*.svg)
PDFS    := $(SVGS:.svg=.pdf)
CHROME  := /Applications/Google Chrome.app/Contents/MacOS/Google Chrome

.PHONY: pdf figures arxiv clean

pdf:
	latexmk -pdf -interaction=nonstopmode $(MAIN).tex

# Regenerate figure PDFs from their SVG sources (headless Chrome, tight crop via
# a zero-margin @page wrapper; rsvg-convert/inkscape are fine alternatives).
figures: $(PDFS)

$(FIGDIR)/%.pdf: $(FIGDIR)/%.svg
	@W=$$(python3 -c "import re,sys;print(re.search(r'viewBox=\"0 0 (\d+) (\d+)\"',open('$<').read()).group(1))"); \
	 H=$$(python3 -c "import re,sys;print(re.search(r'viewBox=\"0 0 (\d+) (\d+)\"',open('$<').read()).group(2))"); \
	 T=$$(mktemp -d)/fig.html; \
	 { echo "<!DOCTYPE html><html><head><style>@page{size:$${W}px $${H}px;margin:0}html,body{margin:0;padding:0}svg{display:block;width:$${W}px;height:$${H}px}</style></head><body>"; cat "$<"; echo "</body></html>"; } > "$$T"; \
	 "$(CHROME)" --headless --disable-gpu --no-pdf-header-footer --print-to-pdf="$(abspath $@)" "file://$$T" 2>/dev/null; \
	 echo "built $@"

# The arXiv upload bundle: sources + PDF figures only, arXiv-safe file names.
arxiv:
	tar czf rgd-arxiv.tar.gz main.tex refs.bib sections/*.tex $(FIGDIR)/*.pdf
	@echo "wrote rgd-arxiv.tar.gz"; tar tzf rgd-arxiv.tar.gz

clean:
	latexmk -C $(MAIN).tex 2>/dev/null || true
	rm -f $(MAIN).bbl $(MAIN).run.xml *.auxlock

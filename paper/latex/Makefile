# Build with pdfLaTeX + BibTeX (the MLSys toolchain).
#   make             -> submission.pdf (anonymous main paper), submission_appendix.pdf (anonymous appendix), arxiv.pdf,
#                       then check_xref.py: fails if an \apprefFallback/\mainrefFallback number in paper.tex
#                       disagrees with the .aux files, a cross-PDF label has no fallback, or a reference is undefined
#   make check       -> check_xref.py alone (needs the .aux files of a full build)
#   make overleaf    -> full build and check, then ../bytes-are-not-latency-overleaf.zip (anonymous sources)
#   make overleaf-signed -> the same plus arxiv-identity.tex: ../bytes-are-not-latency-overleaf-signed.zip
#   make arxiv-src   -> arXiv package ../bytes-are-not-latency-arxiv-src.tar.gz and the PDF arXiv will
#                       produce from it (needs arxiv-identity.tex; see arxiv-identity.example.tex)
#   make figures     -> regenerate figures/*.pdf from ../figures/*.svg (needs cairosvg and Ghostscript)
LATEX  ?= pdflatex
BIBTEX ?= bibtex
FLAGS  := -interaction=nonstopmode -halt-on-error
# Rerun pdfLaTeX until cross-references are stable (at most four extra passes).
SETTLE = for i in 1 2 3 4; do grep -q 'Rerun to get' $(1).log || break; $(LATEX) $(FLAGS) $(1); done
FIGURES := figures/fig1_samework_ceiling.pdf figures/fig2_graph_ab_ratio.pdf figures/fig3_heldout_predicted_vs_measured.pdf
DEPS := paper.tex appendix.tex references.bib mlsys2025.sty mlsys2025.bst $(FIGURES)

OVERLEAF_ZIP := ../bytes-are-not-latency-overleaf.zip
OVERLEAF_FILES := README_EDITING.md arxiv.tex submission.tex submission_appendix.tex paper.tex appendix.tex \
	references.bib mlsys2025.sty mlsys2025.bst fancyhdr.sty algorithm.sty algorithmic.sty $(FIGURES)

# The .aux files are build outputs too, so `make' after `make clean' rebuilds and can run the check.
all: submission.pdf submission_appendix.pdf arxiv.pdf submission.aux submission_appendix.aux arxiv.aux
	python3 check_xref.py

check:
	python3 check_xref.py

arxiv.pdf arxiv.aux &: arxiv.tex $(DEPS) $(wildcard arxiv-identity.tex)
	$(LATEX) $(FLAGS) arxiv
	$(BIBTEX) arxiv
	$(LATEX) $(FLAGS) arxiv
	$(LATEX) $(FLAGS) arxiv
	$(call SETTLE,arxiv)

# The two submission PDFs reference each other (xr-hyper), so they are built together.
submission.pdf submission_appendix.pdf submission.aux submission_appendix.aux &: submission.tex submission_appendix.tex $(DEPS)
	$(LATEX) $(FLAGS) submission
	$(BIBTEX) submission
	$(LATEX) $(FLAGS) submission
	$(LATEX) $(FLAGS) submission_appendix
	$(LATEX) $(FLAGS) submission_appendix
	$(LATEX) $(FLAGS) submission
	$(call SETTLE,submission)
	$(LATEX) $(FLAGS) submission_appendix
	$(call SETTLE,submission_appendix)

figures:
	for f in $(FIGURES); do python3 svg2pdf.py ../$${f%.pdf}.svg $$f; done

# The zip is written only after the build and check_xref.py have passed.
overleaf: all
	rm -f $(OVERLEAF_ZIP)
	zip -X -q $(OVERLEAF_ZIP) $(OVERLEAF_FILES)

# Same package plus arxiv-identity.tex, for a private Overleaf project that should also compile the
# signed arXiv version (main document arxiv.tex); the submission entry points never read that file.
OVERLEAF_SIGNED_ZIP := ../bytes-are-not-latency-overleaf-signed.zip
overleaf-signed: all arxiv-identity.tex
	rm -f $(OVERLEAF_SIGNED_ZIP)
	zip -X -q $(OVERLEAF_SIGNED_ZIP) $(OVERLEAF_FILES) arxiv-identity.tex

# arXiv compiles the single file with \documentclass and does not run BibTeX; make_arxiv_src.py
# builds that file and main.bbl, then recompiles the package as arXiv does and checks the result.
ARXIV_SRC := ../bytes-are-not-latency-arxiv-src.tar.gz
ARXIV_PDF := ../bytes-are-not-latency-arxiv.pdf
arxiv-src: arxiv-identity.tex paper.tex appendix.tex references.bib mlsys2025.sty mlsys2025.bst $(FIGURES)
	python3 make_arxiv_src.py --tarball $(ARXIV_SRC) --pdf $(ARXIV_PDF)

clean:
	rm -f *.aux *.log *.blg *.out *.bbl

.PHONY: all check overleaf overleaf-signed arxiv-src figures clean

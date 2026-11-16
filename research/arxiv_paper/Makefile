# Build of the PINNeAPPle preprint.
#
#   make              build paper.pdf from the committed run records
#   make results      regenerate every table and figure from the records (no LaTeX)
#   make verification re-run the 72-case verification experiment (~4 min) and its analysis
#   make check        regenerate everything, then rebuild — use this before any submission
#   make bundle       build the arXiv submission tarball
#   make clean        remove LaTeX build products only (never the records)
#
# The engine is tectonic: it fetches exactly the packages the document needs and runs bibtex when it
# sees \bibliography, so no TeX distribution has to be installed.
#
# The Python steps need numpy, scipy, matplotlib and the repository importable. Point PY at an
# interpreter that has them, for example:
#
#   make check PY=/opt/homebrew/bin/python3.11

ENGINE  ?= tectonic
PY      ?= python3
PAPER   := paper.tex
PDF     := paper.pdf
BUNDLE  := arxiv/pinneapple-preprint.tar.gz

# the verification code imports the benchmark's reference solutions from the repository root
REPO    := $(abspath $(CURDIR)/../..)
RUN     := PYTHONPATH=$(REPO) $(PY)

TABLES  := $(wildcard tables/*.tex)
FIGS    := $(wildcard figures/*.pdf)
SOURCES := $(PAPER) refs.bib $(TABLES) $(FIGS)

.PHONY: all results verification bundle check clean distclean deps

all: $(PDF)

$(PDF): $(SOURCES)
	$(ENGINE) --keep-logs $(PAPER)

# a clear message instead of a numpy traceback when PY is the wrong interpreter
deps:
	@$(PY) -c "import numpy, scipy, matplotlib" 2>/dev/null || { \
	  echo "error: $(PY) is missing numpy, scipy or matplotlib."; \
	  echo "       run: make $@ PY=/path/to/python3"; exit 1; }

results: deps
	$(RUN) scripts/make_tables.py
	$(RUN) experiments/verification/analyze.py

verification: deps
	$(RUN) experiments/verification/run.py
	$(RUN) experiments/verification/analyze.py

# the numbers in the paper must be exactly the numbers in the records
check: deps results
	$(RUN) scripts/make_tables.py > /dev/null
	$(ENGINE) $(PAPER)
	@echo "ok: every table regenerated from the records and the PDF rebuilt"

# the arXiv bundle: sources and the generated tables/figures, plus the .bbl so the references
# resolve even if the processor does not run bibtex. No compiled PDF, no _work, no caches.
bundle: $(PDF)
	@mkdir -p arxiv
	@rm -f $(BUNDLE)
	@$(ENGINE) --keep-intermediates $(PAPER) > /dev/null 2>&1 || true
	@tar --exclude='_work' --exclude='__pycache__' --exclude='*.pyc' \
	     -czf $(BUNDLE) $(PAPER) refs.bib Makefile REPRODUCE.md README.md \
	     $(if $(wildcard paper.bbl),paper.bbl,) \
	     tables figures scripts experiments
	@echo "-> $(BUNDLE)  ($$(du -h $(BUNDLE) | cut -f1))"
	@tar -tzf $(BUNDLE) | wc -l | xargs echo "   files:"

clean:
	rm -f $(PDF) *.aux *.bbl *.blg *.log *.out *.toc *.xdv

distclean: clean
	rm -rf arxiv experiments/verification/_work

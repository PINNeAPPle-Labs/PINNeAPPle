# Build of the PINNeAPPle preprint.
#
#   make            build paper.pdf from the committed run records
#   make results    regenerate every table and figure from the records (no LaTeX)
#   make verification  re-run the 72-case verification experiment (~3 min) and its analysis
#   make bundle     build the arXiv submission tarball
#   make clean      remove build products of LaTeX only (never the records)
#
# The engine is tectonic: it fetches exactly the packages the document needs and runs bibtex when it
# sees \bibliography, so no TeX distribution has to be installed.

ENGINE  ?= tectonic
PAPER   := paper.tex
PDF     := paper.pdf
PY      ?= python3
BUNDLE  := arxiv/pinneapple-preprint.tar.gz

TABLES  := $(wildcard tables/*.tex)
FIGS    := $(wildcard figures/*.pdf)
SOURCES := $(PAPER) refs.bib $(TABLES) $(FIGS)

.PHONY: all results verification bundle clean check distclean

all: $(PDF)

$(PDF): $(SOURCES)
	$(ENGINE) --keep-logs $(PAPER)

results:
	$(PY) scripts/make_tables.py
	$(PY) experiments/verification/analyze.py

verification:
	$(PY) experiments/verification/run.py
	$(PY) experiments/verification/analyze.py

# the arxiv bundle: sources only, no compiled PDF, no _work, no caches
bundle: $(PDF)
	@mkdir -p arxiv
	@rm -f $(BUNDLE)
	@tar --exclude='_work' --exclude='__pycache__' --exclude='*.pyc' \
	     -czf $(BUNDLE) $(PAPER) refs.bib Makefile REPRODUCE.md README.md \
	     tables figures scripts experiments
	@echo "-> $(BUNDLE)"
	@tar -tzf $(BUNDLE) | sed -n '1,12p'

# the numbers in the paper must be exactly the numbers in the records
check:
	$(PY) scripts/make_tables.py
	$(PY) experiments/verification/analyze.py
	$(ENGINE) $(PAPER)
	@echo "rebuild clean"

clean:
	rm -f $(PDF) *.aux *.bbl *.blg *.log *.out *.toc

distclean: clean
	rm -rf arxiv experiments/verification/_work

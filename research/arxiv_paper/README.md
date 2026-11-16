# PINNeAPPle scientific preprint

Working title: **Reproducible cross-framework evaluation of physics-informed learning: matched
numerics, divergent defaults, and the limits of physics-based error detection**.

Authors: Yan Barros; Enrico Schiassi. Status: **draft, not submitted**. Nothing here claims arXiv
acceptance, and no statement of the paper is a claim of statistical significance.

## What this directory is

The manuscript, the table and figure pipeline, and the one experiment that was run specifically for
the paper (the verification corpus). The cross-framework benchmark it reports on lives in
[`benchmarks/vs_frameworks/`](../../benchmarks/vs_frameworks/) and is not duplicated here.

| file | what it is |
|---|---|
| `paper.tex` | the manuscript (19 pages); every table and figure is `\input`/`\includegraphics` from generated files |
| `refs.bib` | bibliography; every DOI was resolved against Crossref or DataCite when it was added |
| `protocol.md` | the preregistered protocol, including the verification corpus that was added to it |
| `tables/`, `figures/` | **generated**, never edited by hand |
| `scripts/make_tables.py` | `benchmarks/vs_frameworks/results/*.json` → cross-framework tables and figures |
| `experiments/verification/` | the 72-case error-detection study: `corpus.py`, `run.py`, `analyze.py` |
| `benchmark.py`, `test_benchmark.py` | the original dependency-light 1D Poisson sanity case this project started from |
| `REPRODUCE.md` | what regenerates what, and how long it takes |
| `arxiv/` | the submission metadata and the source bundle |

## Build

```bash
make                                  # tables and figures are committed; this only builds the PDF
make results PY=/path/to/python3      # regenerate every table and figure from the records
make check PY=/path/to/python3        # regenerate, rebuild, and prove the two agree
make bundle                           # the arXiv source bundle
```

`make` needs [tectonic](https://tectonic-typesetting.github.io) and nothing else; it fetches the LaTeX
packages the document uses. See `REPRODUCE.md` for the Python side.

## Where the numbers come from

Two immutable record sets, both committed:

* `benchmarks/vs_frameworks/results/*.json` — one record per benchmark run, with its configuration,
  package versions, thread count, wall and CPU time, and the load average before and after.
* `experiments/verification/results/verification.json` — one record per rule outcome for each of the
  72 corpus cases.

If a table and a record disagree, the record is right and the table is a bug in
`scripts/make_tables.py` or `analyze.py`.

## What the paper reports

1. Under an identical configuration the three PINN libraries are numerically the same; their cost per
   iteration is not.
2. Under each library's own recipe and an equal 240 s CPU budget, the defaults are worth a factor of
   ten on Poisson.
3. Applying the protocol to the inverse problem found and diagnosed a defect in PINNeAPPle's own
   adaptive loss weighting.
4. On a controlled 72-case corpus, reference-free physics checks catch 28 of 36 single-defect
   predictions; the residual check is scale-invariant under a homogeneous equation, amplifies small
   errors, and is bounded below by the finite-difference floor of the grid.

## Before submitting

See `NEXT_STEPS.md`. The short version: both authors must confirm authorship, order, affiliations
and funding; one independent reviewer in scientific ML should read it; and the arXiv category and
endorsement status must be checked against the current rules at submission time.

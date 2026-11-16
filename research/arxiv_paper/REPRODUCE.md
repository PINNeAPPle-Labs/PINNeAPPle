# Reproducing the preprint

Everything in the manuscript comes from two sets of committed records. Nothing is typed by hand and
no table is edited after generation.

```
records                                        what regenerates them
─────────────────────────────────────────────  ────────────────────────────────────────────────────
benchmarks/vs_frameworks/results/*.json        research/arxiv_paper/scripts/make_tables.py
research/arxiv_paper/experiments/               research/arxiv_paper/experiments/verification/
  verification/results/*.json                     analyze.py
```

| what | command | machine time |
|---|---|---|
| rebuild every cross-framework table and figure | `python research/arxiv_paper/scripts/make_tables.py` | seconds |
| rebuild the verification tables and figures | `python research/arxiv_paper/experiments/verification/analyze.py` | ~3 min (the Allen--Cahn reference and its resolution study dominate) |
| re-run the 72-case verification experiment, then analyse it | `python research/arxiv_paper/experiments/verification/run.py` | ~4 min, CPU, no GPU, no network |
| build the PDF | `make` (tectonic) or `tectonic paper.tex` | ~1 min |
| build the arXiv source bundle | `make bundle` | seconds |

The two `analyze`/`make_tables` steps need the repository importable: run them from the repository
root.

## Why the benchmark itself is not re-run here

The 10.2 process-hours of cross-framework runs behind `results/*.json` are recorded, not repeated:
each record carries its configuration, the versions of every package, the thread count, the load
average before and after, and the driver's return code. Re-running them needs a machine with
DeepXDE 1.15.0, PhysicsNeMo 2.2.2, PhysicsNeMo-Sym 2.4.0 and neuraloperator 2.0.0 installed side by
side; `benchmarks/vs_frameworks/PROTOCOL.md` records how, including the CUDA-compiler workaround that
PhysicsNeMo-Sym needs on a CPU-only host. The commands are in Appendix C of the manuscript
(`paper.tex`) and below.

## Re-running one track of the benchmark

```bash
python -m benchmarks.vs_frameworks.driver env
python -m benchmarks.vs_frameworks.driver run --track A1 --iters 2000 --seeds 0 1 2 --jobs 2
python -m benchmarks.vs_frameworks.driver collect
```

`driver run` writes into `_work/`; `driver collect` merges the new runs into `results/*.json`.
`--device cuda` runs the same code on a GPU, which none of the numbers in this manuscript did.

## Verification experiment

The corpus is deterministic: the same 72 cases are rebuilt byte-identically on any machine, because
they are analytic functions evaluated on fixed grids. `run.py` writes `results/verification.json`
(every rule outcome for every case); `analyze.py` writes `results/summary.json`, the LaTeX tables and
the figures.

Nothing in the corpus needs a GPU, a dataset or a network. It needs the repository importable, which
is how it reaches the benchmark's own reference solutions
(`pinneapple_tools.benchmark_suite.crossframework.problems`).

## Building the PDF

`tectonic` fetches exactly the packages the document uses, runs bibtex when it sees
`\bibliography`, and needs no TeX distribution:

```bash
brew install tectonic    # or: cargo install tectonic
make
```

`make check` regenerates every table and figure first, then rebuilds the PDF, so a stale table cannot
survive into a submission.

## What is deliberately absent

`_work/` (the per-case directories `run.py` writes) and `paper.pdf` are build products and are not
committed; the arXiv bundle excludes them. Raw per-run prediction arrays (`.npz`) of the
cross-framework benchmark are several GB and are not in the repository either: only the metrics,
provenance and configuration of each run are, which is what the tables use.

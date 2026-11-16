# Publication roadmap — PINNeAPPle arXiv preprint

**Working title:** PINNeAPPle: Towards Reproducible and Physics-Verified Physics-Informed Learning  
**Authors (requested order):** Yan Barros; Enrico Schiassi  
**Status:** Research draft; no submission or benchmark superiority claims.  
**Branch:** `research/arxiv-reproducible-physics-ai-paper`

> Authorship names and order are supplied by the project lead. Confirm both authors' consent, final contributions, affiliations, ORCID identifiers and corresponding author before submission. Do not invent affiliations.

## P0 — Scientific positioning and authorship
- [ ] Confirm Yan Barros and Enrico Schiassi approve authorship, order and manuscript scope.
- [ ] Collect accurate affiliations, emails, ORCIDs, funding, competing interests and contributions.
- [ ] Formulate a falsifiable central hypothesis and clearly differentiate PINNeAPPle from DeepXDE, PhysicsNeMo and NeuralOperator.
- [ ] Audit actual PINNeAPPle modules at pinned SHA; document implemented versus experimental features and code paths.
- [ ] Select primary arXiv category based on paper's actual content; check current moderation, endorsement and submission requirements.
- [ ] Identify appropriate venue after arXiv (e.g. JOSS subject to its current eligibility requirements or a scientific computing journal).

**Exit:** Authors approve a one-page scientific claims document with traceable evidence.

## P0 — Reproducible experiment infrastructure
- [ ] Create `experiments/` with versioned YAML configs, adapters, unified CLI, metrics, dataset manifests and plotting scripts.
- [ ] Implement PINNeAPPle, DeepXDE and NVIDIA PhysicsNeMo adapters; include NeuralOperator for matched operator-learning tasks.
- [ ] Pin exact package versions, git SHAs, Python/CUDA versions, GPU type, numerical precision and OS/container digest.
- [ ] Set deterministic seeds for Python, NumPy and each framework; record known nondeterminism.
- [ ] Save run manifests, raw predictions, train/validation/test indices, loss curves, hardware utilization and failures.
- [ ] Provide `make smoke`, `make benchmark`, `make figures` (or documented equivalents) and clean-environment CI.
- [ ] Add automated tests for shape, units, coordinate frames, BC enforcement, metrics and file checksums.

**Exit:** An independent contributor can reproduce a smoke run from a clean checkout.

## P0 — Benchmark design and reference validation
- [ ] Poisson 1D: manufactured exact solution, BC and PDE residual checks.
- [ ] Burgers 1D: documented numerical reference and convergence study.
- [ ] Heat equation 2D: manufactured solution, temporal and spatial error.
- [ ] Darcy 2D: held-out coefficient fields, fixed dataset provenance and splits.
- [ ] Flow case: independent solver, mesh/time-step convergence and Reynolds-number definition.
- [ ] Neural operators: FNO/DeepONet on compatible tasks, avoiding invalid task comparisons.
- [ ] Pre-register OOD scenarios (coefficient shifts, geometries, forcing, Reynolds number) and thresholds.
- [ ] Separate matched architecture comparisons from framework-native tuned comparisons.

**Exit:** All datasets and reference solutions have provenance, units, licenses and independent correctness checks.

## P0 — Experimental execution and statistical analysis
- [ ] Run at least five independent seeds per configuration; report all failures and exclusions.
- [ ] Equalize device, precision, training data, collocation budgets and compute budgets where appropriate.
- [ ] Report relative L2 and absolute errors, physical-unit residuals, BC error, conservation violations and OOD degradation.
- [ ] Measure train time, inference throughput/latency, peak GPU memory and accelerator-hours.
- [ ] Report median, IQR, uncertainty intervals and effect sizes; use statistical tests only when justified.
- [ ] Evaluate VeriPhysics with deliberately invalid predictions and false-positive/false-negative rates.
- [ ] Run ablations: without verification, without progressive fidelity, and other features only if implemented.
- [ ] Archive immutable result manifests and generate all tables/plots from them.

**Exit:** Every quantitative statement in the manuscript maps to an immutable run artifact.

## P1 — Manuscript completion
- [ ] Rewrite abstract after results exist; state measured contributions rather than aspirational features.
- [ ] Write related work with verified primary sources for PINNs, DeepXDE, PhysicsNeMo, FNO, DeepONet and verification.
- [ ] Document software architecture and actual public APIs.
- [ ] Add benchmark setup, hyperparameters, hardware and computational budget tables.
- [ ] Add accuracy-vs-cost, OOD, verification calibration, variability and failure-case figures.
- [ ] Explain limitations, negative findings, threats to validity, scope of physical guarantees and generalization limits.
- [ ] Include reproducibility statement, software/data licensing and archived DOI.
- [ ] Review English, notation, captions, accessibility, bibliography and LaTeX compilation.

**Exit:** All coauthors approve a complete manuscript without placeholders.

## P1 — Independent review and submission
- [ ] Obtain at least two independent technical reviews, including one expert in scientific ML/PINNs.
- [ ] Address review comments and verify figures are regenerated from released artifacts.
- [ ] Confirm authorship, intellectual-property permissions, institutional policies and preprint posting rights.
- [ ] Check current arXiv source-package limits, category, endorsement needs and metadata.
- [ ] Compile and inspect the exact submission bundle; submit only with both authors' approval.
- [ ] After announcement, link arXiv ID, release/tag, archival DOI and citation metadata; track corrections.

**Exit:** Public arXiv record exists and linked artifacts reproduce the claims. Acceptance cannot be guaranteed.

## Proposed directory layout

```text
research/arxiv_paper/
  README.md
  NEXT_STEPS.md
  protocol.md
  paper.tex
  benchmark.py
  test_benchmark.py
  experiments/       # TODO: configs and framework adapters
  datasets/          # TODO: manifests only; large data external
  results/           # TODO: manifests and aggregated outputs
  figures/           # TODO: generated, never manually fabricated
  environment/       # TODO: lockfiles/container recipes
```

## Immediate implementation order
1. Author approval and code audit.
2. Benchmark API plus independent analytical/reference tests.
3. PINNeAPPle and DeepXDE matched Poisson/Burgers adapters.
4. PhysicsNeMo adapter and operator-learning track.
5. Multi-seed execution, OOD cases and VeriPhysics tests.
6. Analysis, manuscript revision, independent review and submission.

## Non-negotiable scientific safeguards
No invented measurements, no fabricated citations, no implied physical guarantees without proof, no selective reporting of successful seeds, and no claim that arXiv acceptance is certain.

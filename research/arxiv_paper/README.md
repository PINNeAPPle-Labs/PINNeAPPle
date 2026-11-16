# PINNeAPPle scientific preprint project

Working title: **PINNeAPPle: Reproducible Physics-Informed Learning and Physics-Verified Neural Operators**.

This is a research workspace, NOT a claim of validated performance or arXiv acceptance.

## Step-by-step publication TODO

- [ ] Confirm authorship, affiliations, contributor permissions, funding, conflicts and software license.
- [ ] Audit implementation and version pinning of PINNeAPPle, DeepXDE, PhysicsNeMo and neuraloperator.
- [ ] Define a falsifiable research question and a precise novelty claim; distinguish implemented features from proposals.
- [ ] Choose arXiv primary category (likely cs.LG, physics.comp-ph or cs.NA depending on scientific emphasis), verify current category rules and endorsement eligibility.
- [ ] Freeze commit hashes, dependencies, container images, datasets, reference solvers and hardware inventory.
- [ ] Run deterministic smoke tests and independently verify analytical solutions.
- [ ] Execute all benchmark configurations with at least five independent seeds, record failed runs and wall time.
- [ ] Use identical data, physical domains, precision, boundary conditions, evaluation grids and compute budgets across frameworks.
- [ ] Compare matched architectures and also framework-native best-practice configurations; clearly separate both tracks.
- [ ] Measure relative L2, absolute error, PDE residual in physical units, boundary error, conservation violations, OOD degradation, training/inference time, peak GPU memory and failure rate.
- [ ] Include confidence intervals, effect sizes and failure analysis; avoid unsupported superiority claims.
- [ ] Generate plots and tables only from immutable run artifacts.
- [ ] Release reproducibility scripts, dataset provenance, configuration, licenses and a documented one-command reproduction workflow.
- [ ] Complete manuscript with related work, methods, limitations, ethics, availability, author contributions and references.
- [ ] Check LaTeX build, bibliography, figures, accessibility, anonymization if needed, and arXiv submission constraints.
- [ ] Obtain independent technical review and author approval before submitting.
- [ ] Upload arXiv source manually; inspect compiled preview and respond to moderation requests.

## Layout

- `benchmark.py`: dependency-light analytical benchmark baseline; emits JSON with machine and run provenance.
- `test_benchmark.py`: correctness and reproducibility checks.
- `protocol.md`: preregistered comparison methodology.
- `paper.tex`: manuscript skeleton, deliberately without fabricated numerical results.

## Quick start

```bash
python research/arxiv_paper/benchmark.py --output /tmp/pinneapple-poisson.json
python -m unittest discover -s research/arxiv_paper -p 'test_*.py' -v
```

The baseline is **not** a PINNeAPPle/DeepXDE/PhysicsNeMo comparison. Implement and execute adapters before reporting cross-framework results.

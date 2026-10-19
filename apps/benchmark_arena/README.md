# Physics AI Benchmark Arena (day 22)

**Live:** https://arena.pinneapple.org · **Port:** 8105 · **Prefix:** `ARN`

Which architecture works best for each physical problem, under one protocol, with every number from a run that was
actually executed.

## Pain

Method comparisons in papers and vendor pages use different data, budgets, stopping rules and lucky seeds, and
rarely say how a method behaves with little data, outside its training range or on a new geometry. Teams pick a method
by reputation and find out later.

## Input

- Public benchmarks: two problems run by `run_public_benchmarks.py`, plus the published results of
  pinneapple-benchmark experiment 09 (`results/external/bench09/`, unchanged, with the source commit).
- The two arena problems are stored in `results/` with date, hardware and the dataset hash:
  - The OpenFOAM airfoil campaign of the zoo: scalars, 100 shapes.
  - Exact viscous Burgers fields with a declared PDE and a finite-difference solver baseline.
- Private benchmarks: your table of runs (CSV, Parquet, JSON, HDF5, NPZ, or a `DatasetSpec`), with:
  - the inputs, outputs and an optional group column;
  - an optional field grid and PDE;
  - constraints, the methods and regimes to run, the compute budget and the number of seeds.

## Output

- **Rankings per regime**, as mean ± std over seeds with ranks: interpolation, few data, extrapolation (one end of an
  input or the edge of the space), geometry change (held-out groups) and steep fronts (fields). The overall mean rank
  is also given.
- **Seed spread** per regime: one dot per seed.
- **Cost and physics**: training time, inference time per query, model size, parameters, the share of predictions
  breaking the declared constraints, and the PDE residual relative to the data. Numerical-solver baselines are scored
  on the same samples.
- **Audit**: an independent audit of the splits by `pinneapple_data.audit`, i.e. `pinneapple check` PC451–PC453.
- **Not run**: what was not compared, and why (inapplicable methods, missing regimes, GNN and R-OAMS).
- **Downloads**: `report.md`, `study_result.json` (a `pinneapple_data.contracts.StudyResult`, `study_type="benchmark"`)
  and `result.json`.

## Methods (PINNeAPPle)

`pinneapple_arena.regimes.run_regime_arena` runs the candidates of `pinneapple_analysis.surrogate.methods`:

- polynomial ridge, Gaussian process and gradient-boosted trees (classical ML);
- MLP;
- FNO and DeepONet (neural operators);
- PINN, when a PDE is declared;
- numerical solvers as baselines (the Burgers FDM).

Every run uses equal data (one split for all methods and seeds) and equal compute (the same `ComputeBudget`, enforced
by `train_within_budget`). The stopping rule is the same for all: the best of ten validation checkpoints.

Not compared:
- Graph networks need mesh connectivity, which parametric tables and grids lack. `pp.benchmark` runs MeshGraphNet on
  mesh fields.
- R-OAMS is not in the library.
- Published runs of pinneapple-benchmark experiment 09 (PINN vs FNO vs DeepONet, MLP and MeshGraphNet on five
  problems, `pp.benchmark` protocol) are shown as their own view, copied unchanged from the repository at a pinned
  commit with their hardware record (`results/external/bench09/`). The arena did not rerun them.

## Integrations

REST and Python (`run_regime_arena`); CSV / Parquet / HDF5 / NPZ; the shared contracts (`DatasetSpec` in,
`StudyResult` out); scikit-learn, XGBoost (optional) and PyTorch.

## Monetization

Private benchmarks on customer data (free tier: 5 per month), cloud runs with larger budgets, seeds and hardware, and
comparative reports for procurement and method selection.

## Limits

- Hyper-parameters are the library defaults; there is no per-method search. A tuned method can rank higher.
  `pp.benchmark` adds equal-trial searches and budget curves for field-operator problems.
- The budget is wall-clock on the hardware shown. Rankings between methods with close errors can change on other
  hardware or with other budgets; the seed spread shows how settled they are.
- "Steep fronts" is a proxy for discontinuities: the top quartile of held-out fields by maximum gradient. True shocks
  need a dataset that has them.
- The physics checks are the declared constraints and the declared PDE residual (finite differences on the grid).
- The public results are two problems; they say nothing about other physics.
- Thread jobs are cancelled at their next progress report; `ARN_JOB_EXECUTOR=process` kills them at once.

## Refresh the public results

```bash
cd apps/benchmark_arena && python run_public_benchmarks.py --budget 10 --seeds 3
```

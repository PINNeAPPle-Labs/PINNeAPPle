# Running a benchmark

`pp.benchmark` compares physics-AI methods (PINNs, neural operators, graph networks, MLPs, your own) on physics
problems (built-in or your own) under **the same compute budget**, and tells you whether the comparison can be
trusted. One call does everything; each stage can also be run alone.

```
prepare  ->  validate  ->  run (gates, search, final, equal FLOPs)  ->  report (tables, figures, diagnostics)
```

## In one call

```python
import pinneapple as pp

res = pp.benchmark.run(
    problems=["burgers", "darcy", "navier_stokes"],   # built-in ids, or Problem objects (yours)
    methods=["pinn", "fno", "deeponet"],               # built-in ids, or Method objects (yours)
    budget=pp.ComputeBudget(wall_s=300),               # per training run; or flops=..., steps=...
    budget_levels=[1/9, 1/3, 1.0],                     # error-versus-budget curves
    seeds=3, n_trials=4, search_fraction=1/3,          # the same search effort for every method
    equal_flops=True,                                  # an extra arm at equal FLOPs
    n_workers=4, threads_per_run=1,                    # every run gets the same hardware share
    out_dir="results/my_benchmark",                    # re-running resumes from here
)
res.table()            # pandas DataFrame
res.to_markdown()      # the report as Markdown; res.to_csv()
res.report()           # (re)writes table.md, table.csv, summary.json and figures/ in out_dir
res.diagnostics        # the advisor's findings (below)
```

From the shell, with a YAML file holding the same fields
([example](https://github.com/PINNeAPPle-Labs/PINNeAPPle/blob/main/examples/benchmark/bench.yaml)):

```bash
pinneapple benchmark prepare --config bench.yaml --out results/x     # artifacts + validation only
pinneapple benchmark run     --config bench.yaml --out results/x     # train and report
pinneapple benchmark report  results/x                               # report again from the cache
pinneapple benchmark run --config bench.yaml --out results/x --smoke # tiny sizes: a pipeline check in minutes
```

Options of `run`: `--force` (reuse an `out_dir` made with another configuration), `--skip-checks`,
`--allow-failed-validation`, `--apply-suggestions` (writes `config.suggested.yaml` with the advisor's edits),
`--no-prepare` (refuse to run when the artifacts are not prepared yet).

## What ends up in `out_dir`

| Path | Content |
|---|---|
| `config.yaml` | the resolved configuration; its hash guards the cache (a different configuration is refused unless `force`) |
| `hardware.json` | CPU, cores, memory, GPU, library versions |
| `prepared/<problem>/` | solver configuration, data, geometry (for the airfoil: shapes, a complete OpenFOAM case), data-generation cost, `manifest.json` with the SHA-256 of every file and the validation status |
| `validation/` | per problem: solver verification, data checks, the validation stack; `architecture.json`, `determinism.json` |
| `runs/` | one JSON per training job (spec, result, hardware, wall time): the cache. A job that evaluated its final model also leaves `runs/<job>.npz` (the reference and predicted field of the instance the report draws, in the same ndarray sidecar the problems' own campaigns use) |
| `table.md`, `table.csv`, `summary.json` | the report |
| `figures/` | `summary_table.png` (shareable), `error_vs_budget.png`, `pareto_error_vs_inference.png`, and `comparison_<problem>.png` for every case with a planar field |

## Stage 1: prepare

`pp.benchmark.prepare(config, out_dir)` creates the artifacts **before any training** and saves them for you to
inspect: the solver configuration, the geometry, the generated data, what generating it cost (core-seconds, measured)
and the hashes. `run` consumes only prepared artifacts: it prepares what is missing, and refuses to train when they are
stale (a hash no longer matches the configuration or a file changed) or failed validation, unless you pass
`allow_failed_validation=True`, which the table then shows on every page.

## Stage 2: validation, with PINNeAPPle's own validation stack

Run on the prepared artifacts, stored in `out_dir/validation/` and in the report's **Validation** section:

| Check | Library entry point | Applies to |
|---|---|---|
| Reference solver against an exact or manufactured solution, observed order | the problem's `verify_solver` (Cole-Hopf, Taylor-Green, a manufactured Darcy solution, NACA 0012 vs the NASA TMR value) | built-in problems |
| Grid convergence (GCI, observed order, asymptotic range) | `pinneapple_design.fitmesh.gci`, `pinneapple_analysis.verification.richardson_extrapolate` | problems with grid levels |
| Simulation preflight of the solver case | `pinneapple_data.preflight` | OpenFOAM / CalculiX problems |
| Mesh quality | `pinneapple_data.cae.mesh_report` | problems with a mesh |
| First-cell y+ | `verification.mesh_intelligence.estimate_required_first_cell_height` | wall-bounded flows |
| Physics audit of the solver's assumptions | `pinneapple_physics.physics_audit` (`ModelProfile`, `audit_assumptions`) | every built-in problem |
| Applicability domain: test inside, OOD outside the training envelope | `physics_audit.Envelope` | every problem |
| Data sanity: finite, split sizes, no duplicates or leakage between splits (hashes), OOD really out of distribution, normalisation from the training split | `FieldOperatorProblem.data_checks` | every problem (including your arrays) |
| Six-stage trust card | `pinneapple_lab.curation.trust_stages` | the report's Trust card section |

A failed blocking item stops the run. For a problem you bring (an `ArrayProblem` or your own `Problem`), the same
checks run against what you provide, and the report says which ones did not apply and why.

## Stage 3: run, with correctness gates

The gates run by default, are recorded with their measured values, and are never skipped silently:

* **Architecture (before training)**: forward shapes and dtype, parameter count, FNO modes at most the Nyquist limit and
  output invariant to the input resolution, DeepONet branch and trunk shapes (and its basis against the POD rank of the
  outputs), MeshGraphNet permutation equivariance, PINN derivatives against finite differences and **the PDE residual of
  an exact solution through the same compiled residual** (a wrong equation or a sign error fails here), and every
  method overfitting a tiny batch (to 10 % of its initial loss; a model that cannot even halve it is a broken setup,
  one that learns but more slowly is a warning). A pair that fails an error-level gate is not trained and is INVALID
  in the table.
* **Training (from every run)**: loss finite and decreasing, divergence, "did not learn" (judged on the full-budget
  final runs; the search runs and the short runs of the error-versus-budget curve are expected to learn less), the budget enforced, identical
  accounting across methods (hardware, threads, timer, FLOP counter), evaluation off the clock, the search touching
  validation data only, the seed spread (veriphysics tolerance), and a determinism spot-check (same seed, same steps,
  same loss).

Then the search (`n_trials` configurations per method, each for `search_fraction` of the budget, on validation data),
the final runs (`seeds` x `budget_levels`, test sets) and the equal-FLOPs arm. A row whose error is no better than a
trivial prediction is labelled **did not converge**, not shown as a plain number. `skip_checks=True` puts a "checks
skipped" banner on the table and the summary figure.

## Stage 4: report, and the advisor

`table.md` has one section per problem with **n in every column header**, integral quantities and fields in separate
tables (a method that predicts only integrals shows "integrals only" in the field columns), the simulation-data cost,
the time per new instance, the break-even number of queries against a per-instance method, a checks badge per row,
and, with `equal_flops`, an **Equal FLOPs** table (every method trained to the same FLOP count, with the error on
the instances all methods share, the FLOPs and wall time used, and what stopped each run).

**Diagnostics and suggestions** lists what in the setup is likely to hurt the result, each with a severity, the
measured evidence, the suggested change and, where possible, the exact configuration edit
(`res.apply_suggestions()` writes them to a new configuration for you to review). It looks for: a budget too small (the
error still falling between the last two budget levels), a model that did not learn or converge (with the likely cause
for a PINN: loss-term imbalance, nondimensionalisation, collocation density), too few seeds, an OOD set that is not
out of distribution or has a single case, leakage, normalisation from all data, FNO modes against the output spectrum,
a DeepONet basis smaller than the field's rank, a learning rate that spikes or makes no progress early, and solver or
mesh problems (GCI, y+, unconverged runs, CFL).

### Budget study and setup audit

Two optional parts answer "would this method do better with more compute, or with another setting?":

```python
pp.benchmark.run(...,
    budget_study={"airfoil": {"pinn_rans": [3600, 14400], "pinn": [3600]}},       # wall-clock seconds, seed 0
    ablations={"airfoil": {"pinn": {"lr1e-3": {"lr": 1e-3}, "lbfgs": {"lbfgs": 0.3}}}})   # changes to the best config
```

`budget_study` trains the best configuration again at each listed budget (seed 0, on the problem's study instances)
and adds those points to the seed-0 runs of `budget_levels`: a table per instance and `figures/budget_study.png`.
`ablations` trains the best configuration with each change at the full budget on the problem's audit instance and
puts it next to the unchanged run (the **Setup audit** table). Both only add jobs: adding budgets, problems, methods or variants to them reuses an existing `out_dir`;
changing or removing an entry is refused like any other configuration change (it would reinterpret cached runs).

### Field comparison

Every case that has a planar field gets the same picture in `figures/comparison_<problem>.png` -- the reference, the
surrogate and `|surrogate - reference|` side by side, the two solutions on **one shared colour scale**, and the error
as a number in the title -- drawn with `pp.viz.compare_solutions` and measured with `pp.viz.field_error_metrics`, so
the figure and the table cannot quote two different errors. The surrogate is the case's best field model on the metric
that judges it (integrals-only methods and rows that failed a gate are left out), from its full-budget, seed-0 run;
the reference is the case's own. The report's **Field comparison** section says how many cases have the figure and, for
every case that has not, why: `burgers` is a 1-D profile and the two CalculiX parts carry their fields on a 3-D node
cloud, so they declare it in `Problem.comparison_absent` instead of getting a picture that means nothing.

```python
@pp.benchmark.register_problem("my_case")
class MyCase(pp.benchmark.FieldOperatorProblem):
    comparison_absent = ""          # empty: the figure applies. A case without a 2-D field writes the reason here
```

A problem that has the field stores it with `comparison.save_comparison(path, reference, prediction, x=..., y=...,
quantity=..., unit=...)` from its trainer (the job's `comparison_path`), and the report draws it.

### Example: what the gates and the advisor said on PINNeAPPle-Benchmark experiment 09

[Experiment 09](https://github.com/PINNeAPPle-Labs/PINNeAPPle-Benchmark/tree/main/experiments/09_equal_budget_pinn_fno_deeponet)
is five problems and seven methods through `pp.benchmark`. From its report:

- **error** `[airfoil/pinn_rans]` Did not converge: the error is no better than a trivial prediction. Evidence: mean
  velocity rel. L2 on the held-out runs = 0.96 (>= 0.9). Suggestion: check the nondimensionalisation, the loss weights
  and the collocation density near walls and leading edges; consider more budget or a formulation with fewer stiff
  terms. With the training gate "no divergence" failing in 2 of its 29 runs, the row is INVALID. The budget study
  (`budget_study`: 1 h, 4 h) and the setup audit (`ablations`: learning rate, Adam -> L-BFGS) show neither more compute
  nor those settings help: the PINN collapsed to the zero-flow solution of its own equations.
- **warning** `[airfoil/mgn]` Still improving at the full budget: velocity rel. L2 0.368 at 1/3 of the budget, 0.169 at
  the full one (54 % lower for 3x). Edit: `problem_budgets.airfoil.wall_s: 900 -> 2700`.
- **warning** `[darcy]` The OOD set is not clearly outside the training data: 9 % of the OOD inputs outside the training
  distribution (threshold 50 %), 1 of 100 outside the applicability envelope. Suggestion: move the OOD inputs further
  out. (Its OOD column is read as a second test set.)
- The setup audit of the potential-flow PINN found that wall and Kutta weights of 1 instead of 10 lower its velocity
  error from 0.81 to 0.51: a wider search would have helped it.

The gates also caught problems in the benchmark itself before it was published (a preflight that did not know
`freestreamPressure`, a y+ estimate fed the wrong argument, a convergence metric a potential-flow model fails by
construction); each was fixed with a test.

## Your own problem or method

Wrap your arrays (inputs and outputs on a grid) and FNO and DeepONet apply at once; pass `pinn_setup` to add PINNs:

```python
prob = pp.benchmark.ArrayProblem("my_flow", train=(X_tr, Y_tr), val=(X_va, Y_va), test=(X_te, Y_te),
                                 ood=(X_ood, Y_ood), data_cost_s=3600.0)
pp.benchmark.run(problems=[prob], methods=["fno", "deeponet"], budget=pp.ComputeBudget(wall_s=120), out_dir="r")
```

Or register new ones (they are then usable by id, in YAML and in parallel workers):

```python
@pp.benchmark.register_method("my_net")
class MyNet(pp.benchmark.Method):
    title = "My net"
    def supports(self, problem): return isinstance(problem, pp.benchmark.FieldOperatorProblem)
    def space(self, problem): return {"lr": [1e-3, 3e-3], "width": [64, 128]}
    def train(self, problem, prepared, cfg, seed, budget, final=True, **job):
        ...  # train under `budget` with pinneapple_analysis.cost.train_within_budget; return a result dict
    def architecture_checks(self, problem, prepared, cfg): ...  # your gates (shapes, overfit, ...)

@pp.benchmark.register_problem("my_problem")
class MyProblem(pp.benchmark.FieldOperatorProblem):
    def generate(self, split, n, n_workers): ...   # (inputs, outputs, solver core-seconds)
    def verify_solver(self): ...                   # checks of your solver against a known answer
```

The built-in problems: `burgers`, `darcy`, `navier_stokes` (the operator-learning standards), `airfoil` (the aero app's
RANS airfoil; needs OpenFOAM), `inverse_heat` (h from sensors), `calculix_beam` and `calculix_bracket` (structural FEM
with CalculiX as the reference solver; needs `ccx`, see below). Methods: `pinn`, `fno`, `deeponet`, `mgn`, `mlp`,
`mlp_fields`, `pinn_rans`, `mlp_calculix`, `pod_mlp_calculix`. See [Choosing a method](choosing_a_method.md) for what
the published benchmark found.

## Structural: CalculiX as the reference solver

`calculix_beam` (a cantilever box beam, load spread over the tip face) and `calculix_bracket` (an L bracket, base
clamped, side load on the upright) sweep the geometry and the load in a Latin hypercube and solve every sample with
`ccx` -- the same protocol as the airfoil's OpenFOAM campaign, on the implicit side of the house. Each sample keeps the
nodal displacement, the von Mises field and two integrals (the free face's displacement, the bending stress of the mid
section); the meshes are structured C3D8I hexahedra whose topology does not change with the design, so a surrogate maps
the four design numbers to the fields directly.

```python
res = pp.benchmark.run(problems=["calculix_beam", "calculix_bracket"],
                       methods=["mlp_calculix", "pod_mlp_calculix"],
                       budget=pp.ComputeBudget(wall_s=120), seeds=3, n_workers=4, out_dir="results/ccx")
```

The report's cost table carries what a structural result has to carry: the solver's version (the native binary, or
`2.20` from the bundled image) and the wall time of one `ccx` run, which is the "time per new instance" the surrogates
are measured against. The tables also carry **beam theory** as a no-data row (Timoshenko's deflection, M c / I): a
surrogate has to beat it on the same columns to be worth its data.

Needs `ccx`: a native binary on `PATH`, or Docker, in which case
`pinneapple_simulation.external_solvers.calculix.runner` builds the bundled Debian image. On macOS, raise `JOB_MAX_PROCS`
(the RLIMIT_NPROC) if `ccx` fails to create its processes.

The image itself has a gate, because an image nobody runs is a file:

```bash
python scripts/calculix_docker_smoke.py                                      # the host's platform
python scripts/calculix_docker_smoke.py --platforms linux/amd64 linux/arm64  # both, the x86 one under emulation
```

It builds the image for that platform and runs a real cantilever inside it, comparing the tip deflection with
Timoshenko's (measured: `ccx` 2.20, -1.5128 mm against 1.5268 mm, -0.92 %). This is the same function
`.github/workflows/calculix-image.yml` calls on a runner; while the repository's runners are unavailable it is the gate
to quote, and `tests/test_calculix_image.py` runs it too (skipping without Docker).

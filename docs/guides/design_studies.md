# Design studies: one problem, many optimisers, verified candidates

`pinneapple_design.design_optimizer` describes a design optimisation problem once and runs any method on it under
the same evaluation budget. Every evaluated design is recorded, so the Pareto front, the convergence history, the
feasibility report and the cost are computed the same way whatever produced them. Candidates found on a surrogate
are re-evaluated with a reference model or solver before anyone calls them optimal.

It is the engine behind two apps: the Engineering Design Optimizer (black-box, surrogate + high-fidelity
verification) and the Physics Design Optimizer (gradient-aware, method comparison, design files).

## The problem

```python
from pinneapple_design.design_optimizer import (Constraint, DesignProblem, DesignSpace, FunctionEvaluator,
                                                Objective, Variable, run)

space = DesignSpace([Variable("x", -1.0, 3.0), Variable("y", -1.0, 1.0)])
f = lambda X: {"f1": X[:, 0] ** 2 + X[:, 1] ** 2, "f2": (X[:, 0] - 2) ** 2 + X[:, 1] ** 2, "g": X[:, 0] + X[:, 1]}
problem = DesignProblem("demo", space, [Objective("f1"), Objective("f2")], [Constraint("g", upper=1.5)],
                        FunctionEvaluator(f, ["f1", "f2", "g"], torch_fn=f))   # torch_fn: gradients via autograd
result = run(problem, "nsga2", budget=400, seed=0)
print(result.summary(), result.evaluations.rows(space, result.pareto())[:3])
```

* `Variable(kind="integer")` rounds a variable (fin counts, ply counts). Gradient methods relax it when the
  evaluator accepts fractional values (`relaxable=True`), mark those iterates `relaxed` and evaluate the rounded end
  point; relaxed designs never enter a Pareto set.
* `Constraint(kind=...)` labels a limit as `physical`, `manufacturing` or `validity`; `feasibility_report` lists
  each constraint's value, margin and status (`active` within 5 % of the bound, `inactive`, `violated`).
* An evaluator may return `"_valid"`; `False` (a surrogate outside its training range, a geometry that does not
  exist) makes the design infeasible with that reason.

## Evaluators

| Evaluator | Use |
|---|---|
| `FunctionEvaluator(fn, outputs, torch_fn=None)` | any vectorised Python model; `torch_fn` makes it differentiable |
| `ZooEvaluator(name, space, fixed=...)` | a `pp.zoo` model; its validity range marks designs invalid |
| `TorchModuleEvaluator(module, outputs, ...)` | an MLP, PINN, neural operator or differentiable physics model |
| `CommandEvaluator(command, space, outputs)` | an external solver: designs in a CSV, outputs back in a CSV |
| `from_model_ref(ref, space)` | `zoo:<name>`, a TorchScript file or `cmd:<command>` from a model reference |
| `GPSurrogate(problem, outputs)` | Gaussian-process surrogate trained on your evaluated designs |

## Methods and budget

`run(problem, method, budget=..., seed=...)` with `random` (Latin hypercube), `nsga2`, `pymoo-nsga2` (optional
pymoo), `scipy-de`, `scipy-slsqp` (finite-difference gradients), `bayesian` (GP, ParEGO) and `gradient` (Adam
through autograd). One evaluation is one design passed to the evaluator: a finite-difference gradient costs `d + 1`,
an autograd step costs one (and its wall time is reported). `compare_methods` runs several on the same budget and
seeds and reports the hypervolume (normalised over all runs), the evaluations needed to reach 95 % of the best, and
whether each run converged (hypervolume grew less than 1 % over its last quarter).

## Verification and active learning

```python
from pinneapple_design.design_optimizer import presets, run, select_candidates, verify_candidates

problem = presets.heatsink_problem("surrogate")                 # zoo MLP of the heat-sink FVM engine
res = run(problem, "nsga2", budget=600, initial=problem.space.from_dicts([presets.heatsink_baseline()]))
chosen = select_candidates(res.evaluations, res.pareto(), 5)
report = verify_candidates(problem, res.evaluations, chosen, presets.HeatsinkEvaluator("fvm"))
print([r["status"] for r in report.rows], report.errors["t_source_c"])
```

Statuses: `verified` (the reference ran and the design is feasible under it), `rejected by reference`,
`reference failed`, `only surrogate`. `surrogate_assisted(problem, reference)` fits a GP on reference runs,
optimises on it with each constraint applied to a pessimistic value (mean plus or minus 2 standard deviations),
verifies designs along the front and adds them to the data; the error it reports per round is measured at designs
the GP had not been trained on.

## Presets

* `heatsink_problem(fidelity)`: plate-fin heat sink in ducted air, minimise mass and cost (or pressure drop) within
  source temperature, pressure drop, fin stress and extrusion limits. Fidelities `surrogate` (zoo MLP, differentiable),
  `closed_form` (resistance network) and `fvm` (3-D finite-volume base, the reference). The cost model is an
  assumption the user sets (`CostModel`).
* `airfoil_problem()`: CST airfoil at Re 4e6, minimise drag and maximise spar depth at a lift target, on the zoo
  ensemble `airfoil_cst_rans_mlp` (differentiable). No reference solver ships with it, so its candidates stay
  "only surrogate" until verified with OpenFOAM.
* `plate_fin_bodies`, `airfoil_bodies` and `design_file(bodies, "stl" | "step")` write the chosen design as a binary
  STL or an AP214 faceted STEP file through the geometry library's writers;
  `preset_bodies(preset, design)` picks the right one.
* `preset_info(preset)`: what a form needs (variables, operating point, limits, cost model, objectives, baseline
  design and the reference that verifies candidates).

## After a run: what an app shows and exports

The design apps are built on these calls, so a run gives the same files and tables from Python:

* `verify_front(problem, ev, front, n, reference)` returns the candidates spread along the front, the verification
  report (None without a reference) and the status of each candidate;
* `write_design_files(folder, problem, ev, candidates, status, labels=...)` writes `evaluations.csv`,
  `candidates.csv` (with the verification status and, from a comparison, the run that found each one),
  `candidates_to_run.csv` (the variables only, to run in your solver) and `parameter_space.json`;
* `chart_points(problem, ev, front)` thins the evaluations for a trade-off plot and keeps the front;
* `verify_uploaded_results(space, candidates, results, objectives, constraints)` verifies the exported candidates
  against the solver results you ran offline, matching rows by their variable values;
* `MethodComparison.union_result()` and `.fronts(seed)` give a comparison as one result (for `study_result`) and
  each method's front.

## Limits

The heat-sink model assumes ducted laminar/developing flow without radiation; the fin stress is root bending of a
cantilever fin under a uniform lateral load. The hypervolume above two objectives is a seeded Monte Carlo estimate.
Bayesian optimisation refits a GP at every proposal, so it is meant for budgets of a few hundred evaluations at
most.

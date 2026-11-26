# Surrogate pipeline

`pp.SurrogatePipeline` takes one problem from its definition to a verified, exportable surrogate model, with
VeriPhysics as a layer over every step instead of a check at the end. It orchestrates what the library already does —
problem presets, reference solvers, designs of experiments, architectures and training, the supervised surrogate
study, the PINN validation suite, model export — and adds the part that was missing: the seven stages as one
reproducible run, and the ten gates that decide what the run may be used for.

```python
import pinneapple as pp

result = pp.SurrogatePipeline.from_preset("burgers_in_viscosity").run()
print(result.summary())          # burgers_in_viscosity: completed (7 pass, 1 warn, 2 not applicable)
print(result.to_markdown())      # the report an engineer files
pipeline.predict_verified(nu=0.03)
```

Three levels of use, one object:

```python
# beginner — a preset is a whole configuration
pp.SurrogatePipeline.from_preset("burgers_in_viscosity").run()

# engineer — the blocks that matter for the case, as keyword arguments or YAML
pp.SurrogatePipeline(
    name="fin_temperature",
    problem={"source": "preset", "preset": "burgers_1d"},
    parameters={"parameters": {"nu": [0.005, 0.06]}},
    data={"source": "simulation", "probes": {"x": 7}, "probe_at": {"t": 1.0}},
    architecture={"family": "auto", "candidates": ["gp", "mlp"]},
    verification={"mode": "strict", "acceptance": {"nrmse": 0.05}},
).run()

# researcher — replace a stage, or reach the objects the stages built
pipeline.register_stage(MyTrainingStage())
pipeline.run()
pipeline.state.bundle, pipeline.state.dataset, pipeline.state.model
```

`pp.SurrogateConfig` is **not** this configuration: that name is `pinneapple_design`'s surrogate settings. The
pipeline's aggregate is `pp.PipelineConfig` (also `pp.pipeline.SurrogateConfig`).

## The stages

| stage | what it does | what it writes |
|---|---|---|
| `problem` | builds the problem (preset, `ProblemSpec`, or equations written as text) and applies the conditions of the configuration | `problem.json` |
| `data` | designs the cases (`pinneapple_data.doe`) and solves each one with the reference solver | `data/dataset.npz`, `data/simulation.json`, `data/probe_points.npy` |
| `preprocess` | builds the dataset object and the train/validation/test/distribution-shift split | `data/dataset_report.json` |
| `model` | resolves the architecture and checks it fits the problem before any training time is spent | `model/architecture.json` |
| `training` | a PINN on the equations, or a study over the dataset (`pinneapple_analysis.surrogate`) | `model/training_history.json`, `model/study.md`, `model/surrogate/` |
| `export` | checkpoint, portable formats, figures and the model card | `export/model.pt`, `export/model.onnx`, `export/model_card.json`, `figures/` |
| `verification` | runs the gates and writes the report | `report.md`, `report.json`, `state.json` |

`run_until("data")` stops after the reference campaign; `resume()` continues without repeating it. `status()` and
`history()` read `state.json`, so an interrupted run is inspectable from a later session.

A physics-only run (`data.source: physics_only`) skips `data` and `preprocess` and records why:
*the model is trained on the equations, so no case is solved*.

## The gates

Every gate returns one of five statuses. The last two matter as much as the others: a gate that could not be
evaluated is never reported as one that passed.

| gate | establishes | not applicable when |
|---|---|---|
| `problem` | the configuration and the problem agree (dimension, names, parameters, conditions) | never |
| `geometry` | the geometry is the one declared | the domain is a box |
| `discretization` | the reference solution does not move when its grid is refined | the reference is a closed form, or there is none |
| `simulation` | every reference solve the data rests on finished | physics-only run |
| `dataset` | the cases cover the space, are distinct, and are split without leakage | physics-only run |
| `model` | the error on data the model never saw, against the criterion of the use case | no independent reference |
| `physics` | PDE residual and boundary/initial misfit of the trained model | the surrogate was fitted to reference solutions |
| `applicability` | where the model stays valid, and what happens outside it | physics-only run |
| `export` | the exported model predicts what the in-memory model predicts | export disabled |
| `inference` | a fresh query returns finite values | no model |

Thresholds come from `verification.acceptance` and are the use case's, not universal tolerances:

```yaml
verification:
  mode: strict          # strict fails on any WARN; balanced fails on FAIL; report never fails the run
  fail_policy: block    # block raises PipelineRejected; warn marks the run completed-with-findings
  acceptance:
    nrmse: 0.05         # parametric surrogate, held-out normalised RMSE
    relative_l2: 0.03   # field model against an independent reference
    residual: 0.05      # relative PDE residual (symbolic backend, from validate_pinn)
    bc: 0.01            # max boundary/initial misfit relative to the field scale
    refinement: 0.05    # QoI change between a grid and its refinement
    export_equivalence: 1e-4
```

A profile cannot loosen a tolerance: `execution.profile` changes threads and defaults, nothing else.

## Two backends, and why

* **`preset` / `spec`** — a `PhysicalProblem` that `pinneapple_physics.solving` can solve and compare. Training is
  `pp.solve(problem, method="pinn")`; the error is measured by `pp.compare` against the `analytic` or `reference`
  method.
* **`symbolic`** — equations written as text, parsed by `pinneapple_physics.symbolic_pde`, trained and validated by
  `pinneapple_analysis.validation.pinn` (`train_pinn` / `validate_pinn`). This is the only backend whose `physics`
  gate compares the residual against a threshold, because `validate_pinn` is the only one that computes a
  non-dimensional residual.

The equation language is the parser's, not SymPy's:

```
d(u, t) - alpha * lap(u)          # ✓
Derivative(u, t) - alpha * ...    # ✗ EquationSyntaxError: unknown function 'Derivative'
```

## The supervised surrogate

When the data come from solves, the fitting is `pinneapple_analysis.surrogate.run_study`: every applicable method
on the same splits and budget, held-out and distribution-shift error, its own trust checks, an out-of-distribution
gate and a validity envelope. The pipeline lifts all of that into the `dataset`, `model` and `applicability` gates
instead of duplicating it.

```yaml
data:      {source: simulation, probes: {x: 7}, probe_at: {t: 1.0}}
parameters: {parameters: {nu: [0.005, 0.06]}}
architecture: {family: auto, candidates: [gp, mlp]}
training:  {budget_seconds: 2.0}
```

`train`/`preprocess` never split a dataset by rows across cases: the distribution-shift set is removed first, then
train/validation/test, and the `dataset` gate fails on any overlap between them.

## Command line

```bash
python -m pinneapple.pipeline presets                      # the four ready configurations
python -m pinneapple.pipeline validate experiment.yaml     # configuration, problem and dependencies
python -m pinneapple.pipeline plan experiment.yaml         # the stages and the cost
python -m pinneapple.pipeline run experiment.yaml          # the whole thing, then the report
python -m pinneapple.pipeline resume experiment.yaml
python -m pinneapple.pipeline report experiment.yaml
```

Exit codes: `0` passed, `1` a stage failed, `2` the configuration is wrong, `3` rejected by a gate.

## What is not implemented

The configuration schema is deliberately smaller than the design that motivated it. These blocks are **absent**,
not silently ignored: a field that does nothing is worse than an absent one.

* `geometry`, `materials` — there is no geometry/material backend on this path yet; a preset that carries one reaches
  the run, and the `geometry` gate says the pipeline does not verify it.
* `optimization` (HPO), `adaptive_learning` — the study already compares candidates at equal budget; an
  infill loop over `pinneapple_data.doe.design(method="infill")` is the natural next step.
* `deployment` — export writes the artefacts and the card; serving them is the engineering API's job.
* `boundary_conditions` accepts `constant`, `symbolic` and `callable` sources on `preset`/`spec` problems only;
  a `symbolic` problem declares its conditions in `problem.conditions`.
* `early_stopping` raises `ConfigError`: the backends train for a fixed budget (`training.epochs`, or
  `training.budget_seconds` on the study backend).
* `export` for the symbolic backend: there is no portable export of a `PINNProblem` network yet.

## Where the parts live

```
pinneapple_pipeline/
├── configs.py       the configuration dataclasses (+ YAML/JSON round trip)
├── stages.py        the seven stages and the PipelineStage contract
├── adapters.py      the only place a configuration meets a module that computes
├── verification.py  the ten gates
├── results.py       VerificationResult, StageRecord, PipelineResult and its report
├── pipeline.py      SurrogatePipeline (run / run_until / resume / status / predict_verified)
├── presets.py       four ready configurations
├── figures.py       the optional plots
└── cli.py           python -m pinneapple.pipeline
```

Examples: [`examples/surrogate_pipeline/`](https://github.com/PINNeAPPle-Labs/PINNeAPPle/tree/main/examples/surrogate_pipeline).

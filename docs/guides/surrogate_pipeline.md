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
pipeline.deploy().predict(nu=0.03)        # the same answer, as a model service with the run's trust
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

`run(verbose=True)` prints one line per stage as it finishes — what it did and how long it took — which is what
makes a long run watchable from a terminal or a notebook. `run_until("data")` stops after the reference campaign; `resume()` continues without repeating it. `status()` and
`history()` read `state.json`, so an interrupted run is inspectable from a later session.

A physics-only run (`data.source: physics_only`) skips `data` and `preprocess` and records why:
*the model is trained on the equations, so no case is solved*.

## The gates

Every gate returns one of five statuses. The last two matter as much as the others: a gate that could not be
evaluated is never reported as one that passed.

| gate | establishes | not applicable when |
|---|---|---|
| `problem` | the configuration and the problem agree (dimension, names, parameters, conditions) | never |
| `geometry` | the surface of `geometry.path` — or the one `geometry.builder` builds — is watertight, manifold and in the stated units | no `geometry` block, or a box domain |
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

## Geometry, materials, and the search over the run

Four blocks go beyond "train the thing and score it":

```yaml
geometry:   {source: file, path: cases/bracket.stl, units: mm}     # checked, not meshed
geometry:   {source: builder, builder: cylinder, parameters: {radius: 0.05, length: 0.3, axis: y, n: 36}}
materials:  {materials: {body: al6061_t6}, assignments: {kb: conductivity}}
optimization:        {enabled: true, strategy: grid, search_space: {lr: [1e-3, 5e-3], hidden_dim: [32, 48]}}
adaptive_learning:   {enabled: true, max_rounds: 3, samples_per_round: 4, target_error: 0.02}
deployment:          {enabled: true, target: http, port: 8000}
```

* **`geometry`** reads the surface with `pinneapple_data.cae.io.read_stl` and runs
  `pinneapple_data.cae.quality.surface_checks`: open and non-manifold edges, shells, winding, degenerate faces, the
  enclosed volume and the bounding box in metres. A surface that does not bound a volume fails the `geometry` gate
  instead of being meshed. `source: builder` builds the surface here with
  `pinneapple_design.geometry.bodies` (`cylinder`, `box`, `sphere`, `annulus`) and writes it to
  `<run>/geometry/surface.stl` — the gate then measures that file, so the surface under test is the one on disk and
  the `geometry` figure draws it. The pipeline checks the surface; meshing stays `pinneapple_design.geometry`'s job.
* **`materials`** resolves a key of `pinneapple_design.intent.materials.MATERIALS` and sets the problem's own
  parameters — only on parameters named exactly like the catalogue property, because `nu` is the Poisson ratio of a
  solid and the kinematic viscosity of a flow. Anything named differently goes through `assignments`. Every number
  arrives with the datasheet it was read from, and the Lamé constants derived from `E` and `nu` are recomputed so
  they cannot be left stale.
* **`optimization`** trains one model per point of `search_space` (grid, or `n_trials` random draws), scores each on
  validation points, and trains once more with the winner. The points the `model` gate reports on are drawn with
  another seed, so the reported error is not the one the selection was made on. It searches the PINN's training
  knobs; a supervised run already searches its candidates at equal budget.
* **`adaptive_learning`** adds cases where the response varies most: `doe.design(method="infill")` over the cases
  already solved, weighted by the outputs. Each round is solved, the dataset grows, the study is repeated, and the
  loop stops on the target, on `max_rounds` or on `max_total_simulations`. It is not a variance-driven active
  learning criterion — the per-round table shows a flat curve instead of assuming one away.
* **`early_stopping`** means three different things, and each says which: the symbolic backend runs `PINNMonitor`
  (it stops when a *physics* check has been failing for `patience` checks); the `preset`/`spec` backend trains in
  chunks of `check_every` epochs with `pinneapple_physics.solve_pde`, keeps the best held-out physics loss and stops
  when it stops falling; the study backend keeps its `budget_seconds`.
* **`deployment`** serves the run through `pinneapple_tools.model_services`: `pipeline.deploy()` returns a
  `ModelService` (and, with `target: http`, a live URL) whose trust function re-reads the validated domain and the
  gates. A query outside the domain is refused (`on_ood: reject`) or answered and flagged. `serve.py` is written into
  the run folder, so the endpoint can be restarted from the artefacts alone
  (`pp.SurrogatePipeline.from_run(folder).deploy()`).

## `pinneapple check` as a second opinion

Every run writes `verification/predict.py`, `verification/queries.csv` and `verification/pinneapple-check.toml`, and
runs `pinneapple_check` over them. That is an independent implementation that loads the model through a plain
`(N, n_inputs) -> (N, n_outputs)` function and applies its own rules: bounds (PC4xx), out-of-distribution queries
(PC6xx) and checkpoint integrity (PC8xx) always; boundary/residual (PC1xx-PC3xx) when
`verification.check_physics` names the physics the outputs belong to. The `applicability`, `export` and `physics`
gates quote those outcomes when they exist and fall back to their own measurement when they do not — and the report
says which one it used. A checker that cannot run becomes a note, never a silent pass.

## Seeing the run: `pipeline.view()`

```python
pipeline.view()                              # writes <run>/viewer/index.html and opens it
pipeline.view(open_browser=False)            # the page, without the browser
```

One `file://` page, no server and no internet: the stages, the ten gates with their values and the reason each one
gives, the figures the run wrote, the surface the `geometry` gate measured (the library's own three.js viewer, inlined
as `srcdoc` because a `file://` page cannot frame another), and — the part that moves — the model itself.

The explorer is the model evaluated on a grid of the space it was trained on: the first two inputs of the dataset, or
the coordinates of a physics-only model. Drag on the map and the readouts follow; two cuts through the point are drawn
under it (`u(x)` at that `t` and `u(t)` at that `x` for a PINN, each output for a parametric surrogate). Every cell is
a real query — `pinneapple_pipeline.viewer.payload` calls `predict_array` and rounds what came back — so the page
cannot show anything the checkpoint does not answer. A run with no trained model still opens, with its gates and a
note saying why there is nothing to sweep. `scene=False` skips the 3-D surface, which is the slow part of building
the page.

## Command line

```bash
python -m pinneapple.pipeline presets                      # the four ready configurations
python -m pinneapple.pipeline validate experiment.yaml     # configuration, problem and dependencies
python -m pinneapple.pipeline plan experiment.yaml         # the stages and the cost
python -m pinneapple.pipeline run experiment.yaml          # the whole thing, then the report
python -m pinneapple.pipeline run experiment.yaml --verbose # ... printing each stage as it finishes
python -m pinneapple.pipeline resume experiment.yaml
python -m pinneapple.pipeline report experiment.yaml
python -m pinneapple.pipeline view runs/demo/burgers_in_viscosity   # the run in a browser
```

Exit codes: `0` passed, `1` a stage failed, `2` the configuration is wrong, `3` rejected by a gate.

## What is not implemented

The configuration accepts nothing that silently does nothing: a field the library cannot back raises `ConfigError`
with the reason, and a gate that cannot be measured says so.

* `geometry.format` is STL only (`pinneapple_data.cae.io.read_stl`), and `geometry.repair` is refused — a surface
  that is not watertight is *reported*, not stitched. STEP inputs and the repair tools live in
  `pinneapple_design.geometry`.
* `materials.temperature_dependent` is refused: the catalogue carries single-temperature properties.
* `optimization` searches the PINN's training knobs, not the surrogate methods' internals (the study already
  compares those at equal budget).
* `adaptive_learning.strategy` is `infill` only, and it applies to parametric surrogates (there is no case to add
  to a physics-only model).
* `boundary_conditions` accepts `constant`, `symbolic` and `callable` sources on `preset`/`spec` problems only;
  a `symbolic` problem declares its conditions in `problem.conditions`.
* `export` for the symbolic backend: there is no portable export of a `PINNProblem` network yet.
* the viewer's explorer sweeps two inputs at a time (the first two); the rest stay at the middle of their range,
  and the page says so instead of pretending the others do not exist.
* `pinneapple check`'s physics rules need the field names its equations know (`T` for heat, `U`/`p` for flow,
  `u`/`v`/`w` or `D` for elasticity); when they do not match, the run records that the declared physics was not
  applied and the `physics` gate measures the residual itself.

## Where the parts live

```
pinneapple_pipeline/
├── configs.py       the configuration dataclasses (+ YAML/JSON round trip)
├── stages.py        the seven stages and the PipelineStage contract
├── adapters.py      the only place a configuration meets a module that computes
├── verification.py  the ten gates
├── checks.py        `pinneapple check` over the run folder, its rules read as gate evidence
├── deploy.py        the run as a model service, with the run's own trust function
├── results.py       VerificationResult, StageRecord, PipelineResult and its report
├── pipeline.py      SurrogatePipeline (run / run_until / resume / status / predict_verified / deploy)
├── presets.py       four ready configurations
├── figures.py       the optional plots
└── cli.py           python -m pinneapple.pipeline
```

Examples: [`examples/surrogate_pipeline/`](https://github.com/PINNeAPPle-Labs/PINNeAPPle/tree/main/examples/surrogate_pipeline).

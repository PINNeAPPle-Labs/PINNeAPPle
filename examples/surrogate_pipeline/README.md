# Surrogate pipeline — use cases

Seven runs that exercise the pipeline end to end and show what it refuses. Each one prints the stage table and the
VeriPhysics gates; the artefacts land in `_out/<name>/` (git-ignored), with the report at `_out/<name>/report.md`.

| script | what it exercises | backend | time |
|---|---|---|---|
| `01_parametric_surrogate.py` | a supervised surrogate over a parameter space: reference campaign → dataset → study (GP vs MLP) → gates | `preset` + `simulation` | ~15 s |
| `02_physics_only_pinn.py` | a PINN with no data, scored against the Cole-Hopf solution, exported to ONNX and compared back | `preset` + `physics_only` | ~90 s |
| `03_symbolic_heat.py` | a PDE written as text, trained and validated by the PINN validation suite — and the same run undertrained, refused | `symbolic` | ~2 min |
| `04_yaml_and_cli.py` | the pipeline as a YAML file: `run_until("data")`, `resume()`, and `python -m pinneapple.pipeline` | `preset` + `simulation` | ~20 s |
| `05_every_kind_of_surrogate.py` | the tour: parametric surrogate, physics-only PINN, a PDE as text, adaptive learning, `pinneapple check`, and the run served with its own trust | all three | ~2 min |
| `06_case_matrix.py` | the matrix: flow and structural, geometry built or imported, dataset solved or loaded, FNO, DeepONet, PINN — the same object every time | all three | ~3 min |
| `07_aircraft_and_crash.py` | solver data → surrogate → both drawn in 3-D: the library airliner through its vortex lattice, and the pole impact through the explicit crash solver | both solvers | ~9 min (or `--reuse`) |

```bash
python examples/surrogate_pipeline/01_parametric_surrogate.py
python examples/surrogate_pipeline/02_physics_only_pinn.py
python examples/surrogate_pipeline/03_symbolic_heat.py
python examples/surrogate_pipeline/04_yaml_and_cli.py [--config my.yaml]
python examples/surrogate_pipeline/05_every_kind_of_surrogate.py [--fast] [--only serve]
python examples/surrogate_pipeline/06_case_matrix.py [--fast] [--only fno]
python examples/surrogate_pipeline/07_aircraft_and_crash.py [--only aircraft|crash] [--reuse]
```

## What each one is meant to show

**01** — the surrogate is scored on cases it never trained on, and the pipeline says where it stops being valid: the
`applicability` gate warns that the corners of the viscosity range were held out as a distribution-shift set and
measured 20-odd times worse than the interior. That is the answer to "can I use this at nu = 0.06", and it comes from
the run, not from a threshold someone typed.

**02** — the training loss of a PINN goes to ~1e-2 and means nothing on its own. What licenses the model is the
`model` gate (relative L2 against a closed form the network never saw), the `physics` gate (the boundary and initial
conditions of the trained network, evaluated on fresh points) and the `export` gate (the ONNX file predicts what the
in-memory model predicts, to 1e-7).

**03** — the same run at 300 epochs instead of 2000: the residual passes, and the boundary and initial conditions are
traded off against it. `validate_pinn` names which term was lost, the `physics` gate fails, and the report says so
instead of shipping the model. The undertrained run is `fail_policy: report`, so it finishes and shows the verdict
rather than raising.

**04** — the expensive part of a parametric study is the reference campaign; `run_until("data")` does it and stops,
`resume()` picks the run up and does not solve anything twice (`state.json` records what finished).

**05** — the tour that doubles as the demo: the three kinds of model the pipeline builds (a study over a reference
campaign, a PINN with no data, a PDE written as text), then the four things that make it a pipeline and not a
training script: `doe.design(method="infill")` adding the cases where the response varies most, `pinneapple check`
loading the trained model through a plain array function and agreeing with the gates, `pipeline.deploy()` serving it
with the run's own trust function, and the refusal of a query outside the validated domain.

**06** — one pipeline, every kind of problem it covers, each case a few lines of configuration and a `run()`:
flow mechanics from 12 reference solves and from the dataset that came out of them; geometry built by the run (a
cylinder) and imported (a Wigley hull); the same campaign as a 64-point field for a Fourier neural operator and for a
DeepONet; a steel bar under a distributed load with the physics only, scored against its closed form; and a PINN that
never sees a case at all. The figures of each run are written next to it — held-out parity, where the error sits, the
training loss, the surface the `geometry` gate measured.

**07** — the shape the pipeline is for, outside the pipeline: a solver the library already ships is the reference,
a surrogate is fitted to the fields it produced, and the comparison is drawn on the geometry — reference, surrogate
and the difference on one colour scale — so "the surrogate agrees with the solver" is something you look at. The
airliner comes from the geometry library and the reference is its vortex lattice (40 strips, held-out nrmse 3e-7 for
a ridge over the incidences; the VLM is linear in alpha, so the surrogate is exact and the figure says so). The pole
impact comes from the explicit crash solver: 30 designs of the bumper beam and its crash boxes, and a surrogate that
reproduces the |u| field of a held-out design to 11 %.

## Configuration

`configs/` holds the same pipelines as files:

* `burgers_in_viscosity.yaml` — the supervised surrogate of example 01, commented.
* `heat_2d_symbolic.yaml` — the symbolic problem of example 03, including why `lbfgs_steps` and the boundary weights
  are set the way they are.

Any block left out takes its default; every block can be overridden (`from_yaml(path, training={"epochs": 500})`).

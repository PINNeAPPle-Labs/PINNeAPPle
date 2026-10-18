# AGENTS.md

Guide for coding agents (Gemini CLI, Antigravity, Claude Code, Codex, Cursor, ...) working with or on PINNeAPPle.
Every Python block in this file is run by `tests/test_agents_md.py`; keep them short and correct.

## What PINNeAPPle is

A Physics AI library: one problem description (`pp.PhysicalProblem`) solved by physics-informed networks, neural
operators, graph networks or classical solvers (FDM, FEM, OpenFOAM, CalculiX bridges) through one call, and a trust
layer that checks every result against the physics (residuals, conservation, boundary conditions, references,
applicability) before anyone uses it.

## Install

```bash
pip install pinneapple                 # core: numpy, scipy, torch, matplotlib
pip install "pinneapple[mcp]"          # + the MCP server (pinneapple-mcp)
pip install -e ".[dev]"                # from a checkout, with pytest and ruff
```

On Colab: the notebooks in [`notebooks/colab/`](notebooks/colab/README.md) install it in their first cell.

## Use these APIs first

| Task | API |
|---|---|
| Use a pre-trained, verified model or a dataset | `pp.zoo.list()`, `pp.zoo.load(name).predict(...)`, `pp.zoo.validate(name, inputs)`, `pp.datasets.load(name, split=...)` |
| Describe a problem | `pp.PhysicalProblem.from_preset(name)`, `ProblemBuilder(...)...build()` + `pp.PhysicalProblem.from_pde_spec`, `pp.list_presets()` |
| Solve it | `pp.solve(problem, method, **options)`: `"pinn"`, `"fem"`, `"analytic"`, `"reference"`, `"exact"`, `"external"` (`pp.list_methods()`) |
| Compare methods on the same points | `pp.compare(problem, methods, reference=...)` |
| Check a result | `pp.trust_card(grid, values, pde=..., dirichlet=..., reference=..., train_bounds=...)` (decision APPROVED / REVIEW / REJECT) |
| Validate a model (zoo, PyTorch, ONNX, callable) | `pp.validate(model, problem, data=..., query=..., level="quick" / "standard" / "thorough")` (credibility per factor, decision pass / warn / fail; [docs](docs/guides/pp_validate.md)) |
| Validate an autoregressive rollout (trust horizon, drift, long-run statistics) | `pinneapple_analysis.validation.rollout.validate_rollout(pred, ndim, ref, long_ref=...)` or `pp.validate(step, {"rollout": {...}}, data=...)` ([docs](docs/guides/rollout_validation.md)) |
| Validate a trained PINN (fresh-point residual, failure modes, exact solutions) | `pinneapple_analysis.validation.pinn.validate_pinn(net, problem)` or `pp.validate(net, {"pinn": "burgers_1d"})`; `PINNMonitor` during training ([docs](docs/guides/pinn_validation.md)) |
| Physics checks during training (Lightning, PhysicsNeMo, Hugging Face) | `pinneapple_neural.callbacks.PhysicsMonitor(problem, reference=..., every=N)` with `PhysicsValidationCallback` / `PhysicsNeMoHook` / `PhysicsTrainerCallback` ([docs](docs/guides/training_callbacks.md)) |
| Worked validation examples per model family and physics | `examples/validation/NN_*.py` (`run(smoke=True)`), each ending in `pp.validate` + trust card + `pinneapple check` ([docs](docs/guides/validation_gallery.md)) |
| Deploy decision from your own checks | `pinneapple_analysis.trust.TrustReport`, `Check` |
| Which methods are validated | `pinneapple_catalog.validated_methods()`, `pinneapple_catalog.methods.method_status()` |
| Equal-budget benchmark | `pp.compare` with the same epochs / collocation / seeds per method (below); `pp.benchmark.run(...)` where installed |
| A field, a prediction against its reference, or a deformed shape on its mesh in an app | `pp.viz.field_view(mesh, fields, deform=..., markers=...)` + `pinneapple_analysis.validation.error_map.error_map(points, pred, ref, regions=...)`; page module `/shared/fieldview3d.js` ([docs](docs/core_concepts/studio.md)) |
| Animations with synced charts | `pp.dashboard.Dashboard(...)`: `.to_png`, `.to_html`, `.to_video(gif=...)` |
| Lint results and models for physics in CI | `pinneapple check [paths]` (conservation, BCs, residual, units, OOD, regression; [docs](docs/guides/physics_checks_in_ci.md)) |
| Helpers that used to live in apps and examples | `pp.to_jsonable`, `pp.as_tensor`, `pp.dominant_frequency`, `pp.closed_form` (pipe flow, compressible, heat), `pp.uncertainty` (resampling, drift, tolerances), `pp.assess_drift`, `pp.propagate_tolerances`, `pp.score_against_reference`, `pp.excitation`, `pp.reporting.paperkit` ([inventory](docs/dev/LIBRARY_CONSOLIDATION.md)) |
| Electromagnetics: references, materials, solvers, physics checks of any field | `pp.em` (statics, waves, Mie, media, FEM), `pp.maxwell.maxwell_checks(grid, fields, ...)`, `pinneapple check` with `physics = "maxwell"`, `pp.em.verification.run()` ([docs](docs/guides/electromagnetics.md)) |
| Electric machines: interior-PM FEA (torque, cogging, L_d/L_q, MTPA, losses) | `pinneapple_simulation.electromagnetics.machines`: `IPMMachine(prius_2004())`, `.operating_point(I_peak=, gamma=)`, `.mtpa(I)`, `.inductances(i_d, i_q)`, `.losses(rpm, i_d, i_q)`; drive: `machines.drive.FluxMap.from_fea(m, I_max)`, `Drive(fm, R, I_max, V_max).max_torque(w)` / `.operating_point(T, w)` / `.efficiency_map(...)`, `cycle_energy(drive, Vehicle(...), *wltc_class3b())`; verification `machines.verification.run()` ([docs](docs/guides/electromagnetics.md#interior-pm-traction-motor-the-2004-toyota-prius)) |
| Run FreeCAD / CfdOF / OpenFOAM / ParaView and keep the evidence | `pinneapple_tools.automation`: `Workflow.load(...)`, `Runner(wf, folder, inputs).run(approve=[...])`, `interpret(task, catalog())` ([guide](docs/guides/engineering_gui_agent.md)) |
| Flight dynamics: 6-DOF aircraft on the vortex lattice, MIL-F-8785C turbulence, take-off, gust encounters, animated 3-D scenes | `pinneapple_simulation.flight`: `catalog.build("a320")`, `trim(ac, V, h)`, `stability_derivatives`, `simulate_takeoff(ac)` + `takeoff_metrics` + `raymer_takeoff`, `turbulence_encounter(ac, V, h, turbulence_spec(h, "severe"))`, `pratt_check`, `visuals.flight_scene(ac, hist, wind=...)`, `verification.run()` ([docs](docs/guides/flight_dynamics.md)) |
| Interlocking-brick builds: statics with stud clutch, Newton replay, self-modifying search | `pinneapple_design.bricks`: `Assembly`, `analyse_assembly`, `CantileverGoal` / `BridgeGoal` / `TowerGoal`, `evolve`, `surrogate_search`, `compare`, `visuals` ([docs](docs/guides/interlocking_bricks.md)) |
| A plain-language goal to a verified, buildable design (or the questions it needs, or a refusal saying what is missing) | `pp.design(text, answers=..., out_dir=...)`, `pinneapple design "<goal>"`, MCP `design`; templates in `pinneapple_design.intent` ([docs](docs/guides/design_from_intent.md)) |
| Forecast a noisy or near-random series without fooling yourself (predictability first, purged walk-forward, untouched test, DM tests) | `pinneapple_systems.time_series.predictability.predictability_report(y)`, `pinneapple_systems.time_series.forecast_lab` (`study.run_zoo(data, specs, Protocol(horizons))`, `presets.case(name)`) ([docs](docs/guides/near_random_forecasting.md)) |
| Predict El Niño: ENSO data (OISST, WOD thermocline, ERA5), recharge / delayed oscillators, an intermediate coupled model with checks against theory, PINNs, LIM / CNN, EnKF, the forecast_lab protocol | `pinneapple_systems.time_series.enso`: `dataset.build_dataset()`, `dataset.to_series(ds)`, `study.protocol(ds.months)`, `study.run_model(name, data, prot)`, `pipeline.solver_checks()` ([docs](docs/guides/enso_forecasting.md)) |
| Record an experiment so the Lab, the site and the Flow Studio see it (manifest, A-F grade, next steps) | `with pinneapple_lab.experiment("name", title=..., domain=...) as ex:` + `ex.log_metric` / `ex.check` (figures saved with `savefig` are recorded); `python -m pinneapple_lab scaffold <name>`, `register <folder>`, `seal --explain <item>`; format `pinneapple_lab.manifest` ([docs](docs/guides/adding_an_experiment.md)) |
| Everything from an agent tool call | the MCP server `pinneapple-mcp` ([docs](docs/guides/mcp_server.md)) |

Check the zoo before training a surrogate: a zoo model comes with its held-out errors, physics checks and validity
range. The geometry library (`pp.geometries`) is being added; use it once `hasattr(pp, "geometries")` is true.

### Pre-trained models and datasets

```python
import pinneapple as pp

print(pp.zoo.list(), pp.datasets.list())
model = pp.zoo.load("airfoil_cst_rans_mlp")           # weights checked against their SHA-256
naca2412 = dict(T0=0.169591, T1=0.144959, T2=0.131433, M0=0.024922, M1=0.071088, M2=0.058960)
out = model.predict(**naca2412, alpha_deg=[4.0])
print(out["Cl"], out["Cl_std"])                         # mean and ensemble spread (OpenFOAM: Cl 0.6625)
report = pp.zoo.validate("airfoil_cst_rans_mlp", inputs={**naca2412, "alpha_deg": [4.0]})
assert report.decisao == "APPROVED"                     # card metrics reproduced, physics checks, validity range
test = pp.datasets.load("airfoil_rans_cst_re4e6", split="test")
print(test.n_rows, test.columns[:4])
```

### Solve and compare

```python
import numpy as np
import pinneapple as pp
from pinneapple_physics.pde_environment.builder import ProblemBuilder

# laplace(u) = f on the unit square, u = 0 on the boundary, exact u = sin(pi x) sin(pi y)
spec = (ProblemBuilder("poisson").domain(x=(0, 1), y=(0, 1)).fields("u").pde("poisson")
        .bc("dirichlet", field="u", value=0.0, on="x_boundary")
        .bc("dirichlet", field="u", value=0.0, on="y_boundary").build())
problem = pp.PhysicalProblem.from_pde_spec(spec)
ctx = {"source_fn": lambda X, ctx=None: -2 * np.pi**2 * np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])}
exact = lambda X: (np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1]))[:, None]

fem = pp.solve(problem, "fem", ctx=ctx, n=24)
pinn = pp.solve(problem, "pinn", ctx=ctx, epochs=200, hidden_dim=16, n_layers=2, n_collocation=256)
table = pp.compare(problem, [fem, pinn], reference="exact", options={"exact": {"fn": exact}})
print(table)                                   # relative L2, RMSE, max error, train and inference cost
assert table.best() == "fem"
```

### Validate any result (never report a field without this)

```python
import numpy as np
import pinneapple as pp

x = y = np.linspace(0, 1, 41)
X, Y = np.meshgrid(x, y, indexing="ij")
T = np.sin(np.pi * X) * np.sinh(np.pi * Y) / np.sinh(np.pi)          # any field on a grid, from any code

card = pp.trust_card({"x": x, "y": y}, T, pde="laplace",
                     dirichlet={"x_min": 0.0, "x_max": 0.0, "y_min": 0.0,
                                "y_max": lambda P: np.sin(np.pi * P[:, 0])},
                     train_bounds={"x": (0, 1), "y": (0, 1)})
print(card)                                    # one line per check, PASS / FAIL / N/A
assert card.decision == "APPROVED"

bad = pp.trust_card({"x": x, "y": y}, 1.2 * T, pde="laplace", dirichlet={"y_max": lambda P: np.sin(np.pi * P[:, 0])})
assert bad.decision == "REJECT"                # harmonic, but the boundary condition is wrong
```

The checks a result needs, and where they come from:

| Check | How |
|---|---|
| PDE residual | `pp.trust_card(..., pde="laplace" / "poisson" / "heat", source=...)` on a grid; for a model, `pinneapple_analysis.validation.PhysicsValidator` |
| Boundary conditions | `dirichlet={face: value or callable}`; `PhysicsValidator.add_boundary_check` |
| Conservation | automatic in `trust_card` for steady problems (net boundary flux = integral of the source); `PhysicsValidator.add_conservation_check` |
| Reference comparison | `reference=` array or callable; `pp.compare(..., reference=...)` |
| Applicability (OOD) | `train_bounds=` (and `query_points=`); `pinneapple_analysis.trust.TrustGate` for model inputs |

### Equal-budget comparison

```python
import numpy as np
import pinneapple as pp

problem = pp.PhysicalProblem.from_preset("burgers_1d", nu=0.01 / np.pi)
budget = {"epochs": 100, "n_collocation": 256, "hidden_dim": 16, "n_layers": 2}
runs = [pp.solve(problem, "pinn", architecture=arch, seed=s, **budget)
        for arch in ("modified_mlp", "siren") for s in (0, 1)]
cmp = pp.compare(problem, runs, reference="analytic")
print(cmp)                                     # same points, same reference, same budget for every run
```

### Dashboard of a time-dependent result

```python
import numpy as np
import pinneapple as pp
from pinneapple_tools.visualization.dashboard import Chart, Dashboard, Series, Tile

t = np.linspace(0, 2, 12)
theta = 0.5 * np.cos(3 * t)
frames = [np.full((90, 160, 3), int(100 + 100 * (v + 0.5)), dtype=np.uint8) for v in theta]
db = Dashboard("Pendulum", t, frames, charts=[Chart("Angle", "angle [rad]", "time [s]", [Series("theta", theta)])],
               tiles=[Tile("Max angle", float(np.abs(theta).max()), "rad")])
db.to_png("pendulum.png", k=5)
```

### Validation status of a method

```python
from pinneapple_catalog import methods

status = methods.method_status()
validated = [m.id for m in methods.validated_methods("solver")]
print(len(validated), "validated solvers, e.g.", validated[:5])
assert status["S2"]["status"] == "validated"   # linear finite elements: compared with a reference in a test
```

## Project layout

| Path | Contents |
|---|---|
| `pinneapple/` | the `pp` namespace: re-exports and lazy shortcuts (`pp.solve`, `pp.trust_card`, `pp.dashboard`, ...) |
| `pinneapple_physics/` | problems and presets, the PDE compiler, `solve`/`compare`, metrics, physics audit |
| `pinneapple_neural/` | architectures (model registry), trainers, predictors |
| `pinneapple_simulation/` | numerical solvers, external solvers (OpenFOAM, CalculiX), terramechanics, multibody, vehicles |
| `pinneapple_analysis/` | validation, trust (`TrustReport`, `trust_card`), uncertainty, inverse problems |
| `pinneapple_design/` | geometry, CAD, aero (CST airfoils), design optimisation |
| `pinneapple_tools/` | visualisation (studio, dashboard), export, benchmark suite |
| `pinneapple_catalog/` | method catalog and validation status |
| `pinneapple_lab/` | experiment runner, database, trust card per run |
| `pinneapple_mcp/` | the MCP server for agents |
| `tests/`, `examples/`, `notebooks/colab/`, `docs/` | tests, runnable scripts, Colab notebooks, documentation (mkdocs) |

## Tests and lint

```bash
pytest tests/test_public_api.py tests/test_manufactured_solutions.py -q    # the must-pass core
pytest tests/test_agents_md.py tests/test_mcp_server.py -q                  # this file and the MCP server
python scripts/run_colab_notebooks.py --smoke                               # every Colab notebook, tiny sizes
pinneapple check <result dirs or models>                                    # physics linter (pinneapple_check)
python scripts/check_app_helpers.py                                         # apps/examples only compose library calls
python scripts/check_experiment_manifests.py                                # every examples/* folder has a valid experiment.json
ruff check <changed files>
```

CI (`.github/workflows/tests.yml`): public API, distribution contents, manufactured-solution physics tests, the first
example from a clean wheel, and the Colab notebooks in smoke mode. Changes go under `## [Unreleased]` in
`CHANGELOG.md`.

## Never do

- Do not put reusable logic (physics, numerics, geometry, data processing, metrics, plotting helpers, trust checks,
  IO, solvers, training loops) in `apps/`, `examples/`, `scripts/` or notebooks: it goes in a `pinneapple*` package
  with a public API and tests, and the app or example imports it. Apps keep only glue (routes, schemas, UI, deploy).
  `python scripts/check_app_helpers.py` flags new local helper modules and copies of library helpers; the inventory is
  [docs/dev/LIBRARY_CONSOLIDATION.md](docs/dev/LIBRARY_CONSOLIDATION.md).
- Do not write a raw PyTorch training loop when `pp.solve`, a trainer or a registered model already does the job.
- Do not report a result (a number, a field, a plot, "it works") without its physics checks: a trust card or the
  equivalent residual, boundary-condition, conservation and reference checks, and say which ones failed.
- Do not call a method validated unless `pinneapple_catalog` says so; "tested" is not "validated".
- Do not copy GPL (or other copyleft) code into this Apache-2.0 repository; reimplement from the paper and cite it.
- Do not invent reference values or citations; every reference number needs its source.
- Do not widen tolerances or skip a failing check to get a pass; fix the cause or report the failure.
- Do not add an `examples/*` folder without a valid `experiment.json`: wrap the script in
  `pinneapple_lab.experiment(...)` (or start from `python -m pinneapple_lab scaffold <name>`, or run
  `python -m pinneapple_lab register <folder>`). CI fails otherwise (`scripts/check_experiment_manifests.py`), and a
  metric in a manifest needs its measured value, unit and how it was measured.

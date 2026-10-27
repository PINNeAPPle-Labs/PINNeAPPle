# Getting started with PINNeAPPle

Ten runnable recipes, each with the output it actually prints. Every command below was
executed against `pinneapple 0.7.0` (Python 3.14, PyTorch 2.x, Apple MPS backend) right
before this page was written. The numbers are real; timings will differ on your hardware.

```bash
pip install pinneapple            # or: pip install -e ".[dev]" from a clone
python -c "import pinneapple as pp; pp.info()"
```

The scripts are in [`examples/quickstart/`](examples/quickstart/); each is self-contained
and run from the repository root. Everything they write goes to `runs/` (git-ignored).

| # | Recipe | Answers |
|---|--------|---------|
| 01 | [First look](#01--first-look) | What do I have installed, and what is already solved for me? |
| 02 | [Solve and compare](#02--solve-and-compare) | How do I solve a problem and score the answer? |
| 03 | [Trust card](#03--trust-card) | Can this field be used, or not? |
| 04 | [Your own problem and data](#04--your-own-problem-geometry-and-data) | How do I describe a PDE and generate training data? |
| 05 | [Inverse problem](#05--inverse-problem) | How do I recover an unknown parameter from measurements? |
| 06 | [Use the model](#06--use-the-model) | How do I predict, export and reload a trained model? |
| 07 | [Benchmark architectures](#07--benchmark-architectures) | Which model should I use for this problem? |
| 08 | [Uncertainty and validation](#08--uncertainty-and-validation) | How much should I trust a prediction? |
| 09 | [Industrial preset](#09--industrial-preset) | Is there a ready-made case close to mine? |
| 10 | [Digital twin](#10--digital-twin) | How do I put a trained model live on a sensor stream? |

---

## 01 — First look

```python
import pinneapple as pp

print(f"pinneapple {pp.__version__}")
pp.info()                                        # device, packages, missing extras

print(len(pp.list_presets()), "presets")
print(len(pp.list_models()), "architectures")
print(pp.zoo.list())                             # pre-trained models with validity ranges
print(pp.presets.names())                        # industrial cases with data sheets

spec = pp.get_preset("burgers_1d", nu=0.01)
print(spec.coords, spec.fields, spec.domain_bounds, spec.pde.kind)
```

```text
pinneapple 0.7.0
pinneapple v0.7.0
  Best device  : mps
  GPUs         : 0
  Problem presets  : 72
  Registered models: 97

  Packages:
    pinneapple_analysis.uncertainty            [OK]  — Uncertainty quantification
    pinneapple_adaptation.transfer_learning    [OK]  — Transfer learning
    pinneapple_adaptation.meta_learning        [OK]  — Meta-learning (MAML/Reptile)
    pinneapple_analysis.validation             [OK]  — Physical validation
    pinneapple_tools.model_export              [OK]  — Model export (ONNX/TorchScript)
    pinneapple_quantum                         [OK]  — Hybrid classical–quantum ML (PQM)
    pinneapple_physics.symbolic_pde            [OK]  — Symbolic PDE compiler + HardBC/PeriodicBC
    pinneapple_tools.compute_backends          [OK]  — Multi-backend (PyTorch + JAX)
    pinneapple_simulation.particle_dynamics    [OK]  — Differentiable dynamics (rigid body, MPM, SPH)
    pinneapple_worldmodel                      [OK]  — Physics world models
    pinneapple_decision                        [OK]  — Decision layer (which experiment to run next)

  Optional dependencies:
    jax          [OK]
    pyvista      [not installed]  → pip install "pinneapple[pyvista]"
    matplotlib   [OK]
    cadquery     [OK]
    trimesh      [OK]
    onnx         [not installed]  → pip install "pinneapple[export]"
    fastapi      [OK]
    mujoco       [not installed]  → pip install "pinneapple[mujoco]"

72 problem presets, e.g. ['aircraft_wing_aerodynamics', 'aircraft_wing_structural', ...]
97 model architectures, e.g. ['ae_rom_hybrid', 'afno', 'arima', 'autoencoder_2d']

pre-trained models in the zoo   : ['airfoil_cst_rans_mlp', 'heatsink_forced_mlp',
                                   'heatsink_natural_mlp', 'rover_contact_grc1_viper']
industrial cases ready to run   : ['airfoil_section_re4e6', 'eddy_current_skin_inspection',
                                   'ipm_traction_motor', 'plate_fin_heatsink_forced', ...]

burgers_1d
  coordinates : ('x', 't')
  fields      : ('u',)
  domain      : {'x': (-1.0, 1.0), 't': (0.0, 1.0)}
  PDE         : burgers {'nu': 0.01}
  conditions  : ['u_init', 'bc_left', 'bc_right']
```

`pp.info()` also lists which optional dependency is missing and the exact `pip install`
line for each — that is where to start when an import fails.

---

## 02 — Solve and compare

`pp.solve(problem, method=...)` takes a preset name, a `ProblemSpec` or a
`PhysicalProblem`. `pp.list_methods()` → `analytic`, `exact`, `external`, `fem`, `pinn`,
`reference`, plus anything you register with `pp.register_method`.

```python
import numpy as np
import pinneapple as pp

closed = pp.solve("burgers_1d", method="analytic")        # closed form, instant
pinn   = pp.solve("burgers_1d", method="pinn", epochs=800)

print(pp.compare("burgers_1d", methods=["pinn", "analytic"], reference="analytic",
                 options={"pinn": {"epochs": 800}}))

X = np.array([[-0.5, 0.0], [0.0, 0.5], [0.5, 1.0]], dtype=np.float32)
print(pinn.predict(X).ravel(), closed.predict(X).ravel())
```

```text
analytic : analytic 0.000 s
pinn     : neural 19.1 s

Comparison on 'burgers_1d_default' against 'analytic' at 4096 points
method          kind       status     train [s]  infer [us/pt]  ref sims  relL2(u)
pinn            neural     ok              19.6          0.494         0  1.504e-01
analytic        analytic   ok                 0           1.82         0  0.000e+00
pinn.predict at 3 points : [ 0.9939  0.1161 -0.3712]
analytic at the same ones: [ 1.     -0.     -0.3744]
```

`pp.compare` scores every method on the same points and reports a failing method instead
of stopping. 800 epochs gives ~15 % relative error on this problem — that is a recipe for
*how* to run it, not a converged result. Recipe 08 shows what the trust layer says about
a model in that state.

---

## 03 — Trust card

This is the part that makes the library different: any field, from any code, goes through
the checks it needs before anyone uses it — PDE residual, boundary conditions,
conservation, maximum principle, reference comparison, applicability (OOD) — and comes
back with one decision: **APPROVED / REVIEW / REJECT**.

```python
import numpy as np
import pinneapple as pp

x = y = np.linspace(0.0, 1.0, 41)
X, Y = np.meshgrid(x, y, indexing="ij")
u = np.sin(np.pi * X) * np.sinh(np.pi * Y) / np.sinh(np.pi)     # harmonic on the square

bcs = {"x_min": 0.0, "x_max": 0.0, "y_min": 0.0, "y_max": lambda P: np.sin(np.pi * P[:, 0])}

good = pp.trust_card({"x": x, "y": y}, u, pde="laplace", dirichlet=bcs,
                     train_bounds={"x": (0.0, 1.0), "y": (0.0, 1.0)})
bad  = pp.trust_card({"x": x, "y": y}, 1.2 * u, pde="laplace",
                     dirichlet={"y_max": lambda P: np.sin(np.pi * P[:, 0])})
```

```text
Trust card 'field': APPROVED
  [PASS] finite values: 0 (limit 0)
  [PASS] PDE residual: 0.000514 (limit 0.05) -- laplace(u) = f, three-point finite differences, ...
  [PASS] boundary condition on x_min: 0 (limit 0.02)
  [PASS] boundary condition on x_max: 1.22e-16 (limit 0.02)
  [PASS] boundary condition on y_min: 0 (limit 0.02)
  [PASS] boundary condition on y_max: 1.11e-16 (limit 0.02)
  [PASS] conservation (net boundary flux = integral of the source): 0.00229 (limit 0.05)
  [PASS] maximum principle (no interior extremum): 0 (limit 0.01)
  [N/A ] reference (relative L2): -- no reference given
  [PASS] applicability: query inside the training bounds (OOD): 1 (limit 1) -- 0 of 1681 points outside

Trust card 'field': REJECT
  [PASS] finite values: 0 (limit 0)
  [PASS] PDE residual: 0.000514 (limit 0.05) -- laplace(u) = f, ...
  [FAIL] boundary condition on y_max: 0.167 (limit 0.02) -- Dirichlet, max |u - g| / range of u
  [PASS] conservation (net boundary flux = integral of the source): 0.00229 (limit 0.05)
  [PASS] maximum principle (no interior extremum): 0 (limit 0.01)
  [N/A ] reference (relative L2): -- no reference given
  [N/A ] applicability: query inside the training bounds (OOD): -- no training bounds given

correct field: APPROVED
scaled field : REJECT
```

A check without its input is reported `N/A` with the reason, never silently dropped, and
a field that is harmonic but scaled 20 % still gets `REJECT` — the residual check alone
would have passed it.

---

## 04 — Your own problem, geometry and data

`ProblemBuilder` describes a PDE as data; `CSG*` shapes describe a geometry;
`generate_pinn_dataset` turns both into collocation / boundary / initial batches.

Problem: `u_t = alpha u_xx` on `[0,1]²`, `u(x,0) = sin(pi x)`, `u = 0` at `x = 0, 1`;
exact `u(x,t) = exp(-alpha pi² t) sin(pi x)`.

```python
from pinneapple_physics import ProblemBuilder, metrics
from pinneapple_simulation.numerical_solvers import generate_pinn_dataset, register_all
from pinneapple_design.geometry.csg import CSGCircle, CSGDifference

ALPHA = 0.05
spec = (
    ProblemBuilder("heat_1d_custom")
    .domain(x=(0.0, 1.0), t=(0.0, 1.0))
    .fields("u")
    .pde("heat_equation", alpha=ALPHA)
    .ic(field="u", fn=lambda X: np.sin(np.pi * X[:, 0:1]))
    .bc(kind="dirichlet", field="u", value=0.0, on="x_boundary")
    .build()
)
sol = pp.solve(spec, method="pinn", epochs=2000)

batch = generate_pinn_dataset(spec, n_col=4000, n_bc=400, n_ic=400, seed=0)

for r in (0.10, 0.20, 0.30):                       # geometry is a parameter
    shape = CSGDifference(CSGCircle(0.0, 0.0, 0.5), CSGCircle(0.0, 0.0, r))
    shape.sample_interior(2000, seed=0)
    shape.sample_boundary(200, seed=0)
```

```text
problem    : heat_1d_custom | coordinates: ('x', 't')
conditions : [('ic_0', 'initial'), ('bc_dirichlet_1', 'dirichlet')]
solved in  : 65.7 s
relative L2: 2.705e-04
dataset    : {'x_col': (4000, 2), 'x_bc': (400, 2), 'y_bc': (400, 1), 'x_ic': (400, 2), 'y_ic': (400, 1)}
  inner r=0.10  area=0.7540  interior (2000, 2)  boundary (200, 2)
  inner r=0.20  area=0.6597  interior (2000, 2)  boundary (200, 2)
  inner r=0.30  area=0.5027  interior (2000, 2)  boundary (200, 2)
```

Two things that cost an hour if you hit them blind:

* `.bc(..., on="boundary")` (the default) creates a **tag** selector — it names regions of
  a real mesh, and `pp.solve` refuses it when all you have is a box, with an explicit
  `TagConditionsUnresolved`. Use `on="x_boundary"`, `on="x_min"`, `on=("x", "max")` or a
  callable.
* `pde(...)` takes the *compiler's* kind (`heat_equation`, `poisson`, `burgers`,
  `navier_stokes_incompressible`, ...). `get_preset(name).pde.kind` is the fastest way to
  find the right string for a preset you want to imitate.
* sample from the **shape** (`shape.sample_interior`) — `Domain(bounds, shape=...)` returns
  points over the bounding box and ignores the CSG. The script asserts
  `shape.contains(points).all()` so a regression fails loudly.

---

## 05 — Inverse problem

The library ships inverse-problem presets ("the Inverse Heat Lab"): a real sensor layout,
a forward model, and a known true value to check the recovered answer against.

```python
import pinneapple_analysis.inverse_problems as pinv

preset  = pinv.get_preset("fin_h")              # 5 thermocouples on a pin fin
built   = preset.build()
problem = pinv.InverseProblem(model=built["model"], observations=built["observations"],
                              parameters=built["parameters"], validation=built["validation"],
                              name="fin_h")

result = pinv.solve_inverse(problem, "least_squares")
comparison = pinv.compare_methods(problem, methods=["least_squares", "bayesian"])
```

```text
problem : Pin fin: h from five thermocouples
unknown : ['h', 'k', 'T_base', 'T_air']
methods : {'least_squares': True, 'adjoint': False, 'autodiff': True, 'bayesian': True, 'pinn': True}

h = 24.97 W/m²K  (true 25.0, error -0.11 %)  95% CI [24.15, 25.82]
truth inside the CI   : True
held-out sensors RMSE : 0.410 degC

method                fit RMSE  val RMSE   seconds  truth
least_squares           0.1923    0.4099    0.0047  True
bayesian                0.1924    0.4107    0.0920  True
best: least_squares
```

Each preset carries a **lesson**: `fin_h_k` shows two parameters that no amount of
thermocouples can separate, `plate_h_table` shows the interpolation error a table of
solver runs hides, `arrhenius_two_T` shows the measurement that removes an ambiguity.
`pinv.list_presets()` lists all eleven with their summary.

---

## 06 — Use the model

```python
from pinneapple_neural import build_model
from pinneapple_neural.predictor import infer_on_grid_2d
from pinneapple_tools.model_export import export_csv, export_npz

sol = pp.solve("burgers_1d", method="pinn", epochs=400,
               architecture="vanilla_pinn", hidden_dim=64, n_layers=3)
result = infer_on_grid_2d(sol.model, x_range=(-1.0, 1.0), y_range=(0.0, 1.0),
                          nx=120, ny=80, field_names=["u"])
export_csv(coords, values, "runs/quickstart_field.csv")
sol.model.save_checkpoint("runs/quickstart_model.pt", metadata={"problem": "burgers_1d"})

reloaded = build_model("vanilla_pinn", in_dim=2, out_dim=1, hidden_dim=64, n_layers=3)
reloaded.load_state_dict(torch.load("runs/quickstart_model.pt", weights_only=False)["state_dict"])
```

```text
model: VanillaPINN | {'backend': 'pinneapple.pinn', 'architecture': 'vanilla_pinn', 'epochs': 400,
                      'n_collocation': 2048, 'seed': 0, 'n_parameters': 8577}
grid         : (120, 80)
field arrays : {'u': (120, 80)}
saved        : runs/quickstart_field.csv, runs/quickstart_field.npz
trust card   : REJECT
checkpoint   : ['class_name', 'metadata', 'state_dict'] | pinneapple_neural.architectures.pinns.vanilla.VanillaPINN
reload equal : True
```

`trust card : REJECT` is the honest answer for a 400-epoch field: the card compares it
with the closed form and refuses it. Export a field only after the card approves it —
that is the rule the rest of this library follows.

A checkpoint stores `state_dict`, the dotted `class_name` and your metadata. Rebuild the
**same architecture with the same kwargs** (`hidden_dim`, `n_layers`) and call
`load_state_dict`; the registry is keyed by short names (`vanilla_pinn`), not the dotted
path. ONNX export (`export_onnx`) needs `pip install "pinneapple[export]"`; serve a model
over HTTP with `pinneapple_tools.model_services`.

---

## 07 — Benchmark architectures

```python
import pinneapple_arena as ar

config = ar.ArenaConfig.from_dict({
    "problem": {"name": "burgers_1d", "params": {"nu": 0.01}},
    "models": [
        {"name": "VanillaPINN", "type": "vanilla_pinn", "training": {"epochs": 600, "lr": 1e-3}},
        {"name": "SIREN",       "type": "siren",        "training": {"epochs": 600, "lr": 1e-3}},
        {"name": "HashGridMLP", "type": "hash_grid_mlp","training": {"epochs": 600, "lr": 1e-3}},
    ],
    "output": {"dir": "runs/quickstart_arena", "prefix": "guide"},
})
arena = ar.Arena(config).run()
arena.compare()

for run in arena.results:
    print(run["name"], run["metrics"])
```

```text
======================================================================
  ARENA RESULTS  >>  burgers_1d
======================================================================
  Model                   L2-u         rel-u       Time(s)  Physics-Res
----------------------------------------------------------------------
  VanillaPINN             2.328e-01    3.924e-01      14.3  1.070e-01
  SIREN                   1.879e-01    3.166e-01      10.8  4.935e-06
  HashGridMLP             2.666e-01    4.492e-01      58.1  n/a
======================================================================

VanillaPINN    L2 2.328e-01   rel 3.924e-01   Linf 7.117e-01
SIREN          L2 1.879e-01   rel 3.166e-01   Linf 6.451e-01
HashGridMLP    L2 2.666e-01   rel 4.492e-01   Linf 1.445e+00
```

SIREN wins *and* drives the PDE residual down by four orders of magnitude, while
HashGridMLP is the most accurate on the supervised L2 in some runs and has no physics
residual at all (`n/a` — it does not support a physics loss). Three figures (fields, loss
curves, metric bars) land in the output directory.

Related: `pinneapple_arena.nas.search_architecture` (architecture + hyperparameters as one
Optuna search space), `run_parallel_sweep`, and `Arena.run_decision(budget=2)` where the
decision layer picks *which* models to train next.

---

## 08 — Uncertainty and validation

Two different questions: how uncertain is a prediction, and does it satisfy the physics.

```python
from pinneapple_analysis.uncertainty import uq_predict

sol = pp.solve("burgers_1d", method="pinn", epochs=800)

X = rng.uniform([-1.0, 0.0], [1.0, 1.0], size=(2000, 2)).astype(np.float32)
uq = uq_predict(sol.model, torch.tensor(X), method="mc_dropout", n_samples=50, dropout_p=0.1, seed=0)
lo, hi = uq.confidence_interval(alpha=0.05)

print(pp.validate(sol.model, {"pinn": "burgers_1d"}))
```

```text
model: ModifiedMLP | trained in 33.8 s
prediction std     : 0.1177
95% interval width : 0.4612
least certain point: x=-0.500, t=0.013, std=0.1859

validation of modified_mlp on PINN, burgers_1d (standard): fail: PDE residual at points not used
in training; error against an exact solution
  code_verification      level 0  (no evidence)
  solution_verification  level 1  (evidence contradicts the model (a critical check failed))
  validation             level 3  (passes quantitative checks)
  input_pedigree         level 0  (no evidence)
  uncertainty            level 0  (no evidence)
  robustness             level 0  (no evidence)
  applicability          level 0  (no evidence)
  NOT-APPLICABLE  method.status: no catalog method id given (validate(..., method='S2'))
  FAIL            pinn.residual: relative residual at fresh points: RMS 1.11e-01 (<= 0.05), max 1.19e+00 (<= 0.3, adaptive points included)
  PASS            pinn.trivial: spread and gradient energy 0.84 x the exact solution's (>= 0.2)
  PASS            pinn.causality: residual in the last quarter 1.2e-01, first quarter 8.5e-02 (late/early 1.5, <= 10); error early 6.4e-02 -> late 7.7e-02
  PASS            pinn.boundary: boundary condition error 7.89e-04 of the solution scale (<= 0.01)
  PASS            pinn.initial: initial condition error 3.54e-03 of the solution scale (<= 0.01)
  PASS            pinn.spectrum: high-wavenumber energy -4% relative to the exact solution (within +/-25%)
  FAIL            pinn.exact: relative L2 error 1.23e-01 (<= 0.02), max 4.39e-01 of the scale
```

The report **fails on purpose**: at 800 epochs the loss curve looks converged and the
boundary and initial conditions are met to 1e-3, yet the PDE residual at points the model
never saw is 1.1e-1 and the error against the exact solution is 12 %. That gap between
"the loss went down" and "the model is right" is exactly what the trust layer is for.
Train longer (or use causal training — `CausalPINNTrainer`) and run it again.

`pp.validate` also accepts the `level` (`"quick"` / `"standard"` / `"thorough"`), a
`reference=`, and the `validate` entry point works on zoo models, PyTorch modules, ONNX
files and plain callables.

---

## 09 — Industrial preset

`pp.presets` holds industrial cases with a data sheet, sourced defaults, a solver,
references with their measured error, a surrogate, inversion and optimisation.

```python
import pinneapple as pp

preset = pp.presets.load("water_hammer_valve_closure")
print(preset.card()["title"])

print(preset.run(t_close=10.0))          # slow closure: inside the pipe rating
fast = preset.run(t_close=2.0)           # fast closure: too fast
print(fast.decision["verdict"], fast.trust.decisao)

print(preset.validate().get("all_pass")) # against Joukowsky and rigid-column theory
```

```text
case    : Golpe de aríete no fechamento de válvula de adutora de aço
sector  : Fluidos / saneamento e óleo e gás

water_hammer_valve_closure (solver): OK | trust APPROVED
  inner_diameter = 0.48894
  wave_speed = 1211.84
  period = 1.98046
  reynolds = 730629
  friction_factor = 0.0137365
  max_head = 82.2931
  surge_head = 26.1607
  p_max = 805567
  p_min_abs = 471087
  joukowsky_head = 185.36
  michaud_head = 36.7098
  h_allow = 163.449
  t_close_safe = 3.37008

t_close=2.0 s -> NOT OK | trust: REVIEW

preset.validate(): True
```

`pp.presets.names()` lists the rest (airfoil section, IPM traction motor, PCB junction
temperature, ASME pressure vessel, pump NPSH cavitation, shell-and-tube cooler, ...).
Each result carries `.decision`, `.trust`, `.outputs` and `.save(dir)`.

---

## 10 — Digital twin

A digital twin is a trained surrogate plus a running loop: sensor data in, field
re-evaluated, observations compared with predictions, anomalies flagged.

```python
from pinneapple_systems.digital_twin import build_digital_twin
from pinneapple_systems.digital_twin.io import MockStream

sol = pp.solve("burgers_1d", method="pinn", epochs=400)
twin = build_digital_twin(sol.model, field_names=["u"], coord_names=["x", "t"],
                          update_interval=0.1, anomaly_z_threshold=3.0)
twin.set_domain_coords({"x": ..., "t": ...})
twin.add_stream(MockStream("probe", ["u"], sensor, coords={"x": 0.0, "t": 0.0}))

with twin:
    time.sleep(3.0)

print(twin.anomaly_monitor.recent_events())
```

```text
[ANOMALY/ZScore] probe/u: z=3.75 > threshold=3.0
surrogate: ModifiedMLP
field mean      : 0.0175
history records : 29
anomalies       : 1
  u: observed 5.106, predicted 0.000, score 3.75 (zscore)
```

`MockStream` needs no broker. In a real deployment swap it for `MQTTStream`,
`KafkaStream`, `OPCUAStream`, `ModbusStream`, `HTTPPollStream` or `FileWatchStream` — the
interface is the same. Add `assimilation="ekf"` or `"enkf"` to fuse observations with the
model instead of only comparing them.

---

## Where to go next

| I want to... | Start here |
|---|---|
| A working script for a technique | [`templates/`](templates/) — 35 templates (causal training, DDP, HardBC, transfer learning, UQ, export, serving, RANS, 3-D, JAX ...) |
| A technique applied end to end | [`examples/`](examples/) — grouped by area (`arena_pipelines`, `numerical_solvers`, `data_pipeline`, `validation`, `use_cases`, ...) |
| A pre-trained model or dataset | `pp.zoo.list()`, `pp.zoo.load(name).predict(...)`, `pp.datasets.list()` |
| The design of the library | [`docs/core_concepts/`](docs/core_concepts/) and [`docs/architecture/package_layers.md`](docs/architecture/package_layers.md) |
| A classical solver instead of a network | `pinneapple_simulation.numerical_solvers` — FDM, FEM, FVM, spectral, LBM, SPH, X-TFC, plus OpenFOAM / FEniCS / CalculiX bridges |
| Operator learning instead of pointwise | `deeponet`, `fno`, `mesh_graph_net`, `uno`, `pino` in `pp.list_models()` |
| To know which methods are actually validated | `pinneapple_catalog.methods.validated_methods()`, [`docs/dev/CATALOGO_METODOS.md`](docs/dev/CATALOGO_METODOS.md) |
| To know what does *not* work | [Known limits](#known-limits) and `CHANGELOG.md` → "Known issues" |
| Physics checks in CI | `pinneapple check <dirs>` ([guide](docs/guides/physics_checks_in_ci.md)) |

---

## Known limits

Found while writing and running these recipes on 0.7.0. None blocks the workflows above;
all of them cost an hour if you meet them blind.

* **Presets with tag selectors cannot be solved from `domain_bounds` alone.**
  `pp.solve("poisson_2d", method="pinn")` raises `TagConditionsUnresolved`, because the
  preset's boundary condition names a mesh region. Build the batch from a real geometry
  (`STLDomainBatchBuilder`) or select faces explicitly (`on="x_min"`). This is a
  deliberate refusal: fabricating the mask would produce a physically invalid condition.
* **`pp.validate(net, {"pinn": ...})` wants a preset name**, not a bare `ProblemSpec`
  (`AttributeError: 'ProblemSpec' object has no attribute 'mesh'`). For a custom problem,
  use `pp.trust_card` on the field instead.
* **`Domain(bounds, shape=...)` samples the bounding box**, not the CSG shape — use
  `shape.sample_interior` / `shape.sample_boundary`.
* **A checkpoint needs the same constructor kwargs** (`hidden_dim`, `n_layers`) to reload;
  `class_name` is recorded but is the dotted path, not the registry key.
* **`Arena` cannot build `ModifiedMLP`** in 0.7.0: its factory passes `hidden=`, which
  `ModifiedMLP.__init__` does not accept (it takes `hidden_dim=` in `build_model`). Other
  families build fine.
* **`industrial_furnace_thermal` does not compile** (its hot face combines convection and
  radiation, which `thermal_bc` does not support), and the `reference` method for
  `burgers_1d` is refused because the FDM grid covers `x` only — use `method="analytic"`.

---

## Running the recipes

```bash
git clone https://github.com/PINNeAPPle-Labs/PINNeAPPle.git
cd PINNeAPPle
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

for f in examples/quickstart/*.py; do python "$f"; done
```

Recipes 02, 04, 07, 08 and 10 train for 20–70 s each on CPU; the rest finish in seconds.

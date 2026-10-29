"""Build notebooks/colab/06_quickstart_ten_recipes.ipynb from the recipes in GETTING_STARTED.md.

Kept as a script so the ten recipes are reviewable as text and the notebook can be rebuilt
after an edit. **It writes the notebook without outputs** — the committed copy carries its
outputs (``metadata.pinneapple.keep_outputs``), so after running this you must execute it:

    python tools/build_quickstart_notebook.py
    python scripts/run_colab_notebooks.py 06 --out /tmp/nb && \
      cp /tmp/nb/06_quickstart_ten_recipes.ipynb notebooks/colab/

Not wired into CI: the notebook is committed, and CI checks it by running it.
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notebooks", "colab", "06_quickstart_ten_recipes.ipynb")

BADGE = ("[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
         "(https://colab.research.google.com/github/PINNeAPPle-Labs/PINNeAPPle/blob/main/"
         "notebooks/colab/06_quickstart_ten_recipes.ipynb)")

cells: list[tuple[str, str]] = []


def md(text: str) -> None:
    cells.append(("markdown", text.strip("\n")))
def code(text: str) -> None:
    cells.append(("code", text.strip("\n")))


# --------------------------------------------------------------------------- 0
md(f"""# Thirteen recipes: from `pip install` to a live digital twin

{BADGE}

This is the notebook version of [`GETTING_STARTED.md`](../../GETTING_STARTED.md): the same thirteen recipes, in the order
a new user needs them, each one printing the numbers it produced. Every cell below was executed before the notebook
was committed, and the outputs you see are what it printed.

1. what is installed, and what is already solved for you
2. solve a problem and compare methods
3. decide whether a field can be trusted (the trust card)
4. your own problem, your own geometry, and the data to train on
5. an inverse problem: recover a physical parameter from measurements
6. use the model: grid prediction, export, checkpoint
7. benchmark architectures on the same problem
8. uncertainty and validation
9. a ready-made industrial case
10. a digital twin on a sensor stream
11. 1-D, 2-D, 3-D and 4-D (animated) visualisation
12. one sentence -> quantities -> 3-D geometry -> solver -> surrogate -> twin
13. geometry variants -> benchmark -> inverse model -> verification

Run it top to bottom (Runtime > Run all). On a free Colab CPU it takes a few minutes; on a T4 GPU, less.""")

# --------------------------------------------------------------------------- 1
code('''# Setup: on Colab (or wherever PINNeAPPle is missing) install it from PyPI, then pick the device.
import importlib.util, os, subprocess, sys

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB or importlib.util.find_spec("pinneapple") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pinneapple>=0.7.0"], check=True)

import numpy as np
import torch
import pinneapple as pp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PINNeAPPle {pp.__version__} on {DEVICE}"
      + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else " (no GPU: slower, still fine)"))''')

# --------------------------------------------------------------------------- 2
code('''# Parameters. SMOKE=True shrinks every run so the notebook finishes in seconds (CI runs it that way).
SMOKE = os.environ.get("PINNEAPPLE_NB_SMOKE", "0") == "1"

EPOCHS_SHORT = 100 if SMOKE else 400      # recipe 6 and 10
EPOCHS_PINN = 150 if SMOKE else 800       # recipe 2 and 8
EPOCHS_LONG = 300 if SMOKE else 2000      # recipe 4
ARENA_EPOCHS = 50 if SMOKE else 600       # recipe 7
TRAINER_EPOCHS = 12 if SMOKE else 150     # recipe 12
VARIANT_EPOCHS = 10 if SMOKE else 80      # recipe 13
N_VARIANTS = 2 if SMOKE else 4            # recipe 13
GRID = 21 if SMOKE else 41                # points per axis of a checking grid
print(f"SMOKE={SMOKE}  epochs: short={EPOCHS_SHORT} pinn={EPOCHS_PINN} long={EPOCHS_LONG} arena={ARENA_EPOCHS}")''')

# --------------------------------------------------------------------------- 3
md("""## 1. What is installed

`pp.info()` reports the device, which packages import, and which optional dependency is missing — with the exact
`pip install` line for each. `pp.zoo` and `pp.presets` are the shortest paths to a result: a pre-trained model with a
validity range, or a full industrial case with a data sheet.""")

code('''print(f"pinneapple {pp.__version__}")
pp.info()

print(f"\\n{len(pp.list_presets())} problem presets, e.g. {pp.list_presets()[:4]}")
print(f"{len(pp.list_models())} model architectures, e.g. {pp.list_models()[:4]}")
print(f"pre-trained models in the zoo : {pp.zoo.list()}")
print(f"industrial cases ready to run : {pp.presets.names()[:6]}")

spec = pp.get_preset("burgers_1d", nu=0.01)
print("\\nburgers_1d")
print("  coordinates :", spec.coords)
print("  fields      :", spec.fields)
print("  domain      :", spec.domain_bounds)
print("  PDE         :", spec.pde.kind, spec.pde.params)
print("  conditions  :", [c.name for c in spec.conditions])''')

# --------------------------------------------------------------------------- 3
md("""## 2. Solve a problem and compare methods

`pp.solve(problem, method=...)` takes a preset name, a `ProblemSpec` or a `PhysicalProblem`. `pp.list_methods()` gives
`analytic`, `exact`, `external`, `fem`, `pinn`, `reference` — and anything you register with `pp.register_method`.
`pp.compare` scores every method on the same points.""")

code('''PROBLEM = "burgers_1d"

closed = pp.solve(PROBLEM, method="analytic")             # closed form, instant
pinn = pp.solve(PROBLEM, method="pinn", epochs=EPOCHS_PINN)
print("analytic :", closed.metadata()["kind"], f"{closed.metadata()['wall_time_s']:.3f} s")
print("pinn     :", pinn.metadata()["kind"], f"{pinn.metadata()['wall_time_s']:.1f} s")

print()
print(pp.compare(PROBLEM, methods=["pinn", "analytic"], reference="analytic",
                 options={"pinn": {"epochs": EPOCHS_PINN}}))

X = np.array([[-0.5, 0.0], [0.0, 0.5], [0.5, 1.0]], dtype=np.float32)
print("pinn.predict at 3 points :", np.round(pinn.predict(X).ravel(), 4))
print("analytic at the same ones:", np.round(closed.predict(X).ravel(), 4))''')

md('''**Reading the table.** 800 epochs on this problem leaves ~15 % relative error. That is a recipe for *how* to run
it, not a converged result — recipe 8 shows what the trust layer says about a model in that state.''')

# --------------------------------------------------------------------------- 4
md("""## 3. The trust card: can this field be used?

Any field, from any code, goes through the checks it needs — PDE residual, boundary conditions, conservation,
maximum principle, reference comparison, applicability — and comes back with one decision:
**APPROVED / REVIEW / REJECT**. The second field below is the first one scaled by 1.2: still harmonic, still wrong.""")

code('''x = y = np.linspace(0.0, 1.0, GRID)
X, Y = np.meshgrid(x, y, indexing="ij")
u = np.sin(np.pi * X) * np.sinh(np.pi * Y) / np.sinh(np.pi)      # harmonic on the square

bcs = {"x_min": 0.0, "x_max": 0.0, "y_min": 0.0, "y_max": lambda P: np.sin(np.pi * P[:, 0])}

good = pp.trust_card({"x": x, "y": y}, u, pde="laplace", dirichlet=bcs,
                     train_bounds={"x": (0.0, 1.0), "y": (0.0, 1.0)})
print(good)

bad = pp.trust_card({"x": x, "y": y}, 1.2 * u, pde="laplace",
                    dirichlet={"y_max": lambda P: np.sin(np.pi * P[:, 0])})
print()
print(bad)

print("\\ncorrect field:", good.decision, "| scaled field:", bad.decision)
assert good.decision == "APPROVED" and bad.decision == "REJECT"''')

# --------------------------------------------------------------------------- 5
md("""## 4. Your own problem, your own geometry, and the data to train on

`ProblemBuilder` writes the PDE as data; `CSG*` shapes describe a geometry; `generate_pinn_dataset` turns both into
collocation / boundary / initial batches. Problem: $u_t = \\alpha u_{xx}$ on $[0,1]^2$ with
$u(x,0)=\\sin(\\pi x)$ and $u=0$ at $x=0,1$, whose exact solution is $u=e^{-\\alpha\\pi^2 t}\\sin(\\pi x)$.""")

code('''from pinneapple_physics import ProblemBuilder, metrics
from pinneapple_simulation.numerical_solvers import generate_pinn_dataset, register_all
from pinneapple_design.geometry.csg import CSGCircle, CSGDifference

ALPHA = 0.05

def exact(X):
    return np.exp(-ALPHA * np.pi**2 * X[:, 1:2]) * np.sin(np.pi * X[:, 0:1])

spec_heat = (
    ProblemBuilder("heat_1d_custom")
    .domain(x=(0.0, 1.0), t=(0.0, 1.0))
    .fields("u")
    .pde("heat_equation", alpha=ALPHA)
    .ic(field="u", fn=lambda X: np.sin(np.pi * X[:, 0:1]))
    .bc(kind="dirichlet", field="u", value=0.0, on="x_boundary")
    .build()
)
print("problem    :", spec_heat.name or spec_heat.problem_id, "| coordinates:", spec_heat.coords)
print("conditions :", [(c.name, c.kind) for c in spec_heat.conditions])

sol_heat = pp.solve(spec_heat, method="pinn", epochs=EPOCHS_LONG)
Xq = np.random.default_rng(0).uniform([0.0, 0.0], [1.0, 1.0], size=(5000, 2)).astype(np.float32)
rel = float(np.ravel(metrics.relative_l2(sol_heat.predict(Xq), exact(Xq)))[0])
print(f"solved in  : {sol_heat.metadata()['wall_time_s']:.1f} s")
print("relative L2:", f"{rel:.3e}")''')

code('''# The same spec produces a training dataset — collocation, boundary and initial batches.
register_all()
batch = generate_pinn_dataset(spec_heat, n_col=4000, n_bc=400, n_ic=400, seed=0)
print("dataset    :", {k: np.shape(v) for k, v in batch.items() if hasattr(v, "shape")})

# Geometry is data too: a parameterised annulus, sampled by the shape (not by the bounding box).
for r in (0.10, 0.20, 0.30):
    shape = CSGDifference(CSGCircle(0.0, 0.0, 0.5), CSGCircle(0.0, 0.0, r))
    interior = shape.sample_interior(2000, seed=0)
    boundary = shape.sample_boundary(200, seed=0)
    print(f"  inner r={r:.2f}  area={np.pi * (0.5**2 - r**2):.4f}  "
          f"interior {interior.shape}  boundary {boundary.shape}")
    assert bool(shape.contains(interior).all())     # every point really is inside''')

md('''**Two limits that cost an hour if you meet them blind.**

* `.bc(..., on="boundary")` is a **tag** selector: it names a mesh region. With only a box, `pp.solve` refuses it
  with `TagConditionsUnresolved`. Use `on="x_boundary"`, `on="x_min"`, `on=("x", "max")` or a callable.
* `pde(...)` takes the *compiler's* kind (`heat_equation`, `poisson`, `burgers`, ...), not the preset name.
  `get_preset(name).pde.kind` gives the right string for a preset you want to imitate.
* sample from the **shape** (`shape.sample_interior`) — `Domain(bounds, shape=...)` returns points over the
  bounding box and ignores the CSG ([#938](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/938)).''')

# --------------------------------------------------------------------------- 6
md("""## 5. Inverse problem: recover a parameter from measurements

The library ships inverse-problem presets (the "Inverse Heat Lab"): a real sensor layout, a forward model, and a known
true answer to check the recovered value against. Here: five thermocouples on a stainless pin fin, unknown
convection coefficient `h`.""")

code('''import pinneapple_analysis.inverse_problems as pinv

preset = pinv.get_preset("fin_h")
built = preset.build()
problem = pinv.InverseProblem(model=built["model"], observations=built["observations"],
                              parameters=built["parameters"], validation=built["validation"], name="fin_h")
print("problem :", preset.title)
print("unknown :", [p.name for p in built["parameters"]])
print("methods :", {k: v["available"] for k, v in pinv.available_methods(problem).items()})

result = pinv.solve_inverse(problem, "least_squares")
for p in result["parameters"]:
    lo, hi = p["interval95"]
    print(f"\\n{p['name']} = {p['estimate']:.2f} {p['unit']}  (true {p['truth']}, "
          f"error {p['error_pct']:+.2f} %)  95% CI [{lo:.2f}, {hi:.2f}]")
print("truth inside the CI   :", result["truth_recovered"])
print("held-out sensors RMSE :", f"{result['validation']['rmse']:.3f} degC")''')

code('''comparison = pinv.compare_methods(problem, methods=["least_squares", "bayesian"])
print(f"{'method':<20}{'fit RMSE':>10}{'val RMSE':>10}{'seconds':>10}  truth")
for row in comparison["table"]:
    print(f"{row['method']:<20}{row['fit_rmse']:>10.4f}{row['validation_rmse']:>10.4f}"
          f"{row['seconds']:>10.4f}  {row['truth_recovered']}")
print("best:", comparison["best"])''')

code('''# Each preset carries a lesson. pinv.list_presets() lists all eleven:
#   fin_h_k          two parameters no amount of thermocouples can separate (not identifiable)
#   plate_h_table    the interpolation error a table of solver runs hides
#   arrhenius_two_T  the measurement that removes an ambiguity
print([p["id"] for p in pinv.list_presets()])''')

# --------------------------------------------------------------------------- 7
md("""## 6. Use the model: predict on a grid, export, reload

A checkpoint stores `state_dict`, the dotted `class_name` and your metadata. Rebuild the **same architecture with the
same kwargs** and load the weights — the registry is keyed by short names (`vanilla_pinn`), not the dotted path.""")

code('''from pinneapple_neural import build_model
from pinneapple_neural.predictor import infer_on_grid_2d
from pinneapple_tools.model_export import export_csv, export_npz

ARCH, HIDDEN, LAYERS = "vanilla_pinn", 64, 3

sol = pp.solve("burgers_1d", method="pinn", epochs=EPOCHS_SHORT,
               architecture=ARCH, hidden_dim=HIDDEN, n_layers=LAYERS)
print("model:", type(sol.model).__name__, "|", sol.metadata()["info"])

result = infer_on_grid_2d(sol.model, x_range=(-1.0, 1.0), y_range=(0.0, 1.0),
                          nx=120, ny=80, field_names=["u"])
print("grid         :", result.grid_shape)
print("field arrays :", {k: np.asarray(v).shape for k, v in result.fields.items()})

XX, YY = np.meshgrid(result.coords["x"], result.coords["y"], indexing="ij")
coords = np.column_stack([XX.ravel(), YY.ravel()])
values = np.asarray(result.fields["u"])''')

code('''export_csv(coords, values.ravel(), "quickstart_field.csv", coord_names=["x", "t"], field_names=["u"])
export_npz(coords, values.ravel(), "quickstart_field.npz", coord_names=["x", "t"], field_names=["u"])
print("saved        : quickstart_field.csv, quickstart_field.npz")

# Check the field you are about to hand over. REJECT here is the honest answer for a short run:
# export a field only after the card approves it.
reference = pp.solve("burgers_1d", method="analytic").predict(coords).reshape(values.shape)
card = pp.trust_card({"x": result.coords["x"], "t": result.coords["y"]}, values,
                     reference=reference, train_bounds={"x": (-1.0, 1.0), "t": (0.0, 1.0)})
print("trust card   :", card.decision)''')

code('''sol.model.save_checkpoint("quickstart_model.pt", metadata={"problem": "burgers_1d"})
checkpoint = torch.load("quickstart_model.pt", weights_only=False)
print("checkpoint   :", sorted(checkpoint.keys()), "|", checkpoint["class_name"])

reloaded = build_model(ARCH, in_dim=2, out_dim=1, hidden_dim=HIDDEN, n_layers=LAYERS)
reloaded.load_state_dict(checkpoint["state_dict"])
reloaded.eval()
with torch.no_grad():
    a = sol.model(torch.tensor(coords[:4], dtype=torch.float32)).y
    b = reloaded(torch.tensor(coords[:4], dtype=torch.float32)).y
print("reload equal :", bool(torch.allclose(a, b, atol=1e-6)))''')

# --------------------------------------------------------------------------- 8
md("""## 7. Benchmark architectures on the same problem

The Arena trains several registered models on one problem with the same budget, scores them against the same
reference and ranks them — three figures (fields, loss curves, metric bars) are written next to the notebook.""")

code('''import pinneapple_arena as ar

config = ar.ArenaConfig.from_dict({
    "problem": {"name": "burgers_1d", "params": {"nu": 0.01}},
    "models": [
        {"name": "VanillaPINN", "type": "vanilla_pinn", "training": {"epochs": ARENA_EPOCHS, "lr": 1e-3}},
        {"name": "SIREN", "type": "siren", "training": {"epochs": ARENA_EPOCHS, "lr": 1e-3}},
        {"name": "HashGridMLP", "type": "hash_grid_mlp", "training": {"epochs": ARENA_EPOCHS, "lr": 1e-3}},
    ],
    "output": {"dir": "arena", "prefix": "quickstart"},
})

arena = ar.Arena(config).run()
arena.compare()

for run in arena.results:
    m = run["metrics"]
    print(f"{run['name']:<14} L2 {m['L2_u']:.3e}   rel {m['rel_u']:.3e}   Linf {m['Linf_u']:.3e}")''')

md('''**Reading the table.** SIREN is both the most accurate and the one that drives the PDE residual down by orders of
magnitude. HashGridMLP shows `n/a` under Physics-Res because that family does not support a physics loss — it is a
supervised model.''')

# --------------------------------------------------------------------------- 9
md("""## 8. Uncertainty and validation

Two different questions: how uncertain is a prediction (`uq_predict` with MC Dropout), and does it satisfy the
physics (`pp.validate`, which re-checks at points the model never saw). The report below **fails on purpose** — a
converged-looking loss curve is not a validated model.""")

code('''from pinneapple_analysis.uncertainty import uq_predict

sol_mc = pp.solve("burgers_1d", method="pinn", epochs=EPOCHS_PINN)
print("model:", type(sol_mc.model).__name__, "| trained in", f"{sol_mc.metadata()['wall_time_s']:.1f} s")

rng = np.random.default_rng(0)
Xm = rng.uniform([-1.0, 0.0], [1.0, 1.0], size=(2000, 2)).astype(np.float32)
uq = uq_predict(sol_mc.model, torch.tensor(Xm), method="mc_dropout", n_samples=50, dropout_p=0.1, seed=0)
lo, hi = uq.confidence_interval(alpha=0.05)
print(f"prediction std     : {uq.std.mean():.4f}")
print(f"95% interval width : {(hi - lo).mean():.4f}")
worst = int(np.argmax(uq.std.ravel()))
print(f"least certain point: x={Xm[worst, 0]:.3f}, t={Xm[worst, 1]:.3f}, std={uq.std.ravel()[worst]:.4f}")''')

code('''print(pp.validate(sol_mc.model, {"pinn": "burgers_1d"}))''')

md('''**Reading the report.** The gap between "the loss went down" and "the model is right" is what the trust layer
exists for: boundary and initial conditions are met to ~1e-3, yet the error against the exact solution is ~12 %.
Train longer, or use causal training (`CausalPINNTrainer`), and run it again. `pp.validate` also takes
`level="quick" / "standard" / "thorough"`, and works on zoo models, PyTorch modules, ONNX files and plain
callables.''')

# --------------------------------------------------------------------------- 10
md("""## 9. A ready-made industrial case

`pp.presets` holds industrial cases with a data sheet, sourced defaults, a solver, references with their measured
error, a surrogate, inversion and optimisation. Water hammer in a DN500 steel line: slow closure stays inside the
pipe rating, fast closure does not.""")

code('''preset_wh = pp.presets.load("water_hammer_valve_closure")
card_wh = preset_wh.card()
print("case    :", card_wh["title"])
print("sector  :", card_wh["sector"])

print()
print(preset_wh.run(t_close=10.0))

fast = preset_wh.run(t_close=2.0)
print()
print("t_close=2.0 s ->", fast.decision["verdict"], "| trust:", fast.trust.decisao)

print("\\npreset.validate():", preset_wh.validate().get("all_pass"))''')

code('''# pp.presets.names() lists the rest: airfoil section, IPM traction motor, PCB junction temperature,
# ASME pressure vessel, pump NPSH cavitation, shell-and-tube cooler.
# Each result carries .decision, .trust, .outputs and .save(dir).
print(pp.presets.names())''')

# --------------------------------------------------------------------------- 11
md("""## 10. A digital twin on a sensor stream

A digital twin is a trained surrogate plus a running loop: sensor data in, field re-evaluated, observations compared
with predictions, anomalies flagged. `MockStream` needs no broker; in a real deployment swap it for `MQTTStream`,
`KafkaStream`, `OPCUAStream`, `ModbusStream`, `HTTPPollStream` or `FileWatchStream` — the interface is the same.""")

code('''import time
from pinneapple_systems.digital_twin import build_digital_twin
from pinneapple_systems.digital_twin.io import MockStream

sol_twin = pp.solve("burgers_1d", method="pinn", epochs=EPOCHS_SHORT)
print("surrogate:", type(sol_twin.model).__name__)

twin = build_digital_twin(sol_twin.model, field_names=["u"], coord_names=["x", "t"],
                          update_interval=0.1, anomaly_z_threshold=3.0)
twin.set_domain_coords({"x": np.tile(np.linspace(-1.0, 1.0, 40), 20),
                        "t": np.repeat(np.linspace(0.0, 1.0, 20), 40)})

def sensor(t):
    value = float(sol_twin.predict([[0.0, min(t / 3.0, 1.0)]])[0, 0])
    return {"u": value + (5.0 if t > 1.5 else 0.0)}      # the probe fails after 1.5 s

twin.add_stream(MockStream("probe", ["u"], sensor, coords={"x": 0.0, "t": 0.0}))''')

code('''with twin:                    # the twin updates its field until the block exits
    time.sleep(3.0)

print(f"field mean      : {np.mean(twin.state.fields['u']):.4f}")
print(f"history records : {len(twin.get_history_df())}")
events = twin.anomaly_monitor.recent_events()
print(f"anomalies       : {len(events)}")
for event in events[-3:]:
    print(f"  {event.field_name}: observed {event.observed:.3f}, predicted {event.predicted:.3f}, "
          f"score {event.score:.2f} ({event.detector})")''')


# --------------------------------------------------------------------------- 12 (viz)
md("""## 11. Visualisation: 1-D, 2-D, 3-D and 4-D

Each block writes a file under `visualisation/`. The 4-D case is the interesting one: a field
that depends on `(x, y, t)` is *frames plus a time axis*, not a 3-D scatter — `Dashboard` takes
the frames as RGB arrays and writes a GIF with the charts beside them.""")

code('''import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_VIS = "visualisation"
os.makedirs(OUT_VIS, exist_ok=True)

# --- 1-D: a PINN profile against the closed form -------------------------------------
pinn_v = pp.solve("burgers_1d", method="pinn", epochs=EPOCHS_PINN)
closed_v = pp.solve("burgers_1d", method="analytic")

x = np.linspace(-1.0, 1.0, 300, dtype=np.float32)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, tk in zip(axes, (0.25, 0.75)):
    X = np.column_stack([x, np.full_like(x, tk)])
    ax.plot(x, closed_v.predict(X).ravel(), "k-", lw=2, label="Cole-Hopf (exact)")
    ax.plot(x, pinn_v.predict(X).ravel(), "C3--", lw=2, label="PINN")
    ax.set_title(f"u(x, t={tk})"); ax.set_xlabel("x"); ax.grid(alpha=0.3)
axes[0].set_ylabel("u"); axes[0].legend()
fig.savefig(f"{OUT_VIS}/01_field_1d.png", dpi=120, bbox_inches="tight"); plt.close(fig)
print("1-D ->", f"{OUT_VIS}/01_field_1d.png")

# --- 2-D: the field and its error map -----------------------------------------------
# A sourced Poisson problem, solved with finite elements (same problem object as recipe 2).
from pinneapple_physics import ProblemBuilder

src2 = lambda X, ctx=None: -2.0 * np.pi**2 * np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])
b2 = ProblemBuilder("poisson_showcase").domain(x=(0, 1), y=(0, 1)).fields("u").pde("poisson")
for face in (("x", "min"), ("x", "max"), ("y", "min"), ("y", "max")):
    b2 = b2.bc(kind="dirichlet", field="u", value=0.0, on=face)
fem = pp.solve(b2.build(), method="fem", ctx={"source_fn": src2}, n=16 if SMOKE else 64)

gx2 = np.linspace(0.0, 1.0, GRID); gy2 = np.linspace(0.0, 1.0, GRID)
GX, GY = np.meshgrid(gx2, gy2, indexing="ij")
pts2 = np.column_stack([GX.ravel(), GY.ravel()])
fld = fem.predict(pts2).reshape(GX.shape)
ref2 = np.sin(np.pi * GX) * np.sin(np.pi * GY)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
im0 = axes[0].pcolormesh(GX, GY, fld, shading="auto", cmap="viridis"); axes[0].set_title("u(x, y) — FEM")
fig.colorbar(im0, ax=axes[0])
im1 = axes[1].pcolormesh(GX, GY, np.abs(fld - ref2), shading="auto", cmap="magma"); axes[1].set_title("|error|")
fig.colorbar(im1, ax=axes[1])
for ax in axes: ax.set_aspect("equal"); ax.set_xlabel("x")
axes[0].set_ylabel("y")
fig.savefig(f"{OUT_VIS}/02_field_2d.png", dpi=120, bbox_inches="tight"); plt.close(fig)
print(f"2-D -> {OUT_VIS}/02_field_2d.png   max |error| = {np.max(np.abs(fld - ref2)):.2e}")

# --- 3-D: three orthogonal slices through a solver volume ---------------------------
from pinneapple_simulation.numerical_solvers import HeatConduction3D, HeatConfig3D

n3 = 16 if SMOKE else 28
vol = HeatConduction3D(HeatConfig3D(nx=n3, ny=n3, nz=n3, nt=20 if SMOKE else 60, dt=2e-3,
                                    alpha=0.01, source_fn=lambda X, Y, Z, t: 1.0)).solve()
Tv = vol.u[-1]; k = Tv.shape[2] // 2
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, plane, label in ((axes[0], Tv[:, :, k].T, f"z = {k}"),
                         (axes[1], Tv[:, k, :].T, f"y = {k}"),
                         (axes[2], Tv[k, :, :].T, f"x = {k}")):
    im = ax.imshow(plane, origin="lower", cmap="inferno"); ax.set_title(f"slice at {label}")
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.savefig(f"{OUT_VIS}/03_volume_3d.png", dpi=120, bbox_inches="tight"); plt.close(fig)
print(f"3-D -> {OUT_VIS}/03_volume_3d.png   volume {np.shape(Tv)}, T in [{Tv.min():.4f}, {Tv.max():.4f}]")

# --- 4-D: (x, y, t) as an animated GIF ----------------------------------------------
from pinneapple_tools.visualization.dashboard import Chart, Dashboard, Series, Tile

nxy = 32 if SMOKE else 48
xs4 = np.linspace(-1.0, 1.0, nxy); X4, Y4 = np.meshgrid(xs4, xs4, indexing="ij")
t4 = np.linspace(0.0, 1.0, 8 if SMOKE else 24)
frames4, peaks4 = [], []
for tk in t4:
    width = 0.25 + 0.35 * np.sqrt(tk)
    Z4 = np.exp(-((X4 - 0.3) ** 2 + (Y4 - 0.2) ** 2) / (2 * width ** 2)) * np.exp(-tk)
    peaks4.append(float(Z4.max()))
    Z4 = (Z4 - Z4.min()) / (Z4.max() - Z4.min() + 1e-12)
    frames4.append((plt.get_cmap("viridis")(Z4)[..., :3] * 255).astype(np.uint8))

db = Dashboard("Diffusing blobs", t4, frames4,
               subtitle="T(x, y, t): a Gaussian blob diffusing and decaying",
               charts=[Chart("peak amplitude", "amplitude [a.u.]", "time [s]",
                             [Series("peak", np.asarray(peaks4))])],
               tiles=[Tile("final peak", peaks4[-1], "a.u."), Tile("frames", len(frames4), "")])
written = db.to_video(gif=f"{OUT_VIS}/04_field_4d.gif")
print("4-D ->", {k: v for k, v in written.items()})''')

# --------------------------------------------------------------------------- 13 (text->twin)
md("""## 12. One sentence to a live digital twin

Text -> the quantities it extracts (with provenance) -> a 3-D geometry -> a 3-D solve -> a
training set -> a surrogate -> a twin. The parser is offline and rule-based: a keyword score
picks the template, and it asks for the one slot it cannot default.""")

code('''import time
from pinneapple_neural import Trainer, TrainConfig
from pinneapple_design.intent import select_template
from pinneapple_systems.digital_twin import build_digital_twin
from pinneapple_systems.digital_twin.io import MockStream

REQUEST = "size a heat sink for a 50 W chip in still air, keep it under 60 mm wide"
print("request :", REQUEST)
print("template:", select_template(REQUEST)[:3])

needs = pp.design(REQUEST, out_dir="text_to_twin", verify=False, n_candidates=5, budget=10)
print("asks    :", [q.text for q in needs.questions] if hasattr(needs, "questions") else needs)

# this slot is in kelvin: 70 degC = 343.15 K
design = pp.design(REQUEST, answers={"t_limit": 343.15}, out_dir="text_to_twin",
                   verify=False, n_candidates=10, budget=20)
print("geometry:", design.geometry)
for q in design.spec.values.values():
    shown = "-" if q.value is None else f"{q.value:.6g}"
    print(f"  {q.name:<14} {shown:<10} {str(q.unit):<8} [{q.provenance}]")''')

code('''import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

cfg12 = HeatConfig3D(nx=14, ny=14, nz=14, nt=20 if SMOKE else 40, dt=2e-3, alpha=0.02,
                     source_fn=lambda X, Y, Z, t: 1.0)
u12 = HeatConduction3D(cfg12).solve().u
print("solver :", u12.shape, f"T in [{u12.min():.3f}, {u12.max():.3f}]")

nt12, nx12, ny12, nz12 = u12.shape
ax12 = [np.linspace(0.0, 1.0, n) for n in (nt12, nx12, ny12, nz12)]
T12, X12, Y12, Z12 = np.meshgrid(*ax12, indexing="ij")
Xa = np.column_stack([X12.ravel(), Y12.ravel(), Z12.ravel(), T12.ravel()]).astype(np.float32)
ya = u12.reshape(-1, 1).astype(np.float32)

rng12 = np.random.default_rng(0)
keep = min(3000 if SMOKE else 40000, len(Xa))
sel = rng12.choice(len(Xa), size=keep, replace=False)
Xa, ya = Xa[sel], ya[sel]
cut = int(0.85 * keep)
tr_dl = DataLoader(TensorDataset(torch.tensor(Xa[:cut]), torch.tensor(ya[:cut])), batch_size=1024, shuffle=True)
va_dl = DataLoader(TensorDataset(torch.tensor(Xa[cut:]), torch.tensor(ya[cut:])), batch_size=2048)

sur = build_model("vanilla_pinn", in_dim=4, out_dim=1, hidden_dim=48, n_layers=3)
tr12 = Trainer(sur, loss_fn=lambda m, yh, b: F.mse_loss(yh, b["y"]))
hist12 = tr12.fit(tr_dl, va_dl, TrainConfig(epochs=TRAINER_EPOCHS, lr=2e-3, device="cpu", log_dir="text_to_twin"))
last12 = hist12["history"][-1]
print(f"training: epoch {last12['epoch']}, train MSE {last12['train_total']:.3e}, val MSE {last12['val_total']:.3e}")

with torch.no_grad():
    pred12 = sur(torch.tensor(Xa[cut:])).y.numpy()
print(f"held-out relative L2: {float(np.linalg.norm(pred12 - ya[cut:]) / np.linalg.norm(ya[cut:])):.3e}")''')

code('''twin = build_digital_twin(sur, field_names=["T"], coord_names=["x", "y", "z", "t"],
                          update_interval=0.1, anomaly_z_threshold=3.0)
g12 = np.array(np.meshgrid(np.linspace(0, 1, 6), np.linspace(0, 1, 6), np.linspace(0, 1, 6),
                           np.linspace(0, 1, 3), indexing="ij")).reshape(4, -1)
twin.set_domain_coords({n: g12[i] for i, n in enumerate(["x", "y", "z", "t"])})

probe = (0.5, 0.5, 0.5, 1.0)
def sensor(t):
    with torch.no_grad():
        base = float(sur(torch.tensor([probe], dtype=torch.float32)).y)
    return {"T": base + (0.6 if t > 1.2 else 0.0)}

twin.add_stream(MockStream("die", ["T"], sensor,
                           coords={"x": probe[0], "y": probe[1], "z": probe[2], "t": probe[3]}))
with twin:
    time.sleep(2.5)
events12 = twin.anomaly_monitor.recent_events()
print(f"twin    : {len(twin.get_history_df())} updates, field mean {float(np.mean(twin.state.fields['T'])):.4f}, "
      f"{len(events12)} anomalies")
for ev in events12[-2:]:
    print(f"  {ev.field_name}: observed {ev.observed:.3f}, predicted {ev.predicted:.3f}, z {ev.score:.2f}")''')

# --------------------------------------------------------------------------- 14 (variants)
md("""## 13. Geometry variants, benchmark, inverse model, verification

Sweep a design parameter, simulate each variant, assemble the runs into one dataset that
carries the parameter, train three architectures on the same data and budget, keep the best,
and recover the parameter from a handful of probe readings.""")

code('''from pinneapple_design.geometry import sdf3d_sphere
from pinneapple_analysis.inverse_problems import EnsembleKalmanInversion, EKIConfig

RADII = [0.12, 0.18, 0.24, 0.30][:N_VARIANTS]

def run_case(r):
    cfg = HeatConfig3D(nx=12, ny=12, nz=12, nt=15 if SMOKE else 30, dt=2e-3, alpha=0.02,
        source_fn=lambda X, Y, Z, t, r=r: (sdf3d_sphere(
            np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]), (0.5, 0.5, 0.5), r) <= 0.0
        ).reshape(np.shape(X)).astype(float))
    return HeatConduction3D(cfg).solve().u

fields13 = {}
for r in RADII:
    fields13[r] = run_case(r)
    print(f"  r={r:.2f}: field {fields13[r].shape}, final peak T {fields13[r][-1].max():.4f}")''')

code('''u13 = fields13[RADII[0]]
nt13, nx13, ny13, nz13 = u13.shape
ax13 = [np.linspace(0.0, 1.0, n) for n in (nt13, nx13, ny13, nz13)]
T13, X13, Y13, Z13 = np.meshgrid(*ax13, indexing="ij")
c13 = np.column_stack([X13.ravel(), Y13.ravel(), Z13.ravel(), T13.ravel()]).astype(np.float32)

Xs13, ys13 = [], []
rng13 = np.random.default_rng(0)
per = 2000 if SMOKE else 8000
for r in RADII:
    rows = rng13.choice(len(c13), size=min(per, len(c13)), replace=False)
    Xs13.append(np.column_stack([c13[rows], np.full(len(rows), r, dtype=np.float32)]))
    ys13.append(fields13[r].reshape(-1, 1)[rows].astype(np.float32))

X13a, y13a = np.concatenate(Xs13), np.concatenate(ys13)
print("dataset:", X13a.shape, "(x, y, z, t, r) -> T")

perm = rng13.permutation(len(X13a)); cut13 = int(0.85 * len(X13a))
tr13, va13 = perm[:cut13], perm[cut13:]
trdl = DataLoader(TensorDataset(torch.tensor(X13a[tr13]), torch.tensor(y13a[tr13])), batch_size=1024, shuffle=True)
vadl = DataLoader(TensorDataset(torch.tensor(X13a[va13]), torch.tensor(y13a[va13])), batch_size=2048)

results = {}
for arch in ("vanilla_pinn", "siren", "modified_mlp"):
    m13 = build_model(arch, in_dim=5, out_dim=1, hidden_dim=40, n_layers=3)
    tr = Trainer(m13, loss_fn=lambda mm, yh, b: F.mse_loss(yh, b["y"]))
    h13 = tr.fit(trdl, vadl, TrainConfig(epochs=VARIANT_EPOCHS, lr=2e-3, device="cpu", log_dir=f"variants/{arch}"))
    with torch.no_grad():
        p13 = m13(torch.tensor(X13a[va13])).y.numpy()
    rel13 = float(np.linalg.norm(p13 - y13a[va13]) / np.linalg.norm(y13a[va13]))
    results[arch] = (rel13, m13)
    print(f"  {arch:<14} held-out rel L2 {rel13:.3e}")

best13 = min(results, key=lambda k: results[k][0])
sur13 = results[best13][1]
print(f"  best: {best13}")''')

code('''probes13 = np.array([[0.5, 0.5, 0.5, 1.0], [0.6, 0.5, 0.5, 1.0], [0.5, 0.6, 0.5, 1.0],
                        [0.5, 0.5, 0.6, 1.0], [0.5, 0.5, 0.5, 0.5], [0.7, 0.5, 0.5, 0.5]], dtype=np.float32)
R_TRUE = RADII[-2] if len(RADII) > 2 else RADII[-1]

def forward13(theta):
    theta = np.asarray(theta, dtype=np.float32).reshape(-1, 1)
    n = len(theta)
    pts = np.hstack([np.tile(probes13, (n, 1)), np.repeat(theta, len(probes13), axis=0)])
    with torch.no_grad():
        return sur13(torch.tensor(pts, dtype=torch.float32)).y.numpy().reshape(n, len(probes13))

y_obs13 = forward13(np.array([R_TRUE]))[0]
eki = EnsembleKalmanInversion(forward13, EKIConfig(n_ensemble=40, n_iterations=25, noise_std=0.005, seed=0))
hist13 = eki.run(y_obs13, theta_init=np.array([0.16]),
                 ensemble=np.linspace(0.10, 0.35, 40).reshape(-1, 1))
r_hat = float(np.mean(hist13.theta_mean_history[-1]))
print(f"true r = {R_TRUE:.4f}  recovered r = {r_hat:.4f}  error {100 * (r_hat - R_TRUE) / R_TRUE:+.2f} %")''')

code('''from pinneapple_decision import VeriPhysicsVerifier, ThresholdVerifier, ExecutionResult

grid13 = {"x": ax13[1], "y": ax13[2], "z": ax13[3]}
gx13, gy13, gz13 = np.meshgrid(grid13["x"], grid13["y"], grid13["z"], indexing="ij")
gp = np.column_stack([gx13.ravel(), gy13.ravel(), gz13.ravel(),
                      np.ones(gx13.size, dtype=np.float32), np.full(gx13.size, R_TRUE, dtype=np.float32)])
with torch.no_grad():
    field13 = sur13(torch.tensor(gp, dtype=torch.float32)).y.numpy().reshape(gx13.shape)

card13 = pp.trust_card(grid13, field13, reference=fields13[R_TRUE][-1],
                       train_bounds={"x": (0, 1), "y": (0, 1), "z": (0, 1)})
print("trust card (surrogate field vs solver field):", card13.decision)

verifier = VeriPhysicsVerifier(min_trust=70.0, min_coverage=0.5,
                               fallback=ThresholdVerifier(thresholds={"rel_l2": 0.15}))
result13 = ExecutionResult(status="ok",
                           metrics={"rel_l2": results[best13][0],
                                    "theta_error_pct": abs(100 * (r_hat - R_TRUE) / R_TRUE)},
                           diagnostics={}, artifacts={}, decision_id="inverse-r")
verification = verifier.verify(result13, {"name": "sphere_radius_inverse"})
print("VeriPhysics installed:", verifier.veriphysics_installed())
print("decision:", verification.passed, "| source:", verification.source)''')

# --------------------------------------------------------------------------- 12
md(f"""## Where to go next

| I want to... | Start here |
|---|---|
| The prose version of this notebook | [`GETTING_STARTED.md`](../../GETTING_STARTED.md) |
| A working script per technique | [`templates/`](../../templates/) — 35 templates |
| A technique applied end to end | [`examples/`](../../examples/) |
| A pre-trained model or dataset | `pp.zoo.list()`, `pp.zoo.load(name).predict(...)`, `pp.datasets.list()` |
| A classical solver instead of a network | `pinneapple_simulation.numerical_solvers` |
| To know which methods are validated | `pinneapple_catalog.methods.validated_methods()` |
| To know what does not work | the "Known limits" section of `GETTING_STARTED.md` |
| Physics checks in CI | `pinneapple check <dirs>` |

The other notebooks in this folder go deeper on one subject each: a first PINN with its checks, an airfoil surrogate
against CFD and theory, an equal-budget benchmark, a lunar rover, and the trust card catching broken models.""")


# --------------------------------------------------------------------------- build
def to_nb(cells_) -> dict:
    out = []
    for kind, source in cells_:
        cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        out.append(cell)
    return {
        "cells": out,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "pinneapple": {"keep_outputs": True},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(to_nb(cells), f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {OUT} ({len(cells)} cells)")

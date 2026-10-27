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
md(f"""# Ten recipes: from `pip install` to a live digital twin

{BADGE}

This is the notebook version of [`GETTING_STARTED.md`](../../GETTING_STARTED.md): the same ten recipes, in the order
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

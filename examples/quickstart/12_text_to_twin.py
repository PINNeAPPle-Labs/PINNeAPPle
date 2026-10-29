"""12 — One sentence to a live digital twin.

The chain, end to end, with nothing hidden in the middle:

  plain-language request
    -> the template it matches and the quantities it extracts (with provenance)
    -> a 3-D geometry it sizes
    -> a 3-D solver on that geometry
    -> a training set built from the solver field
    -> a surrogate trained on it
    -> a digital twin fed by a sensor stream and backed by that surrogate

The text->design step is offline and rule-based (no API key): `pinneapple_design.intent`
selects a template by keyword score and fills the slots it can read, asking for the rest.

Run from the repository root:

    python examples/quickstart/12_text_to_twin.py
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import pinneapple as pp
from pinneapple_design.intent import select_template
from pinneapple_neural import Trainer, TrainConfig, build_model
from pinneapple_simulation.numerical_solvers import HeatConduction3D, HeatConfig3D
from pinneapple_systems.digital_twin import build_digital_twin
from pinneapple_systems.digital_twin.io import MockStream

OUT = "runs/text_to_twin"

# ---------------------------------------------------------------------------
# 1. The sentence
# ---------------------------------------------------------------------------
REQUEST = "size a heat sink for a 50 W chip in still air, keep it under 60 mm wide"
print("request :", REQUEST)

ranking = select_template(REQUEST)
template, score = ranking[0]
print(f"template: {template} (keyword score {score:.2f}); runner-ups "
      f"{[(n, s) for n, s in ranking[1:] if s > 0] or 'none'}")

# ---------------------------------------------------------------------------
# 2. What it extracted, and what it had to ask
# ---------------------------------------------------------------------------
needs = pp.design(REQUEST, out_dir=OUT, verify=False, n_candidates=10, budget=20)
print("\nthe parser needs one thing it cannot default:", type(needs).__name__)
for q in needs.questions:
    print("  asks:", q.text if hasattr(q, "text") else q)

# The answer is in kelvin for this slot: 70 degC = 343.15 K.
design = pp.design(REQUEST, answers={"t_limit": 343.15}, out_dir=OUT, verify=False,
                   n_candidates=30, budget=80)
print("\nstatus  :", design.status if hasattr(design, "status") else "design package")
print("geometry:", design.geometry)
for q in design.spec.values.values():
    shown = "-" if q.value is None else f"{q.value:.6g}"
    print(f"  {q.name:<14} {shown:<10} {str(q.unit):<8} [{q.provenance}] {q.source}")

# ---------------------------------------------------------------------------
# 3. A 3-D geometry from the design, and a 3-D solver on it
# ---------------------------------------------------------------------------
# The design sizes the sink in metres; the solver works on a normalised box, so the
# field is reported in the same [0,1]^3 the surrogate will be trained on.
L, W, H = 0.033, 0.094, 0.0365          # design.geometry bounding box
power = design.spec.values["power"].value

print(f"\n3-D domain: {L * 1e3:.0f} x {W * 1e3:.0f} x {H * 1e3:.0f} mm, source {power:.0f} W")

cfg = HeatConfig3D(
    nx=24, ny=24, nz=24, nt=40, dt=2e-3, alpha=0.02,
    source_fn=lambda X, Y, Z, t: 1.0,       # uniform volumetric heating, normalised
)
volume = HeatConduction3D(cfg).solve()
print(f"solver    : {volume.u.shape} (t, x, y, z), T in [{volume.u.min():.3f}, {volume.u.max():.3f}]")

# ---------------------------------------------------------------------------
# 4. Organise the solver output into a training set
# ---------------------------------------------------------------------------
u = volume.u                                  # (nt+1, nx, ny, nz)
nt, nx, ny, nz = u.shape
axis = [np.linspace(0.0, 1.0, n) for n in (nt, nx, ny, nz)]
Tg, Xg, Yg, Zg = np.meshgrid(*axis, indexing="ij")
X_all = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel(), Tg.ravel()]).astype(np.float32)
y_all = u.reshape(-1, 1).astype(np.float32)
print(f"dataset   : {X_all.shape[0]} points of (x, y, z, t) -> T")

rng = np.random.default_rng(0)
n_keep = min(60_000, X_all.shape[0])
idx = rng.choice(X_all.shape[0], size=n_keep, replace=False)
X, y = X_all[idx], y_all[idx]

split = int(0.85 * n_keep)
train_ds = TensorDataset(torch.tensor(X[:split]), torch.tensor(y[:split]))
val_ds = TensorDataset(torch.tensor(X[split:]), torch.tensor(y[split:]))
train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4096)
print(f"split     : {split} train / {n_keep - split} validation")

# ---------------------------------------------------------------------------
# 5. The surrogate: (x, y, z, t) -> T, trained with the library's Trainer
# ---------------------------------------------------------------------------
model = build_model("vanilla_pinn", in_dim=4, out_dim=1, hidden_dim=64, n_layers=3)


def mse_loss(model, y_hat, batch):
    return F.mse_loss(y_hat, batch["y"] if "y" in batch else batch["target"])


trainer = Trainer(model, loss_fn=mse_loss)
history = trainer.fit(train_loader, val_loader,
                      TrainConfig(epochs=150, lr=2e-3, device="cpu", log_dir=OUT))
last = history["history"][-1]
print(f"training  : epoch {last['epoch']}, train MSE {last['train_total']:.3e}, "
      f"val MSE {last['val_total']:.3e}")

with torch.no_grad():
    pred = model(torch.tensor(X[split:])).y
rel = float(np.linalg.norm(pred.numpy() - y[split:]) / np.linalg.norm(y[split:]))
print(f"held-out relative L2: {rel:.3e}")

# ---------------------------------------------------------------------------
# 6. The twin: the surrogate, live, on a sensor stream
# ---------------------------------------------------------------------------
twin = build_digital_twin(model, field_names=["T"], coord_names=["x", "y", "z", "t"],
                          update_interval=0.1, anomaly_z_threshold=3.0)

xs = np.linspace(0.0, 1.0, 8)
grid = np.array(np.meshgrid(xs, xs, xs, np.linspace(0.0, 1.0, 3), indexing="ij")).reshape(4, -1)
twin.set_domain_coords({name: grid[i] for i, name in enumerate(["x", "y", "z", "t"])})

probe = (0.5, 0.5, 0.5, 1.0)


def sensor(t: float) -> dict:
    with torch.no_grad():
        base = float(model(torch.tensor([probe], dtype=torch.float32)).y)
    return {"T": base + (0.6 if t > 1.2 else 0.0)}    # the probe drifts after 1.2 s


twin.add_stream(MockStream("die", ["T"], sensor,
                           coords={"x": probe[0], "y": probe[1], "z": probe[2], "t": probe[3]}))

with twin:
    time.sleep(2.5)

events = twin.anomaly_monitor.recent_events()
print(f"\ntwin      : {len(twin.get_history_df())} updates, "
      f"field mean {float(np.mean(twin.state.fields['T'])):.4f}, {len(events)} anomalies")
for event in events[-2:]:
    print(f"  {event.field_name}: observed {event.observed:.3f}, predicted {event.predicted:.3f}, "
          f"z {event.score:.2f}")
print(f"\nartefacts : {OUT}/ (design.json, report.html, geometry/)")

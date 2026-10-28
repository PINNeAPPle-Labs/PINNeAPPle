"""06 — Use a trained model: predict on a grid, save it, check what you saved.

Run from the repository root:

    python examples/quickstart/06_use_the_model.py
"""
from __future__ import annotations

import numpy as np
import torch

import pinneapple as pp
from pinneapple_neural import build_model
from pinneapple_neural.predictor import infer_on_grid_2d
from pinneapple_tools.model_export import export_csv, export_npz

ARCH, HIDDEN, LAYERS = "vanilla_pinn", 64, 3

# --- 1. Train a model whose architecture you can rebuild exactly --------
sol = pp.solve("burgers_1d", method="pinn", epochs=400,
               architecture=ARCH, hidden_dim=HIDDEN, n_layers=LAYERS)
model = sol.model
print("model:", type(model).__name__, "|", sol.metadata()["info"])

# --- 2. Evaluate on a grid: the shape plots and exports expect ----------
result = infer_on_grid_2d(model, x_range=(-1.0, 1.0), y_range=(0.0, 1.0),
                          nx=120, ny=80, field_names=["u"])
print("grid         :", result.grid_shape)
print("field arrays :", {k: np.asarray(v).shape for k, v in result.fields.items()})

XX, YY = np.meshgrid(result.coords["x"], result.coords["y"], indexing="ij")
coords = np.column_stack([XX.ravel(), YY.ravel()])
values = np.asarray(result.fields["u"])

# --- 3. Save the field for other tools ----------------------------------
export_csv(coords, values.ravel(), "runs/quickstart_field.csv",
           coord_names=["x", "t"], field_names=["u"])
export_npz(coords, values.ravel(), "runs/quickstart_field.npz",
           coord_names=["x", "t"], field_names=["u"])
print("saved        : runs/quickstart_field.csv, runs/quickstart_field.npz")

# --- 4. Check the field you are about to hand over ----------------------
reference = pp.solve("burgers_1d", method="analytic").predict(coords).reshape(values.shape)
card = pp.trust_card({"x": result.coords["x"], "t": result.coords["y"]}, values,
                     reference=reference, train_bounds={"x": (-1.0, 1.0), "t": (0.0, 1.0)})
print("trust card   :", card.decision)

# --- 5. Save the model and reload it ------------------------------------
model.save_checkpoint("runs/quickstart_model.pt", metadata={"problem": "burgers_1d"})
checkpoint = torch.load("runs/quickstart_model.pt", weights_only=False)
print("checkpoint   :", sorted(checkpoint.keys()), "|", checkpoint["class_name"])

reloaded = build_model(ARCH, in_dim=2, out_dim=1, hidden_dim=HIDDEN, n_layers=LAYERS)
reloaded.load_state_dict(checkpoint["state_dict"])
reloaded.eval()
with torch.no_grad():
    a = model(torch.tensor(coords[:4], dtype=torch.float32)).y
    b = reloaded(torch.tensor(coords[:4], dtype=torch.float32)).y
print("reload equal :", bool(torch.allclose(a, b, atol=1e-6)))

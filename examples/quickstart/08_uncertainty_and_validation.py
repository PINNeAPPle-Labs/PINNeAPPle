"""08 — How much should you trust a prediction?

Two different questions, two tools:
  * uncertainty — `uq_predict(..., method="mc_dropout")` gives a per-point std;
  * validation — `pp.validate(model, {"pinn": <preset>})` re-checks the physics
    at points the model never saw, against the exact solution when one exists.

The report below FAILS on purpose: a converged-looking loss curve is not a
validated model. That is the point of the trust layer.

Run from the repository root:

    python examples/quickstart/08_uncertainty_and_validation.py
"""
from __future__ import annotations

import numpy as np
import torch

import pinneapple as pp
from pinneapple_analysis.uncertainty import uq_predict

EPOCHS = 800

# --- 1. A trained model to inspect --------------------------------------
sol = pp.solve("burgers_1d", method="pinn", epochs=EPOCHS)
print("model:", type(sol.model).__name__, "| trained in", f"{sol.metadata()['wall_time_s']:.1f} s")

# --- 2. Uncertainty: MC Dropout over the same network -------------------
rng = np.random.default_rng(0)
X = rng.uniform([-1.0, 0.0], [1.0, 1.0], size=(2000, 2)).astype(np.float32)
uq = uq_predict(sol.model, torch.tensor(X), method="mc_dropout", n_samples=50, dropout_p=0.1, seed=0)
lo, hi = uq.confidence_interval(alpha=0.05)
print(f"prediction std     : {uq.std.mean():.4f}")
print(f"95% interval width : {(hi - lo).mean():.4f}")
worst = int(np.argmax(uq.std.ravel()))
print(f"least certain point: x={X[worst, 0]:.3f}, t={X[worst, 1]:.3f}, std={uq.std.ravel()[worst]:.4f}")

# --- 3. Validation: does the model satisfy the physics? -----------------
report = pp.validate(sol.model, {"pinn": "burgers_1d"})
print()
print(report)

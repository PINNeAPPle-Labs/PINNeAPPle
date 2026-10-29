"""02 — Solve a problem and compare methods on the same points.

`pp.solve(problem, method=...)` accepts a preset name, a `ProblemSpec` or a
`PhysicalProblem`. Methods: `pinn`, `fem`, `analytic`, `reference`, `exact`,
`external` (`pp.list_methods()`), and anything you add with `pp.register_method`.

Run from the repository root:

    python examples/quickstart/02_solve_and_compare.py
"""
from __future__ import annotations

import numpy as np

import pinneapple as pp

PROBLEM = "burgers_1d"
EPOCHS = 800

# --- 1. Closed form: instant, no training -------------------------------
closed = pp.solve(PROBLEM, method="analytic")
print("analytic :", closed.metadata()["kind"], f"{closed.metadata()['wall_time_s']:.3f} s")

# --- 2. A PINN trained from the PDE alone (no data) ---------------------
pinn = pp.solve(PROBLEM, method="pinn", epochs=EPOCHS)
print("pinn     :", pinn.metadata()["kind"], f"{pinn.metadata()['wall_time_s']:.1f} s")

# --- 3. Score them against the analytic reference on the same points ----
print()
print(pp.compare(PROBLEM, methods=["pinn", "analytic"], reference="analytic",
                 options={"pinn": {"epochs": EPOCHS}}))

# --- 4. Any solution predicts on arbitrary points -----------------------
X = np.array([[-0.5, 0.0], [0.0, 0.5], [0.5, 1.0]], dtype=np.float32)
print("pinn.predict at 3 points :", np.round(pinn.predict(X).ravel(), 4))
print("analytic at the same ones:", np.round(closed.predict(X).ravel(), 4))

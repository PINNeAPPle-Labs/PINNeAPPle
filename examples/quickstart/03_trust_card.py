"""03 — Decide whether a field can be trusted, before anyone uses it.

`pp.trust_card(grid, values, ...)` runs the checks a field needs — PDE residual,
boundary conditions, conservation, bounds, reference, applicability — and returns
one decision: APPROVED / REVIEW / REJECT. It works on any field from any code: a
PINN, a surrogate, a solver, a spreadsheet export.

Run from the repository root:

    python examples/quickstart/03_trust_card.py
"""
from __future__ import annotations

import numpy as np

import pinneapple as pp

# A harmonic field on the unit square: laplace(u) = 0, u = sin(pi x) on y = 1.
x = y = np.linspace(0.0, 1.0, 41)
X, Y = np.meshgrid(x, y, indexing="ij")
u = np.sin(np.pi * X) * np.sinh(np.pi * Y) / np.sinh(np.pi)

bcs = {"x_min": 0.0, "x_max": 0.0, "y_min": 0.0, "y_max": lambda P: np.sin(np.pi * P[:, 0])}

# --- 1. A correct field -------------------------------------------------
good = pp.trust_card({"x": x, "y": y}, u, pde="laplace", dirichlet=bcs,
                     train_bounds={"x": (0.0, 1.0), "y": (0.0, 1.0)})
print(good)

# --- 2. The same field with a 20 % scaling error ------------------------
#      Still harmonic (the residual check passes) — and still wrong.
bad = pp.trust_card({"x": x, "y": y}, 1.2 * u, pde="laplace",
                    dirichlet={"y_max": lambda P: np.sin(np.pi * P[:, 0])})
print()
print(bad)

# --- 3. The decision is a value, not console output ---------------------
print("\ncorrect field:", good.decision)
print("scaled field :", bad.decision)
assert good.decision == "APPROVED" and bad.decision == "REJECT"

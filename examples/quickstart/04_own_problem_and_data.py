"""04 — Your own problem, your own geometry, and the data to train on.

`ProblemBuilder` describes a PDE as data; `CSG*` shapes describe a geometry;
`generate_pinn_dataset` turns both into collocation / boundary / initial batches
ready for training.

Problem:  1-D heat equation   u_t = alpha * u_xx    on x in [0,1], t in [0,1]
          u(x,0) = sin(pi x),   u(0,t) = u(1,t) = 0
          exact:  u(x,t) = exp(-alpha pi^2 t) sin(pi x)

Run from the repository root:

    python examples/quickstart/04_own_problem_and_data.py
"""
from __future__ import annotations

import numpy as np

import pinneapple as pp
from pinneapple_physics import ProblemBuilder, metrics
from pinneapple_simulation.numerical_solvers import generate_pinn_dataset, register_all
from pinneapple_design.geometry.csg import CSGCircle, CSGDifference

ALPHA = 0.05


def exact(X: np.ndarray) -> np.ndarray:
    return np.exp(-ALPHA * np.pi**2 * X[:, 1:2]) * np.sin(np.pi * X[:, 0:1])


# --- 1. Describe the problem --------------------------------------------
spec = (
    ProblemBuilder("heat_1d_custom")
    .domain(x=(0.0, 1.0), t=(0.0, 1.0))
    .fields("u")
    .pde("heat_equation", alpha=ALPHA)
    .ic(field="u", fn=lambda X: np.sin(np.pi * X[:, 0:1]))
    .bc(kind="dirichlet", field="u", value=0.0, on="x_boundary")
    .build()
)
print("problem    :", spec.name or spec.problem_id, "| coordinates:", spec.coords)
print("conditions :", [(c.name, c.kind) for c in spec.conditions])

# --- 2. Solve it with a PINN and score it against the closed form --------
sol = pp.solve(spec, method="pinn", epochs=2000)
X = np.random.default_rng(0).uniform([0.0, 0.0], [1.0, 1.0], size=(5000, 2)).astype(np.float32)
rel = float(np.ravel(metrics.relative_l2(sol.predict(X), exact(X)))[0])
print(f"solved in  : {sol.metadata()['wall_time_s']:.1f} s")
print("relative L2:", f"{rel:.3e}")

# --- 3. The same spec produces a training dataset -----------------------
register_all()
batch = generate_pinn_dataset(spec, n_col=4000, n_bc=400, n_ic=400, seed=0)
print("dataset    :", {k: np.shape(v) for k, v in batch.items() if hasattr(v, "shape")})

# --- 4. Geometry is data too: a parameterised annulus -------------------
for r in (0.10, 0.20, 0.30):
    shape = CSGDifference(CSGCircle(0.0, 0.0, 0.5), CSGCircle(0.0, 0.0, r))
    interior = shape.sample_interior(2000, seed=0)
    boundary = shape.sample_boundary(200, seed=0)
    area = np.pi * (0.5**2 - r**2)
    print(f"  inner r={r:.2f}  area={area:.4f}  interior {interior.shape}  boundary {boundary.shape}")
    assert bool(shape.contains(interior).all())      # every point really is inside

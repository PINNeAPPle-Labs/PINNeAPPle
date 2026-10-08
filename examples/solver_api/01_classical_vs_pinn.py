"""The same problem through one call: a classical FEM solve and a PINN, compared against the exact solution.

    PYTHONPATH=. python examples/solver_api/01_classical_vs_pinn.py
"""
import numpy as np

import pinneapple as pp
from pinneapple_physics.pde_environment.builder import ProblemBuilder

# laplace(u) = f on the unit square, u = 0 on the boundary, f chosen so that u = sin(pi x) sin(pi y)
spec = (ProblemBuilder("poisson_manufactured").domain(x=(0, 1), y=(0, 1)).fields("u").pde("poisson")
        .bc("dirichlet", field="u", value=0.0, on="x_boundary")
        .bc("dirichlet", field="u", value=0.0, on="y_boundary").build())
problem = pp.PhysicalProblem.from_pde_spec(spec)

ctx = {"source_fn": lambda X, ctx=None: -2 * np.pi**2 * np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])}
exact = lambda X: (np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1]))[:, None]

print(pp.compare(
    problem, ["fem", "pinn"], reference="exact",
    options={"exact": {"fn": exact}, "fem": {"ctx": ctx},
             "pinn": {"ctx": ctx, "epochs": 2000, "hidden_dim": 32, "n_layers": 3}},
))
print()
for method, opts in (("fem", {}), ("pinn", {"epochs": 50, "hidden_dim": 16, "n_layers": 2})):
    meta = pp.solve(problem, method, ctx=ctx, **opts).metadata()
    print(method, {k: meta[k] for k in ("kind", "wall_time_s", "fingerprint")}, meta["info"])

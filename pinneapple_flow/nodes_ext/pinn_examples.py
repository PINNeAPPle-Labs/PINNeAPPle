"""The one-call solver examples: ``examples/first_example`` (a Burgers PINN against the Cole-Hopf solution with
``pp.solve``) and ``examples/solver_api`` (a classical FEM and a PINN on the manufactured Poisson problem with
``pp.compare``)."""
from __future__ import annotations

import time

import numpy as np

from ..registry import Param, Port, node


@node("pinn.burgers_first", "Burgers PINN against the exact solution (first example)", category="PINN",
      outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("epochs", "int", 1500, minimum=1), Param("seed", "int", 0), Param("nu_over_pi", "float", 0.01,
                                                                                      help="nu = value / pi"),
              Param("t", "float", 0.75, minimum=0.0, maximum=1.0, help="time of the profile compared")],
      calls="pinneapple.PhysicalProblem.from_preset('burgers_1d') + pinneapple.solve ('pinn', 'analytic') + "
            "pinneapple.metrics.relative_l2")
def burgers_first(ctx, inputs, params):
    """examples/first_example/first_example.py: u_t + u u_x = nu u_xx on [-1, 1]; the PINN's profile at t = 0.75 (the
    steep front) against the Cole-Hopf closed form."""
    import pinneapple as pp
    prob = pp.PhysicalProblem.from_preset("burgers_1d", nu=float(params["nu_over_pi"]) / np.pi)
    t0 = time.perf_counter()
    pinn = pp.solve(prob, "pinn", epochs=int(params["epochs"]), seed=int(params["seed"]))
    ctx.metric("train_seconds", round(time.perf_counter() - t0, 1))
    exact = pp.solve(prob, "analytic")
    x = np.linspace(-1, 1, 201)
    pts = np.stack([x, np.full_like(x, float(params["t"]))], axis=1)
    u_p, u_e = pinn.predict(pts)[:, 0], exact.predict(pts)[:, 0]
    ctx.metric("relative_l2_at_t", float(pp.metrics.relative_l2(u_p, u_e)), f"at t = {params['t']:g}, 201 points")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(x, u_e, "k-", label="exact (Cole-Hopf)")
    ax.plot(x, u_p, "r--", label="PINN")
    ax.set(xlabel="x", ylabel=f"u(x, t = {params['t']:g})")
    ax.legend()
    f = ctx.figure(fig, "first_example", "PINN against the exact solution")
    return {"metrics": ctx.report["metrics"], "figure": f}


@node("pinn.poisson_fem_vs_pinn", "FEM and PINN on the manufactured Poisson problem (pp.compare)", category="PINN",
      outputs=[Port("metrics", "metrics")],
      params=[Param("epochs", "int", 2000, minimum=1), Param("hidden_dim", "int", 32, minimum=4),
              Param("n_layers", "int", 3, minimum=1)],
      calls="pinneapple_physics.pde_environment.builder.ProblemBuilder + pinneapple.PhysicalProblem.from_pde_spec + "
            "pinneapple.compare (fem, pinn, reference exact) + pinneapple.solve(...).metadata")
def poisson_fem_vs_pinn(ctx, inputs, params):
    """examples/solver_api/01_classical_vs_pinn.py: laplace(u) = f on the unit square with u = sin(pi x) sin(pi y),
    one call for both methods against the exact solution."""
    import pinneapple as pp
    from pinneapple_physics.pde_environment.builder import ProblemBuilder
    spec = (ProblemBuilder("poisson_manufactured").domain(x=(0, 1), y=(0, 1)).fields("u").pde("poisson")
            .bc("dirichlet", field="u", value=0.0, on="x_boundary")
            .bc("dirichlet", field="u", value=0.0, on="y_boundary").build())
    problem = pp.PhysicalProblem.from_pde_spec(spec)
    sctx = {"source_fn": lambda X, ctx=None: -2 * np.pi ** 2 * np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])}

    def exact(X):
        return (np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1]))[:, None]
    table = pp.compare(problem, ["fem", "pinn"], reference="exact",
                       options={"exact": {"fn": exact}, "fem": {"ctx": sctx},
                                "pinn": {"ctx": sctx, "epochs": int(params["epochs"]),
                                         "hidden_dim": int(params["hidden_dim"]), "n_layers": int(params["n_layers"])}})
    for r in table.rows:
        if r["status"] != "ok":
            ctx.warn(f"{r['method']}: {r['status']}")
            continue
        for metric, per_field in r["metrics"].items():
            vals = list(per_field.values()) if isinstance(per_field, dict) else [per_field]
            if vals and all(isinstance(v, (int, float)) for v in vals):
                ctx.metric(f"{r['method']}.{metric}", float(np.mean(vals)), "mean over fields, against exact")
    ctx.metric("best", str(table.best()))
    ctx.table("pp.compare", ["result"], [[line] for line in str(table).splitlines()])
    return {"metrics": ctx.report["metrics"]}


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="pinn.burgers_first", title="Your first result: a Burgers PINN against the exact solution", domain="PINN",
        example="first_example", keywords={"first example": 3, "first result": 3, "burgers": 1.5, "cole-hopf": 2,
                                           "getting started": 1.5, "pinn": 1}, min_score=3,
        nodes=("pinn.burgers_first",), references=("raissi2019pinn", "cole1951burgers"),
        template="first_example/first_example", quick=True, description="examples/first_example/first_example.py"))
    register_capability(Capability(
        id="pinn.poisson_fem_vs_pinn", title="A classical solver and a PINN on the same problem, one call",
        domain="PINN", example="solver_api",
        keywords={"manufactured": 3, "fem vs pinn": 3, "classical vs pinn": 3, "pp.compare": 2, "solver api": 3},
        min_score=3, nodes=("pinn.poisson_fem_vs_pinn",), references=("brenner2008fem", "raissi2019pinn"),
        template="solver_api/01_classical_vs_pinn", quick=True,
        description="examples/solver_api/01_classical_vs_pinn.py"))


_register_capabilities()

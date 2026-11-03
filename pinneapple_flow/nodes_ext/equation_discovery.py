"""The standard equation-discovery workflow (:mod:`pinneapple_analysis.inverse_problems.discovery`, #1192): SINDy
for ODEs and PDE-FIND for PDEs, the model chosen on held-out data, bootstrap inclusion and intervals, the discovered
system simulated against data it never fitted, and the equations as text and LaTeX."""
from __future__ import annotations

from ..registry import Param, Port, node


def _lorenz(noise: float, seed: int):
    import numpy as np
    from scipy.integrate import solve_ivp

    def f(_t, s):
        x, y, z = s
        return [10 * (y - x), x * (28 - z) - y, x * y - 8 / 3 * z]
    t = np.arange(0, 20, 0.002)
    Y = solve_ivp(f, (0, 20), [-8, 8, 27], t_eval=t, rtol=1e-10, atol=1e-10).y.T
    return t, Y + noise * Y.std(0) * np.random.default_rng(seed).standard_normal(Y.shape), ["x", "y", "z"]


@node("discover.equations", "Equation discovery (SINDy / PDE-FIND), checked on held-out data", category="Inverse",
      outputs=[Port("metrics", "metrics")],
      params=[Param("kind", "enum", "ode", choices=["ode", "pde"]),
              Param("csv", "str", "", help="ODE: columns t and the states; PDE: an .npz with u (nx, nt), x, t"),
              Param("states", "list", [], help="ODE CSV: the state columns"),
              Param("degree", "int", 2, minimum=1, maximum=4, help="ODE: polynomial library degree"),
              Param("noise", "float", 0.01, minimum=0.0, maximum=0.2,
                    help="without data: the demo (Lorenz / Burgers) with this relative noise, stated"),
              Param("holdout", "float", 0.2, minimum=0.05, maximum=0.5), Param("n_bootstrap", "int", 50, minimum=0),
              Param("seed", "int", 0)],
      calls="pinneapple_analysis.inverse_problems.discovery.discover_ode / discover_pde")
def discover_equations(ctx, inputs, params):
    import numpy as np

    from pinneapple_analysis.inverse_problems.discovery import discover_ode, discover_pde
    kind = str(params["kind"])
    if kind == "ode":
        if params.get("csv"):
            import csv
            with open(params["csv"], newline="") as f:
                rows = list(csv.DictReader(f))
            t = np.array([float(r["t"]) for r in rows])
            names = list(params["states"])
            Y = np.array([[float(r[k]) for k in names] for r in rows])
            src = f"{len(t)} samples from {params['csv']}"
        else:
            t, Y, names = _lorenz(float(params["noise"]), int(params["seed"]))
            src = f"Lorenz (sigma 10, rho 28, beta 8/3), {len(t)} samples, {params['noise']:.1%} noise (synthetic)"
        r = discover_ode(t, Y, names, degree=int(params["degree"]), holdout=float(params["holdout"]),
                         n_bootstrap=int(params["n_bootstrap"]), seed=int(params["seed"]))
        for e in r["equations"]:
            ctx.metric(f"equation.{e.split('=')[0].strip()}", e)
        ctx.metric("savgol_window", r["window"], "chosen by the held-out rollout")
    else:
        if params.get("csv"):
            d = np.load(params["csv"])
            u, x, t = d["u"], d["x"], d["t"]
            src = f"field {u.shape} from {params['csv']}"
        else:
            from pinneapple_analysis.validation.pinn import cole_hopf
            x = np.linspace(-1, 1, 256)
            t = np.linspace(0, 1, 101)
            X, T = np.meshgrid(x, t, indexing="ij")
            u = cole_hopf(X, T, 0.1)
            u = u + float(params["noise"]) * np.random.default_rng(int(params["seed"])).standard_normal(u.shape)
            src = f"viscous Burgers (nu 0.1, Cole-Hopf exact), {u.shape} grid, {params['noise']:.1%} noise (synthetic)"
        r = discover_pde(u, x, t, holdout=float(params["holdout"]), n_bootstrap=int(params["n_bootstrap"]),
                         seed=int(params["seed"]))
        ctx.metric("equation", r["equation"])
    ctx.metric("data", src)
    ctx.table("checks", ["check", "status", "detail"], [[c["code"], c["status"], c["detail"]] for c in r["checks"]])
    bad = [c["code"] for c in r["checks"] if c["status"] == "fail"]
    ctx.metric("verdict", "fail" if bad else "pass", ", ".join(bad) or "every check passes")
    ctx.summary((" ; ".join(r["equations"]) if kind == "ode" else r["equation"]) + (
        "; fail: " + ", ".join(bad) if bad else "; verified on held-out data"))
    return {"metrics": ctx.report["metrics"]}


def build_flow(**params):
    from ..planner.build import FlowBuilder
    title = "Equation discovery: sparse regression checked on held-out data"
    fb = FlowBuilder(title, "SINDy (ODEs) or PDE-FIND (PDEs): the model chosen on held-out data, bootstrap "
                            "inclusion and intervals, the discovered system simulated against data it never fitted "
                            "(pinneapple_analysis.inverse_problems.discovery).")
    fb.add("discover", "discover.equations", dict(params), label="Equation discovery")
    fb.add("report", "export.report", {"title": title}, label="Report")
    fb.connect("discover", "metrics", "report", "items")
    return fb


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="discover.equations", title="Equation discovery from data", domain="inverse",
        keywords={"equation discovery": 3, "descoberta de equações": 3, "descobrir equações": 3,
                  "descobrir a equação": 3, "discover the equation": 3, "sindy": 3, "pde-find": 3,
                  "governing equations": 2, "equações governantes": 2, "symbolic regression": 2,
                  "regressão simbólica": 2},
        min_score=3, nodes=("discover.equations",), build=lambda text, det: build_flow(),
        inputs=("time series of the states (ODE) or a field on a grid (PDE)",),
        uncited="the method cites Brunton et al. (2016), Rudy et al. (2017) and Fasel et al. (2022) in "
                "pinneapple_analysis.inverse_problems.discovery; they are not in the flow reference file yet",
        description="pinneapple_analysis.inverse_problems.discovery"))


_register_capabilities()

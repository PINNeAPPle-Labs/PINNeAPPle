"""Nodes for ``examples/geophysics`` (shallow water on the sphere, 4D-Var on Lorenz-96, Richards infiltration),
``examples/physics_optimizer`` (:class:`pinneapple_core.optim.PhysicsOptimizer` on a constrained shape),
``examples/physics_module`` (:class:`pinneapple_core.module.Hybrid`: a coarse FEM with a learned correction) and
``examples/time_series`` (the naive baselines of the quickstart, :class:`AdaptiveForecaster` on a series that changes
regime). The synthetic inputs of each example are node parameters, so the flow is the example's study."""
from __future__ import annotations

import math

import numpy as np

from ..registry import Param, Port, node


@node("geo.earth_system", "Earth-system blocks: sphere, 4D-Var, infiltration", category="Geophysics",
      outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("days", "float", 4.0, minimum=0.1, maximum=60.0), Param("truncation", "int", 42, minimum=10),
              Param("dt", "float", 600.0, minimum=10.0), Param("da_n", "int", 40, minimum=4),
              Param("da_window", "int", 20, minimum=2), Param("seed", "int", 0)],
      calls="pinneapple_simulation.geophysics.SphericalHarmonics + williamson_case6 + solve_richards (CELIA_1990_SOIL, "
            "Boundary); pinneapple_analysis.data_assimilation.twin_experiment")
def earth_system(ctx, inputs, params):
    """examples/geophysics/01_sphere_and_assimilation.py: the Rossby-Haurwitz wave (Williamson case 6) and the drift
    of its conserved quantities, 4D-Var on Lorenz-96 (background and analysis RMSE), Richards infiltration (Celia et
    al. 1990) and its mass-balance ratio."""
    from pinneapple_analysis.data_assimilation import twin_experiment
    from pinneapple_simulation.geophysics import (
        CELIA_1990_SOIL,
        Boundary,
        SphericalHarmonics,
        solve_richards,
        williamson_case6,
    )
    sh = SphericalHarmonics(int(params["truncation"]))
    sw = williamson_case6(sh)
    sw.run(float(params["days"]) * 86400, float(params["dt"]), diagnostics_every=86400)
    d0, d1 = sw.history[0], sw.history[-1]
    for k in ("mass", "energy", "potential_enstrophy"):
        ctx.metric(f"drift.{k}", (d1[k] - d0[k]) / d0[k], f"relative change after {params['days']:g} days")
    da = twin_experiment(n=int(params["da_n"]), window_steps=int(params["da_window"]), seed=int(params["seed"]))
    for k, v in da.items():
        if k.startswith("rmse"):
            ctx.metric(f"da.{k}", float(v))
    soil = solve_richards(CELIA_1990_SOIL, 100.0, 100, -1000.0, 86400.0, Boundary("head", -75.0),
                          Boundary("free_drainage"), t_out=np.array([0, 21600, 43200, 86400]))
    ctx.metric("mbr", float(soil.mass_balance_ratio), "Richards mass-balance ratio")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    c = ax[0].contourf(np.rad2deg(sh.lon), np.rad2deg(sh.lat), sw.fields()["h"], 20, cmap="jet")
    fig.colorbar(c, ax=ax[0], label="h [m]")
    ax[0].set(title=f"Rossby-Haurwitz wave, day {params['days']:g}")
    k = int(params["da_window"])
    ax[1].plot(da["truth"][k].numpy(), "k", lw=2, label="truth")
    ax[1].plot(da["background_trajectory"][k].numpy(), "C0--", label="background")
    ax[1].plot(da["analysis_trajectory"][k].numpy(), "C1", label="4D-Var analysis")
    ax[1].legend(fontsize=8)
    for j, t in enumerate(soil.t[1:], start=1):
        ax[2].plot(soil.theta[j], soil.z, label=f"{t / 3600:g} h")
    ax[2].legend()
    fig.tight_layout()
    f = ctx.figure(fig, "sphere_and_assimilation", "shallow water, 4D-Var, infiltration")
    return {"metrics": ctx.report["metrics"], "figure": f}


@node("core.torsion_optimizer", "Constrained shape optimization, six methods", category="Optimization",
      outputs=[Port("metrics", "metrics")],
      params=[Param("n", "int", 10, minimum=2, maximum=100, help="P1 mesh cells per side"),
              Param("start", "list", [1.6, 0.8], help="a, b at the start"), Param("bounds", "list", [0.5, 2.0])],
      calls="pinneapple_core.Mesh.structured + fem.solve_poisson / integrate_p1 + optim.PhysicsOptimizer.compare (eq "
            "constraint, an adjoint from torch.func.grad)")
def torsion_optimizer(ctx, inputs, params):
    """Maximise the torsional rigidity of an a x b rectangle with a b = 1 by SLSQP, the adjoint, a penalty, an
    augmented Lagrangian, differential evolution and Bayesian optimisation (examples/physics_optimizer)."""
    import torch

    from pinneapple_core import Mesh
    from pinneapple_core.fem import integrate_p1, solve_poisson
    from pinneapple_core.optim import PhysicsOptimizer, eq
    torch.set_default_dtype(torch.float64)
    n = int(params["n"])
    ref = Mesh.structured([0, 0], [1, 1], (n, n))
    base, boundary = torch.tensor(ref.points), ref.boundary_nodes()

    def rigidity(x):
        pts = base * x
        return integrate_p1(pts, ref.cells, solve_poisson(pts, ref.cells, 1.0, boundary))
    lo, hi = map(float, params["bounds"])
    a0, b0 = map(float, params["start"])
    opt = PhysicsOptimizer({"a": (lo, hi, a0), "b": (lo, hi, b0)}, objective=lambda x: -rigidity(x),
                           constraints=[eq(lambda x: x[0] * x[1] - 1.0)])
    res = opt.compare(grad=lambda z: torch.func.grad(lambda x: -rigidity(x))(torch.tensor(z)).numpy())
    rows = []
    for name, r in res.items():
        ctx.metric(f"{name}.value", float(r.fun))
        ctx.metric(f"{name}.a", float(r.x[0]))
        ctx.metric(f"{name}.b", float(r.x[1]))
        ctx.metric(f"{name}.violation", float(r.violation))
        ctx.metric(f"{name}.evaluations", int(r.n_evals))
        rows.append([name, float(r.x[0]), float(r.x[1]), float(r.fun), float(r.violation), int(r.n_evals)])
    ctx.table("methods", ["method", "a", "b", "objective (-rigidity)", "violation", "evaluations"], rows)
    torch.set_default_dtype(torch.float32)
    return {"metrics": ctx.report["metrics"]}


@node("core.hybrid_fem", "Coarse FEM plus a learned correction", category="Hybrid models",
      outputs=[Port("metrics", "metrics")],
      params=[Param("n_coarse", "int", 6, minimum=2), Param("n_fine", "int", 128, minimum=8),
              Param("n_samples", "int", 160, minimum=8), Param("n_train", "int", 128, minimum=4),
              Param("epochs", "int", 600, minimum=1), Param("lr", "float", 1e-2), Param("seed", "int", 0)],
      calls="pinneapple_core.Mesh.interval + fem.solve_poisson; pinneapple_core.module.Hybrid(SolverModule, "
            "Sequential corrector).fit / describe")
def hybrid_fem(ctx, inputs, params):
    """-(k(x) u')' = 1 with k = 1 + 0.8 sin(2 pi f x + phase): a 6-cell FEM and a dilated Conv1d corrector trained
    end to end against a 128-cell FEM; held-out relative error of the coarse FEM alone and of the hybrid
    (examples/physics_module/01_coarse_fem_plus_neural_correction.py)."""
    import torch
    import torch.nn as nn

    from pinneapple_core import Mesh
    from pinneapple_core.fem import solve_poisson
    from pinneapple_core.module import Hybrid, Sequential, SolverModule
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(int(params["seed"]))
    NF = int(params["n_fine"]) + 1
    xf = torch.linspace(0, 1, NF)

    def k_fun(th, x):
        return 1 + 0.8 * torch.sin(2 * math.pi * th[:, :1] * x + th[:, 1:])

    def fem(n):
        m = Mesh.interval(0, 1, n)
        pts = torch.tensor(m.points)
        xc = (pts[1:] + pts[:-1]).T / 2

        def solve(th):
            u = torch.stack([solve_poisson(pts, m.cells, 1.0, m.boundary_nodes(), kappa=k) for k in k_fun(th, xc)])
            return torch.nn.functional.interpolate(u[:, None], size=NF, mode="linear", align_corners=True)[:, 0]
        return solve
    theta = torch.rand(int(params["n_samples"]), 2) * torch.tensor([3.0, 6.28]) + torch.tensor([2.0, 0.0])
    truth = fem(int(params["n_fine"]))(theta)
    corr = Sequential(nn.Conv1d(3, 32, 5, padding=2), nn.Tanh(), nn.Conv1d(32, 32, 5, padding=4, dilation=2), nn.Tanh(),
                      nn.Conv1d(32, 32, 5, padding=8, dilation=4), nn.Tanh(), nn.Conv1d(32, 1, 5, padding=2),
                      lambda c: c[:, 0])
    model = Hybrid(SolverModule(fem(int(params["n_coarse"]))), corr,
                   corrector_input=lambda a, u0: torch.stack([u0, k_fun(a[0], xf), 0 * u0 + xf], 1))
    nt = int(params["n_train"])
    model.fit(theta[:nt], truth[:nt], epochs=int(params["epochs"]), lr=float(params["lr"]), batch_size=32)

    def rel(p, t):
        return ((p - t).norm(dim=1) / t.norm(dim=1)).mean().item()
    with torch.no_grad():
        ctx.metric("coarse_fem_rel_l2", rel(model.solver(theta[nt:]), truth[nt:]), "held out")
        ctx.metric("hybrid_rel_l2", rel(model(theta[nt:]), truth[nt:]), "held out")
    torch.set_default_dtype(torch.float32)
    return {"metrics": ctx.report["metrics"]}


def regime_series(n: int = 900, seed: int = 0) -> np.ndarray:
    """The series of examples/time_series/adaptive_forecasting.py: seasonal, then a trend, then a random walk."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = np.empty(n)
    a, b = n // 3, 2 * n // 3
    y[:a] = 10 + 5 * np.sin(2 * np.pi * t[:a] / 12) + rng.normal(0, 0.5, a)
    y[a:b] = y[a - 1] + 0.4 * np.arange(1, b - a + 1) + rng.normal(0, 0.5, b - a)
    y[b:] = y[b - 1] + np.cumsum(rng.normal(0, 1.5, n - b))
    return y


@node("ts.adaptive", "Adaptive forecasting across regime changes", category="Time series",
      outputs=[Port("metrics", "metrics")],
      params=[Param("n", "int", 900, minimum=60), Param("seed", "int", 0), Param("season_length", "int", 12),
              Param("horizon", "int", 3, minimum=1), Param("start", "int", 60, minimum=1)],
      calls="pinneapple_systems.time_series.AdaptiveForecaster(default_experts(...)).run + errors / coverage / "
            "switches")
def adaptive(ctx, inputs, params):
    """Experts weighted by their recent errors on a seasonal -> trend -> random-walk series: the ensemble's MAE
    against every expert, the 90 % interval coverage and when the leading model changes
    (examples/time_series/adaptive_forecasting.py)."""
    from pinneapple_systems.time_series import AdaptiveForecaster, default_experts
    y = regime_series(int(params["n"]), int(params["seed"]))
    run = AdaptiveForecaster(default_experts(season_length=int(params["season_length"])),
                             horizon=int(params["horizon"])).run(y, start=int(params["start"]))
    err = run.errors(h=1)
    for name, e in err.items():
        ctx.metric(f"mae.{name}", float(e["mae"]))
    ctx.metric("coverage_90", float(run.coverage(1)))
    ctx.metric("switches", len(run.switches()))
    ctx.table("MAE at h = 1", ["model", "MAE"], sorted([[k, v["mae"]] for k, v in err.items()], key=lambda r: r[1]))
    return {"metrics": ctx.report["metrics"]}


@node("ts.naive_baselines", "Naive forecasting baselines (quickstart)", category="Time series",
      outputs=[Port("metrics", "metrics")],
      params=[Param("T", "int", 4000, minimum=200), Param("seed", "int", 7), Param("input_len", "int", 96),
              Param("horizon", "int", 24), Param("season", "int", 24)],
      calls="pinneapple_systems.time_series.TimeSeriesSpec + TSDataModule + NaiveForecaster / SeasonalNaiveForecaster "
            "/ DriftForecaster; metrics_ext.point.mae / rmse / smape")
def naive_baselines(ctx, inputs, params):
    """examples/time_series/00_quickstart.py: a seasonal + trend + noise series, time-ordered train / validation
    windows, the naive, seasonal-naive and drift baselines and their errors."""
    import torch

    from pinneapple_systems.time_series import (
        DriftForecaster,
        NaiveForecaster,
        SeasonalNaiveForecaster,
        TimeSeriesSpec,
        TSDataModule,
    )
    from pinneapple_systems.time_series.metrics_ext.point import mae, rmse, smape
    rng = np.random.default_rng(int(params["seed"]))
    T = int(params["T"])
    t = np.arange(T, dtype=float)
    y = 0.0008 * t + 0.8 * np.sin(2 * math.pi * t / 24.0) + 0.3 * np.sin(2 * math.pi * t / 168.0) + \
        0.25 * rng.standard_normal(T)
    spec = TimeSeriesSpec(input_len=int(params["input_len"]), horizon=int(params["horizon"]), stride=1, target_offset=0)
    _, val = TSDataModule(series=torch.tensor(y.reshape(T, 1), dtype=torch.float32), spec=spec, batch_size=128,
                          val_ratio=0.2).make_loaders()
    Xv = torch.cat([xb for xb, _ in val])[..., 0].numpy()
    Yv = torch.cat([yb for _, yb in val])[..., 0].numpy()
    H = spec.horizon
    preds = {"naive": [NaiveForecaster().fit(h).predict(horizon=H) for h in Xv],
             f"seasonal_naive_{params['season']}": [SeasonalNaiveForecaster(season_length=int(params["season"]))
                                                    .fit(h).predict(horizon=H) for h in Xv],
             "drift": [DriftForecaster().fit(h).predict(horizon=H) for h in Xv]}
    for name, p in preds.items():
        p = np.stack(p)
        ctx.metric(f"{name}.mae", float(mae(Yv, p)))
        ctx.metric(f"{name}.rmse", float(rmse(Yv, p)))
        ctx.metric(f"{name}.smape", float(smape(Yv, p)))
    return {"metrics": ctx.report["metrics"]}


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="geo.earth_system", title="Earth-system blocks: shallow water on the sphere, 4D-Var, infiltration",
        domain="geophysics", example="geophysics",
        keywords={"williamson": 3, "rossby-haurwitz": 3, "sphere": 1.5, "4d-var": 3, "4dvar": 3, "lorenz-96": 2,
                  "richards": 3, "infiltration": 2, "data assimilation": 2}, min_score=3, nodes=("geo.earth_system",),
        references=("evensen1994enkf",), template="geophysics/01_sphere_and_assimilation",
        description="examples/geophysics/01_sphere_and_assimilation.py"))
    register_capability(Capability(
        id="core.torsion_optimizer", title="Constrained shape optimization compared across six methods",
        domain="optimization", example="physics_optimizer",
        keywords={"torsional rigidity": 3, "shape optimization": 2.5, "constrained optimization": 2.5,
                  "augmented lagrangian": 2, "slsqp": 2, "adjoint": 1.5}, min_score=3,
        nodes=("core.torsion_optimizer",), references=("brenner2008fem",),
        template="physics_optimizer/01_constrained_shape_six_methods",
        description="examples/physics_optimizer/01_constrained_shape_six_methods.py"))
    register_capability(Capability(
        id="core.hybrid_fem", title="Coarse FEM with a learned correction (hybrid model)", domain="hybrid models",
        example="physics_module", keywords={"coarse fem": 3, "neural correction": 3, "hybrid model": 2,
                                            "corrector": 2, "solver in the loop": 2.5}, min_score=3,
        nodes=("core.hybrid_fem",), references=("willard2022physicsml", "brenner2008fem"),
        template="physics_module/01_coarse_fem_plus_neural_correction", quick=True,
        description="examples/physics_module/01_coarse_fem_plus_neural_correction.py"))
    register_capability(Capability(
        id="ts.adaptive", title="Adaptive forecasting across regime changes", domain="time series",
        example="time_series", keywords={"adaptive forecast": 3, "regime change": 3, "regime": 1.5, "experts": 1.5,
                                         "ensemble forecast": 2}, min_score=3, nodes=("ts.adaptive",),
        references=("hyndman2021fpp",), template="time_series/adaptive_forecasting",
        description="examples/time_series/adaptive_forecasting.py"))


_register_capabilities()

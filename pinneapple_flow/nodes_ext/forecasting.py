"""Forecasting nodes: the ENSO solver checks of ``examples/enso_forecasting`` (no data needed) and the
``forecast_lab`` studies of ``examples/near_random_forecasting`` (one case: predictability, model zoo, surrogate-trained
twins; the overfitting study; the Mackey-Glass Lyapunov convergence). The ENSO model zoo itself is the existing
``forecast.enso`` node. Metric names follow the keys of the examples' results JSON."""
from __future__ import annotations

import time

import numpy as np

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node


def _cases():
    try:
        from pinneapple_systems.time_series.forecast_lab.presets import CASES
        return list(CASES)
    except Exception:  # noqa: BLE001 - an older checkout
        return ["lorenz63_noise0.1"]


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)


@node("enso.solver_checks", "ENSO model checks against theory", category="Time series",
      outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      calls="pinneapple_systems.time_series.enso.pipeline.solver_checks + enso.plots.plot_validation")
def enso_solver_checks(ctx, inputs, params):
    """The intermediate coupled model's shallow-water solver (Kelvin and Rossby wave speeds, convergence order, the
    Gill steady response against the analytic solution) and the recharge / delayed oscillators against their
    eigenvalues (examples/enso_forecasting/run_study.py, results/solver_checks.json). No observations needed."""
    from pinneapple_systems.time_series.enso import pipeline, plots
    sc = pipeline.solver_checks()
    k, r = sc["kelvin"], sc["rossby"]
    ctx.metric("kelvin_speed", k["speed_measured"], "shallow-water solver, crossing time of the basin")
    ctx.metric("kelvin_speed_theory", k["speed_theory"])
    ctx.metric("kelvin_rel_error", k["rel_error"])
    ctx.metric("rossby_speed", r["speed_measured"])
    ctx.metric("rossby_rel_error", r["rel_error"])
    for i, o in enumerate(sc["kelvin_convergence"].get("observed_order") or []):
        ctx.metric(f"kelvin_observed_order_{i}", o)
    for row in sc["gill"]["rows"]:
        ctx.metric(f"gill_rel_l2_u_nlon{row['nlon']}", row["rel_l2_u"], "Gill (1980) steady response vs analytic")
    ro = sc["recharge_oscillator"]
    ctx.metric("recharge_period_theory", ro["period_theory"])
    for row in ro["rows"]:
        ctx.metric(f"recharge_period_rel_error_sub{row['sub']}", row["period_rel_error"], "recharge oscillator "
                                                                                          "period vs theory")
    for row in sc["delayed_oscillator"]["rows"]:
        ctx.metric(f"delayed_period_rel_error_dt{row['dt']:g}", row["period_rel_error"])
    p = plots.plot_validation(sc["kelvin"], sc["gill"], sc["kelvin_convergence"], ro, str(ctx.path("solver_checks.png")))
    ctx.report["figures"].append({"path": p, "name": "solver_checks", "caption": "solver checks against theory"})
    from ..runtime import artifact
    ctx.summary(f"Kelvin {k['speed_measured']:.4f} vs {k['speed_theory']:.4f}, Rossby error {r['rel_error']:+.3f}")
    return {"metrics": ctx.report["metrics"], "figure": artifact(p, "png")}


@node("forecast_lab.case", "Near-random series: predictability, model zoo, surrogate twins", category="Time series",
      outputs=[Port("metrics", "metrics"), Port("figure", "figure", required=False)],
      params=[Param("case", "enum", "lorenz63_noise0.1", choices=_cases(),
                    help="pinneapple_systems.time_series.forecast_lab.presets.CASES (some real cases download data)"),
              Param("n", "int", 6000, minimum=200, maximum=100000, help="series length"),
              Param("budget", "int", 8, minimum=1, maximum=64, help="tuning configurations per model"),
              Param("n_surrogates", "int", 99, minimum=9, maximum=999), Param("seed", "int", 0, minimum=0),
              Param("models", "list", [], help="report these models' test RMSE (empty: every model)")],
      calls="pinneapple_systems.time_series.forecast_lab.pipeline.run_case (presets.case, predictability_report, "
            "study.run_zoo, surrogate_trained, fan_panels); forecast_lab.plots.plot_skill_vs_horizon")
def forecast_case(ctx, inputs, params):
    """examples/near_random_forecasting/run_study.py for one case: predictability first, then every model under the
    purged walk-forward protocol with an untouched test period."""
    from pinneapple_systems.time_series.forecast_lab import pipeline
    t0 = time.perf_counter()
    try:
        res = pipeline.run_case(params["case"], n=int(params["n"]), budget=int(params["budget"]),
                                seed=int(params["seed"]), n_surrogates=int(params["n_surrogates"]), log=ctx.log)
    except OSError as exc:
        ctx.not_run(f"case {params['case']} needs its data download ({exc})")
    ctx.metric("seconds", round(time.perf_counter() - t0, 1))
    for k in ("lyapunov", "lyapunov_time", "noise_level"):
        if _num((res.get("truth") or {}).get(k)):
            ctx.metric(f"truth.{k}", res["truth"][k])
    for k, v in (res.get("predictability") or {}).items():
        if _num(v):
            ctx.metric(f"predictability.{k}", v)
    models = res["zoo"]["models"]
    H = list(res["zoo"]["protocol"]["horizons"])
    want = params["models"] or list(models)
    rows = []
    for name in want:
        m = models.get(name)
        if not m or "test" not in m:
            continue
        rmse = m["test"].get("rmse") or []
        for h, v in zip(H, rmse, strict=False):
            if _num(v):
                ctx.metric(f"zoo.{name}.test_rmse_h{h}", v, "untouched test period")
        sk = m.get("skill_vs_best_baseline") or []
        rows.append([name, m.get("family"), rmse[:4], sk[:4] if isinstance(sk, list) else sk])
    ctx.table(f"{res['title']}: test RMSE (first horizons {H[:4]}) and skill vs the best baseline",
              ["model", "family", "RMSE", "skill vs best baseline"], rows)
    ctx.metric("best_baseline_by_val", str(res["zoo"].get("best_baseline_by_val")))
    out = {"metrics": ctx.report["metrics"]}
    try:
        from pinneapple_systems.time_series.forecast_lab import plots
        p = plots.plot_skill_vs_horizon([res], str(ctx.path("skill_vs_horizon.png")))
        ctx.report["figures"].append({"path": p, "name": "skill_vs_horizon", "caption": "skill against horizon"})
        from ..runtime import artifact
        out["figure"] = artifact(p, "png")
    except Exception as exc:  # noqa: BLE001 - a figure never fails the study
        ctx.warn(f"no skill figure: {exc}")
    ctx.summary(f"{res['title']}: {len(models)} models, n = {res['n']}")
    return out


@node("forecast_lab.lyapunov", "Mackey-Glass Lyapunov exponent: convergence", category="Time series",
      outputs=[Port("metrics", "metrics")],
      params=[Param("dt_ints", "list", [0.1, 0.05], help="integration steps"),
              Param("total_times", "list", [10000.0, 50000.0, 200000.0], help="run lengths"),
              Param("seeds", "list", [0, 1, 2])],
      calls="pinneapple_systems.time_series.forecast_lab.pipeline.lyapunov_study (synthetic.mackey_glass_lyapunov)")
def lyapunov(ctx, inputs, params):
    """Benettin estimate of the largest Lyapunov exponent of Mackey-Glass (tau = 17) against the step, the length
    of the run and the seed (results/lyapunov_mackey_glass.json)."""
    from pinneapple_systems.time_series.forecast_lab import pipeline
    res = pipeline.lyapunov_study(dt_ints=tuple(map(float, params["dt_ints"])),
                                  total_times=tuple(map(float, params["total_times"])),
                                  seeds=tuple(map(int, params["seeds"])), log=ctx.log)
    rows = []
    for r in res["rows"]:
        tag = f"dt{r['dt_int']:g}_T{r['total_time']:g}"
        ctx.metric(f"{tag}.mean", r["mean"], f"mean over seeds {list(params['seeds'])}")
        for i, v in enumerate(r["values"]):
            ctx.metric(f"{tag}.seed{params['seeds'][i]}", v)
        rows.append([r["dt_int"], r["total_time"], r["mean"], r.get("std")])
    ctx.table("Mackey-Glass largest Lyapunov exponent", ["dt", "T", "mean", "std"], rows)
    return {"metrics": ctx.report["metrics"]}


@node("forecast_lab.overfitting", "How cross-validation schemes fool you (overfitting study)", category="Time series",
      outputs=[Port("metrics", "metrics")],
      params=[Param("case", "enum", "lorenz63_noise0.1", choices=_cases()),
              Param("n", "int", 6000, minimum=200, maximum=100000), Param("seed", "int", 0, minimum=0)],
      calls="pinneapple_systems.time_series.forecast_lab.pipeline.run_overfitting (presets.overfitting_variants, "
            "study.overfitting_study)")
def overfitting(ctx, inputs, params):
    """Shuffled k-fold, walk-forward without purge, purged walk-forward and purged k-fold scores of deliberately
    overfitting and regularised variants, against the untouched test score."""
    from pinneapple_systems.time_series.forecast_lab import pipeline
    try:
        st = pipeline.run_overfitting(params["case"], n=int(params["n"]), seed=int(params["seed"]), log=ctx.log)
    except OSError as exc:
        ctx.not_run(f"case {params['case']} needs its data download ({exc})")
    variants = st.get("variants") or st.get("rows") or {}
    if isinstance(variants, dict):
        for name, v in variants.items():
            for k, x in (v.items() if isinstance(v, dict) else []):
                if _num(x):
                    ctx.metric(f"{name}.{k}", x)
    if not ctx.report["metrics"]:
        raise StepFailed("the overfitting study returned no numbers")
    return {"metrics": ctx.report["metrics"]}


# ------------------------------------------------------------------------------------------- planner capabilities
def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="enso.solver_checks", title="ENSO models checked against theory (waves, Gill, oscillators)",
        keywords={"enso": 2, "el niño": 2, "el nino": 2, "kelvin wave": 2.5, "rossby": 2, "gill": 2.5,
                  "recharge oscillator": 2.5, "delayed oscillator": 2, "solver check": 2, "verify": 1, "check": 0.5},
        requires_all=(("kelvin", "rossby", "gill", "oscillator", "solver", "check", "verif"),), min_score=3,
        nodes=("enso.solver_checks", "examples.compare"), domain="time series", example="enso_forecasting",
        uncited="Jin (1997), Gill (1980) and Matsuno (1966) are cited by the ENSO guide but not in the flow "
                "reference registry", template="enso_forecasting/solver_checks",
        description="the intermediate coupled model's solver and the oscillators against their closed forms"))
    register_capability(Capability(
        id="enso.forecast", title="El Niño forecast: Niño 3.4 with the ENSO model zoo", domain="time series",
        keywords={"enso": 3, "el niño": 3, "el nino": 3, "niño 3.4": 3, "nino 3.4": 3, "la niña": 2, "la nina": 2,
                  "sea surface temperature": 1.5, "pacific": 1}, min_score=3,
        nodes=("forecast.enso", "enso.solver_checks", "examples.compare"), example="enso_forecasting",
        inputs=("the OISST / WOD / ERA5 observations (downloaded once into the data cache)",),
        references=("diebold1995dm", "evensen1994enkf"), template="enso_forecasting/forecast", quick=True,
        description="baselines, oscillators, LIM, CNN, hybrids under purged walk-forward with an untouched test "
                    "period; DM tests (examples/enso_forecasting/run_study.py)"))
    register_capability(Capability(
        id="forecast_lab.case", title="Forecast a near-random series without fooling yourself", domain="time series",
        keywords={"near-random": 3, "near random": 3, "chaotic": 2, "lorenz": 2.5, "mackey": 2.5, "white noise": 2,
                  "random walk": 2, "predictability": 2.5, "surrogate test": 2, "walk-forward": 2, "forecast": 1,
                  "time series": 1}, min_score=3,
        nodes=("forecast_lab.case", "forecast_lab.lyapunov", "examples.compare"), example="near_random_forecasting",
        references=("theiler1992surrogate", "diebold1995dm", "hyndman2021fpp"),
        template="near_random_forecasting/lorenz63_case", quick=True,
        description="predictability first, then the model zoo under purged walk-forward "
                    "(examples/near_random_forecasting/run_study.py)"))


_register_capabilities()

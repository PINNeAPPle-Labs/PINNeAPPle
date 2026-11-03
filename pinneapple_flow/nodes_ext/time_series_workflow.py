"""The standard time-series prediction and modeling workflow (:mod:`pinneapple_systems.time_series.workflow`, #1197):
analysis (missing values, outliers, seasonality, stationarity, predictability horizon, leakage), baselines every
model must beat, statistical / learned / physical models on one rolling-origin backtest, skill against the
baselines and conformal intervals with their measured coverage."""
from __future__ import annotations

from ..registry import Param, Port, node

_MODELS = ["persistence", "seasonal_naive", "mean", "arima", "ets", "gbm", "tcn", "ode"]


@node("ts.workflow", "Time series: analysis, baselines, models, rolling backtest, conformal intervals",
      category="Time series", outputs=[Port("metrics", "metrics")],
      params=[Param("series", "enum", "seasonal", choices=["seasonal", "arma", "lorenz"],
                    help="bundled series with known ground truth"),
              Param("csv_path", "str", "", help="instead of a bundled series: a CSV (one row per time step)"),
              Param("value_column", "str", "", help="CSV column of the target"),
              Param("covariates", "list", [], help="CSV columns of exogenous covariates"),
              Param("horizons", "list", [1, 2, 4, 8, 16], help="forecast steps"),
              Param("models", "list", list(_MODELS)),
              Param("n_folds", "int", 3, minimum=1),
              Param("test_fraction", "float", 0.3, minimum=0.05, maximum=0.8),
              Param("period", "int", 0, minimum=0, help="seasonal period (0: from the spectrum)"),
              Param("level", "float", 0.9, minimum=0.5, maximum=0.99, help="conformal interval level"),
              Param("coverage_tolerance", "float", 0.05, minimum=0.0, maximum=0.5),
              Param("leak_corr", "float", 0.98, minimum=0.5, maximum=1.0, advanced=True,
                    help="a covariate this correlated with the target or a future target is dropped"),
              Param("epochs", "int", 40, minimum=1, advanced=True, help="TCN training epochs"),
              Param("seed", "int", 0, advanced=True)],
      calls="pinneapple_systems.time_series.workflow.time_series_workflow")
def ts_workflow(ctx, inputs, params):
    from pinneapple_systems.time_series.workflow import as_tables, time_series_workflow
    r = time_series_workflow(series=str(params["series"]), csv_path=str(params.get("csv_path") or ""),
                             value_column=str(params.get("value_column") or ""),
                             covariates=list(params.get("covariates") or []),
                             horizons=[int(h) for h in params["horizons"]], models=list(params["models"]),
                             n_folds=int(params["n_folds"]), test_fraction=float(params["test_fraction"]),
                             period=int(params["period"]), level=float(params["level"]),
                             coverage_tolerance=float(params["coverage_tolerance"]),
                             leak_corr=float(params["leak_corr"]), epochs=int(params["epochs"]),
                             seed=int(params["seed"]))
    a = r["analysis"]
    ctx.metric("data", r["data"]["name"], r["data"]["description"])
    ctx.metric("missing", sum(v["missing"] for v in a["missing"].values()), "filled by linear interpolation")
    ctx.metric("outliers", a["outliers"]["count"], a["outliers"]["rule"])
    ctx.metric("period", a["seasonality"]["period"], f"autocorrelation {a['seasonality']['acf_at_period']:.3f}")
    st = a["stationarity"]
    ctx.metric("stationarity", st["verdict"], f"ADF {st['adf']['statistic']:.3f} (5 % {st['adf']['critical']['5%']:.3f}),"
                                              f" KPSS {st['kpss']['statistic']:.3f} (5 % 0.463)")
    p = a["predictability"]
    ctx.metric("permutation_entropy", p["permutation_entropy"], "order 5, normalised (1 = white noise)")
    if p.get("lyapunov") is not None:
        ctx.metric("lyapunov_time", p["lyapunov_time"], f"1 / lambda, lambda = {p['lyapunov']:.4g} (Rosenstein)")
    ctx.metric("skill_horizon", p["skill_horizon_steps"], "largest tested horizon where the recommended model beats "
                                                          "the mean")
    if r["data"]["dropped_columns"]:
        ctx.metric("leakage.dropped", r["data"]["dropped_columns"])
    t = as_tables(r)
    ctx.table("backtest (per horizon " + " / ".join(str(h) for h in r["horizons"]) + ")",
              ["model", "beats baselines", "RMSE", "skill vs best baseline", f"{r['level']:.0%} coverage",
               "fit + predict s"], t["backtest"])
    if t["skipped"]:
        ctx.table("models not run", ["model", "reason"], t["skipped"])
    ctx.table("checks", ["check", "status", "detail"], t["checks"])
    ctx.metric("recommended", r["recommended"], "lowest mean RMSE among the models that beat the baselines")
    ctx.metric("passed", r["passed"])
    ctx.summary(r["summary"])
    return {"metrics": ctx.report["metrics"]}


def build_flow(**params):
    from ..planner.build import FlowBuilder
    title = "Time-series prediction: analysis, baselines, models, backtest and calibrated intervals"
    fb = FlowBuilder(title, "Missing values, outliers, seasonality, stationarity, the predictability horizon and "
                            "leakage checks; persistence / seasonal-naive / mean baselines; ARIMA, ETS, gradient "
                            "boosting, a TCN and a fitted ODE when one exists, on one rolling-origin backtest with "
                            "skill against the baselines and split-conformal intervals with their coverage "
                            "(pinneapple_systems.time_series.workflow).")
    fb.add("ts", "ts.workflow", dict(params), label="Time-series workflow")
    fb.add("report", "export.report", {"title": title}, label="Report")
    fb.connect("ts", "metrics", "report", "items")
    return fb


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="ts.workflow", title="Time-series prediction and modeling (standard workflow)", domain="time series",
        keywords={"time series": 2, "série temporal": 2, "séries temporais": 2, "forecast": 2, "previsão": 2,
                  "prever": 1, "backtest": 3, "rolling origin": 3, "origem móvel": 3, "arima": 3, "arma": 2,
                  "exponential smoothing": 2, "suavização exponencial": 2, "conformal": 3, "conformal prediction": 3,
                  "intervalo conformal": 3, "baselines": 1, "seasonal naive": 2, "sazonalidade": 1,
                  "seasonality": 1, "stationarity": 1, "estacionariedade": 1, "data leakage": 2,
                  "vazamento de dados": 2},
        min_score=4, nodes=("ts.workflow",), build=lambda text, det: build_flow(),
        inputs=("a series or a table (target column, optional covariates)", "the forecast horizons",
                "the interval level"),
        uncited="the protocol (rolling-origin backtest, split-conformal intervals) is described in the library "
                "module; it is not in the flow reference file yet",
        description="pinneapple_systems.time_series.workflow.time_series_workflow"))


_register_capabilities()

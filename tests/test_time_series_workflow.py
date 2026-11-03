"""The time-series workflow (#1197): ARMA order selection, stationarity tests against statsmodels, the Lyapunov
time of Lorenz-63, leakage and outlier checks, skill against the baselines and conformal coverage."""
import numpy as np
import pytest

from pinneapple_systems.time_series.statistical import (
    adf_test,
    arma_sample,
    conformal_quantile,
    hampel,
    kpss_test,
    select_arma,
)
from pinneapple_systems.time_series.workflow import load_series, time_series_workflow

QUICK = ("persistence", "seasonal_naive", "mean", "arima", "ets")


def test_bic_selects_the_true_arma_order():
    hits = sum(select_arma(arma_sample(1000, [0.75, -0.5], [0.6], mu=5.0, rng=np.random.default_rng(s)))["order"]
               == (2, 1) for s in range(10))
    assert hits == 10


def test_stationarity_tests_match_statsmodels():
    sm = pytest.importorskip("statsmodels.tsa.stattools")
    y = arma_sample(800, [0.5], [0.3], rng=np.random.default_rng(2))
    rw = np.cumsum(np.random.default_rng(3).standard_normal(800))
    for z in (y, rw):
        a, b = adf_test(z), sm.adfuller(z, autolag="AIC", regression="c")
        assert abs(a["statistic"] - b[0]) < 1e-8 and a["lags"] == b[2]
        k = kpss_test(z)
        assert abs(k["statistic"] - sm.kpss(z, regression="c", nlags="legacy")[0]) < 1e-10
    assert adf_test(y)["unit_root_rejected_5pct"] and not adf_test(rw)["unit_root_rejected_5pct"]
    assert kpss_test(rw)["stationarity_rejected_5pct"] and not kpss_test(y)["stationarity_rejected_5pct"]


def test_conformal_quantile_and_hampel():
    r = np.arange(1, 100, dtype=float)                   # n = 99: the ceil(100 * 0.9) = 90th smallest
    assert conformal_quantile(r, 0.1) == 90.0
    y = 0.01 * np.arange(500) + 0.05 * np.sin(1.7 * np.arange(500))      # bounded noise: no false flag
    y[[100, 300]] += 10
    h = hampel(y)
    assert h["index"] == [100, 300] and abs(h["cleaned"][100] - 1.0) < 0.1      # the window median of a trend


def test_arma_series_selects_the_order_and_beats_the_baselines():
    r = time_series_workflow(series="arma", models=QUICK)
    assert r["passed"], [c for c in r["checks"] if c["status"] == "fail"]
    assert r["recommended"] == "arima"
    assert r["models"]["arima"]["info"]["order"] == [2, 0, 1]
    assert r["models"]["arima"]["skill"][0] > 0.25


def test_lorenz_lyapunov_time_matches_the_equations_and_the_ode_wins_short_range():
    r = time_series_workflow(series="lorenz", models=("persistence", "mean", "arima", "ode"))
    assert r["passed"], [c for c in r["checks"] if c["status"] == "fail"]
    p = r["analysis"]["predictability"]
    truth = load_series("lorenz")["truth"]
    assert abs(p["lyapunov_time"] - truth["lyapunov_time"]) / truth["lyapunov_time"] < 0.15
    assert r["models"]["ode"]["skill"][0] > 0.7 and r["models"]["ode"]["beats_baselines"]


def test_seasonal_series_drops_the_leaky_column_finds_outliers_and_covers():
    r = time_series_workflow(series="seasonal", models=QUICK)
    assert r["passed"], [c for c in r["checks"] if c["status"] == "fail"]
    assert r["data"]["dropped_columns"] == ["y_next_reported"]
    assert r["analysis"]["seasonality"]["period"] == 24
    cov = np.array(r["models"][r["recommended"]]["coverage"])
    assert np.all(np.abs(cov - 0.9) <= 0.05)


def test_learned_models_run_when_their_libraries_exist():
    pytest.importorskip("sklearn")
    r = time_series_workflow(series="seasonal", models=("persistence", "mean", "gbm", "tcn"), epochs=10)
    assert {"gbm", "tcn"} <= set(r["models"])
    assert r["models"]["gbm"]["beats_baselines"]


def test_csv_input_and_bad_inputs(tmp_path):
    y = arma_sample(600, [0.6], [], mu=2.0, rng=np.random.default_rng(0))
    x = np.random.default_rng(1).standard_normal(600)
    p = tmp_path / "s.csv"
    p.write_text("t,load,temp\n" + "".join(f"{i},{a:.5f},{b:.5f}\n" for i, (a, b) in enumerate(zip(y, x))))
    r = time_series_workflow(csv_path=str(p), value_column="load", covariates=["temp"], models=QUICK)
    assert r["models"]["arima"]["info"]["order"][0] >= 1
    with pytest.raises(ValueError):
        time_series_workflow(csv_path=str(p), value_column="nope")
    with pytest.raises(ValueError):
        time_series_workflow(series="nope")
    with pytest.raises(ValueError):
        time_series_workflow(series="arma", horizons=[0])


def test_the_planner_proposes_the_workflow_in_portuguese_and_english():
    import pinneapple_flow.nodes_ext  # noqa: F401
    from pinneapple_flow.planner.extensions import match
    for text in ("previsão de série temporal com backtest e intervalo conformal",
                 "forecast this time series with a rolling origin backtest and conformal intervals"):
        assert match(text)[0][1].id == "ts.workflow"

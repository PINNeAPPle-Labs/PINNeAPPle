"""The ENSO forecasting study: the protocol, the model zoo, and one call per model (so the zoo runs in parallel
processes and is merged by :func:`~pinneapple_systems.time_series.forecast_lab.study.merge_zoo`).

* :func:`protocol` -- horizons 1-12, 15, 18, 21, 24 months; the test period starts at ``test_start`` (default
  2012-01) and is untouched until the final scoring; the last 15 % of the development period is the validation
  block; purged walk-forward tuning on the rest (forecast_lab rules).
* :func:`zoo` -- the specs by family: baselines, data-driven (GBM, MLP, LSTM, TCN, transformer, LIM, CNN),
  data + physics (CNN pre-trained on the intermediate model, PINN-identified recharge oscillator with and without a
  learned closure, PINN extrapolation, MLP with a recharge-oscillator residual loss, GBM with physics features),
  physical (recharge oscillator deterministic / stochastic / seasonal with multiplicative noise, the RO fitted by
  multiple shooting, the delayed oscillator, the intermediate model with SST nudging), hybrid (RO + GBM residual,
  UDE closure of the RO, the intermediate model with EnKF, a multi-model ensemble).
* :func:`run_model` -- tune / fit / score one spec; also the forecasts from the case-study origins.
* :func:`event_hindcast` -- refit a spec on data that end before an event (1997/98) and forecast it.
"""
from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

import numpy as np

from ..forecast_lab import presets as FP
from ..forecast_lab.base import SeriesData, forecast_origins
from ..forecast_lab.hybrid import ResidualCorrectionForecaster, UDEForecaster
from ..forecast_lab.learned import GBMForecaster, PhysicsLossMLP
from ..forecast_lab.physical import DelayODEForecaster, ODEForecaster, PhysicsFeatures
from ..forecast_lab.study import ModelSpec, Protocol, run_zoo
from . import forecasters as F
from . import oscillators as O
from .pinn import PINNExtrapolationForecaster, PINNRechargeForecaster

__all__ = ["HORIZONS", "protocol", "zoo", "spec", "run_model", "event_hindcast", "CASE_ORIGINS"]

HORIZONS = list(range(1, 13)) + [15, 18, 21, 24]
#: origins of the case-study forecasts (plumes): month -> label
CASE_ORIGINS = {"2015/16": ["2015-02", "2015-05", "2015-08"], "2023/24": ["2023-02", "2023-05", "2023-08"],
                "1997/98": ["1997-02", "1997-05", "1997-08"]}


def _grid(**axes) -> list[dict[str, Any]]:
    keys = list(axes)
    return [dict(zip(keys, v, strict=False)) for v in itertools.product(*axes.values())]


def protocol(months: list[str], test_start: str = "2012-01", horizons=HORIZONS, min_history: int = 36) -> Protocol:
    n = len(months)
    return Protocol(list(horizons), test_frac=1 - months.index(test_start) / n, val_frac=0.15, n_folds=3,
                    fold_frac=0.12, min_history=min_history, test_stride=1)


def _th(make, name, family):
    """A forecast_lab forecaster run on the (nino34, wwv) pair only."""
    return lambda **k: F.ColumnSubset(make(**k), ["nino34", "wwv"], name=name, family=family)


def _ro_ode():
    return ODEForecaster(F.ro_rhs, {"R": -0.05, "F1": 0.02, "F2": 1.5, "eps": 0.03}, name="ro_multiple_shooting",
                         sub=4, fit_steps=3)


def zoo(quick: bool = False) -> list[ModelSpec]:
    """The model zoo. ``quick``: tiny epochs / members for smoke runs."""
    ep = 8 if quick else 150
    pe = 3 if quick else 15
    mem = 1 if quick else 5
    pinn_ep = 200 if quick else 4000
    specs = FP.baseline_specs(arima=True)
    specs += FP.data_specs(lags=(6, 12), seq_lags=24, epochs=20 if quick else 80, patience=pe, max_train=None)
    specs += [
        ModelSpec("lim", "data", lambda **k: F.LIMForecaster(**k),
                  _grid(n_eofs=[6, 10, 15], extra_cols=[("wwv",), ("h_west", "h_east")])),
        ModelSpec("cnn", "data", lambda **k: F.MapCNNForecaster(epochs=ep, patience=pe, n_members=mem, **k),
                  _grid(n_lags=[3, 6], channels=[(8, 16), (16, 32)], weight_decay=[1e-3, 1e-2])),
        # data + physics
        ModelSpec("cnn_pretrained", "data+physics",
                  lambda **k: F.MapCNNForecaster(epochs=ep, patience=pe, n_members=mem, family="data+physics",
                                                 pretrain={"years": 20 if quick else 300, "members": 2 if quick else 6},
                                                 **k)),
        ModelSpec("pinn_ro", "data+physics", lambda: PINNRechargeForecaster(epochs=pinn_ep)),
        ModelSpec("pinn_ro_closure", "data+physics", lambda: PINNRechargeForecaster(epochs=pinn_ep, closure=True)),
        ModelSpec("pinn_extrapolation", "data+physics",
                  lambda: PINNExtrapolationForecaster(epochs=150 if quick else 2500)),
        ModelSpec("mlp_physics_loss", "data+physics",
                  _th(lambda **k: PhysicsLossMLP(_ro_ode(), epochs=ep, patience=pe, **k), "mlp_physics_loss",
                      "data+physics"),
                  _grid(lam=[0.0, 0.1, 1.0], n_lags=[6], hidden=[64], weight_decay=[1e-3])),
        ModelSpec("gbm_physics_features", "data+physics",
                  _th(lambda **k: GBMForecaster(features=PhysicsFeatures(_ro_ode()), name="gbm_physics_features",
                                                family="data+physics", **k), "gbm_physics_features", "data+physics"),
                  _grid(n_lags=[6, 12], max_iter=[100, 300], max_leaf_nodes=[7, 15], min_samples_leaf=[20])),
        # physical
        ModelSpec("ro", "physical", lambda: F.RechargeOscillatorForecaster(stochastic=False, name="ro")),
        ModelSpec("ro_stochastic", "physical", lambda: F.RechargeOscillatorForecaster(name="ro_stochastic")),
        ModelSpec("ro_seasonal_multiplicative", "physical",
                  lambda: F.RechargeOscillatorForecaster(seasonal=("R", "F1", "F2", "eps"), multiplicative=True,
                                                         name="ro_seasonal_multiplicative")),
        ModelSpec("ro_multiple_shooting", "physical", _th(lambda: _ro_ode(), "ro_multiple_shooting", "physical")),
        ModelSpec("delayed_oscillator", "physical",
                  lambda **k: DelayODEForecaster(O.do_rhs, {"a": 0.2, "b": 0.3, "c": 0.05}, dt_int=0.25,
                                                 name="delayed_oscillator", **k),
                  _grid(tau=[3.0, 6.0, 9.0, 12.0])),
        ModelSpec("im_nudged", "physical", lambda: F.IntermediateModelForecaster(init="nudged")),
        # hybrid
        ModelSpec("ro+gbm", "hybrid",
                  lambda **k: ResidualCorrectionForecaster(F.RechargeOscillatorForecaster(stochastic=False), "gbm",
                                                           name="ro+gbm", **k),
                  _grid(n_lags=[6, 12], max_iter=[100, 300], max_leaf_nodes=[7, 15])),
        ModelSpec("ude_ro", "hybrid",
                  _th(lambda **k: UDEForecaster(_ro_ode(), window=6, epochs=ep, patience=pe, name="ude_ro", **k),
                      "ude_ro", "hybrid"),
                  _grid(hidden=[16, 32], weight_decay=[1e-4, 1e-2])),
        ModelSpec("im_enkf", "hybrid",
                  lambda: F.IntermediateModelForecaster(init="enkf", n_members=4 if quick else 10,
                                                        enkf_members=8 if quick else 24)),
        ModelSpec("ensemble", "hybrid",
                  lambda: F.EnsembleForecaster([F.RechargeOscillatorForecaster(seasonal=("R", "F1", "F2", "eps"),
                                                                               multiplicative=True),
                                                F.LIMForecaster(n_eofs=10, extra_cols=("h_west", "h_east")),
                                                GBMForecaster(n_lags=12, max_iter=300, max_leaf_nodes=7)],
                                               name="ensemble(ro_seasonal+lim+gbm)")),
    ]
    return specs


def spec(name: str, quick: bool = False) -> ModelSpec:
    for s in zoo(quick):
        if s.name == name:
            return s
    raise KeyError(name)


def run_model(name: str, data: SeriesData, prot: Protocol, *, budget: int = FP.BUDGET, quick: bool = False,
              case_origins: dict[str, list[str]] | None = None, log: Callable[[str], None] = print,
              seed: int = 0) -> dict[str, Any]:
    """Tune, fit and score one spec under ``prot`` (forecasts kept for :func:`merge_zoo`), plus its forecasts from
    the case-study origins that lie after the development period (``_cases``: label -> origin -> Forecast)."""
    s = spec(name, quick)
    res = run_zoo(data, [s], prot, budget=budget, keep_forecasts=[name], keep_models=True, log=log, seed=seed)
    model = res["_models"].pop(name, None)
    res["_cases"] = {}
    months = data.meta["months"]
    dev_end = prot.final_split(len(data))["dev_end"]
    if model is not None:
        for label, ors in (case_origins or CASE_ORIGINS).items():
            idx = [months.index(m) for m in ors if m in months and months.index(m) >= dev_end]
            if idx:
                res["_cases"][label] = {"origins": idx, "forecast": model.predict(data, np.asarray(idx))}
        if hasattr(model, "inner"):
            model = model.inner
        res["_model_info"] = model.describe()
    return res


def event_hindcast(name: str, data: SeriesData, event_start: str, origins: list[str], *, quick: bool = False,
                   horizons=HORIZONS, min_history: int = 36, val_frac: float = 0.15) -> dict[str, Any]:
    """Refit ``name`` (its first grid configuration) on data whose labels end before ``event_start`` and forecast
    from ``origins`` (for an event inside the development period, e.g. 1997/98, so no forecast uses a model that saw
    it). Returns the forecasts and the fitted model's description."""
    s = spec(name, quick)
    months = data.meta["months"]
    H = np.asarray(horizons)
    end = months.index(event_start)
    o = forecast_origins(0, end, H, min_history=min_history)
    k = int(round(val_frac * len(o)))
    val = o[-k:]
    train = o[o < val[0] - H.max()]
    m = s.factory(**s.grid[0]).fit(data, train, val, H)
    idx = np.array([months.index(x) for x in origins])
    return {"origins": idx.tolist(), "forecast": m.predict(data, idx), "train_last_label": months[int(train[-1] + H.max())],
            "data_end": months[end - 1], "info": m.describe()}

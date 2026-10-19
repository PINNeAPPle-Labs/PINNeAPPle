"""The series and model zoos of the near-random forecasting study (``examples/near_random_forecasting``).

:func:`case` builds one study case by name: the series (synthetic with known ground truth, or real), the protocol
(horizons, test period) and the list of :class:`~.study.ModelSpec` with their tuning grids. Every tuned model gets
the same budget of configurations (``BUDGET``) drawn from its grid, scored by purged walk-forward CV.

Prior knowledge given to the physical models is stated per case: the structure of the equations (or a deliberately
wrong structure), initial guesses that are off by 20-30 % for the parameters that are estimated, and which
parameters are taken as known (the Mackey-Glass delay and exponent, the Lorenz-96 dimension).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from .. import synthetic as syn
from .base import SeriesData
from .baselines import (
    ARForecaster,
    ARIMAForecaster,
    Climatology,
    ExponentialSmoothing,
    Persistence,
    PhaseClimatology,
    SeasonalNaive,
)
from .hybrid import ResidualCorrectionForecaster, UDEForecaster
from .learned import GBMForecaster, NeuralForecaster, PhysicsLossMLP
from .physical import (
    DelayODEForecaster,
    EnKFForecaster,
    LinearSSMForecaster,
    ODEForecaster,
    PhysicsFeatures,
    SDEDriftForecaster,
    SINDyForecaster,
)
from .study import ModelSpec, Protocol

__all__ = ["BUDGET", "Case", "CASES", "case", "lorenz63_missing_xz_rhs", "baseline_specs", "data_specs",
           "overfitting_variants"]

BUDGET = 4                     # configurations per tuned model


def lorenz63_missing_xz_rhs(x, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """Lorenz-63 without the -x z term of dy/dt: a deliberately misspecified physical model."""
    a, b, c = x[..., 0], x[..., 1], x[..., 2]
    return syn._stack([sigma * (b - a), rho * a - b, a * b - beta * c], x)


def _grid(**axes) -> list[dict[str, Any]]:
    keys = list(axes)
    return [dict(zip(keys, v, strict=False)) for v in itertools.product(*axes.values())]


@dataclass
class Case:
    name: str
    data: SeriesData
    protocol: Protocol
    specs: list[ModelSpec]
    truth: dict[str, Any] = field(default_factory=dict)
    title: str = ""


# ---------------------------------------------------------------------------
# Zoo building blocks
# ---------------------------------------------------------------------------

def baseline_specs(arima: bool = True) -> list[ModelSpec]:
    out = [ModelSpec("persistence", "baseline", Persistence), ModelSpec("climatology", "baseline", Climatology),
           ModelSpec("ar", "baseline", ARForecaster), ModelSpec("exp_smoothing", "baseline", ExponentialSmoothing),
           ModelSpec("seasonal_naive", "baseline", SeasonalNaive),
           ModelSpec("phase_climatology", "baseline", lambda **k: PhaseClimatology(**k), _grid(fit_cycles=[1, 3]))]
    if arima:
        out.append(ModelSpec("arima", "baseline", ARIMAForecaster))
    return out


def data_specs(lag_steps=(1,), lags=(8, 16), seq_lags: int = 32, epochs: int = 60, patience: int = 8,
               max_train: int | None = 3000, cols=None) -> list[ModelSpec]:
    """GBM, MLP, LSTM, TCN and transformer on lagged windows, each with a grid sampled to ``BUDGET`` configs."""
    common = dict(epochs=epochs, patience=patience, max_train=max_train, cols=cols)
    return [
        ModelSpec("gbm", "data", lambda **k: GBMForecaster(cols=cols, **k),
                  _grid(n_lags=list(lags) + [4], lag_step=list(lag_steps), max_iter=[100, 300],
                        max_leaf_nodes=[7, 15], min_samples_leaf=[20])),
        ModelSpec("mlp", "data", lambda **k: NeuralForecaster("mlp", **common, **k),
                  _grid(n_lags=list(lags), lag_step=list(lag_steps), hidden=[32, 128], weight_decay=[1e-4, 1e-2],
                        dropout=[0.0, 0.1])),
        ModelSpec("lstm", "data", lambda **k: NeuralForecaster("lstm", layers=1, **common, **k),
                  _grid(n_lags=[seq_lags // 2, seq_lags], lag_step=list(lag_steps), hidden=[32, 64],
                        weight_decay=[1e-4])),
        ModelSpec("tcn", "data", lambda **k: NeuralForecaster("tcn", **common, **k),
                  _grid(n_lags=[seq_lags], lag_step=list(lag_steps), hidden=[16, 32], layers=[3, 5],
                        weight_decay=[1e-4])),
        ModelSpec("transformer", "data", lambda **k: NeuralForecaster("transformer", **common, **k),
                  _grid(n_lags=[seq_lags // 2, seq_lags], lag_step=list(lag_steps), hidden=[16, 32], layers=[1, 2],
                        weight_decay=[1e-4])),
    ]


def _gbm_physics(name, make_physical, lags=(4, 8), lag_steps=(1,)):
    return ModelSpec(name, "data+physics",
                     lambda **k: GBMForecaster(features=PhysicsFeatures(make_physical()), name=name,
                                               family="data+physics", **k),
                     _grid(n_lags=list(lags), lag_step=list(lag_steps), max_iter=[100, 300], max_leaf_nodes=[7, 15]))


def _correction(name, make_physical, lags=(4, 8), lag_steps=(1,)):
    return ModelSpec(name, "hybrid",
                     lambda **k: ResidualCorrectionForecaster(make_physical(), "gbm", name=name, **k),
                     _grid(n_lags=list(lags), lag_step=list(lag_steps), max_iter=[100, 300], max_leaf_nodes=[7, 15]))


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def _lorenz63(noise: float, n: int, seed: int) -> Case:
    s = syn.lorenz63(n=n, noise=noise, seed=seed)
    data = SeriesData(s.obs, 0, s.dt, s.name)
    guess = {"sigma": 8.0, "rho": 22.0, "beta": 2.0}
    ode = lambda: ODEForecaster(syn.lorenz63_rhs, guess)  # noqa: E731
    wrong = lambda: ODEForecaster(lorenz63_missing_xz_rhs, guess, name="ode_missing_xz")  # noqa: E731
    H = [1, 2, 4, 8, 16, 32, 64, 128]
    specs = baseline_specs() + data_specs() + [
        _gbm_physics("gbm_physics_features", ode),
        ModelSpec("mlp_physics_loss", "data+physics",
                  lambda **k: PhysicsLossMLP(ode(), epochs=60, patience=8, max_train=3000, **k),
                  _grid(lam=[0.0, 0.01, 0.1, 1.0], n_lags=[4], hidden=[128], weight_decay=[1e-4])),
        ModelSpec("ode_fitted", "physical", ode),
        ModelSpec("ode_missing_xz", "physical", wrong),
        ModelSpec("sindy", "physical", lambda **k: SINDyForecaster(**k), _grid(threshold=[0.02, 0.05, 0.1, 0.2])),
        ModelSpec("enkf", "physical", lambda: EnKFForecaster(ode(), n_ens=40)),
        _correction("ode_fitted+gbm", ode),
        _correction("ode_missing_xz+gbm", wrong),
        ModelSpec("ude_missing_xz", "hybrid",
                  lambda **k: UDEForecaster(wrong(), window=3, epochs=60, patience=8, **k),
                  _grid(hidden=[32, 64], weight_decay=[1e-5, 1e-3])),
    ]
    lt = 1 / s.lyapunov
    return Case(s.name, data, Protocol(H, min_history=32, test_stride=2), specs,
                {"kind": "chaotic", "lyapunov": s.lyapunov, "lyapunov_time": lt, "noise_level": noise,
                 "params": s.params, "physical_prior": guess},
                f"Lorenz-63, noise {noise:g} std")


def _lorenz96(noise: float, n: int, seed: int) -> Case:
    s = syn.lorenz96(n=n, noise=noise, seed=seed)
    data = SeriesData(s.obs, 0, s.dt, s.name)
    ode = lambda: ODEForecaster(syn.lorenz96_rhs, {"F": 6.0})  # noqa: E731
    H = [1, 2, 4, 8, 16, 32, 64]
    specs = baseline_specs() + data_specs(lags=(4, 8), seq_lags=16) + [
        _gbm_physics("gbm_physics_features", ode, lags=(2, 4)),
        ModelSpec("mlp_physics_loss", "data+physics",
                  lambda **k: PhysicsLossMLP(ode(), epochs=60, patience=8, max_train=3000, **k),
                  _grid(lam=[0.0, 0.01, 0.1, 1.0], n_lags=[2], hidden=[256], weight_decay=[1e-4])),
        ModelSpec("ode_fitted", "physical", ode),
        ModelSpec("sindy", "physical", lambda **k: SINDyForecaster(**k), _grid(threshold=[0.05, 0.1, 0.2, 0.4])),
        ModelSpec("enkf", "physical", lambda: EnKFForecaster(ode(), n_ens=40)),
        _correction("ode_fitted+gbm", ode, lags=(2, 4)),
    ]
    return Case(s.name, data, Protocol(H, min_history=32, test_stride=2), specs,
                {"kind": "chaotic", "lyapunov": s.lyapunov, "lyapunov_time": 1 / s.lyapunov, "noise_level": noise,
                 "params": s.params, "physical_prior": {"F": 6.0}},
                f"Lorenz-96 (K=40), noise {noise:g} std")


def _mackey_glass(noise: float, n: int, seed: int) -> Case:
    s = syn.mackey_glass(n=n, noise=noise, seed=seed)
    data = SeriesData(s.obs, 0, s.dt, s.name)
    guess = {"beta": 0.15, "gamma": 0.08, "n": 10.0}
    dde = lambda: DelayODEForecaster(syn.mackey_glass_rhs, guess, tau=17.0, fit_params=["beta", "gamma"])  # noqa
    wrong = lambda: DelayODEForecaster(syn.mackey_glass_rhs, guess, tau=14.0,  # noqa: E731
                                       fit_params=["beta", "gamma"], name="dde_wrong_delay")
    H = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    specs = baseline_specs() + data_specs(lag_steps=(1, 2)) + [
        _gbm_physics("gbm_physics_features", dde),
        ModelSpec("dde_fitted", "physical", dde),
        ModelSpec("dde_wrong_delay", "physical", wrong),
        _correction("dde_fitted+gbm", dde),
        _correction("dde_wrong_delay+gbm", wrong),
    ]
    return Case(s.name, data, Protocol(H, min_history=64, test_stride=2), specs,
                {"kind": "chaotic", "lyapunov": s.lyapunov, "lyapunov_time": 1 / s.lyapunov, "noise_level": noise,
                 "params": s.params, "physical_prior": {**guess, "tau": 17.0, "known": ["tau", "n"]}},
                f"Mackey-Glass, noise {noise:g} std")


def _ou(n: int, seed: int) -> Case:
    s = syn.ornstein_uhlenbeck(n=n, noise=0.1, seed=seed)
    data = SeriesData(s.obs, 0, s.dt, s.name)
    kal = lambda: LinearSSMForecaster("ou")  # noqa: E731
    H = [1, 2, 4, 8, 16, 32]
    specs = baseline_specs() + data_specs() + [
        _gbm_physics("gbm_physics_features", kal), ModelSpec("ou_kalman", "physical", kal),
        _correction("ou_kalman+gbm", kal)]
    return Case(s.name, data, Protocol(H, min_history=32, test_stride=2), specs,
                {"kind": "stochastic", "params": s.params, "noise_level": 0.1}, "Ornstein-Uhlenbeck")


def _oscillator(n: int, seed: int) -> Case:
    s = syn.stochastic_oscillator(n=n, noise=0.1, seed=seed)
    data = SeriesData(s.obs, 0, s.dt, s.name)
    osc = lambda: LinearSSMForecaster("oscillator")  # noqa: E731
    ou = lambda: LinearSSMForecaster("ou")  # noqa: E731
    H = [1, 2, 4, 8, 16, 32, 64]
    specs = baseline_specs() + data_specs(lags=(16, 32), seq_lags=32) + [
        _gbm_physics("gbm_physics_features", osc), ModelSpec("oscillator_kalman", "physical", osc),
        ModelSpec("ou_kalman", "physical", ou), _correction("oscillator_kalman+gbm", osc),
        _correction("ou_kalman+gbm", ou)]
    return Case(s.name, data, Protocol(H, min_history=32, test_stride=2), specs,
                {"kind": "stochastic", "params": s.params, "noise_level": 0.1}, "Noise-driven damped oscillator")


def _double_well(n: int, seed: int) -> Case:
    s = syn.double_well(n=n, noise=0.1, seed=seed)
    data = SeriesData(s.obs, 0, s.dt, s.name)
    H = [1, 2, 4, 8, 16, 32, 64, 128]
    specs = baseline_specs() + data_specs() + [
        _gbm_physics("gbm_physics_features", lambda: SDEDriftForecaster(3)),
        ModelSpec("sde_poly3", "physical", lambda: SDEDriftForecaster(3)),
        ModelSpec("sde_poly1", "physical", lambda: SDEDriftForecaster(1)),
        ModelSpec("ou_kalman", "physical", lambda: LinearSSMForecaster("ou")),
        ModelSpec("sde_poly1_nn_closure", "hybrid", lambda **k: SDEDriftForecaster(1, closure=k),
                  _grid(hidden=[16, 32], weight_decay=[1e-4, 1e-2])),
        _correction("sde_poly3+gbm", lambda: SDEDriftForecaster(3, n_samples=100)),
    ]
    return Case(s.name, data, Protocol(H, min_history=32, test_stride=2), specs,
                {"kind": "stochastic", "params": s.params, "noise_level": 0.1}, "Double-well (regime switching)")


def _noise(kind: str, n: int, seed: int) -> Case:
    s = syn.white_noise(n=n, seed=seed) if kind == "white_noise" else syn.random_walk(n=n, seed=seed)
    data = SeriesData(s.obs, 0, s.dt, s.name)
    H = [1, 2, 4, 8, 16]
    return Case(s.name, data, Protocol(H, min_history=32, test_stride=2), baseline_specs() + data_specs(),
                {"kind": "noise"}, "White noise" if kind == "white_noise" else "Random walk")


def _fx(seed: int, cache_dir) -> Case:
    from ..datasets.public import load_usd_eur_daily, log_returns
    s = load_usd_eur_daily(cache_dir)
    r = 100 * log_returns(s.values)
    data = SeriesData(r[:, None], 0, 1.0, "usd_eur_log_returns",
                      meta={"start": s.time[1], "end": s.time[-1], "dates": list(s.time[1:])})
    H = [1, 2, 5, 10, 20]
    return Case(data.name, data, Protocol(H, min_history=32, test_stride=2), baseline_specs() + data_specs(),
                {"kind": "real", "source": s.source, "units": "percent log-return per business day",
                 "first": s.time[1], "last": s.time[-1]}, "USD/EUR daily log-returns")


def _sunspots(seed: int, cache_dir) -> Case:
    from ..datasets.public import load_sunspots_yearly
    s = load_sunspots_yearly(cache_dir)
    data = SeriesData(s.values[:, None], 0, 1.0, "sunspots_yearly", meta={"first": s.time[0], "last": s.time[-1]})
    osc = lambda: LinearSSMForecaster("oscillator", name="oscillator_kalman")  # noqa: E731
    H = [1, 2, 3, 5, 8, 11]
    specs = baseline_specs() + data_specs(lags=(8, 16), seq_lags=16, max_train=None) + [
        _gbm_physics("gbm_physics_features", osc), ModelSpec("oscillator_kalman", "physical", osc),
        _correction("oscillator_kalman+gbm", osc)]
    return Case(data.name, data, Protocol(H, min_history=16, test_stride=1, val_frac=0.2, fold_frac=0.2,
                                          n_folds=2), specs,
                {"kind": "real", "source": s.source, "units": "yearly mean sunspot number", "first": s.time[0],
                 "last": s.time[-1]}, "Sunspots (yearly)")


def _youtube(kind: str, cache_dir) -> Case:
    """Daily aggregates of the US trending list (205 trending days, 2017-11-14 to 2018-06-14; 8 calendar days are
    missing from the source and are skipped, so one step is one trending day)."""
    import numpy as _np

    from ..datasets.public import load_youtube_trending_us
    y = load_youtube_trending_us(cache_dir)
    if kind == "log_views":
        v, name, title, units = _np.log10(y.total_views), "youtube_us_log_views", \
            "YouTube US trending: log10 total daily views", "log10(sum of views of the trending entries)"
    else:
        v, name, title, units = y.category_counts[24].astype(float), "youtube_us_entertainment_slots", \
            "YouTube US trending: Entertainment slots per day", "trending entries in category 24 (of ~200)"
    data = SeriesData(v[:, None], 0, 1.0, name, meta={"dates": list(y.dates), "missing_days": y.missing_days()})
    H = [1, 2, 3, 5, 7]
    specs = baseline_specs() + data_specs(lags=(7, 14), seq_lags=14, max_train=None)
    return Case(name, data, Protocol(H, min_history=16, test_stride=1, val_frac=0.2, fold_frac=0.2, n_folds=2),
                specs, {"kind": "real", "source": y.source, "units": units, "first": y.dates[0],
                        "last": y.dates[-1], "missing_days": y.missing_days()}, title)


CASES = ["lorenz63_noise0.01", "lorenz63_noise0.1", "lorenz63_noise0.3", "mackey_glass_noise0.01",
         "mackey_glass_noise0.1", "mackey_glass_noise0.3", "lorenz96_noise0.1", "ornstein_uhlenbeck",
         "stochastic_oscillator", "double_well", "white_noise", "random_walk", "usd_eur_log_returns",
         "sunspots_yearly", "youtube_us_log_views", "youtube_us_entertainment_slots"]


def case(name: str, *, n: int = 6000, seed: int = 0, cache_dir: str | None = None) -> Case:
    """Build the study case ``name`` (one of :data:`CASES`); ``n`` is the length of the synthetic series."""
    if name.startswith("lorenz63_noise"):
        return _lorenz63(float(name.split("noise")[1]), n, seed)
    if name.startswith("mackey_glass_noise"):
        return _mackey_glass(float(name.split("noise")[1]), n, seed)
    if name.startswith("lorenz96_noise"):
        return _lorenz96(float(name.split("noise")[1]), n, seed)
    table = {"ornstein_uhlenbeck": lambda: _ou(n, seed), "stochastic_oscillator": lambda: _oscillator(n, seed),
             "double_well": lambda: _double_well(n, seed), "white_noise": lambda: _noise("white_noise", n, seed),
             "random_walk": lambda: _noise("random_walk", n, seed), "usd_eur_log_returns": lambda: _fx(seed, cache_dir),
             "sunspots_yearly": lambda: _sunspots(seed, cache_dir),
             "youtube_us_log_views": lambda: _youtube("log_views", cache_dir),
             "youtube_us_entertainment_slots": lambda: _youtube("entertainment", cache_dir)}
    if name not in table:
        raise KeyError(f"unknown case {name!r}; choose from {CASES}")
    return table[name]()


def overfitting_variants(c: Case, physics: bool = False) -> dict[str, Any]:
    """Training variants of an MLP (and of gradient boosting) for :func:`~.study.overfitting_study`: from a
    deliberately overfit network (wide, deep, long window, no early stopping, no weight decay, 200 epochs) to small,
    regularised, early-stopped and ensembled ones. ``physics`` adds the physics-loss MLP of a Lorenz case."""
    base = dict(n_lags=16, hidden=64, layers=2, epochs=150, patience=10, max_train=3000, weight_decay=1e-4)
    v = {
        "overfit: MLP 512x3, 32 lags, no early stop": lambda: NeuralForecaster(
            "mlp", n_lags=32, hidden=512, layers=3, epochs=200, patience=None, weight_decay=0.0, max_train=3000),
        "MLP 512x3 + early stopping": lambda: NeuralForecaster(
            "mlp", n_lags=32, hidden=512, layers=3, epochs=200, patience=10, weight_decay=0.0, max_train=3000),
        "MLP 512x3 + early stop + wd 1e-2 + dropout 0.2": lambda: NeuralForecaster(
            "mlp", n_lags=32, hidden=512, layers=3, epochs=200, patience=10, weight_decay=1e-2, dropout=0.2,
            max_train=3000),
        "MLP 16x1 + early stopping": lambda: NeuralForecaster("mlp", **{**base, "hidden": 16, "layers": 1}),
        "MLP 64x2 + early stopping": lambda: NeuralForecaster("mlp", **base),
        "MLP 64x2 + early stopping, 5-member ensemble": lambda: NeuralForecaster("mlp", n_members=5, **base),
        "overfit: GBM 255 leaves, 1000 rounds, leaf>=1": lambda: GBMForecaster(
            n_lags=16, max_iter=1000, learning_rate=0.1, max_leaf_nodes=255, min_samples_leaf=1),
        "GBM 7 leaves, 100 rounds, leaf>=50, l2=1": lambda: GBMForecaster(
            n_lags=8, max_iter=100, learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=50, l2=1.0),
    }
    if physics:
        ode = lambda: ODEForecaster(syn.lorenz63_rhs, c.truth["physical_prior"])  # noqa: E731
        v["MLP 128x2, 4 lags + early stop (no physics)"] = lambda: PhysicsLossMLP(
            ode(), lam=0.0, n_lags=4, hidden=128, epochs=150, patience=10, max_train=3000, weight_decay=1e-4)
        v["MLP 128x2, 4 lags + early stop + physics loss 0.1"] = lambda: PhysicsLossMLP(
            ode(), lam=0.1, n_lags=4, hidden=128, epochs=150, patience=10, max_train=3000, weight_decay=1e-4)
    return v

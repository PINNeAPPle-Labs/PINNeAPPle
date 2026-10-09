"""Adaptive forecasting: switch between and combine forecasting models online, using only past out-of-sample errors.

Every expert (any object with ``fit(y) -> self`` and ``predict(horizon) -> array``: the built-in fast models below,
the baselines and ML forecasters of ``pinneapple_systems.time_series``, or your own) is refitted on the data seen so
far and issues forecasts for horizons 1..H. When the observation for a forecast arrives, its error updates the
weights. Nothing ever looks at data the forecast could not have known: the procedure is prequential (Dawid 1984),
which is what prevents overfitting the model choice to the evaluation period.

Three levels:

1. **Fixed-Share exponential weights per horizon** (Herbster & Warmuth 1998, Machine Learning 32:151). Weights
   shrink by exp(-eta * loss) and a share alpha is redistributed every step, so the mixture can move to a different
   model when the regime changes (a plain exponential-weights / "best model so far" rule takes far too long to switch).
   Regret is bounded against the best *sequence* of models with a limited number of switches.
2. **A grid of (eta, alpha) aggregators combined by AdaHedge** (de Rooij, van Erven, Grunwald & Koolen 2014, JMLR
   15:1281), which needs no tuning: the learning rate and the switching rate are themselves chosen online, with a
   regret bound against the best (eta, alpha) in hindsight.
3. **Adaptive conformal intervals** (Gibbs & Candes 2021, NeurIPS): the interval level is corrected online after every
   miss or hit, so coverage stays near the target even when the distribution shifts.

Losses are scale-free: absolute (or squared) error divided by the mean absolute one-step change of the series so far
(the MASE scale), clipped to keep them bounded as the theory assumes.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["SimpleExpSmoothing", "HoltDamped", "HoltWintersAdditive", "Theta", "MovingAverage", "RidgeAR",
           "LagRegressorExpert", "default_experts", "FixedShare", "AdaHedge", "AdaptiveConformal", "AdaptiveForecaster", "AdaptiveRun"]


# ============================================================================================== fast experts
def _clean(y) -> np.ndarray:
    y = np.asarray(y, dtype=float).reshape(-1)
    return y[~np.isnan(y)]


@dataclass
class SimpleExpSmoothing:
    alpha: float = 0.3
    level_: float | None = None

    def fit(self, y):
        y = _clean(y)
        lev = y[0]
        for v in y[1:]:
            lev = lev + self.alpha * (v - lev)
        self.level_ = float(lev)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(int(horizon), self.level_)


@dataclass
class HoltDamped:
    """Holt's linear trend with damping phi (Gardner & McKenzie 1985)."""
    alpha: float = 0.3
    beta: float = 0.1
    phi: float = 0.95
    level_: float = 0.0
    trend_: float = 0.0

    def fit(self, y):
        y = _clean(y)
        lev, tr = y[0], (y[1] - y[0]) if len(y) > 1 else 0.0
        for v in y[1:]:
            prev = lev
            lev = self.alpha * v + (1 - self.alpha) * (lev + self.phi * tr)
            tr = self.beta * (lev - prev) + (1 - self.beta) * self.phi * tr
        self.level_, self.trend_ = float(lev), float(tr)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        damp = np.cumsum(self.phi ** np.arange(1, int(horizon) + 1))
        return self.level_ + damp * self.trend_


@dataclass
class HoltWintersAdditive:
    season_length: int = 12
    alpha: float = 0.2
    beta: float = 0.05
    gamma: float = 0.2
    level_: float = 0.0
    trend_: float = 0.0
    seasonal_: np.ndarray | None = None
    n_: int = 0

    def fit(self, y):
        y = _clean(y)
        m = self.season_length
        if len(y) < 2 * m:                                         # not enough data: behave as SES
            self.seasonal_ = np.zeros(m)
            ses = SimpleExpSmoothing(self.alpha).fit(y)
            self.level_, self.trend_, self.n_ = ses.level_, 0.0, len(y)
            return self
        lev = y[:m].mean()
        tr = (y[m:2 * m].mean() - y[:m].mean()) / m
        s = list(y[:m] - lev)
        for t in range(m, len(y)):
            prev = lev
            st = s[t - m]
            lev = self.alpha * (y[t] - st) + (1 - self.alpha) * (lev + tr)
            tr = self.beta * (lev - prev) + (1 - self.beta) * tr
            s.append(self.gamma * (y[t] - lev) + (1 - self.gamma) * st)
        self.level_, self.trend_, self.seasonal_, self.n_ = float(lev), float(tr), np.array(s[-m:]), len(y)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        h = np.arange(1, int(horizon) + 1)
        return self.level_ + h * self.trend_ + self.seasonal_[(h - 1) % self.season_length]


@dataclass
class Theta:
    """Theta method as SES with drift equal to half the regression slope (Assimakopoulos & Nikolopoulos 2000;
    Hyndman & Billah 2003, Int. J. Forecasting 19:287)."""
    alpha: float = 0.3
    level_: float = 0.0
    slope_: float = 0.0
    n_: int = 0

    def fit(self, y):
        y = _clean(y)
        t = np.arange(len(y))
        self.slope_ = float(np.polyfit(t, y, 1)[0]) if len(y) > 2 else 0.0
        self.level_ = SimpleExpSmoothing(self.alpha).fit(y).level_
        self.n_ = len(y)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        h = np.arange(1, int(horizon) + 1)
        a = self.alpha
        return self.level_ + 0.5 * self.slope_ * (h - 1 + 1 / a - (1 - a) ** self.n_ / a)


@dataclass
class MovingAverage:
    window: int = 12
    value_: float = 0.0

    def fit(self, y):
        self.value_ = float(_clean(y)[-self.window:].mean())
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(int(horizon), self.value_)


@dataclass
class RidgeAR:
    """Autoregression of order p with an intercept, ridge-regularised, fitted on the last ``window`` points; multi-step
    forecasts are recursive."""
    p: int = 12
    window: int = 300
    ridge: float = 1e-2
    coef_: np.ndarray | None = None
    tail_: np.ndarray | None = None

    def fit(self, y):
        y = _clean(y)[-self.window:]
        p = min(self.p, max(1, len(y) // 3))
        X = np.column_stack([y[p - k - 1: len(y) - k - 1] for k in range(p)] + [np.ones(len(y) - p)])
        target = y[p:]
        sc = np.std(y) or 1.0
        A = X.T @ X + self.ridge * len(target) * sc ** 2 * np.diag(np.r_[np.ones(p), 0.0])
        self.coef_ = np.linalg.solve(A, X.T @ target)
        self.tail_ = y[-p:][::-1].copy()                           # most recent first
        return self

    def predict(self, horizon: int) -> np.ndarray:
        lags, out = list(self.tail_), []
        p = len(lags)
        for _ in range(int(horizon)):
            v = float(np.dot(self.coef_[:p], lags) + self.coef_[-1])
            out.append(v)
            lags = [v] + lags[:-1]
        return np.array(out)


class LagRegressorExpert:
    """Any tabular regressor (scikit-learn style ``fit(X, y)`` / ``predict(X)``, including the ML forecasters of
    ``pinneapple_systems.time_series.models``) as an expert: lagged values as features, recursive multi-step forecasts.
    The estimator is refitted every ``refit_every`` calls (on the last ``window`` points); in between only the lag
    tail is updated, so an expensive model does not slow every step."""

    def __init__(self, estimator: Any, n_lags: int = 12, refit_every: int = 24, window: int = 500):
        self.estimator, self.n_lags, self.refit_every, self.window = estimator, int(n_lags), int(refit_every), int(window)
        self._calls, self._fitted, self.tail_ = 0, False, None

    def fit(self, y):
        y = _clean(y)[-self.window:]
        p = self.n_lags
        if len(y) <= p + 2:
            raise ValueError("not enough history for the lags")
        if not self._fitted or self._calls % self.refit_every == 0:
            X = np.column_stack([y[p - k - 1: len(y) - k - 1] for k in range(p)])
            self.estimator.fit(X, y[p:])
            self._fitted = True
        self._calls += 1
        self.tail_ = y[-p:][::-1].copy()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        lags, out = list(self.tail_), []
        for _ in range(int(horizon)):
            v = float(np.asarray(self.estimator.predict(np.array([lags]))).reshape(-1)[0])
            out.append(v)
            lags = [v] + lags[:-1]
        return np.array(out)


def default_experts(season_length: int = 1) -> dict[str, Any]:
    """A diverse, cheap pool. Hyper-parameter variants are separate experts, so choosing a smoothing constant is part
    of the online selection instead of an in-sample fit."""
    from .baselines.naive import DriftForecaster, NaiveForecaster, SeasonalNaiveForecaster

    m = int(season_length)
    ex: dict[str, Any] = {"naive": NaiveForecaster(), "drift": DriftForecaster(),
                          "ses_0.1": SimpleExpSmoothing(0.1), "ses_0.3": SimpleExpSmoothing(0.3),
                          "ses_0.7": SimpleExpSmoothing(0.7), "holt_damped": HoltDamped(0.3, 0.1, 0.95),
                          "theta": Theta(0.3), "mean": MovingAverage(max(m, 10)),
                          "ar": RidgeAR(p=max(2 * m, 6) if m > 1 else 6)}
    if m > 1:
        ex["seasonal_naive"] = SeasonalNaiveForecaster(m)
        ex["holt_winters"] = HoltWintersAdditive(m)
    return ex


# ============================================================================================== aggregation
from pinneapple_physics.online_learning import (  # noqa: E402,F401  (re-exported)
    AdaHedge,
    AdaptiveConformal,
    FixedShare,
)


# ============================================================================================== forecaster
@dataclass
class AdaptiveRun:
    """Result of :meth:`AdaptiveForecaster.run`: one row per forecast origin t (forecasts for t+1..t+H)."""
    origins: np.ndarray
    forecast: np.ndarray                    # (n_origins, H) ensemble
    expert_forecasts: np.ndarray            # (n_origins, n_experts, H)
    lower: np.ndarray
    upper: np.ndarray
    weights: np.ndarray                     # (n_origins, n_experts) effective weights for horizon 1
    active: list[str]                       # expert with the largest horizon-1 weight at each origin
    expert_names: list[str]
    y: np.ndarray

    def errors(self, h: int = 1) -> dict[str, dict[str, float]]:
        """MAE, RMSE and MASE at horizon ``h`` for the ensemble and every expert, on the origins whose target is
        observed; plus the best single expert in hindsight (not available online: an upper bar, not a competitor
        the ensemble could have known)."""
        idx = self.origins + h
        ok = idx < len(self.y)
        truth = self.y[idx[ok]]
        scale = float(np.mean(np.abs(np.diff(self.y[: self.origins[0] + 1])))) or 1.0
        out = {}

        def stats(pred):
            e = pred - truth
            return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e * e))),
                    "mase": float(np.mean(np.abs(e)) / scale)}

        out["ensemble"] = stats(self.forecast[ok, h - 1])
        for i, n in enumerate(self.expert_names):
            out[n] = stats(self.expert_forecasts[ok, i, h - 1])
        best = min(self.expert_names, key=lambda n: out[n]["mae"])
        out["best_single_in_hindsight"] = {**out[best], "name": best}
        return out

    def coverage(self, h: int = 1) -> float:
        idx = self.origins + h
        ok = (idx < len(self.y)) & np.isfinite(self.lower[:, h - 1])
        y = self.y[idx[ok]]
        return float(np.mean((y >= self.lower[ok, h - 1]) & (y <= self.upper[ok, h - 1])))

    def switches(self) -> list[tuple[int, str]]:
        """(origin, expert) each time the leading expert changes."""
        out, prev = [], None
        for t, a in zip(self.origins.tolist(), self.active, strict=True):
            if a != prev:
                out.append((t, a))
                prev = a
        return out


class AdaptiveForecaster:
    """Online selection and combination of forecasting models.

    >>> af = AdaptiveForecaster(default_experts(season_length=12), horizon=6)
    >>> run = af.run(y, start=48)          # backtest: every origin uses only data up to that origin
    >>> run.errors(h=1)["ensemble"], run.switches()
    >>> af.forecast()                      # next H values, interval and current weights

    ``mode``: "combine" (weighted average, default) or "select" (the forecast of the leading expert, i.e. switching
    between models). ``etas`` / ``alphas``: the (learning rate, switching rate) grid of the Fixed-Share aggregators,
    combined by AdaHedge. ``refit_every``: refit experts every k steps (expensive experts); forecasts are reissued from
    the last fit in between."""

    def __init__(self, experts: Mapping[str, Any] | Sequence[Any], horizon: int = 1, mode: str = "combine",
                 etas: Sequence[float] = (0.5, 2.0, 8.0), alphas: Sequence[float] = (0.0, 0.01, 0.05),
                 loss: str = "absolute", clip: float = 10.0, interval_level: float | None = 0.9,
                 conformal_gamma: float = 0.01, refit_every: int = 1, max_history: int | None = 1000,
                 on_error: str = "fallback"):
        if mode not in ("combine", "select"):
            raise ValueError("mode must be 'combine' or 'select'")
        if loss not in ("absolute", "squared"):
            raise ValueError("loss must be 'absolute' or 'squared'")
        if not isinstance(experts, Mapping):
            experts = {f"{type(e).__name__}_{i}": e for i, e in enumerate(experts)}
        if len(experts) < 2:
            raise ValueError("give at least two experts")
        self.names = list(experts)
        self._proto = [experts[n] for n in self.names]
        self.H, self.mode, self.loss, self.clip = int(horizon), mode, loss, float(clip)
        self.grid = [(e, a) for e in etas for a in alphas]
        self.refit_every, self.max_history, self.on_error = max(1, int(refit_every)), max_history, on_error
        n, k = len(self.names), len(self.grid)
        self.agg = [[FixedShare(n, e, a) for e, a in self.grid] for _ in range(self.H)]
        self.top = [AdaHedge(k) for _ in range(self.H)]
        self.conf = [AdaptiveConformal(interval_level, conformal_gamma) for _ in range(self.H)] if interval_level else None
        self.y: list[float] = []
        self._pending: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}   # origin -> (experts, ens, hw)
        self._models = [copy.deepcopy(p) for p in self._proto]
        self._expert_fc: np.ndarray | None = None
        self._abs_diff_sum, self._steps_since_fit = 0.0, 0
        self.failures: dict[str, int] = {n: 0 for n in self.names}

    # ------------------------------------------------------------------------------------------ internals
    def _scale(self) -> float:
        n = len(self.y) - 1
        return self._abs_diff_sum / n if n > 0 and self._abs_diff_sum > 0 else 1.0

    def _norm_loss(self, err: np.ndarray) -> np.ndarray:
        z = np.abs(err) / self._scale()
        z = z if self.loss == "absolute" else z * z
        return np.where(np.isfinite(z), np.minimum(z, self.clip), self.clip)   # a failed expert gets the worst loss

    def _refit(self) -> np.ndarray:
        hist = np.asarray(self.y if self.max_history is None else self.y[-self.max_history:])
        fc = np.empty((len(self.names), self.H))
        for i, m in enumerate(self._models):
            try:
                self._models[i] = m.fit(hist) or m
                p = np.asarray(self._models[i].predict(self.H), dtype=float).reshape(-1)[: self.H]
                if p.shape[0] != self.H or not np.all(np.isfinite(p)):
                    raise ValueError("non-finite or short forecast")
                fc[i] = p
            except Exception:
                if self.on_error == "raise":
                    raise
                self.failures[self.names[i]] += 1
                fc[i] = np.nan                                     # left out of this step and scored as the worst
        return fc

    def _effective_weights(self, h: int) -> np.ndarray:
        tw = self.top[h].weights()
        return sum(tw[k] * self.agg[h][k].w for k in range(len(self.grid)))

    def _combine(self, fc: np.ndarray) -> np.ndarray:
        out = np.empty(self.H)
        for h in range(self.H):
            if self.mode == "select":
                w = np.where(np.isfinite(fc[:, h]), self._effective_weights(h), -1.0)
                out[h] = fc[int(np.argmax(w)), h]
            else:
                tw = self.top[h].weights()
                out[h] = float(sum(tw[k] * self.agg[h][k].predict(fc[:, h]) for k in range(len(self.grid))))
        return out

    # ------------------------------------------------------------------------------------------ public API
    def update(self, value: float) -> None:
        """Feed one new observation: score the forecasts that targeted it, then refit and issue new forecasts."""
        value = float(value)
        if self.y:
            self._abs_diff_sum += abs(value - self.y[-1])
        self.y.append(value)
        t = len(self.y) - 1                                        # index of the new observation
        for h in range(self.H):
            origin = t - (h + 1)
            if origin not in self._pending:
                continue
            efc, ens, hw = self._pending[origin]
            lossv = self._norm_loss(efc[:, h] - value)
            agg_losses = np.array([self._norm_loss(np.array([a.predict(efc[:, h]) - value]))[0]
                                   for a in self.agg[h]])
            for a in self.agg[h]:
                a.update(lossv)
            self.top[h].update(agg_losses)
            if self.conf is not None:
                self.conf[h].update(value, ens[h], hw[h])
        for origin in [o for o in self._pending if o <= t - self.H]:
            del self._pending[origin]
        if len(self.y) < 3:
            return
        if self._expert_fc is None or self._steps_since_fit + 1 >= self.refit_every:
            self._expert_fc, self._steps_since_fit = self._refit(), 0
        else:
            self._steps_since_fit += 1
            self._expert_fc = np.column_stack([self._expert_fc[:, 1:], self._expert_fc[:, -1:]])  # shift horizons
        ens = self._combine(self._expert_fc)
        hw = np.array([c.halfwidth() for c in self.conf]) if self.conf is not None else np.full(self.H, np.nan)
        self._pending[t] = (self._expert_fc.copy(), ens, hw)

    def forecast(self) -> dict[str, Any]:
        """Forecast for the next H steps from everything fed so far, with interval and weights."""
        if not self._pending:
            raise RuntimeError("feed at least 3 observations first")
        t = max(self._pending)
        efc, ens, hw = self._pending[t]
        w1 = self._effective_weights(0)
        return {"forecast": ens, "lower": ens - hw, "upper": ens + hw,
                "expert_forecasts": dict(zip(self.names, efc, strict=True)),
                "weights": dict(zip(self.names, w1.tolist(), strict=True)),
                "active": self.names[int(np.argmax(w1))],
                "strategy_weights": {f"eta={e:g},alpha={a:g}": float(w) for (e, a), w in
                                     zip(self.grid, self.top[0].weights(), strict=True)}}

    def run(self, y: Sequence[float], start: int | None = None) -> AdaptiveRun:
        """Backtest: feed ``y`` one value at a time; record the forecasts issued at every origin >= ``start``."""
        y = np.asarray(y, dtype=float)
        start = max(3, int(start if start is not None else min(len(y) // 5, 50)))
        origins, ens, efcs, lo, hi, ws, act = [], [], [], [], [], [], []
        for t, v in enumerate(y):
            self.update(v)
            if t + 1 < start or t == len(y) - 1:
                continue
            f = self.forecast()
            origins.append(t)
            ens.append(f["forecast"])
            efcs.append(np.array([f["expert_forecasts"][n] for n in self.names]))
            lo.append(f["lower"])
            hi.append(f["upper"])
            ws.append([f["weights"][n] for n in self.names])
            act.append(f["active"])
        return AdaptiveRun(np.array(origins), np.array(ens), np.array(efcs), np.array(lo), np.array(hi),
                           np.array(ws), act, list(self.names), y)

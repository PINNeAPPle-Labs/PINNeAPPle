"""Online learning with expert advice, shared by every adaptive method in the library (time-series forecasting in
``pinneapple_systems.time_series.adaptive`` and physics-model ensembles in ``pinneapple_physics.ensemble``).

* :class:`FixedShare`: exponentially weighted average with fixed share (Herbster & Warmuth 1998, Machine Learning
  32:151); tracks the best *sequence* of experts, so it can switch when the regime changes.
* :class:`AdaHedge`: parameter-free Hedge (de Rooij, van Erven, Grunwald & Koolen 2014, JMLR 15:1281); used to choose
  the learning and switching rates of a grid of Fixed-Share aggregators online.
* :class:`AdaptiveConformal`: adaptive conformal inference (Gibbs & Candes 2021, NeurIPS); interval level corrected
  after every observation so coverage holds under distribution shift.
* :class:`FixedShareGrid`: a grid of Fixed-Share aggregators weighted by AdaHedge, the combination used by both
  adaptive methods.

All decisions use only losses already observed (prequential), so they cannot overfit the evaluation period.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

__all__ = ["FixedShare", "AdaHedge", "AdaptiveConformal", "FixedShareGrid"]


class FixedShare:
    """Exponentially weighted average with fixed share (one per horizon)."""

    def __init__(self, n: int, eta: float, alpha: float):
        self.w = np.full(n, 1.0 / n)
        self.eta, self.alpha = float(eta), float(alpha)

    def predict(self, forecasts: np.ndarray) -> float:
        ok = np.isfinite(forecasts)                                # experts that failed this step are left out
        w = self.w * ok
        return float(w[ok] @ forecasts[ok] / w.sum()) if w.sum() > 0 else float(np.nanmean(forecasts))

    def update(self, losses: np.ndarray) -> None:
        v = self.w * np.exp(-self.eta * (losses - losses.min()))
        v /= v.sum()
        self.w = (1 - self.alpha) * v + self.alpha / len(v)


class AdaHedge:
    """Parameter-free Hedge: follow the leader while it works, hedge with eta = ln K / Delta otherwise."""

    def __init__(self, k: int):
        self.L = np.zeros(k)
        self.delta = 0.0
        self.k = k

    @property
    def eta(self) -> float:
        return math.inf if self.delta <= 0 else math.log(self.k) / self.delta

    def weights(self) -> np.ndarray:
        if math.isinf(self.eta):
            best = self.L == self.L.min()
            return best / best.sum()
        z = np.exp(-self.eta * (self.L - self.L.min()))
        return z / z.sum()

    def update(self, losses: np.ndarray) -> None:
        w, eta = self.weights(), self.eta
        h = float(w @ losses)
        if math.isinf(eta):
            mix = float(losses[w > 0].min())
        else:
            pos = w > 0                                                # log-sum-exp: no underflow for large eta
            z = np.log(w[pos]) - eta * losses[pos]
            zmax = z.max()
            mix = -(zmax + math.log(float(np.exp(z - zmax).sum()))) / eta
        self.delta += max(h - mix, 0.0)
        self.L += losses


class AdaptiveConformal:
    """Adaptive conformal inference on absolute residuals: alpha_{t+1} = alpha_t + gamma (alpha - err_t)."""

    def __init__(self, level: float = 0.9, gamma: float = 0.01, window: int = 500):
        self.target = 1 - level
        self.alpha_t = self.target
        self.gamma, self.window = gamma, window
        self.resid: list[float] = []
        self.hits: list[int] = []

    def halfwidth(self) -> float:
        if len(self.resid) < 10:
            return float("nan")
        r = np.asarray(self.resid[-self.window:])
        if self.alpha_t <= 0:
            return float(r.max() * 1.5)
        if self.alpha_t >= 1:
            return 0.0
        return float(np.quantile(r, 1 - self.alpha_t, method="higher"))

    def update(self, y: float, yhat: float, halfwidth: float) -> None:
        if np.isfinite(halfwidth):
            miss = int(abs(y - yhat) > halfwidth)
            self.hits.append(1 - miss)
            self.alpha_t += self.gamma * (self.target - miss)
        self.resid.append(abs(y - yhat))


class FixedShareGrid:
    """Fixed-Share aggregators over ``n`` experts for every (eta, alpha) in the grid, weighted by AdaHedge."""

    def __init__(self, n: int, etas: Sequence[float] = (0.5, 2.0, 8.0), alphas: Sequence[float] = (0.0, 0.01, 0.05)):
        self.grid: list[tuple[float, float]] = [(e, a) for e in etas for a in alphas]
        self.aggs = [FixedShare(n, e, a) for e, a in self.grid]
        self.top = AdaHedge(len(self.grid))

    def expert_weights(self) -> np.ndarray:
        """Effective weight of every expert (AdaHedge mixture of the Fixed-Share weights)."""
        tw = self.top.weights()
        return sum(tw[k] * a.w for k, a in enumerate(self.aggs))

    def strategy_weights(self) -> np.ndarray:
        return self.top.weights()

    def update(self, expert_losses: np.ndarray, aggregator_losses: np.ndarray) -> None:
        """``expert_losses``: loss of each expert; ``aggregator_losses``: loss of each aggregator's combined prediction
        (for convex losses this is at most its weighted expert loss, so combining can beat every expert)."""
        for a in self.aggs:
            a.update(expert_losses)
        self.top.update(aggregator_losses)

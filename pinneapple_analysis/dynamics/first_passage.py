r"""First-passage and hitting times: :math:`\tau = \inf\{t \ge 0 : X(t) \in \partial\Omega\}`.

Three ways to get the distribution of :math:`\tau`, one analyzer (:class:`FirstPassageAnalyzer`):

* **Deterministic crossing** of one trajectory (:meth:`FirstPassageAnalyzer.crossing`): the first sample at or
  past the boundary, with the crossing time interpolated linearly inside the sampling interval that brackets it.
  Works on irregular grids and skips missing samples (the bracket then spans the gap and is reported).
* **Empirical distribution** of an ensemble of observed trajectories (:meth:`FirstPassageAnalyzer.from_trajectories`):
  one crossing per member; a member that has not crossed by the end of its window is right-censored there, and the
  survival function is the Kaplan-Meier estimate with Greenwood (log-log) bands.
* **Monte Carlo** of a stochastic differential equation :math:`dX = a(t, X)\,dt + b(t, X)\,dW` with diagonal noise
  (:meth:`FirstPassageAnalyzer.monte_carlo`): Euler-Maruyama paths, with a Brownian-bridge correction for scalar
  thresholds and intervals (the probability that the path crossed and came back between two steps,
  :math:`\exp(-2 d_i d_{i+1} / (b^2 \Delta t))`), optional reflecting boundaries, and paths that never reach the
  boundary censored at the simulation horizon.

Each returns a :class:`FirstPassageResult` (a :class:`~pinneapple_analysis.dynamics.DynamicsResult`) whose
:class:`HittingTimeDistribution` gives the survival function :math:`S(t) = P(\tau > t)`
(:class:`SurvivalFunction`), hitting probabilities :math:`P(\tau \le T)` with confidence intervals
(:class:`CrossingProbability`), quantiles, prediction intervals for a new realisation, and the mean (or, when paths
are censored, the restricted mean :math:`\int_0^T S(t)\,dt`, which is labelled as such: the full mean is not
identifiable from censored data).

Boundaries (:class:`BoundaryCondition`): an upper, lower or two-sided scalar threshold on one component, an interval
:math:`(l, u)` to exit, or a level set :math:`\phi(x) \ge 0` in any dimension. Several boundaries can be analysed on
the same trajectories (one :math:`\tau` per boundary). Reflecting thresholds change the dynamics of a simulation; they
are not targets.

Closed forms, used as references in the tests and available to callers:

* Brownian motion with drift to one barrier: the inverse-Gaussian density and its CDF
  (:func:`brownian_first_passage_pdf`, :func:`brownian_first_passage_cdf`, :func:`brownian_first_passage_mean`);
  Karatzas & Shreve, *Brownian Motion and Stochastic Calculus* (1991), sec. 3.5.C.
* Brownian motion with drift in an interval: exit probabilities, mean exit time and the survival series
  (:func:`brownian_exit_probability`, :func:`brownian_exit_mean`, :func:`brownian_exit_survival`); Cox & Miller,
  *The Theory of Stochastic Processes* (1965), sec. 5.7; Redner, *A Guide to First-Passage Processes* (2001), ch. 2.
* Ornstein-Uhlenbeck mean first-passage time to a level (:func:`ou_mean_first_passage`) from the backward equation
  :math:`D T'' + a T' = -1` (Gardiner, *Stochastic Methods*, 4th ed., sec. 6.2), evaluated by quadrature.

Limits: Euler-Maruyama has weak order 1 in :math:`\Delta t`; with discrete monitoring alone the first-passage time is
biased late by :math:`O(\sqrt{\Delta t})` (Asmussen, Glynn & Pitman 1995), which the bridge correction reduces but does
not remove (it uses the diffusion at the start of the step and places a bridge crossing in the middle of the step;
for an interval it adds the two one-sided probabilities). :meth:`FirstPassageAnalyzer.dt_convergence` measures the
remaining step-size dependence instead of assuming it. A deterministic crossing assumes the trajectory does not leave
and re-enter between two samples; that cannot be checked from the samples.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy import integrate, special, stats

from ._contract import Assumption, Diagnostic, DynamicsResult, MethodCard, make_rng, reproducibility
from .trajectories import Trajectory, TrajectoryEnsemble

__all__ = [
    "BoundaryCondition", "SDE", "SurvivalFunction", "CrossingProbability", "HittingTimeDistribution",
    "FirstPassageResult", "FirstPassageAnalyzer", "kaplan_meier", "wilson_interval",
    "brownian_first_passage_pdf", "brownian_first_passage_cdf", "brownian_first_passage_mean",
    "brownian_exit_probability", "brownian_exit_mean", "brownian_exit_survival", "ou_mean_first_passage",
    "CARD_CROSSING", "CARD_EMPIRICAL", "CARD_MONTE_CARLO",
]

Direction = Literal["up", "down", "either"]


# =========================================================================== boundaries
@dataclass(frozen=True)
class BoundaryCondition:
    """The boundary :math:`\\partial\\Omega` (or a reflecting wall). Build one with the class methods.

    ``threshold(level, direction)``: ``"up"`` is hit when ``x >= level``, ``"down"`` when ``x <= level``,
    ``"either"`` when the component reaches ``level`` from the side it started on. ``interval(lower, upper)``: hit
    on leaving ``(lower, upper)``. ``level_set(phi)``: hit when ``phi(x) >= 0``, with ``phi`` mapping ``(m, d)``
    states to ``(m,)`` (e.g. ``|x| - R`` for leaving a ball). ``behaviour="reflecting"`` (thresholds only) makes it
    a wall for :meth:`FirstPassageAnalyzer.monte_carlo` instead of a target."""
    kind: Literal["threshold", "interval", "level_set"]
    name: str
    component: int | str = 0
    level: float | None = None
    direction: Direction = "up"
    lower: float | None = None
    upper: float | None = None
    phi: Callable[[np.ndarray], np.ndarray] | None = None
    behaviour: Literal["absorbing", "reflecting"] = "absorbing"

    @classmethod
    def threshold(cls, level: float, direction: Direction = "up", component: int | str = 0, *,
                  name: str | None = None, behaviour: str = "absorbing") -> BoundaryCondition:
        if direction not in ("up", "down", "either"):
            raise ValueError("direction must be 'up', 'down' or 'either'")
        if behaviour == "reflecting" and direction == "either":
            raise ValueError("a reflecting threshold needs a side: direction 'up' (wall above) or 'down' (below)")
        return cls("threshold", name or f"{component}{'>=' if direction == 'up' else '<=' if direction == 'down' else '=='}"
                   f"{level:g}", component, float(level), direction, behaviour=behaviour)

    @classmethod
    def interval(cls, lower: float, upper: float, component: int | str = 0, *, name: str | None = None
                 ) -> BoundaryCondition:
        if not lower < upper:
            raise ValueError(f"interval needs lower < upper, got ({lower}, {upper})")
        return cls("interval", name or f"exit({lower:g},{upper:g})", component, lower=float(lower),
                   upper=float(upper))

    @classmethod
    def level_set(cls, phi: Callable[[np.ndarray], np.ndarray], name: str = "phi>=0") -> BoundaryCondition:
        if not callable(phi):
            raise TypeError("phi must be callable: (m, d) states -> (m,) values, hit where phi >= 0")
        return cls("level_set", name, phi=phi)

    def _index(self, names: Sequence[str] | None) -> int:
        if isinstance(self.component, (int, np.integer)):
            return int(self.component)
        if names is None or self.component not in names:
            raise KeyError(f"boundary {self.name!r}: no component {self.component!r} in {names}")
        return list(names).index(self.component)

    def gauge(self, X: np.ndarray, start: np.ndarray, names: Sequence[str] | None = None) -> np.ndarray:
        """Signed distance-like value per state: ``>= 0`` on or past the boundary. ``start`` (the initial states)
        fixes the side of a two-sided threshold."""
        X = np.atleast_2d(X)
        if self.kind == "level_set":
            g = np.asarray(self.phi(X), dtype=np.float64).reshape(-1)
            if g.size != X.shape[0]:
                raise ValueError(f"phi returned {g.size} values for {X.shape[0]} states")
            return g
        c = X[:, self._index(names)]
        if self.kind == "interval":
            return np.maximum(self.lower - c, c - self.upper)
        if self.direction == "up":
            return c - self.level
        if self.direction == "down":
            return self.level - c
        s = np.atleast_2d(start)[:, self._index(names)]
        side = np.where(s < self.level, 1.0, np.where(s > self.level, -1.0, 0.0))
        return np.where(side == 0.0, 0.0, side * (c - self.level))

    @property
    def bridge_supported(self) -> bool:
        return self.kind in ("threshold", "interval")

    def bridge_probability(self, x_old: np.ndarray, x_new: np.ndarray, sigma: np.ndarray, dt: float,
                           names: Sequence[str] | None = None) -> np.ndarray:
        """Probability that a Brownian bridge between two inside states touched the boundary (scalar boundaries)."""
        j = self._index(names)
        a, b, s2 = x_old[:, j], x_new[:, j], np.maximum(sigma[:, j] ** 2, 1e-300)

        def one(level, side):
            d0 = side * (level - a)
            d1 = side * (level - b)
            return np.where((d0 > 0) & (d1 > 0), np.exp(-2.0 * d0 * d1 / (s2 * dt)), 0.0)

        if self.kind == "interval":
            return np.minimum(one(self.upper, 1.0) + one(self.lower, -1.0), 1.0)
        if self.direction == "up":
            return one(self.level, 1.0)
        if self.direction == "down":
            return one(self.level, -1.0)
        return np.where(a < self.level, one(self.level, 1.0), one(self.level, -1.0))

    def reflect(self, X: np.ndarray, names: Sequence[str] | None = None) -> np.ndarray:
        """Mirror states that crossed a reflecting threshold back into the domain (in place)."""
        j = self._index(names)
        c = X[:, j]
        if self.direction == "up":                          # wall above: x > level is forbidden
            X[:, j] = np.where(c > self.level, 2.0 * self.level - c, c)
        else:                                               # wall below
            X[:, j] = np.where(c < self.level, 2.0 * self.level - c, c)
        return X

    def describe(self) -> dict[str, Any]:
        d = {"name": self.name, "kind": self.kind, "behaviour": self.behaviour}
        if self.kind == "threshold":
            d.update(component=self.component, level=self.level, direction=self.direction)
        elif self.kind == "interval":
            d.update(component=self.component, lower=self.lower, upper=self.upper)
        else:
            d["phi"] = getattr(self.phi, "__name__", "callable")
        return d


# =========================================================================== stochastic model
@dataclass(frozen=True)
class SDE:
    """:math:`dX = a(t, X)\\,dt + b(t, X)\\,dW` with diagonal noise (independent Wiener processes per component).

    ``drift(t, X)`` and ``diffusion(t, X)`` take ``X`` of shape ``(m, dim)`` and return arrays broadcastable to it.
    ``noise`` documents the hypothesis (``"additive"``: ``b`` does not depend on ``X``; ``"multiplicative"``
    otherwise); the Euler-Maruyama step reads ``b`` at the start of the step (Ito)."""
    drift: Callable[[float, np.ndarray], np.ndarray]
    diffusion: Callable[[float, np.ndarray], np.ndarray]
    dim: int = 1
    name: str = "sde"
    params: Mapping[str, float] = field(default_factory=dict)
    noise: Literal["additive", "multiplicative"] = "additive"
    names: tuple[str, ...] | None = None

    @classmethod
    def brownian(cls, mu: float = 0.0, sigma: float = 1.0) -> SDE:
        """:math:`dX = \\mu\\,dt + \\sigma\\,dW`."""
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        return cls(lambda t, x: np.full_like(x, mu), lambda t, x: np.full_like(x, sigma), 1, "brownian",
                   {"mu": float(mu), "sigma": float(sigma)})

    @classmethod
    def ornstein_uhlenbeck(cls, theta: float, mean: float = 0.0, sigma: float = 1.0) -> SDE:
        """:math:`dX = -\\theta (X - m)\\,dt + \\sigma\\,dW`."""
        if theta <= 0 or sigma <= 0:
            raise ValueError("theta and sigma must be positive")
        return cls(lambda t, x: -theta * (x - mean), lambda t, x: np.full_like(x, sigma), 1, "ornstein_uhlenbeck",
                   {"theta": float(theta), "mean": float(mean), "sigma": float(sigma)})

    @classmethod
    def geometric_brownian(cls, mu: float, sigma: float) -> SDE:
        """:math:`dX = \\mu X\\,dt + \\sigma X\\,dW` (multiplicative noise)."""
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        return cls(lambda t, x: mu * x, lambda t, x: sigma * x, 1, "geometric_brownian",
                   {"mu": float(mu), "sigma": float(sigma)}, noise="multiplicative")


# =========================================================================== statistics
def wilson_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k / n`` (Wilson 1927, JASA 22:209)."""
    if n <= 0:
        return (0.0, 1.0)
    z = stats.norm.ppf(0.5 + level / 2)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class SurvivalFunction:
    """A right-continuous step function :math:`S(t) = P(\\tau > t)` with pointwise confidence bands.

    ``t``: event times (increasing); ``S[k]`` holds on ``[t[k], t[k+1])`` and ``S = 1`` before ``t[0]``.
    ``horizon``: beyond it nothing was observed and :meth:`__call__` returns ``NaN``."""
    t: np.ndarray
    S: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    at_risk: np.ndarray
    events: np.ndarray
    level: float
    horizon: float
    method: str

    def _eval(self, values: np.ndarray, t) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        k = np.searchsorted(self.t, t, side="right") - 1
        out = np.where(k >= 0, values[np.clip(k, 0, None)] if values.size else 1.0, 1.0)
        return np.where(t > self.horizon, np.nan, out)

    def __call__(self, t) -> np.ndarray:
        return self._eval(self.S, t)

    def band(self, t) -> tuple[np.ndarray, np.ndarray]:
        return self._eval(self.lower, t), self._eval(self.upper, t)


def kaplan_meier(times, observed, *, level: float = 0.95, horizon: float | None = None) -> SurvivalFunction:
    """Kaplan-Meier estimate of the survival function from right-censored times (Kaplan & Meier 1958, JASA 53:457).

    ``observed[i]`` is ``True`` when ``times[i]`` is an event and ``False`` when it is a censoring time. Bands:
    Greenwood variance on the log-log scale (Kalbfleisch & Prentice 2002, sec. 1.4), which stays inside [0, 1].
    Censored times tied with an event time are counted at risk at that event (the usual convention)."""
    t = np.asarray(times, dtype=np.float64)
    d = np.asarray(observed, dtype=bool)
    if t.shape != d.shape or t.ndim != 1 or t.size == 0:
        raise ValueError("times and observed must be 1-D arrays of the same, non-zero length")
    if np.any(~np.isfinite(t)):
        raise ValueError("times must be finite (put never-observed events at their censoring time)")
    z = stats.norm.ppf(0.5 + level / 2)
    ev, n_ev = np.unique(t[d], return_counts=True)
    n_risk = (t.size - np.searchsorted(np.sort(t), ev, side="left")).astype(float)   # O(n log n)
    n_ev = n_ev.astype(float)
    S = np.cumprod(1.0 - n_ev / n_risk) if ev.size else np.array([])
    with np.errstate(divide="ignore", invalid="ignore"):
        gw = np.cumsum(np.where(n_risk > n_ev, n_ev / (n_risk * (n_risk - n_ev)), np.inf)) if ev.size else S
        logS = np.log(S)
        se = np.sqrt(gw) / np.abs(logS)
        ll = np.log(-logS)
        lo = np.exp(-np.exp(ll + z * se))
        hi = np.exp(-np.exp(ll - z * se))
    lo = np.where(S <= 0, 0.0, np.where(np.isfinite(lo), lo, 0.0))
    hi = np.where(S <= 0, 0.0, np.where(np.isfinite(hi), hi, 1.0))
    hz = float(t.max()) if horizon is None else float(horizon)
    return SurvivalFunction(ev, S, lo, hi, n_risk, n_ev, level, hz, "Kaplan-Meier, Greenwood log-log band")


@dataclass(frozen=True)
class CrossingProbability:
    """:math:`P(\\tau \\le T)` with a confidence interval; :attr:`no_crossing` is the complement
    :math:`P(\\tau > T)` (probability of not reaching the boundary within the horizon)."""
    horizon: float
    p: float
    ci: tuple[float, float]
    level: float
    n: int
    method: str

    @property
    def no_crossing(self) -> tuple[float, tuple[float, float]]:
        return 1.0 - self.p, (1.0 - self.ci[1], 1.0 - self.ci[0])


class HittingTimeDistribution:
    """The distribution of one first-passage time, from event and right-censoring times.

    ``times[i]`` is the hitting time of sample ``i`` when ``observed[i]``, else the time its observation stopped
    (``tau > times[i]``). ``source`` says where the samples come from (``"monte carlo"``, ``"observed trajectories"``).
    """

    def __init__(self, times, observed, *, horizon: float | None = None, source: str = "samples", level: float = 0.95):
        self.times = np.asarray(times, dtype=np.float64).copy()
        self.observed = np.asarray(observed, dtype=bool).copy()
        if self.times.shape != self.observed.shape or self.times.ndim != 1 or self.times.size == 0:
            raise ValueError("times and observed must be 1-D, of equal non-zero length")
        self.horizon = float(self.times.max()) if horizon is None else float(horizon)
        self.source = source
        self.level = level
        self._km = kaplan_meier(self.times, self.observed, level=level, horizon=self.horizon)

    @property
    def n(self) -> int:
        return int(self.times.size)

    @property
    def n_events(self) -> int:
        return int(self.observed.sum())

    @property
    def n_censored(self) -> int:
        return self.n - self.n_events

    @property
    def event_times(self) -> np.ndarray:
        return np.sort(self.times[self.observed])

    @property
    def uniformly_censored(self) -> bool:
        """True when every censored sample was censored at the horizon (e.g. Monte Carlo with one end time)."""
        return bool(np.all(self.times[~self.observed] >= self.horizon))

    def survival_function(self) -> SurvivalFunction:
        return self._km

    def survival(self, t) -> np.ndarray:
        return self._km(t)

    def cdf(self, t) -> np.ndarray:
        return 1.0 - self._km(t)

    def hitting_probability(self, T: float | None = None, level: float | None = None) -> CrossingProbability:
        """:math:`P(\\tau \\le T)` (default: the horizon). Without censoring before ``T`` the estimate is the binomial
        proportion with a Wilson interval; otherwise it is the Kaplan-Meier estimate with its band."""
        T = self.horizon if T is None else float(T)
        level = self.level if level is None else level
        if T > self.horizon:
            raise ValueError(f"T = {T} is beyond the horizon {self.horizon}: nothing was observed there")
        early_censor = np.any((~self.observed) & (self.times < T))
        if not early_censor:
            k = int(np.sum(self.observed & (self.times <= T)))
            lo, hi = wilson_interval(k, self.n, level)
            return CrossingProbability(T, k / self.n, (lo, hi), level, self.n, "binomial proportion, Wilson interval")
        km = self._km if level == self.level else kaplan_meier(self.times, self.observed, level=level,
                                                               horizon=self.horizon)
        s = float(km(T))
        slo, shi = km.band(T)
        return CrossingProbability(T, 1.0 - s, (1.0 - float(shi), 1.0 - float(slo)), level, self.n,
                                   "Kaplan-Meier, Greenwood log-log band")

    def quantile(self, q: float) -> float:
        """Smallest ``t`` with :math:`P(\\tau \\le t) \\ge q`; ``inf`` when the estimate never reaches ``q`` within
        the horizon (the quantile lies beyond what was observed)."""
        if not 0 < q < 1:
            raise ValueError("q must be in (0, 1)")
        hit = np.nonzero(1.0 - self._km.S >= q - 1e-12)[0]
        return float(self._km.t[hit[0]]) if hit.size else math.inf

    def quantile_ci(self, q: float) -> tuple[float, float]:
        """Confidence interval of the ``q`` quantile from the survival band (Brookmeyer & Crowley 1982)."""
        lo_hit = np.nonzero(1.0 - self._km.lower >= q - 1e-12)[0]
        hi_hit = np.nonzero(1.0 - self._km.upper >= q - 1e-12)[0]
        lo = float(self._km.t[hi_hit[0]]) if hi_hit.size else math.inf
        hi = float(self._km.t[lo_hit[0]]) if lo_hit.size else math.inf
        return (min(lo, hi), max(lo, hi))

    @property
    def median(self) -> float:
        return self.quantile(0.5)

    def prediction_interval(self, level: float = 0.9) -> tuple[float, float]:
        """Interval that contains the first-passage time of a new realisation with probability ``level`` (central,
        from the estimated quantiles; the upper end is ``inf`` when it lies beyond the horizon)."""
        a = (1 - level) / 2
        return (self.quantile(a), self.quantile(1 - a))

    def mean(self) -> dict[str, Any]:
        """The mean first-passage time when every sample hit; otherwise the restricted mean
        :math:`\\int_0^{T} S(t)\\,dt` up to the horizon, labelled ``restricted`` (a lower bound of the true mean)."""
        if self.n_censored == 0:
            x = self.times
            se = float(x.std(ddof=1) / math.sqrt(x.size)) if x.size > 1 else math.nan
            return {"value": float(x.mean()), "se": se, "kind": "sample mean", "restricted": False}
        km = self._km
        knots = np.concatenate([[0.0], km.t[km.t <= self.horizon], [self.horizon]])
        vals = np.concatenate([[1.0], km.S[km.t <= self.horizon]])
        rmst = float(np.sum(np.diff(knots) * vals))
        if self.uniformly_censored and self.n > 1:         # i.i.d. min(tau, T): plain standard error
            se = float(np.minimum(self.times, self.horizon).std(ddof=1) / math.sqrt(self.n))
        else:
            se = math.nan
        return {"value": rmst, "se": se, "kind": f"restricted mean up to t = {self.horizon:g}", "restricted": True}

    def density(self, bins: int | Sequence[float] = 50) -> tuple[np.ndarray, np.ndarray]:
        """Histogram density of the event times normalised by *all* samples (a defective density when some never
        hit): ``(bin_edges, density)``. Only meaningful without early censoring."""
        e = self.event_times
        hist, edges = np.histogram(e, bins=bins, range=(0.0, self.horizon) if isinstance(bins, int) else None)
        return edges, hist / (self.n * np.diff(edges))

    def summary(self) -> dict[str, Any]:
        p = self.hitting_probability()
        return {"n": self.n, "events": self.n_events, "censored": self.n_censored, "horizon": self.horizon,
                "p_hit": p.p, "p_hit_ci": p.ci, "median": self.median, "mean": self.mean(),
                "prediction_interval_90": self.prediction_interval(0.9)}


# =========================================================================== result
@dataclass
class FirstPassageResult(DynamicsResult):
    """A :class:`DynamicsResult` with the hitting-time distribution of each boundary (``distributions``)."""
    distributions: dict[str, HittingTimeDistribution] = field(default_factory=dict)

    @property
    def distribution(self) -> HittingTimeDistribution:
        if len(self.distributions) != 1:
            raise ValueError(f"{len(self.distributions)} boundaries: use distributions[name], names "
                             f"{list(self.distributions)}")
        return next(iter(self.distributions.values()))


# =========================================================================== method cards
_A_CONT = Assumption("no unobserved excursion", "the trajectory does not reach the boundary and come back between "
                     "two consecutive samples (cannot be checked from the samples)")
_A_INDEP_CENSOR = Assumption("independent censoring", "the end of a member's observation window does not depend on "
                             "how close it is to the boundary (Kaplan-Meier requirement)")
_A_IID = Assumption("exchangeable members", "the trajectories are independent realisations of the same process")
_A_MODEL = Assumption("model adequacy", "the SDE (drift, diffusion, noise type) describes the process; the result "
                      "is a property of the model, not of the physical system")
_A_MARKOV = Assumption("Markov / Ito", "the process is Markov and the noise enters in the Ito sense (b read at the "
                       "start of each step)")

CARD_CROSSING = MethodCard(
    "first_passage.crossing", "trajectory_operator",
    inputs={"trajectory": "Trajectory x(t_i), irregular grid and NaN gaps allowed",
            "boundaries": "one or more BoundaryCondition (threshold / interval / level set)"},
    assumptions=[_A_CONT],
    config={"boundaries": "targets; reflecting boundaries are not used here"},
    outputs={"tau": "interpolated first crossing time per boundary, or None when not crossed",
             "bracket": "sampling interval that contains the crossing", "censored_at": "end of the window"},
    failure_modes=["fewer than two usable samples", "the trajectory starts on or past the boundary (tau = t0)",
                   "a crossing bracketed by a long gap of missing samples (reported, not refused)"])

CARD_EMPIRICAL = MethodCard(
    "first_passage.empirical", "trajectory_operator",
    inputs={"ensemble": "TrajectoryEnsemble of observed or simulated trajectories, each with its own window",
            "boundaries": "targets"},
    assumptions=[_A_CONT, _A_IID, _A_INDEP_CENSOR],
    config={"level": "confidence level of the bands"},
    outputs={"distribution": "Kaplan-Meier survival function with Greenwood bands per boundary",
             "p_hit": "P(tau <= horizon) with CI", "mean": "mean or restricted mean"},
    failure_modes=["all members censored before any event (no estimate of S below 1)",
                   "dependent censoring (biased, not detectable from the data)", "fewer than 2 members"],
    references=["Kaplan & Meier 1958, JASA 53:457", "Kalbfleisch & Prentice 2002, The Statistical Analysis of "
                "Failure Time Data, sec. 1.4"])

CARD_MONTE_CARLO = MethodCard(
    "first_passage.monte_carlo", "trajectory_operator",
    inputs={"sde": "SDE dX = a dt + b dW (diagonal noise)", "x0": "initial state(s)",
            "boundaries": "targets", "reflecting": "reflecting thresholds (optional)"},
    assumptions=[_A_MODEL, _A_MARKOV],
    config={"t_max": "simulation horizon (paths not hit are censored there)", "dt": "Euler-Maruyama step",
            "n_paths": "Monte Carlo sample size", "bridge": "Brownian-bridge crossing correction", "seed": "RNG seed"},
    outputs={"distribution": "hitting-time distribution per boundary", "p_hit": "P(tau <= T) with Wilson CI",
             "mean": "mean or restricted mean with Monte Carlo standard error"},
    failure_modes=["dt not small against the time scale of tau (weak-order-1 bias; check with dt_convergence)",
                   "level-set boundaries are monitored only at the steps (late bias O(sqrt dt))",
                   "horizon shorter than most hitting times (mean not identifiable; restricted mean reported)",
                   "non-diagonal noise is not supported"],
    references=["Kloeden & Platen 1992, Numerical Solution of SDEs, ch. 9", "Asmussen, Glynn & Pitman 1995, Ann. "
                "Appl. Probab. 5:875", "Gobet 2000, Stoch. Proc. Appl. 87:167 (bridge correction)"])


# =========================================================================== analyzer
def _as_boundaries(boundaries) -> list[BoundaryCondition]:
    bcs = [boundaries] if isinstance(boundaries, BoundaryCondition) else list(boundaries)
    if not bcs:
        raise ValueError("at least one boundary is needed")
    names = [b.name for b in bcs]
    if len(set(names)) != len(names):
        raise ValueError(f"boundary names must be unique, got {names}")
    if any(b.behaviour != "absorbing" for b in bcs):
        raise ValueError("target boundaries must be absorbing; pass reflecting walls as reflecting=[...]")
    return bcs


def _first_crossing(t: np.ndarray, g: np.ndarray) -> tuple[float | None, tuple[float, float] | None]:
    """First crossing of g from < 0 to >= 0 on samples (t, g); linear interpolation inside the bracket."""
    hit = np.nonzero(g >= 0)[0]
    if hit.size == 0:
        return None, None
    i = int(hit[0])
    if i == 0:
        return float(t[0]), (float(t[0]), float(t[0]))
    g0, g1 = g[i - 1], g[i]
    frac = 0.0 if g1 == g0 else -g0 / (g1 - g0)
    return float(t[i - 1] + frac * (t[i] - t[i - 1])), (float(t[i - 1]), float(t[i]))


class FirstPassageAnalyzer:
    """First-passage analysis of one or more target boundaries; see the module docstring.

    ``horizon``: the ``T`` of :math:`P(\\tau \\le T)` (default: the end of the data or of the simulation)."""

    cards = {"crossing": CARD_CROSSING, "empirical": CARD_EMPIRICAL, "monte_carlo": CARD_MONTE_CARLO}

    def __init__(self, boundaries: BoundaryCondition | Sequence[BoundaryCondition], *, horizon: float | None = None,
                 level: float = 0.95):
        self.boundaries = _as_boundaries(boundaries)
        self.horizon = horizon
        self.level = level

    # ------------------------------------------------------------------ deterministic, one trajectory
    def _crossing_one(self, tr: Trajectory, bc: BoundaryCondition) -> dict[str, Any]:
        X = tr.x
        if bc.kind == "level_set":
            usable = ~np.isnan(X).any(axis=1)
        else:
            usable = ~np.isnan(X[:, bc._index(tr.names)])
        t, Xu = tr.t[usable], X[usable]
        if t.size == 0:
            return {"tau": None, "crossed": False, "bracket": None, "censored_at": None, "usable": 0,
                    "gap_bracket": False, "started_inside": False}
        g = bc.gauge(Xu, Xu[:1], tr.names)
        tau, bracket = _first_crossing(t, g)
        gap = False
        if bracket is not None and bracket[1] > bracket[0]:
            inside = (tr.t > bracket[0]) & (tr.t < bracket[1])
            gap = bool(inside.any())                       # samples skipped as missing inside the bracket
        return {"tau": tau, "crossed": tau is not None, "bracket": bracket,
                "censored_at": None if tau is not None else tr.observed_until, "usable": int(t.size),
                "gap_bracket": gap, "started_inside": bool(g[0] >= 0)}

    def crossing(self, trajectory: Trajectory) -> FirstPassageResult:
        """First crossing of each boundary by one trajectory (deterministic: no distribution is inferred)."""
        t0 = time.perf_counter()
        tr = trajectory
        est, unc, diags, failures = {}, {}, [], []
        for bc in self.boundaries:
            r = self._crossing_one(tr, bc)
            est[bc.name] = {"tau": r["tau"], "crossed": r["crossed"], "censored_at": r["censored_at"]}
            unc[bc.name] = {"bracket": r["bracket"], "kind": "sampling interval containing the crossing (exact "
                            "bounds if there is no unobserved excursion)"}
            if r["usable"] < 2:
                failures.append(f"{bc.name}: {r['usable']} usable samples")
            if r["started_inside"]:
                diags.append(Diagnostic(f"{bc.name}: started_on_boundary", True, "warn",
                                        "the first usable sample is already on or past the boundary: tau = t0"))
            if r["gap_bracket"]:
                w = r["bracket"][1] - r["bracket"][0]
                diags.append(Diagnostic(f"{bc.name}: crossing_across_gap", w, "warn",
                                        "missing samples inside the bracket: the crossing time is uncertain by the "
                                        "gap width"))
            if r["bracket"] is not None:
                diags.append(Diagnostic(f"{bc.name}: bracket_width", r["bracket"][1] - r["bracket"][0], "info"))
        samp = tr.sampling()
        diags.append(Diagnostic("sampling_irregularity", samp.get("irregularity", 1.0), "info"))
        cfg = {"boundaries": [b.describe() for b in self.boundaries]}
        rep = reproducibility(CARD_CROSSING.name, cfg, [tr], wall_time_s=time.perf_counter() - t0)
        return FirstPassageResult(CARD_CROSSING.name, est, unc, [_A_CONT], cfg, diags, {},
                                  applicable=not failures, failures=failures,
                                  limits=["linear interpolation inside the bracket: error O(dt^2) for a smooth "
                                          "trajectory, unbounded for a rough one"],
                                  reproducibility=rep, details={"source": dict(tr.source)})

    # ------------------------------------------------------------------ empirical, many trajectories
    def from_trajectories(self, ensemble: TrajectoryEnsemble | Sequence[Trajectory]) -> FirstPassageResult:
        """Hitting-time distributions from observed trajectories, each censored at the end of its own window."""
        t0 = time.perf_counter()
        ens = ensemble if isinstance(ensemble, TrajectoryEnsemble) else TrajectoryEnsemble(list(ensemble))
        horizon = float(ens.horizons.max()) if self.horizon is None else float(self.horizon)
        dists, est, unc, diags, failures = {}, {}, {}, [], []
        if len(ens) < 2:
            failures.append(f"{len(ens)} trajectory: a distribution needs at least 2")
        for bc in self.boundaries:
            times, obs, widths, gaps = [], [], [], 0
            for tr in ens:
                r = self._crossing_one(tr, bc)
                if r["crossed"]:
                    times.append(r["tau"])
                    obs.append(True)
                    widths.append(r["bracket"][1] - r["bracket"][0])
                    gaps += r["gap_bracket"]
                else:
                    times.append(tr.observed_until)
                    obs.append(False)
            times, obs = np.array(times), np.array(obs)
            obs = obs & (times <= horizon)                 # events after the horizon are censored at it
            times = np.minimum(times, horizon)
            dist = HittingTimeDistribution(times, obs, horizon=horizon, source="observed trajectories",
                                           level=self.level)
            dists[bc.name] = dist
            est[bc.name], unc[bc.name] = self._summarise(dist)
            if dist.n_events == 0:
                failures.append(f"{bc.name}: no member crossed within its window")
            diags.append(Diagnostic(f"{bc.name}: censored_fraction", dist.n_censored / dist.n,
                                    "warn" if dist.n_censored else "ok"))
            if widths:
                diags.append(Diagnostic(f"{bc.name}: max_bracket_width", float(max(widths)), "info",
                                        "interval censoring by the sampling, ignored by the interpolated times"))
            if gaps:
                diags.append(Diagnostic(f"{bc.name}: crossings_across_gaps", gaps, "warn"))
        cfg = {"boundaries": [b.describe() for b in self.boundaries], "horizon": horizon, "level": self.level}
        rep = reproducibility(CARD_EMPIRICAL.name, cfg, [ens], wall_time_s=time.perf_counter() - t0)
        return FirstPassageResult(CARD_EMPIRICAL.name, est, unc, [_A_CONT, _A_IID, _A_INDEP_CENSOR], cfg, diags, {},
                                  applicable=not failures, failures=failures,
                                  limits=["crossing times are interpolated inside the sampling interval; interval "
                                          "censoring is not modelled"],
                                  reproducibility=rep, distributions=dists)

    # ------------------------------------------------------------------ Monte Carlo of an SDE
    def simulate(self, sde: SDE, x0, *, t_max: float, dt: float, n_paths: int, seed: int | None = 0,
                 bridge: bool = True, reflecting: Sequence[BoundaryCondition] = (), keep_paths: int = 0
                 ) -> dict[str, Any]:
        """Raw Monte Carlo: hitting times ``(n_boundaries, n_paths)`` (``inf`` when not hit by ``t_max``) and
        diagnostics. :meth:`monte_carlo` wraps it into a result."""
        if t_max <= 0 or dt <= 0 or n_paths < 1:
            raise ValueError("t_max, dt and n_paths must be positive")
        if dt > t_max:
            raise ValueError(f"dt = {dt} is larger than t_max = {t_max}")
        for w in reflecting:
            if w.kind != "threshold" or w.behaviour != "reflecting":
                raise ValueError("reflecting walls must be BoundaryCondition.threshold(..., behaviour='reflecting')")
        rng, seed = make_rng(seed)
        d = sde.dim
        X = np.broadcast_to(np.asarray(x0, dtype=np.float64), (n_paths, d)).copy() if np.ndim(x0) < 2 \
            else np.asarray(x0, dtype=np.float64).copy()
        if X.shape != (n_paths, d):
            raise ValueError(f"x0 must be a scalar, a ({d},) state or ({n_paths}, {d}) states, got {np.shape(x0)}")
        names = sde.names
        start = X.copy()
        bcs = self.boundaries
        nb = len(bcs)
        tau = np.full((nb, n_paths), np.inf)
        g_old = np.stack([bc.gauge(X, start, names) for bc in bcs])
        tau[g_old >= 0] = 0.0
        n_steps = int(math.ceil(t_max / dt - 1e-12))
        h = t_max / n_steps
        sq = math.sqrt(h)
        stride = max(1, n_steps // 200)
        kept = [X[:keep_paths].copy()] if keep_paths else []
        kept_t = [0.0]
        n_bridge = np.zeros(nb, dtype=int)
        alive = np.isinf(tau).any(axis=0)
        t = 0.0
        for k in range(n_steps):
            if not alive.any() and not keep_paths:
                break
            idx = np.nonzero(alive)[0] if not keep_paths else np.arange(n_paths)
            Xa = X[idx]
            a = np.broadcast_to(np.asarray(sde.drift(t, Xa), dtype=np.float64), Xa.shape)
            b = np.broadcast_to(np.asarray(sde.diffusion(t, Xa), dtype=np.float64), Xa.shape)
            Xn = Xa + a * h + b * (sq * rng.standard_normal(Xa.shape))
            for w in reflecting:
                w.reflect(Xn, names)
            for j, bc in enumerate(bcs):
                open_ = np.isinf(tau[j, idx])
                if not open_.any():
                    continue
                gn = bc.gauge(Xn, start[idx], names)
                go = g_old[j, idx]
                crossed = open_ & (gn >= 0)
                if crossed.any():
                    frac = np.where(gn[crossed] == go[crossed], 0.0, -go[crossed] / (gn[crossed] - go[crossed]))
                    tau[j, idx[crossed]] = t + np.clip(frac, 0.0, 1.0) * h
                if bridge and bc.bridge_supported:
                    still = open_ & ~crossed
                    if still.any():
                        p = bc.bridge_probability(Xa[still], Xn[still], b[still], h, names)
                        hit = rng.random(p.size) < p
                        if hit.any():
                            tau[j, idx[still][hit]] = t + 0.5 * h
                            n_bridge[j] += int(hit.sum())
                g_old[j, idx] = gn
            X[idx] = Xn
            t = (k + 1) * h
            alive = np.isinf(tau).any(axis=0)
            if keep_paths and ((k + 1) % stride == 0 or k == n_steps - 1):
                kept.append(X[:keep_paths].copy())
                kept_t.append(t)
        out = {"tau": tau, "seed": seed, "dt": h, "n_steps": n_steps, "t_max": float(t_max), "n_paths": n_paths,
               "bridge_hits": n_bridge}
        if keep_paths:
            out["paths"] = TrajectoryEnsemble.from_paths(np.array(kept_t), np.stack(kept, axis=1), names,
                                                         kind="stochastic", sde=sde.name, seed=seed)
        return out

    def monte_carlo(self, sde: SDE, x0, *, t_max: float, dt: float, n_paths: int = 10_000, seed: int | None = 0,
                    bridge: bool = True, reflecting: Sequence[BoundaryCondition] = (), keep_paths: int = 0
                    ) -> FirstPassageResult:
        """Monte Carlo hitting-time distributions of ``sde`` from ``x0``; see the module docstring."""
        t0 = time.perf_counter()
        sim = self.simulate(sde, x0, t_max=t_max, dt=dt, n_paths=n_paths, seed=seed, bridge=bridge,
                            reflecting=reflecting, keep_paths=keep_paths)
        horizon = float(t_max) if self.horizon is None else min(float(self.horizon), float(t_max))
        dists, est, unc, diags, failures = {}, {}, {}, [], []
        for j, bc in enumerate(self.boundaries):
            tj = sim["tau"][j]
            obs = np.isfinite(tj) & (tj <= horizon)
            dist = HittingTimeDistribution(np.where(obs, tj, horizon), obs, horizon=horizon, source="monte carlo",
                                           level=self.level)
            dists[bc.name] = dist
            est[bc.name], unc[bc.name] = self._summarise(dist)
            if dist.n_events == 0:
                failures.append(f"{bc.name}: no path reached the boundary by t = {horizon:g}")
            cf = dist.n_censored / dist.n
            diags.append(Diagnostic(f"{bc.name}: censored_fraction", cf,
                                    "fail" if cf == 1 else "warn" if cf > 0.01 else "ok",
                                    "paths not hit by the horizon: the mean reported is the restricted mean up to "
                                    "the horizon (a lower bound), and upper quantiles may be infinite" if cf else "",
                                    threshold=0.01))
            m = dist.mean()
            if np.isfinite(m["se"]) and m["value"] > 0:
                rse = m["se"] / m["value"]
                diags.append(Diagnostic(f"{bc.name}: mc_relative_se_mean", rse, "ok" if rse <= 0.02 else "warn",
                                        threshold=0.02))
            med = dist.median
            if np.isfinite(med) and med > 0:
                r = sim["dt"] / med
                diags.append(Diagnostic(f"{bc.name}: dt_over_median_tau", r, "ok" if r <= 0.01 else "warn",
                                        "step not small against the hitting time scale: run dt_convergence"
                                        if r > 0.01 else "", threshold=0.01))
            monitor = "Brownian bridge" if (bridge and bc.bridge_supported) else "discrete steps only"
            diags.append(Diagnostic(f"{bc.name}: monitoring", monitor,
                                    "info" if monitor == "Brownian bridge" else "warn",
                                    "" if monitor == "Brownian bridge" else "first passage biased late by O(sqrt dt)"))
            started = float(np.mean(sim["tau"][j] == 0.0))
            if started:
                diags.append(Diagnostic(f"{bc.name}: started_on_boundary", started, "warn"))
        cfg = {"sde": sde.name, "params": dict(sde.params), "noise": sde.noise, "t_max": float(t_max), "dt": sim["dt"],
               "n_paths": n_paths, "bridge": bridge, "horizon": horizon, "level": self.level,
               "boundaries": [b.describe() for b in self.boundaries],
               "reflecting": [w.describe() for w in reflecting]}
        x0_arr = np.asarray(x0, dtype=np.float64)
        rep = reproducibility(CARD_MONTE_CARLO.name, cfg, [x0_arr], seed=sim["seed"],
                              wall_time_s=time.perf_counter() - t0)
        conv = {"n_paths": n_paths, "dt": sim["dt"], "n_steps": sim["n_steps"],
                "bridge_hits": {bc.name: int(sim["bridge_hits"][j]) for j, bc in enumerate(self.boundaries)},
                "note": "step-size dependence not measured here; see FirstPassageAnalyzer.dt_convergence"}
        details = {"paths": sim["paths"]} if keep_paths else {}
        return FirstPassageResult(CARD_MONTE_CARLO.name, est, unc, [_A_MODEL, _A_MARKOV], cfg, diags, conv,
                                  applicable=not failures, failures=failures,
                                  limits=["Euler-Maruyama: weak order 1 in dt", "results describe the SDE model"],
                                  reproducibility=rep, details=details, distributions=dists)

    def _summarise(self, dist: HittingTimeDistribution) -> tuple[dict[str, Any], dict[str, Any]]:
        p = dist.hitting_probability(level=self.level)
        m = dist.mean()
        est = {"p_hit": p.p, "p_no_hit": 1.0 - p.p, "horizon": dist.horizon, "median": dist.median,
               "mean": m["value"], "mean_kind": m["kind"], "n": dist.n, "events": dist.n_events}
        unc = {"p_hit_ci": p.ci, "p_hit_method": p.method, "level": p.level, "mean_se": m["se"],
               "median_ci": dist.quantile_ci(0.5) if dist.n_events else (math.inf, math.inf),
               "prediction_interval_90": dist.prediction_interval(0.9)}
        return est, unc

    # ------------------------------------------------------------------ convergence and sensitivity
    def _statistic(self, res: FirstPassageResult, statistic: str, name: str | None) -> tuple[float, float]:
        nm = name or self.boundaries[0].name
        e, u = res.estimate[nm], res.uncertainty[nm]
        if statistic == "p_hit":
            lo, hi = u["p_hit_ci"]
            z = stats.norm.ppf(0.5 + self.level / 2)
            return e["p_hit"], (hi - lo) / (2 * z)
        if statistic == "mean":
            return e["mean"], u["mean_se"]
        if statistic == "median":
            lo, hi = u["median_ci"]
            z = stats.norm.ppf(0.5 + self.level / 2)
            return e["median"], (hi - lo) / (2 * z) if np.isfinite(hi - lo) else math.nan
        raise ValueError("statistic must be 'p_hit', 'mean' or 'median'")

    def dt_convergence(self, sde: SDE, x0, *, t_max: float, dts: Sequence[float], n_paths: int = 20_000,
                       seed: int = 0, statistic: str = "mean", boundary: str | None = None, bridge: bool = True,
                       reflecting: Sequence[BoundaryCondition] = ()) -> DynamicsResult:
        """The statistic for each step in ``dts`` (largest first), its Monte Carlo error, the successive changes,
        the observed order when three steps with a constant ratio are given, and a Richardson extrapolation."""
        t0 = time.perf_counter()
        dts = sorted((float(v) for v in dts), reverse=True)
        if len(dts) < 2:
            raise ValueError("give at least two step sizes")
        vals, ses = [], []
        for k, h in enumerate(dts):
            r = self.monte_carlo(sde, x0, t_max=t_max, dt=h, n_paths=n_paths, seed=seed + k, bridge=bridge,
                                 reflecting=reflecting)
            v, s = self._statistic(r, statistic, boundary)
            vals.append(v)
            ses.append(s)
        vals, ses = np.array(vals), np.array(ses)
        diffs = np.abs(np.diff(vals))
        conv: dict[str, Any] = {"dt": dts, "value": vals, "mc_se": ses, "successive_change": diffs}
        diags = []
        order = math.nan
        ratios = np.array(dts[:-1]) / np.array(dts[1:])
        if len(dts) >= 3 and np.allclose(ratios, ratios[0], rtol=1e-6) and diffs[-1] > 0:
            order = float(np.log(diffs[-2] / diffs[-1]) / np.log(ratios[0]))
        conv["observed_order"] = order
        noise_floor = float(2 * np.sqrt(ses[-1] ** 2 + ses[-2] ** 2))
        resolved = bool(diffs[-1] > noise_floor)
        diags.append(Diagnostic("last_change_vs_mc_noise", float(diffs[-1]), "info" if resolved else "ok",
                                "the last refinement still changes the result beyond Monte Carlo noise" if resolved
                                else "the last refinement is within Monte Carlo noise", threshold=noise_floor))
        est = {"finest": float(vals[-1]), "dt_finest": dts[-1]}
        if np.isfinite(order) and order > 0:
            est["richardson"] = float(vals[-1] + (vals[-1] - vals[-2]) / (ratios[0] ** order - 1))
        cfg = {"sde": sde.name, "params": dict(sde.params), "t_max": t_max, "dts": dts, "n_paths": n_paths,
               "statistic": statistic, "bridge": bridge}
        rep = reproducibility("first_passage.dt_convergence", cfg, [np.asarray(x0, float)], seed=seed,
                              wall_time_s=time.perf_counter() - t0)
        return DynamicsResult("first_passage.dt_convergence", est, {"mc_se_finest": float(ses[-1])},
                              [_A_MODEL, _A_MARKOV], cfg, diags, conv,
                              limits=["the observed order is noisy when changes approach the Monte Carlo error"],
                              reproducibility=rep)

    def sensitivity(self, models: Mapping[Any, SDE] | Callable[[float], SDE], values: Sequence[float] | None = None,
                    *, x0, t_max: float, dt: float, n_paths: int = 20_000, seed: int = 0, statistic: str = "p_hit",
                    boundary: str | None = None, bridge: bool = True) -> DynamicsResult:
        """The statistic across models with common random numbers (same seed for every model).

        ``models`` is either a mapping ``{label: SDE}`` (competing stochastic hypotheses, e.g. additive vs
        multiplicative noise) or a callable ``value -> SDE`` evaluated at ``values`` (a parameter sweep). For a sweep
        the derivative ``d statistic / d value`` is estimated by central differences."""
        t0 = time.perf_counter()
        if callable(models) and not isinstance(models, Mapping):
            if values is None or len(values) < 2:
                raise ValueError("a parameter sweep needs at least two values")
            labels = [float(v) for v in values]
            sdes = [models(v) for v in labels]
        else:
            labels = list(models)
            sdes = [models[k] for k in labels]
        vals, ses = [], []
        for s in sdes:
            r = self.monte_carlo(s, x0, t_max=t_max, dt=dt, n_paths=n_paths, seed=seed, bridge=bridge)
            v, e = self._statistic(r, statistic, boundary)
            vals.append(v)
            ses.append(e)
        vals, ses = np.array(vals), np.array(ses)
        est: dict[str, Any] = {"labels": labels, "value": vals, "mc_se": ses}
        diags = []
        if all(isinstance(v, float) for v in labels):
            x = np.array(labels)
            est["derivative"] = np.gradient(vals, x) if x.size > 2 else (vals[1:] - vals[:-1]) / (x[1:] - x[:-1])
        spread = float(vals.max() - vals.min())
        noise = float(2 * np.sqrt(np.max(ses) ** 2 * 2))
        diags.append(Diagnostic("spread_vs_mc_noise", spread, "info", "differences larger than the threshold are "
                                "resolved by this sample size (independent-error bound; common random numbers make "
                                "it conservative)", threshold=noise))
        cfg = {"labels": [str(v) for v in labels], "t_max": t_max, "dt": dt, "n_paths": n_paths, "seed": seed,
               "statistic": statistic, "bridge": bridge}
        rep = reproducibility("first_passage.sensitivity", cfg, [np.asarray(x0, float)], seed=seed,
                              wall_time_s=time.perf_counter() - t0)
        return DynamicsResult("first_passage.sensitivity", est, {"mc_se": ses}, [_A_MODEL], cfg, diags,
                              {"n_paths": n_paths, "dt": dt}, limits=["each model is simulated with the same seed "
                                                                       "(common random numbers)"],
                              reproducibility=rep)


# =========================================================================== closed forms
def _barrier(x0: float, b: float, mu: float) -> tuple[float, float]:
    """Distance to the barrier and the drift towards it (mirrored for a barrier below)."""
    return (b - x0, mu) if b >= x0 else (x0 - b, -mu)


def brownian_first_passage_pdf(t, x0: float, b: float, mu: float, sigma: float) -> np.ndarray:
    """Density of the first time :math:`X = x_0 + \\mu t + \\sigma W` reaches ``b``:
    :math:`f(t) = a / (\\sigma \\sqrt{2\\pi t^3})\\, \\exp(-(a - \\nu t)^2 / (2\\sigma^2 t))`, ``a = |b - x0|``,
    ``nu`` the drift towards ``b``. Defective (integrates to :math:`e^{2\\nu a/\\sigma^2} < 1`) when ``nu < 0``."""
    a, nu = _barrier(x0, b, mu)
    t = np.asarray(t, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = a / (sigma * np.sqrt(2 * np.pi * t ** 3)) * np.exp(-((a - nu * t) ** 2) / (2 * sigma ** 2 * t))
    return np.where(t > 0, f, 0.0)


def brownian_first_passage_cdf(t, x0: float, b: float, mu: float, sigma: float) -> np.ndarray:
    """:math:`P(\\tau \\le t) = \\Phi((\\nu t - a)/(\\sigma\\sqrt t)) + e^{2\\nu a/\\sigma^2}\\,\\Phi((-a - \\nu t)/(\\sigma\\sqrt t))`."""
    a, nu = _barrier(x0, b, mu)
    t = np.asarray(t, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        st = sigma * np.sqrt(t)
        first = stats.norm.cdf((nu * t - a) / st)
        # e^{2 nu a / s^2} Phi(-(a + nu t)/st), evaluated in logs to avoid overflow for large nu a
        log2 = 2 * nu * a / sigma ** 2 + stats.norm.logcdf(-(a + nu * t) / st)
        F = first + np.exp(log2)
    return np.where(t > 0, np.clip(F, 0.0, 1.0), 0.0 if a > 0 else 1.0)


def brownian_first_passage_mean(x0: float, b: float, mu: float, sigma: float) -> float:
    """:math:`E[\\tau] = a / \\nu` for drift towards the barrier; ``inf`` otherwise (``nu = 0``: finite almost surely
    but infinite mean; ``nu < 0``: the barrier may never be reached)."""
    a, nu = _barrier(x0, b, mu)
    return a / nu if nu > 0 else math.inf


def brownian_exit_probability(x0: float, lower: float, upper: float, mu: float, sigma: float) -> float:
    """Probability of leaving ``(lower, upper)`` through ``upper`` (scale function of drifted Brownian motion)."""
    if not lower <= x0 <= upper:
        raise ValueError("x0 must lie in [lower, upper]")
    L, y = upper - lower, x0 - lower
    k = 2 * mu / sigma ** 2
    if abs(k * L) < 1e-12:
        return y / L
    return float(np.expm1(-k * y) / np.expm1(-k * L))


def brownian_exit_mean(x0: float, lower: float, upper: float, mu: float, sigma: float) -> float:
    """Mean exit time from ``(lower, upper)``: :math:`y(L-y)/\\sigma^2` without drift,
    :math:`(L\\,P_{up} - y)/\\mu` with drift (``y = x0 - lower``, ``L = upper - lower``)."""
    L, y = upper - lower, x0 - lower
    if abs(2 * mu * L / sigma ** 2) < 1e-8:
        return y * (L - y) / sigma ** 2
    return float((L * brownian_exit_probability(x0, lower, upper, mu, sigma) - y) / mu)


def brownian_exit_survival(t, x0: float, lower: float, upper: float, mu: float, sigma: float,
                           n_terms: int | None = None) -> np.ndarray:
    """:math:`P(\\tau > t)` of exit from ``(lower, upper)`` by eigenfunction expansion of the backward equation.

    With ``k = mu/sigma^2``, ``S = e^{-k y - \\mu^2 t/(2\\sigma^2)} \\sum_n c_n \\sin(n\\pi y/L) e^{-\\lambda_n t}``,
    ``\\lambda_n = n^2\\pi^2\\sigma^2/(2L^2)``, ``c_n = (2/L)\\,a_n (1 - (-1)^n e^{kL})/(k^2 + a_n^2)``,
    ``a_n = n\\pi/L``. The series converges fast for ``t`` away from 0; ``n_terms`` defaults to enough terms for
    ``exp(-lambda_n t) < 1e-16`` at the smallest positive ``t`` (at most 20 000)."""
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    L, y = upper - lower, x0 - lower
    if not 0 <= y <= L:
        raise ValueError("x0 must lie in [lower, upper]")
    k = mu / sigma ** 2
    lam1 = np.pi ** 2 * sigma ** 2 / (2 * L ** 2)
    tpos = t[t > 0]
    if n_terms is None:
        tmin = tpos.min() if tpos.size else 1.0
        n_terms = int(min(20_000, max(50, math.ceil(math.sqrt(37.0 / (lam1 * tmin))))))
    n = np.arange(1, n_terms + 1)
    a = n * np.pi / L
    c = (2 / L) * a * (1 - (-1.0) ** n * np.exp(k * L)) / (k ** 2 + a ** 2)
    modes = c * np.sin(a * y)
    S = np.exp(-k * y - mu ** 2 * t[:, None] / (2 * sigma ** 2)) * modes * np.exp(-lam1 * n ** 2 * t[:, None])
    out = S.sum(axis=1)
    return np.where(t > 0, np.clip(out, 0.0, 1.0), 1.0 if 0 < y < L else 0.0)


def ou_mean_first_passage(x0: float, level: float, theta: float, mean: float = 0.0, sigma: float = 1.0) -> float:
    """Mean first-passage time of :math:`dX = -\\theta (X - m)dt + \\sigma dW` from ``x0`` to ``level``.

    From the backward equation :math:`(\\sigma^2/2) T'' - \\theta (x - m) T' = -1` with :math:`T(\\text{level}) = 0`
    and a natural boundary on the far side:
    :math:`T(x_0) = \\frac{\\sqrt{\\pi}}{\\sigma\\sqrt{\\theta}} \\int_{x_0}^{\\text{level}}
    \\operatorname{erfcx}(-(y - m)\\sqrt{\\theta}/\\sigma)\\,dy` for an upper level (mirrored for a lower one),
    evaluated by adaptive quadrature (``scipy.integrate.quad``)."""
    if theta <= 0 or sigma <= 0:
        raise ValueError("theta and sigma must be positive")
    if level < x0:                                         # mirror: X -> -X
        return ou_mean_first_passage(-x0, -level, theta, -mean, sigma)
    if level == x0:
        return 0.0
    r = math.sqrt(theta) / sigma
    val, err = integrate.quad(lambda y: special.erfcx(-(y - mean) * r), x0, level, limit=200)
    return float(math.sqrt(math.pi) / (sigma * math.sqrt(theta)) * val)

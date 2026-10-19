"""Monte-Carlo propagation of input (manufacturing, operating) tolerances through any model.

Each input ``i`` is drawn independently around its nominal value ``x0_i``:

* ``"normal"``: ``x_i ~ N(x0_i, t_i^2)`` -- the tolerance ``t_i`` is the standard deviation;
* ``"uniform"``: ``x_i ~ U(x0_i - t_i, x0_i + t_i)`` -- the tolerance is the half-width, variance ``t_i^2 / 3``
  (``(b - a)^2 / 12`` with ``b - a = 2 t_i``).

The model is evaluated once on the ``(n, d)`` sample matrix and the output statistics are the sample mean, standard
deviation and percentiles (JCGM 101:2008, "Propagation of distributions using a Monte Carlo method"). The
probability of meeting the specification limits ``lo <= y <= hi`` (jointly over the outputs and per output) comes
with the Wilson score interval for a binomial proportion (Wilson 1927; Brown, Cai & DasGupta 2001), which stays
inside [0, 1] and keeps its coverage when the proportion is close to 0 or 1 -- the usual case for a yield.

References
----------
JCGM 101:2008. *Evaluation of measurement data -- Supplement 1 to the GUM -- Propagation of distributions using a
Monte Carlo method*. BIPM.

Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *JASA* 22(158),
209-212.

Brown, L. D., Cai, T. T. & DasGupta, A. (2001). Interval estimation for a binomial proportion. *Statistical
Science* 16(2), 101-133.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

__all__ = ["ToleranceResult", "propagate", "wilson_interval"]


def wilson_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k / n`` (Wilson 1927; Brown, Cai & DasGupta 2001, eq. 4):

        (p + z^2 / 2n -+ z sqrt(p (1 - p) / n + z^2 / 4n^2)) / (1 + z^2 / n),   z = Phi^-1((1 + level) / 2).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    z = stats.norm.ppf(0.5 + level / 2.0)
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


@dataclass
class ToleranceResult:
    """Result of :func:`propagate`. Output arrays have shape ``(m,)`` (``m = 1`` for a scalar model).

    ``p_within`` is the probability that all outputs meet their limits jointly, ``p_within_ci`` its Wilson interval,
    and ``p_within_each`` the per-output probabilities; they are ``None`` without ``limits``.
    """
    mean: np.ndarray
    std: np.ndarray
    percentiles: dict[float, np.ndarray]
    n: int
    inputs: np.ndarray = field(repr=False)
    outputs: np.ndarray = field(repr=False)
    p_within: float | None = None
    p_within_ci: tuple[float, float] | None = None
    p_within_each: np.ndarray | None = None


def _sample(nominal: np.ndarray, tol: np.ndarray, dist, n: int, rng: np.random.Generator) -> np.ndarray:
    d = nominal.size
    kinds = [dist] * d if isinstance(dist, str) else list(dist)
    if len(kinds) != d:
        raise ValueError(f"dist has {len(kinds)} entries for {d} inputs")
    X = np.empty((n, d))
    for i, kind in enumerate(kinds):
        if kind == "normal":
            X[:, i] = rng.normal(nominal[i], tol[i], size=n)
        elif kind == "uniform":
            X[:, i] = rng.uniform(nominal[i] - tol[i], nominal[i] + tol[i], size=n)
        else:
            raise ValueError(f"unknown dist {kind!r}; use 'normal' or 'uniform'")
    return X


def propagate(fn: Callable[[np.ndarray], np.ndarray], nominal, tolerances, n: int = 10000,
              dist: str | Sequence[str] = "normal", limits=None, seed: int = 0, level: float = 0.95,
              percentiles: Sequence[float] = (1, 5, 50, 95, 99)) -> ToleranceResult:
    """Propagate independent input tolerances through ``fn`` by Monte-Carlo sampling (JCGM 101:2008).

    Parameters
    ----------
    fn : model evaluated on an ``(n, d)`` array, returning ``(n,)`` or ``(n, m)``.
    nominal : nominal inputs (d,).
    tolerances : per-input standard deviation (``"normal"``) or half-width (``"uniform"``), shape (d,) or scalar.
    n : number of Monte-Carlo samples.
    dist : ``"normal"``, ``"uniform"``, or one of those per input.
    limits : optional specification ``(lo, hi)``, each a scalar or (m,) array; ``None`` / ``+-inf`` for one-sided.
    seed : seed of ``numpy.random.default_rng``.
    level : confidence level of the Wilson interval on the probability of meeting the limits.
    percentiles : output percentiles to report (in percent).

    Returns
    -------
    ToleranceResult
    """
    x0 = np.atleast_1d(np.asarray(nominal, dtype=float))
    tol = np.broadcast_to(np.asarray(tolerances, dtype=float), x0.shape)
    if np.any(tol < 0):
        raise ValueError("tolerances must be non-negative")
    rng = np.random.default_rng(seed)
    X = _sample(x0, tol, dist, int(n), rng)
    return _evaluate(fn, X, limits, level, percentiles)


def _evaluate(fn, X: np.ndarray, limits, level: float, percentiles: Sequence[float]) -> ToleranceResult:
    Y = np.asarray(fn(X), dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    if Y.ndim != 2 or Y.shape[0] != X.shape[0]:
        raise ValueError(f"fn must return (n,) or (n, m); got {Y.shape} for n = {X.shape[0]}")
    res = ToleranceResult(mean=Y.mean(axis=0), std=Y.std(axis=0, ddof=1),
                          percentiles={float(p): np.percentile(Y, p, axis=0) for p in percentiles},
                          n=Y.shape[0], inputs=X, outputs=Y)
    if limits is not None:
        lo, hi = limits
        lo = np.broadcast_to(-np.inf if lo is None else np.asarray(lo, dtype=float), (Y.shape[1],))
        hi = np.broadcast_to(np.inf if hi is None else np.asarray(hi, dtype=float), (Y.shape[1],))
        ok = (Y >= lo) & (Y <= hi)
        res.p_within_each = ok.mean(axis=0)
        k = int(ok.all(axis=1).sum())
        res.p_within = k / Y.shape[0]
        res.p_within_ci = wilson_interval(k, Y.shape[0], level)
    return res


def propagate_space(fn: Callable[[np.ndarray], np.ndarray], space, n: int = 4096, method: str = "lhs",
                    limits=None, seed: int = 0, level: float = 0.95,
                    percentiles: Sequence[float] = (1, 5, 50, 95, 99)) -> ToleranceResult:
    """Propagate independent inputs with any distribution (see
    :mod:`pinneapple_analysis.uncertainty.distributions`) through ``fn``.

    ``method`` is ``"mc"`` (plain Monte Carlo), ``"lhs"`` (Latin hypercube: same estimator, lower variance of the
    mean for smooth models) or ``"sobol"`` (scrambled quasi-Monte Carlo). The probability of meeting ``limits`` and
    its Wilson interval assume independent draws, exact for ``"mc"`` and conservative for the other two.
    """
    from pinneapple_analysis.uncertainty.distributions import as_space, to_values, unit_sample

    sp = as_space(space)
    X = to_values(sp, unit_sample(len(sp), int(n), method, seed))
    return _evaluate(fn, X, limits, level, percentiles)

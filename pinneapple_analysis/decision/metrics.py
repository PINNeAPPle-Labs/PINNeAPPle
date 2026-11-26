"""Does the surrogate change the design you would pick? Error metrics that speak in decisions.

A field error of 5 % says nothing about whether a surrogate is worth paying for: a model that is wrong everywhere
but orders every design the same way gives the same design as the reference solver, and one that is right on
average but wrong on the two candidates that matter gives a different one. The two are indistinguishable from an
L2 number and obvious from a flip rate. This module holds the three decision-level metrics and the map of where
they break down, all measured against the reference solver on the same design set:

``flip_rate(truth, prediction, *, margin=0.0, lower_is_better=True)``
    over every pair of designs, the fraction whose ordering the surrogate reverses. A pair whose two reference
    values differ by no more than ``margin`` times the standard deviation of the reference vector is declared
    equivalent and leaves the calculation: choosing either design is not a mistake. The indices of the designs
    involved in a reversal are returned with the number, and the reversal of the chosen design itself (the argmin
    changing) is reported separately as ``choice_flip``, because a surrogate can reverse a pair far from the
    optimum and still pick the right design.
``rank_preservation(truth, prediction, top_k=None)``
    Spearman rho and Kendall tau-b over the design set, plus the fraction of the true top-k that the surrogate
    keeps in its own top-k (``k`` defaults to 10 % of the designs, at least one).
``regret(truth, prediction, lower_is_better=True)``
    how much worse the design the surrogate would have chosen is, in the objective's own units and as a fraction
    of the spread of the reference values.
``failure_map(coords, error, *, bins=20, quantile=0.9, n_regions=5)``
    the mean error and the count per cell of a 1-D, 2-D or 3-D design space, and the cells above a quantile of the
    occupied cells' mean error, ready to be drawn as a figure.

Everything here is arithmetic on the arrays given: no model is fitted, nothing is interpolated and no number is
estimated. The reference values must come from the reference solver on a test split disjoint from the fit split,
otherwise the card they produce is a description of the training set. See ``card`` for the one-page summary and
``pinneapple_analysis.validation.error_map`` for the field-level map these numbers exist to de-emphasise.

Usage
-----
>>> from pinneapple_analysis.decision import decision_card
>>> truth = [1.0, 2.0, 3.0, 4.0]
>>> prediction = [1.1, 2.0, 3.0, 4.2]
>>> card = decision_card(truth, prediction, names=["bend_r40", "bend_r50", "bend_r60", "bend_r80"])
>>> card.verdict
'decision-safe'
>>> print(card.summary())
DECISION-SAFE: flip rate 0.0 %, Spearman 1.000, Kendall 1.000, top-1 kept 100 %, regret 0.0 % of the spread -- 4 designs, lower is better, margin 0 sd
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "FlipRate",
    "RankPreservation",
    "Regret",
    "FailureMap",
    "flip_rate",
    "rank_preservation",
    "regret",
    "failure_map",
]


# --------------------------------------------------------------------------- helpers
def _values(truth, prediction, *, minimum: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """The two vectors as 1-D float arrays of the same length, checked for finite entries."""
    t = np.asarray(truth, float).reshape(-1)
    p = np.asarray(prediction, float).reshape(-1)
    if t.shape != p.shape:
        raise ValueError(f"reference {t.shape} and prediction {p.shape} differ: they must be one value per design")
    if t.size < minimum:
        raise ValueError(f"at least {minimum} designs are needed, got {t.size}")
    if not np.all(np.isfinite(t)):
        raise ValueError("the reference vector has non-finite entries: report those designs as not run")
    if not np.all(np.isfinite(p)):
        raise ValueError("the prediction vector has non-finite entries: report those designs as not run")
    return t, p


def _best_index(values: np.ndarray, lower_is_better: bool) -> int:
    return int(np.argmin(values)) if lower_is_better else int(np.argmax(values))


def _best_value(values: np.ndarray, lower_is_better: bool) -> float:
    return float(np.min(values)) if lower_is_better else float(np.max(values))


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Ordinal ranks with ties replaced by the mean of the ranks they occupy (the ranking Spearman is defined on)."""
    n = x.size
    order = np.argsort(x, kind="stable")
    ranks = np.empty(n, float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    xs = x[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and xs[j] == xs[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = 0.5 * (i + 1 + j)  # mean of the 1-based ranks i+1 ... j
        i = j
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho as the Pearson correlation of the average ranks; ``nan`` when a vector is constant."""
    rx, ry = _average_ranks(x), _average_ranks(y)
    sx, sy = rx - rx.mean(), ry - ry.mean()
    denom = float(np.sqrt(np.sum(sx * sx) * np.sum(sy * sy)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(sx * sy) / denom)


def _kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall tau-b, counted pair by pair (``n`` passes of vectorised comparisons: O(n^2), no scipy).

    tau-b is the form that corrects for ties in either vector, which a design sweep always has (duplicate
    geometries, a solver that returns the same pressure drop twice, a coarse objective).
    """
    n = x.size
    n0 = n * (n - 1) / 2
    concordant = discordant = tie_x = tie_y = 0
    for i in range(n - 1):
        dx = np.sign(x[i + 1:] - x[i])
        dy = np.sign(y[i + 1:] - y[i])
        tie_x += int(np.count_nonzero(dx == 0))
        tie_y += int(np.count_nonzero(dy == 0))
        concordant += int(np.count_nonzero((dx != 0) & (dx == dy)))
        discordant += int(np.count_nonzero((dx != 0) & (dx != dy)))
    denom = np.sqrt((n0 - tie_x) * (n0 - tie_y))
    if denom == 0.0:
        return float("nan")
    return float((concordant - discordant) / denom)


# --------------------------------------------------------------------------- flip rate
@dataclass(frozen=True)
class FlipRate:
    """Result of :func:`flip_rate`.

    ``flip_rate``
        fraction of the comparable design pairs whose ordering the surrogate reverses (0 to 1). Pairs the
        reference itself cannot separate within the tolerance are not in the denominator.
    ``inverted``
        indices of the designs that take part in at least one reversal, ascending: the designs whose ranking
        against another design the surrogate gets wrong.
    ``choice_flip``
        whether the design the surrogate would pick (argmin, or argmax when ``lower_is_better=False``) differs
        from the one the reference picks.
    ``n_pairs``, ``n_flipped_pairs``, ``n_equivalent_pairs``, ``n_prediction_ties``
        the counts behind the fraction: comparable pairs, reversed pairs, pairs declared equivalent inside the
        tolerance, and pairs the surrogate cannot order because it predicts the same value for both.
    """

    flip_rate: float
    inverted: tuple[int, ...]
    n_pairs: int
    n_flipped_pairs: int
    n_equivalent_pairs: int
    n_prediction_ties: int
    tolerance: float
    margin: float
    lower_is_better: bool
    choice_flip: bool
    best_truth: int
    best_prediction: int

    def __float__(self) -> float:
        return self.flip_rate

    def summary(self) -> str:
        head = (f"Flip rate {100 * self.flip_rate:.1f} %: {self.n_flipped_pairs} of {self.n_pairs} comparable "
                f"design pairs are reversed (tolerance {self.tolerance:.3g}, margin {self.margin:g} sd)")
        if self.n_equivalent_pairs:
            head += f"; {self.n_equivalent_pairs} pairs are equivalent inside the tolerance and are not counted"
        if self.n_prediction_ties:
            head += f"; {self.n_prediction_ties} comparable pairs are tied in the prediction"
        choice = ("the surrogate picks a different design" if self.choice_flip
                  else "the surrogate picks the same design")
        return f"{head}. Chosen design {self.best_prediction} against {self.best_truth}: {choice}."

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)


def flip_rate(truth, prediction, *, margin: float = 0.0, lower_is_better: bool = True) -> FlipRate:
    """Fraction of design pairs whose ordering the surrogate reverses, and which designs that costs you.

    The unit is the design pair, because that is the unit of a design decision: a study never asks "is this design
    good", it asks "is A better than B", and a surrogate that gets that question wrong sends the customer to the
    other geometry. With ``n`` designs there are ``n (n - 1) / 2`` pairs; a pair is reversed when the two vectors
    order it differently, and ``flip_rate`` is the number of reversed pairs over the comparable ones.

    Parameters
    ----------
    truth, prediction
        One value per design, from the reference solver and from the surrogate, on the same designs. Lower is
        better (minimisation) unless ``lower_is_better=False``; the pairwise test is direction-agnostic, the flag
        only decides which design counts as the chosen one in ``choice_flip``.
    margin
        The decision tolerance, as a fraction of the standard deviation of the reference vector (``ddof=1``).
        Pairs whose reference values differ by no more than ``margin`` times that deviation are declared
        equivalent: choosing either is not a mistake, so they are neither reversed nor comparable, and they are
        dropped from the denominator. ``margin=0`` (the default) keeps every pair the reference separates and
        drops only exact ties. The sample standard deviation is used so that the tolerance does not depend on
        how many designs were sampled; when ``margin`` is zero no standard deviation is needed at all.
    lower_is_better
        Only affects ``choice_flip`` and ``best_truth``/``best_prediction``.

    Notes
    -----
    A pair the surrogate ties exactly (same predicted value for both designs) is not a reversal: the surrogate
    has failed to separate them but has not been shown to order them the wrong way. Those pairs are counted in
    ``n_prediction_ties`` so that a degenerate surrogate cannot hide behind the denominator.

    The tolerance belongs to the customer: sweep it and report the flip rate as a curve, since a single number
    quoted without the tolerance that produced it is a sales claim, not a measurement.

    Examples
    --------
    >>> fr = flip_rate([1.0, 2.0, 3.0], [1.0, 3.0, 2.0])
    >>> fr.flip_rate, fr.inverted
    (0.3333333333333333, (1, 2))
    """
    t, p = _values(truth, prediction)
    n = t.size
    tol = float(abs(margin)) * float(np.std(t, ddof=1)) if n > 1 else 0.0
    if not np.isfinite(tol):
        tol = 0.0

    i, j = np.triu_indices(n, k=1)
    d_truth = t[i] - t[j]
    d_pred = p[i] - p[j]
    equivalent = np.abs(d_truth) <= tol
    comparable = ~equivalent
    reversed_pairs = comparable & (np.sign(d_truth) * np.sign(d_pred) < 0)

    n_pairs = int(np.count_nonzero(comparable))
    n_flipped = int(np.count_nonzero(reversed_pairs))
    involved = np.unique(np.concatenate([i[reversed_pairs], j[reversed_pairs]])) if n_flipped else np.empty(0, int)

    best_truth = _best_index(t, lower_is_better)
    best_pred = _best_index(p, lower_is_better)
    return FlipRate(
        flip_rate=(n_flipped / n_pairs) if n_pairs else 0.0,
        inverted=tuple(int(k) for k in involved),
        n_pairs=n_pairs,
        n_flipped_pairs=n_flipped,
        n_equivalent_pairs=int(np.count_nonzero(equivalent)),
        n_prediction_ties=int(np.count_nonzero(comparable & (np.sign(d_pred) == 0))),
        tolerance=tol,
        margin=float(margin),
        lower_is_better=bool(lower_is_better),
        choice_flip=bool(best_truth != best_pred),
        best_truth=best_truth,
        best_prediction=best_pred,
    )


# --------------------------------------------------------------------------- ranking
@dataclass(frozen=True)
class RankPreservation:
    """Result of :func:`rank_preservation`.

    ``spearman``, ``kendall_tau``
        rank correlations over the whole design set, in [-1, 1]; 1 means the surrogate orders the designs exactly
        as the reference does, -1 that it orders them backwards. ``nan`` when one of the two vectors is constant,
        in which case there is no ranking to correlate.
    ``top_k``, ``top_k_kept``
        the number of designs kept in the top-k and the fraction of the reference's top-k that the surrogate also
        puts in its own top-k. This is the number a reader cares about: a model can have rho = 0.9 and still miss
        the optimum.
    ``top_k_truth``, ``top_k_prediction``
        the indices of the two top-k sets, ascending.
    """

    spearman: float
    kendall_tau: float
    top_k: int
    top_k_kept: float
    n: int
    top_k_truth: tuple[int, ...]
    top_k_prediction: tuple[int, ...]
    lower_is_better: bool

    def summary(self) -> str:
        return (f"Spearman {self.spearman:.3f}, Kendall tau-b {self.kendall_tau:.3f}; the surrogate keeps "
                f"{100 * self.top_k_kept:.0f} % of the reference top-{self.top_k} "
                f"({len(set(self.top_k_truth) & set(self.top_k_prediction))} of {self.top_k}) "
                f"over {self.n} designs")

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)


def rank_preservation(truth, prediction, top_k: int | None = None,
                      lower_is_better: bool = True) -> RankPreservation:
    """Rank correlations over the design set plus the fraction of the true top-k the surrogate keeps.

    A good rank correlation and a missed optimum are different failures and mean different things to a customer:
    rho is how usable the surrogate is for ranking a sweep, the top-k fraction is whether the design it sends to
    the shop is among the best ones.

    Parameters
    ----------
    truth, prediction
        One value per design, from the reference and from the surrogate, on the same designs.
    top_k
        Size of the top-k set; defaults to 10 % of the designs, at least one (``max(1, n // 10)``). For a
        minimisation the top-k are the ``k`` smallest reference values and the ``k`` smallest predictions; for
        ``lower_is_better=False`` the ``k`` largest.
    lower_is_better
        Direction of "better", used only to select the top-k sets.

    Notes
    -----
    Both correlations are implemented here in numpy (average ranks for Spearman, pair counting of tau-b for
    Kendall), so a metric bug cannot hide in a library call, and exact ties are handled the way the statistics
    literature defines them: ties get the mean rank and tau-b corrects its denominator for them.
    """
    t, p = _values(truth, prediction)
    n = t.size
    k = max(1, n // 10) if top_k is None else int(top_k)
    if not 1 <= k <= n:
        raise ValueError(f"top_k must be between 1 and {n} designs, got {k}")

    order_t = np.argsort(t, kind="stable")
    order_p = np.argsort(p, kind="stable")
    top_t = order_t[:k] if lower_is_better else order_t[::-1][:k]
    top_p = order_p[:k] if lower_is_better else order_p[::-1][:k]
    kept = len(set(int(v) for v in top_t) & set(int(v) for v in top_p))
    return RankPreservation(
        spearman=_spearman(t, p),
        kendall_tau=_kendall_tau_b(t, p),
        top_k=k,
        top_k_kept=kept / k,
        n=n,
        top_k_truth=tuple(sorted(int(v) for v in top_t)),
        top_k_prediction=tuple(sorted(int(v) for v in top_p)),
        lower_is_better=bool(lower_is_better),
    )


# --------------------------------------------------------------------------- regret
@dataclass(frozen=True)
class Regret:
    """Result of :func:`regret`.

    ``absolute``
        how much worse the design the surrogate picks is than the best design in the reference, in the
        objective's own units (non-negative by construction: a surrogate cannot beat the reference minimum).
    ``relative``
        ``absolute`` divided by the spread of the reference values over the design set, so 0 is the optimum and
        1 means the surrogate picked the worst design of the sweep. ``nan`` when the reference values are all
        equal (no spread to normalise by) unless ``absolute`` is also zero.
    ``chosen``, ``best_truth``
        indices of the design the surrogate picks and of the true best design.
    """

    absolute: float
    relative: float
    chosen: int
    best_truth: int
    truth_at_chosen: float
    best_value: float
    span: float

    def summary(self) -> str:
        return (f"Regret {self.absolute:.4g} ({100 * self.relative:.1f} % of the reference spread of "
                f"{self.span:.4g}): the surrogate picks design {self.chosen} at {self.truth_at_chosen:.4g} "
                f"against design {self.best_truth} at {self.best_value:.4g}")

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)


def regret(truth, prediction, lower_is_better: bool = True) -> Regret:
    """How much of the achievable improvement the surrogate's chosen design gives up.

    The chosen design is the one the surrogate ranks best; the regret is the reference value of that design minus
    the best reference value, so it is what you pay, in objective units, for trusting the model. Reported as a
    fraction of the spread of the reference values, which makes it comparable across objectives and is the number
    a business case can be built on. A surrogate that always returns the mean of the training set has a regret
    near the full spread, which is what calibrates the scale of the metric.
    """
    t, p = _values(truth, prediction)
    best_truth = _best_index(t, lower_is_better)
    chosen = _best_index(p, lower_is_better)
    best = _best_value(t, lower_is_better)
    span = float(np.max(t) - np.min(t))
    if lower_is_better:
        absolute = float(t[chosen] - best)
    else:
        absolute = float(best - t[chosen])
    absolute = max(absolute, 0.0)
    if span > 0.0:
        relative = absolute / span
    else:
        relative = 0.0 if absolute == 0.0 else float("nan")
    return Regret(
        absolute=absolute,
        relative=relative,
        chosen=chosen,
        best_truth=best_truth,
        truth_at_chosen=float(t[chosen]),
        best_value=best,
        span=span,
    )


# --------------------------------------------------------------------------- failure map
@dataclass(frozen=True)
class FailureMap:
    """Result of :func:`failure_map`: the error binned over the design space.

    ``mean``, ``count``
        arrays of shape ``(bins,) * dim``: mean error per cell (``nan`` where the cell holds no design) and the
        number of designs in it. Plot ``mean`` for the picture, ``count`` to see which cells are barely sampled
        and therefore not evidence of anything.
    ``top_regions``
        the worst cells, each a dict with ``index``, ``low``, ``high``, ``centre``, ``count``, ``mean_error`` and
        ``max_error``, ordered by decreasing mean error and limited to the cells above ``quantile`` of the occupied
        cells' mean error.
    ``edges``
        the bin edges per dimension, for drawing the figure with the same axes.
    """

    dim: int
    bins: int
    edges: tuple[np.ndarray, ...]
    mean: np.ndarray
    count: np.ndarray
    top_regions: tuple[dict[str, Any], ...]
    quantile: float
    n: int
    n_dropped: int
    error_scale: float

    def occupancy(self) -> float:
        """Fraction of the grid cells that hold at least one design."""
        return float(np.count_nonzero(self.count) / self.count.size)

    def worst(self) -> dict[str, Any] | None:
        """The single worst cell above the quantile, or ``None`` when the map is empty."""
        return dict(self.top_regions[0]) if self.top_regions else None

    def summary(self) -> str:
        occ = self.occupancy()
        head = (f"Failure map over {self.n} designs in {self.dim}-D on {self.bins} bins per axis: "
                f"{100 * occ:.0f} % of the cells are occupied, mean error per occupied cell up to "
                f"{float(np.nanmax(self.mean)):.4g} against a global {self.error_scale:.4g}")
        if not self.top_regions:
            return head
        parts = []
        for region in self.top_regions[:3]:
            centre = ", ".join(f"{c:.4g}" for c in region["centre"])
            parts.append(f"({centre}) mean {region['mean_error']:.4g} over {region['count']} designs")
        return f"{head}. Worst regions: " + "; ".join(parts) + "."

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)


def failure_map(coords, error, *, bins: int = 20, quantile: float = 0.9,
                n_regions: int = 5) -> FailureMap:
    """Grid the surrogate's decision error over the design space, so the customer sees where the boundary is.

    The average error hides the boundary of the trustworthy region; a binned map shows it. Every design gets the
    per-design error the caller computed against the reference (``|prediction - reference|``, or the norm of the
    difference for a vector objective, as ``pinneapple_analysis.validation.error_map.pointwise_error`` returns
    it), and the map is the mean and the count of that error per cell.

    Parameters
    ----------
    coords
        Design coordinates, ``(n, d)`` with ``d`` in {1, 2, 3}; a flat ``(n,)`` array is read as ``d = 1``.
    error
        One non-negative error per design, or an array whose last-but-one axis aligns with ``coords`` when the
        objective is a vector: it is reduced to a single number per design with ``np.max`` only when its shape is
        not ``(n,)``, otherwise it is used as given.
    bins
        Number of cells per axis, the same in every axis.
    quantile
        Cells whose mean error is at or above this quantile of the occupied cells' mean error are candidates for
        ``top_regions``; ``0.9`` marks the worst tenth of the sampled space.
    n_regions
        How many of those candidates to return, the worst first.

    Notes
    -----
    Empty cells are ``nan`` in ``mean`` and zero in ``count``: a cell with no design is not evidence of a small
    error, and drawing it as zero is the standard way a failure map lies. Designs with a non-finite coordinate or
    error are dropped and counted in ``n_dropped`` rather than binned somewhere arbitrary. A dimension in which
    every design has the same coordinate is given a width of one unit centred on that value, so that a constant
    parameter does not collapse the grid.
    """
    C = np.asarray(coords, float)
    if C.ndim == 1:
        C = C[:, None]
    if C.ndim != 2:
        raise ValueError(f"coords must be (n, d) with d in 1, 2, 3; got shape {C.shape}")
    n, d = C.shape
    if d not in (1, 2, 3):
        raise ValueError(f"coords must have 1, 2 or 3 columns; got {d}")
    if bins < 1:
        raise ValueError(f"bins must be at least 1, got {bins}")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")

    e = np.asarray(error, float)
    if e.shape[:1] != (n,):
        raise ValueError(f"error must have one value per design ({n}); got shape {e.shape}")
    if e.ndim > 1:
        e = e.reshape(n, -1).max(axis=1)
    if not np.allclose(e, np.abs(e)):
        raise ValueError("error must be non-negative (an absolute difference or a norm)")

    ok = np.all(np.isfinite(C), axis=1) & np.isfinite(e)
    n_dropped = int(n - np.count_nonzero(ok))
    C, e = C[ok], e[ok]
    shape = (bins,) * d

    if C.shape[0] == 0:
        empty = np.full(shape, np.nan)
        return FailureMap(dim=d, bins=bins, edges=tuple(np.zeros(bins + 1) for _ in range(d)),
                          mean=empty, count=np.zeros(shape, int), top_regions=(), quantile=quantile,
                          n=0, n_dropped=n_dropped, error_scale=float("nan"))

    edges: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for axis in range(d):
        lo, hi = float(np.min(C[:, axis])), float(np.max(C[:, axis]))
        if hi <= lo:
            lo, hi = lo - 0.5, hi + 0.5
        ed = np.linspace(lo, hi, bins + 1)
        edges.append(ed)
        idx = np.clip(np.searchsorted(ed, C[:, axis], side="right") - 1, 0, bins - 1)
        indices.append(idx)

    flat = np.ravel_multi_index(tuple(indices), shape)
    count = np.bincount(flat, minlength=int(np.prod(shape))).reshape(shape)
    totals = np.bincount(flat, weights=e, minlength=int(np.prod(shape))).reshape(shape)
    maxima = np.full(int(np.prod(shape)), -np.inf)
    np.maximum.at(maxima, flat, e)
    maxima = maxima.reshape(shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, totals / np.maximum(count, 1), np.nan)

    occupied = mean[count > 0]
    top: list[dict[str, Any]] = []
    if occupied.size:
        threshold = float(np.quantile(occupied, quantile))
        candidates = np.argwhere((count > 0) & (mean >= threshold))
        candidates = sorted(candidates.tolist(), key=lambda ix: -float(mean[tuple(ix)]))[:max(1, n_regions)]
        for cell in candidates:
            ix = tuple(int(v) for v in cell)
            low = tuple(float(edges[a][ix[a]]) for a in range(d))
            high = tuple(float(edges[a][ix[a] + 1]) for a in range(d))
            top.append({
                "index": ix,
                "low": low,
                "high": high,
                "centre": tuple(0.5 * (lo + hi) for lo, hi in zip(low, high, strict=True)),
                "count": int(count[ix]),
                "mean_error": float(mean[ix]),
                "max_error": float(maxima[ix]),
            })

    return FailureMap(dim=d, bins=bins, edges=tuple(edges), mean=mean, count=count,
                      top_regions=tuple(top), quantile=quantile, n=int(C.shape[0]),
                      n_dropped=n_dropped, error_scale=float(np.mean(e)))


def _plain(obj: Any) -> Any:
    """Recursively convert dataclasses and numpy values to plain Python for JSON."""
    import dataclasses

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    return obj

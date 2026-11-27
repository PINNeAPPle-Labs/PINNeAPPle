"""Where a model holds, and how badly it fails outside that: the validity domain and the measured extrapolation error
(#58, roadmap D7).

A benchmark row that says "rel. L2 = 0.02" without saying *where* it holds invites the reader to assume everywhere.
This module makes the bound part of the case:

- :class:`ValidityDomain` — the envelope a case declares, as ranges per variable, with **where the ranges came from**
  (the generator's own bounds, or the measured range of the training split). It answers ``contains(point)`` and
  ``distance(point)``, the last one a normalized distance outside the box: 0 inside, 1 one full range beyond the
  bound, and so on. Normalizing is what makes "how far outside" comparable across variables and across cases.
- :func:`study` — the measurement. It walks one axis outward from the edge of the envelope, calls the case's own
  model and its reference at every step, and records the relative error (and, when a trust gate is passed, its OOD
  score). Nothing is extrapolated by formula: every number in the table is an error that was computed.
- :func:`valid_until` — the point where the measured error first crosses the case's tolerance, refined between the two
  measured steps around the crossing (linear in ``distance`` against ``log error``). ``None`` means "still inside the
  tolerance at the farthest distance measured", which is a statement about the sampled range, not a promise.

The benchmark report prints, for every case, the envelope (``describe()``) and an **extrapolation column**: the error
outside the envelope over the error inside it, from the OOD and in-domain sets the problem already measures. A factor
near 1 says the method carries its accuracy across the boundary; a factor of 3 says it does not, and the column says
so instead of leaving it to the reader.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["ExtrapolationStudy", "ValidityDomain", "extrapolation_factor", "study", "valid_until"]


@dataclass(frozen=True)
class ValidityDomain:
    """The envelope of a case: one range per variable, and the reason those are the bounds.

    ``source`` is not decoration: a domain measured on the training data is a different claim from one declared by
    the generator, and a reader deciding whether to trust an extrapolation needs to know which it was.
    """
    name: str
    bounds: Mapping[str, tuple[float, float]]
    source: str                       # "declared by <the generator>" | "measured on the training split of <prepared>"
    note: str = ""

    def __post_init__(self):
        if not isinstance(self.bounds, Mapping):
            raise ValueError(f"validity domain of {self.name}: bounds must map variable -> (low, high), "
                             f"got {type(self.bounds).__name__}")
        for var, (lo, hi) in self.bounds.items():
            if not float(lo) < float(hi):
                raise ValueError(f"validity domain of {self.name}: {var} has an empty range ({lo}, {hi})")

    @property
    def variables(self) -> list[str]:
        return list(self.bounds)

    def contains(self, point: Mapping[str, float], *, tol: float = 1e-12) -> bool:
        return all(lo - tol <= float(point[v]) <= hi + tol for v, (lo, hi) in self.bounds.items() if v in point)

    def distance(self, point: Mapping[str, float]) -> float:
        """How far outside the envelope a point sits, in units of the variable's own range (0 = inside)."""
        worst = 0.0
        for var, (lo, hi) in self.bounds.items():
            if var not in point:
                continue
            v = float(point[var])
            width = float(hi) - float(lo)
            if v < lo:
                worst = max(worst, (lo - v) / width)
            elif v > hi:
                worst = max(worst, (v - hi) / width)
        return worst

    def describe(self) -> str:
        parts = [f"{v} in [{lo:g}, {hi:g}]" for v, (lo, hi) in self.bounds.items()]
        text = f"{self.name}: " + "; ".join(parts) + f" ({self.source})"
        return f"{text} -- {self.note}" if self.note else text

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "bounds": {k: list(v) for k, v in self.bounds.items()},
                "source": self.source, "note": self.note}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ValidityDomain:
        return cls(d["name"], {k: tuple(v) for k, v in d["bounds"].items()}, d["source"], d.get("note", ""))

    @classmethod
    def from_training_data(cls, name: str, X: np.ndarray, variables: Sequence[str], *, where: str) -> ValidityDomain:
        """The envelope measured on the training inputs: the coordinate-wise min and max of what the model was fitted
        on. Use it when the case's data, not its generator, defines the range."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(variables):
            raise ValueError(f"{where}: training inputs have shape {X.shape}, expected (n, {len(variables)})")
        bounds = {v: (float(X[:, i].min()), float(X[:, i].max())) for i, v in enumerate(variables)}
        return cls(name, bounds, f"measured on the training split of {where}")


def valid_until(distances: Iterable[float], errors: Iterable[float], tolerance: float) -> float | None:
    """The largest distance whose measured error is still within ``tolerance``, with the crossing refined.

    ``None`` means the error was still inside the tolerance at the farthest distance measured: a statement about the
    sampled range, not a promise about further out.
    """
    d = np.asarray(list(distances), dtype=float)
    e = np.asarray(list(errors), dtype=float)
    if not len(d):
        return None
    ok = e <= tolerance
    if ok.all():
        return float(d.max())
    first_bad = int(np.argmax(~ok))
    if first_bad == 0:
        return 0.0 if d[0] > 0 else float(d[0])
    lo, hi = d[first_bad - 1], d[first_bad]
    e_lo, e_hi = max(e[first_bad - 1], 1e-300), max(e[first_bad], 1e-300)
    if e_hi <= e_lo:
        return float(lo)
    # linear in distance against log error: errors grow by orders of magnitude, distances by ones
    t = (math.log(tolerance) - math.log(e_lo)) / (math.log(e_hi) - math.log(e_lo))
    return float(lo + max(0.0, min(1.0, t)) * (hi - lo))


@dataclass
class ExtrapolationStudy:
    """A measured walk out of the envelope along one axis: error (and OOD score) against normalized distance."""
    case: str
    domain: ValidityDomain
    axis: str
    tolerance: float
    distances: list[float] = field(default_factory=list)
    errors: list[float] = field(default_factory=list)
    ood_scores: list[float | None] = field(default_factory=list)
    error_inside: float | None = None
    valid_until: float | None = None

    def rows(self) -> list[dict[str, Any]]:
        return [{"distance": d, "rel_error": e, "ood_score": s}
                for d, e, s in zip(self.distances, self.errors, self.ood_scores, strict=True)]

    @property
    def factor(self) -> float | None:
        """Farthest error over the in-envelope error: how much of its accuracy the model keeps at the far end."""
        if self.error_inside is None or not self.errors or self.error_inside <= 0:
            return None
        return float(max(self.errors) / self.error_inside)

    def summary(self) -> str:
        inside = "-" if self.error_inside is None else f"{self.error_inside:.3g}"
        until = "not within the sampled range" if self.valid_until is None else f"{self.valid_until:.2f} ranges out"
        return (f"{self.case} along {self.axis}: rel. error {inside} inside the envelope; stays within "
                f"{self.tolerance:g} until {until}; x{(self.factor or float('nan')):.2f} at the farthest step")

    def to_dict(self) -> dict[str, Any]:
        return {"case": self.case, "domain": self.domain.to_dict(), "axis": self.axis,
                "tolerance": self.tolerance, "error_inside": self.error_inside, "valid_until": self.valid_until,
                "factor": self.factor, "rows": self.rows()}


def _points_for(domain: ValidityDomain, axis: str, distance: float, n: int, rng: np.random.Generator) -> dict:
    """``n`` points sitting ``distance`` envelope-widths beyond the upper bound of ``axis``, others uniform inside."""
    lo, hi = domain.bounds[axis]
    point = {v: rng.uniform(a, b, n) for v, (a, b) in domain.bounds.items()}
    point[axis] = np.full(n, hi + distance * (hi - lo))
    return point


def _rel_error(model: Callable[[dict], Any], reference: Callable[[dict], Any], points: dict) -> float:
    got = np.asarray(model(points), dtype=float).ravel()
    want = np.asarray(reference(points), dtype=float).ravel()
    denom = float(np.linalg.norm(want))
    return float(np.linalg.norm(got - want) / denom) if denom > 0 else float("nan")


def study(case: str, domain: ValidityDomain, model: Callable[[dict], Any], reference: Callable[[dict], Any], *,
          axis: str | None = None, distances: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 2.0), n: int = 64,
          tolerance: float = 0.05, seed: int = 0,
          ood_score: Callable[[dict], Any] | None = None) -> ExtrapolationStudy:
    """Measure the error of ``model`` against ``reference`` at increasing distance outside ``domain``.

    ``model`` and ``reference`` take a mapping of variable name to an array of ``n`` points and return values; the
    distance 0 step is the in-envelope control, so every later number is comparable with a measured baseline rather
    than with an assumption. ``ood_score`` (a ``TrustGate.ood_p_values``-shaped callable) is recorded per step when
    given: it is the score a gate would have produced, next to the error that actually happened.
    """
    axis = axis or domain.variables[0]
    if axis not in domain.bounds:
        raise ValueError(f"{case}: {axis!r} is not a variable of the domain {domain.variables}")
    rng = np.random.default_rng(seed)
    out = ExtrapolationStudy(case=case, domain=domain, axis=axis, tolerance=float(tolerance))
    for d in distances:
        points = _points_for(domain, axis, float(d), n, rng)
        out.distances.append(float(d))
        out.errors.append(_rel_error(model, reference, points))
        if ood_score is not None:
            scores = np.asarray(ood_score(points), dtype=float).ravel()
            out.ood_scores.append(float(np.mean(scores)))
        else:
            out.ood_scores.append(None)
    out.error_inside = out.errors[0] if out.errors else None
    out.valid_until = valid_until(out.distances, out.errors, tolerance)
    return out


def extrapolation_factor(inside: Sequence[float], outside: Sequence[float]) -> float | None:
    """Error outside the envelope over error inside it, from the two sets a benchmark case already measures.

    ``None`` when there is nothing to divide (an empty set, or an in-domain error of zero): a made-up factor would be
    worse than the absence of one.
    """
    a = np.asarray([v for v in inside if v is not None and np.isfinite(v)], dtype=float)
    b = np.asarray([v for v in outside if v is not None and np.isfinite(v)], dtype=float)
    if not len(a) or not len(b):
        return None
    base = float(a.mean())
    return float(b.mean() / base) if base > 0 else None

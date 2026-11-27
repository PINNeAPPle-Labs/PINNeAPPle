"""The load on a mooring line as the vessel moves: the quasi-static catenary, and what a lost line does.

**Quasi-static** is the honest word. The line is a chain of constant submerged weight in a plane, hanging between an
anchor on the seabed and the fairlead, and the tension is whatever that shape requires at the given geometry. Nothing
here integrates the line's own dynamics: a real line has added mass along its length, a drag that grows with the square
of the fairlead velocity, and a snap load when the vessel surges away faster than the line can pay out. That is a
coupled dynamic analysis (OrcaFlex and its kind), and a quasi-static solution *under*-predicts the peak. It is used here
because it is exact for the geometry it describes, and because the shape is the point: the tension grows nonlinearly
with the offset, and the nonlinearity is what an operability decision is about.

The formulation, for a line of submerged weight ``w`` [N/m] and length ``L`` [m] with the fairlead at the height ``h``
above the anchor and the anchor at the horizontal distance ``X``:

* the catenary ``y = a cosh(x/a)`` has horizontal tension ``H = w a`` everywhere, and the tension at a point is
  ``w`` times its height above the lowest point;
* **fully suspended** (the whole line off the bottom): with ``delta = atanh(h/L)`` the parameter is ``a = X/(2 delta)``,
  and the fairlead tension is ``w a cosh(sigma + delta)`` with ``sigma = asinh(sqrt(L^2 - h^2)/(2 a))``. The line
  touches the seabed exactly when ``X`` reaches ``a_t acosh(h/a_t + 1)`` with ``a_t = (L^2 - h^2)/(2h)``;
* **with bottom contact** (a shorter span): the part that hangs does so from a touchdown point where the tangent is
  horizontal, so ``a = (L_s^2 - h^2)/(2h)`` and the fairlead tension is simply ``w (h + a)``; the rest lies on the
  seabed and carries no tension of its own beyond the friction the catenary does not model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import spectrum as sp

__all__ = ["Line", "Mooring", "catenary", "fairlead_tension", "CHAIN_76", "CHAIN_56"]

G = sp.G

#: A 76 mm R4 studlink chain in sea water: submerged weight and minimum breaking load, per the catalogue values a
#: mooring design starts from (the weight is the dry weight less the buoyancy of the steel it is made of).
CHAIN_76 = {"weight": 1106.0, "breaking": 5.83e6}
CHAIN_56 = {"weight": 600.0, "breaking": 3.16e6}


@dataclass(frozen=True)
class Line:
    """One mooring line: its length, its submerged weight and the load it is allowed to see."""

    length: float                  # m, from the anchor to the fairlead
    weight: float | None = None    # N/m, submerged; None is a 76 mm chain
    breaking: float | None = None  # N, minimum breaking load; None is a 76 mm chain
    name: str = "line"

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError("a line needs a positive length")
        if self.weight is not None and self.weight <= 0:
            raise ValueError("the weight has to be positive")
        if self.breaking is not None and self.breaking <= 0:
            raise ValueError("the breaking load has to be positive")

    @property
    def properties(self) -> tuple[float, float]:
        return (CHAIN_76["weight"] if self.weight is None else float(self.weight),
                CHAIN_76["breaking"] if self.breaking is None else float(self.breaking))


def catenary(length: float, depth: float, span: float, weight: float) -> dict[str, float]:
    """The shape of one line between the anchor and the fairlead, and the tension at its two ends.

    Parameters
    ----------
    length : float [m]
        Length from the anchor to the fairlead.
    depth : float [m]
        Height of the fairlead above the seabed: the vertical span of the line.
    span : float [m]
        Horizontal distance from the anchor to the fairlead.
    weight : float [N/m]
        Submerged weight per unit length.

    Returns
    -------
    dict
        ``a`` (``H/w``), ``H`` (horizontal tension, the same everywhere), ``T_anchor`` and ``T_fairlead`` [N],
        ``bottom`` (the length lying on the seabed), ``suspended``, and ``reach`` (the longest span at which the line
        still just touches the bottom -- a longer one lifts the whole line off it).
    """
    if not length > depth:
        raise ValueError(f"a line of {length:.1f} m cannot reach a fairlead {depth:.1f} m above the anchor")
    if span < 0:
        raise ValueError("the span cannot be negative")
    a_touch = (length ** 2 - depth ** 2) / (2.0 * depth)
    reach = a_touch * math.acosh(depth / a_touch + 1.0)
    if span >= reach:
        delta = math.atanh(depth / length)
        a = span / (2.0 * delta) if span > 0 else 0.0
        sigma = math.asinh(math.sqrt(length ** 2 - depth ** 2) / (2.0 * a)) if a > 0 else 0.0
        bottom = 0.0
        suspended = length
        tension_fairlead = weight * a * math.cosh(sigma + delta) if a > 0 else weight * depth
        tension_anchor = weight * a * math.cosh(sigma - delta) if a > 0 else weight * depth
    else:
        bottom, suspended, a = _bottom_contact(length, depth, span)
        tension_fairlead = weight * (depth + a)
        tension_anchor = weight * a
    return {"a": a, "H": weight * a, "T_anchor": tension_anchor, "T_fairlead": tension_fairlead,
            "bottom": bottom, "suspended": suspended, "reach": reach, "weight": weight, "span": span}


def _bottom_contact(length: float, depth: float, span: float) -> tuple[float, float, float]:
    """``(bottom length, suspended length, a)`` for a line whose part lies on the seabed."""
    from scipy.optimize import brentq

    if span <= length - depth:
        # slack enough to hang vertically from the fairlead, the rest on the bottom
        return length - depth, depth, 0.0

    def span_for(suspended: float) -> float:
        a = (suspended ** 2 - depth ** 2) / (2.0 * depth)
        return 0.0 if a <= 0 else a * math.acosh(depth / a + 1.0)

    def residual(suspended: float) -> float:
        return span_for(suspended) + (length - suspended) - span

    suspended = brentq(residual, depth, length, xtol=1e-10)
    return length - suspended, suspended, (suspended ** 2 - depth ** 2) / (2.0 * depth)


def fairlead_tension(length: float, depth: float, span, weight: float) -> np.ndarray:
    """The fairlead tension for a line, over an array of horizontal spans."""
    return np.array([catenary(length, depth, float(max(s, 0.0)), weight)["T_fairlead"]
                     for s in np.atleast_1d(np.asarray(span, dtype=float))])


@dataclass(frozen=True)
class Mooring:
    """A spread of identical lines holding the vessel, and how the tension changes as it moves.

    ``direction`` is, per line, the projection of the line's horizontal direction (anchor to vessel) on the excursion
    axis: ``+1`` for a line the vessel moves away from as it moves along that axis, ``-1`` for one it moves towards.
    """

    lines: tuple[Line, ...]
    depth: float
    span: float
    direction: np.ndarray

    def __post_init__(self) -> None:
        if not self.lines:
            raise ValueError("a mooring needs at least one line")
        if self.depth <= 0 or self.span < 0:
            raise ValueError("depth has to be positive and the span non-negative")
        if np.asarray(self.direction, dtype=float).size != len(self.lines):
            raise ValueError("one direction per line")

    def tension(self, surge) -> dict[str, np.ndarray]:
        """Every line's fairlead tension for a set of excursions ``surge`` [m] along the spread's axis."""
        surge = np.atleast_1d(np.asarray(surge, dtype=float))
        direction = np.asarray(self.direction, dtype=float)
        spans = self.span + np.outer(direction, surge)
        tensions = np.zeros((len(self.lines), surge.size))
        for i, line in enumerate(self.lines):
            weight, _ = line.properties
            tensions[i] = fairlead_tension(line.length, self.depth, spans[i], weight)
        return {"surge": surge, "span": spans, "tension": tensions, "worst": tensions.max(axis=0),
                "total_along_axis": -(tensions * direction[:, None]).sum(axis=0)}

    def equilibrium(self, steady_load: float, *, low: float = -200.0, high: float = 200.0) -> dict[str, float]:
        """The excursion at which the spread's horizontal pull balances a steady environmental load.

        This is what a lost line is about: the load does not change, so the vessel moves until the lines that are left
        carry it, and at a larger offset each of them is stiffer. ``steady_load`` [N] pulls the vessel along the
        excursion axis, away from the anchors it is directed at, so the equilibrium it produces is a positive surge.
        """
        from scipy.optimize import brentq

        def out_of_balance(surge: float) -> float:
            # at equilibrium the spread's pull cancels the load: the pull is reported as a force on the vessel, so the
            # sign that balances a positive load is the negative one
            return float(self.tension([surge])["total_along_axis"][0]) + steady_load

        try:
            surge = brentq(out_of_balance, low, high, xtol=1e-6)
        except ValueError as error:
            raise ValueError(f"the spread cannot hold {steady_load / 1e3:.0f} kN within +-{high:.0f} m") from error
        loaded = self.utilisation([surge])
        return {"surge": float(surge), "tension": float(loaded["worst"][0]),
                "utilisation": float(loaded["worst_utilisation"][0])}

    def utilisation(self, surge) -> dict[str, np.ndarray]:
        """Tension as a fraction of each line's breaking load, and the worst of them."""
        out = self.tension(surge)
        breaking = np.array([line.properties[1] for line in self.lines])[:, None]
        ratio = out["tension"] / breaking
        return {**out, "utilisation": ratio, "worst_utilisation": ratio.max(axis=0)}


def spread(n: int = 8, *, length: float = 437.0, depth: float = 120.0, span: float = 400.0,
           chain: dict | None = None, direction=None) -> Mooring:
    """A symmetric spread of ``n`` identical chain lines -- the geometry the example uses."""
    properties = dict(CHAIN_76 if chain is None else chain)
    lines = tuple(Line(length=length, name=f"line {i + 1}", **properties) for i in range(n))
    if direction is None:
        direction = np.cos(np.linspace(0.0, math.pi, n))
    return Mooring(lines=lines, depth=depth, span=span, direction=np.asarray(direction, dtype=float))

r"""Dimensional analysis, Buckingham pi, scaling laws and curve collapse.

**Dimensional analysis** is exact algebra on the dimensions of the quantities that describe a problem. With ``n``
quantities over ``k`` base dimensions, the dimensional matrix :math:`D` (``k x n``) holds the exponent of each base
dimension in each quantity. A product :math:`\Pi = \prod_i q_i^{\alpha_i}` is dimensionless exactly when
:math:`D\alpha = 0`, so the dimensionless groups are the null space of :math:`D`: ``n - rank(D)`` independent ones
(Buckingham 1914, Phys. Rev. 4:345). Everything here is computed in exact rational arithmetic (``fractions.Fraction``),
so a group's exponents are exact, and the basis is made of small integers.

Buckingham's theorem says that a physically complete relation among the quantities can be rewritten as a relation
among the groups. It does **not** say which quantities are relevant: that is the modelling hypothesis, always listed
as unchecked. The analyzer reports when the list is visibly insufficient: a base dimension carried by a single
quantity (that quantity cannot enter any group: it is irrelevant or a quantity is missing), or a target whose
dimension cannot be built from the others.

**Empirical normalisation** (dividing a curve by its maximum, z-scoring, min-max scaling) is something else: it uses
the data, not dimensions, and its result is not a pi group even when it is dimensionless. :class:`TrajectoryNormalizer`
keeps the two apart and labels every normalised trajectory with which one produced it.

**Curve collapse** (:class:`CurveCollapseAnalyzer`) measures how well a family of curves falls onto one after a
rescaling, against two controls: the raw curves, and the same rescaling with its scale factors permuted among the
members (a collapse no better than permuted scales is not evidence for the scaling). A collapse that flattens every
curve towards a constant is flagged as degenerate rather than reported as success.

Objects:

* :class:`Dimension`: exponents over named base dimensions (any system: SI ``L, M, T, Theta, I, N, J``, or e.g. a
  force-length-time system), from a unit string through :mod:`pinneapple_data.physical_units`.
* :class:`PhysicalQuantity`: name, dimension, optional value (SI) and symbol.
* :class:`DimensionalMatrix`: ``D``, its exact rank and null space.
* :class:`BuckinghamPiAnalyzer` -> :class:`DimensionlessGroup` s (with recognition of named groups: Re, Pe, Fo, Ra,
  Pr, Nu, Ma, Fr, We, St, Bi, Gr, Sc), characteristic scales of a target dimension, the implied :class:`ScalingLaw`.
* :class:`ScalingLaw`: a power law, fitted in log space with confidence intervals, or implied exactly by a single group.
* :class:`TrajectoryNormalizer`, :class:`CurveCollapseAnalyzer`, :class:`SimilarityMetrics`.
"""
from __future__ import annotations

import itertools
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from functools import reduce
from typing import Any

import numpy as np
from scipy import optimize, stats

from ._contract import Assumption, Diagnostic, DynamicsResult, MethodCard, make_rng, reproducibility
from .trajectories import Trajectory, TrajectoryEnsemble

__all__ = [
    "SI_BASES", "Dimension", "DimensionError", "PhysicalQuantity", "DimensionalMatrix", "DimensionlessGroup",
    "BuckinghamPiAnalyzer", "ScalingLaw", "TrajectoryNormalizer", "CurveCollapseAnalyzer", "SimilarityMetrics",
    "check_homogeneity", "quantities", "KNOWN_GROUPS", "CARD_BUCKINGHAM", "CARD_COLLAPSE",
]

SI_BASES: tuple[str, ...] = ("L", "M", "T", "Theta", "I", "N", "J")
_PU_ORDER = ("L", "M", "T", "Theta", "I", "N")          # pinneapple_data.physical_units: (m, kg, s, K, A, mol)
_SUPER = str.maketrans("0123456789-/", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻ᐟ")


class DimensionError(ValueError):
    """Dimensions that do not match where they must."""


def _frac(x) -> Fraction:
    if isinstance(x, Fraction):
        return x
    if isinstance(x, float):
        f = Fraction(x).limit_denominator(1000)
        if abs(float(f) - x) > 1e-12:
            raise ValueError(f"exponent {x!r} is not a simple rational; pass a Fraction")
        return f
    return Fraction(x)


# =========================================================================== dimensions
@dataclass(frozen=True)
class Dimension:
    """Exponents over named base dimensions; zero exponents are dropped. ``Dimension()`` is dimensionless.

    Arithmetic: ``a * b``, ``a / b``, ``a ** p`` (``p`` rational). Any base names may be used, so non-SI systems
    (``Dimension({"F": 1, "L": -2})``) and extra bases (information, currency) work; mixing systems is the caller's
    choice and is not converted."""
    exponents: Mapping[str, Fraction] = field(default_factory=dict)

    def __post_init__(self):
        clean = {str(k): _frac(v) for k, v in dict(self.exponents).items() if _frac(v) != 0}
        object.__setattr__(self, "exponents", dict(sorted(clean.items())))

    @classmethod
    def of(cls, **exponents) -> Dimension:
        """``Dimension.of(L=1, T=-1)``."""
        return cls(exponents)

    @classmethod
    def from_unit(cls, unit: str) -> Dimension:
        """Dimension of a unit string (``"kg/(m·s)"``, ``"W/m2/K"``), parsed by
        :func:`pinneapple_data.physical_units.parse_unit` (SI bases ``L, M, T, Theta, I, N``)."""
        from pinneapple_data.physical_units import parse_unit

        u = parse_unit(unit)
        return cls(dict(zip(_PU_ORDER, u.dim, strict=False)))

    @property
    def is_dimensionless(self) -> bool:
        return not self.exponents

    def exponent(self, base: str) -> Fraction:
        return self.exponents.get(base, Fraction(0))

    def vector(self, bases: Sequence[str]) -> list[Fraction]:
        extra = set(self.exponents) - set(bases)
        if extra:
            raise DimensionError(f"base dimensions {sorted(extra)} are not among {list(bases)}")
        return [self.exponent(b) for b in bases]

    def __mul__(self, other: Dimension) -> Dimension:
        keys = set(self.exponents) | set(other.exponents)
        return Dimension({k: self.exponent(k) + other.exponent(k) for k in keys})

    def __truediv__(self, other: Dimension) -> Dimension:
        return self * other ** -1

    def __pow__(self, p) -> Dimension:
        p = _frac(p)
        return Dimension({k: v * p for k, v in self.exponents.items()})

    def __str__(self) -> str:
        if not self.exponents:
            return "1"
        return " ".join(k if v == 1 else f"{k}{str(v).translate(_SUPER)}" for k, v in self.exponents.items())

    def __repr__(self) -> str:
        return f"Dimension({str(self)!r})"


@dataclass(frozen=True)
class PhysicalQuantity:
    """A named quantity with a dimension, and optionally a value in SI (scalar or array) and a symbol.

    ``PhysicalQuantity.from_unit("mu", 1.8e-5, "Pa s")`` converts the value to SI and takes the dimension from the
    unit. Values are optional: Buckingham analysis needs dimensions only."""
    name: str
    dimension: Dimension
    value: Any = None
    unit: str | None = None
    symbol: str | None = None
    description: str = ""

    @classmethod
    def from_unit(cls, name: str, value: Any, unit: str, **kw) -> PhysicalQuantity:
        from pinneapple_data.physical_units import parse_unit

        u = parse_unit(unit)
        if u.offset and value is not None:
            raise DimensionError(f"{name}: unit {unit!r} has an offset (e.g. °C); give a temperature difference in K "
                                 "or an absolute temperature in K")
        v = None if value is None else np.asarray(value, dtype=np.float64) * u.factor
        if v is not None and v.ndim == 0:
            v = float(v)
        return cls(name, Dimension(dict(zip(_PU_ORDER, u.dim, strict=False))), v, unit, **kw)

    @classmethod
    def of(cls, name: str, value: Any = None, **exponents) -> PhysicalQuantity:
        """``PhysicalQuantity.of("U", 2.0, L=1, T=-1)``: value already in consistent units."""
        return cls(name, Dimension(exponents), value)

    def sympy(self):
        """The quantity as a positive SymPy symbol (requires sympy)."""
        import sympy

        return sympy.Symbol(self.symbol or self.name, positive=True)


def check_homogeneity(terms: Mapping[str, Dimension | PhysicalQuantity]) -> DynamicsResult:
    """Do all ``terms`` of a sum (or both sides of an equation) have the same dimension?

    Example: Bernoulli ``{"p": Pa, "rho U^2/2": ..., "rho g h": ...}``. The result's ``estimate`` maps each term to
    its dimension; ``applicable`` is ``False`` and ``failures`` names the terms that differ from the first."""
    dims = {k: (v.dimension if isinstance(v, PhysicalQuantity) else v) for k, v in terms.items()}
    if not dims:
        raise ValueError("no terms")
    ref_name, ref = next(iter(dims.items()))
    bad = [f"{k}: {d} differs from {ref_name}: {ref}" for k, d in dims.items() if d != ref]
    diags = [Diagnostic("homogeneous", not bad, "ok" if not bad else "fail")]
    return DynamicsResult("dimensional.homogeneity", {k: str(d) for k, d in dims.items()}, diagnostics=diags,
                          applicable=not bad, failures=bad,
                          assumptions=[Assumption("dimensions as declared", "each term's dimension is the one given; "
                                                  "numeric constants are dimensionless")])


# =========================================================================== exact linear algebra
def _rref(M: list[list[Fraction]]) -> tuple[list[list[Fraction]], list[int]]:
    A = [row[:] for row in M]
    rows = len(A)
    cols = len(A[0]) if rows else 0
    piv: list[int] = []
    r = 0
    for c in range(cols):
        p = next((i for i in range(r, rows) if A[i][c] != 0), None)
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        inv = 1 / A[r][c]
        A[r] = [v * inv for v in A[r]]
        for i in range(rows):
            if i != r and A[i][c] != 0:
                f = A[i][c]
                A[i] = [a - f * b for a, b in zip(A[i], A[r], strict=False)]
        piv.append(c)
        r += 1
        if r == rows:
            break
    return A, piv


def _integerise(v: Sequence[Fraction]) -> list[Fraction]:
    """Scale a rational vector to coprime integers with a positive first non-zero entry."""
    den = reduce(lambda a, b: a * b // math.gcd(a, b), (x.denominator for x in v), 1)
    ints = [int(x * den) for x in v]
    g = reduce(math.gcd, (abs(i) for i in ints if i), 0) or 1
    ints = [i // g for i in ints]
    first = next((i for i in ints if i), 1)
    if first < 0:
        ints = [-i for i in ints]
    return [Fraction(i) for i in ints]


def _solve(A: list[list[Fraction]], b: list[Fraction]) -> list[Fraction] | None:
    """One exact solution of ``A x = b`` (free variables at 0), or ``None`` when inconsistent."""
    n = len(A[0]) if A else 0
    aug = [row[:] + [bi] for row, bi in zip(A, b, strict=False)]
    R, piv = _rref(aug)
    if n in piv:
        return None
    x = [Fraction(0)] * n
    for i, c in enumerate(piv):
        x[c] = R[i][n]
    return x


class DimensionalMatrix:
    """The dimensional matrix ``D`` (base dimensions x quantities) with exact rank and null space."""

    def __init__(self, quantities: Sequence[PhysicalQuantity], bases: Sequence[str] | None = None):
        if not quantities:
            raise ValueError("no quantities")
        names = [q.name for q in quantities]
        if len(set(names)) != len(names):
            raise ValueError(f"quantity names must be unique, got {names}")
        self.quantities = list(quantities)
        used = sorted({b for q in quantities for b in q.dimension.exponents},
                      key=lambda b: (SI_BASES.index(b) if b in SI_BASES else len(SI_BASES), b))
        self.bases = list(bases) if bases is not None else used
        self.D = [[q.dimension.exponent(b) for q in quantities] for b in self.bases]
        for q in quantities:
            q.dimension.vector(self.bases)               # raises on a base outside the list

    @property
    def names(self) -> list[str]:
        return [q.name for q in self.quantities]

    @property
    def rank(self) -> int:
        return len(_rref(self.D)[1]) if self.D else 0

    def array(self) -> np.ndarray:
        return np.array([[float(v) for v in row] for row in self.D]) if self.D else np.zeros((0, len(self.names)))

    def nullspace(self) -> list[list[Fraction]]:
        """An integer basis of ``{alpha : D alpha = 0}`` (one vector per independent dimensionless group)."""
        n = len(self.quantities)
        if not self.D:
            return [[Fraction(int(i == j)) for i in range(n)] for j in range(n)]
        R, piv = _rref(self.D)
        free = [j for j in range(n) if j not in piv]
        basis = []
        for f in free:
            v = [Fraction(0)] * n
            v[f] = Fraction(1)
            for i, c in enumerate(piv):
                v[c] = -R[i][f]
            basis.append(_integerise(v))
        return basis

    def is_dimensionless(self, alpha: Sequence[Fraction]) -> bool:
        return all(sum(d * a for d, a in zip(row, alpha, strict=False)) == 0 for row in self.D)

    def __repr__(self) -> str:
        return f"DimensionalMatrix(bases={self.bases}, quantities={self.names}, rank={self.rank})"


# =========================================================================== groups
# Named groups by the conventional quantity names; recognition matches exponents up to inversion.
KNOWN_GROUPS: dict[str, dict[str, int]] = {
    "Reynolds": {"rho": 1, "U": 1, "L": 1, "mu": -1},
    "Reynolds (kinematic)": {"U": 1, "L": 1, "nu": -1},
    "Peclet": {"U": 1, "L": 1, "alpha": -1},
    "Peclet (mass)": {"U": 1, "L": 1, "D": -1},
    "Fourier": {"alpha": 1, "t": 1, "L": -2},
    "Fourier (mass)": {"D": 1, "t": 1, "L": -2},
    "Prandtl": {"nu": 1, "alpha": -1},
    "Schmidt": {"nu": 1, "D": -1},
    "Rayleigh": {"g": 1, "beta": 1, "dT": 1, "L": 3, "nu": -1, "alpha": -1},
    "Grashof": {"g": 1, "beta": 1, "dT": 1, "L": 3, "nu": -2},
    "Nusselt": {"h": 1, "L": 1, "k": -1},
    "Biot": {"h": 1, "L": 1, "k_s": -1},
    "Mach": {"U": 1, "c": -1},
    "Froude": {"U": 2, "g": -1, "L": -1},
    "Weber": {"rho": 1, "U": 2, "L": 1, "sigma": -1},
    "Strouhal": {"f": 1, "L": 1, "U": -1},
}


@dataclass(frozen=True)
class DimensionlessGroup:
    """``Pi = prod q_i ** exponents[q_i]``, exponents exact (``Fraction``)."""
    exponents: Mapping[str, Fraction]
    name: str = ""

    def __post_init__(self):
        object.__setattr__(self, "exponents", {k: _frac(v) for k, v in self.exponents.items() if _frac(v) != 0})

    def formula(self) -> str:
        num = [f"{k}" if v == 1 else f"{k}^{v}" for k, v in self.exponents.items() if v > 0]
        den = [f"{k}" if v == -1 else f"{k}^{-v}" for k, v in self.exponents.items() if v < 0]
        top = " ".join(num) or "1"
        return top if not den else f"{top} / ({' '.join(den)})" if len(den) > 1 else f"{top} / {den[0]}"

    def dimension(self, quantities: Mapping[str, PhysicalQuantity]) -> Dimension:
        return reduce(lambda a, b: a * b, (quantities[k].dimension ** v for k, v in self.exponents.items()),
                      Dimension())

    def evaluate(self, values: Mapping[str, Any]) -> np.ndarray | float:
        """``Pi`` from values in consistent units (arrays broadcast). ``values`` may map names to numbers or to
        :class:`PhysicalQuantity` with a value."""
        out: Any = 1.0
        for k, e in self.exponents.items():
            if k not in values:
                raise KeyError(f"group {self.formula()} needs {k!r}")
            v = values[k].value if isinstance(values[k], PhysicalQuantity) else values[k]
            if v is None:
                raise ValueError(f"{k!r} has no value")
            out = out * np.asarray(v, dtype=np.float64) ** float(e)
        return float(out) if np.ndim(out) == 0 else out

    def recognize(self, aliases: Mapping[str, str] | None = None) -> str | None:
        """The conventional name of this group (or of its inverse / a power of it) when the quantity names follow
        :data:`KNOWN_GROUPS` (``aliases`` maps your names to those). ``None`` if it is not a known group."""
        al = dict(aliases or {})
        mine = {al.get(k, k): v for k, v in self.exponents.items()}
        for name, ref in KNOWN_GROUPS.items():
            if set(ref) != set(mine):
                continue
            ratios = {Fraction(mine[k]) / ref[k] for k in ref}
            if len(ratios) == 1:
                p = ratios.pop()
                return name if p == 1 else f"{name}^{p}" if p != -1 else f"1/{name}"
        return None

    def power(self, p) -> DimensionlessGroup:
        p = _frac(p)
        return DimensionlessGroup({k: v * p for k, v in self.exponents.items()}, self.name)

    def __mul__(self, other: DimensionlessGroup) -> DimensionlessGroup:
        keys = set(self.exponents) | set(other.exponents)
        return DimensionlessGroup({k: self.exponents.get(k, 0) + other.exponents.get(k, 0) for k in keys})

    def sympy(self, quantities: Mapping[str, PhysicalQuantity] | None = None):
        import sympy

        expr = sympy.Integer(1)
        for k, v in self.exponents.items():
            sym = quantities[k].sympy() if quantities and k in quantities else sympy.Symbol(k, positive=True)
            expr *= sym ** sympy.Rational(v.numerator, v.denominator)
        return expr

    def __str__(self) -> str:
        return f"{self.name + ' = ' if self.name else ''}{self.formula()}"


# =========================================================================== scaling laws
@dataclass
class ScalingLaw:
    """``y = C * prod x_i ** b_i``.

    ``exact=True`` when the exponents follow from dimensional analysis alone (a single group: only ``C`` is
    unknown); otherwise they are fitted. ``ci`` holds confidence intervals (``level``) of ``log C`` and each
    exponent from ordinary least squares in log space, which assumes multiplicative log-normal errors."""
    target: str
    exponents: dict[str, float]
    constant: float | None = None
    exact: bool = False
    ci: dict[str, tuple[float, float]] = field(default_factory=dict)
    level: float = 0.95
    r2: float | None = None
    residual_std: float | None = None
    n: int = 0

    def predict(self, values: Mapping[str, Any]) -> np.ndarray | float:
        if self.constant is None:
            raise ValueError("the constant is not known; fit it with ScalingLaw.fit_constant(...) first")
        out: Any = self.constant
        for k, b in self.exponents.items():
            out = out * np.asarray(values[k], dtype=np.float64) ** b
        return float(out) if np.ndim(out) == 0 else out

    @classmethod
    def fit(cls, y, x: Mapping[str, Any], target: str = "y", level: float = 0.95) -> ScalingLaw:
        """Least squares of ``log y`` on ``log x_i`` with a t-based confidence interval per coefficient."""
        y = np.asarray(y, dtype=np.float64).ravel()
        names = list(x)
        X = np.column_stack([np.log(np.asarray(x[k], dtype=np.float64).ravel()) for k in names]) if names \
            else np.zeros((y.size, 0))
        if np.any(y <= 0) or (X.size and not np.all(np.isfinite(X))):
            raise ValueError("a power law needs positive y and x")
        A = np.column_stack([np.ones(y.size), X])
        n, p = A.shape
        if n <= p:
            raise ValueError(f"{n} points for {p} coefficients")
        coef, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
        res = np.log(y) - A @ coef
        s2 = float(res @ res / (n - p))
        cov = s2 * np.linalg.pinv(A.T @ A)
        tq = stats.t.ppf(0.5 + level / 2, n - p)
        se = np.sqrt(np.diag(cov))
        ss_tot = float(np.sum((np.log(y) - np.log(y).mean()) ** 2))
        ci = {"log_C": (float(coef[0] - tq * se[0]), float(coef[0] + tq * se[0]))}
        ci.update({k: (float(coef[i + 1] - tq * se[i + 1]), float(coef[i + 1] + tq * se[i + 1]))
                   for i, k in enumerate(names)})
        return cls(target, {k: float(coef[i + 1]) for i, k in enumerate(names)}, float(np.exp(coef[0])), False, ci,
                   level, 1 - float(res @ res) / ss_tot if ss_tot > 0 else None, math.sqrt(s2), n)

    def fit_constant(self, y, values: Mapping[str, Any]) -> ScalingLaw:
        """The constant ``C`` of an exact law from data, by the geometric mean of ``y / prod x^b``, with a t interval
        on ``log C``."""
        y = np.asarray(y, dtype=np.float64).ravel()
        base = np.ones_like(y)
        for k, b in self.exponents.items():
            base = base * np.asarray(values[k], dtype=np.float64).ravel() ** b
        r = np.log(y / base)
        n = r.size
        m = float(r.mean())
        sd = float(r.std(ddof=1)) if n > 1 else math.nan
        tq = stats.t.ppf(0.5 + self.level / 2, max(n - 1, 1))
        half = tq * sd / math.sqrt(n) if n > 1 else math.nan
        return replace(self, constant=math.exp(m), ci={**self.ci, "log_C": (m - half, m + half)},
                       residual_std=sd, n=n)

    def __str__(self) -> str:
        c = "C" if self.constant is None else f"{self.constant:.6g}"
        terms = " ".join(f"{k}^{b:.4g}" for k, b in self.exponents.items())
        return f"{self.target} = {c} {terms}" + (" (exponents exact)" if self.exact else "")


# =========================================================================== analyzer
_A_COMPLETE = Assumption("complete list of quantities", "every quantity the target depends on is in the list, and "
                         "none is irrelevant (Buckingham's theorem only rewrites the relation among those given)")
_A_DIMS = Assumption("dimensions as declared", "each quantity's dimension is the one given; dimensional constants "
                     "(g, a material constant) are listed as quantities when they matter")

CARD_BUCKINGHAM = MethodCard(
    "dimensional.buckingham_pi", "primitive",
    inputs={"quantities": "PhysicalQuantity list with dimensions (values optional)",
            "target": "the dependent quantity (optional)", "repeating": "repeating variables (optional)"},
    assumptions=[_A_COMPLETE, _A_DIMS],
    config={"bases": "base dimensions (default: those used)", "repeating": "chosen to exclude the target"},
    outputs={"groups": "n - rank(D) independent DimensionlessGroup with exact integer exponents",
             "rank": "rank of D", "recognized": "conventional names of the groups that match"},
    failure_modes=["a base dimension carried by only one quantity (that quantity cannot enter any group)",
                   "the target's dimension cannot be formed from the other quantities (a quantity is missing)",
                   "rank(D) = n: no group exists",
                   "the repeating variables given are not dimensionally independent"],
    references=["Buckingham 1914, Phys. Rev. 4:345", "Barenblatt 1996, Scaling, self-similarity and intermediate "
                "asymptotics, ch. 1", "Bridgman 1922, Dimensional Analysis"])


class BuckinghamPiAnalyzer:
    """Dimensionless groups of a set of quantities; see the module docstring.

    ``target``: the dependent quantity. The groups are built with repeating variables (Rayleigh's method): ``r``
    dimensionally independent quantities other than the target, and one group per remaining quantity, so the target
    appears in exactly one group. ``repeating`` chooses them; by default the first independent set in the given order
    is used (any choice gives an equivalent basis: the groups are unique only up to products and powers)."""

    card = CARD_BUCKINGHAM

    def __init__(self, quantities: Sequence[PhysicalQuantity], target: str | None = None,
                 repeating: Sequence[str] | None = None, bases: Sequence[str] | None = None):
        self.matrix = DimensionalMatrix(quantities, bases)
        self.quantities = {q.name: q for q in quantities}
        if target is not None and target not in self.quantities:
            raise KeyError(f"target {target!r} is not among {list(self.quantities)}")
        self.target = target
        self.repeating = list(repeating) if repeating is not None else None

    # ------------------------------------------------------------------ helpers
    def _columns(self, names: Sequence[str]) -> list[list[Fraction]]:
        idx = [self.matrix.names.index(n) for n in names]
        return [[row[i] for i in idx] for row in self.matrix.D]

    def _independent(self, names: Sequence[str]) -> bool:
        return len(_rref(self._columns(names))[1]) == len(names) if names else True

    def _choose_repeating(self) -> tuple[list[str], int]:
        r = self.matrix.rank
        pool = [n for n in self.matrix.names if n != self.target]
        count = 0
        chosen = None
        for combo in itertools.combinations(pool, r):
            if self._independent(combo) and len(_rref(self._columns(combo))[1]) == r:
                count += 1
                if chosen is None:
                    chosen = list(combo)
        if chosen is None:
            raise DimensionError("no set of repeating variables excluding the target spans the dimensions")
        return chosen, count

    def characteristic_scales(self, dimension: Dimension, using: Sequence[str] | None = None
                              ) -> dict[str, Any]:
        """Products of the quantities (``using``, default all but the target) with the given dimension: one
        particular solution and the groups that make it ambiguous. With no ambiguity the scale is unique (e.g. the
        only time scale from ``L`` and ``alpha`` is ``L^2/alpha``); otherwise every scale times any power of the
        groups is another valid scale (``L/U`` and ``L^2/alpha`` differ by Pe)."""
        names = list(using) if using is not None else [n for n in self.matrix.names if n != self.target]
        A = self._columns(names)
        unknown = set(dimension.exponents) - set(self.matrix.bases)
        if unknown:
            return {"exists": False, "scale": None, "ambiguity": [],
                    "reason": f"no quantity carries the base dimension(s) {sorted(unknown)}"}
        b = dimension.vector(self.matrix.bases)
        x = _solve(A, b)
        if x is None:
            return {"exists": False, "scale": None, "ambiguity": [],
                    "reason": f"no product of {names} has dimension {dimension}"}
        sub = DimensionalMatrix([self.quantities[n] for n in names], self.matrix.bases)
        amb = [DimensionlessGroup(dict(zip(names, v, strict=False))) for v in sub.nullspace()]
        return {"exists": True, "scale": DimensionlessGroup(dict(zip(names, x, strict=False))), "ambiguity": amb,
                "unique": not amb}

    # ------------------------------------------------------------------ main
    def analyze(self, aliases: Mapping[str, str] | None = None) -> DynamicsResult:
        t0 = time.perf_counter()
        D = self.matrix
        n, r = len(D.names), D.rank
        diags = [Diagnostic("n_quantities", n, "info"), Diagnostic("rank", r, "info"),
                 Diagnostic("n_groups", n - r, "ok" if n > r else "fail")]
        failures, limits = [], ["groups are unique only up to products and powers of each other"]
        carriers = {b: [q for q in D.names if self.quantities[q].dimension.exponent(b) != 0] for b in D.bases}
        isolated = {b: qs[0] for b, qs in carriers.items() if len(qs) == 1}
        for b, q in isolated.items():
            diags.append(Diagnostic(f"isolated_dimension:{b}", q, "warn",
                                    f"{q} is the only quantity with dimension {b}: it cannot enter any group, so "
                                    f"either it is irrelevant or a quantity carrying {b} is missing"))
        if self.target is not None:
            others = [q for q in D.names if q != self.target]
            cs = self.characteristic_scales(self.quantities[self.target].dimension, others) if others else \
                {"exists": False}
            if not cs["exists"]:
                failures.append(f"insufficient quantities: the dimension of the target {self.target!r} "
                                f"({self.quantities[self.target].dimension}) cannot be formed from {others}")
        if n == r:
            failures.append(f"rank(D) = n = {n}: no dimensionless group can be formed")
        groups: list[DimensionlessGroup] = []
        repeating_used, n_choices = None, None
        if not failures:
            if self.target is not None:
                if self.repeating is not None:
                    if self.target in self.repeating:
                        raise ValueError("the target cannot be a repeating variable")
                    if len(self.repeating) != r or not self._independent(self.repeating):
                        raise DimensionError(f"repeating variables {self.repeating} are not {r} dimensionally "
                                             "independent quantities")
                    repeating_used = self.repeating
                else:
                    repeating_used, n_choices = self._choose_repeating()
                A = self._columns(repeating_used)
                for q in [self.target] + [q for q in D.names if q != self.target and q not in repeating_used]:
                    rhs = [-v for v in self.quantities[q].dimension.vector(D.bases)]
                    x = _solve(A, rhs)
                    ex = dict(zip(repeating_used, x, strict=False))
                    ex[q] = Fraction(1)
                    v = _integerise([ex.get(k, Fraction(0)) for k in D.names])
                    if v[D.names.index(q)] < 0:
                        v = [-a for a in v]
                    groups.append(DimensionlessGroup(dict(zip(D.names, v, strict=False))))
            else:
                groups = [DimensionlessGroup(dict(zip(D.names, v, strict=False))) for v in D.nullspace()]
            named = []
            for i, g in enumerate(groups):
                nm = g.recognize(aliases)
                groups[i] = replace(g, name=nm or f"Pi_{i + 1}")
                named.append(nm)
            for g in groups:
                assert D.is_dimensionless([g.exponents.get(k, Fraction(0)) for k in D.names])
        est = {"groups": [str(g) for g in groups], "rank": r, "n_groups": len(groups),
               "recognized": {g.name: g.formula() for g in groups if not g.name.startswith("Pi_")},
               "repeating": repeating_used, "target_group": groups[0].name if (groups and self.target) else None}
        if n_choices is not None:
            est["repeating_alternatives"] = n_choices
        cfg = {"quantities": {q: str(self.quantities[q].dimension) for q in D.names}, "target": self.target,
               "repeating": self.repeating, "bases": D.bases}
        rep = reproducibility(CARD_BUCKINGHAM.name, cfg, [], wall_time_s=time.perf_counter() - t0)
        return DynamicsResult(CARD_BUCKINGHAM.name, est, {"kind": "exact rational arithmetic: no numerical error"},
                              [_A_COMPLETE, _A_DIMS], cfg, diags, {}, applicable=not failures, failures=failures,
                              limits=limits, reproducibility=rep,
                              details={"groups": groups, "matrix": D.array(), "isolated": isolated})

    def groups(self, aliases: Mapping[str, str] | None = None) -> list[DimensionlessGroup]:
        r = self.analyze(aliases)
        if not r.applicable:
            raise DimensionError("; ".join(r.failures))
        return r.details["groups"]

    def express(self, group: DimensionlessGroup, basis: Sequence[DimensionlessGroup] | None = None
                ) -> dict[str, Fraction] | None:
        """Exponents ``c`` with ``group = prod basis_j ** c_j`` (exact), or ``None`` when ``group`` is not a product of
        the basis (it is then not dimensionless in these quantities, or uses a quantity not in the list). Shows,
        e.g., that the Rayleigh number is a product of the groups found for ``g, beta, dT, L, nu, alpha``."""
        basis = list(basis) if basis is not None else self.groups()
        names = self.matrix.names
        if set(group.exponents) - set(names):
            return None
        A = [[g.exponents.get(k, Fraction(0)) for g in basis] for k in names]
        x = _solve(A, [group.exponents.get(k, Fraction(0)) for k in names])
        return None if x is None else {g.name or g.formula(): c for g, c in zip(basis, x, strict=True)}

    def scaling_law(self) -> ScalingLaw:
        """The power law implied when the target's group is the only group (``Pi_target = const``): exact exponents,
        unknown constant. Raises when other groups exist (the law is then ``Pi_target = f(other groups)``, whose form
        dimensional analysis does not give)."""
        if self.target is None:
            raise ValueError("a scaling law needs a target")
        gs = self.groups()
        if len(gs) != 1:
            raise DimensionError(f"{len(gs)} groups: Pi_target = f({', '.join(g.name for g in gs[1:])}), whose form "
                                 "is not fixed by dimensions; fit it with ScalingLaw.fit or a model")
        g = gs[0]
        a = g.exponents[self.target]
        return ScalingLaw(self.target, {k: float(-v / a) for k, v in g.exponents.items() if k != self.target},
                          exact=True)


# =========================================================================== normalisation
class TrajectoryNormalizer:
    """Rescales trajectories, either **dimensionally** (by a time scale and state scales built from physical
    quantities, whose dimensions are checked) or **empirically** (from the data: min-max, z-score, peak, initial
    value). The two are labelled differently in ``source["normalization"]``; an empirical normalisation is never
    called a pi representation."""

    def __init__(self, *, time_scale: float | None = None, state_scales: Mapping[str, float] | None = None,
                 time_offset: float = 0.0, state_offsets: Mapping[str, float] | None = None,
                 kind: str = "dimensional", provenance: Mapping[str, Any] | None = None):
        if kind not in ("dimensional", "empirical"):
            raise ValueError("kind must be 'dimensional' or 'empirical'")
        if time_scale is not None and not time_scale > 0:
            raise ValueError("time_scale must be positive")
        self.time_scale = time_scale
        self.state_scales = dict(state_scales or {})
        self.time_offset = time_offset
        self.state_offsets = dict(state_offsets or {})
        self.kind = kind
        self.provenance = dict(provenance or {})

    @classmethod
    def dimensional(cls, quantities: Mapping[str, PhysicalQuantity], *, time: DimensionlessGroup | None = None,
                    states: Mapping[str, DimensionlessGroup] | None = None,
                    state_dimensions: Mapping[str, Dimension] | None = None,
                    state_offsets: Mapping[str, float] | None = None) -> TrajectoryNormalizer:
        """Scales built from quantity products. ``time`` must have the dimension of time (``T``) and each state scale
        the dimension in ``state_dimensions`` (when given): otherwise :class:`DimensionError`. Values come from the
        quantities' ``value`` (SI)."""
        prov: dict[str, Any] = {}
        ts = None
        if time is not None:
            d = time.dimension(quantities)
            if d != Dimension.of(T=1):
                raise DimensionError(f"time scale {time.formula()} has dimension {d}, not T")
            ts = float(time.evaluate(quantities))
            prov["time_scale"] = time.formula()
        ss = {}
        for name, g in (states or {}).items():
            d = g.dimension(quantities)
            want = (state_dimensions or {}).get(name)
            if want is not None and d != want:
                raise DimensionError(f"state scale for {name}: {g.formula()} has dimension {d}, not {want}")
            ss[name] = float(g.evaluate(quantities))
            prov[f"scale[{name}]"] = g.formula()
        return cls(time_scale=ts, state_scales=ss, state_offsets=state_offsets, kind="dimensional", provenance=prov)

    @classmethod
    def empirical(cls, trajectory: Trajectory, method: str = "minmax", time: str | None = None
                  ) -> TrajectoryNormalizer:
        """Scales from the data of ``trajectory``: ``"minmax"`` (to [0, 1]), ``"zscore"``, ``"peak"`` (divide by the
        largest absolute value), ``"initial"`` (divide by the first value). ``time``: ``"span"`` divides time by the
        observed span, ``None`` leaves it. These are data-dependent and **not** dimensional analysis."""
        ss, so = {}, {}
        for nm in trajectory.names:
            v = trajectory.component(nm)
            v = v[~np.isnan(v)]
            if v.size == 0:
                continue
            if method == "minmax":
                so[nm], ss[nm] = float(v.min()), float(v.max() - v.min()) or 1.0
            elif method == "zscore":
                so[nm], ss[nm] = float(v.mean()), float(v.std()) or 1.0
            elif method == "peak":
                ss[nm] = float(np.abs(v).max()) or 1.0
            elif method == "initial":
                if v[0] == 0:
                    raise ValueError(f"{nm}: initial value is 0, cannot normalise by it")
                ss[nm] = float(v[0])
            else:
                raise ValueError("method must be minmax, zscore, peak or initial")
        ts, to = None, 0.0
        if time == "span":
            to, ts = trajectory.t[0], float(trajectory.t[-1] - trajectory.t[0]) or 1.0
        return cls(time_scale=ts, time_offset=to, state_scales=ss, state_offsets=so, kind="empirical",
                   provenance={"method": method, "time": time})

    def apply(self, trajectory: Trajectory) -> Trajectory:
        t = (trajectory.t - self.time_offset) / (self.time_scale or 1.0)
        X = trajectory.x.copy()
        for j, nm in enumerate(trajectory.names):
            X[:, j] = (X[:, j] - self.state_offsets.get(nm, 0.0)) / self.state_scales.get(nm, 1.0)
        source = dict(trajectory.source)
        source["normalization"] = {"kind": self.kind, "time_scale": self.time_scale, "state_scales": self.state_scales,
                                   "time_offset": self.time_offset, "state_offsets": self.state_offsets,
                                   **self.provenance,
                                   "note": "dimensional: scales built from physical quantities" if self.kind ==
                                   "dimensional" else "empirical: scales from the data; not a pi representation"}
        units = {"t": "1"} if self.time_scale is not None else {k: v for k, v in trajectory.units.items() if k == "t"}
        units.update({nm: "1" for nm in trajectory.names if nm in self.state_scales})
        end = (trajectory.observed_until - self.time_offset) / (self.time_scale or 1.0)
        return Trajectory(t, X, trajectory.names, units, end, source)

    def invert(self, trajectory: Trajectory) -> Trajectory:
        t = trajectory.t * (self.time_scale or 1.0) + self.time_offset
        X = trajectory.x.copy()
        for j, nm in enumerate(trajectory.names):
            X[:, j] = X[:, j] * self.state_scales.get(nm, 1.0) + self.state_offsets.get(nm, 0.0)
        source = {k: v for k, v in trajectory.source.items() if k != "normalization"}
        end = trajectory.observed_until * (self.time_scale or 1.0) + self.time_offset
        return Trajectory(t, X, trajectory.names, {}, end, source)


# =========================================================================== collapse
_A_COMMON = Assumption("common support", "the rescaled curves overlap on a common range of the abscissa; the "
                       "collapse is measured only there")
_A_SAME_LAW = Assumption("one law", "the members obey the same scaling function; the collapse tests this, it does "
                         "not explain why")

CARD_COLLAPSE = MethodCard(
    "dimensional.curve_collapse", "validation",
    inputs={"curves": "ensemble of curves y_k(x_k), one per member (Trajectory: x = t, y = one component)",
            "scales": "per-member abscissa and ordinate scale factors (from groups or fitted)"},
    assumptions=[_A_COMMON, _A_SAME_LAW],
    config={"n_grid": "points of the common grid", "n_permutations": "size of the permutation control",
            "log_x": "measure on a log abscissa"},
    outputs={"collapse": "mean across-member variance / variance of the master curve (0 = perfect)",
             "raw": "the same metric before rescaling", "p_value": "share of permuted scalings that collapse as well",
             "degenerate": "true when the master curve is flat"},
    failure_modes=["no common support after rescaling", "fewer than 3 members (no meaningful permutation control)",
                   "degenerate collapse: the rescaling flattens every curve (variance of the master curve near 0)",
                   "scale factors all equal (a permutation control cannot distinguish anything)"],
    references=["Bhattacharjee & Seno 2001, J. Phys. A 34:6375 (collapse measure)",
                "Barenblatt 1996, Scaling, self-similarity and intermediate asymptotics"])


def _collapse_metric(xs: list[np.ndarray], ys: list[np.ndarray], n_grid: int, log_x: bool
                     ) -> tuple[float, float, np.ndarray, np.ndarray] | None:
    lo = max(float(x.min()) for x in xs)
    hi = min(float(x.max()) for x in xs)
    if not hi > lo:
        return None
    grid = np.geomspace(lo, hi, n_grid) if log_x and lo > 0 else np.linspace(lo, hi, n_grid)
    Y = np.stack([np.interp(grid, x, y) for x, y in zip(xs, ys, strict=False)])
    master = Y.mean(axis=0)
    spread = float(np.mean(Y.var(axis=0)))
    signal = float(master.var())
    return spread, signal, grid, master


class CurveCollapseAnalyzer:
    """Does a family of curves collapse onto one after rescaling? See the module docstring.

    Each member ``k`` is a curve ``y_k(x)`` (a :class:`Trajectory` component against its time). A rescaling divides
    ``x`` by ``a_k`` and ``y`` by ``b_k``. The collapse measure is ``mean_x Var_k[y] / Var_x[mean_k y]`` on the
    common support (0: all curves identical; about 1 or more: no collapse)."""

    card = CARD_COLLAPSE

    def __init__(self, ensemble: TrajectoryEnsemble | Sequence[Trajectory], component: str | int = 0, *,
                 n_grid: int = 200, log_x: bool = False):
        self.ens = ensemble if isinstance(ensemble, TrajectoryEnsemble) else TrajectoryEnsemble(list(ensemble))
        self.component = component
        self.n_grid = n_grid
        self.log_x = log_x
        self._curves = []
        for tr in self.ens:
            y = tr.component(component)
            ok = ~np.isnan(y)
            self._curves.append((tr.t[ok], y[ok]))

    def _metric(self, a: np.ndarray, b: np.ndarray):
        xs = [x / ak for (x, _), ak in zip(self._curves, a, strict=False)]
        ys = [y / bk for (_, y), bk in zip(self._curves, b, strict=False)]
        return _collapse_metric(xs, ys, self.n_grid, self.log_x)

    def measure(self, x_scales: Sequence[float] | None = None, y_scales: Sequence[float] | None = None) -> float:
        """The collapse measure for these scale factors (``inf`` without common support)."""
        m = len(self._curves)
        a = np.ones(m) if x_scales is None else np.asarray(x_scales, dtype=float)
        b = np.ones(m) if y_scales is None else np.asarray(y_scales, dtype=float)
        res = self._metric(a, b)
        if res is None:
            return math.inf
        spread, signal, _, _ = res
        return spread / signal if signal > 0 else math.inf

    def analyze(self, x_scales: Sequence[float] | None = None, y_scales: Sequence[float] | None = None, *,
                label: str = "", origin: str = "dimensional", n_permutations: int = 999, seed: int | None = 0,
                degenerate_tol: float = 1e-8) -> DynamicsResult:
        """Collapse under the given scales, against the raw curves and a permutation control (the same scale
        factors assigned to the wrong members).

        ``origin`` says where the scales come from: ``"dimensional"`` (built from physical quantities, independent of
        these curves), ``"empirical"`` (from the curves themselves: min-max, peak, span; part of the agreement is then
        forced by construction, whatever the law) or ``"fitted"`` (exponents fitted on these curves: the permutation
        p-value is optimistic). The last two are reported as warnings."""
        if origin not in ("dimensional", "empirical", "fitted"):
            raise ValueError("origin must be 'dimensional', 'empirical' or 'fitted'")
        t0 = time.perf_counter()
        m = len(self._curves)
        a = np.ones(m) if x_scales is None else np.asarray(x_scales, dtype=float)
        b = np.ones(m) if y_scales is None else np.asarray(y_scales, dtype=float)
        if a.shape != (m,) or b.shape != (m,):
            raise ValueError(f"need one x and one y scale per member ({m})")
        failures, diags = [], []
        if m < 3:
            failures.append(f"{m} members: a collapse test needs at least 3")
        res = self._metric(a, b)
        raw = self.measure()
        if res is None:
            failures.append("no common support of the rescaled curves")
            value, signal, master, grid = math.inf, 0.0, None, None
        else:
            spread, signal, grid, master = res
            value = spread / signal if signal > 0 else math.inf
        yrange = max(float(np.ptp(y / bk)) for (_, y), bk in zip(self._curves, b, strict=False)) if m else 0.0
        degenerate = bool(res is not None and signal <= degenerate_tol * max(yrange ** 2, 1e-300))
        if degenerate:
            failures.append("degenerate collapse: the master curve is flat, so a small spread says nothing")
        rng, seed = make_rng(seed)
        perm_vals = []
        distinct = np.unique(np.round(np.column_stack([a, b]), 12), axis=0).shape[0] > 1
        if not distinct:
            diags.append(Diagnostic("permutation_control", "not informative", "warn",
                                    "all members have the same scales: permuting them changes nothing"))
        elif not failures:
            for _ in range(n_permutations):
                p = rng.permutation(m)
                perm_vals.append(self.measure(a[p], b[p]))
        perm = np.array(perm_vals)
        p_value = float((1 + np.sum(perm <= value)) / (1 + perm.size)) if perm.size else math.nan
        if perm.size:
            diags.append(Diagnostic("permutation_p_value", p_value, "ok" if p_value <= 0.05 else "warn",
                                    "share of permuted scalings that collapse at least as well", threshold=0.05))
            diags.append(Diagnostic("permuted_median", float(np.median(perm)), "info"))
        if origin == "empirical":
            diags.append(Diagnostic("collapse_by_construction", True, "warn",
                                    "scales taken from the curves themselves force extremes or end points to agree: "
                                    "curves following different laws can also collapse; this is not a pi scaling"))
        elif origin == "fitted":
            diags.append(Diagnostic("scales_fitted_on_these_curves", True, "warn",
                                    "the permutation control does not account for the fit: validate on held-out "
                                    "members"))
        diags.append(Diagnostic("raw_collapse", raw, "info"))
        diags.append(Diagnostic("improvement_over_raw", raw / value if value > 0 else math.inf, "info"))
        est = {"collapse": value, "raw": raw, "p_value": p_value, "degenerate": degenerate, "label": label,
               "origin": origin, "members": m}
        cfg = {"component": self.component, "n_grid": self.n_grid, "log_x": self.log_x,
               "n_permutations": n_permutations, "label": label, "origin": origin}
        rep = reproducibility(CARD_COLLAPSE.name, cfg, [self.ens, a, b], seed=seed,
                              wall_time_s=time.perf_counter() - t0)
        return DynamicsResult(CARD_COLLAPSE.name, est,
                              {"permutation_null": {"median": float(np.median(perm)) if perm.size else None,
                                                    "q05": float(np.quantile(perm, 0.05)) if perm.size else None}},
                              [_A_COMMON, _A_SAME_LAW], cfg, diags, {}, applicable=not failures,
                              failures=failures,
                              limits=["the measure is computed on the common support only",
                                      "a good collapse supports the scaling; it does not prove the physics"],
                              reproducibility=rep, details={"grid": grid, "master": master})

    def fit_exponents(self, parameter: Sequence[float], *, x0: tuple[float, float] = (0.0, 0.0),
                      fit_y: bool = True) -> dict[str, Any]:
        """Exponents ``(p, q)`` of the rescaling ``x / P^p``, ``y / P^q`` that best collapse the curves, with ``P`` a
        per-member parameter (Nelder-Mead on the collapse measure). Also returns the curvature of the measure at the
        optimum (a flat minimum means the exponents are poorly determined: an unstable collapse)."""
        P = np.asarray(parameter, dtype=float)
        if np.any(P <= 0):
            raise ValueError("the parameter must be positive")

        def f(v):
            p, q = (v[0], v[1]) if fit_y else (v[0], 0.0)
            return self.measure(P ** p, P ** q)

        start = np.array(x0 if fit_y else x0[:1], dtype=float)
        opt = optimize.minimize(f, start, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-12,
                                                                         "maxiter": 2000})
        h = 1e-3
        curv = []
        for i in range(opt.x.size):
            e = np.zeros_like(opt.x)
            e[i] = h
            curv.append((f(opt.x + e) - 2 * opt.fun + f(opt.x - e)) / h ** 2)
        return {"x_exponent": float(opt.x[0]), "y_exponent": float(opt.x[1]) if fit_y else 0.0,
                "collapse": float(opt.fun), "curvature": curv, "converged": bool(opt.success)}

    def sensitivity(self, parameter: Sequence[float], x_exponent: float, y_exponent: float = 0.0,
                    deltas: Sequence[float] = (-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2)) -> dict[str, Any]:
        """The collapse measure when the exponents are perturbed: a sharp minimum at the nominal exponents is a
        stable collapse, a flat one is not (the data cannot pin the scaling)."""
        P = np.asarray(parameter, dtype=float)
        dx = [self.measure(P ** (x_exponent + d), P ** y_exponent) for d in deltas]
        dy = [self.measure(P ** x_exponent, P ** (y_exponent + d)) for d in deltas]
        return {"deltas": list(deltas), "x_exponent": dx, "y_exponent": dy}


# =========================================================================== similarity
class SimilarityMetrics:
    """Are two systems (or datasets) similar? Dynamic similarity holds when every dimensionless group has the same
    value; :meth:`groups` compares them by log-ratio. :meth:`curves` compares two normalised curves on their common
    support (relative L2 and max distance)."""

    @staticmethod
    def groups(groups: Sequence[DimensionlessGroup], a: Mapping[str, Any], b: Mapping[str, Any], *,
               rtol: float = 0.05) -> DynamicsResult:
        rows = {}
        diags = []
        for g in groups:
            va, vb = g.evaluate(a), g.evaluate(b)
            lr = float(np.log(vb / va))
            rows[g.name or g.formula()] = {"a": va, "b": vb, "log_ratio": lr}
            ok = abs(lr) <= math.log1p(rtol)
            diags.append(Diagnostic(f"match:{g.name or g.formula()}", lr, "ok" if ok else "warn",
                                    threshold=math.log1p(rtol)))
        similar = all(d.status == "ok" for d in diags)
        return DynamicsResult("dimensional.similarity", {"similar": similar, "groups": rows}, diagnostics=diags,
                              assumptions=[_A_COMPLETE],
                              limits=["similarity in the listed groups only; a group missing from the list is not "
                                      "compared"], config={"rtol": rtol})

    @staticmethod
    def curves(a: Trajectory, b: Trajectory, component: str | int = 0, n_grid: int = 200) -> dict[str, float]:
        ya, yb = a.component(component), b.component(component)
        lo, hi = max(a.t[0], b.t[0]), min(a.t[-1], b.t[-1])
        if not hi > lo:
            raise ValueError("the curves have no common support")
        g = np.linspace(lo, hi, n_grid)
        fa, fb = np.interp(g, a.t, ya), np.interp(g, b.t, yb)
        scale = max(float(np.sqrt(np.mean(fa ** 2))), 1e-300)
        return {"relative_l2": float(np.sqrt(np.mean((fa - fb) ** 2)) / scale),
                "max_abs": float(np.max(np.abs(fa - fb))), "support": (float(lo), float(hi))}


def quantities(*items: PhysicalQuantity | Iterable[PhysicalQuantity]) -> dict[str, PhysicalQuantity]:
    """``{name: quantity}`` from quantities or iterables of them (a convenience for ``evaluate``)."""
    out: dict[str, PhysicalQuantity] = {}
    for it in items:
        for q in ([it] if isinstance(it, PhysicalQuantity) else it):
            out[q.name] = q
    return out


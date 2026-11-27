"""Interpolation and approximation, with the Lebesgue constant and the Runge phenomenon measured (#1058).

Three questions, and this module answers each with a number:

1. **Does adding nodes help?** For equispaced nodes it eventually does not: the Lebesgue constant grows
   exponentially with the degree, so the interpolant's error can grow even as the data becomes denser.
   That is the Runge phenomenon, and :func:`runge_study` measures it on the classic ``1/(1+25x^2)``.
2. **How fast does an expansion converge?** A Chebyshev series converges geometrically for an analytic
   function and algebraically (like ``n^-2``) for one with a kink or a singularity on the interval.
   :func:`chebyshev_convergence_study` measures the exponent instead of claiming "spectral".
3. **Is the fit stable?** The normal equations square the condition number; a QR solve does not.
   :func:`least_squares_conditioning` measures the digits each one costs on the same data.

The limitation, once: every result here is about *polynomial* approximation on an interval. Extrapolating
(the nodes are outside the data range) and multi-dimensional interpolation are different problems with
different constants, and neither is covered.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .convergence import fit_order
from .diagnostics import Diagnostic, explain

__all__ = [
    "CubicSpline",
    "Pchip",
    "RungeStudy",
    "barycentric_interpolate",
    "barycentric_weights",
    "chebyshev_coefficients",
    "chebyshev_convergence_study",
    "chebyshev_evaluate",
    "chebyshev_lobatto_nodes",
    "chebyshev_nodes",
    "dft",
    "equispaced_nodes",
    "gibbs_overshoot",
    "lagrange_interpolate",
    "lebesgue_constant",
    "least_squares_conditioning",
    "least_squares_normal_equations",
    "least_squares_qr",
    "l_curve",
    "newton_divided_differences",
    "newton_evaluate",
    "pade_coefficients",
    "pade_evaluate",
    "pchip",
    "runge_study",
    "tikhonov_solve",
    "trigonometric_interpolant",
    "vandermonde",
]


# --------------------------------------------------------------------------- nodes
def equispaced_nodes(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """``n + 1`` equally spaced nodes on ``[a, b]`` -- the natural choice, and the one that breaks."""
    if n < 0:
        raise ValueError(f"n must be non-negative; got {n}")
    return np.linspace(float(a), float(b), int(n) + 1)


def chebyshev_nodes(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """``n + 1`` Chebyshev points of the first kind, ``x_j = cos(pi j / n)`` mapped to ``[a, b]``.

    Their Lebesgue constant grows like ``log n`` instead of exponentially, and they are the nodes that
    make the interpolant converge for any continuous function (and the ones for which the Chebyshev
    series coefficients are computable by a cosine transform).
    """
    if n < 1:
        raise ValueError(f"n must be at least 1; got {n}")
    j = np.arange(int(n) + 1)
    return 0.5 * (a + b) + 0.5 * (b - a) * np.cos(np.pi * j / n)


def chebyshev_lobatto_nodes(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """The same ``cos(pi j/n)`` points in ascending order, i.e. the second kind (Lobatto) set.

    The two sets are the same points indexed from opposite ends; the ascending order is what an
    interpolation or quadrature routine wants, and the descending one is what the cosine transform wants.
    """
    return chebyshev_nodes(n, a, b)[::-1].copy()


# --------------------------------------------------------------------------- Lagrange / barycentric
def barycentric_weights(nodes: Sequence[float]) -> np.ndarray:
    """``w_j = 1 / prod_{k != j} (x_j - x_k)`` -- the weights of the second barycentric form.

    The formula is a product of differences, which is stable for the Chebyshev sets; for very large
    ``n`` the products underflow or overflow and a scaled version is needed, which is stated here
    because it is the one case where this function is not enough.
    """
    x = np.asarray(nodes, dtype=np.float64).ravel()
    if x.size < 2:
        raise ValueError("at least two nodes are needed")
    n = x.size
    w = np.ones(n)
    for j in range(n):
        differences = x[j] - np.delete(x, j)
        if np.any(differences == 0.0):
            raise ValueError(f"duplicate node {x[j]!r}: interpolation nodes must be distinct")
        w[j] = 1.0 / float(np.prod(differences))
    return w


def barycentric_interpolate(
    nodes: Sequence[float],
    values: Sequence[float],
    x: Sequence[float] | float,
    *,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate the interpolating polynomial through ``(nodes, values)`` at ``x``.

    ``p(x) = sum_j w_j y_j / (x - x_j) / sum_j w_j / (x - x_j)`` for ``x`` off the nodes, and ``y_j``
    exactly at a node. This is the form to use: it costs ``O(n)`` per point (against ``O(n^2)`` for the
    Lagrange sum), and its rounding error stays at the level of the Lebesgue constant instead of growing
    with the degree.
    """
    xn = np.asarray(nodes, dtype=np.float64).ravel()
    yn = np.asarray(values, dtype=np.float64).ravel()
    if xn.size != yn.size:
        raise ValueError(f"nodes and values must have the same length; got {xn.size} and {yn.size}")
    w = barycentric_weights(xn) if weights is None else np.asarray(weights, dtype=np.float64)
    xt = np.atleast_1d(np.asarray(x, dtype=np.float64))
    out = np.empty_like(xt)

    on_node = np.isin(xt, xn)
    if np.any(on_node):
        position = {float(v): k for k, v in enumerate(xn)}
        for i in np.flatnonzero(on_node):
            out[i] = yn[position[float(xt[i])]]

    off = ~on_node
    if np.any(off):
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = w[None, :] / (xt[off, None] - xn[None, :])
        numerator = terms @ yn
        denominator = terms.sum(axis=1)
        out[off] = numerator / denominator
    return out if np.ndim(x) else out[0]


def lagrange_interpolate(
    nodes: Sequence[float], values: Sequence[float], x: Sequence[float] | float
) -> np.ndarray:
    """Evaluate the interpolant through the Lagrange cardinal basis -- the textbook ``O(n^2)`` form.

    Present to be compared with :func:`barycentric_interpolate`, not to be used: it builds every cardinal
    function explicitly, so its error and its cost both grow faster.
    """
    xn = np.asarray(nodes, dtype=np.float64).ravel()
    yn = np.asarray(values, dtype=np.float64).ravel()
    xt = np.atleast_1d(np.asarray(x, dtype=np.float64))
    out = np.zeros_like(xt)
    for j in range(xn.size):
        basis = np.ones_like(xt)
        for k in range(xn.size):
            if k != j:
                basis *= (xt - xn[k]) / (xn[j] - xn[k])
        out += yn[j] * basis
    return out if np.ndim(x) else out[0]


def newton_divided_differences(
    nodes: Sequence[float], values: Sequence[float]
) -> np.ndarray:
    """Divided differences ``f[x_0, ..., x_k]`` -- the coefficients of the Newton form.

    Nesting is what makes this cheap: each new node adds one column of the difference table, so a fit
    refined with more data keeps the coefficients it already had. The Newton form is the ``O(n^2)``
    construction that the barycentric form is the ``O(n)`` evaluation of.
    """
    xn = np.asarray(nodes, dtype=np.float64).ravel()
    yn = np.asarray(values, dtype=np.float64).ravel()
    if xn.size != yn.size:
        raise ValueError(f"nodes and values must have the same length; got {xn.size} and {yn.size}")
    n = xn.size
    table = np.zeros((n, n))
    table[:, 0] = yn
    for k in range(1, n):
        table[: n - k, k] = (
            table[1 : n - k + 1, k - 1] - table[: n - k, k - 1]
        ) / (xn[k:] - xn[: n - k])
    return table[0, :]


def newton_evaluate(
    nodes: Sequence[float], coefficients: Sequence[float], x: Sequence[float] | float
) -> np.ndarray:
    """Evaluate the Newton form by Horner's nesting: ``c_0 + (x-x_0)(c_1 + (x-x_1)(...))``."""
    xn = np.asarray(nodes, dtype=np.float64).ravel()
    c = np.asarray(coefficients, dtype=np.float64).ravel()
    if xn.size != c.size:
        raise ValueError(f"nodes and coefficients must have the same length; got {xn.size} and {c.size}")
    xt = np.atleast_1d(np.asarray(x, dtype=np.float64))
    out = np.full_like(xt, c[-1])
    for k in range(c.size - 2, -1, -1):
        out = c[k] + (xt - xn[k]) * out
    return out if np.ndim(x) else out[0]


def lebesgue_constant(nodes: Sequence[float], *, samples: int = 2001) -> float:
    """``Lambda_n = max_x sum_j |L_j(x)|`` on a dense grid -- the error amplification of the node set.

    If ``p`` interpolates ``f`` on the nodes then ``||f - p||_inf <= (1 + Lambda_n) ||f - p*||_inf``:
    the constant is how much worse than the best polynomial a *given* node set can be. It depends only on
    the nodes, which is why it is the right way to compare node families without a function.
    """
    xn = np.asarray(nodes, dtype=np.float64).ravel()
    grid = np.linspace(float(np.min(xn)), float(np.max(xn)), int(samples))
    w = barycentric_weights(xn)
    with np.errstate(divide="ignore", invalid="ignore"):
        # L_j(x) = (w_j/(x-x_j)) / sum_k (w_k/(x-x_k)): the second barycentric form, term by term
        terms = w[None, :] / (grid[:, None] - xn[None, :])
        # on a node both sums are infinite, and the ratio below is 0/0 -- handled right after
        sums = np.abs(terms).sum(axis=1) / np.abs(terms.sum(axis=1))
    on_node = np.isin(grid, xn)
    sums[on_node] = 1.0  # exactly on a node one cardinal function is 1 and the rest vanish
    finite = sums[np.isfinite(sums)]
    return float(np.max(finite))


# --------------------------------------------------------------------------- Runge
@dataclass(frozen=True)
class RungeStudy:
    """Maximum interpolation error against the degree, for two node families."""

    degrees: np.ndarray
    equispaced_errors: np.ndarray
    chebyshev_errors: np.ndarray
    equispaced_lebesgue: np.ndarray
    chebyshev_lebesgue: np.ndarray
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    @property
    def equispaced_diverges(self) -> bool:
        """Whether the error grew somewhere as the degree was raised -- the Runge phenomenon."""
        return bool(np.max(np.diff(self.equispaced_errors)) > 0.0)

    @property
    def chebyshev_converges(self) -> bool:
        """Whether the Chebyshev error decreased monotonically (it should, for a function this smooth)."""
        return bool(np.all(np.diff(self.chebyshev_errors) < 0.0))

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form for the lab page."""
        return {
            "degrees": [int(d) for d in self.degrees],
            "equispaced_errors": [float(v) for v in self.equispaced_errors],
            "chebyshev_errors": [float(v) for v in self.chebyshev_errors],
            "equispaced_lebesgue": [float(v) for v in self.equispaced_lebesgue],
            "chebyshev_lebesgue": [float(v) for v in self.chebyshev_lebesgue],
            "equispaced_diverges": self.equispaced_diverges,
            "chebyshev_converges": self.chebyshev_converges,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def runge_study(
    degrees: Sequence[int] = (4, 8, 12, 16, 20, 24, 30, 40),
    *,
    samples: int = 2001,
) -> RungeStudy:
    """Interpolate ``1/(1 + 25 x^2)`` on ``[-1, 1]`` with both node families and measure the maximum error.

    This function is analytic in the complex plane except for poles at ``x = +-i/5``, which is exactly why
    equispaced interpolation diverges: the Runge constant is governed by the Bernstein ellipse through
    those poles. On Chebyshev nodes the same function converges geometrically, and the errors below show
    both facts. The Lebesgue constant is measured for each degree as the mechanism behind the divergence.
    """
    f = lambda t: 1.0 / (1.0 + 25.0 * t**2)  # noqa: E731
    grid = np.linspace(-1.0, 1.0, int(samples))
    exact = f(grid)

    rows_equi: list[float] = []
    rows_cheb: list[float] = []
    leb_equi: list[float] = []
    leb_cheb: list[float] = []
    for degree in degrees:
        for nodes, errors, lebesgue in (
            (equispaced_nodes(degree), rows_equi, leb_equi),
            (chebyshev_nodes(degree), rows_cheb, leb_cheb),
        ):
            values = f(nodes)
            errors.append(float(np.max(np.abs(barycentric_interpolate(nodes, values, grid) - exact))))
            lebesgue.append(lebesgue_constant(nodes, samples=min(2001, 20 * (degree + 1))))

    diagnostics: list[Diagnostic] = []
    study = RungeStudy(
        degrees=np.asarray(degrees),
        equispaced_errors=np.asarray(rows_equi),
        chebyshev_errors=np.asarray(rows_cheb),
        equispaced_lebesgue=np.asarray(leb_equi),
        chebyshev_lebesgue=np.asarray(leb_cheb),
    )
    if study.equispaced_diverges:
        diagnostics.append(
            explain(
                "INTERP.RUNGE",
                max_degree=int(np.max(degrees)),
                max_error=float(np.max(rows_equi)),
                lebesgue_at_max=float(leb_equi[int(np.argmax(degrees))]),
                chebyshev_error=float(rows_cheb[-1]),
            )
        )
    return RungeStudy(
        degrees=study.degrees,
        equispaced_errors=study.equispaced_errors,
        chebyshev_errors=study.chebyshev_errors,
        equispaced_lebesgue=study.equispaced_lebesgue,
        chebyshev_lebesgue=study.chebyshev_lebesgue,
        diagnostics=tuple(diagnostics),
    )


# --------------------------------------------------------------------------- splines
class CubicSpline:
    """Piecewise cubic interpolant with continuous second derivative, in three boundary flavours.

    With ``M_i = s''(x_i)`` the interpolant on ``[x_i, x_{i+1}]`` (``h_i = x_{i+1}-x_i``) is

        s(x) = M_i (x_{i+1}-x)^3/(6h_i) + M_{i+1}(x-x_i)^3/(6h_i)
               + (y_i - M_i h_i^2/6)(x_{i+1}-x)/h_i + (y_{i+1} - M_{i+1} h_i^2/6)(x-x_i)/h_i

    and the ``M_i`` solve one tridiagonal system whose rows are the continuity of the first derivative,
    closed by the boundary condition: ``"natural"`` (``M_0 = M_n = 0``), ``"clamped"`` (given end slopes)
    or ``"not-a-knot"`` (third derivative continuous across ``x_1`` and ``x_{n-1}``, which is what
    ``scipy.interpolate.CubicSpline`` defaults to and what removes the two extra conditions without
    touching the data).

    The measured convergence on a smooth function is ``O(h^4)``; the limitation is that a cubic spline
    *overshoots* near a jump even where the data is monotone (:class:`Pchip` is the answer there).
    """

    def __init__(
        self,
        nodes: Sequence[float],
        values: Sequence[float],
        *,
        kind: str = "not-a-knot",
        left_slope: float | None = None,
        right_slope: float | None = None,
    ) -> None:
        x = np.asarray(nodes, dtype=np.float64).ravel()
        y = np.asarray(values, dtype=np.float64).ravel()
        if x.size != y.size:
            raise ValueError(f"nodes and values must have the same length; got {x.size} and {y.size}")
        if x.size < 3:
            raise ValueError("a cubic spline needs at least three nodes")
        if np.any(np.diff(x) <= 0.0):
            raise ValueError("nodes must be strictly increasing")
        self.nodes = x
        self.values = y
        self.kind = kind
        self.second_derivatives = self._solve_second_derivatives(kind, left_slope, right_slope)

    def _solve_second_derivatives(
        self, kind: str, left_slope: float | None, right_slope: float | None
    ) -> np.ndarray:
        x, y = self.nodes, self.values
        n = x.size - 1
        h = np.diff(x)
        if kind not in ("natural", "clamped", "not-a-knot"):
            raise ValueError(f"kind must be 'natural', 'clamped' or 'not-a-knot'; got {kind!r}")
        if kind == "clamped" and (left_slope is None or right_slope is None):
            raise ValueError("the clamped spline needs both left_slope and right_slope")

        A = np.zeros((n + 1, n + 1))
        rhs = np.zeros(n + 1)
        for i in range(1, n):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2.0 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
            rhs[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

        if kind == "natural":
            A[0, 0] = 1.0
            A[n, n] = 1.0
        elif kind == "clamped":
            A[0, 0], A[0, 1] = 2.0, 1.0
            rhs[0] = 6.0 / h[0] * ((y[1] - y[0]) / h[0] - float(left_slope))
            A[n, n - 1], A[n, n] = 1.0, 2.0
            rhs[n] = 6.0 / h[n - 1] * (float(right_slope) - (y[n] - y[n - 1]) / h[n - 1])
        else:  # not-a-knot: continuity of the third derivative, expressed in terms of M
            A[0, 0], A[0, 1], A[0, 2] = -h[1], h[0] + h[1], -h[0]
            rhs[0] = 0.0
            A[n, n - 2], A[n, n - 1], A[n, n] = -h[n - 1], h[n - 2] + h[n - 1], -h[n - 2]
            rhs[n] = 0.0
        return np.linalg.solve(A, rhs)

    def __call__(self, x: Sequence[float] | float) -> np.ndarray:
        """Evaluate the spline, clamping inputs to the node range (no extrapolation is attempted)."""
        xt = np.atleast_1d(np.asarray(x, dtype=np.float64))
        nodes, values, m = self.nodes, self.values, self.second_derivatives
        idx = np.clip(np.searchsorted(nodes, xt) - 1, 0, nodes.size - 2)
        h = nodes[idx + 1] - nodes[idx]
        a = (nodes[idx + 1] - xt) / h
        b = (xt - nodes[idx]) / h
        result = (
            m[idx] * a**3 * h**2 / 6.0
            + m[idx + 1] * b**3 * h**2 / 6.0
            + (values[idx] - m[idx] * h**2 / 6.0) * a
            + (values[idx + 1] - m[idx + 1] * h**2 / 6.0) * b
        )
        result = np.where(np.isclose(xt, nodes[0]), values[0], result)
        result = np.where(np.isclose(xt, nodes[-1]), values[-1], result)
        return result if np.ndim(x) else result[0]


class Pchip:
    """Piecewise cubic Hermite with limited slopes: shape preserving, so no new extrema appear.

    The interior slope at ``x_i`` is the weighted harmonic mean of the two adjacent secants
    (Fritsch-Carlson), and it is set to zero whenever the secants have opposite signs -- which is exactly
    what keeps the interpolant monotone on every interval where the data is monotone. The convergence
    order drops to ``O(h^3)`` at best, and to ``O(h^2)`` where the data has an extremum: that is the price
    of not overshooting, and it is a price worth paying for tabulated physical data.
    """

    def __init__(self, nodes: Sequence[float], values: Sequence[float]) -> None:
        x = np.asarray(nodes, dtype=np.float64).ravel()
        y = np.asarray(values, dtype=np.float64).ravel()
        if x.size != y.size or x.size < 2:
            raise ValueError("PCHIP needs at least two nodes and matching values")
        if np.any(np.diff(x) <= 0.0):
            raise ValueError("nodes must be strictly increasing")
        self.nodes = x
        self.values = y
        self.slopes = self._slopes(x, y)

    @staticmethod
    def _slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        h = np.diff(x)
        d = np.diff(y) / h
        m = np.zeros_like(y)
        if y.size == 2:
            m[:] = d[0]
            return m
        for i in range(1, y.size - 1):
            if d[i - 1] * d[i] <= 0.0:
                m[i] = 0.0  # a local extremum: no slope, so no overshoot
            else:
                w1 = 2.0 * h[i] + h[i - 1]
                w2 = h[i] + 2.0 * h[i - 1]
                m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])

        def endpoint(h0: float, h1: float, d0: float, d1: float) -> float:
            slope = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
            if np.sign(slope) != np.sign(d0):
                return 0.0
            if np.sign(d0) != np.sign(d1) and abs(slope) > 3.0 * abs(d0):
                return 3.0 * d0
            return float(slope)

        m[0] = endpoint(h[0], h[1], d[0], d[1])
        m[-1] = endpoint(h[-1], h[-2], d[-1], d[-2])
        return m

    def __call__(self, x: Sequence[float] | float) -> np.ndarray:
        """Evaluate with the cubic Hermite basis on each interval."""
        xt = np.atleast_1d(np.asarray(x, dtype=np.float64))
        nodes, values, m = self.nodes, self.values, self.slopes
        idx = np.clip(np.searchsorted(nodes, xt) - 1, 0, nodes.size - 2)
        h = nodes[idx + 1] - nodes[idx]
        t = (xt - nodes[idx]) / h
        h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
        h10 = t**3 - 2.0 * t**2 + t
        h01 = -2.0 * t**3 + 3.0 * t**2
        h11 = t**3 - t**2
        result = h00 * values[idx] + h10 * h * m[idx] + h01 * values[idx + 1] + h11 * h * m[idx + 1]
        return result if np.ndim(x) else result[0]


def pchip(nodes: Sequence[float], values: Sequence[float]) -> Pchip:
    """Monotone cubic Hermite interpolant (see :class:`Pchip`)."""
    return Pchip(nodes, values)


# --------------------------------------------------------------------------- Chebyshev
def chebyshev_coefficients(f: Callable[[np.ndarray], np.ndarray], n: int) -> np.ndarray:
    """Chebyshev coefficients of ``f`` on ``[-1, 1]`` from ``n + 1`` points, by the cosine sum

        a_k = (2/n) * sum_{j=0}^{n}'' f(cos(pi j/n)) cos(k pi j/n)

    where the double prime halves the ``j = 0`` and ``j = n`` terms -- and, because ``T_0`` and ``T_n``
    themselves carry half weight in the discrete inner product, the ``k = 0`` and ``k = n`` coefficients
    are halved as well. Dropping that last halving is the classic sign error: the series then reproduces
    the function everywhere except in its mean, which is exactly what evaluating at ``x = 0`` reveals.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1; got {n}")
    j = np.arange(int(n) + 1)
    samples = np.asarray(f(np.cos(np.pi * j / n)), dtype=np.float64).ravel()
    weights = np.ones(j.size)
    weights[0] = weights[-1] = 0.5
    k = np.arange(int(n) + 1)
    cosines = np.cos(np.outer(k, j) * np.pi / n)
    coefficients = (2.0 / n) * (cosines @ (weights * samples))
    coefficients[0] *= 0.5  # the k = 0 and k = N terms of the discrete series carry half the weight
    coefficients[-1] *= 0.5
    return coefficients


def chebyshev_evaluate(coefficients: Sequence[float], x: Sequence[float] | float) -> np.ndarray:
    """Evaluate a Chebyshev series by Clenshaw's recurrence -- stable and ``O(n)`` per point.

    ``sum a_k T_k(x)`` with ``b_{k} = 2x b_{k+1} - b_{k+2} + a_k`` and the result ``x b_1 - b_2 + a_0``.
    Building ``T_k`` explicitly or evaluating ``cos(k arccos x)`` both lose accuracy near ``|x| = 1``.
    """
    c = np.asarray(coefficients, dtype=np.float64).ravel()
    xt = np.atleast_1d(np.asarray(x, dtype=np.float64))
    b_kplus1 = np.zeros_like(xt)
    b_kplus2 = np.zeros_like(xt)
    for k in range(c.size - 1, 0, -1):
        b_k = 2.0 * xt * b_kplus1 - b_kplus2 + c[k]
        b_kplus2, b_kplus1 = b_kplus1, b_k
    out = xt * b_kplus1 - b_kplus2 + c[0]
    return out if np.ndim(x) else out[0]


def chebyshev_convergence_study(
    f: Callable[[np.ndarray], np.ndarray],
    degrees: Sequence[int] = (4, 8, 16, 32, 64),
    *,
    samples: int = 4001,
) -> dict[str, Any]:
    """Measure how fast the Chebyshev series of ``f`` converges: geometric for analytic, algebraic otherwise.

    The well-posed measurement is on the **coefficients**, not on the error against the true function
    (which hits the rounding floor and then says nothing): the fitted exponent of ``|a_k|`` against ``k``
    is about ``-2`` for ``|x|`` (whose coefficients are ``-2/(pi(k^2-1))``) and steeply negative for
    ``exp``, whose coefficients ``2 I_k(1)`` decay geometrically. A geometric decay appears as a large
    negative log-log slope, which is the honest way to read "spectral convergence" off a fit.

    ``error_at_degree`` is the maximum error of the interpolant on a dense grid; it is reported for the
    degrees where it is still above the rounding floor, and the degree where it saturates is named.
    """
    grid = np.linspace(-1.0, 1.0, int(samples))
    exact = np.asarray(f(grid), dtype=np.float64)
    errors: list[float] = []
    for n in degrees:
        coefficients = chebyshev_coefficients(f, int(n))
        errors.append(float(np.max(np.abs(chebyshev_evaluate(coefficients, grid) - exact))))

    finest = chebyshev_coefficients(f, int(max(degrees)))
    orders = np.arange(finest.size, dtype=np.float64)
    magnitude = np.abs(finest)
    usable = (magnitude > 1e-14) & (orders >= 1)
    exponent = float("nan")
    if usable.sum() >= 2:
        exponent = fit_order(orders[usable], magnitude[usable], name="chebyshev coefficient").order
    tail = magnitude[magnitude > 1e-14]
    floor_index = int(np.argmax(np.asarray(errors) < 1e-13)) if errors else 0
    return {
        "degrees": [int(d) for d in degrees],
        "error_at_degree": errors,
        "coefficient_exponent": exponent,
        "coefficients": [float(v) for v in finest],
        "tail_ratio": float(tail[-2] / tail[-1]) if tail.size >= 2 else float("nan"),
        "floor_reached_at": int(degrees[floor_index]) if errors else None,
    }


# --------------------------------------------------------------------------- least squares
def vandermonde(x: Sequence[float], degree: int) -> np.ndarray:
    """The monomial design matrix ``V[i, k] = x_i^k`` -- and its condition number is the whole problem.

    On equispaced points in ``[-1, 1]`` its condition number grows like ``2^n / sqrt(n)``: degree 15
    already exceeds ``10^6``, so a normal-equations solve there has no digits left.
    """
    xv = np.asarray(x, dtype=np.float64).ravel()
    return np.vstack([xv**k for k in range(int(degree) + 1)]).T


def least_squares_qr(A: np.ndarray, b: Sequence[float]) -> np.ndarray:
    """Least squares through a QR factorisation -- the numerically stable way.

    Solving ``R c = Q^T b`` never forms ``A^T A``, so the condition number of the problem stays
    ``kappa(A)`` instead of being squared.
    """
    from .linalg import householder_qr

    Am = np.asarray(A, dtype=np.float64)
    rhs = np.asarray(b, dtype=np.float64).ravel()
    Q, R = householder_qr(Am)
    m, n = Am.shape
    return np.linalg.solve(R[:n, :], (Q.T @ rhs)[:n])


def least_squares_normal_equations(A: np.ndarray, b: Sequence[float]) -> np.ndarray:
    """The same fit through the normal equations ``A^T A c = A^T b`` -- present to be measured, not used."""
    Am = np.asarray(A, dtype=np.float64)
    rhs = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.solve(Am.T @ Am, Am.T @ rhs)


def least_squares_conditioning(
    degree: int = 12, *, samples: int = 40, high_precision: bool = True
) -> dict[str, Any]:
    """Compare QR against the normal equations on the same ill-conditioned polynomial fit.

    The data is a polynomial whose coefficients are known, sampled on equispaced points; the reference
    solution comes from a 60-digit ``decimal`` solve, so the error being measured is the algorithm's and
    not the reference's. The expected ratio between the two errors is about ``kappa(A)``, because the
    normal equations square the condition number.
    """
    from decimal import Decimal, localcontext

    x = np.linspace(-1.0, 1.0, int(samples))
    A = vandermonde(x, degree)
    true_coefficients = np.zeros(degree + 1)
    true_coefficients[0] = 1.0
    true_coefficients[1] = -0.5
    true_coefficients[min(2, degree)] = 0.25
    b = A @ true_coefficients

    kappa_A = float(np.linalg.cond(A))
    kappa_normal = kappa_A**2
    qr = least_squares_qr(A, b)
    normal = least_squares_normal_equations(A, b)

    reference = true_coefficients
    if high_precision:
        with localcontext() as ctx:
            ctx.prec = 60
            Ad = [[Decimal(float(v)) for v in row] for row in A]
            bd = [Decimal(float(v)) for v in b]
            n = A.shape[1]
            # normal equations in 60 digits: the reference is accurate enough for both algorithms
            AtA = [[sum(Ad[k][i] * Ad[k][j] for k in range(A.shape[0])) for j in range(n)] for i in range(n)]
            Atb = [sum(Ad[k][i] * bd[k] for k in range(A.shape[0])) for i in range(n)]
            for i in range(n):  # Gaussian elimination
                pivot = max(range(i, n), key=lambda r: abs(AtA[r][i]))
                AtA[i], AtA[pivot] = AtA[pivot], AtA[i]
                Atb[i], Atb[pivot] = Atb[pivot], Atb[i]
                for r in range(i + 1, n):
                    factor = AtA[r][i] / AtA[i][i]
                    for c in range(i, n):
                        AtA[r][c] -= factor * AtA[i][c]
                    Atb[r] -= factor * Atb[i]
            solution = [Decimal(0)] * n
            for i in range(n - 1, -1, -1):
                total = Atb[i] - sum(AtA[i][j] * solution[j] for j in range(i + 1, n))
                solution[i] = total / AtA[i][i]
            reference = np.array([float(v) for v in solution])

    qr_error = float(np.linalg.norm(qr - reference) / np.linalg.norm(reference))
    normal_error = float(np.linalg.norm(normal - reference) / np.linalg.norm(reference))
    return {
        "degree": int(degree),
        "condition_number": kappa_A,
        "condition_number_normal_equations": kappa_normal,
        "qr_error": qr_error,
        "normal_equations_error": normal_error,
        "error_ratio": float(normal_error / qr_error) if qr_error > 0 else float("inf"),
        "diagnostics": [
            d.to_dict()
            for d in (
                [explain("INTERP.ILL_CONDITIONED_VANDERMONDE", degree=int(degree),
                         condition_number=kappa_A, normal_equations_error=normal_error)]
                if kappa_A > 1e6
                else []
            )
        ],
    }


def tikhonov_solve(A: np.ndarray, b: Sequence[float], lam: float) -> np.ndarray:
    """Tikhonov-regularised least squares: minimise ``||A c - b||^2 + lam ||c||^2``.

    Solved as the augmented least-squares problem ``[A; sqrt(lam) I] c ~ [b; 0]`` through QR, which is
    stable for every ``lam``; forming ``A^T A + lam I`` would square the condition number again. ``lam``
    trades residual for solution norm -- the L-curve plots one against the other.
    """
    Am = np.asarray(A, dtype=np.float64)
    rhs = np.asarray(b, dtype=np.float64).ravel()
    if lam < 0:
        raise ValueError(f"lam must be non-negative; got {lam}")
    augmented = np.vstack([Am, np.sqrt(lam) * np.eye(Am.shape[1])])
    target = np.concatenate([rhs, np.zeros(Am.shape[1])])
    return least_squares_qr(augmented, target)


def l_curve(
    A: np.ndarray, b: Sequence[float], lambdas: Sequence[float] | None = None
) -> dict[str, Any]:
    """Solution norm against residual norm over a range of ``lam``: the L whose corner picks the weight.

    ``lambdas`` defaults to a logarithmic sweep over ``[1e-12, 1e2]`` scaled by the largest singular
    value squared, which is the range where the curve actually bends for a given matrix.
    """
    Am = np.asarray(A, dtype=np.float64)
    rhs = np.asarray(b, dtype=np.float64).ravel()
    if lambdas is None:
        scale = float(np.linalg.svd(Am, compute_uv=False)[0] ** 2)
        lambdas = scale * np.logspace(-12, 2, 25)
    residuals: list[float] = []
    norms: list[float] = []
    for lam in lambdas:
        coefficients = tikhonov_solve(Am, rhs, float(lam))
        residuals.append(float(np.linalg.norm(Am @ coefficients - rhs)))
        norms.append(float(np.linalg.norm(coefficients)))
    curvature: list[float] = []
    for i in range(1, len(norms) - 1):
        v1 = np.array([np.log(residuals[i]) - np.log(residuals[i - 1]),
                       np.log(norms[i]) - np.log(norms[i - 1])])
        v2 = np.array([np.log(residuals[i + 1]) - np.log(residuals[i]),
                       np.log(norms[i + 1]) - np.log(norms[i])])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 0 and n2 > 0:
            curvature.append(float(np.degrees(np.arccos(np.clip(v1 @ v2 / (n1 * n2), -1.0, 1.0)))))
        else:
            curvature.append(0.0)
    best = int(np.argmax(curvature)) + 1 if curvature else 0
    return {
        "lambdas": [float(v) for v in lambdas],
        "residuals": residuals,
        "solution_norms": norms,
        "curvature": curvature,
        "corner_index": best,
        "corner_lambda": float(np.asarray(lambdas)[best]) if len(lambdas) else float("nan"),
    }


# --------------------------------------------------------------------------- Pade
def pade_coefficients(series: Sequence[float], m: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """``[m/n]`` Pade approximant: the rational function whose Taylor series matches ``series`` to order m+n.

    With ``p(x)/q(x)`` and ``q_0 = 1``, the ``n`` conditions ``sum_{j<=n} q_j c_{k-j} = -c_k``
    (``k = m+1..m+n``) determine ``q``, and ``p_k = sum_j q_j c_{k-j}`` gives ``p``. It is the cheapest
    way to turn a truncated series into something that stays bounded where the polynomial diverges: for
    ``exp`` on the positive axis the Taylor polynomial is hard to beat, but at ``x = -5`` the ``[2/2]``
    Pade still returns a positive number of the right order of magnitude while the Taylor polynomial has
    already gone negative and wrong by three orders.

    Raises ``ValueError`` if the linear system for ``q`` is singular (the series does not determine a
    unique approximant at that order).
    """
    c = np.asarray(series, dtype=np.float64).ravel()
    if c.size < m + n + 1:
        raise ValueError(f"{m + n + 1} series coefficients are needed; got {c.size}")
    if n > 0:
        A = np.zeros((n, n))
        rhs = np.zeros(n)
        for i in range(n):
            for j in range(n):
                index = m + i - j  # q_j multiplies c_{m + (i+1) - (j+1)}
                A[i, j] = c[index] if index >= 0 else 0.0
            rhs[i] = -c[m + 1 + i]
        try:
            q_tail = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError as error:
            raise ValueError(f"the Pade system is singular for order [{m}/{n}]") from error
        q = np.concatenate([[1.0], q_tail])
    else:
        q = np.array([1.0])
    p = np.zeros(m + 1)
    for k in range(m + 1):
        p[k] = sum(q[j] * c[k - j] for j in range(min(k, n) + 1))
    return p, q


def pade_evaluate(p: Sequence[float], q: Sequence[float], x: Sequence[float] | float) -> np.ndarray:
    """Evaluate ``p(x)/q(x)`` (ascending coefficients) with Horner's rule in both polynomials."""
    pv = np.asarray(p, dtype=np.float64).ravel()
    qv = np.asarray(q, dtype=np.float64).ravel()
    xt = np.atleast_1d(np.asarray(x, dtype=np.float64))
    numerator = np.full_like(xt, pv[-1])
    for k in range(pv.size - 2, -1, -1):
        numerator = numerator * xt + pv[k]
    denominator = np.full_like(xt, qv[-1])
    for k in range(qv.size - 2, -1, -1):
        denominator = denominator * xt + qv[k]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = numerator / denominator
    return out if np.ndim(x) else out[0]


# --------------------------------------------------------------------------- Fourier
def dft(y: Sequence[float]) -> np.ndarray:
    """The discrete Fourier transform, normalised by ``1/N`` so that ``dft(1) = 1``.

    ``c_k = (1/N) sum_j y_j exp(-2 pi i j k / N)``, i.e. the coefficients of the trigonometric
    interpolant on equispaced nodes -- the same object as the Chebyshev series, with cosines replaced by
    complex exponentials. Using the FFT is the only reason this is practical. The limitation is aliasing:
    ``c_k`` and ``c_{k+N}`` are indistinguishable in the samples.
    """
    values = np.asarray(y, dtype=np.float64).ravel()
    return np.fft.fft(values) / values.size


def trigonometric_interpolant(
    values: Sequence[float], x: Sequence[float] | float
) -> np.ndarray:
    """Evaluate the trigonometric interpolant of equispaced samples at ``x`` (in cycles over the period).

    Direct evaluation of ``sum_k c_k exp(2 pi i k x)`` over the frequencies the samples can represent,
    including the Nyquist mode. ``x`` is measured in periods ``[0, 1)``; values outside are wrapped,
    because a trigonometric interpolant is periodic by construction.
    """
    coefficients = dft(values)
    n = coefficients.size
    xt = np.atleast_1d(np.asarray(x, dtype=np.float64))
    positions = np.arange(n)
    # ifft-shifted frequencies: 0..N/2 then -N/2..-1
    frequencies = np.where(positions <= n // 2, positions, positions - n)
    out = np.exp(2.0j * np.pi * np.outer(xt, frequencies)) @ coefficients
    return np.real(out) if np.ndim(x) else float(np.real(out[0]))


def aliasing_demo(true_cycles: float, n_samples: int) -> dict[str, float]:
    """What frequency a sampled cosine *appears* to have: ``|f - round(f/fs) fs|``.

    ``fs = n_samples`` cycles per period, so any frequency above the Nyquist limit ``fs/2`` is folded
    back. The function measures the peak of the DFT and compares it with the predicted alias, which is a
    check of the sampling theorem rather than of the FFT.
    """
    if n_samples < 4:
        raise ValueError("at least four samples are needed to resolve a peak")
    j = np.arange(int(n_samples))
    samples = np.cos(2.0 * np.pi * true_cycles * j / n_samples)
    spectrum = np.abs(dft(samples))
    half = spectrum.size // 2 + 1
    peak = int(np.argmax(spectrum[1:half])) + 1
    alias = abs(true_cycles - round(true_cycles / n_samples) * n_samples)
    alias = min(alias, n_samples - alias)
    return {
        "true_cycles": float(true_cycles),
        "apparent_cycles": float(peak),
        "predicted_alias": float(alias),
        "nyquist": n_samples / 2.0,
        "aliased": bool(abs(alias - peak) < 1.0 and abs(true_cycles - alias) > 1e-9),
    }


def gibbs_overshoot(n_terms: int) -> float:
    """Overshoot of the truncated Fourier series of a unit square wave, as a fraction of its *jump*.

    The partial sum overshoots by about 8.95 % of the jump -- 0.179 for a wave that jumps by 2 -- however
    many terms are kept (Wilbraham-Gibbs); only the width of the overshoot shrinks. The jump is 2 here,
    which is why the raw maximum is 1.179 and the returned value is ``(max - 1)/2``.
    """
    if n_terms < 1:
        raise ValueError(f"n_terms must be positive; got {n_terms}")
    t = np.linspace(1e-6, 0.5 - 1e-6, 200_001)
    partial = np.zeros_like(t)
    for k in range(1, int(n_terms) + 1, 2):  # odd harmonics of a square wave
        partial += 4.0 / (np.pi * k) * np.sin(2.0 * np.pi * k * t)
    return float((np.max(partial) - 1.0) / 2.0)

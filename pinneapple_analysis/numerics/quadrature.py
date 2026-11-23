"""Numerical differentiation and quadrature, with the order and the cost of each rule measured (#1059).

Differentiation
---------------
The finite-difference weights for *any* derivative order on *any* set of nodes come from Fornberg's
algorithm, so a stencil is never hardcoded and its truncation order can be measured rather than trusted.
Two facts are demonstrated here with numbers:

* the truncation/round-off trade-off is a V, and :func:`optimal_step_study` locates its vertex;
* the complex-step derivative has no subtractive cancellation at all, so it is exact to rounding at any
  small step -- the difference between ``O(h^p)`` and "no ``h`` in the error".

Quadrature
----------
Newton-Cotes (midpoint, trapezoid, Simpson), Romberg, Gauss-Legendre, Gauss-Lobatto, Gauss-Laguerre and
Gauss-Hermite (the last two for infinite intervals), Clenshaw-Curtis, adaptive Simpson, and Monte Carlo
and quasi-Monte Carlo. The Gauss nodes come from the Golub-Welsch construction -- eigenvalues of the
Jacobi matrix of the orthogonal polynomial -- not from a table of constants, so every degree is available
and the weights are exact to rounding.

The limitation, once: a composite rule's error estimate assumes a bounded derivative on the interval. At
an endpoint singularity the observed order silently falls (it is measured, and the
``QUAD.SINGULARITY`` diagnostic fires); the fix is a substitution, and
:func:`integrate_endpoint_singularity` shows exactly that.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .convergence import VCurveResult, fit_order
from .diagnostics import Diagnostic, explain

__all__ = [
    "adaptive_simpson",
    "clenshaw_curtis",
    "composite_midpoint",
    "composite_simpson",
    "composite_trapezoid",
    "complex_step_derivative",
    "fd_weights",
    "finite_difference",
    "gauss_hermite",
    "gauss_laguerre",
    "gauss_legendre",
    "gauss_legendre_nodes",
    "gauss_lobatto",
    "halton_sequence",
    "integrate_endpoint_singularity",
    "monte_carlo_integrate",
    "optimal_step_study",
    "quasi_monte_carlo_integrate",
    "romberg",
    "sampling_error_rate",
]

#: finite-difference stencils, as offsets from the evaluation point, with the derivative order and the
#: nominal truncation order of the scheme (all of them verified by measurement in the tests)
_SCHEMES: dict[str, tuple[tuple[int, ...], int, int]] = {
    "forward": ((0, 1), 1, 1),
    "backward": ((-1, 0), 1, 1),
    "central": ((-1, 0, 1), 1, 2),
    "central4": ((-2, -1, 0, 1, 2), 1, 4),
    "second-central": ((-1, 0, 1), 2, 2),
    "second-central4": ((-2, -1, 0, 1, 2), 2, 4),
    "third-central": ((-2, -1, 0, 1, 2), 3, 2),
}


# --------------------------------------------------------------------------- finite differences
def fd_weights(nodes: Sequence[float], derivative: int, x: float = 0.0) -> np.ndarray:
    """Fornberg's weights: ``f^(m)(x) ~ sum_j w_j f(x_j)`` on an arbitrary node set.

    The algorithm builds the weights by recursion on the nodes, so any ``m`` and any nodes (equispaced,
    Chebyshev, one-sided, staggered) work, and the truncation order follows from the number of nodes:
    ``n`` nodes give at least order ``n - m``. This is the routine that removes stencils from the code
    and puts them in a formula.

    Raises ``ValueError`` for duplicate nodes or a negative derivative order.
    """
    if derivative < 0:
        raise ValueError(f"derivative order must be non-negative; got {derivative}")
    xs = np.asarray(nodes, dtype=np.float64).ravel()
    if xs.size != np.unique(xs).size:
        raise ValueError("nodes must be distinct")
    n = xs.size
    m = int(derivative)
    c = np.zeros((n, m + 1))
    c[0, 0] = 1.0
    c1 = 1.0
    c4 = xs[0] - float(x)
    for i in range(1, n):
        mn = min(i, m)
        c2 = 1.0
        c5 = c4
        c4 = xs[i] - float(x)
        for j in range(i):
            c3 = xs[i] - xs[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[i, k] = (c1 * (k * c[i - 1, k - 1] - c5 * c[i - 1, k])) / c2
                c[i, 0] = -c1 * c5 * c[i - 1, 0] / c2
            for k in range(mn, 0, -1):
                c[j, k] = (c4 * c[j, k] - k * c[j, k - 1]) / c3
            c[j, 0] = c4 * c[j, 0] / c3
        c1 = c2
    return c[:, m]


def finite_difference(
    f: Callable[[np.ndarray], np.ndarray],
    x: float,
    h: float,
    *,
    derivative: int = 1,
    scheme: str = "central",
) -> float:
    """Approximate the ``derivative``-th derivative of ``f`` at ``x`` with one of the named stencils.

    ``scheme`` is one of ``forward``, ``backward``, ``central``, ``central4``, ``second-central``,
    ``second-central4`` or ``third-central``; the weights come from :func:`fd_weights` on ``x + k h``, so
    the nominal order is a property of the stencil and the measured order is checked in the tests.
    """
    if scheme not in _SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; use one of {sorted(_SCHEMES)}")
    offsets, order, _ = _SCHEMES[scheme]
    weights = fd_weights([float(k) for k in offsets], order, 0.0)
    values = np.asarray(f(float(x) + h * np.asarray(offsets, dtype=np.float64)), dtype=np.float64)
    return float(weights @ values / h**order)


def measure_differentiation_order(
    f: Callable[[np.ndarray], np.ndarray],
    exact: Callable[[float], float],
    x: float,
    steps: Sequence[float],
    *,
    scheme: str = "central",
) -> dict[str, Any]:
    """Measure the observed order of a stencil on ``f`` against its closed-form derivative.

    ``steps`` must run from coarse to fine. Only the steps on the truncation side of the V are fitted --
    past its minimum the error grows again with refinement, and including those points would average a
    ``+p`` slope with a ``-1`` one and report an order that no analysis predicts.
    """
    errors = [abs(finite_difference(f, x, float(h), scheme=scheme) - exact(x)) for h in steps]
    if all(e == 0.0 for e in errors):
        raise ValueError("every error is exactly zero: the order cannot be measured from this data")
    # steps must be given coarse-to-fine. Two exclusions keep the fit on the truncation branch: past the
    # V minimum the error grows again, and within a factor of ~20 of that minimum the round-off term is
    # already comparable with the truncation term, so neither is measured there.
    truncation = errors[: int(np.argmin(errors)) + 1]
    minimum = float(np.min(truncation))
    threshold = 20.0 * minimum if minimum > 0.0 else 0.0
    usable = [i for i, e in enumerate(truncation) if e > threshold]
    if len(usable) < 2:
        usable = [i for i, e in enumerate(truncation) if e > 0.0]
    if len(usable) < 2:
        raise ValueError("fewer than two usable steps above the round-off floor: no order to measure")
    finest = usable[-4:] if len(usable) >= 4 else usable
    estimate = fit_order(
        np.asarray(steps, dtype=np.float64)[finest], np.asarray(errors, dtype=np.float64)[finest],
        name=f"{scheme} error",
    )
    return {
        "scheme": scheme,
        "steps": [float(h) for h in steps],
        "errors": errors,
        "order": estimate.order,
        "ci95": list(estimate.ci95),
        "r_squared": estimate.r_squared,
    }


def optimal_step_study(
    f: Callable[[np.ndarray], np.ndarray],
    exact: Callable[[float], float],
    x: float,
    *,
    scheme: str = "central",
    steps: Sequence[float] | None = None,
) -> VCurveResult:
    """The V-shaped error curve of one stencil: locate the step where truncation and round-off balance.

    For a stencil of order ``p`` the truncation error is ``C h^p`` and the round-off is ``eps |f| / h``
    (each function value carries ``eps |f|``, and the difference divides by ``h``), so the total is
    minimised at ``h* ~ (eps |f| / C)^(1/(p+1))``. Both branches and the vertex are measured by
    :func:`pinneapple_analysis.numerics.convergence.truncation_roundoff_balance`.
    """
    from .convergence import truncation_roundoff_balance

    grid = steps if steps is not None else np.logspace(-12, 0, 70)
    scale = max(abs(float(exact(x))), abs(float(f(np.asarray([x], dtype=np.float64))[0])), 1e-30)
    return truncation_roundoff_balance(
        grid,
        lambda h: finite_difference(f, x, float(h), scheme=scheme) - exact(x),
        scale=scale,
    )


def complex_step_derivative(
    f: Callable[[np.ndarray], complex], x: float, h: float = 1e-20
) -> float:
    """``Im f(x + i h) / h``: the derivative without any subtraction, hence without cancellation.

    Since ``f(x + ih) = f(x) + i h f'(x) - h^2 f''(x)/2 + ...``, dividing the imaginary part by ``h``
    leaves an ``O(h^2)`` truncation error and **no** subtractive cancellation: the real part
    ``f(x)`` is never subtracted from anything. Taking ``h = 1e-20`` therefore gives a derivative accurate
    to the last bit, which no real-step difference can. The limitation is that ``f`` must be analytic in
    a complex neighbourhood of ``x`` and must accept complex input.
    """
    value = complex(f(np.asarray([x + 1j * h], dtype=np.complex128))[0])
    return float(value.imag / h)


# --------------------------------------------------------------------------- Newton-Cotes
def composite_trapezoid(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    """Composite trapezoid rule: order 2, and the rule a spectral method is compared against."""
    if n < 1:
        raise ValueError(f"n must be at least 1; got {n}")
    x = np.linspace(float(a), float(b), int(n) + 1)
    weights = np.ones(x.size)
    weights[0] = weights[-1] = 0.5
    return float((float(b) - float(a)) / n * (weights @ np.asarray(f(x), dtype=np.float64)))


def composite_midpoint(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    """Composite midpoint rule: order 2, with half the constant of the trapezoid for smooth integrands."""
    if n < 1:
        raise ValueError(f"n must be at least 1; got {n}")
    h = (float(b) - float(a)) / n
    x = float(a) + h * (np.arange(int(n)) + 0.5)
    return float(h * np.sum(np.asarray(f(x), dtype=np.float64)))


def composite_simpson(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    """Composite Simpson rule: order 4 when ``n`` is even (it is rounded up to an even number here).

    ``(h/3)[f_0 + 4f_1 + 2f_2 + 4f_3 + ... + f_n]``. The order is 4 for a smooth integrand and 2 as soon
    as the fourth derivative is unbounded -- the singularity diagnostic exists because of that.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1; got {n}")
    n = int(n) + (int(n) % 2)
    x = np.linspace(float(a), float(b), n + 1)
    weights = np.ones(n + 1)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    h = (float(b) - float(a)) / n
    return float(h / 3.0 * (weights @ np.asarray(f(x), dtype=np.float64)))


def romberg(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, levels: int = 6) -> dict[str, Any]:
    """Romberg integration: Richardson extrapolation on the trapezoid rule, applied repeatedly.

    ``R[k, 0]`` is the trapezoid rule with ``2^k`` intervals and each extra column removes one power of
    ``h^2``, so ``R[k, k]`` is a ``2k``-order rule built from reused function values. The table is returned
    because the diagonal's convergence is the interesting part: the error falls like ``h^{2k}``.
    """
    if levels < 1:
        raise ValueError(f"levels must be at least 1; got {levels}")
    R = np.zeros((levels, levels))
    h = float(b) - float(a)
    R[0, 0] = 0.5 * h * (float(f(np.asarray([a], dtype=np.float64))[0])
                         + float(f(np.asarray([b], dtype=np.float64))[0]))
    for k in range(1, levels):
        n = 2 ** (k - 1)
        x = float(a) + (np.arange(n) + 0.5) * (h / n)
        R[k, 0] = 0.5 * R[k - 1, 0] + 0.5 * (h / n) * float(np.sum(np.asarray(f(x), dtype=np.float64)))
        for j in range(1, k + 1):
            R[k, j] = R[k, j - 1] + (R[k, j - 1] - R[k - 1, j - 1]) / (4.0**j - 1.0)
    diagonal = np.array([R[k, k] for k in range(levels)])
    return {
        "value": float(R[levels - 1, levels - 1]),
        "table": R,
        "diagonal": diagonal,
        "levels": levels,
    }


# --------------------------------------------------------------------------- Gaussian rules
def _golub_welsch(diagonal: np.ndarray, off_diagonal: np.ndarray, mu0: float) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights from the Jacobi matrix of the orthogonal polynomials (Golub-Welsch).

    The nodes are the eigenvalues of the symmetric tridiagonal matrix of the three-term recurrence and the
    weights are ``mu0`` times the squared first components of its eigenvectors. No constants are copied
    from a table, every degree is available, and the accuracy is that of ``eigh`` -- the reference values
    in ``scipy.special`` agree to the last few bits, which is what the tests check.
    """
    n = diagonal.size
    J = np.diag(diagonal)
    if n > 1:
        J += np.diag(np.sqrt(off_diagonal[1:]), 1) + np.diag(np.sqrt(off_diagonal[1:]), -1)
    values, vectors = np.linalg.eigh(J)
    return values, mu0 * vectors[0, :] ** 2


def gauss_legendre_nodes(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights of the ``n``-point Gauss-Legendre rule on ``[-1, 1]`` (exact to degree ``2n-1``)."""
    if n < 1:
        raise ValueError(f"n must be at least 1; got {n}")
    k = np.arange(n, dtype=np.float64)
    diagonal = np.zeros(n)
    off_diagonal = np.ones(n)
    if n > 1:
        j = np.arange(1, n, dtype=np.float64)
        off_diagonal[1:] = j**2 / (4.0 * j**2 - 1.0)
    return _golub_welsch(diagonal, off_diagonal, 2.0)


def gauss_legendre(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    """``n``-point Gauss-Legendre quadrature on ``[a, b]``.

    Converges geometrically for an analytic integrand and exactly for polynomials up to degree ``2n-1``;
    the cost is ``n`` evaluations with no subdivision, which is what makes it the rule of choice for smooth
    integrands over a finite interval.
    """
    nodes, weights = gauss_legendre_nodes(n)
    half = 0.5 * (float(b) - float(a))
    mid = 0.5 * (float(a) + float(b))
    return float(half * np.sum(weights * np.asarray(f(mid + half * nodes), dtype=np.float64)))


def gauss_lobatto(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    """``n``-point Gauss-Lobatto rule (``n >= 2``): the endpoints are nodes, exact to degree ``2n-3``.

    The interior nodes are the roots of ``P'_{n-1}``, equivalently the Gauss-Jacobi(1,1) nodes, and the
    weights are ``2 / (n (n-1) P_{n-1}(x_i)^2)``. Including the endpoints costs one order of exactness
    compared with Gauss-Legendre, and buys the property every collocation method needs: the integrand is
    evaluated where the boundary conditions live.
    """
    if n < 2:
        raise ValueError(f"a Lobatto rule needs at least 2 points; got {n}")
    if n == 2:
        nodes = np.array([-1.0, 1.0])
    else:
        # the n-2 interior nodes are the Gauss-Jacobi(1,1) nodes: b_k = k(k+2)/((2k+1)(2k+3)), mu0 = 4/3
        interior_count = n - 2
        diagonal = np.zeros(interior_count)
        off_diagonal = np.ones(interior_count)
        k = np.arange(1, interior_count, dtype=np.float64)
        if interior_count > 1:
            off_diagonal[1:] = k * (k + 2.0) / ((2.0 * k + 1.0) * (2.0 * k + 3.0))
        interior, _ = _golub_welsch(diagonal, off_diagonal, 4.0 / 3.0)
        nodes = np.concatenate(([-1.0], interior, [1.0]))

    weights = np.empty(n)
    for i, node in enumerate(nodes):
        # P_{n-1}(x_i) by the three-term recurrence, then w_i = 2 / (n (n-1) P_{n-1}(x_i)^2)
        p_prev, p_curr = 1.0, float(node)
        for k in range(2, n):
            p_prev, p_curr = p_curr, ((2.0 * k - 1.0) * node * p_curr - (k - 1.0) * p_prev) / k
        legendre = 1.0 if n - 1 == 0 else p_curr
        weights[i] = 2.0 / (n * (n - 1) * legendre**2)
    half = 0.5 * (float(b) - float(a))
    mid = 0.5 * (float(a) + float(b))
    return float(half * np.sum(weights * np.asarray(f(mid + half * nodes), dtype=np.float64)))


def gauss_laguerre(f: Callable[[np.ndarray], np.ndarray], n: int) -> float:
    """``n``-point Gauss-Laguerre: ``int_0^inf exp(-x) g(x) dx``, for a decaying integrand on a half line.

    Infinite intervals are handled by the weight, not by truncation, which is why the rule keeps its
    ``2n-1`` polynomial exactness. The limitation is the ``exp(-x)`` factor: an integrand that decays
    more slowly (or not at all) must be transformed first, and one that decays faster wastes nodes.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1; got {n}")
    k = np.arange(n, dtype=np.float64)
    diagonal = 2.0 * k + 1.0
    off_diagonal = np.ones(n)
    if n > 1:
        off_diagonal[1:] = k[1:] ** 2
    nodes, weights = _golub_welsch(diagonal, off_diagonal, 1.0)
    return float(np.sum(weights * np.asarray(f(nodes), dtype=np.float64)))


def gauss_hermite(f: Callable[[np.ndarray], np.ndarray], n: int) -> float:
    """``n``-point Gauss-Hermite: ``int_{-inf}^{inf} exp(-x^2) g(x) dx``, the Gaussian-weighted integral.

    Uses the physicists' Hermite polynomials, whose recurrence gives ``b_k = k/2`` and ``mu0 = sqrt(pi)``.
    The nodes are the roots of ``H_n`` and lie symmetrically about zero, so an odd integrand is integrated
    to rounding rather than to truncation -- a useful self-check.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1; got {n}")
    k = np.arange(n, dtype=np.float64)
    diagonal = np.zeros(n)
    off_diagonal = np.ones(n)
    if n > 1:
        off_diagonal[1:] = k[1:] / 2.0
    nodes, weights = _golub_welsch(diagonal, off_diagonal, math.sqrt(math.pi))
    return float(np.sum(weights * np.asarray(f(nodes), dtype=np.float64)))


def clenshaw_curtis(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    """Clenshaw-Curtis: integrate the Chebyshev interpolant on ``n+1`` Lobatto points, O(n log n).

    Built by integrating the interpolant exactly, ``int T_k = 2/(1-k^2)`` for even ``k`` and 0 for odd, so
    the weights are a transform of the function values. Converges geometrically for analytic integrands,
    like Gauss-Legendre, with cheaper weights and nodes that are reusable across levels -- which is why it
    is the rule inside nested quadrature and spectral codes.
    """
    from .interpolation import chebyshev_coefficients

    if n < 2:
        raise ValueError(f"Clenshaw-Curtis needs at least 2 points (n >= 2); got {n}")
    n = int(n) + (int(n) % 2)
    g = lambda t: f(0.5 * (float(a) + float(b)) + 0.5 * (float(b) - float(a)) * t)  # noqa: E731
    coefficients = chebyshev_coefficients(g, n)
    integral = 0.0
    for k in range(0, n + 1, 2):
        factor = 2.0 / (1.0 - k**2) if k != 1 else 0.0
        integral += coefficients[k] * factor
    return float(0.5 * (float(b) - float(a)) * integral)


# --------------------------------------------------------------------------- adaptive rules
def adaptive_simpson(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    *,
    tol: float = 1e-10,
    max_depth: int = 40,
) -> dict[str, Any]:
    """Adaptive Simpson with a local error estimate from the difference between one and two halvings.

    The estimate ``|S(h) - S(h/2)| / 15`` is the classic fifth-order Richardson estimate; it is returned
    alongside the value so a caller can compare it with the true error (which the tests do). The recursion
    stops when the estimate is below ``tol`` on a subinterval or the depth limit is reached, so an
    integrand with a singularity returns the best value found and a flag, not an exception.
    """
    evaluations = 0

    def evaluate(x: np.ndarray) -> np.ndarray:
        nonlocal evaluations
        evaluations += int(np.size(x))
        return np.asarray(f(x), dtype=np.float64)

    fa, fm, fb = evaluate(np.array([float(a)]))[0], evaluate(np.array([0.5 * (a + b)]))[0], evaluate(
        np.array([float(b)])
    )[0]
    whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb)

    def recurse(left: float, right: float, f_left: float, f_mid: float, f_right: float,
                estimate: float, tolerance: float, depth: int) -> tuple[float, float]:
        mid = 0.5 * (left + right)
        left_mid = 0.5 * (left + mid)
        right_mid = 0.5 * (mid + right)
        f_left_mid, f_right_mid = evaluate(np.array([left_mid, right_mid]))
        left_rule = (mid - left) / 6.0 * (f_left + 4.0 * f_left_mid + f_mid)
        right_rule = (right - mid) / 6.0 * (f_mid + 4.0 * f_right_mid + f_right)
        refined = left_rule + right_rule
        error = abs(refined - estimate) / 15.0
        if depth <= 0 or error <= tolerance:
            return refined + (refined - estimate) / 15.0, error
        left_value, left_error = recurse(left, mid, f_left, f_left_mid, f_mid, left_rule,
                                         tolerance / 2.0, depth - 1)
        right_value, right_error = recurse(mid, right, f_mid, f_right_mid, f_right, right_rule,
                                           tolerance / 2.0, depth - 1)
        return left_value + right_value, left_error + right_error

    value, error_estimate = recurse(float(a), float(b), fa, fm, fb, whole, tol, max_depth)
    diagnostics: list[Diagnostic] = []
    if error_estimate > tol:
        diagnostics.append(explain("QUAD.ERROR_ESTIMATE_OPTIMISTIC", estimated_error=error_estimate,
                                   tolerance=tol))
    return {
        "value": float(value),
        "error_estimate": float(error_estimate),
        "evaluations": int(evaluations),
        "diagnostics": [d.to_dict() for d in diagnostics],
    }


# --------------------------------------------------------------------------- singular integrands
def integrate_endpoint_singularity(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    *,
    exponent: float = 0.5,
    n: int = 200,
) -> float:
    """Integrate ``int_a^b (x-a)^(-exponent) g(x) dx`` by substituting ``x = a + t^(1/(1-exponent))``.

    The substitution removes the singularity from the integrand and puts the weight into the Jacobian:
    with ``x - a = t^q`` and ``q = 1/(1-exponent)``, ``dx = q t^(q-1) dt``, so
    ``(x-a)^(-exponent) g = t^(q-1-exponent q) g = t^(q-1) g`` -- regular. Composed with the trapezoid rule
    the observed order returns to the rule's nominal order, which is the measurement this function exists
    to make possible.

    Raises ``ValueError`` for an exponent outside ``(0, 1)`` (no substitution is valid there).
    """
    if not 0.0 < exponent < 1.0:
        raise ValueError(f"exponent must be in (0, 1); got {exponent}")
    q = 1.0 / (1.0 - exponent)
    nodes = np.linspace(0.0, 1.0, int(n) + 1)

    def transformed(t: np.ndarray) -> np.ndarray:
        # x - a = (b-a) t^q and (x-a)^(-exponent) dx = q (b-a) t^(q(1-exponent)-1) dt = q (b-a) dt:
        # the singular power cancels exactly, which is the whole point of the substitution
        x = a + (b - a) * t**q
        return q * (b - a) * np.asarray(f(x), dtype=np.float64)

    weights = np.ones(nodes.size)
    weights[0] = weights[-1] = 0.5
    return float(1.0 / int(n) * (weights @ transformed(nodes)))


# --------------------------------------------------------------------------- sampling methods
def monte_carlo_integrate(
    f: Callable[[np.ndarray], np.ndarray],
    a: Sequence[float],
    b: Sequence[float],
    n: int,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Plain Monte Carlo on a box: the mean of the samples times the volume, with its own standard error.

    The error falls like ``N^{-1/2}`` in **any** dimension, which is the trade it makes against the
    ``N^{-4/d}`` of a product Gauss rule: better above ``d = 8``, hopeless below it. The returned standard
    error is the sample estimate ``s / sqrt(N)`` -- an estimate, not a bound.
    """
    lo = np.atleast_1d(np.asarray(a, dtype=np.float64))
    hi = np.atleast_1d(np.asarray(b, dtype=np.float64))
    if lo.shape != hi.shape:
        raise ValueError("a and b must have the same length")
    if n < 2:
        raise ValueError(f"at least two samples are needed; got {n}")
    rng = np.random.default_rng(seed)
    samples = rng.uniform(lo, hi, size=(int(n), lo.size))
    values = np.asarray(f(samples), dtype=np.float64).ravel()
    volume = float(np.prod(hi - lo))
    mean = float(np.mean(values))
    return {
        "value": volume * mean,
        "standard_error": float(volume * np.std(values, ddof=1) / np.sqrt(n)),
        "samples": int(n),
        "dimension": int(lo.size),
    }


def halton_sequence(n: int, dimension: int, *, start: int = 1) -> np.ndarray:
    """The Halton low-discrepancy sequence: each coordinate uses the van der Corput sequence in a prime base.

    Deterministic but not random, and *more evenly spread* than pseudorandom points: for a smooth integrand
    its error is closer to ``N^-1`` in low dimension. It degrades as the dimension grows, because the
    bases of the later coordinates are large and their cycles long -- which the measured exponents show.
    """
    if n < 1 or dimension < 1:
        raise ValueError("n and dimension must be positive")
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89]
    if dimension > len(primes):
        raise ValueError(f"at most {len(primes)} dimensions are supported; got {dimension}")
    points = np.empty((int(n), int(dimension)))
    for d in range(int(dimension)):
        base = primes[d]
        indices = np.arange(int(start), int(start) + int(n))
        values = np.zeros(int(n))
        factor = 1.0 / base
        while np.any(indices > 0):
            values += factor * (indices % base)
            indices //= base
            factor /= base
        points[:, d] = values
    return points


def quasi_monte_carlo_integrate(
    f: Callable[[np.ndarray], np.ndarray],
    a: Sequence[float],
    b: Sequence[float],
    n: int,
) -> dict[str, Any]:
    """Quasi-Monte Carlo on a box using the Halton sequence (deterministic, no standard error)."""
    lo = np.atleast_1d(np.asarray(a, dtype=np.float64))
    hi = np.atleast_1d(np.asarray(b, dtype=np.float64))
    if lo.shape != hi.shape:
        raise ValueError("a and b must have the same length")
    points = lo + halton_sequence(int(n), lo.size) * (hi - lo)
    values = np.asarray(f(points), dtype=np.float64).ravel()
    volume = float(np.prod(hi - lo))
    return {
        "value": volume * float(np.mean(values)),
        "samples": int(n),
        "dimension": int(lo.size),
    }


def sampling_error_rate(
    dimensions: Sequence[int] = (1, 2, 4, 8),
    sizes: Sequence[int] = (1_000, 10_000, 100_000),
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Measure the error rate of Monte Carlo against Halton as the dimension grows.

    The test integrand is the tensor product ``prod_i |4 x_i - 2|`` over ``[0, 1]^d``, whose exact integral
    is 1 in every dimension -- a known answer and a bounded variance, so what is being measured is the
    sampling method, not the integrand. Monte Carlo's exponent should sit near ``-0.5`` regardless of
    ``d``; Halton's should be clearly better in one dimension and approach Monte Carlo as ``d`` grows.
    """
    exact = 1.0

    def integrand(points: np.ndarray) -> np.ndarray:
        return np.prod(np.abs(4.0 * np.asarray(points, dtype=np.float64) - 2.0), axis=1)

    monte_carlo: dict[int, list[float]] = {}
    quasi: dict[int, list[float]] = {}
    for dimension in dimensions:
        lo = np.zeros(int(dimension))
        hi = np.ones(int(dimension))
        mc_errors: list[float] = []
        qmc_errors: list[float] = []
        for n in sizes:
            mc = monte_carlo_integrate(integrand, lo, hi, int(n), seed=seed)
            qmc = quasi_monte_carlo_integrate(integrand, lo, hi, int(n))
            mc_errors.append(abs(mc["value"] - exact))
            qmc_errors.append(abs(qmc["value"] - exact))
        monte_carlo[int(dimension)] = mc_errors
        quasi[int(dimension)] = qmc_errors

    sizes_array = np.asarray(sizes, dtype=np.float64)

    def exponent(errors: list[float]) -> float:
        positive = np.asarray(errors) > 0
        if positive.sum() < 2:
            return float("nan")
        return fit_order(sizes_array[positive], np.asarray(errors)[positive], name="sampling error").order

    return {
        "dimensions": [int(d) for d in dimensions],
        "sizes": [int(n) for n in sizes],
        "monte_carlo_errors": {str(d): v for d, v in monte_carlo.items()},
        "quasi_monte_carlo_errors": {str(d): v for d, v in quasi.items()},
        "monte_carlo_exponents": {str(d): exponent(v) for d, v in monte_carlo.items()},
        "quasi_monte_carlo_exponents": {str(d): exponent(v) for d, v in quasi.items()},
    }


# --------------------------------------------------------------------------- errors against cost
@dataclass(frozen=True)
class QuadratureComparison:
    """Error against the number of function evaluations, for several rules on the same integrand."""

    rules: dict[str, dict[str, Any]]
    dominant: str
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form for the lab page."""
        return {
            "rules": {name: dict(payload) for name, payload in self.rules.items()},
            "dominant": self.dominant,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def compare_rules_at_equal_cost(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    exact: float,
    *,
    budgets: Sequence[int] = (8, 16, 32, 64, 128),
) -> QuadratureComparison:
    """Measure error against evaluation count for trapezoid, Simpson, Gauss-Legendre and Clenshaw-Curtis.

    "Cost" is the number of function evaluations, which is the only fair axis: it is what the answer costs
    the caller. On a smooth integrand Gauss-Legendre should dominate; the point of running the comparison
    is to see how much, and to see Simpson lose its fourth order once the budget is small.
    """
    payload: dict[str, dict[str, Any]] = {}
    for name in ("trapezoid", "simpson", "gauss_legendre", "clenshaw_curtis"):
        errors: list[float] = []
        costs: list[int] = []
        for budget in budgets:
            if name == "trapezoid":
                value = composite_trapezoid(f, a, b, int(budget))
                cost = int(budget) + 1
            elif name == "simpson":
                value = composite_simpson(f, a, b, int(budget))
                cost = int(budget) + 1 + (int(budget) % 2)
            elif name == "gauss_legendre":
                value = gauss_legendre(f, a, b, max(1, int(budget)))
                cost = max(1, int(budget))
            else:
                value = clenshaw_curtis(f, a, b, max(2, int(budget)))
                cost = max(2, int(budget)) + (max(2, int(budget)) % 2) + 1
            errors.append(abs(value - exact))
            costs.append(int(cost))
        # a rule that reaches the rounding floor has no measurable exponent left: fit only above it
        usable = np.asarray(errors) > 1e-14
        order = float("nan")
        if usable.sum() >= 2:
            order = fit_order(np.asarray(costs, dtype=np.float64)[usable],
                              np.asarray(errors)[usable], name=name).order
        payload[name] = {"errors": errors, "costs": costs, "error_exponent": order}

    dominant = min(payload, key=lambda name: payload[name]["errors"][-1])
    diagnostics: list[Diagnostic] = []
    if payload["gauss_legendre"]["errors"][-1] > payload["trapezoid"]["errors"][-1]:
        diagnostics.append(
            explain("QUAD.ERROR_ESTIMATE_OPTIMISTIC", note="Gauss-Legendre losing to the trapezoid rule",
                    error=payload["gauss_legendre"]["errors"][-1])
        )
    return QuadratureComparison(rules=payload, dominant=dominant, diagnostics=tuple(diagnostics))

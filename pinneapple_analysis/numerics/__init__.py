"""Numerical methods as a verified, self-explaining toolkit (#1048).

The library uses numerical methods everywhere -- RK4, Gauss-Legendre collocation, ETDRK4, stiff kinetics,
quadrature inside ``immersed``, FEM and convergence studies in ``pinneapple_analysis.verification`` -- but
it had no single place where a user could *pick* a method, see its order, stability and cost, and measure
those on their own problem. This package is that place.

Layout
------
``diagnostics``
    The explanation layer. Every failure in every module returns a :class:`~.diagnostics.Diagnostic` with
    a stable code, a severity, what happened mathematically and what to change. It is the text the Studio
    page and the Flow node show.
``convergence``
    The measurement layer: ``error ~ C h^p`` fit in log-log with a confidence interval, the V-curve of
    truncation against round-off with its measured optimum, and the order of an iteration from its
    iterates. Every other module reports through these.
``floating_point``
    Machine epsilon and range per dtype, catastrophic cancellation (quadratic formula, ``log1p``/``expm1``,
    Welford against the naive one-pass variance, compensated summation) and the condition-number /
    backward-error verdict that says whether the problem or the algorithm took the digits.
``roots``
    Bisection, regula falsi (plain and Illinois), secant, Newton, Halley and Brent for one equation;
    Newton with a line search, Broyden and Jacobian-free Newton-Krylov for systems; natural-parameter and
    pseudo-arclength continuation through a fold; companion-matrix roots and Wilkinson's sensitivity.
``interpolation``
    Lagrange, barycentric and Newton divided differences; the Runge phenomenon and the Lebesgue constant
    measured; cubic splines (natural, clamped, not-a-knot) and PCHIP; Chebyshev approximation and its
    coefficient decay; least squares by QR against the normal equations; Pade; the trigonometric
    interpolant, aliasing and Gibbs.
``quadrature``
    Finite-difference weights for any order on any nodes, the V-curve of the optimal step, Richardson
    extrapolation and the complex-step derivative; rectangle/trapezoid/Simpson, Romberg, Gauss-Legendre,
    Gauss-Lobatto, Gauss-Laguerre, Gauss-Hermite, Clenshaw-Curtis, adaptive Simpson; Monte Carlo and
    quasi-Monte Carlo with the error rate against dimension.
``ode``
    Euler, Heun, RK4, embedded pairs with step control, Adams-Bashforth/Moulton; backward Euler, the
    trapezoid rule, BDF and Radau IIA; sampled stability regions, A/L stability, stiffness benchmarks
    (Robertson, Van der Pol), symplectic integrators with the measured energy drift, events, and an
    index-1 DAE.
``linalg``
    LU with pivoting and the growth factor, Cholesky, Householder QR against Gram-Schmidt, conditioning
    against the measured error, Jacobi/Gauss-Seidel/SOR with the spectral radius predicting the rate, CG
    against ``sqrt(kappa)``, GMRES, BiCGStab, ILU(0), a multigrid V-cycle measured to be h-independent,
    power/inverse/Rayleigh/QR/Lanczos eigensolvers, and the SVD with Eckart-Young checked.
``pde``
    Heat, advection and wave equations in finite differences: truncation error, von Neumann amplification,
    the CFL limit with the blow-up just above it, upwind/Lax-Friedrichs/Lax-Wendroff with their measured
    numerical diffusion and dispersion, TVD limiters, and manufactured solutions reporting the observed
    order.
``regimes``
    Which regime a problem is in and which strategy answers it (#207): the stiffness ratio and the Jacobian
    spectrum for an ODE, the gradient concentration, layer thickness, location, jump, turning points and
    two-band spectrum for a field, and a rule table that turns the numbers into ``stiff``, ``oscillatory``,
    ``boundary_layer``, ``shock``, ``high_frequency`` or ``multiscale`` with the remedies attached.
Not here yet
------------
The app-facing glue (#29): a catalogue of (problem, method) pairs and a single call that returns the
error-against-step and error-against-cost curves, the observed order with its interval and the
diagnostics as one JSON-able record. Everything it needs is in these modules -- ``convergence`` measures,
``diagnostics`` explains, each module exposes the curves -- but the catalogue itself is not written, so
nothing here claims to be that entry point.

Design rules this package follows
---------------------------------
* **Pure numpy** in the numerical code (scipy is used only in tests, as an independent reference); no
  torch, no plot calls, no I/O.
* **Nothing is assumed.** An order, a stability limit or a convergence rate is *measured* on a problem
  with a known solution and returned with its uncertainty; when a method does not converge, the result
  says so instead of returning the last iterate as if it were the answer.
* **Failures are data.** Every result carries the diagnostics that explain it, and results are dataclasses
  with ``to_dict()`` so the interface serialises them without re-deriving anything.

Reuse, not duplication: ``pinneapple_simulation.numerical_solvers`` keeps the production solvers
(``irk_gauss_legendre``, ``etdrk4``, ``stiff_kinetics_ode``, ``tvd_advection``, ``symbolic_fd``) and
``pinneapple_analysis.verification.convergence`` keeps Roache's GCI; this package provides the modular,
measurable primitives those are built from and the explanation of when they fail.

References for the claims made in the docstrings
------------------------------------------------
* Butcher, J.C., *Numerical Methods for Ordinary Differential Equations*, 3rd ed., Wiley, 2016.
* Trefethen, L.N. and Bau, D., *Numerical Linear Algebra*, SIAM, 1997; Trefethen, *Approximation Theory
  and Approximation Practice*, SIAM, 2013.
* Higham, N.J., *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002.
* Hairer, E. and Wanner, G., *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic
  Problems*, 2nd ed., Springer, 1996.
* LeVeque, R.J., *Finite Difference Methods for Ordinary and Partial Differential Equations*, SIAM, 2007.
* Roache, P.J., *Verification and Validation in Computational Science and Engineering*, Hermosa, 1998.
"""
from __future__ import annotations

from . import (
    convergence,
    diagnostics,
    floating_point,
    interpolation,
    linalg,
    ode,
    pde,
    quadrature,
    regimes,
    roots,
)
from .convergence import (
    ErrorCurve,
    OrderEstimate,
    VCurveResult,
    fit_order,
    order_from_iterates,
    richardson_extrapolate,
    truncation_roundoff_balance,
)
from .diagnostics import Diagnostic, Severity, catalogue, explain

__all__ = [
    "Diagnostic",
    "ErrorCurve",
    "OrderEstimate",
    "Severity",
    "VCurveResult",
    "catalogue",
    "convergence",
    "diagnostics",
    "explain",
    "fit_order",
    "floating_point",
    "interpolation",
    "linalg",
    "ode",
    "order_from_iterates",
    "pde",
    "quadrature",
    "regimes",
    "richardson_extrapolate",
    "roots",
    "truncation_roundoff_balance",
]

"""Why a numerical method failed, in a form an interface can show.

Every module of :mod:`pinneapple_analysis.numerics` that can fail returns a list of :class:`Diagnostic`
instead of only a flag: a stable ``code`` to key on, a severity, one sentence of *what happened*, one
of *why*, and one of *what to do*. The Studio page of the numerical lab and the Flow node render these
directly; a CLI prints :meth:`Diagnostic.format`.

    >>> from pinneapple_analysis.numerics.diagnostics import explain
    >>> d = explain("ODE.STIFF", ratio=1.2e6)
    >>> d.severity, d.code
    ('warning', 'ODE.STIFF')

The catalogue is data, not behaviour: a diagnostic never decides anything. The method that produced it
decides what to return; the diagnostic only says what a person reading the result needs to know. Adding
a code means adding one row to ``_CATALOGUE`` -- nothing else in the package changes.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = ["CATALOGUE", "Diagnostic", "Severity", "catalogue", "explain", "has_code", "worst"]

Severity = Literal["info", "warning", "error"]

#: code -> (severity, title, explanation, remedy). Explanation is what went wrong mathematically;
#: remedy is the concrete change that fixes it.
_CATALOGUE: dict[str, tuple[Severity, str, str, str]] = {
    # ---------------------------------------------------------------- floating point
    "FLOAT.CANCELLATION": (
        "warning",
        "Catastrophic cancellation",
        "Two nearly equal numbers were subtracted, so the leading digits cancel and the surviving digits "
        "are the rounding noise of the inputs. The relative error of the result is amplified by the ratio "
        "of the operands to the (small) difference.",
        "Rewrite the expression so the subtraction disappears: a factored form, log1p/expm1, Welford's "
        "recurrence, or a stable formula such as the q-form of the quadratic roots.",
    ),
    "FLOAT.NO_SIGNIFICANT_DIGITS": (
        "error",
        "No reliable digits in the result",
        "The amplification is large enough that not a single significant digit of the computed value is "
        "trustworthy. The number that came out is arithmetic noise, not an approximation of the answer.",
        "Reformulate the computation (a mathematically equivalent but stable expression), raise the working "
        "precision, or accept that this quantity cannot be evaluated from these inputs in this precision.",
    ),
    "FLOAT.WELL_CONDITIONED": (
        "info",
        "Well-conditioned computation",
        "The measured forward error is at the level the condition number allows, and the condition number "
        "is close to one. Rounding is not the limiting factor here; any remaining error comes from the "
        "method or the model.",
        "Nothing to fix on the floating-point side.",
    ),
    "FLOAT.PROBLEM_ILL_CONDITIONED": (
        "warning",
        "The problem is ill-conditioned, not the algorithm",
        "The forward error is large but consistent with ``condition_number x backward_error``: the "
        "algorithm delivers as much accuracy as the problem permits. Blaming the implementation would be "
        "wrong -- this problem amplifies any perturbation, including rounding.",
        "Change the problem, not the code: rescale the data, reformulate the model, or work in higher "
        "precision. Reordering operations will not buy digits.",
    ),
    "FLOAT.ALGORITHM_UNSTABLE": (
        "error",
        "The algorithm loses accuracy beyond the conditioning",
        "The forward error is much larger than ``condition_number x backward_error``. The problem itself "
        "is not the obstacle; this particular sequence of operations amplifies rounding far more than the "
        "condition number requires.",
        "Replace the algorithm with a backward-stable one (stable quadratic formula, Welford variance, "
        "compensated summation, pivoted factorisation).",
    ),
    "FLOAT.SUBNORMAL_UNDERFLOW": (
        "info",
        "Subnormal range or underflow",
        "An intermediate value fell below the smallest normal number of the dtype and either became "
        "subnormal (fewer digits) or flushed to zero. Results near this scale cannot be trusted digit for "
        "digit.",
        "Rescale the problem so the intermediates stay normal, or use a wider dtype.",
    ),
    "FLOAT.OVERFLOW": (
        "error",
        "Overflow",
        "An intermediate value exceeded the largest finite number of the dtype and became infinity, which "
        "then propagates through the rest of the computation.",
        "Rescale the problem, work in log space, or use a wider dtype.",
    ),
    "FLOAT.MIXED_PRECISION_RESIDUAL": (
        "warning",
        "Precision floor in the residual",
        "Evaluated in reduced precision, this residual never falls below about ``eps/precision`` no matter "
        "how accurate the model becomes: the difference it is built from is already at the rounding level "
        "of the dtype. A training loss built on it stops improving for a numerical reason, not a modelling "
        "one.",
        "Form the residual in float64 even when the network runs in float32, use a residual form that "
        "subtracts less (non-dimensionalised, non-conservative split), or accept the floor as the target.",
    ),
    # ---------------------------------------------------------------- roots
    "ROOTS.NO_BRACKET": (
        "error",
        "The initial interval does not bracket a root",
        "``f(a)`` and ``f(b)`` have the same sign, so no odd number of roots is guaranteed inside and "
        "bisection has no direction to move in.",
        "Scan the interval for a sign change first, or start from a point where the function changes sign.",
    ),
    "ROOTS.NEWTON_CYCLE": (
        "error",
        "Newton's method is cycling",
        "The iterates repeat after a finite number of steps instead of converging. This is a genuine "
        "2-cycle of the Newton map, common for cubic-like functions when the start point lies in the "
        "basin of a saddle, not a slow-convergence symptom.",
        "Change the start point, bracket the root and use a bracketed method (bisection, Brent), or damp the "
        "step (line search).",
    ),
    "ROOTS.FLAT_DERIVATIVE": (
        "error",
        "Derivative too small to divide by",
        "``f'(x)`` is at or near zero, so the Newton step ``-f/f'`` is enormous or undefined. This happens "
        "at an extremum of ``f`` (a start point with no information) or at a multiple root.",
        "Start from a point with a well-scaled derivative, use a bracketed method, or use the "
        "multiple-root form ``f/f'`` when the multiplicity is known.",
    ),
    "ROOTS.MULTIPLE_ROOT_ORDER_LOSS": (
        "warning",
        "Order loss at a multiple root",
        "At a root of multiplicity ``m > 1`` Newton's method converges linearly with ratio ``1 - 1/m`` "
        "instead of quadratically, because ``f'`` also vanishes there. The convergence is still real, it "
        "is just one order lower than the theory for a simple root.",
        "Apply Newton to ``f/f'`` (or to ``f * f' / (f'^2 - f f'')``), which restores the order, or use "
        "bisection which is unaffected.",
    ),
    "ROOTS.STALLED": (
        "warning",
        "The bracket stopped shrinking",
        "Regula falsi keeps one endpoint fixed and the other converges onto the root, so the bracket width "
        "-- and therefore the guaranteed accuracy -- stops improving even though the estimate looks "
        "converged.",
        "Use the Illinois variant (halve the stale endpoint's value) or switch to Brent.",
    ),
    "ROOTS.NOT_CONVERGED": (
        "error",
        "Iteration limit reached before the tolerance",
        "The method used its whole iteration budget without meeting the tolerance. The returned value is "
        "the best iterate, not a certified root.",
        "Look at the residual history: if it decreases linearly, the method is converging but slowly; if it "
        "stagnates, the tolerance is below the achievable accuracy or the method is wrong for this problem.",
    ),
    "ROOTS.JACOBIAN_SINGULAR": (
        "error",
        "The Jacobian is singular or near-singular",
        "The linear system inside the Newton step cannot be solved reliably: either the iteration has "
        "reached a turning point of the solution branch (a fold, where the Jacobian genuinely vanishes) or "
        "the model is degenerate at this point.",
        "At a fold, parameter continuation is the fix (pseudo-arclength lets the branch turn); otherwise "
        "reformulate, rescale, or use a least-squares (Levenberg-Marquardt) step.",
    ),
    "ROOTS.FOLD": (
        "info",
        "Turning point of the solution branch",
        "The parameter cannot be advanced further in this direction: the Jacobian is singular here, so the "
        "branch turns back. Natural-parameter continuation stops at this point by construction.",
        "Switch to pseudo-arclength continuation, which treats the parameter as an unknown and passes "
        "through the fold.",
    ),
    "ROOTS.POLYNOMIAL_ILL_CONDITIONED": (
        "warning",
        "Roots are ill-conditioned with respect to the coefficients",
        "A small relative change in one coefficient moves the roots by ``1/|p'(r)|`` times as much. For a "
        "polynomial with widely separated roots (Wilkinson's), that amplification can reach ten or more "
        "orders of magnitude: the single-precision coefficients do not determine the roots.",
        "Do not compute roots from coefficients in low precision. Use a companion-matrix eigenvalue solve "
        "in double precision, a basis where the problem is well conditioned, or accept an interval bound.",
    ),
    # ---------------------------------------------------------------- interpolation
    "INTERP.RUNGE": (
        "warning",
        "Runge phenomenon: equispaced interpolation diverges",
        "On equispaced nodes the Lebesgue constant grows exponentially with the degree, so increasing the "
        "number of nodes eventually makes the interpolant worse everywhere, with the error largest near the "
        "ends of the interval. This is a property of the node distribution, not of the arithmetic.",
        "Use Chebyshev (or Chebyshev-Lobatto) nodes, which keep the Lebesgue constant logarithmic, or use a "
        "spline, which converges at fixed low order without the oscillation.",
    ),
    "INTERP.ILL_CONDITIONED_VANDERMONDE": (
        "warning",
        "Vandermonde system is ill-conditioned",
        "Solving the interpolation/least-squares problem through the monomial Vandermonde matrix squares "
        "the condition number when the normal equations are formed, and the nodes make it bad to start "
        "with. Above degree ~10 in double precision the coefficients carry no information.",
        "Use the barycentric form for interpolation, and a QR (or SVD) factorisation -- never the normal "
        "equations -- for least squares. A Chebyshev basis lowers the condition number further.",
    ),
    "INTERP.LEBESGUE_GROWTH": (
        "info",
        "Lebesgue constant measured",
        "The Lebesgue constant bounds the interpolation error amplification: ``||p|| <= Lambda * ||f||`` "
        "on the nodes. Its growth rate with the degree is what separates a usable node set from an "
        "unusable one.",
        "Compare node families at the same degree when choosing one.",
    ),
    "INTERP.NON_MONOTONE": (
        "warning",
        "Monotonicity lost",
        "The interpolant overshoots between nodes where the data is monotone. A plain cubic spline has "
        "this behaviour by construction; a shape-preserving scheme must not.",
        "Use PCHIP (monotone cubic Hermite), which limits the slopes so no new extremum appears.",
    ),
    # ---------------------------------------------------------------- quadrature
    "QUAD.ROUNDOFF_FLOOR": (
        "warning",
        "Round-off floor reached",
        "Refining further makes the quadrature/differentiation error grow: the truncation error is already "
        "below the rounding noise accumulated over more and more operations. This is the right-hand branch "
        "of the V-shaped error curve.",
        "Stop refining at the measured optimum, or raise the working precision if that optimum is not "
        "accurate enough.",
    ),
    "QUAD.SINGULARITY": (
        "error",
        "Integrand is singular in the interval",
        "A composite rule assumes a bounded derivative on the interval. With an endpoint or interior "
        "singularity its error estimate collapses and the observed order falls below the theoretical one.",
        "Substitute the variable to remove the singularity (``x = t^2`` for ``1/sqrt(x)``, tanh-sinh for "
        "endpoint behaviour), or split the interval and integrate the singular part in closed form.",
    ),
    "QUAD.ALIASING": (
        "warning",
        "Aliasing in a Fourier/trigonometric interpolation",
        "Frequencies above the Nyquist limit of the sampling are folded onto lower ones, so the computed "
        "spectrum shows energy at frequencies the data never contained.",
        "Sample at least twice the highest frequency present, low-pass filter before sampling, or average "
        "over several phases.",
    ),
    "QUAD.GIBBS": (
        "info",
        "Gibbs overshoot at a discontinuity",
        "A truncated Fourier series overshoots a jump by about 9 % of its height no matter how many terms "
        "are kept; only the width of the overshoot shrinks.",
        "Filter the series (Fejer/Lanczos sigma factors) or use a scheme designed for discontinuities "
        "(TVD limiter, front tracking).",
    ),
    "QUAD.ERROR_ESTIMATE_OPTIMISTIC": (
        "warning",
        "The embedded error estimate is optimistic",
        "The estimated error is much smaller than the true error against the exact value. Embedded "
        "estimates compare two rules of the same family, so they miss the error the family itself makes "
        "(a singularity, a kink, a fast oscillation).",
        "Cross-check with a different rule family or against a manufactured solution before trusting the "
        "estimate for a tolerance.",
    ),
    # ---------------------------------------------------------------- ODE
    "ODE.STIFF": (
        "warning",
        "The problem is stiff",
        "The eigenvalues of the local Jacobian span a wide range of time scales, so an explicit method is "
        "forced to the fastest one for stability while the solution varies on the slow one. The cost is "
        "set by stability, not accuracy.",
        "Use an implicit or linearly implicit method (backward Euler, BDF, Radau, Rosenbrock), which is "
        "stable at the step the accuracy dictates.",
    ),
    "ODE.STABILITY_LIMIT": (
        "error",
        "Step size outside the stability region",
        "``h * lambda`` lies outside the absolute stability region of the method, so the parasitic "
        "component is amplified at every step and the numerical solution blows up even though the exact "
        "one decays.",
        "Reduce the step to the stability limit, or use a method whose region contains the relevant "
        "eigenvalues (A-stable, or L-stable for very stiff modes).",
    ),
    "ODE.ENERGY_DRIFT": (
        "warning",
        "Secular energy drift in a conservative system",
        "A non-symplectic method on a Hamiltonian problem loses or gains energy steadily: the error is "
        "secular in time, not bounded, so long integrations drift away from the true orbit.",
        "Use a symplectic or geometric integrator (Stormer-Verlet, symplectic Euler, Gauss-Legendre), "
        "whose energy error is bounded and oscillatory.",
    ),
    "ODE.ORDER_REDUCTION": (
        "warning",
        "Order reduction on a stiff problem",
        "The method's classical order is derived in the non-stiff limit. On a stiff problem the observed "
        "order can fall well below it (Dahlquist's order reduction), so a step sized for the nominal order "
        "misses the tolerance.",
        "Judge the method by its measured order on this problem, not by the textbook one, or use a method "
        "without stiff order reduction (Radau, SDIRK of sufficient stage order).",
    ),
    "ODE.EVENT": (
        "info",
        "Event located by root finding",
        "The step that crossed the event was cut back and the event time refined, so the discontinuity is "
        "applied at the event and not at the end of a full step.",
        "None; this is the correct handling. Check that the reported event count matches the physics.",
    ),
    "ODE.NOT_CONVERGED": (
        "error",
        "Step below the minimum allowed",
        "The step controller asked for a step smaller than the minimum, usually because the right-hand "
        "side is discontinuous, singular, or the tolerance is unattainable in this precision.",
        "Check the right-hand side near the failure time, loosen the tolerance, or split the integration at "
        "the discontinuity.",
    ),
    # ---------------------------------------------------------------- linear algebra
    "LINALG.SINGULAR": (
        "error",
        "Matrix is singular",
        "The factorisation produced a zero pivot (or a zero diagonal in Cholesky): the matrix does not have "
        "full rank and the system has either no solution or infinitely many.",
        "Check the model formulation; if the system is a least-squares problem, use QR/SVD with a "
        "rank-revealing threshold instead of a direct solve.",
    ),
    "LINALG.ILL_CONDITIONED": (
        "warning",
        "Matrix is ill-conditioned",
        "``kappa(A) * eps`` bounds the relative error of the computed solution. When that product "
        "approaches or exceeds one, the solve can return a solution with no correct digits even though the "
        "factorisation is backward stable.",
        "Rescale rows and columns, use a preconditioner, regularise, or raise the precision. Iterative "
        "refinement recovers accuracy only while ``kappa * eps < 1``.",
    ),
    "LINALG.NO_CONVERGENCE": (
        "error",
        "Iterative solver did not converge",
        "The residual did not fall below the tolerance within the iteration budget. For a stationary "
        "iteration the spectral radius of the iteration matrix is >= 1; for CG the matrix is not positive "
        "definite or is too ill-conditioned.",
        "Use a preconditioner, a Krylov method that tolerates the spectrum (GMRES), or a direct solve.",
    ),
    "LINALG.ORTHOGONALITY_LOSS": (
        "warning",
        "Loss of orthogonality in Gram-Schmidt",
        "Classical Gram-Schmidt loses orthogonality proportionally to ``kappa(A)^2 * eps`` because it "
        "projects against the already-perturbed vectors; reorthogonalisation (or Householder) is what "
        "keeps the basis usable.",
        "Use Householder QR, or modified Gram-Schmidt with a second reorthogonalisation pass.",
    ),
    "LINALG.SLOW_CG": (
        "warning",
        "Slow conjugate-gradient convergence",
        "CG's iteration count grows with ``sqrt(kappa)``: doubling the condition number roughly doubles the "
        "iterations. A large count is a statement about the spectrum, not about the implementation.",
        "Precondition (Jacobi is cheap, incomplete Cholesky is usually decisive, multigrid gives "
        "h-independent counts on a Poisson-like operator).",
    ),
    "LINALG.GROWTH_FACTOR": (
        "info",
        "Pivot growth measured",
        "The growth factor is the largest multiplier a pivot sees relative to the largest entry of ``A``. "
        "Partial pivoting keeps it modest in practice; a large value is a warning that rounding is being "
        "amplified by the elimination order.",
        "Reorder (or equilibrate) the matrix before factorising.",
    ),
    "LINALG.NEAR_DEFECTIVE": (
        "warning",
        "The matrix is near-defective: its eigenvalues are ill-conditioned",
        "The eigenvectors are close to linearly dependent (the condition number of the eigenvector matrix is "
        "large), so a perturbation of the matrix of size ``eps`` moves the eigenvalues by something like "
        "``eps ** (1/m)`` for a Jordan block of size ``m`` -- not ``eps``. The eigenvalues are therefore not "
        "a continuous function of the matrix entries in the way a well-conditioned matrix's are, and digits "
        "that look significant in the computation need not be significant in the answer.",
        "Do not read individual eigenvalues as physical modes when they sit in a defective cluster: report "
        "the invariant subspace instead, or perturb the model to break the degeneracy on purpose.",
    ),
    "LINALG.MULTIGRID_H_DEPENDENT": (
        "warning",
        "Multigrid convergence depends on the mesh",
        "The V-cycle iteration count grew with the mesh size, which means one of the components is not "
        "doing its job (smoother too weak, transfer operators inconsistent, coarse grid too coarse), or "
        "the boundary conditions are not resolved on the coarsest grid.",
        "Check the smoother's smoothing factor and the coarse-grid operator; a correct V-cycle on a "
        "Poisson operator needs a constant number of iterations at every h.",
    ),
    # ---------------------------------------------------------------- PDE
    "PDE.CFL_VIOLATION": (
        "error",
        "CFL / stability limit exceeded",
        "The amplification factor of the scheme has modulus above one for at least one wavenumber, so every "
        "step multiplies that error mode. The solution grows without bound even though the exact solution "
        "is bounded: this is numerical instability, not physics.",
        "Reduce ``dt`` to the limit the scheme allows (``c dt/dx <= 1`` for upwind and Lax-Friedrichs, "
        "``2 alpha dt/dx^2 <= 1`` for explicit heat), or use an unconditionally stable (implicit) scheme.",
    ),
    "PDE.NUMERICAL_DIFFUSION": (
        "warning",
        "Numerical diffusion dominates the physical one",
        "The leading truncation term of the upwind scheme is a second derivative with coefficient "
        "``c dx (1 - C)/2``. On a coarse mesh that artificial viscosity is larger than the physical one and "
        "smooths the solution: profiles spread and sharp fronts disappear.",
        "Refine the mesh, use a higher-order upwind reconstruction, or a scheme with less numerical "
        "dissipation (Lax-Wendroff, MUSCL) where it stays stable.",
    ),
    "PDE.DISPERSION": (
        "warning",
        "Dispersion: the numerical phase speed is wrong",
        "The truncation error is a third-derivative term, so waves of different wavelengths travel at "
        "different speeds and the packet develops oscillations that the exact solution does not have.",
        "Refine the mesh, reduce the Courant number, or use a scheme with a corrected dispersion relation.",
    ),
    "PDE.CONSISTENCY": (
        "info",
        "Truncation error measured",
        "The truncation error is what remains when the exact solution is substituted into the discrete "
        "operator. Its order in ``dx`` and ``dt`` is the consistency order; it is a property of the scheme, "
        "independent of the solution being computed.",
        "None; compare it with the observed convergence order to check the Lax equivalence theorem holds "
        "for this problem.",
    ),
    "PDE.INCONSISTENT": (
        "error",
        "Scheme is not consistent",
        "The truncation error does not vanish as the mesh is refined, so stability can never give "
        "convergence: the scheme is solving a different equation. A typical cause is a misaligned boundary "
        "stencil or a scheme whose order collapses at the boundary.",
        "Derive the truncation error for the scheme as written, including the boundary treatment.",
    ),
    "PDE.BLOW_UP": (
        "error",
        "Solution blew up",
        "The discrete solution grew past any physical bound. Above the stability limit this happens at "
        "machine speed; below it, it can also be a genuinely unstable initial-boundary-value problem.",
        "Check the stability limit first (compare the measured amplification factor with one), then check "
        "that the discrete boundary conditions are consistent with the interior scheme.",
    ),
    "PDE.OSCILLATION": (
        "warning",
        "Overshoot next to a discontinuity",
        "A second-order scheme without limiting produces new extrema near a jump (Godunov's barrier: no "
        "linear scheme is both second-order and monotone). The overshoot does not shrink with refinement, "
        "only its width does.",
        "Use a TVD limiter (minmod, van Leer, superbee) or a WENO reconstruction where the solution is "
        "non-smooth.",
    ),
    # ---------------------------------------------------------------- regimes (which strategy fits the problem)
    "REGIME.BOUNDARY_LAYER": (
        "warning",
        "Thin layer at a boundary",
        "The field changes over a thickness that is a small fraction of the domain, and the change happens "
        "against a wall or an edge. Uniform sampling spends almost every point on the region where the "
        "solution is flat and a handful on the one where it is not, so the loss is dominated by the flat "
        "part and the layer is learned late or not at all.",
        "Concentrate points in the layer (residual- or curvature-based resampling), scale the wall-normal "
        "coordinate so the layer has thickness O(1) before training, or ramp the problem: a thickened layer "
        "first, the true one afterwards.",
    ),
    "REGIME.SHOCK": (
        "warning",
        "Interior jump in the field",
        "The field goes from one plateau to another across a thickness that is a small fraction of the "
        "domain, away from any boundary. A smooth-network residual is large in a band around the jump and "
        "the minimisation spreads that error over the whole domain.",
        "Add points where the residual is largest, enforce time causality when the jump forms in time, "
        "split the domain (XPINN) at the jump, or continue in viscosity: solve a smeared jump first and "
        "reduce the viscosity towards the target.",
    ),
    "REGIME.OSCILLATORY": (
        "warning",
        "The solution turns before it decays",
        "The local Jacobian has an eigenvalue whose imaginary part is comparable to its modulus, so the "
        "dynamics rotate: the solution oscillates on the same time scale on which it decays. A monotone "
        "decay is a poor model for it, and in an oscillatory problem the phase error is as important as "
        "the amplitude error.",
        "Give the model sinusoidal features at the measured frequency, weight time so that earlier times "
        "are learned first, and sample where the residual is largest over a cycle.",
    ),
    "REGIME.HIGH_FREQUENCY": (
        "warning",
        "Oscillatory field",
        "The field oscillates across the domain, so the number of points per wavelength is what limits how "
        "much of it any model can represent. A plain MLP is spectrally biased towards low frequencies: it "
        "fits the envelope and misses the oscillation.",
        "Provide Fourier features or a sinusoidal activation, check that the sampling resolves every "
        "wavelength, and evaluate the error per wavelength rather than as one global number.",
    ),
    "REGIME.SHARP_FEATURE": (
        "warning",
        "Thin interior feature with no net change",
        "The field has a core thinner than a small fraction of the domain, in the interior, and returns to "
        "the level it started from -- a bump, a spot, a hot streak. There is no jump, so this is not a "
        "shock and the shock-specific remedies do not apply, but the residual has the same shape: large in "
        "a band around the core and averaged away by a global loss.",
        "Add points around the core, or split the domain so one network is responsible for it; measure the "
        "thickness again after each cycle, because the measured one is a lower bound.",
    ),
    "REGIME.MULTISCALE": (
        "warning",
        "Two separated scales in the spectrum",
        "The spectrum has two bands with a gap between them: the field is a large-scale structure plus a "
        "fine scale that carries a small but non-negligible part of the energy. Training on the large "
        "scale first and then adding the fine one is not the same problem as training on both at once.",
        "Use a curriculum (large scale first, fine scale added afterwards), or domain decomposition so "
        "that each network sees one scale.",
    ),
    "REGIME.UNRESOLVED": (
        "warning",
        "The layer is not resolved on this grid",
        "Fewer than three grid points lie across the layer, so the measured thickness is close to the grid "
        "spacing and not the feature. The sharpness, the thickness and the strategies derived from them are "
        "therefore lower bounds.",
        "Refine the grid across the layer before drawing a conclusion from the numbers, or read them as "
        "'at least this thin'.",
    ),
    "REGIME.SMOOTH": (
        "info",
        "No difficult regime detected",
        "The stiffness ratio, the gradient concentration, the layer thickness and the spectrum are all "
        "inside the bands where a uniform method behaves well. This is a statement about the evidence that "
        "was measured, not a guarantee: a regime these detectors do not test for can still be present.",
        "Nothing to change on the basis of this evidence.",
    ),
}


@dataclass(frozen=True)
class Diagnostic:
    """One explanation of a numerical result: what happened, why, and what to do about it."""

    code: str
    severity: Severity
    title: str
    explanation: str
    remedy: str
    context: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form (what the Studio page and the Flow node receive)."""
        return {
            "code": self.code,
            "severity": self.severity,
            "title": self.title,
            "explanation": self.explanation,
            "remedy": self.remedy,
            "context": dict(self.context),
        }

    def format(self) -> str:
        """One line for a CLI: ``[warning] ODE.STIFF: The problem is stiff -- use an implicit method``."""
        return f"[{self.severity}] {self.code}: {self.title} -- {self.remedy}"


def explain(code: str, **context: Any) -> Diagnostic:
    """Look ``code`` up in the catalogue and attach the measured numbers as ``context``.

    Raises ``KeyError`` for an unknown code: a diagnostic with no explanation would be a silent failure,
    which is exactly what this module exists to avoid.
    """
    try:
        severity, title, explanation, remedy = _CATALOGUE[code]
    except KeyError:  # pragma: no cover - a typo in a code is a programming error, not a runtime condition
        raise KeyError(f"unknown diagnostic code {code!r}; see diagnostics.catalogue()") from None
    return Diagnostic(
        code=code,
        severity=severity,
        title=title,
        explanation=explanation,
        remedy=remedy,
        context=dict(context),
    )


def catalogue() -> tuple[str, ...]:
    """Every code this package can emit, sorted -- the contract with the interface."""
    return tuple(sorted(_CATALOGUE))


CATALOGUE: Mapping[str, tuple[Severity, str, str, str]] = _CATALOGUE


def has_code(code: str) -> bool:
    """Whether ``code`` is in the catalogue."""
    return code in _CATALOGUE


def worst(diagnostics: list[Diagnostic] | tuple[Diagnostic, ...] | None) -> Diagnostic | None:
    """The most severe diagnostic of a list (error > warning > info), or ``None`` when there is none.

    Result objects expose this as a convenience: an interface that shows one line shows this one.
    """
    if not diagnostics:
        return None
    rank = {"info": 0, "warning": 1, "error": 2}
    return max(diagnostics, key=lambda d: rank[d.severity])

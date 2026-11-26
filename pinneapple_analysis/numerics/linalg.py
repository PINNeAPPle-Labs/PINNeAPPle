"""Numerical linear algebra: factorisations, conditioning, iterative solvers, eigenvalues, SVD (#1061).

The organising question of this module is the one a solver user actually asks: **where did my digits
go?** Every routine here returns the evidence for that answer, not only the answer.

* Direct solvers -- LU with partial pivoting, Cholesky, Householder QR, Gram-Schmidt -- with the
  **growth factor** of the elimination, or the **loss of orthogonality** of the basis they built.
* Conditioning -- ``kappa(A) * eps`` against the relative error actually observed, so "the answer is
  wrong" becomes "the answer is wrong by as much as this matrix allows", plus iterative refinement,
  which recovers the digits that remain recoverable.
* Iterative solvers -- Jacobi, Gauss-Seidel, SOR with the **spectral radius** of the iteration matrix
  predicting the measured convergence rate, CG with its iteration count against ``sqrt(kappa)``, GMRES
  and BiCGStab for the non-symmetric case, and a multigrid V-cycle whose iteration count is measured to
  be *independent of h*.
* Eigenvalues -- power and inverse iteration with the measured rate ``|lambda_2/lambda_1|``, Rayleigh
  quotient iteration, the shifted QR algorithm, and Lanczos for the extreme eigenvalues of a large
  symmetric matrix.
* SVD -- Eckart-Young low-rank truncation with the measured (not merely claimed) Frobenius error, the
  pseudo-inverse, and truncated-SVD regularisation.

The limitation, once for the module: none of this is sparse. Every matrix here is dense and every
factorisation fills in; the complexity claims are for the teaching-sized problems these functions are
verified on. ``pinneapple_simulation.numerical_solvers.sparse_direct`` is the sparse counterpart, and
the multigrid V-cycle below is the one genuinely ``O(n)`` algorithm in the file.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Union

import numpy as np

from .diagnostics import Diagnostic, explain

MatrixLike = Union[np.ndarray, Callable[[np.ndarray], np.ndarray]]  # noqa: UP007

__all__ = [
    "ConditioningStudy",
    "EigenResult",
    "IterativeResult",
    "LUFactorization",
    "SVDResult",
    "apply_ilu",
    "bicgstab",
    "cholesky",
    "cholesky_solve",
    "condition_number",
    "conditioning_study",
    "conjugate_gradient",
    "gauss_seidel",
    "gmres",
    "householder_qr",
    "ilu0",
    "inverse_iteration",
    "iterative_matrix",
    "iterative_refinement",
    "jacobi",
    "jacobi_preconditioner",
    "lanczos",
    "lu_factor",
    "lu_solve",
    "modified_gram_schmidt",
    "multigrid_h_independence",
    "multigrid_poisson_1d",
    "orthogonality_loss",
    "power_iteration",
    "qr_algorithm",
    "rayleigh_quotient_iteration",
    "sor",
    "sor_optimal_omega",
    "svd_low_rank",
    "svd_pseudoinverse",
    "truncated_svd_solve",
]


def _matvec(A: MatrixLike, x: np.ndarray) -> np.ndarray:
    """Apply ``A`` to ``x``, whether ``A`` is a dense array or a matrix-vector callable."""
    xv = np.asarray(x, dtype=np.float64).ravel()
    out = A @ xv if isinstance(A, np.ndarray) else np.asarray(A(xv), dtype=np.float64)
    return np.asarray(out, dtype=np.float64).ravel()


def _as_matrix(A: np.ndarray, *, square: bool = True) -> np.ndarray:
    M = np.asarray(A, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError(f"A must be two-dimensional; got shape {M.shape}")
    if square and M.shape[0] != M.shape[1]:
        raise ValueError(f"A must be square; got shape {M.shape}")
    return M


# --------------------------------------------------------------------------- direct solvers
@dataclass(frozen=True)
class LUFactorization:
    """The factors and, as importantly, the growth factor that says how much rounding they amplify."""

    lu: np.ndarray
    pivots: tuple[int, ...]
    growth_factor: float
    n_pivots_changed: int
    singular: bool
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form."""
        return {
            "growth_factor": self.growth_factor,
            "n_pivots_changed": self.n_pivots_changed,
            "singular": self.singular,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def lu_factor(A: np.ndarray) -> LUFactorization:
    """LU with partial pivoting, plus the growth factor ``max|U_ij| / max|A_ij|``.

    Partial pivoting keeps ``|L_ij| <= 1`` and makes the factorisation backward stable; the price of the
    error is measured by the growth factor, which is the largest multiplier any pivot saw. It is
    ``O(1)`` in practice and ``O(2^n)`` in the worst case (Wilkinson's matrix), and that worst case is
    what makes "LU is stable" a statement about the algorithm plus an assumption about the pivoting.

    A zero pivot with nothing to swap is reported as ``singular`` rather than raising: a caller doing a
    residual check usually wants to see the factors.
    """
    M = _as_matrix(A).copy()
    n = M.shape[0]
    scale = float(np.max(np.abs(M))) if M.size else 0.0
    pivots: list[int] = []
    singular = False
    changed = 0

    for k in range(n):
        p = int(np.argmax(np.abs(M[k:, k]))) + k
        pivots.append(p)
        if p != k:
            changed += 1
            M[[k, p], :] = M[[p, k], :]
        pivot = M[k, k]
        if pivot == 0.0:
            singular = True
            continue
        M[k + 1:, k] /= pivot
        if k + 1 < n:
            M[k + 1:, k + 1:] -= np.outer(M[k + 1:, k], M[k, k + 1:])

    growth = float(np.max(np.abs(M))) / scale if scale > 0 else float("nan")
    diagnostics: list[Diagnostic] = []
    if singular:
        diagnostics.append(explain("LINALG.SINGULAR", size=n))
    elif growth > 20.0:
        diagnostics.append(explain("LINALG.GROWTH_FACTOR", growth_factor=growth))
    return LUFactorization(
        lu=M, pivots=tuple(pivots), growth_factor=growth, n_pivots_changed=changed,
        singular=singular, diagnostics=tuple(diagnostics),
    )


def lu_solve(fact: LUFactorization, b: Sequence[float] | np.ndarray) -> np.ndarray:
    """Solve ``A x = b`` from an :class:`LUFactorization` by forward/back substitution with the pivots.

    Raises ``ValueError`` when the factorisation found the matrix singular: a solution would be an
    arbitrary representative of an infinite set.
    """
    if fact.singular:
        raise ValueError("cannot solve with a singular LU factorisation")
    lu = fact.lu
    n = lu.shape[0]
    rhs = np.asarray(b, dtype=np.float64).ravel()
    if rhs.size != n:
        raise ValueError(f"b has {rhs.size} entries for a {n}x{n} system")
    y = rhs.copy()
    for k in range(n - 1):  # apply the pivoting as it happened
        p = fact.pivots[k]
        if p != k:
            y[k], y[p] = y[p], y[k]
    y = _forward_substitution(lu, y, unit_diagonal=True)
    return _back_substitution(lu, y)


def _forward_substitution(L: np.ndarray, b: np.ndarray, *, unit_diagonal: bool = False) -> np.ndarray:
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        s = float(b[i])
        for j in range(i):
            s -= L[i, j] * y[j]
        y[i] = s if unit_diagonal else s / L[i, i]
    return y


def _back_substitution(U: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = float(b[i])
        for j in range(i + 1, n):
            s -= U[i, j] * x[j]
        if U[i, i] == 0.0:
            raise ValueError("zero on the diagonal of U: the system is singular")
        x[i] = s / U[i, i]
    return x


def cholesky(A: np.ndarray) -> np.ndarray:
    """Lower triangular ``L`` with ``A = L L^T`` for symmetric positive definite ``A``.

    Uses half the work of LU and needs no pivoting, and the algorithm *detects* indefiniteness: a
    non-positive pivot under the square root means the matrix is not positive definite, which is reported
    as ``ValueError`` instead of returning a complex factor.

    Limitation: symmetry is assumed, not checked -- pass a symmetric matrix, or symmetrise it first.
    """
    M = _as_matrix(A)
    n = M.shape[0]
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            s = float(M[i, j]) - float(L[i, :j] @ L[j, :j])
            if i == j:
                if s <= 0.0:
                    raise ValueError(
                        f"not positive definite: pivot {s:.3e} at index {i} -- Cholesky needs a positive "
                        "definite matrix (a semidefinite one needs pivoting or an LDL^T factorisation)"
                    )
                L[i, i] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    return L


def cholesky_solve(L: np.ndarray, b: Sequence[float] | np.ndarray) -> np.ndarray:
    """Solve ``L L^T x = b`` by one forward and one back substitution."""
    y = _forward_substitution(L, np.asarray(b, dtype=np.float64).ravel())
    return _back_substitution(L.T, y)


def householder_qr(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """QR by Householder reflections: ``Q`` orthogonal to machine precision, ``R`` upper triangular.

    Each step reflects a column onto the coordinate axis, which is unconditionally stable: no subtraction
    of nearly equal numbers is involved in building the reflectors. This is why QR is the recommended
    way to solve least squares, and why its orthogonality does not degrade as ``kappa(A)`` grows (compare
    :func:`modified_gram_schmidt`).
    """
    M = _as_matrix(A, square=False).copy()
    m, n = M.shape
    Q = np.eye(m)
    for k in range(min(m - 1, n)):
        x = M[k:, k]
        norm = float(np.linalg.norm(x))
        if norm == 0.0:
            continue
        sign = 1.0 if x[0] >= 0 else -1.0
        v = x.copy()
        v[0] += sign * norm
        v_norm = float(np.linalg.norm(v))
        if v_norm == 0.0:
            continue
        v = v / v_norm
        M[k:, k:] -= 2.0 * np.outer(v, v @ M[k:, k:])
        Q[:, k:] -= 2.0 * np.outer(Q[:, k:] @ v, v)
    return Q, np.triu(M)


def modified_gram_schmidt(A: np.ndarray, *, reorthogonalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """QR by (modified) Gram-Schmidt, optionally with a second reorthogonalisation pass.

    Classical Gram-Schmidt loses orthogonality like ``kappa(A)^2 * eps`` because it projects against
    vectors that are already wrong; the modified variant re-projects after each subtraction, which
    reduces the loss to ``kappa(A) * eps``. A second pass restores it to machine precision even for
    ``kappa`` near ``1/eps``. The three variants are here so the loss can be *measured* on a badly
    conditioned matrix instead of asserted.
    """
    M = _as_matrix(A, square=False).copy()
    m, n = M.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = M[:, j].copy()
        for i in range(j):
            R[i, j] = float(Q[:, i] @ v)
            v -= R[i, j] * Q[:, i]
        if reorthogonalize:
            for i in range(j):
                correction = float(Q[:, i] @ v)
                R[i, j] += correction
                v -= correction * Q[:, i]
        R[j, j] = float(np.linalg.norm(v))
        Q[:, j] = v / R[j, j] if R[j, j] > 0 else 0.0
    return Q, R


def orthogonality_loss(Q: np.ndarray) -> float:
    """``max|Q^T Q - I|``: the measured departure of a computed basis from orthonormality.

    This is the number that separates Householder QR (``~eps``) from classical Gram-Schmidt
    (``~kappa^2 eps``) on the same matrix, and the reason least squares is done with the former.
    """
    Qm = np.asarray(Q, dtype=np.float64)
    return float(np.max(np.abs(Qm.T @ Qm - np.eye(Qm.shape[1]))))


# --------------------------------------------------------------------------- conditioning
def condition_number(A: np.ndarray) -> float:
    """2-norm condition number ``sigma_max / sigma_min`` from the SVD.

    This is the ratio by which a relative perturbation of the data can be amplified in the solution, so
    ``kappa * eps`` is the accuracy ceiling of *any* backward-stable solve of this system -- including a
    perfect one. ``inf`` means exactly singular.
    """
    s = np.linalg.svd(_as_matrix(A), compute_uv=False)
    if s[-1] == 0.0:
        return float("inf")
    return float(s[0] / s[-1])


@dataclass(frozen=True)
class ConditioningStudy:
    """Measured solve error against the condition number -- the slope should be 1 while rounding rules."""

    condition_numbers: np.ndarray
    relative_errors: np.ndarray
    predicted_errors: np.ndarray
    slope: float
    observations: int
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form for the lab page."""
        return {
            "condition_numbers": [float(v) for v in self.condition_numbers],
            "relative_errors": [float(v) for v in self.relative_errors],
            "predicted_errors": [float(v) for v in self.predicted_errors],
            "slope": self.slope,
            "observations": self.observations,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def conditioning_study(matrices: Sequence[np.ndarray], *, eps: float = float(np.finfo(float).eps)) -> ConditioningStudy:
    """Solve each system with a known exact solution and plot the error against ``kappa``.

    For each matrix ``A`` the exact solution is ``x = 1`` (all ones), the right-hand side is ``b = A x``
    computed in double precision, and the solve is ``numpy.linalg.solve`` -- i.e. a backward-stable LU.
    The relative error of the recovered ``x`` is then expected to be ``~ kappa * eps``; the measured
    slope of ``log(error)`` against ``log(kappa)`` is the *experimental* confirmation that the bound is
    tight, and it should be 1.

    A slope well below 1 means the largest condition numbers are still in the regime where the error is
    set by the ``O(1)`` factor rather than by ``kappa``; a slope above 1 means the systems are so
    ill-conditioned that the solve is no longer backward stable (`numpy` uses the same LU throughout, so
    that would point at the *problem*, not the routine).
    """
    conds: list[float] = []
    errors: list[float] = []
    for A in matrices:
        Am = _as_matrix(A)
        x_true = np.ones(Am.shape[1])
        b = Am @ x_true
        x = np.linalg.solve(Am, b)
        conds.append(condition_number(Am))
        errors.append(float(np.linalg.norm(x - x_true) / np.linalg.norm(x_true)))

    cond_arr = np.asarray(conds)
    err_arr = np.maximum(np.asarray(errors), np.finfo(float).tiny)
    finite = np.isfinite(cond_arr) & (cond_arr > 0)
    if finite.sum() >= 2:
        slope = float(np.polyfit(np.log(cond_arr[finite]), np.log(err_arr[finite]), 1)[0])
    else:
        slope = float("nan")

    diagnostics: list[Diagnostic] = []
    if finite.any() and np.max(cond_arr[finite]) * eps > 1e-3:
        diagnostics.append(
            explain("LINALG.ILL_CONDITIONED", condition_number=float(np.max(cond_arr[finite])),
                    eps=eps, bound=float(np.max(cond_arr[finite]) * eps))
        )
    return ConditioningStudy(
        condition_numbers=cond_arr,
        relative_errors=err_arr,
        predicted_errors=cond_arr * eps,
        slope=slope,
        observations=int(cond_arr.size),
        diagnostics=tuple(diagnostics),
    )


def iterative_refinement(
    A: np.ndarray, b: Sequence[float] | np.ndarray, *, max_iter: int = 10
) -> dict[str, Any]:
    """Solve, then repeatedly correct ``x += A^{-1} (b - A x)`` using the residual in the original precision.

    The residual ``b - A x`` is computed at the *working* precision, so the correction carries the
    information the first solve lost: refinement recovers accuracy as long as ``kappa(A) * eps < 1``.
    It cannot beat that bound (no algorithm can, for this problem), which is why the returned
    ``improved`` flag compares the final error with the first, not with zero.

    Returns the error history against the exact solution (``x = 1``), the residuals, and whether the
    error actually decreased.
    """
    Am = _as_matrix(A)
    rhs = np.asarray(b, dtype=np.float64).ravel()
    n = Am.shape[1]
    x_true = np.ones(n)
    factor = lu_factor(Am)
    if factor.singular:
        raise ValueError("cannot refine: the matrix is singular")
    x = lu_solve(factor, rhs)
    errors = [float(np.linalg.norm(x - x_true))]
    residuals = [float(np.linalg.norm(rhs - Am @ x))]
    for _ in range(max_iter):
        correction = lu_solve(factor, rhs - Am @ x)
        x = x + correction
        errors.append(float(np.linalg.norm(x - x_true)))
        residuals.append(float(np.linalg.norm(rhs - Am @ x)))
        if errors[-1] >= errors[-2] or errors[-1] == 0.0:
            break
    diagnostics: list[Diagnostic] = []
    if errors[-1] > errors[0]:
        diagnostics.append(explain("LINALG.ILL_CONDITIONED", condition_number=condition_number(Am),
                                   note="refinement increased the error"))
    return {
        "errors": errors,
        "residuals": residuals,
        "improved": bool(errors[-1] < errors[0]),
        "iterations": len(errors) - 1,
        "condition_number": condition_number(Am),
        "diagnostics": [d.to_dict() for d in diagnostics],
    }


# --------------------------------------------------------------------------- stationary iterations
@dataclass(frozen=True)
class IterativeResult:
    """An iterative solve: the history, the measured rate, and the spectral radius that predicted it."""

    x: np.ndarray
    iterations: int
    converged: bool
    residual_history: tuple[float, ...]
    error_history: tuple[float, ...]
    measured_rate: float
    spectral_radius: float
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form."""
        return {
            "x": [float(v) for v in self.x],
            "iterations": self.iterations,
            "converged": self.converged,
            "residual_history": [float(v) for v in self.residual_history],
            "error_history": [float(v) for v in self.error_history],
            "measured_rate": self.measured_rate,
            "spectral_radius": self.spectral_radius,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def iterative_matrix(A: np.ndarray, method: str, *, omega: float = 1.0) -> np.ndarray:
    """The iteration matrix ``T`` with ``x_{k+1} = T x_k + c``: its spectral radius is the rate.

    ``A = D + L + U`` (diagonal, strict lower, strict upper):

    * Jacobi: ``T = -D^{-1}(L+U)``;
    * Gauss-Seidel: ``T = -(D+L)^{-1} U``;
    * SOR: ``T = (D + omega L)^{-1} ((1-omega) D - omega U)``.

    The spectral radius is the asymptotic per-iteration error reduction, which is what makes the two
    numbers in :class:`IterativeResult` (predicted and measured) comparable. ``omega = 1`` is exactly
    Gauss-Seidel, which is a useful identity to check before trusting an SOR result.
    """
    Am = _as_matrix(A)
    D = np.diag(np.diag(Am))
    L = np.tril(Am, -1)
    U = np.triu(Am, 1)
    if method == "jacobi":
        return -np.linalg.solve(D, L + U)
    if method == "gauss_seidel":
        return -np.linalg.solve(D + L, U)
    if method == "sor":
        if not 0.0 < omega < 2.0:
            raise ValueError(f"SOR requires 0 < omega < 2 for convergence; got {omega}")
        return np.linalg.solve(D + omega * L, (1.0 - omega) * D - omega * U)
    raise ValueError(f"unknown method {method!r}; use 'jacobi', 'gauss_seidel' or 'sor'")


def _stationary_solve(
    A: np.ndarray,
    b: np.ndarray,
    method: str,
    *,
    omega: float,
    tol: float,
    max_iter: int,
    x_true: np.ndarray | None,
) -> IterativeResult:
    """Shared driver for Jacobi/Gauss-Seidel/SOR: sweep updates, record residuals and the measured rate."""
    Am = _as_matrix(A)
    rhs = np.asarray(b, dtype=np.float64).ravel()
    n = rhs.size
    x = np.zeros(n)
    residuals = [float(np.linalg.norm(rhs - Am @ x))]
    errors = [float(np.linalg.norm(x - x_true))] if x_true is not None else []
    T = iterative_matrix(Am, method, omega=omega)
    radius = float(np.max(np.abs(np.linalg.eigvals(T)))) if n <= 512 else float("nan")
    bnorm = max(residuals[0], np.finfo(float).tiny)
    converged = False

    for _ in range(max_iter):
        if method == "jacobi":
            x = x + (rhs - Am @ x) / np.diag(Am)
        else:
            for i in range(n):
                left = float(Am[i] @ x) - Am[i, i] * x[i]
                x[i] = (1.0 - omega) * x[i] + omega * (rhs[i] - left) / Am[i, i]
        residuals.append(float(np.linalg.norm(rhs - Am @ x)))
        if x_true is not None:
            errors.append(float(np.linalg.norm(x - x_true)))
        if residuals[-1] <= tol * bnorm:
            converged = True
            break

    rate = float("nan")
    if len(residuals) > 2 and residuals[0] > 0 and residuals[-1] > 0:
        rate = float((residuals[-1] / residuals[0]) ** (1.0 / (len(residuals) - 1)))
    diagnostics: list[Diagnostic] = []
    if not converged:
        diagnostics.append(explain("LINALG.NO_CONVERGENCE", method=method, spectral_radius=radius,
                                   iterations=len(residuals) - 1))
    return IterativeResult(
        x=x, iterations=len(residuals) - 1, converged=converged,
        residual_history=tuple(residuals), error_history=tuple(errors),
        measured_rate=rate, spectral_radius=radius, diagnostics=tuple(diagnostics),
    )


def jacobi(
    A: np.ndarray,
    b: Sequence[float] | np.ndarray,
    *,
    tol: float = 1e-10,
    max_iter: int = 500,
    x_true: Sequence[float] | np.ndarray | None = None,
) -> IterativeResult:
    """Jacobi iteration: diagonal preconditioner as the whole method, and the reference for the others.

    Converges for strictly diagonally dominant ``A``, with the per-iteration reduction given by the
    spectral radius of ``-D^{-1}(L+U)``. Pass ``x_true`` to also record the *error* history, which is
    what shows the difference between the residual falling and the answer improving.
    """
    return _stationary_solve(A, np.asarray(b, dtype=np.float64), "jacobi", omega=1.0, tol=tol,
                             max_iter=max_iter,
                             x_true=None if x_true is None else np.asarray(x_true, dtype=np.float64))


def gauss_seidel(
    A: np.ndarray,
    b: Sequence[float] | np.ndarray,
    *,
    tol: float = 1e-10,
    max_iter: int = 500,
    x_true: Sequence[float] | np.ndarray | None = None,
) -> IterativeResult:
    """Gauss-Seidel: uses each freshly updated unknown immediately, halving the iteration matrix's radius.

    For the model problem its spectral radius is the square of Jacobi's, which is a measurable factor of
    two in iteration count -- the number reported here.
    """
    return _stationary_solve(A, np.asarray(b, dtype=np.float64), "gauss_seidel", omega=1.0, tol=tol,
                             max_iter=max_iter,
                             x_true=None if x_true is None else np.asarray(x_true, dtype=np.float64))


def sor(
    A: np.ndarray,
    b: Sequence[float] | np.ndarray,
    *,
    omega: float = 1.5,
    tol: float = 1e-10,
    max_iter: int = 500,
    x_true: Sequence[float] | np.ndarray | None = None,
) -> IterativeResult:
    """Successive over-relaxation, with the over-relaxation factor as the knob.

    For the 1-D Poisson matrix the optimal factor is ``omega_opt = 2 / (1 + sin(pi h))``, giving a
    spectral radius ``1 - 2 pi h`` -- an order of magnitude fewer iterations than Gauss-Seidel at the
    same ``h``. Raising ``omega`` past ``omega_opt`` raises the radius again, so 1.9 is not automatically
    better than 1.5; that curve is what :func:`sor_optimal_omega` measures.
    """
    return _stationary_solve(A, np.asarray(b, dtype=np.float64), "sor", omega=omega, tol=tol,
                             max_iter=max_iter,
                             x_true=None if x_true is None else np.asarray(x_true, dtype=np.float64))


def sor_optimal_omega(A: np.ndarray, *, samples: int = 19) -> dict[str, Any]:
    """Scan ``omega`` and return the one that minimises the spectral radius, with the measured curve.

    The scan is the honest way to get ``omega_opt`` for a matrix that is not the model Poisson problem:
    the closed form ``2/(1+sqrt(1-rho_J^2))`` holds only under the assumptions of consistently ordered
    matrices with real Jacobi eigenvalues.
    """
    omegas = np.linspace(0.1, 1.95, samples)
    radii = [float(np.max(np.abs(np.linalg.eigvals(iterative_matrix(A, "sor", omega=float(w))))))
             for w in omegas]
    best = int(np.argmin(radii))
    return {
        "omegas": [float(w) for w in omegas],
        "spectral_radii": radii,
        "optimal_omega": float(omegas[best]),
        "min_spectral_radius": radii[best],
        "gauss_seidel_radius": float(np.max(np.abs(np.linalg.eigvals(iterative_matrix(A, "gauss_seidel"))))),
    }


# --------------------------------------------------------------------------- Krylov methods
def conjugate_gradient(
    A: np.ndarray,
    b: Sequence[float] | np.ndarray,
    *,
    tol: float = 1e-10,
    max_iter: int | None = None,
    M: Callable[[np.ndarray], np.ndarray] | None = None,
) -> IterativeResult:
    """Conjugate gradients for symmetric positive definite ``A``.

    The error is minimised in the ``A``-norm over a growing Krylov subspace, which is why the iteration
    count grows like ``sqrt(kappa)``: ``~ (sqrt(kappa)/pi) log(2/tol)``. The measured count is returned
    so that claim can be checked on the caller's own operator; ``max_iter`` defaults to ``2n`` because CG
    is exact in at most ``n`` steps in exact arithmetic and needs a little slack in floating point.

    ``M`` is an optional preconditioner given as a callable applying ``M^{-1}``. A non-SPD matrix makes CG
    break down (``p^T A p <= 0``), which is reported through ``converged=False`` and the diagnostic
    rather than a silent wrong answer.
    """
    Am = _as_matrix(A)
    rhs = np.asarray(b, dtype=np.float64).ravel()
    n = rhs.size
    limit = max_iter if max_iter is not None else 2 * n
    x = np.zeros(n)
    r = rhs.copy()
    z = M(r) if M is not None else r.copy()
    p = z.copy()
    rz = float(r @ z)
    residuals = [float(np.linalg.norm(r))]
    bnorm = max(residuals[0], np.finfo(float).tiny)
    converged = residuals[0] <= tol * bnorm
    breakdown = False

    for _ in range(limit):
        if converged:
            break
        Ap = Am @ p
        pAp = float(p @ Ap)
        if pAp <= 0.0:
            breakdown = True
            break
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        residuals.append(float(np.linalg.norm(r)))
        if residuals[-1] <= tol * bnorm:
            converged = True
            break
        z = M(r) if M is not None else r
        rz_new = float(r @ z)
        if rz == 0.0:
            breakdown = True
            break
        p = z + (rz_new / rz) * p
        rz = rz_new

    diagnostics: list[Diagnostic] = []
    if breakdown:
        diagnostics.append(explain("LINALG.NO_CONVERGENCE", method="conjugate gradient",
                                   note="p^T A p <= 0: A is not positive definite"))
    elif not converged:
        kappa = condition_number(Am)
        diagnostics.append(explain("LINALG.SLOW_CG", condition_number=kappa,
                                   predicted_iterations=float(np.sqrt(kappa)),
                                   iterations=len(residuals) - 1))
    rate = float("nan")
    if len(residuals) > 2 and residuals[-1] > 0:
        rate = float((residuals[-1] / residuals[0]) ** (1.0 / (len(residuals) - 1)))
    return IterativeResult(
        x=x, iterations=len(residuals) - 1, converged=converged,
        residual_history=tuple(residuals), error_history=(), measured_rate=rate,
        spectral_radius=float("nan"), diagnostics=tuple(diagnostics),
    )


def gmres(
    A: MatrixLike,
    b: Sequence[float] | np.ndarray,
    *,
    tol: float = 1e-10,
    restart: int | None = None,
    max_iter: int | None = None,
) -> tuple[np.ndarray, int]:
    """GMRES with restarts, for a general (non-symmetric) operator given as a matrix **or a callable**.

    Builds an orthonormal Krylov basis with Arnoldi and solves the least-squares problem in that basis
    through Givens rotations, which is what makes the residual available for free at every step. The
    callable form is what makes it usable inside a Jacobian-free Newton-Krylov iteration, where the
    matrix is never formed.

    Returns ``(x, info)`` with ``info = 0`` on convergence and ``info = 1`` when the iteration budget ran
    out -- a return code rather than an exception, because the caller (JFNK) has a meaningful fallback.
    The limitation: without a preconditioner, the iteration count still grows with ``kappa``; GMRES buys
    applicability to non-symmetric problems, not conditioning.
    """
    rhs = np.asarray(b, dtype=np.float64).ravel()
    n = rhs.size
    bnorm = float(np.linalg.norm(rhs))
    if bnorm == 0.0:
        return np.zeros(n), 0
    m = int(restart) if restart else n
    budget = int(max_iter) if max_iter else 10 * n
    x = np.zeros(n)
    used = 0

    while used < budget:
        r = rhs - _matvec(A, x)
        beta = float(np.linalg.norm(r))
        if beta <= tol * bnorm:
            return x, 0
        width = min(m, budget - used)
        V = np.zeros((n, width + 1))
        H = np.zeros((width + 1, width))
        cs = np.zeros(width)
        sn = np.zeros(width)
        g = np.zeros(width + 1)
        V[:, 0] = r / beta
        g[0] = beta
        k = 0
        for j in range(width):
            w = _matvec(A, V[:, j])
            for i in range(j + 1):
                H[i, j] = float(V[:, i] @ w)
                w -= H[i, j] * V[:, i]
            for i in range(j + 1):  # one reorthogonalisation pass: cheap, and keeps the basis usable
                correction = float(V[:, i] @ w)
                H[i, j] += correction
                w -= correction * V[:, i]
            H[j + 1, j] = float(np.linalg.norm(w))
            if H[j + 1, j] > 0:
                V[:, j + 1] = w / H[j + 1, j]
            for i in range(j):  # apply the rotations from previous steps to this column
                temp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = temp
            denom = float(np.hypot(H[j, j], H[j + 1, j]))
            cs[j] = H[j, j] / denom if denom > 0 else 1.0
            sn[j] = H[j + 1, j] / denom if denom > 0 else 0.0
            H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = 0.0
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]
            k = j + 1
            if abs(g[j + 1]) <= tol * bnorm:
                break

        try:
            y = _back_substitution(H[:k, :k], g[:k]) if k > 0 else np.zeros(0)
        except ValueError:  # a zero pivot in the Krylov problem: report breakdown, do not invent a step
            return x, 1
        x = x + V[:, :k] @ y
        used += max(k, 1)
        if abs(g[k]) <= tol * bnorm:
            return x, 0

    return x, 1


def bicgstab(
    A: np.ndarray,
    b: Sequence[float] | np.ndarray,
    *,
    tol: float = 1e-10,
    max_iter: int | None = None,
) -> IterativeResult:
    """BiCGStab: short recurrence, non-symmetric, no growth in memory -- the workhorse for real systems.

    Combines a bi-Lanczos step (one matrix-vector product per step in each of the two spaces) with a
    local steepest-descent stabilisation. It converges in ``O(n)`` steps for the model problems here,
    but it can break down (``rho`` or ``omega`` zero) on an unlucky system, which is reported rather
    than smoothed over.
    """
    Am = _as_matrix(A)
    rhs = np.asarray(b, dtype=np.float64).ravel()
    n = rhs.size
    limit = int(max_iter) if max_iter else 2 * n
    x = np.zeros(n)
    r = rhs.copy()
    r_hat = r.copy()
    rho = alpha = omega = 1.0
    v = np.zeros(n)
    p = np.zeros(n)
    residuals = [float(np.linalg.norm(r))]
    bnorm = max(residuals[0], np.finfo(float).tiny)
    converged = residuals[0] <= tol * bnorm
    breakdown = False

    for _ in range(limit):
        if converged:
            break
        rho_new = float(r_hat @ r)
        if rho_new == 0.0:
            breakdown = True
            break
        beta = (rho_new / rho) * (alpha / omega) if rho != 0 and omega != 0 else 0.0
        p = r + beta * (p - omega * v)
        v = Am @ p
        denom = float(r_hat @ v)
        if denom == 0.0:
            breakdown = True
            break
        alpha = rho_new / denom
        s = r - alpha * v
        if float(np.linalg.norm(s)) <= tol * bnorm:
            x = x + alpha * p
            residuals.append(float(np.linalg.norm(s)))
            converged = True
            break
        t = Am @ s
        tt = float(t @ t)
        omega = float(t @ s) / tt if tt > 0 else 0.0
        if omega == 0.0:
            breakdown = True
            break
        x = x + alpha * p + omega * s
        r = s - omega * t
        residuals.append(float(np.linalg.norm(r)))
        rho = rho_new
        if residuals[-1] <= tol * bnorm:
            converged = True
            break

    diagnostics: list[Diagnostic] = []
    if breakdown:
        diagnostics.append(explain("LINALG.NO_CONVERGENCE", method="BiCGStab",
                                   note="breakdown: rho or omega became zero"))
    elif not converged:
        diagnostics.append(explain("LINALG.NO_CONVERGENCE", method="BiCGStab",
                                   iterations=len(residuals) - 1))
    rate = float("nan")
    if len(residuals) > 2 and residuals[-1] > 0:
        rate = float((residuals[-1] / residuals[0]) ** (1.0 / (len(residuals) - 1)))
    return IterativeResult(
        x=x, iterations=len(residuals) - 1, converged=converged,
        residual_history=tuple(residuals), error_history=(), measured_rate=rate,
        spectral_radius=float("nan"), diagnostics=tuple(diagnostics),
    )


def jacobi_preconditioner(A: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """The cheapest preconditioner: ``M^{-1} r = r / diag(A)``.

    It removes the scaling of the diagonal, which is all a diagonally dominant operator needs; for a
    Poisson operator it changes the CG count by a constant factor at most. The real gains come from
    :func:`ilu0` or multigrid.
    """
    d = np.diag(_as_matrix(A)).astype(np.float64)

    def apply(r: np.ndarray) -> np.ndarray:
        return np.asarray(r, dtype=np.float64) / d

    return apply


def ilu0(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Incomplete LU with zero fill-in: the factors keep exactly the sparsity pattern of ``A``.

    The elimination drops every update outside the pattern, so ``L U`` approximates ``A`` with an error
    concentrated where the fill would have been. For a banded matrix (the usual Poisson discretisation)
    that is a good approximation, and it is the preconditioner that makes CG practical on large systems.

    Returns ``(L, U)``; apply with :func:`apply_ilu`.
    """
    Am = _as_matrix(A).copy()
    n = Am.shape[0]
    pattern = Am != 0.0
    L = np.eye(n)
    U = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if not pattern[i, j]:
                continue
            s = float(Am[i, j])
            for k in range(min(i, j)):
                s -= L[i, k] * U[k, j]
            if i > j:
                if U[j, j] == 0.0:
                    raise ValueError("zero pivot in ILU(0): the matrix needs reordering or pivoting")
                L[i, j] = s / U[j, j]
            else:
                U[i, j] = s
    return L, U


def apply_ilu(L: np.ndarray, U: np.ndarray, r: Sequence[float] | np.ndarray) -> np.ndarray:
    """Apply ``(L U)^{-1} r`` by forward/back substitution -- the preconditioner's action."""
    y = _forward_substitution(L, np.asarray(r, dtype=np.float64).ravel(), unit_diagonal=True)
    return _back_substitution(U, y)


def multigrid_poisson_1d(
    n: int,
    *,
    levels: int | None = None,
    tol: float = 1e-8,
    max_iter: int = 100,
    pre_smooth: int = 2,
    post_smooth: int = 2,
    omega: float = 2.0 / 3.0,
) -> dict[str, Any]:
    """Solve ``-u'' = pi^2 sin(pi x)``, ``u(0) = u(1) = 0`` with a multigrid V-cycle.

    Coarse-grid correction is what makes the iteration count independent of ``h``: weighted Jacobi removes
    the high-frequency part of the error, the coarse grid removes the low-frequency part, and each level
    sees only its own band. Weighted Jacobi with ``omega = 2/3`` is the smoother, full weighting the
    restriction, linear interpolation the prolongation, and the coarsest grid is solved directly.

    Requires an odd ``n`` (so that ``(n+1)/2`` is an integer at every level); the returned iteration count
    and the error against ``sin(pi x)`` are what the h-independence check compares.
    """
    if n < 5 or n % 2 == 0:
        raise ValueError(f"n must be odd and at least 5 (so every level halves exactly); got {n}")
    h = 1.0 / (n - 1)
    xs = np.linspace(0.0, 1.0, n)
    f = np.pi**2 * np.sin(np.pi * xs)
    u_exact = np.sin(np.pi * xs)

    available = 0
    size = n
    while size > 5:
        size = (size + 1) // 2
        available += 1
    depth = available if levels is None else max(0, min(int(levels), available))

    def smoother(u: np.ndarray, rhs: np.ndarray, hh: float, sweeps: int) -> np.ndarray:
        w = u.copy()
        for _ in range(sweeps):
            w[1:-1] = (1.0 - omega) * w[1:-1] + omega * (
                0.5 * (w[:-2] + w[2:]) + 0.5 * hh * hh * rhs[1:-1]
            )
        return w

    def vcycle(u: np.ndarray, rhs: np.ndarray, hh: float, level: int) -> np.ndarray:
        if level >= depth or u.size <= 5:
            m = u.size - 2
            A = (np.diag(np.full(m, 2.0)) - np.diag(np.ones(m - 1), 1) - np.diag(np.ones(m - 1), -1)) / hh**2
            out = u.copy()
            out[1:-1] = np.linalg.solve(A, rhs[1:-1])
            return out
        u = smoother(u, rhs, hh, pre_smooth)
        residual = np.zeros_like(rhs)
        residual[1:-1] = (2.0 * u[1:-1] - u[:-2] - u[2:]) / hh**2 - rhs[1:-1]
        coarse_size = (u.size + 1) // 2
        coarse = np.zeros(coarse_size)
        for i in range(1, coarse_size - 1):
            # full weighting of the defect, with the sign of the defect equation A e = -r
            coarse[i] = -0.25 * (residual[2 * i - 1] + 2.0 * residual[2 * i] + residual[2 * i + 1])
        e_coarse = vcycle(np.zeros(coarse_size), coarse, 2.0 * hh, level + 1)
        correction = np.zeros(u.size)
        correction[0::2] = e_coarse                     # fine even nodes take the coarse value
        correction[1::2] = 0.5 * (e_coarse[:-1] + e_coarse[1:])  # fine odd nodes interpolate
        u = smoother(u + correction, rhs, hh, post_smooth)
        u[0] = u[-1] = 0.0
        return u

    u = np.zeros(n)
    residuals = [float(np.linalg.norm(f))]
    bnorm = max(residuals[0], np.finfo(float).tiny)
    for _ in range(max_iter):
        u = vcycle(u, f, h, 0)
        r = np.zeros(n)
        r[1:-1] = f[1:-1] - (2.0 * u[1:-1] - u[:-2] - u[2:]) / h**2
        residuals.append(float(np.linalg.norm(r)))
        if residuals[-1] <= tol * bnorm:
            break
    return {
        "n": n,
        "u": u,
        "iterations": len(residuals) - 1,
        "residual_history": residuals,
        "error": float(np.max(np.abs(u - u_exact))),
        "levels": depth,
    }


def multigrid_h_independence(sizes: Sequence[int] = (33, 65, 129, 257), **kwargs: Any) -> dict[str, Any]:
    """Run the V-cycle across mesh sizes and check that the iteration count does not grow with ``n``.

    This is the one measurement that distinguishes a genuine multigrid from a smoother with a coarse-grid
    decoration: iterations constant across a 8x range of ``n``, and the error falling like ``h^2`` (the
    truncation order of the scheme) rather than stagnating.
    """
    runs = [multigrid_poisson_1d(int(n), **kwargs) for n in sizes]
    counts = [r["iterations"] for r in runs]
    errors = [r["error"] for r in runs]
    hs = [1.0 / (n - 1) for n in sizes]
    finite = [i for i, e in enumerate(errors) if e > 0]
    order = float("nan")
    if len(finite) >= 2:
        idx = finite[-2:]
        order = float(np.log(errors[idx[0]] / errors[idx[1]]) / np.log(hs[idx[0]] / hs[idx[1]]))
    diagnostics: list[Diagnostic] = []
    if max(counts) > 2.0 * max(min(counts), 1):
        diagnostics.append(explain("LINALG.MULTIGRID_H_DEPENDENT", counts=counts, sizes=list(sizes)))
    return {
        "sizes": [int(n) for n in sizes],
        "iterations": counts,
        "errors": errors,
        "error_order": order,
        "h_independent": max(counts) <= 2 * max(min(counts), 1),
        "diagnostics": [d.to_dict() for d in diagnostics],
    }


# --------------------------------------------------------------------------- eigenvalues
@dataclass(frozen=True)
class EigenResult:
    """An eigenvalue computation with the rate it converged at."""

    eigenvalue: float
    eigenvector: np.ndarray
    iterations: int
    converged: bool
    measured_rate: float
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form."""
        return {
            "eigenvalue": float(self.eigenvalue),
            "eigenvector": [float(v) for v in self.eigenvector],
            "iterations": self.iterations,
            "converged": self.converged,
            "measured_rate": self.measured_rate,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def power_iteration(
    A: np.ndarray, *, tol: float = 1e-13, max_iter: int = 1000, x0: Sequence[float] | None = None
) -> EigenResult:
    """The dominant eigenvalue by repeated multiplication, converging at the ratio ``|lambda_2/lambda_1|``.

    The measured rate is the geometric decay of the *eigenvalue estimate*, which with a Rayleigh quotient
    is ``(lambda_next/lambda_1)^2`` and not ``lambda_next/lambda_1``: the quotient has a second-order
    error in the eigenvector error, so the exponent doubles. ``lambda_next`` is the second eigenvalue the
    start vector is **not** orthogonal to -- the default all-ones vector is orthogonal to a
    sign-alternating mode, and the rate then reflects the mode after it, which is measured, not assumed.

    That ratio is the whole story of the method's cost: if the two largest eigenvalues are close, the
    method is hopeless and no number of iterations fixes it. The rate is fit on the geometric decay of
    the estimates, excluding the last few steps where the error has reached the rounding floor --
    including those would report a rate of zero for every matrix.

    Limitations: a start vector orthogonal to the dominant eigenvector never converges (the default
    ``x0`` is all ones, which is orthogonal to a sign-alternating dominant mode), and for a non-symmetric
    matrix the eigenvalue estimate can converge to a complex pair only in the real part.
    """
    Am = _as_matrix(A)
    n = Am.shape[0]
    x = np.ones(n) if x0 is None else np.asarray(x0, dtype=np.float64).ravel()
    x = x / np.linalg.norm(x)
    estimates: list[float] = []
    for k in range(1, max_iter + 1):
        y = Am @ x
        lam = float(x @ y)  # Rayleigh quotient: the best estimate for a symmetric A
        norm = float(np.linalg.norm(y))
        if norm == 0.0:
            break
        x = y / norm
        estimates.append(lam)
        if len(estimates) >= 3 and abs(estimates[-1] - estimates[-2]) <= tol * max(1.0, abs(lam)):
            return EigenResult(eigenvalue=lam, eigenvector=x, iterations=k, converged=True,
                               measured_rate=_rate_from_history(estimates))
    diagnostics = (explain("LINALG.NO_CONVERGENCE", method="power iteration", iterations=max_iter),)
    return EigenResult(eigenvalue=estimates[-1] if estimates else float("nan"), eigenvector=x,
                       iterations=max_iter, converged=False,
                       measured_rate=_rate_from_history(estimates), diagnostics=diagnostics)


def _rate_from_history(estimates: Sequence[float]) -> float:
    """Geometric decay rate of a converging scalar estimate, ignoring the rounding plateau at the end.

    Errors below ``1e-6`` of the largest error are at the floor of the measurement (they are the noise of
    the last few iterations), and they are what would make every rate look like zero.
    """
    if len(estimates) < 4:
        return float("nan")
    final = estimates[-1]
    errors = np.abs(np.asarray(estimates, dtype=np.float64) - final)
    keep = errors > 1e-6 * max(float(np.max(errors)), np.finfo(float).tiny)
    e = errors[keep]
    if e.size < 3:
        return float("nan")
    ratios = e[1:] / e[:-1]
    return float(np.median(ratios[-3:]))


def inverse_iteration(
    A: np.ndarray, sigma: float = 0.0, *, tol: float = 1e-12, max_iter: int = 100
) -> EigenResult:
    """The eigenvalue closest to ``sigma``, by power iteration on ``(A - sigma I)^{-1}``.

    The shift is what makes it useful: the rate is ``|lambda_near - sigma| / |lambda_next - sigma|``, so
    a shift near the wanted eigenvalue converges in a handful of steps. It also gives eigenvectors when
    the eigenvalues are clustered, where the plain power method cannot separate them.
    """
    base = _as_matrix(A)
    n = base.shape[0]
    shifted = base - float(sigma) * np.eye(n)
    x = np.ones(n) / np.sqrt(n)
    lam_old = float("inf")
    ratios: list[float] = []
    lam = float("nan")
    for k in range(1, max_iter + 1):
        try:
            y = np.linalg.solve(shifted, x)
        except np.linalg.LinAlgError:
            diagnostics = (explain("LINALG.SINGULAR", note="the shift is an exact eigenvalue"),)
            return EigenResult(eigenvalue=float(sigma), eigenvector=x, iterations=k, converged=True,
                               measured_rate=0.0, diagnostics=diagnostics)
        mu = float(x @ y)  # x^T (A - sigma I)^{-1} x, with x already normalised
        norm = float(np.linalg.norm(y))
        if norm == 0.0 or mu == 0.0:
            break
        x = y / norm
        lam = float(sigma) + 1.0 / mu
        if np.isfinite(lam_old):
            ratios.append(abs((lam - lam_old) / lam) if lam != 0 else float("inf"))
        if abs(lam - lam_old) <= tol * max(1.0, abs(lam)):
            rate = float(np.median(ratios[-3:])) if len(ratios) >= 3 else float("nan")
            return EigenResult(eigenvalue=lam, eigenvector=x, iterations=k, converged=True,
                               measured_rate=rate)
        lam_old = lam
    rate = float(np.median(ratios[-3:])) if len(ratios) >= 3 else float("nan")
    return EigenResult(eigenvalue=lam, eigenvector=x, iterations=max_iter, converged=False,
                       measured_rate=rate,
                       diagnostics=(explain("LINALG.NO_CONVERGENCE", method="inverse iteration",
                                            iterations=max_iter),))


def rayleigh_quotient_iteration(
    A: np.ndarray, *, tol: float = 1e-14, max_iter: int = 100, x0: Sequence[float] | None = None
) -> EigenResult:
    """Inverse iteration with the shift updated to the current Rayleigh quotient: cubic convergence.

    Updating ``sigma`` to ``x^T A x / x^T x`` every step turns the linear rate of a fixed shift into
    order 3 for a symmetric matrix -- the fastest way to a single eigenpair without a full
    factorisation, and the reason it is used for large problems where only one pair is wanted.
    """
    Am = _as_matrix(A)
    n = Am.shape[0]
    x = np.ones(n) / np.sqrt(n) if x0 is None else np.asarray(x0, dtype=np.float64).ravel()
    x = x / np.linalg.norm(x)
    lam = float(x @ Am @ x)
    for k in range(1, max_iter + 1):
        try:
            y = np.linalg.solve(Am - lam * np.eye(n), x)
        except np.linalg.LinAlgError:
            return EigenResult(eigenvalue=lam, eigenvector=x, iterations=k, converged=True,
                               measured_rate=0.0)
        x = y / np.linalg.norm(y)
        lam_new = float(x @ Am @ x)
        if abs(lam_new - lam) <= tol * max(1.0, abs(lam_new)):
            return EigenResult(eigenvalue=lam_new, eigenvector=x, iterations=k, converged=True,
                               measured_rate=0.0)
        lam = lam_new
    return EigenResult(eigenvalue=lam, eigenvector=x, iterations=max_iter, converged=False,
                       measured_rate=float("nan"),
                       diagnostics=(explain("LINALG.NO_CONVERGENCE", method="Rayleigh quotient iteration",
                                            iterations=max_iter),))


def qr_algorithm(A: np.ndarray, *, tol: float = 1e-14, max_iter: int = 2000) -> np.ndarray:
    """All eigenvalues of a symmetric matrix by the shifted QR algorithm, deflating as it goes.

    The shift is the Rayleigh quotient of the trailing ``2x2`` block (Wilkinson's shift), which avoids
    the slow convergence of the unshifted iteration on close eigenvalues. Symmetry is what makes QR
    preserve the symmetric structure and the eigenvalues real; a non-symmetric matrix is symmetrised for
    the purpose of the computation, which is stated here because it would otherwise be a silent change to
    the problem.
    """
    Am = _as_matrix(A)
    if not np.allclose(Am, Am.T, atol=1e-12):
        Am = 0.5 * (Am + Am.T)
    T = Am.copy()
    n = T.shape[0]
    eigenvalues = np.zeros(n)
    m = n
    iterations = 0
    while m > 1 and iterations < max_iter:
        if abs(T[m - 1, m - 2]) <= tol * (abs(T[m - 2, m - 2]) + abs(T[m - 1, m - 1]) + 1e-300):
            eigenvalues[m - 1] = T[m - 1, m - 1]
            T = T[: m - 1, : m - 1]
            m -= 1
            continue
        a, b = T[m - 2, m - 2], T[m - 2, m - 1]
        c, d = T[m - 1, m - 2], T[m - 1, m - 1]
        delta = (a - d) / 2.0
        disc = np.sqrt(delta**2 + b * c)
        mu = d - b * c / (delta + np.copysign(disc, delta)) if disc != 0.0 else d
        Q, R = np.linalg.qr(T - mu * np.eye(m))
        T = R @ Q + mu * np.eye(m)
        iterations += 1
    eigenvalues[:m] = np.diag(T)
    return np.sort(eigenvalues)


def lanczos(A: np.ndarray, m: int | None = None) -> dict[str, Any]:
    """Lanczos: a symmetric matrix reduced to tridiagonal form, whose eigenvalues are the extremes.

    After ``m`` steps the eigenvalues of the small tridiagonal ``T`` (the Ritz values) approximate the
    extreme eigenvalues of ``A`` -- the outer ones first, which is why Lanczos is used for the largest or
    smallest few eigenvalues of a large sparse matrix and not for all of them. The measured error of the
    extremes against a direct solve is reported.

    Limitation: in floating point Lanczos loses orthogonality and duplicates eigenvalues without
    explicit reorthogonalisation; the Gram-Schmidt pass below costs ``O(m^2 n)`` and is what makes the
    Ritz values trustworthy at teaching size.
    """
    Am = _as_matrix(A)
    n = Am.shape[0]
    steps = int(m) if m else min(n, 50)
    steps = max(1, min(steps, n))
    Q = np.zeros((n, steps + 1))
    alpha = np.zeros(steps)
    beta = np.zeros(steps + 1)
    Q[:, 0] = np.ones(n) / np.sqrt(n)
    used = steps

    for j in range(steps):
        w = Am @ Q[:, j]
        alpha[j] = float(Q[:, j] @ w)
        w = w - alpha[j] * Q[:, j]
        if j > 0:
            w = w - beta[j] * Q[:, j - 1]
        for i in range(j + 1):  # full reorthogonalisation
            w -= float(Q[:, i] @ w) * Q[:, i]
        beta[j + 1] = float(np.linalg.norm(w))
        if beta[j + 1] < 1e-12:  # invariant subspace reached: the Krylov space is exhausted
            used = j + 1
            break
        Q[:, j + 1] = w / beta[j + 1]

    T = (np.diag(alpha[:used])
         + np.diag(beta[1:used], 1)
         + np.diag(beta[1:used], -1))
    ritz = np.sort(np.linalg.eigvalsh(T))
    direct = np.sort(np.linalg.eigvalsh(Am))
    return {
        "ritz": ritz,
        "steps": used,
        "T": T,
        "exact": direct,
        "extreme_error": float(max(abs(ritz[0] - direct[0]), abs(ritz[-1] - direct[-1]))),
    }


# --------------------------------------------------------------------------- SVD
@dataclass(frozen=True)
class SVDResult:
    """A low-rank approximation with the error the truncation actually made."""

    approximation: np.ndarray
    rank: int
    singular_values: np.ndarray
    frobenius_error: float
    predicted_error: float
    relative_error: float
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form."""
        return {
            "approximation": [[float(v) for v in row] for row in self.approximation],
            "rank": self.rank,
            "singular_values": [float(v) for v in self.singular_values],
            "frobenius_error": self.frobenius_error,
            "predicted_error": self.predicted_error,
            "relative_error": self.relative_error,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def svd_low_rank(A: np.ndarray, rank: int) -> SVDResult:
    """Best rank-``r`` approximation, with Eckart-Young checked against the measured error.

    Eckart-Young says the truncated SVD is optimal and that its Frobenius error is exactly
    ``sqrt(sum of the discarded squared singular values)``. Both the measured error and that prediction
    are returned: a mismatch (beyond rounding) would mean the implementation, not the theorem, is wrong.
    """
    Am = _as_matrix(A, square=False)
    U, s, Vt = np.linalg.svd(Am, full_matrices=False)
    r = int(rank)
    if not 1 <= r <= s.size:
        raise ValueError(f"rank must be between 1 and {s.size}; got {rank}")
    approx = (U[:, :r] * s[:r]) @ Vt[:r, :]
    measured = float(np.linalg.norm(Am - approx, "fro"))
    predicted = float(np.sqrt(np.sum(s[r:] ** 2)))
    total = float(np.linalg.norm(Am, "fro"))
    diagnostics: list[Diagnostic] = []
    if total > 0 and measured / total > 0.5:
        diagnostics.append(
            explain("LINALG.ILL_CONDITIONED", note="the truncation discards more than half the energy",
                    relative_error=measured / total, rank=r)
        )
    return SVDResult(
        approximation=approx, rank=r, singular_values=s, frobenius_error=measured,
        predicted_error=predicted, relative_error=float(measured / total) if total > 0 else float("nan"),
        diagnostics=tuple(diagnostics),
    )


def svd_pseudoinverse(A: np.ndarray, *, rcond: float = 1e-12) -> np.ndarray:
    """Moore-Penrose pseudo-inverse, with singular values below ``rcond * sigma_max`` dropped.

    The threshold is the rank decision, and it is the only free parameter that matters: too small and the
    ``1/sigma`` factors amplify noise, too large and real directions are discarded. Returning it as an
    explicit argument is the point -- there is no such thing as "the" pseudo-inverse of a noisy matrix.
    """
    Am = _as_matrix(A, square=False)
    U, s, Vt = np.linalg.svd(Am, full_matrices=False)
    cutoff = rcond * (s[0] if s.size else 0.0)
    inverse = np.array([1.0 / v if v > cutoff else 0.0 for v in s])
    return (Vt.T * inverse) @ U.T


def truncated_svd_solve(A: np.ndarray, b: Sequence[float] | np.ndarray, rank: int) -> dict[str, Any]:
    """Truncated-SVD regularised solve of ``A x = b``: a Tikhonov alternative with a hard rank cut.

    Returns the solution, its norm and the residual for each rank up to ``rank`` -- the two curves whose
    L-shape is the usual way to pick the truncation (see ``pinneapple_analysis.inverse_problems`` for the
    L-curve machinery itself).
    """
    Am = _as_matrix(A, square=False)
    rhs = np.asarray(b, dtype=np.float64).ravel()
    U, s, Vt = np.linalg.svd(Am, full_matrices=False)
    solutions = []
    for r in range(1, int(rank) + 1):
        coeff = np.array([(U[:, i] @ rhs) / s[i] if s[i] > 0 else 0.0 for i in range(r)])
        x = Vt[:r, :].T @ coeff
        solutions.append({
            "rank": r,
            "solution": [float(v) for v in x],
            "solution_norm": float(np.linalg.norm(x)),
            "residual": float(np.linalg.norm(rhs - Am @ x)),
        })
    return {
        "singular_values": [float(v) for v in s],
        "solutions": solutions,
    }

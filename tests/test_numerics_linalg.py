"""Tests for ``pinneapple_analysis.numerics.linalg`` (#1061).

The measurements that matter here are comparisons, not isolated values: the computed factorisation
against the matrix it came from, the observed convergence rate against the spectral radius that predicts
it, the CG iteration count against ``sqrt(kappa)``, and the multigrid iteration count across mesh sizes.
Where a closed form exists (Hilbert's condition number, the model Poisson spectral radii, Eckart-Young)
it is used as the reference.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_analysis.numerics.linalg import (
    apply_ilu,
    bicgstab,
    cholesky,
    cholesky_solve,
    condition_number,
    conditioning_study,
    conjugate_gradient,
    gauss_seidel,
    gmres,
    householder_qr,
    ilu0,
    inverse_iteration,
    iterative_matrix,
    iterative_refinement,
    jacobi,
    jacobi_preconditioner,
    lanczos,
    lu_factor,
    lu_solve,
    modified_gram_schmidt,
    multigrid_h_independence,
    multigrid_poisson_1d,
    orthogonality_loss,
    power_iteration,
    qr_algorithm,
    rayleigh_quotient_iteration,
    sor,
    sor_optimal_omega,
    svd_low_rank,
    svd_pseudoinverse,
    truncated_svd_solve,
)


def _hilbert(n: int) -> np.ndarray:
    return np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])


def _poisson_1d(n: int) -> np.ndarray:
    return 2.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)


def _convection_diffusion(n: int, peclet: float = 8.0) -> np.ndarray:
    """Non-symmetric: ``-u'' + Pe u'`` central differences, Dirichlet."""
    h = 1.0 / (n + 1)
    A = (2.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)) / h**2
    A += peclet * (np.eye(n, k=1) - np.eye(n, k=-1)) / (2.0 * h)
    return A


# --------------------------------------------------------------------------- direct solvers
def test_lu_partial_pivoting_solves_and_reports_a_small_growth_factor() -> None:
    """For a diagonally dominant matrix the growth factor is 1 and the residual is at the rounding level."""
    A = np.array([[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 4.0]])
    b = np.array([1.0, 2.0, 3.0])
    factor = lu_factor(A)
    x = lu_solve(factor, b)
    assert np.allclose(A @ x, b, rtol=1e-14)
    assert factor.growth_factor == pytest.approx(1.0)
    assert not factor.singular
    # a matrix that *needs* pivoting: without it the first pivot would be 1e-18 and the growth enormous
    tiny = np.array([[1e-18, 1.0], [1.0, 1.0]])
    assert lu_factor(tiny).growth_factor < 10.0
    assert np.allclose(lu_solve(lu_factor(tiny), np.ones(2)), np.linalg.solve(tiny, np.ones(2)), rtol=1e-10)


def test_lu_flags_a_singular_matrix() -> None:
    factor = lu_factor(np.array([[1.0, 2.0], [2.0, 4.0]]))
    assert factor.singular
    assert "LINALG.SINGULAR" in [d.code for d in factor.diagnostics]
    with pytest.raises(ValueError):
        lu_solve(factor, np.ones(2))


def test_cholesky_reproduces_the_matrix_and_refuses_an_indefinite_one() -> None:
    A = _poisson_1d(6)
    L = cholesky(A)
    assert np.allclose(L @ L.T, A)
    assert cholesky_solve(L, np.ones(6)) == pytest.approx(np.linalg.solve(A, np.ones(6)))
    with pytest.raises(ValueError):
        cholesky(np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_householder_qr_keeps_orthogonality_where_gram_schmidt_loses_it() -> None:
    """On a Hilbert matrix the two methods differ by many orders of magnitude in ``max|Q^T Q - I|``."""
    A = _hilbert(12)
    q_householder, r_householder = householder_qr(A)
    q_gram, r_gram = modified_gram_schmidt(A, reorthogonalize=False)
    q_reorth, _ = modified_gram_schmidt(A, reorthogonalize=True)
    assert np.allclose(q_householder @ r_householder, A, atol=1e-12)
    assert np.allclose(q_gram @ r_gram, A, atol=1e-12)
    assert orthogonality_loss(q_householder) < 1e-14
    assert orthogonality_loss(q_gram) > 1e3 * orthogonality_loss(q_householder)
    # a second reorthogonalisation pass is what recovers it
    assert orthogonality_loss(q_reorth) < 1e-14


# --------------------------------------------------------------------------- conditioning
def test_condition_number_matches_the_closed_form_for_hilbert() -> None:
    kappa = condition_number(_hilbert(5))
    assert kappa == pytest.approx(4.7660725e5, rel=1e-4)
    assert condition_number(np.eye(4)) == pytest.approx(1.0)
    assert condition_number(np.zeros((2, 2))) == float("inf")


def test_solve_error_grows_like_the_condition_number() -> None:
    """``error ~ kappa * eps``: the log-log slope of the measured error against ``kappa`` must be ~1."""
    study = conditioning_study([_hilbert(n) for n in (4, 6, 8, 10, 12)])
    assert 0.7 < study.slope < 1.35, study.slope
    # and the bound is not wildly pessimistic: the measured error is within four decades of kappa * eps
    ratios = study.relative_errors / study.predicted_errors
    assert np.all(ratios > 1e-3) and np.all(ratios < 1e2)
    assert "LINALG.ILL_CONDITIONED" in [d.code for d in study.diagnostics]


def test_iterative_refinement_helps_only_while_kappa_eps_is_below_one() -> None:
    """Refinement recovers digits from the residual, and stops being able to when ``kappa * eps ~ 1``.

    Both outcomes are measured: Hilbert(8) has ``kappa * eps = 3.4e-6`` and the error drops; Hilbert(10)
    has ``kappa * eps = 3.6e-3`` and no amount of refinement in the same precision improves it.
    """
    x_true = np.ones(8)
    fixable = iterative_refinement(_hilbert(8), _hilbert(8) @ x_true)
    assert fixable["condition_number"] * np.finfo(float).eps < 1e-4
    assert fixable["improved"]
    # it stops as soon as a further correction makes the error worse, so the best iterate is the one to look at
    assert min(fixable["errors"]) < fixable["errors"][0] / 2.0

    hopeless = iterative_refinement(_hilbert(10), _hilbert(10) @ np.ones(10))
    assert hopeless["condition_number"] * np.finfo(float).eps > 1e-4
    assert not hopeless["improved"]
    # the residual is at the rounding level in both cases: it is not the residual that was ever wrong
    assert fixable["residuals"][-1] < 1e-14 and hopeless["residuals"][-1] < 1e-14


# --------------------------------------------------------------------------- stationary iterations
def test_stationary_iteration_rate_is_the_spectral_radius() -> None:
    """The measured per-iteration reduction must equal the radius that predicted it (both measured)."""
    A = _poisson_1d(20)
    b = np.ones(20)
    for solver in (jacobi, gauss_seidel):
        result = solver(A, b, tol=1e-8, max_iter=2000)
        assert result.converged
        assert result.measured_rate == pytest.approx(result.spectral_radius, rel=0.25)
        assert 0.0 < result.spectral_radius < 1.0


def test_gauss_seidel_radius_is_the_square_of_jacobi_for_the_model_problem() -> None:
    """A closed-form prediction, measured: ``rho_GS = rho_Jacobi^2`` for the 1-D Poisson matrix."""
    A = _poisson_1d(20)
    rho_jacobi = float(np.max(np.abs(np.linalg.eigvals(iterative_matrix(A, "jacobi")))))
    rho_gs = float(np.max(np.abs(np.linalg.eigvals(iterative_matrix(A, "gauss_seidel")))))
    assert rho_gs == pytest.approx(rho_jacobi**2, rel=1e-10)
    # and the iteration counts follow: GS needs about half the Jacobi count for the same reduction
    b = np.ones(20)
    gs = gauss_seidel(A, b, tol=1e-10, max_iter=4000)
    jc = jacobi(A, b, tol=1e-10, max_iter=4000)
    assert gs.converged and jc.converged
    assert gs.iterations < 0.75 * jc.iterations


def test_sor_optimal_omega_matches_the_model_problem_formula() -> None:
    """``omega_opt = 2/(1+sin(pi h))`` for the 1-D Poisson matrix: the scan must find it."""
    n = 30
    h = 1.0 / (n + 1)
    A = _poisson_1d(n)
    scan = sor_optimal_omega(A, samples=39)
    assert scan["optimal_omega"] == pytest.approx(2.0 / (1.0 + np.sin(np.pi * h)), abs=0.05)
    assert scan["min_spectral_radius"] < scan["gauss_seidel_radius"]
    # SOR at the measured optimum beats Gauss-Seidel in real iterations, which is the point of the knob
    b = np.ones(n)
    fast = sor(A, b, omega=scan["optimal_omega"], tol=1e-10, max_iter=4000)
    slow = gauss_seidel(A, b, tol=1e-10, max_iter=4000)
    assert fast.converged and slow.converged
    assert fast.iterations < 0.6 * slow.iterations


def test_sor_rejects_an_out_of_range_omega() -> None:
    with pytest.raises(ValueError):
        sor(_poisson_1d(5), np.ones(5), omega=2.5)


# --------------------------------------------------------------------------- Krylov
def test_conjugate_gradient_iterations_grow_like_sqrt_kappa() -> None:
    """The classic prediction, measured: doubling ``sqrt(kappa)`` roughly doubles the iteration count."""
    counts = []
    roots = []
    for n in (10, 20, 40, 80):
        A = _poisson_1d(n)
        b = np.ones(n)
        result = conjugate_gradient(A, b, tol=1e-10)
        assert result.converged
        assert np.linalg.norm(b - A @ result.x) < 1e-8
        counts.append(result.iterations)
        roots.append(np.sqrt(condition_number(A)))
    slope = np.polyfit(np.log(roots), np.log(counts), 1)[0]
    assert 0.6 < slope < 1.4, (slope, counts)


def test_jacobi_preconditioning_cuts_the_cg_iteration_count_on_a_scaled_operator() -> None:
    """Scaling the diagonal out is the cheapest preconditioner and the easiest gain to measure."""
    n = 24
    A = _poisson_1d(n)
    scaling = np.diag(np.logspace(0.0, 3.0, n))  # a badly scaled but still SPD operator
    A = scaling @ A @ scaling
    b = np.ones(n)
    # without a preconditioner this operator needs ~sqrt(kappa) ~ 2700 iterations; the default budget of
    # 2n = 48 is deliberately not enough, so it is raised here
    plain = conjugate_gradient(A, b, tol=1e-10, max_iter=4000)
    preconditioned = conjugate_gradient(A, b, tol=1e-10, M=jacobi_preconditioner(A))
    assert plain.converged and preconditioned.converged
    assert preconditioned.iterations < plain.iterations
    assert np.allclose(A @ preconditioned.x, b, atol=1e-6)


def test_conjugate_gradient_reports_a_non_positive_definite_matrix() -> None:
    """``p^T A p <= 0`` is a breakdown of the assumption, not a slow convergence."""
    # indefinite, eigenvalues 3 and -1: start along the negative-curvature direction and CG breaks down
    A = np.array([[1.0, 2.0], [2.0, 1.0]])
    result = conjugate_gradient(A, np.array([1.0, -1.0]))
    assert not result.converged
    assert "LINALG.NO_CONVERGENCE" in [d.code for d in result.diagnostics]


def test_gmres_and_bicgstab_solve_a_non_symmetric_system() -> None:
    A = _convection_diffusion(30, peclet=8.0)
    b = np.ones(30)
    x, info = gmres(A, b, tol=1e-12)
    assert info == 0
    assert np.linalg.norm(b - A @ x) / np.linalg.norm(b) < 1e-10
    result = bicgstab(A, b, tol=1e-12)
    assert result.converged
    assert np.linalg.norm(b - A @ result.x) / np.linalg.norm(b) < 1e-9
    # a restart is a memory/orthogonality trade-off, not a different answer
    x_restarted, info_restarted = gmres(A, b, tol=1e-10, restart=5)
    assert info_restarted == 0
    assert np.allclose(x_restarted, x, atol=1e-6)


def test_ilu0_preconditions_better_than_jacobi_on_a_banded_matrix() -> None:
    """Zero fill-in keeps the pattern of ``A``, and the preconditioned operator is far closer to identity.

    On a tridiagonal matrix there is no fill to drop, so ILU(0) is an exact factorisation and the
    preconditioned operator *is* the identity -- a useful limit to verify before trusting the 2-D case,
    where the dropped fill shows up as a real residual spectral radius.
    """
    A = _poisson_1d(40)
    L, U = ilu0(A)
    assert np.array_equal(np.tril(A, -1) != 0.0, np.tril(L, -1) != 0.0)  # exactly the pattern of A
    assert np.array_equal(np.triu(A, 1) != 0.0, np.triu(U, 1) != 0.0)
    tridiagonal_rho = np.max(np.abs(np.linalg.eigvals(np.eye(40) - np.linalg.solve(L @ U, A))))
    jacobi_rho = np.max(np.abs(np.linalg.eigvals(np.eye(40) - np.diag(1.0 / np.diag(A)) @ A)))
    assert tridiagonal_rho < 1e-12
    assert jacobi_rho > 0.9

    # the 5-point Laplacian on a 7x7 grid: fill really is dropped, and ILU(0) still beats the diagonal
    n = 7
    T = _poisson_1d(n)
    A2 = np.kron(np.eye(n), T) + np.kron(T, np.eye(n))
    L2, U2 = ilu0(A2)
    ilu_rho = np.max(np.abs(np.linalg.eigvals(np.eye(n * n) - np.linalg.solve(L2 @ U2, A2))))
    jacobi_rho_2d = np.max(np.abs(np.linalg.eigvals(np.eye(n * n) - np.diag(1.0 / np.diag(A2)) @ A2)))
    assert 0.0 < ilu_rho < 0.7 < jacobi_rho_2d
    # here (LU)^{-1} A is close to the identity but not equal to it: the dropped fill is exactly ilu_rho
    assert np.allclose(apply_ilu(L2, U2, A2 @ np.ones(n * n)), np.ones(n * n), atol=ilu_rho * 4.0)


def test_multigrid_iteration_count_is_independent_of_the_mesh() -> None:
    """The defining property: the same number of V-cycles at every h, with second-order accuracy."""
    report = multigrid_h_independence((33, 65, 129, 257))
    assert report["h_independent"]
    counts = report["iterations"]
    assert max(counts) - min(counts) <= 2
    assert report["error_order"] == pytest.approx(2.0, abs=0.15)
    assert not report["diagnostics"]
    # a single run reports an error consistent with the h^2 truncation of the scheme
    single = multigrid_poisson_1d(65)
    assert single["error"] < 5e-4
    assert single["residual_history"][-1] < single["residual_history"][0] * 1e-7


# --------------------------------------------------------------------------- eigenvalues
def test_power_iteration_rate_is_the_squared_eigenvalue_ratio() -> None:
    """With a Rayleigh quotient the estimate converges at ``(lambda_2/lambda_1)^2``, and that is measured."""
    A = np.array([[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 4.0]])
    exact = np.sort(np.linalg.eigvalsh(A))
    default_start = power_iteration(A)  # all-ones: orthogonal to the sign-alternating eigenvector
    other_start = power_iteration(A, x0=[2.0, 1.0, 1.0])
    for result in (default_start, other_start):
        assert result.converged
        assert result.eigenvalue == pytest.approx(exact[-1], rel=1e-12)
    # the rate is (ratio of the second eigenvalue *present in the start vector*) squared, so the default
    # start skips lambda_2 entirely and converges at the ratio of lambda_3
    assert other_start.measured_rate == pytest.approx((exact[1] / exact[-1]) ** 2, rel=0.05)
    assert default_start.measured_rate == pytest.approx((exact[0] / exact[-1]) ** 2, rel=0.05)


def test_inverse_and_rayleigh_quotient_iteration_find_the_wanted_eigenpair() -> None:
    A = np.diag([1.0, 2.0, 5.0, 9.0])
    nearest = inverse_iteration(A, sigma=4.6)
    assert nearest.converged
    assert nearest.eigenvalue == pytest.approx(5.0, rel=1e-10)
    fastest = rayleigh_quotient_iteration(A, x0=[0.0, 0.1, 1.0, 0.2])
    assert fastest.eigenvalue == pytest.approx(5.0, rel=1e-10)
    assert fastest.iterations <= nearest.iterations + 2


def test_qr_algorithm_reproduces_all_eigenvalues() -> None:
    rng = np.random.default_rng(5)
    B = rng.normal(size=(10, 10))
    A = 0.5 * (B + B.T)  # symmetric, so the eigenvalues are real
    assert np.allclose(qr_algorithm(A), np.sort(np.linalg.eigvalsh(A)), atol=1e-10)


def test_lanczos_extremes_converge_before_the_interior_ones() -> None:
    """``m`` steps reproduce the outer eigenvalues and leave the middle ones alone -- the reason to use it."""
    n = 100
    A = np.diag(np.linspace(1.0, 2.0, n))
    exact = np.sort(np.linalg.eigvalsh(A))
    report = lanczos(A, m=25)
    assert report["steps"] == 25
    assert report["extreme_error"] < 1e-2
    interior_error = abs(report["ritz"][report["ritz"].size // 2] - exact[n // 2])
    assert interior_error > report["extreme_error"]


# --------------------------------------------------------------------------- SVD
def test_low_rank_svd_error_equals_the_eckart_young_prediction() -> None:
    """The theorem gives the error in closed form; the computation must match it, not merely be small."""
    rng = np.random.default_rng(9)
    A = rng.normal(size=(12, 8))
    for rank in (1, 3, 6):
        result = svd_low_rank(A, rank)
        assert result.frobenius_error == pytest.approx(result.predicted_error, rel=1e-10)
        assert result.rank == rank
    # truncating to the full rank reproduces the matrix exactly
    assert svd_low_rank(A, 8).frobenius_error < 1e-12


def test_pseudoinverse_inverts_the_restricted_operator() -> None:
    rng = np.random.default_rng(13)
    U = rng.normal(size=(6, 2))
    A = U @ U.T  # rank 2, symmetric, singular
    pinv = svd_pseudoinverse(A)
    # A pinv A = A for a rank-2 matrix, and pinv A pinv = pinv
    assert np.allclose(A @ pinv @ A, A, atol=1e-10)
    assert np.allclose(pinv @ A @ pinv, pinv, atol=1e-10)
    with pytest.raises(ValueError):
        svd_low_rank(A, 0)


def test_truncated_svd_solve_trades_residual_for_norm() -> None:
    """The L-curve, in two numbers: a higher rank lowers the residual and raises the solution norm."""
    rng = np.random.default_rng(17)
    A = rng.normal(size=(20, 6))
    b = A @ np.array([1.0, -1.0, 0.5, 0.0, 2.0, 1.0])
    report = truncated_svd_solve(A, b, rank=6)
    residuals = [s["residual"] for s in report["solutions"]]
    norms = [s["solution_norm"] for s in report["solutions"]]
    assert residuals[-1] < residuals[0]
    assert norms[-1] > norms[0]
    assert residuals[-1] < 1e-10  # full rank: the system is consistent, so the residual vanishes

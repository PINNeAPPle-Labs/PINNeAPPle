"""Tests for ``pinneapple_analysis.linalg`` -- the ``pp.linalg`` layer (X18, #202).

Each piece is checked against a reference that exists independently of the implementation:

* POD against Eckart-Young (the truncation error is ``sqrt`` of the energy it discarded) and against a
  dataset built to have exactly three modes.
* Modal analysis against the hand-solved eigenvalues of a two-degree-of-freedom spring-mass-damper, and
  against a Jordan block, whose eigenvalues are the textbook example of ones that cannot be quoted.
* The condition estimator against ``np.linalg.cond(A, 1)``, which is the number it is trying to reach.
* Arnoldi against the exact spectrum of a diagonal matrix, where a full Krylov space must terminate with
  a breakdown and reproduce every eigenvalue.
* The adjoint sensitivity against a central difference of the re-solved system.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from pinneapple_analysis.linalg import (
    LinearOperator,
    adjoint_sensitivity,
    arnoldi,
    condition_estimate,
    condition_number,
    modal_analysis,
    pod,
)
from pinneapple_analysis.numerics.linalg import svd_low_rank


# --------------------------------------------------------------------------- LinearOperator
def test_an_operator_reproduces_its_matrix_and_its_adjoint():
    A = np.array([[2.0, -1.0, 0.0], [0.0, 3.0, 1.0]])
    op = LinearOperator.from_matrix(A, name="M")
    x = np.array([1.0, 2.0, 3.0])
    assert op.shape == (2, 3) and not op.is_square
    assert np.allclose(op @ x, A @ x)
    assert np.allclose(op.rmatvec(np.array([1.0, 1.0])), A.T @ np.array([1.0, 1.0]))
    assert np.allclose(op.adjoint().to_dense(), A.T)
    assert np.allclose(op.to_dense(), A)


def test_an_operator_without_an_adjoint_refuses_instead_of_transposing():
    """A transpose is not the adjoint of a map that is not known to be symmetric."""
    op = LinearOperator(lambda x: np.array([x[0] + 2.0 * x[1]]), (1, 2), name="row")
    assert op.has_adjoint is False
    with pytest.raises(NotImplementedError, match="has no adjoint"):
        op.adjoint()
    with pytest.raises(NotImplementedError, match="has no adjoint"):
        op.rmatvec(np.array([1.0]))


def test_an_operator_checks_the_shapes_it_was_given_and_the_shapes_it_returns():
    op = LinearOperator(lambda x: x[:2], (2, 3))
    with pytest.raises(ValueError, match="acts on 3 entries"):
        op.matvec(np.ones(4))
    wrong = LinearOperator(lambda x: x, (3, 3))
    with pytest.raises(ValueError, match="returned 3 entries for a 2-row operator"):
        LinearOperator(lambda x: wrong.matvec(x), (2, 3)).matvec(np.ones(3))
    with pytest.raises(ValueError, match="positive"):
        LinearOperator(lambda x: x, (0, 3))


def test_a_matrix_free_operator_is_a_matrix_where_it_matters():
    """The point of the class: the same call with a callable instead of a stored matrix."""
    rng = np.random.default_rng(7)
    A = rng.normal(size=(30, 30))
    A = A @ A.T + 30 * np.eye(30)                        # symmetric positive definite
    op = LinearOperator(lambda x: A @ x, A.shape, rmatvec=lambda y: A.T @ y)
    assert np.allclose(op.to_dense(), A)
    estimate = condition_estimate(op, solve=lambda r: np.linalg.solve(A, r),
                                  solve_adjoint=lambda r: np.linalg.solve(A.T, r))
    assert estimate.estimate == pytest.approx(condition_estimate(A).estimate, rel=1e-12)
    assert estimate.exact is None                     # a matrix-free estimate has no truth to compare


# --------------------------------------------------------------------------- POD
def test_pod_reproduces_a_dataset_that_has_exactly_three_modes():
    """Four snapshots spanning a three-dimensional subspace: after the mean, the rank is three."""
    x = np.linspace(0.0, 1.0, 80)
    modes = np.column_stack([np.sin(np.pi * x), np.cos(2 * np.pi * x), np.exp(-((x - 0.5) / 0.1) ** 2)])
    coefficients = np.array([[1.0, 0.5, 0.2, -0.4],
                             [0.3, -1.0, 0.4, 0.9],
                             [2.0, 0.1, -0.5, 0.7]])
    snapshots = modes @ coefficients                   # (80, 4): one snapshot per column

    result = pod(snapshots, energy=0.999)
    assert result.rank == 3
    assert result.energy_captured == pytest.approx(1.0, abs=1e-10)
    assert result.relative_error < 1e-12
    assert np.allclose(result.reconstruct(), snapshots)
    assert result.coefficients.shape == (3, 4)


def test_the_pod_error_is_the_energy_it_discarded():
    """Eckart-Young: the error is ``sqrt(1 - captured)`` times the centred norm, against the raw norm."""
    rng = np.random.default_rng(3)
    snapshots = rng.normal(size=(40, 25))
    result = pod(snapshots, energy=0.9)
    assert 1 <= result.rank < 25
    # the same truncation through the module that measures it independently (#1061)
    reference = svd_low_rank(snapshots - snapshots.mean(axis=1, keepdims=True), result.rank)
    assert result.relative_error == pytest.approx(
        reference.frobenius_error / np.linalg.norm(snapshots, "fro"))
    centred = np.linalg.norm(snapshots - snapshots.mean(axis=1, keepdims=True), "fro")
    assert result.relative_error == pytest.approx(
        np.sqrt(max(0.0, 1.0 - result.energy_captured)) * centred / np.linalg.norm(snapshots, "fro"))


def test_pod_rank_is_the_numerical_rank_of_the_data_not_the_number_asked_for():
    x = np.linspace(0.0, 1.0, 30)
    snapshots = np.outer(np.sin(np.pi * x), [1.0, 0.5])      # rank 1 before and after the mean
    result = pod(snapshots, rank=10)
    assert result.rank == 1
    assert result.relative_error < 1e-14
    assert result.diagnostics[0].code == "LINALG.ILL_CONDITIONED"    # asked for 10, got 1


def test_pod_refuses_input_it_cannot_decompose():
    with pytest.raises(ValueError, match="two-dimensional"):
        pod(np.zeros(10))
    with pytest.raises(ValueError, match="non-finite"):
        pod(np.array([[1.0, 2.0], [np.nan, 3.0]]))
    with pytest.raises(ValueError, match="energy must be"):
        pod(np.zeros((4, 4)), energy=0.0)
    constant = pod(np.ones((5, 4)))
    assert constant.rank == 0 and constant.diagnostics[0].code == "LINALG.SINGULAR"


# --------------------------------------------------------------------------- modal analysis
def _two_dof() -> np.ndarray:
    """Mass-spring-damper, ``m = 1``, ``k = [2, 2]``, damping ``c = 0.05`` on the first mass.

    The undamped natural frequencies solve ``w^4 - 6 w^2 + 4 = 0``, so ``w^2 = 3 +- sqrt(5)``.
    """
    k1 = k2 = 2.0
    c1 = 0.05
    return np.array([[0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0],
                     [-(k1 + k2), k2, -c1, 0.0],
                     [k2, -k2, 0.0, 0.0]])


def test_modal_analysis_finds_the_two_natural_frequencies_of_a_two_dof_system():
    report = modal_analysis(_two_dof())
    assert report.exact is True and report.ritz_approximation is False
    omega_squared = np.sort(3.0 + np.array([-1.0, 1.0]) * np.sqrt(5.0))
    # the conjugate pair of each mode gives the same frequency to rounding, so it is counted once
    measured = np.unique(np.round([m.frequency * 2 * np.pi for m in report.modes if m.frequency > 0], 4))
    # the damped frequencies are the undamped ones to within zeta^2 (6e-5 here); the damping is not
    # proportional, so there is no reason for them to sit on one particular side
    assert np.allclose(measured, np.sqrt(omega_squared), rtol=1e-3)
    # and the polynomial is the exact statement: every eigenvalue is a root of the characteristic one
    characteristic = np.poly(_two_dof())
    for lam in report.eigenvalues:
        assert abs(np.polyval(characteristic, lam)) < 1e-12

    frequencies_hz = np.unique(np.round([m.frequency for m in report.modes if m.frequency > 0], 4))
    assert np.allclose(frequencies_hz, measured / (2 * np.pi), rtol=1e-3)

    # the damping ratio of a mode is -Re/|lambda|; for this system both modes give c phi1^2 / (2 w)
    for mode in report.modes:
        assert mode.damping_ratio == pytest.approx(0.00791, rel=0.01)
        assert 0.0 < mode.participation <= 1.0 + 1e-12
        assert mode.residual < 1e-12


def test_a_defective_matrix_says_its_eigenvalues_are_ill_conditioned():
    """A Jordan block: the eigenvalues exist (2, 2, 2) and are exactly the number not to trust."""
    jordan = np.array([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0], [0.0, 0.0, 2.0]])
    report = modal_analysis(jordan)
    assert np.allclose(report.eigenvalues, 2.0)
    assert report.basis_condition > 1e10
    assert any(d.code == "LINALG.NEAR_DEFECTIVE" for d in report.diagnostics)


def test_modal_analysis_of_an_operator_reports_ritz_values_and_says_so():
    M = _two_dof()
    op = LinearOperator(lambda x: M @ x, M.shape, rmatvec=lambda y: M.T @ y)
    report = modal_analysis(op, krylov=4)
    assert report.ritz_approximation is True and report.exact is False
    # with a full Krylov space (n = 4) the Ritz values are the eigenvalues, and the residuals say so
    assert np.allclose(np.sort_complex(report.eigenvalues), np.sort_complex(np.linalg.eigvals(M)))
    assert max(m.residual for m in report.modes) < 1e-10


def test_modal_analysis_refuses_a_matrix_it_cannot_diagonalise():
    with pytest.raises(ValueError, match="square matrix"):
        modal_analysis(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="square operator"):
        modal_analysis(LinearOperator(lambda x: x[:2], (2, 3)))


# --------------------------------------------------------------------------- Krylov
def test_arnoldi_is_exact_and_breaks_down_on_an_invariant_subspace():
    eigenvalues = np.array([1.0, 2.0, 3.0, 4.0])
    run = arnoldi(np.diag(eigenvalues), np.ones(4))
    assert run.breakdown is True and run.dimension == 4
    assert np.allclose(np.sort(np.real(run.ritz_values)), eigenvalues, rtol=1e-12)
    assert np.allclose(run.basis[:, :-1].T @ run.basis[:, :-1], np.eye(4), atol=1e-12)


def test_arnoldi_with_fewer_steps_finds_the_right_ritz_values_and_their_residuals():
    rng = np.random.default_rng(11)
    A = rng.normal(size=(40, 40))
    run = arnoldi(A, np.ones(40), 12)
    eigenvalues = np.linalg.eigvals(A)
    assert run.dimension == 12 and run.breakdown is False
    # the Ritz values closest to the boundary of the spectrum are the accurate ones
    assert np.allclose(np.sort(np.abs(run.ritz_values))[-1], np.max(np.abs(eigenvalues)), rtol=5e-2)
    assert np.all(np.asarray(run.residuals) < 10.0)


def test_arnoldi_refuses_a_start_vector_it_cannot_use():
    with pytest.raises(ValueError, match="the start vector has 2 entries"):
        arnoldi(np.eye(3), np.ones(2))
    with pytest.raises(ValueError, match="start vector is zero"):
        arnoldi(np.eye(3), np.zeros(3))
    with pytest.raises(ValueError, match="m must be at least 1"):
        arnoldi(np.eye(3), np.ones(3), 0)


# --------------------------------------------------------------------------- conditioning
def test_the_condition_estimate_is_a_lower_bound_of_the_exact_one_norm_value():
    """Hager's estimator: never above the truth, and in practice within a factor of a few of it."""
    rng = np.random.default_rng(5)
    ratios = []
    for n in (5, 20, 40):
        A = rng.normal(size=(n, n))
        estimate = condition_estimate(A)
        exact = float(np.linalg.cond(A, 1))
        assert estimate.exact == pytest.approx(exact, rel=1e-9)
        assert estimate.estimate <= exact * (1 + 1e-6)      # a lower bound, by construction
        assert estimate.iterations >= 1
        ratios.append(estimate.ratio)
    # measured here: the estimator lands between 0.46 and 1.0 times the exact value on random matrices,
    # so "a lower bound" is what it promises and "the exact number" is not
    assert min(ratios) > 0.25 and max(ratios) <= 1 + 1e-6


def test_the_estimator_finds_the_exact_value_where_the_answer_is_known_by_hand():
    identity = condition_estimate(np.eye(6))
    assert identity.exact == pytest.approx(1.0) and identity.estimate == pytest.approx(1.0)
    # a diagonal with a known spread: kappa_1 = max|d| / min|d| exactly
    diagonal = np.diag([1.0, 2.0, 50.0, 1e-2])
    estimate = condition_estimate(diagonal)
    assert estimate.exact == pytest.approx(50.0 / 1e-2)
    assert estimate.estimate == pytest.approx(50.0 / 1e-2)


def test_the_estimate_is_a_lower_bound_between_the_two_norms():
    """``kappa_1 >= kappa_2`` always; a 1-norm estimate below the 2-norm value would be a bug."""
    rng = np.random.default_rng(9)
    A = rng.normal(size=(15, 15))
    estimate = condition_estimate(A)
    assert estimate.estimate >= condition_number(A) * (1 - 1e-9)


def test_a_hilbert_matrix_is_reported_as_the_ill_conditioned_problem_it_is():
    H = np.array([[1.0 / (i + j + 1) for j in range(8)] for i in range(8)])
    estimate = condition_estimate(H)
    assert estimate.exact > 1e10
    assert estimate.estimate == pytest.approx(estimate.exact, rel=1e-6)


def test_a_matrix_free_estimate_needs_a_solver_and_says_how_well_it_did():
    rng = np.random.default_rng(13)
    A = rng.normal(size=(12, 12))
    op = LinearOperator(lambda x: A @ x, A.shape, rmatvec=lambda y: A.T @ y)
    with pytest.raises(ValueError, match="needs a ``solve`` callable"):
        condition_estimate(op)
    refined = condition_estimate(op, solve=lambda r: np.linalg.solve(A, r),
                                 solve_adjoint=lambda r: np.linalg.solve(A.T, r))
    assert refined.exact is None                     # nothing to compare against, and it does not pretend
    assert refined.estimate <= float(np.linalg.cond(A, 1)) * (1 + 1e-6)
    assert refined.estimate == pytest.approx(condition_estimate(A).estimate, rel=1e-12)
    plain = condition_estimate(op, solve=lambda r: np.linalg.solve(A, r))
    assert plain.iterations == 1                     # one right-hand side: a valid but weaker bound
    assert 0.0 < plain.estimate <= refined.estimate * (1 + 1e-9)


def test_the_condition_estimate_refuses_a_rectangular_operator():
    with pytest.raises(ValueError, match="needs a square operator"):
        condition_estimate(np.zeros((3, 4)))


# --------------------------------------------------------------------------- adjoint sensitivity
def test_the_adjoint_derivative_agrees_with_re_solving_the_system():
    """``d(c^T x)/dp`` for ``A(p) x = b``: one solve for every parameter, checked against a difference."""
    A = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    b = np.array([1.0, 0.0, 2.0])
    output = np.array([1.0, -1.0, 0.5])
    deriv_A = [np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
               np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, -1.0]])]
    result = adjoint_sensitivity(A, b, output, deriv_A, None)

    # the analytic reference for the first parameter: d(c^T x)/dp = -lambda^T E11 x
    x = np.linalg.solve(A, b)
    lam = np.linalg.solve(A.T, output)
    assert result.values[0] == pytest.approx(-float(lam @ np.diag([1.0, 0.0, 0.0]) @ x))
    assert result.values[1] == pytest.approx(
        -float(lam @ np.diag([0.0, 2.0, -1.0]) @ x))
    assert np.all(result.relative_error < 1e-5)
    assert result.solves_forward == 1
    assert result.solves_adjoint == 1 + 2 * 2          # one adjoint solve, then the check


def test_the_adjoint_sensitivity_covers_a_right_hand_side_derivative():
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 0.0])
    output = np.array([1.0, 0.0])
    result = adjoint_sensitivity(A, b, output, None, [np.array([0.0, 1.0])])
    lam = np.linalg.solve(A.T, output)
    assert result.values[0] == pytest.approx(float(lam @ np.array([0.0, 1.0])))
    assert result.relative_error[0] < 1e-6


def test_the_adjoint_sensitivity_refuses_what_it_cannot_differentiate():
    A = np.eye(3)
    with pytest.raises(ValueError, match="no parameter derivatives"):
        adjoint_sensitivity(A, np.ones(3), np.ones(3))
    with pytest.raises(ValueError, match="b has 2 entries"):
        adjoint_sensitivity(A, np.ones(2), np.ones(3), np.eye(3))
    with pytest.raises(NotImplementedError, match="no adjoint"):
        adjoint_sensitivity(LinearOperator(lambda x: A @ x, (3, 3)), np.ones(3), np.ones(3), np.eye(3))


# --------------------------------------------------------------------------- as data
def test_every_result_serialises():
    snapshots = np.random.default_rng(0).normal(size=(20, 8))
    payload = json.loads(json.dumps(pod(snapshots, energy=0.95).to_dict()))
    assert payload["rank"] >= 1 and 0.0 <= payload["energy_captured"] <= 1.0

    modal = json.loads(json.dumps(modal_analysis(_two_dof()).to_dict()))
    assert len(modal["modes"]) == 4 and modal["exact"] is True

    estimate = json.loads(json.dumps(condition_estimate(np.eye(4) + 0.1).to_dict()))
    assert estimate["exact"] == pytest.approx(estimate["estimate"], rel=1e-9)

    krylov = json.loads(json.dumps(arnoldi(np.eye(4) * 2.0, np.ones(4)).to_dict()))
    assert krylov["breakdown"] is True and krylov["dimension"] == 1    # 2*b is b again: invariant at once

    sensitivity = json.loads(json.dumps(adjoint_sensitivity(
        np.eye(2) * 2.0, np.ones(2), np.ones(2), np.eye(2)).to_dict()))
    assert sensitivity["solves_forward"] == 1

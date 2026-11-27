"""The four filters behind one name, and the cases where each is the right one (#215).

The done-when of the issue is "EKF and EnKF recover a known hidden state on a synthetic twin case", so that is the
first test. The rest pin the two reasons the new filters exist at all: the UKF is exact where the EKF linearises a
strong nonlinearity badly, and the particle filter can hold a belief with two modes, which no Kalman filter can.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_analysis.state_estimation import (
    METHODS,
    EstimateReport,
    ExtendedKalmanFilter,
    ParticleFilter,
    StateEstimator,
    UnscentedKalmanFilter,
)

G, L, DT = 9.81, 1.0, 0.05


def pendulum(x):
    """One explicit step of a pendulum: state ``[theta, omega]``, the nonlinearity is the ``sin``."""
    theta, omega = x
    return np.array([theta + DT * omega, omega - DT * (G / L) * np.sin(theta)])


def pendulum_case(n=400, sigma=0.05, seed=3):
    rng = np.random.default_rng(seed)
    truth = np.zeros((n, 2))
    truth[0] = [1.0, 0.0]
    for k in range(1, n):
        truth[k] = pendulum(truth[k - 1])
    noisy = truth[:, [0]] + rng.normal(0.0, sigma, (n, 1))     # only the angle is measured
    return truth, noisy, sigma


def estimator(method, **options):
    return StateEstimator(method, n_state=2, n_obs=1, f=pendulum, h=lambda x: np.array([x[0]]),
                          Q=np.diag([1e-6, 1e-4]), R=np.array([[0.05 ** 2]]), seed=0, **options)


@pytest.mark.parametrize("method", METHODS)
def test_every_filter_recovers_the_hidden_state(method):
    """A twin case: the truth is known, only a noisy angle is measured, and the filter has to do better than that."""
    truth, noisy, sigma = pendulum_case()
    est = estimator(method).initialize(np.zeros(2), np.diag([0.5, 0.5]))
    report = est.run(noisy, truth=truth)
    theta_rmse, omega_rmse = report.rmse
    assert theta_rmse < 0.6 * sigma, f"{method}: {theta_rmse:.4f} against a measurement noise of {sigma}"
    assert omega_rmse < 0.15, f"{method}: the unmeasured velocity came out at {omega_rmse:.4f}"
    assert report.n_steps == len(noisy) and report.errors.shape == truth.shape
    assert report.innovations.shape == (len(noisy), 1)


def test_the_report_says_what_it_measured():
    truth, noisy, _ = pendulum_case(n=60)
    report = estimator("ukf").initialize(np.zeros(2)).run(noisy, truth=truth)
    assert isinstance(report, EstimateReport)
    text = str(report)
    assert "ukf" in text and "rmse" in text
    assert report.to_dict()["n_steps"] == 60
    silent = estimator("ukf").initialize(np.zeros(2)).run(noisy)
    assert silent.rmse is None and "no truth" in str(silent)


# --------------------------------------------------------------------------- the UKF is exact where it should be
def test_the_ukf_matches_the_analytic_kalman_on_a_linear_model():
    """On a linear-Gaussian model the unscented transform introduces no error: it has to agree with the closed form."""
    q, r, p0 = 1e-4, 4e-2, 1.0
    ukf = UnscentedKalmanFilter(1, 1, f=lambda x: x, h=lambda x: x[:1], Q=np.array([[q]]), R=np.array([[r]]),
                                alpha=1.0, kappa=0.0)
    ukf.initialize(np.array([0.0]), np.array([[p0]]))
    x, p = 0.0, p0
    for y in (0.6, 1.1, 0.4, 2.0):
        x, p = x, p + q                                   # predict (f is the identity)
        s = p + r
        k = p / s
        x, p = x + k * (y - x), (1 - k) * p               # closed-form update
        out = ukf.step(np.array([y]))
        assert out["x"][0] == pytest.approx(x, abs=1e-9)
        assert out["P"][0, 0] == pytest.approx(p, abs=1e-9)


def test_the_ukf_beats_the_ekf_on_a_strongly_nonlinear_observation():
    """``h(x) = x**3`` from a poor prior: linearising at the prior is what costs the EKF, and nothing else changes."""
    rng = np.random.default_rng(1)
    n = 60
    truth = np.full((n, 1), 1.3)
    noisy = truth ** 3 + rng.normal(0.0, 0.01, (n, 1))
    scored = {}
    for method in ("ekf", "ukf"):
        est = StateEstimator(method, n_state=1, n_obs=1, f=lambda x: x, h=lambda x: np.array([x[0] ** 3]),
                             Q=np.array([[1e-8]]), R=np.array([[1e-4]]))
        est.initialize(np.array([2.0]), np.array([[1.0]]))
        scored[method] = est.run(noisy, truth=truth).rmse[0]
    assert scored["ukf"] < 0.6 * scored["ekf"], scored


# --------------------------------------------------------------------------- the particle filter keeps the modes
def test_the_particle_filter_can_hold_two_modes():
    """A Kalman filter would put the estimate between the modes; the cloud can say "one or the other"."""
    pf = ParticleFilter(1, 1, f=lambda x: x, h=lambda x: x[:1], Q=np.array([[1e-6]]), R=np.array([[0.25]]),
                        n_particles=4000, ess_fraction=0.0, seed=0)
    rng = np.random.default_rng(0)
    low = -2.0 + 0.1 * rng.standard_normal(2000)
    high = 2.0 + 0.1 * rng.standard_normal(2000)
    pf.initialize_from_samples(np.concatenate([low, high])[None, :])
    out = pf.update(np.array([0.0]))                       # an observation that tells the two apart from the middle
    mass_low = float(np.sum(pf.weights[pf.particles[0] < 0.0]))
    mass_high = float(np.sum(pf.weights[pf.particles[0] > 0.0]))
    assert mass_low > 0.05 and mass_high > 0.05, (mass_low, mass_high)
    assert out["x"][0] == pytest.approx(pf.mean[0])


def test_the_particle_filter_resamples_only_when_the_cloud_collapses():
    pf = ParticleFilter(1, 1, f=lambda x: x, h=lambda x: x[:1], Q=np.array([[1e-4]]), R=np.array([[1e-3]]),
                        n_particles=500, ess_fraction=0.5, seed=0)
    pf.initialize(np.array([0.0]), np.array([[1.0]]))
    assert pf.ess == pytest.approx(500, rel=1e-9)
    with pytest.warns(RuntimeWarning, match="collapsed the cloud"):   # far from the cloud: a few particles carry it
        out = pf.step(np.array([4.0]))
    assert out["ess"] < 250 and bool(np.asarray(out["resampled"]).ravel()[0])
    assert pf.ess == pytest.approx(500, rel=1e-9), "after resampling the weights are uniform again"
    assert np.all(pf.weights == pytest.approx(1.0 / 500))


def test_the_particle_filter_refuses_samples_of_the_wrong_shape():
    pf = ParticleFilter(2, 1, f=lambda x: x, h=lambda x: x[:1], n_particles=100)
    with pytest.raises(ValueError, match="samples must be"):
        pf.initialize_from_samples(np.zeros((3, 100)))


def test_an_observation_that_does_not_fit_warns_instead_of_passing_quietly():
    """The weights stay well defined, so this is not an error — but a cloud collapsing onto one particle means the
    model or R does not fit the data, and the filter says so."""
    rng = np.random.default_rng(0)
    pf = ParticleFilter(1, 1, f=lambda x: x, h=lambda x: x[:1], Q=np.array([[1e-12]]), R=np.array([[1e-12]]),
                        n_particles=100, seed=0)
    pf.initialize_from_samples(rng.standard_normal((1, 100)))     # a spread cloud ...
    with pytest.warns(RuntimeWarning, match="collapsed the cloud"):
        out = pf.update(np.array([1.0e3]))                        # ... and an observation a thousand away
    assert out["ess"] == pytest.approx(1.0)


def test_a_non_finite_observation_is_an_error_not_a_silent_zero():
    pf = ParticleFilter(1, 1, f=lambda x: x, h=lambda x: x[:1], n_particles=100, seed=0)
    pf.initialize_from_samples(np.zeros((1, 100)))
    with pytest.raises(FloatingPointError, match="non-finite"):
        pf.update(np.array([np.nan]))


# --------------------------------------------------------------------------- the facade
def test_the_facade_lists_its_filters_and_refuses_an_unknown_one():
    with pytest.raises(ValueError, match="one of ekf"):
        StateEstimator("nao_existe", n_state=1, n_obs=1, f=lambda x: x, h=lambda x: x)


def test_the_facade_exposes_the_state_of_each_filter():
    for method in METHODS:
        est = StateEstimator(method, n_state=2, n_obs=1, f=pendulum, h=lambda x: np.array([x[0]]))
        est.initialize(np.array([0.3, 0.1]), np.eye(2))
        est.step(np.array([0.3]))
        assert est.state.shape == (2,), method


def test_the_ekf_and_enkf_classes_still_behave_as_before():
    """The two that already existed keep their contract: this is a regression guard, not a new claim."""
    ekf = ExtendedKalmanFilter(2, 1, f=pendulum, h=lambda x: np.array([x[0]]))
    ekf.initialize(np.zeros(2), np.eye(2))
    out = ekf.step(np.array([0.1]))
    assert set(out) == {"x", "P", "K", "innovation", "innovation_cov"}

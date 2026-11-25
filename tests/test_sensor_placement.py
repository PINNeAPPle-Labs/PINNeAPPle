"""Tests of pinneapple_analysis.data_assimilation.sensor_placement (GP reconstruction, LOO, greedy placement)."""
import numpy as np
import pytest

from pinneapple_analysis.data_assimilation.sensor_placement import (
    GaussianProcessField,
    greedy_sensor_placement,
    matern_kernel,
)


def _field(X):
    return np.sin(3 * X[:, 0]) * np.cos(2 * X[:, 1])


def test_matern_closed_forms():
    a, b = np.array([[0.0, 0.0]]), np.array([[0.3, 0.4]])
    r = np.hypot(0.3 / 0.5, 0.4 / 2.0)
    assert matern_kernel(a, b, [0.5, 2.0], 2.0, 0.5)[0, 0] == pytest.approx(2 * np.exp(-r))
    s = np.sqrt(5) * r
    assert matern_kernel(a, b, [0.5, 2.0], 2.0, 2.5)[0, 0] == pytest.approx(2 * (1 + s + s * s / 3) * np.exp(-s))
    assert matern_kernel(a, b, [0.5, 2.0], 1.0, np.inf)[0, 0] == pytest.approx(np.exp(-r * r / 2))


def test_posterior_mean_interpolates_noiseless_data():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(25, 2))
    gp = GaussianProcessField([0.3, 0.5], nu=2.5, noise=1e-10, mean="data").fit(X, _field(X))
    mu, sd = gp.predict(X, return_std=True)
    assert mu == pytest.approx(_field(X), abs=1e-6)
    assert np.all(sd < 1e-3)


def test_variance_decreases_as_sensors_are_added():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, size=(30, 2))
    grid = rng.uniform(0, 1, size=(200, 2))
    prev = None
    for k in (2, 5, 10, 20, 30):
        gp = GaussianProcessField([0.2, 0.4], noise=1e-6).fit(X[:k], _field(X[:k]))
        _, sd = gp.predict(grid, return_std=True)
        if prev is not None:
            assert np.all(sd <= prev + 1e-9)
        prev = sd


def test_greedy_placement_spreads_sensors_1d():
    x = np.linspace(0, 1, 101)
    res = greedy_sensor_placement(GaussianProcessField(0.2, nu=2.5, noise=1e-6), x, 3)
    p = res.points.ravel()
    assert abs(p[0] - 0.5) <= 0.02
    assert sorted([p[1], p[2]])[0] < 0.3 and sorted([p[1], p[2]])[1] > 0.7
    assert np.all(np.diff(res.integrated_variance) <= 1e-12)
    # integrated variance after the picks equals the mean posterior variance of a GP fitted at those points
    gp = GaussianProcessField(0.2, nu=2.5, noise=1e-6).fit(res.points, np.zeros(3))
    _, sd = gp.predict(x, return_std=True)
    assert res.integrated_variance[-1] == pytest.approx(np.mean(sd ** 2), rel=1e-6, abs=1e-9)


def test_greedy_with_existing_sensors_and_domain():
    x = np.linspace(0, 1, 51)
    gp = GaussianProcessField(0.15, noise=1e-6)
    res = greedy_sensor_placement(gp, x, 1, existing=[[0.5]], domain=np.linspace(0, 1, 201))
    assert abs(res.points[0, 0] - 0.5) > 0.2   # not on top of the installed sensor


def test_loo_matches_brute_force():
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 1, size=(15, 2))
    y = _field(X) + 0.01 * rng.normal(size=15)
    gp = GaussianProcessField([0.3, 0.6], variance=0.5, nu=1.5, noise=1e-4, mean=0.1).fit(X, y)
    err, var = gp.loo()
    for i in range(len(X)):
        keep = np.arange(len(X)) != i
        g = GaussianProcessField([0.3, 0.6], variance=0.5, nu=1.5, noise=1e-4, mean=0.1).fit(X[keep], y[keep])
        mu, sd = g.predict(X[i:i + 1], return_std=True)
        assert err[i] == pytest.approx(y[i] - mu[0], rel=1e-6, abs=1e-9)
        assert var[i] == pytest.approx(sd[0] ** 2 + 1e-4, rel=1e-5)


def test_hyperparameter_fit_improves_likelihood():
    rng = np.random.default_rng(3)
    X = rng.uniform(0, 1, size=(40, 2))
    y = _field(X)
    gp = GaussianProcessField([2.0, 2.0], variance=0.1, noise=1e-6).fit(X, y)
    before = gp.log_marginal_likelihood()
    gp.fit_hyperparameters(X, y)
    assert gp.log_marginal_likelihood() > before
    Xt = rng.uniform(0, 1, size=(200, 2))
    assert np.sqrt(np.mean((gp.predict(Xt) - _field(Xt)) ** 2)) < 0.05


def test_hyperparameter_fit_does_not_depend_on_the_starting_guess():
    """A length scale of twice the domain used to be a local optimum of the likelihood: the fit collapsed one axis
    (0.025), the likelihood improved and the prediction was the mean far from the samples (0.246 against 0.007)."""
    rng = np.random.default_rng(3)
    X = rng.uniform(0, 1, size=(40, 2))
    y = _field(X)
    Xt = rng.uniform(0, 1, size=(200, 2))
    rmses = []
    for ls in ([2.0, 2.0], [0.3, 0.3], [5.0, 5.0]):
        gp = GaussianProcessField(ls, variance=0.1, noise=1e-6).fit(X, y)
        gp.fit_hyperparameters(X, y)
        rmses.append(float(np.sqrt(np.mean((gp.predict(Xt) - _field(Xt)) ** 2))))
    assert max(rmses) < 0.05

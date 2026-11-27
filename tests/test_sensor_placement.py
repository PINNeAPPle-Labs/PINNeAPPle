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


# ------------------------------------------------------------------ designs: criteria, constraints, strategies

from pinneapple_analysis.data_assimilation.sensor_placement import (  # noqa: E402
    Layout,
    closest_pair,
    evolutionary_sensor_placement,
    layout_curves,
    maximin_sensor_placement,
    random_sensor_placement,
)


def _plate():
    """A 32 x 20 grid on a 1 x 0.6 plate, and the quantity of interest: the mean over the hot corner at x = L."""
    L, H = 1.0, 0.6
    xs, ys = np.linspace(0.0, L, 32), np.linspace(0.0, H, 20)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], axis=-1)
    qoi = np.exp(-((X.ravel() - 0.9) ** 2 + (Y.ravel() - 0.5) ** 2) / 0.02)
    return GaussianProcessField(lengthscales=[0.18, 0.18], variance=1.0, nu=2.5), pts, qoi / qoi.sum()


def _slope():
    """80 stations down a 3 m slope, and the weighted mean over the toe -- a layered soil, so a long kernel."""
    z = np.linspace(0.0, 3.0, 80)
    w = np.exp(-0.5 * (z - 2.2) ** 2 / 0.09)
    return GaussianProcessField(lengthscales=[0.35], variance=1.0, nu=1.5), z[:, None], w / w.sum()


def test_a_quantity_of_interest_changes_where_the_sensors_go():
    gp, pts, qoi = _plate()
    spread = greedy_sensor_placement(gp, pts, 4)
    focused = greedy_sensor_placement(gp, pts, 4, qoi=qoi)
    assert np.mean(focused.points[:, 0]) > np.mean(spread.points[:, 0])
    assert focused.integrated_variance[-1] < spread.integrated_variance[-1]
    with pytest.raises(ValueError, match="qoi must have one coefficient"):
        greedy_sensor_placement(gp, pts, 2, qoi=qoi[:-1])


def test_a_budget_stops_the_walk_and_the_gain_is_per_unit_of_cost():
    gp, pts, _ = _plate()
    full = greedy_sensor_placement(gp, pts, 6)
    price = np.full(len(pts), 3.0)
    poor = greedy_sensor_placement(gp, pts, 6, cost=price, budget=10.0)
    assert len(poor.indices) == 3                              # three fit; the fourth would cost 12
    assert price[poor.indices].sum() <= 10.0
    assert poor.integrated_variance[-1] > full.integrated_variance[-1]
    with pytest.raises(ValueError, match="budget must be positive"):
        greedy_sensor_placement(gp, pts, 2, budget=0.0)


def test_the_baselines_are_what_they_say():
    gp, pts, _ = _plate()
    r1 = random_sensor_placement(gp, pts, 5, seed=3)
    r2 = random_sensor_placement(gp, pts, 5, seed=3)
    assert isinstance(r1, Layout) and r1.strategy == "random"
    assert np.array_equal(r1.indices, r2.indices) and r1.criterion == r2.criterion
    u = maximin_sensor_placement(gp, pts, 5)
    assert u.strategy == "uniform" and closest_pair(u.points) > closest_pair(r1.points)
    assert len(set(u.indices.tolist())) == 5


def test_the_evolutionary_search_has_to_be_better_than_random():
    gp, z, w = _slope()
    ga = evolutionary_sensor_placement(gp, z, 3, weights=w, n_generations=12, seed=0)
    best_random = min(random_sensor_placement(gp, z, 3, weights=w, seed=s).criterion for s in range(20))
    assert ga.strategy == "evolutionary" and len(ga.history) == 12
    assert ga.criterion < best_random
    assert np.isfinite(ga.criterion)
    assert np.all(np.diff(ga.history) <= 0.0)                                     # the best never gets worse


def test_a_hard_spacing_the_greedy_walk_cannot_hold():
    gp, pts, qoi = _plate()
    greedy = greedy_sensor_placement(gp, pts, 8, qoi=qoi)
    assert closest_pair(greedy.points) < 0.15                       # what the greedy walk does not see
    held = evolutionary_sensor_placement(gp, pts, 8, qoi=qoi, min_spacing=0.15, n_generations=15)
    assert closest_pair(held.points) >= 0.15 - 1e-9                 # close to it, and feasible
    spread = maximin_sensor_placement(gp, pts, 8, qoi=qoi)
    assert closest_pair(spread.points) >= 0.15
    free = evolutionary_sensor_placement(gp, pts, 8, qoi=qoi, n_generations=15)
    assert free.criterion < held.criterion                          # the constraint costs something


def test_the_curves_of_the_two_cases_are_the_deliverable():
    """The measured comparison the issue asks for: uncertainty against the number of sensors, four strategies, on
    the plate and on the slope."""
    ks = [1, 2, 4, 8]
    for gp, cand, kw in ((_plate()[0], _plate()[1], {"qoi": _plate()[2]}),
                         (_slope()[0], _slope()[1], {"weights": _slope()[2]})):
        curves = layout_curves(gp, cand, ks, n_generations=10, n_random=10, **kw)
        assert set(curves) == {"prior", "greedy", "evolutionary", "random", "uniform"}
        for name in ("greedy", "uniform"):                           # the deterministic designs only: a search run
            assert np.all(np.diff(curves[name]) <= 1e-12)            # once per k has no reason to be monotone
            assert curves[name][-1] < curves["prior"][0]
        assert curves["evolutionary"][-1] < curves["prior"][0]
        assert curves["greedy"][-1] <= curves["random"][-1]          # greedy beats the sensor lottery
        assert curves["greedy"][-1] <= curves["uniform"][-1]
    with pytest.raises(ValueError, match="unknown strategies"):
        layout_curves(_plate()[0], _plate()[1], ks, strategies=("greedy", "sobol"))
    with pytest.raises(ValueError, match="empty"):
        layout_curves(_plate()[0], _plate()[1], [])


def test_the_documented_table_is_what_the_strategies_reach():
    """The digits in ``docs/product/core-shared-pieces.md``, measured here so they cannot drift away from it.

    A table of numbers in a document is a claim like any other: if a change moves one of them, the document is wrong
    and this fails. The search runs at its defaults (``population_size = 30``, ``n_generations = 25``, ``seed = 0``),
    and the lottery is reported as the median of twenty draws because one draw is not a number -- the spread is wide
    enough to overturn a single comparison (the luckiest draw at two sensors beats the walk's two-sensor criterion on
    the slope's scale, and the unluckiest is worse than the even spread).
    """
    gp, pts, qoi = _plate()
    ks = (1, 2, 4, 8)
    greedy = [greedy_sensor_placement(gp, pts, k, qoi=qoi).integrated_variance[-1] for k in ks]
    assert np.allclose(greedy, [0.0539, 0.0341, 0.0089, 0.0006], atol=5e-5)
    assert np.allclose([maximin_sensor_placement(gp, pts, k, qoi=qoi).criterion for k in ks],
                       [0.6470, 0.3799, 0.3778, 0.2681], atol=5e-5)
    assert np.allclose([evolutionary_sensor_placement(gp, pts, k, qoi=qoi, seed=0).criterion for k in ks],
                       [0.2130, 0.1492, 0.1018, 0.0233], atol=5e-4)
    draws = np.array([[random_sensor_placement(gp, pts, k, qoi=qoi, seed=s).criterion for s in range(20)]
                      for k in ks])
    assert np.allclose(np.median(draws, axis=1), [0.6034, 0.4087, 0.2550, 0.1906], atol=5e-4)
    assert draws[0].min() == pytest.approx(0.0916, abs=5e-4) and draws[1].min() == pytest.approx(0.0605, abs=5e-4)
    assert draws[1].min() < 0.2 * np.median(draws[1])      # the luckiest draw is far below the median ...
    assert draws[1].min() > greedy[1]                      # ... and still behind the walk
    assert draws[:, -1].max() > 20.0 * greedy[-1]          # and no draw comes near it at eight sensors

    gp, z, w = _slope()
    greedy = [greedy_sensor_placement(gp, z, k, weights=w).integrated_variance[-1] for k in ks]
    assert np.allclose(greedy, [0.4629, 0.2913, 0.0965, 0.0193], atol=5e-5)
    assert np.allclose([maximin_sensor_placement(gp, z, k, weights=w).criterion for k in ks],
                       [1.0000, 0.9443, 0.3542, 0.1020], atol=5e-5)
    search = [evolutionary_sensor_placement(gp, z, k, weights=w, seed=0).criterion for k in ks]
    assert np.allclose(search, [0.4629, 0.2368, 0.0872, 0.0244], atol=5e-4)
    assert search[1] < greedy[1] and search[2] < greedy[2]  # the search beats the walk here ...
    assert search[3] > greedy[3]                            # ... and loses at eight

    # the spacing numbers the document quotes, from the same test the constraint is built for
    gp, pts, qoi = _plate()
    assert closest_pair(greedy_sensor_placement(gp, pts, 8, qoi=qoi).points) == pytest.approx(0.063, abs=5e-4)
    held = evolutionary_sensor_placement(gp, pts, 8, qoi=qoi, min_spacing=0.15, seed=0)
    assert closest_pair(held.points) == pytest.approx(0.173, abs=5e-4)
    assert held.criterion == pytest.approx(0.104, abs=5e-4)
    spread = maximin_sensor_placement(gp, pts, 8, qoi=qoi)
    assert closest_pair(spread.points) == pytest.approx(0.374, abs=5e-4)
    assert spread.criterion == pytest.approx(0.268, abs=5e-4)

"""PosteriorBench-style posterior metrics: correctness and the point-estimate-vs-posterior gap."""
import numpy as np
import pytest
from scipy.stats import wasserstein_distance

from pinneapple_analysis.uncertainty.posterior_metrics import (
    evaluate_posterior, grf_direction, weighted_mmd, weighted_wasserstein_1d,
)


def _grf_fields(n, h=16, w=16, seed=0, scale=1.0, shift=0.0):
    rng = np.random.default_rng(seed)
    return np.stack([scale * grf_direction(h, w, rng) * np.sqrt(h * w) + shift for _ in range(n)])


def test_1d_wasserstein_matches_scipy_with_weights():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=40), rng.normal(0.5, 2.0, size=60)
    wx, wy = rng.random(40), rng.random(60)
    assert weighted_wasserstein_1d(x, y, wx, wy) == pytest.approx(wasserstein_distance(x, y, wx, wy), rel=1e-10)


def test_identical_ensembles_score_zero_and_mmd_grows_with_shift():
    a = _grf_fields(30)
    r = evaluate_posterior(a, a, n_projections=16)
    assert r["mean_rel_l2"] == 0 and r["std_rel_l2"] == 0 and r["mmd"] < 1e-6 and r["swd"] == 0
    assert weighted_mmd(a, a + 0.5) < weighted_mmd(a, a + 2.0)


def test_collapsed_posterior_is_caught_even_when_the_mean_is_right():
    """The PosteriorBench point: an under-dispersed posterior (MC-dropout-like) with the right mean."""
    ref = _grf_fields(60, seed=0) + 3.0
    good = _grf_fields(60, seed=1) + 3.0  # same distribution, other draws
    mean = ref.mean(0)
    collapsed = mean + 0.1 * (_grf_fields(60, seed=2))  # right mean, 10x too narrow
    g = evaluate_posterior(good, ref, n_projections=32)
    c = evaluate_posterior(collapsed, ref, n_projections=32)
    assert c["mean_rel_l2"] < 0.1 and g["mean_rel_l2"] < 0.1  # both point estimates within sampling noise (~sigma/sqrt(60))
    assert c["std_rel_l2"] > 5 * g["std_rel_l2"]
    assert c["swd"] > 2 * g["swd"] and c["mmd"] > g["mmd"]

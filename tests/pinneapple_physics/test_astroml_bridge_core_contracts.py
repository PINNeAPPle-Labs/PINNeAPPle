"""Core coverage (#1129, toward 1.0) for the astroML bridge of the astrophysics presets. astroML is installed by the
CI step that runs this file, so these tests run there rather than being skipped.

References: a known period recovered from an irregularly sampled, two-harmonic signal; a density change found by
Bayesian blocks at the true change point; a straight line recovered by a fit with errors on both axes; the means of
an extreme-deconvolution fit near the true centres.
"""
import numpy as np
import pytest

from pinneapple_physics.pde_environment.presets import astroml_bridge as AB

pytest.importorskip("astroML")


def test_availability_is_reported():
    assert AB.astroml_available() is True


def test_period_recovery_reference_case():
    r = AB.check_period_recovery()
    assert r["passed"] and r["reference"] == 0.613 and r["error"] < r["tolerance"]           # published check


def test_multiterm_periodogram_finds_a_non_sinusoidal_period():
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 50, 200))
    y = np.sin(2 * np.pi * t / 1.7) + 0.5 * np.sin(4 * np.pi * t / 1.7) + rng.normal(0, 0.05, 200)
    pg = AB.multiterm_periodogram(t, y, np.full(200, 0.05), n_terms=2, period_range=(0.5, 5.0))
    assert abs(1 / pg.freq[np.argmax(pg.power)] - 1.7) < 0.01 and pg.n_terms == 2


def test_bayesian_blocks_find_the_change_in_density():
    rng = np.random.default_rng(0)
    ev = np.r_[rng.uniform(0, 1, 200), rng.uniform(1, 2, 20)]                                # rate drops 10x at 1
    e = AB.bayesian_block_edges(ev)
    interior = e[1:-1]
    assert len(interior) >= 1 and np.min(np.abs(interior - 1.0)) < 0.02


def test_line_fit_with_errors_on_both_axes():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 40)
    lf = AB.fit_line_tls(x + rng.normal(0, 0.1, 40), 2 * x + 1 + rng.normal(0, 0.1, 40), 0.1, 0.1)
    assert abs(lf.slope - 2.0) < 0.02 and abs(lf.intercept - 1.0) < 0.1


def test_extreme_deconvolution_result_samples_its_mixture():
    rng = np.random.default_rng(1)
    X = np.r_[rng.normal([0, 0], 0.3, (150, 2)), rng.normal([3, 3], 0.3, (150, 2))]
    Xerr = np.tile(np.eye(2) * 0.05 ** 2, (300, 1, 1))
    res = AB.xd_fit(X + rng.normal(0, 0.05, X.shape), Xerr, n_components=2, max_iter=50, random_state=0)
    assert isinstance(res, AB.XDResult) and res.weights.sum() == pytest.approx(1.0)
    centres = sorted(res.means.tolist())
    assert np.allclose(centres, [[0, 0], [3, 3]], atol=0.15)                                # the true centres
    assert res.sample(10, random_state=0).shape == (10, 2)

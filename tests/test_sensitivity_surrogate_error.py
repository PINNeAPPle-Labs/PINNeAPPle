"""Measured checks that the surrogate's own error reaches the Sobol intervals and the ranking (#927).

The point of the module is not that the interval exists but that it covers at the rate it claims, so the tests count
coverage against the exact Ishigami indices instead of asserting the interval is merely wider.

Measured on this machine (``2026-11-25``) by the run that produced the thresholds below — a GP is a Matern 2.5 with
the noise kernel of ``_fit`` below, fitted on ``n_train`` points, and the indices are computed on the Saltelli design
with ``N = 256`` over 12 training sets (8 for ``n_train = 300``):

=================  ======================  =====================  =====================
GP training points  coverage, bootstrap     coverage, combined     surrogate variance
=================  ======================  =====================  =====================
30                  0.431                   0.611                  0.731
100                 0.792                   0.861                  0.476
300                 1.000                   1.000                  0.079
=================  ======================  =====================  =====================

Sobol' G (a = 1, 2, 5), same design and 8 training sets: 0.77 -> 0.88 at 25 points (surrogate share 0.79), 0.98 at 120
points (share 0.29).

The residual gap at 30 and 100 points is the surrogate's *bias*: :func:`gp_posterior_paths` is itself calibrated
pointwise (94-95 % of the truth inside its own 95 % band, checked below), so what the combined interval does not
cover is the part of the model error the model does not know about. That is why the study also reports which source
dominates, and says to improve the model when the surrogate is the larger one.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

pytest.importorskip("sklearn")
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel  # noqa: E402

from pinneapple_analysis.sensitivity import (  # noqa: E402
    ensemble_paths,
    gp_posterior_paths,
    paths_from_residuals,
    ranking_stability,
    saltelli_design,
    sensitivity_study,
    sobol_from_paths,
    sobol_indices,
    sobol_with_paths,
)
from pinneapple_analysis.sensitivity.testfunctions import (  # noqa: E402
    ishigami,
    ishigami_indices,
    ishigami_space,
    sobol_g,
    sobol_g_indices,
    sobol_g_space,
)
from pinneapple_analysis.study_models import StudyModel  # noqa: E402

D = 3
SPACE = ishigami_space()
EXACT = ishigami_indices()


def _fit(n_train: int, seed: int = 0) -> GaussianProcessRegressor:
    """A GP of the kind the library fits, on ``n_train`` points of Ishigami."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-np.pi, np.pi, (n_train, D))
    kernel = (ConstantKernel(1.0, (1e-3, 1e3)) * Matern(np.ones(D), (1e-1, 1e1), nu=2.5)
              + WhiteKernel(1e-4, (1e-6, 1e-1)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                     # a hyper-parameter at its bound is not an error
        return GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=1,
                                        random_state=0).fit(X, ishigami(X))


def _indices(paths: np.ndarray, n: int = 256, seed: int = 0):
    _, layout = saltelli_design(SPACE, n, False, seed)
    return sobol_from_paths(paths, layout, level=0.95, n_draws=100, seed=seed)


def _inside(ci: np.ndarray, truth: float, output: int, i: int) -> bool:
    return bool(ci[0, output, i] <= truth <= ci[1, output, i])


# ------------------------------------------------------------------------------------ the source, verified alone
def test_gp_paths_are_calibrated_pointwise_and_agree_with_the_mean():
    """What is propagated has to be honest: the 95 % band of the paths must hold the truth ~95 % of the time."""
    gp = _fit(100)
    rng = np.random.default_rng(7)
    Xte = rng.uniform(-np.pi, np.pi, (2000, D))
    paths = gp_posterior_paths(gp, Xte, n_paths=400, seed=0)
    assert paths.shape == (400, 2000, 1)
    # 400 draws of a mean with posterior std ~2.5: the Monte Carlo error of the path mean is ~0.13
    np.testing.assert_allclose(paths.mean(axis=0)[:, 0], gp.predict(Xte), rtol=0, atol=0.5)
    lo, hi = np.quantile(paths[:, :, 0], [0.025, 0.975], axis=0)
    truth = ishigami(Xte)
    coverage = float(np.mean((lo <= truth) & (truth <= hi)))
    assert 0.90 <= coverage <= 0.99, f"the paths cover the truth {coverage:.3f} of the time, not ~0.95"


def test_gp_paths_reproduce_the_predictive_std_of_sklearn():
    gp = _fit(60)
    X = np.random.default_rng(3).uniform(-np.pi, np.pi, (200, D))
    paths = gp_posterior_paths(gp, X, n_paths=2000, seed=0)
    _, sklearn_std = gp.predict(X, return_std=True)
    got = paths[:, :, 0].std(axis=0)
    np.testing.assert_allclose(got, sklearn_std, rtol=0.1)


def test_identical_paths_give_exactly_the_bootstrap_interval():
    """With no surrogate error the combined interval must collapse onto the old one: no width is invented."""
    X, layout = saltelli_design(SPACE, 256, False, 0)
    s = sobol_from_paths(np.repeat(ishigami(X)[None], 8, axis=0), layout, level=0.95, n_draws=100)
    plain = sobol_indices(ishigami, SPACE, n=256, n_boot=100, level=0.95)
    np.testing.assert_allclose(s.ST_ci, plain.ST_ci, atol=1e-12)      # the same design and rows: same interval
    np.testing.assert_allclose(s.ST, plain.ST, atol=1e-12)
    assert np.all(s.surrogate_error.width["surrogate_only"]["ST"] < 1e-12)
    assert not s.surrogate_error.dominates


# ------------------------------------------------------------------------------------ coverage, measured
def test_combined_interval_covers_the_exact_ishigami_indices_at_a_higher_rate_than_the_bootstrap():
    """The acceptance criterion of #927, counted: 6 indices x 12 training sets per surrogate quality."""
    reps = {30: 12, 100: 12, 300: 8}
    got = {}
    for n_train, R in reps.items():
        boot = comp = share = 0.0
        for r in range(R):
            gp = _fit(n_train, seed=1000 + r)
            Xd, _ = saltelli_design(SPACE, 256, False, 0)
            s = _indices(gp_posterior_paths(gp, Xd, n_paths=100, seed=r), seed=r)
            e = s.surrogate_error
            for j in range(D):
                boot += _inside(e.bootstrap_only["ST_ci"], EXACT["ST"][j], 0, j)
                boot += _inside(e.bootstrap_only["S1_ci"], EXACT["S1"][j], 0, j)
                comp += _inside(s.ST_ci, EXACT["ST"][j], 0, j)
                comp += _inside(s.S1_ci, EXACT["S1"][j], 0, j)
            share += float(np.mean(e.variance_share["surrogate"]["ST"][0]))
        got[n_train] = {"bootstrap": boot / (6 * R), "combined": comp / (6 * R), "share": share / R}

    assert got[30]["combined"] > got[30]["bootstrap"] + 0.10, got
    assert got[100]["combined"] > got[100]["bootstrap"] + 0.03, got
    assert got[30]["bootstrap"] < 0.60, f"the bootstrap interval did not under-cover: {got}"
    # the share of the width that is the surrogate falls as the surrogate improves, and says who dominates
    assert got[30]["share"] > got[100]["share"] > got[300]["share"], got
    assert got[300]["share"] < 0.30, got
    assert got[300]["combined"] >= 0.90, got


def test_sobol_g_with_surrogates_of_controlled_quality():
    """The same counting on the second benchmark, 6 indices x 8 training sets: 0.77 -> 0.88 at 25 points, 0.98 at 120."""
    a = np.array([1.0, 2.0, 5.0])
    space, exact = sobol_g_space(3), sobol_g_indices(a)
    got = {}
    for n_train, R in ((25, 8), (120, 8)):
        boot = comp = share = 0.0
        for r in range(R):
            rng = np.random.default_rng(3000 + r)
            Xd, layout = saltelli_design(space, 256, False, 0)
            Xtr = rng.uniform(0.0, 1.0, (n_train, 3))
            kernel = (ConstantKernel(1.0, (1e-3, 1e3)) * Matern(np.ones(3), (1e-1, 1e1), nu=2.5)
                      + WhiteKernel(1e-4, (1e-6, 1e-1)))
            gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=1,
                                          random_state=0).fit(Xtr, sobol_g(Xtr, a))
            s = sobol_from_paths(gp_posterior_paths(gp, Xd, n_paths=100, seed=r), layout, n_draws=100, seed=r)
            e = s.surrogate_error
            for j in range(3):
                boot += _inside(e.bootstrap_only["ST_ci"], exact["ST"][j], 0, j)
                boot += _inside(e.bootstrap_only["S1_ci"], exact["S1"][j], 0, j)
                comp += _inside(s.ST_ci, exact["ST"][j], 0, j)
                comp += _inside(s.S1_ci, exact["S1"][j], 0, j)
            share += float(np.mean(e.variance_share["surrogate"]["ST"][0]))
        got[n_train] = {"bootstrap": boot / (6 * R), "combined": comp / (6 * R), "share": share / R}
    assert got[25]["combined"] > got[25]["bootstrap"] + 0.08, got
    assert got[25]["share"] > 0.6 and got[120]["share"] < 0.35, got
    assert got[120]["combined"] >= 0.90, got


def test_a_poor_surrogate_is_reported_as_dominating_the_interval():
    gp = _fit(30, seed=1000)
    Xd, _ = saltelli_design(SPACE, 256, False, 0)
    s = _indices(gp_posterior_paths(gp, Xd, n_paths=100, seed=0))
    e = s.surrogate_error
    assert e.dominates
    assert np.nanmax(e.variance_share["surrogate"]["ST"]) > 0.5
    json.dumps(e.to_dict())
    assert "surrogate" in s.to_dict()["sources"] and "monte_carlo" in s.to_dict()["sources"]
    # the decomposition is closed: the measured variance is the sum of the two parts
    total = e.variance_share["monte_carlo"]["ST"] + e.variance_share["surrogate"]["ST"]
    np.testing.assert_allclose(total, 1.0, atol=1e-9)
    assert np.nanmax(e.width["combined"]["ST"]) > np.nanmax(e.width["bootstrap_only"]["ST"])


def test_a_good_surrogate_does_not_widen_the_interval():
    gp = _fit(300, seed=2000)
    Xd, _ = saltelli_design(SPACE, 512, False, 0)
    s = _indices(gp_posterior_paths(gp, Xd, n_paths=100, seed=0), n=512)
    e = s.surrogate_error
    assert not e.dominates
    assert np.all(e.variance_share["surrogate"]["ST"][0] < 0.5)
    # the indices of a surrogate this close to the truth are the exact ones, within the bootstrap interval
    for j in range(D):
        assert abs(s.ST[0, j] - EXACT["ST"][j]) < 0.15


# ------------------------------------------------------------------------------------ the other two sources
def test_ensemble_paths_are_one_index_set_per_member():
    members = [(lambda Z, a=a: ishigami(Z, a=a)) for a in (5.0, 7.0, 9.0)]
    Xd, layout = saltelli_design(SPACE, 256, False, 0)
    paths = ensemble_paths(members, Xd)
    assert paths.shape == (3, Xd.shape[0], 1)
    s = sobol_from_paths(paths, layout, level=0.95, n_draws=100, seed=0, sources=["monte_carlo", "ensemble"])
    assert s.surrogate_error.n_paths == 3
    assert "ensemble" in s.sources
    # the ensemble mean path is what the point estimate comes from
    np.testing.assert_allclose(s.Y, paths.mean(axis=0), rtol=1e-12)
    # a member whose parameter differs has different indices, and the interval spans them
    single = sobol_from_paths(np.repeat(paths[:1], 2, axis=0), layout, n_draws=20)
    assert abs(single.ST[0, 1] - s.ST[0, 1]) > 0.005
    assert s.ST_ci[1, 0, 1] > single.ST_ci[1, 0, 1]


def test_residual_paths_carry_the_measured_error_of_the_model():
    """The weak source: residuals re-sampled at each design point. Its width must follow their scale."""
    Xd, layout = saltelli_design(SPACE, 256, False, 0)
    pred = ishigami(Xd)
    rng = np.random.default_rng(0)
    narrow = 0.01 * rng.standard_normal(200)             # a shift of the output alone would change no index
    wide = 2.0 * rng.standard_normal(200)
    small = sobol_from_paths(paths_from_residuals(pred, narrow, 60, seed=0), layout, n_draws=60)
    big = sobol_from_paths(paths_from_residuals(pred, wide, 60, seed=0), layout, n_draws=60)
    assert np.max(big.surrogate_error.width["combined"]["ST"]) > np.max(small.surrogate_error.width["combined"]["ST"])
    assert big.surrogate_error.dominates and not small.surrogate_error.dominates
    with pytest.raises(ValueError):
        paths_from_residuals(pred, [1.0], 10)


# ------------------------------------------------------------------------------------ the ranking
def test_ranking_stability_counts_what_it_says():
    """A constructed case with a known answer: x2 is tied with x1 in half the draws, x3 is always last."""
    D_draws, m, d = 100, 1, 3
    ST = np.zeros((D_draws, m, d))
    ST[:50, 0] = [0.6, 0.3, 0.1]
    ST[50:, 0] = [0.3, 0.6, 0.1]                                   # x1 and x2 swap in half the draws
    out = ranking_stability(ST, ["x1", "x2", "x3"], top_k=(1, 2), outputs=["y"])["y"]
    assert out["reference"] == ["x1", "x2", "x3"]                  # the median ties, broken by input order
    assert out["rank_stability"] == [0.5, 0.5, 1.0]
    assert out["p_top_k"][1] == [0.5, 0.5, 0.0]
    assert out["p_top_k"][2] == [1.0, 1.0, 0.0]
    # tau is 1 in the first half and (1 - 1) / 3 in the second, so the mean over the draws is 2/3
    assert abs(out["kendall_tau"]["mean"] - 2.0 / 3.0) < 1e-9
    assert 0.0 <= out["kendall_tau"]["p05"] <= out["kendall_tau"]["p95"] <= 1.0


def test_ranking_stability_of_a_separated_ranking_is_perfect():
    ST = np.array([[[0.5, 0.3, 0.2]]] * 50) + np.linspace(0, 0.01, 50)[:, None, None]
    out = ranking_stability(ST, ["a", "b", "c"], outputs=["z"])["z"]
    assert out["rank_stability"] == [1.0, 1.0, 1.0]
    assert out["kendall_tau"]["mean"] == 1.0


# ------------------------------------------------------------------------------------ in the study
def test_study_records_the_sources_and_freezes_on_the_combined_upper_bound():
    """A member ensemble that disagrees about an inert input turns its freeze into a keep, and the JSON says why."""
    def predict(X):
        return (3.0 * np.atleast_2d(X)[:, 0])[:, None]

    model = StudyModel.from_function(predict, ["x1", "x2"], ["y"])
    space = [{"name": "x1", "dist": "uniform", "low": 0.0, "high": 1.0},
             {"name": "x2", "dist": "uniform", "low": 0.0, "high": 1.0}]
    plain = sensitivity_study(model, space, n_sobol=256, morris_on=False, effects_on=False, threshold=0.05)
    assert [r["name"] for r in plain["freeze"]["rows"] if r["decision"] == "freeze"] == ["x2"]
    assert plain["sobol"]["sources"] == ["monte_carlo"]

    members = [(lambda X, s=s: 3.0 * np.atleast_2d(X)[:, 0] + s * np.atleast_2d(X)[:, 1]) for s in (0.0, 0.6, 1.2)]
    study = sensitivity_study(model, space, n_sobol=256, morris_on=False, effects_on=False, threshold=0.05,
                              paths_fn=lambda X: ensemble_paths(members, X), n_draws=100,
                              surrogate_sources=["monte_carlo", "ensemble"])
    se = study["sobol"]["surrogate_error"]
    assert study["sobol"]["sources"] == ["monte_carlo", "ensemble"]
    assert se["n_paths"] == 3
    row = {r["name"]: r for r in study["surrogate_error_tables"]["y"]}
    assert row["x1"]["width_combined"] > row["x1"]["width_bootstrap_only"]
    x2 = [r for r in study["freeze"]["rows"] if r["name"] == "x2"][0]
    x2_plain = [r for r in plain["freeze"]["rows"] if r["name"] == "x2"][0]
    assert x2["decision"] != "freeze", "an input the ensemble disagrees about cannot be frozen"
    assert x2["max_ST_upper"] > x2_plain["max_ST_upper"], "the combined upper bound did not move"
    assert study["ranking_stability"]["y"]["kendall_tau"]["p05"] >= -1.0
    json.dumps(study)                                             # the study is what the app ships


def test_study_with_paths_uses_the_combined_interval_in_the_exact_block():
    gp = _fit(100, seed=5)
    model = StudyModel.from_function(lambda X: gp.predict(np.atleast_2d(X))[:, None],
                                     ["x1", "x2", "x3"], ["y"])
    out = sensitivity_study(model, SPACE, n_sobol=256, morris_on=False, effects_on=False, exact=EXACT,
                            paths_fn=lambda X: gp_posterior_paths(gp, X, n_paths=60, seed=0), n_draws=100)
    assert out["sobol"]["surrogate_error"]["n_paths"] == 60
    assert len(out["sobol"]["ST_ci"]) == 2
    assert out["n_evals"]["sobol"] == 60 * 256 * 5                 # surrogate evaluations, not model calls


# ------------------------------------------------------------------------------------ the contract
def test_sobol_with_paths_builds_the_design_itself():
    gp = _fit(80, seed=11)
    s = sobol_with_paths(lambda X: gp_posterior_paths(gp, X, n_paths=40, seed=0), SPACE, n=128, n_draws=50)
    assert s.n_base == 128 and s.surrogate_error.n_paths == 40
    assert s.names == ["x1", "x2", "x3"]
    rows = s.surrogate_error.table(0)
    assert len(rows) == 3 and "share_surrogate" in rows[0]


def test_sobol_from_paths_rejects_what_it_cannot_use():
    _, layout = saltelli_design(SPACE, 64, False, 0)
    n = 64 * (D + 2)
    with pytest.raises(ValueError, match="paths must be"):
        sobol_from_paths(np.zeros((5, n + 1)), layout)
    with pytest.raises(ValueError, match="at least two paths"):
        sobol_from_paths(np.zeros((1, n)), layout)
    bad = np.zeros((5, n))
    bad[0, 3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        sobol_from_paths(bad, layout)
    with pytest.raises(ValueError, match="at least two members"):
        ensemble_paths([ishigami], np.zeros((4, D)))
    with pytest.raises(ValueError, match="fitted"):
        gp_posterior_paths(GaussianProcessRegressor(), np.zeros((4, D)))

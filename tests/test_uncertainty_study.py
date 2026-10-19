"""Uncertainty-study helpers: distributions, split-conformal coverage on held-out data, propagation (MC, LHS,
polynomial chaos) against exact moments, applicability gate and routing rule."""
import numpy as np
import pytest

from pinneapple_analysis.sensitivity.testfunctions import ishigami, ishigami_indices, ishigami_space
from pinneapple_analysis.uncertainty.applicability import (
    ApplicabilityDomain,
    route_cases,
    routing_summary,
)
from pinneapple_analysis.uncertainty.conformal_study import (
    conformal_quantile,
    conformal_study,
    min_calibration_size,
    split_indices,
)
from pinneapple_analysis.uncertainty.distributions import as_space, sample_space
from pinneapple_analysis.uncertainty.pce import polynomial_chaos
from pinneapple_analysis.uncertainty.tolerance import propagate, propagate_space


def test_distributions_sample_their_moments_and_reject_bad_specs():
    sp = [{"name": "a", "dist": "normal", "mean": 1, "std": 0.1},
          {"name": "b", "dist": "triangular", "low": 0, "mode": 1, "high": 3},
          {"name": "c", "dist": "truncnormal", "mean": 0, "std": 1, "low": -1, "high": 2},
          {"name": "d", "dist": "lognormal", "median": 2, "gsd": 1.3}]
    X = sample_space(sp, 20000, "lhs")
    assert X[:, 0].mean() == pytest.approx(1.0, abs=1e-3)
    assert X[:, 1].mean() == pytest.approx(4 / 3, abs=1e-3)
    assert X[:, 2].min() >= -1 and X[:, 2].max() <= 2
    assert np.median(X[:, 3]) == pytest.approx(2.0, rel=0.01)
    assert [p.name for p in as_space({"x": (0, 1), "y": (2, 5)})] == ["x", "y"]
    with pytest.raises(ValueError, match="needs std"):
        as_space([{"name": "z", "dist": "normal", "mean": 0}])
    with pytest.raises(ValueError, match="duplicate"):
        as_space([{"name": "z", "low": 0, "high": 1}, {"name": "z", "low": 0, "high": 1}])


def test_conformal_quantile_is_the_finite_sample_order_statistic():
    s = np.arange(1, 16, dtype=float)                       # n = 15, alpha = 0.1: k = ceil(16 * 0.9) = 15 -> max
    assert conformal_quantile(s, 0.1) == 15.0
    assert min_calibration_size(0.1) == 9


def test_split_conformal_coverage_measured_on_held_out_data():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, 4000)
    sd = 0.1 + 0.2 * x
    y = np.sin(6 * x) + rng.normal(0, sd)
    pred = np.sin(6 * x)
    cal, test = split_indices(4000, (0.5, 0.5), seed=0)
    assert not set(cal) & set(test)
    st = conformal_study(y[cal], pred[cal], y[test], pred[test], alpha=0.1, group_by=x[test], group_name="x")
    assert st.n_cal == st.n_test == 2000
    assert st.meets_target and 0.88 < st.coverage < 0.93
    # constant width under-covers where the noise is large: the per-group report shows it
    assert st.groups[-1]["coverage"] < 0.86 and st.groups[0]["coverage"] > 0.97
    # normalised scores with the right scale fix the conditional coverage
    st2 = conformal_study(y[cal], pred[cal], y[test], pred[test], alpha=0.1, scale_cal=sd[cal], scale_test=sd[test],
                          group_by=x[test])
    assert all(0.85 < g["coverage"] < 0.96 for g in st2.groups)
    assert [r["nominal"] for r in st.reliability] == sorted(r["nominal"] for r in st.reliability)
    with pytest.raises(ValueError, match="too few"):
        conformal_study(y[:5], pred[:5], y[test], pred[test], alpha=0.1)


def test_propagation_recovers_the_ishigami_moments():
    ex = ishigami_indices()
    for method in ("mc", "lhs", "sobol"):
        r = propagate_space(ishigami, ishigami_space(), n=2 ** 14, method=method)
        assert r.mean[0] == pytest.approx(3.5, abs=0.1)
        assert r.std[0] ** 2 == pytest.approx(ex["V"], rel=0.03)
    lim = propagate_space(lambda X: X[:, 0], [{"name": "u", "low": 0, "high": 1}], n=10000, method="mc",
                          limits=(None, 0.25))
    assert lim.p_within == pytest.approx(0.25, abs=0.015)
    assert lim.p_within_ci[0] < 0.25 < lim.p_within_ci[1]
    # the original nominal +- tolerance interface is unchanged
    old = propagate(lambda X: X.sum(axis=1), [1.0, 2.0], [0.1, 0.2], n=5000)
    assert old.mean[0] == pytest.approx(3.0, abs=0.01)


def test_polynomial_chaos_moments_and_indices():
    ex = ishigami_indices()
    p = polynomial_chaos(ishigami, ishigami_space(), degree=9, n=1200)
    assert p.mean[0] == pytest.approx(3.5, abs=0.02)
    assert p.std[0] ** 2 == pytest.approx(ex["V"], rel=0.01)
    np.testing.assert_allclose(p.S1[0], ex["S1"], atol=0.01)
    np.testing.assert_allclose(p.ST[0], ex["ST"], atol=0.01)
    assert p.loo_q2[0] > 0.99
    # a polynomial inside the basis is reproduced exactly: f = x1 + 2 x2^2 + x1 x3 on [0, 1]^3
    unit = [{"name": f"x{i}", "low": 0, "high": 1} for i in range(3)]
    q = polynomial_chaos(lambda X: X[:, 0] + 2 * X[:, 1] ** 2 + X[:, 0] * X[:, 2], unit, degree=3)
    # exact mean 1/2 + 2/3 + 1/4 and variance by expansion: var(x1 (1 + x3)) + var(2 x2^2), independent terms
    var = (1 / 3) * (1 + 1 + 1 / 3) - (0.5 * 1.5) ** 2 + 4 * (1 / 5 - 1 / 9)
    assert q.mean[0] == pytest.approx(17 / 12, abs=1e-10)
    assert q.std[0] ** 2 == pytest.approx(var, abs=1e-10)
    with pytest.raises(ValueError, match="chaos terms"):
        polynomial_chaos(ishigami, [{"name": f"x{i}", "low": 0, "high": 1} for i in range(12)], degree=5)


def test_applicability_gate_flags_holes_and_out_of_range_points():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (800, 2))
    X = X[np.hypot(X[:, 0] - 0.5, X[:, 1] - 0.5) > 0.2]            # a hole in the middle of the range
    dom = ApplicabilityDomain.fit(X, ["u", "v"], bounds={"u": (0, 1), "v": (0, 1)})
    a = dom.assess([[0.5, 0.5], [0.1, 0.9], [1.5, 0.5]])
    assert list(a["status"]) == ["sparse", "inside", "outside"]
    assert a["out_of_range"][2] == ["u"]
    assert a["density_ratio"][1] >= 1 > a["density_ratio"][0]
    assert dom.bounds_source.startswith("declared")


def test_routing_rule_reasons():
    routes = route_cases(["inside", "inside", "sparse", "outside"], pred=[1, 2, 1, 1],
                         lower=[0.9, 1.5, 0.95, 0.9], upper=[1.1, 2.5, 1.05, 1.1], tolerance=0.3, limits=(None, 2.2))
    assert [r["route"] for r in routes] == ["surrogate", "high_fidelity", "high_fidelity", "high_fidelity"]
    assert routes[0]["verdict"] == "meets limits"
    assert set(routes[1]["codes"]) == {"too_uncertain", "straddles_limit"}
    s = routing_summary(routes)
    assert s["high_fidelity"] == 3 and s["reasons"]["sparse_data"] == 1 and s["reasons"]["outside_range"] == 1


def test_uncertainty_study_end_to_end_with_a_known_truth():
    from pinneapple_analysis.study_models import StudyModel
    from pinneapple_analysis.uncertainty.study import uncertainty_study

    def truth(X):
        return np.sin(3 * X[:, 0]) + X[:, 1] ** 2

    def surrogate(X):                                       # exact inside [0, 1], drifts outside
        return truth(X) + 0.3 * np.clip(X[:, 0] - 1, 0, None)

    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (1200, 2))
    y = truth(X) + rng.normal(0, 0.05, 1200)
    model = StudyModel.from_function(surrogate, ["a", "b"], ["y"], bounds={"a": (0, 1), "b": (0, 1)})
    cases = np.array([[0.1, 0.1], [1.6, 0.1], [0.5, 0.52]])
    out = uncertainty_study(model, "y", X[:600], y[:600], X[600:], y[600:], alpha=0.1, X_ref=X[:600],
                            bounds=model.bounds, cases=cases, case_names=["mid", "beyond", "near limit"],
                            tolerance=0.5, limits=(None, float(surrogate(cases[2:])[0]) + 0.01),
                            high_fidelity=truth, hf_scope="all", nominal={"b": 0.5},
                            uncertain=[{"name": "a", "dist": "uniform", "low": 0.2, "high": 0.8}], n_propagation=2048)
    c = out["conformal"]
    assert c["n_cal"] == 600 and c["n_test"] == 600
    assert abs(c["coverage"] - 0.9) < 0.05                           # one calibration draw: marginal guarantee only
    assert c["q"] == pytest.approx(1.645 * 0.05, rel=0.15)          # noise quantile
    rows = {r["name"]: r for r in out["cases"]["rows"]}
    assert rows["mid"]["route"] == "surrogate" and rows["mid"]["status"] == "inside"
    assert rows["beyond"]["route"] == "high_fidelity" and "outside_range" in rows["beyond"]["codes"]
    assert rows["beyond"]["hf_error"] == pytest.approx(0.3 * 0.6, rel=1e-9)
    assert "straddles_limit" in rows["near limit"]["codes"]
    prop = out["propagation"]
    assert prop["n"] == 2048 and prop["share_outside"] == 0.0
    assert prop["pce"]["loo_q2"] > 0.99
    assert prop["p_within_with_model_error"] <= prop["p_within"]
    with pytest.raises(ValueError, match="nominal"):
        uncertainty_study(model, "y", X[:600], y[:600], X[600:], y[600:],
                          uncertain=[{"name": "a", "dist": "uniform", "low": 0.2, "high": 0.8}])

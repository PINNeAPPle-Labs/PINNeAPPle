"""Sensitivity estimators against exact answers: Ishigami and Sobol' G indices, a linear model for Morris."""
import numpy as np
import pytest

from pinneapple_analysis.sensitivity import (
    effect_curves,
    first_order_given_data,
    freeze_recommendation,
    local_derivatives,
    morris,
    saltelli_design,
    sobol_from_evaluations,
    sobol_indices,
    verify_freeze,
)
from pinneapple_analysis.sensitivity.testfunctions import (
    ishigami,
    ishigami_indices,
    ishigami_space,
    sobol_g,
    sobol_g_indices,
    sobol_g_space,
)

G_A = [0.0, 1.0, 4.5, 9.0, 99.0, 99.0]


def test_ishigami_first_total_and_second_order_match_the_exact_values():
    r = sobol_indices(ishigami, ishigami_space(), n=4096, second_order=True, n_boot=100)
    ex = ishigami_indices()
    assert r.n_evals == 4096 * (2 * 3 + 2)
    np.testing.assert_allclose(r.S1[0], ex["S1"], atol=0.02)
    np.testing.assert_allclose(r.ST[0], ex["ST"], atol=0.02)
    assert r.S2[0, 0, 2] == pytest.approx(ex["S2"][(0, 2)], abs=0.03)
    assert abs(r.S2[0, 0, 1]) < 0.03 and abs(r.S2[0, 1, 2]) < 0.03
    # the bootstrap intervals contain the exact values
    assert np.all(r.S1_ci[0, 0] <= ex["S1"] + 1e-9) and np.all(ex["S1"] <= r.S1_ci[1, 0] + 1e-9)
    assert np.all(r.ST_ci[0, 0] <= ex["ST"] + 1e-9) and np.all(ex["ST"] <= r.ST_ci[1, 0] + 1e-9)
    assert r.ranking() == ["x1", "x2", "x3"]
    np.testing.assert_allclose(r.interaction[0], ex["ST"] - ex["S1"], atol=0.03)


def test_sobol_g_indices_and_freeze_recommendation():
    def fn(X):
        return sobol_g(X, G_A)

    r = sobol_indices(fn, sobol_g_space(6), n=4096, n_boot=100)
    ex = sobol_g_indices(G_A)
    np.testing.assert_allclose(r.S1[0], ex["S1"], atol=0.02)
    np.testing.assert_allclose(r.ST[0], ex["ST"], atol=0.02)
    rec = {row["name"]: row["decision"] for row in freeze_recommendation(r, threshold=0.005)}
    assert rec["x1"] == rec["x2"] == rec["x3"] == "keep"
    assert rec["x5"] == rec["x6"] == "freeze"
    check = verify_freeze(fn, sobol_g_space(6), ["x5", "x6"], n=512)
    assert 0.9 < check["variance_ratio"][0] < 1.1
    assert check["p95_change_sd"][0] < 0.1


def test_indices_from_external_runs_equal_the_direct_call():
    X, layout = saltelli_design(ishigami_space(), 256, seed=3)
    a = sobol_from_evaluations(ishigami(X), layout, n_boot=0)
    b = sobol_indices(ishigami, ishigami_space(), n=256, n_boot=0, seed=3)
    np.testing.assert_allclose(a.ST, b.ST)
    with pytest.raises(ValueError, match="NaN"):
        y = ishigami(X)
        y[3] = np.nan
        sobol_from_evaluations(y, layout)


def test_multi_output_and_non_uniform_inputs():
    space = [{"name": "a", "dist": "normal", "mean": 0, "std": 1}, {"name": "b", "dist": "normal", "mean": 0, "std": 2}]
    def fn(X):
        return np.column_stack([X[:, 0] + X[:, 1], 3 * X[:, 0]])

    r = sobol_indices(fn, space, n=2048, n_boot=50, outputs=["sum", "a3"])
    # y = a + b with var(a) = 1, var(b) = 4: S = (0.2, 0.8); second output depends on a only
    np.testing.assert_allclose(r.S1[0], [0.2, 0.8], atol=0.02)
    np.testing.assert_allclose(r.ST[1], [1.0, 0.0], atol=0.02)
    assert r.outputs == ["sum", "a3"]


def test_morris_on_a_linear_model_returns_the_coefficients():
    c = np.array([3.0, -2.0, 0.5])
    m = morris(lambda X: X @ c, sobol_g_space(3), r=10)
    np.testing.assert_allclose(m.mu[0], c, atol=1e-9)
    np.testing.assert_allclose(m.mu_star[0], np.abs(c), atol=1e-9)
    np.testing.assert_allclose(m.sigma[0], 0.0, atol=1e-9)
    assert m.n_evals == 10 * 4


def test_morris_ranks_the_g_function_inputs_by_importance():
    m = morris(lambda X: sobol_g(X, G_A), sobol_g_space(6), r=60, seed=1)
    assert m.ranking()[:4] == ["x1", "x2", "x3", "x4"]
    assert np.all(m.mu_star_ci[0, 0] <= m.mu_star[0]) and np.all(m.mu_star[0] <= m.mu_star_ci[1, 0])


def test_given_data_first_order_on_a_random_ishigami_sample():
    X = np.random.default_rng(0).uniform(-np.pi, np.pi, (20000, 3))
    out = first_order_given_data(X, ishigami(X), n_boot=20)
    np.testing.assert_allclose(out["S1"], ishigami_indices()["S1"], atol=0.02)
    assert abs(out["max_input_correlation"]["r"]) < 0.05                # independent sample


def test_local_derivatives_autodiff_and_finite_differences_agree_with_the_analytic_gradient():
    torch = pytest.importorskip("torch")
    x0 = np.array([0.4, -1.1, 2.0])
    exact = np.array([np.cos(0.4) * (1 + 0.1 * 2.0 ** 4), 7 * 2 * np.sin(-1.1) * np.cos(-1.1),
                      0.4 * 2.0 ** 3 * np.sin(0.4)])
    fd = local_derivatives(ishigami, x0, ishigami_space())
    def tf(x):
        return torch.sin(x[:, 0]) + 7 * torch.sin(x[:, 1]) ** 2 + 0.1 * x[:, 2] ** 4 * torch.sin(x[:, 0])

    ad = local_derivatives(ishigami, x0, ishigami_space(), torch_fn=tf)
    np.testing.assert_allclose(fd["jacobian"][0], exact, rtol=1e-5)
    np.testing.assert_allclose(ad["jacobian"][0], exact, rtol=1e-5)
    assert "automatic" in ad["method"] and "finite" in fd["method"]
    np.testing.assert_allclose(ad["scaled"][0], exact * 2 * np.pi, rtol=1e-5)


def test_effect_curves_show_additive_and_interacting_inputs():
    ec = effect_curves(ishigami, ishigami_space(), n_grid=11, n_background=64)
    by = {c["name"]: c for c in ec["curves"]}
    assert ec["n_evals"] == 3 * 11 * 64
    assert by["x2"]["ice_spread"][0] < 1e-9           # x2 enters additively: parallel ICE curves
    assert by["x3"]["ice_spread"][0] > 1.0            # x3 acts only through x1 sin(x1): crossing ICE curves
    assert by["x3"]["pdp_range"][0] < 0.1             # ... and has no main effect


def test_indices_from_runs_done_elsewhere_in_any_order():
    from pinneapple_analysis.sensitivity import sobol_from_runs
    X, _ = saltelli_design(ishigami_space(), 512, seed=2)
    order = np.random.default_rng(0).permutation(X.shape[0])
    r = sobol_from_runs(ishigami_space(), 512, order, ishigami(X[order]), seed=2, n_boot=0)
    np.testing.assert_allclose(r.ST, sobol_indices(ishigami, ishigami_space(), n=512, seed=2, n_boot=0).ST)
    with pytest.raises(ValueError, match="missing"):
        sobol_from_runs(ishigami_space(), 512, order[:-3], ishigami(X[order[:-3]]), seed=2)


def test_study_models_builtin_zoo_and_onnx():
    from pinneapple_analysis.study_models import StudyModel, builtin
    ish = builtin("ishigami")
    assert ish.predict([[0.0, 0.0, 0.0]]).shape == (1, 1)
    np.testing.assert_allclose(ish.metadata["exact"]["S1"], ishigami_indices()["S1"])
    hs = StudyModel.from_zoo("heatsink_forced_mlp", outputs=["r_base_k_w", "pressure_drop_pa"])
    X, Y = hs.reference("test")
    assert X.shape == (900, 12) and hs.torch_fn is not None and hs.std_fn is None
    rel = np.abs(hs.predict(X) - Y) / Y
    assert rel[:, 0].mean() < 0.01                       # the card's held-out 0.71 % mean error
    ld = local_derivatives(hs.predict, X[0], torch_fn=hs.torch_fn)
    fd = local_derivatives(hs.predict, X[0], rel_step=1e-3)
    np.testing.assert_allclose(ld["jacobian"], fd["jacobian"], rtol=0.05, atol=1e-6)
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from pinneapple_hub import zoo
    from pinneapple_hub.zoo.export import export_bytes
    graph = StudyModel.from_onnx(export_bytes(zoo.load("heatsink_forced_mlp")), input_names=hs.input_names,
                                 output_names=["r_base_k_w", "pressure_drop_pa", "r_upper"])
    np.testing.assert_allclose(graph.predict(X[:50])[:, :2], hs.predict(X[:50]), rtol=1e-4)


def test_sensitivity_study_on_ishigami_reports_errors_against_the_exact_indices():
    from pinneapple_analysis.sensitivity import sensitivity_study, study_cost
    from pinneapple_analysis.study_models import builtin
    b = builtin("ishigami")
    out = sensitivity_study(b, b.metadata["space"], n_sobol=2048, second_order=True, n_grid=9, n_background=32,
                            exact=b.metadata["exact"])
    assert out["exact"]["max_abs_error_S1"] < 0.02 and out["exact"]["max_abs_error_ST"] < 0.02
    assert out["sobol"]["ranking"]["y"] == ["x1", "x2", "x3"]
    cost = study_cost(3, 2048, True, n_grid=9, n_background=32)
    assert out["n_evals"]["sobol"] == cost["parts"]["sobol"] and out["n_evals"]["effects"] == cost["parts"]["effects"]
    assert {r["name"]: r["decision"] for r in out["freeze"]["rows"]} == {"x1": "keep", "x2": "keep", "x3": "keep"}


def test_sensitivity_study_freezes_an_inert_input_and_checks_it():
    from pinneapple_analysis.sensitivity import sensitivity_study
    from pinneapple_analysis.study_models import StudyModel

    def f(X):
        return np.column_stack([X[:, 0] ** 2 + 0.5 * X[:, 1], X[:, 0] + X[:, 2] * 0.0])

    m = StudyModel.from_function(f, ["a", "b", "c", "k"], ["y", "z"])
    space = [{"name": n, "low": 0, "high": 1} for n in ("a", "b", "c")]
    out = sensitivity_study(m, space, fixed={"k": 3.0}, n_sobol=512, effects_on=False)
    dec = {r["name"]: r["decision"] for r in out["freeze"]["rows"]}
    assert dec == {"a": "keep", "b": "keep", "c": "freeze"}
    assert out["freeze"]["check"]["max_change_sd"] == [0.0, 0.0]
    with pytest.raises(ValueError, match="fixed value"):
        sensitivity_study(m, space)

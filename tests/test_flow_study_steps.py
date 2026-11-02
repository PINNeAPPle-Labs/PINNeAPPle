"""Study steps that take a model as data (pinneapple_flow.study_steps). References: the Ishigami function's exact Sobol
indices; a bending stress linear in the load (6 F L / (w t^2)), so its propagated mean and spread are exact; a known
thickness recovered from noisy stresses; an AR(2) series a forecast must beat persistence on, and white noise it
must not claim skill on."""
import math

import numpy as np
import pytest

from pinneapple_analysis.sensitivity.testfunctions import ishigami_indices
from pinneapple_flow.study_steps import (
    analyze_step,
    forecast_step,
    identify_step,
    model_from_ref,
    sensitivity_step,
    uncertainty_step,
)

STRESS = {"tool": "pinneapple_physics.closed_form.cantilever_fatigue:cantilever_root_bending_stress_pa",
          "fixed": {"width_m": 0.05, "thickness_m": 0.01, "length_m": 0.5}, "inputs": ["load_n"],
          "outputs": ["sigma"]}


def _sigma(load, length=0.5, w=0.05, t=0.01):
    return 6 * load * length / (w * t * t)


def test_a_model_reference_runs_a_library_tool_and_nothing_else():
    f = model_from_ref(STRESS)
    assert f({"load_n": 1000.0})["sigma"] == pytest.approx(_sigma(1000.0))
    with pytest.raises(PermissionError):
        model_from_ref({"tool": "os:system", "inputs": ["x"], "outputs": ["y"]})


def test_sensitivity_recovers_the_ishigami_indices():
    ref = {"tool": "pinneapple_analysis.sensitivity.testfunctions:ishigami", "call": "array",
           "inputs": ["x1", "x2", "x3"], "outputs": ["y"]}
    r = sensitivity_step(ref, {k: [-math.pi, math.pi] for k in ("x1", "x2", "x3")}, "y", n=1024)
    exact = ishigami_indices()
    for i, k in enumerate(("x1", "x2", "x3")):
        assert r["S1"][k] == pytest.approx(exact["S1"][i], abs=0.06)
        assert r["ST"][k] == pytest.approx(exact["ST"][i], abs=0.06)
    assert r["ranking"][0] in ("x1", "x2") and all(c["status"] == "pass" for c in r["checks"])


def test_uncertainty_of_a_linear_response_is_exact_within_sampling_error():
    r = uncertainty_step(STRESS, {"load_n": {"dist": "normal", "mean": 1000.0, "std": 50.0}}, "sigma", n=4000,
                         limit=_sigma(1100.0))
    assert r["mean"] == pytest.approx(_sigma(1000.0), rel=0.005)
    assert r["std"] == pytest.approx(_sigma(50.0), rel=0.05)
    assert r["p_exceed"]["p"] == pytest.approx(0.02275, abs=4 * r["p_exceed"]["se"])     # P(Z > 2)
    assert r["checks"][0]["status"] == "pass"


def test_identification_recovers_a_known_thickness_and_checks_held_out_rows():
    rng = np.random.default_rng(0)
    loads = np.linspace(200.0, 2000.0, 30)
    noise = 0.01 * _sigma(1000.0)
    meas = {"inputs": {"load_n": loads.tolist()}, "values": (_sigma(loads, t=0.012) +
                                                              rng.normal(0, noise, 30)).tolist(), "sigma": noise}
    model = {"tool": STRESS["tool"], "fixed": {"width_m": 0.05, "length_m": 0.5}, "outputs": ["sigma"]}
    r = identify_step(model, {"thickness_m": {"initial": 0.008, "lower": 0.002, "upper": 0.05, "unit": "m"}}, meas,
                      "sigma", bootstrap=0)
    assert r["estimates"]["thickness_m"] == pytest.approx(0.012, rel=0.01)
    lo, hi = r["intervals"]["thickness_m"]
    assert lo < 0.012 < hi
    assert {c["code"]: c["status"] for c in r["checks"]} == {"identify.fits_data": "pass",
                                                            "identify.identifiable": "pass",
                                                            "verify.held_out": "pass"}


def test_forecast_skill_is_claimed_only_when_it_beats_the_baselines():
    rng = np.random.default_rng(1)
    x = np.zeros(600)
    for t in range(2, 600):                                          # AR(2) with oscillation: predictable
        x[t] = 1.6 * x[t - 1] - 0.8 * x[t - 2] + rng.normal()
    good = forecast_step(x, 5)
    assert good["order"] >= 2 and good["checks"][1]["status"] == "pass"
    noise = forecast_step(rng.normal(size=600), 5)
    assert noise["checks"][1]["status"] == "fail"                    # white noise: no skill over the mean


def test_analysis_tests_a_random_walk_on_its_differences():
    walk = np.cumsum(np.random.default_rng(2).normal(size=2000))                # AR(1) estimate above 0.99
    r = analyze_step(walk)
    assert r["report"]["tested_on"] == "differences" and r["checks"][0]["value"] == "differences"


def test_a_constant_series_is_refused_not_divided_by_zero():
    with pytest.raises(ValueError, match="constant"):
        forecast_step(np.zeros(200), 3)

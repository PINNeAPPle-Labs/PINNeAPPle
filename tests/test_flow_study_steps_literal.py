"""Literal calls of the study kinds and study steps (the Studio harvests its smoke inputs from them); small sizes,
the reference checks are in test_flow_studies.py and test_flow_study_steps.py."""
from pinneapple_flow.studies import classify_study, role_checks, roles_for_kind
from pinneapple_flow.study_steps import (
    analyze_step,
    forecast_step,
    identify_step,
    model_from_ref,
    sensitivity_step,
    uncertainty_step,
)


def test_kinds_literal():
    assert classify_study("estimate the conductivity from the measurements")[0].kind == "inverse"
    assert roles_for_kind("forecast")[-1] == "verify"
    assert role_checks("forecast", "verify")[0][0] == "verify.beats_baseline"


def test_steps_literal():
    f = model_from_ref({"tool": "pinneapple_physics.closed_form.cantilever_fatigue:cantilever_root_bending_stress_pa",
                        "fixed": {"width_m": 0.05, "thickness_m": 0.01, "length_m": 0.5}, "inputs": ["load_n"],
                        "outputs": ["sigma"]})
    assert abs(f({"load_n": 1000.0})["sigma"] / 6e8 - 1) < 1e-12
    r = uncertainty_step({"tool": "pinneapple_physics.closed_form.cantilever_fatigue:cantilever_root_bending_stress_pa",
                          "fixed": {"width_m": 0.05, "thickness_m": 0.01, "length_m": 0.5}, "inputs": ["load_n"],
                          "outputs": ["sigma"]}, {"load_n": {"dist": "normal", "mean": 1000.0, "std": 50.0}}, "sigma",
                         n=400)
    assert abs(r["mean"] / 6e8 - 1) < 0.02
    s = sensitivity_step({"tool": "pinneapple_analysis.sensitivity.testfunctions:ishigami", "call": "array",
                          "inputs": ["x1", "x2", "x3"], "outputs": ["y"]},
                         {"x1": [-3.1416, 3.1416], "x2": [-3.1416, 3.1416], "x3": [-3.1416, 3.1416]}, "y", n=128)
    assert set(s["S1"]) == {"x1", "x2", "x3"}
    i = identify_step({"tool": "pinneapple_physics.closed_form.cantilever_fatigue:cantilever_root_bending_stress_pa",
                       "fixed": {"width_m": 0.05, "length_m": 0.5}, "outputs": ["sigma"]},
                      {"thickness_m": {"initial": 0.008, "lower": 0.002, "upper": 0.05, "unit": "m"}},
                      {"inputs": {"load_n": [200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0, 1400.0, 1600.0, 1800.0,
                                             2000.0]},
                       "values": [8.33e7, 1.667e8, 2.5e8, 3.333e8, 4.167e8, 5.0e8, 5.833e8, 6.667e8, 7.5e8, 8.333e8],
                       "sigma": 2e6}, "sigma", bootstrap=0)
    assert abs(i["estimates"]["thickness_m"] - 0.012) < 2e-4
    a = analyze_step([0.0, 0.0, 2.041, 0.71, -0.079, -1.262, -2.409, -3.06, -4.989, -5.766, -6.1, -1.824, 2.187, 4.606, 5.339, 4.189, 1.376, -1.54, -3.083, -3.939, -2.879, -1.655, -0.32, 2.358, 4.573, 4.926, 4.04, 3.064, 3.605, 3.048, 1.748, 1.362, -0.106, -1.551, -1.514, -0.602, 0.34, 1.696, -0.387, -0.954, -2.177, -4.389, -5.004, -3.794, -2.513, -2.061, -1.262, -0.423, 1.739, 3.868, 4.991, 6.003, 5.407, 2.922, 0.934, -0.26, -1.379, -2.781, -3.117, -5.256, -5.227, -3.666, -3.323, -2.323, -2.022, -0.62, -1.408, -2.672, -2.439, -0.608, -1.18, -1.899, -1.767, -1.917, -0.063, 0.242, 0.792, 0.025, 0.812, 1.258, 0.991, -1.139, -0.933, 0.17, 1.773, 3.838, 5.072, 4.406, 2.191, -0.819, -1.693, -3.514, -4.865, -5.293, -4.353, -2.155, -1.214, -1.949, -2.151, -0.67, 1.407, 3.002, 3.361, 3.269, 2.298, 1.879, 0.374, -0.771, -1.644, -1.47, -0.812, 2.427, 6.031, 9.205, 7.864, 4.877, 0.904, -1.922, -6.078, -7.013])
    assert a["report"]["tested_on"] in ("levels", "differences")
    fc = forecast_step([0.0, 0.0, 2.041, 0.71, -0.079, -1.262, -2.409, -3.06, -4.989, -5.766, -6.1, -1.824, 2.187, 4.606, 5.339, 4.189, 1.376, -1.54, -3.083, -3.939, -2.879, -1.655, -0.32, 2.358, 4.573, 4.926, 4.04, 3.064, 3.605, 3.048, 1.748, 1.362, -0.106, -1.551, -1.514, -0.602, 0.34, 1.696, -0.387, -0.954, -2.177, -4.389, -5.004, -3.794, -2.513, -2.061, -1.262, -0.423, 1.739, 3.868, 4.991, 6.003, 5.407, 2.922, 0.934, -0.26, -1.379, -2.781, -3.117, -5.256, -5.227, -3.666, -3.323, -2.323, -2.022, -0.62, -1.408, -2.672, -2.439, -0.608, -1.18, -1.899, -1.767, -1.917, -0.063, 0.242, 0.792, 0.025, 0.812, 1.258, 0.991, -1.139, -0.933, 0.17, 1.773, 3.838, 5.072, 4.406, 2.191, -0.819, -1.693, -3.514, -4.865, -5.293, -4.353, -2.155, -1.214, -1.949, -2.151, -0.67, 1.407, 3.002, 3.361, 3.269, 2.298, 1.879, 0.374, -0.771, -1.644, -1.47, -0.812, 2.427, 6.031, 9.205, 7.864, 4.877, 0.904, -1.922, -6.078, -7.013], 3)
    assert fc["order"] >= 1 and len(fc["forecast"]) == 3

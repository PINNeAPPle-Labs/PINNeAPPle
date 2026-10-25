"""pinneapple_analysis.uncertainty.margin on the brake rotor of pinneapple_design.surrogate_inverse.brake_rotor (ported
from the tests of PINNeAPPle-apps poc_verification_gap)."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_analysis.uncertainty.margin import corner_stackup, margin_study, summarize_samples
from pinneapple_design.surrogate_inverse.brake_rotor import (
    BASELINE_DESIGN,
    RotorDesign,
    rotor_temperature_c,
    thermal_field,
)

R = BASELINE_DESIGN
TOL = [0.05e-3, 0.1e-3]


def temp(X, h=40.0):
    return np.array([rotor_temperature_c(RotorDesign(R.vane_count, g, t), 1480.0, h) for t, g in X])


def test_nominal_matches_the_source_projects_baseline():
    assert rotor_temperature_c(R, 1480.0, 40.0) == pytest.approx(495.90246514504753, abs=1e-9)


def test_highway_airflow_is_much_cooler_and_the_vane_is_hottest_at_the_root():
    assert rotor_temperature_c(R, 1480.0, 120.0) < rotor_temperature_c(R, 1480.0, 40.0) - 200
    p = thermal_field(R, 1480.0, 40.0)
    assert p["theta_vane_base_c"] > 0 and p["lc_radial_m"] > 0


def test_worst_corner_is_thinnest_vane_and_smallest_gap():
    c = corner_stackup(temp, [0.004, 0.008], TOL)
    assert c["worst_corner"] == pytest.approx([0.004 - TOL[0], 0.008 - TOL[1]])
    assert c["worst_value"] == pytest.approx(temp([[0.004 - TOL[0], 0.008 - TOL[1]]])[0])
    assert c["worst_value"] > rotor_temperature_c(R, 1480.0, 40.0) > c["best_value"]


def test_seeded_study_is_reproducible_and_marginal():
    a = margin_study(temp, [0.004, 0.008], TOL, 500.0, seed=42, check_seed=2)
    b = margin_study(temp, [0.004, 0.008], TOL, 500.0, seed=42, check_seed=2)
    assert a["samples"] == b["samples"]
    s = a["summary"]
    assert s["verdict"] == "MARGINAL" and s["safe_fraction"] == 1.0
    assert 497.5 < s["p95"] < 498.4 < 500.0          # corner stack-up 498.38 C bounds the sample p95
    assert len({c["name"] for c in a["checks"]}) == 6
    assert a["overall_verdict"] == "WARNING" and a["decision"] == "REVIEW"
    assert s["p95"] <= a["corner"]["worst_value"]


def test_fresh_draws_differ_without_a_seed():
    a = margin_study(temp, [0.004, 0.008], TOL, 500.0, n=50)
    b = margin_study(temp, [0.004, 0.008], TOL, 500.0, n=50)
    assert a["samples"] != b["samples"]


def test_verdicts_and_decision_follow_the_p95():
    assert summarize_samples([505.0] * 200, 500.0)["verdict"] == "FRAGILE"
    assert summarize_samples([400.0] * 200, 500.0)["verdict"] == "ROBUST"
    hot = margin_study(lambda X: 505.0 + 0 * X[:, 0], [1.0], [0.1], 500.0, seed=0, check_seed=1)
    assert hot["overall_verdict"] == "FAIL" and hot["decision"] == "REJECT"
    cool = margin_study(lambda X: 400.0 + X[:, 0], [1.0], [0.1], 500.0, seed=0, check_seed=1)
    assert cool["overall_verdict"] == "PASS" and cool["decision"] == "APPROVED"

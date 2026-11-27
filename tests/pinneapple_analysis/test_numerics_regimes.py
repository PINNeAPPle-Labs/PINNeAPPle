"""Tests for ``pinneapple_analysis.numerics.regimes`` (#207, roadmap X23).

The claim the issue asks to be checked is the diagnosis report on three known cases -- a stiff ODE, a
boundary layer and a shock -- listing the expected strategies. Every case here is a problem with a known
answer, so the tests compare the measured numbers with the analytic ones and the strategies with the ones
the regime calls for. A detector that fires on everything would pass a "is it detected" test; that is why
each working case has a neighbour that must report ``smooth`` and no strategy at all.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from pinneapple_analysis.numerics.diagnostics import explain
from pinneapple_analysis.numerics.ode import robertson_rhs, stiffness_ratio, van_der_pol_rhs
from pinneapple_analysis.numerics.regimes import (
    STRATEGIES,
    Thresholds,
    diagnose,
    diagnose_field,
    diagnose_ode,
    strategies_for,
)

#: a uniform line fine enough to resolve a layer of 1 % of the domain (five points across it)
FINE = np.linspace(0.0, 1.0, 2001)
COARSE = np.linspace(0.0, 1.0, 201)
DELTA = 1e-2


def boundary_layer(x: np.ndarray, delta: float = DELTA) -> np.ndarray:
    """``1 - exp(-x/delta)``: an exponential layer against the wall at ``x = 0``, monotone, full jump."""
    return 1.0 - np.exp(-x / delta)


def shock(x: np.ndarray, width: float = DELTA) -> np.ndarray:
    """``tanh((x - 1/2)/width)``: a jump from -1 to +1 across the middle of the domain."""
    return np.tanh((x - 0.5) / width)


# --------------------------------------------------------------------------- stiff ODE
def test_stiff_ode_reports_stiff_with_the_implicit_strategies():
    """Robertson's kinetics: the measured ratio is the 1e9 the module docstring of the RHS claims."""
    report = diagnose_ode(robertson_rhs, [1.0, 0.0, 0.0], (0.0, 1e5))
    assert report.regimes == ("stiff",)
    assert report.evidence.stiffness_ratio > 1e6
    # the reference: the eigenvalues of the local Jacobian are ~ -1e7 and ~ -1e-2
    assert report.evidence.stiffness_ratio > 1e8
    assert "implicit_solver" in report.codes()
    assert "adaptive_time_stepping" in report.codes()
    assert "time_marching" in report.codes()
    assert report.diagnostics[0].code == "ODE.STIFF"


def test_a_non_stiff_oscillator_is_not_reported_stiff_but_oscillatory():
    """Van der Pol at mu = 1: ratio 1 (not stiff), eigenvalues -0.5 +- 0.866i (turning)."""

    def harmonic(t, y):
        return np.array([y[1], -y[0]])

    report = diagnose_ode(harmonic, [1.0, 0.0], (0.0, 2.0 * np.pi))
    assert report.regimes == ("oscillatory",)
    assert report.evidence.oscillation_index == pytest.approx(1.0)
    assert "fourier_features" in report.codes()
    assert "causal_training" in report.codes()

    vdp = diagnose_ode(van_der_pol_rhs(1.0), [1.0, 0.0], (0.0, 10.0))
    assert vdp.regimes == ("oscillatory",)
    assert vdp.evidence.stiffness_ratio < 1e3


def test_van_der_pol_at_large_mu_is_stiff():
    """mu = 1000: the relaxation oscillation is fast in one branch and slow in the other."""
    report = diagnose_ode(van_der_pol_rhs(1e3), [2.0, 0.0], (0.0, 3000.0))
    assert report.regimes == ("stiff",)
    assert report.evidence.stiffness_ratio > 1e3


def test_the_stiffness_is_local_and_the_report_samples_it():
    """A local measure changes along the trajectory, which is why the diagnosis samples it (#1060)."""
    from pinneapple_analysis.numerics.ode import stiffness_report

    # at the initial condition y2 = y3 = 0, so only the -0.04 mode exists: this point is not stiff
    at_start = stiffness_ratio(robertson_rhs, 0.0, np.array([1.0, 0.0, 0.0]))
    assert at_start < 1e3
    report = stiffness_report(robertson_rhs, [1.0, 0.0, 0.0], (0.0, 1e5))
    assert report["max_stiffness_ratio"] > 1e8
    assert report["is_stiff"] is True
    # the states are returned so a second measurement (the spectrum) needs no second integration
    assert len(report["states"]) == len(report["times"]) == 11


# --------------------------------------------------------------------------- boundary layer
def test_boundary_layer_thickness_matches_the_analytic_half_width():
    """The measured thickness is the FWHM of the gradient: for ``exp(-x/d)`` that is ``d ln 2``."""
    report = diagnose_field(boundary_layer(FINE), FINE)
    assert report.regimes == ("boundary_layer",)
    assert report.evidence.layer_location == "boundary"
    assert report.evidence.jump_ratio == pytest.approx(1.0)
    assert report.evidence.layer_thickness == pytest.approx(DELTA * np.log(2), rel=0.05)
    # 1/(d) divided by (range/L) = 1/d: the gradient is this many times the field's typical slope
    assert report.evidence.gradient_concentration == pytest.approx(1.0 / DELTA, rel=0.05)
    assert report.diagnostics[0].code == "REGIME.BOUNDARY_LAYER"


def test_boundary_layer_lists_the_strategies_the_issue_expects():
    report = diagnose_field(boundary_layer(FINE), FINE)
    codes = report.codes()
    for expected in ("adaptive_sampling", "coordinate_scaling", "curriculum", "xpinn"):
        assert expected in codes, codes
    assert [s.code for s in report.strategies_for("boundary_layer")] == list(codes)


# --------------------------------------------------------------------------- shock
def test_shock_is_told_apart_from_a_boundary_layer_by_where_the_jump_is():
    """Same thickness, different location: the profile, not the sharpness, decides."""
    interior = diagnose_field(shock(FINE), FINE)
    assert interior.regimes == ("shock",)
    assert interior.evidence.layer_location == "interior"
    assert interior.evidence.jump_ratio == pytest.approx(1.0)     # -1 to +1 across the line
    # the FWHM of d/dx tanh(x/w) is 2 w atanh(1/sqrt 2) = 1.7627 w
    assert interior.evidence.layer_thickness == pytest.approx(2 * DELTA * np.arctanh(1 / np.sqrt(2)),
                                                              rel=0.05)


def test_shock_lists_the_strategies_the_issue_expects():
    report = diagnose_field(shock(FINE), FINE)
    codes = report.codes()
    for expected in ("adaptive_sampling", "causal_training", "viscosity_continuation", "xpinn"):
        assert expected in codes, codes
    assert "coordinate_scaling" not in codes    # that is the boundary-layer remedy, not this one


def test_a_thin_bump_is_not_a_shock_because_the_field_comes_back():
    """A thin interior core with no net change: sharp, but no Rankine-Hugoniot relation to impose."""
    bump = np.exp(-(((FINE - 0.5) / 0.005) ** 2))
    report = diagnose_field(bump, FINE)
    assert report.evidence.layer_location == "interior"
    assert report.evidence.layer_thickness < 0.02
    assert report.evidence.jump_ratio == pytest.approx(0.0, abs=1e-12)
    assert report.regimes == ("sharp_feature",)
    assert "viscosity_continuation" not in report.codes()
    assert "adaptive_sampling" in report.codes()


# --------------------------------------------------------------------------- what must not fire
@pytest.mark.parametrize(
    "field",
    [
        FINE ** 2 * (1.0 - FINE),                       # a cubic: fills the domain
        1.0 - np.exp(-FINE / 0.3),                      # a layer a third of the domain thick
        np.exp(-(((FINE - 0.5) / 0.02) ** 2)),          # a bump with 20 points across it
        np.full_like(FINE, 3.0),                        # a constant field
    ],
    ids=["cubic", "thick-layer", "wide-bump", "constant"],
)
def test_a_benign_field_reports_smooth_and_no_strategy(field):
    report = diagnose_field(field, FINE)
    assert report.regimes == ("smooth",), report.evidence.to_dict()
    assert report.codes() == ()
    assert report.diagnostics[0].code == "REGIME.SMOOTH"
    assert "[warning]" not in report.format()


# --------------------------------------------------------------------------- the other two regimes
def test_many_cycles_report_high_frequency_with_the_wavenumber_measured():
    field = np.sin(2 * np.pi * 40 * FINE)
    report = diagnose_field(field, FINE)
    assert report.regimes == ("high_frequency",)
    assert report.evidence.oscillation_count == 40
    assert report.evidence.to_dict()["dominant_wavenumber"] == 40
    assert report.codes() == ("fourier_features", "adaptive_sampling")


def test_two_separated_bands_are_multiscale_and_a_smooth_decay_is_not():
    two_band = np.sin(2 * np.pi * FINE) + 0.2 * np.sin(2 * np.pi * 30 * FINE)
    report = diagnose_field(two_band, FINE)
    assert report.regimes == ("multiscale",)
    assert report.evidence.band_ratio >= 10.0
    assert report.evidence.spectral_gap <= 1e-4
    assert report.codes() == ("curriculum", "xpinn", "adaptive_sampling")

    # a single exponential decays from the peak to the floor with no dip: band ratio high, gap not
    smooth_decay = diagnose_field(1.0 - np.exp(-FINE / 0.3), FINE)
    assert smooth_decay.evidence.band_ratio > 10.0
    assert smooth_decay.evidence.spectral_gap > 1e-4
    assert smooth_decay.regimes == ("smooth",)


# --------------------------------------------------------------------------- honesty of the numbers
def test_an_unresolved_layer_says_so_instead_of_reporting_a_confident_thickness():
    """201 points across 1 % of the domain: the measured thickness is barely more than the grid."""
    report = diagnose_field(boundary_layer(COARSE), COARSE)
    assert report.regimes == ("boundary_layer",)
    assert report.evidence.layer_points < 3.0
    codes = [d.code for d in report.diagnostics]
    assert "REGIME.UNRESOLVED" in codes


def test_the_thresholds_are_the_caller_s_and_change_the_answer():
    report = diagnose_field(boundary_layer(FINE), FINE)
    assert report.regimes == ("boundary_layer",)
    strict = diagnose_field(boundary_layer(FINE), FINE,
                            thresholds=Thresholds(layer_thickness=1e-3))
    assert strict.regimes == ("smooth",)


def test_evidence_from_someone_elses_measurement_goes_through_the_same_rules():
    """The rule table is separate from the measurement: a caller who already has the numbers can ask."""
    from pinneapple_analysis.numerics.regimes import RegimeEvidence

    evidence = RegimeEvidence(source="openfoam-slice", layer_thickness=0.004,
                              gradient_concentration=250.0, layer_location="boundary",
                              jump_ratio=1.0, layer_points=12.0, samples=400)
    report = diagnose(evidence)
    assert report.regimes == ("boundary_layer",)
    assert report.evidence.source == "openfoam-slice"


def test_a_rule_whose_number_is_missing_does_not_fire():
    from pinneapple_analysis.numerics.regimes import RegimeEvidence

    report = diagnose(RegimeEvidence(source="nothing"))
    assert report.regimes == ("smooth",)
    assert report.codes() == ()


# --------------------------------------------------------------------------- the report as data
def test_the_report_serialises_and_the_catalogue_covers_every_regime():
    report = diagnose_field(shock(FINE), FINE)
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["regimes"] == ["shock"]
    assert payload["codes"] == list(report.codes())
    assert payload["evidence"]["layer_location"] == "interior"
    assert all(s["regime"] in payload["regimes"] for s in payload["strategies"])

    # every regime a strategy is attached to is one the rules can report
    reported = {"stiff", "oscillatory", "boundary_layer", "shock", "sharp_feature", "high_frequency",
                "multiscale"}
    assert set(STRATEGIES) == reported
    assert strategies_for("smooth") == ()
    assert report.format().startswith("regimes: shock")


def test_the_diagnostic_codes_of_the_regimes_exist_in_the_catalogue():
    for code in ("REGIME.BOUNDARY_LAYER", "REGIME.SHOCK", "REGIME.SHARP_FEATURE",
                 "REGIME.HIGH_FREQUENCY", "REGIME.MULTISCALE", "REGIME.OSCILLATORY",
                 "REGIME.SMOOTH", "REGIME.UNRESOLVED"):
        assert explain(code).severity in ("info", "warning")


# --------------------------------------------------------------------------- refusals
def test_a_field_that_cannot_have_a_gradient_is_refused():
    with pytest.raises(ValueError, match="at least 8 samples"):
        diagnose_field([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="non-finite"):
        diagnose_field(np.r_[np.zeros(4), np.nan, np.zeros(4)])
    with pytest.raises(ValueError, match="strictly increasing"):
        diagnose_field(np.zeros(16), np.zeros(16))
    with pytest.raises(ValueError, match="entries for"):
        diagnose_field(np.zeros(16), np.linspace(0, 1, 15))

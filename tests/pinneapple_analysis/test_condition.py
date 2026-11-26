"""Tests for ``pinneapple_analysis.condition`` -- automatic scaling and conditioning (#206).

The number every test here is checked against is computed by hand in the test:

* the Reynolds, Peclet and Froude numbers of a stated set of SI parameters, each written out as the formula;
* the loss-term ratio of the conduction problem, ``(f / (k dT / L^2))^2`` before the report and ``1`` after it;
* the weight the report returns, ``(target / magnitude)^2``, which is exact arithmetic and not an estimate.

The claim of the issue -- "on a benchmark PINN the auto-conditioned run converges faster or equal to the
hand-tuned baseline" -- is the last test, which trains the same network three ways and compares when each
arm first reached the tolerance.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from pinneapple_analysis.condition import (
    Scale,
    condition_report,
    dimensionless_groups,
    scaling_benchmark,
)

#: one engineering-scale conduction problem: a 0.1 m slab, k = 0.15 W/(m K), f = 3000 W/m^3
PROBLEM = {"length": 0.1, "conductivity": 0.15, "source": 3000.0, "ambient": 20.0}


# --------------------------------------------------------------------------- dimensionless groups
def test_reynolds_number_from_geometry_velocity_density_and_viscosity():
    """Re = rho U L / mu, the example the issue gives: 1000 x 2 x 1 / 0.1 = 20 000."""
    groups, missing = dimensionless_groups(
        {"density": 1000.0, "velocity": 2.0, "length": 1.0, "dynamic_viscosity": 0.1})
    assert groups["Re"] == pytest.approx(1000.0 * 2.0 * 1.0 / 0.1)
    assert groups["Re"] == pytest.approx(20000.0)
    # Fr = U / sqrt(g L) with the default g = 9.81
    assert groups["Fr"] == pytest.approx(2.0 / np.sqrt(9.81 * 1.0))
    assert "density" not in missing.get("Re", ())


def test_peclet_and_prandtl_are_the_ratio_the_formula_says():
    alpha = 1.4e-7
    groups, _ = dimensionless_groups({"density": 1000.0, "velocity": 2.0, "length": 1.0,
                                      "dynamic_viscosity": 0.1, "thermal_diffusivity": alpha})
    assert groups["Pe"] == pytest.approx(2.0 * 1.0 / alpha)
    assert groups["Pr"] == pytest.approx((0.1 / 1000.0) / alpha)
    assert groups["Pe"] == pytest.approx(groups["Re"] * groups["Pr"])


def test_a_group_that_cannot_be_built_says_what_it_needed():
    groups, missing = dimensionless_groups({"velocity": 2.0, "length": 1.0})
    assert "Re" not in groups
    assert missing["Re"] == ("density", "dynamic_viscosity")
    assert missing["Ma"] == ("sound_speed",)
    assert "Bi" in missing and set(missing["Bi"]) == {"h", "conductivity"}
    assert "gravity" not in " ".join(sum(missing.values(), ()))   # it has a default, so never the reason


def test_two_viscosities_that_contradict_each_other_are_refused():
    with pytest.raises(ValueError, match="disagrees with dynamic_viscosity/density"):
        dimensionless_groups({"density": 1000.0, "dynamic_viscosity": 0.1,
                              "kinematic_viscosity": 1e-3, "velocity": 1.0, "length": 1.0})


# --------------------------------------------------------------------------- the scale
def test_a_scale_sends_the_declared_range_to_minus_one_and_one():
    scale = Scale.from_range("x", 0.0, 0.1)
    assert scale.centre == pytest.approx(0.05) and scale.scale == pytest.approx(0.05)
    assert np.allclose(scale.apply([0.0, 0.05, 0.1]), [-1.0, 0.0, 1.0])
    assert np.allclose(scale.invert(scale.apply([0.02, 0.07])), [0.02, 0.07])


def test_a_scale_that_cannot_be_built_is_refused():
    with pytest.raises(ValueError, match="zero width"):
        Scale.from_range("T", 5.0, 5.0)
    with pytest.raises(ValueError, match="must be finite"):
        Scale.from_range("T", 0.0, np.inf)
    with pytest.raises(ValueError, match="no values"):
        Scale.from_values("T", [])


# --------------------------------------------------------------------------- the report
def test_the_weight_is_the_arithmetic_that_makes_the_term_of_order_one():
    """weight * magnitude^2 == target, by hand: (1/3000)^2 * 3000^2 = 1."""
    report = condition_report(residual_terms={"conduction": PROBLEM["source"]})
    weight = report.weight("conduction")
    assert weight == pytest.approx((1.0 / PROBLEM["source"]) ** 2)
    assert weight * PROBLEM["source"] ** 2 == pytest.approx(1.0)


def test_the_report_measures_the_ratio_the_conditioning_has_to_fix():
    """The two loss terms of this problem differ by a factor computed here, and the report levels them."""
    length, conductivity, source, ambient = 0.1, 0.15, 3000.0, 20.0
    rise = source * length**2 / (2.0 * conductivity)          # peak rise of the exact solution
    diffusion = conductivity * rise / length**2               # the magnitude of k T''
    boundary = rise                                           # the magnitude of T - ambient
    # raw loss terms: mean(r^2) of each, so the ratio is the square of the magnitudes
    raw_ratio = (source**2) / (boundary**2)
    assert raw_ratio == pytest.approx(900.0)                  # 9e6 against 1e4, exactly

    report = condition_report(
        coordinates={"x": (0.0, length)},
        fields={"T": (ambient, ambient + 0.5 * rise)},
        residual_terms={"conduction": diffusion},
        target=1.0,
    )
    # after the report, the same ratio is weight * source^2 / (rise_scale^2)
    weighted_residual = report.weight("conduction") * diffusion**2
    scaled_boundary = (report.scale("T").scale) ** 2
    assert weighted_residual == pytest.approx(1.0)
    assert scaled_boundary == pytest.approx(report.scale("T").scale**2)
    assert abs(weighted_residual / scaled_boundary - 1.0) < 1.0     # within a factor of one, not of 900


def test_the_report_scales_the_coordinates_and_the_fields_it_was_given():
    report = condition_report(coordinates={"x": (0.0, 0.1)}, fields={"T": (20.0, 45.0)})
    assert np.allclose(report.apply("x", [0.0, 0.1]), [-1.0, 1.0])
    assert np.allclose(report.apply("T", [20.0, 45.0]), [-1.0, 1.0])
    with pytest.raises(KeyError, match="neither a coordinate nor a field"):
        report.scale("q")
    with pytest.raises(KeyError, match="not a residual term"):
        report.weight("q")


def test_the_report_refuses_input_a_weight_cannot_be_defined_for():
    with pytest.raises(ValueError, match="nothing to condition"):
        condition_report()
    with pytest.raises(ValueError, match="not a conditioning decision"):
        condition_report(residual_terms={"conduction": 0.0})
    with pytest.raises(ValueError, match="target must be positive"):
        condition_report(residual_terms={"conduction": 1.0}, target=0.0)


def test_the_report_serialises_with_the_groups_and_the_missing_ones():
    report = condition_report(
        coordinates={"x": (0.0, 0.1)}, fields={"T": (20.0, 120.0)},
        residual_terms={"conduction": 3000.0},
        parameters={"density": 1000.0, "velocity": 2.0, "length": 1.0, "dynamic_viscosity": 0.1})
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["groups"]["Re"] == pytest.approx(20000.0)
    assert payload["residuals"]["conduction"] == pytest.approx((1.0 / 3000.0) ** 2)
    assert payload["coordinates"]["x"]["scale"] == pytest.approx(0.05)
    assert "Re" not in payload["missing"]
    assert "sound_speed" in payload["missing"]["Ma"]
    assert report.format().startswith("coordinates:")


def test_weights_that_span_decades_are_reported_rather_than_accepted():
    report = condition_report(residual_terms={"tiny": 1e-6, "huge": 1e6})
    assert report.diagnostics[0].code == "LINALG.ILL_CONDITIONED"


# --------------------------------------------------------------------------- the issue's done-when
def test_the_auto_conditioned_run_converges_as_fast_as_the_hand_tuned_one():
    """The claim of #206, measured: same net, seed, optimiser and budget; only the scaling differs."""
    result = scaling_benchmark(steps=600, seed=0)
    raw, hand, auto = (result.first_below["raw"], result.first_below["hand"], result.first_below["auto"])
    # the raw problem never reaches the tolerance in this budget; the two conditioned ones do
    assert not np.isfinite(raw)
    assert np.isfinite(hand) and np.isfinite(auto)
    # and the automatic one is at least as fast as the hand-tuned one
    assert auto <= hand
    # the final error of the two conditioned arms is comparable (both far below the raw one)
    assert result.final["auto"] < result.final["raw"] / 10
    assert result.final["auto"] <= 10 * result.final["hand"]
    assert result.steps == 600 and result.tolerance == pytest.approx(0.05)
    assert result.format().count("first below tolerance") == 3
    assert set(result.to_dict()["errors"]) == {"raw", "hand", "auto"}

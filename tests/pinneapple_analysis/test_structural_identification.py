"""Tests for ``pinneapple_analysis.inverse_problems.structural`` -- inverse identification (#177, S28).

The data are the stored output of a real CalculiX run (``tests/data/calculix_cantilever_displacements.json``,
written by ``scripts/structural_identification.py --write``): a C3D20R brick cantilever, 1 x 0.1 x 0.1 m,
E = 70 GPa, 800 N tip load, read at four gauge stations with a micrometre of gauge noise. The truth is known
because the data were generated at known values, so the acceptance criterion of the issue is checkable:
**with synthetic data, the true parameters fall inside the reported uncertainty interval.**

The last test is the one the study exists for: with both unknowns the displacements only see ``P/E``, and
the report says so instead of quoting two confident intervals.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from pinneapple_analysis.inverse_problems.structural import (
    IDENTIFICATION_FIXTURE,
    ObservationSet,
    cantilever_measurements,
    identify_cantilever,
    load_measurements,
    timoshenko_deflection,
    timoshenko_model,
)

L, B, H, NU = 1.0, 0.1, 0.1, 0.33
TRUE = {"E": 70e9, "P": 800.0}


# --------------------------------------------------------------------------- the beam it is built on
def test_the_tip_deflection_is_the_timoshenko_formula_written_out_by_hand():
    E, P = TRUE["E"], TRUE["P"]
    inertia = B * H**3 / 12.0
    shear_modulus = E / (2.0 * (1.0 + NU))
    bending = P * L**3 / (3.0 * E * inertia)
    transverse = P * L / ((5.0 / 6.0) * shear_modulus * B * H)
    measured = float(timoshenko_deflection([L], E=E, P=P, nu=NU)[0])
    assert measured == pytest.approx(bending + transverse, rel=1e-12)
    # at L/h = 10 shear carries 0.8 % of the tip deflection -- small, and exactly the share a model that
    # omitted it would push into the recovered modulus
    assert transverse / bending == pytest.approx(0.008, abs=0.001)


def test_the_deflection_is_linear_in_load_and_in_compliance():
    """The reason E and P are not separately identifiable: only their ratio reaches the gauge."""
    a = timoshenko_deflection([0.3, 0.7, 1.0], E=70e9, P=800.0)
    b = timoshenko_deflection([0.3, 0.7, 1.0], E=140e9, P=1600.0)     # both doubled
    assert np.allclose(a, b, rtol=1e-12)


def test_the_model_exposes_the_two_parameters_and_predicts_at_the_stations():
    model = timoshenko_model(stations=[0.5, 1.0], E=70e9, P=800.0)
    names = [p.name for p in model.default_parameters()]
    assert names == ["E", "P"]
    prediction = model.predict({"E": 70e9, "P": 800.0}, {"x": np.array([0.5, 1.0])})
    assert prediction == pytest.approx(timoshenko_deflection([0.5, 1.0], E=70e9, P=800.0))


# --------------------------------------------------------------------------- the measured data
def test_the_stored_measurements_came_from_the_solver_and_carry_their_uncertainty():
    measurements = load_measurements()
    assert measurements.stations.tolist() == [0.25, 0.5, 0.75, 1.0]
    assert measurements.truth_parameters["E"] == pytest.approx(TRUE["E"])
    assert measurements.truth_parameters["P"] == pytest.approx(TRUE["P"])
    assert np.all(measurements.sigma > 0)
    # the sigma is the gauge noise plus the measured model-form gap between the 3-D solid and the beam
    assert measurements.model_form > 0
    assert np.all(measurements.sigma >= measurements.noise)
    # every reading is within three sigma of the beam value it is meant to represent
    assert np.all(np.abs(measurements.measured - measurements.truth) <= 3 * measurements.sigma)
    # and the measured gap is the few per cent that makes that sigma necessary
    assert 0.005 < measurements.model_form < 0.1


def test_the_measurement_set_round_trips_through_json():
    payload = json.loads(IDENTIFICATION_FIXTURE.read_text(encoding="utf-8"))
    again = ObservationSet.from_dict(payload)
    assert np.allclose(again.stations, load_measurements().stations)
    assert np.allclose(again.sigma, load_measurements().sigma)
    assert json.loads(json.dumps(again.to_dict()))["model_form"] == pytest.approx(again.model_form)


def test_a_missing_measurement_file_says_how_to_make_one():
    with pytest.raises(FileNotFoundError, match="--write"):
        load_measurements("/nonexistent/cantilever.json")


def test_synthetic_measurements_keep_the_noise_and_the_model_form_apart():
    measurements = cantilever_measurements(E=70e9, P=800.0, stations=[0.5, 1.0], noise=1e-6, model_form=0.02)
    expected = np.sqrt(1e-6**2 + (0.02 * np.abs(measurements.truth)) ** 2)
    assert np.allclose(measurements.sigma, expected)
    assert measurements.noise == pytest.approx(1e-6)
    # a different seed gives different readings for the same sigma
    other = cantilever_measurements(E=70e9, P=800.0, stations=[0.5, 1.0], noise=1e-6, model_form=0.02, seed=7)
    assert not np.allclose(other.measured, measurements.measured)
    assert np.allclose(other.sigma, measurements.sigma)


# --------------------------------------------------------------------------- the acceptance criterion
def test_the_modulus_comes_back_with_the_truth_inside_its_interval():
    """Load on a load cell, material unknown: E is identifiable and the truth must be in the interval."""
    verdict = identify_cantilever(load_measurements(), unknown="modulus").verdict()
    row = verdict["parameters"]["E"]
    assert row["truth"] == pytest.approx(TRUE["E"])
    assert row["inside"] is True, row
    assert abs(row["value"] - TRUE["E"]) / TRUE["E"] < 0.05
    assert row["low"] <= TRUE["E"] <= row["high"]
    assert verdict["dead"] == []


def test_the_load_comes_back_with_the_truth_inside_its_interval():
    """Material certified, actuator force unknown: P is identifiable and the truth must be in the interval."""
    verdict = identify_cantilever(load_measurements(), unknown="load").verdict()
    row = verdict["parameters"]["P"]
    assert row["truth"] == pytest.approx(TRUE["P"])
    assert row["inside"] is True, row
    assert abs(row["value"] - TRUE["P"]) / TRUE["P"] < 0.05
    assert verdict["dead"] == []


def test_the_two_identifiable_cases_agree_on_the_ratio_the_data_actually_fix():
    """Both runs see the same ``P/E``; measuring one and fitting the other is where the consistency shows."""
    modulus = identify_cantilever(load_measurements(), unknown="modulus").verdict()
    load = identify_cantilever(load_measurements(), unknown="load").verdict()
    ratio_from_modulus = TRUE["P"] / modulus["parameters"]["E"]["value"]
    ratio_from_load = load["parameters"]["P"]["value"] / TRUE["E"]
    assert ratio_from_modulus == pytest.approx(ratio_from_load, rel=1e-3)


def test_with_both_unknowns_the_report_says_the_data_cannot_separate_them():
    """The identifiability check of [G5]: an infinite interval and a named dead direction, not two numbers."""
    verdict = identify_cantilever(load_measurements(), unknown="both").verdict()
    assert verdict["dead"], "the degeneracy between E and P was not detected"
    assert any("P" in text and "E" in text for text in verdict["dead"])
    assert verdict["condition_number"] > 1e8
    for name in ("E", "P"):
        row = verdict["parameters"][name]
        assert row["relative_width"] is None            # an open interval, not a confident one
        assert not np.isfinite(row["low"]) and not np.isfinite(row["high"])


def test_the_identification_refuses_an_unknown_it_does_not_know():
    with pytest.raises(ValueError, match="'modulus', 'load' or 'both'"):
        identify_cantilever(load_measurements(), unknown="poisson")


def test_the_verdict_serialises():
    payload = json.loads(json.dumps(identify_cantilever(load_measurements(), unknown="modulus").to_dict()))
    assert payload["unknown"] == "modulus"
    assert payload["interval"] == "profile"
    assert payload["parameters"]["E"]["inside"] is True
    assert payload["condition_number"] == pytest.approx(1.0)

"""The trust layer as one object (#345): a ``Trust`` that a prediction carries and a consumer reads.

What this file holds the object to:

* two genuinely different models produce two different ``Trust`` — different residuals, different gates, a
  different badge reason;
* a field nobody measured comes out ``not_measured`` with the reason, and never as a zero standing in for "unknown";
* a measured value without the source it came from is refused, which is how the "no number without provenance" rule
  is enforced rather than documented;
* the adapters read the number from the module that already computes it — ``weather.evaluate.horizons``,
  ``uncertainty.calibration.CalibrationMetrics`` and ``decision.verifiers`` — instead of computing a second one;
* the JSON round trips and carries the schema version.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from pinneapple_analysis.trust.adapters import (
    calibration_uncertainty,
    domain_from_gate,
    from_ensemble,
    rollout_horizon,
    trust,
    verification_measurement,
    weather_horizon,
)
from pinneapple_analysis.trust.record import (
    FADING,
    GOOD,
    NO_SKILL,
    SCHEMA,
    Horizon,
    Measurement,
    Residual,
    Trust,
    TrustValue,
    Uncertainty,
)
from pinneapple_analysis.trust.trust_gate import TrustGate
from pinneapple_decision.schema import ExecutionResult
from pinneapple_decision.verifiers import ThresholdVerifier
from pinneapple_physics.ensemble import PhysicsEnsemble, from_callable
from pinneapple_physics.weather.evaluate import horizons as weather_horizons


# ------------------------------------------------------------------------------------------ the two models
def _residual(prediction, query):
    """Relative deviation of the prediction from the case's own value: 0 for an exact model, 2 for one 3x off."""
    return float(abs(np.mean(prediction) - query["u"]) / abs(query["u"]))


def _model(scale: float, *, measurement_check=None) -> PhysicsEnsemble:
    """An ensemble of two experts that both predict ``scale * u``: ``scale = 1`` is the model that is right,
    ``3`` the one that is not. Same experts, same residual definition, same declared in-domain levels — only the
    predictions differ, so anything the ``Trust`` says about them comes from the models and not from the setup."""
    experts = [from_callable("fno", lambda q: np.full(8, scale * q["u"]), kind="operator"),
               from_callable("gnn", lambda q: np.full(8, 1.001 * scale * q["u"]), kind="graph")]
    return PhysicsEnsemble(experts, residual_fn=_residual, residual_lookahead=1.0,
                           residual_baseline={"fno": 0.05, "gnn": 0.05},
                           measurement_check=measurement_check)


def _points(*rows) -> np.ndarray:
    return np.asarray(rows, dtype=float)


def _gate() -> TrustGate:
    """A gate fitted on the unit box: points inside it are inside the training region, points far away are not."""
    return TrustGate().fit_from_bounds({"a": (0.0, 1.0), "b": (0.0, 1.0)})


# ------------------------------------------------------------------------------------------ the two Trusts
def test_two_models_produce_two_trusts():
    """The whole point: a model that is right and a model that is wrong cannot yield the same Trust."""
    good = from_ensemble(_model(1.0), {"u": 1.0}, gate=_gate(), domain_points=_points((0.5, 0.5), (0.4, 0.6)))
    bad = from_ensemble(_model(3.0), {"u": 1.0}, gate=_gate(), domain_points=_points((0.5, 0.5), (0.4, 0.6)))

    good_residuals = {r.expert: r.value.value for r in good.residuals}
    bad_residuals = {r.expert: r.value.value for r in bad.residuals}
    assert good_residuals["fno"] == pytest.approx(0.0, abs=1e-12)
    assert bad_residuals["fno"] == pytest.approx(2.0, rel=1e-9)

    good_gates = {g.name: g.status for g in good.gates()}
    bad_gates = {g.name: g.status for g in bad.gates()}
    assert good_gates["residual:fno/pde"] == "pass"
    assert bad_gates["residual:fno/pde"] == "fail"
    assert good.badge_reason() != bad.badge_reason()
    assert good.to_json() != bad.to_json()


def test_residual_is_judged_against_the_in_domain_level_of_the_same_model():
    """A residual is a number with no absolute scale: the only honest comparison is the level the same expert
    reaches inside its own domain, and the object records where that level came from."""
    t = from_ensemble(_model(1.0), {"u": 2.0})
    r = t.residual("pde", "fno")
    assert r is not None and r.value.is_measured
    assert r.value.reference == pytest.approx(0.05)
    assert "residual_baseline" in r.value.reference_source
    assert "PhysicsEnsemble.predict" in r.value.source


def test_a_residual_without_an_in_domain_level_is_not_a_pass():
    """No level to compare with is not "it passed": the gate says it was not measured and why."""
    ensemble = PhysicsEnsemble([from_callable("a", lambda q: np.full(4, q["u"])),
                                from_callable("b", lambda q: np.full(4, 1.001 * q["u"]))],
                               residual_fn=_residual, residual_lookahead=1.0)   # no residual_baseline
    t = from_ensemble(ensemble, {"u": 1.0})
    gate = {g.name: g for g in t.gates()}["residual:a/pde"]
    assert gate.status == "unmeasured"
    assert "in-domain level is not" in gate.evidence
    assert t.residual("pde", "a").value.is_measured          # the number is there, only its scale is not


# ------------------------------------------------------------------------------------------ not measured, never zero
def test_a_field_nobody_measured_says_so_instead_of_zero():
    """The rule of the issue, checked on the field that is easiest to fake: coverage of the interval is not
    measured for a single prediction (the ensemble only reports it for a scored case), so it must say so."""
    t = from_ensemble(_model(1.0), {"u": 1.0})
    coverage = t.uncertainty.coverage
    assert coverage.status == "not_measured"
    assert coverage.value is None
    assert coverage.reason and "coverage" in coverage.reason
    assert t.uncertainty.calibrated().status == "not_measured"

    document = json.loads(t.to_json())
    assert document["uncertainty"]["coverage"]["status"] == "not_measured"
    assert "value" not in document["uncertainty"]["coverage"]
    assert document["uncertainty"]["coverage"]["reason"]


def test_the_fields_that_were_not_measured_never_claim_a_good_badge():
    """An object that measured one residual and nothing else cannot say "good": a missing check is not a pass."""
    t = from_ensemble(_model(1.0), {"u": 1.0})
    assert t.badge() == FADING
    assert "not measured" in t.badge_reason() or "unmeasured" in t.badge_reason()
    assert t.horizon.useful.status == "not_measured"
    assert t.in_domain.status == "not_measured"


def test_an_interval_that_is_not_finite_is_not_reported_as_a_width_of_zero():
    """Before the conformal factor exists the ensemble returns NaN bounds; that is an absence, not a zero width."""
    t = from_ensemble(_model(1.0), {"u": 1.0})
    assert t.uncertainty.width.status == "not_measured"
    assert "not finite" in t.uncertainty.width.reason


# ------------------------------------------------------------------------------------------ the badge
def test_badge_good_needs_every_gate_measured_and_passing():
    """The only route to "good": the residual passes, the input is inside the training region, the forecast lead is
    inside the no-skill lead, the reading passed its check and the interval covered what it claims."""
    t = from_ensemble(
        _model(1.0, measurement_check=lambda q: False), {"u": 1.0},
        gate=_gate(), domain_points=_points((0.5, 0.5), (0.9, 0.1)),
        horizon=weather_horizon({"useful_hours": 120.0, "no_skill_hours": 240.0}, requested_lead=48.0))
    t = t.with_(uncertainty=calibration_uncertainty(np.zeros(50), np.zeros(50), np.full(50, 10.0), level=0.9))
    assert t.badge() == GOOD, t.badge_reason()
    assert "all" in t.badge_reason() and "passing" in t.badge_reason()


def test_badge_no_skill_when_the_input_is_outside_the_training_region():
    """Away from the distribution where the model was validated there is no evidence of skill, so the badge says
    no skill — decided by the measured domain score, not by a threshold somebody picked."""
    t = from_ensemble(_model(1.0), {"u": 1.0}, gate=_gate(), domain_points=_points((25.0, -12.0)))
    assert t.domain_score.value == pytest.approx(0.0)
    assert t.in_domain.value == pytest.approx(0.0)
    assert t.badge() == NO_SKILL
    assert "domain" in t.badge_reason()


def test_badge_no_skill_when_the_lead_is_past_the_no_skill_lead():
    horizon = weather_horizon({"useful_hours": 120.0, "no_skill_hours": 240.0}, requested_lead=360.0)
    t = from_ensemble(_model(1.0), {"u": 1.0}, horizon=horizon)
    assert t.badge() == NO_SKILL
    assert "horizon" in t.badge_reason()


def test_badge_fading_when_the_residual_left_its_in_domain_level():
    t = from_ensemble(_model(3.0), {"u": 1.0})
    assert t.badge() == FADING
    assert "residual:fno/pde fail" in t.badge_reason()


# ------------------------------------------------------------------------------------------ the adapters read, not recompute
def test_weather_horizon_reads_what_weather_evaluate_measured():
    leads = np.array([0.0, 24.0, 48.0, 72.0, 96.0, 120.0, 240.0])
    acc = np.array([1.0, 0.95, 0.9, 0.8, 0.7, 0.65, 0.5])
    rmse = np.array([1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 3.0])
    measured = weather_horizons(leads, acc, rmse, clim_rmse=2.5)

    horizon = weather_horizon(measured, requested_lead=72.0)
    assert horizon.useful.value == pytest.approx(measured["useful_hours"])
    assert horizon.no_skill.value == pytest.approx(measured["no_skill_hours"])
    assert horizon.unit == "h"
    assert "weather.evaluate.horizons" in horizon.useful.source
    assert horizon.covers().value == 1.0


def test_weather_horizon_of_a_lead_never_crossed_is_not_measured():
    """``inf`` means the threshold was not crossed inside the leads scored: the horizon is beyond the window, which
    is not the same number as the last lead, so no number is written."""
    horizon = weather_horizon({"useful_hours": float("inf"), "no_skill_hours": float("inf")})
    assert horizon.useful.status == "not_measured"
    assert "never crossed" in horizon.useful.reason


def test_calibration_uncertainty_reads_the_calibration_metrics():
    rng = np.random.default_rng(7)
    y_pred = np.zeros(400)
    y_true = rng.normal(0.0, 1.0, 400)
    y_std = np.ones(400)
    u = calibration_uncertainty(y_pred, y_true, y_std, level=0.9)
    assert u.coverage.is_measured
    assert 0.0 <= u.coverage.value <= 1.0
    assert u.coverage.reference == pytest.approx(0.9)
    assert "CalibrationMetrics" in u.coverage.source
    assert u.ece.is_measured
    assert u.calibrated().value == float(u.coverage.value >= 0.9)


def test_a_calibrated_interval_passes_the_calibration_gate_and_a_narrow_one_does_not():
    wide = calibration_uncertainty(np.zeros(200), np.zeros(200), np.full(200, 10.0), level=0.9)
    narrow = calibration_uncertainty(np.zeros(200), np.full(200, 50.0), np.full(200, 0.001), level=0.9)
    assert wide.calibrated().value == 1.0
    assert narrow.calibrated().value == 0.0


def test_verification_measurement_reads_the_decision_verifier():
    result = ExecutionResult(status="ok", metrics={"residual": 1e-4, "rel_l2": 0.4})
    verification = ThresholdVerifier({"residual": 1e-3, "rel_l2": 0.05}).verify(result)
    m = verification_measurement(verification)
    assert m.checks["residual"].value == 1.0
    assert m.checks["rel_l2"].value == 0.0
    assert "threshold" in m.checks["residual"].source
    assert m.suspect.status == "not_measured"     # a threshold verifier does not look at sensor readings


def test_a_verifier_check_that_failed_shows_up_as_a_failing_gate():
    result = ExecutionResult(status="ok", metrics={"residual": 1.0})
    m = verification_measurement(ThresholdVerifier({"residual": 1e-3}).verify(result))
    t = from_ensemble(_model(1.0), {"u": 1.0}).with_(measurement=m)
    status = {g.name: g.status for g in t.gates()}
    assert status["check:residual"] == "fail"
    assert t.badge() == FADING


def test_rollout_horizon_reads_the_divergence_time():
    horizon = rollout_horizon({"times": [0.0, 1.0, 2.0], "trust_horizon": 1.5, "horizon_end": 2.0})
    assert horizon.useful.value == pytest.approx(1.5)
    assert "rollout.divergence_time" in horizon.useful.source
    assert horizon.no_skill.status == "not_measured"   # divergence_time reports no climatology lead


def test_rollout_that_never_crossed_has_no_horizon_number():
    horizon = rollout_horizon({"times": [0.0, 1.0], "trust_horizon": None, "horizon_end": 1.0})
    assert horizon.useful.status == "not_measured"
    assert "beyond" in horizon.useful.reason


# ------------------------------------------------------------------------------------------ the rules of the object
def test_a_measured_value_without_a_source_is_refused():
    with pytest.raises(TypeError):                      # measured() will not even build one without a source
        TrustValue.measured(1.0)
    with pytest.raises(ValueError, match="source"):
        TrustValue("measured", 1.0)                     # and the constructor behind it refuses it too
    with pytest.raises(ValueError, match="reason"):
        TrustValue.not_measured("")
    with pytest.raises(ValueError, match="finite"):
        TrustValue.measured(float("nan"), source="somewhere")


def test_a_not_measured_field_cannot_carry_a_value():
    with pytest.raises(ValueError, match="cannot carry a value"):
        TrustValue("not_measured", 0.0, "", "", "", None, "", "", "because")


def test_a_reference_needs_its_own_source():
    with pytest.raises(ValueError, match="reference_source"):
        TrustValue.measured(1.0, source="somewhere", reference=0.5)


def test_empty_trust_says_not_measured_for_every_field():
    t = Trust.empty(kind="pinn", model="a PINN", reason="no adapter reads a PINN yet")
    assert t.in_domain.status == "not_measured"
    assert t.domain_score.value is None
    assert t.horizon.useful.value is None
    assert t.measurement.suspect.value is None
    assert t.uncertainty.coverage.value is None
    assert t.badge() == FADING
    assert t.provenance.kind == "pinn"


def test_an_uncovered_model_kind_is_refused_by_name():
    from pinneapple_core.errors import NotSupported
    with pytest.raises(NotSupported, match="does not read the trust of"):
        trust(object(), {"u": 1.0})


# ------------------------------------------------------------------------------------------ the schema
def test_the_json_carries_its_schema_and_round_trips():
    t = from_ensemble(_model(1.0), {"u": 1.0}, gate=_gate(), domain_points=_points((0.5, 0.5)))
    document = t.to_dict()
    assert document["schema"] == SCHEMA
    assert document["badge"] == t.badge()
    assert document["kind"] == "ensemble"
    again = Trust.from_json(t.to_json())
    assert again.to_dict() == document


def test_a_document_of_another_schema_is_refused():
    document = from_ensemble(_model(1.0), {"u": 1.0}).to_dict()
    document["schema"] = "pinneapple.other/1"
    with pytest.raises(ValueError, match="pinneapple.other"):
        Trust.from_dict(document)


def test_the_provenance_names_what_it_could_not_record():
    t = from_ensemble(_model(1.0), {"u": 1.0})
    provenance = t.provenance
    assert provenance.model.startswith("PhysicsEnsemble(")
    assert len(provenance.model_fingerprint) == 64
    assert len(provenance.data_fingerprint) == 64
    assert "sha256" in provenance.model_fingerprint_method
    assert set(provenance.not_recorded()) <= {"kind", "model", "model_fingerprint", "data_fingerprint",
                                              "library_version", "created"}


def test_pp_trust_is_the_entry_point():
    import pinneapple as pp
    assert callable(pp.trust)
    t = pp.trust(_model(1.0), {"u": 1.0})
    assert isinstance(t, Trust)
    assert t.kind == "ensemble"


def test_a_query_mapping_with_a_reference_is_unpacked():
    t = trust(_model(1.0), {"query": {"u": 1.0}, "reference": np.full(8, 1.0)})
    assert t.uncertainty.coverage.status in ("measured", "not_measured")
    assert t.provenance.data_fingerprint


# ------------------------------------------------------------------------------------------ building blocks
def test_horizon_covers_needs_both_leads():
    no_request = Horizon.not_measured("nothing was asked")
    assert no_request.covers().status == "not_measured"
    assert "requested lead" in no_request.covers().reason


def test_measurement_and_uncertainty_not_measured_helpers():
    assert Measurement.not_measured("no check").suspect.reason == "no check"
    assert Uncertainty.not_measured("no interval").coverage.reason == "no interval"


def test_domain_from_gate_on_points_the_gate_cannot_score():
    gate = _gate()
    in_domain, score = domain_from_gate(gate, {"u": 1.0})
    assert in_domain.status == "not_measured" and score.status == "not_measured"
    assert "not a set of coordinates" in in_domain.reason


def test_residual_name_includes_the_expert_when_there_is_one():
    r = Residual("mass", TrustValue.measured(1e-6, source="x"), expert="fno")
    assert r.name == "fno/mass"
    assert Residual("mass", TrustValue.measured(1e-6, source="x")).name == "mass"

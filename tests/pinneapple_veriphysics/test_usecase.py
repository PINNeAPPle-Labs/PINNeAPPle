"""Use-case specification (pinneapple_veriphysics.usecase): success is declared before anything is verified."""
from __future__ import annotations

import pytest

from pinneapple_veriphysics.usecase import (OBJECTIVES, PROFILE_PRESETS, REFERENCE_KINDS, ValidationProfile,
                                            profile_presets)


def test_a_tolerance_without_a_quantity_is_refused():
    with pytest.raises(ValueError, match="primary_metric"):
        ValidationProfile(name="x", objective="industrial", primary_metric="  ", target_error_pct=5.0)
    with pytest.raises(ValueError, match="target_error_pct"):
        ValidationProfile(name="x", objective="industrial", primary_metric="error", target_error_pct=0.0)
    with pytest.raises(ValueError, match="objective"):
        ValidationProfile(name="x", objective="marketing", primary_metric="error", target_error_pct=5.0)
    with pytest.raises(ValueError, match="reference kind"):
        ValidationProfile(name="x", objective="academic", primary_metric="error", target_error_pct=1.0,
                          reference_evidence=("vibes",))


def test_no_independent_reference_is_stated_as_a_limit_not_a_score():
    exploratory = PROFILE_PRESETS["exploration"]
    assert not exploratory.needs_reference_at_all and exploratory.problems() == []
    academic = PROFILE_PRESETS["academic_cfd"].with_changes(reference_evidence=("none",))
    problems = academic.problems()
    assert problems and "no independent reference" in problems[0]
    text = academic.describe()
    assert "cannot be reported as validated against reality" in text
    assert "warning: no independent reference" in text


def test_high_consequence_demands_a_declared_critical_scope():
    p = PROFILE_PRESETS["industrial_screening"].with_changes(consequence="safety", decision="size the manifold")
    assert any("critical quantity" in x for x in p.problems())
    ok = p.with_changes(critical_quantities=("peak wall temperature",))
    assert ok.problems() == [] and ok.declares_critical_scope


def test_the_illustrative_tolerance_is_labelled_as_illustrative():
    p = PROFILE_PRESETS["industrial_screening"]
    text = p.describe()
    assert "declared by the caller" in text and f"{OBJECTIVES['industrial']['tolerance_pct']:g} % is the illustrative" in \
        text
    assert "never a rule" in text                      # the objective note travels with the number
    assert set(PROFILE_PRESETS) == set(profile_presets())


def test_profile_round_trips_and_changes_make_a_new_object():
    p = PROFILE_PRESETS["academic_cfd"]
    assert ValidationProfile.from_dict(p.to_dict()) == p
    q = p.with_changes(target_error_pct=2.0)
    assert q.target_error_pct == 2.0 and p.target_error_pct == 0.5


def test_reference_kinds_are_a_closed_vocabulary():
    assert REFERENCE_KINDS[-1] == "none"
    p = PROFILE_PRESETS["academic_cfd"]
    assert all(k in REFERENCE_KINDS for k in p.reference_evidence)

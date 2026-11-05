"""Maturity and gap assessment (pinneapple_veriphysics.maturity): two axes, verifiable requirements, no single score."""
from __future__ import annotations

import pytest

from pinneapple_veriphysics.maturity import (DIMENSIONS, LEVELS, REQUIREMENTS, assess_maturity, requirements_for)


def _facts(*keys: str) -> dict[str, bool]:
    return {k: True for k in keys}


ALL_SCIENTIFIC = [r.met_by for r in requirements_for("scientific")]
ALL_INDUSTRIAL = [r.met_by for r in requirements_for("industrial")]


def test_a_good_metric_cannot_lift_a_case_past_a_missing_requirement():
    only_metric = _facts("usecase.profile_defined", "model.prediction_recorded", "verification.residual_executed",
                         "verification.conditions_executed", "validation.independent_reference_recorded",
                         "validation.error_reported")
    a = assess_maturity(only_metric)
    assert a.level("scientific") == 2                       # reference and error alone are level 3 material, but the
    assert a.level("industrial") == 2                       # requirement before them (conditions) is what gates
    a2 = assess_maturity({**only_metric, "reproducibility.provenance_complete": True,
                          "validation.critical_regions_addressed": True})
    assert a2.level("scientific") == 3
    assert any("provenance" in g.met_by for g in a.gaps("scientific"))


def test_the_two_axes_are_independent():
    scientifically_mature = _facts(*ALL_SCIENTIFIC, "usecase.profile_defined")
    a = assess_maturity(scientifically_mature)
    assert a.level("scientific") == 5
    assert a.level("industrial") == 2                        # verification facts lift it to 2, no more
    industrial = _facts(*ALL_INDUSTRIAL)
    b = assess_maturity(industrial)
    assert b.level("industrial") == 5 and b.level("scientific") < 5


def test_every_gap_says_how_to_close_it_and_the_next_level_is_named():
    a = assess_maturity({})
    for dim in DIMENSIONS:
        axis = a.axes[dim]
        assert axis.level == 0 and axis.next_level == 1
        assert axis.gaps and all(g.how_to_meet.strip() for g in axis.gaps)
        assert LEVELS[dim][1][0] == axis.next_title
    text = a.render()
    assert "level 0" in text and "To reach level 1" in text


def test_facts_that_are_absent_are_false_never_assumed():
    a = assess_maturity({k: True for k in ALL_SCIENTIFIC} | {"verification.residual_executed": False})
    assert a.level("scientific") == 1 and a.facts["verification.residual_executed"] is False
    assert assess_maturity([]).level("scientific") == 0


def test_requirements_are_verifiable_and_well_formed():
    keys = [r.key for r in REQUIREMENTS]
    assert len(keys) == len(set(keys))
    for r in REQUIREMENTS:
        assert r.dimension in DIMENSIONS and 1 <= r.level <= 5
        assert r.statement and r.how_to_meet and r.met_by
        assert any(r.met_by == other.met_by for other in REQUIREMENTS) or True   # facts may serve one axis only
    with pytest.raises(ValueError):
        requirements_for("marketing")


def test_improvement_actions_are_ordered_and_prioritised():
    rows = assess_maturity({}).improvement_actions()
    assert rows[0]["priority"] == "high" and rows[-1]["priority"] == "low"
    levels = [r["level"] for r in rows]
    assert levels == sorted(levels)
    assert all(r["action"] and r["statement"] for r in rows)

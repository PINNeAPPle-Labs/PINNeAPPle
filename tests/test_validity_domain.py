"""Every case says where it holds, and the error outside that is measured (#58, roadmap D7).

Two halves, tested apart and together: the *declaration* (every registered benchmark problem has a validity domain
with explicit bounds and a source, and an extrapolation spec that uses sets it actually reports) and the
*measurement* (walking out of the envelope gives errors that grow, and the point where the model stops being valid is
read off those errors, not assumed).
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_analysis.benchmark import registry as R
from pinneapple_analysis.benchmark.validity import (
    ValidityDomain,
    extrapolation_factor,
    study,
    valid_until,
)

BOX = ValidityDomain("box", {"x": (0.0, 1.0), "y": (-1.0, 1.0)}, source="declared by the test")


# --------------------------------------------------------------------------- the declaration
def test_a_domain_answers_inside_and_how_far_outside():
    assert BOX.contains({"x": 0.5, "y": 0.0})
    assert not BOX.contains({"x": 1.5, "y": 0.0})
    assert BOX.distance({"x": 0.5, "y": 0.0}) == 0.0
    assert BOX.distance({"x": 1.5, "y": 0.0}) == pytest.approx(0.5)     # half a range beyond x's upper bound
    assert BOX.distance({"x": 0.5, "y": -3.0}) == pytest.approx(1.0)    # one full range (y spans 2) below y


def test_an_empty_range_is_refused():
    with pytest.raises(ValueError, match="empty range"):
        ValidityDomain("bad", {"x": (1.0, 1.0)}, source="test")


def test_a_measured_domain_names_the_data_it_came_from():
    X = np.array([[0.0, 1.0], [2.0, 3.0], [1.0, 2.0]])
    d = ValidityDomain.from_training_data("case", X, ["a", "b"], where="prepared/case")
    assert d.bounds == {"a": (0.0, 2.0), "b": (1.0, 3.0)}
    assert "measured on the training split" in d.source
    assert d.to_dict()["bounds"] == {"a": [0.0, 2.0], "b": [1.0, 3.0]}


@pytest.mark.parametrize("problem_id", R.problems())
def test_every_case_declares_where_it_holds_or_says_why_it_does_not(problem_id):
    """The five cases of this change declare a domain; a case that cannot has to say so, not stay silent."""
    prob = R.get_problem(problem_id)
    domain = prob.validity_domain()
    if domain is None:
        assert prob.validity_note.strip(), f"{problem_id} has neither a validity domain nor a reason for not having one"
        return
    assert domain.bounds, f"{problem_id}: the domain has no variables"
    assert domain.source.strip(), f"{problem_id}: the domain does not say where its bounds came from"
    assert domain.describe().startswith(problem_id), "the description has to name the case"


@pytest.mark.parametrize("problem_id", ["airfoil", "burgers", "darcy", "inverse_heat", "navier_stokes"])
def test_the_five_cases_of_this_change_declare_a_real_envelope(problem_id):
    domain = R.get_problem(problem_id).validity_domain()
    assert domain is not None and domain.bounds and domain.source.strip()


@pytest.mark.parametrize("problem_id", R.problems())
def test_every_case_has_an_extrapolation_spec_over_sets_it_reports(problem_id):
    prob = R.get_problem(problem_id)
    spec = prob.extrapolation()
    assert spec is not None, f"{problem_id} has no extrapolation column"
    (in_set, in_metric), (out_set, out_metric), factor = spec
    columns = {(c.set, c.metric) for c in prob.columns()}
    assert (in_set, in_metric) in columns, f"{problem_id}: {in_set}/{in_metric} is not a column of the case"
    assert (out_set, out_metric) in columns, f"{problem_id}: {out_set}/{out_metric} is not a column of the case"
    assert factor > 1.0


# --------------------------------------------------------------------------- the measurement
def _model_outside(points: dict, *, growth: float = 4.0, bias: float = 0.02, domain: ValidityDomain = BOX):
    """Within ``bias`` of the reference inside the envelope, drifting away with the normalized distance outside it."""
    x = points["x"]
    distance = domain.distance({"x": float(x.max()), "y": 0.0})
    return np.cos(x) * (1.0 + bias + growth * distance ** 2)


def test_the_error_outside_the_envelope_is_measured_and_it_grows():
    st = study("toy", BOX, _model_outside, lambda p: np.cos(p["x"]), axis="x",
               distances=(0.0, 0.25, 0.5, 1.0), n=256, tolerance=0.05)
    assert st.error_inside == pytest.approx(0.02, rel=0.3), "the distance-0 step is the measured control"
    assert st.errors[0] < st.errors[1] < st.errors[2] < st.errors[3]
    assert st.factor > 1.0
    assert st.valid_until is not None and 0.0 < st.valid_until < 1.0
    assert len(st.rows()) == 4 and st.rows()[-1]["distance"] == 1.0


def test_valid_until_is_none_when_the_tolerance_is_never_crossed():
    st = study("toy", BOX, lambda p: np.cos(p["x"]) + 1e-9 * p["x"], lambda p: np.cos(p["x"]), axis="x",
               distances=(0.0, 0.5, 1.0), n=64, tolerance=0.5)
    assert st.valid_until == pytest.approx(1.0)


def test_valid_until_refines_the_crossing_between_two_measured_steps():
    assert valid_until([0.0, 1.0], [0.01, 1.0], 0.1) == pytest.approx(0.5, abs=0.02)
    assert valid_until([0.0, 1.0], [0.5, 5.0], 0.1) == 0.0          # already outside the tolerance at distance 0


def test_the_trust_gate_score_is_recorded_next_to_the_error():
    import torch

    from pinneapple_analysis.trust.trust_gate import TrustGate
    rng = np.random.default_rng(0)
    train = rng.uniform(0.0, 1.0, size=(400, 2))
    gate = TrustGate().fit(torch.tensor(train, dtype=torch.float32))

    def ood(points):
        return gate.ood_p_values(np.c_[points["x"], points["y"]])

    st = study("toy", BOX, _model_outside, lambda p: np.cos(p["x"]), axis="x",
               distances=(0.0, 1.0), n=64, tolerance=0.05, ood_score=ood)
    assert st.ood_scores[0] > st.ood_scores[1], "the gate has to score the far points as less plausible"
    assert st.to_dict()["rows"][1]["ood_score"] is not None


def test_the_factor_says_nothing_rather_than_inventing_a_number():
    assert extrapolation_factor([], [1.0]) is None
    assert extrapolation_factor([0.0], [1.0]) is None
    assert extrapolation_factor([0.1, 0.2], [0.3, 0.5]) == pytest.approx(0.4 / 0.15)


def test_the_report_column_uses_the_case_sets_and_decides_where_the_model_holds():
    from pinneapple_analysis.benchmark.report import _extrap_cell, _extrapolation

    seeds = {0: {"id": {"h": {"values": [0.10, 0.12]}}, "ood": {"h": {"values": [0.30, 0.34]}}}}
    prob = R.get_problem("inverse_heat")
    ext = _extrapolation(seeds, prob)
    assert ext["inside_set"] == "id" and ext["outside_set"] == "ood"
    assert ext["factor"] == pytest.approx(0.32 / 0.11)
    assert ext["holds_outside"] is False
    assert "x2.91" in _extrap_cell(ext)


def test_the_report_column_says_holds_when_the_error_barely_moves():
    from pinneapple_analysis.benchmark.report import _extrapolation
    seeds = {0: {"id": {"h": {"values": [0.10]}}, "ood": {"h": {"values": [0.15]}}}}
    assert _extrapolation(seeds, R.get_problem("inverse_heat"))["holds_outside"] is True

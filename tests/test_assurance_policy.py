from pinneapple_analysis.verification.assurance_policy import RequiredCheck, evaluate_gate
from pinneapple_analysis.verification.pipeline_assurance import Check, Status


def test_missing_checks_block():
    result = evaluate_gate([], [RequiredCheck("solver", "conservation")])
    assert result["decision"] == "blocked"
    assert result["missing"] == [("solver", "conservation")]


def test_pass_still_requires_review():
    c = Check("conservation", "solver", Status.PASS)
    result = evaluate_gate([c], [RequiredCheck("solver", "conservation")])
    assert result["decision"] == "eligible_for_review"
    assert result["review_required"] is True


def test_empirical_gate_and_duplicates():
    c = Check("conservation", "solver", Status.PASS)
    required = [RequiredCheck("solver", "conservation")]
    assert evaluate_gate([c], required, empirical_validation_required=True)["decision"] == "blocked"
    assert evaluate_gate([c, c], required)["decision"] == "blocked"


def test_inconclusive_blocks():
    c = Check("conservation", "solver", Status.NOT_APPLICABLE)
    assert evaluate_gate([c], [RequiredCheck("solver", "conservation")])["decision"] == "blocked"

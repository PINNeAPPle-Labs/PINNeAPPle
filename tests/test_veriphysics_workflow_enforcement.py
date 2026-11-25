import pytest

from pinneapple_core.workflows import StepFailed
from pinneapple_flow.nodes_ext.veriphysics_grade import enforce_grade


@pytest.mark.parametrize("decision", ["REVIEW", "REJECT", "", "unknown", None])
def test_strict_mode_blocks_nonapproval(decision):
    with pytest.raises(StepFailed):
        enforce_grade(decision)


def test_approved_and_diagnostic_modes():
    enforce_grade("APPROVED")
    enforce_grade("REJECT", "report_only")


def test_unknown_mode_blocks():
    with pytest.raises(StepFailed):
        enforce_grade("APPROVED", "typo")

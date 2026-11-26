"""ASME V&V 20 validation and the V&V 40 credibility block (#106).

The point of the standard is one sentence: an error without an uncertainty is not a validation. So every test here
is about the budget being complete and the verdict following from it — including the cases where a model looks wrong
until the uncertainty is counted, and looks fine until it is not.
"""
from __future__ import annotations

import pytest

from pinneapple_analysis.verification import (
    CredibilityMatrix,
    UncertaintyBudget,
    validation,
    validation_from_gci,
    validation_report,
)


def test_the_comparison_error_and_the_uncertainty_are_both_reported():
    r = validation(simulation=0.98, data=1.0, u_numerical=0.02, u_input=0.01, u_reference=0.01, quantity="Cd")
    assert r.e == pytest.approx(-0.02)
    assert r.relative_error == pytest.approx(-0.02)
    assert r.u_val == pytest.approx((0.02 ** 2 + 0.01 ** 2 + 0.01 ** 2) ** 0.5)
    assert r.ratio == pytest.approx(abs(r.e) / r.u_val)
    assert r.achieved is True
    assert "validated at u_val" in r.verdict


def test_a_model_that_looks_wrong_is_validated_once_the_uncertainty_is_counted():
    """The mistake the standard exists to prevent: 5 % away from the data, 8 % of uncertainty."""
    r = validation(simulation=0.95, data=1.0, u_numerical=0.06, u_input=0.04, u_reference=0.03)
    assert r.relative_error == pytest.approx(-0.05)
    assert r.u_val > abs(r.e)
    assert r.achieved is True


def test_a_model_that_looks_close_is_not_validated_when_the_uncertainty_is_small():
    r = validation(simulation=1.001, data=1.0, u_numerical=0.0005, u_reference=0.0001)
    assert r.achieved is False
    assert "not validated" in r.verdict and "times u_val" in r.verdict


def test_the_budget_says_where_the_uncertainty_comes_from():
    r = validation(1.0, 1.02, u_numerical=0.01, u_input=0.04, u_reference=0.01)
    shares = r.budget.shares()
    assert sum(shares.values()) == pytest.approx(1.0)
    assert r.budget.dominant() == "input"
    assert "input-dominated" in r.to_markdown()


def test_the_numerical_uncertainty_has_no_default():
    """A validation without it is the report the standard replaces, so building one is a TypeError, not a 0."""
    with pytest.raises(TypeError):
        validation(1.0, 1.0)


def test_a_negative_or_non_finite_uncertainty_is_refused():
    with pytest.raises(ValueError, match="non-negative"):
        UncertaintyBudget(-0.1)
    with pytest.raises(ValueError, match="finite"):
        UncertaintyBudget(float("nan"))


def test_the_data_being_zero_does_not_produce_a_ratio():
    r = validation(0.5, 0.0, u_numerical=0.1)
    assert r.relative_error is None and r.achieved is False      # |E| = 0.5 > 0.1
    assert "—" in r.to_markdown()


def test_the_report_is_one_row_per_quantity_plus_the_credibility_block():
    rows = [validation(0.98, 1.0, u_numerical=0.03, quantity="Cd"),
            validation(0.62, 0.60, u_numerical=0.05, quantity="Cl")]
    matrix = CredibilityMatrix(influence="high", consequence="high").add("verification", "high", "tests/gci")
    text = validation_report(rows, credibility=matrix, title="Ahmed body")
    assert "| Quantity | S | D | E | E/D | u_val |" in text
    assert "2 of 2 quantities validated" in text
    assert "Credibility matrix" in text


# --------------------------------------------------------------------------- the mesh study feeds the budget
def test_the_gci_gives_the_numerical_uncertainty():
    """A GCI in percent is a fraction of the fine-grid value: that is the numerical uncertainty of the number."""
    class Conv:
        gci_fine = 2.17

    class Study:
        convergence = Conv()

    r = validation_from_gci(simulation=6.1685, data=6.2, study=Study(), u_reference=0.01, quantity="f")
    assert r.budget.u_numerical == pytest.approx(6.1685 * 0.0217)
    assert "GCI_fine = 2.17" in r.budget.notes["numerical"]
    assert r.achieved is True


def test_a_study_without_a_usable_gci_is_refused():
    class Conv:
        gci_fine = float("nan")

    with pytest.raises(ValueError, match="non-finite GCI"):
        validation_from_gci(1.0, 1.0, Conv())


# --------------------------------------------------------------------------- the credibility matrix
def test_the_required_level_follows_influence_and_consequence():
    assert CredibilityMatrix(influence="high", consequence="high").needed == "high"
    assert CredibilityMatrix(influence="low", consequence="low").needed == "low"
    assert CredibilityMatrix(influence="low", consequence="high").needed == "medium"


def test_gaps_are_named_instead_of_scored():
    m = (CredibilityMatrix(influence="high", consequence="high")
         .add("verification", "high", "docs/gci.md")
         .add("validation", "medium", "wind tunnel 2026-10"))
    result = m.evaluate()
    assert result.sufficient is False
    assert result.achieved == "medium"
    assert any("validation: medium recorded, high needed" in g for g in result.missing)
    assert any("uncertainty: no evidence recorded" in g for g in result.missing)


def test_enough_evidence_is_sufficient():
    m = CredibilityMatrix(influence="medium", consequence="low")
    for category in CredibilityMatrix.CATEGORIES:
        m.add(category, "low", f"{category}.md")
    assert m.evaluate().sufficient is True


def test_a_category_can_require_more_than_the_decision_baseline():
    m = CredibilityMatrix(influence="low", consequence="low").require("verification", "high")
    m.add("verification", "medium", "unit tests only")
    result = m.evaluate()
    assert any("verification: medium recorded, high needed" in g for g in result.missing)


def test_the_matrix_refuses_what_it_cannot_judge():
    with pytest.raises(ValueError, match="influence must be"):
        CredibilityMatrix(influence="enormous")
    with pytest.raises(ValueError, match="unknown category"):
        CredibilityMatrix().add("marketing", "high", "a deck")
    with pytest.raises(ValueError, match="evidence needs something"):
        CredibilityMatrix().add("verification", "high", "")


def test_the_matrix_is_serialisable_for_other_tooling():
    m = CredibilityMatrix(influence="high", consequence="medium").add("verification", "high", "gci.md")
    payload = m.to_dict()
    assert payload["needed"] == "high" and payload["evidence"]["verification"]["ref"] == "gci.md"
    assert payload["missing"] and payload["sufficient"] is False

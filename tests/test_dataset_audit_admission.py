import pytest
import numpy as np

from pinneapple_data.audit import audit_dataset
from pinneapple_data.audit_admission import assess_audit


def card(status="pass"):
    return {"schema": "dataset-audit/v1", "decision": "pass",
            "checks": [{"id": "finite", "status": "pass"},
                       {"id": "physics.solve", "status": status}]}


@pytest.mark.parametrize("status", ["warn", "fail", "not-applicable", None])
def test_required_check_must_pass(status):
    assert not assess_audit(card(status), required_checks=["physics.solve"]).approved


def test_pass_and_missing():
    assert assess_audit(card(), required_checks=["finite", "physics.solve"]).approved
    assert not assess_audit(card(), required_checks=["leakage"]).approved


def test_duplicate_and_nonmandatory_failure():
    c = card()
    c["checks"].append({"id": "finite", "status": "pass"})
    assert not assess_audit(c, required_checks=["finite"]).approved
    assert not assess_audit(card("fail"), required_checks=["finite"]).approved


def test_real_audit_nonfinite_data():
    c = audit_dataset({"train": {"x": np.array([1., np.nan])}}).to_dict()
    assert not assess_audit(c, required_checks=["finite"]).approved


def test_empty_requirements():
    with pytest.raises(ValueError):
        assess_audit(card(), required_checks=[])

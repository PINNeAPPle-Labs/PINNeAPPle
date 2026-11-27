"""The data readiness gate (#1491): a dataset with no verdict, a failed verdict or unreviewed warnings cannot train.

The acceptance of the issue is a set of refusals, so most of this file asserts that something *does not* happen: a
units mismatch, a missing boundary condition and an operating point outside the declared validity each stop a run
through the API that performs runs, and each refusal names the finding ids, the evidence and the fix. Two properties
are asserted on their own because they are the ones that make the gate trustworthy:

* the **verdict is not the score** — a dataset that fails one critical check is ``FAIL`` while its score is 95;
* the **verdict is bound to the data** — approving dataset A does not approve A' with one value changed.
"""
from __future__ import annotations

import pandas as pd
import pytest

import pinneapple as pp
from pinneapple_data.readiness import ReadinessError, ReadinessPolicy, readiness, require

MANIFEST = {
    "schema": "pinneapple.readiness.dataset/1",
    "name": "plate_sweep",
    "version": "1.0",
    "columns": [{"name": "Re", "role": "input", "unit": "1"},
                {"name": "T_max", "role": "output", "unit": "K"}],
    "boundary_conditions": [{"field": "T_max", "kind": "dirichlet", "value": 300.0}],
    "validity": {"Re": [1.0e4, 1.0e5]},
    "operating_point": {"Re": 5.0e4},
}


def table() -> pd.DataFrame:
    return pd.DataFrame({"Re [1]": [1e4, 5e4, 1e5], "T_max [K]": [300.0, 310.0, 320.0]})


def manifest(**over) -> dict:
    m = {k: (list(v) if isinstance(v, list) else v) for k, v in MANIFEST.items()}
    m.update(over)
    return m


# ------------------------------------------------------------------ the verdict itself
def test_a_good_dataset_passes_and_says_which_groups_ran():
    report = readiness(manifest(), table=table())
    assert report.status == "PASS", report.explain()
    assert set(report.groups) == {"schema", "units", "conditions", "validity", "table"}
    assert report.digest.startswith("sha256:")
    assert require(report) is report


def test_nothing_assessed_is_not_a_pass():
    """A report with no group at all is NOT_ASSESSED, and NOT_ASSESSED stops a run like FAIL does."""
    silent = readiness({"schema": "pinneapple.readiness.dataset/1", "name": "nothing"})
    assert silent.status == "NOT_ASSESSED"
    with pytest.raises(ReadinessError, match="nothing was assessed"):
        require(silent)


def test_the_verdict_is_not_derived_from_the_score():
    """Two datasets with the *same score* and different verdicts -- which a score-driven verdict cannot produce."""
    critical_manifest = manifest(columns=[{"name": "Re", "role": "input", "unit": "1"},
                                          {"name": "T_max", "role": "output", "unit": "banana"}])
    warning_manifest_ = manifest(columns=[{"name": "Re", "role": "input", "unit": "1"},
                                          {"name": "T_max", "role": "output"},
                                          {"name": "T_min", "role": "output"},
                                          {"name": "T_avg", "role": "output"}],
                                 boundary_conditions=[{"field": f, "kind": "dirichlet", "value": 300.0}
                                                      for f in ("T_max", "T_min", "T_avg")])
    failed = readiness(critical_manifest)
    warned = readiness(warning_manifest_)
    assert failed.status == "FAIL" and warned.status == "WARNING"
    assert failed.score["overall"] == warned.score["overall"] == 70.0, (failed.score, warned.score)
    assert "units.unparseable" in failed.ids("critical")


# ------------------------------------------------------------------ the three refusals of the acceptance
def test_a_units_mismatch_is_a_finding_with_both_units_in_the_evidence():
    bad = manifest(columns=[{"name": "Re", "role": "input", "unit": "1"},
                            {"name": "T_max", "role": "output", "unit": "Pa"}])
    report = readiness(bad, table=table())
    finding = next(f for f in report.critical() if f.id == "units.mismatch")
    assert finding.evidence == {"column": "T_max", "manifest_unit": "Pa", "table_unit": "K"}
    with pytest.raises(ReadinessError) as excinfo:
        require(report)
    message = str(excinfo.value)
    assert "units.mismatch" in message and "evidence" in message and "fix:" in message


def test_a_missing_boundary_condition_is_critical_and_names_the_field():
    report = readiness(manifest(boundary_conditions=[]), table=table())
    assert report.status == "FAIL"
    finding = next(f for f in report.critical() if f.id == "conditions.missing")
    assert finding.evidence["field"] == "T_max"
    assert "Add the condition" in finding.fix


def test_an_operating_point_outside_the_declared_validity_refuses_the_run():
    report = readiness(manifest(operating_point={"Re": 3.0e5}), table=table())
    assert report.status == "FAIL"
    finding = next(f for f in report.critical() if f.id == "validity.outside")
    assert finding.evidence == {"key": "Re", "operating_point": 3.0e5, "validity": [1.0e4, 1.0e5]}
    with pytest.raises(ReadinessError, match="validity.outside"):
        require(report)


# ------------------------------------------------------------------ warnings and the review record
def warning_manifest() -> dict:
    return manifest(columns=[{"name": "Re", "role": "input", "unit": "1"},
                             {"name": "T_max", "role": "output"}])          # the unit is the warning


def test_a_warning_trains_only_with_a_review_that_accepts_exactly_those_findings(tmp_path):
    report = readiness(warning_manifest(), table=pd.DataFrame({"Re [1]": [1e4, 5e4], "T_max": [300.0, 310.0]}))
    assert report.status == "WARNING" and report.ids() == ["units.missing"]
    with pytest.raises(ReadinessError, match="engineering review"):
        require(report)

    store = pp.ReviewStore(tmp_path / "reviews.jsonl")
    record = pp.ReviewRecord(subject=report.digest, decision="accept_exception", reviewer="A. Ribeiro",
                             role="thermal", scope=[{"finding": "units.missing", "reason": "K in the case sheet"}])
    store.save(record)
    accepted = require(report, review=store)
    assert accepted.status == "WARNING"
    assert accepted.score["review"]["approved"] is True
    assert accepted.score["review"]["record"] == record.id
    assert accepted.score["review"]["accepted"] == ["units.missing"]


def test_modifying_the_dataset_invalidates_the_approval(tmp_path):
    first = readiness(warning_manifest(), table=pd.DataFrame({"Re [1]": [1e4, 5e4], "T_max": [300.0, 310.0]}))
    store = pp.ReviewStore(tmp_path / "reviews.jsonl")
    store.save(pp.ReviewRecord(subject=first.digest, decision="approve", reviewer="A. Ribeiro"))
    assert require(first, review=store).status == "WARNING"

    changed = readiness(warning_manifest(), table=pd.DataFrame({"Re [1]": [1e4, 5e4], "T_max": [300.0, 311.0]}))
    assert changed.digest != first.digest
    with pytest.raises(ReadinessError, match="not accepted for this dataset"):
        require(changed, review=store)


# ------------------------------------------------------------------ the training path
def _tiny_trainer(tmp_path, report=None, **kw):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from pinneapple_train.trainer import Trainer, TrainConfig

    torch.manual_seed(0)
    loader = DataLoader(TensorDataset(torch.randn(32, 3), torch.randn(32, 1)), batch_size=8)

    def loss_fn(model, y_hat, batch):
        return torch.nn.functional.mse_loss(y_hat, batch["y"])

    trainer = Trainer(model=torch.nn.Linear(3, 1), loss_fn=loss_fn, readiness=report, **kw)
    cfg = TrainConfig(epochs=1, device="cpu", log_dir=str(tmp_path), run_name="gate", save_best=False)
    return trainer, loader, cfg


def test_training_refuses_a_failed_dataset_before_the_first_epoch(tmp_path):
    report = readiness(manifest(operating_point={"Re": 3.0e5}), table=table())
    trainer, loader, cfg = _tiny_trainer(tmp_path, report)
    with pytest.raises(ReadinessError):
        trainer.fit(loader, cfg)


def test_training_carries_the_verdict_and_the_reviewer_into_the_run(tmp_path):
    report = readiness(warning_manifest(), table=pd.DataFrame({"Re [1]": [1e4, 5e4], "T_max": [300.0, 310.0]}))
    store = pp.ReviewStore(tmp_path / "reviews.jsonl")
    record = store.save(pp.ReviewRecord(subject=report.digest, decision="accept_exception", reviewer="M. Costa",
                                        scope=[{"finding": "units.missing", "reason": "documented in the sheet"}]))
    trainer, loader, cfg = _tiny_trainer(tmp_path, report, review=store)
    out = trainer.fit(loader, cfg)
    assert out["readiness"]["verdict"] == "WARNING"
    assert out["readiness"]["digest"] == report.digest
    assert out["readiness"]["review"]["record"] == record.id
    assert out["readiness"]["review"]["accepted"] == ["units.missing"]
    assert "M. Costa" in out["readiness"]["review"]["reviewer"]


def test_the_train_model_shortcut_is_not_a_way_around_the_gate():
    """``train_model`` builds a Trainer; if it did not forward the report, it would be the way around the gate."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from pinneapple_neural import train_model

    report = readiness(manifest(operating_point={"Re": 3.0e5}), table=table())
    loader = DataLoader(TensorDataset(torch.randn(8, 1), torch.randn(8, 1)), batch_size=4)
    with pytest.raises(ReadinessError):
        train_model(torch.nn.Linear(1, 1), lambda m, y, b: torch.nn.functional.mse_loss(y, b["y"]), loader,
                    epochs=1, readiness=report)


def test_without_a_report_nothing_changes(tmp_path):
    trainer, loader, cfg = _tiny_trainer(tmp_path)
    out = trainer.fit(loader, cfg)
    assert out["readiness"] is None and len(out["history"]) == 1


def test_a_policy_can_be_loaded_from_data_and_names_what_it_changes():
    policy = ReadinessPolicy.from_dict({"name": "thermal", "critical": ["units.missing"], "disabled": ["table"]})
    report = readiness(warning_manifest(), policy=policy)
    assert report.status == "FAIL", "the policy promoted the missing unit to critical"
    assert report.policy.name == "thermal" and "table" not in report.groups
    with pytest.raises(ValueError, match="do not exist"):
        ReadinessPolicy.from_dict({"critical": ["units.typo"]})

"""The engineering review record (#1493): a person, a decision, the evidence, and a chain that notices edits.

Approval is the one thing in this pipeline no metric can produce, so the tests here are about what the record refuses
(a record with no reviewer, an exception that does not name what it overrides) and what the store makes impossible
(silently editing or removing a decision). An approval that can be edited after the fact is not an approval.
"""
from __future__ import annotations

import json

import pytest

from pinneapple_data.review import (ReviewError, ReviewRecord, ReviewStore, approval_for, digest_of)

DIGEST = "sha256:" + "a" * 64


def record(**kw):
    body = {"subject": DIGEST, "decision": "approve", "reviewer": "A. Ribeiro", "role": "thermal"}
    body.update(kw)
    return ReviewRecord(**body)


def test_a_record_needs_a_person_and_a_decision():
    with pytest.raises(ReviewError, match="reviewer"):
        ReviewRecord(subject=DIGEST, decision="approve", reviewer="  ")
    with pytest.raises(ReviewError, match="decision"):
        ReviewRecord(subject=DIGEST, decision="looks_fine", reviewer="A. Ribeiro")
    with pytest.raises(ReviewError, match="subject"):
        ReviewRecord(subject="", decision="approve", reviewer="A. Ribeiro")


def test_an_exception_has_to_name_what_it_overrides():
    with pytest.raises(ReviewError, match="findings it overrides"):
        record(decision="accept_exception")
    with pytest.raises(ReviewError, match="finding id"):
        record(decision="accept_exception", scope=[{"reason": "because"}])
    with pytest.raises(ReviewError, match="needs a reason"):
        record(decision="accept_exception", scope=[{"finding": "units.missing"}])
    with pytest.raises(ReviewError, match="does not override"):
        record(decision="approve", scope=[{"finding": "units.missing", "reason": "x"}])


def test_the_same_decision_saved_twice_is_one_record(tmp_path):
    store = ReviewStore(tmp_path / "reviews.jsonl")
    first = store.save(record())
    again = store.save(record())
    assert first.id == again.id
    assert len(store) == 1
    assert store.verify() == []


def test_a_new_decision_supersedes_the_old_one_and_both_stay_readable(tmp_path):
    store = ReviewStore(tmp_path / "reviews.jsonl")
    old = store.save(record())
    new = store.save(record(decision="reject", notes="the wall temperature was assumed, not measured"))
    assert new.id != old.id
    assert len(store) == 2
    assert [r["decision"] for r in store.decisions(DIGEST)] == ["approve", "reject"]
    assert store.latest(DIGEST)["id"] == new.id
    assert store.verify() == []


def test_editing_a_saved_record_is_detected(tmp_path):
    store = ReviewStore(tmp_path / "reviews.jsonl")
    store.save(record())
    rows = [json.loads(line) for line in store.path.read_text().splitlines()]
    rows[0]["reviewer"] = "Somebody Else"
    store.path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    problems = store.verify()
    assert any("altered after saving" in p for p in problems)


def test_removing_a_record_breaks_the_chain(tmp_path):
    store = ReviewStore(tmp_path / "reviews.jsonl")
    store.save(record())
    store.save(record(decision="reject"))
    store.save(record(decision="approve", notes="second look"))
    lines = store.path.read_text().splitlines()
    store.path.write_text("\n".join(lines[:1] + lines[2:]) + "\n", encoding="utf-8")
    problems = store.verify()
    assert any("previous record is" in p for p in problems)


def test_an_approval_covers_the_findings_of_its_subject_and_nothing_else(tmp_path):
    store = ReviewStore(tmp_path / "reviews.jsonl")
    store.save(record(decision="accept_exception",
                      scope=[{"finding": "units.missing", "reason": "K is in the case sheet"}]))
    covered = approval_for(store, DIGEST, ["units.missing"])
    assert covered["approved"] and covered["accepted"] == ["units.missing"]
    partial = approval_for(store, DIGEST, ["units.missing", "conditions.missing"])
    assert not partial["approved"] and partial["missing"] == ["conditions.missing"]
    elsewhere = approval_for(store, digest_of({"other": "dataset"}), ["units.missing"])
    assert not elsewhere["approved"], "an approval is for one dataset, not for a finding id in general"


def test_a_reject_never_approves_anything(tmp_path):
    store = ReviewStore(tmp_path / "reviews.jsonl")
    store.save(record(decision="reject"))
    assert approval_for(store, DIGEST, ["units.missing"])["approved"] is False


def test_the_records_carry_a_readable_timestamp_and_a_stable_id():
    r = record()
    assert r.at.startswith("20") and "T" in r.at
    assert r.id.startswith("sha256:") and len(r.id) == 71
    assert record().id == r.id, "the same decision is the same id, whatever the clock said"

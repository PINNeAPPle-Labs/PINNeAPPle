import pytest

from pinneapple_veriphysics.ledger import EvidenceLedger, utc_now

HASH = "a" * 64


def record(ledger, passed=True, digest=HASH):
    return ledger.record("physics", passed=passed, inputs_sha256=digest,
                         outputs={"residual": 0.01},
                         started_at=utc_now(), finished_at=utc_now())


def test_admission_requires_current_evidence():
    ledger = EvidenceLedger()
    assert not ledger.admits(["physics"], inputs_sha256=HASH)
    record(ledger)
    assert ledger.admits(["physics"], inputs_sha256=HASH)
    assert not ledger.admits(["physics", "mesh"], inputs_sha256=HASH)
    record(ledger, passed=False)
    assert not ledger.admits(["physics"], inputs_sha256=HASH)


def test_tampered_latest_does_not_fall_back():
    ledger = EvidenceLedger()
    record(ledger)
    entry = record(ledger)
    entry.outputs["residual"] = 99
    assert ledger.status("physics") == "passed"
    assert not ledger.admits(["physics"], inputs_sha256=HASH)


def test_other_artifact_and_crash_block():
    ledger = EvidenceLedger()
    record(ledger)
    record(ledger, digest="b" * 64)
    assert not ledger.admits(["physics"], inputs_sha256=HASH)
    record(ledger)
    with pytest.raises(RuntimeError):
        with ledger.run("physics", inputs_sha256=HASH):
            raise RuntimeError("solver crashed")
    assert not ledger.admits(["physics"], inputs_sha256=HASH)


def test_empty_policy_blocks():
    with pytest.raises(ValueError):
        EvidenceLedger().admits([], inputs_sha256=HASH)

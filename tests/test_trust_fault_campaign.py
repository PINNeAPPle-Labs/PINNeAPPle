from pinneapple_veriphysics.fault_campaign import FaultCase, run_campaign
from pinneapple_veriphysics.ledger import EvidenceLedger, utc_now


def ledger_gate(tampered=False, wrong_artifact=False):
    ledger = EvidenceLedger()
    digest = "a" * 64
    entry = ledger.record("physics", passed=True, inputs_sha256=digest,
                          outputs={"residual": 0.01},
                          started_at=utc_now(), finished_at=utc_now())
    if tampered:
        entry.outputs["residual"] = 99
    current = ledger.latest("physics")
    return bool(current is not None and current.passed is True
                and current.inputs_sha256 == ("b" * 64 if wrong_artifact else digest))


def test_existing_ledger_controls():
    report = run_campaign([
        FaultCase("evidence", "clean", True, ledger_gate),
        FaultCase("evidence", "tampered", False, lambda: ledger_gate(tampered=True)),
        FaultCase("evidence", "wrong artifact", False, lambda: ledger_gate(wrong_artifact=True)),
    ])
    assert report["passed"]
    assert report["stages"]["evidence"]["detection_rate"] == 1


def test_unsafe_acceptance_and_false_rejection():
    report = run_campaign([
        FaultCase("gate", "clean", True, lambda: False),
        FaultCase("gate", "fault", False, lambda: True),
    ])
    assert not report["passed"]
    assert report["stages"]["gate"]["unsafe_acceptances"] == 1
    assert report["stages"]["gate"]["false_rejections"] == 1


def test_crash_is_not_detection():
    def crash():
        raise RuntimeError("failure")
    report = run_campaign([
        FaultCase("gate", "clean", True, lambda: True),
        FaultCase("gate", "fault", False, crash),
    ])
    assert not report["passed"]
    assert report["stages"]["gate"]["errors"] == 1
    assert report["stages"]["gate"]["detection_rate"] == 0


def test_missing_clean_control():
    assert not run_campaign([FaultCase("gate", "fault", False, lambda: False)])["passed"]

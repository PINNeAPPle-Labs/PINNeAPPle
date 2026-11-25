from datetime import datetime, timedelta, timezone

import pytest

from pinneapple_veriphysics.trust_policy import (
    CheckEvidence, TrustPolicy, Verdict,
)

HASH = "a" * 64
POLICY = TrustPolicy("v1", ("physics", "numerics"))


def evidence(status="pass", digest=HASH):
    return (
        CheckEvidence("physics", "e1", digest, status),
        CheckEvidence("numerics", "e2", HASH, "pass"),
    )


@pytest.mark.parametrize("status,verdict", [
    ("pass", Verdict.APPROVED), ("fail", Verdict.REJECTED),
    ("not_run", Verdict.REVIEW), ("inconclusive", Verdict.REVIEW),
    ("unknown", Verdict.REVIEW),
])
def test_policy(status, verdict):
    expiry = datetime.now(timezone.utc) + timedelta(hours=1)
    result = POLICY.evaluate(HASH, "design", evidence(status), expires_at=expiry)
    assert result.verdict is verdict
    assert result.permits(HASH, "design", "v1") == (status == "pass")
    assert not result.permits(HASH, "production", "v1")
    assert not result.permits(HASH, "design", "v2")
    assert not result.permits(HASH, "design", "v1", now=expiry)


@pytest.mark.parametrize("items", [
    (), evidence(digest="b" * 64), evidence() + evidence(),
])
def test_missing_or_mismatched_evidence_blocks(items):
    result = POLICY.evaluate(
        HASH, "design", items,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    assert not result.permits(HASH, "design", "v1")


def test_empty_policy_rejected():
    with pytest.raises(ValueError):
        TrustPolicy("v1", ())

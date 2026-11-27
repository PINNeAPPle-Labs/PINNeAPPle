from dataclasses import replace

import pytest

from pinneapple_veriphysics.attestation import Attestation, permits, sign

KEY = b"k" * 32
A = Attestation("cert-1", "a" * 64, "design", "v1", "b" * 64, "lab", 100, 200)


def check(a=A, signature=None, **overrides):
    kwargs = dict(key=KEY, artifact_hash=A.artifact_hash, scope=A.scope,
                  policy_version=A.policy_version, evidence_hash=A.evidence_hash,
                  issuer=A.issuer, revoked_ids=frozenset(), now=150)
    kwargs.update(overrides)
    return permits(a, sign(A, KEY) if signature is None else signature, **kwargs)


def test_valid_attestation():
    assert check()


@pytest.mark.parametrize("overrides", [
    {"revoked_ids": frozenset({"cert-1"})}, {"revoked_ids": None},
    {"now": 200}, {"now": 99}, {"scope": "production"},
    {"artifact_hash": "c" * 64}, {"evidence_hash": "c" * 64},
    {"policy_version": "v2"}, {"issuer": "other"}, {"key": b"x" * 32},
])
def test_reject(overrides):
    assert not check(**overrides)


def test_tampered_expiry_and_signature():
    assert not check(replace(A, expires_at=300))
    assert not check(signature="0" * 64)


def test_short_key():
    with pytest.raises(ValueError):
        sign(A, b"short")

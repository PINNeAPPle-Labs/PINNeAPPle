"""Authenticated internal validation attestations (HMAC, not public certificates).

Only a trusted issuer may sign after independently approving evidence.
The verifier must obtain a current revocation snapshot from a trusted store.
"""
from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
import re
import time


@dataclass(frozen=True)
class Attestation:
    certificate_id: str
    artifact_hash: str
    scope: str
    policy_version: str
    evidence_hash: str
    issuer: str
    issued_at: int
    expires_at: int

    def __post_init__(self):
        for digest in (self.artifact_hash, self.evidence_hash):
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("Expected SHA-256 digest")
        if not all(s.strip() for s in (self.certificate_id, self.scope,
                                       self.policy_version, self.issuer)):
            raise ValueError("Attestation identity and scope required")
        if (type(self.issued_at) is not int or type(self.expires_at) is not int
                or self.expires_at <= self.issued_at):
            raise ValueError("Invalid validity period")

    def payload(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True).encode("ascii")


def sign(attestation: Attestation, key: bytes) -> str:
    if not isinstance(key, bytes) or len(key) < 32:
        raise ValueError("Use a securely generated key of at least 32 bytes")
    return hmac.new(key, b"PINNeAPPle-attestation-v1:" + attestation.payload(),
                    hashlib.sha256).hexdigest()


def permits(attestation: Attestation, signature: str, *, key: bytes,
            artifact_hash: str, scope: str, policy_version: str,
            evidence_hash: str, issuer: str,
            revoked_ids: frozenset[str] | None,
            now: int | None = None) -> bool:
    """Fail closed on unavailable revocation state or mismatched claims."""
    now = int(time.time()) if now is None else now
    return (
        revoked_ids is not None
        and attestation.certificate_id not in revoked_ids
        and attestation.issued_at <= now < attestation.expires_at
        and attestation.artifact_hash == artifact_hash
        and attestation.evidence_hash == evidence_hash
        and attestation.scope == scope
        and attestation.policy_version == policy_version
        and attestation.issuer == issuer
        and isinstance(signature, str)
        and re.fullmatch(r"[0-9a-f]{64}", signature) is not None
        and hmac.compare_digest(sign(attestation, key), signature)
    )

"""Fail-closed authorization over measured, artifact-bound evidence.

This policy does not perform scientific checks or authenticate evidence.
Callers must supply evidence from a trusted verifier and enforce the result.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import re


class Verdict(str, Enum):
    APPROVED = "approved"
    REVIEW = "review"
    REJECTED = "rejected"


@dataclass(frozen=True)
class CheckEvidence:
    check_id: str
    evidence_id: str
    artifact_hash: str
    status: str


@dataclass(frozen=True)
class TrustDecision:
    artifact_hash: str
    policy_version: str
    scope_of_use: str
    verdict: Verdict
    evidence_ids: tuple[str, ...]
    reasons: tuple[str, ...]
    expires_at: datetime

    def permits(self, artifact_hash: str, scope: str, policy_version: str,
                *, now: datetime | None = None) -> bool:
        now = datetime.now(timezone.utc) if now is None else now
        return (
            self.verdict is Verdict.APPROVED
            and self.artifact_hash == artifact_hash
            and self.scope_of_use == scope
            and self.policy_version == policy_version
            and now < self.expires_at
        )


@dataclass(frozen=True)
class TrustPolicy:
    version: str
    required_checks: tuple[str, ...]

    def __post_init__(self):
        if (not self.version.strip() or not self.required_checks
                or any(not c.strip() for c in self.required_checks)
                or len(set(self.required_checks)) != len(self.required_checks)):
            raise ValueError("Policy requires a version and unique nonempty checks")

    def evaluate(self, artifact_hash: str, scope: str,
                 evidence: tuple[CheckEvidence, ...], *,
                 expires_at: datetime) -> TrustDecision:
        if not re.fullmatch(r"[0-9a-f]{64}", artifact_hash):
            raise ValueError("Expected lowercase SHA-256")
        if not scope.strip() or expires_at.utcoffset() is None:
            raise ValueError("Scope and timezone-aware expiry are required")
        failures = []
        gaps = []
        ids = []
        for check in self.required_checks:
            matches = [e for e in evidence if e.check_id == check]
            if len(matches) != 1:
                gaps.append(f"{check}: missing or ambiguous evidence")
                continue
            item = matches[0]
            if item.artifact_hash != artifact_hash or not item.evidence_id.strip():
                gaps.append(f"{check}: invalid evidence binding")
                continue
            ids.append(item.evidence_id)
            if item.status == "fail":
                failures.append(f"{check}: failed")
            elif item.status != "pass":
                gaps.append(f"{check}: {item.status}")
        if expires_at <= datetime.now(timezone.utc):
            gaps.append("Decision expired")
        verdict = (Verdict.REJECTED if failures else
                   Verdict.REVIEW if gaps else Verdict.APPROVED)
        return TrustDecision(artifact_hash, self.version, scope, verdict,
                             tuple(ids), tuple(failures + gaps), expires_at)

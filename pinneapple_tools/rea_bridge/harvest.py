"""Harvest ledger: what was studied, under which license, and what came of it.

Each :class:`HarvestEntry` records one studied project (URL, commit, license
policy, the REA Evidence id/SHA-256 when REA was used) and its
:class:`Finding` items -- an idea, a technique or a feature, where it lives in
the original, and what PINNeAPPle did with it.

``Finding.status`` values:

``idea``           noted, nothing implemented yet
``adapted``        PINNeAPPle code derived from the original code (needs ``adapt``)
``reimplemented``  written from the published method / observed behaviour
``exists``         PINNeAPPle already had it; the finding only cross-checks
``rejected``       studied and not taken (``note`` says why)

The ledger enforces the license gate: an ``adapted`` finding under a
``clean_room`` or ``reference_only`` policy is refused at construction time,
so the rule cannot be bypassed by editing a JSON file and reloading it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .licenses import POLICIES

__all__ = ["Finding", "HarvestEntry", "HarvestLedger", "STATUSES", "LicenseGateError"]

STATUSES = ("idea", "adapted", "reimplemented", "exists", "rejected")
#: How the finding was obtained -- REA Evidence, or a direct reading of the
#: source when REA has no analyser for that kind of target (e.g. Python).
METHODS = ("rea", "manual")


class LicenseGateError(ValueError):
    """A finding claims more reuse than the project's license allows."""


@dataclass
class Finding:
    title: str
    summary: str
    source_refs: list[str]
    status: str = "idea"
    method: str = "manual"
    pinneapple_refs: list[str] = field(default_factory=list)
    note: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {self.status!r}")
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if not self.source_refs:
            raise ValueError(f"finding {self.title!r} has no source reference")


@dataclass
class HarvestEntry:
    project: str
    url: str
    commit: str
    licenses: list[str]
    policy: str
    findings: list[Finding] = field(default_factory=list)
    rea_version: str | None = None
    evidence_id: str | None = None
    subject_sha256: str | None = None
    studied_on: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {self.policy!r}")
        self.findings = [f if isinstance(f, Finding) else Finding(**f) for f in self.findings]
        for f in self.findings:
            self._check(f)

    def _check(self, f: Finding) -> None:
        if f.status == "adapted" and self.policy != "adapt":
            raise LicenseGateError(
                f"{self.project}: finding {f.title!r} is 'adapted' but the policy is "
                f"{self.policy!r} ({', '.join(self.licenses) or 'no license'}); only "
                "'reimplemented' (from the published method), 'idea', 'exists' or "
                "'rejected' are allowed")

    def add(self, finding: Finding) -> Finding:
        self._check(finding)
        self.findings.append(finding)
        return finding


class HarvestLedger:
    """A JSON file of :class:`HarvestEntry` items keyed by project name."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.entries: dict[str, HarvestEntry] = {}
        if self.path.is_file():
            raw = json.loads(self.path.read_text())
            for e in raw.get("entries", []):
                entry = HarvestEntry(**e)
                self.entries[entry.project] = entry

    def put(self, entry: HarvestEntry) -> HarvestEntry:
        self.entries[entry.project] = entry
        return entry

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"schema": "pinneapple.rea_harvest/1",
                "entries": [asdict(e) for e in sorted(self.entries.values(), key=lambda e: e.project)]}
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        return self.path

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for e in self.entries.values():
            counts: dict[str, int] = {}
            for f in e.findings:
                counts[f.status] = counts.get(f.status, 0) + 1
            out[e.project] = {"policy": e.policy, "findings": counts}
        return out

"""The engineering review record: a person looked at the evidence and said so (#1493, part of #1497).

Automated checks answer *is this data well formed*; they cannot answer *is this data right for the engineering
question*, because that is a judgement about the regime, the assumptions behind a boundary condition, and where the
answer is used. So approval is a **record**, not a flag:

* it names a **reviewer** and a **decision** -- there is no default and no auto-approve path, and nothing about the
  metrics can produce the reviewer;
* it is tied to the **evidence** it was given (the dataset digest and the digests of the reports), so approving one
  version of a dataset does not approve the next one;
* an exception has to name **which findings** it overrides, each with a reason -- "accept" without saying what it
  accepts is how a critical check disappears from a project;
* the store is **append-only and chained**: every record carries the digest of the previous one, so an edited or
  removed record is detected by :meth:`ReviewStore.verify` instead of being silently trusted.

The vocabulary is the one the repository already uses for operational evidence (``reviewer``, ``reviewed_at``,
``environment`` in ``pinneapple_llm/qa/levels.py``) rather than a second one.

>>> store = ReviewStore("/tmp/reviews.jsonl")                    # doctest: +SKIP
>>> rec = ReviewRecord(subject="sha256:ab...", decision="accept_exception", reviewer="A. Ribeiro",
...                    scope=[{"finding": "boundary_conditions.missing", "reason": "wall measured in test 42"}])
>>> store.save(rec).id[:12]                                      # doctest: +SKIP
'3f9c0a1b2d4e'
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

DECISIONS = ("approve", "reject", "request_more_data", "accept_exception")


class ReviewError(ValueError):
    """The record cannot be built or saved: a missing reviewer, an unscoped exception, a broken chain."""


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def digest_of(obj: Any) -> str:
    """``sha256:`` digest of a canonical JSON dump -- the same convention the rest of the library uses."""
    return "sha256:" + hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()


@dataclass
class ReviewRecord:
    """One decision about one subject, with the evidence that was in front of the reviewer.

    ``subject`` is the digest of what is reviewed (a dataset digest from :mod:`pinneapple_data.readiness`, a model id
    or a study reference). ``scope`` is required for ``accept_exception`` and must name findings; ``annotations``
    marks regions, fields or results as ``acceptable`` / ``suspect`` / ``incorrect``.
    """
    subject: str
    decision: str
    reviewer: str
    role: str = ""
    notes: str = ""
    scope: List[Dict[str, str]] = field(default_factory=list)      # [{"finding": id, "reason": why}]
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    evidence_seen: List[str] = field(default_factory=list)         # digests of the reports shown
    checklist: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    at: str = ""

    def __post_init__(self) -> None:
        if not str(self.reviewer).strip():
            raise ReviewError("a review record needs a reviewer: approval cannot be derived from the metrics")
        if self.decision not in DECISIONS:
            raise ReviewError(f"decision must be one of {DECISIONS}, got {self.decision!r}")
        if not str(self.subject).strip():
            raise ReviewError("a review record needs a subject (the digest of what is reviewed)")
        if self.decision == "accept_exception":
            if not self.scope:
                raise ReviewError("accept_exception has to name the findings it overrides")
            for item in self.scope:
                if not str(item.get("finding") or "").strip():
                    raise ReviewError("every accepted exception needs a finding id")
                if not str(item.get("reason") or "").strip():
                    raise ReviewError(f"exception on {item.get('finding')!r} needs a reason")
        elif self.scope:
            raise ReviewError(f"a '{self.decision}' decision does not override findings; use accept_exception")
        self.at = self.at or datetime.now(timezone.utc).isoformat(timespec="seconds")

    def overridden(self) -> List[str]:
        """The finding ids this record accepts."""
        return [str(item["finding"]) for item in self.scope]

    @property
    def id(self) -> str:
        body = asdict(self)
        body.pop("at", None)                       # the same decision saved twice is the same record
        return digest_of(body)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, **asdict(self)}

    def applies_to(self, subject: str) -> bool:
        """A record applies only to the evidence it was made against."""
        return self.subject == subject

    def accepts(self, finding_ids: Iterable[str]) -> List[str]:
        """Which of ``finding_ids`` this record covers -- empty when it covers none of them."""
        if self.decision == "approve":
            return list(finding_ids)
        return sorted(set(finding_ids) & set(self.overridden()))


class ReviewStore:
    """Append-only, chained store of :class:`ReviewRecord` on disk (one JSON object per line).

    Append-only means an edit is not offered: a new decision supersedes the old one and both stay readable. The chain
    makes the edit visible anyway -- every record holds the digest of the record before it, and
    :meth:`verify` recomputes the whole chain.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    # -- reading ----------------------------------------------------------------------------------------
    def records(self) -> List[Dict[str, Any]]:
        if not self.path.is_file():
            return []
        out: List[Dict[str, Any]] = []
        for i, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:                 # a broken line is evidence of tampering, not a crash
                raise ReviewError(f"{self.path.name}: line {i} is not JSON ({exc})") from exc
        return out

    def latest(self, subject: str) -> Optional[Dict[str, Any]]:
        found = [r for r in self.records() if r.get("subject") == subject]
        return found[-1] if found else None

    def decisions(self, subject: str) -> List[Dict[str, Any]]:
        return [r for r in self.records() if r.get("subject") == subject]

    # -- writing ----------------------------------------------------------------------------------------
    def save(self, record: ReviewRecord) -> ReviewRecord:
        """Append the record. Saving the same decision twice is idempotent; it is never an edit."""
        existing = self.records()
        for row in existing:
            if row.get("id") == record.id:
                return record                          # idempotent: the same decision, already recorded
        # a different decision is a new record that supersedes the old one; both stay readable, and the chain makes
        # rewriting a saved line visible instead of silent
        payload = {**record.to_dict(), "prev": existing[-1]["id"] if existing else ""}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(_canonical(payload) + "\n")
        return record

    # -- integrity --------------------------------------------------------------------------------------
    def verify(self) -> List[str]:
        """Every problem with the chain. An empty list means the records are exactly the ones that were written."""
        problems: List[str] = []
        prev = ""
        for i, row in enumerate(self.records(), start=1):
            body = {k: v for k, v in row.items() if k not in ("id", "prev")}
            body.pop("at", None)
            if row.get("id") != digest_of(body):
                problems.append(f"record {i} does not match its own id: it was altered after saving")
            if row.get("prev", "") != prev:
                problems.append(f"record {i} points at {row.get('prev')!r}, the previous record is {prev!r}")
            prev = row.get("id", "")
        return problems

    def __len__(self) -> int:
        return len(self.records())


def approval_for(store: Optional[ReviewStore], subject: str, finding_ids: Sequence[str], *,
                 required: bool = True) -> Dict[str, Any]:
    """The review that lets ``finding_ids`` through for ``subject``, or a refusal.

    Returns ``{"approved": bool, "record": id | None, "reviewer": ..., "accepted": [...], "missing": [...]}``; the gate
    in :func:`pinneapple_data.readiness.require` turns it into an exception.
    """
    ids = list(finding_ids)
    out: Dict[str, Any] = {"approved": False, "record": None, "reviewer": "", "accepted": [], "missing": ids}
    if store is None:
        return out
    for row in reversed(store.decisions(subject)):
        if row.get("decision") not in ("approve", "accept_exception"):
            continue
        accepted = ids if row["decision"] == "approve" else sorted(set(ids) & set(
            str(item.get("finding")) for item in row.get("scope") or ()))
        if set(ids) <= set(accepted):
            out.update({"approved": True, "record": row.get("id"), "reviewer": row.get("reviewer") or "",
                        "accepted": accepted, "missing": []})
            return out
        out.update({"record": row.get("id"), "reviewer": row.get("reviewer") or "", "accepted": accepted,
                    "missing": sorted(set(ids) - set(accepted))})
        if required:
            return out
    return out

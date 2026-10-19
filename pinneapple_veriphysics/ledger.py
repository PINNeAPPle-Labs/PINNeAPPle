"""Evidence ledger: a test may be reported as passed only when a record shows that it ran.

Every entry holds what ran (``check_id``, ``tool``), on what (``inputs_sha256``: SHA-256 of the exact inputs),
what came out (``outputs``), when (``started_at`` / ``finished_at``, UTC ISO 8601) and its verdict (``passed``).
The entry is sealed with ``entry_sha256``, the SHA-256 of its canonical JSON, so a record edited after the fact
no longer counts.

:meth:`EvidenceLedger.status` is the only way reports in this package turn a check into words: ``"passed"`` or
``"failed"`` when a complete, intact entry exists, ``"not run"`` otherwise. A source that merely says a test
passed (a CI badge, a JSON without the record) is "not run" here::

    from pinneapple_veriphysics.ledger import EvidenceLedger, sha256_bytes

    ledger = EvidenceLedger()
    with ledger.run("residual", inputs_sha256=sha256_bytes(data), tool="my-check") as rec:
        rec.outputs = {"rel_residual": 3e-4}
        rec.passed = 3e-4 < 1e-3
    assert ledger.status("residual") == "passed"
    assert ledger.status("conservation") == "not run"
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import re
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["EvidenceEntry", "EvidenceLedger", "sha256_bytes", "sha256_file", "sha256_json", "canonical_json",
           "utc_now", "record_checks", "PASSED", "FAILED", "NOT_RUN", "LEDGER_SCHEMA"]

PASSED, FAILED, NOT_RUN = "passed", "failed", "not run"
LEDGER_SCHEMA = "pinneapple-evidence-ledger/v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _clean(v: Any) -> Any:
    """JSON-safe copy: numpy scalars to Python, non-finite floats to strings, tuples to lists."""
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if hasattr(v, "tolist") and callable(v.tolist):                 # numpy arrays and scalars
        return _clean(v.tolist())
    if isinstance(v, Path):
        return v.as_posix()
    return v


def canonical_json(obj: Any) -> str:
    """Sorted keys, no whitespace: the same object always gives the same bytes."""
    return json.dumps(_clean(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def utc_now(t: float | None = None) -> str:
    return _dt.datetime.fromtimestamp(time.time() if t is None else t, _dt.timezone.utc).isoformat(
        timespec="milliseconds").replace("+00:00", "Z")


@dataclass
class EvidenceEntry:
    """One recorded execution of a check. ``passed`` is None while it runs or when it crashed."""
    check_id: str
    name: str = ""
    category: str = ""
    tool: str = ""
    tool_version: str = ""
    inputs_sha256: str = ""
    outputs: dict[str, Any] | None = None
    started_at: str = ""
    finished_at: str = ""
    passed: bool | None = None
    error: str = ""
    entry_sha256: str = ""

    def _body(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("entry_sha256")
        return d

    def seal(self) -> EvidenceEntry:
        self.outputs = _clean(self.outputs) if self.outputs is not None else None
        self.entry_sha256 = sha256_json(self._body())
        return self

    def problems(self) -> list[str]:
        """Why this entry cannot back a claim (empty when it can)."""
        out = []
        if not _HEX64.match(self.inputs_sha256 or ""):
            out.append("no SHA-256 of the inputs")
        if self.outputs is None:
            out.append("no recorded outputs")
        if not self.started_at or not self.finished_at:
            out.append("no start/finish time")
        if self.passed is None:
            out.append("no verdict" + (f" (error: {self.error})" if self.error else ""))
        if not self.entry_sha256 or self.entry_sha256 != sha256_json(self._body()):
            out.append("record altered after it was sealed")
        return out

    @property
    def valid(self) -> bool:
        return not self.problems()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvidenceEntry:
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


@dataclass
class EvidenceLedger:
    """An append-only list of :class:`EvidenceEntry`. Several entries may share a ``check_id`` (re-runs): the
    status of a check is that of its latest valid entry."""
    entries: list[EvidenceEntry] = field(default_factory=list)

    def add(self, entry: EvidenceEntry) -> EvidenceEntry:
        self.entries.append(entry)
        return entry

    def record(self, check_id: str, *, passed: bool, inputs_sha256: str, outputs: dict[str, Any],
               started_at: str, finished_at: str, name: str = "", category: str = "", tool: str = "",
               tool_version: str = "") -> EvidenceEntry:
        """Record a check that already ran (times are those of the run, not of this call)."""
        return self.add(EvidenceEntry(check_id=check_id, name=name or check_id, category=category, tool=tool,
                                      tool_version=tool_version, inputs_sha256=inputs_sha256, outputs=outputs,
                                      started_at=started_at, finished_at=finished_at, passed=bool(passed)).seal())

    @contextmanager
    def run(self, check_id: str, *, inputs_sha256: str, name: str = "", category: str = "", tool: str = "",
            tool_version: str = "") -> Iterator[EvidenceEntry]:
        """Time a check while it runs. Set ``rec.outputs`` and ``rec.passed`` inside the block; an exception is
        recorded (no verdict, so the check reads "not run") and re-raised."""
        e = EvidenceEntry(check_id=check_id, name=name or check_id, category=category, tool=tool,
                          tool_version=tool_version, inputs_sha256=inputs_sha256, started_at=utc_now())
        try:
            yield e
        except BaseException as exc:
            e.passed, e.error = None, f"{type(exc).__name__}: {exc}"[:500]
            raise
        finally:
            e.finished_at = utc_now()
            if e.passed is not None:
                e.passed = bool(e.passed)
            self.add(e.seal())

    def latest(self, check_id: str) -> EvidenceEntry | None:
        """Latest valid entry of ``check_id`` (None if none)."""
        for e in reversed(self.entries):
            if e.check_id == check_id and e.valid:
                return e
        return None

    def status(self, check_id: str) -> str:
        e = self.latest(check_id)
        if e is None:
            return NOT_RUN
        return PASSED if e.passed else FAILED

    def claim(self, check_id: str) -> str:
        """One line a report may print about ``check_id``; "not run" unless the ledger proves otherwise."""
        e = self.latest(check_id)
        if e is None:
            bad = [x for x in self.entries if x.check_id == check_id]
            why = f" (record invalid: {', '.join(bad[-1].problems())})" if bad else ""
            return f"{check_id}: not run{why}"
        return (f"{check_id}: {PASSED if e.passed else FAILED} -- ran {e.started_at} to {e.finished_at}, "
                f"inputs sha256 {e.inputs_sha256[:12]}")

    def check_ids(self) -> list[str]:
        return list(dict.fromkeys(e.check_id for e in self.entries))

    def invalid(self) -> list[EvidenceEntry]:
        return [e for e in self.entries if not e.valid]

    def extend(self, other: EvidenceLedger | Iterable[EvidenceEntry]) -> None:
        for e in (other.entries if isinstance(other, EvidenceLedger) else other):
            self.add(e)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": LEDGER_SCHEMA, "entries": [e.to_dict() for e in self.entries]}

    @classmethod
    def from_dict(cls, d: dict[str, Any] | list[dict[str, Any]]) -> EvidenceLedger:
        """Entries are taken as stored (never re-sealed), so an edited entry stays invalid."""
        items = d.get("entries", []) if isinstance(d, dict) else d
        return cls([EvidenceEntry.from_dict(x) for x in items if isinstance(x, dict) and x.get("check_id")])


def _check_category(name: str, kind: str) -> str:
    n, k = name.lower(), kind.lower()
    if "residual" in n or k in ("sanity", "numerics", "verification"):
        return "verification"
    if k in ("reference", "generalization", "validation", "benchmark"):
        return "validation"
    return "physics" if k else ""


def record_checks(ledger: EvidenceLedger, checks: Iterable[Any], *, prefix: str, inputs_sha256: str,
                  started_at: str, finished_at: str, tool: str = "", tool_version: str = "") -> list[EvidenceEntry]:
    """Record the checks of a trust card (``pp.trust_card(...).checks`` or its ``to_dict()["checks"]``): one entry
    per check that ran, under ``f"{prefix}/{name}"``. A check with ``passed is None`` (N/A) did not run and gets no
    entry, so it reads "not run"."""
    out = []
    for c in checks:
        d = c if isinstance(c, dict) else vars(c)
        if d.get("passed") is None:
            continue
        out.append(ledger.record(f"{prefix}/{d['name']}", passed=bool(d["passed"]), inputs_sha256=inputs_sha256,
                                 outputs={k: d.get(k) for k in ("value", "threshold", "critical", "detail")},
                                 started_at=started_at, finished_at=finished_at, name=str(d["name"]),
                                 category=_check_category(str(d["name"]), str(d.get("kind", ""))), tool=tool,
                                 tool_version=tool_version))
    return out

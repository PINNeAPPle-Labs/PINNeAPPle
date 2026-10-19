"""What a run leaves behind: one :class:`StepRecord` per step (state, commands, logs, artifacts with their SHA-256,
screenshots, checks, outputs, error) and the :class:`RunRecord` around them, saved as ``run.json`` in the run folder.

A step is ``passed`` only if its action ran, every check it produced held and every artifact it was expected to
leave exists; nothing else may be reported as passed.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Check", "Evidence", "FileRecord", "StepRecord", "RunRecord", "STATES", "file_record"]

STATES = ("pending", "running", "passed", "failed", "skipped", "awaiting_confirmation", "declined", "blocked")


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""
    value: float | None = None
    threshold: Any = None
    source: str = ""                         # which library function computed it


@dataclass
class Evidence:
    """An image of what happened. ``kind`` says what it is, so a report never passes a plot off as a screenshot:
    ``screen`` (capture of the whole virtual display), ``window`` (one application window), ``render`` (the
    application's own image export), ``plot`` (a chart drawn from the step's log or data)."""
    path: str                                # relative to the run folder
    kind: str
    caption: str = ""
    sha256: str = ""


@dataclass
class FileRecord:
    path: str                                # relative to the run folder
    bytes: int
    sha256: str
    kind: str = "file"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path, root: Path) -> FileRecord:
    """Size and SHA-256 of a file; for a folder, the total size and a hash over its sorted file hashes."""
    rel = path.resolve().relative_to(root.resolve()).as_posix() if path.resolve().is_relative_to(root.resolve()) \
        else str(path)
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file())
        h = hashlib.sha256()
        for p in files:
            h.update(p.relative_to(path).as_posix().encode() + b"\0" + _sha(p).encode())
        return FileRecord(rel, sum(p.stat().st_size for p in files), h.hexdigest(), "dir")
    return FileRecord(rel, path.stat().st_size, _sha(path))


def _clean(v: Any) -> Any:
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if hasattr(v, "item") and callable(v.item):
        try:
            return _clean(v.item())
        except (TypeError, ValueError):
            pass
    if isinstance(v, Path):
        return str(v)
    return v


@dataclass
class StepRecord:
    id: str
    title: str
    action: str
    mode: str = "api"                        # api | gui
    state: str = "pending"
    started: float | None = None
    ended: float | None = None
    commands: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    artifacts: list[FileRecord] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    actions: list[str] = field(default_factory=list)       # what the agent did, in words (incl. compatibility fixes)
    confirmation: dict[str, Any] | None = None             # {"categories", "reasons", "approved_by", "at"}
    error: str | None = None

    @property
    def duration_s(self) -> float | None:
        return None if self.started is None or self.ended is None else round(self.ended - self.started, 2)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["duration_s"] = self.duration_s
        return _clean(d)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StepRecord:
        d = dict(d)
        d.pop("duration_s", None)
        d["artifacts"] = [FileRecord(**a) for a in d.get("artifacts", [])]
        d["evidence"] = [Evidence(**e) for e in d.get("evidence", [])]
        d["checks"] = [Check(**c) for c in d.get("checks", [])]
        return cls(**d)


@dataclass
class RunRecord:
    run_id: str
    workflow: dict[str, Any]
    inputs: dict[str, Any]
    folder: str
    state: str = "pending"   # pending running passed rejected (steps ran, a criterion not met)
    # failed awaiting_confirmation declined blocked paused
    created: float = field(default_factory=time.time)
    finished: float | None = None
    steps: list[StepRecord] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)  # tool versions, host
    criteria: list[Check] = field(default_factory=list)
    message: str = ""

    def step(self, step_id: str) -> StepRecord:
        for s in self.steps:
            if s.id == step_id:
                return s
        raise KeyError(step_id)

    def to_dict(self) -> dict[str, Any]:
        return _clean({"run_id": self.run_id, "workflow": self.workflow, "inputs": self.inputs, "folder": self.folder,
                       "state": self.state, "created": self.created, "finished": self.finished,
                       "steps": [s.to_dict() for s in self.steps], "environment": self.environment,
                       "criteria": [asdict(c) for c in self.criteria], "message": self.message})

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunRecord:
        d = dict(d)
        d["steps"] = [StepRecord.from_dict(s) for s in d.get("steps", [])]
        d["criteria"] = [Check(**c) for c in d.get("criteria", [])]
        return cls(**d)

    def save(self) -> Path:
        p = Path(self.folder) / "run.json"
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        tmp.replace(p)
        return p

    @classmethod
    def load(cls, folder: str | Path) -> RunRecord:
        return cls.from_dict(json.loads((Path(folder) / "run.json").read_text(encoding="utf-8")))

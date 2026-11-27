"""A run result becomes evidence: a record with a hash, its environment, its parameters and the command that
re-runs it (#38, roadmap B12).

A number in a report is worth what you can check about it. This module makes the check possible without trusting the
person who produced the number: every run writes a :class:`RunRecord` next to its output, and the record carries

- **what was run**: ``kind`` and ``id`` (an example, a benchmark job, a case), plus the parameters it ran with;
- **on what**: the environment (interpreter, platform, accelerator, relevant package versions) and the commit;
- **the output, by hash**: each file the run produced, with its sha256 and size, so a changed file is detectable
  without re-running anything;
- **the command that re-executes it**: a list of arguments, not a sentence, so a tool can run it;
- **the numbers it publishes**, under ``metrics``, so a number in a report can be traced back to the record that
  produced it and from the record to the file and the command;
- **a digest of all of the above** (``hash``), computed over the canonical JSON of the record without the hash
  field: editing any field, any metric included, invalidates it.

:func:`verify` returns the problems rather than raising one, because verification tooling wants the whole list:
a missing output, a changed file, a digest that no longer matches, a record with no command to re-run.

The Benchmark writes one record per job (``records/<job id>.json``) and the examples runner writes one per script
(``run_results/records/<category>__<name>.json``), which is what makes "every published number points to a verifiable
record" a property of the tree rather than an intention.

    python -m pinneapple_core.run_record verify run_results      # check every record under a folder
    python -m pinneapple_core.run_record show <record.json>      # what it ran, with what, and its digest
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["RunRecord", "digest_of", "file_hash", "find", "index", "load", "record", "verify", "verify_tree"]

RECORD_SUFFIX = ".record.json"


# --------------------------------------------------------------------------- hashing
def file_hash(path: str | Path) -> str:
    """sha256 of a file, read in chunks (a checkpoint is not a small file)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_of(payload: Any) -> str:
    """The digest of any JSON-able object, independent of key order and whitespace."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _git_commit(cwd: Path | None = None) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return ""


def environment() -> dict[str, Any]:
    """What the run depended on outside its own code, recorded so two runs can be compared honestly."""
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cwd": os.getcwd(),
    }
    for mod in ("torch", "numpy", "scipy"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    try:
        import torch
        env["cuda"] = bool(torch.cuda.is_available())
        env["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        env["cuda"] = False
    return env


# --------------------------------------------------------------------------- the record
@dataclass
class RunRecord:
    """One run, as evidence: what ran, on what, what it produced (by hash), the numbers, and how to redo it.

    ``outputs`` is a list of ``{"path": ..., "sha256": ..., "bytes": ...}``, each path relative to ``root`` -- itself
    relative to the folder the record is written in -- so a record stays valid when the whole run folder is moved.
    ``metrics`` maps a published number's name to its value: something a report can print and a checker can compare.
    """
    kind: str
    id: str
    root: str = "."
    command: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    commit: str = ""
    created: str = ""
    note: str = ""
    hash: str = ""

    def payload(self) -> dict[str, Any]:
        """Everything the digest covers: the record without its own hash."""
        d = self.to_dict()
        d.pop("hash", None)
        return d

    def compute_hash(self) -> str:
        return digest_of(self.payload())

    def sealed(self) -> RunRecord:
        """The record with its digest filled in (what gets written)."""
        self.hash = self.compute_hash()
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "id": self.id, "root": self.root, "hash": self.hash, "command": list(self.command),
            "parameters": self.parameters, "outputs": list(self.outputs), "metrics": self.metrics,
            "environment": self.environment, "commit": self.commit, "created": self.created, "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunRecord:
        return cls(kind=d.get("kind", ""), id=d.get("id", ""), root=d.get("root", "."),
                   command=list(d.get("command") or []),
                   parameters=dict(d.get("parameters") or {}), outputs=[dict(o) for o in d.get("outputs") or []],
                   metrics=dict(d.get("metrics") or {}), environment=dict(d.get("environment") or {}),
                   commit=d.get("commit", ""), created=d.get("created", ""), note=d.get("note", ""),
                   hash=d.get("hash", ""))

    # -- writing and reading -------------------------------------------------
    def write(self, path: str | Path) -> Path:
        """Write the record, first deriving ``root`` from where it lands so the output paths survive a move."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        base = getattr(self, "_base", None)
        if base:
            self.root = os.path.relpath(base, path.parent.resolve())
            self.sealed()
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    @property
    def digest_ok(self) -> bool:
        return bool(self.hash) and self.hash == self.compute_hash()


def record(kind: str, id: str, *, command: Iterable[str], parameters: dict | None = None,
           outputs: Iterable[str | Path] = (), root: str | Path = ".", metrics: dict | None = None,
           environment_vars: dict | None = None, commit: str | None = None,
           note: str = "") -> RunRecord:
    """Build a sealed :class:`RunRecord`: the outputs are hashed from disk, relative to ``root``."""
    root = Path(root)
    files = []
    for out in outputs:
        p = Path(out)
        if not p.exists():
            continue
        rel = str(p.relative_to(root)) if p.is_relative_to(root) else str(p)
        files.append({"path": rel, "sha256": file_hash(p), "bytes": p.stat().st_size})
    rec = RunRecord(
        kind=kind, id=id, command=[str(c) for c in command], parameters=dict(parameters or {}), outputs=files,
        metrics=dict(metrics or {}), environment=environment_vars if environment_vars is not None else environment(),
        commit=commit if commit is not None else _git_commit(root),
        created=__import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"), note=note,
    )
    rec._base = str(root.resolve())          # not part of the digest: it only says where the paths were taken from
    return rec.sealed()


# --------------------------------------------------------------------------- verification
def verify(rec: RunRecord | dict, base: str | Path = ".") -> list[str]:
    """Everything wrong with this record as evidence: an empty list means it verifies.

    ``base`` is the folder the record lives in (``.`` for one you just built in memory); the outputs resolve as
    ``base / record.root / path``. Checks the digest, the presence of a command that re-executes the run, the
    environment, and every output file's hash. It does not re-run anything: re-execution is ``record.command``,
    kept as data for whoever wants to do it.
    """
    if isinstance(rec, dict):
        rec = RunRecord.from_dict(rec)
    problems: list[str] = []
    if not rec.hash:
        problems.append("no digest: the record was never sealed, so nothing in it can be checked")
    elif not rec.digest_ok:
        problems.append(f"digest mismatch: the record says {rec.hash[:12]}, its content hashes to "
                        f"{rec.compute_hash()[:12]} (a field was edited after it was written)")
    if not rec.command:
        problems.append("no command: nothing says how to re-execute this run")
    if not rec.environment:
        problems.append("no environment: the numbers cannot be tied to a machine or a library version")
    root = Path(base) / rec.root
    for out in rec.outputs:
        p = Path(out.get("path", ""))
        p = p if p.is_absolute() else root / p
        if not p.exists():
            problems.append(f"missing output: {out.get('path')}")
        elif out.get("sha256") and file_hash(p) != out["sha256"]:
            problems.append(f"changed output: {out.get('path')} no longer hashes to {out['sha256'][:12]}")
    return problems


def load(path: str | Path) -> RunRecord:
    return RunRecord.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _record_files(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if folder.is_file():
        return [folder]
    return sorted(p for p in folder.rglob(f"*{RECORD_SUFFIX}") if p.is_file())


def verify_tree(folder: str | Path) -> dict[str, list[str]]:
    """``{record path: problems}`` for every record under ``folder``; only the ones with problems are returned."""
    out: dict[str, list[str]] = {}
    for p in _record_files(folder):
        try:
            problems = verify(load(p), p.parent)
        except Exception as exc:                              # a broken file is a finding, not a crash
            problems = [f"unreadable record: {type(exc).__name__}: {exc}"]
        if problems:
            out[str(p)] = problems
    return out


# --------------------------------------------------------------------------- querying
def index(folder: str | Path) -> list[dict[str, Any]]:
    """One row per record under ``folder``: the shape the Benchmark and verification tooling query."""
    rows = []
    for p in _record_files(folder):
        try:
            rec = load(p)
        except Exception:
            continue
        rows.append({"path": str(p), "kind": rec.kind, "id": rec.id, "hash": rec.hash, "commit": rec.commit,
                     "created": rec.created, "metrics": rec.metrics, "digest_ok": rec.digest_ok,
                     "n_outputs": len(rec.outputs)})
    return rows


def find(folder: str | Path, *, id: str | None = None, kind: str | None = None,
         metric: str | None = None) -> list[RunRecord]:
    """Records under ``folder`` matching an id, a kind, or a metric name (the three questions a report asks)."""
    out = []
    for p in _record_files(folder):
        try:
            rec = load(p)
        except Exception:
            continue
        if id is not None and rec.id != id:
            continue
        if kind is not None and rec.kind != kind:
            continue
        if metric is not None and metric not in rec.metrics:
            continue
        out.append(rec)
    return out


# --------------------------------------------------------------------------- CLI
def _main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("verify", "show", "index"):
        print(__doc__.split("    python -m")[-1].strip())
        return 2
    if argv[0] == "verify":
        target = argv[1] if len(argv) > 1 else "."
        bad = verify_tree(target)
        for path, problems in bad.items():
            for problem in problems:
                print(f"{path}: {problem}")
        print(f"{len(_record_files(target))} record(s), {len(bad)} with problems")
        return 1 if bad else 0
    if argv[0] == "index":
        for row in index(argv[1] if len(argv) > 1 else "."):
            print(f"{row['id']:<50} {row['kind']:<16} {row['hash'][:12]} {'ok' if row['digest_ok'] else 'EDITED'}")
        return 0
    rec = load(argv[1])
    print(f"{rec.kind}:{rec.id}")
    print(f"  digest   {rec.hash or '(unsealed)'} {'ok' if rec.digest_ok else 'MISMATCH'}")
    print(f"  commit   {rec.commit}")
    print(f"  command  {' '.join(rec.command)}")
    print(f"  metrics  {json.dumps(rec.metrics, sort_keys=True)}")
    for out in rec.outputs:
        print(f"  output   {out['path']} {out['sha256'][:12]} {out['bytes']} B")
    problems = verify(rec, Path(argv[1]).parent)
    print("  verify   " + ("ok" if not problems else "; ".join(problems)))
    return 0 if not problems else 1


if __name__ == "__main__":                                   # python -m pinneapple_core.run_record ...
    sys.exit(_main(sys.argv[1:]))

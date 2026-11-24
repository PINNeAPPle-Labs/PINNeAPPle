"""The supported versions matrix (roadmap I6, #93): which Python, PyTorch and operating system combinations
PINNeAPPle promises, and which of them somebody actually ran.

Two things, kept apart on purpose:

* **declared** — what a user is told they can rely on. The same range is in the ``classifiers`` of
  ``pyproject.toml``, in ``docs/getting_started/installation.md`` and in ``docs/support_matrix.md``.
* **executed** — the cells a person ran the smoke on, on a real machine, with the record of the run: date,
  interpreter, torch version, operating system and architecture.

A cell is never green for having been declared. ``scripts/check_matrix.py`` runs the smoke in the environment it
is started in, says whether that environment is in the matrix and, with ``--record``, writes the run into
``support_matrix.json``. The page ``docs/support_matrix.md`` is rendered from that file, and a test fails when the
two drift, so the page cannot claim a run that did not happen.

Nothing here imports torch: the version is read from the installed metadata, so the check costs milliseconds and
works before the first import of the heavy stack.
"""
from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "SCHEMA", "MATRIX_PATH", "matrix", "cells", "runs", "Cell", "Verdict", "classify", "probe", "record",
    "render_markdown", "os_name", "version_line",
]

SCHEMA = "pinneapple.support_matrix/1"
MATRIX_PATH = Path(__file__).with_name("support_matrix.json")
_CACHE: dict[str, Any] | None = None


# ── the declaration ────────────────────────────────────────────────────────────────────────────────────────
def matrix(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """The declared matrix as a dict (``support_matrix.json``), loaded once.

    ``path`` reads another file and does not touch the cache: the tests use it to check the committed one against a
    scratch copy.
    """
    global _CACHE
    if path is not None:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    if _CACHE is None:
        _CACHE = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    return _CACHE


def _reload(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Drop the cache (after a record, so the next call sees the new run)."""
    global _CACHE
    _CACHE = None
    return matrix(path)


def _write(data: dict[str, Any], path: str | os.PathLike[str] | None = None) -> Path:
    target = Path(path or MATRIX_PATH)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def version_line(version: Any) -> str:
    """``"3.11.16"`` -> ``"3.11"``; ``"2.15.0+cpu"`` -> ``"2.15"``; ``"2.13.0rc1"`` -> ``"2.13"``."""
    text = str(version or "").strip()
    parts = []
    for chunk in text.replace("+", ".").replace("-", ".").split("."):
        digits = ""
        for char in chunk:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
    return ".".join(str(p) for p in parts[:2])


def _tuple(version: Any) -> tuple[int, ...]:
    return tuple(int(p) for p in version_line(version).split(".")) if version_line(version) else ()


def os_name(system: str | None = None) -> str:
    """The matrix spelling of the operating system: ``linux``, ``macos``, ``windows`` (or the raw name)."""
    name = (system or platform.system()).strip().lower()
    if name.startswith("darwin") or name == "macos":
        return "macos"
    if name.startswith("linux"):
        return "linux"
    if name.startswith("win"):
        return "windows"
    return name


# ── cells and runs ────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Cell:
    """One combination of the minimal matrix: a Python version, a torch line and an operating system."""

    python: str
    torch: str          # "current" or "previous", as the keys of matrix()["torch"]["lines"]
    os: str
    status: str         # "executed" (a run matches it) or "declared" (nobody ran it yet)
    run: dict[str, Any] | None = None

    @property
    def torch_version(self) -> str:
        return str(matrix()["torch"]["lines"].get(self.torch, self.torch))

    def label(self) -> str:
        return f"{self.os} · python {self.python} · torch {self.torch} ({self.torch_version})"

    def as_dict(self) -> dict[str, Any]:
        return {"python": self.python, "torch": self.torch, "os": self.os, "status": self.status,
                "run": self.run}


def runs() -> list[dict[str, Any]]:
    """The recorded runs, newest last (each is an environment somebody ran the smoke on)."""
    return list(matrix().get("runs") or [])


def executed(python: Any, torch: Any, system: Any) -> dict[str, Any] | None:
    """The recorded run of exactly this environment, if there is one."""
    line, want_os = version_line(python), os_name(str(system))
    for run in runs():
        if version_line(run.get("python")) == line and version_line(run.get("torch")) == version_line(torch) \
                and run.get("os") == want_os:
            return run
    return None


def cells() -> list[Cell]:
    """The cells of the minimal matrix, each labelled executed or declared from the recorded runs."""
    out = []
    for cell in matrix()["cells"]:
        run = executed(cell["python"], matrix()["torch"]["lines"].get(cell["torch"]), cell["os"])
        out.append(Cell(cell["python"], cell["torch"], cell["os"], "executed" if run else "declared", run))
    return out


# ── the current environment ──────────────────────────────────────────────────────────────────────────────
def probe(torch_version: str | None = None) -> dict[str, Any]:
    """The current interpreter, torch and machine, in the matrix vocabulary.

    The torch version comes from the installed metadata (``importlib.metadata``), not from an import: the check
    stays instant and the heavy stack is imported by the smoke, not by the probe.
    """
    if torch_version is None:
        try:
            from importlib.metadata import version

            torch_version = version("torch")
        except Exception:  # noqa: BLE001 - no torch, or a torch installed without metadata
            torch_version = None
    return {
        "python": platform.python_version(),
        "torch": torch_version,
        "os": os_name(),
        "arch": platform.machine() or None,
        "system": platform.system(),
        "release": platform.release(),
        "executable": sys.executable,
    }


@dataclass(frozen=True)
class Verdict:
    """Whether an environment is in the matrix, and the sentence that says why."""

    env: dict[str, Any]
    status: str                  # "executed" | "declared" | "outside" | "incomplete"
    reasons: tuple[str, ...]
    cell: Cell | None = None

    @property
    def in_matrix(self) -> bool:
        return self.status in ("executed", "declared")

    @property
    def executed(self) -> bool:
        return self.status == "executed"

    def as_dict(self) -> dict[str, Any]:
        return {"env": self.env, "status": self.status, "reasons": list(self.reasons),
                "cell": self.cell.as_dict() if self.cell else None}


def classify(env: dict[str, Any] | None = None) -> Verdict:
    """Place one environment in the matrix: executed cell, declared cell, or outside it.

    The order of the checks is the order a user hits them: the Python version, then torch, then the cell.
    """
    env = dict(env or probe())
    data = matrix()
    py_min, py_max = data["python"]["min"], data["python"]["max"]
    floor = str(data["torch"].get("floor", ""))
    line = version_line(env.get("python"))
    torch_line = version_line(env.get("torch"))

    if not line:
        return Verdict(env, "incomplete", ("the Python version could not be read",))
    if _tuple(line) < _tuple(py_min):
        return Verdict(env, "outside", (f"python {line} is older than the oldest declared version, {py_min}",))
    if _tuple(line) > _tuple(py_max):
        return Verdict(env, "outside", (
            f"python {line} is newer than the newest declared version, {py_max}",
            "the package installs there (requires-python has no upper bound) but the matrix says nothing about it",
        ))
    if not torch_line:
        return Verdict(env, "incomplete", ("torch is not installed in this interpreter",))
    if floor and _tuple(torch_line) < _tuple(floor):
        return Verdict(env, "outside", (f"torch {torch_line} is older than the declared floor, {floor}",))

    want = os_name(str(env.get("os") or ""))
    for cell in cells():
        if version_line(cell.python) == line and cell.torch_version == torch_line \
                and os_name(cell.os) == want:
            if cell.status == "executed":
                return Verdict(env, "executed", (f"these are the versions of the run recorded on "
                                                 f"{cell.run.get('at', '?')}",), cell)
            return Verdict(env, "declared", (
                f"this is cell {cell.label()}: declared, and nobody has recorded a smoke run on it",
                "run scripts/check_matrix.py on this machine and record it",
            ), cell)

    if executed(env.get("python"), env.get("torch"), want):
        return Verdict(env, "executed", ("a run of this exact environment is recorded, outside the declared cells",))
    lines = ", ".join(f"{k} {v}" for k, v in data["torch"]["lines"].items())
    return Verdict(env, "declared", (
        f"python {line} on {want} with torch {torch_line} is inside the declared range "
        f"(python {py_min}-{py_max}, torch >= {floor}), but no cell covers it: the cells use {lines}",
    ))


# ── evidence ─────────────────────────────────────────────────────────────────────────────────────────────
def record(env: dict[str, Any], results: list[dict[str, Any]], *, at: str | None = None,
           note: str = "", path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Write one smoke run into the matrix file and return the record.

    A run of the same environment replaces the earlier one (the newest evidence is the one that counts); the file
    stays sorted, so a new run is a one-line diff in the pull request. Only a run in which every step passed is
    recorded: the caller checks that before calling.
    """
    data = matrix()
    run = {
        "at": at or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "python": platform.python_version() if env.get("python") is None else str(env["python"]),
        "torch": None if env.get("torch") is None else str(env["torch"]),
        "os": os_name(env.get("os")),
        "arch": env.get("arch"),
        "smoke": [r["id"] for r in results if r.get("ok")],
        "failed": [r["id"] for r in results if not r.get("ok")],
        "seconds": round(sum(float(r.get("seconds") or 0.0) for r in results), 1),
    }
    if note:
        run["note"] = note
    kept = [r for r in data.get("runs") or []
            if not (version_line(r.get("python")) == version_line(run["python"])
                    and version_line(r.get("torch")) == version_line(run["torch"])
                    and r.get("os") == run["os"])]
    data["runs"] = sorted(kept + [run], key=lambda r: (r.get("os", ""), version_line(r.get("python")),
                                                       version_line(r.get("torch"))))
    _write(data, path)
    _reload()
    return run


# ── the page ─────────────────────────────────────────────────────────────────────────────────────────────
def _row(cell: Cell) -> str:
    mark = "✅ executed" if cell.status == "executed" else "⬜ declared"
    when = cell.run.get("at", "") if cell.run else ""
    return f"| {cell.os} | {cell.python} | {cell.torch} ({cell.torch_version}) | {mark} | {when} |"


def render_markdown() -> str:
    """The page ``docs/support_matrix.md``, from the declaration and the recorded runs."""
    data = matrix()
    cells_ = cells()
    total, done = len(cells_), sum(1 for c in cells_ if c.status == "executed")
    py_min, py_max = data["python"]["min"], data["python"]["max"]
    floor = data["torch"]["floor"]
    lines = ", ".join(f"**{k}** {v}" for k, v in data["torch"]["lines"].items())
    missing_os = sorted({c.os for c in cells_ if c.status == "declared"})

    out = [
        "# Supported versions: the matrix, and what was actually run",
        "",
        "Generated by `scripts/check_matrix.py --markdown` from `pinneapple_stability/support_matrix.json` and the "
        "runs recorded there (see issue #93). Do not edit by hand.",
        "",
        "**The rule.** A cell is green only when a run of `scripts/check_matrix.py` from that environment is "
        "recorded in the file. Being declared is not being green: the page says *executed* for the cells somebody "
        "ran and *declared* for the ones nobody did, and never the other way around.",
        "",
        "## Declared",
        "",
        "| Axis | Promise |",
        "|---|---|",
        f"| Python | {py_min} to {py_max} |",
        f"| PyTorch | {floor} or newer; the matrix covers {lines} |",
        f"| Operating system | {', '.join(data['os'])} |",
        "",
        f"`requires-python` in `pyproject.toml` is `>=3.10`, with no upper bound, and the classifiers list "
        f"{py_min} to {py_max}: that is the declaration this page and the check read.",
        "",
        "## The minimal matrix",
        "",
        f"{total} cells. Every Python version in the range is covered once on Linux, both torch lines are covered "
        f"on macOS, and Windows is covered on the middle Python version.",
        "",
        "| OS | Python | Torch | Status | Run |",
        "|---|---|---|---|---|",
        *[_row(c) for c in cells_],
        "",
        f"**{done} of {total} cells executed.** The other {total - done} are declared only: nothing has been run "
        f"on {' or '.join(missing_os) if missing_os else 'no OS'}, so this page does not claim they work.",
        "",
        "## What the smoke runs",
        "",
        "The same steps in every environment, chosen because they are the ones a torch or numpy change breaks "
        "first, and because they finish in seconds:",
        "",
        "| Step | What it does |",
        "|---|---|",
        *[f"| `{step['id']}` | {step['what']} |" for step in data["smoke"]],
        "",
        "Nothing in it trains a network to convergence or solves a 3-D case: those are hours, and the question "
        "here is whether the environment runs the library at all.",
        "",
        "## Recorded runs",
        "",
    ]
    if not runs():
        out += ["No run recorded yet.", ""]
    else:
        out += [
            "| Date | OS | Arch | Python | Torch | Steps | Seconds |",
            "|---|---|---|---|---|---|---|",
        ]
        for run in runs():
            steps = f"{len(run.get('smoke') or [])} ok"
            if run.get("failed"):
                steps += f", {len(run['failed'])} failed"
            out.append(f"| {run.get('at', '')} | {run.get('os', '')} | {run.get('arch') or ''} | "
                       f"{version_line(run.get('python'))} | {version_line(run.get('torch')) or 'none'} | "
                       f"{steps} | {run.get('seconds', '')} |")
        out.append("")
        for run in runs():
            if run.get("note"):
                out.append(f"- {run.get('os')} python {version_line(run.get('python'))}: {run['note']}")
        out.append("")

    out += [
        "## Running the check",
        "",
        "```bash",
        "python scripts/check_matrix.py              # is this environment in the matrix? + the smoke",
        "python scripts/check_matrix.py --json       # the same, machine-readable",
        "python scripts/check_matrix.py --record     # write this passing run into support_matrix.json",
        "python scripts/check_matrix.py --markdown   # regenerate this page",
        "```",
        "",
        "The exit code is `0` when the environment is in the matrix and every step passed, `2` when the smoke "
        "passed but the environment is outside the matrix (a newer Python, a torch older than the floor, no "
        "torch), and `1` when a step failed.",
        "",
        "## How to close a declared cell",
        "",
        "A declared cell becomes executed with one command, in an environment of that combination:",
        "",
        "```bash",
        "python3.12 -m venv /tmp/m && /tmp/m/bin/pip install -e .",
        "/tmp/m/bin/python scripts/check_matrix.py --record --note \"ubuntu 24.04, ARM runner\"",
        "```",
        "",
        "The plan of 2026-10-03 runs the matrix in GitHub Actions, which is off in this repository for billing "
        "reasons, so the Linux and Windows cells need a machine or a runner outside it. Until then they stay "
        "declared on this page, which is the honest reading of them.",
        "",
    ]
    return "\n".join(out)

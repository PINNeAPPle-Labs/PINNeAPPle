#!/usr/bin/env python3
"""The example catalog: every example either runs in CI or is marked manual with a reason that was measured (#41).

``examples/`` holds about a hundred folders, each with an ``experiment.json`` manifest. Nothing said which of them
still work: the import check (``scripts/check_example_imports.py``) only proves a name exists, and the manifests say
what an example is about, not whether anyone ran it recently. So a folder could rot for a year and still look
presentable in the docs.

``examples/run_tiers.json`` is the declaration, and this script is what keeps it honest:

* **ci** — the entries the tier runs, each with a command and a wall-clock cap. The whole tier has a budget
  (``budget_seconds``), so "runs in CI" stays affordable.
* **manual** — every other example, each with a reason. The reasons are *measured*: ``--measure`` runs the script
  with a timeout and records what actually happened ("took longer than 20 s", "failed: ModuleNotFoundError..."), or
  derives it from a checked fact (needs a package that is not part of PINNeAPPle).

Usage
-----
    python scripts/example_catalog.py --check        # the gate: coverage, caps, broken imports, budget
    python scripts/example_catalog.py --run          # run the ci tier (what the CI job does)
    python scripts/example_catalog.py --measure      # re-derive the reasons (slow: runs every example)
    python scripts/example_catalog.py --write        # write examples/run_tiers.json from the measurement
    python scripts/example_catalog.py --docs         # regenerate docs/guides/example_catalog.md

An example that imports a name this repository does not have stays in ``scripts/example_imports_baseline.txt`` and is
reported here as such: the two gates share the same checker, so a fixed import cannot linger in one of them.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
TIERS = EXAMPLES / "run_tiers.json"
DOCS = ROOT / "docs" / "guides" / "example_catalog.md"
SCHEMA = "pinneapple.example_runs/1"
MEASURE_TIMEOUT = 12

#: Top-level packages that ship with PINNeAPPle (everything under ``pinneapple*``). An example whose code needs
#: anything else is manual by definition: the CI install would have to grow a dependency for it.
FAMILY = "pinneapple"


def _env() -> dict[str, str]:
    """The environment an example runs in: this checkout on ``PYTHONPATH`` (an example is not installed) and no
    interactive backend. ``PYTHONPATH`` must be absolute -- a relative ``.`` resolves against the example's own
    folder once ``cwd`` moves there, which is exactly what made the first measurement report 72 missing packages."""
    env = dict(os.environ)
    parts = [str(ROOT)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and p != "."]
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env.setdefault("MPLBACKEND", "Agg")
    return env


def _run(argv: list[str], folder: Path, cap: float) -> subprocess.CompletedProcess | None:
    """Run one example in its own session and kill the whole group on timeout.

    ``subprocess.run(timeout=...)`` kills only the direct child: an example that spawns a process pool leaves the
    grandchildren holding the pipe open, and the call blocks on the pipe long after the timeout fired -- which is how
    the first two measurements of this catalog hung on the same example every time.
    """
    proc = subprocess.Popen(argv, cwd=str(folder), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env=_env(), start_new_session=True)
    try:
        out, err = proc.communicate(timeout=cap)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        with contextlib.suppress(Exception):
            proc.communicate(timeout=5)
        return None
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)


def _manifest(folder: Path) -> dict[str, Any]:
    path = folder / "experiment.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def examples() -> list[Path]:
    """Every example folder: a directory under ``examples/`` with a manifest (the declaration that it is one)."""
    return sorted(p for p in EXAMPLES.iterdir() if p.is_dir() and (p / "experiment.json").is_file()
                  and not p.name.startswith(("_", ".")))


def scripts(folder: Path) -> list[Path]:
    """The scripts an example declares, falling back to what is in the folder."""
    declared = list(_manifest(folder).get("scripts") or [])
    found = [folder / name for name in declared if (folder / name).is_file()]
    return found or sorted(p for p in folder.glob("*.py") if not p.name.startswith("_"))


def load_tiers() -> dict[str, Any]:
    if not TIERS.is_file():
        return {"schema": SCHEMA, "budget_seconds": 600, "ci": [], "manual": {}}
    return json.loads(TIERS.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ classification
def _external_modules(folder: Path) -> list[str]:
    """Top-level packages the example needs that are not part of PINNeAPPle."""
    wanted = [str(m) for m in (_manifest(folder).get("modules") or [])]
    out = []
    for name in wanted:
        top = name.split(".")[0]
        if not top.startswith(FAMILY) and top not in out:
            out.append(top)
    return out


def _broken_imports() -> set[str]:
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import check_example_imports as chk
    except Exception:  # pragma: no cover - the checker moved
        return set()
    return {problem.split(":")[0] for problem in chk.check_all()}


def classify(folder: Path, broken: set[str], run: bool = False) -> dict[str, Any]:
    """What the evidence says about this example: ``{"id", "mode", "reason", ...}``."""
    rel_folder = folder.relative_to(ROOT).as_posix()
    entry: dict[str, Any] = {"id": rel_folder, "mode": "manual"}
    files = scripts(folder)
    if not files:
        entry["reason"] = "no script: the folder declares an example but has no .py to run"
        return entry
    hits = sorted(p for p in files if p.relative_to(ROOT).as_posix() in broken)
    if hits:
        names = ", ".join(p.name for p in hits)
        entry["reason"] = (f"imports a name this repository does not have yet ({names}); listed in "
                           "scripts/example_imports_baseline.txt")
        return entry
    external = _external_modules(folder)
    if external:
        entry["reason"] = ("needs " + ", ".join(f"`{m}`" for m in external) +
                           ", which is not part of PINNeAPPle; run it where that is installed")
        return entry
    if not run:
        entry["reason"] = "not in the ci tier and not measured yet: run scripts/example_catalog.py --measure"
        return entry
    only = files[0]
    started = time.perf_counter()
    # A declared script can sit in a subfolder of the example (``classical_cases/run_cases.py``), so it is run by its
    # path inside the folder, not by its bare name: with the bare name every example that keeps its scripts one level
    # down was measured as "can't open file" and filed as failing (#625).
    proc = _run([sys.executable, only.relative_to(folder).as_posix()], folder, MEASURE_TIMEOUT)
    if proc is None:
        entry["reason"] = f"takes longer than {MEASURE_TIMEOUT} s ({only.name}): too slow for the ci tier"
        return entry
    elapsed = round(time.perf_counter() - started, 1)
    if proc.returncode == 0:
        entry.update(mode="manual", seconds=elapsed,
                     reason=f"runs clean in {elapsed} s but is not in the ci tier yet: promote it with --promote")
        entry["promotable"] = True
        return entry
    last = (proc.stderr.strip().splitlines() or ["(no output)"])[-1]
    entry["reason"] = f"fails when run: {last[:200]}"
    return entry


def build(run: bool = False) -> dict[str, Any]:
    broken = _broken_imports()
    tiers = load_tiers()
    ci_ids = {entry["id"] for entry in tiers.get("ci") or []}
    manual: dict[str, str] = {}
    for folder in examples():
        rel = folder.relative_to(ROOT).as_posix()
        if rel in ci_ids:
            continue
        result = classify(folder, broken, run=run)
        manual[rel] = result["reason"]
    return {"schema": SCHEMA, "budget_seconds": int(tiers.get("budget_seconds", 600)),
            "ci": list(tiers.get("ci") or []), "manual": manual}


# ------------------------------------------------------------------ the gate
def check(tiers: dict[str, Any] | None = None) -> list[str]:
    """Every problem with the declaration, one readable line each (empty list = the gate passes)."""
    tiers = tiers if tiers is not None else load_tiers()
    problems: list[str] = []
    known = {folder.relative_to(ROOT).as_posix(): folder for folder in examples()}
    if tiers.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA!r}")

    ci = tiers.get("ci") or []
    ids = [entry.get("id") for entry in ci]
    for duplicate in {i for i in ids if ids.count(i) > 1}:
        problems.append(f"{duplicate} is in the ci tier twice")
    budget = int(tiers.get("budget_seconds") or 0)
    declared = 0
    broken = _broken_imports()
    for entry in ci:
        rel = entry.get("id", "")
        if rel not in known:
            problems.append(f"ci entry {rel!r} is not an example folder with an experiment.json")
            continue
        command = entry.get("command") or []
        if not command:
            problems.append(f"ci entry {rel} has no command")
        for name in command:
            if name.endswith(".py") and not (known[rel] / name).is_file():
                problems.append(f"ci entry {rel} names {name!r}, which is not in the folder")
            if (known[rel] / name).relative_to(ROOT).as_posix() in broken:
                problems.append(f"ci entry {rel} imports a name that does not exist yet ({name}); "
                                "the tier may not contain a known-broken example")
        cap = int(entry.get("max_seconds") or 0)
        if cap <= 0:
            problems.append(f"ci entry {rel} has no max_seconds cap")
        declared += cap
    if budget and declared > budget:
        problems.append(f"the ci tier declares {declared} s of caps against a {budget} s budget: either promote "
                        "fewer examples or raise budget_seconds deliberately")

    manual = tiers.get("manual") or {}
    for rel in sorted(set(manual) - set(known)):
        problems.append(f"manual entry {rel!r} is not an example folder any more: remove it")
    for rel, folder in sorted(known.items()):
        if rel in ids:
            continue
        reason = (manual.get(rel) or "").strip()
        if not reason:
            problems.append(f"{rel} is neither in the ci tier nor marked manual with a reason")
        elif "not measured yet" in reason:
            problems.append(f"{rel} has no measured reason yet: run --measure")
    return problems


# ------------------------------------------------------------------ run / write / docs
def run_ci(tiers: dict[str, Any] | None = None) -> int:
    """Run every ci entry; a failure or a timeout fails the tier."""
    tiers = tiers if tiers is not None else load_tiers()
    failures = 0
    total = 0.0
    for entry in tiers.get("ci") or []:
        folder = ROOT / entry["id"]
        cap = int(entry.get("max_seconds") or 120)
        started = time.perf_counter()
        proc = _run([sys.executable, *entry["command"]], folder, cap)
        if proc is None:
            status, detail = "TIMEOUT", f"over its {cap} s cap"
        else:
            status = "PASS" if proc.returncode == 0 else "FAIL"
            detail = "" if proc.returncode == 0 else (proc.stderr.strip().splitlines() or ["?"])[-1][:200]
        elapsed = time.perf_counter() - started
        total += elapsed
        failures += status != "PASS"
        print(f"{status} {entry['id']} ({elapsed:.1f} s of {cap} s){' -- ' + detail if detail else ''}")
    print(f"\n{len(tiers.get('ci') or [])} example(s), {failures} failed, {total:.1f} s total")
    return 1 if failures else 0


def write(tiers: dict[str, Any]) -> None:
    TIERS.write_text(json.dumps(tiers, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def docs(tiers: dict[str, Any] | None = None) -> str:
    tiers = tiers if tiers is not None else load_tiers()
    lines = ["# Example catalog", "",
             "Every folder under `examples/` is either run by CI or marked manual with a reason. The reasons are "
             "measured, not assumed: `python scripts/example_catalog.py --measure` runs each example and records "
             "what happened, and `--check` is the gate that keeps this page true.", "",
             f"**In the CI tier: {len(tiers.get('ci') or [])} · manual: {len(tiers.get('manual') or {})} · "
             f"budget: {int(tiers.get('budget_seconds') or 0)} s.**", "",
             "## Runs in CI", "", "| example | command | cap |", "|---|---|---|"]
    for entry in tiers.get("ci") or []:
        lines.append(f"| `{entry['id']}` | `{' '.join(entry['command'])}` | {entry.get('max_seconds')} s |")
    lines += ["", "## Manual, with the reason", "", "| example | why it is not in the tier |", "|---|---|"]
    for rel, reason in sorted((tiers.get("manual") or {}).items()):
        lines.append(f"| `{rel}` | {reason} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="The example catalog: ci tier, manual reasons, and the gate.")
    ap.add_argument("--check", action="store_true", help="the gate: coverage, caps, broken imports, budget")
    ap.add_argument("--run", action="store_true", help="run the ci tier")
    ap.add_argument("--measure", action="store_true", help="re-derive the reasons (slow)")
    ap.add_argument("--write", action="store_true", help="with --measure: write examples/run_tiers.json")
    ap.add_argument("--docs", action="store_true", help="regenerate docs/guides/example_catalog.md")
    args = ap.parse_args(argv)

    if args.measure:
        fresh = build(run=True)
        if args.write:
            write(fresh)
        print(json.dumps({"ci": len(fresh["ci"]), "manual": len(fresh["manual"]),
                          "promotable": [k for k, v in fresh["manual"].items()
                                         if "runs clean" in v]}, indent=1))
    if args.docs:
        DOCS.parent.mkdir(parents=True, exist_ok=True)
        DOCS.write_text(docs(), encoding="utf-8")
        print(f"wrote {DOCS.relative_to(ROOT)}")
    if args.check:
        problems = check()
        for p in problems:
            print(f"problem: {p}")
        print(f"{len(problems)} problem(s)" if problems else "the example catalog is complete")
        return 1 if problems else 0
    if args.run:
        return run_ci()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

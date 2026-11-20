#!/usr/bin/env python3
"""Test shards: the wide suite blocks in the parts that are green, and says why the rest does not (#90).

`tests.yml` runs about thirty selections as must-pass steps. Everything else — roughly four thousand tests — ran only
in the nightly job, with a single failure ceiling for the whole thing. That has two costs. A shard that is perfectly
green cannot block a pull request, so a regression in it is caught a day later, if at all. And a single number ("81
failures allowed") hides which part is failing: the ceiling tolerates a regression in a green shard as long as some
other shard improved.

`tests/tier_shards.json` splits `tests/` into shards. A shard is **blocking** when a recorded measurement shows it
green: then every pull request runs it and a failure fails the build. A shard that is not blocking carries a reason
and a ceiling, and stays in the nightly job. The gate refuses a test file that belongs to no shard (a new test cannot
land unclassified), a blocking shard with no measurement, a non-blocking shard with no reason, and a path that does
not exist.

Usage
-----
    python scripts/tier_shards.py --check            # the gate
    python scripts/tier_shards.py --run-blocking     # what the pull-request job runs
    python scripts/tier_shards.py --measure data     # run one shard, print the numbers to record
    python scripts/tier_shards.py --promote data     # make it blocking, with the measurement it just took
    python scripts/tier_shards.py --docs             # regenerate docs/dev/test_shards.md

Promoting is a measurement, not a wish: `--promote` runs the shard first and refuses to promote one that fails.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import signal
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
DECLARATION = TESTS / "tier_shards.json"
DOCS = ROOT / "docs" / "dev" / "test_shards.md"
SCHEMA = "pinneapple.test_shards/1"


def load() -> dict[str, Any]:
    if not DECLARATION.is_file():
        return {"schema": SCHEMA, "shards": []}
    return json.loads(DECLARATION.read_text(encoding="utf-8"))


def write(declaration: dict[str, Any]) -> None:
    DECLARATION.write_text(json.dumps(declaration, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def files_of(shard: dict[str, Any]) -> list[str]:
    """The test files a shard owns, relative to the repository root."""
    out: list[str] = []
    for path in shard.get("paths") or []:
        target = ROOT / path
        if target.is_dir():
            out += [p.relative_to(ROOT).as_posix() for p in sorted(target.rglob("test_*.py"))]
        else:
            out += [Path(p).relative_to(ROOT).as_posix() if Path(p).is_absolute() else p
                    for p in sorted(glob.glob(str(ROOT / path)))]
    return out


def all_test_files() -> list[str]:
    return [p.relative_to(ROOT).as_posix() for p in sorted(TESTS.rglob("test_*.py"))]


# ------------------------------------------------------------------ the gate
def check(declaration: dict[str, Any] | None = None) -> list[str]:
    declaration = declaration if declaration is not None else load()
    problems: list[str] = []
    if declaration.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA!r}")
    shards = declaration.get("shards") or []
    if not shards:
        return problems + ["the declaration has no shards"]

    owner: dict[str, str] = {}
    for shard in shards:
        sid = shard.get("id") or "<no id>"
        paths = shard.get("paths") or []
        if not paths:
            problems.append(f"{sid}: no paths")
        for path in paths:
            if not glob.glob(str(ROOT / path)) and not (ROOT / path).exists():
                problems.append(f"{sid}: path {path!r} matches nothing")
        for rel in files_of(shard):
            if rel in owner:
                problems.append(f"{rel} is claimed by both {owner[rel]} and {sid}")
            owner[rel] = sid
        if shard.get("blocking"):
            if not str(shard.get("measured") or "").strip():
                problems.append(f"{sid}: blocking without a recorded measurement -- promote it with --promote")
            if int(shard.get("ceiling") or 0) != 0:
                problems.append(f"{sid}: a blocking shard has ceiling 0; {shard.get('ceiling')} would allow a failure")
        else:
            if not str(shard.get("reason") or "").strip():
                problems.append(f"{sid}: not blocking and no reason why")
            if shard.get("ceiling") is None:
                problems.append(f"{sid}: not blocking and no ceiling -- the nightly job needs a number to allow")

    for rel in all_test_files():
        if rel not in owner:
            problems.append(f"{rel} belongs to no shard: add it to one in tests/tier_shards.json")
    for rel in sorted(set(owner) - set(all_test_files())):
        problems.append(f"{owner[rel]} claims {rel}, which is not a test file any more")
    ids = [s.get("id") for s in shards]
    for duplicate in {i for i in ids if ids.count(i) > 1}:
        problems.append(f"shard id {duplicate!r} appears twice")
    return problems


# ------------------------------------------------------------------ running
#: A shard is measured with a cap: the wide suite has tests that wait on a solver or a network, and a measurement
#: that never returns is worse than a shard that is not blocking yet.
SHARD_TIMEOUT = 600


def _pytest(paths: list[str], extra: list[str] | None = None,
            timeout: float = SHARD_TIMEOUT) -> subprocess.CompletedProcess | None:
    """Run pytest for one shard, killing the whole process group on timeout (see #41 for why the group)."""
    # no `-q` here: pyproject.toml already sets it, and a second one takes the quiet level to -qq, which drops the
    # "218 passed in 35.7s" line this measurement exists to record
    argv = [sys.executable, "-m", "pytest", *paths, "--tb=no", "-p", "no:warnings",
            "--continue-on-collection-errors", *(extra or [])]
    proc = subprocess.Popen(argv, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        with contextlib.suppress(Exception):
            proc.communicate(timeout=5)
        return None
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)


def measure(shard: dict[str, Any]) -> dict[str, Any]:
    """Run one shard and return what a promotion needs: the counts and a one-line evidence string."""
    started = time.perf_counter()
    proc = _pytest([p for p in shard.get("paths") or []])
    elapsed = time.perf_counter() - started
    if proc is None:
        return {"returncode": 124, "failed": [], "seconds": round(elapsed, 1),
                "summary": f"over the {SHARD_TIMEOUT} s cap", "tail": f"over the {SHARD_TIMEOUT} s cap"}
    out = proc.stdout + proc.stderr
    failed = [line.split(" - ")[0][len("FAILED "):] for line in out.splitlines() if line.startswith("FAILED ")]
    summary = next((line.strip(" =") for line in reversed(out.splitlines())
                    if any(word in line for word in ("passed", "failed", "error", "no tests ran"))), "")
    return {"returncode": proc.returncode, "failed": failed, "seconds": round(elapsed, 1), "summary": summary,
            "tail": out.strip().splitlines()[-1] if out.strip() else ""}


def run_blocking(declaration: dict[str, Any] | None = None, only: str | None = None) -> int:
    declaration = declaration if declaration is not None else load()
    failures = 0
    total = 0.0
    for shard in declaration.get("shards") or []:
        if only and shard.get("id") != only:
            continue
        if not only and not shard.get("blocking"):
            continue
        started = time.perf_counter()
        result = measure(shard)
        elapsed = time.perf_counter() - started
        total += elapsed
        ok = result["returncode"] == 0 and not result["failed"]
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'} {shard['id']} ({elapsed:.1f} s) {result['summary']}".rstrip())
        for name in result["failed"][:10]:
            print(f"     failed: {name}")
    print(f"\n{len([s for s in declaration.get('shards') or [] if s.get('blocking') or only])} shard(s), "
          f"{failures} failing, {total:.1f} s")
    return 1 if failures else 0


def promote(declaration: dict[str, Any], shard_id: str) -> int:
    shard = next((s for s in declaration["shards"] if s.get("id") == shard_id), None)
    if shard is None:
        print(f"no shard {shard_id!r}; known: {[s['id'] for s in declaration['shards']]}")
        return 2
    result = measure(shard)
    if result["returncode"] != 0 or result["failed"]:
        print(f"{shard_id} is not green ({result['summary'] or result['tail']}); fixing it comes before blocking on it")
        for name in result["failed"][:15]:
            print(f"  failed: {name}")
        return 1
    shard["blocking"] = True
    shard["ceiling"] = 0
    shard.pop("reason", None)
    shard["measured"] = f"{date.today().isoformat()}: {result['summary']} ({result['seconds']} s wall)"
    write(declaration)
    print(f"{shard_id} is blocking: {shard['measured']}")
    return 0


def docs(declaration: dict[str, Any] | None = None) -> str:
    declaration = declaration if declaration is not None else load()
    shards = declaration.get("shards") or []
    blocking = [s for s in shards if s.get("blocking")]
    lines = ["# Test shards", "",
             "The wide suite blocks in the parts that have been measured green, and says why the rest does not. "
             "`python scripts/tier_shards.py --check` is the gate; the pull-request job runs `--run-blocking`.", "",
             f"**Blocking: {len(blocking)} of {len(shards)} shards · tests covered: {len(all_test_files())} files.**", "",
             "## Blocking (every pull request runs these)", "",
             "| shard | paths | measured |", "|---|---|---|"]
    for shard in blocking:
        lines.append(f"| `{shard['id']}` | {', '.join(f'`{p}`' for p in shard['paths'])} | {shard['measured']} |")
    lines += ["", "## Not blocking yet, and why", "", "| shard | paths | ceiling | reason |", "|---|---|---|---|"]
    for shard in shards:
        if shard.get("blocking"):
            continue
        lines.append(f"| `{shard['id']}` | {', '.join(f'`{p}`' for p in shard['paths'])} | "
                     f"{shard.get('ceiling')} | {shard.get('reason')} |")
    lines += ["", "## Promoting a shard", "",
              "```bash", "python scripts/tier_shards.py --measure <id>    # see what it does today",
              "python scripts/tier_shards.py --promote <id>    # runs it again and, if green, makes it blocking",
              "```", "",
              "A shard is promoted by a measurement, never by opinion: `--promote` refuses a shard that fails, and the"
              " record it writes (date, counts, seconds) is what the gate reads.", ""]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Test shards: block the wide suite in the parts that are green.")
    ap.add_argument("--check", action="store_true", help="the gate")
    ap.add_argument("--run-blocking", action="store_true", help="run every blocking shard")
    ap.add_argument("--measure", metavar="ID", help="run one shard and print what it did")
    ap.add_argument("--promote", metavar="ID", help="measure a shard and, if it is green, make it blocking")
    ap.add_argument("--docs", action="store_true", help="regenerate docs/dev/test_shards.md")
    args = ap.parse_args(argv)

    declaration = load()
    if args.measure:
        shard = next((s for s in declaration["shards"] if s.get("id") == args.measure), None)
        if shard is None:
            print(f"no shard {args.measure!r}")
            return 2
        result = measure(shard)
        print(json.dumps(result, indent=1))
        return 0 if result["returncode"] == 0 else 1
    if args.promote:
        return promote(declaration, args.promote)
    if args.docs:
        DOCS.parent.mkdir(parents=True, exist_ok=True)
        DOCS.write_text(docs(declaration), encoding="utf-8")
        print(f"wrote {DOCS.relative_to(ROOT)}")
    if args.check:
        problems = check(declaration)
        for p in problems:
            print(f"problem: {p}")
        print(f"{len(problems)} problem(s)" if problems else "the shard declaration is complete")
        return 1 if problems else 0
    if args.run_blocking:
        return run_blocking(declaration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

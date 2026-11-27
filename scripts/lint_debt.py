#!/usr/bin/env python3
"""The lint debt, recorded and decreasing (#51).

The repository has two kinds of ruff finding: the ones a pull request introduces, and the tens of thousands that were
already there. The first kind has to fail the build; the second cannot, or nothing ever merges. `pyproject.toml`
declares the selection and `.pre-commit-config.yaml` runs it on the files a commit touches — that is the "new
violations are blocked" half. This script is the other half: **the old debt is written down, and the gate refuses to
let it grow**.

The baseline is `docs/dev/lint_debt.json`, one count per (file, rule). Comparing per pair instead of by total means
the gate cannot be satisfied by trading: removing five `UP006` in one module does not buy five in another.

Usage
-----
    python scripts/lint_debt.py --record     # write the baseline (a decision: say why in the commit)
    python scripts/lint_debt.py --check      # fail on any (file, rule) above its baseline
    python scripts/lint_debt.py --summary    # the numbers, by rule, with what is left to pay

`--check` also prints what the current tree *could* ratchet down to, so the next commit that removes debt can record
it instead of waiting for it to melt back.
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "docs" / "dev" / "lint_debt.json"
SCHEMA = "pinneapple.lint_debt/1"


def _ruff() -> list[str]:
    """The ruff command: the module if this interpreter has it, the binary otherwise."""
    try:
        subprocess.run([sys.executable, "-m", "ruff", "--version"], capture_output=True, check=True)
        return [sys.executable, "-m", "ruff"]
    except (subprocess.CalledProcessError, OSError):
        exe = shutil.which("ruff")
        if not exe:
            raise SystemExit("ruff is not installed: pip install ruff, or run this with the interpreter that has it")
        return [exe]


def inventory() -> dict[str, dict[str, int]]:
    """file -> {rule: count}, for the selection declared in pyproject.toml (ruff reads it from there)."""
    out = subprocess.run([*_ruff(), "check", ".", "--output-format", "json", "--no-cache"],
                         cwd=ROOT, capture_output=True, text=True)
    if out.returncode not in (0, 1):                      # 1 is "findings", not an error
        raise SystemExit(f"ruff failed ({out.returncode}): {out.stderr.strip()[:400]}")
    try:
        findings = json.loads(out.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ruff did not return JSON: {exc}: {out.stdout[:200]}")
    table: dict[str, dict[str, int]] = collections.defaultdict(collections.Counter)
    for f in findings:
        name = Path(f["filename"]).resolve().relative_to(ROOT).as_posix()
        table[name][f.get("code") or "?"] += 1
    return {name: dict(sorted(rules.items())) for name, rules in sorted(table.items())}


def total(table: dict[str, dict[str, int]]) -> int:
    return sum(sum(rules.values()) for rules in table.values())


def by_rule(table: dict[str, dict[str, int]]) -> dict[str, int]:
    out: collections.Counter = collections.Counter()
    for rules in table.values():
        out.update(rules)
    return dict(out.most_common())


def compare(baseline: dict[str, dict[str, int]], current: dict[str, dict[str, int]]) -> tuple[list, list]:
    """(increases, decreases). An increase is a (file, rule, baseline, current); a decrease is the ratchet."""
    worse, better = [], []
    for name in sorted(set(baseline) | set(current)):
        was, now = baseline.get(name, {}), current.get(name, {})
        for rule in sorted(set(was) | set(now)):
            before, after = was.get(rule, 0), now.get(rule, 0)
            if after > before:
                worse.append((name, rule, before, after))
            elif after < before:
                better.append((name, rule, before, after))
    return worse, better


def _where(path: Path) -> str:
    """The path as a reader should see it: relative to the repository when it is inside it, absolute otherwise."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def load(path: Path = BASELINE) -> dict[str, dict[str, int]]:
    if not path.is_file():
        raise SystemExit(f"no baseline at {_where(path)}: run --record first, and say why in the commit")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise SystemExit(f"{_where(path)} is {data.get('schema')!r}, this script writes {SCHEMA!r}")
    return data["files"]


def record(path: Path = BASELINE) -> int:
    table = inventory()
    payload = {"schema": SCHEMA, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "how": "ruff check . (selection from pyproject.toml), one count per file and rule",
               "total": total(table), "by_rule": by_rule(table), "files": table}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    print(f"recorded {payload['total']} finding(s) in {len(table)} file(s) -> {_where(path)}")
    return 0


def check(path: Path = BASELINE) -> int:
    baseline = load(path)
    current = inventory()
    worse, better = compare(baseline, current)
    was, now = total(baseline), total(current)
    if worse:
        print(f"new lint debt: {len(worse)} (file, rule) pair(s) above the baseline of "
              f"{_where(path)} -- fix it, or record a new baseline and say why:")
        for name, rule, before, after in worse[:40]:
            print(f"  {name}: {rule} {before} -> {after}")
        if len(worse) > 40:
            print(f"  ... and {len(worse) - 40} more")
        return 1
    print(f"lint debt ok: {now} finding(s) (baseline {was}), nothing above its count")
    if better:
        print(f"  {len(better)} pair(s) below the baseline, ready to ratchet down "
              f"({sum(b - a for _, _, b, a in better)} finding(s)): run --record and commit the drop")
    return 0


def summary(path: Path = BASELINE) -> int:
    current = inventory()
    rules = by_rule(current)
    print(f"{total(current)} finding(s) in {len(current)} file(s), {len(rules)} rule(s)")
    for rule, count in list(rules.items())[:25]:
        print(f"  {rule:8s} {count:6d}")
    if path.is_file():
        worse, better = compare(load(path), current)
        print(f"against {_where(path)}: {len(worse)} above, {len(better)} below")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--record", action="store_true", help="write the baseline from the current tree")
    ap.add_argument("--check", action="store_true", help="fail when a (file, rule) count grows past the baseline")
    ap.add_argument("--summary", action="store_true", help="print the debt by rule")
    a = ap.parse_args(argv)
    if a.record:
        return record()
    if a.check:
        return check()
    if a.summary:
        return summary()
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

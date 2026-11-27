#!/usr/bin/env python3
"""Measure the verification certificate: detection and false alarms on the five-family benchmark and on one CFD
surrogate (#341).

    python scripts/certificate_benchmark.py                      # write results/certificate_five_families.json
    python scripts/certificate_benchmark.py --out other.json     # write somewhere else
    python scripts/certificate_benchmark.py --check              # fail if the committed file is out of date

The cases, the noise levels and the tolerance are in ``pinneapple_analysis.trust.benchmark``; this script only runs
them and writes the document the guide prints, so the numbers in the guide are the numbers a rerun produces.

What the document answers, per family: of the fields that do **not** satisfy the declared physics, how many the
certificate refuses, and of the fields that do, how many it flags. A family where the two populations do not separate
is reported as such -- the misses are listed by label, with the number behind each.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinneapple_analysis.trust.benchmark import SCHEMA, measure, table, write  # noqa: E402

DEFAULT_OUT = ROOT / "results" / "certificate_five_families.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="where to write the JSON (default: %(default)s)")
    ap.add_argument("--check", action="store_true",
                    help="do not write; compare the committed JSON with a fresh measurement and fail on any difference")
    args = ap.parse_args(argv)

    report = measure()
    if args.check:
        path = Path(args.out)
        if not path.exists():
            print(f"{path} is missing: run python scripts/certificate_benchmark.py", file=sys.stderr)
            return 1
        committed = json.loads(path.read_text(encoding="utf-8"))
        fresh = json.loads(json.dumps(report, allow_nan=False))
        if committed.get("schema") != SCHEMA:
            print(f"{path} is a {committed.get('schema')!r} document, this script writes {SCHEMA!r}", file=sys.stderr)
            return 1
        if committed != fresh:
            print(f"{path} is out of date: rerun python scripts/certificate_benchmark.py", file=sys.stderr)
            return 1
        print(f"{path}: up to date")
        return 0

    path = write(args.out, report=report)
    print(table(report))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

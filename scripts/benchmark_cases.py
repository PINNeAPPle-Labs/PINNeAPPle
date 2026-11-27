"""Official benchmark cases: run them, publish the table, reproduce it, or check it (#30).

    python scripts/benchmark_cases.py --run        # train every case, rewrite the results file and the page
    python scripts/benchmark_cases.py --docs       # regenerate the page from the results file (no training)
    python scripts/benchmark_cases.py --reproduce  # train again and compare against what is published
    python scripts/benchmark_cases.py --check      # declaration + page, no training (this is the CI gate)

``--only`` picks cases, ``--seeds`` and ``--budget`` override the protocol for a quick look. An override is not a
benchmark: the page and the results file are only written when the protocol ran as declared.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pinneapple_tools.benchmark_suite import cases as cases_mod   # noqa: E402


def _print_table(results) -> None:
    print(cases_mod.table(results))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="The official benchmark cases and their protocol.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true", help="train every case, then write the results and the page")
    group.add_argument("--docs", action="store_true", help="regenerate the page from the results file")
    group.add_argument("--reproduce", action="store_true", help="train again and compare with the published table")
    group.add_argument("--check", action="store_true", help="verify the declaration and the page, no training")
    parser.add_argument("--only", nargs="*", metavar="ID", help="case ids, e.g. --only 01 03")
    parser.add_argument("--seeds", nargs="*", type=int, help="override the protocol seeds (a quick look)")
    parser.add_argument("--budget", type=json.loads, metavar="JSON", help='override the budget, e.g. \'{"epochs": 50}\'')
    args = parser.parse_args(argv)

    if args.check:
        problems = cases_mod.check(ROOT)
        for problem in problems:
            print(f"problem: {problem}")
        print(f"{len(problems)} problem(s)" if problems else "the benchmark cases are declared and published")
        return 1 if problems else 0

    if args.docs:
        results_path = ROOT / cases_mod.RESULTS
        if not results_path.is_file():
            print(f"problem: {cases_mod.RESULTS.as_posix()} is missing; run --run first")
            return 1
        results = json.loads(results_path.read_text(encoding="utf-8"))
        written, page_path = cases_mod.write(results, root=ROOT)
        print(f"wrote {page_path.relative_to(ROOT)} from {written.relative_to(ROOT)}")
        return 0

    results = cases_mod.run_all(seeds=args.seeds, budget=args.budget, only=args.only)
    _print_table(results)

    if args.reproduce:
        published_path = ROOT / cases_mod.RESULTS
        if not published_path.is_file():
            print(f"problem: nothing published at {cases_mod.RESULTS.as_posix()} to reproduce")
            return 1
        published = json.loads(published_path.read_text(encoding="utf-8"))
        problems = cases_mod.reproduces(published, results)
        for problem in problems:
            print(f"problem: {problem}")
        print(f"{len(problems)} cell(s) outside the tolerance" if problems else "the table reproduces")
        return 1 if problems else 0

    if args.seeds or args.budget or args.only:
        print("an override is not a benchmark: results and page were not written")
        return 0
    results_path, page_path = cases_mod.write(results, root=ROOT)
    print(f"wrote {results_path.relative_to(ROOT)} and {page_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

"""The trust benchmark (#349): run it, publish it, or check it against the record.

    python scripts/trust_benchmark.py --run     # fit the candidates, run the four failure kinds, write record + page
    python scripts/trust_benchmark.py --docs    # regenerate the page from the record (no fits)
    python scripts/trust_benchmark.py --check   # declaration, record and page agree -- the gate, no fits

``--only`` picks cases, ``--kinds`` picks failure kinds, ``--reduce`` runs a short stream for a look. A reduced run
never writes the record: the published numbers are only written when the protocol ran as declared, so a quick look
cannot replace the benchmark by accident.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pinneapple_tools.benchmark_suite import trust  # noqa: E402

#: The reduced protocol: a tenth of the stream and a smaller bottom, for a look at a slow machine.
REDUCED = {"samples": {"train": 120, "validation": 100, "calibration": 200},
           "stream": {"n_clean": 40, "n_after": 40}, "monitor": {"window": 10, "calibration_windows": 60}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="The trust benchmark: four failure kinds, one protocol, one board.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true", help="fit the candidates, run the suite, write the record")
    group.add_argument("--docs", action="store_true", help="regenerate the page from the record")
    group.add_argument("--check", action="store_true", help="verify the declaration, the record and the page")
    parser.add_argument("--only", nargs="*", metavar="ID", help="case ids, e.g. --only dust_ramp")
    parser.add_argument("--kinds", nargs="*", metavar="KIND", help=f"failure kinds: {', '.join(trust.KINDS)}")
    parser.add_argument("--reduce", action="store_true", help="a short stream: prints, never publishes")
    args = parser.parse_args(argv)

    if args.check:
        problems = trust.check(ROOT)
        for problem in problems:
            print(f"problem: {problem}")
        print(f"{len(problems)} problem(s)" if problems else "the trust benchmark is declared, recorded and in sync")
        return 1 if problems else 0

    if args.docs:
        path = ROOT / trust.RESULTS
        if not path.is_file():
            print(f"{trust.RESULTS.as_posix()} is missing: run --run first")
            return 1
        trust.write(json.loads(path.read_text(encoding="utf-8")), root=ROOT)
        print(f"rewrote {trust.PAGE.as_posix()} from {trust.RESULTS.as_posix()}")
        return 0

    protocol = REDUCED if args.reduce else None
    candidates = trust.default_candidates(protocol)
    results = trust.run_suite(candidates, protocol=protocol, kinds=args.kinds, only=args.only,
                              log=lambda s: print(s, flush=True))
    print(trust.table(results))
    if args.reduce or args.only or args.kinds:
        print("a reduced run (--reduce, --only or --kinds): the record and the page were not touched")
        return 0
    path, page = trust.write(results, root=ROOT)
    print(f"wrote {path.relative_to(ROOT).as_posix()} and {page.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

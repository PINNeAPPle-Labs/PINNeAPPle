"""``python -m pinneapple_reason``: run a case, list the library, explain a saved chain.

    python -m pinneapple_reason run roof --out roof_report [--lang en]
    python -m pinneapple_reason modes
    python -m pinneapple_reason explain roof_report/chain.json
"""
from __future__ import annotations

import argparse
import json
import sys

from . import Chain, cases, describe_library, run_case, write_report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pinneapple_reason")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="reason through a case and write its report")
    r.add_argument("case", choices=sorted(cases.CASES))
    r.add_argument("--out", default=None)
    r.add_argument("--lang", choices=("pt", "en"), default="pt")
    m = sub.add_parser("modes", help="the library of failure modes and countermeasures")
    m.add_argument("--lang", choices=("pt", "en"), default="pt")
    e = sub.add_parser("explain", help="explain a saved chain.json")
    e.add_argument("path")
    e.add_argument("--lang", choices=("pt", "en"), default="pt")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        chain = run_case(a.case, progress=lambda s: print(s, file=sys.stderr))
        print(chain.explain(a.lang))
        if a.out:
            paths = write_report(chain, a.out, lang=a.lang)
            print("\n" + "\n".join(f"{k}: {v}" for k, v in paths.items()))
        return 0 if chain.resolved else 1
    if a.cmd == "modes":
        print(json.dumps(describe_library(a.lang), indent=1, ensure_ascii=False))
        return 0
    print(Chain.load(a.path).explain(a.lang))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

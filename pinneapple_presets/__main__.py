"""python -m pinneapple_presets list | card NAME | run NAME [k=v ...] [--surrogate] | validate NAME |
surrogate NAME | report NAME|all [--build-surrogate] [--no-3d] | index"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import warnings

from . import load, names
from .registry import list_presets, write_index


def _value(s: str):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pinneapple_presets", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("index")
    for c in ("card", "validate", "surrogate"):
        sub.add_parser(c).add_argument("name")
    r = sub.add_parser("run")
    r.add_argument("name")
    r.add_argument("params", nargs="*", help="name=value overrides (SI units)")
    r.add_argument("--surrogate", action="store_true")
    r.add_argument("--json", action="store_true")
    rep = sub.add_parser("report")
    rep.add_argument("name")
    rep.add_argument("--build-surrogate", action="store_true")
    rep.add_argument("--no-3d", action="store_true")
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    if a.cmd == "list":
        for row in list_presets():
            caps = ", ".join(k for k, v in row["capabilities"].items() if v)
            print(f"{row['name']:34s} {row['sector']:38s} [{caps}]\n    {row['title']}")
    elif a.cmd == "card":
        print(json.dumps(load(a.name).card(), indent=1, ensure_ascii=False, default=str))
    elif a.cmd == "run":
        kw = dict(x.split("=", 1) for x in a.params)
        res = load(a.name).run(fidelity="surrogate" if a.surrogate else "solver",
                               **{k: _value(v) for k, v in kw.items()})
        print(json.dumps(res.to_dict(), indent=1, ensure_ascii=False) if a.json else res.summary())
        for x in res.decision.get("reasons", []):
            print("  -", x)
    elif a.cmd == "validate":
        v = load(a.name).validate()
        for row in v["references"]:
            print(f"{'PASS' if row['pass'] else 'FAIL'}  {row['case']}: {row['computed']:.6g} vs {row['reference']:.6g} "
                  f"{row['unit']} ({100 * row['rel_error']:.3g} %, tol {100 * row['tol']:.3g} %) [{row['source']}]")
        if v.get("convergence"):
            c = v["convergence"]
            print(f"{'PASS' if c['pass'] else 'FAIL'}  convergence in {c['parameter']}: last change "
                  f"{c['rel_change_last']:.2e}")
        return 0 if v["all_pass"] else 1
    elif a.cmd == "surrogate":
        from .surrogate import build_surrogate
        s = build_surrogate(load(a.name))
        print(json.dumps(s.meta["holdout"], indent=1))
    elif a.cmd == "report":
        from .report import write_report
        for n in (names() if a.name == "all" else [a.name]):
            s = write_report(n, surrogate="build" if a.build_surrogate else "load", three_d=not a.no_3d)
            print(f"{n}: {s['decision']['verdict']}, trust {s['trust']['decision']}, validation "
                  f"{'pass' if s['validation']['all_pass'] else 'FAIL'}, {s['wall_time_total_s']:.0f} s")
        from .report import write_catalogue
        write_catalogue()
        write_index()
    elif a.cmd == "index":
        print(write_index())
    return 0


if __name__ == "__main__":
    sys.exit(main())

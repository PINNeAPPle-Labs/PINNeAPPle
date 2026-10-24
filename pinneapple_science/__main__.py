"""``python -m pinneapple_science atlas [--html PATH] | list [--area A] [--dim D] | run NAME [--dim D] [--quick]
| report NAME|all [--quick] [--no-register] | table``."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pinneapple_science")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("atlas", help="print the atlas (and write it as a page with --html)")
    a.add_argument("--html")
    a.add_argument("--check", action="store_true", help="validate every pointer; exit 1 on a problem")
    ls = sub.add_parser("list", help="experiments")
    ls.add_argument("--area")
    ls.add_argument("--dim")
    ls.add_argument("--level")
    ls.add_argument("--kind")
    r = sub.add_parser("run", help="run one experiment")
    r.add_argument("name")
    r.add_argument("--dim")
    r.add_argument("--quick", action="store_true")
    r.add_argument("--out")
    rp = sub.add_parser("report", help="committed results, manifests, seals and the catalogue")
    rp.add_argument("names", nargs="+", help="experiment names or 'all'")
    rp.add_argument("--quick", action="store_true")
    rp.add_argument("--no-register", action="store_true")
    sub.add_parser("table", help="experiment x reference x measured error from the committed results (JSON)")
    ns = ap.parse_args(argv)
    if ns.cmd == "atlas":
        from .atlas import load, validate
        at = load()
        print(at)
        if ns.html:
            at.to_html(ns.html)
            print("wrote", ns.html)
        if ns.check:
            problems = validate(at)
            print("\n".join(problems) or "atlas ok")
            return 1 if problems else 0
        return 0
    if ns.cmd == "list":
        from .registry import experiments
        for e in experiments(ns.area, level=ns.level, kind=ns.kind, dim=ns.dim):
            print(f"{e['name']:32s} {e['level']:9s} {e['kind']:9s} {'/'.join(e['dims']):14s} {e['title']}")
        return 0
    if ns.cmd == "run":
        from .registry import run
        res = run(ns.name, ns.dim, quick=ns.quick)
        print(res)
        if ns.out:
            print(json.dumps(res.save(ns.out), indent=1))
        return 0 if res.passed else 1
    if ns.cmd == "report":
        from . import report
        from .registry import names
        todo = names() if ns.names == ["all"] else ns.names
        for n in todo:
            out = report.write(n, quick=ns.quick, register=not ns.no_register)
            print(n, "->", out["folder"], "item", out["item"],
                  "; ".join(f"{r.dim}: {'PASS' if r.passed else 'FAIL'} {r.trust.decisao}" for r in out["results"]))
        print(report.write_catalogue(register=not ns.no_register))
        return 0
    from .report import table
    print(json.dumps(table(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

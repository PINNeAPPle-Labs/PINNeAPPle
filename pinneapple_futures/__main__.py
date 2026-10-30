"""``python -m pinneapple_futures atlas [--check] | list [--area A] | run NAME [--quick] [--out DIR]
| ask QUESTION [--run] | publish NAME|all --out DIR``."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pinneapple_futures",
                                 description="PINNeAPPle Futures Lab: speculative physics, investigated.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("atlas", help="print the directions of the Lab")
    a.add_argument("--markdown", action="store_true", help="print the table used by the guide")
    a.add_argument("--check", action="store_true", help="validate every pointer; exit 1 on a problem")

    ls = sub.add_parser("list", help="the registered experiments")
    ls.add_argument("--area")

    r = sub.add_parser("run", help="run one experiment")
    r.add_argument("name")
    r.add_argument("--quick", action="store_true")
    r.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="an experiment parameter (repeatable)")
    r.add_argument("--out", help="save the result, figures and page here")

    k = sub.add_parser("ask", help="turn a what-if question into an experiment")
    k.add_argument("question")
    k.add_argument("--run", action="store_true", help="run the experiment the question maps to")
    k.add_argument("--quick", action="store_true")
    k.add_argument("--json", action="store_true", help="print the plan as JSON")

    p = sub.add_parser("publish", help="run, save and write the publication page")
    p.add_argument("names", nargs="+", help="experiment names or 'all'")
    p.add_argument("--out", default="out/futures")
    p.add_argument("--quick", action="store_true")

    ns = ap.parse_args(argv)
    from .atlas import load as load_atlas, validate as validate_atlas

    if ns.cmd == "atlas":
        at = load_atlas()
        print(at.to_markdown() if ns.markdown else at)
        if ns.check:
            problems = validate_atlas(at)
            print("\n".join(problems) or "atlas ok")
            return 1 if problems else 0
        return 0

    if ns.cmd == "list":
        from .core import experiments

        for e in experiments(ns.area):
            print(f"{e['name']:32s} {e['area']:22s} {e['level']:9s} {'/'.join(e['dims']):8s} {e['title']}")
        return 0

    if ns.cmd == "run":
        from .core import run

        params = {}
        for item in ns.set:
            if "=" not in item:
                raise SystemExit(f"--set needs KEY=VALUE, got {item!r}")
            key, value = item.split("=", 1)
            params[key] = float(value)
        res = run(ns.name, quick=ns.quick, **params)
        print(res)
        if ns.out:
            from .report import publish

            info = publish(ns.name, root=ns.out, quick=ns.quick)
            print(json.dumps(info, indent=1, default=str))
        return 0 if res.passed else 1

    if ns.cmd == "ask":
        from .whatif import ask

        brief = ask(ns.question)
        if ns.json:
            print(json.dumps(brief.to_dict(), indent=1))
        else:
            print(brief)
        if not ns.run:
            return 0 if brief.understood else 1
        if not brief.answerable:
            print("\nnothing to run: no registered experiment contains this parameter", file=sys.stderr)
            return 1
        res = brief.run(quick=ns.quick)
        print("\n" + repr(res))
        return 0 if res.passed else 1

    if ns.cmd == "publish":
        from .core import names
        from .report import publish, write_catalogue

        todo = names() if ns.names == ["all"] else ns.names
        for name in todo:
            info = publish(name, root=ns.out, quick=ns.quick)
            print(f"{name:34s} {info['decision']:9s} {info['page']}")
        if ns.names == ["all"]:
            print("catalogue", write_catalogue(ns.out, quick=ns.quick))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

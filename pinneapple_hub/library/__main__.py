"""``python -m pinneapple_hub.library build | run [--stale] [ids ...] | check`` (see the package docstring)."""
from __future__ import annotations

import argparse
import sys

from . import INDEX, build, check, collect, load_index, record_runs, write_index


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pinneapple_hub.library")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="regenerate data/web/hub/index.json from the sources")
    b.add_argument("--only", nargs="*", help="collectors to run (models, geometries, ...)")
    r = sub.add_parser("run", help="run the snippets and record their output (runs.json), then rebuild the index")
    r.add_argument("ids", nargs="*", help="only these item ids")
    r.add_argument("--stale", action="store_true", help="only snippets without a run of their current code")
    r.add_argument("--timeout", type=float, default=300.0)
    r.add_argument("--workers", type=int, default=4)
    f = sub.add_parser("flowruns", help="run the bundled flows headless and record each node's metrics")
    f.add_argument("names", nargs="*", help="only these flows")
    f.add_argument("--timeout", type=float, default=900.0)
    sub.add_parser("check", help="the committed index is current and every snippet has a passing run")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        idx = build(a.only)
        print(write_index(idx), idx["counts"])
        return 0
    if a.cmd == "flowruns":
        from .flowruns import record
        record(a.names or None, timeout=a.timeout)
        write_index(build())
        return 0
    if a.cmd == "run":
        items = collect()
        runs = record_runs(items, stale_only=a.stale, timeout=a.timeout, ids=a.ids or None,
                           workers=a.workers)
        write_index(build(runs=runs))
        return 0
    problems = check(load_index(INDEX))
    fresh = build()
    if [i["id"] for i in fresh["items"]] != [i["id"] for i in load_index(INDEX)["items"]]:
        problems.append("index.json is not current: python -m pinneapple_hub.library build")
    print("\n".join(problems) or "index ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

"""``pinneapple studio`` / ``python -m pinneapple_studio``: the catalog, the gallery and the local Studio server.

    pinneapple studio                         # serve on http://127.0.0.1:8765 (Flow Studio UI + /api/studio)
    pinneapple studio summary                 # tools per category and status, gallery per state
    pinneapple studio search "dissipador de calor" --category executable
    pinneapple studio tool pinneapple_presets.cases.water_hammer:WaterHammer
    pinneapple studio gallery --state blocked
    pinneapple studio run-tool <tool id> --inputs '{"x": 1}'
    pinneapple studio run example:lbm --quick
    pinneapple studio inventory -o inventory.json
"""
from __future__ import annotations

import argparse
import json
import sys


def _print(obj) -> None:
    print(json.dumps(obj, indent=1, ensure_ascii=False, default=str))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pinneapple studio", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="the local Studio server (default)")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.add_argument("--no-browser", action="store_true")
    s.add_argument("--insecure-no-auth", action="store_true")
    sub.add_parser("summary", help="tools per category / status, gallery per state").add_argument("--rebuild",
                                                                                                    action="store_true")
    s = sub.add_parser("search", help="search the tools (Portuguese or English)")
    s.add_argument("query", nargs="?", default="")
    for f in ("category", "status", "task", "area", "product", "package", "kind"):
        s.add_argument(f"--{f}")
    s.add_argument("--limit", type=int, default=20)
    sub.add_parser("tool", help="one tool's card").add_argument("id")
    s = sub.add_parser("gallery", help="examples, reproductions and apps")
    s.add_argument("--kind")
    s.add_argument("--state")
    s = sub.add_parser("run-tool", help="run an executable tool as a job and wait")
    s.add_argument("id")
    s.add_argument("--inputs", default="{}")
    s = sub.add_parser("run", help="run a gallery entry as a job and wait")
    s.add_argument("entry")
    s.add_argument("--full", action="store_true", help="the full configuration (default: quick)")
    s.add_argument("--script")
    s = sub.add_parser("smoke", help="smoke-run executable tools and record the outcome")
    s.add_argument("--all", action="store_true", help="also tools whose current code already has a run")
    s.add_argument("--package")
    s.add_argument("--id", action="append")
    s.add_argument("--workers", type=int, default=4)
    sub.add_parser("inputs", help="harvest example inputs from tests, examples and Hub snippets into tool_inputs.json")
    s = sub.add_parser("examples", help="run the scripts of examples without a flow and record the outcome")
    s.add_argument("--id", action="append")
    s.add_argument("--timeout", type=float, default=900)
    s = sub.add_parser("coverage", help="coverage tables; exit 1 on a problem")
    s.add_argument("-o", "--out", default="docs/studio/COVERAGE.md")
    s.add_argument("--write-baseline", action="store_true", help="once, when the gate is introduced")
    s.add_argument("--record-history", action="store_true", help="append today's counts to the baseline curve")
    s = sub.add_parser("inventory", help="write the static inventory")
    s.add_argument("-o", "--out", default="-")
    a = ap.parse_args(argv)

    if a.cmd in (None, "serve"):
        from pinneapple_flow.studio import serve

        from .api import router
        return serve(getattr(a, "host", None), getattr(a, "port", None),
                     open_browser=not getattr(a, "no_browser", False),
                     allow_insecure=getattr(a, "insecure_no_auth", False), routers=[("/api/studio", router())],
                     name="PINNeAPPle Studio")
    if a.cmd == "smoke":
        from . import catalog, smoke
        _print(smoke.record(catalog.load(), stale_only=not a.all, package=a.package, ids=a.id, workers=a.workers,
                            log=lambda m: print(m, flush=True)))
        return 0
    if a.cmd == "inputs":
        from . import catalog, inputs
        _print(inputs.write(catalog.load()))
        return 0
    if a.cmd == "examples":
        from . import coverage, gallery
        coverage.record_examples(gallery.load(), ids=a.id, timeout=a.timeout)
        return 0
    if a.cmd == "coverage":
        from pathlib import Path

        from . import catalog, coverage, gallery
        cat, gal = catalog.load(), gallery.load()
        if a.write_baseline:
            print(f"baseline: {coverage.write_baseline(cat, gal)} lines", file=sys.stderr)
        if a.record_history:
            print(f"history: {coverage.record_history(cat, gal)}", file=sys.stderr)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(coverage.markdown(cat, gal), encoding="utf-8")
        _print(coverage.summary(cat, gal))
        bad = coverage.problems(cat, gal)
        for p in bad[:200]:
            print(f"problem: {p}", file=sys.stderr)
        if len(bad) > 200:
            print(f"... and {len(bad) - 200} more", file=sys.stderr)
        return 1 if bad else 0
    if a.cmd == "inventory":
        from .inventory import build
        text = json.dumps(build(), ensure_ascii=False, indent=1)
        if a.out == "-":
            print(text)
        else:
            open(a.out, "w", encoding="utf-8").write(text)
        return 0
    if a.cmd == "summary":
        from . import catalog, gallery
        c = catalog.load(rebuild=a.rebuild)
        _print({"commit": c["commit"], "tools": c["counts"], "gallery": gallery.summary(gallery.load(rebuild=a.rebuild)),
                "prose_unmatched": {k: len(v) for k, v in c["prose_unmatched"].items()},
                "prose_missing": {k: len(v) for k, v in c["prose_missing"].items()}})
        return 0
    if a.cmd == "search":
        from .catalog import load
        from .search import Index
        hits = Index(load()).search(a.query, category=a.category, status=a.status, task=a.task, area=a.area,
                                    product=a.product, package=a.package, kind=a.kind, limit=a.limit)
        for h in hits:
            print(f"{h['score']:7.2f}  {h['category']:<10} {h['status']:<10} {h['id']}")
            if h["pt"] or h["doc1"]:
                print(f"         {(h['pt'] or h['doc1'])[:150]}")
        return 0
    if a.cmd == "tool":
        from .catalog import load
        t = load()["tools"].get(a.id)
        if t is None:
            print(f"unknown tool {a.id}", file=sys.stderr)
            return 2
        _print(t)
        return 0
    if a.cmd == "gallery":
        from .gallery import load
        for e in load():
            if (a.kind and e["kind"] != a.kind) or (a.state and e["state"] != a.state):
                continue
            why = "; ".join(f"{r['name']}: {r['fix']}" for r in e["blocked_by"])
            print(f"{e['state']:<8} {e['id']:<45} {e['title'][:60]}" + (f"\n         needs {why}" if why else ""))
        return 0
    if a.cmd in ("run-tool", "run"):
        from .runner import queue
        q = queue()
        if a.cmd == "run-tool":
            job = q.submit("tool", {"tool": a.id, "inputs": json.loads(a.inputs)})
        else:
            params = {"entry": a.entry, "quick": not a.full}
            if a.script:
                params["script"] = a.script
            job = q.submit("gallery", params)
        print(f"job {job}", file=sys.stderr)
        res = q.wait(job, timeout=7200)
        _print(res)
        return 0 if res["status"] == "done" else 1
    ap.print_help()
    return 2

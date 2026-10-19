"""Command line: ``python -m pinneapple_flow <command>``.

    run flow.json [--workdir DIR] [--from NODE] [--only NODE ...] [--json OUT] [--no-cache]
    plan "description" [--backend heuristic|llm|auto] [--provider ...] [--model ...] [--base-url ...] [-o flow.json]
    validate flow.json
    catalog [--json]
    schema
    examples                                     # the bundled example flows
    coverage [-o FILE]                           # which library modules are nodes (markdown)
    autocalls [--no-check]                       # regenerate the nodes generated from the public API
    studio [--host 127.0.0.1] [--port 8765] [--no-browser] [--env .env]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _print_states(run_id: str, states: dict) -> None:
    line = "  ".join(f"{k}:{v['status']}" for k, v in states.items())
    print(f"\r[{run_id}] {line}"[:240], end="", file=sys.stderr, flush=True)


def cmd_run(a) -> int:
    from .runtime import run
    from .service import FLOWS_DIR
    path = Path(a.flow)
    if not path.exists() and (FLOWS_DIR / f"{a.flow}.json").exists():
        path = FLOWS_DIR / f"{a.flow}.json"
    res = run(path, workdir=a.workdir, from_node=a.from_node, only=a.only or None, cache=not a.no_cache,
              on_update=None if a.quiet else _print_states)
    print(file=sys.stderr)
    for nid, s in res.nodes.items():
        mets = s["report"].get("metrics") or {}
        short = ", ".join(f"{k}={_short(v)}" for k, v in list(mets.items())[:6])
        print(f"{nid:<14} {s['status']:<9} {(s['seconds'] or 0):7.2f}s  {short}")
        if s["status"] == "failed":
            print(f"    error: {s['error']}")
    print(f"run {res.run_id}: {res.status} in {res.seconds:.1f} s (folder {res.workdir})")
    if a.json:
        Path(a.json).write_text(json.dumps(res.to_dict(), indent=1, default=str))
    return 0 if res.status == "done" else 1


def _short(v):
    if isinstance(v, dict) and "value" in v:
        v = v["value"]
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)[:40]


def cmd_plan(a) -> int:
    from .planner import plan
    res = plan(a.text, backend=a.backend, provider=a.provider or "", model=a.model or "", base_url=a.base_url or "")
    print(f"planner: {res['planner']}" + (f" (fallback from {res['fallback_from']})" if res.get("fallback_from")
                                          else ""))
    for w in res.get("warnings", []):
        print(f"warning: {w}")
    for i, p in enumerate(res["proposals"]):
        print(f"[{i}] {p['title']} ({len(p['flow']['nodes'])} nodes): {p['rationale']}")
    if a.output:
        k = min(a.pick, len(res["proposals"]) - 1)
        Path(a.output).write_text(json.dumps(res["proposals"][k]["flow"], indent=1))
        print(f"wrote proposal {k} to {a.output}")
    return 0


def cmd_validate(a) -> int:
    from .schema import validate_flow
    issues = validate_flow(json.loads(Path(a.flow).read_text()))
    for i in issues:
        print(f"{i.level}: {i.node or i.edge or 'flow'}: {i.message}")
    print("valid" if not any(i.level == "error" for i in issues) else "invalid")
    return 1 if any(i.level == "error" for i in issues) else 0


def cmd_catalog(a) -> int:
    from .registry import REGISTRY
    if a.json:
        print(json.dumps(REGISTRY.catalog(), indent=1))
        return 0
    for s in REGISTRY.specs():
        ok, why = s.available()
        print(f"{s.type:<20} {s.category:<14} {s.title}{'' if ok else '  [unavailable: ' + why + ']'}")
    return 0


def cmd_examples(a) -> int:
    from .service import FLOWS_DIR
    for p in sorted(FLOWS_DIR.glob("*.json")):
        d = json.loads(p.read_text())
        print(f"{p.stem:<28} {d.get('name', '')}")
    return 0


def cmd_coverage(a) -> int:
    from .coverage import markdown
    text = markdown()
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(text)
        print(f"wrote {a.output}")
    else:
        print(text)
    return 0


def cmd_autocalls(a) -> int:
    from .autocalls import generate, prune, smoke, write
    data = generate()
    if not a.no_check:
        smoke(data["calls"])
        prune(data)
    write(data)
    ok = sum(c.get("check", "").startswith("runs") for c in data["calls"])
    print(f"{len(data['calls'])} nodes from the public API ({ok} run with their defaults), "
          f"{len(data['reasons'])} modules with a recorded reason -> pinneapple_flow/autocalls.json")
    return 0


def cmd_references(a) -> int:
    from .references import registry, verify_registry
    if not a.verify:
        for k, v in registry().items():
            print(f"{k:<26} {v.get('year')}  {v.get('title', '')[:80]}")
        return 0
    res = verify_registry()
    for k, v in res.items():
        tm = "" if v.get("title_match") is None else ("  title ok" if v["title_match"] else
                                                      f"  title differs: {v.get('registered_title')!r}")
        print(f"{k:<26} {v['status']:<11} {v.get('source', '')}{tm}")
    bad = [k for k, v in res.items() if v["status"] == "not found" or v.get("title_match") is False]
    print(f"{len(res)} DOIs: {sum(v['status'] == 'verified' for v in res.values())} verified, "
          f"{sum(v['status'] == 'unverified' for v in res.values())} not checkable here, {len(bad)} problems")
    return 1 if bad else 0


def cmd_studio(a) -> int:
    from .studio import serve
    return serve(host=a.host, port=a.port, open_browser=not a.no_browser, env_file=a.env,
                 allow_insecure=a.insecure_no_auth)


def main(argv=None) -> int:
    from .config import load_dotenv
    p = argparse.ArgumentParser(prog="python -m pinneapple_flow", description="PINNeAPPle Flow")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a flow headless")
    r.add_argument("flow")
    r.add_argument("--workdir")
    r.add_argument("--from", dest="from_node")
    r.add_argument("--only", nargs="*")
    r.add_argument("--json")
    r.add_argument("--no-cache", action="store_true")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(fn=cmd_run)
    pl = sub.add_parser("plan", help="proposals for a description")
    pl.add_argument("text")
    pl.add_argument("--backend", default="auto", choices=["auto", "heuristic", "llm"])
    pl.add_argument("--provider")
    pl.add_argument("--model")
    pl.add_argument("--base-url")
    pl.add_argument("-o", "--output")
    pl.add_argument("--pick", type=int, default=0)
    pl.set_defaults(fn=cmd_plan)
    v = sub.add_parser("validate")
    v.add_argument("flow")
    v.set_defaults(fn=cmd_validate)
    c = sub.add_parser("catalog")
    c.add_argument("--json", action="store_true")
    c.set_defaults(fn=cmd_catalog)
    s = sub.add_parser("schema")
    s.set_defaults(fn=lambda a: print(json.dumps(__import__("pinneapple_flow").flow_json_schema(), indent=1)) or 0)
    e = sub.add_parser("examples")
    e.set_defaults(fn=cmd_examples)
    cv = sub.add_parser("coverage", help="which library capabilities are nodes (markdown table)")
    cv.add_argument("-o", "--output")
    cv.set_defaults(fn=cmd_coverage)
    rf = sub.add_parser("references", help="the curated reference registry; --verify resolves the DOIs online")
    rf.add_argument("--verify", action="store_true")
    rf.set_defaults(fn=cmd_references)
    ac = sub.add_parser("autocalls", help="regenerate the nodes generated from the public API (autocalls.json)")
    ac.add_argument("--no-check", action="store_true", help="skip calling each function with its defaults")
    ac.set_defaults(fn=cmd_autocalls)
    st = sub.add_parser("studio", help="serve Flow Studio (backend + built frontend)")
    st.add_argument("--host")
    st.add_argument("--port", type=int)
    st.add_argument("--no-browser", action="store_true")
    st.add_argument("--env", default=".env")
    st.add_argument("--insecure-no-auth", action="store_true",
                    help="allow a non-localhost bind without FLOW_USER / FLOW_PASSWORD_HASH (not recommended)")
    st.set_defaults(fn=cmd_studio)
    a = p.parse_args(argv)
    load_dotenv(getattr(a, "env", ".env") or ".env")
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    sys.exit(main())

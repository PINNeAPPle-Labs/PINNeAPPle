"""Command line: ``python -m pinneapple_unity {serve, jobs, verify, ingest, link}``.

    python -m pinneapple_unity serve  --port 8766           # the live bridge Unity calls while playing
    python -m pinneapple_unity jobs                         # the registered physics jobs
    python -m pinneapple_unity verify out/bend_unity        # re-read an exported twin folder
    python -m pinneapple_unity ingest out/gripper_unity --out out/gripper_twin   # Unity -> twin + OpenUSD
    python -m pinneapple_unity link   out/gripper_unity     # print the pinneapple_link.json
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _cmd_serve(a: argparse.Namespace) -> int:
    from .bridge import serve_bridge

    srv = serve_bridge(port=a.port, host=a.host, block=False, token=a.token, allow_folders=not a.no_folders)
    print(f"PINNeAPPle <-> Unity bridge: {srv.url}")
    print(f"jobs: {', '.join(sorted(__import__('pinneapple_unity').list_jobs()))}")
    print("Ctrl+C to stop")
    try:
        import time

        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()
    return 0


def _cmd_jobs(a: argparse.Namespace) -> int:
    from .jobs import list_jobs

    jobs = list_jobs()
    if a.json:
        print(json.dumps(jobs, indent=1))
        return 0
    for name, doc in sorted(jobs.items()):
        print(f"{name}\n    {doc}")
    return 0


def _cmd_verify(a: argparse.Namespace) -> int:
    from .export import verify_unity_folder

    problems = verify_unity_folder(a.folder)
    for p in problems:
        print(p)
    print(f"{len(problems)} problem(s)" if problems else f"{a.folder}: every file checks out")
    return 1 if problems else 0


def _cmd_ingest(a: argparse.Namespace) -> int:
    from .ingest import unity_to_usd

    out: dict[str, Any] = unity_to_usd(a.folder, a.out, usd=not a.no_usd)
    for rep in out["reports"]:
        print(f"{rep['job']} on {rep['part']}: {rep['decision']}")
        for c in rep.get("checks", []):
            mark = {True: "PASS", False: "FAIL", None: "N/A "}[c.get("passed")]
            print(f"   {mark} {c['name']}")
        if rep.get("error"):
            print(f"   error: {rep['error']}")
    print("twin:", out["manifest"])
    if out["usd"]:
        print("usd: ", out["usd"])
    print("decision:", out["decision"])
    return 0 if out["decision"] == "APPROVED" else 1


def _cmd_link(a: argparse.Namespace) -> int:
    from .ingest import read_link

    print(json.dumps(read_link(a.folder), indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pinneapple_unity",
                                 description="PINNeAPPle <-> Unity: twin export, physics on a Unity scene, live bridge")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="start the HTTP bridge the Unity runtime calls")
    s.add_argument("--port", type=int, default=8766)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--token", default=None, help="require X-Pinneapple-Token (or set PINNEAPPLE_UNITY_TOKEN)")
    s.add_argument("--no-folders", action="store_true", help="refuse requests that name a folder on this machine")
    s.set_defaults(fn=_cmd_serve)

    s = sub.add_parser("jobs", help="list the registered physics jobs")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=_cmd_jobs)

    s = sub.add_parser("verify", help="check an exported twin folder (GLB, field offsets, package)")
    s.add_argument("folder")
    s.set_defaults(fn=_cmd_verify)

    s = sub.add_parser("ingest", help="Unity folder -> twin folder (+ OpenUSD), running the link's jobs")
    s.add_argument("folder")
    s.add_argument("--out", required=True)
    s.add_argument("--no-usd", action="store_true")
    s.set_defaults(fn=_cmd_ingest)

    s = sub.add_parser("link", help="print the pinneapple_link.json of a Unity export")
    s.add_argument("folder")
    s.set_defaults(fn=_cmd_link)

    a = ap.parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    sys.exit(main())

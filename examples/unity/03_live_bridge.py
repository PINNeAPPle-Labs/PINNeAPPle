"""The live lane: Unity asks PINNeAPPle for a field while it plays, and gets the checks with it.

    python examples/unity/03_live_bridge.py

Starts the bridge (`pinneapple_unity.serve_bridge`, the same server `python -m pinneapple_unity serve` runs), then
behaves like the Unity component `PinneappleBridgeClient` does:

    1. GET  /v1/health                      which jobs this machine serves
    2. POST /v1/run    (geometry inline)    the thermal transient of the spool from demo 01, as the C# client sends it
    3. POST /v1/scene  (folder, save)       a whole twin from disk, every field, every check, saved back as a
                                            Unity project folder
    4. POST /v1/run    (overloaded bracket) the same job twice, one of them past yield -> REJECT, and it says why

The point is not that a number arrives in Unity: it is that the number arrives with its boundary conditions, its
residual and a decision, so a game can refuse to show a field the physics does not support.

One warm-up request is paid at start-up (~20 s on this machine: it imports torch). After that a job is milliseconds.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from pinneapple_unity import decode_field, serve_bridge

HERE = os.path.dirname(os.path.abspath(__file__))
SPOOL_TWIN = os.path.join(HERE, "_out", "spool_unity")       # from 01_engineering_twin_to_unity.py
TIMES = [0.0, 30.0, 120.0, 600.0, 1800.0, 3600.0]


def show_checks(checks, indent: str = "   ") -> None:
    for c in checks:
        mark = {True: "PASS", False: "FAIL", None: "n/a "}[c["passed"]]
        value = "" if c.get("value") is None else f"  value {c['value']:.4g}"
        detail = f"   {c['detail']}" if c.get("detail") and c["passed"] is False else ""
        print(f"{indent}{mark}  {c['name']}{value}{detail}")


def main() -> None:
    t0 = time.perf_counter()
    srv = serve_bridge(port=0, block=False)                  # port 0: the OS picks a free one
    print(f"bridge up at {srv.url}  (start-up including the warm-up: {time.perf_counter() - t0:.1f} s)")
    print("Unity: point the PinneappleBridgeClient at that URL, or run `python -m pinneapple_unity serve --port 8766`")

    # 1. what this machine serves
    health = srv.post("/v1/health", {})
    print("\n/v1/health:", json.dumps(health, indent=None)[:160], "...")
    print("  jobs:", ", ".join(health["jobs"]))

    # 2. a run with the geometry inline -- exactly the body PinneappleBridgeClient builds
    vertices = [[float(x), 0.0, 0.0] for x in np.linspace(0.0, 0.4, 41)]
    body = {"protocol": "pinneapple-unity/1", "job": "slab_transient", "part": "spool", "up_axis": "z",
            "times": TIMES, "vertices": vertices,
            "params": {"alpha": 4e-6, "t_left": 180.0, "t_right": 30.0, "t0": 30.0, "axis": "x", "length": 0.4}}
    t = time.perf_counter()
    run = srv.post("/v1/run", body)
    print(f"\n/v1/run  {run['job']} on {run['part']}: {run['decision']}   "
          f"({(time.perf_counter() - t) * 1000:.0f} ms round trip, {run['seconds'] * 1000:.0f} ms of job)")
    show_checks(run["checks"])
    temp = decode_field(run["fields"]["temperature"])
    print(f"   temperature [{run['fields']['temperature']['unit']}]: {temp.shape[0]} steps x {temp.shape[1]} "
          f"vertices, hot flange {temp[:, 0].round(1).tolist()} ... cold end {temp[:, -1].round(1).tolist()}")

    # 3. a whole twin from disk: the folder demo 01 exported (run it first), with a job run again on it
    if os.path.exists(SPOOL_TWIN):
        t = time.perf_counter()
        scene = srv.post("/v1/scene", {
            "folder": SPOOL_TWIN,
            "jobs": [{"job": "slab_transient", "part": "spool",
                      "params": {"alpha": 2e-5, "t_left": 180.0, "t_right": 30.0, "t0": 30.0,
                                 "axis": "x", "length": 0.4, "sensor": "TC-01"}}],
            "save": os.path.join(HERE, "_out", "bridge_twin")})     # write the result where Unity can reopen it
        print(f"\n/v1/scene  {scene['title']}: {scene['decision']}  ({scene['seconds'] * 1000:.0f} ms)")
        for rep in scene["reports"]:
            print(f"   {rep['job']} on {rep['part']}: {rep['decision']}, {len(rep['checks'])} check(s)")
        for part, fields in scene["parts"].items():
            print(f"   {part}: {sorted(fields)}")
        print("   sensors:", [(s["id"], None if s["series"] is None else [round(v, 1) for v in s["series"]])
                              for s in scene["sensors"]])
        twin = scene.get("twin")
        if twin:
            print(f"   saved: {os.path.relpath(twin['manifest'], HERE)} ({twin['problems'] or 'every file checks out'})")
    else:
        print(f"\n(skipping /v1/scene: run 01_engineering_twin_to_unity.py first, it writes {SPOOL_TWIN})")

    # 4. a design that does not hold: the same bracket, twice the load, past yield
    print("\nthe same bracket at two loads (/v1/run, cantilever_stress)")
    section = [[float(x), 0.1, 0.0] for x in np.linspace(0.0, 0.25, 41)]
    for load in (40.0, 1000.0):
        t = time.perf_counter()
        res = srv.post("/v1/run", {"job": "cantilever_stress", "part": "bracket", "up_axis": "z",
                                   "times": [0.0, 3600.0], "vertices": section,
                                   "params": {"width": 0.05, "thickness": 0.01, "load": load, "length": 0.25,
                                              "axis": "x", "yield_mpa": 250.0}})
        peak = float(np.frombuffer(__import__("base64").b64decode(res["fields"]["bending_stress"]["data"]),
                                   dtype="<f4").max())
        print(f"  load {load:6.0f} N -> peak {peak:6.1f} MPa: {res['decision']}  "
              f"({(time.perf_counter() - t) * 1000:.0f} ms)")
        show_checks([c for c in res["checks"] if c["passed"] is False] or res["checks"][-1:], indent="     ")

    print(f"\n{srv.requests} requests served, {len(srv.errors)} error(s)")
    print("""
Try it by hand
  python -m pinneapple_unity serve --port 8766
  curl -s localhost:8766/v1/health | python -m json.tool
  curl -s -X POST localhost:8766/v1/run -H 'Content-Type: application/json' \\
       -d '{"job":"slab_transient","part":"bar","up_axis":"z","times":[0,3600],
            "vertices":[[0,0,0],[0.2,0,0]],"params":{"alpha":1e-5,"t_left":120,"t_right":20,"t0":20}}'

  serve --token ... requires X-Pinneapple-Token; the bridge binds 127.0.0.1 by default because it runs physics
  on demand -- opening it to a network is a decision, not a default.""")


if __name__ == "__main__":
    main()

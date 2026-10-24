"""Recorded runs of the bundled flows (``pinneapple_flow/flows``), shown in the Hub on each flow's page and on the
page of every node type the flow uses.

    python -m pinneapple_hub.library flowruns [names ...]    # run them headless, write data/web/hub/flow_runs.json
    from pinneapple_hub.library.flowruns import read_flow_runs, runs_for_node
    runs_for_node(read_flow_runs(), "geometry.mesh")          # [{flow, node, status, metrics, summary}, ...]

Each flow runs in the sandbox (:func:`pinneapple_hub.library.sandbox.run_snippet`, its own interpreter and time limit)
with ``pinneapple_flow.run``; what is kept per node is its status and the metrics and summary its report holds. A run
is current while the flow file has the SHA-256 it was recorded with; a flow that fails or times out is recorded as
such, with the error, and its nodes show nothing.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .sandbox import run_snippet

ROOT = Path(__file__).resolve().parents[2]
FLOW_RUNS = ROOT / "data" / "web" / "hub" / "flow_runs.json"
MARK = "@@FLOWRUN@@"

_CODE = '''import json, time
import pinneapple_flow as pf
from pinneapple_flow.service import FLOWS_DIR

flow = pf.Flow.load(FLOWS_DIR / "{name}.json")
t0 = time.perf_counter()
res = pf.run(flow, workdir="work")
out = {{"status": res.status, "seconds": round(time.perf_counter() - t0, 2), "nodes": {{}}}}
for n in flow.nodes:
    rep = res.report(n.id)
    m = {{k: v for k, v in (rep.get("metrics") or {{}}).items()
          if isinstance(v, (int, float, str, bool)) and not (isinstance(v, str) and len(v) > 120)}}
    st = res.nodes.get(n.id) or {{}}
    out["nodes"][n.id] = {{"type": n.type, "status": st.get("status"), "error": (st.get("error") or "")[:300],
                         "metrics": dict(list(m.items())[:12]), "summary": (rep.get("summary") or "")[:300]}}
print("{mark}" + json.dumps(out, default=str))
'''


def flow_hash(name: str) -> str:
    from pinneapple_flow.service import FLOWS_DIR
    return hashlib.sha256((Path(FLOWS_DIR) / f"{name}.json").read_bytes()).hexdigest()[:16]


def read_flow_runs(path: Path = FLOW_RUNS) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def current(runs: dict[str, Any], name: str) -> dict[str, Any] | None:
    """The recorded run of ``name`` if it was recorded for the flow file as it is now."""
    r = runs.get(name)
    return r if r and r.get("flow_hash") == flow_hash(name) else None


def record(names: list[str] | None = None, *, timeout: float = 900.0, workers: int = 2, log=print) -> dict[str, Any]:
    """Run the bundled flows (all, or ``names``) and write ``flow_runs.json``."""
    from concurrent.futures import ThreadPoolExecutor

    from pinneapple_flow.service import FLOWS_DIR
    runs = read_flow_runs()
    todo = names or sorted(p.stem for p in Path(FLOWS_DIR).glob("*.json"))

    def one(name: str) -> None:
        r = run_snippet(_CODE.format(name=name, mark=MARK), timeout=timeout, max_figures=0)
        line = next((ln for ln in r["stdout"].splitlines()[::-1] if ln.startswith(MARK)), "")
        rec: dict[str, Any] = {"flow_hash": flow_hash(name), "recorded": time.strftime("%Y-%m-%d", time.gmtime()),
                               "source": "python -m pinneapple_hub.library flowruns"}
        if r["ok"] and line:
            rec.update(json.loads(line[len(MARK):]))
            if rec["status"] != "done":
                rec["error"] = next((n["error"] for n in rec["nodes"].values() if n.get("error")), "")
        else:
            rec.update({"status": "timed out" if r["timed_out"] else "failed", "seconds": r["seconds"], "nodes": {},
                        "error": (r["stderr"].splitlines() or [""])[-1][:300]})
        runs[name] = rec
        log(f"{rec['status']:>10} {rec.get('seconds', 0):7.1f}s {name}")

    with ThreadPoolExecutor(max(1, workers)) as pool:
        list(pool.map(one, todo))
    FLOW_RUNS.parent.mkdir(parents=True, exist_ok=True)
    FLOW_RUNS.write_text(json.dumps(runs, indent=1, sort_keys=True, default=str) + "\n")
    return runs


def runs_for_node(runs: dict[str, Any], type_: str) -> list[dict[str, Any]]:
    """Every node of ``type_`` in a current recorded flow run: flow, node id, status, metrics, summary."""
    out = []
    for name in sorted(runs):
        r = current(runs, name)
        for nid, n in ((r or {}).get("nodes") or {}).items():
            if n.get("type") == type_:
                out.append({"flow": name, "node": nid, "status": n.get("status"), "error": n.get("error", ""),
                            "metrics": n.get("metrics", {}),
                            "summary": n.get("summary", ""), "flow_status": r.get("status"),
                            "seconds": r.get("seconds")})
    return out


__all__ = ["record", "read_flow_runs", "current", "runs_for_node", "flow_hash", "FLOW_RUNS"]

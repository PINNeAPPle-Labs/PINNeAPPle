"""Every ``examples/*`` study as a flow: the inventory, the quick configurations, the measured comparisons.

* the flows are ``pinneapple_flow/flows/examples/<domain>/<name>.json``; ``meta.example`` names the example folder,
  ``meta.quick`` the parameter overrides of its quick configuration (``{node id: {param: value}}``);
* :data:`INVENTORY` (``flows/examples/inventory.json``) gives, per example folder, its status when a flow does not
  reproduce all of it (``needs`` a tool, ``not_convertible`` with the reason) and notes;
* :data:`MEASURED` (``flows/examples/measured.json``) is written by :func:`run_example`: the outcome of each flow run
  (full or quick), its seconds and the ``examples.compare`` rows, flow number against the example's committed one;
* :func:`scan` lists the library calls of an example's scripts and whether a node's declared calls cover them;
* :func:`markdown` renders ``docs/flow_studio/EXAMPLES_AS_FLOWS.md`` from these three; :func:`problems` is the CI
  gate (an example folder with neither a flow nor a stated reason).

    python -m pinneapple_flow example-flows list
    python -m pinneapple_flow example-flows run flight_dynamics/03_takeoff [--quick]
    python -m pinneapple_flow example-flows doc -o docs/flow_studio/EXAMPLES_AS_FLOWS.md
    python -m pinneapple_flow example-flows check
"""
from __future__ import annotations

import ast
import copy
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

FLOWS = Path(__file__).parent / "flows" / "examples"
INVENTORY_FILE = FLOWS / "inventory.json"
MEASURED_FILE = FLOWS / "measured.json"
ROOT = Path(__file__).resolve().parents[1]
NOT_EXAMPLES = re.compile(r"^(_.*|scratch|results.*|.*_out|.*_runs.*)$")
STATUSES = ("runs", "runs-quick", "needs", "not_convertible")


def _load(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text()) if p.is_file() else {}


def inventory() -> dict[str, Any]:
    return _load(INVENTORY_FILE)


def measured() -> dict[str, Any]:
    return _load(MEASURED_FILE)


def example_folders(root: Path | None = None) -> list[str]:
    """The ``examples/*`` folders that are examples (scratch, ``_out``, ``_runs`` and results folders are not)."""
    base = (root or ROOT) / "examples"
    return sorted(p.name for p in base.iterdir() if p.is_dir() and not NOT_EXAMPLES.match(p.name)
                  and not p.name.startswith("."))


def flows() -> list[dict[str, Any]]:
    """Every example flow: ``{"name": "<domain dir>/<stem>", "path", "title", "description", "example", "domain",
    "nodes", "quick"}``."""
    out = []
    for p in sorted(FLOWS.glob("*/*.json")):
        d = json.loads(p.read_text())
        m = d.get("meta") or {}
        out.append({"name": f"{p.parent.name}/{p.stem}", "path": str(p), "title": d.get("name", p.stem),
                    "description": d.get("description", ""), "example": m.get("example", p.parent.name),
                    "domain": m.get("domain", p.parent.name), "scripts": m.get("scripts", []),
                    "nodes": len(d.get("nodes", [])), "quick": bool(m.get("quick"))})
    return out


def flow_path(name: str) -> Path:
    p = (FLOWS / f"{name}.json").resolve()
    if FLOWS.resolve() not in p.parents or not p.is_file():
        raise KeyError(f"unknown example flow {name!r}")
    return p


def with_quick(doc: dict[str, Any]) -> dict[str, Any]:
    """The flow with its quick configuration applied (``meta.quick``), marked ``meta.configuration = "quick"``."""
    d = copy.deepcopy(doc)
    q = (d.get("meta") or {}).get("quick") or {}
    for n in d["nodes"]:
        n["params"] = {**(n.get("params") or {}), **q.get(n["id"], {})}
    d["meta"]["configuration"] = "quick"
    d["id"] = f"{d['id']}-quick"
    d["name"] = f"{d['name']} [quick]"
    return d


def _git() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001 - not a checkout
        return ""


def run_example(name: str, *, quick: bool = False, workdir: str | Path | None = None, record: bool = True,
                on_update=None) -> dict[str, Any]:
    """Run an example flow (full or quick) and, with ``record``, store its outcome and comparisons in MEASURED."""
    from .runtime import run
    doc = json.loads(flow_path(name).read_text())
    if quick:
        doc = with_quick(doc)
    t0 = time.time()
    res = run(doc, workdir=workdir, on_update=on_update)
    rec: dict[str, Any] = {"configuration": "quick" if quick else "full", "status": res.status,
                           "seconds": round(time.time() - t0, 1), "commit": _git(),
                           "date": time.strftime("%Y-%m-%d"), "nodes": {}, "comparisons": [],
                           "manifest": res.hooks.get("register_example_run", "")}
    for nid, st in res.nodes.items():
        rep = st.get("report") or {}
        rec["nodes"][nid] = st["status"] + (f": {rep.get('not_run_reason')}" if st["status"] == "not_run" else "") \
            + (f": {st.get('error')}" if st["status"] == "failed" else "")
        n = next(x for x in doc["nodes"] if x["id"] == nid)
        if n["type"] == "examples.compare" and st["status"] in ("done", "cached"):
            for t in rep.get("tables") or []:
                for r in t["rows"]:
                    rec["comparisons"].append(dict(zip(("flow", "ref", "value", "reference", "rel_diff", "outcome"),
                                                       r, strict=True)))
    rec["matched"] = sum(c["outcome"] == "match" for c in rec["comparisons"])
    if record:
        m = measured()
        m.setdefault(name, {})[rec["configuration"]] = rec
        MEASURED_FILE.write_text(json.dumps(m, indent=1, sort_keys=True, default=str) + "\n")
    rec["run_id"], rec["workdir"] = res.run_id, res.workdir
    return rec


# -------------------------------------------------------------------------------------------- after a run
def register_example_run(result, flow) -> str:
    """After-run hook (:data:`pinneapple_flow.runtime.RUN_FINISHED_HOOKS`): a finished run of an example flow is
    written as its own experiment manifest (``<workdir>/<flow id>.experiment.json``, id ``flow/<flow id>``, the domain
    and the example folder it reproduces) and imported into the Flow Studio's Lab (``pinneapple_lab.capture.
    register_manifest``), which grades it A-F, so it shows on the experiments page with its seal."""
    meta = flow.meta or {}
    if not meta.get("example"):
        return "skipped: not an example flow"
    if result.status != "done":
        return f"skipped: the run is {result.status}"
    from pinneapple_lab import manifest as M
    from pinneapple_lab.capture import register_manifest
    from pinneapple_lab.store import LabStore

    from .config import home
    from .runtime import _flow_manifest_run
    run = _flow_manifest_run(result, flow)
    run.update(id=f"flow/{flow.id}", domain=meta.get("domain", ""))
    run["links"].update(example=f"examples/{meta['example']}", configuration=meta.get("configuration", "full"),
                        scripts=list(meta.get("scripts") or []))
    m = M.from_flow_run(run)
    m.references = run["references"]
    m.status = "curated"
    path = M.save(m, str(Path(result.workdir) / f"{flow.id}.experiment.json"))
    problems_ = M.validate(m, base=result.workdir)
    if problems_:
        return f"manifest {path}, not registered: {problems_[:3]}"
    item = register_manifest(path, LabStore(str(home() / "lab")))
    return f"manifest {path}, Lab item {item}"


def register_hook() -> None:
    from .runtime import RUN_FINISHED_HOOKS
    if register_example_run not in RUN_FINISHED_HOOKS:
        RUN_FINISHED_HOOKS.append(register_example_run)


# -------------------------------------------------------------------------------------------- library calls
def _calls_of(src: str) -> tuple[set[str], set[str]]:
    """(``pinneapple*`` modules imported, ``module.name`` of the names imported from them) of a script."""
    mods, names = set(), set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return mods, names
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.split(".")[0].startswith("pinneapple"):
            mods.add(n.module)
            names |= {f"{n.module}.{a.name}" for a in n.names}
        elif isinstance(n, ast.Import):
            mods |= {a.name for a in n.names if a.name.split(".")[0].startswith("pinneapple")}
    pp_calls = {f"pp.{n.attr}" for n in ast.walk(tree) if isinstance(n, ast.Attribute)
                and isinstance(n.value, ast.Name) and n.value.id == "pp"}
    return mods, names | pp_calls


def scan(example: str) -> dict[str, Any]:
    """Per script of ``examples/<example>``: the library calls it imports and which of them a node declares."""
    from .registry import REGISTRY
    declared = " ".join(s.calls for s in REGISTRY.specs())
    out = {}
    for p in sorted((ROOT / "examples" / example).rglob("*.py")):
        if any(NOT_EXAMPLES.match(part) for part in p.relative_to(ROOT / "examples" / example).parts[:-1]):
            continue
        mods, names = _calls_of(p.read_text(errors="replace"))
        covered = sorted(n for n in names if n.rsplit(".", 1)[-1] in declared)
        out[str(p.relative_to(ROOT / "examples" / example))] = {
            "modules": sorted(mods), "calls": sorted(names), "covered": covered,
            "uncovered": sorted(set(names) - set(covered))}
    return out


# -------------------------------------------------------------------------------------------- inventory
def status(example: str, inv: dict | None = None, meas: dict | None = None) -> dict[str, Any]:
    """The inventory row of an example: its flows, status, reason and the measured comparison."""
    inv = inventory() if inv is None else inv
    meas = measured() if meas is None else meas
    entry = dict(inv.get(example) or {})
    fl = [f for f in flows() if f["example"] == example]
    row = {"example": example, "flows": [f["name"] for f in fl], "domain": entry.get("domain")
           or (fl[0]["domain"] if fl else ""), "reason": entry.get("reason", ""), "notes": entry.get("notes", ""),
           "needs": entry.get("needs", "")}
    runs = {f["name"]: meas.get(f["name"], {}) for f in fl}
    full_ok = [n for n, r in runs.items() if r.get("full", {}).get("status") == "done"]
    quick_ok = [n for n, r in runs.items() if r.get("quick", {}).get("status") == "done"]
    if entry.get("status"):
        row["status"] = entry["status"]
    elif fl and len(full_ok) == len(fl):
        row["status"] = "runs"
    elif fl and len(set(full_ok) | set(quick_ok)) == len(fl):
        row["status"] = "runs-quick"
    elif fl:
        row["status"] = "not yet run"
    else:
        row["status"] = ""
    cmp_ = [c for r in runs.values() for c in (r.get("full") or r.get("quick") or {}).get("comparisons", [])]
    row["compared"], row["matched"] = len(cmp_), sum(c["outcome"] == "match" for c in cmp_)
    row["runs"] = runs
    return row


def problems() -> list[str]:
    """Example folders with neither a flow nor a stated reason, statuses outside :data:`STATUSES`, flows naming an
    example that does not exist."""
    inv, out = inventory(), []
    folders = set(example_folders())
    for ex in sorted(folders):
        e = inv.get(ex) or {}
        has_flow = any(f["example"] == ex for f in flows())
        if not has_flow and not e.get("reason"):
            out.append(f"examples/{ex}: no flow and no reason in {INVENTORY_FILE.name}")
        if e.get("status") and e["status"] not in STATUSES:
            out.append(f"examples/{ex}: status {e['status']!r} is not one of {STATUSES}")
        if e.get("status") in ("needs", "not_convertible") and not e.get("reason"):
            out.append(f"examples/{ex}: status {e['status']} needs a reason")
    out += [f"{f['name']}: names example {f['example']!r}, which is not an examples/* folder" for f in flows()
            if f["example"] not in folders]
    out += [f"{INVENTORY_FILE.name}: {ex} is not an examples/* folder" for ex in inv if ex not in folders]
    return out


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def markdown() -> str:
    """``docs/flow_studio/EXAMPLES_AS_FLOWS.md``."""
    inv, meas = inventory(), measured()
    rows = [status(ex, inv, meas) for ex in example_folders()]
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"] or "no flow"] = counts.get(r["status"] or "no flow", 0) + 1
    L = ["# Examples as flows", "",
         "Every `examples/*` folder, the flow that reproduces it in Flow Studio (*Open... -> Examples*) and the outcome "
         "of running that flow. Generated by `python -m pinneapple_flow example-flows doc` from "
         "`pinneapple_flow/flows/examples/` (the flows), `inventory.json` (statuses and reasons) and `measured.json` "
         "(written by `example-flows run`: every number below was measured by a run of the flow, none is typed in). "
         "`tests/test_flow_examples.py` fails when a folder has neither a flow nor a reason.", "",
         "Statuses: **runs** (the flow runs in its full configuration), **runs-quick** (it runs in its documented quick "
         "configuration, `meta.quick` of the flow; the full one is too long for this machine), **needs X** (a tool that "
         "is not installed here), **not_convertible** (the reason says why).", "",
         "| status | examples |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in sorted(counts.items())]
    L += ["", "| example | domain | flow(s) | status | same numbers | notes |", "|---|---|---|---|---|---|"]
    for r in rows:
        fl = "<br>".join(f"`{f}`" for f in r["flows"]) or "-"
        st = r["status"] + (f" {r['needs']}" if r["status"] == "needs" and r["needs"] else "")
        same = f"{r['matched']} / {r['compared']} match" if r["compared"] else ""
        note = "; ".join(x for x in (r["reason"], r["notes"]) if x)
        L.append(f"| `{r['example']}` | {r['domain']} | {fl} | {st} | {same} | {note} |")
    L += ["", "## Measured comparisons", "",
          "Per flow run: the flow's number, the example's committed number (results file or manifest metric) and the "
          "relative difference; the tolerance of each comparison is in the flow (`examples.compare` node).", ""]
    for r in rows:
        for name, runs in r["runs"].items():
            for conf in ("full", "quick"):
                rec = runs.get(conf)
                if not rec:
                    continue
                L += [f"### `{name}` ({conf})", "",
                      f"{rec['status']} in {rec['seconds']:.0f} s (commit {rec.get('commit') or '?'}, {rec['date']}); "
                      f"{rec['matched']} of {len(rec['comparisons'])} numbers match.", ""]
                bad = {k: v for k, v in rec["nodes"].items() if not v.startswith(("done", "cached"))}
                if bad:
                    L += ["Nodes not done: " + "; ".join(f"`{k}` {v}" for k, v in bad.items()), ""]
                if rec["comparisons"]:
                    L += ["| flow metric | example reference | flow | example | rel. diff | outcome |",
                          "|---|---|---|---|---|---|"]
                    L += [f"| `{c['flow']}` | `{c['ref']}` | {_fmt(c['value'])} | {_fmt(c['reference'])} | "
                          f"{_fmt(c['rel_diff'])} | {c['outcome']} |" for c in rec["comparisons"]]
                    L.append("")
    L += ["## Library calls per script", "",
          "The `pinneapple*` names each script imports and whether a node declares them (`calls` of the node types).",
          ""]
    for ex in example_folders():
        sc = scan(ex)
        if not sc:
            continue
        L += [f"<details><summary><code>{ex}</code></summary>", "", "| script | covered by a node | not covered |",
              "|---|---|---|"]
        for s, d in sc.items():
            L.append(f"| `{s}` | {', '.join(f'`{c}`' for c in d['covered']) or '-'} | "
                     f"{', '.join(f'`{c}`' for c in d['uncovered']) or '-'} |")
        L += ["", "</details>", ""]
    return "\n".join(L) + "\n"


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m pinneapple_flow example-flows")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    r = sub.add_parser("run")
    r.add_argument("names", nargs="+", help="<domain>/<stem>, or a domain directory for all its flows")
    r.add_argument("--quick", action="store_true")
    r.add_argument("--workdir")
    r.add_argument("--no-record", action="store_true")
    d = sub.add_parser("doc")
    d.add_argument("-o", "--out", default=str(ROOT / "docs" / "flow_studio" / "EXAMPLES_AS_FLOWS.md"))
    sub.add_parser("check")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for f in flows():
            print(f"{f['name']:<55} {f['nodes']:3d} nodes  {'quick' if f['quick'] else '     '}  {f['title']}")
        return 0
    if a.cmd == "run":
        names = []
        for n in a.names:
            names += [f["name"] for f in flows() if f["name"].startswith(n.rstrip("/") + "/")] or [n]
        rc = 0
        for n in names:
            rec = run_example(n, quick=a.quick, workdir=a.workdir, record=not a.no_record)
            print(f"{n}: {rec['status']} in {rec['seconds']:.0f} s, {rec['matched']} / {len(rec['comparisons'])} "
                  f"numbers match; {rec['manifest']}")
            for k, v in rec["nodes"].items():
                if not v.startswith(("done", "cached")):
                    print(f"   {k}: {v}")
            for c in rec["comparisons"]:
                if c["outcome"] != "match":
                    print(f"   differs: {c['flow']} = {c['value']} vs {c['reference']} ({c['ref']})")
            rc |= rec["status"] != "done"
        return rc
    if a.cmd == "doc":
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(markdown())
        print(a.out)
        return 0
    errs = problems()
    print("\n".join(errs) or "ok")
    return 1 if errs else 0

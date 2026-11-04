"""Gallery groups of the library's closed-loop flows: ``examples/<group>/<entry>/`` with an ``experiment.json`` of
measured runs (#1244), for apps that list ``examples/<group>/`` entries (the engineering app's gallery).

Each entry is one bundled flow (``pinneapple_flow/flows/<flow>.json``) with a **quick configuration** (parameter
overrides per node). :func:`write_group` runs every entry through :func:`pinneapple_flow.examples.run_example`
(full and / or quick: the run is recorded in ``flows/examples/measured.json`` too) and writes, from what the run
returned and nothing else:

* ``pinneapple_flow/flows/examples/<group>/<entry>.json``: the flow as an example flow (``meta.example`` = the
  group, ``meta.quick`` = the quick configuration), so Flow Studio's *Examples* and ``example-flows`` list it;
* ``examples/<group>/<entry>/results/result.json``: per configuration, the run's status, seconds, commit, date, every
  numeric node metric with its "how", and the rows of the nodes' ``checks`` tables;
* ``examples/<group>/<entry>/README.md`` and ``experiment.json`` (a :mod:`pinneapple_lab` manifest: the seconds and
  metrics of the runs, one check per recorded check row);
* ``examples/<group>/README.md`` and ``experiment.json``: the group, one check per entry (its run finished).

    python -m pinneapple_flow example-flows gallery industrial            # quick and full runs of every entry
    python -m pinneapple_flow example-flows gallery base_workflows --quick-only
    python -m pinneapple_flow example-flows gallery industrial --entries wing_design
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

from .examples import FLOWS, ROOT

BUNDLED = Path(__file__).parent / "flows"

GROUPS: dict[str, dict[str, Any]] = {
    "industrial": {
        "title": "Industrial closed-loop flows",
        "description": "Reactor sizing (cooled packed bed and CSTR), computational lithography (ILT mask correction), "
                       "wing planform design and the least-drag Ahmed body: each a model, an optimisation and the "
                       "design re-verified by the model, with the measured run times and results.",
        "domain": "engineering",
        "entries": {
            "reactor_packed_bed": {"flow": "reactor_sizing_packed_bed", "quick": {"design": {"maxiter": 20,
                                                                                            "popsize": 8}}},
            "reactor_cstr": {"flow": "reactor_sizing_cstr", "quick": {"design": {"maxiter": 8, "popsize": 6}}},
            "lithography_opc": {"flow": "lithography_opc", "quick": {}},
            "wing_design": {"flow": "wing_design", "quick": {"wing": {"maxiter": 12, "popsize": 8}}},
            "ahmed_design": {"flow": "ahmed_body_end_to_end", "quick": {"geometry": {"cad": False},
                                                                        "models": {"budget_s": 3.0}}},
        },
    },
    "base_workflows": {
        "title": "Base workflows",
        "description": "The library's method workflows as flows: surrogate modeling, a trained or zoo model used "
                       "instead of training, VeriPhysics for a model trained elsewhere, inverse modeling, equation "
                       "discovery, model benchmarking, design optimization, reduced-order modeling and multiphysics "
                       "coupling, each with its checks and the measured run times and results.",
        "domain": "methods",
        "entries": {
            "surrogate_modeling": {"flow": "surrogate_modeling", "quick": {}},
            "use_trained_model": {"flow": "use_trained_model", "quick": {}},
            "veriphysics_grade": {"flow": "veriphysics_grade", "quick": {"grade": {"steps": 400}}},
            "inverse_modeling": {"flow": "inverse_modeling", "quick": {"inverse": {"bootstrap": 20,
                                                                                   "n_recovery": 4}}},
            "equation_discovery": {"flow": "equation_discovery", "quick": {"discover": {"n_bootstrap": 20}}},
            "model_benchmarking": {"flow": "model_benchmarking", "quick": {"bench": {"n_test": 60}}},
            "design_optimization": {"flow": "design_optimization", "quick": {}},
            "reduced_order_modeling": {"flow": "reduced_order_modeling", "quick": {"rom": {"repeats": 1}}},
            "multiphysics_coupling": {"flow": "multiphysics_coupling", "quick": {}},
        },
    },
}


def example_flow(group: str, entry: str) -> dict[str, Any]:
    """The bundled flow of a gallery entry as an example flow (``meta.example``, ``meta.quick``)."""
    spec = GROUPS[group]["entries"][entry]
    doc = json.loads((BUNDLED / f"{spec['flow']}.json").read_text())
    ids = {n["id"] for n in doc["nodes"]}
    bad = {k for k in spec["quick"] if k not in ids}
    if bad:
        raise KeyError(f"{group}/{entry}: the quick configuration names nodes {sorted(bad)} the flow does not have")
    out = copy.deepcopy(doc)
    out["id"] = f"example.{group}.{entry}"
    meta = dict(out.get("meta") or {})
    meta.pop("decisions", None)
    meta.update(example=group, domain=GROUPS[group]["domain"], entry=entry, bundled=spec["flow"],
                quick=copy.deepcopy(spec["quick"]), scripts=[])
    out["meta"] = meta
    return out


def write_flows(group: str, entries: list[str] | None = None) -> list[Path]:
    out = []
    (FLOWS / group).mkdir(parents=True, exist_ok=True)
    for e in entries or list(GROUPS[group]["entries"]):
        p = FLOWS / group / f"{e}.json"
        p.write_text(json.dumps(example_flow(group, e), indent=1) + "\n")
        out.append(p)
    return out


def _metrics(res) -> dict[str, dict[str, Any]]:
    """Every numeric node metric of a finished run: ``"<node>.<name>" -> {"value", "how"}``."""
    out = {}
    for nid, st in res.nodes.items():
        for k, v in ((st.get("report") or {}).get("metrics") or {}).items():
            how = ""
            if isinstance(v, dict):
                v, how = v.get("value"), v.get("how", "")
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            out[f"{nid}.{k}"] = {"value": v, "how": how}
    return out


def _checks(res) -> list[dict[str, Any]]:
    out = []
    for nid, st in res.nodes.items():
        for t in (st.get("report") or {}).get("tables") or []:
            cols = [c.lower() for c in t["columns"]]
            if "status" not in cols or not ({"check", "code"} & set(cols)):
                continue
            ci = cols.index("check") if "check" in cols else cols.index("code")
            si = cols.index("status")
            di = cols.index("detail") if "detail" in cols else None
            for r in t["rows"]:
                out.append({"node": nid, "check": str(r[ci]), "status": str(r[si]),
                            "detail": str(r[di]) if di is not None else ""})
    return out


def run_entry(group: str, entry: str, *, quick: bool, workdir: str | Path | None = None) -> dict[str, Any]:
    """One configuration of an entry, run and recorded (``measured.json``); the record of ``result.json``."""
    from .examples import run_example
    results: list[Any] = []
    rec = run_example(f"{group}/{entry}", quick=quick, workdir=workdir, results=results, cache=False)
    res = results[0]
    return {"configuration": rec["configuration"], "status": rec["status"], "seconds": rec["seconds"],
            "commit": rec["commit"], "date": rec["date"], "nodes": rec["nodes"], "metrics": _metrics(res),
            "checks": _checks(res)}


def _title(group: str, entry: str) -> tuple[str, str]:
    doc = json.loads((BUNDLED / f"{GROUPS[group]['entries'][entry]['flow']}.json").read_text())
    return doc.get("name", entry), doc.get("description", "")


def _fmt(v) -> str:
    return f"{v:.6g}" if isinstance(v, float) else str(v)


def write_entry(group: str, entry: str, runs: dict[str, dict[str, Any]]) -> Path:
    """``results/result.json``, ``README.md`` and ``experiment.json`` of one entry from its recorded runs."""
    from pinneapple_lab import experiment
    base = ROOT / "examples" / group / entry
    (base / "results").mkdir(parents=True, exist_ok=True)
    old = {}
    rp = base / "results" / "result.json"
    if rp.is_file():
        old = json.loads(rp.read_text()).get("runs", {})
    runs = {**old, **runs}
    spec = GROUPS[group]["entries"][entry]
    title, desc = _title(group, entry)
    data = {"group": group, "entry": entry, "flow": spec["flow"], "example_flow": f"{group}/{entry}",
            "quick": spec["quick"], "runs": runs}
    rp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    L = [f"# {title}", "", desc, "",
         f"Flow: `pinneapple_flow/flows/{spec['flow']}.json` (as the example flow `{group}/{entry}`). Run it with "
         f"`python -m pinneapple_flow example-flows run {group}/{entry} [--quick]`; regenerate this folder with "
         f"`python -m pinneapple_flow example-flows gallery {group} --entries {entry}`.", "",
         "Quick configuration (parameter overrides per node): `" + json.dumps(spec["quick"]) + "`.", "",
         "Every number below was measured by a run of the flow on the commit and date stated; none is typed in.", ""]
    for conf in ("quick", "full"):
        r = runs.get(conf)
        if not r:
            L += [f"## {conf}", "", "not run", ""]
            continue
        L += [f"## {conf}", "", f"{r['status']} in {r['seconds']:g} s (commit {r['commit']}, {r['date']}).", "",
              "| metric | value | how |", "|---|---|---|"]
        L += [f"| `{k}` | {_fmt(m['value'])} | {m['how']} |" for k, m in r["metrics"].items()]
        bad = {k: v for k, v in r["nodes"].items() if not v.startswith(("done", "cached"))}
        if bad:
            L += ["", "Nodes not done: " + "; ".join(f"`{k}` {v}" for k, v in bad.items())]
        if r["checks"]:
            L += ["", "| check | status | detail |", "|---|---|---|"]
            L += [f"| {c['check']} | {c['status']} | {c['detail'].replace('|', '/')} |" for c in r["checks"]]
        L.append("")
    (base / "README.md").write_text("\n".join(L))
    main = runs.get("full") if (runs.get("full") or {}).get("status") == "done" else runs.get("quick")
    with experiment(f"examples/{group}/{entry}", title=title, description=desc, domain=GROUPS[group]["domain"],
                    methods=[spec["flow"]], modules=["pinneapple_flow"], out_dir=str(base), register=False) as ex:
        ex.param(**{f"seconds_{c}": r["seconds"] for c, r in runs.items()})
        for c, r in runs.items():
            ex.log_metric(f"run_seconds_{c}", r["seconds"], "s", f"wall time of the {c} configuration, {r['date']}, "
                                                                   f"commit {r['commit']}")
        if main:
            for k, m in main["metrics"].items():
                ex.log_metric(k, m["value"], "", f"{main['configuration']} run: {m['how']}".strip(": "))
            for c in main["checks"]:
                if c["status"] in ("pass", "fail"):
                    ex.check(f"{c['node']}.{c['check']}", c["status"] == "pass", detail=c["detail"][:300])
        for c, r in runs.items():
            ex.check(f"run ({c})", r["status"] == "done", detail=f"{r['status']} in {r['seconds']:g} s")
        ex.add_file(str(rp), "data")
    return base


def write_group_index(group: str) -> Path:
    from pinneapple_lab import experiment
    g = GROUPS[group]
    base = ROOT / "examples" / group
    rows = []
    for e in g["entries"]:
        rp = base / e / "results" / "result.json"
        runs = json.loads(rp.read_text())["runs"] if rp.is_file() else {}
        rows.append((e, runs))
    L = [f"# {g['title']}", "", g["description"], "",
         f"Regenerate with `python -m pinneapple_flow example-flows gallery {group}` (runs every entry, quick and "
         "full, and rewrites these files from the runs).", "",
         "| entry | flow | quick | full |", "|---|---|---|---|"]
    for e, runs in rows:
        cells = [f"{runs[c]['status']}, {runs[c]['seconds']:g} s" if c in runs else "not run" for c in ("quick", "full")]
        L.append(f"| [{e}]({e}/README.md) | `{g['entries'][e]['flow']}` | {cells[0]} | {cells[1]} |")
    (base / "README.md").write_text("\n".join(L) + "\n")
    with experiment(f"examples/{group}", title=g["title"], description=g["description"], domain=g["domain"],
                    modules=["pinneapple_flow"], out_dir=str(base), register=False) as ex:
        for e, runs in rows:
            for c, r in runs.items():
                ex.check(f"{e}: {c} run", r["status"] == "done", detail=f"{r['seconds']:g} s, {r['date']}")
    return base


def write_group(group: str, entries: list[str] | None = None, *, quick: bool = True, full: bool = True,
                workdir: str | Path | None = None, log=print) -> list[Path]:
    """Write the example flows, run the entries (quick and / or full) and write the gallery files from the runs."""
    entries = entries or list(GROUPS[group]["entries"])
    write_flows(group, entries)
    out = []
    for e in entries:
        runs = {}
        for conf, on in (("quick", quick), ("full", full)):
            if not on:
                continue
            t0 = time.time()
            runs[conf] = run_entry(group, e, quick=conf == "quick", workdir=workdir)
            log(f"{group}/{e} [{conf}]: {runs[conf]['status']} in {time.time() - t0:.1f} s")
        out.append(write_entry(group, e, runs))
    out.append(write_group_index(group))
    return out


__all__ = ["GROUPS", "example_flow", "write_flows", "run_entry", "write_entry", "write_group_index", "write_group"]

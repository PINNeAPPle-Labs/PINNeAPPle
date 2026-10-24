""""Create your own": a custom tool composed from existing blocks (Flow Studio node types, optionally a bundled flow
to start from and a library geometry), returned as a flow document and the Python that runs it.

    from pinneapple_hub.library.compose import compose
    out = compose({"name": "Heat rod", "blocks": [
        {"type": "geometry.domain", "params": {"dimension": 1}},
        {"type": "geometry.mesh", "params": {"n": 41}}]})
    out["flow"]          # pinneapple-flow/1 document (open it in Flow Studio, or pf.run it)
    out["code"]          # Python: build the flow, validate, run, print each node's metrics
    out["issues"]        # validation issues (pinneapple_flow.validate_flow) and inputs that could not be wired

A whole flow can be the start too (``"flow": <pinneapple-flow/1 document>``, what Flow Studio's "Publish to Hub"
sends), and ``"expose": [{"node": id, "param": name}]`` turns those parameters into variables at the top of the
generated Python, so the published tool reads as a function of them.

Blocks are appended in order; each new block's required inputs are wired from the block before it, then from that
block's ancestors, then from any earlier block with a compatible output
(:func:`pinneapple_flow.lab_status.insert_nodes`). Only registered node types are accepted, and parameters are
checked against each node's schema, so a composed tool never contains code a visitor typed.
"""
from __future__ import annotations

import pprint
import re
from typing import Any

MAX_BLOCKS = 24


class ComposeError(ValueError):
    pass


def blocks() -> list[dict[str, Any]]:
    """The blocks the composer offers: every node type of Flow Studio's palette (curated nodes first, then the
    generated library calls), by palette section, with its ports and parameters."""
    from pinneapple_flow import REGISTRY
    from pinneapple_flow.palette import is_auto, sections
    order = {s["id"]: k for k, s in enumerate(sections())}
    titles = {s["id"]: s.get("title", s["id"]) for s in sections()}
    out = []
    for spec in REGISTRY.specs():
        if spec.type.startswith("custom."):
            continue
        d = spec.to_dict()
        sec = d["palette"]["section"]
        out.append({"type": spec.type, "title": d["palette"]["title"] or spec.title, "section": sec,
                    "section_title": titles.get(sec, sec), "summary": d["palette"]["summary"],
                    "available": d["available"], "install": d["palette"]["install"],
                    "inputs": [{"name": p.name, "type": p.type, "required": p.required} for p in spec.inputs],
                    "outputs": [{"name": p.name, "type": p.type} for p in spec.outputs],
                    "params_schema": d["params_schema"], "defaults": spec.defaults(), "auto": is_auto(spec),
                    "group": d["palette"].get("group", "")})
    out.sort(key=lambda b: (b["auto"], order.get(b["section"], 99), b["group"], b["title"].lower()))
    return out


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40] or "my_tool"


def compose(spec: dict[str, Any]) -> dict[str, Any]:
    """See the module doc. ``spec``: ``name``, ``description``, ``blocks`` ([{type, params}]), optional ``start``
    (a bundled flow name to append the blocks to) and ``geometry`` (a geometry library entry the code loads)."""
    from pinneapple_flow import REGISTRY
    from pinneapple_flow.lab_status import insert_nodes
    from pinneapple_flow.schema import SCHEMA_ID, validate_flow
    name = str(spec.get("name") or "My tool").strip()[:80]
    items = list(spec.get("blocks") or [])
    if len(items) > MAX_BLOCKS:
        raise ComposeError(f"at most {MAX_BLOCKS} blocks")
    doc: dict[str, Any] = {"schema": SCHEMA_ID, "name": name, "description": str(spec.get("description", ""))[:500],
                           "nodes": [], "edges": [], "meta": {"created_with": "PINNeAPPle Hub"}}
    if spec.get("flow"):
        base = _checked_flow(spec["flow"])
        doc["nodes"], doc["edges"] = base["nodes"], base["edges"]
        doc["name"] = name if spec.get("name") else base.get("name") or name
        doc["description"] = str(spec.get("description") or base.get("description", ""))[:500]
    elif spec.get("start"):
        from pinneapple_flow.service import FlowService
        try:
            base = FlowService.example(None, str(spec["start"]))       # reads pinneapple_flow/flows only
        except KeyError as exc:
            raise ComposeError(str(exc)) from None
        doc["nodes"], doc["edges"] = base["nodes"], base["edges"]
    issues: list[dict[str, str]] = []
    anchor = doc["nodes"][-1]["id"] if doc["nodes"] else None
    for k, b in enumerate(items):
        t = str(b.get("type", ""))
        s = REGISTRY.get(t)
        if s is None or t.startswith("custom."):
            raise ComposeError(f"block {k + 1}: unknown node type {t!r}")
        params = dict(b.get("params") or {})
        for p, msg in s.check_params(params):
            issues.append({"level": "error", "message": f"{s.title}: {msg}", "param": p})
        before = {n["id"] for n in doc["nodes"]}
        doc, missing = insert_nodes(doc, anchor, [{"type": t, "params": params}])
        anchor = next(n["id"] for n in doc["nodes"] if n["id"] not in before)
        for n in doc["nodes"]:
            if n["id"] == anchor:
                n["label"] = str(b.get("label") or s.title)[:60]
                n["position"] = {"x": 40.0 + 270.0 * (len(doc["nodes"]) - 1), "y": 60.0}
        for m in missing:
            issues.append({"level": "warning", "message": f"{s.title}: input {m} is not connected", "node": anchor})
    doc.setdefault("meta", {})["created_with"] = "PINNeAPPle Hub"
    issues += [i.to_dict() for i in validate_flow(doc)]
    geometry = str(spec.get("geometry") or "")
    if geometry:
        from pinneapple_design.geometry_library import list as geometry_names
        if geometry not in geometry_names():
            raise ComposeError(f"unknown geometry {geometry!r}")
    expose = _exposed(doc, spec.get("expose") or [], issues)
    return {"flow": doc, "code": code_for(doc, geometry=geometry, expose=expose), "issues": issues,
            "ok": not any(i["level"] == "error" for i in issues), "file_stem": _slug(doc["name"]),
            "expose": expose}


MAX_NODES = 60


def _checked_flow(flow: Any) -> dict[str, Any]:
    """A flow document from outside (Flow Studio): only its graph is kept, every node must be a registered type."""
    from pinneapple_flow import REGISTRY
    from pinneapple_flow.schema import FlowError, as_flow
    if not isinstance(flow, dict):
        raise ComposeError("flow: a pinneapple-flow/1 document")
    try:
        f = as_flow({k: flow[k] for k in ("schema", "id", "name", "description", "nodes", "edges") if k in flow})
    except (FlowError, TypeError, ValueError, KeyError) as exc:
        raise ComposeError(f"flow: {exc}") from None
    if len(f.nodes) > MAX_NODES:
        raise ComposeError(f"flow: at most {MAX_NODES} nodes")
    unknown = sorted({n.type for n in f.nodes if n.type not in REGISTRY or n.type.startswith("custom.")})
    if unknown:
        raise ComposeError(f"flow: unknown node types {', '.join(unknown)}")
    return f.to_dict()


def _exposed(doc: dict[str, Any], expose: list[Any], issues: list[dict[str, str]]) -> list[dict[str, Any]]:
    """The exposed parameters that exist, with their current value and a Python name."""
    from pinneapple_flow import REGISTRY
    by_id = {n["id"]: n for n in doc["nodes"]}
    out, names = [], set()
    for e in expose[:24]:
        nid, pname = str((e or {}).get("node", "")), str((e or {}).get("param", ""))
        node = by_id.get(nid)
        spec = REGISTRY.get(node["type"]) if node else None
        p = spec.param(pname) if spec else None
        if p is None:
            issues.append({"level": "warning", "message": f"exposed parameter {nid}.{pname} does not exist"})
            continue
        var = re.sub(r"[^a-z0-9_]", "_", f"{nid}_{pname}".lower())
        if var[0].isdigit() or var in names:
            var = f"p_{var}_{len(names)}"
        names.add(var)
        out.append({"node": nid, "param": pname, "var": var,
                    "value": node.get("params", {}).get(pname, p.default), "help": p.help})
    return out


def code_for(doc: dict[str, Any], *, geometry: str = "", workdir: str = "hub_tool",
             expose: list[dict[str, Any]] | None = None) -> str:
    """Python that rebuilds the flow, validates it, runs it headless and prints what each node measured; exposed
    parameters are variables set before the run."""
    flow = {k: v for k, v in doc.items() if k != "meta"}
    lit = pprint.pformat(flow, sort_dicts=False, width=110, compact=True)
    head = ["import time", "", "import pinneapple as pp", "import pinneapple_flow as pf", ""]
    if expose:
        head += ["# parameters of this tool"]
        head += [f"{e['var']} = {e['value']!r}" + (f"    # {e['help'][:70]}" if e["help"] else "") for e in expose]
        head += [""]
    sets = [f'next(n for n in FLOW["nodes"] if n["id"] == {e["node"]!r})["params"][{e["param"]!r}] = {e["var"]}'
            for e in expose or []]
    if geometry:
        head += [f'stl = pp.geometries.load("{geometry}", format="stl")      # library geometry for this tool',
                 f'print("geometry:", pp.geometries.info("{geometry}")["dimensions_m"], stl)', ""]
    return "\n".join(head + [
        f"FLOW = {lit}",
        *sets,
        "",
        "flow = pf.Flow.from_dict(FLOW)",
        'print("issues:", [i.message for i in pf.validate_flow(flow)])',
        "t0 = time.perf_counter()",
        f'result = pf.run(flow, workdir="{workdir}")',
        'print("run:", result.status, f"{time.perf_counter() - t0:.1f} s")',
        "for node in flow.nodes:",
        '    state = result.nodes.get(node.id, {})',
        '    print(node.id, state.get("status"), result.report(node.id).get("metrics", {}))',
        "",
    ])

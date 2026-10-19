"""The Lab seal of a flow's validated item and what would raise it, read through ``pinneapple_lab``.

Flow Studio never grades anything itself: the letter comes from the Lab curation (``pinneapple_lab.curation``) and
the seal API (``pinneapple_lab.seal``). Both are feature-detected, so the same code shows the A-F seal scale and
``curation.next_steps`` once the Lab provides them, and today's curation tier (A-D) or "not curated" before that.

* :func:`seal_status` - letter, its name, the reason; for the lowest letters the failing check and how to raise it.
* :func:`next_steps` - the gaps to the next letter with the Lab's suggested action and, when that action is a
  registered node type (or a library call a registered node wraps), the nodes to insert.
* :func:`insert_nodes` - add those nodes to a flow, wired by port type from an anchor node.
"""
from __future__ import annotations

import re
from typing import Any

NOT_AVAILABLE = ("pinneapple_lab.curation.next_steps is not available in this installation yet: the gaps the "
                 "curation lists are shown, without suggested nodes")


def _load(lab: Any):
    from pinneapple_lab.seal import load_curation
    return load_curation(lab)


def _attrs(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(obj)
    return {k: v for k, v in vars(obj).items() if not k.startswith("_")} if hasattr(obj, "__dict__") else {}


def seal_status(item: str, lab: Any = None) -> dict[str, Any]:
    """The seal letter of a Lab item with its reason, as the Lab states it. ``letter`` is None when the item was
    never curated."""
    from pinneapple_lab import curation
    out: dict[str, Any] = {"item": item, "letter": None, "name": "", "scale": "", "reason": "", "failing_check": "",
                           "hint": "", "sealed": False, "source": ""}
    names = getattr(curation, "SEALS", None) or curation.TIERS          # the A-F scale when the Lab has it
    out["scale"] = "".join(names)
    try:
        cur = _load(lab)
    except FileNotFoundError:
        out["reason"] = "not curated: the Lab has no curation record yet"
        return out
    rec = cur.get("items", {}).get(item)
    if rec is None:
        out["reason"] = "not curated"
        return out
    letter = rec.get("seal") or rec.get("letter") or rec.get("tier")
    hl = rec.get("headline") or {}
    out.update(letter=letter, name=rec.get("seal_name") or rec.get("tier_name") or names.get(letter, ""),
               reason=rec.get("reason") or hl.get("label") or hl.get("text", ""),
               failing_check=rec.get("failing_check") or "", hint=rec.get("how_to_raise") or rec.get("hint") or "",
               source="pinneapple_lab.curation", curated=str(cur.get("generated", "")))
    try:
        from pinneapple_lab.seal import seal_for
        s = _attrs(seal_for(item, lab))
    except Exception:  # noqa: BLE001 - SealError / KeyError: no seal at this letter, the curation record stands
        return out
    out["sealed"] = True
    out["source"] = "pinneapple_lab.seal"
    for k_out, keys in (("letter", ("letter", "seal", "tier")), ("reason", ("reason",)),
                        ("failing_check", ("failing_check",)), ("hint", ("how_to_raise", "hint"))):
        v = next((s[k] for k in keys if s.get(k)), None)
        if v:
            out[k_out] = v
    if s.get("reason_kind") == "failing" and not out["failing_check"]:       # E / F: the reason is the failing check
        out["failing_check"] = s.get("reason", "")
    out["name"] = names.get(out["letter"], out["name"])
    return out


# ------------------------------------------------------------------------------------------------ next steps
def _norm_calls(text: str) -> str:
    return re.sub(r"\bpp\.", "pinneapple.", (text or "").replace(":", "."))


def nodes_for(action: str = "", command: str = "", params: dict[str, Any] | None = None,
              target: str = "") -> list[dict[str, Any]]:
    """Registered node types that carry out a Lab action: the action names a node type, or its ``target`` (a dotted
    import path such as ``pinneapple.compare``) is a library call a registered node declares (its ``calls``: the
    full path, or the module and the function name). Curated nodes come before generated ``auto.*`` ones;
    parameters are kept only where the node has them."""
    from .registry import REGISTRY
    hits = [REGISTRY[w] for w in (action, command, target) if w and w in REGISTRY]
    if not hits and target and not target.startswith("cli:"):
        t = _norm_calls(target)
        mod, _, name = t.rpartition(".")
        specs = sorted((sp for sp in REGISTRY.specs() if not sp.type.startswith("custom.")),
                       key=lambda sp: sp.type.startswith("auto."))
        for sp in specs:
            c = _norm_calls(sp.calls)
            if re.search(rf"(^|[^\w.]){re.escape(t)}($|[^\w])", c) or (
                    mod and mod in c and re.search(rf"\b{re.escape(name)}\b", c)):
                hits.append(sp)
                break
    out = []
    for sp in hits[:1]:
        names = {p.name for p in sp.params}
        out.append({"type": sp.type, "params": {k: v for k, v in (params or {}).items() if k in names}})
    return out


def next_steps(item: str, lab: Any = None) -> dict[str, Any]:
    """What the item misses for the next seal letter, with the Lab's action and the nodes that carry it out."""
    from pinneapple_lab import curation
    fn = getattr(curation, "next_steps", None)
    if fn is None:
        try:
            rec = _load(lab).get("items", {}).get(item) or {}
        except FileNotFoundError:
            rec = {}
        return {"available": False, "message": NOT_AVAILABLE,
                "gaps": [{"gap": g if isinstance(g, str) else str(g), "action": "", "command": "", "actions": [],
                          "nodes": []} for g in rec.get("gaps", [])]}
    try:
        try:
            gaps = fn(item, lab=lab)
        except TypeError:
            gaps = fn(item)
    except Exception as exc:  # noqa: BLE001 - an unknown item or no curation: say so, keep the panel working
        return {"available": True, "message": f"{type(exc).__name__}: {exc}", "gaps": []}
    out = []
    for g in gaps or []:
        d = _attrs(g)
        acts = []
        for a in d.get("actions") or []:
            a = _attrs(a)
            acts.append({"id": a.get("id", ""), "kind": a.get("kind", ""), "target": a.get("target", ""),
                         "command": a.get("command", ""), "text": a.get("text", ""),
                         "nodes": nodes_for(a.get("id", ""), "", a.get("params") or {}, a.get("target", ""))})
        if not acts and (d.get("action") or d.get("command")):        # a flat gap (action / command / params)
            acts.append({"id": str(d.get("action", "")), "kind": "", "target": "", "command": str(d.get("command", "")),
                         "text": str(d.get("action", "")),
                         "nodes": nodes_for(str(d.get("action", "")), str(d.get("command", "")),
                                            d.get("params") if isinstance(d.get("params"), dict) else {})})
        nodes = []
        for a in acts:
            nodes += [n for n in a["nodes"] if n not in nodes]
        out.append({"gap": str(d.get("short") or d.get("gap") or d.get("missing") or d.get("why") or ""),
                    "why": str(d.get("why", "")), "to": str(d.get("to", "")), "check": str(d.get("check", "")),
                    "measured": d.get("measured"), "actions": acts,
                    "action": acts[0]["text"] if acts else "", "command": acts[0]["command"] if acts else "",
                    "nodes": nodes})
    return {"available": True, "message": "", "gaps": out}


def lab_items(reports: dict[str, dict[str, Any]]) -> list[str]:
    """The Lab items recorded by the validate.lab nodes of a run (``report["lab"]["item"]``)."""
    return sorted({r["lab"]["item"] for r in reports.values() if isinstance(r, dict) and isinstance(r.get("lab"), dict)
                   and r["lab"].get("item")})


# ------------------------------------------------------------------------------------------------ insert nodes
def insert_nodes(flow, anchor: str | None, items: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    """Add registered nodes ``[{"type", "params"}]`` to a flow and wire each required input from the anchor node,
    then from the anchor's upstream nodes, then from any node with a compatible output. Returns the new flow
    document and the inputs that could not be wired."""
    from .registry import REGISTRY
    from .schema import as_flow
    from .types import compatible
    doc = as_flow(flow).to_dict()
    nodes, edges = doc["nodes"], doc["edges"]
    by_id = {n["id"]: n for n in nodes}
    order = []
    if anchor in by_id:
        order.append(anchor)
        frontier = [anchor]
        while frontier:                                            # upstream of the anchor, nearest first
            ups = [e["source"] for e in edges if e["target"] in frontier and e["source"] not in order]
            order += list(dict.fromkeys(ups))
            frontier = ups
    order += [n["id"] for n in nodes if n["id"] not in order]
    base = by_id.get(anchor, {}).get("position") or {"x": 0, "y": 0}
    missing = []
    for k, it in enumerate(items):
        spec = REGISTRY[it["type"]]                                # KeyError: only registered types
        stem = re.sub(r"[^A-Za-z0-9_]", "_", spec.type.split(".")[-1])[:40] or "n"
        nid, i = stem, 1
        while nid in by_id:
            i += 1
            nid = f"{stem}_{i}"
        names = {p.name for p in spec.params}
        node = {"id": nid, "type": spec.type, "params": {a: b for a, b in (it.get("params") or {}).items()
                                                          if a in names},
                "position": {"x": base.get("x", 0) + 280, "y": base.get("y", 0) + 140 * (k + 1)}}
        nodes.append(node)
        by_id[nid] = node
        used: set[tuple[str, str]] = set()                        # one output feeds one input of a new node
        for port in spec.inputs:
            src = None
            for cand in order:
                cs = REGISTRY.get(by_id[cand]["type"])
                out = next((o for o in (cs.outputs if cs else []) if compatible(o.type, port.type)
                            and (cand, o.name) not in used), None)
                if out is not None:
                    src = (cand, out.name)
                    break
            if src:
                used.add(src)
                edges.append({"id": f"{src[0]}.{src[1]}->{nid}.{port.name}", "source": src[0],
                              "source_port": src[1], "target": nid, "target_port": port.name})
            elif port.required:
                missing.append(f"{nid}.{port.name} ({port.type})")
    return doc, missing

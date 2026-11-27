"""Which workflow stages a proposed flow spans, and what the stage map says is available at each one.

The planner builds a flow out of nodes and explains each choice with a decision. That explains *this* flow; it does
not say whether the workflow it describes can actually be executed here, which is the question
:mod:`pinneapple_catalog.stages` answers — the open tools and public datasets available at each stage, and how many of
those tools this library really calls (``wired``).

This module joins the two. Given the nodes of a flow it reports, per stage the flow touches:

* how many of the map's tools at that stage are **wired** in this library, and how many are only candidates;
* which public datasets the map knows for that stage, split into the ones this library can read
  (``readable``: there is an adapter) and the ones it merely catalogues;
* the stage's own ``gap`` sentence, which is what a person needs when the answer is "one wired tool, and not the kind
  this step needs".

``thin`` is the operational finding, and it is deliberately not "zero wired tools": the inverse stage has exactly one
(the equation-discovery package, not a sampler), and a stage where a single external tool is wired offers no choice
at all. A stage with ``wired <= THIN`` is reported with its gap.

    >>> from pinneapple_flow.stage_fit import fit
    >>> f = fit([{"type": "inverse.calibrate"}, {"type": "geometry.domain"}])
    >>> [s["id"] for s in f["stages"]]
    ['problem', 'inverse']
    >>> [s["id"] for s in f["thin"]]
    ['inverse']

The join is optional at runtime: without ``pinneapple_catalog`` installed, :func:`fit` still reports which stages the
flow spans and says in ``reason`` why no coverage came back, rather than returning an empty map that would read as
"nothing is available anywhere".
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

#: a stage with this many wired tools or fewer is reported as thin: one wired tool is not a choice
THIN = 1


def _stage_catalog():
    try:
        from pinneapple_catalog import stages as catalog_stages
    except Exception as exc:                                    # partial install: the join degrades, it does not fail
        return None, f"{type(exc).__name__}: {exc}"
    return catalog_stages, ""


def node_stage(node_type: str) -> str | None:
    """The workflow stage a node type belongs to, or ``None`` for a type the palette does not place.

    The palette is the single source of placement (``SECTION_OF`` / category), so a node shown under "Inverse &
    calibration" in the Studio is the same node this reports as the inverse stage."""
    from . import palette
    try:
        spec = palette.REGISTRY.get(node_type)
    except Exception:
        return None
    if spec is None:
        return None
    try:
        return palette.section_of(spec)
    except Exception:                                            # an unplaced type is not an error here
        return None


def flow_stages(nodes: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    """Stage id -> the node ids of a flow that sit there, in workflow order (unplaced types are left out)."""
    from .palette import SECTION_IDS
    found: dict[str, list[str]] = {}
    for node in nodes:
        stage_id = node_stage(str(node.get("type", "")))
        if stage_id is None:
            continue
        found.setdefault(stage_id, []).append(str(node.get("id", "")))
    return {s: found[s] for s in SECTION_IDS if s in found}


def fit(nodes: Iterable[dict[str, Any]], *, with_datasets: bool = True) -> dict[str, Any]:
    """What the stage map says about the stages a flow spans, the thin ones named, and the datasets on offer."""
    nodes = list(nodes)
    spanned = flow_stages(nodes)
    catalog, reason = _stage_catalog()
    if catalog is None:
        return {"stages": [{"id": s, "nodes": ids} for s, ids in spanned.items()],
                "thin": [], "available": False,
                "reason": f"the stage map could not be read ({reason})"}

    rows: list[dict[str, Any]] = []
    thin: list[dict[str, Any]] = []
    for stage_id, node_ids in spanned.items():
        st = catalog.stage(stage_id)
        tools = catalog.tools(stage_id=stage_id)
        wired = [t for t in tools if t.integration == "wired"]
        row: dict[str, Any] = {
            "id": stage_id, "title": st.title, "question": st.question, "nodes": node_ids,
            "tools": len(tools), "wired": len(wired),
            "wired_tools": [t.id for t in wired],
            "candidates": sum(1 for t in tools if t.integration == "candidate"),
            "gap": st.gap,
        }
        if with_datasets:
            datasets = catalog.datasets(stage_id)
            row["datasets"] = [d["id"] for d in datasets]
            row["readable_datasets"] = [d["id"] for d in datasets if d.get("readable")]
        rows.append(row)
        if len(wired) <= THIN:
            thin.append({"id": stage_id, "title": st.title, "wired": len(wired), "gap": st.gap,
                         "wired_tools": [t.id for t in wired]})

    return {"stages": rows, "thin": thin, "available": True, "reason": "",
            "note": ("'thin' is a property of this library, not of the flow: a stage where at most one external tool "
                     "is wired offers no choice, so the flow runs that step on PINNeAPPle's own code and the gap "
                     "says what is missing.")}


def fit_message(result: dict[str, Any]) -> str:
    """One sentence for a decision record or a warning line, from :func:`fit`'s answer."""
    if not result.get("available"):
        return f"stage coverage not checked: {result.get('reason', '')}"
    thin = result.get("thin") or []
    if not thin:
        return "every stage of this flow has more than one wired external tool"
    return "; ".join(f"{t['id']}: {t['wired']} wired tool{'s' if t['wired'] != 1 else ''}" for t in thin)

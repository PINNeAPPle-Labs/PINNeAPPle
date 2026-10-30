"""Publishing an experiment of the Lab: one folder with its result, its figures and the page that explains it.

The page is the publication: the question, the scenario with the provenance of every number, the equations, the
result with its comparisons and checks, the six evidence axes with their two verdicts, and -- in its own section,
never as a footnote -- what the experiment does **not** show.

    python -m pinneapple_futures publish futures.mars_habitat --out out/futures
    python -m pinneapple_futures publish all --out out/futures

Layout: ``<out>/<experiment>/`` holds ``result.json``, the figures and ``README.md``; ``<out>/Catalogue.md`` holds
the table of everything published, and ``<out>/ATLAS.md`` the atlas of the directions.
"""
from __future__ import annotations

import json
import os
from typing import Any

from .atlas import load as load_atlas
from .core import get, names

__all__ = ["publish", "publish_all", "write_catalogue", "folder_of"]

LIMITS = (
    "The equations solved are the ones written on this page. A result that matches its closed form shows that the "
    "model was solved correctly, not that the model is the world.",
    "No measurement was compared: nothing here is validated against an experiment on a real system.",
    "No technological feasibility was assessed: nothing here says the system can be built, powered or afforded.",
)


def folder_of(name: str, root: str) -> str:
    """The folder an experiment publishes into."""
    return os.path.join(root, name.split(".", 1)[-1])


def publish(name: str, *, root: str = "out/futures", quick: bool = False) -> dict[str, Any]:
    """Run ``name``, save its result and figures, and write the publication page. Returns the paths written."""
    exp = get(name)
    folder = folder_of(name, root)
    os.makedirs(folder, exist_ok=True)
    res = exp.run(quick=quick)
    paths = res.save(folder)
    page = os.path.join(folder, "README.md")
    with open(page, "w", encoding="utf-8") as fh:
        fh.write(to_markdown(exp, res))
    return {"folder": folder, "page": page, "result": paths.get("result"), "passed": res.passed,
            "decision": res.decision, "assessment": None if res.assessment is None else res.assessment.to_dict()}


def to_markdown(exp, res) -> str:
    """The publication page of one result."""
    world = exp.world()
    out = [f"# {exp.title}", "", f"*{exp.area} -- {exp.topic} -- {exp.level} -- {exp.kind}*", "",
           exp.description, "", f"**Verdict:** {res.assessment.headline() if res.assessment else res.decision}", ""]
    if world is not None:
        out += ["## The world it runs in", "", f"**{world.title or world.name}.** {world.description}", "",
                "| Quantity | Value | Unit | Provenance | Source |", "|---|---|---|---|---|"]
        for i in world.inputs:
            out.append(f"| {i.name} | {i.value:g} | {i.unit or '-'} | {i.provenance} | {i.source} |")
        out += ["", "| Derived | Value | Unit | How |", "|---|---|---|---|"]
        for d in world.derived().values():
            out.append(f"| {d.name} | {d.value:.6g} | {d.unit} | {d.how} |")
        out += ["", "### What the world rests on", ""]
        for c in world.claims:
            law = f" (`{c.law}`)" if c.law else ""
            out.append(f"- **{c.plausibility}** -- {c.statement}{law}")
            if c.source:
                out.append(f"  - source: {c.source}")
        out.append("")

    out += ["## What was measured", "", "| Quantity | Value | Unit | How |", "|---|---|---|---|"]
    for k, v in res.values.items():
        out.append(f"| {k} | {v['value']:.6g} | {v['unit'] or '-'} | {v['how']} |")
    out += ["", "## Against the references", "",
            "| Quantity | Computed | Reference | Rel. error | Tolerance | Kind | Source |",
            "|---|---|---|---|---|---|---|"]
    for c in res.comparisons:
        mark = "" if c.passed else " **FAIL**"
        out.append(f"| {c.quantity}{mark} | {c.computed:.6g} | {c.reference:.6g} | {c.rel_error:.2e} | "
                   f"{c.tol:.2g} | {c.kind} | {c.source} |")
    out += ["", "## Physics checks", "", "| Check | Result | Value | Limit | Evidence class |", "|---|---|---|---|---|"]
    for k in res.checks:
        state = "not run" if k["passed"] is None else ("pass" if k["passed"] else "**FAIL**")
        val = "-" if k["value"] is None else f"{k['value']:.4g}"
        lim = "-" if k["limit"] is None else f"{k['limit']:.4g}"
        out.append(f"| {k['name']} | {state} | {val} | {lim} | {k['source']} |")
    if res.assessment is not None:
        a = res.assessment
        out += ["", "## Evidence and verdict", "", f"**{a.headline()}**", "",
                "| Axis | Passed | Assessed | Evidence |", "|---|---|---|---|"]
        for axis in a.axes.values():
            out.append(f"| {axis.axis.replace('_', ' ')} | {axis.passed} | {axis.total} | "
                       f"{', '.join(axis.evidence) or 'not assessed'} |")
        for n in a.notes:
            out.append(f"\n> {n}")
    out += ["", "## What this does not show", ""]
    for line in LIMITS:
        out.append(f"- {line}")
    for n in res.notes:
        out.append(f"- {n}")
    out += ["", "## References", ""]
    for ref in exp.references:
        out.append(f"- {ref}")
    out += ["", f"*Trust decision: {res.decision}. Generated by ``python -m pinneapple_futures publish "
                f"{res.experiment}``.*", ""]
    return "\n".join(out)


def write_catalogue(root: str = "out/futures", *, quick: bool = False) -> str:
    """Publish every experiment and write the catalogue of the Lab; returns the catalogue path."""
    rows = []
    for name in names():
        info = publish(name, root=root, quick=quick)
        rows.append((name, info))
    os.makedirs(root, exist_ok=True)
    page = os.path.join(root, "Catalogue.md")
    out = ["# PINNeAPPle Futures Lab", "", "*Exploring the physics of possible worlds.*", "",
           "| Experiment | Area | Verdict | Physics | Engineering | Class | Passed |",
           "|---|---|---|---|---|---|---|"]
    for name, info in rows:
        a = info["assessment"] or {}
        out.append(f"| [{name}]({name.split('.', 1)[-1]}/README.md) | {get(name).area} | "
                   f"{a.get('physics_verdict', '-')} | {a.get('engineering_verdict', '-')} | "
                   f"{a.get('epistemic', '-')} | {info['passed']} |")
    out += ["", "## Directions", "", load_atlas().to_markdown(), ""]
    with open(page, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    with open(os.path.join(root, "ATLAS.md"), "w", encoding="utf-8") as fh:
        fh.write("# Directions of the Futures Lab\n\n" + load_atlas().to_markdown() + "\n")
    return page


def publish_all(root: str = "out/futures", *, quick: bool = False) -> list[dict[str, Any]]:
    """Every experiment of the Lab, published; returns one entry per experiment."""
    return [publish(n, root=root, quick=quick) for n in names()]


def write_json(path: str, payload: Any) -> str:
    """Write JSON with the Lab's own schema marker (used by the command line)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    return path

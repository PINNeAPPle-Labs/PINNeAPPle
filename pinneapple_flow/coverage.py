"""Which PINNeAPPle capabilities are flow nodes, and why the others are not.

:func:`coverage` walks every ``pinneapple*`` package and its first-level modules, maps the registered node types onto
them (by the library calls each node declares) and attaches a curated reason (:data:`REASONS`) to whatever has no
node. ``python -m pinneapple_flow coverage -o docs/flow_studio/COVERAGE.md`` writes the table.
"""
from __future__ import annotations

import pkgutil
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# package or package.module -> why there is no node (or "covered by <node types>" when the coverage is indirect)
REASONS: dict[str, str] = {
    # compatibility shims and infrastructure
    "pinneapple_models": "shim: re-exports pinneapple_neural.architectures (covered there)",
    "pinneapple_train": "shim: re-exports pinneapple_neural.trainer (covered there)",
    "pinneapple_solvers": "shim: re-exports pinneapple_simulation.numerical_solvers (covered there)",
    "pinneapple_orchestration": "pipeline wiring for Prefect; flows are orchestrated by pinneapple_flow itself",
    "pinneapple_mcp": "the MCP server for coding agents (a protocol front end, like Flow Studio)",
    "pinneapple_flow": "this package",
    "pinneapple_registry": "artifact store and model gateway services (deployment, not a computation step)",
    "pinneapple_security": "integrity, signing, web security and audit (infrastructure used by the apps)",
    "pinneapple_core.service": "job queue / API keys of the app servers (infrastructure)",
    "pinneapple_core.workflows": "the execution engine every flow runs on",
    "pinneapple_problemdesign": "conversational problem design with an external LLM agent; the planner and "
                                "problem.generic cover the same ground offline",
    "pinneapple_quantum": "needs a quantum SDK (Qiskit / PennyLane), not installed by the flow extra",
    "pinneapple_worldmodel": "foundation-model training at GPU scale; no CPU-sized entry point",
    "pinneapple_perception": "image / video / audio inputs: no media port type in the flow schema yet",
    "pinneapple_blender": "needs Blender's bpy; viz.render uses it when installed",
    "pinneapple_twin3d": "3-D twin scene server; viz.field3d gives the in-studio 3-D view",
    "pinneapple_hub": "Hugging Face Hub transfers (network access and credentials, not a computation step)",
    "pinneapple_catalog": "method / resource catalogue with license gates (read by the planner and docs)",
    "pinneapple_decision": "experiment-selection loop driven by an LLM backend; the planner covers routing",
    "pinneapple_arena": "YAML benchmark campaigns (long runs); single benchmark runs are benchmark.arena nodes",
    "pinneapple_pdb": "dataset builder for benchmark publication (sharding, upload)",
    "pinneapple_veriphysics": "report layer over results; decision records + export.report cover it here",
    "pinneapple_check": "the CI physics linter (runs on files in a repository, not in a flow)",
    "pinneapple_lab": "covered by validate.lab (run, curation tiers, seal)",
    "pinneapple_llm": "LLM drafting: used by the LLM planner, problem.generic and twin.auto",
    "pinneapple_app": "a separate web application (not a library package)",
}


def packages() -> list[str]:
    return sorted(p.name for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("pinneapple")
                  and (p / "__init__.py").is_file())


def modules(pkg: str) -> list[str]:
    path = ROOT / pkg
    return sorted(f"{pkg}.{m.name}" for m in pkgutil.iter_modules([str(path)]) if not m.name.startswith("_"))


_MOD = re.compile(r"\bpp\.[a-z_]+|\bpinneapple[a-z_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")


def node_targets(auto: bool = True) -> dict[str, set[str]]:
    """module path -> node types that call into it (``auto``: with the nodes generated from the public API)."""
    from .registry import REGISTRY
    out: dict[str, set[str]] = {}
    pp = {"pp.solve": "pinneapple_physics.solving", "pp.trust_card": "pinneapple_analysis.trust",
          "pp.sobol_indices": "pinneapple_analysis.sensitivity", "pp.viz": "pinneapple_tools.visualization"}
    for s in REGISTRY.specs():
        if s.type.startswith("custom.") or (not auto and s.type.startswith("auto.")):
            continue                              # user-defined types reuse allowlisted calls; not library coverage
        for m in _MOD.findall(s.calls or ""):
            if m.startswith("pp."):
                out.setdefault("pinneapple", set()).add(s.type)       # the pp namespace itself
            m = pp.get(m, m)
            out.setdefault(m, set()).add(s.type)
    return out


def coverage(auto: bool = True) -> list[dict[str, str]]:
    """One row per package and first-level module: the node types that reach it, or why there are none (a curated
    reason, or the one the public-API scan of :mod:`pinneapple_flow.autocalls` recorded)."""
    targets = node_targets(auto)
    reasons = dict(REASONS)
    if auto:
        from .autocalls import load
        reasons.update(load()["reasons"])
    rows = []
    for pkg in packages():
        mods = modules(pkg) or [pkg]
        for m in [pkg] + [x for x in mods if x != pkg]:
            nodes = sorted({t for k, v in targets.items() for t in v if k == m or k.startswith(m + ".")})
            reason = reasons.get(m) or (reasons.get(pkg) if m != pkg else "") or ""
            rows.append({"module": m, "nodes": ", ".join(nodes), "reason": reason if not nodes else ""})
    return rows


def counts(rows: list[dict[str, str]] | None = None) -> dict[str, int]:
    rows = coverage() if rows is None else rows
    return {"modules": len(rows), "with_nodes": sum(1 for r in rows if r["nodes"]),
            "with_reason": sum(1 for r in rows if not r["nodes"] and r["reason"]),
            "unreached": sum(1 for r in rows if not r["nodes"] and not r["reason"])}


def markdown() -> str:
    rows = coverage()
    c = counts(rows)
    from .autocalls import load
    auto = load()["calls"]
    ok = sum(a.get("check", "").startswith("runs") for a in auto)
    out = ["# Flow Studio coverage of the library", "",
           f"**{c['modules']} packages / modules: {c['with_nodes']} with nodes, {c['with_reason']} with a stated "
           f"reason, {c['unreached']} unreached.**", "",
           "Generated by `python -m pinneapple_flow coverage` from the node registry (each node declares the library "
           "calls it wraps), the curated reasons in `pinneapple_flow/coverage.py` and the public-API scan of "
           "`pinneapple_flow/autocalls.py` (`python -m pinneapple_flow autocalls`). Curated nodes come first in each "
           f"row; `auto.*` nodes are generated from type-hinted public functions ({len(auto)} nodes, {ok} of them run "
           "with their default form values, the rest need input: each node shows its check). A module without a node "
           "says why: a curated reason, or what the scan found (only classes, or functions that take objects, files "
           "or code). `tests/test_flow_library_nodes.py` fails when a public module is neither.", "",
           "| package / module | node types | why not a node |", "|---|---|---|"]
    for r in rows:
        nodes = r["nodes"].split(", ") if r["nodes"] else []
        cur = [n for n in nodes if not n.startswith("auto.")]
        au = [n for n in nodes if n.startswith("auto.")]
        cell = ", ".join(cur + ([f"{len(au)} auto (" + ", ".join(a.rsplit(".", 1)[1] for a in au[:6])
                                 + (", ..." if len(au) > 6 else "") + ")"] if au else []))
        out.append(f"| `{r['module']}` | {cell} | {r['reason']} |")
    return "\n".join(out) + "\n"

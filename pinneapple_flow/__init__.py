"""PINNeAPPle Flow: engineering flows as typed node graphs, planned from a description, run headless or in Flow Studio.

    import pinneapple_flow as pf

    plan = pf.plan("heat in a 2D plate with fixed edges, compare a PINN with finite differences")
    flow = plan.proposals[0].flow          # a Flow (JSON schema pinneapple-flow/1)
    result = pf.run(flow, workdir="work")   # on the pinneapple_core.workflows engine (cache, logs, retries)
    result.report("validate")["metrics"]

    python -m pinneapple_flow run flow.json        # the same, from the shell
    python -m pinneapple_flow studio               # the GUI on http://127.0.0.1:8765

=====================  ==============================================================================================
:mod:`.schema`         the flow document (``Flow``, ``Node``, ``Edge``), ``validate_flow``, ``flow_json_schema``
:mod:`.types`          port types and which output may feed which input
:mod:`.registry`       node types (``NodeSpec``, ``Port``, ``Param``, ``REGISTRY``, ``@node``)
:mod:`.nodes`          the built-in catalog: each node wraps a library call
:mod:`.runtime`        ``run``: flows on the workflow engine, per-node status, artifacts
:mod:`.planner`        description -> proposals (heuristic offline planner, LLM planner with validation/repair)
:mod:`.store`          flows, versions, runs and node outputs in SQLite or any SQLAlchemy database
:mod:`.learn`          surrogate architectures, preview, training, tuning
:mod:`.templates`      equation templates per PDE family and dimension
:mod:`.service`        the operations Flow Studio exposes (the app's routes call these)
=====================  ==============================================================================================
"""
from __future__ import annotations

from .registry import REGISTRY, NodeSpec, Param, Port, node
from .schema import (
    SCHEMA_ID,
    Edge,
    Flow,
    FlowError,
    Issue,
    Node,
    as_flow,
    check_flow,
    flow_json_schema,
    validate_flow,
)
from .types import PORT_TYPES, compatible, register_port_type

__all__ = ["Flow", "Node", "Edge", "FlowError", "Issue", "SCHEMA_ID", "as_flow", "check_flow", "validate_flow",
           "flow_json_schema", "REGISTRY", "NodeSpec", "Port", "Param", "node", "PORT_TYPES", "compatible", "run",
           "plan", "catalog", "FlowStore", "register_port_type", "register_references", "register_rule", "Rule",
           "NotRun"]
__version__ = "0.1.0"


def register_references(entries):
    """Add curated references for another package's rules (see :func:`pinneapple_flow.references.register_references`)."""
    from .references import register_references as _r
    return _r(entries)


def register_rule(r):
    """Add a heuristic rule that cites registered references (see :func:`pinneapple_flow.rules.register_rule`)."""
    from .rules import register_rule as _r
    return _r(r)


def __getattr__(name):
    if name == "Rule":
        from .rules import Rule
        return Rule
    if name == "NotRun":
        from .runtime import NotRun
        return NotRun
    if name == "FlowStore":
        from .store import FlowStore
        return FlowStore
    raise AttributeError(name)


def run(flow, **kw):
    """Run a flow headless (see :func:`pinneapple_flow.runtime.run`)."""
    from .runtime import run as _run
    return _run(flow, **kw)


def plan(text: str, **kw):
    """Proposals for a problem description (see :func:`pinneapple_flow.planner.plan`)."""
    from .planner import plan as _plan
    return _plan(text, **kw)


def catalog() -> list[dict]:
    """Every node type as JSON (ports, parameter schema, availability)."""
    return REGISTRY.catalog()


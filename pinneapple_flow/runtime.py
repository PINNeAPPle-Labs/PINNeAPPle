"""Run a flow on the workflow engine of :mod:`pinneapple_core.workflows`.

Each node becomes a :class:`~pinneapple_core.workflows.Step` of kind ``pinneapple_flow.node``; the engine gives the
order, per-node logs and states, retries, cancel and **skip-done**: a node whose type, parameters and upstream are
unchanged reuses its earlier outputs (state ``cached``). ``from_node`` re-runs one node and everything downstream of it
(upstream nodes are reused from the cache when they ran before).

    from pinneapple_flow import run
    result = run("flow.json", workdir="work")
    result.status, result.nodes["pinn"]["status"], result.nodes["pinn"]["report"]["metrics"]
    u = result.load("fd", "solution")            # arrays of an artifact output

Values on edges are JSON; arrays, weights and figures are files in the node's folder, referenced as
``{"$artifact": "<absolute path>", "kind": "npz" | "json" | "png" | "pt"}``.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pinneapple_core.workflows import (
    Adapter,
    Step,
    StepContext,
    StepFailed,
    Workflow,
    WorkflowRunner,
    register_adapter,
)
from pinneapple_core.workflows.adapters import ADAPTERS

from .registry import REGISTRY
from .schema import Flow, as_flow, check_flow

ADAPTER_KIND = "pinneapple_flow.node"
STATUS = {"pending": "queued", "running": "running", "done": "done", "failed": "failed", "skipped": "cached",
          "cancelled": "cancelled"}
NODE_STATES = ("queued", "running", "done", "degraded", "not_run", "failed", "cached", "cancelled")


class NotRun(Exception):
    """Raised by a node (``ctx.not_run(reason)``) that cannot run here: the node finishes ``not_run`` with the
    reason, outputs nothing, and the nodes that need its outputs finish ``not_run`` too (not ``failed``)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = str(reason)


def default_workdir() -> Path:
    return Path(os.environ.get("PINNEAPPLE_FLOW_HOME", Path.home() / ".pinneapple_flow")) / "work"


# ----------------------------------------------------------------------------------------------- node context
class NodeContext:
    """What a node function gets besides its inputs and parameters: a folder, a log, and helpers that turn
    arrays, figures and tables into artifacts and report entries (shown when the node is clicked)."""

    def __init__(self, node_id: str, node_type: str, workdir: Path, log: Callable[[str], None],
                 should_stop: Callable[[], bool] = lambda: False, meta: dict | None = None):
        self.node_id, self.node_type, self.workdir = node_id, node_type, Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._log = log
        self.should_stop = should_stop
        self.meta = meta or {}
        self.report: dict[str, Any] = {"metrics": {}, "figures": [], "tables": [], "views": {}, "warnings": [],
                                       "summary": "", "decisions": []}
        self.upstream_reports: dict[str, dict[str, Any]] = {}
        self.explicit: set[str] = set()                 # parameters the flow document set (not defaults)

    # -- outcome ---------------------------------------------------------------------------------------------------
    def not_run(self, reason: str):
        """Finish this node as ``not_run`` (it did not run, for ``reason``: a missing tool, a missing input)."""
        raise NotRun(reason)

    def degrade(self, reason: str) -> None:
        """Mark this node ``degraded``: it ran, but on a declared fallback (``reason`` says which)."""
        self.report["state"] = "degraded"
        self.report.setdefault("degraded", []).append(str(reason))
        self.warn(f"degraded: {reason}")

    # -- logging and report ----------------------------------------------------------------------------------------
    def log(self, msg: str) -> None:
        self._log(str(msg))

    def metric(self, name: str, value: Any, how: str = "") -> None:
        v = _plain(value)
        self.report["metrics"][name] = v if not how else {"value": v, "how": how}

    def warn(self, msg: str) -> None:
        self.report["warnings"].append(str(msg))
        self.log(f"warning: {msg}")

    def summary(self, text: str) -> None:
        self.report["summary"] = str(text)

    def table(self, title: str, columns: list[str], rows: list[list[Any]]) -> None:
        self.report["tables"].append({"title": title, "columns": list(columns),
                                      "rows": [[_plain(v) for v in r] for r in rows]})

    def view(self, kind: str, data: Any) -> None:
        """A rich result for the GUI: ``"architecture"`` (layer graph), ``"scene"`` (three.js payload),
        ``"equations"`` (parsed system), ``"plan"``..."""
        self.report["views"][kind] = data

    def decide(self, rule_id: str, what: str, value: Any = None, *, rationale: str = "",
               extra: tuple[str, ...] = ()) -> None:
        """Record a decision this node takes while running (a rule with its references and evidence)."""
        from .decisions import rule_decision
        self.report["decisions"].append(rule_decision(rule_id, what=what, node=self.node_id, value=_plain(value),
                                                      rationale=rationale, extra=extra).to_dict())

    def auto(self, rule_id: str, what: str, *keys: str, value: Any = None, rationale: str = "") -> None:
        """Record a decision for parameters the node filled with its defaults (those the flow set explicitly are
        already recorded by the planner or as the user's edits)."""
        if not any(k in self.explicit for k in keys):
            self.decide(rule_id, what, value, rationale=rationale)

    def user_choice(self, topic: str, what: str, value: Any = None) -> None:
        """Record that a value the node used was set by the user (not by a rule)."""
        from .decisions import user_decision
        self.report["decisions"].append(user_decision(self.node_id, topic, what, _plain(value)).to_dict())

    def check_stop(self) -> None:
        if self.should_stop():
            from pinneapple_core.workflows import StepCancelled
            raise StepCancelled("cancelled")

    # -- artifacts -------------------------------------------------------------------------------------------------
    def path(self, name: str) -> Path:
        p = (self.workdir / name).resolve()
        if self.workdir.resolve() not in p.parents and p != self.workdir.resolve():
            raise StepFailed(f"artifact name {name!r} leaves the node folder")
        return p

    def save_npz(self, name: str, **arrays) -> dict[str, Any]:
        p = self.path(f"{name}.npz")
        np.savez_compressed(p, **{k: np.asarray(v) for k, v in arrays.items()})
        return artifact(p, "npz")

    def save_json(self, name: str, obj: Any) -> dict[str, Any]:
        p = self.path(f"{name}.json")
        p.write_text(json.dumps(obj, indent=1, default=_plain))
        return artifact(p, "json")

    def save_torch(self, name: str, obj: Any) -> dict[str, Any]:
        import torch
        p = self.path(f"{name}.pt")
        torch.save(obj, p)
        return artifact(p, "pt")

    def figure(self, fig, name: str, caption: str = "", dpi: int = 110) -> dict[str, Any]:
        import matplotlib.pyplot as plt
        p = self.path(f"{name}.png")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        a = artifact(p, "png")
        self.report["figures"].append({"path": str(p), "name": name, "caption": caption})
        return a


def artifact(path: str | Path, kind: str) -> dict[str, Any]:
    return {"$artifact": str(Path(path).resolve()), "kind": kind}


def is_artifact(v: Any) -> bool:
    return isinstance(v, dict) and "$artifact" in v


def load_artifact(v: Any):
    """The content of an artifact value: npz -> dict of arrays, json -> object, pt -> torch object, png -> path."""
    if not is_artifact(v):
        return v
    p = Path(v["$artifact"])
    if not p.is_file():
        raise StepFailed(f"artifact {p} is missing (the run folder was removed?)")
    kind = v.get("kind")
    if kind == "npz":
        with np.load(p, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    if kind == "json":
        return json.loads(p.read_text())
    if kind == "pt":
        import torch
        return torch.load(p, weights_only=False)
    return p


def _plain(v: Any):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if hasattr(v, "item") and callable(v.item):
        try:
            return v.item()
        except (TypeError, ValueError, RuntimeError):
            pass
    return v


# ----------------------------------------------------------------------------------------------- engine adapter
class FlowNodeAdapter(Adapter):
    """Runs one flow node in-process: resolves its inputs from upstream outputs and calls the registered type."""
    kind = ADAPTER_KIND

    def available(self, step: Step | None = None):
        # never refuse a step: a node whose library is missing finishes ``not_run`` with the reason (see run)
        if step is not None and REGISTRY.get(step.params.get("node_type", "")) is None:
            return False, f"unknown node type {step.params.get('node_type')!r}"
        return True, ""

    @staticmethod
    def _finish(ctx: StepContext, report: dict[str, Any], ports: dict[str, Any]) -> dict[str, Any]:
        report = _plain(report)
        (Path(ctx.workdir) / "report.json").write_text(json.dumps(report, indent=1))
        return {"ports": _plain(ports), "report": report}

    def run(self, ctx: StepContext) -> dict[str, Any]:
        p = ctx.params
        spec = REGISTRY[p["node_type"]]
        nctx = NodeContext(ctx.step.id, spec.type, ctx.workdir, ctx.log, ctx.should_stop, p.get("meta") or {})
        try:
            ok, why = spec.available()
            if not ok:
                raise NotRun(why or "not available here")
            inputs: dict[str, Any] = {}
            for port, sources in (p.get("inputs") or {}).items():
                pspec = spec.input(port)
                vals = []
                for src, src_port in sources:
                    up = ctx.upstream.get(src)
                    if up is None:
                        raise StepFailed(f"upstream node {src!r} has no outputs")
                    urep = (up.get("outputs") or {}).get("report") or {}
                    if urep.get("state") == "not_run":
                        if pspec is not None and not pspec.required:
                            continue                             # an optional input from a node that did not run
                        raise NotRun(f"upstream node {src!r} did not run: {urep.get('not_run_reason', '')}")
                    ports = (up.get("outputs") or {}).get("ports") or {}
                    if src_port not in ports:
                        raise StepFailed(f"upstream node {src!r} produced no {src_port!r}")
                    vals.append(ports[src_port])
                inputs[port] = vals if (pspec is not None and pspec.multiple) else (vals[0] if vals else None)
            nctx.upstream_reports = {k: (v.get("outputs") or {}).get("report") or {} for k, v in ctx.upstream.items()}
            nctx.explicit = set((p.get("params") or {}).keys())
            params = spec.resolved(p.get("params") or {})
            ctx.log(f"{spec.type}: {spec.calls or spec.title}")
            out = spec.fn(nctx, inputs, params) or {}
        except NotRun as nr:
            ctx.log(f"not run: {nr.reason}")
            nctx.report.update(state="not_run", not_run_reason=nr.reason, summary=f"not run: {nr.reason}")
            return self._finish(ctx, nctx.report, {})
        unknown = set(out) - {o.name for o in spec.outputs}
        if unknown:
            raise StepFailed(f"node returned unknown outputs {sorted(unknown)}")
        return self._finish(ctx, nctx.report, out)


if ADAPTER_KIND not in ADAPTERS:
    register_adapter(FlowNodeAdapter())


# ----------------------------------------------------------------------------------------------- compile and run
COLLECTORS = {"export.report"}


def _flow_decisions(flow: Flow) -> list[dict[str, Any]]:
    from .decisions import collect
    return collect(flow)


def to_workflow(flow: Flow, *, nodes: set[str] | None = None, force: set[str] = frozenset(), token: str = "",
                max_parallel: int = 1) -> Workflow:
    """The engine workflow of ``flow`` (only ``nodes`` when given); ``force`` nodes get a fresh token in their
    parameters so their hash changes and they run again."""
    steps = []
    for nid in flow.order():
        if nodes is not None and nid not in nodes:
            continue
        n = flow.node(nid)
        ins: dict[str, list[list[str]]] = {}
        for e in flow.incoming(nid):
            ins.setdefault(e.target_port, []).append([e.source, e.source_port])
        meta: dict[str, Any] = {"flow_id": flow.id}
        needs = flow.upstream_ids(nid)
        if n.type in COLLECTORS:
            # a report collects the whole flow: it runs after every node that is not downstream of it, and gets
            # the flow-level decisions (planner, user edits)
            later = flow.downstream(nid) | {nid}
            needs = list(dict.fromkeys(needs + [x for x in flow.order() if x not in later
                                                and (nodes is None or x in nodes)]))
            meta["flow_decisions"] = _flow_decisions(flow)
            meta["flow_name"] = flow.name
        params = {"node_type": n.type, "params": n.params, "inputs": ins, "meta": meta}
        if nid in force:
            params["force"] = token
        steps.append(Step(id=nid, kind=ADAPTER_KIND, params=params, needs=needs))
    return Workflow(flow.name or "flow", steps, max_parallel=max_parallel, metadata={"flow_id": flow.id})


@dataclass
class FlowRunResult:
    run_id: str
    status: str
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    workdir: str = ""
    seconds: float = 0.0
    hooks: dict[str, str] = field(default_factory=dict)     # what each after-run hook did (on_run_finished)

    def output(self, node: str, port: str):
        return ((self.nodes.get(node) or {}).get("outputs") or {}).get(port)

    def load(self, node: str, port: str):
        return load_artifact(self.output(node, port))

    def report(self, node: str) -> dict[str, Any]:
        return (self.nodes.get(node) or {}).get("report") or {}

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "status": self.status, "nodes": self.nodes, "workdir": self.workdir,
                "seconds": self.seconds, "hooks": self.hooks}


def node_states(runner: WorkflowRunner, run_id: str, *, with_logs: bool = False) -> dict[str, dict[str, Any]]:
    out = {}
    for s in runner.steps(run_id):
        outputs = s.get("outputs") or {}
        status = STATUS.get(s["status"], s["status"])
        state = (outputs.get("report") or {}).get("state")
        if state in ("not_run", "degraded") and status in ("done", "cached"):
            status = state                                  # did not run / ran on a declared fallback
        d = {"status": status, "error": s.get("error"),
             "seconds": s.get("seconds"), "attempts": s.get("attempts"), "outputs": outputs.get("ports") or {},
             "report": outputs.get("report") or {}, "workdir": str(runner.step_workdir(run_id, s["step_id"])),
             "reused_from": s.get("reused_from")}
        if with_logs:
            d["log"] = runner.log(run_id, s["step_id"])[-20000:]
        out[s["step_id"]] = d
    return out


def register_custom_nodes(docs: list[dict[str, Any]]) -> list[str]:
    """Register user-defined node types (validated by :func:`pinneapple_flow.libcalls.custom_spec`: only
    allowlisted library calls); returns their types."""
    from .libcalls import custom_spec
    out = []
    for d in docs or []:
        REGISTRY.register_custom(custom_spec(d))
        out.append(d["type"])
    return out


def run(flow: Flow | dict | str | Path, *, workdir: str | Path | None = None, from_node: str | None = None,
        only: list[str] | None = None, run_id: str | None = None, cache: bool = True, max_parallel: int = 1,
        on_update: Callable[[str, dict[str, dict[str, Any]]], None] | None = None,
        should_stop: Callable[[], bool] = lambda: False, store=None, flow_version: int | None = None
        ) -> FlowRunResult:
    """Run a flow headless and return its :class:`FlowRunResult`.

    ``from_node``: run that node and everything downstream (and its ancestors, reused from the cache when unchanged).
    ``only``: run just these nodes and their ancestors. ``on_update(run_id, node_states)`` is called while it runs.
    ``store``: a :class:`pinneapple_flow.store.FlowStore` recording the run and each node's outputs."""
    flow = as_flow(flow)
    register_custom_nodes(flow.meta.get("custom_nodes") or [])
    if store is not None:
        register_custom_nodes(store.custom_nodes())
    flow = check_flow(flow)
    root = Path(workdir) if workdir else default_workdir()
    runner = WorkflowRunner(root / "engine", cache=cache)
    nodes = None
    force: set[str] = set()
    if from_node:
        if from_node not in [n.id for n in flow.nodes]:
            raise KeyError(f"unknown node {from_node!r}")
        force = {from_node} | flow.downstream(from_node)
        nodes = force | flow.ancestors(from_node)
        for d in list(force):
            nodes |= flow.ancestors(d)
    if only:
        nodes = set(only)
        for o in only:
            nodes |= flow.ancestors(o)
    run_id = run_id or f"flow-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    wf = to_workflow(flow, nodes=nodes, force=force, token=run_id, max_parallel=max_parallel)
    runner.create(wf, run_id, metadata={"flow_id": flow.id, "from_node": from_node})
    if store is not None:
        store.record_run(run_id, flow, version=flow_version, status="running", workdir=str(root))
    t0 = time.time()
    last = [0.0]
    lock = threading.Lock()

    def progress(_counts):
        if on_update is None and store is None:
            return
        now = time.time()
        with lock:
            if now - last[0] < 0.3:
                return
            last[0] = now
        states = node_states(runner, run_id)
        if store is not None:
            store.update_nodes(run_id, states)
        if on_update is not None:
            on_update(run_id, states)

    summary = runner.run(run_id=run_id, should_stop=should_stop, progress=progress)
    states = node_states(runner, run_id, with_logs=True)
    status = summary["status"]
    if store is not None:
        store.update_nodes(run_id, states)
        store.record_run_decisions(run_id, flow.id, states)
        store.finish_run(run_id, status)
    if on_update is not None:
        on_update(run_id, states)
    result = FlowRunResult(run_id=run_id, status=status, nodes=states, workdir=str(root),
                           seconds=time.time() - t0)
    on_run_finished(result, flow)
    return result


# ---------------------------------------------------------------------------------------------- after a run
RUN_FINISHED_HOOKS: list[Callable[[FlowRunResult, Flow], None]] = []


def on_run_finished(result: FlowRunResult, flow: Flow) -> None:
    """The one place a finished run is handed on (headless, service and GUI runs all pass here). The export of each
    run as a PINNeAPPle Lab experiment (``pinneapple_lab.experiment``, so flow runs appear on the experiments site
    with their seal) plugs in here once that API is on main; until then :func:`lab_export_status` says it is not
    available and nothing is written. A hook that fails is logged in ``result.hooks`` and never fails the run."""
    result.hooks = {}
    for hook in [export_to_lab, *RUN_FINISHED_HOOKS]:
        name = getattr(hook, "__name__", repr(hook))
        try:
            result.hooks[name] = hook(result, flow) or "done"
        except Exception as exc:  # noqa: BLE001 - a hook never breaks a run that already finished
            result.hooks[name] = f"failed: {type(exc).__name__}: {exc}"


def lab_export_status() -> str:
    try:
        from pinneapple_lab import manifest
    except ImportError:
        return "pinneapple_lab.manifest is not available"
    return "available" if callable(getattr(manifest, "from_flow_run", None)) else \
        "pinneapple_lab.manifest.from_flow_run is not available"


def _flow_manifest_run(result: FlowRunResult, flow: Flow) -> dict[str, Any]:
    """The run as the dict :func:`pinneapple_lab.manifest.from_flow_run` takes: nodes, the library modules they
    call, parameters, measured metrics (numbers only, each with how it was measured), checks, figures / videos
    (relative to the run folder) and the decision records with their references (``decisions.json``)."""
    import re as _re

    from .decisions import collect
    out = Path(result.workdir)
    rel = lambda p: os.path.relpath(p, out) if Path(p).resolve().is_relative_to(out.resolve()) else None  # noqa: E731
    metrics, checks, figures, videos, modules = [], [], [], [], set()
    for nid, st in result.nodes.items():
        n = flow.node(nid)
        spec = REGISTRY.get(n.type)
        if spec is not None:
            modules |= set(_re.findall(r"\bpinneapple[a-z_]*(?:\.[a-z_]+)*", spec.calls or ""))
        rep = st.get("report") or {}
        for k, v in (rep.get("metrics") or {}).items():
            val, how = (v.get("value"), v.get("how", "")) if isinstance(v, dict) else (v, "")
            if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val)):
                metrics.append({"name": f"{nid}.{k}", "value": val, "unit": "",
                                "how": how or f"measured by the {n.type} node"})
        for port in (st.get("outputs") or {}).values():
            if isinstance(port, dict) and isinstance(port.get("checks"), list):
                checks += [{"name": f"{nid}.{c.get('name')}", "passed": bool(c.get("passed")),
                            **{k: c[k] for k in ("value", "detail", "kind") if k in c}} for c in port["checks"]
                           if isinstance(c, dict) and "name" in c]
        figures += [r for f in rep.get("figures") or [] if (r := rel(f.get("path", "")))]
        vid = (rep.get("views") or {}).get("video")
        if vid and (r := rel(vid.get("path", ""))):
            videos.append(r)
    decisions = collect(flow, {k: v.get("report") or {} for k, v in result.nodes.items()})
    (out / "decisions.json").write_text(json.dumps(decisions, indent=1, default=str))
    refs = sorted({r.get("key") or r.get("link") for d in decisions for r in d.get("references", [])
                   if r.get("key") or r.get("link")})
    return {"id": f"flow-{flow.id}-{result.run_id}", "title": flow.name or flow.id, "description": flow.description,
            "nodes": [{"type": n.type} for n in flow.nodes], "modules": sorted(modules),
            "params": {n.id: n.params for n in flow.nodes}, "metrics": metrics, "checks": checks,
            "files": {"figures": figures, "videos": videos, "data": ["decisions.json"]},
            "links": {"flow_id": flow.id, "run_id": result.run_id}, "references": refs}


def export_to_lab(result: FlowRunResult, flow: Flow) -> str:
    """Write the run's experiment manifest (``experiment.json`` in the run folder) with the Lab's own
    ``pinneapple_lab.manifest.from_flow_run``, so flow runs appear on the experiments site."""
    status = lab_export_status()
    if status != "available":
        return f"skipped: {status}"
    if result.status != "done":
        return f"skipped: the run is {result.status}"
    from pinneapple_lab import manifest as M
    run = _flow_manifest_run(result, flow)
    m = M.from_flow_run(run)
    m.references = run["references"]
    path = M.save(m, result.workdir)
    problems = M.validate(m, base=result.workdir)
    return f"manifest {path}" + (f" ({len(problems)} issues: {problems[:3]})" if problems else "")

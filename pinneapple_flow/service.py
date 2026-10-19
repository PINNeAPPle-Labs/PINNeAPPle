"""The operations Flow Studio (or any other front end, e.g. the PINNeAPPle-CFD app) exposes, as plain methods.

The app's HTTP routes are one-liners over :class:`FlowService`: planning, validation, the node catalog, flow
storage with versions, background runs with live per-node status, node results, architecture and equation previews,
and safe access to run artifacts (figures, 3-D scenes) inside the run folder only.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .config import Settings, home
from .registry import REGISTRY
from .schema import Flow, FlowError, as_flow, flow_json_schema, validate_flow
from .types import CONVERSIONS, PORT_TYPES

FLOWS_DIR = Path(__file__).parent / "flows"


class FlowService:
    def __init__(self, *, db_url: str | None = None, workdir: str | Path | None = None,
                 settings: Settings | None = None, max_runs: int = 1):
        from .store import FlowStore
        self.settings = settings or Settings.from_env()
        self.store = FlowStore(db_url or self.settings.db_url or None)
        self.workdir = Path(workdir) if workdir else home() / "work"
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._pool = ThreadPoolExecutor(max_workers=max(1, max_runs), thread_name_prefix="flow-run")
        self._live: dict[str, dict[str, Any]] = {}
        self._stop: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        from .runtime import register_custom_nodes
        for d in self.store.custom_nodes():
            try:
                register_custom_nodes([d])
            except ValueError:
                pass                                    # a stale definition (its base call was removed)

    # -- catalog, schema ------------------------------------------------------------------------------------------
    def catalog(self) -> dict[str, Any]:
        return {"nodes": REGISTRY.catalog(), "port_types": PORT_TYPES,
                "conversions": sorted([list(c) for c in CONVERSIONS])}

    # -- palette: sections, search, per-user favourites and recent node types -----------------------------------
    PALETTE_LIMITS = {"favourites": 60, "recent": 12}

    def palette(self, user: str = "local") -> dict[str, Any]:
        from .palette import layout
        return {"sections": layout(REGISTRY.catalog()), **self.palette_prefs(user)}

    def palette_search(self, q: str, limit: int = 50) -> list[dict[str, Any]]:
        from .palette import search
        return [{"type": e["type"], "section": e["palette"]["section"]} for e in search(q[:200], limit=limit)]

    def palette_prefs(self, user: str = "local") -> dict[str, list[str]]:
        return {k: [t for t in self.store.get_pref(user, f"palette.{k}", []) if REGISTRY.get(t) is not None]
                for k in self.PALETTE_LIMITS}

    def set_palette_prefs(self, user: str, prefs: dict[str, Any]) -> dict[str, list[str]]:
        """Store ``favourites`` and/or ``recent`` (lists of registered node types; unknown ones are dropped)."""
        for k, n in self.PALETTE_LIMITS.items():
            if k in prefs:
                v = prefs[k]
                if not isinstance(v, list):
                    raise FlowError(f"{k} must be a list of node types")
                seen: list[str] = []
                for t in v:
                    if isinstance(t, str) and REGISTRY.get(t) is not None and t not in seen:
                        seen.append(t)
                self.store.set_pref(user, f"palette.{k}", seen[:n])
        return self.palette_prefs(user)

    def schema(self) -> dict[str, Any]:
        return flow_json_schema()

    def info(self) -> dict[str, Any]:
        from . import __version__
        return {"version": __version__, "settings": self.settings.public(),
                "planners": ["heuristic"] + ([f"llm:{self.settings.llm_provider}"]
                                             if self.settings.llm_provider not in ("", "none") else [])}

    # -- planning and validation ----------------------------------------------------------------------------------
    def plan(self, text: str, backend: str = "auto", **overrides) -> dict[str, Any]:
        from .planner import plan
        s = self.settings
        kw = {"provider": overrides.get("provider") or (s.llm_provider if s.llm_provider != "none" else ""),
              "model": overrides.get("model") or s.llm_model, "base_url": overrides.get("base_url") or s.llm_base_url,
              "api_key": overrides.get("api_key") or s.llm_api_key}
        if backend == "auto":
            backend = "llm" if kw["provider"] else "heuristic"
        return plan(text, backend=backend, **kw)

    def validate(self, flow: dict[str, Any]) -> dict[str, Any]:
        issues = validate_flow(flow)
        return {"valid": not any(i.level == "error" for i in issues), "issues": [i.to_dict() for i in issues]}

    def equations_preview(self, geometry: dict[str, Any], equations: dict[str, Any]) -> dict[str, Any]:
        """Parse (safely) what the equations panel holds, for live feedback while the user edits."""
        from pinneapple_physics.symbolic_pde.text import EquationSyntaxError, parse_system

        from .nodes.problem import geometry_doc
        from .templates import template
        try:
            geo = geometry_doc({**REGISTRY["geometry.domain"].defaults(), **(geometry or {})})
        except Exception as exc:  # noqa: BLE001 - shown to the user
            return {"ok": False, "error": f"domain: {exc}"}
        eq = {**REGISTRY["equations.system"].defaults(), **(equations or {})}
        if eq.get("template", "custom") != "custom" and (eq.get("follow_template") or not eq.get("pdes")):
            try:
                t = template(eq["template"], geo["dimension"])
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            eq = {**eq, "pdes": t["pdes"], "conditions": t["conditions"], "fields": t["fields"],
                  "coefficients": {**t["coefficients"], **(eq.get("coefficients") or {})}, "exact": t["exact"] or ""}
        try:
            sys_ = parse_system(eq["pdes"], eq.get("conditions") or [], coords=geo["coords"],
                                fields=eq.get("fields") or ["u"], coefficients=eq.get("coefficients") or {},
                                unknowns=eq.get("unknowns") or [])
        except EquationSyntaxError as exc:
            return {"ok": False, "error": str(exc)}
        doc = sys_.to_dict()
        return {"ok": not doc["issues"], "system": doc, "resolved": eq, "geometry": geo}

    def architecture(self, kind: str, config: dict[str, Any] | None = None, in_dim: int = 2,
                     out_dim: int = 64, n_fields: int = 1) -> dict[str, Any]:
        from . import learn
        cfg = {**learn.default_config(kind, int(in_dim), int(out_dim)), **(config or {}), "kind": kind}
        return learn.architecture(cfg, int(in_dim), int(out_dim), n_fields=int(n_fields))

    # -- decisions and references ---------------------------------------------------------------------------------
    def decisions(self, flow: dict[str, Any], run_id: str | None = None) -> dict[str, Any]:
        """Every decision of a flow (planner, LLM, user edits) and of a run's nodes, grouped by node."""
        from .decisions import by_node, collect, summary
        reports = {}
        if run_id:
            try:
                reports = {k: v["report"] for k, v in self.store.get_run(run_id)["nodes"].items()}
            except KeyError:
                reports = {}
        ds = collect(as_flow(flow), reports)
        return {"decisions": ds, "by_node": by_node(ds), "summary": summary(ds)}

    def references(self) -> list[dict[str, Any]]:
        from .references import link, registry
        return [{**r, "link": link(r)} for r in registry().values()]

    def check_reference(self, reference: dict[str, Any]) -> dict[str, Any]:
        from .references import check
        return check(reference)

    # -- Lab seal and how to raise it --------------------------------------------------------------------------
    def lab_root(self) -> str:
        return str(home() / "lab")

    def seal(self, item: str) -> dict[str, Any]:
        from .lab_status import seal_status
        return seal_status(item, self.lab_root())

    def next_steps(self, item: str) -> dict[str, Any]:
        from .lab_status import next_steps
        return next_steps(item, self.lab_root())

    def insert_nodes(self, flow: dict[str, Any], anchor: str | None, items: list[dict[str, Any]]) -> dict[str, Any]:
        """Add registered node types to a flow, wired by port type from ``anchor`` (the GUI's "add node" buttons
        of a seal gap or a low-confidence decision)."""
        from .lab_status import insert_nodes
        for it in items:
            if not isinstance(it, dict) or it.get("type") not in REGISTRY:
                raise ValueError(f"not a registered node type: {it!r}")
        doc, missing = insert_nodes(flow, anchor, items)
        return {"flow": doc, "unwired": missing, **self.validate(doc)}

    # -- library calls and user-defined node types -------------------------------------------------------------
    def library_calls(self) -> list[dict[str, Any]]:
        from .libcalls import CALLS, introspect
        out = []
        for c in CALLS.values():
            try:
                params, skipped = introspect(c)
            except Exception:  # noqa: BLE001 - not importable here
                params, skipped = [], []
            out.append({"key": c.key, "target": c.target, "title": c.title, "category": c.category,
                        "params": [p.name for p in params], "not_exposed": skipped})
        return out

    def custom_nodes(self) -> list[dict[str, Any]]:
        return self.store.custom_nodes()

    def save_custom_node(self, doc: dict[str, Any]) -> dict[str, Any]:
        from .runtime import register_custom_nodes
        register_custom_nodes([doc])                    # validates: ValueError on anything not allowlisted
        self.store.save_custom_node(doc)
        return REGISTRY[doc["type"]].to_dict()

    def delete_custom_node(self, type_: str) -> None:
        self.store.delete_custom_node(type_)
        REGISTRY.unregister_custom(type_)

    # -- flows ----------------------------------------------------------------------------------------------------
    def save_flow(self, flow: dict[str, Any], note: str = "") -> dict[str, Any]:
        f = as_flow(flow)
        v = self.store.save_flow(f, note)
        return {"id": f.id, "version": v}

    def get_flow(self, flow_id: str, version: int | None = None) -> dict[str, Any]:
        return self.store.get_flow(flow_id, version)

    def list_flows(self) -> list[dict[str, Any]]:
        return self.store.list_flows()

    def versions(self, flow_id: str) -> list[dict[str, Any]]:
        return self.store.list_versions(flow_id)

    def delete_flow(self, flow_id: str) -> None:
        self.store.delete_flow(flow_id)

    def examples(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted(FLOWS_DIR.glob("*.json")):
            d = json.loads(p.read_text())
            out.append({"name": p.stem, "title": d.get("name"), "description": d.get("description", ""),
                        "nodes": len(d.get("nodes", []))})
        return out

    def example(self, name: str) -> dict[str, Any]:
        p = (FLOWS_DIR / f"{name}.json").resolve()
        if p.parent != FLOWS_DIR.resolve() or not p.is_file():
            raise KeyError(f"unknown example {name!r}")
        return json.loads(p.read_text())

    # -- runs -----------------------------------------------------------------------------------------------------
    def start_run(self, flow: dict[str, Any], *, from_node: str | None = None, save: bool = True) -> dict[str, Any]:
        from .runtime import run
        from .schema import check_flow
        f = check_flow(flow)
        version = self.store.save_flow(f, "run") if save else None
        run_id = f"flow-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        stop = threading.Event()
        with self._lock:
            self._stop[run_id] = stop
            self._live[run_id] = {"run_id": run_id, "flow_id": f.id, "status": "queued", "nodes": {
                n.id: {"status": "queued"} for n in f.nodes}, "started": time.time()}

        def update(rid, states):
            with self._lock:
                live = self._live.get(rid)
                if live is not None:
                    live["status"] = "running"
                    live["nodes"].update({k: _brief(v) for k, v in states.items()})

        def job():
            try:
                res = run(f, workdir=self.workdir, from_node=from_node, run_id=run_id, on_update=update,
                          should_stop=stop.is_set, store=self.store, flow_version=version)
                status = res.status
            except Exception as exc:  # noqa: BLE001 - reported to the GUI
                status = "failed"
                with self._lock:
                    self._live[run_id]["error"] = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._live[run_id]["status"] = status
                self._live[run_id]["finished"] = time.time()

        self._pool.submit(job)
        return {"run_id": run_id, "flow_id": f.id, "version": version}

    def cancel(self, run_id: str) -> bool:
        ev = self._stop.get(run_id)
        if ev is None:
            return False
        ev.set()
        return True

    def run_status(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            live = self._live.get(run_id)
            if live is not None:
                return json.loads(json.dumps(live, default=str))
        r = self.store.get_run(run_id)
        return {"run_id": run_id, "flow_id": r["flow_id"], "status": r["status"],
                "nodes": {k: _brief(v) for k, v in r["nodes"].items()}}

    def runs(self, flow_id: str | None = None) -> list[dict[str, Any]]:
        return self.store.list_runs(flow_id)

    def node_result(self, run_id: str, node_id: str) -> dict[str, Any]:
        r = self.store.get_run(run_id)
        n = r["nodes"].get(node_id)
        if n is None:
            live = self._live.get(run_id, {}).get("nodes", {}).get(node_id)
            if live is None:
                raise KeyError(f"node {node_id!r} has no result in run {run_id!r}")
            return {"node": node_id, **live}
        log = ""
        wd = n.get("workdir")
        if wd and (Path(wd) / "log.txt").is_file():
            log = (Path(wd) / "log.txt").read_text(errors="replace")[-30000:]
        return {"node": node_id, "status": n["status"], "error": n["error"], "seconds": n["seconds"],
                "report": n["report"], "outputs": n["outputs"], "log": log, "lab": self._run_lab(r)}

    def _run_lab(self, run: dict[str, Any]) -> list[dict[str, Any]]:
        """The current seal of every Lab item the run's validate.lab nodes recorded (shown on each result panel)."""
        from .lab_status import lab_items
        return [self.seal(i) for i in lab_items({k: v.get("report") or {} for k, v in run["nodes"].items()})]

    def artifact_path(self, path: str) -> Path:
        """A file of a run (figure, scene, report): only inside the run folders or the Lab folder."""
        p = Path(path).resolve()
        roots = [self.workdir.resolve(), (home() / "lab").resolve()]
        if not any(r == p or r in p.parents for r in roots) or not p.is_file():
            raise PermissionError("not a run artifact")
        return p

    def html_artifact(self, path: str) -> Path:
        """An HTML report of a run, to be shown in a sandboxed frame (same rules as :meth:`artifact_path`)."""
        p = self.artifact_path(path)
        if p.suffix.lower() not in (".html", ".htm"):
            raise PermissionError("not an HTML artifact")
        return p

    def shutdown(self):
        for ev in self._stop.values():
            ev.set()
        self._pool.shutdown(wait=False, cancel_futures=True)


def _brief(state: dict[str, Any]) -> dict[str, Any]:
    rep = state.get("report") or {}
    return {"status": state.get("status"), "error": state.get("error"), "seconds": state.get("seconds"),
            "summary": rep.get("summary", ""), "metrics": rep.get("metrics", {}),
            "warnings": rep.get("warnings", [])}


__all__ = ["FlowService", "FlowError", "Flow"]

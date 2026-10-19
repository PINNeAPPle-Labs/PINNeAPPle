"""Where flows, their versions, runs and per-node outputs live: SQLite by default, any SQLAlchemy URL otherwise.

    store = FlowStore()                                            # ~/.pinneapple_flow/flow.sqlite
    store = FlowStore("postgresql+psycopg://user:pw@host/db")      # Postgres (pip install psycopg)
    v = store.save_flow(flow, note="first draft")                  # -> version 1, 2, ...
    store.get_flow(flow.id)                                        # latest version (a flow document)
    store.get_run(run_id)["nodes"]["pinn"]["report"]["metrics"]

The URL comes from the argument, else ``PINNEAPPLE_FLOW_DB_URL``, else the SQLite file under
``PINNEAPPLE_FLOW_HOME`` (default ``~/.pinneapple_flow``). Artifacts (arrays, figures, weights) stay on disk in the run
folder; the store keeps their paths.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from .schema import Flow, as_flow


def default_url() -> str:
    url = os.environ.get("PINNEAPPLE_FLOW_DB_URL", "").strip()
    if url:
        return url
    home = Path(os.environ.get("PINNEAPPLE_FLOW_HOME", Path.home() / ".pinneapple_flow"))
    home.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{home / 'flow.sqlite'}"


def redact(url: str) -> str:
    """The URL with any password replaced (for logs and the GUI)."""
    import re
    return re.sub(r"(://[^:/@]+:)[^@]*@", r"\1***@", url)


_PREFIX = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


class FlowStore:
    """``table_prefix``: put the tables under a prefix (``flow_`` -> ``flow_flows``, ``flow_runs``, ...) when they
    share an application's database; default ``$PINNEAPPLE_FLOW_TABLE_PREFIX`` or none."""

    def __init__(self, url: str | None = None, *, table_prefix: str | None = None):
        import sqlalchemy as sa
        prefix = os.environ.get("PINNEAPPLE_FLOW_TABLE_PREFIX", "") if table_prefix is None else table_prefix
        if prefix and not _PREFIX.match(prefix):
            raise ValueError(f"table prefix {prefix!r}: lower-case letters, digits and _, starting with a letter")
        self.table_prefix = prefix
        self.url = url or default_url()
        kw: dict[str, Any] = {"future": True}
        if self.url.startswith("sqlite"):
            kw["connect_args"] = {"check_same_thread": False, "timeout": 30}
        self.engine = sa.create_engine(self.url, **kw)
        if self.url.startswith("sqlite"):
            @sa.event.listens_for(self.engine, "connect")
            def _wal(dbapi_conn, _rec):
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.close()
        md = sa.MetaData()
        self.flows = sa.Table(
            prefix + "flows", md, sa.Column("id", sa.String(64), primary_key=True),
            sa.Column("name", sa.String(200)),
            sa.Column("description", sa.Text), sa.Column("created", sa.Float), sa.Column("updated", sa.Float),
            sa.Column("latest_version", sa.Integer))
        self.versions = sa.Table(
            prefix + "flow_versions", md, sa.Column("flow_id", sa.String(64), primary_key=True),
            sa.Column("version", sa.Integer, primary_key=True), sa.Column("document", sa.Text),
            sa.Column("note", sa.String(500)), sa.Column("created", sa.Float))
        self.runs = sa.Table(
            prefix + "runs", md, sa.Column("id", sa.String(80), primary_key=True), sa.Column("flow_id", sa.String(64)),
            sa.Column("version", sa.Integer), sa.Column("status", sa.String(20)), sa.Column("created", sa.Float),
            sa.Column("finished", sa.Float), sa.Column("workdir", sa.Text), sa.Column("document", sa.Text))
        self.nodes = sa.Table(
            prefix + "node_results", md, sa.Column("run_id", sa.String(80), primary_key=True),
            sa.Column("node_id", sa.String(64), primary_key=True), sa.Column("status", sa.String(20)),
            sa.Column("outputs", sa.Text), sa.Column("report", sa.Text), sa.Column("error", sa.Text),
            sa.Column("seconds", sa.Float), sa.Column("workdir", sa.Text), sa.Column("updated", sa.Float))
        self.decisions = sa.Table(
            prefix + "decisions", md, sa.Column("id", sa.String(40), primary_key=True), sa.Column("flow_id", sa.String(64)),
            sa.Column("version", sa.Integer), sa.Column("run_id", sa.String(80)), sa.Column("node_id", sa.String(64)),
            sa.Column("topic", sa.String(40)), sa.Column("confidence", sa.Float), sa.Column("no_reference", sa.Boolean),
            sa.Column("record", sa.Text), sa.Column("created", sa.Float))
        self.custom = sa.Table(
            prefix + "custom_nodes", md, sa.Column("type", sa.String(64), primary_key=True), sa.Column("doc", sa.Text),
            sa.Column("created", sa.Float))
        self.prefs = sa.Table(
            prefix + "user_prefs", md, sa.Column("user", sa.String(120), primary_key=True),
            sa.Column("key", sa.String(64), primary_key=True), sa.Column("value", sa.Text), sa.Column("updated", sa.Float))
        md.create_all(self.engine)
        self.sa = sa

    # -- flows -----------------------------------------------------------------------------------------------------
    def save_flow(self, flow: Flow | dict, note: str = "") -> int:
        """Store a new version of the flow (created on first save); returns the version number. A document equal
        to the latest version creates nothing and returns that version."""
        flow = as_flow(flow)
        sa, now = self.sa, time.time()
        doc = flow.to_json(0)
        with self.engine.begin() as c:
            row = c.execute(sa.select(self.flows.c.latest_version).where(self.flows.c.id == flow.id)).first()
            if row:
                last = c.execute(sa.select(self.versions.c.document).where(
                    (self.versions.c.flow_id == flow.id) & (self.versions.c.version == row[0]))).first()
                if last and last[0] == doc:
                    return int(row[0])
            version = (row[0] if row else 0) + 1
            if row:
                c.execute(self.flows.update().where(self.flows.c.id == flow.id).values(
                    name=flow.name, description=flow.description, updated=now, latest_version=version))
            else:
                c.execute(self.flows.insert().values(id=flow.id, name=flow.name, description=flow.description,
                                                     created=now, updated=now, latest_version=version))
            c.execute(self.versions.insert().values(flow_id=flow.id, version=version, document=doc,
                                                    note=note[:500], created=now))
        from .decisions import collect
        self.record_decisions(flow.id, collect(flow), version=version)
        return version

    # -- per-user preferences (palette favourites and recent node types) ----------------------------------------
    def get_pref(self, user: str, key: str, default: Any = None) -> Any:
        with self.engine.connect() as c:
            row = c.execute(self.sa.select(self.prefs.c.value).where(
                (self.prefs.c.user == user) & (self.prefs.c.key == key))).first()
        return json.loads(row[0]) if row else default

    def set_pref(self, user: str, key: str, value: Any) -> None:
        with self.engine.begin() as c:
            c.execute(self.prefs.delete().where((self.prefs.c.user == user) & (self.prefs.c.key == key)))
            c.execute(self.prefs.insert().values(user=user, key=key, value=json.dumps(value), updated=time.time()))

    # -- user-defined node types -----------------------------------------------------------------------------------
    def save_custom_node(self, doc: dict[str, Any]) -> None:
        with self.engine.begin() as c:
            c.execute(self.custom.delete().where(self.custom.c.type == doc["type"]))
            c.execute(self.custom.insert().values(type=doc["type"], doc=json.dumps(doc), created=time.time()))

    def custom_nodes(self) -> list[dict[str, Any]]:
        with self.engine.connect() as c:
            return [json.loads(r[0]) for r in c.execute(self.sa.select(self.custom.c.doc))]

    def delete_custom_node(self, type_: str) -> None:
        with self.engine.begin() as c:
            c.execute(self.custom.delete().where(self.custom.c.type == type_))

    # -- decisions -------------------------------------------------------------------------------------------------
    def record_decisions(self, flow_id: str, decisions: list[dict[str, Any]], *, version: int | None = None,
                         run_id: str | None = None) -> None:
        """Keep decision records (pinneapple_flow.decisions) with the flow version or the run they belong to."""
        sa, now = self.sa, time.time()
        with self.engine.begin() as c:
            for d in decisions:
                rid = f"{run_id or 'v' + str(version)}:{d['id']}"[:40]
                c.execute(self.decisions.delete().where(self.decisions.c.id == rid))
                c.execute(self.decisions.insert().values(
                    id=rid, flow_id=flow_id, version=version, run_id=run_id, node_id=d.get("node") or "",
                    topic=d.get("topic"), confidence=d.get("confidence"), no_reference=bool(d.get("no_reference")),
                    record=json.dumps(d, default=str), created=now))
        del sa

    def get_decisions(self, flow_id: str, *, version: int | None = None, run_id: str | None = None
                      ) -> list[dict[str, Any]]:
        sa = self.sa
        q = sa.select(self.decisions.c.record).where(self.decisions.c.flow_id == flow_id)
        if version is not None:
            q = q.where(self.decisions.c.version == version)
        if run_id is not None:
            q = q.where(self.decisions.c.run_id == run_id)
        with self.engine.connect() as c:
            return [json.loads(r[0]) for r in c.execute(q.order_by(self.decisions.c.created))]

    def get_flow(self, flow_id: str, version: int | None = None) -> dict[str, Any]:
        sa = self.sa
        q = sa.select(self.versions.c.document, self.versions.c.version).where(self.versions.c.flow_id == flow_id)
        q = q.where(self.versions.c.version == version) if version else q.order_by(self.versions.c.version.desc())
        with self.engine.connect() as c:
            row = c.execute(q.limit(1)).first()
        if row is None:
            raise KeyError(f"unknown flow {flow_id!r}" + (f" version {version}" if version else ""))
        doc = json.loads(row[0])
        doc.setdefault("meta", {})["version"] = row[1]
        return doc

    def list_flows(self) -> list[dict[str, Any]]:
        sa = self.sa
        with self.engine.connect() as c:
            rows = c.execute(sa.select(self.flows).order_by(self.flows.c.updated.desc())).mappings().all()
        return [dict(r) for r in rows]

    def list_versions(self, flow_id: str) -> list[dict[str, Any]]:
        sa = self.sa
        q = sa.select(self.versions.c.version, self.versions.c.note, self.versions.c.created).where(
            self.versions.c.flow_id == flow_id).order_by(self.versions.c.version.desc())
        with self.engine.connect() as c:
            return [dict(r) for r in c.execute(q).mappings().all()]

    def delete_flow(self, flow_id: str) -> None:
        with self.engine.begin() as c:
            c.execute(self.versions.delete().where(self.versions.c.flow_id == flow_id))
            c.execute(self.flows.delete().where(self.flows.c.id == flow_id))

    # -- runs ------------------------------------------------------------------------------------------------------
    def record_run(self, run_id: str, flow: Flow, *, version: int | None = None, status: str = "running",
                   workdir: str = "") -> None:
        with self.engine.begin() as c:
            c.execute(self.runs.insert().values(id=run_id, flow_id=flow.id, version=version, status=status,
                                                created=time.time(), workdir=workdir, document=flow.to_json(0)))

    def update_nodes(self, run_id: str, states: dict[str, dict[str, Any]]) -> None:
        sa, now = self.sa, time.time()
        with self.engine.begin() as c:
            existing = {r[0] for r in c.execute(sa.select(self.nodes.c.node_id).where(self.nodes.c.run_id == run_id))}
            for nid, s in states.items():
                vals = dict(status=s.get("status"), outputs=json.dumps(s.get("outputs") or {}),
                            report=json.dumps(s.get("report") or {}), error=s.get("error"),
                            seconds=s.get("seconds"), workdir=s.get("workdir"), updated=now)
                if nid in existing:
                    c.execute(self.nodes.update().where((self.nodes.c.run_id == run_id)
                                                        & (self.nodes.c.node_id == nid)).values(**vals))
                else:
                    c.execute(self.nodes.insert().values(run_id=run_id, node_id=nid, **vals))

    def record_run_decisions(self, run_id: str, flow_id: str, states: dict[str, dict[str, Any]]) -> None:
        decs = []
        for nid, s in states.items():
            for d in (s.get("report") or {}).get("decisions") or []:
                decs.append({**d, "node": d.get("node") or nid})
        if decs:
            self.record_decisions(flow_id, decs, run_id=run_id)

    def finish_run(self, run_id: str, status: str) -> None:
        with self.engine.begin() as c:
            c.execute(self.runs.update().where(self.runs.c.id == run_id).values(status=status, finished=time.time()))

    def get_run(self, run_id: str) -> dict[str, Any]:
        sa = self.sa
        with self.engine.connect() as c:
            row = c.execute(sa.select(self.runs).where(self.runs.c.id == run_id)).mappings().first()
            if row is None:
                raise KeyError(f"unknown run {run_id!r}")
            nodes = c.execute(sa.select(self.nodes).where(self.nodes.c.run_id == run_id)).mappings().all()
        d = dict(row)
        d["document"] = json.loads(d["document"]) if d.get("document") else None
        d["nodes"] = {n["node_id"]: {"status": n["status"], "outputs": json.loads(n["outputs"] or "{}"),
                                     "report": json.loads(n["report"] or "{}"), "error": n["error"],
                                     "seconds": n["seconds"], "workdir": n["workdir"]} for n in nodes}
        return d

    def list_runs(self, flow_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        sa = self.sa
        q = sa.select(self.runs.c.id, self.runs.c.flow_id, self.runs.c.version, self.runs.c.status,
                      self.runs.c.created, self.runs.c.finished)
        if flow_id:
            q = q.where(self.runs.c.flow_id == flow_id)
        with self.engine.connect() as c:
            return [dict(r) for r in c.execute(q.order_by(self.runs.c.created.desc()).limit(limit)).mappings()]

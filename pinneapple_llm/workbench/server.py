"""The workbench web app: FastAPI routes over :class:`~.runtime.Session` and the :class:`~.store.Store`, plus the
static single-page frontend in ``static/`` (no build step).

The database is a local SQLite file or a PostgreSQL server (``db="postgresql://..."``); every conversation and
model call is kept there for fine-tuning. It binds to 127.0.0.1 by default: the agents run commands on your machine, so do not expose it to a network. Set
``token`` to require ``Authorization: Bearer <token>`` (or ``?token=``) on every API call.
"""
import asyncio
import json
import threading
from dataclasses import asdict
from pathlib import Path

from .agents import USER_DIR, load_agents, parse_agent
from .backend import Backend, make_backend
from .models import detect_machine, recommend
from .runtime import Session, SessionSettings
from .store import DEFAULT_DB, Store
from .tools import ApprovalPolicy

__all__ = ["create_app", "Manager"]

STATIC = Path(__file__).with_name("static")


class Manager:
    """Live sessions (one worker thread per running turn) over the store."""

    def __init__(self, store: Store, backend_factory, default_workspace: str, language: str = ""):
        self.store, self.backend_factory = store, backend_factory
        self.default_workspace, self.language = default_workspace, language
        self.live: dict[str, Session] = {}
        self.busy: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def session(self, sid: str) -> Session:
        with self._lock:
            if sid in self.live:
                return self.live[sid]
            row = self.store.get(sid)
            if row is None:
                raise KeyError(sid)
            s = row["settings"]
            sess = Session(self.backend_factory(s.get("model") or None), row["workspace"],
                           policy=ApprovalPolicy(s.get("policy", "auto")),
                           settings=SessionSettings(language=s.get("language", self.language)),
                           on_event=lambda e, sid=sid: self.store.add_event(sid, e), history=row["history"],
                           on_model_call=lambda c, sid=sid: self.store.add_model_call(sid, c),
                           hooks=self._hooks(sid), web=s.get("web", True))
            self.live[sid] = sess
            return sess

    # -- tools that reach beyond one session ---------------------------------------------------------------------
    def _hooks(self, sid: str) -> dict:
        def schedule(minutes: float, message: str) -> str:
            t = threading.Timer(max(0.0, float(minutes)) * 60, self._deliver, args=(sid, f"(scheduled) {message}"))
            t.daemon = True
            t.start()
            return f"will send this message back in {float(minutes):g} min"

        def start_session(title: str, task: str, workspace: str = "") -> str:
            parent = self.store.get(sid) or {}
            ws = workspace or parent.get("workspace") or self.default_workspace
            new = self.store.create(str(Path(ws).expanduser().resolve()), dict(parent.get("settings") or {}), title)
            self.send(new, task)
            return f"started session {new} ({title})"

        def send_to_session(session_id: str, message: str) -> str:
            self._deliver(session_id, message)
            return f"sent to {session_id}"

        def list_sessions() -> str:
            return "\n".join(f"{r['id']}  {'working' if self.is_busy(r['id']) else 'idle':8s} {r['title']}"
                             for r in self.store.list()) or "(none)"

        def session_status(session_id: str) -> str:
            if self.store.get(session_id) is None:
                return f"no session {session_id}"
            live = self.live.get(session_id)
            msgs = [e for e in self.store.events(session_id) if e["type"] in ("message", "error")
                    and not e.get("depth")][-3:]
            plan = "\n".join(f"[{i['status']}] {i['text']}" for i in (live.plan if live else []))
            return (f"{'working' if self.is_busy(session_id) else 'idle'}\nplan:\n{plan or '(none)'}\nlast:\n"
                    + "\n---\n".join(m.get("content", "")[:1500] for m in msgs))

        return {"schedule": schedule, "start_session": start_session, "send_to_session": send_to_session,
                "list_sessions": list_sessions, "session_status": session_status}

    def _deliver(self, sid: str, text: str) -> None:
        """Send now, or as soon as the session's current turn ends."""
        def go():
            th = self.busy.get(sid)
            if th is not None:
                th.join()
            try:
                self.send(sid, text)
            except (KeyError, RuntimeError) as e:
                self.store.add_event(sid, {"type": "error", "content": f"could not deliver a message: {e}"})
        threading.Thread(target=go, daemon=True).start()

    def is_busy(self, sid: str) -> bool:
        return sid in self.busy and self.busy[sid].is_alive()

    def send(self, sid: str, text: str) -> None:
        sess = self.session(sid)
        if self.is_busy(sid):
            raise RuntimeError("this session is already working; wait or stop it")
        row = self.store.get(sid)
        if row and row["title"] == "New session":
            self.store.update(sid, title=text.strip().splitlines()[0][:80])

        def work():
            try:
                sess.send(text)
            except Exception as e:  # noqa: BLE001 - the error is shown in the session
                self.store.add_event(sid, {"type": "error", "content": f"{type(e).__name__}: {e}"})
                self.store.add_event(sid, {"type": "done", "usage": dict(sess.usage)})
            finally:
                self.store.update(sid, history=sess.history)

        th = threading.Thread(target=work, daemon=True)
        self.busy[sid] = th
        th.start()


def create_app(*, backend: str = "ollama", model: str = "", host: str = "", base_url: str = "", api_key: str = "",
               workspace: str = ".", db: str | Path = DEFAULT_DB, token: str = "", language: str = "",
               backend_obj: Backend | None = None):
    from fastapi import Depends, FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    rec = recommend()
    default_model = model or rec.model
    num_ctx = rec.context

    def factory(m: str | None = None) -> Backend:
        if backend_obj is not None:
            return backend_obj
        kw = {"num_ctx": num_ctx}
        if host:
            kw["host"] = host
        if base_url:
            kw["base_url"] = base_url
        if api_key:
            kw["api_key"] = api_key
        return make_backend(backend, m or default_model, **kw)

    store = Store(db)                # a SQLite path or a postgresql:// URL
    mgr = Manager(store, factory, str(Path(workspace).expanduser().resolve()), language)
    app = FastAPI(title="PINNeAPPle Agent Workbench")
    app.state.manager = mgr

    def auth(request: Request):
        if not token:
            return
        got = request.headers.get("authorization", "").removeprefix("Bearer ").strip() or \
            request.query_params.get("token", "")
        if got != token:
            raise HTTPException(401, "missing or wrong token")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/api/info", dependencies=[Depends(auth)])
    def info():
        be = factory()
        agents, _ = load_agents(mgr.default_workspace)
        return {"machine": asdict(detect_machine()), "recommended": asdict(rec), "backend": backend,
                "model": default_model, "installed_models": be.models(), "workspace": mgr.default_workspace,
                "language": language,
                "agents": [{"name": a.name, "title": a.title, "description": a.description, "tools": a.tools,
                            "source": a.source} for a in agents.values()]}

    @app.get("/api/sessions", dependencies=[Depends(auth)])
    def sessions():
        return [{**s, "busy": mgr.is_busy(s["id"])} for s in store.list()]

    @app.post("/api/sessions", dependencies=[Depends(auth)])
    async def new_session(request: Request):
        body = await request.json()
        ws = Path(body.get("workspace") or mgr.default_workspace).expanduser().resolve()
        if not ws.is_dir():
            raise HTTPException(400, f"{ws} is not a folder")
        policy = body.get("policy", "auto")
        if policy not in ("ask", "auto", "yolo"):
            raise HTTPException(400, "policy must be ask, auto or yolo")
        settings = {"policy": policy, "model": body.get("model") or default_model,
                    "language": body.get("language", language), "web": bool(body.get("web", True))}
        return {"id": store.create(str(ws), settings, body.get("title", ""))}

    @app.get("/api/sessions/{sid}", dependencies=[Depends(auth)])
    def get_session(sid: str):
        row = store.get(sid)
        if row is None:
            raise HTTPException(404, "no such session")
        live = mgr.live.get(sid)
        return {**{k: row[k] for k in ("id", "title", "workspace", "settings", "created", "updated")},
                "plan": live.plan if live else [], "busy": mgr.is_busy(sid)}

    @app.delete("/api/sessions/{sid}", dependencies=[Depends(auth)])
    def delete_session(sid: str):
        if sid in mgr.live:
            mgr.live.pop(sid).close()
        store.delete(sid)
        return {"ok": True}

    @app.post("/api/sessions/{sid}/messages", dependencies=[Depends(auth)])
    async def send(sid: str, request: Request):
        text = (await request.json()).get("text", "").strip()
        if not text:
            raise HTTPException(400, "empty message")
        try:
            mgr.send(sid, text)
        except KeyError:
            raise HTTPException(404, "no such session") from None
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from None
        return {"ok": True}

    @app.post("/api/sessions/{sid}/answer", dependencies=[Depends(auth)])
    async def answer(sid: str, request: Request):
        body = await request.json()
        try:
            ok = mgr.session(sid).answer(body.get("id", ""), body.get("value"))
        except KeyError:
            raise HTTPException(404, "no such session") from None
        if not ok:
            raise HTTPException(404, "nothing is waiting for that answer")
        return {"ok": True}

    @app.post("/api/sessions/{sid}/cancel", dependencies=[Depends(auth)])
    def cancel(sid: str):
        if sid in mgr.live:
            mgr.live[sid].cancel()
        return {"ok": True}

    @app.get("/api/sessions/{sid}/events", dependencies=[Depends(auth)])
    def events(sid: str, after: int = 0, stream: bool = True):
        if store.get(sid) is None:
            raise HTTPException(404, "no such session")
        if not stream:
            return store.events(sid, after)

        async def gen():
            last, idle = after, 0
            while True:
                rows = store.events(sid, last)
                for r in rows:
                    last = r["seq"]
                    yield f"id: {last}\ndata: {json.dumps(r)}\n\n"
                idle = 0 if rows else idle + 1
                if idle % 30 == 29:
                    yield ": keep-alive\n\n"
                await asyncio.sleep(0.3)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/api/sessions/{sid}/feedback", dependencies=[Depends(auth)])
    async def feedback(sid: str, request: Request):
        body = await request.json()
        if store.get(sid) is None:
            raise HTTPException(404, "no such session")
        store.set_feedback(sid, str(body.get("turn_id", "")), int(body.get("rating", 0)), str(body.get("note", "")))
        return {"ok": True}

    @app.get("/api/sessions/{sid}/file", dependencies=[Depends(auth)])
    def session_file(sid: str, path: str):
        row = store.get(sid)
        if row is None:
            raise HTTPException(404, "no such session")
        root = Path(row["workspace"]).resolve()
        f = (root / path).resolve()
        if root not in f.parents or not f.is_file():
            raise HTTPException(404, "no such file in the workspace")
        return FileResponse(f)

    @app.get("/api/training", dependencies=[Depends(auth)])
    def training():
        return store.stats()

    @app.post("/api/training/export", dependencies=[Depends(auth)])
    async def training_export(request: Request):
        from .finetune import export_dataset

        body = await request.json()
        out = Path(body.get("out_dir") or (Path.home() / ".pinneapple" / "workbench" / "dataset")).expanduser()
        return export_dataset(store, out, min_rating=int(body.get("min_rating", 0)),
                              mode=body.get("mode", "trajectories"))

    @app.get("/api/agents/{name}", dependencies=[Depends(auth)])
    def get_agent(name: str):
        agents, _ = load_agents(mgr.default_workspace)
        if name not in agents:
            raise HTTPException(404, "no such agent")
        a = agents[name]
        return {"name": a.name, "source": a.source, "text": Path(a.source).read_text()}

    @app.put("/api/agents/{name}", dependencies=[Depends(auth)])
    async def put_agent(name: str, request: Request):
        text = (await request.json()).get("text", "")
        try:
            a = parse_agent(text, f"{name}.md")
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        if a.name != name or not name.replace("_", "").replace("-", "").isalnum():
            raise HTTPException(400, "the name in the front matter must match the URL and be alphanumeric")
        USER_DIR.mkdir(parents=True, exist_ok=True)
        (USER_DIR / f"{name}.md").write_text(text)
        for s in mgr.live.values():      # specialists use it from their next delegation; the lead in new sessions
            s.agents[name] = parse_agent(text, str(USER_DIR / f"{name}.md"))
        return {"ok": True, "saved": str(USER_DIR / f"{name}.md")}

    return app

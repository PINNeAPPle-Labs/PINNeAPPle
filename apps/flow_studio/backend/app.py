"""PINNeAPPle Flow Studio: HTTP routes over pinneapple_flow.service.FlowService and the built frontend.

Glue only: every route calls one library method. Security comes from the shared appkit (prefix ``FLOW``): headers
and CSP, body limit, rate limits, login with hashed passwords when FLOW_USER is set, same-origin check on writes.
Start it with ``python -m pinneapple_flow studio``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent.parent / "_shared"))
from appkit import install  # noqa: E402

from pinneapple_flow import FlowError  # noqa: E402
from pinneapple_flow.service import FlowService  # noqa: E402

DIST = HERE.parent / "frontend" / "dist"
JSONBody = Annotated[dict[str, Any], Body()]


def create_app(service: FlowService | None = None) -> FastAPI:
    svc = service or FlowService(max_runs=int(os.environ.get("FLOW_MAX_RUNS", "1")))
    app = FastAPI(title="PINNeAPPle Flow Studio", version="0.1.0")
    app.add_middleware(GZipMiddleware, minimum_size=2000)
    install(app, prefix="FLOW", max_body_mb=float(os.environ.get("FLOW_MAX_BODY_MB", "20")))
    app.state.service = svc

    def bad(exc: Exception):
        raise HTTPException(400, str(exc)) from None

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/api/info")
    def info():
        return svc.info()

    @app.get("/api/catalog")
    def catalog():
        return svc.catalog()

    def who(request: Request) -> str:
        p = getattr(request.state, "principal", "") or ""
        return p if p.startswith(("user:", "key:")) else "local"      # one profile for the local, unauthenticated user

    @app.get("/api/palette")
    def palette(request: Request):
        return svc.palette(who(request))

    @app.get("/api/palette/search")
    def palette_search(q: str = ""):
        return svc.palette_search(q)

    @app.put("/api/palette/prefs")
    def palette_prefs(request: Request, body: JSONBody):
        try:
            return svc.set_palette_prefs(who(request), body)
        except FlowError as e:
            bad(e)

    @app.get("/api/schema")
    def schema():
        return svc.schema()

    @app.post("/api/plan")
    def plan(body: JSONBody):
        try:
            return svc.plan(str(body.get("text", "")), backend=str(body.get("backend", "auto")),
                            provider=body.get("provider"), model=body.get("model"), base_url=body.get("base_url"))
        except ValueError as exc:
            bad(exc)

    @app.post("/api/validate")
    def validate(body: JSONBody):
        return svc.validate(body)

    @app.post("/api/preview/equations")
    def preview_equations(body: JSONBody):
        return svc.equations_preview(body.get("geometry") or {}, body.get("equations") or {})

    @app.post("/api/preview/architecture")
    def preview_architecture(body: JSONBody):
        try:
            return svc.architecture(str(body.get("kind", "mlp")), body.get("config") or {},
                                    int(body.get("in_dim", 2)), int(body.get("out_dim", 64)),
                                    int(body.get("n_fields", 1)))
        except (ValueError, TypeError) as exc:
            bad(exc)

    @app.post("/api/decisions")
    def decisions(body: JSONBody):
        try:
            return svc.decisions(body.get("flow") or {}, body.get("run_id"))
        except FlowError as exc:
            bad(exc)

    @app.get("/api/references")
    def references():
        return svc.references()

    @app.post("/api/references/check")
    def check_reference(body: JSONBody):
        return svc.check_reference(body)

    @app.get("/api/lab/seal")
    def lab_seal(item: str):
        return svc.seal(item)

    @app.get("/api/lab/next-steps")
    def lab_next_steps(item: str):
        return svc.next_steps(item)

    @app.post("/api/flows/insert-nodes")
    def insert_nodes(body: JSONBody):
        try:
            return svc.insert_nodes(body.get("flow") or {}, body.get("anchor"), list(body.get("nodes") or []))
        except (ValueError, KeyError, FlowError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/library-calls")
    def library_calls():
        return svc.library_calls()

    @app.get("/api/custom-nodes")
    def custom_nodes():
        return svc.custom_nodes()

    @app.post("/api/custom-nodes")
    def save_custom_node(body: JSONBody):
        try:
            return svc.save_custom_node(body)
        except ValueError as exc:
            bad(exc)

    @app.delete("/api/custom-nodes/{type_}")
    def delete_custom_node(type_: str):
        svc.delete_custom_node(type_)
        return {"deleted": type_}

    @app.get("/api/flows")
    def flows():
        return svc.list_flows()

    @app.post("/api/flows")
    def save_flow(body: JSONBody):
        try:
            return svc.save_flow(body.get("flow") or body, str(body.get("note", "")) if "flow" in body else "")
        except FlowError as exc:
            bad(exc)

    @app.get("/api/flows/{flow_id}")
    def get_flow(flow_id: str, version: int | None = None):
        try:
            return svc.get_flow(flow_id, version)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from None

    @app.get("/api/flows/{flow_id}/versions")
    def versions(flow_id: str):
        return svc.versions(flow_id)

    @app.delete("/api/flows/{flow_id}")
    def delete_flow(flow_id: str):
        svc.delete_flow(flow_id)
        return {"deleted": flow_id}

    @app.get("/api/examples")
    def examples():
        return svc.examples()

    @app.get("/api/examples/{name}")
    def example(name: str):
        try:
            return svc.example(name)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from None

    @app.post("/api/runs")
    def start_run(body: JSONBody):
        try:
            return svc.start_run(body["flow"], from_node=body.get("from_node"))
        except (FlowError, KeyError) as exc:
            bad(exc)

    @app.get("/api/runs")
    def runs(flow_id: str | None = None):
        return svc.runs(flow_id)

    @app.get("/api/runs/{run_id}")
    def run_status(run_id: str):
        try:
            return svc.run_status(run_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from None

    @app.post("/api/runs/{run_id}/cancel")
    def cancel(run_id: str):
        return {"cancelled": svc.cancel(run_id)}

    @app.get("/api/runs/{run_id}/nodes/{node_id}")
    def node_result(run_id: str, node_id: str):
        try:
            return svc.node_result(run_id, node_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from None

    # an HTML report (the rover dashboard) in a frame of the app: its own opaque origin (CSP sandbox, no
    # allow-same-origin), so its inline scripts run without reaching the app's API or cookies
    HTML_CSP = ("sandbox allow-scripts allow-downloads; default-src 'none'; script-src 'unsafe-inline' blob:; "
                "style-src 'unsafe-inline'; img-src data: blob:; media-src data: blob:; font-src data:; "
                "connect-src 'none'; worker-src blob:; frame-ancestors 'self'")

    @app.get("/api/artifact/html")
    def artifact_html(path: str):
        try:
            return FileResponse(svc.html_artifact(path), media_type="text/html",
                                headers={"content-security-policy": HTML_CSP, "x-frame-options": "SAMEORIGIN",
                                         "cross-origin-opener-policy": "unsafe-none"})
        except PermissionError:
            return JSONResponse({"detail": "not a run artifact"}, status_code=403)

    @app.get("/api/artifact")
    def artifact(path: str):
        try:
            return FileResponse(svc.artifact_path(path))
        except PermissionError:
            return JSONResponse({"detail": "not a run artifact"}, status_code=403)

    if (DIST / "index.html").is_file():
        app.mount("/", StaticFiles(directory=DIST, html=True), name="ui")
    return app

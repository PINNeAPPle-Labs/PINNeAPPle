"""Live lane: an HTTP bridge the Unity runtime calls while it plays.

    python -m pinneapple_unity serve --port 8766
    # or from Python
    from pinneapple_unity import serve_bridge
    srv = serve_bridge(port=8766, block=False)      # srv.url -> http://127.0.0.1:8766/

Endpoints (protocol ``pinneapple-unity/1``, JSON in and out; field payloads base64 float32 by default):

``GET /v1/health``   ``{ok, protocol, package, jobs, requests}``
``GET /v1/jobs``     every registered job with its parameters and doc
``POST /v1/run``     one job on one part (vertices inline, or ``folder`` = a twin/Unity folder on this machine)
``POST /v1/scene``   a whole scene and its list of jobs -> every field, with every check

Why HTTP and not a socket protocol: Unity's ``UnityWebRequest`` already speaks it, the C# side needs no package,
every request is reproducible from ``curl``, and the bridge stays a plain library object a test can drive. It binds
``127.0.0.1`` by default -- it runs physics on demand, so an open port is a real decision, not a default.

Nothing here blocks the game: the C# client sends a request from a coroutine and applies the field when it lands.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

from .jobs import JobRequest, list_jobs, run_job
from .protocol import PROTOCOL, encode_field, y_up_to_z_up

__all__ = ["Bridge", "serve_bridge"]

_MAX_BODY = 64 * 1024 * 1024          # 64 MB: a twin with a real mesh, not a point cloud of a whole plant

#: What went wrong in a warm-up on this machine (reported by /v1/health, never silent).
warnings: list[str] = []


@dataclass
class Bridge:
    """A running (or startable) bridge: the server, its counters, and the last response it built."""

    host: str = "127.0.0.1"
    port: int = 8766
    token: str | None = None
    allow_folders: bool = True
    requests: int = 0
    seconds: float = 0.0
    last: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    _httpd: ThreadingHTTPServer | None = field(default=None, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # -- lifecycle -------------------------------------------------------------------
    def start(self) -> Bridge:
        """Start serving in a background thread (idempotent)."""
        with self._lock:
            if self._httpd is not None:
                return self
            httpd = ThreadingHTTPServer((self.host, self.port), _handler(self))
            self.port = int(httpd.server_address[1])
            self._httpd = httpd
            self._thread = threading.Thread(target=httpd.serve_forever, daemon=True, name="pinneapple-unity-bridge")
            self._thread.start()
        return self

    def stop(self) -> None:
        with self._lock:
            httpd, self._httpd = self._httpd, None
            thread, self._thread = self._thread, None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if thread is not None:
            thread.join(timeout=5)

    def __enter__(self) -> Bridge:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    @property
    def running(self) -> bool:
        return self._httpd is not None

    # -- protocol --------------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        from . import __version__

        return {"ok": True, "protocol": PROTOCOL, "package": __version__, "bridge": "pinneapple-unity",
                "jobs": sorted(list_jobs()), "requests": self.requests, "seconds": round(self.seconds, 6),
                "allow_folders": self.allow_folders, "warnings": list(self.warnings), "errors": self.errors[-5:]}

    def handle(self, path: str, payload: dict[str, Any] | None) -> tuple[int, dict[str, Any]]:
        """Route one request; returns ``(http_status, body)``. Kept public so a test needs no socket."""
        t0 = time.perf_counter()
        try:
            status, body = self._route(path.rstrip("/") or "/", payload or {})
        except KeyError as exc:
            status, body = 404, {"ok": False, "error": str(exc)}
        except (ValueError, TypeError) as exc:
            status, body = 400, {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        except Exception as exc:  # noqa: BLE001 - a failing job is reported, the bridge keeps serving
            self.errors.append(f"{path}: {type(exc).__name__}: {exc}")
            status, body = 500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        dt = time.perf_counter() - t0
        with self._lock:
            self.requests += 1
            self.seconds = dt
        body.setdefault("protocol", PROTOCOL)
        body["seconds"] = round(dt, 6)
        body["run_id"] = str(uuid.uuid4())[:8]
        self.last = body
        return status, body

    def _route(self, path: str, p: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if path in ("/", "/v1/health"):
            return 200, self.health()
        if path == "/v1/jobs":
            return 200, {"ok": True, "jobs": list_jobs()}
        if path == "/v1/run":
            return 200, self._run(p)
        if path == "/v1/scene":
            return 200, self._scene(p)
        raise KeyError(f"no route {path!r} (v1: /v1/health, /v1/jobs, /v1/run, /v1/scene)")

    # -- the two verbs ---------------------------------------------------------------
    def _run(self, p: dict[str, Any]) -> dict[str, Any]:
        encoding = str(p.get("encoding", "base64"))
        if "folder" in p:
            scene, times, up = self._from_folder(str(p["folder"]))
            part_name = str(p.get("part") or scene.parts[0].name)
            part = next((x for x in scene.parts if x.name == part_name), None)
            if part is None:
                raise ValueError(f"no part {part_name!r} (have {[x.name for x in scene.parts]})")
            vertices, faces = part.vertices, part.faces
        else:
            if "vertices" not in p:
                raise ValueError("send 'vertices' (part geometry) or 'folder' (a twin folder on this machine)")
            up = str(p.get("up_axis", "y")).lower()
            vertices = np.asarray(p["vertices"], float).reshape(-1, 3)
            if up == "y":
                vertices = y_up_to_z_up(vertices)
            faces = np.asarray(p.get("faces", []), np.int64).reshape(-1, 3)
            part_name = str(p.get("part") or "part")
            times = [float(t) for t in p.get("times", [])] or [0.0]
        if p.get("times"):
            times = [float(t) for t in p["times"]]
        res = run_job(JobRequest(job=str(p.get("job", "")), part=part_name, vertices=vertices,
                                 times=times, params=dict(p.get("params", {})), faces=faces))
        fields = {name: {**encode_field(vals, encoding, unit=unit), "name": name}
                  for name, (vals, unit) in res.fields.items()}
        return {"ok": True, "job": res.job, "part": part_name, "encoding": encoding, "up_axis": up,
                "steps": len(times), "fields": fields, "checks": res.checks, "decision": res.decision,
                "notes": res.notes, "detail": res.detail}

    def _scene(self, p: dict[str, Any]) -> dict[str, Any]:
        from .protocol import LINK_FILE, LinkJob

        encoding = str(p.get("encoding", "base64"))
        jobs_spec = list(p.get("jobs", []))
        if "folder" in p:
            if not self.allow_folders:
                raise ValueError("this bridge does not read folders (start it with allow_folders=True)")
            folder = str(p["folder"])
            if os.path.exists(os.path.join(folder, LINK_FILE)):      # a scene authored in Unity
                from .ingest import apply_link, from_unity

                scene, link = from_unity(folder)
                if jobs_spec:
                    link.jobs = [LinkJob(str(j["job"]), str(j.get("part", "")), dict(j.get("params", {})))
                                 for j in jobs_spec]
                reports = apply_link(scene, link)
            else:                                                    # an exported twin: run the jobs it is asked for
                from .export import scene_from_twin

                scene = scene_from_twin(folder)
                reports = self._run_jobs(scene, jobs_spec)
            times = [float(t) for t in (scene.times if scene.times is not None else [0.0])]
        else:
            scene, times, _ = self._from_inline(p.get("scene", p))
            reports = self._run_jobs(scene, jobs_spec)
        parts = {}
        for part in scene.parts:
            if not part.fields:
                continue
            parts[part.name] = {name: {**encode_field(vals, encoding, unit=scene.field_units.get(name, "")),
                                       "name": name} for name, vals in part.fields.items()}
        decision = _decision(reports)
        out: dict[str, Any] = {
            "ok": True, "title": scene.title, "encoding": encoding, "steps": len(times),
            "parts": parts, "reports": reports, "decision": decision,
            "sensors": [{"id": s.id, "unit": s.unit, "series": None if s.series is None else
                         [float(v) for v in s.series]} for s in scene.sensors]}
        if p.get("save"):
            # write the computed twin as a Unity project folder, so the result can be reopened straight away
            if not self.allow_folders:
                raise ValueError("this bridge does not write folders (start it with allow_folders=True)")
            from .export import export_for_unity

            exp = export_for_unity(scene, str(p["save"]), usd=bool(p.get("usd", False)))
            out["twin"] = {"folder": exp.folder, "manifest": exp.scene_json, "summary": exp.summary(),
                           "usd": os.path.join(exp.data_dir, "scene.usda") if p.get("usd") else None,
                           "problems": exp.problems}
        return out

    # -- inputs ----------------------------------------------------------------------
    def _run_jobs(self, scene, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run ``jobs`` ([{job, part, params}]) on ``scene``, adding each field to its part; one report per job."""
        times = [float(t) for t in (scene.times if scene.times is not None else [0.0])]
        reports: list[dict[str, Any]] = []
        for j in jobs:
            part = next((x for x in scene.parts if x.name == str(j.get("part"))), None)
            if part is None:
                reports.append({"job": j.get("job"), "part": j.get("part"), "decision": "FAILED",
                                "error": f"no part {j.get('part')!r} (have {[x.name for x in scene.parts]})",
                                "checks": []})
                continue
            try:
                res = run_job(JobRequest(job=str(j.get("job", "")), part=part.name, vertices=part.vertices,
                                         times=times, params=dict(j.get("params", {})), faces=part.faces))
            except Exception as exc:  # noqa: BLE001 - one bad job must not take the whole scene down
                reports.append({"job": j.get("job"), "part": part.name, "decision": "FAILED",
                                "error": f"{type(exc).__name__}: {exc}", "checks": []})
                continue
            for name, (vals, unit) in res.fields.items():
                scene.add_field(part.name, name, vals, unit=unit)
            rep = res.report()
            rep["fields"] = sorted(res.fields)
            reports.append(rep)
        return reports

    def _from_folder(self, folder: str):
        if not self.allow_folders:
            raise ValueError("this bridge does not read folders (start it with allow_folders=True)")
        from .ingest import from_unity

        scene, link = from_unity(folder)
        times = [float(t) for t in (scene.times if scene.times is not None else [0.0])]
        return scene, times, link.up_axis

    def _from_inline(self, p: dict[str, Any]):
        from pinneapple_twin3d import Scene

        d = p.get("scene", p)
        times = [float(t) for t in d.get("times", [])] or [0.0]
        up = str(d.get("up_axis", "y")).lower()
        scene = Scene(str(d.get("title", "Unity scene")), length_unit=str(d.get("length_unit", "m")),
                      times=np.asarray(times, float) if len(times) > 1 else None,
                      time_unit=str(d.get("time_unit", "s")), source="Unity bridge (inline geometry)")
        for part in d.get("parts", []):
            v = np.asarray(part["vertices"], float).reshape(-1, 3)
            if up == "y":
                v = y_up_to_z_up(v)
            f = np.asarray(part.get("faces", []), np.int64).reshape(-1, 3)
            scene.add_part(str(part["name"]), v, f, group=str(part.get("group", "")),
                           color=tuple(float(c) for c in part.get("color", (0.75, 0.77, 0.8))))
        for s in d.get("sensors", []):
            pos = np.asarray(s.get("position", (0.0, 0.0, 0.0)), float)
            scene.add_sensor(str(s.get("id", "sensor")), y_up_to_z_up(pos) if up == "y" else pos,
                             label=str(s.get("label", "")), unit=str(s.get("unit", "")),
                             quantity=str(s.get("quantity", "")))
        return scene, times, up

    # -- client helper ---------------------------------------------------------------
    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Call this bridge over real HTTP (what the C# client does); returns the decoded JSON body."""
        import urllib.error
        import urllib.request

        data = json.dumps(payload).encode()
        req = urllib.request.Request(self.url.rstrip("/") + path, data=data, method="POST",
                                     headers={"Content-Type": "application/json", **_auth(self.token)})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"{path}: HTTP {exc.code}: {body}") from None


def _decision(reports: list[dict[str, Any]]) -> str:
    """The worst decision a set of job reports carries (``NO JOBS`` when there was nothing to check)."""
    decisions = [r.get("decision") for r in reports]
    if not decisions:
        return "NO JOBS"
    if "FAILED" in decisions:
        return "FAILED"
    return "REJECT" if "REJECT" in decisions else "APPROVED"


def _auth(token: str | None) -> dict[str, str]:
    return {"X-Pinneapple-Token": token} if token else {}


def _handler(bridge: Bridge):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):        # a bridge in a game loop should not spam the console
            return

        def _send(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):                     # a browser client (the twin viewer) may preflight
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Pinneapple-Token")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()

        def do_GET(self):
            if bridge.token and self.headers.get("X-Pinneapple-Token") != bridge.token:
                return self._send(401, {"ok": False, "error": "bad or missing X-Pinneapple-Token",
                                        "protocol": PROTOCOL})
            status, body = bridge.handle(self.path, {})
            self._send(status, body)

        def do_POST(self):
            if bridge.token and self.headers.get("X-Pinneapple-Token") != bridge.token:
                return self._send(401, {"ok": False, "error": "bad or missing X-Pinneapple-Token",
                                        "protocol": PROTOCOL})
            n = int(self.headers.get("Content-Length") or 0)
            if n > _MAX_BODY:
                return self._send(413, {"ok": False, "error": f"body of {n} bytes exceeds {_MAX_BODY}",
                                        "protocol": PROTOCOL})
            raw = self.rfile.read(n) if n else b"{}"
            try:
                payload = json.loads(raw or b"{}")
            except json.JSONDecodeError as exc:
                return self._send(400, {"ok": False, "error": f"invalid JSON: {exc}", "protocol": PROTOCOL})
            status, body = bridge.handle(self.path, payload)
            self._send(status, body)

    return Handler


def warm_up() -> float:
    """Pay the imports now, not on the first request: ``import pinneapple`` pulls torch and costs ~20 s here.

    Runs one real job on two vertices, which also proves the jobs work on this machine before a game waits on one.
    Returns the seconds it took.
    """
    t0 = time.perf_counter()
    try:
        import pinneapple  # noqa: F401  (the heavy import: torch, scipy, the trust layer)
        run_job(JobRequest("slab_transient", "warm-up", np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
                           times=[0.0, 3600.0], params={"alpha": 1e-5, "length": 0.2, "axis": "x"}))
    except Exception as exc:  # noqa: BLE001 - a machine that cannot run a job still serves /v1/health, and says so
        warnings.append(f"warm-up failed: {type(exc).__name__}: {exc}")
    return time.perf_counter() - t0


def serve_bridge(port: int = 8766, *, host: str = "127.0.0.1", block: bool = True, token: str | None = None,
                 allow_folders: bool = True, warm: bool = True) -> Bridge:
    """Start the bridge; with ``block=False`` returns it running in a thread (``.url``, ``.stop()``)."""
    token = token if token is not None else os.environ.get("PINNEAPPLE_UNITY_TOKEN")
    seconds = warm_up() if warm else 0.0
    bridge = Bridge(host=host, port=port, token=token or None, allow_folders=allow_folders,
                    warnings=list(warnings)).start()
    if block:
        print(f"PINNeAPPle <-> Unity bridge: {bridge.url}   (jobs: {', '.join(sorted(list_jobs()))}, "
              f"warmed up in {seconds:.1f} s)")
        print("Ctrl+C to stop")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            bridge.stop()
    return bridge

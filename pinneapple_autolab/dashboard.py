"""The lab's dashboard: four synchronised areas over the real state of a session.

    from pinneapple_autolab.dashboard import build_dashboard, serve
    build_dashboard("runs/aero", "runs/aero/dashboard", scenes={"baseline": sc0, "best": sc1})   # static page
    serve("runs/aero", "runs/aero/dashboard", port=8120)            # live: the page polls /state.json

* **Geometry**: the original and the optimised design in 3-D (the studio viewer, skin Cp, streamlines, slices) and
  the variables that changed.
* **Physics**: drag and downforce per component (CFD of both designs), the force history of a run with its averaging
  window and iteration noise, the aero balance.
* **Models**: every surrogate family's held-out error per output, cross-validated predictions against CFD with their
  95 % intervals, the error of each prediction made before its simulation.
* **Decisions**: drag x downforce (candidates, surrogate front, simulated designs), the budget, the decision
  timeline (label before -> after, rule, reason), the validation criteria and the three trust cases.

The page shows what the ledger says and nothing else: a simulation in progress is shown as running, a failed run as
failed (red), training as training; a design that was only predicted is labelled "predicted, not simulated" and is
never drawn as verified. Selecting a run anywhere (Pareto point, timeline row) selects it in every area.
"""
from __future__ import annotations

import base64
import http.server
import json
import os
import socketserver
import threading
from collections.abc import Mapping
from typing import Any

__all__ = ["live_state", "build_dashboard", "serve", "capture", "record_replay"]

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(path: str) -> Any:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _events(path: str, tail: int = 400) -> list[dict[str, Any]]:
    try:
        with open(path) as f:
            lines = f.readlines()[-tail:]
    except OSError:
        return []
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except ValueError:
            pass
    return out


def live_state(folder: str, out_dir: str | None = None, tail: int = 120) -> dict[str, Any]:
    """The session's state from its files: ``ledger.json`` (runs), ``events.jsonl`` (what is happening),
    ``snapshot.json`` / ``summary.json`` when written (in ``folder`` or ``out_dir``). ``status`` says what is going
    on now: ``running`` (the run being simulated), ``training``, ``phase`` (last stage), ``failed`` runs."""
    led = _load(os.path.join(folder, "ledger.json")) or {"runs": [], "budget": {}, "environment": {}}
    ev = _events(os.path.join(folder, "events.jsonl"), tail=max(tail, 400))
    snap = summ = None
    for d in filter(None, (out_dir, folder)):
        snap = snap or _load(os.path.join(d, "snapshot.json")) or _load(os.path.join(d, "snapshot_live.json"))
        summ = summ or _load(os.path.join(d, "summary.json"))
    running = None
    training = False
    done_ids = {r["id"] for r in led["runs"]}
    for e in ev:
        if e.get("stage") == "simulate" and e.get("running"):
            rid = e["message"].split(":")[0]
            running = None if rid in done_ids else {"id": rid, "params": e.get("params"), "since": e.get("time"),
                                                    "purpose": e.get("purpose")}
        if e.get("stage") == "learn":
            training = bool(e.get("training"))
        if e.get("stage") == "run":
            rid = e.get("run", {}).get("id")
            if running and running["id"] == rid:
                running = None
    if running and running["id"] in done_ids:
        running = None
    phase = ev[-1]["stage"] if ev else "idle"
    status = {"phase": phase, "running": running, "training": training and phase == "learn",
              "failed": [r["id"] for r in led["runs"] if not r["ok"]], "finished": phase == "deliver",
              "last_event": ev[-1] if ev else None}
    return {"ledger": led, "events": ev[-tail:], "snapshot": snap, "summary": summ, "status": status}


def _embed(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), default=float).replace("</", "<\\/")


def build_dashboard(folder: str, out_dir: str, *, scenes: Mapping[str, Any] | None = None, title: str = "",
                    state: Mapping[str, Any] | None = None, live: bool = False, replay: bool = False,
                    figures: Mapping[str, str] | None = None, figure_runs: Mapping[str, str] | None = None) -> str:
    """Write ``out_dir/index.html``: the dashboard with the current state embedded (``live=True``: it also polls
    ``state.json`` next to it, see :func:`serve`; ``replay=True``: a slider and a play button step through the
    recorded events, every number the recorded one). ``scenes``: name -> studio Scene (e.g. "baseline", "best"),
    each written as a self-contained 3-D viewer ``viewer_<name>.html`` shown in the Geometry area; ``figures``:
    name -> PNG embedded in the Models area (e.g. the surface-Cp triptych); ``figure_runs``: figure name -> the run
    it shows (a replay shows a figure only once its run exists)."""
    from pinneapple_tools.visualization.studio.web import web_viewer
    os.makedirs(out_dir, exist_ok=True)
    st = dict(state or live_state(folder, out_dir, tail=100000 if replay else 120))
    viewers = {}
    for name, sc in (scenes or {}).items():
        web_viewer(sc, out_dir, single_file=True, filename=f"viewer_{name}.html")
        viewers[name] = f"viewer_{name}.html"
    imgs = {}
    for k, p in (figures or {}).items():
        if p and os.path.exists(p):
            with open(p, "rb") as f:
                imgs[k] = "data:image/png;base64," + base64.b64encode(f.read()).decode()
    with open(os.path.join(HERE, "web", "dashboard.html"), encoding="utf-8") as f:
        page = f.read()
    summ = st.get("summary") or {}
    designs = {k: (summ.get(k) or {}).get("id") for k in ("baseline", "best")}
    if not designs["baseline"]:
        designs["baseline"] = next((r["id"] for r in (st.get("ledger") or {}).get("runs", []) if r.get("purpose") == "baseline"), None)
    cfg = {"title": title or "PINNeAPPle Autonomous Physics Lab", "viewers": viewers, "figures": imgs, "live": live,
           "replay": replay, "designs": designs,   # renders and viewers belong to these runs; shown once they exist
           "figure_runs": dict(figure_runs or {})}
    m = summ.get("mission") or {}
    if m.get("constraints_absolute"):
        cfg["limits"] = {c["name"]: [c["lower"], c["upper"]] for c in m["constraints_absolute"]}
        cfg["objective"] = m["objectives"][0]["name"]
    page = page.replace("/*__STATE__*/null", _embed(st)).replace("/*__CONFIG__*/null", _embed(cfg))
    path = os.path.join(out_dir, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path


def serve(folder: str, out_dir: str, port: int = 8120, host: str = "127.0.0.1", block: bool = True, **kw: Any):
    """Serve the dashboard of ``folder`` live: ``/`` the page (rebuilt once), ``/state.json`` the state read from
    the session's files at every request. ``block=False`` returns the server (``.shutdown()`` to stop)."""
    build_dashboard(folder, out_dir, live=True, **kw)

    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=out_dir, **k)

        def do_GET(self):  # noqa: N802 - http.server naming
            if self.path.split("?")[0] == "/state.json":
                body = _embed(live_state(folder, out_dir)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def log_message(self, *a):
            pass

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    srv = socketserver.ThreadingTCPServer((host, port), H)
    srv.daemon_threads = True
    if not block:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv
    print(f"Autonomous Physics Lab dashboard: http://{host}:{port}/")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()
    return srv


def _browser(pw):
    """Chromium from Playwright, falling back to a browser already installed under PLAYWRIGHT_BROWSERS_PATH."""
    import glob
    args = ["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"]
    try:
        return pw.chromium.launch(args=args)
    except Exception:  # noqa: BLE001 - try a browser of another Playwright version
        root = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "/opt/pw-browsers")
        exe = sorted(glob.glob(os.path.join(root, "chromium-*", "chrome-linux*", "chrome")))
        if not exe:
            raise
        return pw.chromium.launch(executable_path=exe[-1], args=args)


def capture(html: str, png: str, *, width: int = 1600, height: int = 1100, full_page: bool = True,
            replay_to: int | None = None, wait_ms: int = 5000) -> str:
    """Screenshot of a dashboard page (``replay_to``: the replay at that event). Needs ``playwright``."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = _browser(pw)
        page = b.new_page(viewport={"width": width, "height": height})
        page.goto("file://" + os.path.abspath(html))
        page.wait_for_timeout(wait_ms)
        if replay_to is not None:
            page.evaluate("k => window.replayTo(k)", replay_to)
            page.wait_for_timeout(400)
        page.screenshot(path=png, full_page=full_page)
        b.close()
    return png


def record_replay(html: str, mp4: str, *, frames: int = 60, fps: float = 4.0, hold_s: float = 3.0,
                  width: int = 1600, height: int = 2040, out_width: int = 1280, crf: int = 30) -> str:
    """An MP4 of the dashboard's replay (the recorded events, evenly sampled), via ffmpeg. Needs ``playwright``."""
    import shutil
    import subprocess
    import tempfile

    from playwright.sync_api import sync_playwright
    tmp = tempfile.mkdtemp(prefix="autolab_replay_")
    with sync_playwright() as pw:
        b = _browser(pw)
        page = b.new_page(viewport={"width": width, "height": height})
        page.goto("file://" + os.path.abspath(html))
        page.wait_for_timeout(5000)
        total = int(page.evaluate("() => +document.getElementById('slider').max"))
        ks = [round(i * total / max(frames - 1, 1)) for i in range(frames)]
        ks += [ks[-1]] * int(hold_s * fps)
        for i, k in enumerate(ks):
            page.evaluate("k => window.replayTo(k)", k)
            page.wait_for_timeout(200)
            page.screenshot(path=os.path.join(tmp, f"f{i:04d}.png"))
        b.close()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "f%04d.png"),
                    "-vf", f"scale={out_width}:-2,format=yuv420p", "-c:v", "libx264", "-crf", str(crf), "-preset",
                    "slow", mp4], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return mp4

"""Interactive browser view of a :class:`~.scene.Scene` (three.js, shipped with the library; no internet needed).

    web_viewer(scene, "viewer/")       # writes index.html, viewer.js, studio-core.js, scene.glb, scene.json and three.js
    serve(scene)                       # the same in a temporary folder, served on http://localhost:8000
    screenshot("viewer/", "view.png", frame=10)   # headless Chromium (playwright) picture of the page

Modes: the materials ("Realistic"), each surface field on the jet scale with a colour bar, streamlines coloured by
their values, each slice plane. Browsers do not open ES modules from file://, so open the folder through a local
server (``python -m http.server -d viewer``) or use :func:`serve`. A scene with an ``animation``
(:class:`~.animation.Animation`) gets a bottom bar: play / pause, time slider, speed, head-up values, follow camera.
"""
from __future__ import annotations

import http.server
import os
import shutil
import socketserver
import tempfile
import threading
import warnings
import webbrowser
from typing import Optional

HERE = os.path.join(os.path.dirname(__file__), "web")


MAX_FACES = 1_000_000          # beyond this a browser on a laptop stutters; see docs/core_concepts/studio.md


def web_viewer(scene, out_dir: str, title: Optional[str] = None, max_faces: Optional[int] = MAX_FACES) -> str:
    """Write the viewer for ``scene`` into ``out_dir``; returns the path of its index.html. Surfaces above
    ``max_faces`` triangles in total are decimated for the page (the scene itself is not changed); None keeps all."""
    os.makedirs(out_dir, exist_ok=True)
    if title:
        scene.title = title
    if max_faces and scene.n_faces() > max_faces:
        import copy
        n0 = scene.n_faces()
        scene = copy.copy(scene)
        scene.surfaces = list(scene.surfaces)
        anim = getattr(scene, "animation", None)
        scene.decimate(max_faces, keep=list(anim.surface_fields) if anim is not None else ())
        print(f"web_viewer: {n0:,} triangles decimated to {scene.n_faces():,} for the browser (max_faces={max_faces:,})")
    scene.save(os.path.join(out_dir, "scene.glb"))
    scene.save(os.path.join(out_dir, "scene.json"))
    for f in ("index.html", "viewer.js", "studio-core.js"):
        shutil.copy(os.path.join(HERE, f), os.path.join(out_dir, f))
    dst = os.path.join(out_dir, "vendor")
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(os.path.join(HERE, "vendor"), dst)
    return os.path.join(out_dir, "index.html")


def serve(scene, port: int = 8000, open_browser: bool = True, block: bool = True, out_dir: Optional[str] = None,
          max_faces: Optional[int] = MAX_FACES):
    """Write the viewer (to ``out_dir`` or a temporary folder) and serve it on ``port``. ``block=False`` returns the
    server (call ``.shutdown()``); otherwise serves until Ctrl+C."""
    d = out_dir or tempfile.mkdtemp(prefix="pinneapple_viewer_")
    web_viewer(scene, d, max_faces=max_faces)

    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=d, **kw)

        def log_message(self, *a):
            pass

    socketserver.TCPServer.allow_reuse_address = True
    srv = socketserver.TCPServer(("127.0.0.1", port), H)
    url = f"http://localhost:{port}/"
    print(f"PINNeAPPle viewer: {url}  (folder {d})")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    if not block:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()
    return srv


def _server(directory: str):
    """A quiet HTTP server for ``directory`` on a free local port, in a daemon thread (call ``.shutdown()``)."""
    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=directory, **kw)

        def log_message(self, *a):
            pass

    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _chromium_path() -> str | None:
    p = os.environ.get("PLAYWRIGHT_CHROMIUM") or "/opt/pw-browsers/chromium"
    return p if os.path.isfile(p) else None


def screenshot(viewer_dir: str, out_png: str, frame: int | None = None, size: tuple[int, int] = (1280, 720),
               setup_js: str | None = None, timeout_ms: int = 60_000) -> str | None:
    """PNG of the viewer written by :func:`web_viewer` in ``viewer_dir``, taken in headless Chromium (needs
    ``pip install playwright``; ``PLAYWRIGHT_CHROMIUM`` or /opt/pw-browsers/chromium is used when the bundled browser
    is missing). ``frame``: animation frame to show (``window.studio.setFrame``); ``setup_js``: JavaScript run after
    the page is ready (e.g. ``"studio.setMode('real')"``). Returns ``out_png``, or None with a warning when
    playwright is not installed."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        warnings.warn("screenshot: playwright is not installed (pip install playwright); no picture taken", stacklevel=2)
        return None
    srv = _server(os.path.abspath(viewer_dir))
    url = f"http://127.0.0.1:{srv.server_address[1]}/index.html"
    args = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(args=args)
            except Exception:                                    # noqa: BLE001 - bundled browser missing: system one
                exe = _chromium_path()
                if exe is None:
                    raise
                browser = p.chromium.launch(args=args, executable_path=exe)
            try:
                page = browser.new_page(viewport={"width": int(size[0]), "height": int(size[1])})
                errors = []
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.goto(url)
                page.wait_for_function("window.studio !== undefined", timeout=timeout_ms)
                if setup_js:
                    page.evaluate(setup_js)
                if frame is not None:
                    page.evaluate(f"window.studio.setFrame({int(frame)})")
                page.evaluate("new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))")
                os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
                page.screenshot(path=out_png)
                if errors:
                    warnings.warn("screenshot: page errors: " + "; ".join(errors), stacklevel=2)
            finally:
                browser.close()
    finally:
        srv.shutdown()
        srv.server_close()
    return out_png

"""Interactive browser view of a :class:`~.scene.Scene` (three.js, shipped with the library; no internet needed).

    web_viewer(scene, "viewer/")       # writes index.html, viewer.js, studio-core.js, scene.glb, scene.json and three.js
    web_viewer(scene, "v/", single_file=True)   # one self-contained v/index.html (three.js, scene inlined): opens from file://
    serve(scene)                       # the same in a temporary folder, served on http://localhost:8000
    screenshot("viewer/", "view.png", frame=10)   # headless Chromium (playwright) picture of the page

Modes: the materials ("Realistic"), each surface field on the jet scale with a colour bar, streamlines coloured by
their values, each slice plane. Browsers do not open ES modules from file://, so open the folder through a local
server (``python -m http.server -d viewer``), use :func:`serve`, or write a single self-contained page
(``single_file=True``: the scene, the viewer and the vendored three.js inlined; it opens from disk and can be mailed). A scene with an ``animation``
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


def web_viewer(scene, out_dir: str, title: Optional[str] = None, max_faces: Optional[int] = MAX_FACES,
               single_file: bool = False, filename: str = "index.html") -> str:
    """Write the viewer for ``scene`` into ``out_dir``; returns the path of its index.html. Surfaces above
    ``max_faces`` triangles in total are decimated for the page (the scene itself is not changed); None keeps all.
    ``single_file``: write only ``out_dir/filename``, with scene.json, scene.glb, the viewer and three.js inlined
    (ES modules as ``data:`` URLs in an import map), so it opens from file:// with no server."""
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
    if single_file:
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_single_page(scene))
        return path
    scene.save(os.path.join(out_dir, "scene.glb"))
    scene.save(os.path.join(out_dir, "scene.json"))
    for f in ("index.html", "viewer.js", "studio-core.js"):
        shutil.copy(os.path.join(HERE, f), os.path.join(out_dir, f))
    dst = os.path.join(out_dir, "vendor")
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(os.path.join(HERE, "vendor"), dst)
    return os.path.join(out_dir, "index.html")


_MODULES = {   # bare specifier in the single-file page -> vendored file (relative imports rewritten to these)
    "three": "vendor/three/three.module.min.js", "pp/orbit": "vendor/three/OrbitControls.js",
    "pp/gltf": "vendor/three/loaders/GLTFLoader.js", "pp/bgu": "vendor/three/utils/BufferGeometryUtils.js",
    "pp/room": "vendor/three/environments/RoomEnvironment.js", "pp/core": "studio-core.js", "pp/viewer": "viewer.js",
}
_REWRITE = {"./vendor/three/OrbitControls.js": "pp/orbit", "./vendor/three/loaders/GLTFLoader.js": "pp/gltf",
            "./vendor/three/environments/RoomEnvironment.js": "pp/room", "./studio-core.js": "pp/core",
            "../utils/BufferGeometryUtils.js": "pp/bgu"}


def _single_page(scene) -> str:
    """index.html with every module, scene.json and scene.glb inlined (see :func:`web_viewer`)."""
    import base64
    import json
    import re

    def module(rel: str) -> str:
        with open(os.path.join(HERE, rel), encoding="utf-8") as fh:
            src = fh.read()
        for a, b in _REWRITE.items():
            src = re.sub(r"""(from\s*)(['"])""" + re.escape(a) + r"""\2""", lambda m, b=b: f'{m.group(1)}"{b}"', src)
        return "data:text/javascript;base64," + base64.b64encode(src.encode("utf-8")).decode()

    imap = json.dumps({"imports": {k: module(v) for k, v in _MODULES.items()}})
    embed = json.dumps({"meta": scene.extras(), "glb": base64.b64encode(scene.to_glb()).decode()},
                       separators=(",", ":")).replace("</", "<\\/")
    with open(os.path.join(HERE, "index.html"), encoding="utf-8") as fh:
        page = fh.read()
    page = re.sub(r'<script type="importmap">.*?</script>', lambda m: f'<script type="importmap">{imap}</script>',
                  page, flags=re.S)
    return page.replace('<script type="module" src="viewer.js"></script>',
                        f'<script>window.STUDIO_EMBED = {embed};</script>\n<script type="module">import "pp/viewer";</script>')


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
    """PNG of the viewer written by :func:`web_viewer` in ``viewer_dir`` (or of a single-file page: the path of the
    .html, opened from file://), taken in headless Chromium (needs
    ``pip install playwright``; ``PLAYWRIGHT_CHROMIUM`` or /opt/pw-browsers/chromium is used when the bundled browser
    is missing). ``frame``: animation frame to show (``window.studio.setFrame``); ``setup_js``: JavaScript run after
    the page is ready (e.g. ``"studio.setMode('real')"``). Returns ``out_png``, or None with a warning when
    playwright is not installed."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        warnings.warn("screenshot: playwright is not installed (pip install playwright); no picture taken", stacklevel=2)
        return None
    single = os.path.isfile(viewer_dir)
    srv = None if single else _server(os.path.abspath(viewer_dir))
    url = "file://" + os.path.abspath(viewer_dir) if single else f"http://127.0.0.1:{srv.server_address[1]}/index.html"
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
        if srv is not None:
            srv.shutdown()
            srv.server_close()
    return out_png

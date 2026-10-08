"""Interactive browser view of a :class:`~.scene.Scene` (three.js, shipped with the library; no internet needed).

    web_viewer(scene, "viewer/")       # writes index.html, viewer.js, scene.glb, scene.json and three.js
    serve(scene)                       # the same in a temporary folder, served on http://localhost:8000

Modes: the materials ("Realistic"), each surface field on the jet scale with a colour bar, streamlines coloured by
their values, each slice plane. Browsers do not open ES modules from file://, so open the folder through a local
server (``python -m http.server -d viewer``) or use :func:`serve`.
"""
from __future__ import annotations

import http.server
import os
import shutil
import socketserver
import tempfile
import threading
import webbrowser
from typing import Optional

HERE = os.path.join(os.path.dirname(__file__), "web")


def web_viewer(scene, out_dir: str, title: Optional[str] = None) -> str:
    """Write the viewer for ``scene`` into ``out_dir``; returns the path of its index.html."""
    os.makedirs(out_dir, exist_ok=True)
    if title:
        scene.title = title
    scene.save(os.path.join(out_dir, "scene.glb"))
    scene.save(os.path.join(out_dir, "scene.json"))
    for f in ("index.html", "viewer.js"):
        shutil.copy(os.path.join(HERE, f), os.path.join(out_dir, f))
    dst = os.path.join(out_dir, "vendor")
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(os.path.join(HERE, "vendor"), dst)
    return os.path.join(out_dir, "index.html")


def serve(scene, port: int = 8000, open_browser: bool = True, block: bool = True, out_dir: Optional[str] = None):
    """Write the viewer (to ``out_dir`` or a temporary folder) and serve it on ``port``. ``block=False`` returns the
    server (call ``.shutdown()``); otherwise serves until Ctrl+C."""
    d = out_dir or tempfile.mkdtemp(prefix="pinneapple_viewer_")
    web_viewer(scene, d)

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

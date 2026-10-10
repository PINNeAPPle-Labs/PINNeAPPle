"""Record a Twin3D scene to PNG frames / GIF with headless Chromium (Playwright), for docs and posts.

    from pinneapple_physics.blackhole.capture import capture_twin
    frames = capture_twin("out/bh_twin", steps=range(40), field="log10_density", cmap="inferno")
"""
from __future__ import annotations

import contextlib
import functools
import http.server
import os
import socketserver
import threading
from typing import Iterable, List, Optional

__all__ = ["capture_twin"]


@contextlib.contextmanager
def _serve(folder: str):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=folder)
    handler.log_message = lambda *a, **k: None
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as httpd:
        th = threading.Thread(target=httpd.serve_forever, daemon=True)
        th.start()
        try:
            yield f"http://127.0.0.1:{httpd.server_address[1]}/"
        finally:
            httpd.shutdown()


def _not_blank(path: str, min_std: float = 12.0) -> bool:
    import numpy as np
    from PIL import Image
    return float(np.asarray(Image.open(path).convert("L"), float).std()) > min_std


def capture_twin(folder: str, steps: Iterable[int], out_dir: Optional[str] = None, *, field: str = "",
                 cmap: str = "turbo", view: str = "", zoom: float = 1.0, size=(1280, 720), theme: str = "dark",
                 vrange: Optional[tuple] = None,
                 hide_panel: bool = True) -> List[str]:
    """Screenshot the exported scene in ``folder`` at each time step; returns the PNG paths."""
    from playwright.sync_api import sync_playwright

    out_dir = out_dir or os.path.join(folder, "_frames")
    os.makedirs(out_dir, exist_ok=True)
    exe = "/opt/pw-browsers/chromium"
    paths = []
    with _serve(folder) as base, sync_playwright() as pw:
        kw = {"executable_path": exe} if os.path.isfile(exe) else {}
        browser = pw.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"], **kw)
        page = browser.new_page(viewport={"width": size[0], "height": size[1]})
        q = f"?field={field}&cmap={cmap}&zoom={zoom}&theme={theme}" + (f"&view={view}" if view else "") + \
            (f"&range={vrange[0]},{vrange[1]}" if vrange else "")
        page.goto(base + "index.html" + q)
        import json
        with open(os.path.join(folder, "scene.json")) as f:
            nt = len(json.load(f)["times"])
        # buildUi() sets the slider range once geometry and fields are loaded
        page.wait_for_function(f"document.getElementById('time') && +document.getElementById('time').max === {max(nt - 1, 0)}"
                               f" && document.getElementById('legendName').textContent.length > 0", timeout=120000)
        page.wait_for_timeout(2000)
        if hide_panel:
            page.add_style_tag(content="aside, header { display: none !important; } #app { display: block !important; } main#viewport { position: fixed !important; inset: 0 !important; width: 100vw !important; height: 100vh !important; }")
            page.evaluate("window.dispatchEvent(new Event('resize'))")
        for k, s in enumerate(steps):
            page.evaluate("(s) => { const t = document.getElementById('time'); t.value = s; "
                          "t.dispatchEvent(new Event('input')); }", s)
            page.wait_for_timeout(250)
            p = os.path.join(out_dir, f"frame_{k:04d}.png")
            for _ in range(5):                       # software WebGL can lag a frame: retry a blank canvas
                page.screenshot(path=p)
                if _not_blank(p):
                    break
                page.wait_for_timeout(1000)
            paths.append(p)
        browser.close()
    return paths

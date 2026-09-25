"""Serve an exported scene folder over HTTP (the viewer needs HTTP, not file://, to fetch its data)."""
from __future__ import annotations

import functools
import http.server
import threading


def serve(folder: str, port: int = 8765, *, host: str = "127.0.0.1", block: bool = True):
    """Serve ``folder`` at ``http://host:port/``. With ``block=False`` returns the running server."""
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=folder)
    httpd = http.server.ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{httpd.server_address[1]}/"
    if not block:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        httpd.url = url  # type: ignore[attr-defined]
        return httpd
    print(f"PINNeAPPle Twin3D: {url}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return None

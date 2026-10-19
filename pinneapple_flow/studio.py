"""Serve Flow Studio: the app in ``apps/flow_studio`` (a thin FastAPI shell over :class:`FlowService` plus the built
React frontend) on localhost by default.

Binding to anything but localhost requires a login (``FLOW_USER`` + ``FLOW_PASSWORD_HASH``, the appkit security of
the other PINNeAPPle apps) unless ``allow_insecure`` is given explicitly.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

LOCAL = ("127.0.0.1", "localhost", "::1")


def app_dir() -> Path:
    env = os.environ.get("PINNEAPPLE_FLOW_APP")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / "apps" / "flow_studio"


def load_app():
    """The FastAPI app of ``apps/flow_studio/backend/app.py`` (from a repository checkout)."""
    d = app_dir()
    f = d / "backend" / "app.py"
    if not f.is_file():
        raise SystemExit(f"Flow Studio app not found at {d}: run from a repository checkout "
                         "(git clone ...; pip install -e .[flow]) or set PINNEAPPLE_FLOW_APP")
    spec = importlib.util.spec_from_file_location("flow_studio_app", f)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["flow_studio_app"] = mod
    spec.loader.exec_module(mod)
    return mod.create_app()


def serve(host: str | None = None, port: int | None = None, *, open_browser: bool = True, env_file: str = ".env",
          allow_insecure: bool = False) -> int:
    from .config import Settings, load_dotenv
    load_dotenv(env_file)
    s = Settings.from_env()
    host, port = host or s.host, int(port or s.port)
    if host not in LOCAL and not (os.environ.get("FLOW_USER") and (os.environ.get("FLOW_PASSWORD_HASH")
                                                                  or os.environ.get("FLOW_USERS_FILE"))):
        if not allow_insecure:
            print(f"refusing to listen on {host} without a login: set FLOW_USER and FLOW_PASSWORD_HASH "
                  "(python -m pinneapple_security hash-password), or pass --insecure-no-auth", file=sys.stderr)
            return 2
    import uvicorn
    app = load_app()
    dist = app_dir() / "frontend" / "dist"
    if not (dist / "index.html").is_file():
        print(f"note: no built frontend in {dist}; the API runs, build the UI with "
              f"`cd {app_dir() / 'frontend'} && npm install && npm run build`", file=sys.stderr)
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}/"
    print(f"PINNeAPPle Flow Studio on {url}", file=sys.stderr)
    if open_browser:
        import threading
        import webbrowser
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0

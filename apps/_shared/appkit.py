"""Shared plumbing for the PINNeAPPle apps (apps/*): login, busy limiter, static vendor files.

Each app does::

    from appkit import install  # apps/_shared on sys.path
    install(app, prefix="HSS")   # reads HSS_USER / HSS_PASSWORD

Environment (all optional, per app prefix):
  <PREFIX>_USER / <PREFIX>_PASSWORD   HTTP Basic auth on everything except /health
"""
from __future__ import annotations

import base64
import os
import secrets
import threading

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

VENDOR = os.path.join(os.path.dirname(__file__), "static", "vendor")
SHARED = os.path.join(os.path.dirname(__file__), "static", "shared")


def install(app: FastAPI, prefix: str) -> None:
    """Optional HTTP Basic login + the shared static files: ``/vendor`` (three.js)
    and ``/shared`` (UI pieces common to every app)."""
    app.state.auth_user = os.environ.get(f"{prefix}_USER")
    app.state.auth_password = os.environ.get(f"{prefix}_PASSWORD")

    @app.middleware("http")
    async def basic_auth(request: Request, call_next):
        user, password = app.state.auth_user, app.state.auth_password
        if user and password and request.url.path != "/health":
            ok = False
            header = request.headers.get("authorization", "")
            if header.lower().startswith("basic "):
                try:
                    u, _, p = base64.b64decode(header[6:]).decode().partition(":")
                    ok = secrets.compare_digest(u, user) and secrets.compare_digest(p, password)
                except (ValueError, UnicodeDecodeError):
                    ok = False
            if not ok:
                return Response(status_code=401,
                                headers={"WWW-Authenticate": f'Basic realm="{app.title}"'})
        return await call_next(request)

    app.mount("/vendor", StaticFiles(directory=VENDOR), name="vendor")
    app.mount("/shared", StaticFiles(directory=SHARED), name="shared")


class BusyLimiter:
    """Caps concurrent heavy requests per worker; the excess gets 429 instead of queueing."""

    def __init__(self, env_var: str, default: int = 2):
        self.slots = threading.BoundedSemaphore(int(os.environ.get(env_var) or default))

    def __enter__(self):
        if not self.slots.acquire(blocking=False):
            raise HTTPException(status_code=429,
                                detail="Server busy with other runs -- please retry in a few seconds.")
        return self

    def __exit__(self, *exc):
        self.slots.release()
        return False

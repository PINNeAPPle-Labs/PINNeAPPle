"""Shared plumbing for the PINNeAPPle apps (apps/*): login, busy limiter, static vendor files, and the runtime
(job queue, API keys, usage metering, uploads) under /api.

Each app does::

    from appkit import install, install_runtime  # apps/_shared on sys.path
    install(app, prefix="HSS")                    # reads HSS_USER / HSS_PASSWORD
    rt = install_runtime(app, prefix="HSS", name="heatsink", units={"runs": 200})

Environment (all optional, per app prefix; the full list is in apps/_shared/README.md):
  <PREFIX>_USER / <PREFIX>_PASSWORD_HASH   HTTP Basic auth on everything except /health (an X-API-Key also passes);
                                      <PREFIX>_PASSWORD (plain text) still works
  <PREFIX>_RATE_LIMIT, <PREFIX>_LOGIN_MAX_FAILURES, <PREFIX>_CORS_ORIGINS, <PREFIX>_MAX_BODY_MB   security settings
  <PREFIX>_MAX_JOBS                   concurrent jobs of the queue (all server workers together)
  <PREFIX>_FREE_<UNIT>                free use per principal and month, e.g. CAMPAIGN_FREE_CASES_PLANNED
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import math
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from pinneapple_core.service import (
    APIKeyStore,
    JobQueue,
    UploadRejected,
    UploadStore,
    UploadTooLarge,
    UsageLimitExceeded,
    UsageMeter,
    data_dir,
    limits_from_env,
)
from pinneapple_security import web
from pinneapple_security.passwords import is_hash, verify_password


def _vendor_dir() -> str:
    """three.js ships once, inside the library (pinneapple_tools/visualization/studio/web/vendor), so the apps and
    pp.viz's browser viewer serve the same copy. Located without importing the package."""
    import importlib.util
    spec = importlib.util.find_spec("pinneapple_tools")
    root = list(spec.submodule_search_locations)[0]
    return os.path.join(root, "visualization", "studio", "web", "vendor")


log = logging.getLogger("pinneapple.apps")
VENDOR = _vendor_dir()
STUDIO = os.path.dirname(VENDOR)                      # studio-core.js: colour scales, tubes, slices, edges
SHARED = os.path.join(os.path.dirname(__file__), "static", "shared")


def install(app: FastAPI, prefix: str, *, max_body_mb: float | None = None) -> None:
    """Login, security middleware and the shared static files: ``/vendor`` (three.js), ``/studio`` (the studio viewer
    core) and ``/shared`` (UI pieces common to every app).

    Security (``pinneapple_security.web``; settings in apps/_shared/README.md): security headers with a CSP built
    from the app's own static pages, a request body limit, rate limits per caller, lockout after failed logins,
    hashed passwords, a cross-site request check on state-changing requests, an optional CORS allow-list, generic
    500 responses and an audit trail of auth and admin events."""
    env = os.environ
    app.state.auth_user = env.get(f"{prefix}_USER")
    app.state.auth_password = env.get(f"{prefix}_PASSWORD_HASH") or env.get(f"{prefix}_PASSWORD")
    app.state.extra_users = _users_file(env.get(f"{prefix}_USERS_FILE"))
    admins = env.get(f"{prefix}_ADMIN_USERS")
    app.state.admin_users = ({u.strip() for u in admins.split(",") if u.strip()} if admins
                             else None if not app.state.extra_users else {app.state.auth_user} - {None})
    if app.state.auth_password and not is_hash(app.state.auth_password):
        # the message names the variable, never its value
        log.warning("%s_PASSWORD is plain text: set %s_PASSWORD_HASH (python -m pinneapple_security "  # nosemgrep
                    "hash-password) instead", prefix, prefix)
    trusted = [c for c in (env.get(f"{prefix}_TRUSTED_PROXIES") or env.get("PINNEAPPLE_TRUSTED_PROXIES") or
                           ",".join(web.DEFAULT_TRUSTED_PROXIES)).split(",") if c.strip()]
    app.state.trusted_proxies = web._networks(trusted)
    app.state.allowed_origins = [o.strip() for o in (env.get(f"{prefix}_ALLOWED_ORIGINS") or "").split(",")
                                 if o.strip()]
    app.state.throttle = web.LoginThrottle(max_failures=int(env.get(f"{prefix}_LOGIN_MAX_FAILURES") or 10),
                                           window=float(env.get(f"{prefix}_LOGIN_WINDOW_S") or 900),
                                           lock_seconds=float(env.get(f"{prefix}_LOGIN_LOCK_S") or 900))
    app.state.limiter = web.RateLimiter(_rate_rules(prefix))
    app.state.audit = web.AuthAudit(prefix.lower(), path=_audit_path(prefix),
                                    key=(env.get(f"{prefix}_AUDIT_KEY") or "").encode() or None)
    app.state.verifier = _Verifier()
    sec_log = logging.getLogger("pinneapple.security")
    if not sec_log.handlers:                                  # one JSON line per auth event on stderr (docker logs)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        sec_log.addHandler(handler)
        sec_log.setLevel(logging.INFO)
        sec_log.propagate = False

    @app.middleware("http")
    async def basic_auth(request: Request, call_next):
        rt = getattr(app.state, "runtime", None)
        ip = client_ip(request)
        path, method = request.url.path, request.method
        if not web.same_origin_ok(method, request.headers, app.state.allowed_origins):
            app.state.audit.event("csrf_blocked", f"ip:{ip}", ip=ip, target=path, outcome="denied",
                                  origin=request.headers.get("origin", ""))
            return JSONResponse({"detail": "cross-site request refused"}, status_code=403)
        users = _users(app)
        login, tried = (_basic_login(request, users, app.state.verifier) if users else (None, None))
        if tried and not login and path != "/health":
            locked = app.state.throttle.locked_for(ip, tried)
            if locked:
                app.state.audit.event("login_locked", f"user:{tried}", ip=ip, target=path, outcome="denied")
                return JSONResponse({"detail": "too many failed logins; try again later"}, status_code=429,
                                    headers={"Retry-After": web.RateLimiter.retry_after(locked)})
            started = app.state.throttle.failure(ip, tried)
            app.state.audit.event("login_failed", f"user:{tried}", ip=ip, target=path, outcome="denied",
                                  lockout_started=started)
        elif login:
            locked = app.state.throttle.locked_for(ip, login)
            if locked:
                app.state.audit.event("login_locked", f"user:{login}", ip=ip, target=path, outcome="denied")
                return JSONResponse({"detail": "too many failed logins; try again later"}, status_code=429,
                                    headers={"Retry-After": web.RateLimiter.retry_after(locked)})
            app.state.throttle.success(ip, login)
        key = rt.keys.verify(request.headers.get("x-api-key")) if rt is not None else None
        if rt is not None and request.headers.get("x-api-key") and key is None:
            if app.state.throttle.failure(ip, None):
                app.state.audit.event("api_key_lockout", f"ip:{ip}", ip=ip, target=path, outcome="denied")
            return JSONResponse({"detail": "invalid or revoked API key"}, status_code=401)
        request.state.login = login
        request.state.api_key = key
        request.state.principal = (f"key:{key['id']}" if key else f"user:{login}" if login else f"ip:{ip}")
        if path != "/health":
            if users and not (login or key):
                return Response(status_code=401,
                                headers={"WWW-Authenticate": f'Basic realm="{app.title}"'})
            if rt is not None and rt.require_key and path.startswith("/api/") and not (login or key):
                return JSONResponse({"detail": "this server requires an X-API-Key header"}, status_code=401)
            wait, rule = app.state.limiter.check(request.state.principal, method, path)
            if wait:
                app.state.audit.event("rate_limited", request.state.principal, ip=ip, target=path,
                                      outcome="denied", rule=rule.name)
                return JSONResponse({"detail": f"rate limit ({rule.name}): retry in {math.ceil(wait)} s"},
                                    status_code=429, headers={"Retry-After": web.RateLimiter.retry_after(wait)})
        return await call_next(request)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        """No stack traces or exception text to clients: a reference the operator finds in the server log."""
        ref = secrets.token_hex(6)
        log.error("unhandled error %s on %s %s", ref, request.method, request.url.path, exc_info=exc)
        return JSONResponse({"detail": "internal error", "reference": ref}, status_code=500)

    cors = [o.strip() for o in (env.get(f"{prefix}_CORS_ORIGINS") or "").split(",") if o.strip()]
    if cors:
        from fastapi.middleware.cors import CORSMiddleware
        credentials = (env.get(f"{prefix}_CORS_CREDENTIALS") or "").lower() in ("1", "true", "yes")
        app.add_middleware(CORSMiddleware, allow_origins=web.check_cors_origins(cors, credentials),
                           allow_credentials=credentials, allow_methods=["GET", "POST", "DELETE"],
                           allow_headers=["Content-Type", "X-API-Key", "Authorization"], max_age=600)
    app.add_middleware(web.SecurityHeaders, csp=_csp_for(app, prefix),
                       hsts=env.get(f"{prefix}_HSTS", "max-age=31536000; includeSubDomains"))
    body_mb = float(env.get(f"{prefix}_MAX_BODY_MB") or max_body_mb or DEFAULT_MAX_BODY_MB)
    app.add_middleware(web.BodyLimit, max_bytes=int(body_mb * 2**20))

    app.mount("/vendor", StaticFiles(directory=VENDOR), name="vendor")
    app.mount("/studio", StaticFiles(directory=STUDIO), name="studio")
    app.mount("/shared", StaticFiles(directory=SHARED), name="shared")


DEFAULT_MAX_BODY_MB = 600          # above the largest per-app upload default (Interop Hub, 500 MB); Caddy is stricter
DEFAULT_RATES = {"write": "120/min", "read": "1200/min"}


def _rate_rules(prefix: str) -> list[web.RateRule]:
    """``<PREFIX>_RATE_LIMIT`` for POST/PUT/PATCH/DELETE under /api (solver runs, inference, uploads) and
    ``<PREFIX>_RATE_LIMIT_READ`` for every request, per caller; ``off`` disables one."""
    rules = []
    for name, env_name, methods, prefixes in (
            ("write", f"{prefix}_RATE_LIMIT", frozenset({"POST", "PUT", "PATCH", "DELETE"}), ("/api/", "/v1/")),
            ("read", f"{prefix}_RATE_LIMIT_READ", frozenset(), ())):
        rate = web.parse_rate(os.environ.get(env_name), web.parse_rate(DEFAULT_RATES[name]))
        if rate:
            rules.append(web.RateRule(name, rate[0], rate[1], methods, prefixes))
    return rules


def _audit_path(prefix: str) -> str | None:
    """The hash-chained audit trail of auth and admin events: ``<PREFIX>_AUDIT_LOG``, else ``audit.jsonl`` in the
    app's data directory when it is writable; ``off`` keeps only the log lines."""
    v = os.environ.get(f"{prefix}_AUDIT_LOG")
    if v and v.lower() in ("off", "0", "none"):
        return None
    if v:
        return v
    d = os.environ.get(f"{prefix}_DATA_DIR")
    if not d:
        try:
            d = str(data_dir(prefix.lower(), prefix))
        except OSError:
            return None
    if os.access(d, os.W_OK):
        return os.path.join(d, "audit.jsonl")
    return None


def _users_file(path: str | None) -> dict[str, str]:
    """``user:hash`` lines (htpasswd style; hashes from ``python -m pinneapple_security hash-password``)."""
    if not path:
        return {}
    users = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and ":" in line:
                u, _, h = line.partition(":")
                if not is_hash(h.strip()):
                    raise ValueError(f"{path}: the password of {u!r} is not a hash")
                users[u.strip()] = h.strip()
    return users


def _users(app: FastAPI) -> dict[str, str]:
    users = dict(getattr(app.state, "extra_users", None) or {})
    if app.state.auth_user and app.state.auth_password:
        users[app.state.auth_user] = app.state.auth_password
    return users


class _Verifier:
    """Password checks with a short in-memory cache of successful ones: Basic auth sends the password with every
    request, and a memory-hard hash per static file would cost seconds per page. Only digests keyed with a
    per-process random key are kept, never passwords."""

    def __init__(self, ttl: float = 600.0, max_entries: int = 1024):
        self.ttl, self.max_entries, self._key, self._ok = ttl, max_entries, secrets.token_bytes(32), {}

    def __call__(self, password: str, stored: str) -> bool:
        if not is_hash(stored):                               # legacy plain-text <PREFIX>_PASSWORD
            return secrets.compare_digest(password.encode(), stored.encode())
        d = hmac.new(self._key, f"{stored}\0{password}".encode(), hashlib.sha256).digest()
        now = time.monotonic()
        if self._ok.get(d, 0) > now:
            return True
        if verify_password(password, stored):
            if len(self._ok) >= self.max_entries:
                self._ok = {k: t for k, t in self._ok.items() if t > now}
                if len(self._ok) >= self.max_entries:
                    self._ok.clear()
            self._ok[d] = now + self.ttl
            return True
        return False


def _csp_for(app: FastAPI, prefix: str):
    """The CSP (``<PREFIX>_CSP`` overrides it): hashes of the inline scripts and handlers of the HTML files in the
    app's static mounts, collected on first use (the mounts are added after ``install``)."""
    fixed = os.environ.get(f"{prefix}_CSP")
    cache: dict[str, str] = {}

    def csp() -> str:
        if fixed:
            return fixed
        if "v" not in cache:
            scripts, handlers = set(), set()
            for route in app.routes:
                directory = getattr(getattr(route, "app", None), "directory", None)
                if directory and os.path.isdir(directory):
                    for page in Path(directory).rglob("*.html"):
                        s, h = web.inline_hashes(page.read_text(encoding="utf-8", errors="replace"))
                        scripts |= s
                        handlers |= h
            cache["v"] = web.build_csp(scripts, handlers)
        return cache["v"]

    return csp


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


def _basic_login(request: Request, users: dict[str, str], verify) -> tuple[str | None, str | None]:
    """(the logged-in user or None, the user name tried or None)."""
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("basic "):
        return None, None
    try:
        u, _, p = base64.b64decode(header[6:], validate=True).decode().partition(":")
    except (ValueError, UnicodeDecodeError):
        return None, "?"
    stored = users.get(u)
    if stored is None:
        verify(p, _DUMMY_HASH)                                # same work for unknown users (no user enumeration)
        return None, u[:64] or "?"
    return (u if verify(p, stored) else None), u


_DUMMY_HASH = "$scrypt$n=16384,r=8,p=1$AAAAAAAAAAAAAAAAAAAAAA$AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"


def client_ip(request: Request) -> str:
    """The client address. Behind Caddy (a trusted proxy: loopback/private by default, ``<PREFIX>_TRUSTED_PROXIES``
    or ``PINNEAPPLE_TRUSTED_PROXIES`` to change) the right-most X-Forwarded-For entry that is not a proxy; a client
    reaching the app directly can not choose its address with the header."""
    peer = request.client.host if request.client else None
    trusted = getattr(request.app.state, "trusted_proxies", None) or web.DEFAULT_TRUSTED_PROXIES
    return web.client_ip(peer, request.headers.get("x-forwarded-for"), trusted)


def principal(request: Request) -> str:
    """Who use is counted against: ``key:<id>``, ``user:<login>`` or ``ip:<addr>``."""
    return getattr(request.state, "principal", None) or f"ip:{client_ip(request)}"


class Runtime:
    """The job queue, API keys, usage meter and upload store of one app (``app.state.runtime``)."""

    def __init__(self, name: str, prefix: str, root: Path, jobs: JobQueue, keys: APIKeyStore, meter: UsageMeter,
                 uploads: UploadStore, admin_token: str | None, require_key: bool):
        self.name, self.prefix, self.root = name, prefix, root
        self.jobs, self.keys, self.meter, self.uploads = jobs, keys, meter, uploads
        self.admin_token, self.require_key = admin_token, require_key

    def is_admin(self, request: Request) -> bool:
        login = getattr(request.state, "login", None)
        if login:
            admins = getattr(request.app.state, "admin_users", None)
            return admins is None or login in admins
        token = request.headers.get("x-admin-token")
        return bool(self.admin_token and token and secrets.compare_digest(token, self.admin_token))

    def consume(self, request: Request, unit: str, amount: float = 1, ref: str = "") -> None:
        """Count ``amount`` of ``unit`` against the caller; a 429 response if it passes the free limit."""
        self.meter.consume(principal(request), unit, amount, ref)

    def submit(self, request: Request, kind: str, params: dict[str, Any] | None = None, *,
               units: dict[str, float] | None = None, max_retries: int = 0) -> dict[str, Any]:
        """Check the free limits for ``units``, queue the job owned by the caller and count the units."""
        who = principal(request)
        for unit, amount in (units or {}).items():
            self.meter.check(who, unit, amount)
        job_id = self.jobs.submit(kind, params, owner=who, max_retries=max_retries)
        for unit, amount in (units or {}).items():
            self.meter.record(who, unit, amount, ref=job_id)
        return self.jobs.get(job_id)

    def job_for(self, request: Request, job_id: str) -> dict[str, Any]:
        """The job if the caller owns it (or is admin); 404 otherwise."""
        try:
            job = self.jobs.get(job_id)
        except KeyError:
            raise HTTPException(404, f"unknown job {job_id}") from None
        if job["owner"] != principal(request) and not self.is_admin(request):
            raise HTTPException(404, f"unknown job {job_id}")
        return job

    def upload_path(self, request: Request, upload_id: str) -> Path:
        try:
            meta = self.uploads.get(upload_id)
        except KeyError:
            raise HTTPException(404, f"unknown upload {upload_id}") from None
        if meta.get("owner") != principal(request) and not self.is_admin(request):
            raise HTTPException(404, f"unknown upload {upload_id}")
        return self.uploads.path(upload_id)

    def describe(self) -> dict[str, Any]:
        return {"app": self.name, "max_concurrent_jobs": self.jobs.max_concurrent, "executor": self.jobs.executor,
                "job_kinds": sorted(self.jobs.stats()["kinds"]), "free_limits_per_month": self.meter.limits,
                "upload_max_mb": self.uploads.max_bytes / 2**20,
                "upload_suffixes": sorted(self.uploads.allowed) if self.uploads.allowed else None,
                "api_key_required": self.require_key}


def install_runtime(app: FastAPI, prefix: str, name: str, *, units: dict[str, float | None] | None = None,
                    max_jobs: int = 2, executor: str = "thread", job_max_seconds: float | None = None,
                    upload_max_mb: float = 50, upload_suffixes: set[str] | None = None) -> Runtime:
    """Job queue, API keys, usage metering and uploads for an app, with their routes under ``/api``.

    ``units`` are the metered units and their default free limit per principal and month (None = unlimited).
    Call after :func:`install`. Register job functions with ``rt.jobs.register(kind, fn)``.
    """
    env = os.environ
    root = data_dir(name, prefix)
    db = root / "runtime.sqlite"
    jobs = JobQueue(db, app=name, max_concurrent=int(env.get(f"{prefix}_MAX_JOBS") or max_jobs),
                    executor=env.get(f"{prefix}_JOB_EXECUTOR") or executor, workdir=root / "jobs",
                    max_seconds=float(env.get(f"{prefix}_JOB_MAX_SECONDS") or 0) or job_max_seconds,
                    limits={"memory_mb": float(env.get(f"{prefix}_JOB_MAX_MEMORY_MB") or 0),
                            "cpu_seconds": float(env.get(f"{prefix}_JOB_MAX_CPU_S") or 0)},
                    log_tracebacks=(env.get(f"{prefix}_JOB_TRACEBACKS") or "").lower() in ("1", "true", "yes"))
    meter = UsageMeter(db, app=name, limits=limits_from_env(prefix, dict(units or {})))
    uploads = UploadStore(root / "uploads", max_bytes=int(float(env.get(f"{prefix}_UPLOAD_MAX_MB") or upload_max_mb)
                                                          * 2**20), allowed_suffixes=upload_suffixes)
    rt = Runtime(name, prefix, root, jobs, APIKeyStore(db, app=name), meter, uploads,
                 admin_token=env.get(f"{prefix}_ADMIN_TOKEN") or None,
                 require_key=(env.get(f"{prefix}_REQUIRE_API_KEY") or "").lower() in ("1", "true", "yes"))
    app.state.runtime = rt

    @app.exception_handler(UsageLimitExceeded)
    async def _limit(request: Request, exc: UsageLimitExceeded):
        return JSONResponse(exc.to_dict(), status_code=429)

    api = APIRouter(prefix="/api", tags=["runtime"])

    def _audit(request: Request, action: str, **data):
        audit = getattr(app.state, "audit", None)
        if audit is not None:
            audit.event(action, principal(request), ip=client_ip(request), **data)

    def admin(request: Request) -> None:
        if not rt.is_admin(request):
            _audit(request, "admin_denied", target=request.url.path, outcome="denied")
            raise HTTPException(403, f"needs the app login or X-Admin-Token ({prefix}_ADMIN_TOKEN)")

    @api.get("/runtime")
    def runtime_info():
        return rt.describe()

    @api.get("/jobs")
    def list_jobs(request: Request, status: str | None = None, kind: str | None = None, limit: int = 50):
        owner = None if rt.is_admin(request) else principal(request)
        return rt.jobs.list(owner=owner, status=status, kind=kind, limit=min(limit, 500))

    @api.get("/jobs/{job_id}")
    def get_job(request: Request, job_id: str):
        return rt.job_for(request, job_id)

    @api.get("/jobs/{job_id}/logs")
    def job_logs(request: Request, job_id: str, since: int = 0):
        rt.job_for(request, job_id)
        return rt.jobs.logs(job_id, since=since)

    @api.post("/jobs/{job_id}/cancel")
    def cancel_job(request: Request, job_id: str):
        rt.job_for(request, job_id)
        return rt.jobs.cancel(job_id)

    @api.post("/jobs/{job_id}/retry")
    def retry_job(request: Request, job_id: str):
        rt.job_for(request, job_id)
        try:
            return rt.jobs.retry(job_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None

    @api.get("/usage")
    def usage(request: Request):
        out = rt.meter.usage(principal(request))
        if rt.is_admin(request):
            out["totals"] = rt.meter.totals()
        return out

    @api.get("/keys")
    def list_keys(request: Request):
        admin(request)
        return rt.keys.list()

    @api.post("/keys")
    def create_key(request: Request, body: dict[str, Any]):
        admin(request)
        key = rt.keys.create(str(body.get("name") or "unnamed")[:100], scopes=body.get("scopes"))
        _audit(request, "api_key_created", target=str(key.get("id")), name=key.get("name"))
        return key

    @api.delete("/keys/{key_id}")
    def revoke_key(request: Request, key_id: str):
        admin(request)
        try:
            out = rt.keys.revoke(key_id)
        except KeyError:
            raise HTTPException(404, f"unknown key {key_id}") from None
        _audit(request, "api_key_revoked", target=key_id)
        return out

    @api.post("/uploads")
    async def upload(request: Request, file: UploadFile):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > rt.uploads.max_bytes + 64 * 1024:
            raise HTTPException(413, f"upload larger than {rt.uploads.max_bytes / 2**20:g} MB")
        try:
            return rt.uploads.save(file.filename or "upload", file.file, owner=principal(request))
        except UploadTooLarge as exc:
            raise HTTPException(413, str(exc)) from None
        except UploadRejected as exc:
            raise HTTPException(415, str(exc)) from None

    @api.get("/uploads")
    def list_uploads(request: Request):
        return rt.uploads.list(owner=None if rt.is_admin(request) else principal(request))

    @api.delete("/uploads/{upload_id}")
    def delete_upload(request: Request, upload_id: str):
        rt.upload_path(request, upload_id)
        rt.uploads.delete(upload_id)
        return {"deleted": upload_id}

    app.include_router(api)
    return rt

"""HTTP hardening for the PINNeAPPle web apps (ASGI: FastAPI / Starlette), with no dependency beyond the standard
library at import time.

=============================  ========================================================================================
:class:`SecurityHeaders`       CSP (with hashes of the app's own inline scripts), HSTS, nosniff, frame-ancestors,
                               Referrer-Policy, Permissions-Policy, COOP; ``Server`` header removed
:class:`BodyLimit`             413 for a request body above a limit, counted while streaming (not only Content-Length)
:class:`RateLimiter`           token buckets per key (principal, IP), bounded memory; 429 + ``Retry-After``
:class:`LoginThrottle`         lockout after repeated failed logins per IP and per user name
:func:`client_ip`              the caller's address; ``X-Forwarded-For`` is believed only from a trusted proxy
:func:`same_origin_ok`         CSRF check for state-changing requests from browsers (Fetch Metadata + Origin)
:func:`check_cors_origins`     an explicit CORS allow-list; refuses ``*`` together with credentials
:class:`AuthAudit`             auth and admin events as structured log lines and in a hash-chained audit trail
:func:`check_outbound_url`     SSRF guard for a URL the server is asked to fetch (no loopback, link-local, metadata
                               or private addresses unless the host is allow-listed)
=============================  ========================================================================================

The apps get all of this through ``apps/_shared/appkit.py`` ``install()``; the environment variables are listed in
``apps/_shared/README.md``. Rate-limit and lockout state is per worker process (in memory): with ``--workers 2`` the
effective budget is up to twice the configured one. The proxy in front (Caddy) is the place for global limits.
"""
from __future__ import annotations

import base64
import hashlib
import html
import ipaddress
import json
import logging
import math
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["SecurityHeaders", "BodyLimit", "RateLimiter", "RateRule", "LoginThrottle", "client_ip", "parse_rate",
           "same_origin_ok", "check_cors_origins", "inline_hashes", "build_csp", "AuthAudit", "DEFAULT_TRUSTED_PROXIES",
           "PERMISSIONS_POLICY", "check_outbound_url", "UnsafeURL"]

log = logging.getLogger("pinneapple.security")

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

PERMISSIONS_POLICY = ("accelerometer=(), camera=(), geolocation=(self), gyroscope=(), magnetometer=(), "
                      "microphone=(), payment=(), usb=(), fullscreen=(self), browsing-topics=()")
DEFAULT_TRUSTED_PROXIES = ("127.0.0.0/8", "::1/128", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7")


# ------------------------------------------------------------------------------------------------------- client IP
def _networks(cidrs: Iterable[str]) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    return [ipaddress.ip_network(c.strip(), strict=False) for c in cidrs if c and c.strip()]


def client_ip(peer: str | None, forwarded_for: str | None,
              trusted: Sequence[str] | Sequence[ipaddress.IPv4Network | ipaddress.IPv6Network] = DEFAULT_TRUSTED_PROXIES
              ) -> str:
    """The caller's address. ``X-Forwarded-For`` is used only when the direct peer is a trusted proxy (by default a
    loopback or private address: the Docker network Caddy sits on), and then the right-most entry that is not itself
    a trusted proxy is taken. A client that reaches the app directly from a public address can not choose its
    identity by sending the header."""
    nets = trusted if trusted and not isinstance(trusted[0], str) else _networks(trusted)  # type: ignore[arg-type]

    def is_trusted(a: str) -> bool:
        try:
            ip = ipaddress.ip_address(a)
        except ValueError:
            return False
        return any(ip in n for n in nets)

    peer = peer or "unknown"
    if not forwarded_for or not is_trusted(peer):
        return peer
    hops = [h.strip() for h in forwarded_for.split(",") if h.strip()]
    for hop in reversed(hops):
        if not is_trusted(hop):
            return hop
    return hops[0] if hops else peer


# --------------------------------------------------------------------------------------------------------- headers
_SCRIPT = re.compile(r"<script\b(?P<attrs>[^>]*)>(?P<body>.*?)</script\s*>", re.I | re.S)
_HANDLER = re.compile(r"\son[a-z]+\s*=\s*(?:\"(?P<d>[^\"]*)\"|'(?P<s>[^']*)')", re.I)


def _sha(text: str) -> str:
    return "'sha256-" + base64.b64encode(hashlib.sha256(text.encode()).digest()).decode() + "'"


def inline_hashes(page: str) -> tuple[set[str], set[str]]:
    """CSP hashes of the inline ``<script>`` blocks (including import maps) and of the inline event handlers
    (``onsubmit="..."``) of an HTML page."""
    scripts = {_sha(m["body"]) for m in _SCRIPT.finditer(page)
               if not re.search(r"\bsrc\s*=", m["attrs"], re.I) and m["body"].strip()}
    handlers = {_sha(html.unescape(m["d"] if m["d"] is not None else m["s"])) for m in _HANDLER.finditer(page)}
    return scripts, handlers


def build_csp(script_hashes: Iterable[str] = (), handler_hashes: Iterable[str] = (), *,
              connect_src: Iterable[str] = (), img_src: Iterable[str] = (), frame_ancestors: str = "'none'") -> str:
    """The Content-Security-Policy of an app page: everything from the app's own origin, the app's own inline
    scripts by hash (no ``'unsafe-inline'`` for scripts), WebGL/three.js friendly (``blob:`` workers and images)."""
    script = ["'self'", *sorted(script_hashes)]
    if handler_hashes:
        script += ["'unsafe-hashes'", *sorted(handler_hashes)]
    return "; ".join([
        "default-src 'self'",
        "script-src " + " ".join(script),
        "style-src 'self' 'unsafe-inline'",
        "img-src " + " ".join(["'self'", "data:", "blob:", *img_src]),
        "font-src 'self' data:",
        "connect-src " + " ".join(["'self'", *connect_src]),
        "worker-src 'self' blob:",
        "media-src 'self' blob:",
        "frame-src 'self'",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
        f"frame-ancestors {frame_ancestors}",
    ])


class SecurityHeaders:
    """ASGI middleware adding the security headers to every response (an existing header set by the route wins).

    ``csp``: a fixed policy, or a callable returning it (called once per response, so it can be computed lazily from
    the app's static pages). ``hsts``: the Strict-Transport-Security value, or "" to leave it out."""

    def __init__(self, app: ASGIApp, csp: str | Callable[[], str] | None = None,
                 hsts: str = "max-age=31536000; includeSubDomains", frame_options: str = "DENY",
                 extra: Mapping[str, str] | None = None):
        self.app, self.csp, self.hsts, self.frame_options = app, csp, hsts, frame_options
        self.extra = dict(extra or {})

    def headers(self) -> list[tuple[bytes, bytes]]:
        csp = self.csp() if callable(self.csp) else (self.csp if self.csp is not None else build_csp())
        h = {"content-security-policy": csp, "x-content-type-options": "nosniff",
             "x-frame-options": self.frame_options, "referrer-policy": "strict-origin-when-cross-origin",
             "permissions-policy": PERMISSIONS_POLICY, "cross-origin-opener-policy": "same-origin",
             "x-permitted-cross-domain-policies": "none", **self.extra}
        if self.hsts:
            h["strict-transport-security"] = self.hsts
        return [(k.encode(), v.encode()) for k, v in h.items() if v]

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        add = self.headers()

        async def send_wrapper(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", []) if k.lower() != b"server"]
                have = {k.lower() for k, _ in headers}
                headers += [(k, v) for k, v in add if k not in have]
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_wrapper)


# ------------------------------------------------------------------------------------------------------ body limit
async def _send_json(send: Send, status: int, body: dict[str, Any], headers: Sequence[tuple[bytes, bytes]] = ()
                     ) -> None:
    data = json.dumps(body).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(data)).encode()),
                            *headers]})
    await send({"type": "http.response.body", "body": data})


class _TooLarge(Exception):
    pass


def _only_too_large(exc: BaseException) -> bool:
    """True for _TooLarge, also when a task group (Starlette's BaseHTTPMiddleware) wrapped it in an ExceptionGroup."""
    if isinstance(exc, _TooLarge):
        return True
    subs = getattr(exc, "exceptions", None)
    return bool(subs) and all(_only_too_large(e) for e in subs)


class BodyLimit:
    """ASGI middleware: 413 for a request body above ``max_bytes`` (``Content-Length`` checked first, then the bytes
    actually received, so chunked uploads can not pass the limit). ``limits``: per path prefix overrides."""

    def __init__(self, app: ASGIApp, max_bytes: int, limits: Mapping[str, int] | None = None):
        self.app, self.max_bytes = app, int(max_bytes)
        self.limits = sorted((limits or {}).items(), key=lambda kv: -len(kv[0]))

    def limit_for(self, path: str) -> int:
        for prefix, n in self.limits:
            if path.startswith(prefix):
                return int(n)
        return self.max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = self.limit_for(scope.get("path", ""))
        msg = {"detail": f"request body larger than {limit / 2**20:g} MB"}
        for k, v in scope.get("headers", []):
            if k == b"content-length":
                try:
                    if int(v) > limit:
                        await _send_json(send, 413, msg, [(b"connection", b"close")])
                        return
                except ValueError:
                    await _send_json(send, 400, {"detail": "invalid Content-Length"})
                    return
        seen, started = 0, False

        async def counted() -> dict[str, Any]:
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > limit:
                    raise _TooLarge
            return message

        async def tracking_send(message: dict[str, Any]) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, counted, tracking_send)
        except BaseException as exc:
            if not _only_too_large(exc):
                raise
            if not started:
                await _send_json(send, 413, msg, [(b"connection", b"close")])


# ---------------------------------------------------------------------------------------------------- rate limits
def parse_rate(spec: str | None, default: tuple[float, float] | None = None) -> tuple[float, float] | None:
    """``"30/60"`` (30 requests per 60 s), ``"30/min"``, ``"5/s"``, ``"1000/h"``; ``off``/``0`` → None."""
    if spec is None or not str(spec).strip():
        return default
    s = str(spec).strip().lower()
    if s in ("off", "0", "none", "unlimited", "-1"):
        return None
    n, _, per = s.partition("/")
    units = {"s": 1, "sec": 1, "second": 1, "m": 60, "min": 60, "minute": 60, "h": 3600, "hour": 3600}
    seconds = units.get(per, None) if per else 60.0
    if seconds is None:
        seconds = float(per)
    count = float(n)
    if count <= 0 or seconds <= 0:
        raise ValueError(f"invalid rate {spec!r}")
    return count, float(seconds)


@dataclass
class RateRule:
    """``count`` requests per ``seconds`` for requests whose method is in ``methods`` (empty: any) and whose path
    starts with one of ``prefixes`` (empty: any)."""
    name: str
    count: float
    seconds: float
    methods: frozenset[str] = frozenset()
    prefixes: tuple[str, ...] = ()

    def matches(self, method: str, path: str) -> bool:
        return ((not self.methods or method.upper() in self.methods)
                and (not self.prefixes or path.startswith(self.prefixes)))


class RateLimiter:
    """Token buckets per ``(rule, key)``: a burst up to ``count`` and a steady ``count/seconds`` refill. Memory is
    bounded (least recently used keys are dropped past ``max_keys``). Thread-safe."""

    def __init__(self, rules: Iterable[RateRule] = (), max_keys: int = 50_000, clock: Callable[[], float] = time.monotonic):
        self.rules = list(rules)
        self.max_keys, self.clock = max_keys, clock
        self._buckets: OrderedDict[tuple[str, str], tuple[float, float]] = OrderedDict()
        self._lock = threading.Lock()

    def hit(self, key: str, rule: RateRule, cost: float = 1.0) -> float:
        """Take ``cost`` tokens; 0.0 if allowed, else the seconds until it would be."""
        now = self.clock()
        rate = rule.count / rule.seconds
        with self._lock:
            tokens, last = self._buckets.pop((rule.name, key), (rule.count, now))
            tokens = min(rule.count, tokens + (now - last) * rate)
            if tokens >= cost:
                tokens -= cost
                wait = 0.0
            else:
                wait = (cost - tokens) / rate
            self._buckets[(rule.name, key)] = (tokens, now)
            while len(self._buckets) > self.max_keys:
                self._buckets.popitem(last=False)
        return wait

    def check(self, key: str, method: str, path: str) -> tuple[float, RateRule | None]:
        """Apply every matching rule; the longest wait and its rule, or ``(0.0, None)``."""
        worst, which = 0.0, None
        for rule in self.rules:
            if rule.matches(method, path):
                w = self.hit(key, rule)
                if w > worst:
                    worst, which = w, rule
        return worst, which

    @staticmethod
    def retry_after(wait: float) -> str:
        return str(max(1, math.ceil(wait)))


@dataclass
class LoginThrottle:
    """Lock a client IP or a user name out for ``lock_seconds`` after ``max_failures`` failed logins within
    ``window`` seconds. A success clears the IP's and the user's failure count."""
    max_failures: int = 10
    window: float = 900.0
    lock_seconds: float = 900.0
    clock: Callable[[], float] = time.monotonic
    max_keys: int = 50_000
    _fails: OrderedDict = field(default_factory=OrderedDict, repr=False)
    _locked: dict = field(default_factory=dict, repr=False)
    _lock: Any = field(default_factory=threading.Lock, repr=False)

    def _keys(self, ip: str, user: str | None) -> list[str]:
        return [f"ip:{ip}"] + ([f"user:{user}"] if user else [])

    def locked_for(self, ip: str, user: str | None = None) -> float:
        """Seconds left on a lockout of this IP or user (0.0 when not locked)."""
        now = self.clock()
        with self._lock:
            left = [until - now for k in self._keys(ip, user) if (until := self._locked.get(k)) and until > now]
        return max(left, default=0.0)

    def failure(self, ip: str, user: str | None = None) -> bool:
        """Record a failed login; True when it starts a lockout."""
        now, started = self.clock(), False
        with self._lock:
            for k in self._keys(ip, user):
                times = [t for t in self._fails.pop(k, []) if now - t < self.window] + [now]
                self._fails[k] = times
                if len(times) >= self.max_failures:
                    self._locked[k] = now + self.lock_seconds
                    self._fails.pop(k, None)
                    started = True
            while len(self._fails) > self.max_keys:
                self._fails.popitem(last=False)
            if len(self._locked) > self.max_keys:
                self._locked = {k: v for k, v in self._locked.items() if v > now}
        return started

    def success(self, ip: str, user: str | None = None) -> None:
        with self._lock:
            for k in self._keys(ip, user):
                self._fails.pop(k, None)


# ------------------------------------------------------------------------------------------------------ CSRF, CORS
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})


def _origin_of(url: str) -> str:
    m = re.match(r"^([a-z][a-z0-9+.-]*)://([^/?#]+)", url.strip(), re.I)
    return f"{m.group(1).lower()}://{m.group(2).lower()}" if m else ""


def same_origin_ok(method: str, headers: Mapping[str, str], allowed_origins: Iterable[str] = ()) -> bool:
    """False for a state-changing browser request coming from another site (cross-site request forgery: a page
    elsewhere posting to the app with the user's cached Basic-auth credentials). Uses the browser-set
    ``Sec-Fetch-Site`` header, then ``Origin`` compared with ``Host`` (``X-Forwarded-Host`` behind a proxy) and the
    configured ``allowed_origins``. Requests without these headers (curl, scripts, API clients) pass: they carry no
    ambient browser credentials."""
    if method.upper() in _SAFE_METHODS:
        return True
    allowed = {_origin_of(o) for o in allowed_origins if o}
    origin = headers.get("origin")
    if origin and origin != "null" and _origin_of(origin) in allowed:
        return True
    site = (headers.get("sec-fetch-site") or "").lower()
    if site in ("cross-site", "same-site"):
        return False
    if origin:
        if origin == "null":
            return False
        host = (headers.get("x-forwarded-host") or headers.get("host") or "").split(",")[0].strip().lower()
        return _origin_of(origin).split("://", 1)[-1] == host
    return True


def check_cors_origins(origins: Iterable[str], allow_credentials: bool) -> list[str]:
    """Validate a CORS allow-list: full ``scheme://host[:port]`` origins only, and no ``*`` with credentials."""
    out = []
    for o in origins:
        o = o.strip()
        if not o:
            continue
        if o == "*":
            if allow_credentials:
                raise ValueError("CORS: '*' can not be combined with credentials; list the origins")
            out.append(o)
            continue
        if _origin_of(o) != o.lower().rstrip("/") or not o.lower().startswith(("https://", "http://")):
            raise ValueError(f"CORS: {o!r} is not an origin (scheme://host[:port], no path)")
        out.append(o.rstrip("/"))
    return out


# ----------------------------------------------------------------------------------------------------------- SSRF
class UnsafeURL(ValueError):
    """A URL the server must not fetch."""


def check_outbound_url(url: str, allow_hosts: Iterable[str] = (), resolve: Callable[[str], list[str]] | None = None
                       ) -> str:
    """Refuse a URL a user asked the server to call unless it is http(s) to a public address, or its host is in
    ``allow_hosts`` (e.g. ``surrogate`` for an app on the same Docker network). Loopback, link-local (cloud metadata
    at 169.254.169.254), private, multicast and reserved addresses are refused, for every address the name resolves
    to. Returns the URL. The check runs when the URL is accepted; a name whose DNS changes later is not re-checked
    here, so keep the allow-list explicit for internal services."""
    import socket
    from urllib.parse import urlsplit
    parts = urlsplit(str(url))
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise UnsafeURL(f"only http(s) URLs with a host are allowed: {url!r}")
    if parts.username or parts.password:
        raise UnsafeURL("credentials in the URL are not allowed")
    host = parts.hostname.lower().rstrip(".")
    allowed = {h.strip().lower() for h in allow_hosts if h and h.strip()}
    if host in allowed:
        return url
    if resolve is None:
        def resolve(h: str) -> list[str]:
            try:
                return sorted({ai[4][0] for ai in socket.getaddrinfo(h, None)})
            except socket.gaierror as e:
                raise UnsafeURL(f"cannot resolve {h!r}: {e}") from None
    for addr in resolve(host):
        ip = ipaddress.ip_address(addr.split("%", 1)[0])
        if (ip.is_loopback or ip.is_link_local or ip.is_private or ip.is_multicast or ip.is_reserved
                or ip.is_unspecified):
            raise UnsafeURL(f"{host} resolves to a non-public address ({ip}); allow-list the host to use it")
    return url


# ---------------------------------------------------------------------------------------------------------- audit
class AuthAudit:
    """Auth and admin events (login ok/failed/locked, API key created/revoked, admin denied, rate limited) as one JSON
    log line each on the ``pinneapple.security`` logger, and, with ``path``, in a hash-chained
    :class:`pinneapple_security.audit.AuditLog` (``key``: HMAC key for the chain). Secrets are never logged."""

    def __init__(self, app_name: str, path: str | None = None, key: bytes | None = None):
        self.app_name = app_name
        self.trail = None
        if path:
            from .audit import AuditLog
            self.trail = AuditLog(path, key=key)
        self._lock = threading.Lock()

    def event(self, action: str, actor: str, *, ip: str = "", target: str = "", outcome: str = "ok",
              **data: Any) -> None:
        rec = {"event": action, "app": self.app_name, "actor": actor, "ip": ip, "target": target,
               "outcome": outcome, **data}
        level = logging.INFO if outcome == "ok" else logging.WARNING
        log.log(level, json.dumps(rec, sort_keys=True, default=str))
        if self.trail is not None:
            try:
                with self._lock:
                    self.trail.append(action, actor or "anonymous", target=target,
                                      data={"ip": ip, "outcome": outcome, "app": self.app_name, **data})
            except OSError as e:                              # never let the audit trail take the app down
                log.error("audit trail write failed: %s", e)

"""Serve the lab catalogue over HTTP: the browsable page, every run's files, a JSON API and dataset downloads.

    python -m pinneapple_lab serve --root lab --port 8093          # http://localhost:8093

Standard library only (no web framework). The page is rebuilt when the database changes (new runs from a sweep
running next to the server, or a ``reindex`` every ``--refresh`` seconds), thumbnails are made on demand and kept in
memory, and dataset exports are cached on disk under ``<root>/_exports``.

Routes::

    GET /                                    the catalogue page
    GET /health                              liveness
    GET /api/catalog                         everything the page shows (experiments, runs, metrics, checks, datasets)
    GET /api/status                          run counts per experiment and status
    GET /api/runs?experiment=NAME&status=S   run index rows
    GET /api/runs/RUN_ID                     one run: record, metrics, validation, file list
    GET /api/datasets                        dataset cards
    GET /api/curation                        quality tier, dimensions and readiness of every run and experiment
    GET /report.html|.md|.pdf?items=&experiment=&tier=&use=&ready=&tag=   a report for one item or a filtered set
    GET /files/EXPERIMENT/RUN_ID/PATH        a run's file (figures, inputs, outputs, code, logs, cards)
    GET /thumb/EXPERIMENT/RUN_ID/PATH        a JPEG thumbnail of a run's figure
    GET /download/EXPERIMENT/DATASET.npz     a dataset over all completed runs, as one .npz
    GET /experiments                         every experiment: A-F seal, trust card, metrics, next steps, gallery
    GET /api/experiments?letter=&domain=&method=&ready=1&q=&sort=letter|date   the page's entries, filtered
    GET /api/experiment?key=KEY              one entry
    GET /api/coverage                        experiment -> curated yes / no
    GET /stamped/EXPERIMENT/RUN_ID/PATH?key=ITEM   a run's figure stamped with the item's seal (JPEG)
    GET /seal.png?key=ITEM&lang=en|pt        an item's seal
    GET /repo/PATH, /repo-thumb/PATH, /stamped-repo/PATH?key=ITEM   files listed by an experiment manifest

Set ``LAB_USER`` and ``LAB_PASSWORD`` to require HTTP Basic login on everything except ``/health``.
"""
from __future__ import annotations

import base64
import hmac
import json
import mimetypes
import os
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .store import LabStore

_FILE_TYPES = {".png", ".gif", ".jpg", ".jpeg", ".svg", ".mp4", ".webm", ".json", ".jsonl", ".csv", ".txt", ".log",
               ".md", ".py", ".diff", ".npy", ".npz", ".html", ".yaml", ".yml", ".js", ".mjs", ".css",
               ".glb", ".bin", ".wasm"}


class LabServer:
    """The catalogue and files of one lab folder. ``handler()`` gives the request handler class."""

    def __init__(self, root: str | None = None, *, refresh: float = 30.0, user: str | None = None,
                 password: str | None = None):
        self.store = LabStore(root)
        self.root = os.path.abspath(self.store.root)
        self.refresh = refresh
        self.user = user if user is not None else os.environ.get("LAB_USER", "")
        self.password = password if password is not None else (os.environ.get("LAB_PASSWORD_HASH")
                                                               or os.environ.get("LAB_PASSWORD", ""))
        from pinneapple_security.web import LoginThrottle
        self.throttle = LoginThrottle(max_failures=int(os.environ.get("LAB_LOGIN_MAX_FAILURES") or 10))
        self._lock = threading.Lock()
        self._page: tuple[Any, bytes, bytes] | None = None     # (signature, html, catalog json)
        self._thumbs: dict[tuple[str, float], bytes] = {}
        self._last_reindex = 0.0
        self.store.reindex()
        self._last_reindex = time.time()

    # ------------------------------------------------------------------ content
    def _signature(self) -> Any:
        idx = os.path.join(self.root, "index.sqlite")
        return (os.path.getmtime(idx) if os.path.exists(idx) else 0.0, json.dumps(self.store.status(), sort_keys=True))

    def catalog(self) -> tuple[bytes, bytes]:
        """(html, json) of the catalogue, rebuilt when the database changed."""
        if self.refresh and time.time() - self._last_reindex > self.refresh:
            with self._lock:
                if time.time() - self._last_reindex > self.refresh:
                    self.store.reindex()
                    self._last_reindex = time.time()
        sig = self._signature()
        with self._lock:
            if self._page is None or self._page[0] != sig:
                from .html import collect, render_page
                data = collect(self.store, thumbs="url")
                data["served"] = True
                self._page = (sig, render_page(data, standalone=True).encode(), json.dumps(data, default=str).encode())
            return self._page[1], self._page[2]

    def run_detail(self, run_id: str) -> dict[str, Any] | None:
        try:
            row = self.store.get(run_id)
        except KeyError:
            return None
        d = self.store.run_dir(row["experiment"], run_id)
        out: dict[str, Any] = {"index": row}
        for name in ("run.json", "metrics.json", "validation.json"):
            try:
                with open(os.path.join(d, name)) as f:
                    out[name[:-5]] = json.load(f)
            except (OSError, ValueError):
                pass
        out["files"] = sorted(os.path.relpath(os.path.join(dp, f), d) for dp, _, fs in os.walk(d) for f in fs)
        return out

    def resolve(self, rel: str) -> str | None:
        """A file under the lab's runs folder, or None (path traversal, unknown type or missing)."""
        path = os.path.realpath(os.path.join(self.root, "runs", rel))
        runs = os.path.realpath(os.path.join(self.root, "runs")) + os.sep
        if not path.startswith(runs) or not os.path.isfile(path):
            return None
        if os.path.splitext(path)[1].lower() not in _FILE_TYPES:
            return None
        return path

    def thumb(self, path: str) -> bytes | None:
        from .html import thumbnail_bytes
        key = (path, os.path.getmtime(path))
        with self._lock:
            if key in self._thumbs:
                return self._thumbs[key]
        data = thumbnail_bytes(path)
        if data is not None:
            with self._lock:
                if len(self._thumbs) > 4000:
                    self._thumbs.clear()
                self._thumbs[key] = data
        return data

    def experiments(self) -> dict[str, Any]:
        """The experiments page data (server URLs) and HTML, rebuilt with the catalogue."""
        sig = self._signature()
        cur_path = os.path.join(self.root, "curation.json")
        sig = (sig, os.path.getmtime(cur_path) if os.path.exists(cur_path) else 0.0)
        with self._lock:
            if getattr(self, "_exp", None) is None or self._exp[0] != sig:
                from .explorer import collect, media_urls, render
                data = media_urls(collect(self.store), "server")
                allowed = {x["repo"] for e in data["entries"] for x in e["media"] if "repo" in x}
                allowed |= {v for e in data["entries"] if e["source"] == "manifest" for v in e["viewers"]}
                self._exp = (sig, data, render(data).encode(), allowed)
            return {"data": self._exp[1], "html": self._exp[2], "allowed": self._exp[3]}

    def repo_file(self, rel: str) -> str | None:
        """A repository file an experiment manifest lists (nothing else under the checkout is served)."""
        from .manifest import REPO
        if rel not in self.experiments()["allowed"]:
            return None
        path = os.path.realpath(os.path.join(REPO, rel))
        return path if path.startswith(os.path.realpath(REPO) + os.sep) and os.path.isfile(path) else None

    def stamped(self, path: str, key: str) -> bytes | None:
        from .explorer import stamped_jpeg
        from .seal import load_curation
        ck = ("stamped", path, key, os.path.getmtime(path))
        with self._lock:
            if ck in self._thumbs:
                return self._thumbs[ck]
        data = stamped_jpeg(path, key, load_curation(self.store))
        if data is not None:
            with self._lock:
                if len(self._thumbs) > 4000:
                    self._thumbs.clear()
                self._thumbs[ck] = data
        return data

    def export(self, experiment: str, dataset: str) -> str | None:
        rows = self.store.datasets(experiment, dataset)
        if not rows:
            return None
        folder = os.path.join(self.root, "_exports")
        os.makedirs(folder, exist_ok=True)
        stamp = max((r.get("run_id") or "") for r in rows) + f"-{len(rows)}"
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in f"{experiment}__{dataset}__{stamp}")
        path = os.path.join(folder, safe + ".npz")
        if not os.path.exists(path):
            with self._lock:
                if not os.path.exists(path):
                    tmp = path + ".tmp.npz"
                    self.store.export_dataset(experiment, dataset, tmp)
                    os.replace(tmp, path)
        return path

    def authorised(self, header: str | None, ip: str = "") -> bool:
        """Basic login against ``LAB_USER`` and ``LAB_PASSWORD_HASH`` (or the plain ``LAB_PASSWORD``); repeated
        failures lock the client and the user name out (``LAB_LOGIN_MAX_FAILURES``, 15 minutes)."""
        if not (self.user or self.password):
            return True
        if not header or not header.startswith("Basic "):
            return False
        try:
            u, _, p = base64.b64decode(header[6:]).decode().partition(":")
        except (ValueError, UnicodeDecodeError):
            return False
        if self.throttle.locked_for(ip, u):
            return False
        from pinneapple_security.passwords import is_hash, verify_password
        ok = hmac.compare_digest(u, self.user) and (verify_password(p, self.password) if is_hash(self.password)
                                                    else hmac.compare_digest(p, self.password))
        if ok:
            self.throttle.success(ip, u)
        else:
            self.throttle.failure(ip, u)
        return ok

    # ------------------------------------------------------------------ HTTP
    def handler(self) -> type[BaseHTTPRequestHandler]:
        lab = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "PINNeAPPleLab/1"

            def log_message(self, fmt, *args):                       # quieter than the default stderr line per hit
                if os.environ.get("LAB_ACCESS_LOG"):
                    super().log_message(fmt, *args)

            def _send(self, code: int, body: bytes, ctype: str, *, cache: str = "no-cache", extra=None):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", cache)
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
                self.send_header("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            def _json(self, obj, code=200):
                self._send(code, json.dumps(obj, default=str).encode(), "application/json")

            def _file(self, path: str, *, download: str | None = None):
                ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
                if path.endswith((".js", ".mjs")):
                    ctype = "text/javascript; charset=utf-8"
                elif path.endswith(".glb"):
                    ctype = "model/gltf-binary"
                elif os.path.basename(path) == "index.html" and os.sep + "viewer" + os.sep in path:
                    ctype = "text/html; charset=utf-8"                 # a run's 3-D viewer page
                elif ctype.startswith("text/") or path.endswith((".py", ".diff", ".jsonl", ".md", ".log", ".html")):
                    ctype = "text/plain; charset=utf-8"                # never render any other run file as a page
                with open(path, "rb") as f:
                    body = f.read()
                extra = {"Content-Disposition": f'attachment; filename="{download}"'} if download else None
                self._send(200, body, ctype, cache="public, max-age=300", extra=extra)

            def do_HEAD(self):
                self.do_GET()

            def do_GET(self):
                url = urllib.parse.urlsplit(self.path)
                path = urllib.parse.unquote(url.path)
                if path == "/health":
                    return self._json({"ok": True, "runs": sum(sum(v.values()) for v in lab.store.status().values())})
                from pinneapple_security.web import client_ip
                ip = client_ip(self.client_address[0], self.headers.get("X-Forwarded-For"))
                if not lab.authorised(self.headers.get("Authorization"), ip):
                    return self._send(401, b"login required", "text/plain",
                                      extra={"WWW-Authenticate": 'Basic realm="PINNeAPPle Lab"'})
                q = dict(urllib.parse.parse_qsl(url.query))
                try:
                    if path in ("/", "/index.html"):
                        return self._send(200, lab.catalog()[0], "text/html; charset=utf-8")
                    if path == "/api/catalog":
                        return self._send(200, lab.catalog()[1], "application/json")
                    if path.startswith("/report."):
                        import tempfile

                        from .reports import write_report
                        fmt = path.rsplit(".", 1)[1]
                        if fmt not in ("html", "md", "pdf"):
                            return self._send(404, b"unknown format", "text/plain")
                        sp = lambda k: [v for v in q.get(k, "").split(",") if v] or None  # noqa: E731
                        with tempfile.TemporaryDirectory() as td:
                            out = write_report(lab.store, os.path.join(td, "report." + fmt), items=sp("items"),
                                               experiment=sp("experiment"), tier=sp("tier"), use=q.get("use") or None,
                                               ready=q.get("ready") or None, tag=sp("tag"), title=q.get("title"),
                                               filters=", ".join(f"{k}={v}" for k, v in q.items()))
                            with open(out, "rb") as fh:
                                body = fh.read()
                        ctype = {"html": "text/html; charset=utf-8", "md": "text/markdown; charset=utf-8",
                                 "pdf": "application/pdf"}[fmt]
                        extra = None if fmt == "html" else {"Content-Disposition": f'attachment; filename="lab_report.{fmt}"'}
                        return self._send(200, body, ctype, extra=extra)
                    if path == "/experiments":
                        return self._send(200, lab.experiments()["html"], "text/html; charset=utf-8")
                    if path == "/api/experiments":
                        from .explorer import filter_entries
                        ents = filter_entries(lab.experiments()["data"]["entries"], letter=q.get("letter"),
                                              domain=q.get("domain"), method=q.get("method"),
                                              ready=q.get("ready") in ("1", "true"), q=q.get("q"),
                                              sort=q.get("sort", "letter"))
                        return self._json({"count": len(ents), "counts": lab.experiments()["data"]["counts"],
                                           "entries": ents})
                    if path == "/api/experiment":
                        e = next((x for x in lab.experiments()["data"]["entries"] if x["key"] == q.get("key")), None)
                        return self._json(e) if e else self._json({"error": "unknown experiment"}, 404)
                    if path == "/api/coverage":
                        from .explorer import coverage
                        return self._json(coverage(lab.experiments()["data"]))
                    if path == "/seal.png":
                        from .explorer import seal_png
                        from .seal import load_curation
                        try:
                            body = seal_png(q.get("key", ""), load_curation(lab.store), q.get("lang", "en"))
                        except (KeyError, ValueError):
                            return self._send(404, b"unknown item", "text/plain")
                        return self._send(200, body, "image/png", cache="public, max-age=600")
                    if path.startswith("/stamped/") or path.startswith("/stamped-repo/"):
                        f = (lab.resolve(path[len("/stamped/"):]) if path.startswith("/stamped/")
                             else lab.repo_file(path[len("/stamped-repo/"):]))
                        data = lab.stamped(f, q.get("key", "")) if f else None
                        if data is None:
                            return self._send(404, b"not found", "text/plain")
                        return self._send(200, data, "image/jpeg", cache="public, max-age=600")
                    if path.startswith("/repo/") or path.startswith("/repo-thumb/"):
                        thumb = path.startswith("/repo-thumb/")
                        f = lab.repo_file(path[len("/repo-thumb/" if thumb else "/repo/"):])
                        if not f:
                            return self._send(404, b"not found", "text/plain")
                        if thumb:
                            data = lab.thumb(f)
                            return (self._send(200, data, "image/jpeg", cache="public, max-age=3600") if data
                                    else self._send(404, b"not found", "text/plain"))
                        return self._file(f)
                    if path == "/api/curation":
                        from .curation import curate
                        return self._json(curate(lab.store))
                    if path == "/api/status":
                        return self._json(lab.store.status())
                    if path == "/api/runs":
                        return self._json(lab.store.runs(q.get("experiment"), status=q.get("status")))
                    if path.startswith("/api/runs/"):
                        d = lab.run_detail(path[len("/api/runs/"):])
                        return self._json(d) if d else self._json({"error": "unknown run"}, 404)
                    if path == "/api/datasets":
                        return self._json(lab.store.datasets(q.get("experiment"), q.get("name")))
                    if path.startswith("/files/"):
                        f = lab.resolve(path[len("/files/"):])
                        return self._file(f) if f else self._send(404, b"not found", "text/plain")
                    if path.startswith("/thumb/"):
                        f = lab.resolve(path[len("/thumb/"):])
                        data = lab.thumb(f) if f else None
                        if data is None:
                            return self._send(404, b"not found", "text/plain")
                        return self._send(200, data, "image/jpeg", cache="public, max-age=3600")
                    if path.startswith("/download/") and path.endswith(".npz"):
                        parts = path[len("/download/"):-4].split("/")
                        if len(parts) == 2:
                            f = lab.export(*parts)
                            if f:
                                return self._file(f, download=f"{parts[0]}_{parts[1]}.npz")
                        return self._send(404, b"unknown dataset", "text/plain")
                except (BrokenPipeError, ConnectionResetError):       # the browser went away (a cancelled image)
                    return None
                except Exception as exc:                              # noqa: BLE001 - report, keep serving
                    return self._json({"error": f"{type(exc).__name__}: {exc}"}, int(HTTPStatus.INTERNAL_SERVER_ERROR))
                return self._send(404, b"not found", "text/plain")

        return Handler


def serve(root: str | None = None, *, host: str = "127.0.0.1", port: int = 8093, refresh: float = 30.0) -> None:
    """Serve the catalogue until interrupted."""
    lab = LabServer(root, refresh=refresh)
    httpd = ThreadingHTTPServer((host, port), lab.handler())
    print(f"PINNeAPPle Lab: {lab.root} on http://{host}:{port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()

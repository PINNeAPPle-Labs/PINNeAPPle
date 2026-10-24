"""HTTP front end of the model services (standard library only), with the routes of NVIDIA NIM."""
from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .registry import Registry, dumps

__all__ = ["make_server", "serve_in_thread"]


def _schema_errors(obj, schema) -> list[str]:
    """A small JSON-schema check (type, required, properties, enum) for ``nvext.guided_json``."""
    errs = []
    t = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "number": (int, float), "integer": int, "boolean": bool}
    if t in types and not isinstance(obj, types[t]):
        return [f"expected {t}, got {type(obj).__name__}"]
    if "enum" in schema and obj not in schema["enum"]:
        errs.append(f"{obj!r} not in {schema['enum']}")
    if t == "object":
        errs += [f"missing {k!r}" for k in schema.get("required", []) if k not in obj]
        for k, sub in schema.get("properties", {}).items():
            if k in obj:
                errs += [f"{k}: {e}" for e in _schema_errors(obj[k], sub)]
    return errs


def make_server(registry: Registry, host: str = "127.0.0.1", port: int = 8000, api_key: str | None = None):
    """A ``ThreadingHTTPServer`` serving ``registry``. ``api_key``: if set, requests need ``Authorization: Bearer``."""
    started = time.time()

    class Handler(BaseHTTPRequestHandler):
        server_version = "pinneapple-model-services/1"

        def log_message(self, *a):            # quiet
            pass

        def _send(self, code, obj):
            body = dumps(obj)
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _auth(self) -> bool:
            if api_key and self.headers.get("Authorization", "") != f"Bearer {api_key}":
                self._send(401, {"error": {"message": "invalid or missing API key", "type": "auth"}})
                return False
            return True

        def do_GET(self):
            if self.path in ("/v1/health/ready", "/v1/health/live"):
                return self._send(200, {"status": "ready" if "ready" in self.path else "live",
                                        "uptime_s": time.time() - started})
            if not self._auth():
                return
            if self.path == "/v1/models":
                return self._send(200, {"object": "list", "data": [s.info() for s in registry.list()]})
            if self.path.startswith("/v1/models/"):
                name = self.path.split("/v1/models/", 1)[1]
                if name in registry:
                    return self._send(200, registry.get(name).info())
                return self._send(404, {"error": {"message": f"unknown model {name!r}", "type": "not_found"}})
            self._send(404, {"error": {"message": f"no route {self.path}", "type": "not_found"}})

        def do_POST(self):
            if not self._auth():
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(n) or b"{}")
            except ValueError as e:
                return self._send(400, {"error": {"message": f"invalid JSON: {e}", "type": "bad_request"}})
            name = req.get("model", "")
            if name not in registry:
                return self._send(404, {"error": {"message": f"unknown model {name!r}", "type": "not_found"}})
            s = registry.get(name)
            try:
                if self.path == "/v1/infer":
                    if s.kind != "infer":
                        return self._send(400, {"error": {"message": f"{name} is a chat model", "type": "bad_request"}})
                    return self._send(200, s.run(req.get("inputs", {})))
                if self.path == "/v1/chat/completions":
                    if s.kind != "chat":
                        return self._send(400, {"error": {"message": f"{name} is not a chat model",
                                                          "type": "bad_request"}})
                    r = s.run(req)
                    content = json.dumps(r["outputs"]["spec"])
                    schema = (req.get("nvext") or {}).get("guided_json")
                    errs = _schema_errors(r["outputs"]["spec"], schema) if schema else []
                    return self._send(200, {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion",
                        "created": int(time.time()), "model": name,
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        "pinneapple": {"trust": r["trust"], "guided_json_errors": errs, "timing_ms": r["timing_ms"]}})
            except (KeyError, TypeError, ValueError) as e:
                return self._send(400, {"error": {"message": f"{type(e).__name__}: {e}", "type": "bad_request"}})
            self._send(404, {"error": {"message": f"no route {self.path}", "type": "not_found"}})

    return ThreadingHTTPServer((host, port), Handler)


def serve_in_thread(registry: Registry, host: str = "127.0.0.1", port: int = 0, api_key: str | None = None):
    """Start the server on a background thread; returns ``(server, base_url)`` (``base_url`` ends in ``/v1``).
    ``server.shutdown()`` stops it."""
    srv = make_server(registry, host, port, api_key)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://{host}:{srv.server_address[1]}/v1"

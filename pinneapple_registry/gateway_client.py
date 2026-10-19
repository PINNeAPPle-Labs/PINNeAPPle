"""``GatewayClient``: call any server that exposes the gateway API (``/v1/models/...``) from Python.

Standard library only (``urllib``), so it runs wherever PINNeAPPle does::

    from pinneapple_registry.gateway_client import GatewayClient

    gw = GatewayClient("https://models.pinneapple.org", api_key="...")
    print(gw.models())
    r = gw.predict("zoo-airfoil-cst-rans", {"T0": 0.15, ..., "alpha_deg": [2, 4]})
    print(r["outputs"]["Cl"], r["ood"])

Errors from the server raise ``GatewayError`` with the HTTP status and the server's message.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

__all__ = ["GatewayClient", "GatewayError"]


class GatewayError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status, self.message = status, message


class GatewayClient:
    def __init__(self, base_url: str, *, api_key: str | None = None, timeout: float = 30.0,
                 headers: dict[str, str] | None = None):
        self.base = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.headers = {"accept": "application/json", **(headers or {})}
        if api_key:
            self.headers["X-API-Key"] = api_key

    def _call(self, method: str, path: str, body: Any = None, query: dict[str, Any] | None = None) -> Any:
        url = self.base + path + (("?" + urllib.parse.urlencode({k: v for k, v in query.items() if v is not None}))
                                  if query else "")
        data = None if body is None else json.dumps(body).encode()
        headers = dict(self.headers)
        if data is not None:
            headers["content-type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:  # noqa: S310 (caller-chosen URL)
                return json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                msg = json.loads(raw).get("detail", raw.decode(errors="replace"))
            except (ValueError, AttributeError):
                msg = raw.decode(errors="replace")
            raise GatewayError(e.code, str(msg)) from None

    @staticmethod
    def _q(name: str) -> str:
        return urllib.parse.quote(name, safe="")

    def models(self) -> list[dict[str, Any]]:
        return self._call("GET", "/v1/models")["models"]

    def describe(self, name: str, version: str | None = None) -> dict[str, Any]:
        if version:
            return self._call("GET", f"/v1/models/{self._q(name)}/versions/{self._q(version)}")
        return self._call("GET", f"/v1/models/{self._q(name)}")

    def predict(self, name: str, inputs, *, version: str | None = None, ood_policy: str | None = None) -> dict[str, Any]:
        body = {"inputs": inputs}
        if ood_policy:
            body["ood_policy"] = ood_policy
        path = f"/v1/models/{self._q(name)}" + (f"/versions/{self._q(version)}" if version else "") + "/predict"
        return self._call("POST", path, body)

    def runs(self, name: str, limit: int = 100) -> list[dict[str, Any]]:
        return self._call("GET", f"/v1/models/{self._q(name)}/runs", query={"limit": limit})["runs"]

    def metrics(self, model: str | None = None) -> dict[str, Any]:
        return self._call("GET", "/v1/metrics", query={"model": model})

"""One client for PINNeAPPle Model Services and NVIDIA NIM (same routes): health, models, infer, chat."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

__all__ = ["ModelServiceClient", "ServiceError", "NIM_URL"]

NIM_URL = "https://integrate.api.nvidia.com/v1"


class ServiceError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


class ModelServiceClient:
    """``base_url`` ends in ``/v1``. For NVIDIA NIM on build.nvidia.com pass ``NIM_URL`` and an ``api_key``
    (default ``$NVIDIA_API_KEY``)."""

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 60.0):
        self.base = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else (os.environ.get("NVIDIA_API_KEY") if
                                                            self.base.startswith(NIM_URL) else None)
        self.timeout = timeout
        self.last_latency_ms: float | None = None

    def _req(self, method: str, path: str, body: dict | None = None) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method)
        req.add_header("Accept", "application/json")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                out = json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read()).get("error", {}).get("message", e.reason)
            except ValueError:
                msg = str(e.reason)
            raise ServiceError(e.code, msg) from None
        self.last_latency_ms = 1e3 * (time.perf_counter() - t0)
        return out

    def ready(self) -> bool:
        try:
            return self._req("GET", "/health/ready").get("status") == "ready"
        except (ServiceError, OSError):
            return False

    def models(self) -> list[dict[str, Any]]:
        return self._req("GET", "/models")["data"]

    def infer(self, model: str, inputs: dict[str, Any]) -> dict[str, Any]:
        return self._req("POST", "/infer", {"model": model, "inputs": inputs})

    def chat(self, model: str, messages: list[dict[str, str]], *, guided_json: dict | None = None,
             **kw) -> dict[str, Any]:
        """OpenAI-format chat completion; ``guided_json`` goes in ``nvext`` (NIM structured generation)."""
        body = {"model": model, "messages": messages, **kw}
        if guided_json is not None:
            body["nvext"] = {"guided_json": guided_json}
        return self._req("POST", "/chat/completions", body)

    @staticmethod
    def content_json(resp: dict[str, Any]) -> Any:
        return json.loads(resp["choices"][0]["message"]["content"])

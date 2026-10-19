"""The gateway's REST shape (FastAPI), shared by every app that serves models: ``/v1/models/...``.

==========================================================  ==========================================================
``GET  /v1/models``                                          published models: latest version, kind, method, trust
``GET  /v1/models/{name}``                                   the latest version's metadata, ModelRef fields, versions
``GET  /v1/models/{name}/versions/{version}``                one version (``latest``, a stage name or ``v...``)
``POST /v1/models/{name}/predict``                           ``{"inputs": ..., "version"?, "ood_policy"?}``
``POST /v1/models/{name}/versions/{version}/predict``        the same, for one version
``GET  /v1/models/{name}/runs?limit=``                       the run log of the model, newest first
``GET  /v1/metrics?model=``                                  calls, rows, errors, OOD rows, latency p50 / p95
``GET  /v1/models/{name}/versions/{version}/export/{fmt}``   ``torchscript`` or ``onnx`` (PyTorch surrogates, ONNX)
==========================================================  ==========================================================

``inputs`` is ``{input: value or list}``, a list of row objects, or a list of rows in the model's input order.
``on_rows(request, n)`` runs before a prediction (metering: raise to refuse); ``caller(request)`` names the caller
in the run log. (No ``from __future__ import annotations`` here: FastAPI resolves the local body model.)
Errors: unknown model or version 404, bad inputs 422, relaxing an OOD policy 403, no export 409.
"""
import os
import tempfile
from collections.abc import Callable
from typing import Any

from .gateway import ModelGateway

__all__ = ["gateway_router", "count_rows"]


def count_rows(inputs: Any) -> int:
    """Rows a request asks for, before the model is loaded."""
    if isinstance(inputs, dict):
        lens = [len(v) if isinstance(v, (list, tuple)) else 1 for v in inputs.values()]
        return max(lens) if lens else 0
    if isinstance(inputs, (list, tuple)):
        if inputs and not isinstance(inputs[0], (dict, list, tuple)):
            return 1
        return len(inputs)
    return 1


def gateway_router(gateway: ModelGateway | Callable[[], ModelGateway], *, prefix: str = "/v1",
                   on_rows: Callable[[Any, int], None] | None = None,
                   caller: Callable[[Any], str] | None = None, max_rows: int = 10000, tags=("models",)):
    from fastapi import APIRouter, HTTPException, Request
    from fastapi.responses import FileResponse
    from pydantic import BaseModel, Field
    from starlette.background import BackgroundTask

    get = gateway if callable(gateway) and not isinstance(gateway, ModelGateway) else (lambda: gateway)

    class PredictBody(BaseModel):
        inputs: Any = Field(description="{input: value or list}, a list of row objects, or rows in input order")
        version: str | None = Field(default=None, description="latest (default), a stage name or a version")
        ood_policy: str | None = Field(default=None, description="'refuse' to tighten a 'warn' model")

    r = APIRouter(prefix=prefix, tags=list(tags))

    def _err(e: Exception):
        if isinstance(e, KeyError):
            return HTTPException(404, str(e).strip("'\""))
        if isinstance(e, PermissionError):
            return HTTPException(403, str(e))
        if isinstance(e, NotImplementedError):
            return HTTPException(409, str(e))
        if isinstance(e, (ValueError, TypeError)):
            return HTTPException(422, str(e))
        return e

    def _summary(g: ModelGateway, name: str) -> dict[str, Any]:
        m = g.metadata(name)
        return {"name": name, "latest": m["version"], "versions": g.versions(name), "kind": m.get("kind"),
                "method": m.get("method"), "stage": m.get("stage"), "trust": m.get("trust"),
                "inputs": m.get("inputs"), "outputs": m.get("outputs"), "ood_policy": m.get("ood_policy"),
                "title": m.get("title", ""), "created_at": m.get("created_at"), "source": m.get("source", "")}

    @r.get("/models")
    def list_models():
        g = get()
        return {"models": [_summary(g, n) for n in g.models()]}

    @r.get("/models/{name}")
    def model(name: str):
        try:
            return get().describe(name)
        except Exception as e:
            raise _err(e) from None

    @r.get("/models/{name}/versions/{version}")
    def model_version(name: str, version: str):
        try:
            return get().describe(name, version)
        except Exception as e:
            raise _err(e) from None

    def _predict(request: Request, name: str, body: PredictBody, version: str | None):
        g = get()
        n = count_rows(body.inputs)
        if n > max_rows:
            raise HTTPException(422, f"{n} rows; the limit per call is {max_rows}")
        try:
            g.resolve(name, version or body.version)
        except Exception as e:
            raise _err(e) from None
        if on_rows is not None:
            on_rows(request, n)
        try:
            return g.predict(name, body.inputs, version=version or body.version, ood_policy=body.ood_policy,
                             caller=caller(request) if caller else "", max_rows=max_rows)
        except Exception as e:
            raise _err(e) from None

    @r.post("/models/{name}/predict")
    def predict(request: Request, name: str, body: PredictBody):
        return _predict(request, name, body, None)

    @r.post("/models/{name}/versions/{version}/predict")
    def predict_version(request: Request, name: str, version: str, body: PredictBody):
        return _predict(request, name, body, version)

    @r.get("/models/{name}/runs")
    def runs(name: str, limit: int = 100):
        return {"runs": get().runs(name, limit=min(limit, 1000))}

    @r.get("/metrics")
    def metrics(model: str | None = None):
        return get().metrics(model)

    @r.get("/models/{name}/versions/{version}/export/{fmt}")
    def export(name: str, version: str, fmt: str):
        if fmt not in ("torchscript", "onnx"):
            raise HTTPException(422, "fmt must be torchscript or onnx")
        fd, path = tempfile.mkstemp(suffix=".onnx" if fmt == "onnx" else ".pt")
        os.close(fd)
        try:
            v = get().resolve(name, version)
            get().export(name, v, fmt, path)
        except Exception as e:
            os.unlink(path)
            raise _err(e) from None
        fname = f"{name}-{v}.{'onnx' if fmt == 'onnx' else 'pt'}"
        return FileResponse(path, filename=fname, background=BackgroundTask(os.unlink, path))

    return r

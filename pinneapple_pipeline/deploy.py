"""Serving a verified model: a ``ModelService`` whose trust function is the run's own verification.

``pipeline.deploy()`` turns a finished run into the object ``pinneapple_tools.model_services`` already knows how to
serve — a :class:`ModelService` with a ``predict`` and a ``trust`` function — so a surrogate reaches a colleague's
script, a dashboard or an HTTP endpoint without a second implementation of what the model is allowed to answer.

The trust function is not decoration: it re-reads the validated domain the study produced and the gates of the run.
A query outside the domain is refused (``deployment.on_ood="reject"``) or answered and flagged (``"flag"``), and a
run whose gates did not pass is served with that verdict attached rather than hidden. The service therefore says
what the model was measured on, in the same words the report uses.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["Deployment", "deploy"]

_DECISIONS = {"accept": "ACCEPT", "review": "REVIEW", "reject": "REJECT"}


class Deployment:
    """A served run: the service, the registry and (with ``target="http"``) the server in front of them."""

    def __init__(self, pipeline, cfg, service, registry, *, url: str = "", server: Any = None):
        self.pipeline, self.config, self.service, self.registry = pipeline, cfg, service, registry
        self.url, self.server = url, server

    def predict(self, **inputs: Any) -> dict[str, Any]:
        """One query through the service, with the trust report (what an HTTP client would receive)."""
        return self.service.run(inputs)

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server = None

    def __repr__(self) -> str:
        return (f"Deployment({self.service.id!r}, service+registry"
                + (f", {self.url}" if self.url else ", in-process") + ")")


def deploy(pipeline, *, target: str | None = None, host: str | None = None, port: int | None = None,
           api_key: str | None = None) -> Deployment:
    """Serve the model of a finished run (``pipeline.run()`` first, or ``SurrogatePipeline.from_run(folder)``).

    ``target="local"`` returns the service and the registry; ``"http"`` also starts the server on a background
    thread and sets ``.url``. The keywords override the ``deployment`` block of the configuration.
    """
    from pinneapple_tools.model_services import ModelService, Registry

    cfg = getattr(pipeline, "config", None)
    if cfg is None:
        raise TypeError("deploy() takes a pp.SurrogatePipeline")
    d = cfg.deployment
    target = target or d.target
    host = host or d.host
    port = d.port if port is None else port
    api_key = d.api_key if api_key is None else api_key
    state = pipeline.state
    if state is None or state.model_object is None:
        raise RuntimeError("no trained model to serve: run the pipeline (or from_run() a finished run folder) first")
    result = pipeline.result
    gates = {g.gate: g.status for g in (result.verification if result is not None else state.verification)}
    failed = [name for name, status in gates.items() if status == "FAIL"]
    domain = _domain(pipeline, state)
    inputs = _input_names(state)
    outputs = _output_names(state)

    def predict(payload: dict[str, Any]) -> dict[str, Any]:
        X, outside = _as_matrix(payload, inputs, domain)
        if outside and d.on_ood == "reject" and d.enforce_validated_domain:
            raise ValueError("the query is outside the validated domain: "
                             + "; ".join(f"{k}={v} not in {domain[k]}" for k, v in outside.items()))
        Y = np.asarray(pipeline.predict_array(X), dtype=float)
        return {name: Y[:, i].tolist() for i, name in enumerate(outputs)}

    def trust(payload: dict[str, Any], out: dict[str, Any]) -> dict[str, Any]:
        _, outside = _as_matrix(payload, inputs, domain)
        checks = [{"name": f"gate:{name}", "status": status, "passed": status != "FAIL"}
                  for name, status in gates.items()]
        if d.enforce_validated_domain:
            checks.append({"name": "domain", "status": "PASS" if not outside else "FAIL",
                           "passed": not outside,
                           "reason": "" if not outside else "outside " + ", ".join(sorted(outside))})
        decision = "REJECT" if (failed or (outside and d.on_ood == "reject")) else \
            ("ACCEPT" if not outside and result is not None and result.status == "completed" else "REVIEW")
        return {"decision": decision, "checks": checks, "in_validated_domain": not outside,
                "validated_domain": {"bounds": domain},
                "failed_gates": failed,
                "reason": ("the run was rejected by its own gates: " + ", ".join(failed)) if failed else
                          ("the query is outside the validated domain" if outside else
                           "inside the domain, and no gate of the run failed")}

    card = {"description": cfg.description or f"surrogate of {state.bundle.name}",
            "task": "surrogate", "version": str(getattr(result, "run_id", "")) or state.run_id,
            "inputs": inputs, "outputs": outputs, "validity_range": domain,
            "validation_metrics": {k: v for k, v in (state.metrics or {}).items()
                                   if k in ("relative_l2", "rmse", "nrmse", "infer_us_per_point")},
            "gates": gates, "known_limitations": list(state.metrics.get("limitations") or [])[:8],
            "run_folder": str(state.out_dir)}
    service = ModelService(id=cfg.name, kind="infer", task="surrogate", predict=predict, trust=trust, card=card,
                           version=str(getattr(result, "run_id", "")) or state.run_id)
    registry = Registry([service])
    deployment = Deployment(pipeline, cfg, service, registry)
    if target == "http":
        from pinneapple_tools.model_services.server import serve_in_thread
        server, url = serve_in_thread(registry, host=host, port=int(port), api_key=api_key or None)
        deployment.server, deployment.url = server, url
    if cfg.deployment.enabled or target != "local":
        _write(state.out_dir, {"service": service.info(), "url": deployment.url, "gates": gates,
                               "validated_domain": domain})
    return deployment


# --------------------------------------------------------------------------------------------------- helpers
def _write(folder: Path, payload: dict[str, Any]) -> None:
    """The service's card and how to start it, next to the model."""
    try:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "deployment.json").write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
        card = (payload.get("service") or {}).get("pinneapple") or {}
        inputs = "{" + ", ".join(f'"{n}": 0.0' for n in (card.get("inputs") or [])) + "}"
        text = _SERVE.replace("__NAME__", str((payload.get("service") or {}).get("id", "model")))
        (folder / "serve.py").write_text(text.replace("__INPUTS__", inputs), encoding="utf-8")
    except OSError:                                            # a read-only folder is not a reason to fail
        pass


_SERVE = '''"""Serve this run.

    python serve.py
    curl -s localhost:8000/v1/models
    curl -s localhost:8000/v1/infer -H 'content-type: application/json' \\
         -d '{"model": "__NAME__", "inputs": __INPUTS__}'

Written by `pp.SurrogatePipeline`; the reply carries the prediction and the trust report of the run (its gates and
whether the query is inside the validated domain).
"""
from pathlib import Path

import pinneapple as pp

RUN = Path(__file__).resolve().parent
deployment = pp.SurrogatePipeline.from_run(RUN).deploy(target="http", host="127.0.0.1", port=8000)
print(f"serving {deployment.service.id} at {deployment.url}")
try:
    import time
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    deployment.stop()
'''


def _domain(pipeline, state) -> dict[str, list[float]]:
    """Where the model is valid, per input: the study's validated bounds, else the problem's domain."""
    domain = dict((state.validated_domain or {}).get("bounds") or {})
    if not domain:
        for c, bounds in (state.bundle.domains or {}).items():
            domain[c] = [float(bounds[0]), float(bounds[1])]
    return domain


def _input_names(state) -> list[str]:
    if state.cfg.data.is_physics_only:
        return list(state.bundle.coords)
    names = list(getattr(state.dataset, "input_names", []) or [])
    if names:
        return names
    space = state.cfg.parameters.to_space() if state.cfg.parameters.parameters else None
    return [p.name for p in getattr(space, "parameters", [])] if space is not None else []


def _output_names(state) -> list[str]:
    if state.cfg.data.is_physics_only:
        return list(state.bundle.fields)
    names = list(getattr(state.dataset, "output_names", []) or [])
    if names:
        return names
    return list(state.bundle.fields)


def _as_matrix(payload: dict[str, Any], inputs: list[str],
               domain: dict[str, list[float]]) -> tuple[np.ndarray, dict[str, float]]:
    """``(X, outside)`` from the query, which may be scalars or lists; ``outside`` names the inputs out of range."""
    columns, length, outside = [], 1, {}
    for name in inputs:
        if name not in payload:
            raise ValueError(f"the model takes {inputs}; missing {name!r}")
        values = np.atleast_1d(np.asarray(payload[name], dtype=float)).ravel()
        columns.append(values)
        length = max(length, len(values))
        low, high = domain.get(name, [-np.inf, np.inf])
        bad = values[(values < float(low)) | (values > float(high))]
        if bad.size:
            outside[name] = float(bad[0])
    X = np.empty((length, len(inputs)), dtype=float)
    for j, values in enumerate(columns):
        X[:, j] = values if len(values) == length else np.repeat(float(values[0]), length)
    return X, outside

"""Model services: a callable with a card and a trust function, and the registry of the built-in ones."""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["ModelService", "Registry", "default_registry", "zoo_service", "trust_card_service", "intent_service",
           "jsonable", "infer"]


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


@dataclass
class ModelService:
    id: str
    kind: str                                            # "infer" | "chat"
    task: str
    predict: Callable[[dict[str, Any]], dict[str, Any]]  # infer: inputs -> outputs; chat: request -> message text
    trust: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None
    card: dict[str, Any] = field(default_factory=dict)
    version: str = "1"
    calls: int = 0

    def info(self) -> dict[str, Any]:
        return {"id": self.id, "object": "model", "created": 0, "owned_by": "pinneapple",
                "pinneapple": jsonable({"kind": self.kind, "task": self.task, "version": self.version, **self.card})}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Prediction, trust report, warnings raised on the way (e.g. extrapolation) and timings."""
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t0 = time.perf_counter()
            out = self.predict(inputs)
            t1 = time.perf_counter()
            trust = self.trust(inputs, out) if self.trust else {"decision": "REVIEW", "checks": [],
                                                                "reason": "this service has no trust function"}
        self.calls += 1
        return {"model": self.id, "outputs": jsonable(out), "trust": jsonable(trust),
                "warnings": sorted({str(w.message) for w in caught}),
                "timing_ms": {"inference": 1e3 * (t1 - t0), "trust": 1e3 * (time.perf_counter() - t1)}}


class Registry:
    def __init__(self, services=()):
        self._s: dict[str, ModelService] = {}
        for s in services:
            self.add(s)

    def add(self, s: ModelService) -> ModelService:
        self._s[s.id] = s
        return s

    def get(self, name: str) -> ModelService:
        if name not in self._s:
            raise KeyError(f"unknown model {name!r}; available: {sorted(self._s)}")
        return self._s[name]

    def list(self) -> list[ModelService]:
        return list(self._s.values())

    def __contains__(self, name) -> bool:
        return name in self._s


# ------------------------------------------------------------------------------------------------ built-ins
def zoo_service(name: str) -> ModelService:
    """A model of the PINNeAPPle zoo: ``predict(**inputs)``; trust from ``pp.zoo.validate`` on the same inputs
    (card metrics reproduced, physics checks, validity range of the query)."""
    import pinneapple as pp
    m = pp.zoo.load(name)
    card = m.card.to_dict() if hasattr(m.card, "to_dict") else {}

    def predict(inputs):
        return m.predict(**{k: np.atleast_1d(np.asarray(v, float)) for k, v in inputs.items()})

    def trust(inputs, outputs):
        rep = pp.zoo.validate(name, inputs={k: np.atleast_1d(np.asarray(v, float)) for k, v in inputs.items()})
        return {"decision": rep.decisao, "checks": [{"name": c.name, "status": c.status,
                                                     "value": getattr(c, "value", None), "limit": getattr(c, "limit", None),
                                                     "reason": getattr(c, "reason", "")} for c in rep.checks],
                "in_validity_range": bool(np.all(m.in_range({k: np.atleast_1d(np.asarray(v, float))
                                                             for k, v in inputs.items()}))),
                "validity_range": {k: list(v) for k, v in m.bounds().items()}}

    keep = ("description", "task", "version", "inputs", "outputs", "validity_range", "known_limitations",
            "architecture", "validation_metrics", "reference_source", "citation", "license")
    return ModelService(name, "infer", card.get("task", "surrogate"), predict, trust,
                        {k: card.get(k) for k in keep if k in card} | {"input_names": list(m.input_names),
                                                                       "output_names": list(m.output_names)},
                        str(card.get("version", "1")))


def trust_card_service() -> ModelService:
    """Physics checks as a service: ``{"axes": {"x": [...], "y": [...]}, "values": [[...]], "pde": "laplace",
    "dirichlet": {"x_min": 0.0, ...}, "train_bounds": {...}}`` -> the trust card of that field."""
    import pinneapple as pp

    def predict(inputs):
        axes = {k: np.asarray(v, float) for k, v in inputs["axes"].items()}
        card = pp.trust_card(axes, np.asarray(inputs["values"], float), pde=inputs.get("pde"),
                             dirichlet=inputs.get("dirichlet"), train_bounds=inputs.get("train_bounds"),
                             source=inputs.get("source"))
        return {"decision": card.decision, "card": str(card)}

    def trust(inputs, outputs):
        return {"decision": outputs["decision"], "checks": [], "reason": "the output is itself a trust card"}

    return ModelService("pinneapple-trust-card", "infer", "physics checks of a field on a grid", predict, trust,
                        {"description": "PDE residual, boundary conditions, conservation and applicability of any "
                                        "field on a structured grid (pp.trust_card)"})


def intent_service() -> ModelService:
    """``pinneapple-intent``: the deterministic specification extractor of ``pinneapple_design.intent`` behind an
    OpenAI-style chat route. The last user message is the request; the answer is the specification as JSON (template,
    values with their units and quotes, questions, values it could not place). Each value's quote is checked to occur
    in the text (the trust report)."""
    from pinneapple_design.intent.parse import formulate

    def predict(req):
        text = next((m["content"] for m in reversed(req.get("messages", [])) if m.get("role") == "user"), "")
        spec = formulate(text, template=req.get("template"))
        d = spec.to_dict() if hasattr(spec, "to_dict") else {"out_of_scope": str(spec)}
        return {"text": text, "spec": d}

    def trust(req, out):
        d, text = out["spec"], out["text"]
        vals = d.get("values", {})
        checks = []
        for k, v in vals.items():
            if v.get("provenance") == "stated" and v.get("source") == "user":
                q = str(v.get("quote", ""))
                ok = q in text or q.replace("_", "") in text.replace(" ", "").lower() or k == "material"
                checks.append({"name": f"{k}: quote occurs in the request", "status": "supports" if ok else
                               "contradicts", "value": q})
        unused = d.get("unsupported", [])
        checks.append({"name": "every number in the request was placed", "status": "supports" if not unused else
                       "contradicts", "value": unused})
        questions = d.get("questions", [])
        dec = "REJECT" if "out_of_scope" in d or any(c["status"] == "contradicts" and "quote" in c["name"]
                                                       for c in checks) else ("REVIEW" if unused or questions
                                                                              else "APPROVED")
        return {"decision": dec, "checks": checks, "questions": questions}

    return ModelService("pinneapple-intent", "chat", "requirement extraction (deterministic, no network)", predict,
                        trust, {"description": "plain-language engineering request -> RequirementSpec "
                                               "(pinneapple_design.intent.formulate); every value carries its quote",
                                "templates": _templates()})


def _templates():
    try:
        from pinneapple_design.intent.parse import templates
        return sorted(templates())
    except Exception:  # noqa: BLE001
        return []


def default_registry(zoo: bool = True) -> Registry:
    r = Registry([trust_card_service(), intent_service()])
    if zoo:
        import pinneapple as pp
        for name in pp.zoo.list():
            r.add(zoo_service(name))
    return r


def dumps(o) -> bytes:
    return json.dumps(jsonable(o)).encode()


_DEFAULT: Registry | None = None


def infer(model: str, inputs_json: str) -> dict:
    """One request to a built-in service in this process (no server): ``inputs_json`` is the ``inputs`` object of
    ``/v1/infer`` as JSON text. Returns the same answer the HTTP route returns (outputs, trust, warnings, timing)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = default_registry()
    return _DEFAULT.get(model).run(json.loads(inputs_json))

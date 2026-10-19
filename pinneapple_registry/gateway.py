"""``ModelGateway``: publish physical models as versioned endpoints, run them, and record every run.

Built on ``ModelStore`` (versions ``vYYYYMMDD_HHMMSS``, ``latest``, stages, the REJECT-blocks-publication rule).
A published version is a directory with ``metadata.json`` (ModelStore's) and the model files; three kinds of
model are served through one ``predict``:

``pinneapple.surrogate``  a ``pinneapple_analysis.surrogate.SurrogateModel`` (Surrogate Studio, the arena), with its
                          OOD gate (training range + nearest-neighbour distance);
``pinneapple.zoo``        a pre-trained zoo model (``pp.zoo``), with the validity box of its model card;
``onnx``                  an ONNX graph (``inputs (n, d) -> outputs (n, p)``) with declared input / output names and
                          validity bounds, run with onnxruntime;
``remote``                a model served by another app with this API (``https://host/v1/models/<name>``, e.g. the
                          Surrogate Studio): this gateway enforces its validity bounds and OOD policy, logs and meters
                          the call, and forwards the rows it accepts (``GatewayClient``).

``publish_ref`` publishes from a ``ModelRef`` (``pinneapple_data.contracts``): ``zoo:<name>`` or an http(s) gateway
URI; its input ranges become the validity bounds. Publishing a remote model makes this server call that URL: let only
trusted callers publish.

Each version has an ``ood_policy``: ``"warn"`` (predict and flag), or ``"refuse"`` (no prediction for an
out-of-distribution row). A caller may ask for ``"refuse"`` on a ``"warn"`` model, never the reverse.
Every call is appended to ``<root>/_runs.jsonl`` (time, model, version, rows, latency, OOD rows, refusals, error,
caller); ``metrics`` aggregates that log (counts, errors, latency percentiles, OOD rate per version).

Only models published through the gateway are loaded: a surrogate is unpickled after its sha256 is checked against
its card, so never publish a directory you did not produce.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .model_store import ModelStore, _check_publishable

__all__ = ["ModelGateway", "OOD_POLICIES", "KINDS", "ood_verdict"]

OOD_POLICIES = ("warn", "refuse")
KINDS = ("pinneapple.surrogate", "pinneapple.zoo", "onnx", "remote")
_POLICY_ORDER = {"warn": 0, "refuse": 1}


def ood_verdict(flag: dict[str, Any], policy: str = "warn") -> str:
    """What ``predict`` does with one assessed row under ``policy``: ``"in range"`` (predicted), ``"warn"``
    (predicted and flagged as an extrapolation) or ``"refuse"`` (no prediction)."""
    if policy not in OOD_POLICIES:
        raise ValueError(f"ood_policy must be one of {OOD_POLICIES}")
    if not flag.get("ood"):
        return "in range"
    return "refuse" if policy == "refuse" else "warn"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class _Predictor:
    """Loaded model: ``run(X (n, d)) -> {output: (n, ...)}``, ``assess(X) -> per-row OOD dicts``."""

    def __init__(self, inputs, outputs, run, assess, export=None):
        self.inputs, self.outputs, self.run, self.assess, self.export = list(inputs), list(outputs), run, assess, export


def _box_assess(names: Sequence[str], bounds: Mapping[str, Sequence[float]]):
    lo = np.array([float(bounds[n][0]) for n in names])
    hi = np.array([float(bounds[n][1]) for n in names])

    def assess(X):
        rows = []
        for r in np.atleast_2d(X):
            out = [n for n, v, a, b in zip(names, r, lo, hi, strict=True) if not np.isfinite(v) or v < a or v > b]
            rows.append({"ood": bool(out), "outside": out, "reasons": [
                f"{n} = {v:.4g} outside the validity range {a:.4g} .. {b:.4g}"
                for n, v, a, b in zip(names, r, lo, hi, strict=True) if n in out]})
        return rows
    return assess


def _split_uri(uri: str) -> tuple[str, str, str | None]:
    """``https://host[/prefix]/v1/models/<name>[/versions/<v>]`` -> (base, name, version)."""
    from urllib.parse import unquote
    if "/v1/models/" not in uri:
        raise ValueError(f"{uri!r} is not a gateway model URI (…/v1/models/<name>)")
    base, rest = uri.split("/v1/models/", 1)
    parts = [unquote(x) for x in rest.strip("/").split("/") if x]
    if not parts:
        raise ValueError(f"{uri!r} names no model")
    version = parts[2] if len(parts) >= 3 and parts[1] == "versions" else None
    return base, parts[0], version


def _zoo_assess(zm, names: Sequence[str], bounds: Mapping[str, Sequence[float]]):
    """The zoo model's own applicability check (``in_range``: its card can state the range on derived quantities,
    e.g. a fin gap computed from several inputs), with the per-input reasons where an input has its own bounds."""
    named = [n for n in names if n in bounds]
    box = _box_assess(named, bounds)

    def assess(X):
        X = np.atleast_2d(X)
        ok = np.asarray(zm.in_range({n: X[:, j] for j, n in enumerate(names)}), bool).reshape(len(X))
        rows = box(X[:, [names.index(n) for n in named]]) if named else [{"outside": [], "reasons": []}] * len(X)
        out = []
        for good, r in zip(ok, rows, strict=True):
            reasons = list(r["reasons"])
            if not good and not reasons:
                reasons = ["outside the validity range stated in the model card (on derived quantities)"]
            out.append({"ood": bool(not good or r["outside"]), "outside": list(r["outside"]), "reasons": reasons})
        return out
    return assess


class ModelGateway(ModelStore):
    def __init__(self, root: str, *, cache_size: int = 8):
        super().__init__(root)
        self._cache: OrderedDict[tuple[str, str], _Predictor] = OrderedDict()
        self._cache_size = int(cache_size)
        self._lock = threading.Lock()
        self.log_path = os.path.join(root, "_runs.jsonl")

    # ------------------------------------------------------------------------------------------ publish
    def models(self) -> list[str]:
        return sorted(d for d in os.listdir(self.root) if not d.startswith("_") and os.path.isdir(
            os.path.join(self.root, d)) and self.latest(d) is not None)

    def _new_version(self, name: str) -> str:
        v = f"v{datetime.now():%Y%m%d_%H%M%S}"
        k = 1
        while os.path.exists(self._version_dir(name, v if k == 1 else f"{v}_{k}")):
            k += 1
        return v if k == 1 else f"{v}_{k}"

    def publish(self, name: str, model: Any = None, *, kind: str = "pinneapple.surrogate", ood_policy: str = "warn",
                stage: str = "production", description: str = "", source: str = "", override_reason: str | None = None,
                spec: Mapping[str, Any] | None = None) -> str:
        """Publish a new version of ``name`` and make it ``latest``; returns the version.

        ``model``: a ``SurrogateModel`` or the directory it was saved to (kind ``pinneapple.surrogate``); a zoo model
        name (``pinneapple.zoo``); the path of an ``.onnx`` file (``onnx``, with ``spec`` = ``{"inputs": [...],
        "outputs": [...], "validity_bounds": {input: [lo, hi]}}``). A version whose trust decision is REJECT is not
        published to staging or production without ``override_reason``."""
        if not name or not all(c.isalnum() or c in "-_." for c in name) or name.startswith(("_", ".")):
            raise ValueError("model names use letters, digits, '-', '_' and '.', and do not start with '_' or '.'")
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if ood_policy not in OOD_POLICIES:
            raise ValueError(f"ood_policy must be one of {OOD_POLICIES}")
        version = self._new_version(name)
        vdir = self._version_dir(name, version)
        os.makedirs(vdir)
        try:
            meta = self._materialise(kind, model, vdir, spec)
        except Exception:
            shutil.rmtree(vdir, ignore_errors=True)
            raise
        meta.update(name=name, problem_id=name, version=version, kind=kind, stage=stage, ood_policy=ood_policy,
                    description=description, source=source, created_at=_now())
        try:
            _check_publishable(meta, name, stage, override_reason)
        except PermissionError:
            shutil.rmtree(vdir, ignore_errors=True)
            raise
        self._write_metadata(name, version, meta)
        self._set_latest(name, version)
        return version

    def _materialise(self, kind: str, model: Any, vdir: str, spec) -> dict[str, Any]:
        if kind == "pinneapple.surrogate":
            from pinneapple_analysis.surrogate import SurrogateModel
            sm = model if isinstance(model, SurrogateModel) else SurrogateModel.load(model)
            sm.save(os.path.join(vdir, "model"))
            card = sm.card
            trust = card.get("trust") or {}
            return {"inputs": sm.input_names, "outputs": sm.output_names, "grid_shape": list(sm.grid_shape),
                    "validity_bounds": card.get("validity_bounds"), "method": card.get("method"),
                    "metrics": card.get("metrics", {}), "dataset_sha256": card.get("dataset_sha256"),
                    "model_sha256": card.get("model_sha256"), "trust": trust.get("decision"),
                    "trust_report": {"decisao": trust.get("decision"),
                                     "motivos": [c["name"] for c in trust.get("checks", [])
                                                 if c.get("status") == "contradicts"]} if trust else None,
                    "limitations": card.get("limitations"), "title": card.get("title", "")}
        if kind == "pinneapple.zoo":
            import pinneapple as pp
            m = pp.zoo.load(str(model))
            card = m.card
            return {"zoo_name": str(model), "inputs": list(m.input_names), "outputs": list(m.output_names),
                    "grid_shape": [], "validity_bounds": {k: list(v) for k, v in m.bounds().items()},
                    "method": card.architecture, "metrics": card.validation_metrics or {},
                    "trust": card.trust_decision if hasattr(card, "trust_decision") else None,
                    "known_limitations": card.known_limitations, "title": card.description[:200]}
        if kind == "onnx":
            spec = dict(spec or {})
            for k in ("inputs", "outputs", "validity_bounds"):
                if not spec.get(k):
                    raise ValueError(f"an ONNX model needs spec[{k!r}]")
            missing = [n for n in spec["inputs"] if n not in spec["validity_bounds"]]
            if missing:
                raise ValueError(f"validity_bounds missing for inputs {missing}")
            shutil.copyfile(str(model), os.path.join(vdir, "model.onnx"))
            import onnxruntime as ort
            ort.InferenceSession(os.path.join(vdir, "model.onnx"), providers=["CPUExecutionProvider"])
            return {"inputs": list(spec["inputs"]), "outputs": list(spec["outputs"]), "grid_shape": [],
                    "validity_bounds": {k: list(map(float, v)) for k, v in spec["validity_bounds"].items()},
                    "method": spec.get("method", "onnx"), "metrics": spec.get("metrics", {}), "trust": None}
        if kind == "remote":
            from .gateway_client import GatewayClient
            spec = dict(spec or {})
            base, rname, rversion = _split_uri(str(model))
            d = GatewayClient(base, timeout=float(spec.get("timeout", 30))).describe(rname, rversion)
            bounds = spec.get("validity_bounds") or d.get("validity_bounds") or {}
            missing = [n for n in d["inputs"] if n not in bounds]
            if missing:
                raise ValueError(f"no validity bounds for inputs {missing}: give them in spec['validity_bounds']")
            return {"remote": {"base": base, "model": rname, "version": rversion or d.get("version"),
                               "api_key_env": spec.get("api_key_env")}, "inputs": list(d["inputs"]),
                    "outputs": list(d["outputs"]), "grid_shape": d.get("grid_shape") or [],
                    "validity_bounds": {k: list(map(float, v)) for k, v in bounds.items()},
                    "method": d.get("method"), "metrics": d.get("metrics", {}), "trust": d.get("trust"),
                    "title": d.get("title", ""), "remote_ood_policy": d.get("ood_policy")}
        raise ValueError(kind)

    def publish_ref(self, ref: Any, *, name: str | None = None, ood_policy: str = "warn",
                    stage: str = "production", override_reason: str | None = None) -> str:
        """Publish from a ``ModelRef`` (object or dict): ``zoo:<name>`` -> a zoo model, an http(s) gateway URI ->
        a remote model whose validity bounds are the ref's input ranges. File URIs are refused here (a pickled
        surrogate must come from a directory you produced: use ``publish``)."""
        r = ref.to_dict() if hasattr(ref, "to_dict") else dict(ref)
        uri = str(r.get("uri", ""))
        name = name or r["name"]
        src = f"ModelRef {r.get('name')} {r.get('version', '')}".strip()
        trust = r.get("trust_decision")
        override = override_reason
        if uri.startswith("zoo:"):
            return self.publish(name, uri[4:], kind="pinneapple.zoo", ood_policy=ood_policy, stage=stage, source=src,
                                override_reason=override)
        if uri.startswith(("http://", "https://")):
            bounds = {v["name"]: [v["low"], v["high"]] for v in r.get("inputs", [])
                      if v.get("low") is not None and v.get("high") is not None}
            v = self.publish(name, uri, kind="remote", ood_policy=ood_policy, stage="development", source=src,
                             spec={"validity_bounds": bounds or None})
            meta = self.metadata(name, v)
            if trust in ("APPROVED", "REVIEW", "REJECT"):
                meta["trust"] = trust
                meta["trust_report"] = {"decisao": trust, "motivos": [f"trust decision of {src}"]}
            meta["model_ref"] = {k: r.get(k) for k in ("name", "version", "uri", "sha256", "metrics", "metrics_split")}
            self._write_metadata(name, v, meta)
            if stage != "development":
                self.promote(name, v, stage, override_reason=override)
            return v
        raise ValueError(f"ModelRef uri {uri!r}: publish_ref takes zoo:<name> or an http(s) gateway URI")

    # ------------------------------------------------------------------------------------------ resolve / load
    def resolve(self, name: str, version: str | None = None) -> str:
        """``None`` / ``"latest"``: the latest version; a stage name: its newest version; else the version itself."""
        if name not in self.models():
            raise KeyError(f"no model named {name!r}")
        if version in (None, "", "latest"):
            return self.latest(name)
        if version in ("development", "staging", "production", "archived"):
            vs = self.by_stage(name, version)
            if not vs:
                raise KeyError(f"{name} has no version in stage {version!r}")
            return vs[-1]
        if version not in self.versions(name):
            raise KeyError(f"{name} has no version {version!r}; versions: {self.versions(name)}")
        return version

    def _load(self, name: str, version: str) -> _Predictor:
        key = (name, version)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        meta = self.metadata(name, version)
        vdir = self._version_dir(name, version)
        kind = meta["kind"]
        if kind == "pinneapple.surrogate":
            from pinneapple_analysis.surrogate import SurrogateModel
            sm = SurrogateModel.load(os.path.join(vdir, "model"))

            def run(X, sm=sm):
                Y = sm.predict_array(X)
                if sm.grid_shape:
                    return {sm.output_names[0]: Y}
                return {n: Y[:, j] for j, n in enumerate(sm.output_names)}
            p = _Predictor(sm.input_names, sm.output_names, run, sm.gate.assess, sm.export)
        elif kind == "pinneapple.zoo":
            import pinneapple as pp
            zm = pp.zoo.load(meta["zoo_name"])
            names = list(zm.input_names)

            def run(X, zm=zm, names=names):
                out = zm.predict({n: X[:, j] for j, n in enumerate(names)}, warn=False)
                return {k: np.asarray(v) for k, v in out.items()}
            p = _Predictor(names, list(zm.output_names), run, _zoo_assess(zm, names, meta["validity_bounds"]))
        elif kind == "remote":
            from .gateway_client import GatewayClient
            rm = meta["remote"]
            key = os.environ.get(rm["api_key_env"]) if rm.get("api_key_env") else None
            client = GatewayClient(rm["base"], api_key=key)

            def run(X, client=client, rm=rm, ins=meta["inputs"]):
                rows = [dict(zip(ins, map(float, r), strict=True)) for r in X]
                out = client.predict(rm["model"], rows, version=rm.get("version"))["outputs"]
                return {k: np.asarray([np.nan if v is None else v for v in vals], dtype=float)
                        for k, vals in out.items()}
            p = _Predictor(meta["inputs"], meta["outputs"], run, _box_assess(meta["inputs"], meta["validity_bounds"]))
        else:
            import onnxruntime as ort
            sess = ort.InferenceSession(os.path.join(vdir, "model.onnx"), providers=["CPUExecutionProvider"])
            iname = sess.get_inputs()[0].name

            def run(X, sess=sess, iname=iname, outs=meta["outputs"]):
                Y = np.asarray(sess.run(None, {iname: np.asarray(X, np.float32)})[0], float).reshape(len(X), -1)
                return {n: Y[:, j] for j, n in enumerate(outs)}
            p = _Predictor(meta["inputs"], meta["outputs"], run, _box_assess(meta["inputs"], meta["validity_bounds"]))
        with self._lock:
            self._cache[key] = p
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return p

    # ------------------------------------------------------------------------------------------ predict
    @staticmethod
    def _rows(inputs, names: Sequence[str]) -> np.ndarray:
        if isinstance(inputs, Mapping):
            missing = [k for k in names if k not in inputs]
            if missing:
                raise ValueError(f"missing inputs: {missing} (the model needs {list(names)})")
            cols = np.broadcast_arrays(*[np.atleast_1d(np.asarray(inputs[k], float)) for k in names])
            return np.stack([c.ravel() for c in cols], 1)
        if isinstance(inputs, Sequence) and inputs and isinstance(inputs[0], Mapping):
            return np.array([[float(r[k]) for k in names] for r in inputs])
        X = np.atleast_2d(np.asarray(inputs, float))
        if X.shape[1] != len(names):
            raise ValueError(f"inputs have {X.shape[1]} columns; the model needs {len(names)} ({list(names)})")
        return X

    def predict(self, name: str, inputs, *, version: str | None = None, ood_policy: str | None = None,
                caller: str = "", max_rows: int = 10000) -> dict[str, Any]:
        """Run a version. ``inputs``: ``{input: value or list}``, a list of row dicts, or an ``(n, d)`` array."""
        t0 = time.perf_counter()
        rid = uuid.uuid4().hex[:16]
        rec: dict[str, Any] = {"request_id": rid, "time": _now(), "model": name, "version": None, "caller": caller,
                               "rows": 0, "ood_rows": 0, "refused": 0, "error": None}
        try:
            v = self.resolve(name, version)
            rec["version"] = v
            meta = self.metadata(name, v)
            policy = meta.get("ood_policy", "warn")
            if ood_policy is not None:
                if ood_policy not in OOD_POLICIES:
                    raise ValueError(f"ood_policy must be one of {OOD_POLICIES}")
                if _POLICY_ORDER[ood_policy] < _POLICY_ORDER[policy]:
                    raise PermissionError(f"{name} {v} is published with ood_policy={policy!r}; a caller cannot "
                                          "relax it")
                policy = ood_policy
            p = self._load(name, v)
            X = self._rows(inputs, p.inputs)
            if len(X) > max_rows:
                raise ValueError(f"{len(X)} rows; the limit per call is {max_rows}")
            rec["rows"] = int(len(X))
            flags = p.assess(X)
            ood = np.array([f["ood"] for f in flags], bool)
            ok = np.array([ood_verdict(f, policy) != "refuse" for f in flags], bool)
            outs: dict[str, list] = {}
            if ok.any():
                res = p.run(X[ok])
                for k, arr in res.items():
                    full: list = [None] * len(X)
                    for i, val in zip(np.flatnonzero(ok), np.asarray(arr), strict=True):
                        full[i] = val.tolist() if np.ndim(val) else float(val)
                    outs[k] = full
            else:
                outs = {k: [None] * len(X) for k in p.outputs}
            rec.update(ood_rows=int(ood.sum()), refused=int((~ok).sum()))
            warnings = []
            if ood.any():
                warnings.append(f"{int(ood.sum())} of {len(X)} rows are outside the model's validity range" +
                                (" and were refused" if policy == "refuse" else "; their predictions are "
                                                                                   "extrapolations"))
            out = {"request_id": rid, "model": {"name": name, "version": v, "kind": meta["kind"],
                                                "method": meta.get("method"), "stage": meta.get("stage")},
                   "outputs": outs, "ood": flags, "ood_policy": policy, "warnings": warnings, "n": int(len(X))}
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["latency_ms"] = (time.perf_counter() - t0) * 1e3
            self._log(rec)
            raise
        rec["latency_ms"] = (time.perf_counter() - t0) * 1e3
        out["latency_ms"] = rec["latency_ms"]
        self._log(rec)
        return out

    # ------------------------------------------------------------------------------------------ logs / metrics
    def _log(self, rec: dict[str, Any]) -> None:
        line = json.dumps(rec, default=str)
        with self._lock, open(self.log_path, "a") as f:
            f.write(line + "\n")

    def runs(self, name: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path) as f:
            rows = [json.loads(x) for x in f if x.strip()]
        if name:
            rows = [r for r in rows if r.get("model") == name]
        return rows[-limit:][::-1]

    def metrics(self, name: str | None = None) -> dict[str, Any]:
        """Per model and version, from the run log: calls, rows, errors, OOD rows, refusals, latency p50 / p95."""
        rows = self.runs(name, limit=10 ** 9)
        groups: dict[tuple[str, str], list[dict]] = {}
        for r in rows:
            groups.setdefault((r.get("model"), r.get("version") or "-"), []).append(r)
        out = []
        for (m, v), rs in sorted(groups.items()):
            lat = np.array([r["latency_ms"] for r in rs if r.get("error") is None and r.get("latency_ms") is not None])
            out.append({"model": m, "version": v, "calls": len(rs), "errors": sum(r.get("error") is not None for r in rs),
                        "rows": sum(int(r.get("rows") or 0) for r in rs),
                        "ood_rows": sum(int(r.get("ood_rows") or 0) for r in rs),
                        "refused_rows": sum(int(r.get("refused") or 0) for r in rs),
                        "latency_ms_p50": float(np.percentile(lat, 50)) if lat.size else None,
                        "latency_ms_p95": float(np.percentile(lat, 95)) if lat.size else None,
                        "last_call": max(r["time"] for r in rs)})
        return {"by_version": out, "calls": len(rows), "errors": sum(r.get("error") is not None for r in rows)}

    # ------------------------------------------------------------------------------------------ describe / export
    def describe(self, name: str, version: str | None = None) -> dict[str, Any]:
        v = self.resolve(name, version)
        meta = self.metadata(name, v)
        return {**meta, "versions": self.versions(name), "latest": self.latest(name),
                "model_ref": {"name": name, "version": v, "kind": meta["kind"], "uri": f"models/{name}/{v}",
                              "inputs": meta.get("inputs"), "outputs": meta.get("outputs"),
                              "validity_bounds": meta.get("validity_bounds"), "metrics": meta.get("metrics"),
                              "sha256": meta.get("model_sha256"), "trust": meta.get("trust")}}

    def export(self, name: str, version: str | None, fmt: str, path: str) -> str:
        """TorchScript / ONNX of a surrogate built on PyTorch; the published ONNX file of an ``onnx`` model."""
        v = self.resolve(name, version)
        meta = self.metadata(name, v)
        if meta["kind"] == "onnx" and fmt == "onnx":
            shutil.copyfile(os.path.join(self._version_dir(name, v), "model.onnx"), path)
            return path
        p = self._load(name, v)
        if p.export is None:
            raise NotImplementedError(f"{meta['kind']} models have no {fmt} export here")
        return str(p.export(path, fmt))

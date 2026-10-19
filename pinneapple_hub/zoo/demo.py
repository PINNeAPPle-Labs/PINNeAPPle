"""What the zoo's demo Space shows, computed by the library: each model's prediction along a sweep of each input,
with its ensemble spread or conformal bound and where its query checks warn or fail, and the files of the static
Space that shows them.

The Space (``huggingface.co/spaces/PINNeAPPle-Labs/zoo-demo``, a static Space: no server) is the page in
``apps/zoo_space/`` over the files :func:`export_space` writes:

* ``models/<name>.sweeps.json`` (:func:`precompute`): for every input, a sweep from 15 % below to 15 % above its
  range, the other inputs at a held-out test case; the predictions, their band, and the verdict of the query checks
  at every point (pass / warn / fail); the checks at the test case and the model's trust decision;
* ``models/<name>.onnx`` and ``models/<name>.json`` (:func:`pinneapple_hub.zoo.web.export_web`): the model and its
  query checks as one ONNX graph, so the page can evaluate the visitor's own inputs in the browser;
* ``lib/onnx.js``, ``lib/zoo.js``: the browser runtime (a dependency-free ONNX interpreter and the JS twin of
  :func:`pinneapple_hub.zoo.checks.evaluate`; ``tests/js/zoo_space.test.mjs`` checks it against these files).

The same calls work in a notebook:

    from pinneapple_hub.zoo import demo
    c = demo.controls("airfoil_cst_rans_mlp")                      # inputs, units, ranges, a held-out case
    r = demo.sweep("airfoil_cst_rans_mlp", {k["name"]: k["value"] for k in c}, vary="alpha_deg")
    r["status"]                                                     # pass / warn / fail along the sweep
"""
from __future__ import annotations

import functools
import json
import shutil
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import _artifacts as art

__all__ = ["models", "controls", "sweep", "trust", "bands", "precompute", "export_space"]

SWEEP_POINTS = 41


def models() -> list[str]:
    """Names of the zoo models the demo offers."""
    return sorted(art.manifest()["models"])


@functools.cache
def _model(name: str):
    """The model, with its reference data fetched once (the query checks read it offline afterwards)."""
    from . import evaluate, load
    zm = load(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evaluate(zm)
    return zm


@functools.cache
def trust(name: str) -> dict[str, Any]:
    """The model's trust report as a whole (``pp.zoo.validate``): decision and the reasons."""
    from . import validate
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = validate(_model(name))
    return {"decision": r.decisao, "reasons": list(getattr(r, "motivos", []) or [])}


def _range(zm, k: str, X: np.ndarray, j: int) -> tuple[float, float]:
    b = zm.bounds()
    if k in b:
        return float(b[k][0]), float(b[k][1])
    return float(X[:, j].min()), float(X[:, j].max())          # no box bound (derived limits): the held-out span


@functools.cache
def _controls(name: str) -> tuple:
    zm = _model(name)
    X, _ = zm.reference("test")
    out = []
    for j, d in enumerate(zm.card.inputs):
        lo, hi = _range(zm, d["name"], X, j)
        out.append({"name": d["name"], "unit": d.get("unit", ""), "description": d.get("description", ""),
                    "lo": lo, "hi": hi, "value": float(f"{X[0, j]:.6g}")})
    return tuple(out)


def controls(name: str) -> list[dict[str, Any]]:
    """One entry per input of the model: name, unit, description, range (validity box, or the held-out span for
    inputs bounded only through derived quantities) and a starting value (the first held-out test case)."""
    return [dict(c) for c in _controls(name)]


def _checks(zm, X: np.ndarray) -> list[dict[str, Any]]:
    from .checks import query_checks
    return query_checks(zm, X)


def sweep(name: str, values: Mapping[str, float], vary: str, *, n: int = 61, margin: float = 0.15) -> dict[str, Any]:
    """Predictions and query checks at ``values`` and along ``vary`` from ``margin`` below to ``margin`` above its
    range (so the sweep shows where the model stops being trusted)."""
    zm = _model(name)
    names = zm.input_names
    if vary not in names:
        raise KeyError(f"{name} has no input {vary!r}; inputs: {names}")
    c = {d["name"]: d for d in _controls(name)}
    lo, hi = c[vary]["lo"], c[vary]["hi"]
    a, b = lo - margin * (hi - lo), hi + margin * (hi - lo)
    if lo > 0:
        a = max(a, lo / 2)                     # positive physical inputs (log features) stay positive
    x0 = np.array([float(values[k]) for k in names])
    grid = np.linspace(a, b, n)
    X = np.repeat(x0[None], n, 0)
    X[:, names.index(vary)] = grid
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = zm.predict(X, warn=False)
        point = zm.predict(x0[None], warn=False)
    return {"model": name, "vary": vary, "x": grid, "range": (lo, hi), "inputs": dict(zip(names, x0.tolist(), strict=True)),
            "pred": {k: np.asarray(v, float) for k, v in pred.items()},
            "status": [r["verdict"]["status"] for r in _checks(zm, X)],
            "point": {k: float(np.asarray(v).ravel()[0]) for k, v in point.items()},
            "point_checks": _checks(zm, x0[None])[0], "units": {d["name"]: d.get("unit", "") for d in
                                                                [*zm.card.inputs, *zm.card.outputs]}}


def bands(outputs: Sequence[str]) -> dict[str, dict[str, Any]]:
    """The main outputs of a model and how each one's uncertainty is drawn: ``{"std": "<name>_std"}`` (ensemble
    spread, drawn as +-2 std) or ``{"upper": "<name with upper95>"}`` (split-conformal 95 % upper bound)."""
    out: dict[str, dict[str, Any]] = {}
    for k in outputs:
        if k.endswith("_std") or "upper95" in k:
            continue
        up = f"{k.removesuffix('_k_w')}_upper95_k_w" if k.endswith("_k_w") else ""
        out[k] = ({"std": f"{k}_std"} if f"{k}_std" in outputs else {"upper": up} if up in outputs else {})
    return out


def _evaluation(e: Mapping[str, Any]) -> dict[str, Any]:
    keep = ("id", "rule", "name", "severity", "kind", "group", "status", "value", "detail_text")
    return {"verdict": e["verdict"], "checks": [{k: c.get(k) for k in keep} for c in e["checks"]]}


def precompute(name: str, *, n: int = SWEEP_POINTS) -> dict[str, Any]:
    """Everything the static Space shows for one model without running it: the inputs (:func:`controls`), the
    outputs with units and bands (:func:`bands`), the model's trust decision (:func:`trust`), the prediction and the
    query checks at the held-out case, and one :func:`sweep` per input (x, every output, verdict per point)."""
    from .web import _round
    zm = _model(name)
    e = art.entry("models", name)
    c = controls(name)
    values = {d["name"]: d["value"] for d in c}
    sweeps, first = {}, None
    for d in c:
        r = sweep(name, values, d["name"], n=n)
        first = first or r
        sweeps[d["name"]] = {"x": r["x"].tolist(), "range": list(r["range"]),
                             "pred": {k: v.tolist() for k, v in r["pred"].items()}, "status": r["status"]}
    units = first["units"]
    outs = list(first["pred"])
    return _round({
        "model": name, "version": e["version"], "revision": e["revision"], "hub": f"https://huggingface.co/{e['hf_repo']}",
        "task": zm.card.task, "description": zm.card.description, "inputs": c,
        "outputs": [{"name": k, "unit": units.get(k, ""), **b} for k, b in bands(outs).items()],
        "all_outputs": outs, "trust": trust(name),
        "default": {"inputs": values, "pred": first["point"], "checks": _evaluation(first["point_checks"])},
        "sweeps": sweeps, "files": {"onnx": f"{name}.onnx", "meta": f"{name}.json"}})


def export_space(directory: str | Path, models_: Sequence[str] | None = None) -> dict[str, Any]:
    """Write the data and runtime of the static Space into ``directory``: ``models/<name>.{onnx,json,sweeps.json}``,
    ``lib/{onnx,zoo}.js`` and ``index.json`` (the models, in order, with their task and trust decision)."""
    from .web import export_web, runtime_dir
    d = Path(directory)
    (d / "models").mkdir(parents=True, exist_ok=True)
    (d / "lib").mkdir(exist_ok=True)
    for f in ("onnx.js", "zoo.js"):
        shutil.copy2(runtime_dir() / f, d / "lib" / f)
    entries = []
    for name in models_ or models():
        _model(name)                      # loads the model and fetches its reference data (read offline below)
        export_web(name, d / "models")
        pre = precompute(name)
        (d / "models" / f"{name}.sweeps.json").write_text(json.dumps(pre, ensure_ascii=False) + "\n")
        entries.append({"name": name, "task": pre["task"], "version": pre["version"],
                        "revision": pre["revision"], "trust": pre["trust"]["decision"]})
    index = {"models": entries, "generator": "pinneapple_hub.zoo.demo.export_space"}
    (d / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False) + "\n")
    return index

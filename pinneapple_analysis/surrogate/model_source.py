"""Use a model already trained instead of training one (#1190): your own (TorchScript, ONNX, a callable, or a model
exported by :mod:`.workflow`), one from the PINNeAPPle zoo, or train one. Every model passes an applicability gate
before it is used, forward or inverse, and the result is re-verified by the solver when there is one.

* :func:`zoo_candidates` ranks the zoo's models for a problem by the outputs it needs and the inputs it has (names,
  then units), with what is missing.
* :func:`open_model` gives one :class:`Predictor` (inputs, outputs, units, bounds, ``predict``) whatever the source.
* :func:`gate` before use: the query inside the model's validity box (and its extra constraints), the problem's
  units convertible to the model's, and a check against references: the zoo's stored reference (its card's
  metrics recomputed) or a few solves of the reference tool, with the error against the model's claimed accuracy.
* :func:`use_model` runs it forward at the query points or inverse (the parameters that reproduce observations,
  bounded least squares through the model), then re-verifies with the solver when one is given.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Predictor:
    name: str
    source: str                                   # "zoo" | "torchscript" | "onnx" | "callable" | "workflow"
    inputs: list[str]
    outputs: list[str]
    bounds: dict[str, tuple[float, float]]
    predict: Callable[[Mapping[str, np.ndarray]], dict[str, np.ndarray]]
    units: dict[str, str] = field(default_factory=dict)
    claimed_rel_error: float | None = None        # the accuracy the model claims (card), relative
    extra_mask: Callable[[Mapping[str, np.ndarray]], np.ndarray] | None = None
    card: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------------------------- sources


def zoo_candidates(outputs: Sequence[str], inputs: Sequence[str] = (), units: Mapping[str, str] | None = None
                   ) -> list[dict[str, Any]]:
    """Zoo models ranked for a problem: outputs matched (by name, or by unit when ``units`` names them), inputs the
    problem lacks; best first."""
    import pinneapple_hub.zoo as Z
    units = dict(units or {})
    out = []
    for name in Z.list():
        c = Z.info(name)
        mo = {d["name"]: d.get("unit", "") for d in c.outputs}
        mi = {d["name"]: d.get("unit", "") for d in c.inputs}
        hit = [o for o in outputs if o in mo or (units.get(o) and units[o] in mo.values())]
        if not hit:
            continue
        missing = [k for k in mi if inputs and k not in inputs]
        out.append({"name": name, "task": c.task, "outputs_matched": hit, "inputs_missing": missing,
                    "score": len(hit) - 0.1 * len(missing), "validity_range": c.validity_range,
                    "metrics": c.validation_metrics})
    return sorted(out, key=lambda r: -r["score"])


def _zoo(name: str) -> Predictor:
    import pinneapple_hub.zoo as Z
    m = Z.load_model(name)
    card = Z.info(name)
    units = {d["name"]: d.get("unit", "") for d in card.inputs + card.outputs}
    met = card.validation_metrics or {}
    mean_pct = [v for k, v in met.items() if k.startswith("test/") and k.endswith("mean_abs_pct")]

    def predict(x):
        return {k: np.asarray(v, float) for k, v in m.predict(dict(x), warn=False).items()}

    def extra(x):
        X = m.as_array(dict(x))
        return np.asarray(m.extra_range_mask(X), bool)
    return Predictor(name, "zoo", list(m.input_names), list(m.output_names), dict(m.bounds()), predict, units,
                     max(mean_pct) / 100 if mean_pct else None, extra, card.to_dict())


def open_model(source: str, *, path: str | None = None, name: str | None = None, fn: Callable | None = None,
               inputs: Sequence[str] = (), outputs: Sequence[str] = (), bounds: Mapping[str, tuple[float, float]]
               | None = None, units: Mapping[str, str] | None = None, claimed_rel_error: float | None = None
               ) -> Predictor:
    """One predictor from ``source``: ``"zoo"`` (``name``), ``"workflow"`` (``path`` of a model exported by
    :func:`.workflow.surrogate_modeling`), ``"torchscript"`` / ``"onnx"`` (``path`` plus the declared ``inputs``,
    ``outputs``, ``bounds``, ``units``: a file carries no physics, so its card is the caller's), ``"callable"``
    (``fn(**inputs) -> dict``)."""
    if source == "zoo":
        return _zoo(str(name))
    if source == "workflow":
        import json
        import os

        from .workflow import load_surrogate
        pred = load_surrogate(str(path))
        card = json.load(open(os.path.join(os.path.dirname(str(path)), "model_card.json")))
        rel = max((v["rel_rmse"] for v in card.get("results", {}).values()), default=None)

        def predict(x):
            n = len(np.atleast_1d(next(iter(x.values()))))
            rows = [{k: float(np.atleast_1d(x[k])[i]) for k in card["inputs"]} for i in range(n)]
            return {o: np.asarray(v[0]) for o, v in pred(rows).items()}
        return Predictor(os.path.basename(os.path.dirname(str(path))), "workflow", card["inputs"], card["outputs"],
                         {k: tuple(v) for k, v in card["bounds"].items()}, predict, card.get("units", {}), rel,
                         card=card)
    if not inputs or not outputs or not bounds:
        raise ValueError(f"a {source} model needs its declared inputs, outputs and bounds (its model card)")
    if source == "torchscript":
        import torch
        mod = torch.jit.load(str(path)).eval()

        def predict(x):
            X = torch.tensor(np.column_stack([np.atleast_1d(x[k]) for k in inputs]), dtype=torch.float32)
            with torch.no_grad():
                Y = mod(X).numpy().reshape(len(X), -1)
            return {o: Y[:, j] for j, o in enumerate(outputs)}
    elif source == "onnx":
        run = onnx_function(str(path), inputs, outputs)

        def predict(x):
            Y = run(np.column_stack([np.atleast_1d(np.asarray(x[k], float)) for k in inputs]))
            return {o: Y[:, j] for j, o in enumerate(outputs)}
    elif source == "callable":
        if fn is None:
            raise ValueError("a callable model needs fn")

        def predict(x):
            n = len(np.atleast_1d(next(iter(x.values()))))
            rows = [fn(**{k: float(np.atleast_1d(x[k])[i]) for k in inputs}) for i in range(n)]
            return {o: np.array([float(r[o]) for r in rows]) for o in outputs}
    else:
        raise ValueError(f"unknown source {source!r}: zoo, workflow, torchscript, onnx or callable")
    return Predictor(str(name or path or getattr(fn, "__name__", source)), source, list(inputs), list(outputs),
                     {k: tuple(v) for k, v in dict(bounds).items()}, predict, dict(units or {}), claimed_rel_error)


_ONNX_DTYPES = {"tensor(float)": np.float32, "tensor(double)": np.float64, "tensor(float16)": np.float16}


def onnx_function(path: str, inputs: Sequence[str], outputs: Sequence[str]
                  ) -> Callable[[np.ndarray], np.ndarray]:
    """An ONNX file as ``f(X) -> Y``: ``X`` (N, len(inputs)) in the declared input order, ``Y`` (N, len(outputs))
    in float64.

    The file's own signature is checked against the declared card when it is opened, and a mismatch is refused with
    both name lists: one input tensor needs ``len(inputs)`` columns (or one tensor per declared input, by name); one
    output tensor needs ``len(outputs)`` columns (or one tensor per declared output, by name). Each input is cast to
    the tensor's own element type (float64 data into a float32 model, and the reverse). A model exported with a
    fixed batch dimension is run in chunks of that size (the last chunk padded and the padding dropped)."""
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise RuntimeError("an ONNX model needs onnxruntime (pip install onnxruntime, or the [onnx] extra); "
                           "TorchScript or a callable work without it") from e
    inputs, outputs = list(inputs), list(outputs)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ins, outs = sess.get_inputs(), sess.get_outputs()

    def last(shape):
        return shape[-1] if shape and isinstance(shape[-1], int) else None

    def refuse(what, have, want):
        raise ValueError(f"the ONNX model's {what} {have} do not match the declared {what} {want}: give the card the "
                         f"model's own names, or export one tensor with one column per {what[:-1]}")
    if len(ins) == 1:
        if len(ins[0].shape) >= 2 and last(ins[0].shape) not in (None, len(inputs)):
            refuse("inputs", f"[{ins[0].name!r} with {last(ins[0].shape)} columns]", inputs)
        feed_cols = None
    else:
        if sorted(i.name for i in ins) != sorted(inputs):
            refuse("inputs", [i.name for i in ins], inputs)
        feed_cols = {i.name: inputs.index(i.name) for i in ins}
    if len(outs) == 1:
        n_cols = last(outs[0].shape) if len(outs[0].shape) >= 2 else 1
        if n_cols not in (None, len(outputs)):
            refuse("outputs", f"[{outs[0].name!r} with {n_cols} columns]", outputs)
        out_cols = None
    else:
        missing = [o for o in outputs if o not in {t.name for t in outs}]
        if missing:
            refuse("outputs", [t.name for t in outs], outputs)
        out_cols = list(outputs)
    for i in ins:
        if i.type not in _ONNX_DTYPES:
            raise ValueError(f"ONNX input {i.name!r} is {i.type}: only float16/32/64 inputs are supported")
    batch = ins[0].shape[0] if ins[0].shape and isinstance(ins[0].shape[0], int) else None

    def feed(X):
        if feed_cols is None:
            i = ins[0]
            A = X.astype(_ONNX_DTYPES[i.type])
            return {i.name: A if len(i.shape) != 1 else A[:, 0]}
        return {i.name: (X[:, [c]] if len(i.shape) >= 2 else X[:, c]).astype(_ONNX_DTYPES[i.type])
                for i, c in ((i, feed_cols[i.name]) for i in ins)}

    def run_chunk(X):
        res = sess.run(out_cols, feed(X))
        if out_cols is None:
            return np.asarray(res[0], np.float64).reshape(len(X), -1)
        return np.column_stack([np.asarray(r, np.float64).reshape(len(X)) for r in res])

    def f(X):
        X = np.asarray(X, np.float64)
        X = X.reshape(len(X), -1) if X.ndim > 1 else X.reshape(-1, len(inputs))
        if X.shape[1] != len(inputs):
            raise ValueError(f"{X.shape[1]} input columns for the declared inputs {inputs}")
        if batch is None or batch == len(X):
            return run_chunk(X)
        parts = []
        for s in range(0, len(X), batch):
            C = X[s:s + batch]
            pad = batch - len(C)
            if pad:
                C = np.vstack([C, np.repeat(C[-1:], pad, axis=0)])
            parts.append(run_chunk(C)[:batch - pad])
        return np.vstack(parts)
    return f


# ----------------------------------------------------------------------------------------------- the gate


def _units_ok(problem: Mapping[str, str], model: Mapping[str, str]) -> list[str]:
    from pinneapple_data.physical_units import parse_unit
    bad = []
    for k, u in problem.items():
        if k not in model or not u or not model[k]:
            continue
        try:
            if parse_unit(u).dim != parse_unit(model[k]).dim:
                bad.append(f"{k}: problem in {u}, model in {model[k]}")
        except Exception:  # noqa: BLE001 - an unparseable unit is reported, not guessed
            if u != model[k]:
                bad.append(f"{k}: units {u!r} and {model[k]!r} cannot be compared")
    return bad


def gate(p: Predictor, query: Mapping[str, Sequence[float]], *, problem_units: Mapping[str, str] | None = None,
         reference: Callable[..., Mapping[str, float]] | None = None, n_reference: int = 8, tolerance: float | None
         = None, seed: int = 0) -> dict[str, Any]:
    """Whether ``p`` may be used at ``query``: ``{"ok", "checks"}``. ``reference(**inputs) -> outputs`` (a solver)
    is compared with the model at ``n_reference`` query points; without one, a zoo model's stored reference is
    recomputed. ``tolerance``: relative error allowed (default: twice the model's claimed error, else 5 %)."""
    checks: list[dict[str, Any]] = []

    def chk(code, ok, detail, value=None):
        ok = None if ok is None else bool(ok)
        checks.append({"code": code, "status": "pass" if ok is True else ("fail" if ok is False else "inconclusive"),
                       "detail": detail, "value": value})
    missing = [k for k in p.inputs if k not in query]
    chk("gate.inputs", not missing, "every input of the model is given" if not missing else f"missing {missing}")
    if missing:
        return {"ok": False, "checks": checks}
    q = {k: np.atleast_1d(np.asarray(query[k], float)) for k in p.inputs}
    n = len(next(iter(q.values())))
    out = []
    for k, (lo, hi) in p.bounds.items():
        if k in q:
            v = q[k]
            bad = (v < lo - 1e-12) | (v > hi + 1e-12)
            if bad.any():
                out.append(f"{k} = {v[bad][0]:g} outside [{lo:g}, {hi:g}]")
    if p.extra_mask is not None:
        m = p.extra_mask(q)
        if (~m).any():
            out.append(f"{int((~m).sum())} of {n} points outside the model's derived-quantity constraints")
    chk("gate.validity_range", not out, f"all {n} query points inside the model's validity range" if not out
        else "; ".join(out))
    ub = _units_ok(problem_units or {}, p.units)
    chk("gate.units", not ub, "the problem's units convert to the model's" if not ub else "; ".join(ub))
    tol = tolerance if tolerance is not None else (2 * p.claimed_rel_error if p.claimed_rel_error else 0.05)
    if reference is not None:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=min(n_reference, n), replace=False)
        sub = {k: v[idx] for k, v in q.items()}
        t0 = time.perf_counter()
        ref = [reference(**{k: float(sub[k][i]) for k in p.inputs}) for i in range(len(idx))]
        ref_s = (time.perf_counter() - t0) / len(idx)
        pr = p.predict(sub)
        errs = {}
        for o in p.outputs:
            if all(o in r for r in ref):
                y = np.array([float(r[o]) for r in ref])
                errs[o] = float(np.max(np.abs(pr[o] - y) / np.maximum(np.abs(y), 1e-12)))
        worst = max(errs.values()) if errs else None
        chk("gate.reference", None if worst is None else worst <= tol,
            f"largest relative error against {len(idx)} reference solves {worst:.2%} (allowed {tol:.1%}; reference "
            f"{ref_s:.3g} s per solve)" if worst is not None else "the reference returns none of the outputs",
            errs)
    elif p.source == "zoo":
        import pinneapple_hub.zoo as Z
        ev = Z.evaluate(p.name)
        ev = ev.get("metrics", ev)
        claimed = {k: v for k, v in (p.card.get("validation_metrics") or {}).items() if isinstance(v, (int, float))}
        diffs = {k: abs(float(ev[k]) - float(v)) / max(abs(float(v)), 1e-12) for k, v in claimed.items() if k in ev}
        worst = max(diffs.values()) if diffs else None
        chk("gate.reference", None if worst is None else worst <= 0.05,
            f"the card's {len(diffs)} metrics recomputed on the stored reference: largest change {worst:.2%}"
            if worst is not None else "no stored metrics to recompute", diffs)
    else:
        chk("gate.reference", None, "no reference solver and no stored reference: the model's accuracy is its "
                                    "card's claim, not checked here")
    return {"ok": all(c["status"] != "fail" for c in checks), "checks": checks}


# ----------------------------------------------------------------------------------------------- use


def use_model(p: Predictor, *, mode: str = "forward", query: Mapping[str, Sequence[float]] | None = None,
              observed: Mapping[str, float] | None = None, unknown: Sequence[str] = (),
              known: Mapping[str, float] | None = None, reference: Callable[..., Mapping[str, float]] | None = None,
              problem_units: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Forward: predictions at ``query``. Inverse: the ``unknown`` inputs (inside the model's bounds) that make the
    model reproduce ``observed`` outputs with the ``known`` inputs, by bounded least squares from several starts;
    then the estimate is re-run through ``reference`` when given."""
    from scipy.optimize import least_squares
    if mode == "forward":
        g = gate(p, query or {}, problem_units=problem_units, reference=reference)
        if not g["ok"]:
            return {"mode": mode, "used": False, "gate": g}
        pr = p.predict({k: np.atleast_1d(np.asarray(v, float)) for k, v in (query or {}).items()})
        return {"mode": mode, "used": True, "gate": g, "predictions": {k: v.tolist() for k, v in pr.items()}}
    if mode != "inverse":
        raise ValueError("mode is 'forward' or 'inverse'")
    known = dict(known or {})
    unknown = list(unknown)
    lo = np.array([p.bounds[k][0] for k in unknown])
    hi = np.array([p.bounds[k][1] for k in unknown])
    obs = {k: float(v) for k, v in (observed or {}).items()}
    scale = np.array([max(abs(v), 1e-12) for v in obs.values()])

    def res(z):
        x = {**{k: np.array([v]) for k, v in known.items()}, **{k: np.array([z[i]]) for i, k in enumerate(unknown)}}
        pr = p.predict(x)
        return np.array([float(pr[o][0]) for o in obs]) / scale - np.array(list(obs.values())) / scale
    best = None
    for s in np.linspace(0.2, 0.8, 4):
        # a network answers in float32: a finite-difference step of 1e-3 of the range sees its slope
        r = least_squares(res, lo + s * (hi - lo), bounds=(lo, hi), x_scale=hi - lo, diff_step=1e-3)
        if best is None or r.cost < best.cost:
            best = r
    est = {k: float(v) for k, v in zip(unknown, best.x, strict=True)}
    full = {**known, **est}
    g = gate(p, {k: [v] for k, v in full.items()}, problem_units=problem_units)
    out = {"mode": mode, "used": True, "estimate": est, "misfit_rel": float(np.sqrt(2 * best.cost / len(obs))),
           "at_bound": [k for k, v, a, b in zip(unknown, best.x, lo, hi, strict=True)
                        if min(v - a, b - v) < 1e-6 * (b - a)], "gate": g}
    tol = 2 * p.claimed_rel_error if p.claimed_rel_error else 0.05
    checks = []

    def chk(code, ok, detail, value=None):
        checks.append({"code": code, "status": "pass" if ok else "fail", "detail": detail, "value": value})
    chk("inverse.misfit", out["misfit_rel"] <= tol, f"the model reproduces the observations to {out['misfit_rel']:.2%} "
        f"(allowed {tol:.1%}, twice the model's claimed error)", out["misfit_rel"])
    chk("inverse.inside_bounds", not out["at_bound"], "the estimate is inside the model's range" if not out["at_bound"]
        else f"{out['at_bound']} at the edge of the model's range: the observation may need a value outside it")
    if reference is not None:
        ref = reference(**full)
        out["reverified"] = {o: {"observed": obs[o], "reference_at_estimate": float(ref[o]),
                                 "rel_diff": abs(float(ref[o]) - obs[o]) / max(abs(obs[o]), 1e-12)}
                             for o in obs if ref and o in ref}
        worst = max((v["rel_diff"] for v in out["reverified"].values()), default=None)
        chk("inverse.reverified", worst is not None and worst <= tol,
            f"the reference solver at the estimate reproduces the observations to {worst:.2%} (allowed {tol:.1%})"
            if worst is not None else "the reference solver rejected the estimate", worst)
    out["checks"] = checks
    out["used"] = all(c["status"] == "pass" for c in checks)
    return out

"""The standard surrogate-modeling workflow (#1186): data or a design of experiments over a tool, the model family
``recommend_architecture`` picks (or none), the library's equal-budget comparison, checks on solves never trained
on, and an exported model that refuses to extrapolate.

1. **Data**: an uploaded table (``table``: rows of inputs and outputs) or a Sobol design of experiments over a tool
   (``tool``: a callable or a library tool id, ``bounds`` per input, ``outputs`` it returns). The cost of one tool
   call is measured.
2. **Which model**: :func:`~pinneapple_analysis.verification.architecture_recommendation.recommend_architecture` with
   the measured cost and ``n_uses`` (how many predictions the model will serve). ``none`` stops here with its
   reason: call the tool (``force=True`` trains anyway, to compare).
3. **Families on one protocol**: :func:`pinneapple_autolab.study.benchmark.benchmark` (k-fold CV on train + val,
   cross-conformal 90 % intervals, the selection rule written in the record); the GP, the MLP ensemble and the
   quadratic response surface (the baseline to beat).
4. **Checks** on ``n_fresh`` new tool calls inside the box (or the table's held-out rows): relative RMSE against
   ``accuracy_target``, coverage of the 90 % interval (binomial band at that sample size), and that the selected
   family beats the quadratic baseline.
5. **Export**: the selected models with their intervals and a model card (inputs, ranges, units, errors, checks,
   provenance); :func:`load_surrogate` returns a predictor that raises on inputs outside the trained box.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np


def _call(fn, kw: dict[str, float], outputs: Sequence[str]) -> list[float]:
    r = fn(**kw)
    if isinstance(r, Mapping):
        return [float(r[o]) for o in outputs]
    if hasattr(r, "__dict__") and all(hasattr(r, o) for o in outputs):
        return [float(getattr(r, o)) for o in outputs]
    if len(outputs) == 1:
        return [float(r)]
    return [float(v) for v in r]


def _binomial_band(p: float, n: int, z: float = 2.0) -> tuple[float, float]:
    s = math.sqrt(p * (1 - p) / max(n, 1))
    return p - z * s, p + z * s


def surrogate_modeling(*, tool: str | Callable[..., Any] | None = None, bounds: Mapping[str, tuple[float, float]]
                       | None = None, outputs: Sequence[str] | None = None, fixed: Mapping[str, Any] | None = None,
                       table: Mapping[str, Sequence[float]] | None = None, inputs: Sequence[str] | None = None,
                       n_doe: int = 80, n_fresh: int = 200, n_uses: int = 10_000, accuracy_target: float = 0.05,
                       force: bool = False, seed: int = 0, out_dir: str | None = None,
                       units: Mapping[str, str] | None = None,
                       export_format: str = "pickle") -> dict[str, Any]:
    """Run the workflow; returns a JSON-ready record (``model_path`` when ``out_dir`` is given)."""
    if export_format not in ("pickle", "safe_json"):
        raise ValueError("unsupported surrogate export format")
    from scipy.stats import qmc

    from pinneapple_autolab.study import benchmark as B
    from pinneapple_autolab.study.dataset import StudyDataset, region_split
    fixed = dict(fixed or {})
    fn = None
    solve_s = None
    if tool is not None:
        from pinneapple_autolab.study.verify import resolve_model
        if not bounds or not outputs:
            raise ValueError("a tool needs the bounds of its varied inputs and the outputs to model")
        fn = resolve_model(tool)
        inputs = list(bounds)
        lo = np.array([bounds[k][0] for k in inputs], float)
        hi = np.array([bounds[k][1] for k in inputs], float)
        with warnings.catch_warnings():            # Sobol balance is best at powers of two; any n is valid
            warnings.simplefilter("ignore")
            X = qmc.scale(qmc.Sobol(len(inputs), seed=seed).random(n_doe), lo, hi)
        t0 = time.perf_counter()
        Y = np.array([_call(fn, {**fixed, **dict(zip(inputs, map(float, x), strict=True))}, outputs) for x in X])
        solve_s = (time.perf_counter() - t0) / len(X)
        source = {"kind": "tool", "tool": tool if isinstance(tool, str) else getattr(tool, "__name__", "callable"),
                  "doe": f"Sobol, {n_doe} points, seed {seed}", "fixed": fixed}
    elif table is not None:
        if not inputs or not outputs:
            raise ValueError("a table needs the names of its input and output columns")
        X = np.column_stack([np.asarray(table[k], float) for k in inputs])
        Y = np.column_stack([np.asarray(table[k], float) for k in outputs])
        ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
        X, Y = X[ok], Y[ok]
        bounds = {k: (float(X[:, j].min()), float(X[:, j].max())) for j, k in enumerate(inputs)}
        source = {"kind": "table", "rows": int(len(X)), "dropped_rows": int((~ok).sum())}
    else:
        raise ValueError("give a tool (with bounds and outputs) or a table (with inputs and outputs)")
    inputs, outputs = list(inputs), list(outputs)

    rec: dict[str, Any] = {"recommended": ["kpi_regressor"], "reasoning": "recommendation unavailable"}
    try:
        from pinneapple_analysis.verification.architecture_recommendation import (
            recommend_architecture,
        )
        r = recommend_architecture(n_high_fidelity_simulations=len(X), needs_parameter_generalization=True,
                                   geometry_varies=False, has_analytical_solver=fn is not None,
                                   solve_seconds=solve_s, n_evaluations=n_uses if solve_s is not None else None,
                                   output_kind="scalar", n_inputs=len(inputs), accuracy_target=accuracy_target)
        rec = {"recommended": list(r.recommended), "reasoning": r.reasoning, "decision_path": list(r.decision_path),
               "checks": list(r.checks)}
    except (ImportError, TypeError) as e:
        rec["reasoning"] = f"recommendation unavailable: {e}"
    out: dict[str, Any] = {"source": source, "inputs": inputs, "outputs": outputs, "bounds": dict(bounds),
                           "recommendation": rec, "cost": {"solve_seconds": solve_s, "n_uses": n_uses}}
    if rec["recommended"][0] == "none" and not force:
        out.update(trained=False, checks=[{"code": "surrogate.not_needed", "status": "pass", "detail":
                   (rec.get("decision_path") or [rec["reasoning"]])[0], "value": None}])
        return out

    ids = [f"s{i:04d}" for i in range(len(X))]
    ds = StudyDataset(ids, inputs, outputs, X, Y, units=dict(units or {}), bounds=dict(bounds),
                      provenance=source, purpose=["doe"] * len(X))
    ds = ds.with_split(region_split(ds, None, val=0.15, test=0.15 if fn is None else 0.0, seed=seed))
    board = B.benchmark(ds, outputs, None, log=lambda s: None, seed=seed)
    sel = board.select()
    # checks on solves never trained on
    if fn is not None:
        lo = np.array([bounds[k][0] for k in inputs])
        hi = np.array([bounds[k][1] for k in inputs])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Xf = qmc.scale(qmc.Sobol(len(inputs), seed=seed + 1000).random(n_fresh), lo, hi)
        Yf = np.array([_call(fn, {**fixed, **dict(zip(inputs, map(float, x), strict=True))}, outputs) for x in Xf])
        where = f"{n_fresh} new tool calls (Sobol, seed {seed + 1000})"
    else:
        te = ds.idx("test")
        Xf, Yf = X[te], Y[te]
        where = f"{len(te)} held-out rows of the table"
    Uf = ds.unit_x(Xf)
    checks: list[dict[str, Any]] = []

    def chk(code, ok, detail, value=None):
        ok = None if ok is None else bool(ok)
        checks.append({"code": code, "status": "pass" if ok is True else ("fail" if ok is False else "inconclusive"),
                       "detail": detail, "value": value})
    per: dict[str, Any] = {}
    for j, o in enumerate(outputs):
        fam = sel["choice"][o]
        if fam is None:
            chk(f"surrogate.{o}.eligible", False, sel["reasons"][o])
            continue
        mu, hw = board.predictor(o, fam)(Uf)
        scale = max(float(np.ptp(Y[:, j])), 1e-300)
        rel = float(np.sqrt(np.mean((mu - Yf[:, j]) ** 2)) / scale)
        cover = float(np.mean(np.abs(mu - Yf[:, j]) <= hw))
        mq, _ = board.predictor(o, "quadratic")(Uf)
        rel_q = float(np.sqrt(np.mean((mq - Yf[:, j]) ** 2)) / scale)
        lo_b, hi_b = _binomial_band(0.9, len(Uf))
        per[o] = {"family": fam, "rel_rmse": rel, "coverage90": cover, "baseline_rel_rmse": rel_q,
                  "reason": sel["reasons"][o]}
        chk(f"surrogate.{o}.held_out_error", rel <= accuracy_target,
            f"{fam}: relative RMSE {rel:.2%} of the output range on {where} (target {accuracy_target:.0%})", rel)
        chk(f"surrogate.{o}.coverage", cover >= lo_b,       # over-coverage is conservative, under-coverage is not
            f"90 % interval covers {cover:.1%} of {where} (at least {lo_b:.1%}: the binomial band at that size)",
            cover)
        chk(f"surrogate.{o}.beats_baseline", fam == "quadratic" or rel < rel_q,
            f"{fam} {rel:.2%} vs quadratic response surface {rel_q:.2%}", rel_q)
    chk("surrogate.no_extrapolation", True, "the exported predictor refuses inputs outside "
        + ", ".join(f"{k} in [{a:g}, {b:g}]" for k, (a, b) in bounds.items()))
    out.update(trained=True, board=board.table(), selection=sel, results=per, checks=checks,
               protocol=board.protocol)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if export_format == "safe_json":
            from .safe_native_export import export_native_scalar
            path = os.path.join(out_dir, "surrogate.safe.json")
            export_native_scalar(
                {o: board.models[f"{o}/{per[o]['family']}"] for o in per},
                {o: per[o]["family"] for o in per},
                {o: board.q[f"{o}/{per[o]['family']}"] for o in per},
                inputs, outputs, bounds, path)
        else:
            path = os.path.join(out_dir, "surrogate.pkl")
            payload = {"inputs": inputs, "outputs": outputs, "bounds": dict(bounds),
                       "models": {o: board.models[f"{o}/{per[o]['family']}"] for o in per},
                       "q": {o: board.q[f"{o}/{per[o]['family']}"] for o in per}}
            with open(path, "wb") as f:
                pickle.dump(payload, f)
        card = {k: out[k] for k in ("source", "inputs", "outputs", "bounds", "results", "checks", "recommendation")}
        card["units"] = dict(units or {})
        with open(os.path.join(out_dir, "model_card.json"), "w") as f:
            json.dump(card, f, indent=1, default=float)
        out["model_path"] = path
    return out


def load_surrogate(path: str) -> Callable[[Mapping[str, float] | Sequence[Mapping[str, float]]], dict[str, Any]]:
    """The exported predictor: ``predict({"x": .., ...}) -> {output: (mean, 90 % half width)}``; raises ValueError for
    an input outside the trained box (no silent extrapolation). Load only files you produced: it is a pickle."""
    with open(path, "rb") as f:
        p = pickle.load(f)  # noqa: S301 - the caller's own export

    def predict(x):
        rows = [x] if isinstance(x, Mapping) else list(x)
        X = np.array([[float(r[k]) for k in p["inputs"]] for r in rows])
        lo = np.array([p["bounds"][k][0] for k in p["inputs"]])
        hi = np.array([p["bounds"][k][1] for k in p["inputs"]])
        bad = (X < lo - 1e-12) | (X > hi + 1e-12)
        if bad.any():
            i, j = map(int, np.argwhere(bad)[0])
            k = p["inputs"][j]
            raise ValueError(f"{k} = {X[i, j]:g} is outside the trained range [{lo[j]:g}, {hi[j]:g}]")
        U = (X - lo) / (hi - lo)
        res = {}
        for o, m in p["models"].items():
            mu, s = m.predict(U)
            res[o] = (np.asarray(mu).tolist(), (p["q"][o] * np.asarray(s)).tolist())
        return res
    return predict

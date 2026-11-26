"""``pinneapple check`` as the pipeline's second pair of eyes.

The gates in :mod:`pinneapple_pipeline.verification` measure what the pipeline itself can compute, and each one
says so. ``pinneapple_check`` asks the questions from outside: it loads the trained model through a plain
``(N, n_inputs) -> (N, n_outputs)`` function, queries it, and applies the rules of a physics linter that knows
nothing about how the model was built — bounds (PC4xx), out-of-distribution use (PC6xx), the integrity of the
checkpoint (PC8xx), and, when the run declares its physics, boundary/initial conditions (PC2xx) and the PDE
residual (PC3xx).

This module writes those three files into the run folder::

    verification/predict.py               # the model behind the wrapper the checker loads
    verification/queries.csv              # points the checker queries it at
    verification/pinneapple-check.toml    # [[model]] with the inputs, outputs, training range and physics

and then runs the checker over it. The gates consume the outcomes: ``applicability`` reads PC6xx, ``export`` reads
PC8xx and ``physics`` reads PC2xx/PC3xx — and where the checker has nothing to say, the gate falls back to its own
measurement and the report says which one it used. A checker that fails to run never fails a gate: it becomes a
note with the exception.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["write_check_files", "run_check", "outcomes_for", "first_failure", "summarize", "register_run",
           "predict_for_run", "PREFIX_PHYSICS", "PREFIX_APPLICABILITY", "PREFIX_EXPORT"]

#: Which gate reads which family of rules.
PREFIX_PHYSICS = ("PC1", "PC2", "PC3")
PREFIX_APPLICABILITY = ("PC4", "PC6")
PREFIX_EXPORT = ("PC8",)

_PREDICT = '''"""The model of this run, behind the function `pinneapple check` loads (`predict.py:predict`).

Written by `pp.SurrogatePipeline`; it reloads the run folder it sits in, so it answers exactly what the pipeline
answers — the checkpoint, not a copy of it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parents[1]


def predict(X):
    """`(N, {n_in}) -> (N, {n_out})`: the model, with the output order of the check configuration."""
    import numpy as np

    from pinneapple_pipeline.checks import predict_for_run
    return predict_for_run(RUN, np.asarray(X, dtype=float))
'''


# A run whose model is only in memory (a physics-only PINN is not always reloadable) is still checkable in the
# process that trained it: the folder is registered here and `predict.py` uses it before trying to reload.
_LIVE: dict[str, Any] = {}


def register_run(state) -> None:
    """Make the model of ``state`` the one ``predict.py`` answers with, without going through disk."""
    try:
        _LIVE[str(Path(state.out_dir).resolve())] = state
    except Exception:                                             # pragma: no cover - a missing out_dir
        return


def predict_for_run(run_dir, X):
    """The array prediction of the run in ``run_dir``: the live model when this process trained it, else the
    checkpoint the export stage wrote."""
    from .pipeline import predict_array_with

    live = _LIVE.get(str(Path(run_dir).resolve()))
    if live is not None and live.model_object is not None:
        return predict_array_with(live, X)
    from .pipeline import SurrogatePipeline
    return SurrogatePipeline.from_run(run_dir).predict_array(X)


# --------------------------------------------------------------------------------------------------- the files
def _names(state) -> tuple[list[str], list[str]]:
    """The model's inputs and outputs, in the order the backend takes them.

    A field surrogate answers one row per case with one column per grid point, and the checker speaks in named
    scalar outputs: the field's columns are declared as ``u[0] … u[N-1]`` so its rules (finite values, bounds) apply
    to each grid point instead of being skipped.
    """
    cfg, bundle = state.cfg, state.bundle
    if cfg.data.is_physics_only:
        return list(bundle.coords), list(bundle.fields)
    dataset = state.dataset
    if dataset is not None:
        return list(dataset.input_names), _expand(dataset.output_names, dataset.grid_shape)
    saved = state.stages["preprocess"].artifacts.get("dataset_report")
    if saved and Path(saved).is_file():
        report = json.loads(Path(saved).read_text(encoding="utf-8"))
        return list(report["inputs"]), list(report["outputs"])
    return [], []


def _expand(names: list[str], grid_shape: tuple[int, ...]) -> list[str]:
    """One name per column of a field output."""
    width = int(np.prod(grid_shape)) if grid_shape else 0
    if not width or len(names) != 1:
        return list(names)
    return [f"{names[0]}[{i}]" for i in range(width)]


def _ranges(state, inputs: list[str]) -> dict[str, list[float]]:
    """The range each input was trained over: the design's own, or the problem's domain."""
    cfg, bundle = state.cfg, state.bundle
    out: dict[str, list[float]] = {}
    if cfg.data.is_physics_only:
        for c in inputs:
            lo, hi = bundle.domains.get(c, (None, None))
            if lo is not None:
                out[c] = [float(lo), float(hi)]
        return out
    space = cfg.parameters.to_space() if cfg.parameters.parameters else None
    by_name = {p.name: p for p in getattr(space, "parameters", [])} if space is not None else {}
    for name in inputs:
        p = by_name.get(name)
        if p is not None and getattr(p, "kind", "continuous") == "continuous" and p.low is not None \
                and p.high is not None:
            out[name] = [float(p.low), float(p.high)]
    if not out and state.cases is not None:                       # a categorical or a renamed design
        for name in inputs:
            if name in state.cases.columns:
                values = state.cases[name].to_numpy()
                try:
                    out[name] = [float(np.min(values)), float(np.max(values))]
                except (TypeError, ValueError):
                    continue
    return out


def _physics_parameters(state, physics: str) -> dict[str, float]:
    """The problem's own coefficients that the checker's equation for ``physics`` knows by name.

    A symbolic problem declares ``alpha = 0.1``; the heat equation of ``pinneapple_check`` takes ``alpha`` (and sets
    ``rho_cp = 1``), so the residual it computes is the one the model was trained on. A coefficient the equation does
    not know is not passed: it would be a missing parameter and the checker says so.
    """
    from pinneapple_check.physics import PARAMS

    known = set(PARAMS.get(physics, {}))
    constants = getattr(state.cfg.problem, "constants", None) or {}
    out: dict[str, float] = {}
    for name, value in constants.items():
        try:
            out[str(name)] = float(value)
        except (TypeError, ValueError):
            continue
    return {k: v for k, v in out.items() if k in known}


def _query_points(state, inputs: list[str], ranges: dict[str, list[float]], n: int, seed: int) -> np.ndarray:
    """``n`` queries spread over the training range (the ends included, so the box is what it says)."""
    rng = np.random.default_rng(seed)
    cols = []
    for name in inputs:
        lo, hi = ranges.get(name, [0.0, 1.0])
        col = rng.uniform(float(lo), float(hi), size=n)
        col[0], col[1] = float(lo), float(hi)                     # the two ends, always queried
        cols.append(col)
    return np.stack(cols, axis=1) if cols else np.zeros((0, 0))


def write_check_files(state, *, physics: str = "none", n_queries: int = 32) -> dict[str, str] | None:
    """Write ``predict.py``, ``queries.csv`` and ``pinneapple-check.toml`` into ``verification/``.

    Returns the paths, or ``None`` when the model cannot be expressed this way (no trained object yet, no named
    inputs) — the caller reports that instead of writing files the checker could not use.
    """
    inputs, outputs = _names(state)
    if not inputs or not outputs:
        return None
    ranges = _ranges(state, inputs)
    directory = state.path("verification", "check").parent
    directory.mkdir(parents=True, exist_ok=True)

    predict = directory / "predict.py"
    predict.write_text(_PREDICT.format(n_in=len(inputs), n_out=len(outputs)), encoding="utf-8")

    queries = _query_points(state, inputs, ranges, int(n_queries), int(state.cfg.sampling.seed) + 11)
    queries_csv = directory / "queries.csv"
    header = ",".join(inputs)
    body = "\n".join(",".join(repr(float(v)) for v in row) for row in queries)
    queries_csv.write_text(f"{header}\n{body}\n", encoding="utf-8")

    lines = ["# Written by pp.SurrogatePipeline: the run's model as a target of `pinneapple check`.",
             "[check]", 'select = ["ALL"]', 'fail-on = "never"', "",
             "[[model]]", 'path = "predict.py:predict"', f"inputs = {json.dumps(inputs)}",
             f"outputs = {json.dumps(outputs)}", 'queries = "queries.csv"']
    if ranges:
        rendered = ", ".join(f"{k} = [{v[0]!r}, {v[1]!r}]" for k, v in ranges.items())
        lines.append(f"train-range = {{ {rendered} }}")
    if physics and physics != "none":
        lines.append(f'physics = "{physics}"')
        constants = _physics_parameters(state, physics)
        if constants:
            rendered = ", ".join(f"{k} = {float(v)!r}" for k, v in constants.items())
            lines.append(f"parameters = {{ {rendered} }}")
    config = directory / "pinneapple-check.toml"
    config.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"predict": str(predict), "queries": str(queries_csv), "config": str(config)}


# ------------------------------------------------------------------------------------------------- the checker
def run_check(state, *, physics: str = "none", n_queries: int = 32) -> dict[str, Any] | None:
    """Write the files and run ``pinneapple_check.run``; the outcomes are what the gates read."""
    files = write_check_files(state, physics=physics, n_queries=n_queries)
    if files is None:
        return None
    register_run(state)
    from pinneapple_check import run as run_checks

    try:
        report = run_checks([files["config"]])
    except Exception as exc:                                      # a checker that cannot run is not a failure
        return {"config": files["config"], "error": f"{type(exc).__name__}: {exc}", "outcomes": []}
    outcomes = [{"rule": o.rule, "target": o.target, "subject": o.subject, "passed": bool(o.passed),
                 "message": o.message, "value": _number(o.value), "threshold": _number(o.threshold),
                 "severity": o.severity} for o in report.outcomes]
    summary = summarize(outcomes)
    summary.update({"config": files["config"], "predict": files["predict"], "queries": files["queries"],
                    "report": _write_report(state, report), "notes": list(report.notes)[:8]})
    if physics not in (None, "", "none") and not [o for o in outcomes
                                                   if any(str(o["rule"]).startswith(p) for p in PREFIX_PHYSICS)]:
        summary["physics_not_applied"] = (
            f"verification.check_physics is {physics!r} but the checker found nothing to apply it to "
            f"(it named the fields it looks for in its notes: {summary['notes']}); the physics gate measured the "
            "residual itself")
    return summary


def _write_report(state, report) -> str:
    path = state.path("verification", "check", "report.json")
    try:
        path.write_text(json.dumps(report.to_dict(), indent=1, default=str), encoding="utf-8")
    except Exception:                                             # pragma: no cover - the checker owns its schema
        return ""
    return str(path)


def _number(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def summarize(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts by rule, and the violations (what the gates quote)."""
    by_rule: dict[str, dict[str, Any]] = {}
    for o in outcomes:
        row = by_rule.setdefault(o["rule"], {"rule": o["rule"], "checks": 0, "failed": 0, "severity": o["severity"],
                                             "messages": []})
        row["checks"] += 1
        if not o["passed"]:
            row["failed"] += 1
            if len(row["messages"]) < 3:
                row["messages"].append(o["message"])
    violations = [o for o in outcomes if not o["passed"]]
    return {"outcomes": outcomes, "by_rule": by_rule, "n_checks": len(outcomes), "n_failed": len(violations),
            "errors": [o for o in violations if o["severity"] == "error"],
            "warnings": [o for o in violations if o["severity"] == "warning"]}


# -------------------------------------------------------------------------------------------------- consumers
def outcomes_for(check: dict[str, Any] | None, prefixes: tuple[str, ...]) -> list[dict[str, Any]]:
    """The outcomes of ``check`` whose rule id starts with one of ``prefixes``."""
    if not check:
        return []
    return [o for o in check.get("outcomes") or [] if any(str(o["rule"]).startswith(p) for p in prefixes)]


def first_failure(check: dict[str, Any] | None, prefixes: tuple[str, ...]) -> dict[str, Any] | None:
    """The first failed outcome of those families (errors before warnings)."""
    rows = [o for o in outcomes_for(check, prefixes) if not o["passed"]]
    rows.sort(key=lambda o: 0 if o["severity"] == "error" else 1)
    return rows[0] if rows else None

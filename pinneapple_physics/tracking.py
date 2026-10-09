"""Experiment tracking adapter: send an ``ExperimentResult`` to MLflow or Weights & Biases.

>>> exp = pp.Experiment("burgers_1d", method="pinn", reference="analytic", tracker="mlflow")
>>> exp.run()                       # params, metrics, wall time and the saved record go to the active MLflow setup

``tracker`` is ``"mlflow"``, ``"wandb"`` or any callable ``f(result)``. The trackers are optional dependencies,
imported only when used; their own configuration (``MLFLOW_TRACKING_URI``, ``WANDB_PROJECT``, login) applies.
What is logged is the same for both: the flat config as parameters, every metric as ``<metric>/<field>``, the wall
time, the problem fingerprint as a tag, and the ``Experiment`` record directory (``result.json`` and ``model.pt``)
as artifacts.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any, Callable, Dict, Union

__all__ = ["log_result", "flat_metrics", "flat_params", "TRACKERS"]


def flat_metrics(result: Any) -> Dict[str, float]:
    out = {"wall_time_s": float(result.wall_time_s)}
    for metric, per_field in (result.metrics or {}).items():
        for fld, v in per_field.items():
            out[f"{metric}/{fld}"] = float(v)
    return out


def flat_params(result: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in result.config.items():
        if isinstance(v, dict):
            out.update({f"{k}.{kk}": str(vv) for kk, vv in v.items()})
        else:
            out[k] = str(v)
    return out


def _mlflow(result: Any, **kw: Any) -> str:
    import mlflow

    with mlflow.start_run(run_name=result.config.get("name") or None, **kw) as run:
        mlflow.log_params(flat_params(result))
        mlflow.log_metrics(flat_metrics(result))
        mlflow.set_tags({"problem_fingerprint": result.problem_fingerprint, "library": "pinneapple"})
        with tempfile.TemporaryDirectory() as d:
            result.save(d)
            mlflow.log_artifacts(d, artifact_path="experiment")
        return run.info.run_id


def _wandb(result: Any, **kw: Any) -> str:
    import wandb

    run = wandb.init(name=result.config.get("name") or None, config=flat_params(result), reinit=True, **kw)
    try:
        run.log(flat_metrics(result))
        run.summary["problem_fingerprint"] = result.problem_fingerprint
        with tempfile.TemporaryDirectory() as d:
            result.save(d)
            art = wandb.Artifact(name=f"experiment-{run.id}", type="experiment")
            art.add_dir(d)
            run.log_artifact(art)
        return run.id
    finally:
        run.finish()


TRACKERS: Dict[str, Callable[..., str]] = {"mlflow": _mlflow, "wandb": _wandb}


def log_result(result: Any, tracker: Union[str, Callable[[Any], Any]], **kw: Any) -> Any:
    """Send ``result`` to ``tracker``; returns the tracker's run id (or what the callable returns)."""
    if callable(tracker):
        return tracker(result, **kw)
    if tracker not in TRACKERS:
        raise ValueError(f"unknown tracker {tracker!r}; use one of {sorted(TRACKERS)} or a callable")
    try:
        return TRACKERS[tracker](result, **kw)
    except ImportError as e:
        raise ImportError(f"tracker {tracker!r} needs the {tracker} package: pip install {tracker}") from e

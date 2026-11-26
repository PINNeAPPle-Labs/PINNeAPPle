"""Saving a flow run so it can be reproduced, and the reproducer that re-runs it (#1480).

A flow run leaves the flow document, what every node reported and the machine it ran on. Kept in one directory,
those three files are a record somebody can hand back six months later:

    >>> import pinneapple_flow as pf
    >>> result = pf.run(flow, workdir="runs/heat")             # doctest: +SKIP
    >>> pf.reproducibility.save_run("runs/heat", flow, result)  # doctest: +SKIP
    >>> import pinneapple as pp
    >>> pp.experiments.reproduce("runs/heat").outcome           # doctest: +SKIP
    'reproduced'

The reproducer does not define "reproduced": it re-runs the document with the cache off, reports the metrics and the
environment, and :mod:`pinneapple_physics.experiments` compares them with the record. A flow that cannot run any
more — a dataset that is gone, a tool that is no longer installed — is ``refused`` with the node's own sentence,
which names what is missing.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

__all__ = ["detect", "prepare", "record_paths", "save_run", "workflow_reproducer"]

#: the files a recorded run is made of, and what each one is for
FILES = {"flow": "flow.json", "run": "run.json", "environment": "environment.json"}


def record_paths(directory: str | os.PathLike[str]) -> dict[str, Path]:
    root = Path(directory)
    return {name: root / fname for name, fname in FILES.items()}


def _metrics_of(record: dict[str, Any]) -> dict[str, float]:
    """``{node}.{metric}`` for every number a node reported: one name per compared number, node included so two
    nodes cannot silence each other by reporting the same metric name."""
    out: dict[str, float] = {}
    for node, state in (record.get("nodes") or {}).items():
        for name, value in ((state or {}).get("report") or {}).get("metrics", {}).items():
            try:
                out[f"{node}.{name}"] = float(value)
            except (TypeError, ValueError):
                continue
    return out


def save_run(directory: str | os.PathLike[str], flow: Any, result: Any) -> dict[str, str]:
    """Write ``flow.json``, ``run.json`` and ``environment.json`` for a finished run and return their paths.

    The environment is written here rather than left implicit: without it a re-run cannot say whether the versions
    moved, and "it produced the same number" would be the only thing known."""
    from pinneapple_physics.experiments import environment

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = record_paths(directory)
    document = flow.to_dict() if hasattr(flow, "to_dict") else dict(flow)
    payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    paths["flow"].write_text(json.dumps(document, indent=1, default=str))
    paths["run"].write_text(json.dumps(payload, indent=1, default=str))
    paths["environment"].write_text(json.dumps(environment(), indent=1, default=str))
    return {name: str(path) for name, path in paths.items()}


def detect(directory: Path) -> bool:
    """A recorded flow run: the document and what it reported. The environment file is optional, and its absence is
    reported instead of being taken as agreement."""
    paths = record_paths(directory)
    return paths["flow"].is_file() and paths["run"].is_file()


def prepare(directory: Path):
    from pinneapple_physics.experiments import Measurement, Refused, environment

    paths = record_paths(directory)
    record = json.loads(paths["run"].read_text())
    document = json.loads(paths["flow"].read_text())
    recorded_environment = (json.loads(paths["environment"].read_text())
                            if paths["environment"].is_file() else {})

    def run() -> Measurement:
        from .runtime import run as run_flow

        workdir = tempfile.mkdtemp(prefix="reproduce-flow-")
        result = run_flow(document, workdir=workdir, cache=False)
        if result.status != "done":
            failed = {n: (s.get("error") or "") for n, s in result.nodes.items() if s.get("status") == "failed"}
            detail = "; ".join(f"{n}: {e}" for n, e in failed.items()) or f"status {result.status}"
            raise Refused(f"the flow does not run any more — {detail}")
        return Measurement(metrics=_metrics_of(result.to_dict()), environment=environment())

    return {"id": str(record.get("run_id") or directory.name), "metrics": _metrics_of(record),
            "environment": recorded_environment, "tolerances": dict(record.get("tolerances") or {}),
            "inputs": {}, "recorded_at": ""}, run


def workflow_reproducer():
    """The workflow kind, ready to register with :mod:`pinneapple_physics.experiments`."""
    from pinneapple_physics.experiments import Reproducer

    return Reproducer("workflow", detect, prepare)


def _register() -> None:
    from pinneapple_physics.experiments import register_reproducer
    try:
        register_reproducer(workflow_reproducer())
    except ValueError:                        # registered already (imported twice): the same one, nothing to do
        pass


_register()

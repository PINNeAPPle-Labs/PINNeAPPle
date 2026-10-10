"""Run experiments: one run, or a sweep over a grid or random samples, optionally in parallel."""
from __future__ import annotations

import itertools
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from .context import RunContext, _jsonable
from .spec import get
from .store import LabStore


@dataclass
class RunResult:
    run_id: str
    experiment: str
    status: str                      # completed | failed_validation | failed | cached:<status>
    dir: str
    metrics: dict[str, Any]
    seconds: float
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status.split(":")[-1] == "completed"


def _git() -> str | None:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(["git", "-C", here, "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
                                       timeout=5).decode().strip()
    except Exception:
        return None


def _env() -> dict[str, Any]:
    vers = {}
    for m in ("numpy", "torch", "scipy"):
        mod = sys.modules.get(m)
        if mod is not None:
            vers[m] = getattr(mod, "__version__", "?")
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": vers}


def run(experiment, params: dict[str, Any] | None = None, *, root: str | None = None, force: bool = False,
        tags: list[str] | None = None, raise_errors: bool = False) -> RunResult:
    """Run one experiment (name, ``module:Class`` or class) with ``params`` over its defaults.

    Runs are identified by experiment, version and parameters; a run already completed (or failed validation)
    is not repeated unless ``force``. A crashed run is retried on the next call.
    """
    cls = get(experiment) if isinstance(experiment, str) else experiment
    p = cls.resolve_params(params)
    rid = cls.run_id(p)
    store = LabStore(root)
    d = store.run_dir(cls.name, rid)
    rj = os.path.join(d, "run.json")
    if os.path.exists(rj) and not force:
        prev = json.load(open(rj))
        if prev.get("status") in ("completed", "failed_validation"):
            mets = json.load(open(os.path.join(d, "metrics.json"))) if os.path.exists(os.path.join(d, "metrics.json")) else {}
            return RunResult(rid, cls.name, "cached:" + prev["status"], d, mets, 0.0)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    record = {"run_id": rid, "experiment": cls.name, "version": cls.version, "params": _jsonable(p),
              "status": "running", "started": time.time(), "tags": list(tags or []) + list(cls.tags),
              "git": _git(), "environment": _env(), "description": cls.description}
    with open(rj, "w") as f:
        json.dump(record, f, indent=1)
    store.index_run(record, {}, {})
    ctx = RunContext(d, p, cls.name, rid)
    error = None
    try:
        cls().run(ctx)
    except Exception as exc:                                      # noqa: BLE001 - recorded, not swallowed silently
        error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        ctx.log(error)
    finally:
        ctx.close()
    failed = [c for c in ctx.checks if not c["passed"]]
    status = "failed" if error else ("failed_validation" if failed else "completed")
    record.update({"status": status, "finished": time.time(), "error": error, "stages": ctx.stages,
                   "files": ctx.files, "validation": {"total": len(ctx.checks), "failed": len(failed)}})
    record["seconds"] = round(record["finished"] - record["started"], 3)
    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump(ctx.metrics, f, indent=1)
    with open(os.path.join(d, "validation.json"), "w") as f:
        json.dump(ctx.checks, f, indent=1)
    cards = {}
    for name, ds in ctx.datasets.items():
        card = ds.card()
        card["path"] = os.path.relpath(ds.folder, store.root)
        cards[name] = card
    record["datasets"] = {k: {"n_samples": v["n_samples"], "path": v["path"]} for k, v in cards.items()}
    with open(rj, "w") as f:
        json.dump(record, f, indent=1)
    store.index_run(record, ctx.metrics, cards)
    if error and raise_errors:
        raise RuntimeError(f"{rid} failed:\n{error}")
    return RunResult(rid, cls.name, status, d, ctx.metrics, record["seconds"], error)


# ---------------------------------------------------------------------- sweeps
def _grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*[grid[k] for k in keys])]


def _sample(space: dict[str, Any], n: int, seed: int, method: str = "lhs") -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    keys = list(space)
    u = np.empty((n, len(keys)))
    for j in range(len(keys)):
        if method == "lhs":
            u[:, j] = (rng.permutation(n) + rng.uniform(size=n)) / n
        else:
            u[:, j] = rng.uniform(size=n)
    out = []
    for i in range(n):
        p = {}
        for j, k in enumerate(keys):
            s = space[k]
            if isinstance(s, list):
                p[k] = s[min(int(u[i, j] * len(s)), len(s) - 1)]
            elif isinstance(s, tuple) and len(s) == 3 and s[0] == "log":
                p[k] = float(math.exp(math.log(s[1]) + u[i, j] * (math.log(s[2]) - math.log(s[1]))))
            else:
                lo, hi = s
                v = lo + u[i, j] * (hi - lo)
                p[k] = int(round(v)) if isinstance(lo, int) and isinstance(hi, int) else float(v)
        out.append(p)
    return out


def _run_one(args):
    name, params, root, force = args
    r = run(name, params, root=root, force=force)
    return r


def sweep(experiment: str, *, grid: dict[str, list[Any]] | None = None, samples: int | None = None,
          space: dict[str, Any] | None = None, fixed: dict[str, Any] | None = None, method: str = "lhs",
          seed: int = 0, n_jobs: int = 1, root: str | None = None, force: bool = False) -> list[RunResult]:
    """Run an experiment over a parameter ``grid`` (dict of lists) and/or ``samples`` random draws (Latin
    hypercube by default) from ``space`` (default: the experiment's ``space``). ``fixed`` overrides apply to
    every run. ``n_jobs > 1`` runs in separate processes (the experiment must be importable by name)."""
    cls = get(experiment)
    plist: list[dict[str, Any]] = []
    if grid:
        plist += _grid(grid)
    if samples:
        plist += _sample(space or cls.space, samples, seed, method)
    if not plist:
        plist = [{}]
    plist = [{**p, **(fixed or {})} for p in plist]
    jobs = [(cls.name, p, root, force) for p in plist]
    if n_jobs <= 1:
        return [_run_one(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        return list(ex.map(_run_one, jobs))

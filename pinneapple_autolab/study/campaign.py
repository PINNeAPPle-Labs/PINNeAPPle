"""A design of experiments run on any :class:`~pinneapple_autolab.simulator.Simulator`, resumable and costed.

    space = space_from_variables(VARIABLES)
    plan = plan_doe(space, 48, method="maximin", seed=0)            # cases + space-filling metrics
    recs = run_campaign(sim, plan["cases"], "runs/doe", purpose="doe")   # skips the ids already done
    table = cost_table(load_records("runs/doe"))                  # CPU-hours per purpose

Each run writes ``<folder>/results/<id>.json`` (a :class:`~pinneapple_autolab.ledger.RunRecord`) and its fields to
``<folder>/fields/<id>.npz``; a restart (a killed container, a second worker) skips every id with a result file,
failed runs included (pass ``retry_failed=True`` to run those again). Only these files are evidence: a value that is
not in a record was not measured.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from ..ledger import RunRecord, environment

__all__ = ["space_from_variables", "plan_doe", "run_campaign", "load_records", "cost_table", "records_table"]


def space_from_variables(variables: Sequence[Any], name: str = "design space"):
    """A :class:`pinneapple_data.contracts.ParameterSpace` from design :class:`Variable` objects."""
    from pinneapple_data.contracts import Parameter, ParameterSpace
    return ParameterSpace(name=name, parameters=[Parameter(name=v.name, low=float(v.lower), high=float(v.upper),
                                                           unit=getattr(v, "unit", ""),
                                                           description=getattr(v, "description", ""))
                                                 for v in variables])


def plan_doe(space, n: int, method: str = "maximin", *, seed: int = 0, id_prefix: str = "doe",
             existing=None) -> dict[str, Any]:
    """``n`` space-filling cases (:func:`pinneapple_data.doe.design`) with their coverage metrics (centred L2
    discrepancy, minimum distance; :func:`pinneapple_data.doe.coverage`) and those of a plain random design of the
    same size as the yardstick (median of 20 draws)."""
    from pinneapple_data.doe import coverage, design
    d = design(space, n, method, seed=seed, existing=existing)
    cov = coverage(d.unit, space.names)
    rnd = [coverage(np.random.default_rng(seed + k).random(d.unit.shape), space.names) for k in range(20)]
    cases = [{"id": f"{id_prefix}{i:03d}", **{k: float(v) for k, v in row.items()}}
             for i, row in enumerate(d.cases[space.names].to_dict("records"))]
    keys = [k for k in ("discrepancy", "min_distance", "fill_distance") if k in cov]
    return {"cases": cases, "unit": d.unit, "method": method, "seed": seed, "coverage": cov,
            "random_reference": {k: float(np.median([r[k] for r in rnd])) for k in keys}, "info": d.info}


def _path(folder: str, kind: str, rid: str, ext: str) -> str:
    return os.path.join(folder, kind, f"{rid}.{ext}")


def run_campaign(sim, cases: Iterable[Mapping[str, Any]], folder: str, *, purpose: str = "doe", reason: str = "",
                 retry_failed: bool = False, log: Callable[[str], None] = print,
                 predicted: Mapping[str, Mapping[str, Any]] | None = None) -> list[RunRecord]:
    """Run every case (``{"id": ..., **params}``) not already done; returns the records of all ``cases`` (new and
    old). A case whose ``sim.check`` finds issues is recorded as failed with them and never simulated."""
    for k in ("results", "fields", "cases"):
        os.makedirs(os.path.join(folder, k), exist_ok=True)
    env_p = os.path.join(folder, "environment.json")
    if not os.path.exists(env_p):
        json.dump(environment(), open(env_p, "w"), indent=1, default=str)
    out = []
    for case in cases:
        rid = str(case["id"])
        params = {k: float(v) for k, v in case.items() if k != "id"}
        rp = _path(folder, "results", rid, "json")
        if os.path.exists(rp):
            rec = RunRecord(**json.load(open(rp)))
            if rec.ok or not retry_failed:
                out.append(rec)
                continue
        issues = sim.check(params)
        started = time.strftime("%Y-%m-%dT%H:%M:%S")
        if issues:
            rec = RunRecord(rid, purpose, params, False, message="; ".join(issues), started=started, reason=reason)
        else:
            log(f"[{rid}] {purpose}: {params}")
            o = sim.run(params, os.path.join(folder, "cases", rid))
            fp = ""
            if o.ok and o.fields:
                fp = _path(folder, "fields", rid, "npz")
                np.savez_compressed(fp, **{k: np.asarray(v) for k, v in o.fields.items()})
            rec = RunRecord(rid, purpose, params, bool(o.ok), dict(o.outputs), o.message, float(o.cost_s), started,
                            reason, dict((predicted or {}).get(rid, {})), _jsonable(o.meta), fp)
            log(f"[{rid}] {'ok' if o.ok else 'FAILED'} in {o.cost_s:.0f} s" + ("" if o.ok else f": {o.message}"))
        json.dump(rec.to_dict(), open(rp + ".tmp", "w"), indent=1)
        os.replace(rp + ".tmp", rp)                               # a killed run never leaves half a result
        out.append(rec)
    return out


def load_records(folder: str, purposes: Sequence[str] | None = None, ok_only: bool = False) -> list[RunRecord]:
    """Every record in ``<folder>/results`` (sorted by id), optionally of some purposes / only successful."""
    d = os.path.join(folder, "results")
    recs = [RunRecord(**json.load(open(os.path.join(d, f)))) for f in sorted(os.listdir(d)) if f.endswith(".json")] \
        if os.path.isdir(d) else []
    return [r for r in recs if (purposes is None or r.purpose in purposes) and (r.ok or not ok_only)]


def records_table(records: Sequence[RunRecord], inputs: Sequence[str], outputs: Sequence[str]):
    """``(ids, X, Y)`` of the successful records: inputs and outputs as arrays in the given column order."""
    ok = [r for r in records if r.ok]
    X = np.array([[r.params[k] for k in inputs] for r in ok], float).reshape(len(ok), len(inputs))
    Y = np.array([[r.outputs[k] for k in outputs] for r in ok], float).reshape(len(ok), len(outputs))
    return [r.id for r in ok], X, Y


def cost_table(records: Sequence[RunRecord], extra: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Wall time and CPU-hours (wall x processes) per purpose, with the run count and the failures; ``extra``:
    seconds of other stages measured elsewhere (training, rendering) on one core."""
    rows: dict[str, dict[str, float]] = {}
    for r in records:
        p = rows.setdefault(r.purpose, {"runs": 0, "failed": 0, "wall_h": 0.0, "cpu_h": 0.0})
        p["runs"] += 1
        p["failed"] += 0 if r.ok else 1
        p["wall_h"] += r.cost_s / 3600
        p["cpu_h"] += r.cost_s * float(r.meta.get("procs", 1)) / 3600
    for k, s in (extra or {}).items():
        rows[k] = {"runs": 0, "failed": 0, "wall_h": float(s) / 3600, "cpu_h": float(s) / 3600}
    tot = {k: float(sum(v[k] for v in rows.values())) for k in ("runs", "failed", "wall_h", "cpu_h")}
    return {"stages": rows, "total": tot}


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x

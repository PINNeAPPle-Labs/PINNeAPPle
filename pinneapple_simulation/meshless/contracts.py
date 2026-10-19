"""A meshless study as the apps' ``StudyResult`` contract (``pinneapple_data.contracts``): one table row per method
and spacing, the convergence and the winners as metrics, and checks that carry the value they were decided on."""
from __future__ import annotations

import math
from typing import Any

__all__ = ["study_result", "meshless_checks"]


def meshless_checks(res) -> list:
    """Observed order against the expected one where the solution is smooth enough to expect it, and stability."""
    from pinneapple_data.contracts import CheckRecord
    out = []
    expected = {"fd": 2.0, "fem": 2.0}
    for m, c in res.convergence.items():
        if c.get("order") is None:
            continue
        if m in expected:
            ok = c["order"] >= expected[m] - 0.3
            out.append(CheckRecord(name=f"{m}: observed order near the theoretical {expected[m]:g}",
                                   status="PASS" if ok else "WARN", value=float(c["order"]), threshold=expected[m] - 0.3,
                                   detail="below it: a singularity, an unresolved feature or a reference floor"))
    bad = [f"{r['method']} h={r['h']:g}" for r in res.rows if not r.get("stable")]
    out.append(CheckRecord(name="every method produced a finite solution", status="WARN" if bad else "PASS",
                           value=len(bad), threshold=0, detail=", ".join(bad)))
    return out


def study_result(res, *, study_id: str, title: str, limits=(), app: str = "meshless"):
    from pinneapple_data.contracts import Provenance, StudyResult, Table
    keys = ("method", "h", "n_nodes", "error", "error_max", "setup_s", "solve_s", "cond", "noise_amplification")
    cols: dict[str, list[Any]] = {k: [] for k in keys}
    for r in res.rows:
        for k in keys:
            v = r.get(k)
            cols[k].append(None if isinstance(v, float) and not math.isfinite(v) else v)
    metrics: dict[str, Any] = {"problem": res.problem["kind"], "domain": res.domain["name"],
                               "reference": res.reference["kind"], "wall_s": res.wall_s}
    for m, c in res.convergence.items():
        if c.get("order") is not None:
            metrics[f"order_{m}"] = float(c["order"])
        metrics[f"finest_error_{m}"] = float(c["finest_error"])
    for w in res.wins:
        if w.get("method"):
            metrics["winner: " + w["criterion"]] = w["method"]
    return StudyResult(id=study_id, study_type="benchmark", app=app, title=title, status="done",
                       table=Table(columns=cols), metrics=metrics, checks=meshless_checks(res), limits=list(limits),
                       provenance=Provenance(app=app, study_id=study_id, source="meshless_study"))

"""The verdict rule of the pre-registration (docs/research/cfd_direct_from_stl.md), as code: gate R, the reference
operating point, the candidate operating point, the speed-up and the verdict. Nothing here is tuned after the runs:
the margins, gates and thresholds are the pre-registered numbers.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["MARGINS", "GATES", "QUANTITIES", "profile_errors", "operating_points", "verdict"]

# fixed margins (absolute, on the relative error against the published value)
MARGINS = {"Cd": 0.02, "dp": 0.02, "Cl": 0.10, "x1": 0.05, "profile": 0.03}
QUANTITIES = {"C1": ("Cd", "Cl", "dp"), "C2": ("Cd",), "C3": ("x1",)}


def GATES(case: str, re: float | None = None) -> dict[str, float]:  # noqa: N802 - a table, read like a constant
    """Gate R: the largest error of the reference against the published value, per quantity."""
    if case == "C1":
        return {"Cd": 0.01, "dp": 0.01, "Cl": 0.05}
    if case == "C2":
        return {"Cd": 0.04}
    if case == "C3":
        return {"x1": 0.05 if (re or 0) < 300 else 0.10}
    raise KeyError(case)


def _flat(profiles: dict[str, list]) -> np.ndarray:
    return np.concatenate([np.asarray(v, float).ravel() for _, v in sorted(profiles.items())])


def profile_errors(runs: list[dict[str, Any]]) -> None:
    """Relative L2 error of each run's velocity profiles against the finest reference run of the same case and Re
    (written into each run as ``err_profile``)."""
    refs = [r for r in runs if r["method"] == "reference" and r.get("ok", True)]
    if not refs:
        return
    fine = max(refs, key=lambda r: r["n"])
    u0 = _flat(fine["profiles"])
    for r in runs:
        if not r.get("ok", True) or "profiles" not in r:
            continue
        u = _flat(r["profiles"])
        m = np.isfinite(u) & np.isfinite(u0)
        r["err_profile"] = float(np.linalg.norm(u[m] - u0[m]) / np.linalg.norm(u0[m])) if m.any() else math.nan
        r["profile_reference_n"] = fine["n"]


def _errs(run: dict[str, Any], case: str) -> dict[str, float]:
    out = {q: run.get(f"err_{q}", math.nan) for q in QUANTITIES[case]}
    out["profile"] = run.get("err_profile", math.nan)
    return out


def operating_points(runs: list[dict[str, Any]], case: str, re: float | None = None) -> dict[str, Any]:
    """Gate R, R*, and X* for every candidate in ``runs`` (all of one case and Re)."""
    gate = GATES(case, re)
    ok = [r for r in runs if r.get("ok", True)]
    refs = sorted([r for r in ok if r["method"] == "reference"], key=lambda r: r["times"]["total_s"])
    passing = [r for r in refs if all(r.get(f"err_{q}", math.inf) <= g for q, g in gate.items())]
    out: dict[str, Any] = {"gate": gate, "gate_passed": bool(passing)}
    if not refs:
        out["reference"] = None
        return out
    rstar = passing[0] if passing else max(refs, key=lambda r: r["n"])
    out["reference"] = rstar
    e_r = _errs(rstar, case)
    limits = {q: (e_r[q] if math.isfinite(e_r[q]) else 0.0) + MARGINS[q] for q in e_r}
    out["limits"] = limits
    for method in sorted({r["method"] for r in ok} - {"reference"}):
        cands = sorted([r for r in ok if r["method"] == method], key=lambda r: r["times"]["total_s"])
        meet = [r for r in cands if all(_errs(r, case)[q] <= lim for q, lim in limits.items())]
        out[method] = {"star": meet[0] if meet else None, "runs": cands}
    return out


def verdict(op: dict[str, Any], method: str) -> dict[str, Any]:
    """validated (X* exists and S >= 2), partial (X* exists, 1 <= S < 2), refuted (no X*, or S < 1), not tested."""
    rstar = op.get("reference")
    info = op.get(method)
    if rstar is None or info is None or not info["runs"]:
        return {"verdict": "not tested"}
    x = info["star"]
    if x is None:
        return {"verdict": "refuted", "why": "no rung within the margins of R*"}
    s = rstar["times"]["total_s"] / x["times"]["total_s"]
    s_solve = rstar["times"]["solve_s"] / max(x["times"]["solve_s"], 1e-300)
    v = "validated" if s >= 2 else ("partial" if s >= 1 else "refuted")
    return {"verdict": v, "speedup": s, "speedup_solve": s_solve, "n": x["n"], "n_reference": rstar["n"],
            "gate_passed": op["gate_passed"]}

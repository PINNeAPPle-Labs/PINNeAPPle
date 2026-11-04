"""Convergence assessment from residual histories: a verdict, the reasons and what to do next.

Statuses
--------
converged       residual targets met (or the solver says so), or 3+ orders of drop and flat at the end
still_falling   stopped at the end time/iteration limit while residuals were still dropping: run longer
stalled         residuals flat (or oscillating) above the target: the solution will not improve by iterating
diverged        residuals grew by orders of magnitude, or the run crashed after residuals rose
crashed         the solver stopped with an error before converging
incomplete      the log ends without an end-of-run marker (interrupted, or still running)
completed       transient run that reached its end time (with quality notes on Courant, linear solvers, continuity)
failed          FEA increment did not converge (cutbacks exhausted / error)
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

import numpy as np

from pinneapple_core.series import downsample_indices

LABEL = {"converged": "Converged", "still_falling": "Not converged: residuals still falling", "stalled": "Stalled",
         "diverged": "Diverged", "crashed": "Crashed", "incomplete": "Incomplete log", "completed": "Completed",
         "failed": "Failed to converge", "unknown": "No residual history"}
LEVEL = {"converged": "ok", "completed": "ok", "still_falling": "warn", "stalled": "warn", "incomplete": "warn",
         "diverged": "bad", "crashed": "bad", "failed": "bad", "unknown": "warn"}


def _target(field: str, rc: Optional[Dict[str, Any]]) -> Optional[float]:
    if not rc:
        return None
    for k, v in rc.items():
        pat = str(k).strip('"')
        try:
            if pat == field or re.fullmatch(pat, field):
                return float(str(v).split()[-1])
        except (re.error, ValueError):
            continue
    return None


def _slope(y: np.ndarray, x: np.ndarray) -> float:
    """d log10(residual) per 100 iterations over the given window (robust linear fit)."""
    m = np.isfinite(y) & (y > 0)
    if m.sum() < 5:
        return float("nan")
    return float(np.polyfit(x[m], np.log10(y[m]), 1)[0] * 100)


def downsample(x: List[float], series: Dict[str, List[float]], k: int = 1500) -> Dict[str, Any]:
    idx = downsample_indices(len(x), k, "even")
    clean = lambda v: None if v is None or not np.isfinite(v) else float(v)   # noqa: E731
    return {"x": [float(x[i]) for i in idx], "fields": {f: [clean(v[i]) for i in idx] for f, v in series.items()}}


def assess_steady(log: Dict[str, Any], targets: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    it = np.arange(1, len(log["time_steps"]) + 1, dtype=float)
    res = {f: np.array(v, dtype=float) for f, v in log["residuals"].items()}
    out: Dict[str, Any] = {"iterations": int(len(it)), "fields": {}}
    reasons: List[str] = []
    recs: List[str] = []
    if not len(it) or not res:
        return {**out, "status": "unknown", "reasons": ["No residuals found in the log."], "recommendations": []}
    met_all, any_target = True, False
    growth = False
    tail = max(20, len(it) // 3)
    for f, r in res.items():
        fin = r[np.isfinite(r)]
        if not len(fin):
            continue
        first, last, rmin = float(fin[0]), float(fin[-1]), float(fin.min())
        tgt = _target(f, targets)
        drop = math.log10(max(fin[: max(1, min(10, len(fin)))].max(), 1e-300) / max(last, 1e-300))
        sl = _slope(r[-tail:], it[-tail:])
        info = {"first": first, "final": last, "min": rmin, "orders_dropped": drop, "slope_per_100_it": sl,
                "target": tgt, "met": (last <= tgt) if tgt else None}
        if tgt:
            any_target = True
            met_all &= last <= tgt
        if last > 100 * rmin and last > 1e-3:
            growth = True
        out["fields"][f] = info
    out["final"] = {f: v["final"] for f, v in out["fields"].items()}
    slopes = [v["slope_per_100_it"] for v in out["fields"].values() if np.isfinite(v["slope_per_100_it"])]
    worst = max(out["fields"].items(), key=lambda kv: (kv[1]["final"] / kv[1]["target"]) if kv[1]["target"]
                else kv[1]["final"])
    if log.get("converged_message"):
        status = "converged"
        reasons.append(f"The solver reports: “{log['converged_message']}”.")
        if any_target:
            reasons.append("Every field's initial residual is below its residualControl target.")
    elif log.get("floating_point_exception") or log.get("fatal"):
        status = "diverged" if growth or log.get("floating_point_exception") else "crashed"
        reasons.append("The run stopped with " + ("a floating-point exception (a NaN/Inf in the solution)"
                       if log.get("floating_point_exception") else f"an error: {log['fatal'][:160]}")
                       + f" after {len(it)} iteration{'s' if len(it) != 1 else ''}.")
        if growth:
            reasons.append(f"Residuals rose before the crash ('{worst[0]}' ended at {worst[1]['final']:.2g}, "
                           f"{worst[1]['final'] / max(worst[1]['min'], 1e-300):.0f}× its minimum).")
        recs += ["Lower the under-relaxation factors (e.g. p 0.3, U 0.7) or use SIMPLEC with consistent formulation.",
                 "Start from a converged laminar or first-order solution, then switch to second-order convection.",
                 "Check the mesh (checkMesh: non-orthogonality, skewness) and the boundary conditions at the crash location."]
    elif not log.get("ended"):
        status = "incomplete"
        reasons.append("The log stops without OpenFOAM's 'End' line: the run was interrupted or is still running.")
    elif growth:
        status = "diverged"
        reasons.append(f"'{worst[0]}' grew to {worst[1]['final']:.2g}, {worst[1]['final'] / worst[1]['min']:.0f}× its minimum.")
        recs.append("Reduce under-relaxation and check boundary conditions and mesh quality.")
    elif any_target and met_all:
        status = "converged"
        reasons.append("Every field's initial residual is below its residualControl target.")
    else:
        judged = [v["slope_per_100_it"] for v in out["fields"].values()
                  if np.isfinite(v["slope_per_100_it"]) and (not any_target or (v["target"] and not v["met"]))]
        falling = bool(judged) and max(judged) < -0.05     # the fields that still have to fall are falling
        if any_target:
            unmet = [f for f, v in out["fields"].items() if v["target"] and not v["met"]]
            reasons.append(f"The run reached its end after {len(it)} iterations with "
                           + ", ".join(f"'{f}' at {out['fields'][f]['final']:.2g} (target {out['fields'][f]['target']:.0e})" for f in unmet)
                           + ".")
        else:
            reasons.append(f"No residualControl targets are set; residuals dropped {min(v['orders_dropped'] for v in out['fields'].values()):.1f}–"
                           f"{max(v['orders_dropped'] for v in out['fields'].values()):.1f} orders of magnitude.")
        if not any_target and all(v["orders_dropped"] >= 3 for v in out["fields"].values()) and \
                (not slopes or max(abs(s) for s in slopes) < 0.05):
            status = "converged"
            reasons.append("All residuals fell 3+ orders and are flat at the end.")
        elif falling:
            status = "still_falling"
            need = []
            for f, v in out["fields"].items():
                if v["target"] and not v["met"] and v["slope_per_100_it"] < 0:
                    need.append((math.log10(v["target"] / v["final"]) / v["slope_per_100_it"]) * 100)
            reasons.append("Residuals are still decreasing at the end of the run "
                           f"(slowest field still above target: {max(judged):+.2f} decades per 100 iterations).")
            if need:
                n_more = int(max(need) * 1.2) + 1
                out["estimated_more_iterations"] = n_more
                recs.append(f"Run at least {n_more} more iterations (straight-line extrapolation of the last {tail}; "
                            "residual decay usually slows down, so plan for more), or restart from the last written time.")
            else:
                recs.append("Run more iterations, or set residualControl targets so the solver stops when converged.")
        else:
            status = "stalled"
            reasons.append("Residuals are flat (or oscillating) above the target: more iterations will not help.")
            recs += ["Look for unsteadiness (vortex shedding): a transient solver may be the right model.",
                     "Check mesh quality near the largest residuals and the outlet (backflow); try lower relaxation."]
    if log.get("bounding_warnings"):
        reasons.append(f"{log['bounding_warnings']} 'bounding' messages: turbulence quantities went negative and were "
                       "clipped, a sign of stiffness early in the run.")
    return {**out, "status": status, "reasons": reasons, "recommendations": recs}


def assess_transient(log: Dict[str, Any], tolerances: Dict[str, Optional[float]], max_co: Optional[float],
                     algorithm: Optional[str], reached_end: Optional[bool]) -> Dict[str, Any]:
    steps = log["time_steps"]
    out: Dict[str, Any] = {"time_steps": len(steps), "fields": {}}
    reasons: List[str] = []
    recs: List[str] = []
    if log.get("floating_point_exception") or log.get("fatal"):
        status = "crashed"
        reasons.append("The run stopped with " + ("a floating-point exception" if log.get("floating_point_exception")
                                                   else f"an error: {log['fatal'][:160]}") + f" at t = {steps[-1] if steps else '?'}.")
        recs.append("Reduce the time step (lower maxCo) and check boundary conditions.")
    elif not log.get("ended"):
        status = "incomplete"
        reasons.append("The log stops without 'End': interrupted or still running.")
    else:
        status = "completed"
        reasons.append(f"The run reached its end time after {len(steps)} time steps.")
    for f, r in log["residuals"].items():
        r = np.array(r, dtype=float)
        rf = np.array(log["final_residuals"].get(f, []), dtype=float)
        base = re.sub(r"(x|y|z)$", "", f) if len(f) > 1 else f      # Ux, Uy are solved under the 'U' entry
        tol = next((t for k, t in tolerances.items() if t and (k in (f, base) or _safe_match(k, f) or _safe_match(k, base))), None)
        fin = r[np.isfinite(r)]
        info = {"max_initial": float(fin.max()) if len(fin) else None,
                "median_initial": float(np.median(fin)) if len(fin) else None,
                "final_initial": float(fin[-1]) if len(fin) else None, "tolerance": tol}
        if tol and len(rf):
            missed = int(np.sum(rf[np.isfinite(rf)] > 10 * tol))
            info["steps_linear_solver_short"] = missed
            if missed > 0.05 * len(rf):
                reasons.append(f"'{f}': the linear solver stopped above 10× its tolerance on {missed} steps "
                               "(maxIter reached): the time steps are not fully converged.")
                recs.append(f"Raise maxIter or use a stronger solver/preconditioner for '{f}'.")
        out["fields"][f] = info
    co = np.array(log.get("courant_max") or [], dtype=float)
    if len(co):
        out["courant"] = {"max": float(co.max()), "mean_of_max": float(co.mean())}
        limit = max_co or (1.0 if algorithm in ("PISO", None) else None)
        if limit and co.max() > limit * 1.05:
            reasons.append(f"Max Courant number reached {co.max():.2f} (limit {limit:g}).")
            recs.append("Reduce deltaT or enable adjustTimeStep with maxCo.")
        else:
            reasons.append(f"Max Courant number {co.max():.2f}" + (f" stays within {limit:g}." if limit else "."))
    cont = log.get("continuity") or []
    if cont:
        local = np.array([c[0] for c in cont])
        out["continuity"] = {"max_local": float(local.max()), "final_cumulative": cont[-1][2]}
        if local.max() > 1e-3:
            reasons.append(f"Continuity errors reached {local.max():.1e} (local sum): mass is not well conserved.")
            recs.append("Add pressure correctors (nCorrectors) or tighten the pressure tolerance.")
    if reached_end is False and status == "completed":
        status = "incomplete"
    return {**out, "status": status, "reasons": reasons, "recommendations": recs}


def _safe_match(pat: str, f: str) -> bool:
    try:
        return bool(re.fullmatch(pat.strip('"'), f))
    except re.error:
        return False


def assess_table(iters: List[float], residuals: Dict[str, List[float]], is_log10: bool) -> Dict[str, Any]:
    """Generic residual history (Fluent / STAR-CCM+ / SU2 exports)."""
    res = {f: (10.0 ** np.array(v, dtype=float) if is_log10 else np.array(v, dtype=float)) for f, v in residuals.items()}
    log = {"time_steps": iters, "residuals": {f: r.tolist() for f, r in res.items()}, "ended": True}
    out = assess_steady(log, None)
    out["reasons"].insert(0, "Assessed from the residual history only (no solver log or settings).")
    return out

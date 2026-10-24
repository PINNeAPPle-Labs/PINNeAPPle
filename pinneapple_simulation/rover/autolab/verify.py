"""Trust report of a rover mission: did it work, does the physics hold, and can the learned soil be believed.

Checks (``pinneapple_analysis.trust.Check``; values are measured in this mission's simulations):

* ``goal_reached`` (critical) and ``slip_within_limit`` (critical): the final run reached the goal with the 2-s mean
  front-wheel slip at or below the mission limit everywhere.
* ``energy_covers_climb``: the measured drive energy of the final run is at least ``m g`` times the total climb along
  it (the wheels do not regenerate and the kinetic energy at walking speed is negligible, so less would break energy
  conservation).
* ``time_step_converged``: the first seconds of the final route rerun at half the time step give the same mean
  slip and sinkage (relative change at most 5 %).
* ``planner_vs_multibody_energy`` / ``planner_vs_multibody_slip``: the planner's quasi-static prediction of the
  final route against the multibody measurement.
* ``replanned_route_risk``: the final route's risk under the learned belief is at most ``risk_max``;
  ``failed_route_risk_learned``: the first route's risk under the learned belief exceeds it (the update explains
  the failure).
* ``learned_soil_predictive``: held-out measurements of the stall unit are predicted within 20 % (sinkage RMSE /
  RMS sinkage).
* ``learned_soil_calibrated`` and ``learned_soil_closer_to_truth``: only on a synthetic site -- the error against the
  known true soil is within 3 posterior standard deviations for every parameter, and smaller than the prior's for
  k_phi and phi.
* ``models_used_in_validity``: the planner used the analytic law; the zoo surrogate's share of operating points
  outside its validity box is reported in ``details``.
"""
from __future__ import annotations

import math

import numpy as np

from pinneapple_analysis.trust import Check, TrustReport
from pinneapple_analysis.trust.trust_report import CONTRADICTS, NOT_RUN, SUPPORTS


def _st(ok):
    return NOT_RUN if ok is None else (SUPPORTS if ok else CONTRADICTS)


def monitor_slip(run, front, window_s: float = 2.0, t_min: float = 2.0):
    """The traction monitor's quantity: front-wheel mean slip averaged over the last ``window_s`` seconds (per output
    sample, once ``window_s`` seconds have been monitored from ``t_min`` on, exactly as
    :func:`~.mission.traction_monitor` decides; nan before)."""
    s = run.wheel["slip"][:, front].mean(1)
    t = run.t
    out = np.full(len(s), np.nan)
    on = np.flatnonzero(t >= t_min)
    if len(on):
        k0 = on[0]
        c = np.cumsum(np.r_[0.0, s[k0:]])
        tt = t[k0:]
        lo = np.searchsorted(tt, tt - window_s + 1e-9, side="right")    # samples with t > t_k - window
        hi = np.arange(1, len(tt) + 1)
        out[k0:] = np.where(tt - tt[0] >= window_s - 1e-9, (c[hi] - c[lo]) / (hi - lo), np.nan)
    return out


def timestep_check(mission, route, seconds: float = 6.0) -> dict:
    from ...vehicle import VehicleSimulator
    out = {}
    for name, dt in (("dt", mission.dt), ("dt/2", mission.dt / 2)):
        ter = mission.site.deformable_terrain(cell=mission.terrain_cell, x_range=(route.waypoints[0][0] - 3,
                                                                                  route.waypoints[0][0] + 5),
                                              y_range=(route.waypoints[0][1] - 3, route.waypoints[0][1] + 3))
        sim = VehicleSimulator(mission.model, ter, mission.site.true_soils(), soil_map=mission.site.unit_at,
                               wheel_speeds="kinematic")
        r = sim.run(seconds, speed=mission.spec.speed, dt=dt, output_dt=0.05, route=route.waypoints,
                    record_stresses=False, settle=1.0, ramp=1.0)
        late = r.t > 0.5 * r.t[-1]
        out[name] = {"slip": float(r.wheel["slip"][late].mean()), "sinkage": float(r.wheel["sinkage"][late].mean()),
                     "speed": float(r.chassis["speed"][late].mean())}
    rel = max(abs(out["dt/2"][k] - out["dt"][k]) / max(abs(out["dt/2"][k]), 1e-6) for k in ("sinkage", "speed"))
    out["slip_abs_change"] = abs(out["dt/2"]["slip"] - out["dt"]["slip"])
    out["max_rel_change"] = float(rel)
    return out


def verify_mission(mission, routes, runs, planners, belief, meas_all, comparison, do_timestep: bool = True):
    sp, site = mission.spec, mission.site
    front = mission.front
    final, first = runs[-1], runs[0]
    checks, det = [], {}

    reached = final.meta["ended"] == "goal"
    checks.append(Check("goal_reached", _st(reached), critical=True, value=float(reached), limit=1.0,
                        source="multibody run of the final route", reason=final.meta["ended"]))
    ms = monitor_slip(final, front)
    checks.append(Check("slip_within_limit", _st(bool(np.nanmax(ms) <= sp.slip_limit)), critical=True,
                        value=float(np.nanmax(ms)), limit=sp.slip_limit,
                        source="2-s mean front-wheel slip after the 2-s start-up, final run"))

    E = float(final.energy()[-1])
    d_run = final.distance()
    s_grid = np.arange(0.0, d_run[-1] + 1e-9, 0.25)
    z = site.height(np.interp(s_grid, d_run, final.q[:, 0]), np.interp(s_grid, d_run, final.q[:, 1]))
    climb = float(np.clip(np.diff(z), 0.0, None).sum())
    pe = mission.mass * mission.site_g * climb
    checks.append(Check("energy_covers_climb", _st(E >= pe), value=E, limit=pe,
                        source="drive energy (sum of max(tau omega, 0)) vs m g x total climb of the final run (no "
                               "regeneration; kinetic energy at 0.2 m/s is negligible)"))
    det["energy_balance"] = {"drive_energy_J": E, "climb_potential_J": pe, "total_climb_m": climb}

    if do_timestep:
        ts = timestep_check(mission, routes[-1])
        det["time_step"] = ts
        checks.append(Check("time_step_converged", _st(ts["max_rel_change"] <= 0.05 and ts["slip_abs_change"] <= 0.01),
                            value=ts["max_rel_change"], limit=0.05,
                            source=f"first seconds of the final route at dt = {mission.dt:g} s and dt/2"))
    else:
        checks.append(Check("time_step_converged", NOT_RUN, reason="skipped (fast mode)"))

    # planner (quasi-static, learned belief) vs multibody on the final route
    pl = planners[-1]
    pred = pl.evaluate(final.q[::20, :2] if len(final.q) > 40 else final.q[:, :2])
    d = final.distance()
    rel_E = abs(pred.energy_J - E) / max(E, 1e-9)
    prof = pred.profile
    s_meas = np.interp(prof["s"], d, np.nan_to_num(monitor_slip(final, front), nan=0.0))   # 2-s mean, as monitored
    keep = np.interp(prof["s"], d, final.t) >= 2.0
    slip_rmse = float(np.sqrt(np.mean((prof["slip"][keep] - s_meas[keep]) ** 2)))
    det["planner_vs_multibody"] = {"predicted_energy_J": pred.energy_J, "measured_energy_J": E, "rel_error": rel_E,
                                   "slip_rmse": slip_rmse}
    checks.append(Check("planner_vs_multibody_energy", _st(rel_E <= 0.25), value=rel_E, limit=0.25,
                        source="quasi-static planner (learned belief) vs multibody, final route"))
    checks.append(Check("planner_vs_multibody_slip", _st(slip_rmse <= 0.05), value=slip_rmse, limit=0.05,
                        source="RMS of predicted slip - 2-s mean front-wheel slip along the final route"))

    # risk of both routes under the learned belief
    r_final = pl.evaluate(routes[-1].waypoints)
    checks.append(Check("replanned_route_risk", _st(r_final.max_risk <= sp.risk_max), value=r_final.max_risk,
                        limit=sp.risk_max, source="share of belief samples that exceed the slip limit"))
    if len(routes) > 1:
        r_first = pl.evaluate(routes[0].waypoints)
        det["first_route_under_learned_belief"] = {"max_risk": r_first.max_risk, "max_slip": r_first.max_slip,
                                                   "energy_J": r_first.energy_J}
        checks.append(Check("failed_route_risk_learned", _st(r_first.max_risk > sp.risk_max), value=r_first.max_risk,
                            limit=sp.risk_max, source="first route re-evaluated with the learned belief"))
    else:
        checks.append(Check("failed_route_risk_learned", NOT_RUN, reason="the first route reached the goal"))

    # learned soil
    stall_unit = None
    if len(runs) > 1:
        stall_unit = int(np.bincount(first.wheel["soil_idx"][-1, front]).argmax())
        det["stall_unit"] = site.units[stall_unit].name
        pc = (meas_all[-1] if meas_all else None)
        held = None
        if pc is not None:
            r = np.flatnonzero(pc["unit"] == stall_unit)
            r = r[np.argsort(pc["t"][r], kind="stable")][int(0.75 * len(r)):]
            if len(r):
                held = belief.predictive_check(stall_unit, pc, r)
                rms = float(np.sqrt(np.mean(pc["sinkage"][r] ** 2)))
                held["sinkage_rel"] = held["sinkage_rmse_mm"] / 1e3 / max(rms, 1e-9)
        det["held_out"] = held
        checks.append(Check("learned_soil_predictive", _st(None if held is None else held["sinkage_rel"] <= 0.2),
                            value=None if held is None else held["sinkage_rel"], limit=0.2,
                            source="held-out last quarter of the stall unit's measurements"))
        rec = belief.recovery(stall_unit, site.true_soil(stall_unit))
        det["recovery"] = rec
        zmax = max(abs(v["z"]) for v in rec.values())
        checks.append(Check("learned_soil_calibrated", _st(zmax <= 3.0), value=zmax, limit=3.0,
                            source="known truth of the synthetic site: max |error| / posterior sd"))
        closer = all(abs(rec[p]["error"]) < abs(rec[p]["prior_error"]) for p in ("log10_k_phi", "phi_deg"))
        checks.append(Check("learned_soil_closer_to_truth", _st(closer), source="known truth: |error| vs prior's"))
    else:
        for n in ("learned_soil_predictive", "learned_soil_calibrated", "learned_soil_closer_to_truth"):
            checks.append(Check(n, NOT_RUN, reason="no stall, nothing learned"))

    sur = {n: (c.get("surrogate_zoo") or {}).get("fraction_outside_validity") for n, c in comparison.items()
           if isinstance(c, dict) and "analytic_prior" in c}
    det["surrogate_outside_validity"] = sur
    checks.append(Check("models_used_in_validity", SUPPORTS,
                        source="planner uses the analytic Wong-Reece law; zoo surrogate only compared",
                        reason="; ".join(f"{n}: {v:.0%} of zoo-surrogate points outside its box" for n, v in sur.items()
                                         if v is not None and math.isfinite(v))))
    rep = TrustReport(checks, model="PINNeAPPle Autonomous Physics Lab - rover mission", details=det)
    return rep, {"decision": rep.decisao, "score": rep.score, "coverage": rep.cobertura,
                 "failed": [c.name for c in checks if c.status == CONTRADICTS]}


__all__ = ["verify_mission", "monitor_slip", "timestep_check"]

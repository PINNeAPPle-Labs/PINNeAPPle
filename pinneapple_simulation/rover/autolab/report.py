"""Report of a rover mission: ``summary.json`` (every number), ``trust_report.md``, ``report.md`` and the figures."""
from __future__ import annotations

import json
import os
import platform
import time

import numpy as np

from .mission import to_json
from .verify import monitor_slip


def run_info() -> dict:
    """Versions and hardware of this run."""
    import scipy
    info = {"date_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(), "cpu_count": os.cpu_count()}
    try:
        import torch
        info["torch"] = torch.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import bpy
        info["blender"] = bpy.app.version_string
    except Exception:  # noqa: BLE001
        pass
    try:
        with open("/proc/cpuinfo") as f:
            names = [ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name")]
        if names:
            info["cpu"] = names[0]
    except OSError:
        pass
    return info


def mission_numbers(mission, res) -> dict:
    """The measured and predicted numbers of a mission, labelled by where they come from."""
    site, sp = mission.site, res.spec
    out = {"spec": {k: v for k, v in vars(sp).items()}, "rover": {"model": "NASA VIPER (parametric, public figures)",
                                                                 "mass_kg": mission.mass, "wheels": mission.n_wheels,
                                                                 "wheel_radius_m": mission.wheel.radius,
                                                                 "wheel_width_m": mission.wheel.width}}
    out["routes"] = []
    for i, (r, run) in enumerate(zip(res.routes, res.runs, strict=True)):
        s = run.summary()
        ms = monitor_slip(run, mission.front)
        learned = res.planners[-1].evaluate(r.waypoints)
        out["routes"].append({
            "name": chr(65 + i), "planned_with": r.belief_label, "start": r.waypoints[0], "waypoints": r.waypoints,
            "predicted_when_planned": {"length_m": r.length_m, "energy_J": r.energy_J, "max_risk": r.max_risk,
                                       "max_slip": r.max_slip},
            "predicted_with_learned_belief": {"energy_J": learned.energy_J, "max_risk": learned.max_risk,
                                              "max_slip": learned.max_slip,
                                              "length_on_risk_above_limit_m": float(
                                                  np.sum(np.diff(np.r_[0, learned.profile["s"]])
                                                         [learned.profile["risk"] > sp.risk_max]))},
            "measured_multibody": {"ended": run.meta["ended"], "distance_m": s["distance_m"],
                                   "energy_J": s["energy_J"], "energy_per_m_J": s["energy_per_metre_J_m"],
                                   "max_slip_2s_front": float(np.nanmax(ms)), "max_slip_instant": s["max_slip"],
                                   "max_sinkage_mm": s["max_sinkage_mm"],
                                   "max_sinkage_front_mean_mm": float(
                                       1e3 * run.wheel["sinkage"][:, mission.front].mean(1).max()),
                                   "max_sinkage_wheel": run.model.spec.corners[
                                       int(np.argmax(run.wheel["sinkage"].max(0)))].name,
                                   "sim_time_s": float(run.t[-1]),
                                   "wall_time_s": float(run.meta.get("wall_s", np.nan)),
                                   "end_xy": run.q[-1, :2]}})
    if res.retreat is not None:
        out["retreat_point"] = res.retreat
    det = res.report.details
    out["recovery"] = det.get("recovery")
    out["held_out"] = det.get("held_out")
    out["stall_unit"] = det.get("stall_unit")
    out["planner_vs_multibody"] = det.get("planner_vs_multibody")
    out["energy_balance"] = det.get("energy_balance")
    out["time_step"] = det.get("time_step")
    cmp = {}
    for n, c in res.comparison.items():
        if isinstance(c, dict) and "analytic_prior" in c:
            cmp[n] = {k: {q: v for q, v in d.items() if q != "pred"} for k, d in c.items()
                      if k in ("analytic_prior", "analytic_learned", "surrogate_zoo") and d}
            cmp[n]["n"] = c["n"]
    out["model_comparison"] = cmp
    out["model_comparison_surrogate"] = res.comparison.get("_surrogate")
    out["beliefs"] = {site.units[u].name: {"mean": b.mean, "sd": b.sd, "n_obs": b.n_obs}
                      for u, b in enumerate(res.belief.units)}
    out["trust"] = res.report.to_dict()
    out["log"] = [{"stage": s.stage, "tool": s.tool, "seconds": s.seconds, "summary": s.summary} for s in res.log]
    return to_json(out)


def _fmt(v, f="{:.3g}"):
    return "-" if v is None else f.format(v)


def write_report(mission, res, out_dir, figures: bool = True, info: dict | None = None) -> dict:
    """Write the mission's numbers, figures and reports into ``out_dir``; returns what was written."""
    os.makedirs(out_dir, exist_ok=True)
    nums = mission_numbers(mission, res)
    nums["run_info"] = info or run_info()
    figs = {}
    if figures:
        from . import visuals as V
        site, sp = mission.site, res.spec
        figs["site_map"] = V.plot_site_map(site, res.planners, res.routes, res.runs,
                                           os.path.join(out_dir, "site_map.png"), sp.slip_limit)
        figs["traverse"] = V.plot_traverse(res.runs, site, os.path.join(out_dir, "traverse.png"), sp.slip_limit,
                                           mission.front)
        figs["route_risk"] = V.plot_route_risk(res.planners, res.routes, os.path.join(out_dir, "route_risk.png"),
                                               sp.slip_limit, sp.risk_max)
        if res.comparison:
            figs["models"] = V.plot_model_comparison(res.comparison, os.path.join(out_dir, "wheel_models.png"))
        su = [k for k, u in enumerate(site.units) if u.name == nums.get("stall_unit")]
        if su:
            figs["learning"] = V.plot_learning(res.belief, su[0], site.true_soil(su[0]),
                                               os.path.join(out_dir, "soil_learning.png"),
                                               title=f"What the rover learned about the '{site.units[su[0]].name}' "
                                                     f"(orbital map: {site.units[su[0]].orbital.name}); iterated EKF")
    nums["figures"] = {k: os.path.basename(v) for k, v in figs.items() if v}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(nums, f, indent=1)
    with open(os.path.join(out_dir, "trust_report.md"), "w") as f:
        f.write(res.report.to_markdown())
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write(report_markdown(nums))
    return nums


def report_markdown(n: dict) -> str:
    sp = n["spec"]
    L = ["# The rover that learns the planet", "", f"Mission: *{sp['text']}*", "",
         f"Parsed by `{sp['parser']}`: goal {tuple(sp['goal'])}, start {tuple(sp['start'])}, slip limit "
         f"{sp['slip_limit']:g}, risk limit {sp['risk_max']:g}, objective {sp['objective']}, speed {sp['speed']:g} m/s"
         + (f" (defaults: {', '.join(sp['defaults_used'])})" if sp["defaults_used"] else "") + ".", "",
         f"Rover: {n['rover']['model']}, {n['rover']['mass_kg']:.0f} kg, {n['rover']['wheels']} wheels "
         f"R = {n['rover']['wheel_radius_m']} m. Site: synthetic, truth known (soils listed in `summary.json`).", "",
         "All numbers below were measured in this run (multibody simulation on the true terrain, or the planner's "
         "prediction where marked).", "", "## Routes", "",
         "| Route | Planned with | Predicted max risk (then) | Predicted max risk (learned) | Measured outcome | "
         "Distance [m] | Drive energy [J] | Max 2-s slip | Max sinkage, any wheel [mm] | Max sinkage, front mean [mm] |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in n["routes"]:
        m = r["measured_multibody"]
        L.append(f"| {r['name']} | {r['planned_with']} | {r['predicted_when_planned']['max_risk']:.3f} | "
                 f"{r['predicted_with_learned_belief']['max_risk']:.3f} | {m['ended']} | {m['distance_m']:.1f} | "
                 f"{m['energy_J']:.0f} | {m['max_slip_2s_front']:.2f} | {m['max_sinkage_mm']:.0f} "
                 f"({m['max_sinkage_wheel']}) | {m['max_sinkage_front_mean_mm']:.0f} |")
    if n.get("retreat_point"):
        L += ["", f"After the stall the rover backs out on its own ruts to {tuple(round(v, 2) for v in n['retreat_point'])}"
              " (the reversal is not simulated) and drives route B from there."]
    if len(n["routes"]) > 1:
        a, b = n["routes"][0], n["routes"][-1]
        L += ["", "## Why the new route reduces the risk of getting stuck", "",
              f"* Under the learned belief, route A has a largest risk of "
              f"{a['predicted_with_learned_belief']['max_risk']:.2f} over "
              f"{a['predicted_with_learned_belief']['length_on_risk_above_limit_m']:.1f} m above the risk limit; "
              f"route B {b['predicted_with_learned_belief']['max_risk']:.3f} "
              f"({b['predicted_with_learned_belief']['length_on_risk_above_limit_m']:.1f} m above it).",
              f"* Predicted largest slip (learned belief mean, 1 = cannot climb): A "
              f"{a['predicted_with_learned_belief']['max_slip']:.2f}, B {b['predicted_with_learned_belief']['max_slip']:.2f};"
              f" measured: A {a['measured_multibody']['max_slip_2s_front']:.2f} (stalled), B "
              f"{b['measured_multibody']['max_slip_2s_front']:.2f}.",
              f"* The stall unit was '{n.get('stall_unit')}': the orbital map called it compact regolith; the "
              "rover's measurements say otherwise (table below), so every slope of that unit steeper than it can "
              "climb is now avoided."]
    if n.get("recovery"):
        L += ["", "## What the rover learned (known truth of the synthetic site)", "",
              "| Parameter | Truth | Orbital prior | Learned | Posterior sd | Error / sd |", "|---|---|---|---|---|---|"]
        for p, v in n["recovery"].items():
            L.append(f"| {p} | {v['true']:.3f} | {v['true'] + v['prior_error']:.3f} | {v['estimate']:.3f} | "
                     f"{v['sd']:.3f} | {v['z']:+.2f} |")
        if n.get("held_out"):
            h = n["held_out"]
            L += ["", f"Held-out check ({h['n']} measurements not used by the filter): sinkage RMSE "
                      f"{h['sinkage_rmse_mm']:.1f} mm, torque RMSE {h['torque_rmse_Nm']:.2f} N m."]
    if n.get("model_comparison"):
        L += ["", "## Wheel-soil models against the measurements", "",
              "| Unit | Model | Sinkage RMSE [mm] | Torque RMSE [N m] | Outside validity box |", "|---|---|---|---|---|"]
        for u, c in n["model_comparison"].items():
            for k in ("analytic_prior", "analytic_learned", "surrogate_zoo"):
                if k in c:
                    d = c[k]
                    L.append(f"| {u} (n={c['n']}) | {k} | {d['sinkage_rmse_mm']:.1f} | {d['torque_rmse_Nm']:.2f} | "
                             f"{_fmt(d.get('fraction_outside_validity'), '{:.0%}')} |")
        L += ["", f"Surrogate: {n.get('model_comparison_surrogate')}."]
    t = n["trust"]
    L += ["", "## Trust report", "", f"**Decision: {t['decisao']}** (score {_fmt(t['score'], '{:.2f}')}, coverage "
          f"{t['cobertura']:.0%})", "", "| Check | Status | Value | Limit | Source |", "|---|---|---|---|---|"]
    for c in t["checagens"]:
        L.append(f"| {c['name']} | {c['status']} | {_fmt(c['value'])} | {_fmt(c['limit'])} | {c['source']} |")
    L += ["", "## Agent log", "", "| Stage | Tool | Seconds |", "|---|---|---|"]
    for s in n["log"]:
        L.append(f"| {s['stage']} | {s['tool']} | {s['seconds']:.1f} |")
    ri = n.get("run_info", {})
    L += ["", "## Run", "", ", ".join(f"{k}: {v}" for k, v in ri.items())]
    if n.get("figures"):
        L += ["", "## Figures", ""] + [f"![{k}]({v})" for k, v in n["figures"].items()]
    return "\n".join(L) + "\n"


__all__ = ["write_report", "mission_numbers", "report_markdown", "run_info"]

"""Deliverables of the pipe-loop twin: control-room figures, confusion matrix, trust report, 3-D scene and report.

Every number comes from the tools' results (:mod:`.agent`) or this run's clock. The acceptance criteria of the trust
report are design choices of this demo, written next to each check.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np

from pinneapple_analysis.trust import Check, TrustReport

from .diagnose import RESIDUALS
from .plant import INPUTS, SENSORS, SIGMA, UNITS

KIND_COLOR = {"normal": "#22c55e", "shift": "#ef4444", "spread": "#f59e0b", "model": "#a855f7", "domain": "#3b82f6"}
FOCUS = {"process": ("Q_B", "P_B"), "sensor": ("Q_B", "Q_A"), "domain": ("Q_A", "P_pump"),
         "normal": ("Q_A", "P_tee")}


def _c(name, ok, *, critical=False, value=None, limit=None, source="", reason=""):
    st = "not_run" if ok is None else ("supports" if ok else "contradicts")
    return Check(name, st, critical, 1.0, None if value is None else float(value),
                 None if limit is None else float(limit), source, reason)


def trust_report(state) -> TrustReport:
    tw, mon, b = state["build_twin"], state["monitor"], state["benchmark"]["benchmark"]
    rs = state.get("request_simulation", {})
    M = np.array(b["confusion"])
    iso = b["isolation"]
    n_ok_kind = sum(v["correct_kind"] for v in iso.values())
    n_ok_tgt = sum(v["correct_target"] for v in iso.values())
    normal_row = M[0]
    k = b["kalman_k_abs_error"]
    checks = [
        _c("fast_model_holdout", tw["holdout_max_over_sigma"] <= 3.0, value=tw["holdout_max_over_sigma"], limit=3.0,
           source=f"{tw['n_test']} held-out operating points, max |error| / sensor sigma"),
        _c("fast_model_coverage_2std", tw["coverage_2std"] >= 0.9, value=tw["coverage_2std"], limit=0.9,
           source="held-out errors inside +-2 std"),
        _c("cause_accuracy", b["accuracy"] >= 0.9, critical=True, value=b["accuracy"], limit=0.9,
           source=f"{b['n_cases']} controlled cases, diagnosed kind = true kind"),
        _c("location_accuracy", n_ok_tgt >= 0.9 * max(n_ok_kind, 1), value=n_ok_tgt / max(n_ok_kind, 1), limit=0.9,
           source="right pipe / sensor / input among correctly classified faults"),
        _c("healthy_runs_stay_normal", normal_row[0] == normal_row.sum(), critical=True,
           value=normal_row[0] / max(normal_row.sum(), 1), limit=1.0, source="healthy cases diagnosed normal at the end"),
        _c("false_alarm_cases", b["cases_with_false_alarm"] <= 0.1 * b["n_cases"],
           value=b["cases_with_false_alarm"] / b["n_cases"], limit=0.1,
           source="cases with any alarm before the fault (or at all, when healthy)"),
        _c("kalman_size_inside_2std", None if not k["n"] else k["inside_2std"] >= 0.8 * k["n"],
           value=None if not k["n"] else k["inside_2std"] / k["n"], limit=0.8,
           source="obstructions whose true extra K is inside the filter's +-2 std"),
        _c("showcase_correct", mon["correct"] == mon["n"], value=mon["correct"], limit=mon["n"],
           source="obstruction B, drifting Q_B, overspeed, healthy"),
        _c("reference_newton_residual", b["newton_max_residual_m"] < 1e-8, value=b["newton_max_residual_m"],
           limit=1e-8, source="plant network solve, max head residual [m]"),
    ]
    if rs.get("n"):
        checks.append(_c("out_of_domain_physics_consistent", rs["physics_consistent"],
                         value=next(iter(rs["cases"].values()))["max_abs_z_reference"], limit=4.0,
                         source="reference at the new operating point vs readings (30-step mean, |z|)"))
        checks.append(_c("extended_model_inside_noise", rs["new_error_over_sigma"] <= 3.0,
                         value=rs["new_error_over_sigma"], limit=3.0,
                         source="fast model after adding the reference solves, max |error| / sigma"))
    return TrustReport(checks, model="pipe-loop digital twin")


# ----------------------------------------------------------------------------- figures
def control_room(case: dict[str, Any], path: str, *, title: str = "", cursor: float | None = None) -> str:
    import matplotlib.pyplot as plt
    run, d = case["run"], case["diagnosis"]
    ev, st, th = d["evidence"], d["stats"], d["thresholds"]
    t = run["t"] / 60
    sc = run["scenario"]
    fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True, constrained_layout=True,
                           gridspec_kw={"height_ratios": [2.2, 1.6, 1.6, 1.4]})
    a2 = ax[0].twinx()
    handles = []
    for a, s, col in ((ax[0], FOCUS[sc.kind][0], "#2563eb"), (a2, FOCUS[sc.kind][1], "#ea580c")):
        j = SENSORS.index(s)
        handles += a.plot(t, run["y"][:, j], lw=0.8, color=col, label=f"{s} measured")
        handles += a.plot(t, ev["surrogate_pred"][:, j], "--", lw=1.3, color=col, alpha=0.8, label=f"{s} twin")
        a.set_ylabel(f"{s} [{UNITS[s]}]", color=col)
    ax[0].legend(handles=handles, fontsize=7, ncol=4, loc="upper left")
    ax[0].set_title(title or f"{sc.name}: truth {sc.kind} {sc.target} -> diagnosed {d['kind']} {d.get('target', '')}",
                    fontsize=10)
    mz = st["mean_z"] * np.sqrt(np.minimum(np.arange(1, len(t) + 1), 18))[:, None]
    for i, r in enumerate(RESIDUALS):
        ax[1].plot(t, mz[:, i], lw=0.9, label=r)
    ax[1].axhspan(-3, 3, color="0.9")
    ax[1].set_ylabel("physics residuals\n(window, z)")
    ax[1].legend(fontsize=6, ncol=5, loc="upper left")
    for k, col in (("shift", "#ef4444"), ("spread", "#f59e0b"), ("model", "#a855f7")):
        ax[2].semilogy(t, np.maximum(st[k] / th[k], 1e-3), color=col, lw=1, label=f"{k} / threshold")
    ax[2].semilogy(t, np.maximum(st["ood"] / 0.02, 1e-3), color="#3b82f6", lw=1, label="outside envelope / 0.02")
    ax[2].axhline(1, color="k", lw=0.8, ls="--")
    ax[2].set_ylabel("evidence\n/ threshold")
    ax[2].legend(fontsize=7, ncol=4, loc="upper left")
    for i, k in enumerate(d["kinds"]):
        ax[3].axvspan(t[i] - 5 / 60, t[i] + 5 / 60, ymin=0.8, ymax=1.0, color=KIND_COLOR[k], lw=0)
    for jj, nm in enumerate("ABC"):
        m, s = ev["k_extra"][:, jj], ev["k_extra_std"][:, jj]
        ax[3].plot(t, m, lw=1, label=f"Kalman extra K, pipe {nm}")
        ax[3].fill_between(t, m - 2 * s, m + 2 * s, alpha=0.15)
    ax[3].set_ylim(min(-1.0, float(ev["k_extra"].min()) - 1), max(3.0, 1.25 * float(ev["k_extra"].max()) + 1))
    if sc.kind == "process":
        ax[3].axhline(sc.magnitude, color="k", lw=0.8, ls=":", label=f"true extra K ({sc.magnitude:g})")
    ax[3].legend(fontsize=7, ncol=4, loc="center left")
    ax[3].set_ylabel("Kalman\n(band: verdict)")
    ax[3].set_xlabel("time [min]")
    for a in ax:
        if sc.kind != "normal":
            a.axvline(sc.onset_s / 60, color="k", lw=0.6, ls=":")
        if cursor is not None:
            a.axvline(cursor, color="#111827", lw=1.6)
    fig.text(0.01, 0.003, f"verdict: {d['kind']} {d.get('target', '')} -- {d['action']}", fontsize=8)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def figure_confusion(b: dict[str, Any], path: str) -> str:
    import matplotlib.pyplot as plt
    M = np.array(b["confusion"])
    fig, ax = plt.subplots(figsize=(5.6, 4.8), constrained_layout=True)
    ax.imshow(M, cmap="Greens")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=13,
                    color="white" if M[i, j] > M.max() / 2 else "black")
    ax.set_xticks(range(4), b["kinds"])
    ax.set_yticks(range(4), b["kinds"])
    ax.set_xlabel("diagnosed")
    ax.set_ylabel("true (controlled case)")
    iso = b["isolation"]
    ax.set_title(f"{b['n_cases']} cases: cause right {100 * b['accuracy']:.0f} %; location right "
                 f"{sum(v['correct_target'] for v in iso.values())}/{sum(v['correct_kind'] for v in iso.values())}",
                 fontsize=9)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def figure_request(rs: dict[str, Any], path: str) -> str | None:
    import matplotlib.pyplot as plt
    if not rs.get("n"):
        return None
    name, c = next(iter(rs["cases"].items()))
    x = np.arange(len(SENSORS))
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.bar(x - 0.2, (c["old_prediction"] - c["reference"]) / SIGMA, 0.4, label="fast model before (error / sigma)")
    ax.bar(x + 0.2, (c["new_prediction"] - c["reference"]) / SIGMA, 0.4, label="after adding reference solves")
    ax.plot(x, c["z_reference_vs_readings"], "k*", ms=9, label="readings vs reference (z, 30-step mean)")
    ax.axhspan(-3, 3, color="0.92")
    ax.set_xticks(x, SENSORS, fontsize=8)
    ax.set_yscale("symlog", linthresh=3)
    ax.set_title(f"{name}: outside the envelope ({', '.join(f'{k} = {v:.3g}' for k, v in c['u'].items())}); the "
                 "physics still matches the plant, the fast model did not", fontsize=9)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def impact_gif(case: dict[str, Any], path: str, n: int = 14) -> str:
    import imageio.v2 as imageio
    tmp = path + ".frames"
    os.makedirs(tmp, exist_ok=True)
    t = case["run"]["t"] / 60
    frames = []
    for k, c in enumerate(np.linspace(t[0] + 5, t[-1], n)):
        p = control_room(case, os.path.join(tmp, f"{k:03d}.png"), cursor=float(c))
        frames.append(imageio.imread(p))
    imageio.mimsave(path, frames, duration=0.45, loop=0)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    return path


def render_3d(case: dict[str, Any], out_dir: str, *, render: bool, samples: int, size=(1280, 720)) -> dict[str, Any]:
    from pinneapple_tools.visualization.studio import web_viewer

    from .scene import loop_scene
    sc = loop_scene(case["run"], case["diagnosis"], title="Pumped loop twin: obstruction in branch B")
    files = {"viewer": web_viewer(sc, out_dir, single_file=True, filename="twin_viewer.html")}
    an = sc.animation
    for s in sc.surfaces:                        # a static copy at the last frame for glTF / USD
        if s.name in an.surface_fields:
            s.fields = {"p" if not s.name.startswith("sensor") else "flag": an.surface_fields[s.name]["values"][-1]}
    files["glb"] = sc.save(os.path.join(out_dir, "loop_twin.glb"))
    files["usda"] = sc.save(os.path.join(out_dir, "loop_twin.usda"))
    if not render:
        return {"files": files, "skipped": ["Blender renders: render=False"]}
    try:
        import bpy  # noqa: F401
    except ImportError:
        return {"files": files, "skipped": ["Blender renders: bpy is not installed"]}
    from pinneapple_tools.visualization.studio import blender
    t0 = time.perf_counter()
    sc2 = loop_scene(case["run"], case["diagnosis"])
    r = blender.render_animation(sc2, os.path.join(out_dir, "obstruction"), size=size, samples=samples,
                                 camera="fixed", background="studio", fps=4, gif=True, mp4=True, name="obstruction")
    files["impact_gif_3d"], files["impact_mp4_3d"] = r["gif"], r["mp4"]
    import shutil
    files["still_png"] = shutil.copy(r["frames"][-1], os.path.join(out_dir, "obstruction_last_frame.png"))
    return {"files": files, "skipped": [], "render_seconds": round(time.perf_counter() - t0, 1)}


# ----------------------------------------------------------------------------- report
def deliver(state, out_dir: str, *, render: bool = True, samples: int = 16) -> dict[str, Any]:
    from pinneapple_design.thermal.board_cooling.report import _env
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "3d"), exist_ok=True)
    F = lambda n: os.path.join(out_dir, "figures", n)  # noqa: E731
    mon = state["monitor"]["cases"]
    files = {f"control_room_{k}": control_room(c, F(f"control_room_{k}.png")) for k, c in mon.items()}
    b = state["benchmark"]["benchmark"]
    files["confusion"] = figure_confusion(b, F("confusion_matrix.png"))
    rs = state.get("request_simulation", {})
    fr = figure_request(rs, F("request_simulation.png"))
    if fr:
        files["request_simulation"] = fr
        name = next(iter(rs["cases"]))
        files["control_room_after_new_simulation"] = control_room(
            {"run": mon[name]["run"], "diagnosis": rs["cases"][name]["diagnosis_after"]},
            F("control_room_after_new_simulation.png"), title=f"{name} after the model was extended with the "
            "reference solves")
    files["impact_gif_2d"] = impact_gif(mon["overspeed"], F("impact_overspeed.gif"))
    r3 = render_3d(mon["obstruction_B"], os.path.join(out_dir, "3d"), render=render, samples=samples)
    files.update(r3["files"])
    rep = state["trust"]["report"]
    with open(os.path.join(out_dir, "trust_report.json"), "w") as f:
        json.dump(rep.to_dict(), f, indent=1)
    env = _env()
    import pinneapple as pp
    summary = {"decision": rep.decisao, "benchmark": {k: v for k, v in b.items() if k not in ("rows", "diagnoser")},
               "cases": [{k: v for k, v in r.items()} for r in b["rows"]],
               "showcase": {k: {kk: vv for kk, vv in c["diagnosis"].items()
                                if kk not in ("evidence", "stats", "kinds", "fits")} for k, c in mon.items()},
               "request_simulation": {k: {kk: vv for kk, vv in c.items() if kk != "diagnosis_after"}
                                      for k, c in rs.get("cases", {}).items()},
               "fast_model": {k: v for k, v in state["build_twin"].items() if k != "surrogate"},
               "evidence": [e.__dict__ for e in state.get("_evidence", [])],
               "plan": state["_plan"].to_dict() if state.get("_plan") else None, "environment": env,
               "skipped": r3["skipped"], "render_seconds": r3.get("render_seconds"),
               "files": {k: os.path.relpath(v, out_dir) for k, v in files.items()}}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(pp.to_jsonable(summary), f, indent=1)
    files["report"] = write_report(state, out_dir, files, env)
    return {"files": files, "report": files["report"], "n_files": len(files)}


def write_report(state, out_dir: str, files: dict[str, str], env: dict[str, Any]) -> str:
    b = state["benchmark"]["benchmark"]
    tw = state["build_twin"]
    mon = state["monitor"]["cases"]
    rs = state.get("request_simulation", {})
    rep = state["trust"]["report"]
    plan = state.get("_plan")
    M = np.array(b["confusion"])
    L = ["# The digital twin that knows when it is wrong", "",
         f"**Trust decision: {rep.decisao}.** Every number here was measured in this run.", "",
         "## Mission", "", f"> {state['read_mission']['text']}", "",
         "The loop: pump -> heater -> pipe A with a 90° elbow -> T junction -> branch B (valve, tank at 6 m) and "
         "branch C (valve, tank at 3 m). Ten sensors (4 pressures, 3 flow meters, 3 temperatures); five inputs "
         f"({', '.join(INPUTS)}). Schematic in the 3-D scene, lengths not to scale (A 60 m, B 40 m, C 35 m).", "",
         "## Plan and evidence", "", f"Planner: `{plan.planner if plan else '?'}`."
         + (" " + " ".join(plan.notes) if plan and plan.notes else ""), "",
         "| # | Tool | Why | Result | Time [s] |", "|---|---|---|---|---|"]
    for e in state.get("_evidence", []):
        summ = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in e.summary.items())
        L.append(f"| {e.step + 1} | `{e.tool}` | {e.why} | {summ or ('ERROR ' + e.error if not e.ok else '')} | "
                 f"{e.seconds:.1f} |")
    L += ["", "## The fast model", "",
          f"POD + Gaussian process on {tw['n_train']} reference solves of the operating envelope; on {tw['n_test']} "
          f"held-out points the largest error is {tw['holdout_max_over_sigma']:.2f} sensor sigmas (median "
          f"{tw['holdout_median_over_sigma']:.3f}), +-2 std covers {100 * tw['coverage_2std']:.0f} %.", "",
          "## Three events and a healthy hour", ""]
    for k, c in mon.items():
        d, sc = c["diagnosis"], c["run"]["scenario"]
        extra = ""
        if d["kind"] == "process":
            extra = f" Kalman: extra K = {d['size']:.2f} +- {2 * d['size_std']:.2f} (true {sc.magnitude:g})."
        elif d["kind"] == "sensor":
            extra = f" Estimated {d['mode']}: {d['size']:.3g} (true {sc.magnitude:g} {UNITS.get(sc.target, '')})."
        fa = "-" if d["first_alarm_s"] is None else f"{d['first_alarm_s'] / 60:.1f} min"
        L.append(f"- **{k}** (truth: {sc.kind} {sc.target}, onset {sc.onset_s / 60:.0f} min): diagnosed "
                 f"**{d['kind']} {d.get('target', '')}**, first alarm {fa}.{extra} "
                 f"Next: {d['action']}.")
    if rs.get("n"):
        name, c = next(iter(rs["cases"].items()))
        L += ["", "## Asking the physics", "",
              f"For **{name}** the twin ran the reference at the new operating point: readings vs reference |z| <= "
              f"{c['max_abs_z_reference']:.2f} (the plant still obeys the physics), fast model error "
              f"{c['old_error_over_sigma']:.1f} sigma before, {c['new_error_over_sigma']:.2f} sigma after adding 40 "
              f"reference solves in the widened envelope; diagnosis after: **{c['diagnosis_after']['kind']}**."]
    iso = b["isolation"]
    L += ["", "## Controlled cases", "",
          f"{b['n_cases']} cases ({b['n_cases'] // 4} per kind), random onset (15-40 min), size and location; "
          f"thresholds from {b['n_calibration']} separate healthy runs. Obstructions: extra K 1-12 in a random pipe; "
          "sensors: bias drift 2-12 sigma or noise x2-8 on a random sensor; domain: pump speed 1.03-1.20, valve C "
          "0.12-0.33, inlet 38-60 °C or heater 950-1300 kW (the small end of each range is meant to be hard).", "",
          "| true \\ diagnosed | " + " | ".join(b["kinds"]) + " |", "|---|" + "---|" * 4]
    for i, k in enumerate(b["kinds"]):
        L.append(f"| {k} | " + " | ".join(str(x) for x in M[i]) + " |")
    kk = b["kalman_k_abs_error"]
    L += ["", f"Cause right: **{100 * b['accuracy']:.1f} %**. Location right (pipe / sensor / input): "
          + ", ".join(f"{k} {v['correct_target']}/{v['correct_kind']}" for k, v in iso.items())
          + f"; sensor fault mode (drift / noise) right {iso['sensor'].get('correct_mode', 0)}/"
          f"{iso['sensor']['correct_target']}. Kalman extra-K error: mean {kk['mean']:.3g}, max {kk['max']:.3g}, "
          f"true value inside +-2 std in {kk['inside_2std']}/{kk['n']}. Cases with an alarm before the fault (or any "
          f"alarm when healthy): {b['cases_with_false_alarm']}. Median detection delay after onset [s]: "
          + ", ".join(f"{k} {v:.0f}" for k, v in b["delay_s_median"].items() if v is not None)
          + f". Time: fast model {b['seconds']['fit_fast_model']:.1f} s, calibration {b['seconds']['calibration']:.1f} s, "
          f"cases {b['seconds']['cases']:.1f} s ({b['seconds']['per_case']:.2f} s each).", ""]
    wrong = [r for r in b["rows"] if r["kind"] != r["truth"]["kind"] or r["target"] != r["truth"]["target"]]
    if wrong:
        L += ["Misdiagnosed cases:", ""] + [f"- {r['name']}: truth {r['truth']['kind']} {r['truth']['target']} "
                                            f"({r['truth']['magnitude']:.3g}) -> {r['kind']} {r['target']}"
                                            for r in wrong] + [""]
    L += ["## Trust report", "", rep.to_markdown().split("\n", 2)[2], "", "## Files", ""]
    L += [f"- {k}: `{os.path.relpath(v, out_dir)}`" for k, v in files.items()]
    L += ["", "## Model scope", "",
          "- The plant is simulated (quasi-steady network: pump affinity curve, Darcy-Weisbach / Colebrook pipes, "
          "minor losses, valve law, heater and branch heat loss; noise of typical instrument classes). Faults and "
          "their sizes are injected, so every answer is known; a real plant adds unmodelled effects.",
          "- The quasi-steady model holds for valve moves slower than ten acoustic round trips "
          "(`plant.quasi_steady_valid`); faster moves need the water-hammer solver "
          "(`pinneapple_simulation.numerical_solvers.water_hammer_moc`), not exercised here.",
          "- The elbow streamlines in the 3-D scene are the 1/7 power-law profile, not a CFD solution.", "",
          "## Environment", "", "```", json.dumps(env, indent=1), "```", ""]
    path = os.path.join(out_dir, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(L))
    return path

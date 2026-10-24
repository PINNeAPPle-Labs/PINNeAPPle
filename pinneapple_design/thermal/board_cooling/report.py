"""Deliverables of a cooling study: figures, 3-D scene and viewer, Blender renders, the report and its trust report.

``deliver(state, out_dir)`` is the last tool of the agent (:mod:`.agent`); it only reads what the earlier tools
measured. Every number in ``report.md`` and ``summary.json`` comes from those results or from this run's clock.
"""
from __future__ import annotations

import json
import os
import platform
import time
from typing import Any

import numpy as np

from .simulate import map_axes

T_LIM_COLOR = "#111827"


def _states(state) -> list[tuple[str, dict[str, Any]]]:
    """(label, result) of the story: bare board, best still-air sink, each design round, the verified design."""
    m = state["read_mission"]["mission"]
    out = [("bare board, still air", state["baseline"]["result"])]
    if "screen_natural" in state:
        out.append(("largest heat sink, still air", state["screen_natural"]["best"]["result"]))
    for r in state["optimize"]["loop"]["rounds"]:
        res = {"design": r["design"], "tj_c": r["ref_tj_c"], "tj_pessimistic_c": r["ref_tj_pessimistic_c"],
               "map_c": r["ref_map_c"], "mass_g": r["mass_g"], "fan_power_w": r["fan_power_w"]}
        out.append((f"round {r['round']}: optimum on the surrogate, reference solve", res))
    v = state["verify"]["fine"]
    out.append(("verified design, fine grids", v))
    _ = m
    return out


def figure_maps(state, path: str) -> str:
    import matplotlib.pyplot as plt
    m = state["read_mission"]["mission"]
    sts = _states(state)
    pick = [sts[0], sts[1] if len(sts) > 2 else sts[0], sts[-1]]
    yc, xc = map_axes(m.board)
    vmax = max(float(np.max(r["map_c"])) for _, r in pick)
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.9), constrained_layout=True)
    for a, (lab, r) in zip(ax, pick, strict=True):
        im = a.pcolormesh(xc, yc, r["map_c"], cmap="jet", vmin=m.t_ambient_c, vmax=vmax, shading="auto")
        cs = a.contour(xc, yc, r["map_c"], levels=[m.tj_limit_c], colors=T_LIM_COLOR, linewidths=1.2)
        a.clabel(cs, fmt=f"{m.tj_limit_c:.0f} °C", fontsize=8)
        for c in m.components:
            rc = c.resolved()
            a.add_patch(plt.Rectangle((rc.x_mm - rc.w_mm / 2, rc.y_mm - rc.d_mm / 2), rc.w_mm, rc.d_mm, fill=False,
                                      ec="white", lw=0.8))
            a.text(rc.x_mm, rc.y_mm, f"{c.name}\n{r['tj_c'][c.name]:.0f} °C", color="white", fontsize=7,
                   ha="center", va="center")
        a.set_title(lab, fontsize=10)
        a.set_aspect("equal")
        a.set_xlabel("x [mm]")
    ax[0].set_ylabel("y [mm] (air flows +y)")
    fig.colorbar(im, ax=ax, label="board top temperature [°C]", shrink=0.9)
    fig.suptitle(f"{m.hot} at {m.hot_component().power_w:g} W: board-top temperature (junction temperatures in the "
                 f"boxes; contour = {m.tj_limit_c:.0f} °C)", fontsize=11)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def figure_evolution(state, path: str) -> str:
    """Maximum (hot-part junction) temperature through the study: baseline, still air, the optimiser's best member
    per generation (surrogate, pessimistic) and each reference-verified optimum."""
    import matplotlib.pyplot as plt
    m = state["read_mission"]["mission"]
    loop = state["optimize"]["loop"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True)
    a = ax[0]
    pts = [("bare board", state["baseline"]["tj_hot_c"])]
    if "screen_natural" in state:
        pts.append(("still-air sink", state["screen_natural"]["tj_hot_pessimistic_c"]))
    for i, (_, v) in enumerate(pts):
        a.bar(i, v, color="#ef4444" if v > m.tj_limit_c else "#22c55e")
        a.text(i, v, f"{v:.0f} °C", ha="center", va="bottom", fontsize=9)
    k0 = len(pts)
    for j, r in enumerate(loop["rounds"]):
        v = r["ref_tj_pessimistic_c"][m.hot]
        a.bar(k0 + j, v, color="#ef4444" if v > m.tj_limit_c else "#22c55e")
        a.text(k0 + j, v, f"{v:.1f} °C", ha="center", va="bottom", fontsize=9)
    vf = state["verify"]["tj_fine_pessimistic"][m.hot]
    a.bar(k0 + len(loop["rounds"]), vf, color="#16a34a")
    a.text(k0 + len(loop["rounds"]), vf, f"{vf:.1f} °C", ha="center", va="bottom", fontsize=9)
    a.axhline(m.tj_limit_c, color=T_LIM_COLOR, ls="--", lw=1)
    a.set_xticks(range(k0 + len(loop["rounds"]) + 1),
                 [p[0] for p in pts] + [f"round {r['round']}\n(reference)" for r in loop["rounds"]]
                 + ["verified\n(fine grids)"], fontsize=8)
    a.set_ylabel(f"Tj {m.hot} [°C] (pessimistic cooling where a sink is fitted)")
    a.set_title("Hot-spot temperature, measured at each stage")
    b = ax[1]
    b2 = b.twinx()
    for r in loop["rounds"]:
        h = r["optimizer_history"]
        g = [e["generation"] for e in h]
        b.plot(g, [e["tj_hot_pessimistic_pred_c"] for e in h], lw=1.2, label=f"round {r['round']}: Tj predicted")
        b2.plot(g, [e["mass_g"] for e in h], lw=1, ls=":", color="grey")
    b.axhline(m.tj_limit_c, color=T_LIM_COLOR, ls="--", lw=1, label="limit")
    b.set_xlabel("optimiser generation")
    b.set_ylabel(f"best member: Tj {m.hot} pessimistic, surrogate [°C]")
    b2.set_ylabel("best member: heat-sink mass [g] (dotted)")
    b.legend(fontsize=8, loc="upper right")
    b.set_title("Search on the surrogate (lightest design that meets the limits)")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def figure_holdout(state, path: str) -> str:
    """Surrogate vs reference on the held-out designs, and the surrogate's map error on the verified design."""
    import matplotlib.pyplot as plt

    from .learn import CoolingSurrogate
    m = state["read_mission"]["mission"]
    data = state["sample"]["data"]
    h = state["learn"]["holdout"]
    te = np.array(h["test_index"])
    tr = np.setdiff1d(np.arange(len(data["X"])), te)
    sur = CoolingSurrogate(m.t_ambient_c).fit(data["X"][tr], data["Y"][tr], data["maps"][tr])
    p = sur.predict(data["X"][te], maps=False)
    names = [c.name for c in m.components]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    a = ax[0]
    for j, nm in enumerate(names):
        a.errorbar(data["Y"][te][:, j], p["tj"][:, j], yerr=2 * p["tj_std"][:, j], fmt="o", ms=3.5, lw=0.7,
                   label=nm)
    lo, hi = data["Y"][te][:, :len(names)].min(), data["Y"][te][:, :len(names)].max()
    a.plot([lo, hi], [lo, hi], "k-", lw=0.8)
    s = h["surrogate"]
    a.set_title(f"{h['n_test']} unseen designs: |error| mean {s['tj_mae_k']:.2f} K, max {s['tj_max_abs_k']:.2f} K;"
                f"\n+-2 std covers {100 * h['coverage_2std']:.0f} %; nearest snapshot: mean "
                f"{h['nearest_snapshot']['tj_mae_k']:.2f} K", fontsize=9)
    a.set_xlabel("reference Tj [°C]")
    a.set_ylabel("surrogate Tj [°C] (bars: +-2 std)")
    a.legend(fontsize=8)
    v = state["verify"]
    yc, xc = map_axes(m.board)
    err = v["prediction"]["map"] - v["fine"]["map_c"]
    lim = float(np.abs(err).max())
    im = ax[1].pcolormesh(xc, yc, err, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
    ax[1].set_aspect("equal")
    ax[1].set_title(f"Verified design: surrogate map - fine solve (max |error| {lim:.2f} K)", fontsize=9)
    fig.colorbar(im, ax=ax[1], label="[K]")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def impact_gif(state, path: str, hold: int = 6) -> str:
    """2-D animation of the board map from the bare board to the verified design (one frame per stage, held)."""
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    m = state["read_mission"]["mission"]
    yc, xc = map_axes(m.board)
    frames = []
    vmax = 120.0
    for lab, r in _states(state):
        fig, a = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
        im = a.pcolormesh(xc, yc, np.clip(r["map_c"], m.t_ambient_c, vmax), cmap="jet", vmin=m.t_ambient_c,
                          vmax=vmax, shading="auto")
        hot = r["tj_c"][m.hot]
        a.set_title(f"{lab}\nTj {m.hot} {hot:.1f} °C (limit {m.tj_limit_c:.0f})"
                    + (f", sink {r['mass_g']:.0f} g" if r.get("mass_g") else ""), fontsize=10,
                    color="#b91c1c" if hot > m.tj_limit_c else "#15803d")
        a.set_aspect("equal")
        fig.colorbar(im, ax=a, label=f"board top [°C] (clipped at {vmax:.0f})")
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        frames += [img] * hold
    imageio.mimsave(path, frames, duration=0.25, loop=0)
    return path


def _env() -> dict[str, Any]:
    import numpy
    import scipy
    out = {"python": platform.python_version(), "platform": platform.platform(), "numpy": numpy.__version__,
           "scipy": scipy.__version__, "cpus": os.cpu_count()}
    try:
        with open("/proc/cpuinfo") as f:
            out["cpu"] = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        out["cpu"] = platform.processor()
    try:
        import sklearn
        out["scikit-learn"] = sklearn.__version__
    except ImportError:
        pass
    try:
        import bpy
        out["blender"] = bpy.app.version_string
    except ImportError:
        pass
    return out


def render_3d(state, out_dir: str, *, render: bool = True, samples: int = 24, size=(1280, 720)) -> dict[str, Any]:
    """glTF / USD of the verified design, the self-contained timeline viewer and, with Blender, stills of the bare
    and the verified board and the animation of the stages (GIF + MP4)."""
    from pinneapple_tools.visualization.studio import web_viewer

    from .scene import T_RANGE, board_scene, timeline_scene
    m = state["read_mission"]["mission"]
    sts = _states(state)
    files: dict[str, str] = {}
    skipped: list[str] = []
    final = board_scene(m, sts[-1][1], title="Verified design")
    files["glb"] = final.save(os.path.join(out_dir, "verified_design.glb"))
    files["usda"] = final.save(os.path.join(out_dir, "verified_design.usda"))
    tl = timeline_scene(m, sts)
    files["viewer"] = web_viewer(tl, out_dir, single_file=True, filename="thermal_viewer.html")
    if not render:
        return {"files": files, "skipped": ["Blender renders: render=False"]}
    try:
        import bpy  # noqa: F401
    except ImportError:
        return {"files": files, "skipped": ["Blender renders: bpy is not installed (pip install bpy)"]}
    from pinneapple_tools.visualization.studio import blender
    t0 = time.perf_counter()
    bare = board_scene(m, sts[0][1], title="Bare board", paths=False)
    for key, sc, ttl in (("bare_png", bare, f"Bare board: {m.hot} {sts[0][1]['tj_c'][m.hot]:.0f} °C"),
                         ("verified_png", final, f"Verified: {m.hot} {sts[-1][1]['tj_c'][m.hot]:.1f} °C")):
        sf = {s.name: ("T", *T_RANGE, "jet", f"{ttl} -- temperature [°C]") for s in sc.surfaces if "T" in s.fields}
        lr = (m.t_ambient_c, float(max(v.max() for v in sc.lines[0].values))) if sc.lines else None
        files[key] = blender.render(sc, os.path.join(out_dir, f"{key[:-4]}_board.png"), surface_fields=sf,
                                    lines="air" if sc.lines else False, line_range=lr, view="iso", samples=samples, size=size,
                                    background="studio")
    fans_on = [bool((r.get("design") or {}).get("air_velocity_m_s", 0.0) > 0) and i == len(sts) - 1
               for i, (_, r) in enumerate(sts)]                     # the drawn air paths are the final design's
    r = blender.render_animation(tl, os.path.join(out_dir, "impact"), size=size, samples=max(8, samples // 2),
                                 camera="fixed", background="studio", fps=1, gif=True, mp4=True, name="impact",
                                 lines=fans_on)
    files["impact_gif_3d"], files["impact_mp4_3d"] = r["gif"], r["mp4"]
    return {"files": files, "skipped": skipped, "render_seconds": round(time.perf_counter() - t0, 1)}


def write_report(state, out_dir: str, files: dict[str, str], extra: dict[str, Any]) -> str:
    m = state["read_mission"]["mission"]
    hot = m.hot
    v = state["verify"]
    d = v["design"]
    loop = state["optimize"]["loop"]
    h = state["learn"]["holdout"]
    ev = state.get("_evidence", [])
    plan = state.get("_plan")
    L = [f"# The self-optimizing engineer: cooling the {hot}", "",
         f"**Decision: {v['decision']}** (trust report below). Every number here was measured in this run.", "",
         "## Mission", "", f"> {m.text}", "",
         "| Quantity | Value | Source |", "|---|---|---|",
         f"| {hot} power | {m.hot_component().power_w:g} W | {m.provenance.get('power_w')} |",
         f"| Junction limit | {m.tj_limit_c:g} °C | {m.provenance.get('tj_limit_c')} |",
         f"| Air in the enclosure | {m.t_ambient_c:g} °C | {m.provenance.get('t_ambient_c')} |",
         f"| Heat-sink envelope | {m.max_width_mm:g} x {m.max_depth_mm:g} x {m.max_height_mm:g} mm | "
         f"{m.provenance.get('envelope')} |",
         f"| Fan air power budget | {m.fan_power_budget_w:g} W | {m.provenance.get('fan_power_budget_w')} |",
         f"| Board | {m.board.width_mm:g} x {m.board.depth_mm:g} mm, {m.board.n_copper} layers | "
         f"{m.provenance.get('board')} |", "",
         "Other parts (their own junction limits): " + ", ".join(
             f"{c.name} {c.power_w:g} W ({c.package}, {m.limits()[c.name]:g} °C)" for c in m.components
             if c.name != hot) + ".", "",
         "## Plan and evidence", "",
         f"Planner: `{plan.planner if plan else '?'}`." + (" " + " ".join(plan.notes) if plan and plan.notes else ""),
         "", "| # | Tool | Why | Result | Time [s] |", "|---|---|---|---|---|"]
    for e in ev:
        summ = ", ".join(f"{k}={v_:.4g}" if isinstance(v_, float) else f"{k}={v_}" for k, v_ in e.summary.items())
        L.append(f"| {e.step + 1} | `{e.tool}` | {e.why} | {summ or ('ERROR ' + e.error if not e.ok else '')} | "
                 f"{e.seconds:.1f} |")
    b, nat = state["baseline"], state.get("screen_natural")
    L += ["", "## What happened", "",
          f"1. **Today**: bare board in still air, {hot} junction **{b['tj_hot_c']:.1f} °C**, "
          f"{b['over_limit_k']:+.1f} K over the limit."]
    if nat:
        L.append(f"2. **Passive cooling is not enough**: the largest heat sink the envelope allows, in still air, "
                 f"gives {nat['tj_hot_c']:.1f} °C nominal / {nat['tj_hot_pessimistic_c']:.1f} °C pessimistic "
                 f"(best of {len(nat['rows'])} fin gaps).")
    s = h["surrogate"]
    L += [f"3. **Learn**: {state['sample']['n']} reference solves (median {state['sample']['median_seconds']:.2f} s "
          f"each, board energy balance <= {state['sample']['max_energy_balance']:.1e}); surrogate measured on "
          f"{h['n_test']} held-out designs: junction |error| mean **{s['tj_mae_k']:.2f} K**, max "
          f"**{s['tj_max_abs_k']:.2f} K**; board map RMSE **{s['map_rmse_k']:.2f} K**, max {s['map_max_abs_k']:.2f} K; "
          f"+-2 std covers {100 * h['coverage_2std']:.0f} % of the errors; nearest-snapshot baseline mean "
          f"{h['nearest_snapshot']['tj_mae_k']:.2f} K. Prediction {1e3 * h['predict_seconds_per_design']:.2f} ms vs "
          f"reference {h['reference_seconds_per_design']:.2f} s per design ({h['speedup']:.0f}x).",
          "   Learning curve (hold-out per number of solves): " + "; ".join(
              f"{c['n']} solves: max |error| {c['tj_max_abs_k']:.2f} K, coverage {100 * c['coverage_2std']:.0f} %"
              for c in state["learn"].get("curve", [])) + ("" if state["learn"].get("accepted", True)
                                                          else " -- NOT accepted within the solve budget") + ".",
          f"4. **Optimise and confirm**: {len(loop['rounds'])} round(s) of surrogate search + reference check, "
          f"{'confirmed' if loop['confirmed'] else 'NOT confirmed'}."]
    for r in loop["rounds"]:
        L.append(f"   - round {r['round']} ({r['n_snapshots']} snapshots): {r['mass_g']:.1f} g, fan air "
                 f"{1e3 * r['fan_power_w']:.1f} mW; {hot} pessimistic predicted "
                 f"{r['pred_tj_pessimistic_c'][hot]:.2f} +- {2 * r['pred_std_k'][hot]:.2f} °C, reference "
                 f"{r['ref_tj_pessimistic_c'][hot]:.2f} °C; max |error| over parts {r['max_abs_error_k']:.2f} K; "
                 f"{'meets' if r['meets_limits'] else 'violates'} the limits.")
    fine = v["fine"]
    L += [f"5. **Verify independently** (board grid {fine['grid']['nx']} x {fine['grid']['ny']} x {fine['grid']['nz']},"
          f" heat-sink base grid {fine['fidelity']['hs_grid']}^2): {hot} **{v['tj_fine'][hot]:.2f} °C** nominal, "
          f"**{v['tj_fine_pessimistic'][hot]:.2f} °C** pessimistic (limit {m.tj_limit_c:g}); largest change from the "
          f"training grid {v['grid_max_dtj_k']:.2f} K.", "",
          "## The design", "",
          "| Parameter | Value |", "|---|---|",
          f"| Base | {d.base_width_mm:.1f} x {d.base_depth_mm:.1f} x {d.base_thickness_mm:.2f} mm |",
          f"| Fins | {d.n_fins} x {d.fin_thickness_mm:.2f} mm, {d.fin_height_mm:.1f} mm tall, gap {d.fin_gap_mm:.2f} mm |",
          f"| Total height | {d.height_mm:.1f} mm |",
          f"| Air speed | {d.air_velocity_m_s:.2f} m/s |",
          f"| Material | {d.material} |",
          f"| Mass | {fine['mass_g']:.1f} g |",
          f"| Fan air power (dp x flow) | {1e3 * fine['fan_power_w']:.2f} mW ({fine['pressure_drop_pa']:.1f} Pa) |",
          f"| Heat up through the sink | {fine['q_sink_w']:.1f} of {m.hot_component().power_w:g} W |", "",
          "| Part | Tj nominal [°C] | Tj pessimistic [°C] | Limit [°C] |", "|---|---|---|---|"]
    for c in m.components:
        L.append(f"| {c.name} | {v['tj_fine'][c.name]:.2f} | {v['tj_fine_pessimistic'][c.name]:.2f} | "
                 f"{m.limits()[c.name]:g} |")
    L += ["", "## Trust report", "", v["report"].to_markdown().split("\n", 2)[2], "",
          "## Figures and 3-D", ""]
    for k, f in files.items():
        L.append(f"- {k}: `{os.path.relpath(f, out_dir)}`")
    L += ["", "## Model scope", "",
          "- Reference = coupled 3-D board finite volumes (`pinneapple_physics.pcb`) and plate-fin heat-sink model "
          "(`pinneapple_design.thermal.heatsink`); both are engineering models with published correlations, not CFD. "
          "Pessimistic = board surface coefficients x 0.8, heat-sink coefficient x 0.85 (each engine's stated band).",
          "- Air in the enclosure at a uniform temperature; air heating along the board and bypass around the sink "
          "are not modelled (the heat-sink model assumes ducted flow). The air paths of the 3-D scene are an "
          "energy balance per strip, for illustration, not a flow solution.",
          "- Package thermal data: typical JEDEC values of `pcb_thermal.PACKAGES`; use datasheet values for sign-off.",
          "- No conjugate CFD (OpenFOAM) was run here: it is not installed in this environment.", "",
          "## Environment", "", "```", json.dumps(extra["environment"], indent=1), "```", ""]
    path = os.path.join(out_dir, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(L))
    return path


def deliver(state, out_dir: str, *, render: bool = True, samples: int = 24) -> dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    figs = os.path.join(out_dir, "figures")
    os.makedirs(figs, exist_ok=True)
    files = {"maps": figure_maps(state, os.path.join(figs, "maps_before_after.png")),
             "evolution": figure_evolution(state, os.path.join(figs, "tmax_evolution.png")),
             "holdout": figure_holdout(state, os.path.join(figs, "surrogate_holdout.png")),
             "impact_gif_2d": impact_gif(state, os.path.join(figs, "impact_2d.gif"))}
    td = os.path.join(out_dir, "3d")
    os.makedirs(td, exist_ok=True)
    r3 = render_3d(state, td, render=render, samples=samples)
    files.update(r3["files"])
    extra = {"environment": _env(), "skipped": r3["skipped"], "render_seconds": r3.get("render_seconds")}
    m = state["read_mission"]["mission"]
    v = state["verify"]
    with open(os.path.join(out_dir, "trust_report.json"), "w") as f:
        json.dump(v["report"].to_dict(), f, indent=1)
    summary = {"mission": m.to_dict(), "decision": v["decision"], "design": v["design"].to_dict(),
               "tj_fine_c": v["tj_fine"], "tj_fine_pessimistic_c": v["tj_fine_pessimistic"],
               "grid_max_dtj_k": v["grid_max_dtj_k"], "mass_g": v["fine"]["mass_g"],
               "fan_power_w": v["fine"]["fan_power_w"], "baseline_tj_hot_c": state["baseline"]["tj_hot_c"],
               "natural_best_tj_hot_c": (state.get("screen_natural") or {}).get("tj_hot_c"),
               "holdout": {k: x for k, x in state["learn"]["holdout"].items() if k != "test_index"},
               "rounds": [{k: x for k, x in r.items() if k not in ("pred_map_c", "ref_map_c", "optimizer_history")}
                          for r in state["optimize"]["loop"]["rounds"]],
               "evidence": [e.__dict__ for e in state.get("_evidence", [])],
               "plan": state["_plan"].to_dict() if state.get("_plan") else None, **extra,
               "files": {k: os.path.relpath(f, out_dir) for k, f in files.items()}}
    import pinneapple as pp
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(pp.to_jsonable(summary), f, indent=1)
    rep = write_report(state, out_dir, files, extra)
    files["report"] = rep
    return {"files": files, "report": rep, "n_files": len(files), "skipped": r3["skipped"]}

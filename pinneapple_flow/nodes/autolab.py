"""Autonomous Physics Lab nodes: a mission in plain language planned and run through controlled library tools
(library: :mod:`pinneapple_autolab`, missions in their domain packages)."""
from __future__ import annotations

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node
from ..runtime import artifact

_TAGS = ("section:optimization", "agent", "mission", "surrogate", "verification", "thermal")


@node("autolab.board_cooling", "Self-optimizing engineer (board cooling)", category="Design",
      outputs=[Port("report", "report"), Port("figure", "figure"), Port("metrics", "metrics")],
      params=[Param("mission", "text", "", help="the request in plain language (empty: the built-in example)"),
              Param("n", "int", 64, minimum=12, maximum=400, help="initial reference solves"),
              Param("planner", "enum", "template", choices=["template", "llm"],
                    help="llm: a language model orders the same tools (falls back to the template)"),
              Param("render", "bool", False, help="Blender renders (needs bpy)"),
              Param("workers", "int", 4, minimum=1, maximum=32)],
      calls="pinneapple_design.thermal.board_cooling.run_mission (pinneapple_autolab.execute)", tags=_TAGS)
def board_cooling(ctx, inputs, params):
    """Read the request, solve the bare board, try still air, learn the temperature field from reference solves
    (hold-out measured), find the lightest heat sink and fan speed under the limits, confirm it on the reference,
    verify it on finer grids and write the report with its trust report."""
    from pinneapple_design.thermal.board_cooling import DEFAULT_TEXT, run_mission
    out = ctx.workdir / "board_cooling"
    run = run_mission(params["mission"] or DEFAULT_TEXT, out_dir=str(out), planner=params["planner"],
                      n=int(params["n"]), render=bool(params["render"]), workers=int(params["workers"]), log=ctx.log)
    if not run.ok:
        bad = next(e for e in run.evidence if not e.ok)
        raise StepFailed(f"{bad.tool}: {bad.error}")
    st = run.state
    v = st["verify"]
    ctx.table("Evidence", ["tool", "seconds", "result"],
              [[e.tool, e.seconds, ", ".join(f"{k}={x}" for k, x in e.summary.items())[:160]] for e in run.evidence])
    hot = st["read_mission"]["hot"]
    ctx.summary(f"{v['decision']}: {hot} {v['tj_fine_pessimistic'][hot]:.1f} °C pessimistic (bare board "
                f"{st['baseline']['tj_hot_c']:.0f} °C), heat sink {v['fine']['mass_g']:.0f} g")
    files = st["deliver"]["files"]
    ctx.report["figures"].append({"path": files["evolution"], "name": "evolution", "caption": "hot-spot temperature"})
    return {"report": artifact(files["report"], "md"), "figure": artifact(files["maps"], "png"),
            "metrics": {"decision": v["decision"], "tj_pessimistic_c": v["tj_fine_pessimistic"],
                        "mass_g": v["fine"]["mass_g"], "fan_power_w": v["fine"]["fan_power_w"],
                        "holdout": {k: x for k, x in st["learn"]["holdout"].items() if k != "test_index"}}}


_AERO_TAGS = ("section:optimization", "agent", "mission", "surrogate", "verification", "cfd", "openfoam", "aero")


@node("autolab.aerodynamicist", "The AI Aerodynamicist (race-car CFD lab)", category="Design",
      outputs=[Port("report", "report"), Port("figure", "figure"), Port("metrics", "metrics")],
      params=[Param("source", "enum", "recorded", choices=["recorded", "run"],
                    help="recorded: the committed session (20 OpenFOAM runs) replayed in the dashboard; run: the "
                         "mission now (OpenFOAM, about 10 min per CFD run on 4 cores)"),
              Param("mission", "text", "", help="run only: the request in plain language (empty: the flagship one)"),
              Param("budget", "int", 20, minimum=6, maximum=60, help="run only: reference simulations"),
              Param("procs", "int", 4, minimum=1, maximum=64, advanced=True)],
      calls="pinneapple_autolab.agent.run_agent (pinneapple_autolab.domains.racecar) + "
            "pinneapple_autolab.dashboard.build_dashboard", tags=_AERO_TAGS)
def aerodynamicist(ctx, inputs, params):
    """The Autonomous Physics Lab on a parametric racing car: lower drag at 70 m/s keeping the downforce and the
    aero balance, OpenFOAM as the reference, surrogates compared on held-out runs, every candidate triaged
    (favourable / uncertain / out of domain), the next CFD chosen by what is unknown, the answer audited. Returns the
    four-area dashboard (geometry, physics, models, decisions) with the replay of the session."""
    import json
    from pathlib import Path

    from pinneapple_autolab.domains import racecar
    root = Path(__file__).resolve().parents[2]
    out = Path(ctx.workdir) / "aerodynamicist"
    if params["source"] == "recorded":
        res_dir = root / "examples" / "autolab_aero" / "_out"
        if not (res_dir / "summary.json").exists():
            raise StepFailed(f"no recorded session in {res_dir}")
        page = racecar.recorded_dashboard(str(res_dir), str(out))
        summary = json.load(open(res_dir / "summary.json"))
        fig_png = res_dir / "figures" / "pareto.png"
    else:
        from pinneapple_simulation.numerical_solvers.external_flow import openfoam_available
        if not openfoam_available():
            raise StepFailed("OpenFOAM is not installed here; use source=recorded")
        from pinneapple_autolab.agent import run_agent
        text = params["mission"] or (racecar.flagship_mission().statement + f" Use at most {params['budget']} CFD runs.")
        res = run_agent(text, racecar.RaceCarAero(procs=int(params["procs"])), str(out / "session"),
                        domain=racecar.racecar_domain(), lab_options={"critical_region": "diffuser", "field_tol": 0.05},
                        listeners=[lambda e: ctx.log(f"{e['stage']}: {e['message'][:160]}")])
        if res.refused:
            raise StepFailed(f"mission refused: {res.refused}")
        summary = res.summary
        page = racecar.build_dashboard(str(out / "session"), str(out / "dashboard"), summary, viewers=False)
        fig_png = None
    imp = summary.get("improvement") or {}
    for k, v in (imp.get("drag_N") or {}).items():
        if isinstance(v, (int, float)):
            ctx.metric(f"drag_{k}", float(v), "CFD of the original and of the recommended design")
    ctx.metric("cfd_runs", summary["budget"]["used_runs"], "reference simulations charged to the mission")
    ctx.table("decisions", ["run", "judged before", "after", "drag pred -> CFD"],
              [[d["run"], d["label_before"], d["label_after"],
                f"{d['predicted'].get('drag_N', float('nan')):.0f} -> {d['reference'].get('drag_N', float('nan')):.0f}"]
               for d in summary.get("decisions", [])])
    ctx.view("html", {"path": str(Path(page).resolve()), "title": "The AI Aerodynamicist"})
    ctx.summary(summary.get("conclusion", ""))
    outs = {"report": artifact(page, "html"), "metrics": {"conclusion": summary.get("conclusion"),
                                                          "budget": summary["budget"], "improvement": imp}}
    if fig_png is not None and fig_png.exists():
        import shutil
        out.mkdir(parents=True, exist_ok=True)
        outs["figure"] = artifact(shutil.copy(fig_png, out / fig_png.name), "png")    # artifacts live in the run
    return outs

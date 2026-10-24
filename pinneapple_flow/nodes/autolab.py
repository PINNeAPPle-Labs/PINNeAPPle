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


@node("autolab.pipe_twin", "Digital twin that knows when it is wrong (pipe loop)", category="Design",
      outputs=[Port("report", "report"), Port("figure", "figure"), Port("metrics", "metrics")],
      params=[Param("mission", "text", "", help="the request in plain language (empty: the built-in example)"),
              Param("n_per_kind", "int", 20, minimum=1, maximum=200, help="controlled cases per kind"),
              Param("planner", "enum", "template", choices=["template", "llm"]),
              Param("render", "bool", False, help="Blender renders (needs bpy)"),
              Param("workers", "int", 4, minimum=1, maximum=32)],
      calls="pinneapple_systems.digital_twin.pipe_loop.run_mission (pinneapple_autolab.execute)",
      tags=("section:optimization", "agent", "digital twin", "fault diagnosis", "kalman", "verification"))
def pipe_twin(ctx, inputs, params):
    """Fit the twin's fast model, calibrate its alarms on healthy runs, tell a changed process from a wrong sensor from
    a model outside its domain on three events, ask the physics where the model is out of its domain, and measure the
    diagnosis on controlled cases (confusion matrix) with its trust report."""
    from pinneapple_systems.digital_twin.pipe_loop import DEFAULT_TEXT, run_mission
    run = run_mission(params["mission"] or DEFAULT_TEXT, out_dir=str(ctx.workdir / "pipe_twin"),
                      planner=params["planner"], render=bool(params["render"]), n_per_kind=int(params["n_per_kind"]),
                      workers=int(params["workers"]), log=ctx.log)
    if not run.ok:
        bad = next(e for e in run.evidence if not e.ok)
        raise StepFailed(f"{bad.tool}: {bad.error}")
    st = run.state
    b = st["benchmark"]["benchmark"]
    ctx.table("Confusion (rows: true, columns: diagnosed)", ["true"] + b["kinds"],
              [[k] + row for k, row in zip(b["kinds"], b["confusion"], strict=True)])
    ctx.summary(f"{st['trust']['decision']}: cause right in {100 * b['accuracy']:.0f} % of {b['n_cases']} "
                f"controlled cases; showcase {st['monitor']['correct']}/{st['monitor']['n']}")
    files = st["deliver"]["files"]
    return {"report": artifact(files["report"], "md"), "figure": artifact(files["confusion"], "png"),
            "metrics": {"decision": st["trust"]["decision"], "accuracy": b["accuracy"], "confusion": b["confusion"],
                        "isolation": b["isolation"], "cases_with_false_alarm": b["cases_with_false_alarm"]}}

"""Industrial preset nodes: run a ready-made, validated engineering case with your own parameters, and its
reference validation (library: :mod:`pinneapple_presets`)."""
from __future__ import annotations

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node

#: kept literal so registering the nodes imports nothing heavy; tests/test_presets.py checks it equals
#: pinneapple_presets.names()
PRESETS = ["plate_fin_heatsink_forced", "power_pcb_junction", "pressure_vessel_asme", "pump_npsh_cavitation",
           "shell_tube_cooler", "steam_pipe_insulation", "steel_footbridge_pedestrian", "water_distribution_network",
           "water_hammer_valve_closure"]
_TAGS = ("section:solvers", "industrial presets", "validated cases", "engineering")


@node("preset.run", "Industrial preset", category="Solve",
      inputs=[Port("params", "params", required=False, help="parameter values overriding the data-sheet defaults")],
      outputs=[Port("metrics", "metrics"), Port("report", "report")],
      params=[Param("preset", "enum", "water_hammer_valve_closure", choices=PRESETS, help="the engineering case"),
              Param("overrides", "dict", {}, help="input name -> value in SI units (see the preset data sheet)"),
              Param("fidelity", "enum", "solver", choices=["solver", "surrogate"],
                    help="the real solver, or the trained surrogate (flagged outside its training box)")],
      calls="pinneapple_presets.load(name).run(**overrides)", tags=_TAGS)
def run_preset(ctx, inputs, params):
    """Solve a ready-made industrial case (data sheet, solver, checks, decision) with your own inputs."""
    import pinneapple_presets as P
    over = dict(params.get("overrides") or {})
    over.update((inputs.get("params") or {}) if isinstance(inputs.get("params"), dict) else {})
    try:
        p = P.load(params["preset"])
        r = p.run(fidelity=params["fidelity"], **over)
    except (KeyError, ValueError, FileNotFoundError) as e:
        raise StepFailed(str(e)) from e
    for k, v in r.outputs.items():
        if isinstance(v, (int, float)):
            unit = next((o.unit for o in p.outputs if o.name == k), "")
            ctx.metric(k, v, f"{params['fidelity']} [{unit}]")
    ctx.table("Trust report", ["check", "status", "value", "limit"],
              [[c.name, c.status, c.value, c.limit] for c in r.trust.checks])
    ctx.summary(f"{p.title}: {r.decision['verdict']} (trust {r.trust.decisao}). " + "; ".join(r.decision["reasons"]))
    path = r.save(str(ctx.path("result.json")))
    return {"metrics": r.to_dict(), "report": {"path": path, "kind": "json"}}


@node("preset.validate", "Validate preset", category="Validate",
      outputs=[Port("metrics", "metrics")],
      params=[Param("preset", "enum", "water_hammer_valve_closure", choices=PRESETS, help="the engineering case")],
      calls="pinneapple_presets.load(name).validate()", tags=("section:validation", "industrial presets"))
def validate_preset(ctx, inputs, params):
    """Run the preset's reference cases (standard, benchmark, analytic, independent solver) and measure the error."""
    import pinneapple_presets as P
    v = P.load(params["preset"]).validate()
    ctx.table("References", ["case", "computed", "reference", "rel. error", "tol", "pass", "source"],
              [[r["case"], r["computed"], r["reference"], r["rel_error"], r["tol"], r["pass"], r["source"]]
               for r in v["references"]])
    ctx.metric("all_pass", v["all_pass"])
    return {"metrics": v}

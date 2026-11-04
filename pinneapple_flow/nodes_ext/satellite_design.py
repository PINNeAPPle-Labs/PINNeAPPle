"""The standard satellite-design workflow (:mod:`pinneapple_design.satellite.workflow`, #1207): from the mission (ground
sample distance, revisit, payload power, data per day, lifetime) to the sun-synchronous orbit, eclipse and J2 drift,
the end-of-life solar array and battery, the hot and cold thermal cases, the link budget, the attitude disturbance
torques with wheel and magnetorquer sizing, the lifetime under F10.7 scenarios and the mass closed at its least value
with margins; re-verified by a J2 orbit propagation and the margins at the end-of-life altitude."""
from __future__ import annotations

from ..registry import Param, Port, node


@node("satellite.design", "Satellite design: orbit, power, thermal, link, ADCS, lifetime; mass closed, re-verified",
      category="Design", outputs=[Port("metrics", "metrics")],
      params=[Param("mission", "dict", {}, help="overrides of the Mission fields (gsd_max_m, revisit_max_days, "
                                                  "payload_power_w, data_gbit_per_day, lifetime_years, ...)"),
              Param("altitude_min_km", "float", 400.0, minimum=200.0),
              Param("altitude_max_km", "float", 800.0, minimum=250.0),
              Param("altitude_step_km", "float", 25.0, minimum=1.0),
              Param("propagate_days", "float", 1.0, minimum=0.1, maximum=30.0, advanced=True)],
      calls="pinneapple_design.satellite.workflow.satellite_design_workflow")
def satellite_design_node(ctx, inputs, params):
    import numpy as np

    from pinneapple_design.satellite import satellite_design_workflow
    alts = tuple(float(x) for x in np.arange(float(params["altitude_min_km"]), float(params["altitude_max_km"]) + 1e-9,
                                             float(params["altitude_step_km"])))
    r = satellite_design_workflow(mission=dict(params.get("mission") or {}), altitudes_km=alts,
                                  propagate_days=float(params["propagate_days"]))
    ctx.table("altitude trade", ["h km", "mass kg", "GSD m", "revisit d", "eclipse", "life (high F10.7) y",
                                 "why not"],
              [[t["h_km"], t["mass_kg"], t["gsd_m"], t["revisit_days"], t["eclipse"], t["lifetime_high_years"],
                "; ".join(t["infeasible"]) or "feasible"] for t in r["trade"]])
    d = r.get("design")
    if d:
        ctx.metric("mass_kg", d["mass_kg"], "closed with structure, harness and system margin")
        ctx.metric("altitude_km", d["h_km"], f"sun-synchronous {d['i_deg']:.3f} deg")
        ctx.metric("solar_array_m2", d["solar_array"]["area_m2"], f"{d['solar_array']['p_bol_w']:.1f} W BOL")
        ctx.metric("battery_wh", d["battery"]["capacity_wh"], f"DoD {d['battery']['dod']:.0%}")
        ctx.metric("link_margin_db", d["link"]["margin_db"])
        ctx.metric("lifetime_high_flux_years", r["lifetime_years"]["high"])
        ctx.table("mass budget", ["subsystem", "kg"], [[k, v] for k, v in d["subsystems_kg"].items()])
    ctx.table("checks", ["check", "status", "detail"], [[c["code"], c["status"], c["detail"]] for c in r["checks"]])
    ctx.metric("verdict", r["verdict"], ", ".join(r["failed"]) or "every check passes")
    ctx.summary(r["summary"])
    return {"metrics": ctx.report["metrics"]}


def build_flow(**params):
    from ..planner.build import FlowBuilder
    title = "Satellite design: orbit, power, thermal, link, ADCS and lifetime, mass closed and re-verified"
    fb = FlowBuilder(title, "From the mission (GSD, revisit, payload power, data per day, lifetime): sun-synchronous "
                            "orbit, eclipse and J2 drift, end-of-life array and battery, hot and cold thermal cases, "
                            "link budget, disturbance torques with wheel and magnetorquer sizing, lifetime under F10.7 "
                            "scenarios, the mass closed at its least value with margins, re-verified by a J2 orbit "
                            "propagation and the end-of-life margins (pinneapple_design.satellite.workflow).")
    fb.add("sat", "satellite.design", dict(params), label="Satellite design")
    fb.add("report", "export.report", {"title": title}, label="Report")
    fb.connect("sat", "metrics", "report", "items")
    return fb


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="satellite.design", title="Satellite design (standard workflow)", domain="space",
        keywords={"satellite": 2, "satélite": 2, "cubesat": 3, "smallsat": 3, "spacecraft": 2, "espaçonave": 2,
                  "sun-synchronous": 3, "heliossíncrona": 3, "sol-síncrona": 3, "link budget": 3,
                  "orçamento de enlace": 3, "power budget": 2, "orçamento de potência": 2, "mass budget": 2,
                  "orçamento de massa": 2, "eclipse": 1, "reaction wheel": 3, "roda de reação": 3,
                  "magnetorquer": 3, "magnetotorquer": 3, "orbit decay": 2, "decaimento orbital": 2,
                  "f10.7": 2, "adcs": 3, "earth observation": 2, "observação da terra": 2},
        min_score=3, nodes=("satellite.design",), build=lambda text, det: build_flow(),
        inputs=("the mission: GSD, revisit, payload power and data per day, lifetime",
                "the ground station and the radio band", "the thermal limits and the margins"),
        uncited="the method follows Wertz & Larson, Space Mission Analysis and Design (1999), and uses NRLMSIS 2.0 "
                "(Emmert et al. 2021) in the library modules; they are not in the flow reference file yet",
        description="pinneapple_design.satellite.workflow.satellite_design_workflow"))


_register_capabilities()

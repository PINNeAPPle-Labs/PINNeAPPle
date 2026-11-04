"""The standard rover workflow (:mod:`pinneapple_simulation.vehicle.mission_design`, #1200): planet, terrain and
mission -> mobility (grade, step, tip-over, sinkage with Wong-Reece wheels), power (drive, solar array over the day,
battery for the night) and thermal (survival heater) -> the lightest rover within the margins -> the traverse
simulated on the mission profile, and the model checked on Sojourner's published figures."""
from __future__ import annotations

from ..registry import Param, Port, node

MISSION = {"distance_m": 100.0, "payload_kg": 2.0, "speed_m_s": 0.01, "drive_hours": 2.0, "max_slope_deg": 15.0,
           "rock_m": 0.10, "max_sols": 30, "max_sinkage_m": 0.03}


@node("rover.mission_design", "Rover design for a mission (mobility, power, thermal, traverse re-verified)",
      category="Solvers", tags=("section:optimization", "rover", "rocker-bogie", "solar", "mission"),
      outputs=[Port("metrics", "metrics")],
      params=[Param("planet", "enum", "mars", choices=["mars", "moon"]),
              Param("soil", "str", "", help="database soil ('' : the planet's default)"),
              Param("mission", "dict", MISSION, help="distance, payload, speed, drive hours per sol, slope, rock, "
                                                      "sols, sinkage limit"),
              Param("wheels", "int", 6, minimum=3, maximum=12),
              Param("suspension", "enum", "rocker-bogie", choices=["rocker-bogie", "rigid"]),
              Param("max_slip", "float", 0.3, minimum=0.05, maximum=0.8),
              Param("radius_bounds", "list", [0.05, 0.20], help="wheel radius range (m)"),
              Param("width_bounds", "list", [0.04, 0.15], help="wheel width range (m)"),
              Param("grid", "int", 6, minimum=2, maximum=12, help="grid points per wheel dimension"),
              Param("cell_efficiency", "float", 0.28, minimum=0.01, maximum=0.5),
              Param("stability_factor", "float", 1.5, minimum=1.0, help="tip-over angle / mission slope"),
              Param("assumptions", "dict", {}, help="override the mass, power and thermal assumptions",
                    advanced=True),
              Param("seed", "int", 0, help="slope profile of the traverse")],
      calls="pinneapple_simulation.vehicle.mission_design.rover_workflow (size_rover, grade_capability, "
            "drive_power, traverse, published_rover_check; terramechanics.WongReeceModel)")
def rover_mission_design(ctx, inputs, params):
    """Run the workflow; report the design, its margins, the traverse and the checks."""
    from pinneapple_simulation.vehicle.mission_design import rover_workflow
    r = rover_workflow(planet=str(params["planet"]), soil=str(params.get("soil") or "") or None,
                       mission=dict(params.get("mission") or {}), n_wheels=int(params["wheels"]),
                       suspension=str(params["suspension"]), max_slip=float(params["max_slip"]),
                       radius_bounds=tuple(params["radius_bounds"]), width_bounds=tuple(params["width_bounds"]),
                       grid=int(params["grid"]), cell_efficiency=float(params["cell_efficiency"]),
                       stability_factor=float(params["stability_factor"]),
                       assumptions=dict(params.get("assumptions") or {}), seed=int(params["seed"]))
    d = r.get("design")
    if d:
        ctx.metric("mass_kg", d["mass_kg"], "total, fixed point of the mass build-up")
        ctx.metric("wheel_radius_m", d["wheel"]["radius_m"])
        ctx.metric("wheel_width_m", d["wheel"]["width_m"])
        ctx.metric("array_m2", d["array_m2"], "sized for the energy of a sol x margin and the slope drive")
        ctx.metric("battery_Wh", d["battery_Wh"], "night + the traverse's measured dawn / dusk deficit")
        ctx.metric("grade_deg", d["grade_deg"])
        ctx.metric("step_m", d["step_m"])
        ctx.metric("heater_W", d["heater_W"])
        ctx.table("mass build-up", ["part", "kg"], [[k, round(v, 4)] for k, v in d["parts_kg"].items()])
    tr = r.get("traverse") or {}
    for k in ("distance_m", "sols", "soc_min", "slip_max", "energy_closure"):
        if k in tr:
            ctx.metric(f"traverse.{k}", tr[k], "time-stepped mission (rover_workflow.traverse)")
    pub = r["published"]
    ctx.metric("sojourner.peak_power_W", pub["peak_power_W"], f"reported {pub['reported_peak_W']} W")
    ctx.table("checks", ["check", "status", "detail"], [[c["code"], c["status"], c["detail"]] for c in r["checks"]])
    ctx.metric("verdict", r["verdict"], ", ".join(r["failed"]) or "every check passes")
    ctx.summary((f"{d['mass_kg']:.2f} kg rover, wheels R {d['wheel']['radius_m']:.3f} m x {d['wheel']['width_m']:.3f}"
                 f" m, array {d['array_m2']:.3f} m^2, battery {d['battery_Wh']:.0f} W h; traverse "
                 f"{tr.get('distance_m', 0):.0f} m in {tr.get('sols', 0):.2f} sols" if d else "no feasible rover") +
                ("; fail: " + ", ".join(r["failed"]) if r["failed"] else "; verified"))
    return {"metrics": ctx.report["metrics"]}


def build_flow(**params):
    from ..planner.build import FlowBuilder
    title = "Rover: mission to mobility, power and thermal, rover design, traverse re-verified"
    fb = FlowBuilder(title, "Planet, terrain and mission to the lightest rover: grade and step climbing with "
                            "Wong-Reece wheels, tip-over, sinkage, drive power, solar array over the day, battery "
                            "for the night and a survival heater; the mission traverse time-stepped with the battery "
                            "state of charge, and Sojourner's published power reproduced "
                            "(pinneapple_simulation.vehicle.mission_design).")
    fb.add("rover", "rover.mission_design", dict(params), label="Rover mission design")
    fb.add("report", "export.report", {"title": title}, label="Report")
    fb.connect("rover", "metrics", "report", "items")
    return fb


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="rover.mission_design", title="Rover design for a planetary mission (standard workflow)",
        domain="vehicles",
        keywords={"rover design": 4, "projeto de rover": 4, "projetar um rover": 4, "dimensionar o rover": 4,
                  "planetary rover": 3, "rover planetário": 3, "mars rover": 3, "rover marciano": 3,
                  "lunar rover": 3, "rover lunar": 3, "rocker-bogie": 3, "rocker bogie": 3, "sojourner": 2,
                  "solar array sizing": 2, "dimensionar o painel solar": 2, "survival heater": 2,
                  "aquecedor de sobrevivência": 2, "traverse": 1, "travessia": 1, "mission": 1, "missão": 1},
        min_score=3, nodes=("rover.mission_design",), build=lambda text, det: build_flow(),
        inputs=("the planet and its soil", "the terrain (slope, rock height)",
                "the mission (distance, payload, speed, drive hours, sols)"),
        uncited="the planet constants and Sojourner's figures carry their sources in "
                "pinneapple_simulation.vehicle.mission_design (PLANETS, SOJOURNER); they are not in the flow "
                "reference file yet",
        description="pinneapple_simulation.vehicle.mission_design.rover_workflow"))


_register_capabilities()

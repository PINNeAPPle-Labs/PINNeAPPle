"""The standard multirotor workflow (:mod:`pinneapple_simulation.multirotor.workflow`, #1201): BEMT rotor checked
against momentum theory and the APC 10 x 4.7 published data, motor and battery models, propeller / pack / motor /
arm sizing for a payload and endurance at a thrust-to-weight margin, LQR control of the 6-DOF quadrotor (margins,
step, gust), the drone flown on a mission until its battery is empty, and a published endurance (DJI Phantom 4)."""
from __future__ import annotations

from ..registry import Param, Port, node

MISSION = [{"name": "hover", "target": [0, 0, 0], "duration": 30.0, "speed": 1.0},
           {"name": "climb", "target": [0, 0, 20], "duration": 30.0, "speed": 2.0},
           {"name": "cruise", "target": [100, 0, 20], "duration": 60.0, "speed": 5.0},
           {"name": "return", "target": [0, 0, 20], "duration": 60.0, "speed": 5.0},
           {"name": "descend", "target": [0, 0, 0], "duration": 30.0, "speed": 1.5}]


@node("drone.design", "Multirotor design (rotor, motor, battery, sizing, 6-DOF control, endurance flown)",
      category="Solvers", tags=("section:optimization", "drone", "quadrotor", "bemt", "lqr"),
      outputs=[Port("metrics", "metrics")],
      params=[Param("payload_kg", "float", 0.3, minimum=0.0),
              Param("endurance_min", "float", 15.0, minimum=1.0, help="hover endurance the sizing must reach"),
              Param("thrust_to_weight", "float", 2.0, minimum=1.2, maximum=5.0),
              Param("mission", "list", MISSION, help="legs [{name, target [x, y, z] m, duration s, speed m/s}], "
                                                     "repeated until the battery is empty"),
              Param("gust_m_s", "float", 3.0, minimum=0.0, help="wind gust of the disturbance test"),
              Param("calibrate_polar", "bool", True, help="use the polar fitted to the APC static data"),
              Param("published", "bool", True, help="also hover the Phantom 4 case"),
              Param("mass_model", "dict", {}, help="override the mass and drive coefficients", advanced=True)],
      calls="pinneapple_simulation.multirotor.workflow.drone_workflow (rotor.bemt, rotor.calibrate_airfoil, "
            "power.Motor, power.Battery, quad.lqr_hover, quad.simulate, quad.fly_mission)")
def drone_design(ctx, inputs, params):
    """Run the workflow; report the rotor checks, the design, the control checks and the measured endurance."""
    from pinneapple_simulation.multirotor.workflow import drone_workflow
    r = drone_workflow(payload_kg=float(params["payload_kg"]), endurance_min=float(params["endurance_min"]),
                       thrust_to_weight=float(params["thrust_to_weight"]), mission=list(params.get("mission") or [])
                       or None, gust_m_s=float(params["gust_m_s"]), calibrate_polar=bool(params["calibrate_polar"]),
                       published=bool(params["published"]), mass_model=dict(params.get("mass_model") or {}))
    rot = r["rotor"]
    ctx.metric("rotor.held_out_cT_max_rel", rot["held_out_errors"]["cT_max_rel"], rot["source"])
    ctx.metric("rotor.held_out_cP_max_rel", rot["held_out_errors"]["cP_max_rel"], rot["source"])
    d = r.get("design")
    if d:
        for k in ("mass_kg", "prop_diameter_in", "cells", "capacity_ah", "motor_kv", "arm_m", "hover_power_W",
                  "figure_of_merit", "thrust_to_weight", "estimated_endurance_min"):
            ctx.metric(f"design.{k}", d[k])
        ctx.table("mass build-up", ["part", "kg"], [[k, round(v, 4)] for k, v in d["parts_kg"].items()])
    if r.get("mission"):
        ctx.metric("mission.endurance_min", r["mission"]["endurance_s"] / 60, "flown until the battery was empty")
        ctx.metric("mission.max_tracking_error_m", r["mission"]["max_tracking_error_m"])
    if r.get("control"):
        for mg in r["control"]["margins"]:
            ctx.metric(f"control.{mg['channel']}.phase_margin_deg", mg["phase_margin"], "motor lag included")
    if r.get("published"):
        ctx.metric("published.phantom4_hover_min", r["published"]["hover_minutes"],
                   f"published {r['published']['published_minutes']} min")
    ctx.table("checks", ["check", "status", "detail"], [[c["code"], c["status"], c["detail"]] for c in r["checks"]])
    ctx.metric("verdict", r["verdict"], ", ".join(r["failed"]) or "every check passes")
    ctx.summary((f"{d['mass_kg']:.3f} kg quad, {d['prop_diameter_in']:.1f} in props, {d['cells']}S "
                 f"{d['capacity_ah']:g} A h, Kv {d['motor_kv']:.0f}; flown {r['mission']['endurance_s'] / 60:.1f} min"
                 if d and r.get("mission") else "no feasible design") +
                ("; fail: " + ", ".join(r["failed"]) if r["failed"] else "; verified"))
    return {"metrics": ctx.report["metrics"]}


def build_flow(**params):
    from ..planner.build import FlowBuilder
    title = "Drone: rotor, motor and battery, sizing, 6-DOF control, endurance flown"
    fb = FlowBuilder(title, "A quadrotor for a payload and endurance: BEMT rotor checked against momentum theory and "
                            "the APC 10 x 4.7 published data, motor and LiPo models, propeller / pack / motor / arm "
                            "sizing at a thrust-to-weight margin, LQR control of the 6-DOF dynamics (margins, step, "
                            "gust), the mission flown until the battery is empty, and the Phantom 4's published "
                            "endurance (pinneapple_simulation.multirotor.workflow).")
    fb.add("drone", "drone.design", dict(params), label="Drone design")
    fb.add("report", "export.report", {"title": title}, label="Report")
    fb.connect("drone", "metrics", "report", "items")
    return fb


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="drone.design", title="Multirotor / drone design and endurance (standard workflow)", domain="vehicles",
        keywords={"drone": 3, "drones": 3, "quadcopter": 4, "quadricóptero": 4, "quadrotor": 4, "quadrirrotor": 4,
                  "multirotor": 4, "multirrotor": 4, "uav": 2, "vant": 2, "propeller sizing": 3,
                  "dimensionar a hélice": 3, "hélice": 1, "propeller": 1, "flight time": 2, "tempo de voo": 2,
                  "endurance": 1, "autonomia de voo": 3, "blade element": 2, "elemento de pá": 2, "bemt": 3,
                  "lipo": 1, "kv": 1},
        min_score=3, nodes=("drone.design",), build=lambda text, det: build_flow(),
        inputs=("the payload", "the endurance target", "the thrust-to-weight margin", "the mission legs"),
        uncited="the workflow cites Leishman (2006) for BEMT, the UIUC propeller database (Brandt & Selig) for the "
                "APC data, Drela (2007) for the motor model and DJI's published Phantom 4 figures in the library "
                "modules; they are not in the flow reference file yet",
        description="pinneapple_simulation.multirotor.workflow.drone_workflow"))


_register_capabilities()

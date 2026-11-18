"""Suspension design and sizing as a flow node and a planner capability: one corner, three models, one loop.

The node ``design.suspension_elastokinematics`` runs :func:`pinneapple_design.suspension.study.suspension_study` — the
hardpoints, the arm's tube and the bushes searched for the lightest corner that meets the structural, kinematic and
elastokinematic constraints, with the finite-element check of the winner — and reports its numbers, its own checks and
the limits of what it answers.

The capability ``suspension.design`` proposes that flow when a goal names a suspension problem ("size the wishbones,
camber gain in bump, bump steer, compliance under lateral load"), in English or Portuguese.
"""
from __future__ import annotations

from ..registry import Param, Port, node

__all__ = ["suspension_elastokinematics"]

_SUSPENSION = ("suspension", "suspens", "wishbone", "double wishbone", "bandeja", "hardpoint", "camber",
               "cambagem", "bump steer", "steering arm", "elastokinematic", "elastocinemátic", "elastocinematic",
               "corner mass", "unsprung")
_ROLES = ("design", "size", "sizing", "dimension", "otimiz", "optimi", "minimiz", "lightest", "mais leve",
          "structural", "estrutural", "fem", "multibody", "multicorpo")


@node("design.suspension_elastokinematics",
      "Suspension corner: hardpoints, arm structure and bush compliance in one loop", category="Design",
      outputs=[Port("metrics", "metrics")],
      params=[Param("corner_mass_kg", "float", 350.0, minimum=50.0,
                    help="sprung mass carried by the corner [kg] (the load cases scale with it)"),
              Param("bump_g", "float", 3.0, minimum=1.0, maximum=10.0,
                    help="the vertical load case at the contact patch [g] (a pothole or kerb)"),
              Param("toe_target_deg_per_kn", "float", 0.45, minimum=0.01, maximum=5.0,
                    help="how much the wheel may steer under 1 kN of lateral force [deg/kN]"),
              Param("budget", "int", 28, minimum=4, maximum=200,
                    help="rounds of the pattern search (each one tries every design variable both ways)"),
              Param("mesh", "enum", "coarse", choices=["coarse", "fine"],
                    help="density of the finite-element check of the winning arm")],
      calls="pinneapple_design.suspension.study.suspension_study (design.suspension_design, "
            "structure.fem_lower_arm, kinematics.kinematic_sweep, compliance.wheel_compliance)")
def suspension_elastokinematics(ctx, inputs, params):
    """Run the suspension study and report its numbers, its checks and its limits."""
    from pinneapple_design.suspension.study import suspension_study

    out = suspension_study(corner_mass_kg=float(params["corner_mass_kg"]), bump_g=float(params["bump_g"]),
                           toe_target_deg_per_kn=float(params["toe_target_deg_per_kn"]),
                           budget=int(params["budget"]), mesh=str(params["mesh"]))
    for name, value in out["metrics"].items():
        ctx.metric(name, float(value))
    for check in out["checks"]:
        ctx.metric(f"check.{check['code']}", 1.0 if check["status"] == "pass" else 0.0,
                   f"{check['status']}: {check['detail']}")
    ctx.table("checks", ["check", "status", "value", "what it measured"],
              [[c["code"], c["status"], c.get("value"), c["detail"]] for c in out["checks"]])
    ctx.table("design", ["variable", "value"], [[k, v] for k, v in out["design"].items()])
    ctx.table("limits of the model", ["not answered here"], [[x] for x in out["limits"]])
    ctx.metric("verdict", out["verdict"])
    ctx.summary(out["verdict"])
    return {"metrics": ctx.report["metrics"]}


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="suspension.design", title="Suspension corner: sizing, kinematics and bush compliance", domain="vehicle",
        keywords={**{k: 2 for k in _SUSPENSION}, **{k: 1 for k in _ROLES}},
        requires_all=(_SUSPENSION[:3] + _SUSPENSION[4:9],),
        min_score=3,
        nodes=("design.suspension_elastokinematics",),
        downstream=("export.report",),
        template="vehicle/suspension_elastokinematics",
        inputs=("the corner's sprung mass", "the vertical load case (a pothole or kerb in g)",
                "how much steer the bushes may allow under 1 kN"),
        uncited="no literature reference registered for this capability: the model states its own limits",
        description="the hardpoints, the arm's tube and the bushes searched together for the lightest corner that "
                    "meets the structural margin, the camber-gain band, the bump-steer limit and the steer compliance, "
                    "with the finite-element check of the winning arm",
        example="suspension_elastokinematics"))


_register_capabilities()

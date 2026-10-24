"""PINNeAPPle Autonomous Physics Lab, rover demo: *the rover that learns the planet*.

A mission in plain language ("drive to (37, 15) without getting stuck, slip at most 40 %, minimum energy") runs the
lab loop on a synthetic site whose truth is known: plan a route on the orbital map's soils, drive it with the
multibody VIPER on the true terrain (Wong-Reece wheels, several soils), stop when the measured slip exceeds the
limit, learn the soil from what the wheels measured (iterated EKF on sinkage and torque), replan on the learned
belief with its uncertainty, drive again, and verify everything in a trust report (including the error of the
recovered soil against the known truth).

    from pinneapple_simulation.rover.autolab import RoverMission, demo_site
    res = RoverMission(demo_site(), "Drive the rover to (37, 15) without getting stuck, slip at most 40 %, "
                                    "minimum energy").run(out_dir="out")
    res.report.decisao, [r.meta["ended"] for r in res.runs]

Modules: :mod:`.site` (synthetic site with hidden truth), :mod:`.wheel_models` (operating points per grade; analytic
vs surrogate vs measured), :mod:`.estimation` (sensors, soil belief), :mod:`.planner` (risk-aware routes),
:mod:`.mission` (parser, tools, the loop), :mod:`.verify` (trust report), :mod:`.report`, :mod:`.visuals`
(matplotlib), :mod:`.render` (Blender Cycles). Guide: ``docs/guides/rover_learns_the_planet.md``.
"""
from .estimation import PARAMS, Sensors, SoilBelief, measure
from .mission import (
           TOOLS,
           MissionResult,
           MissionSpec,
           RoverMission,
           RuleParser,
           parse_mission,
           traction_monitor,
)
from .planner import RiskPlanner, Route
from .site import Mound, Patch, SoilUnit, SyntheticSite, demo_site, soft_sand
from .wheel_models import GradeTable, compare_contact_models, grade_table, max_climbable_grade

__all__ = ["SyntheticSite", "SoilUnit", "Patch", "Mound", "demo_site", "soft_sand", "GradeTable", "grade_table",
           "max_climbable_grade", "compare_contact_models", "Sensors", "measure", "SoilBelief", "PARAMS",
           "RiskPlanner", "Route", "RoverMission", "MissionSpec", "MissionResult", "RuleParser", "parse_mission",
           "TOOLS", "traction_monitor"]

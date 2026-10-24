"""The self-optimizing engineer: cool an overheating part on a board with a heat sink and the enclosure fans.

    from pinneapple_design.thermal.board_cooling import run_mission, DEFAULT_TEXT
    run = run_mission(DEFAULT_TEXT, out_dir="out")       # read -> baseline -> still air -> sample -> learn ->
                                                         # optimise + confirm -> verify -> deliver

Modules: ``mission`` (request, design space), ``simulate`` (coupled 3-D board + heat sink reference), ``learn``
(POD + Gaussian-process surrogate of the junction temperatures and the board map, hold-out), ``optimize`` (constrained
search, confirm-on-physics loop), ``verify`` (fine-grid solve, trust report), ``scene`` (3-D), ``report``
(figures, renders, report), ``agent`` (controlled tools and planners).
"""
from .agent import run_mission, template_plan, toolbox
from .learn import CoolingSurrogate, holdout
from .mission import (
           DEFAULT_TEXT,
           CoolingDesign,
           CoolingMission,
           DesignSpace,
           default_mission,
           parse_mission,
)
from .optimize import design_loop, optimize
from .simulate import FIDELITY, MAP_SHAPE, simulate, simulate_many
from .verify import verify

__all__ = ["run_mission", "toolbox", "template_plan", "CoolingSurrogate", "holdout", "DEFAULT_TEXT", "CoolingDesign",
           "CoolingMission", "DesignSpace", "default_mission", "parse_mission", "design_loop", "optimize",
           "FIDELITY", "MAP_SHAPE", "simulate", "simulate_many", "verify"]

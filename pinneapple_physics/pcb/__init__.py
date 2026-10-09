"""Populated-PCB thermal model (the engine of the PCB Hotspot app).

    from pinneapple_physics.pcb import Board, Component, solve, evaluate, what_if, calibrate
    out = evaluate(board, components, environment)      # junction temperatures, maps, heat paths, checks

``solver``: layered 3D finite-volume board with JEDEC two-resistor packages; ``engine``: evaluation, what-if
studies and calibration to measurements with Ensemble Kalman Inversion.
"""
from pinneapple_physics.closed_form.pcb_thermal import Environment

from .engine import calibrate, evaluate, model_scope, what_if
from .solver import Board, Component, overlaps, solve, validate

__all__ = ["Board", "Component", "Environment", "solve", "validate", "overlaps", "evaluate", "what_if", "calibrate",
           "model_scope"]

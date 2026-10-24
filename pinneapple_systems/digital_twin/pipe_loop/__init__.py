"""The digital twin that knows when it is wrong (Autonomous Physics Lab, demo 3): a pumped, heated water loop with
an elbow and a T junction, a fast model of its sensors, and a diagnosis that tells a changed process from a wrong
sensor from a model outside its domain -- with physics residuals, a Kalman filter and the domain of validity.

    from pinneapple_systems.digital_twin.pipe_loop import run_mission, run_benchmark, Loop, Scenario, run_plant
    run = run_mission(out_dir="out")                       # build -> calibrate -> monitor -> ask the physics ->
                                                           # benchmark (confusion matrix) -> trust -> deliver

Modules: ``plant`` (loop, reference simulator, controlled faults), ``diagnose`` (residuals, fast model, Kalman,
decision), ``benchmark`` (controlled cases, confusion matrix), ``scene`` (3-D), ``report``, ``agent``.
"""
from .agent import DEFAULT_TEXT, SHOWCASE, run_mission, template_plan, toolbox
from .benchmark import KINDS, make_cases, run_benchmark
from .diagnose import RESIDUALS, Diagnoser, LossTracker, Residuals, Surrogate
from .plant import (
           INPUTS,
           NORMAL_BOX,
           SENSORS,
           SIGMA,
           Faults,
           Loop,
           Scenario,
           quasi_steady_valid,
           run_plant,
           steady_state,
)

__all__ = ["run_mission", "toolbox", "template_plan", "DEFAULT_TEXT", "SHOWCASE", "KINDS", "make_cases", "run_benchmark",
           "RESIDUALS", "Diagnoser", "LossTracker", "Residuals", "Surrogate", "INPUTS", "NORMAL_BOX", "SENSORS",
           "SIGMA", "Faults", "Loop", "Scenario", "quasi_steady_valid", "run_plant", "steady_state"]

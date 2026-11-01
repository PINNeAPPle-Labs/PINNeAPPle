"""Control engineering for any dynamic model: linear analysis, actuators and loop architecture, closed-loop
simulation, tuning and robustness (simulation only, no path to real equipment).

* :mod:`.analysis`: trim, linearisation, stability (growth rates, doubling times), controllability / observability,
  transfer functions, gain / phase / delay margins and peak sensitivity, step-response metrics.
* :mod:`.loop`:
  - :class:`Plant`, and :class:`Actuator` (range, rate limit, lag, dead band, delay, integrating);
  - controllers: :class:`PID` (derivative on measurement or rate, filter, back-calculation anti-windup,
    setpoint weighting, feed-forward), :class:`Cascade`, :class:`StateFeedback`;
  - :func:`simulate_loop`, :func:`loop_eigenvalues` (the complete discrete closed loop around an equilibrium)
    and :func:`compare_architectures` (which variable the controller commands).
* :mod:`.tuning`:
  - FOPDT fit, SIMC, IMC, Ziegler-Nichols;
  - PID pole placement for second-order plants, stable or not;
  - LQR;
  - tuning by optimisation under constraints, and Monte Carlo robustness.

Domain plants built on it: :mod:`pinneapple_simulation.naval` (submarine depth-keeping, ship roll stability).
"""
from .analysis import (
    controllability,
    equilibrium,
    linearize,
    loop_margins,
    observability,
    simulate_linear_step,
    stability,
    step_metrics,
    transfer_function,
)
from .loop import (
    PID,
    Actuator,
    Cascade,
    Plant,
    StateFeedback,
    compare_architectures,
    loop_eigenvalues,
    simulate_loop,
)
from .tuning import (
    dlqr,
    fit_fopdt,
    imc_pid,
    lqr,
    place_pid_second_order,
    robustness,
    simc_pid,
    tune_by_optimization,
    ultimate_gain,
    ziegler_nichols,
)

__all__ = [
    "equilibrium", "linearize", "stability", "controllability", "observability", "transfer_function",
    "loop_margins", "step_metrics", "simulate_linear_step",
    "Plant", "Actuator", "PID", "Cascade", "StateFeedback", "simulate_loop", "loop_eigenvalues",
    "compare_architectures",
    "fit_fopdt", "simc_pid", "imc_pid", "ziegler_nichols", "ultimate_gain", "place_pid_second_order", "lqr", "dlqr",
    "tune_by_optimization", "robustness",
]

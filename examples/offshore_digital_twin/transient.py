"""Research-only lumped pressure relaxation experiment.

This is a deliberately reduced-order 0D dynamic system, NOT a 1D conservation
PDE or a well-control simulator. The relaxation time is synthetic and cannot
be interpreted as real hydraulic response without calibration.
"""
from dataclasses import dataclass
from math import exp, isfinite
from offshore_twin import WellConfig, evaluate

@dataclass(frozen=True)
class TransientConfig:
    dt_s: float = 1.0
    steps: int = 120
    relaxation_s: float = 20.0
    initial_pressure_pa: float | None = None

    def validate(self):
        if not (isfinite(self.dt_s) and self.dt_s > 0):
            raise ValueError("dt_s must be finite and positive")
        if not isinstance(self.steps, int) or not 1 <= self.steps <= 100000:
            raise ValueError("steps must be an integer in [1,100000]")
        if not (isfinite(self.relaxation_s) and self.relaxation_s > 0):
            raise ValueError("relaxation_s must be finite and positive")
        if self.initial_pressure_pa is not None and not isfinite(self.initial_pressure_pa):
            raise ValueError("initial pressure must be finite")

def simulate(well: WellConfig, transient: TransientConfig = TransientConfig()):
    """Integrate dp/dt=(p_equilibrium-p)/tau exactly for piecewise constant target."""
    transient.validate()
    equilibrium = evaluate(well)["bottomhole_circulating_pressure_pa"]
    pressure = equilibrium if transient.initial_pressure_pa is None else transient.initial_pressure_pa
    rows = [{"time_s": 0.0, "bottomhole_pressure_pa": pressure,
             "equilibrium_pa": equilibrium}]
    decay = exp(-transient.dt_s / transient.relaxation_s)
    for i in range(1, transient.steps + 1):
        pressure = equilibrium + (pressure - equilibrium) * decay
        rows.append({"time_s": i * transient.dt_s,
                     "bottomhole_pressure_pa": pressure,
                     "equilibrium_pa": equilibrium})
    return rows

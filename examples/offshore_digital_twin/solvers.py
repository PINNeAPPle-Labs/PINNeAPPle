"""Research-only minimal component solvers, not certified engineering models."""
from dataclasses import dataclass
from math import pi, isfinite
from offshore_twin import WellConfig, evaluate

@dataclass(frozen=True)
class DrillingState:
    rpm: float = 90.0
    torque_nm: float = 12000.0
    weight_on_bit_n: float = 100000.0

def drillstring_power(state: DrillingState):
    if any(not isfinite(x) or x < 0 for x in vars(state).values()):
        raise ValueError("Drilling inputs must be finite and nonnegative")
    return {"angular_speed_rad_s": state.rpm*2*pi/60,
            "rotational_power_w": state.torque_nm*state.rpm*2*pi/60,
            "weight_on_bit_n": state.weight_on_bit_n,
            "scope": "kinematic power identity only"}

def riser_static_extension(length_m, tension_n, young_pa, area_m2):
    if any(not isfinite(x) or x <= 0 for x in (length_m,tension_n,young_pa,area_m2)):
        raise ValueError("Inputs must be positive and finite")
    return {"extension_m": tension_n*length_m/(young_pa*area_m2),
            "scope": "linear axial bar only; not riser bending/waves"}

def pressure_snapshot(well: WellConfig):
    return evaluate(well)

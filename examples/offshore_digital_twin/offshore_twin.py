"""Educational offshore well hydraulics twin seed.

Not a certified well-control simulator. No BOP automation or operational advice.
SI units throughout. Single-phase incompressible, lumped quasi-steady pressure model.
"""
from dataclasses import dataclass
from math import pi, isfinite

G = 9.80665

@dataclass(frozen=True)
class WellConfig:
    depth_m: float = 3000.0
    water_depth_m: float = 1000.0
    mud_density_kg_m3: float = 1200.0
    flow_m3_s: float = 0.025
    drillpipe_id_m: float = 0.10
    drillpipe_od_m: float = 0.127
    hole_diameter_m: float = 0.216
    friction_factor: float = 0.025
    pore_pressure_pa: float = 30e6
    fracture_pressure_pa: float = 45e6

    def validate(self):
        values = vars(self)
        if any(not isfinite(v) for v in values.values()):
            raise ValueError("Non-finite configuration")
        if self.depth_m <= 0 or self.water_depth_m < 0 or self.water_depth_m > self.depth_m:
            raise ValueError("Invalid depths")
        if not (0 < self.drillpipe_id_m < self.drillpipe_od_m < self.hole_diameter_m):
            raise ValueError("Invalid diameters")
        if self.mud_density_kg_m3 <= 0 or self.flow_m3_s < 0 or self.friction_factor < 0:
            raise ValueError("Invalid hydraulic parameters")
        if not 0 < self.pore_pressure_pa < self.fracture_pressure_pa:
            raise ValueError("Invalid pressure envelope")

def evaluate(config: WellConfig) -> dict:
    """Return a reproducible simplified pressure envelope at measured depth.

    Assumes vertical well, uniform fluid, constant friction factor, no cuttings,
    no thermal effects, no transients and atmospheric surface annulus pressure.
    """
    config.validate()
    a_pipe = pi * config.drillpipe_id_m**2 / 4
    a_annulus = pi * (config.hole_diameter_m**2 - config.drillpipe_od_m**2) / 4
    v_pipe = config.flow_m3_s / a_pipe
    v_annulus = config.flow_m3_s / a_annulus
    d_h_annulus = config.hole_diameter_m - config.drillpipe_od_m
    rho = config.mud_density_kg_m3
    dp_pipe = config.friction_factor * config.depth_m / config.drillpipe_id_m * rho * v_pipe**2 / 2
    dp_annulus = config.friction_factor * config.depth_m / d_h_annulus * rho * v_annulus**2 / 2
    hydrostatic = rho * G * config.depth_m
    bottomhole = hydrostatic + dp_annulus
    return {
        "hydrostatic_pa": hydrostatic,
        "pipe_friction_pa": dp_pipe,
        "annulus_friction_pa": dp_annulus,
        "pump_differential_pa": dp_pipe + dp_annulus,
        "bottomhole_circulating_pressure_pa": bottomhole,
        "equivalent_circulating_density_kg_m3": bottomhole / (G * config.depth_m),
        "pore_margin_pa": bottomhole - config.pore_pressure_pa,
        "fracture_margin_pa": config.fracture_pressure_pa - bottomhole,
        "within_pressure_window": config.pore_pressure_pa <= bottomhole <= config.fracture_pressure_pa,
        "model_validity": "simplified_vertical_single_phase_quasisteady",
    }

"""Five educational, deterministic SI-unit digital-twin reference models.

These are analytical/lumped research seeds, NOT operational or validated twins.
"""
from dataclasses import dataclass, asdict
from math import exp, pi, isfinite

G = 9.80665
SIGMA = 5.670374419e-8

def _check(**kwargs):
    if any(not isfinite(v) or v <= 0 for v in kwargs.values()):
        raise ValueError("All physical parameters must be finite and positive")

@dataclass(frozen=True)
class FPSO:
    oil_kg_s: float = 25.
    water_kg_s: float = 15.
    gas_kg_s: float = 5.
    oil_density_kg_m3: float = 850.
    water_density_kg_m3: float = 1000.
    separator_liquid_capacity_m3_s: float = .06

    def solve(self):
        _check(**asdict(self))
        liquid = self.oil_kg_s/self.oil_density_kg_m3 + self.water_kg_s/self.water_density_kg_m3
        return {"liquid_flow_m3_s":liquid,
                "water_cut_mass_fraction":self.water_kg_s/(self.oil_kg_s+self.water_kg_s),
                "separator_utilization":liquid/self.separator_liquid_capacity_m3_s,
                "mass_balance_residual_kg_s":(self.oil_kg_s+self.water_kg_s+self.gas_kg_s)-sum((self.oil_kg_s,self.water_kg_s,self.gas_kg_s))}

@dataclass(frozen=True)
class BlastFurnace:
    fuel_kg_s: float = 2.
    fuel_lhv_j_kg: float = 25e6
    thermal_efficiency: float = .7
    iron_kg_s: float = 12.
    iron_specific_heat_j_kg_k: float = 820.

    def solve(self):
        _check(**asdict(self))
        if self.thermal_efficiency > 1: raise ValueError("Efficiency must be <= 1")
        power = self.fuel_kg_s*self.fuel_lhv_j_kg*self.thermal_efficiency
        return {"useful_heat_w":power,
                "idealized_iron_temperature_rise_k":power/(self.iron_kg_s*self.iron_specific_heat_j_kg_k),
                "fuel_input_w":self.fuel_kg_s*self.fuel_lhv_j_kg}

@dataclass(frozen=True)
class ChemicalPlant:
    flow_m3_s: float = .1
    reactor_volume_m3: float = 10.
    rate_constant_s_inv: float = .02
    feed_concentration_mol_m3: float = 100.

    def solve(self):
        _check(**asdict(self))
        residence = self.reactor_volume_m3/self.flow_m3_s
        conversion = self.rate_constant_s_inv*residence/(1+self.rate_constant_s_inv*residence)
        return {"residence_time_s":residence,"conversion":conversion,
                "outlet_concentration_mol_m3":self.feed_concentration_mol_m3*(1-conversion),
                "reactant_consumption_mol_s":self.flow_m3_s*self.feed_concentration_mol_m3*conversion}

@dataclass(frozen=True)
class ReusableRocket:
    mass_kg: float = 18000.
    thrust_n: float = 250000.
    drag_coefficient: float = .6
    air_density_kg_m3: float = 1.225
    reference_area_m2: float = 10.
    upward_velocity_m_s: float = 0.

    def solve(self):
        _check(mass_kg=self.mass_kg,thrust_n=self.thrust_n,
               drag_coefficient=self.drag_coefficient,air_density_kg_m3=self.air_density_kg_m3,
               reference_area_m2=self.reference_area_m2)
        if not isfinite(self.upward_velocity_m_s): raise ValueError("Invalid velocity")
        drag = -.5*self.air_density_kg_m3*self.drag_coefficient*self.reference_area_m2*self.upward_velocity_m_s*abs(self.upward_velocity_m_s)
        return {"vertical_acceleration_m_s2":(self.thrust_n+drag-self.mass_kg*G)/self.mass_kg,
                "drag_force_n":drag,"thrust_to_weight":self.thrust_n/(self.mass_kg*G)}

@dataclass(frozen=True)
class LunarBase:
    solar_irradiance_w_m2: float = 1361.
    panel_area_m2: float = 120.
    panel_efficiency: float = .25
    illumination_fraction: float = 1.
    base_load_w: float = 12000.

    def solve(self):
        _check(solar_irradiance_w_m2=self.solar_irradiance_w_m2,
               panel_area_m2=self.panel_area_m2, panel_efficiency=self.panel_efficiency,
               base_load_w=self.base_load_w)
        if not (0 <= self.illumination_fraction <= 1 and self.panel_efficiency <= 1):
            raise ValueError("Invalid fractions")
        generated = self.solar_irradiance_w_m2*self.panel_area_m2*self.panel_efficiency*self.illumination_fraction
        return {"solar_power_w":generated,"net_power_w":generated-self.base_load_w,
                "power_coverage_ratio":generated/self.base_load_w}

TWINS = {"fpso":FPSO,"blast_furnace":BlastFurnace,
         "chemical_plant":ChemicalPlant,"reusable_rocket":ReusableRocket,"lunar_base":LunarBase}

def run(name, overrides=None):
    if name not in TWINS: raise KeyError(name)
    model = TWINS[name](**(overrides or {}))
    return {"schema_version":"0.1","twin":name,"source":"synthetic",
            "verified":False,"parameters":asdict(model),"results":model.solve(),
            "scope":"educational lumped reference only"}

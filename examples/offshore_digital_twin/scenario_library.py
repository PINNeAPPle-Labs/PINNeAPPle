"""Synthetic scenario templates for research. No safety-critical control actions."""
from dataclasses import replace
from offshore_twin import WellConfig, evaluate

SCENARIOS = {
    "nominal": {"flow_factor":1.0,"density_factor":1.0},
    "reduced_circulation": {"flow_factor":0.6,"density_factor":1.0},
    "increased_density": {"flow_factor":1.0,"density_factor":1.1},
    "reduced_density": {"flow_factor":1.0,"density_factor":0.9},
}

def run_scenario(name, base=None):
    if name not in SCENARIOS:
        raise KeyError(name)
    config = base or WellConfig()
    factors = SCENARIOS[name]
    updated = replace(config,flow_m3_s=config.flow_m3_s*factors["flow_factor"],
                      mud_density_kg_m3=config.mud_density_kg_m3*factors["density_factor"])
    return {"name":name,"source":"synthetic","parameters":updated,
            "results":evaluate(updated),"verified":False}

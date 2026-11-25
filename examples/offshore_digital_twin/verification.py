"""Explicit model-scope checks. Not a substitute for VeriPhysics integration."""
from math import isclose
from offshore_twin import G, WellConfig, evaluate

def audit(config: WellConfig, output=None):
    result = output if output is not None else evaluate(config)
    expected_hydro = config.mud_density_kg_m3 * G * config.depth_m
    residual = abs(result["hydrostatic_pa"] - expected_hydro)
    ecd = result["bottomhole_circulating_pressure_pa"] / (G * config.depth_m)
    checks = {
        "hydrostatic_identity": isclose(result["hydrostatic_pa"], expected_hydro, rel_tol=1e-10),
        "ecd_identity": isclose(result["equivalent_circulating_density_kg_m3"], ecd, rel_tol=1e-10),
        "positive_pressure_losses": result["pipe_friction_pa"] >= 0 and result["annulus_friction_pa"] >= 0,
        "pressure_window": bool(result["within_pressure_window"]),
    }
    return {"checks": checks, "passed": all(checks.values()), "hydrostatic_residual_pa": residual,
            "scope": "simplified mathematical consistency only; no independent validation"}

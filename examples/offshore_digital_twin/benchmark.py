"""Analytic regression benchmarks; no industrial validation claimed."""
from math import isclose
from offshore_twin import WellConfig, G, evaluate
from transient import TransientConfig, simulate

def run_benchmarks():
    well = WellConfig(flow_m3_s=0)
    out = evaluate(well)
    hydrostatic = well.mud_density_kg_m3*G*well.depth_m
    static_ok = isclose(out["bottomhole_circulating_pressure_pa"],hydrostatic,rel_tol=1e-12)
    transient = simulate(well,TransientConfig(steps=10))
    steady_ok = all(isclose(x["bottomhole_pressure_pa"],hydrostatic,rel_tol=1e-12) for x in transient)
    return {"hydrostatic_limit":static_ok,"steady_transient_limit":steady_ok,
            "passed":static_ok and steady_ok,
            "evidence_level":"analytic_consistency_only"}

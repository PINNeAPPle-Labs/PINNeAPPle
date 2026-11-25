"""Reproducible synthetic scenario generation for the offshore hydraulic seed.

Educational synthetic data only; NOT field data or well-control guidance.
"""
from dataclasses import asdict, replace
from hashlib import sha256
import json
import random
from offshore_twin import WellConfig, evaluate

def generate_scenarios(config=None, n=128, seed=42):
    if not isinstance(n, int) or not 1 <= n <= 100000:
        raise ValueError("n must be between 1 and 100000")
    base = config or WellConfig()
    base.validate()
    rng = random.Random(seed)
    records = []
    for i in range(n):
        c = replace(
            base,
            mud_density_kg_m3=base.mud_density_kg_m3*rng.uniform(.9, 1.1),
            flow_m3_s=base.flow_m3_s*rng.uniform(.5, 1.5),
            friction_factor=base.friction_factor*rng.uniform(.8, 1.2),
        )
        values = evaluate(c)
        records.append({"id": i, "parameters": asdict(c), "outputs": values,
                        "split": "train" if i % 5 < 3 else ("validation" if i % 5 == 3 else "test")})
    return records

def manifest(records, seed=42):
    payload = json.dumps(records, sort_keys=True, allow_nan=False)
    return {"schema_version": "0.1", "model": "offshore_hydraulics_0d",
            "source": "synthetic", "seed": seed, "count": len(records),
            "sha256": sha256(payload.encode()).hexdigest(),
            "validity": "vertical single-phase quasi-steady; constant friction factor"}

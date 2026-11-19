"""Generate a traceable FEM sample for the structural bracket benchmark.

Run from this directory:
    python generate_dataset.py --output ./artifacts/l_bracket_demo.npz

The mesh is a simplified voxel L bracket, not the original source geometry.
The dataset is a demonstration only and MUST NOT be used as an independent
train/test benchmark without group-level splits and mesh convergence studies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fem_reference import solve_elasticity
from geometry import demo_case


def generate(output: str | Path, young_pa: float = 70e9, poisson: float = 0.33) -> dict:
    output = Path(output)
    if output.suffix != ".npz":
        raise ValueError("Output must have .npz extension")
    nodes, tets, fixed, nodal_forces = demo_case()
    solution = solve_elasticity(nodes, tets, fixed, nodal_forces, young_pa, poisson)
    metadata = {
        "schema_version": "0.1.0",
        "geometry_family": "L",
        "geometry_variant": "simplified_voxel_demo",
        "source_reproduction": False,
        "units": {
            "nodes": "m", "displacement": "m", "stress": "Pa",
            "force": "N", "young_modulus": "Pa",
        },
        "material": {"young_pa": young_pa, "poisson": poisson},
        "boundary_conditions": {
            "fixed_nodes": "x=max, z<=thickness",
            "loaded_nodes": "z=max, uniformly distributed nodal total Fx=100 N",
        },
        "caveats": [
            "This is an illustrative nodal-load case, not the original experiment",
            "No mesh convergence or experimental validation has been performed",
            "Element stress is piecewise constant for linear tetrahedra",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        nodes_m=nodes,
        tetrahedra=tets,
        fixed_nodes=fixed,
        nodal_forces_n=nodal_forces,
        displacement_m=solution["displacement_m"],
        stress_pa=solution["stress_pa"],
        von_mises_pa=solution["von_mises_pa"],
        reaction_n=solution["reaction_n"],
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    return {
        "path": str(output),
        "nodes": len(nodes),
        "elements": len(tets),
        "max_von_mises_pa": float(np.max(solution["von_mises_pa"])),
        "force_balance_n": float(solution["force_balance_n"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/l_bracket_demo.npz")
    args = parser.parse_args()
    print(json.dumps(generate(args.output), indent=2))

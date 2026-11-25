"""End-to-end STEP→tetra mesh→FEM dataset generation for bracket families.

Requires cadquery, gmsh, scipy, numpy. No source-paper reproduction claim.
Run from this directory:
    python run_cad_fem.py --family G --output-dir artifacts/gusset
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from cad_brackets import export_family
from mesh_step import mesh_step
from surface_loads import bracket_boundary_case
from fem_reference import solve_elasticity


def run(family: str, output_dir: str | Path, *, size_mm: float = 4.0,
        young_pa: float = 70e9, poisson: float = 0.33,
        traction_pa: tuple[float, float, float] = (1e6, 0.0, 0.0),
        **cad_params) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cad = export_family(family, out / f"{family}.step", **cad_params)
    mesh = mesh_step(cad["step"], str(out / f"{family}_mesh.npz"), size_mm)
    with np.load(mesh["output"], allow_pickle=False) as data:
        nodes = data["nodes_m"]
        tets = data["tetrahedra"]
    fixed, forces, bc = bracket_boundary_case(nodes, tets, traction_pa)
    result = solve_elasticity(nodes, tets, fixed, forces, young_pa, poisson)
    meta = {
        "family": family, "cad": cad, "mesh_size_mm": size_mm,
        "cad_parameters_mm": cad_params,
        "material": {"young_pa": young_pa, "poisson": poisson},
        "boundary_conditions": bc,
        "units": {"nodes": "m", "displacement": "m", "stress": "Pa",
                  "force": "N", "traction": "Pa"},
        "validation": {"force_balance_n": float(result["force_balance_n"]),
                       "mesh_convergence": "NOT_RUN",
                       "independent_solver_comparison": "NOT_RUN"},
        "source_reproduction": False,
    }
    np.savez_compressed(
        out / f"{family}_fem.npz", nodes_m=nodes, tetrahedra=tets,
        fixed_nodes=fixed, nodal_forces_n=forces,
        displacement_m=result["displacement_m"],
        stress_pa=result["stress_pa"],
        von_mises_pa=result["von_mises_pa"],
        reaction_n=result["reaction_n"],
        metadata_json=np.array(json.dumps(meta, sort_keys=True)),
    )
    (out / "manifest.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    return {"family": family, "elements": len(tets),
            "maximum_von_mises_pa": float(result["von_mises_pa"].max()),
            "force_balance_n": float(result["force_balance_n"]),
            "dataset": str(out / f"{family}_fem.npz")}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=["L", "G", "U", "Z"], default="G")
    p.add_argument("--output-dir", default="artifacts/cad_fem")
    p.add_argument("--size-mm", type=float, default=4.0)
    a = p.parse_args()
    print(json.dumps(run(a.family, a.output_dir, size_mm=a.size_mm), indent=2))

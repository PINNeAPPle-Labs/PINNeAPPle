"""Mesh STEP B-rep with Gmsh and export tetrahedral connectivity.

Run: python mesh_step.py artifacts/cad/L.step artifacts/mesh/L.npz --size-mm 3
The resulting mesh is converted to SI metres for the FEM solver.
Boundary condition selection remains a separate validation step.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def mesh_step(step_path: str, output: str, size_mm: float = 3.0) -> dict:
    if size_mm <= 0:
        raise ValueError("Mesh size must be positive")
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("structural_bracket")
        gmsh.model.occ.importShapes(str(step_path))
        gmsh.model.occ.synchronize()
        volumes = gmsh.model.getEntities(3)
        if len(volumes) != 1:
            raise ValueError(f"Expected one CAD volume, found {len(volumes)}")
        gmsh.model.addPhysicalGroup(3, [volumes[0][1]], name="solid")
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", size_mm)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size_mm)
        gmsh.model.mesh.generate(3)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        points = np.asarray(coords).reshape(-1, 3)
        tag_to_idx = {int(tag): i for i, tag in enumerate(tags)}
        types, _, connectivity = gmsh.model.mesh.getElements(3)
        cells = []
        for kind, conn in zip(types, connectivity):
            name, _, _, node_count, _, _ = gmsh.model.mesh.getElementProperties(kind)
            if name.startswith("Tetrahedron") and node_count == 4:
                cells.append(np.array([tag_to_idx[int(t)] for t in conn], dtype=int).reshape(-1, 4))
            elif name.startswith("Tetrahedron"):
                raise ValueError("Only first-order tetrahedra are supported by fem_reference.py")
        if not cells:
            raise ValueError("No first-order tetrahedra found")
        tets = np.vstack(cells)
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, nodes_m=points*1e-3, tetrahedra=tets,
                            source_step=str(step_path), source_units="mm",
                            mesh_size_mm=size_mm)
        return {"nodes": len(points), "tetrahedra": len(tets), "output": str(output_path)}
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("step")
    p.add_argument("output")
    p.add_argument("--size-mm", type=float, default=3.0)
    a = p.parse_args()
    print(mesh_step(a.step, a.output, a.size_mm))

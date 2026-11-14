"""Create a reproducible, multi-case bracket stress dataset.

This is a reconstructed benchmark, not the original Danilo dataset.
Run: python build_benchmark.py --output artifacts/brackets_prepared.npz
Requires CadQuery and Gmsh. Each geometry is meshed once, then multiple
traction load cases are solved. Geometry groups remain intact for splitting.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from cad_brackets import export_family
from mesh_step import mesh_step
from surface_loads import bracket_boundary_case, exterior_faces, plane_faces
from fem_reference import solve_elasticity
from features import branch_features, trunk_features, BRANCH_FEATURES, TRUNK_FEATURES


def make_benchmark(output: str | Path, *, families=("L", "G", "U", "Z"),
                   variants: int = 3, loads_per_geometry: int = 3,
                   points_per_case: int = 256, mesh_size_mm: float = 5.,
                   seed: int = 42):
    if variants < 1 or loads_per_geometry < 1 or points_per_case < 1:
        raise ValueError("Positive case and sampling counts required")
    rng = np.random.default_rng(seed)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = output.parent / "cad_cache"
    cache.mkdir(exist_ok=True)
    branch, trunk, stress, group, manifests = [], [], [], [], []
    for family in families:
        for variant in range(variants):
            geometry = {
                "width": 50., "length": 75., "height": 75.,
                "thickness": float([5., 6., 7.][variant % 3]),
                "hole_diameter": float([7., 8., 9.][variant % 3]),
                "offset": 35., "gusset_thickness": 6.,
            }
            gid = f"{family}_v{variant:03d}"
            step = cache / f"{gid}.step"
            meshfile = cache / f"{gid}.npz"
            export_family(family, step, **geometry)
            mesh_step(str(step), str(meshfile), mesh_size_mm)
            with np.load(meshfile, allow_pickle=False) as data:
                nodes = data["nodes_m"]
                tets = data["tetrahedra"]
            centroids = nodes[tets].mean(axis=1)
            chosen = rng.choice(len(tets), points_per_case,
                                replace=len(tets) < points_per_case)
            sample_points = centroids[chosen]
            faces = exterior_faces(tets)
            bottom = plane_faces(nodes, faces, 2, nodes[:, 2].min())
            top = plane_faces(nodes, faces, 2, nodes[:, 2].max())
            fixed_surface = nodes[np.unique(bottom)]
            loaded_surface = nodes[np.unique(top)]
            if len(fixed_surface) == 0 or len(loaded_surface) == 0:
                raise ValueError(f"Missing boundary face group for {gid}")
            # The reconstructed concave-edge proxy is sampled near the
            # plate intersection. Not an exact CAD concave-edge extractor.
            corner = np.array([geometry["thickness"]*1e-3, 0.,
                               geometry["thickness"]*1e-3])
            edge_samples = nodes[np.argsort(np.linalg.norm(nodes-corner, axis=1))[:max(4, min(32,len(nodes)))]]
            point_features = trunk_features(sample_points, fixed_surface,
                                            loaded_surface, edge_samples)
            for load_id in range(loads_per_geometry):
                direction = np.array([1., 0., 0.]) if load_id % 2 == 0 else np.array([0., 1., 0.])
                magnitude_pa = float([0.5e6, 1e6, 1.5e6][load_id % 3])
                fixed, forces, bc = bracket_boundary_case(
                    nodes, tets, tuple(direction * magnitude_pa))
                solution = solve_elasticity(nodes, tets, fixed, forces, 70e9, 0.33)
                total_force = forces.sum(axis=0)
                # Moments are computed about the origin from assembled nodal forces.
                total_moment = np.cross(nodes, forces).sum(axis=0)
                branch.append(branch_features(family, geometry, 70e9, 0.33,
                                              total_force, total_moment))
                trunk.append(point_features)
                stress.append(solution["von_mises_pa"][chosen].astype(np.float32))
                group.append(gid)
                manifests.append({
                    "geometry_id": gid, "family": family, "load_id": load_id,
                    "geometry_mm": geometry, "traction_pa": (direction*magnitude_pa).tolist(),
                    "resultant_n": total_force.tolist(),
                    "resultant_moment_nm": total_moment.tolist(),
                    "force_balance_n": float(solution["force_balance_n"]),
                    "mesh_size_mm": mesh_size_mm, "mesh_elements": len(tets),
                })
    np.savez_compressed(output, branch=np.stack(branch), trunk=np.stack(trunk),
                        stress_pa=np.stack(stress), group=np.array(group),
                        metadata_json=np.array(json.dumps({
                            "source_reproduction": False,
                            "feature_encoding": "reconstructed_v1",
                            "concave_edge": "nearest_mesh_nodes_to_plate_intersection_proxy",
                            "branch_features": BRANCH_FEATURES,
                            "trunk_features": TRUNK_FEATURES,
                            "seed": seed, "cases": manifests,
                        }, sort_keys=True)))
    return {"path": str(output), "cases": len(branch),
            "geometry_groups": len(set(group)), "points_per_case": points_per_case}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="artifacts/brackets_prepared.npz")
    p.add_argument("--variants", type=int, default=3)
    p.add_argument("--loads", type=int, default=3)
    p.add_argument("--points", type=int, default=256)
    p.add_argument("--mesh-size-mm", type=float, default=5.)
    a = p.parse_args()
    print(json.dumps(make_benchmark(a.output, variants=a.variants,
         loads_per_geometry=a.loads, points_per_case=a.points,
         mesh_size_mm=a.mesh_size_mm), indent=2))

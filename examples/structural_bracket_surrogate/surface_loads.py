"""Boundary-face extraction and consistent surface traction for P1 tetrahedra.

Each exterior triangular face contributes area * traction / 3 to each of
its three nodes. This avoids mesh-dependent equal-per-node loading.
All geometry must be in metres, traction in Pa (N/m^2).
"""
from __future__ import annotations
import numpy as np


def exterior_faces(tetrahedra: np.ndarray) -> np.ndarray:
    tets = np.asarray(tetrahedra, dtype=np.int64)
    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError("Expected tetrahedra [T,4]")
    faces = np.concatenate((tets[:, [0, 1, 2]], tets[:, [0, 1, 3]],
                            tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]))
    canonical = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(canonical, axis=0, return_inverse=True,
                                    return_counts=True)
    if np.any(counts > 2):
        raise ValueError("Nonmanifold tetrahedral faces detected")
    return faces[counts[inverse] == 1]


def plane_faces(nodes: np.ndarray, faces: np.ndarray, axis: int, value: float,
                tolerance_m: float = 1e-7) -> np.ndarray:
    if axis not in (0, 1, 2) or tolerance_m <= 0:
        raise ValueError("Invalid axis or tolerance")
    return faces[np.all(np.abs(nodes[faces, axis] - value) <= tolerance_m, axis=1)]


def traction_nodal_forces(nodes: np.ndarray, faces: np.ndarray,
                          traction_pa: np.ndarray) -> tuple[np.ndarray, float]:
    nodes = np.asarray(nodes, dtype=float)
    faces = np.asarray(faces, dtype=int)
    traction = np.asarray(traction_pa, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Expected nodes [N,3] and faces [F,3]")
    if traction.shape != (3,):
        raise ValueError("Expected traction vector [3]")
    if len(faces) == 0:
        raise ValueError("No surface faces selected")
    xyz = nodes[faces]
    areas = np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0],
                                     xyz[:, 2] - xyz[:, 0]), axis=1) / 2
    if np.any(areas <= 0):
        raise ValueError("Degenerate boundary face")
    nodal = np.zeros_like(nodes)
    np.add.at(nodal, faces.reshape(-1),
              np.repeat((areas[:, None] * traction / 3), 3, axis=0))
    return nodal, float(areas.sum())


def bracket_boundary_case(nodes: np.ndarray, tetrahedra: np.ndarray,
                          traction_pa: tuple[float, float, float] = (1e6, 0, 0),
                          tolerance_m: float = 1e-7):
    """Clamp bottom z=0 and load topmost horizontal planar surface.

    This is an explicit *demonstration* BC, not the original study's BC.
    For Z brackets the topmost plane is z=max. Use physical CAD tags for
    more complex loading (pin-bearing pressure, bolt pretension).
    """
    faces = exterior_faces(tetrahedra)
    z_min, z_max = float(nodes[:, 2].min()), float(nodes[:, 2].max())
    fixed_faces = plane_faces(nodes, faces, 2, z_min, tolerance_m)
    load_faces = plane_faces(nodes, faces, 2, z_max, tolerance_m)
    if not len(fixed_faces) or not len(load_faces):
        raise ValueError("No boundary faces on clamp or load plane")
    fixed_nodes = np.unique(fixed_faces)
    forces, loaded_area_m2 = traction_nodal_forces(
        nodes, load_faces, np.asarray(traction_pa, dtype=float))
    return fixed_nodes, forces, {
        "clamp": "exterior faces on z_min",
        "traction": "exterior faces on z_max",
        "loaded_area_m2": loaded_area_m2,
        "traction_pa": list(traction_pa),
        "resultant_n": forces.sum(axis=0).tolist(),
    }

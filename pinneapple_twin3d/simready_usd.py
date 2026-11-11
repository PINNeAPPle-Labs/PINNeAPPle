"""Optional OpenUSD SimReady -> Twin3D importer.

This is a visual-mesh importer with explicit, separately retained physics
metadata. It does not interpret visual triangles as certified CAE meshes.
Requires `usd-core` (pxr). Unsupported primvars, skeletons and materials
are retained in the original USD package, not silently reinterpreted.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .scene import Scene, PhysicsProps


def _pxr():
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError as exc:
        raise ImportError("OpenUSD import requires usd-core: pip install usd-core") from exc
    return Usd, UsdGeom, UsdPhysics


def inspect_usd(path: str | Path) -> dict:
    """Read stage geometry and physics schema inventory without conversion."""
    Usd, UsdGeom, UsdPhysics = _pxr()
    path = Path(path).resolve()
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Cannot open USD stage: {path}")
    meshes, bodies, collisions = [], [], []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            meshes.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collisions.append(str(prim.GetPath()))
    return {
        "stage": str(path), "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meshes": meshes, "rigid_bodies": bodies, "colliders": collisions,
        "warnings": ["USD schemas not explicitly mapped are preserved in the source stage"],
    }


def _triangulate(counts, indices):
    """Fan-triangulate polygon faces. Concave faces require external triangulation."""
    triangles = []
    offset = 0
    for count in counts:
        count = int(count)
        face = indices[offset:offset + count]
        offset += count
        for j in range(1, count - 1):
            triangles.append((face[0], face[j], face[j + 1]))
    if offset != len(indices):
        raise ValueError("USD faceVertexCounts do not match faceVertexIndices")
    return np.asarray(triangles, dtype=np.uint32).reshape(-1, 3)


def import_usd_scene(
    usd_path: str | Path, *, title: str | None = None,
    max_triangles: int = 2_000_000,
) -> tuple[Scene, dict]:
    """Convert static UsdGeom.Mesh prims into a Twin3D scene.

    Stage transforms are baked into visual vertices, and the original USD
    remains authoritative. Physics API presence is recorded as provenance,
    not fabricated as simulation-ready numerical properties.
    """
    Usd, UsdGeom, UsdPhysics = _pxr()
    usd_path = Path(usd_path).resolve()
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise ValueError(f"Cannot open USD stage: {usd_path}")
    scene = Scene(title or usd_path.stem, length_unit="m", source=str(usd_path))
    cache = UsdGeom.XformCache()
    scale = float(UsdGeom.GetStageMetersPerUnit(stage))
    report = {"source_usd": str(usd_path), "meters_per_unit": scale,
              "up_axis": str(UsdGeom.GetStageUpAxis(stage)), "parts": [],
              "limitations": ["Static visual meshes only", "Fan triangulation assumes convex polygons",
                              "Physics schemas inventoried but not numerically validated",
                              "USD material and texture bindings remain in original package"]}
    total = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if points is None or counts is None or indices is None:
            continue
        vertices = np.asarray(points, dtype=np.float64)
        faces = _triangulate(counts, indices)
        if not len(vertices) or not len(faces):
            continue
        total += len(faces)
        if total > max_triangles:
            raise ValueError("Triangle budget exceeded; choose a smaller asset or raise max_triangles")
        transform = cache.GetLocalToWorldTransform(prim)
        # Gf.Matrix4d.Transform applies USD's row-vector convention correctly.
        vertices = np.asarray([tuple(transform.Transform(tuple_to_vec(v))) for v in vertices],
                              dtype=np.float64) * scale
        name = str(prim.GetPath()).strip("/").replace("/", "__")
        scene.add_part(name, vertices.astype(np.float32), faces, group="SimReady")
        report["parts"].append({
            "name": name, "usd_prim": str(prim.GetPath()),
            "triangles": len(faces),
            "has_collision_api": bool(prim.HasAPI(UsdPhysics.CollisionAPI)),
            "has_rigid_body_api": bool(prim.HasAPI(UsdPhysics.RigidBodyAPI)),
        })
    if not scene.parts:
        raise ValueError("USD stage contains no supported polygon meshes")
    return scene, report


def tuple_to_vec(x):
    from pxr import Gf
    return Gf.Vec3d(float(x[0]), float(x[1]), float(x[2]))


def export_simready_twin(
    usd_path: str | Path, destination: str | Path, *,
    with_viewer: bool = True, max_triangles: int = 2_000_000,
) -> dict:
    """Write browser geometry and an explicit import report alongside the source."""
    scene, report = import_usd_scene(usd_path, max_triangles=max_triangles)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    scene.export(str(destination), with_viewer=with_viewer)
    (destination / "simready_import_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


__all__ = ["inspect_usd", "import_usd_scene", "export_simready_twin"]

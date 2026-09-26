"""Twin geometry from scans: cleaned scan meshes and MuJoCo scenes (e.g. LiteReality output).

Two entry points (``ROADMAP.md`` §11, LiteReality-Agent: scan -> physics-ready scene):

- :func:`clean_scan` / :func:`add_scan`: a raw scan mesh (PLY/OBJ/GLB/STL from a phone, LiDAR or
  photogrammetry) -> merged duplicate vertices, degenerate and duplicate faces removed, small
  floating fragments dropped, holes filled, optional decimation, floor aligned to +Z (dominant
  area-weighted face normal), converted to metres. Returns a report of everything changed.
- :func:`scene_from_mjcf`: a MuJoCo MJCF scene (what LiteReality exports: articulated assets with
  mesh geoms) -> one Twin3D part per mesh geom, with the body/geom position and orientation chain
  applied. Primitive geoms (box, sphere, cylinder, capsule, plane) become simple meshes.

Requires ``trimesh`` (the ``geom`` extra). Point clouds without faces are rejected with a clear
message: surface reconstruction is not done here.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .scene import Scene

_TO_M = {"m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254, "ft": 0.3048}


@dataclass
class ScanReport:
    vertices_in: int
    faces_in: int
    vertices_out: int = 0
    faces_out: int = 0
    components_dropped: int = 0
    holes_filled: bool = False
    watertight: bool = False
    up_axis_before: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    notes: List[str] = field(default_factory=list)


def _trimesh():
    try:
        import trimesh
    except ImportError as e:
        raise ImportError('scan import needs trimesh: pip install "pinneapple[geom]"') from e
    return trimesh


def _rotation_to_z(n: np.ndarray) -> np.ndarray:
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(n, z), float(np.dot(n, z))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def clean_scan(path_or_mesh, *, unit: str = "m", min_component_fraction: float = 0.01,
               fill_holes: bool = True, target_faces: Optional[int] = None, align_floor: bool = True):
    """Clean a scan mesh. Returns ``(trimesh.Trimesh, ScanReport)`` in metres."""
    trimesh = _trimesh()
    m = trimesh.load(path_or_mesh, force="mesh") if isinstance(path_or_mesh, (str, os.PathLike)) else path_or_mesh.copy()
    if not hasattr(m, "faces") or len(m.faces) == 0:
        raise ValueError("the scan has no faces (point cloud?): reconstruct a surface first "
                         "(e.g. Poisson reconstruction in Open3D/MeshLab), then import the mesh")
    rep = ScanReport(len(m.vertices), len(m.faces))
    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    parts = m.split(only_watertight=False)
    if len(parts) > 1:
        areas = np.array([p.area for p in parts])
        keep = [p for p, a in zip(parts, areas) if a >= min_component_fraction * areas.sum()]
        rep.components_dropped = len(parts) - len(keep)
        m = trimesh.util.concatenate(keep)
    if fill_holes:
        rep.holes_filled = bool(trimesh.repair.fill_holes(m))
    if target_faces and len(m.faces) > target_faces:
        try:
            m = m.simplify_quadric_decimation(face_count=int(target_faces))
        except Exception as e:  # needs fast-simplification / open3d
            rep.notes.append(f"decimation skipped: {e}")
    m.apply_scale(_TO_M[unit])
    if align_floor:
        # dominant orientation = the normal carrying the most area among axis-like face groups
        n = m.face_normals
        w = m.area_faces
        cand = np.vstack([n, -n])
        cw = np.concatenate([w, w])
        best, best_w = None, -1.0
        for axis in np.eye(3).tolist() + (-np.eye(3)).tolist():
            a = np.asarray(axis)
            s = float(cw[(cand @ a) > 0.95].sum())
            if s > best_w:
                best, best_w = a, s
        # refine with the area-weighted mean of the normals close to that axis
        close = (n @ best) > 0.9
        up = (n[close] * w[close, None]).sum(0) if close.any() else best
        rep.up_axis_before = tuple(float(v) for v in up / np.linalg.norm(up))
        R = np.eye(4)
        R[:3, :3] = _rotation_to_z(up)
        m.apply_transform(R)
        m.apply_translation([0.0, 0.0, -m.bounds[0][2]])  # floor at z = 0
    rep.vertices_out, rep.faces_out, rep.watertight = len(m.vertices), len(m.faces), bool(m.is_watertight)
    return m, rep


def add_scan(scene: Scene, name: str, path_or_mesh, *, group: str = "scan", **kw) -> ScanReport:
    mesh, rep = clean_scan(path_or_mesh, **kw)
    scene.add_part(name, mesh.vertices, mesh.faces, group=group)
    return rep


# ── MuJoCo MJCF (LiteReality exports) ───────────────────────────────────
def _quat_to_R(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _frame(el) -> np.ndarray:
    T = np.eye(4)
    if el.get("quat"):
        T[:3, :3] = _quat_to_R(el.get("quat").split())
    if el.get("pos"):
        T[:3, 3] = [float(v) for v in el.get("pos").split()]
    return T


def _primitive(geom, trimesh):
    typ = geom.get("type", "sphere")
    size = [float(v) for v in (geom.get("size") or "0.05").split()]
    if typ == "box":
        return trimesh.creation.box(extents=2 * np.array((size + size * 3)[:3]))
    if typ == "sphere":
        return trimesh.creation.icosphere(radius=size[0], subdivisions=2)
    if typ == "cylinder":
        return trimesh.creation.cylinder(radius=size[0], height=2 * size[1])
    if typ == "capsule":
        return trimesh.creation.capsule(radius=size[0], height=2 * size[1])
    if typ == "plane":
        sx, sy = (size + [1.0, 1.0])[:2]
        return trimesh.Trimesh(vertices=[[-sx, -sy, 0], [sx, -sy, 0], [sx, sy, 0], [-sx, sy, 0]], faces=[[0, 1, 2], [0, 2, 3]])
    return None


def scene_from_mjcf(path: str, *, title: Optional[str] = None, include_primitives: bool = True) -> Scene:
    """One Twin3D part per geom of a MuJoCo scene, in world coordinates."""
    trimesh = _trimesh()
    root = ET.parse(path).getroot()
    base = os.path.dirname(os.path.abspath(path))
    compiler = root.find("compiler")
    meshdir = os.path.join(base, compiler.get("meshdir", "")) if compiler is not None else base
    meshes: Dict[str, Tuple[str, np.ndarray]] = {}
    for m in root.iter("mesh"):
        f = m.get("file")
        if f:
            scale = np.array([float(v) for v in (m.get("scale") or "1 1 1").split()])
            meshes[m.get("name") or os.path.splitext(os.path.basename(f))[0]] = (os.path.join(meshdir, f), scale)
    sc = Scene(title or os.path.basename(path), source=f"MuJoCo MJCF {os.path.abspath(path)}")
    used: Dict[str, int] = {}

    def walk(el, T_parent, body_name):
        for child in el:
            if child.tag == "body":
                walk(child, T_parent @ _frame(child), child.get("name") or body_name)
            elif child.tag == "geom":
                T = T_parent @ _frame(child)
                if child.get("type") == "mesh" or (child.get("mesh") and not child.get("type")):
                    fname, scale = meshes[child.get("mesh")]
                    g = trimesh.load(fname, force="mesh")
                    g.apply_scale(scale) if np.allclose(scale, scale[0]) else g.apply_transform(np.diag(list(scale) + [1.0]))
                elif include_primitives:
                    g = _primitive(child, trimesh)
                    if g is None:
                        continue
                else:
                    continue
                g.apply_transform(T)
                name = child.get("name") or child.get("mesh") or f"{body_name}_{child.get('type', 'geom')}"
                used[name] = used.get(name, 0) + 1
                if used[name] > 1:
                    name = f"{name}_{used[name]}"
                rgba = [float(v) for v in (child.get("rgba") or "0.75 0.77 0.8 1").split()]
                sc.add_part(name, g.vertices, g.faces, group=body_name, color=tuple(rgba[:3]))

    wb = root.find("worldbody")
    if wb is None:
        raise ValueError(f"{path}: no <worldbody>")
    walk(wb, np.eye(4), "world")
    return sc

"""Triangle meshes of bricks and assemblies, and export to glTF (per-brick colours), STL and USD.

The meshes are simplified solids: the outer body (0.1 mm clearance per side) plus cylindrical studs; the hollow
underside and the tubes are not modelled (use :mod:`.ldraw` with the LDraw parts library for exact shapes).
glTF and USD are written in metres (their convention), STL in millimetres.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .assembly import Assembly, Brick
from .parts import CLEARANCE_MM, PITCH_MM, STUD_DIAMETER_MM, STUD_HEIGHT_MM, colour


def box_mesh(lo: Sequence[float], hi: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Closed, outward-oriented box with four vertices per face (sharp edges under smooth-normal shading)."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    quads = [[(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)],        # -z
             [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],        # +z
             [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],        # -y
             [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],        # +x
             [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)],        # +y
             [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)]]        # -x
    V = np.array(quads, float).reshape(-1, 3)
    F = np.array([[4 * k, 4 * k + 1, 4 * k + 2] for k in range(6)] + [[4 * k, 4 * k + 2, 4 * k + 3] for k in range(6)],
                 np.int64)
    return V, F


def cylinder_mesh(cx: float, cy: float, z0: float, r: float, h: float, n: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Closed cylinder on ``z0``: a smooth side and flat caps with their own vertices."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)
    side = np.vstack([np.c_[ring, np.full(n, z0)], np.c_[ring, np.full(n, z0 + h)]])            # 0 .. 2n-1
    caps = np.vstack([np.c_[ring, np.full(n, z0)], np.c_[ring, np.full(n, z0 + h)],
                      [[cx, cy, z0], [cx, cy, z0 + h]]])                                      # 2n .. 4n+1
    V = np.vstack([side, caps])
    b, t, cb, ct = 2 * n, 3 * n, 4 * n, 4 * n + 1
    F = []
    for k in range(n):
        k1 = (k + 1) % n
        F += [[k, k1, n + k1], [k, n + k1, n + k], [cb, b + k1, b + k], [ct, t + k, t + k1]]
    return V, np.array(F, np.int64)


def merge(meshes: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    Vs, Fs, off = [], [], 0
    for V, F in meshes:
        Vs.append(V)
        Fs.append(F + off)
        off += len(V)
    return np.vstack(Vs), np.vstack(Fs)


def brick_mesh(b: Brick, *, studs: bool = True, segments: int = 16, local: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Mesh of one placed brick in millimetres (world coordinates, or relative to its centre with ``local``)."""
    lo, hi = b.box_mm()
    body_lo, body_hi = lo + [CLEARANCE_MM, CLEARANCE_MM, 0], hi - [CLEARANCE_MM, CLEARANCE_MM, 0]
    parts = [box_mesh(body_lo, body_hi)]
    if studs and not b.part.smooth:
        for i in range(b.sx):
            for j in range(b.sy):
                parts.append(cylinder_mesh(lo[0] + (i + 0.5) * PITCH_MM, lo[1] + (j + 0.5) * PITCH_MM, hi[2],
                                           STUD_DIAMETER_MM / 2, STUD_HEIGHT_MM, segments))
    V, F = merge(parts)
    if local:
        V = V - b.com_mm()
    return V, F


def assembly_meshes(asm: Assembly, **kw) -> list[tuple[np.ndarray, np.ndarray, tuple[float, float, float]]]:
    """``(vertices_mm, faces, srgb)`` per brick."""
    return [(*brick_mesh(b, **kw), colour(b.colour).rgb) for b in asm.bricks]


def studio_scene(asm: Assembly, title: str = "brick build", fields: dict[str, Sequence[float]] | None = None):
    """The build as a ``pp.viz`` studio Scene (metres, z up) with one material per colour, and that material table.

    ``fields``: name -> one value per brick, written as a per-vertex field (e.g. joint utilisation)."""
    from pinneapple_tools.visualization.studio.scene import Scene, Surface
    sc = Scene(title=title, axes="z_up")
    mats = {}
    for i, (b, (V, F, rgb)) in enumerate(zip(asm.bricks, assembly_meshes(asm), strict=True)):
        key = f"brick_{b.colour}"
        mats[key] = {"color": list(rgb), "metallic": 0.0, "roughness": 0.35, "clearcoat": 0.6}
        s = Surface(f"brick_{i:03d}_{b.part.name}", V / 1000.0, F, material=key)
        for name, vals in (fields or {}).items():
            s.fields[name] = np.full(len(V), float(vals[i]))
        sc.surfaces.append(s)
    return sc, mats


def write_gltf(asm: Assembly, path: str, title: str = "brick build") -> str:
    """Binary glTF (.glb) with one node per brick and its colour, in metres."""
    sc, mats = studio_scene(asm, title)
    with open(path, "wb") as f:
        f.write(sc.to_glb(materials=mats))
    return path


def write_usd(asm: Assembly, path: str, title: str = "brick build") -> str:
    sc, mats = studio_scene(asm, title)
    with open(path, "w") as f:
        f.write(sc.to_usda(materials=mats))
    return path


def write_stl(asm: Assembly, path: str, title: str = "brick build") -> str:
    """Binary STL of the whole build in millimetres."""
    from pinneapple_tools.visualization.studio.scene import Surface
    from pinneapple_tools.visualization.studio.scene import write_stl as _stl
    surfaces = [Surface(f"brick_{i}", V, F) for i, (V, F, _) in enumerate(assembly_meshes(asm))]
    with open(path, "wb") as f:
        f.write(_stl(surfaces, title))
    return path

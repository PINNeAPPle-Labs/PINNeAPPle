"""3D digital-twin view of an axisymmetric accretion flow with PINNeAPPle-Twin3D.

The simulation lives on an (r, theta) grid; the flow is axisymmetric, so revolving it about the spin axis gives
the 3D picture. Each panel is a cutaway: the two meridional half-planes that bound a 3/4 revolution
(phi = 0 and phi = ``phi_cut``) and the equatorial sector between them, coloured by a per-vertex field over
time, around a black sphere at the event horizon (r = 2 GM/c^2). Panels for the simulation and for the
forecast sit side by side, so the viewer plays both at once; sensors carry the trust signals (forecast
error, rate of change of the window mass) with their envelopes, which the viewer turns red when violated.

    from pinneapple_physics.blackhole.twin import accretion_scene
    sc = accretion_scene(grid, {"simulation": logrho_true, "forecast": logrho_pred}, times, r_view=60)
    sc.export("out/bh_twin")       # then pinneapple_twin3d.serve("out/bh_twin")

Coordinates: y is the spin axis (the viewer's up direction), lengths in GM/c^2.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from pinneapple_twin3d.scene import Scene

__all__ = ["CutawayMesh", "accretion_scene", "sphere"]


def _cell_to_vertex(a: np.ndarray) -> np.ndarray:
    """(..., n, m) cell values -> (..., n+1, m+1) vertex values (mean of the adjacent cells)."""
    p = np.pad(a, [(0, 0)] * (a.ndim - 2) + [(1, 1), (1, 1)], mode="edge")
    return 0.25 * (p[..., :-1, :-1] + p[..., 1:, :-1] + p[..., :-1, 1:] + p[..., 1:, 1:])


def _grid_faces(n: int, m: int, base: int = 0) -> np.ndarray:
    """Two triangles per quad of an (n+1) x (m+1) vertex grid (row-major)."""
    i, j = np.meshgrid(np.arange(n), np.arange(m), indexing="ij")
    v0 = base + i * (m + 1) + j
    v1, v2, v3 = v0 + (m + 1), v0 + (m + 2), v0 + 1
    return np.concatenate([np.stack([v0, v1, v2], -1).reshape(-1, 3), np.stack([v0, v2, v3], -1).reshape(-1, 3)])


class CutawayMesh:
    """Geometry of one cutaway panel and the map from (nr, ntheta) cell fields to its vertices."""

    def __init__(self, r_faces: np.ndarray, theta_faces: np.ndarray, *, r_view: float, phi_cut_deg: float = 270.0,
                 n_phi: int = 72, offset: Sequence[float] = (0.0, 0.0, 0.0)):
        self.nrv = int(np.searchsorted(r_faces, r_view, side="right") - 1)    # cells kept: r_faces[nrv] <= r_view
        rf = np.asarray(r_faces[:self.nrv + 1], float)
        tf = np.asarray(theta_faces, float)
        self.nth = len(tf) - 1
        off = np.asarray(offset, float)
        verts, faces = [], []
        # meridional half-planes at phi = 0 and phi = phi_cut
        for phi in (0.0, math.radians(phi_cut_deg)):
            R = rf[:, None] * np.sin(tf)[None]
            Y = rf[:, None] * np.cos(tf)[None]
            v = np.stack([R * math.cos(phi), Y, R * math.sin(phi)], -1).reshape(-1, 3) + off
            faces.append(_grid_faces(self.nrv, self.nth, sum(len(x) for x in verts)))
            verts.append(v)
        # equatorial sector, phi from phi_cut to 360 (the part the cut leaves in view), radial faces x phi
        phis = np.linspace(math.radians(phi_cut_deg), 2 * math.pi, n_phi + 1)
        v = np.stack([rf[:, None] * np.cos(phis)[None], np.zeros((len(rf), len(phis))), rf[:, None] * np.sin(phis)[None]],
                     -1).reshape(-1, 3) + off
        faces.append(_grid_faces(self.nrv, n_phi, sum(len(x) for x in verts)))
        verts.append(v)
        self.n_phi = n_phi
        self.vertices = np.concatenate(verts).astype(np.float32)
        self.faces = np.concatenate(faces).astype(np.uint32)

    def values(self, cells: np.ndarray) -> np.ndarray:
        """(T, nr, ntheta) or (nr, ntheta) cell values -> (T, V) per-vertex values."""
        a = np.asarray(cells, float)
        if a.ndim == 2:
            a = a[None]
        a = a[:, :self.nrv]
        vv = _cell_to_vertex(a)                                   # (T, nrv+1, nth+1)
        plane = vv.reshape(len(a), -1)
        # equator: theta face pi/2 is vertex column nth/2 (ntheta even)
        eq = vv[:, :, self.nth // 2]                              # (T, nrv+1)
        sector = np.repeat(eq[:, :, None], self.n_phi + 1, axis=2).reshape(len(a), -1)
        return np.concatenate([plane, plane, sector], axis=1).astype(np.float32)


def sphere(radius: float, center: Sequence[float] = (0, 0, 0), n: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    th = np.linspace(0, math.pi, n + 1)
    ph = np.linspace(0, 2 * math.pi, 2 * n + 1)
    T, P = np.meshgrid(th, ph, indexing="ij")
    v = np.stack([np.sin(T) * np.cos(P), np.cos(T), np.sin(T) * np.sin(P)], -1).reshape(-1, 3) * radius
    return (v + np.asarray(center, float)).astype(np.float32), _grid_faces(n, 2 * n).astype(np.uint32)


def accretion_scene(grid: Dict[str, np.ndarray], panels: Dict[str, np.ndarray], times: Sequence[float], *,
                    field: str = "log10_density", unit: str = "", r_view: float = 60.0, gap: float = 0.35,
                    title: str = "Black hole weather", sensors: Optional[Sequence[dict]] = None,
                    phi_cut_deg: float = 270.0, layout: Sequence[float] = (-1.0, 0.0, -1.0)) -> Scene:
    """One cutaway panel per entry of ``panels`` (name -> (T, nr, ntheta) cell values), laid out along ``layout``
    (default: left to right on screen for the view that looks into the cut, ``view=1,0.75,-1`` in the viewer).

    ``sensors``: dicts with keys id, panel (name, for placement), series (T,), unit, label, envelope.
    """
    sc = Scene(title, length_unit="GM/c^2", times=list(times), time_unit="GM/c^3",
               source="pinneapple_physics.blackhole (issue #399)")
    step = 2 * r_view * (1 + gap)
    lay = np.asarray(layout, float) / np.linalg.norm(layout)
    centers = {}
    for k, (name, cells) in enumerate(panels.items()):
        off = tuple(float(x) for x in k * step * lay)
        centers[name] = off
        mesh = CutawayMesh(grid["r_faces"], grid["theta_faces"], r_view=r_view, phi_cut_deg=phi_cut_deg, offset=off)
        sc.add_part(name, mesh.vertices, mesh.faces, group=name)
        sc.add_field(name, field, mesh.values(cells), unit=unit)
        bv, bf = sphere(2.0, off)
        sc.add_part(f"{name} horizon", bv, bf, group=name, color=(0.02, 0.02, 0.02))
    for s in sensors or []:
        c = centers.get(s.get("panel"), (0.0, 0.0, 0.0))
        pos = (c[0] + s.get("dx", 0.0), c[1] + s.get("dy", 1.05 * r_view), c[2])
        sc.add_sensor(s["id"], pos, label=s.get("label", s["id"]), unit=s.get("unit", ""),
                      quantity=s.get("quantity", ""), series=s.get("series"), envelope=s.get("envelope"))
    return sc

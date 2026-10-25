"""Triangle meshes of extended-surface parts for viewers and STL export: a plate-fin heat sink and a vented brake
rotor (two friction rings joined by radial vanes). Display solids (overlapping closed shells, not a boolean union);
lengths in millimetres.
"""
from __future__ import annotations

import math

import numpy as np

from pinneapple_design.bricks.mesh import box_mesh

__all__ = ["plate_fin_mesh", "ring_mesh", "vented_rotor_mesh", "merge", "mesh_payload", "stl_bytes"]


def merge(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate (V, F) meshes."""
    Vs, Fs, off = [], [], 0
    for V, F in parts:
        Vs.append(V)
        Fs.append(F + off)
        off += len(V)
    return np.vstack(Vs), np.vstack(Fs)


def plate_fin_mesh(width_mm: float, depth_mm: float, base_mm: float, n_fins: int, fin_t_mm: float,
                   fin_h_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Base plate (x across the fins, y along them, z up) with ``n_fins`` evenly spaced fins, outer fins flush."""
    parts = [box_mesh((0, 0, 0), (width_mm, depth_mm, base_mm))]
    n = max(int(round(n_fins)), 1)
    step = (width_mm - fin_t_mm) / (n - 1) if n > 1 else 0.0
    for i in range(n):
        x0 = i * step if n > 1 else (width_mm - fin_t_mm) / 2
        parts.append(box_mesh((x0, 0, base_mm), (x0 + fin_t_mm, depth_mm, base_mm + fin_h_mm)))
    return merge(parts)


def ring_mesh(r_in: float, r_out: float, z0: float, z1: float, n: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """Closed annular ring (outward faces)."""
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    c, s = np.cos(a), np.sin(a)
    V = np.vstack([np.c_[r_out * c, r_out * s, np.full(n, z0)], np.c_[r_out * c, r_out * s, np.full(n, z1)],
                   np.c_[r_in * c, r_in * s, np.full(n, z0)], np.c_[r_in * c, r_in * s, np.full(n, z1)]])
    F = []
    for i in range(n):
        j = (i + 1) % n
        ob, ot, ib, it = i, n + i, 2 * n + i, 3 * n + i
        ob2, ot2, ib2, it2 = j, n + j, 2 * n + j, 3 * n + j
        F += [[ob, ob2, ot2], [ob, ot2, ot],           # outer wall
              [ib, it2, ib2], [ib, it, it2],           # inner wall
              [ot, ot2, it2], [ot, it2, it],           # top
              [ob, ib2, ob2], [ob, ib, ib2]]           # bottom
    return V, np.asarray(F, np.int64)


def vented_rotor_mesh(r_mean_mm: float, radial_len_mm: float, ring_t_mm: float, gap_mm: float, n_vanes: int,
                      vane_t_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Two friction rings (radial extent ``radial_len_mm`` around ``r_mean_mm``) and ``n_vanes`` radial vanes in the
    gap between them."""
    r_in, r_out = r_mean_mm - radial_len_mm / 2, r_mean_mm + radial_len_mm / 2
    parts = [ring_mesh(r_in, r_out, 0.0, ring_t_mm), ring_mesh(r_in, r_out, ring_t_mm + gap_mm, 2 * ring_t_mm + gap_mm)]
    V0, F0 = box_mesh((r_in, -vane_t_mm / 2, ring_t_mm), (r_out, vane_t_mm / 2, ring_t_mm + gap_mm))
    for k in range(max(int(round(n_vanes)), 1)):
        a = 2 * math.pi * k / n_vanes
        R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        parts.append((V0 @ R.T, F0))
    return merge(parts)


def mesh_payload(V: np.ndarray, F: np.ndarray, decimals: int = 4) -> dict:
    """JSON-able ``{"positions": [...], "indices": [...]}`` for a three.js BufferGeometry."""
    return {"positions": np.round(np.asarray(V, float), decimals).ravel().tolist(),
            "indices": np.asarray(F, int).ravel().tolist()}


def stl_bytes(V: np.ndarray, F: np.ndarray, header: str = "pinneapple mesh") -> bytes:
    """Binary STL (``pinneapple_design.geometry_library.exporters.stl_bytes``)."""
    from pinneapple_design.geometry_library.exporters import stl_bytes as _stl

    return _stl(V, F, header)

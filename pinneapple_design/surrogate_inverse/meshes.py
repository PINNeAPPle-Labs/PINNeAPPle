"""Triangle meshes of the five designs, in millimetres, for a 3-D viewer or an STL download.

Each part is a set of individually closed bodies (``pinneapple_design.geometry.bodies``) concatenated, not a boolean
union: the fins of a heat sink or the vanes of a rotor touch their base without being fused into one manifold. That
is right for display and for a slicer that accepts several shells; a CNC or FEM pipeline that needs one watertight
body has to fuse them (for example with ``manifold3d``).
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from pinneapple_design.geometry.bodies import annulus, box, cylinder
from pinneapple_design.geometry_library.exporters import stl_bytes

Mesh = tuple[np.ndarray, np.ndarray]


def merge(parts: Iterable[Mesh]) -> Mesh:
    """Concatenate meshes (faces re-indexed)."""
    Vs, Fs, off = [], [], 0
    for V, F in parts:
        Vs.append(np.asarray(V, float))
        Fs.append(np.asarray(F, np.int64) + off)
        off += len(V)
    return np.vstack(Vs), np.vstack(Fs)


def bbox_size(mesh: Mesh) -> tuple[float, float, float]:
    V = mesh[0]
    ext = V.max(axis=0) - V.min(axis=0)
    return float(ext[0]), float(ext[1]), float(ext[2])


def to_stl(mesh: Mesh, name: str) -> bytes:
    """Binary STL (the header starts with the part name, never with ``solid``)."""
    return stl_bytes(mesh[0], mesh[1], f"PINNeAPPle {name}")


def _z_cylinder(radius: float, height: float, z0: float = 0.0, n: int = 64) -> Mesh:
    V, F = cylinder(radius=radius, length=height, axis="z", n=n)
    return V + np.array([0.0, 0.0, z0 + height / 2]), F


def fin_array(base_w_mm: float, base_d_mm: float, base_t_mm: float, n_fins: int, fin_h_mm: float,
              fin_t_mm: float) -> Mesh:
    """Base plate (centred on the origin, top face at ``z = base_t / 2``) with ``n_fins`` straight fins spread evenly
    across its width, first and last flush with the edges."""
    n = max(int(n_fins), 1)
    spacing = (base_w_mm - n * fin_t_mm) / (n - 1) if n > 1 else 0.0
    pitch = fin_t_mm + spacing
    x0 = -(n - 1) * pitch / 2.0
    parts = [box((base_w_mm, base_d_mm, base_t_mm))]
    parts += [box((fin_t_mm, base_d_mm, fin_h_mm), (x0 + i * pitch, 0.0, base_t_mm / 2 + fin_h_mm / 2))
              for i in range(n)]
    return merge(parts)


def vented_rotor(r_mean_mm: float, radial_len_mm: float, ring_t_mm: float, gap_mm: float, n_vanes: int,
                 vane_t_mm: float, n: int = 96) -> Mesh:
    """Two annular friction rings (z = 0 to ``ring_t`` and above the vent gap) joined by ``n_vanes`` radial vane
    blocks in the gap, around the z axis."""
    r_in, r_out = r_mean_mm - radial_len_mm / 2, r_mean_mm + radial_len_mm / 2
    parts = [annulus(r_in, r_out, ring_t_mm, n=n), annulus(r_in, r_out, ring_t_mm, n=n, z0=ring_t_mm + gap_mm)]
    Vb, Fb = box((radial_len_mm, vane_t_mm, gap_mm), (r_mean_mm, 0.0, ring_t_mm + gap_mm / 2))
    for i in range(max(int(n_vanes), 1)):
        a = 2 * math.pi * i / max(int(n_vanes), 1)
        R = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
        parts.append((Vb @ R.T, Fb))
    return merge(parts)


def layered_disc(radius_mm: float, layer_thicknesses_mm: Iterable[float], n: int = 96) -> Mesh:
    """Stacked coaxial discs (z up from 0), e.g. a drug reservoir under a rate-controlling membrane."""
    parts, z = [], 0.0
    for t in layer_thicknesses_mm:
        parts.append(_z_cylinder(radius_mm, t, z0=z, n=n))
        z += t
    return merge(parts)


def rectangular_beam(width_mm: float, thickness_mm: float, length_mm: float) -> Mesh:
    """A plain box: x = width, y = thickness (the bending axis), z = length, centred on the origin (12 triangles)."""
    return box((width_mm, thickness_mm, length_mm))


def flanged_tube(radius_mm: float, length_mm: float, flange_radius_mm: float, flange_t_mm: float,
                 n: int = 64) -> Mesh:
    """A port: a cylinder of ``radius`` x ``length`` along z from 0, with a flange disc at its base."""
    return merge([_z_cylinder(radius_mm, length_mm, n=n), _z_cylinder(flange_radius_mm, flange_t_mm, n=n)])

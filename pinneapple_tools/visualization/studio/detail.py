"""Mechanical detail: the fasteners, welds and bosses a figure needs and a solver mesh must not have.

A CFD or FEA mesh is right to be smooth -- every bolt head is a place where a boundary layer goes wrong. A figure is
the opposite: at photographic distance the eye reads scale from exactly those details, and a casing with no seam, no
fastener and no weld looks like a render of a mesh, which is what it is.

These builders stay out of the physics. They take the geometry you already have, put the detail where a fabricator
would have put it, and hand back :class:`~.scene.Surface` objects to add to the scene:

    from pinneapple_tools.visualization.studio import detail
    sc.surfaces += detail.bolt_circle(hub_centre, normal=(0, 0, 1), radius=0.11, count=8, head_radius=0.008)
    sc.surfaces.append(detail.weld_bead(root_path, radius=0.004))

Everything is generated, so it is reproducible, parameterised and cheap: a detail level is a number in a script, not a
second CAD model to keep in sync.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .scene import Surface, rod_surface, sphere_surface


def _basis(normal: Sequence[float]):
    """Two unit vectors perpendicular to ``normal`` and to each other (the plane a bolt circle lies in)."""
    n = np.asarray(normal, float)
    n = n / (np.linalg.norm(n) or 1.0)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, helper)
    u /= np.linalg.norm(u) or 1.0
    return n, u, np.cross(n, u)


def hex_bolt(centre, normal=(0.0, 0.0, 1.0), *, head_radius: float = 0.01, height: float = 0.006,
             washer: bool = True, material: str = "steel", name: str = "bolt") -> Surface:
    """One hex-head bolt standing on ``centre`` along ``normal``: a hexagonal prism with a flat top, and under it a
    washer a hair wider and a hair shorter, which is the lip that makes the head read at a distance."""
    c = np.asarray(centre, float)
    n, u, v = _basis(normal)
    ang = np.linspace(0.0, 2.0 * np.pi, 7)[:-1]
    ring = np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v
    top = c + n * height + ring * head_radius
    base = c + ring * head_radius
    V = [top, base]
    F = []
    for i in range(6):
        j = (i + 1) % 6
        F.append([0 + i, 0 + j, 6 + j])                      # side, as two triangles
        F.append([0 + i, 6 + j, 6 + i])
    F += [[0, k, k + 1] for k in range(1, 5)]                 # the top, fanned from its first vertex
    V = np.vstack(V)
    if washer:
        wr, wh = head_radius * 1.55, max(height * 0.25, 1e-4)
        wtop = c + n * wh + ring * wr
        wbase = c + ring * wr
        off = len(V)
        V = np.vstack([V, wtop, wbase])
        for i in range(6):
            j = (i + 1) % 6
            F.append([off + i, off + j, off + 6 + j])
            F.append([off + i, off + 6 + j, off + 6 + i])
    return Surface(name, V, np.asarray(F, np.int64), material)


def bolt_circle(centre, normal=(0.0, 0.0, 1.0), *, radius: float = 0.1, count: int = 8,
                head_radius: float = 0.01, height: float = 0.006, washer: bool = True,
                material: str = "steel", name: str = "bolts") -> List[Surface]:
    """``count`` bolts evenly spaced on a circle of ``radius`` around ``centre``, in the plane of ``normal``, the first
    one on the ``u`` axis of that plane (so two runs with the same arguments put them in the same places)."""
    c = np.asarray(centre, float)
    _, u, v = _basis(normal)
    out = []
    for i in range(int(count)):
        a = 2.0 * np.pi * i / int(count)
        p = c + radius * (np.cos(a) * u + np.sin(a) * v)
        out.append(hex_bolt(p, normal, head_radius=head_radius, height=height, washer=washer, material=material,
                            name=f"{name}{i:02d}"))
    return out


def weld_bead(points, *, radius: float = 0.004, material: str = "steel", name: str = "weld",
              segments: int = 12) -> Surface:
    """A weld along ``points``: cylinders between consecutive points with a sphere at each joint, so the bead reads as
    one continuous run rather than a set of tubes. The radius is the bead's, which is how a weld is called out."""
    P = np.asarray(points, float).reshape(-1, 3)
    if len(P) < 2:
        raise ValueError("a weld needs at least two points")
    parts = [rod_surface(P[i], P[i + 1], radius, name=f"{name}_{i}", material=material, n=segments)
             for i in range(len(P) - 1)]
    parts += [sphere_surface(p, radius, name=f"{name}_j{i}", material=material) for i, p in enumerate(P[1:-1])]
    V = np.vstack([p.vertices for p in parts])
    F, off = [], 0
    for p in parts:
        F.append(p.faces + off)
        off += len(p.vertices)
    return Surface(name, V, np.vstack(F), material)


def seam_ring(centre, normal=(0.0, 0.0, 1.0), *, radius: float = 0.1, thickness: float = 0.004,
              material: str = "steel", name: str = "seam", n: int = 48) -> Surface:
    """A raised ring where two cast halves meet: a torus of ``thickness`` around a circle of ``radius``. It is what
    tells a viewer the part is assembled instead of printed."""
    c = np.asarray(centre, float)
    _, u, v = _basis(normal)
    nn = np.cross(u, v)
    ang = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    tube = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)   # a multiple of four: the tube reaches its full width
    V, F = [], []
    for i, a in enumerate(ang):
        ring_c = c + radius * (np.cos(a) * u + np.sin(a) * v)
        radial = np.cos(a) * u + np.sin(a) * v
        for t in tube:
            V.append(ring_c + thickness * (np.cos(t) * radial + np.sin(t) * nn))
    m = len(tube)
    for i in range(len(ang)):
        for j in range(m):
            a0, b0 = i * m + j, i * m + (j + 1) % m
            a1, b1 = ((i + 1) % len(ang)) * m + j, ((i + 1) % len(ang)) * m + (j + 1) % m
            F += [[a0, b0, b1], [a0, b1, a1]]
    return Surface(name, np.asarray(V, float), np.asarray(F, np.int64), material)


__all__ = ["hex_bolt", "bolt_circle", "weld_bead", "seam_ring"]

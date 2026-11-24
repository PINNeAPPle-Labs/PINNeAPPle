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

from typing import List, Optional, Sequence

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


def pad(centre, normal=(0.0, 0.0, 1.0), *, radius: float = 0.02, height: float = 0.006, chamfer: float = 0.0015,
        material: str = "steel", name: str = "pad", n: int = 40) -> Surface:
    """A machined pad: a boss that stands proud of a casting so a bolt or a cover has something flat to land on. The
    top edge is chamfered, which is the difference between a machined face and an extruded circle."""
    c = np.asarray(centre, float)
    nn, u, v = _basis(normal)
    ang = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    ring = np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v
    ch = min(chamfer, height * 0.5, radius * 0.5)
    base = c + ring * radius
    shoulder = c + nn * (height - ch) + ring * radius
    top = c + nn * height + ring * (radius - ch)
    m = len(ang)
    V = np.vstack([base, shoulder, top, c + nn * height])          # 3 rings and the centre of the top face
    apex = 3 * m
    F = []
    for i in range(m):
        j = (i + 1) % m
        F += [[i, j, m + j], [i, m + j, m + i]]                    # the wall
        F += [[m + i, m + j, 2 * m + j], [m + i, 2 * m + j, 2 * m + i]]   # the chamfer
        F.append([2 * m + i, 2 * m + j, apex])                     # the face, fanned from its centre
    return Surface(name, V, np.asarray(F, np.int64), material)


def gusset(a, b, c, *, thickness: float = 0.006, material: str = "steel", name: str = "gusset") -> Surface:
    """A stiffening rib: the triangle ``a``-``b``-``c``, extruded ``thickness`` along its own normal. Two plates at a
    right angle and a rib across the corner is how a welded frame gets its stiffness, and the triangle is what a viewer
    reads as one."""
    A, B, C = (np.asarray(p, float).reshape(3) for p in (a, b, c))
    nv = np.cross(B - A, C - A)
    L = float(np.linalg.norm(nv))
    if L < 1e-12:
        raise ValueError("a gusset needs three points that are not on a line")
    nv = nv / L * float(thickness)
    V = np.vstack([A, B, C, A + nv, B + nv, C + nv])
    F = [[0, 1, 2], [3, 5, 4],
         [0, 3, 4], [0, 4, 1],        # A-B edge
         [1, 4, 5], [1, 5, 2],        # B-C edge
         [2, 5, 3], [2, 3, 0]]        # C-A edge
    return Surface(name, V, np.asarray(F, np.int64), material)


def _cone(a, b, r0: float, r1: float, *, name: str, material: str, n: int = 48) -> Surface:
    """A truncated cone from ``a`` (radius ``r0``) to ``b`` (radius ``r1``): a weld neck, a reducer, a handrail foot."""
    A, B = np.asarray(a, float).reshape(3), np.asarray(b, float).reshape(3)
    d = B - A
    L = float(np.linalg.norm(d))
    d = d / (L or 1.0)
    u = np.cross(d, [0.0, 0.0, 1.0])
    u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-9 else np.array([1.0, 0.0, 0.0])
    v = np.cross(d, u)
    ph = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    ring = np.cos(ph)[:, None] * u + np.sin(ph)[:, None] * v
    V = np.vstack([A + r0 * ring, B + r1 * ring])
    F = []
    for i in range(int(n)):
        j = (i + 1) % int(n)
        F += [[i, j, int(n) + j], [i, int(n) + j, int(n) + i]]
    return Surface(name, V, np.asarray(F, np.int64), material)


def pipe_flange(centre, axis=(0.0, 0.0, 1.0), *, pipe_radius: float = 0.035, flange_radius: float = 0.075,
                thickness: float = 0.018, raised_face: float = 0.002, neck: float = 0.03, bolts: int = 4,
                bolt_radius: Optional[float] = None, head_radius: float = 0.008, material: str = "steel",
                bolt_material: Optional[str] = None, name: str = "flange",
                n: int = 64) -> List[Surface]:
    """A whole connection, not a disc: the plate, the raised face the gasket sits on, the weld neck that tapers into the
    pipe and the bolts on the face. This is the item the audit called "conexões" — a flange drawn as one cylinder reads
    as a puck, and the three parts together read as a joint.

    Returns the surfaces; add them to the scene with ``sc.surfaces += ...``."""
    c = np.asarray(centre, float).reshape(3)
    ax = np.asarray(axis, float).reshape(3)
    ax = ax / (np.linalg.norm(ax) or 1.0)
    bolt_material = bolt_material or material
    bolt_radius = flange_radius * 0.78 if bolt_radius is None else float(bolt_radius)
    face = c + ax * float(thickness)
    out = [rod_surface(c, face, float(flange_radius), name=name, material=material, n=n),
           rod_surface(face, face + ax * float(raised_face), float(flange_radius) * 0.72,
                       name=f"{name}_face", material=material, n=n)]
    if neck > 0.0:
        out.append(_cone(c - ax * float(neck), c, float(pipe_radius) * 1.25, float(pipe_radius) * 1.05,
                         name=f"{name}_neck", material=material, n=max(24, n // 2)))
    out += bolt_circle(face + ax * float(raised_face), ax, radius=bolt_radius, count=int(bolts),
                       head_radius=float(head_radius), height=float(head_radius) * 0.8, washer=True,
                       material=bolt_material, name=f"{name}_bolt")
    return out


#: What each detail level puts on a body. A level is a recipe, not a switch: the figure script applies the builders
#: with the measurements of the part, because where a rib goes is a question about the part. ``draft`` is the solver
#: geometry as it is; from ``review`` up the picture carries what a fabricator would have added.
LEVELS = {
    "draft": (),
    "review": ("seam_ring",),
    "cinematic": ("seam_ring", "bolt_circle", "weld_bead", "pad", "gusset", "pipe_flange"),
}


__all__ = ["hex_bolt", "bolt_circle", "weld_bead", "seam_ring", "pad", "gusset", "pipe_flange", "LEVELS"]


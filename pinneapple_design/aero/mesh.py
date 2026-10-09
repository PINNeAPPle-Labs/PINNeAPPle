"""Structured O-grid around an airfoil, written straight to an OpenFOAM polyMesh.

Every design gets the same topology (ni x nj cells, same numbering, same faces), only the points move. That is what
lets one graph network see every design on the same graph, and lets results be compared cell by cell.

Points: wall offsets along smoothed normals near the wall (orthogonal boundary layer, first cell for y+ ~ 1),
blended into straight rays to a far-field circle of radius R chord lengths.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .geometry import Loop, wall_loop


@dataclass
class GridSpec:
    n_surf: int = 112         # cells along each surface
    n_te: int = 4             # cells across the blunt trailing edge
    nj: int = 88              # cells wall -> far field
    first: float = 4e-6       # first cell height / chord (y+ ~ 1 at Re 4e6)
    radius: float = 200.0     # far field radius / chord (lift needs it far: 25 c gave +48 % drag at 10 deg)
    span: float = 0.05        # 2D: one cell in z

    @property
    def ni(self) -> int:
        return 2 * self.n_surf + self.n_te


def _distances(first: float, radius: float, nj: int) -> np.ndarray:
    lo, hi = 1.0001, 2.0
    for _ in range(100):                                          # growth ratio with first*(r^nj-1)/(r-1) = radius
        r = (lo + hi) / 2
        if first * (r ** nj - 1) / (r - 1) > radius:
            hi = r
        else:
            lo = r
    return np.concatenate([[0], first * (r ** np.arange(1, nj + 1) - 1) / (r - 1)])


def _normals(xy: np.ndarray, smooth: int = 30) -> np.ndarray:
    t = np.roll(xy, -1, 0) - np.roll(xy, 1, 0)
    n = np.stack([-t[:, 1], t[:, 0]], 1)                          # clockwise loop -> outward
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    for _ in range(smooth):                                       # spreads the fan at the trailing-edge corners
        n = 0.5 * n + 0.25 * (np.roll(n, 1, 0) + np.roll(n, -1, 0))
        n /= np.linalg.norm(n, axis=1, keepdims=True)
    return n


def grid_points(shape: np.ndarray, g: GridSpec = GridSpec()) -> np.ndarray:
    """(nj+1, ni, 2) points; j = 0 is the wall."""
    loop: Loop = wall_loop(shape, g.n_surf, g.n_te)
    w = loop.xy
    ni = len(w)
    d = _distances(g.first, g.radius, g.nj)
    n0 = _normals(w, smooth=2)                                    # nearly true normals at the wall
    n1 = _normals(w, smooth=200)                                  # very smooth away from it
    centre = np.array([0.5, 0.0])
    # far-field angle for each wall point: by arc length, starting downstream (TE base at angle 0, clockwise)
    seg = np.linalg.norm(np.roll(w, -1, 0) - w, axis=1)
    s = np.concatenate([[0], np.cumsum(seg)[:-1]]) / seg.sum()
    s = s - s[loop.n_te // 2]                                     # mid trailing-edge face -> angle 0
    th = -2 * np.pi * s
    far = centre + g.radius * np.stack([np.cos(th), np.sin(th)], 1)
    P = np.empty((g.nj + 1, ni, 2))
    for j, dj in enumerate(d):
        a = np.clip(dj / 0.02, 0, 1)                              # normal blend: true normal -> smooth normal
        nrm = (1 - a) * n0 + a * n1
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        near = w + dj * nrm
        ray = w + (far - w) * (dj / g.radius)
        b = 0.5 - 0.5 * np.cos(np.pi * np.clip((np.log10(max(dj, 1e-9)) + 1.3) / 1.6, 0, 1))  # 0 below 0.05c, 1 above 2c
        P[j] = (1 - b) * near + b * ray
    return P


# ------------------------------------------------------------------ topology (identical for every design)
def topology(g: GridSpec = GridSpec()) -> Dict[str, np.ndarray]:
    """Faces of the 2D grid in OpenFOAM order, as (i, j) point indices of the bottom layer; cells c = j*ni + i."""
    ni, nj = g.ni, g.nj
    pid = lambda i, j: j * ni + (i % ni)                          # noqa: E731  bottom-layer point id
    cid = lambda i, j: j * ni + (i % ni)                          # noqa: E731
    internal: List[Tuple[int, int, Tuple[int, int]]] = []          # owner, neighbour, (p0, p1) edge in 2D
    for j in range(nj):
        for i in range(ni):
            a, b = cid(i, j), cid(i + 1, j)                       # i-face between (i,j) and (i+1,j): edge at i+1
            internal.append((min(a, b), max(a, b), (pid(i + 1, j), pid(i + 1, j + 1))))
            if j < nj - 1:
                a, b = cid(i, j), cid(i, j + 1)
                internal.append((a, b, (pid(i, j + 1), pid(i + 1, j + 1))))
    internal.sort(key=lambda f: (f[0], f[1]))
    wall = [(cid(i, 0), (pid(i, 0), pid(i + 1, 0))) for i in range(ni)]
    far = [(cid(i, nj - 1), (pid(i, nj), pid(i + 1, nj))) for i in range(ni)]
    return {"internal": internal, "wall": wall, "far": far}


def write_polymesh(shape: np.ndarray, case: str, g: GridSpec = GridSpec()) -> np.ndarray:
    """Write constant/polyMesh for this shape; returns the (nj+1, ni, 2) points."""
    P2 = grid_points(shape, g)
    ni, nj = g.ni, g.nj
    flat = P2.reshape(-1, 2)
    npl = len(flat)
    pts = np.concatenate([np.c_[flat, np.zeros(npl)], np.c_[flat, np.full(npl, g.span)]])
    topo = topology(g)
    # cell centres (2D) to orient faces owner -> neighbour
    q = P2
    centres = np.empty((nj, ni, 2))
    for i in range(ni):
        a, b = i, (i + 1) % ni
        centres[:, i] = 0.25 * (q[:-1, a] + q[:-1, b] + q[1:, a] + q[1:, b])
    centres = centres.reshape(-1, 2)

    faces, owner, neigh = [], [], []

    def quad(p0, p1, toward):                                     # side face from 2D edge, normal toward a point
        f = [p0, p1, p1 + npl, p0 + npl]
        e = flat[p1] - flat[p0]
        nrm = np.array([e[1], -e[0]])                             # normal of the extruded face (z-extrusion)
        mid = 0.5 * (flat[p0] + flat[p1])
        return f if nrm @ (toward - mid) > 0 else f[::-1]

    for o, nb, (p0, p1) in topo["internal"]:
        faces.append(quad(p0, p1, centres[nb])); owner.append(o); neigh.append(nb)
    n_int = len(faces)
    patches = []
    for name, lst in (("airfoil", topo["wall"]), ("farfield", topo["far"])):
        start = len(faces)
        for o, (p0, p1) in lst:
            mid = 0.5 * (flat[p0] + flat[p1])
            faces.append(quad(p0, p1, 2 * mid - centres[o])); owner.append(o)
        patches.append((name, "wall" if name == "airfoil" else "patch", start, len(lst)))
    start = len(faces)
    for side in (0, 1):                                           # front/back: empty
        for j in range(nj):
            for i in range(ni):
                a, b = j * ni + i, j * ni + (i + 1) % ni
                c, d = (j + 1) * ni + (i + 1) % ni, (j + 1) * ni + i
                f = [a, b, c, d] if side else [a, d, c, b]         # z+ normal on top, z- at the bottom
                if side:
                    f = [v + npl for v in f]
                # check orientation from geometry
                v = pts[f]
                nrm = np.cross(v[1] - v[0], v[2] - v[0])
                if (nrm[2] > 0) != bool(side):
                    f = f[::-1]
                faces.append(f); owner.append(j * ni + i)
    patches.append(("frontAndBack", "empty", start, 2 * ni * nj))

    d = os.path.join(case, "constant", "polyMesh")
    os.makedirs(d, exist_ok=True)
    hdr = lambda cls, obj: ("FoamFile\n{\n    version 2.0;\n    format ascii;\n    class %s;\n    location \"constant/polyMesh\";\n"  # noqa: E731
                            "    object %s;\n}\n" % (cls, obj))
    with open(os.path.join(d, "points"), "w") as f:
        f.write(hdr("vectorField", "points") + f"{len(pts)}\n(\n")
        f.write("\n".join(f"({x:.9g} {y:.9g} {z:.9g})" for x, y, z in pts) + "\n)\n")
    with open(os.path.join(d, "faces"), "w") as f:
        f.write(hdr("faceList", "faces") + f"{len(faces)}\n(\n")
        f.write("\n".join("4(%d %d %d %d)" % tuple(x) for x in faces) + "\n)\n")
    note = f"nPoints:{len(pts)}  nCells:{ni * nj}  nFaces:{len(faces)}  nInternalFaces:{n_int}"
    for name, lst in (("owner", owner), ("neighbour", neigh)):
        with open(os.path.join(d, name), "w") as f:
            f.write(hdr("labelList", name).replace("}\n", f"    note \"{note}\";\n}}\n") + f"{len(lst)}\n(\n")
            f.write("\n".join(map(str, lst)) + "\n)\n")
    with open(os.path.join(d, "boundary"), "w") as f:
        f.write(hdr("polyBoundaryMesh", "boundary") + f"{len(patches)}\n(\n")
        for name, typ, s, n in patches:
            extra = "        inGroups 1(wall);\n" if typ == "wall" else ""
            f.write(f"    {name}\n    {{\n        type {typ};\n{extra}        nFaces {n};\n        startFace {s};\n    }}\n")
        f.write(")\n")
    return P2


def cell_centres(P2: np.ndarray) -> np.ndarray:
    """(nj, ni, 2) cell centres of the grid points from grid_points."""
    q = np.concatenate([P2, P2[:, :1]], 1)
    return 0.25 * (q[:-1, :-1] + q[:-1, 1:] + q[1:, :-1] + q[1:, 1:])

"""Surface-quality diagnostics of exact bodies (#1117): curvature, G0/G1/G2 continuity, zebra stripes, highlight lines.

* :func:`exact_curvature` evaluates the principal curvatures on the B-rep (the kernel's surface derivatives) at the
  vertices of a :class:`~.tessellate.TessMesh`, with the outward normals; :func:`discrete_curvature` estimates them
  from the mesh alone (angle deficit for K, cotangent Laplacian for H). Their difference is where the mesh is too
  coarse to carry the shape.
* :func:`edge_continuity` classifies every edge between two faces as G0 (they meet at an angle), G1 (tangent) or G2
  (curvature continuous), with the measured gap, normal angle and curvature jump along the edge.
* :func:`zebra` and :func:`highlight_lines` are the reflection tests designers judge surfaces by. Zebra: the
  reflection of a cylinder of parallel stripes around the part, seen from an eye point; stripes run smoothly across
  G2 joins, kink at G1 and jump at G0. Highlight lines (Beier & Chen 1994): the distance between the surface-normal
  line through each point and a set of parallel straight light lines; the lines bend where the surface does.
* :func:`render` draws any per-vertex value on the mesh to a PNG.

Units: metres and 1/m. Curvature signs follow the kernel: positive where the surface bends away from its outward
normal (a ball seen from outside), negative in a bore.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .step import MM, CadBody
from .tessellate import TessMesh


def vertex_face(mesh: TessMesh) -> np.ndarray:
    """The body face each vertex is evaluated on (the face of the first triangle that uses it)."""
    vf = np.full(len(mesh.vertices), -1, np.int64)
    flat = mesh.triangles.ravel()
    faces = np.repeat(mesh.face_id, 3)
    order = np.argsort(flat, kind="stable")
    first = np.ones(len(order), bool)
    first[1:] = flat[order][1:] != flat[order][:-1]
    vf[flat[order][first]] = faces[order][first]
    return vf


def refine(mesh: TessMesh, max_edge_m: float | None = None, max_triangles: int = 200_000) -> TessMesh:
    """Split every triangle 1 -> 4 at its edge midpoints until no edge is longer than ``max_edge_m`` (default: the
    bounding-box diagonal / 60), or until another round would exceed ``max_triangles``.

    Face ids carry over, so the exact surface quantities can be evaluated at the new vertices; the midpoints stay on
    the chords (they are not projected), which is within the mesh's own chordal tolerance. Reflection tests need this:
    a planar face is meshed with its corners only, and a zebra stripe cannot be drawn between four vertices."""
    from dataclasses import replace

    V, T, F = mesh.vertices, mesh.triangles, mesh.face_id
    if max_edge_m is None:
        max_edge_m = float(np.linalg.norm(V.max(0) - V.min(0))) / 60.0
    for _ in range(12):
        L = np.max(np.linalg.norm(V[T[:, [1, 2, 0]]] - V[T], axis=2), axis=1)
        if L.max() <= max_edge_m or 4 * len(T) > max_triangles:
            break
        e = np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
        uniq, inv = np.unique(e, axis=0, return_inverse=True)
        mid = len(V) + inv.reshape(3, -1).T                  # midpoint vertex of edges (01, 12, 20) per triangle
        V = np.vstack([V, 0.5 * (V[uniq[:, 0]] + V[uniq[:, 1]])])
        a, b, c = T[:, 0], T[:, 1], T[:, 2]
        ab, bc, ca = mid[:, 0], mid[:, 1], mid[:, 2]
        T = np.vstack([np.c_[a, ab, ca], np.c_[ab, b, bc], np.c_[ca, bc, c], np.c_[ab, bc, ca]])
        F = np.tile(F, 4)
    return replace(mesh, vertices=V, triangles=T, face_id=F)


def exact_curvature(body: CadBody, mesh: TessMesh) -> dict[str, np.ndarray]:
    """Principal curvatures k1 >= k2 [1/m], Gaussian K = k1 k2 [1/m^2], mean H = (k1 + k2)/2 [1/m] and the outward
    unit normal at every vertex, from the exact surface of the face the vertex lies on."""
    vf = vertex_face(mesh)
    n = len(mesh.vertices)
    k1, k2 = np.zeros(n), np.zeros(n)
    nrm = np.zeros((n, 3))
    pts_mm = mesh.vertices / MM
    for i in range(n):
        f = body.faces[vf[i]].shape
        c = f.curvature_at(pts_mm[i].tolist())
        k1[i], k2[i] = c.k1 / MM, c.k2 / MM
        nrm[i] = list(c.normal)
    return {"k1": k1, "k2": k2, "K": k1 * k2, "H": 0.5 * (k1 + k2), "normal": nrm, "vertex_face": vf}


def _cot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(1) / np.maximum(np.linalg.norm(np.cross(a, b), axis=1), 1e-300)


def discrete_curvature(mesh: TessMesh) -> dict[str, np.ndarray]:
    """Gaussian (angle deficit over the barycentric area) and mean curvature (cotangent Laplacian) from the mesh only.

    Mean curvature is signed with the mesh orientation (positive for a convex body); feature edges (sharp creases)
    show up as large values, which is expected of a discrete estimate there."""
    V, T = mesh.vertices, mesh.triangles
    n = len(V)
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    A = np.bincount(T.ravel(), weights=np.repeat(area / 3.0, 3), minlength=n)

    def ang(p, q, r):
        u, v = q - p, r - p
        return np.arctan2(np.linalg.norm(np.cross(u, v), axis=1), (u * v).sum(1))

    angle_sum = (np.bincount(T[:, 0], ang(a, b, c), n) + np.bincount(T[:, 1], ang(b, c, a), n)
                 + np.bincount(T[:, 2], ang(c, a, b), n))
    K = (2 * np.pi - angle_sum) / np.maximum(A, 1e-300)
    L = np.zeros((n, 3))
    for (i, j, k) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        pi_, pj, pk = V[T[:, i]], V[T[:, j]], V[T[:, k]]
        w = 0.5 * _cot(pi_ - pk, pj - pk)                 # weight of edge (i, j), angle at k
        d = pj - pi_
        np.add.at(L, T[:, i], w[:, None] * d)
        np.add.at(L, T[:, j], -w[:, None] * d)
    Hn = -L / (2 * np.maximum(A, 1e-300))[:, None]       # = H * outward normal for a closed, outward mesh
    fn = np.cross(b - a, c - a)
    vn = np.zeros((n, 3))
    for k in range(3):
        np.add.at(vn, T[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-300)
    H = (Hn * vn).sum(1)
    return {"K": K, "H": H, "normal": vn, "area": A}


def edge_continuity(body: CadBody, samples: int = 16, angle_tolerance_deg: float = 0.5,
                    curvature_tolerance: float = 0.01) -> list[dict[str, Any]]:
    """How the two faces meet along every edge: level G0 / G1 / G2 (None when they do not meet), the largest gap [m],
    normal angle [deg] and curvature jump [1/m] along the edge, and the indices of the two faces.

    ``curvature_tolerance`` is relative (the kernel's convention); tolerances are the kernel's defaults."""
    faces_of = {}
    for cf in body.faces:
        for e in cf.shape.edges():
            key = tuple(round(c, 6) for c in e.point_at(0.5))
            faces_of.setdefault(key, []).append(cf.index)
    out = []
    for i, e in enumerate(body.shape.edges()):
        key = tuple(round(c, 6) for c in e.point_at(0.5))
        fs = sorted(set(faces_of.get(key, [])))
        if len(fs) < 2:
            out.append({"edge": i, "faces": fs, "level": None, "gap_m": None, "angle_deg": None,
                        "curvature_jump_per_m": None, "seam": True})
            continue
        c = e.continuity(samples, angle_tolerance=angle_tolerance_deg, curvature_tolerance=curvature_tolerance)
        out.append({"edge": i, "faces": fs, "level": c.level, "gap_m": float(c.gap) * MM, "angle_deg": float(c.angle),
                    "curvature_jump_per_m": float(c.curvature_jump) / MM, "seam": False,
                    "point_m": [float(x) * MM for x in c.point]})
    return out


def zebra(mesh: TessMesh, normals: np.ndarray, *, eye: Sequence[float] | None = None,
          axis: Sequence[float] = (0.0, 1.0, 0.0), n_stripes: int = 16, sharpness: float = 8.0,
          view: Sequence[float] | None = None) -> np.ndarray:
    """Zebra value in [0, 1] per vertex: the reflection of a cylinder of ``n_stripes`` black-and-white stripe pairs
    whose axis is ``axis``, seen from ``eye`` (default: 3 bounding-box sizes in front of the part along +z).

    The stripe index is the azimuth of the reflected view ray around ``axis``; ``sharpness`` controls the edge
    softness (large = hard stripes). ``view``: a direction from the part towards a distant eye instead of ``eye``
    (parallel view rays: every planar face is one uniform value, and stripes appear only where the surface bends)."""
    V = mesh.vertices
    if view is not None:
        d = -np.asarray(view, float) / np.linalg.norm(view)
        d = np.broadcast_to(d, V.shape).copy()
    elif eye is None:
        lo, hi = V.min(0), V.max(0)
        eye = 0.5 * (lo + hi) + np.array([0.0, 0.0, 3.0 * float(np.linalg.norm(hi - lo))])
    if view is None:
        d = V - np.asarray(eye, float)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
    n = np.asarray(normals, float)
    r = d - 2 * (d * n).sum(1, keepdims=True) * n
    ax = np.asarray(axis, float)
    ax /= np.linalg.norm(ax)
    e1 = np.cross(ax, [0.0, 0.0, 1.0] if abs(ax[2]) < 0.9 else [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ax, e1)
    phi = np.arctan2(r @ e2, r @ e1)
    return 0.5 + 0.5 * np.tanh(sharpness * np.sin(n_stripes * phi))


def highlight_lines(mesh: TessMesh, normals: np.ndarray, *, direction: Sequence[float] = (0.0, 1.0, 0.0),
                    plane_normal: Sequence[float] = (0.0, 0.0, 1.0), height: float | None = None,
                    spacing: float | None = None, width: float | None = None) -> np.ndarray:
    """Highlight-line intensity in [0, 1] per vertex (Beier & Chen 1994): parallel light lines along ``direction``
    lie in the plane ``plane_normal . x = height`` (default: above the part), ``spacing`` apart; a vertex is lit by
    exp(-(d/width)^2), d the distance between its surface-normal line and the nearest light line."""
    V = mesh.vertices
    lo, hi = V.min(0), V.max(0)
    size = float(np.linalg.norm(hi - lo))
    a = np.asarray(direction, float)
    a /= np.linalg.norm(a)
    pn = np.asarray(plane_normal, float)
    pn /= np.linalg.norm(pn)
    h = float(hi @ pn + 0.5 * size) if height is None else height
    s = size / 10.0 if spacing is None else spacing
    w = s / 8.0 if width is None else width
    e = np.cross(pn, a)                                     # in the plane, across the lines
    n = np.asarray(normals, float)
    # intersect each normal line x + t n with the light plane, then take the across-line coordinate
    denom = n @ pn
    t = (h - V @ pn) / np.where(np.abs(denom) < 1e-12, np.nan, denom)
    hit = V + t[:, None] * n
    u = hit @ e
    d = np.abs(((u / s + 0.5) % 1.0) - 0.5) * s
    val = np.exp(-(d / w) ** 2)
    return np.where(np.isfinite(val) & (t > 0), val, 0.0)


def view_direction(elev: float, azim: float) -> np.ndarray:
    """Unit vector from the scene towards the camera for elevation/azimuth in degrees (matplotlib's convention)."""
    e, a = np.radians(elev), np.radians(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def render(mesh: TessMesh, values: np.ndarray, path: str, *, cmap: str = "viridis", title: str = "",
           elev: float = 25.0, azim: float = -60.0, vmin: float | None = None, vmax: float | None = None,
           colorbar: str | None = None, shade: float = 0.0) -> str:
    """Draw a per-vertex value on the mesh to a PNG; returns the path.

    Orthographic view from (``elev``, ``azim``); triangles facing away are culled and the rest painted back to front,
    each with the mean of its vertex values. ``shade`` in [0, 1] mixes in Lambert shading from the view direction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    V, T = mesh.vertices, mesh.triangles
    w = view_direction(elev, azim)
    up = np.array([0.0, 0.0, 1.0]) - w[2] * w
    up = up / np.linalg.norm(up) if np.linalg.norm(up) > 1e-9 else np.array([0.0, 1.0, 0.0])
    right = np.cross(up, w)
    P = np.c_[V @ right, V @ up]
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    keep = fn @ w > 0
    Tk = T[keep]
    order = np.argsort((V[Tk] @ w).mean(1))
    Tk = Tk[order]
    tv = np.asarray(values, float)[Tk].mean(1)
    lo = np.nanmin(tv) if vmin is None else vmin
    hi = np.nanmax(tv) if vmax is None else vmax
    cm = plt.get_cmap(cmap)
    col = cm(np.clip((tv - lo) / max(hi - lo, 1e-300), 0, 1))
    if shade > 0:
        f = fn[keep][order]
        lam = (f @ w) / np.linalg.norm(f, axis=1)
        col[:, :3] *= (1 - shade) + shade * lam[:, None]
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=130)
    ax.add_collection(PolyCollection(P[Tk], facecolors=col, edgecolors=col, linewidths=0.2, antialiased=False))
    ax.set_xlim(P[:, 0].min(), P[:, 0].max())
    ax.set_ylim(P[:, 1].min(), P[:, 1].max())
    ax.set_aspect("equal")
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(lo, hi))
        fig.colorbar(sm, ax=ax, shrink=0.7, label=colorbar)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

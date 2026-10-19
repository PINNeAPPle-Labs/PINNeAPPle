"""Triangle meshes for 2D magnetostatics: a polar ring mesher with a moving band, and a gmsh importer.

The polar mesher builds a mesh from concentric rings of nodes. Two consecutive rings are joined by a
"zipper" triangulation that works for any node counts and any angular offsets, so the same routine
gives quad-like layers (equal counts), coarsening towards the centre (fewer nodes inside), a fan to a
centre node, and the sliding (moving) band of a rotating machine: the rotor rings carry an angular
offset ``theta``, the stator rings do not, and only the band layer between them is re-triangulated
when the rotor turns. Rotor and stator elements keep their indices and shapes for every angle.

Geometry is resolved on the polar grid (a "staircase" in angle), so boundaries of slots, teeth and
magnets should fall on ring radii and on multiples of the angular step of their rings.

    from pinneapple_simulation.electromagnetics.mesh import Ring, polar_mesh
    rings = [Ring(0.0, 1), Ring(0.01, 32), Ring(0.02, 64), Ring(0.03, 64)]
    mesh = polar_mesh(rings, classify=lambda layer, r, th: "air")
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Mesh2D", "Ring", "polar_mesh", "zipper", "mesh_from_gmsh", "rect_mesh"]

TWO_PI = 2.0 * math.pi
_TIE = 1e-10  # radians: nodes closer than this in angle count as aligned


@dataclass
class Mesh2D:
    """P1 triangle mesh with one region name per element.

    points: (N, 2) coordinates in metres; tris: (M, 3) counter-clockwise node indices;
    region: (M,) integer region index into ``region_names``; layer: (M,) index of the ring layer an
    element belongs to (polar meshes only, -1 otherwise); boundary: (K,) nodes on the outer boundary.
    """

    points: np.ndarray
    tris: np.ndarray
    region: np.ndarray
    region_names: list[str]
    layer: np.ndarray | None = None
    boundary: np.ndarray | None = None
    rings: list[np.ndarray] = field(default_factory=list)

    @property
    def n_nodes(self) -> int:
        return len(self.points)

    @property
    def n_elements(self) -> int:
        return len(self.tris)

    def mask(self, *names: str) -> np.ndarray:
        """Boolean element mask of the regions in ``names`` (unknown names select nothing)."""
        ids = [self.region_names.index(n) for n in names if n in self.region_names]
        return np.isin(self.region, ids)

    def mask_prefix(self, prefix: str) -> np.ndarray:
        ids = [i for i, n in enumerate(self.region_names) if n.startswith(prefix)]
        return np.isin(self.region, ids)

    def centroids(self) -> np.ndarray:
        return self.points[self.tris].mean(axis=1)

    def areas(self) -> np.ndarray:
        p = self.points[self.tris]
        return 0.5 * ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                      - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))


@dataclass
class Ring:
    """A circle of ``n`` equally spaced nodes at radius ``r`` (``n=1`` with ``r=0`` is the centre).

    ``offset`` is the angle of the first node; ``rotating`` marks rings that turn with the rotor.
    """

    r: float
    n: int
    offset: float = 0.0
    rotating: bool = False


def _orient(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    p = points[tris]
    det = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
           - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    tris = tris.copy()
    neg = det < 0
    tris[neg, 1], tris[neg, 2] = tris[neg, 2].copy(), tris[neg, 1].copy()
    return tris


def zipper(ia: np.ndarray, ang_a: np.ndarray, ib: np.ndarray, ang_b: np.ndarray,
           closed: bool = True) -> np.ndarray:
    """Triangulate the strip between two rings of nodes (indices ``ia``/``ib``, angles ``ang_a``/``ang_b``).

    Walks around both rings at once, starting at the first node of ring a, and always advances the ring
    whose next node has the smaller angle, which gives ``len(ia) + len(ib)`` triangles for a closed
    strip. For an open strip (a sector) the angles must be sorted ascending. Orientation is not fixed here.
    """
    na, nb = len(ia), len(ib)
    tris = []
    if not closed:
        i = j = 0
        while i < na - 1 or j < nb - 1:
            adv_a = j == nb - 1 or (i < na - 1 and (ang_a[i + 1] < ang_b[j + 1] - _TIE or (
                abs(ang_a[i + 1] - ang_b[j + 1]) <= _TIE and i % 2 == 0)))
            if adv_a:
                tris.append((ia[i], ia[i + 1], ib[j]))
                i += 1
            else:
                tris.append((ia[i], ib[j + 1], ib[j]))
                j += 1
        return np.asarray(tris, dtype=np.int64)

    # angles relative to node 0 of ring a: the walk starts there, so the order of the triangles is tied
    # to the rings' own numbering and does not change when both rings turn (rigid rotor layers)
    base = ang_a[0]
    a = np.mod(np.asarray(ang_a) - base, TWO_PI)
    b = np.mod(np.asarray(ang_b) - base, TWO_PI)
    a[np.abs(a - TWO_PI) <= _TIE] = 0.0
    b[np.abs(b - TWO_PI) <= _TIE] = 0.0
    oa, ob = np.argsort(a, kind="stable"), np.argsort(b, kind="stable")
    ia, a, ib, b = np.asarray(ia)[oa], a[oa], np.asarray(ib)[ob], b[ob]
    parity = oa % 2  # parity of each node's own index on its ring: the diagonal pattern moves with the ring

    def ua(k):  # unwrapped angle of ring-a step k
        return a[k % na] + TWO_PI * (k // na)

    def ub(k):
        return b[k % nb] + TWO_PI * (k // nb)

    # j0: last b node at or before a[0] (cyclically, may be negative)
    j0 = int(np.searchsorted(b, a[0] + _TIE, side="right")) - 1
    i, j = 0, j0
    for _ in range(na + nb):
        if i < na and j < j0 + nb:
            xa, xb = ua(i + 1), ub(j + 1)
            # alternate the diagonal of aligned quads so the mesh has no preferred direction
            adv_a = xa < xb - _TIE or (abs(xa - xb) <= _TIE and parity[i % na] == 0)
        else:
            adv_a = i < na
        if adv_a:
            tris.append((ia[i % na], ia[(i + 1) % na], ib[j % nb]))
            i += 1
        else:
            tris.append((ia[i % na], ib[(j + 1) % nb], ib[j % nb]))
            j += 1
    return np.asarray(tris, dtype=np.int64)


def polar_mesh(rings: Sequence[Ring], classify: Callable[[int, float, float], str],
               theta: float = 0.0, span: float = TWO_PI,
               region_names: Sequence[str] | None = None) -> Mesh2D:
    """Mesh the disc (or sector ``[0, span]``) between concentric rings.

    ``classify(layer, r, phi)`` names the region of an element from the index of its ring layer and
    its centroid radius and angle; for elements between two rotating rings ``phi`` is measured in the
    rotor frame (centroid angle minus ``theta``), so the rotor geometry is the same for every angle.
    ``theta`` turns every ring with ``rotating=True``. For a sector (``span < 2*pi``) each ring holds
    ``n + 1`` nodes from 0 to ``span`` (rotation is not supported for sectors).
    """
    closed = span >= TWO_PI - 1e-12
    if not closed and theta != 0.0:
        raise ValueError("rotating a sector mesh is not supported; mesh the full circle")
    pts, ring_nodes, ring_ang = [], [], []
    count = 0
    for rg in rings:
        if rg.r == 0.0:
            ang = np.zeros(1)
        elif closed:
            ang = rg.offset + (theta if rg.rotating else 0.0) + TWO_PI * np.arange(rg.n) / rg.n
        else:
            ang = np.linspace(0.0, span, rg.n + 1)
        xy = np.column_stack([rg.r * np.cos(ang), rg.r * np.sin(ang)])
        pts.append(xy)
        ring_nodes.append(np.arange(count, count + len(ang)))
        ring_ang.append(ang)
        count += len(ang)
    points = np.vstack(pts)

    tris_all, layer_all = [], []
    for k in range(len(rings) - 1):
        ra = rings[k]
        if ra.r == 0.0:
            ib = ring_nodes[k + 1]
            nb = len(ib)
            if closed:
                t = np.column_stack([np.full(nb, ring_nodes[k][0]), ib, np.roll(ib, -1)])
            else:
                t = np.column_stack([np.full(nb - 1, ring_nodes[k][0]), ib[:-1], ib[1:]])
        else:
            t = zipper(ring_nodes[k], ring_ang[k], ring_nodes[k + 1], ring_ang[k + 1], closed=closed)
        tris_all.append(t)
        layer_all.append(np.full(len(t), k))
    tris = _orient(points, np.vstack(tris_all))
    layer = np.concatenate(layer_all)

    c = points[tris].mean(axis=1)
    rc = np.hypot(c[:, 0], c[:, 1])
    phi = np.arctan2(c[:, 1], c[:, 0])
    rot_layer = np.array([rings[k].rotating and rings[k + 1].rotating for k in range(len(rings) - 1)])
    phi = np.where(rot_layer[layer], phi - theta, phi)
    phi = np.mod(phi, TWO_PI)
    names = [classify(int(ly), float(r), float(p)) for ly, r, p in zip(layer, rc, phi, strict=True)]
    rn = list(region_names) if region_names is not None else []
    for n in names:
        if n not in rn:
            rn.append(n)
    lookup = {n: i for i, n in enumerate(rn)}
    region = np.array([lookup[n] for n in names], dtype=np.int64)
    return Mesh2D(points, tris, region, rn, layer=layer, boundary=ring_nodes[-1].copy(),
                  rings=ring_nodes)


def rect_mesh(x, y, classify: Callable[[float, float], str] | None = None) -> Mesh2D:
    """Structured triangle mesh of the rectangle spanned by the node coordinates ``x`` and ``y`` (1-D, increasing;
    any spacing), two triangles per cell with diagonals alternating from cell to cell, so the mesh has no preferred
    direction. ``classify(xc, yc)`` names the region of each element from its centroid. Node
    ``i * len(y) + j`` sits at ``(x[i], y[j])``; ``boundary`` holds the nodes on the four sides."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    X, Y = np.meshgrid(x, y, indexing="ij")
    points = np.column_stack([X.ravel(), Y.ravel()])
    ci, cj = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    ci, cj = ci.ravel(), cj.ravel()
    n00, n10, n01, n11 = ci * ny + cj, (ci + 1) * ny + cj, ci * ny + cj + 1, (ci + 1) * ny + cj + 1
    flip = (ci + cj) % 2 == 1
    t1 = np.where(flip[:, None], np.column_stack([n00, n10, n01]), np.column_stack([n00, n10, n11]))
    t2 = np.where(flip[:, None], np.column_stack([n10, n11, n01]), np.column_stack([n00, n11, n01]))
    tris = _orient(points, np.vstack([t1, t2]))
    c = points[tris].mean(axis=1)
    names = ["domain"] * len(tris) if classify is None else [classify(float(a), float(b)) for a, b in c]
    rn: list[str] = []
    for n in names:
        if n not in rn:
            rn.append(n)
    lookup = {n: i for i, n in enumerate(rn)}
    region = np.array([lookup[n] for n in names], dtype=np.int64)
    on_b = (X == x[0]) | (X == x[-1]) | (Y == y[0]) | (Y == y[-1])
    return Mesh2D(points, tris, region, rn, boundary=np.flatnonzero(on_b.ravel()))


def mesh_from_gmsh(dim: int = 2, model=None) -> Mesh2D:
    """Convert the current gmsh model (already meshed) into a :class:`Mesh2D`.

    Every 2D physical group becomes a region (its name, or ``"phys<tag>"``); triangles outside any
    physical group are dropped. Nodes of 1D physical groups named ``"boundary"`` (or, when there is
    none, of the whole mesh boundary) are returned as ``boundary``. ``gmsh`` is an optional dependency.
    """
    import gmsh  # optional

    m = model or gmsh.model
    tags, coords, _ = m.mesh.getNodes()
    order = np.argsort(tags)
    tags = tags[order]
    xyz = coords.reshape(-1, 3)[order]
    index = {int(t): i for i, t in enumerate(tags)}
    tris, region, names = [], [], []
    for d, ptag in m.getPhysicalGroups(dim):
        name = m.getPhysicalName(d, ptag) or f"phys{ptag}"
        names.append(name)
        for ent in m.getEntitiesForPhysicalGroup(d, ptag):
            etypes, _, enodes = m.mesh.getElements(d, ent)
            for et, en in zip(etypes, enodes, strict=True):
                if et != 2:  # 3-node triangle
                    continue
                t = np.array([index[int(v)] for v in en], dtype=np.int64).reshape(-1, 3)
                tris.append(t)
                region.append(np.full(len(t), len(names) - 1))
    tris = np.vstack(tris)
    used = np.unique(tris)
    remap = -np.ones(len(xyz), dtype=np.int64)
    remap[used] = np.arange(len(used))
    points = xyz[used, :2]
    tris = remap[tris]
    tris = _orient(points, tris)
    region = np.concatenate(region)

    bnodes = []
    for d, ptag in m.getPhysicalGroups(1):
        if m.getPhysicalName(d, ptag) == "boundary":
            nt, _ = m.mesh.getNodesForPhysicalGroup(d, ptag)
            bnodes.extend(int(v) for v in nt)
    if bnodes:
        boundary = np.unique(remap[[index[v] for v in bnodes]])
    else:
        boundary = boundary_nodes(tris)
    return Mesh2D(points, tris, region, names, layer=None, boundary=boundary)


def boundary_nodes(tris: np.ndarray) -> np.ndarray:
    """Nodes on edges that belong to exactly one triangle."""
    e = np.sort(np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    u, cnt = np.unique(e, axis=0, return_counts=True)
    return np.unique(u[cnt == 1])

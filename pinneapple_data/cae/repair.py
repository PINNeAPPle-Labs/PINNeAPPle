"""Mesh repair: turn the suggestion in ``report.ACTIONS`` into an operation, and measure the mesh before and after.

Every operation takes a :class:`~pinneapple_data.cae.model.Mesh` and returns a **new** mesh plus a
:class:`RepairResult` carrying the same metrics as :func:`~pinneapple_data.cae.quality.surface_checks` for the input
(``before``) and the output (``after``), how many faces or vertices it changed, what it guarantees and what it cannot
fix. The input mesh is never modified.

The order matters, and :func:`repair_surface` runs it:

1. :func:`merge_vertices` -- coincident vertices are welded first, because unmelted nodes are what make a surface look
   open and disconnected in the first place.
2. :func:`drop_degenerate_faces` -- a zero-area face has no normal, so it cannot be oriented or closed against.
3. :func:`drop_duplicate_faces` -- the same face twice inflates the edge count and makes an edge look non-manifold.
4. :func:`orient_consistently` -- neighbours are walked once the topology is clean, so the walk does not stop at a
   face that is about to be removed.
5. :func:`close_holes` -- holes are closed last, and each patch inherits the winding the walk just settled.

Only triangle, quad and polygon shells are repaired here. Finite-volume (polyhedral) and volume meshes are refused,
with the reason in the result: a distorted cell is a meshing decision, not a geometric edit (see #1494).
"""
from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .model import CellBlock, Mesh
from .quality import surface_checks

#: The 2-D element types the operations edit; anything else is carried through untouched.
SURFACE_TYPES = ("triangle", "quad", "polygon")

#: Node permutation that reverses an element's winding (VTK ordering: corners first, then the higher-order nodes).
_REVERSE = {
    "triangle": (0, 2, 1),
    "quad": (0, 3, 2, 1),
    "triangle6": (0, 2, 1, 5, 4, 3),
    "quad8": (0, 3, 2, 1, 7, 6, 5, 4),
    "quad9": (0, 3, 2, 1, 7, 6, 5, 4, 8),
}

#: Direction of each surface metric -- only the directional ones take part in the net-loss check.
_BETTER = {"open_edges": "low", "nonmanifold_edges": "low", "inconsistent_orientation_edges": "low",
           "degenerate_faces": "low", "duplicate_faces": "low", "watertight": "high", "faces": "none",
           "shells": "none", "enclosed_volume": "none", "area": "none"}

_SCALAR_METRICS = ("faces", "open_edges", "nonmanifold_edges", "inconsistent_orientation_edges", "shells",
                   "degenerate_faces", "duplicate_faces", "watertight", "enclosed_volume", "area")

_NUMBER = (int, float, np.integer, np.floating)


@dataclass
class RepairResult:
    """What one operation did: the measured before and after, and an honest account of its limits."""
    operation: str
    kind: str = "surface"
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)
    changed: int = 0                                        # faces and vertices the operation altered
    guarantees: list[str] = field(default_factory=list)
    cannot_fix: list[str] = field(default_factory=list)
    regressed: list[str] = field(default_factory=list)      # checks that got worse: a net loss, not a repair
    refused: bool = False
    reason: str = ""                                        # why it refused, when refused is True

    @property
    def net_loss(self) -> bool:
        """True when the operation fixed one check by breaking another."""
        return bool(self.regressed)

    def as_dict(self) -> dict[str, Any]:
        return {"operation": self.operation, "kind": self.kind, "before": self.before, "after": self.after,
                "changed": self.changed, "guarantees": list(self.guarantees),
                "cannot_fix": list(self.cannot_fix), "regressed": list(self.regressed), "refused": self.refused,
                "reason": self.reason, "net_loss": self.net_loss}


class RepairPlan(list):
    """The steps of :func:`repair_surface` in the order they ran, with the consolidated before/after in ``summary``."""

    def __init__(self, results: Sequence[RepairResult], summary: dict[str, Any]):
        super().__init__(results)
        self.summary = summary

    @property
    def no_op(self) -> bool:
        """True when every step left the mesh as it was: a clean mesh comes back unchanged and says so."""
        return bool(self.summary.get("no_op"))


def surface_summary(mesh: Mesh) -> dict[str, Any]:
    """The scalar metrics of :func:`~pinneapple_data.cae.quality.surface_checks` -- what repair measures against."""
    sc = surface_checks(mesh)
    return {k: sc[k] for k in _SCALAR_METRICS if k in sc}


# ------------------------------------------------------------------------------------------ plumbing
def _surface_indices(mesh: Mesh) -> list[int]:
    return [k for k, b in enumerate(mesh.blocks) if b.dim == 2 and b.type in SURFACE_TYPES]


def _refusal(mesh: Mesh) -> str:
    """Why this mesh cannot be repaired here, or an empty string when it can."""
    if mesh.poly is not None:
        return ("polyhedral (finite-volume) meshes are not repaired: a distorted or inverted cell is a meshing "
                "decision, not a geometric edit; re-run the mesher with better parameters")
    if mesh.dim == 3:
        return ("a volume mesh is not repaired: its boundary surface is an output of the volume, so editing the "
                "surface alone leaves the volume inconsistent; repair the surface before meshing it")
    if not _surface_indices(mesh):
        kinds = ", ".join(sorted(mesh.element_counts())) or "none"
        return f"no triangle, quad or polygon blocks found (the mesh has {kinds}): surface shells only"
    return ""


def _diagonal(mesh: Mesh) -> float:
    if not len(mesh.points):
        return 0.0
    return float(np.linalg.norm(mesh.points.max(0) - mesh.points.min(0)))


def _reverse_conn(block: CellBlock) -> np.ndarray:
    """The block's connectivity with every element's winding reversed, VTK ordering preserved."""
    order = _REVERSE.get(block.type)
    if order is None:                                   # polygon: reversing the cycle reverses every edge
        return block.conn[:, ::-1].copy()
    return block.conn[:, list(order)].copy()


def _copy_block(block: CellBlock) -> CellBlock:
    return CellBlock(type=block.type, conn=np.array(block.conn, copy=True),
                     ids=None if block.ids is None else np.asarray(block.ids).copy())


def _without_mesh(operation: str, mesh: Mesh, reason: str) -> tuple[Mesh, RepairResult]:
    """The safe answer to an unsupported mesh: an unchanged copy, and the reason named."""
    summary = surface_summary(mesh)
    unchanged = _rebuild(mesh, [_copy_block(b) for b in mesh.blocks], mesh.points.copy())
    return (unchanged,
            RepairResult(operation=operation, before=summary, after=dict(summary), refused=True, reason=reason,
                         cannot_fix=[reason]))


def _rebuild(mesh: Mesh, blocks: list[CellBlock], points: np.ndarray, point_map: np.ndarray | None = None,
             kept_cells: Sequence[int] | None = None) -> Mesh:
    """A new mesh from finished blocks, with the mesh's fields and sets carried along (or dropped, never guessed).

    ``blocks`` index into ``points``. Points no block references are dropped. ``point_map`` maps an old point index to
    its index in ``points``, so the point fields and node sets follow. ``kept_cells`` lists the old global cell
    indices that survive, in the new order; ``None`` means every cell is kept and the order is unchanged.
    """
    used = np.zeros(len(points), bool)
    for b in blocks:
        used[b.conn.ravel()] = True
    if used.all():
        packed, pack = points, None
    else:
        packed = points[used]
        pack = np.cumsum(used) - 1
    final: list[CellBlock] = []
    for b in blocks:
        conn = b.conn if pack is None else pack[b.conn]
        final.append(CellBlock(type=b.type, conn=conn, ids=None if b.ids is None else np.asarray(b.ids).copy()))
    if point_map is not None and pack is not None:
        point_map = pack[point_map]
    new = Mesh(points=packed, blocks=final, boundary_blocks=[(n, _copy_block(b)) for n, b in mesh.boundary_blocks],
               units=dict(mesh.units), source=dict(mesh.source), poly=mesh.poly)
    for k, v in mesh.point_data.items():
        if isinstance(v, np.ndarray) and v.shape[0] == mesh.n_points:
            new.point_data[k] = v.copy() if point_map is None else v[point_map]
    for k, v in mesh.node_sets.items():
        idx = np.asarray(v, int)
        new.node_sets[k] = np.array(idx, copy=True) if point_map is None else np.unique(point_map[idx])
    if kept_cells is None:
        for k, v in mesh.cell_data.items():
            if isinstance(v, np.ndarray) and v.shape[0] == mesh.n_cells:
                new.cell_data[k] = np.array(v, copy=True)
        new.cell_sets = {k: np.array(v, copy=True) for k, v in mesh.cell_sets.items()}
    else:
        keep = np.asarray(kept_cells, int)
        remap = {int(old): new_i for new_i, old in enumerate(keep)}
        for k, v in mesh.cell_data.items():
            if isinstance(v, np.ndarray) and v.shape[0] == mesh.n_cells:
                new.cell_data[k] = v[keep]
        new.cell_sets = {k: np.array([remap[int(i)] for i in np.asarray(v, int) if int(i) in remap], int)
                         for k, v in mesh.cell_sets.items()}
    return new


def _filtered(mesh: Mesh, rows_of: dict[int, np.ndarray]) -> tuple[Mesh, int]:
    """A new mesh with only the listed rows of the listed blocks, plus how many faces went away."""
    blocks: list[CellBlock] = []
    kept: list[int] = []
    dropped = 0
    start = 0
    for k, b in enumerate(mesh.blocks):
        if k in rows_of:
            rows = np.asarray(rows_of[k], int)
            dropped += len(b.conn) - len(rows)
            blocks.append(CellBlock(type=b.type, conn=b.conn[rows],
                                    ids=None if b.ids is None else np.asarray(b.ids)[rows]))
            kept.extend((start + rows).tolist())
        else:
            blocks.append(_copy_block(b))
            kept.extend(range(start, start + len(b.conn)))
        start += len(b.conn)
    return _rebuild(mesh, blocks, mesh.points.copy(), kept_cells=kept), dropped


def _regressions(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """The checks this operation made worse: the repair fixed one thing and broke another."""
    worse: list[str] = []
    for k, direction in _BETTER.items():
        if direction == "none" or k not in before or k not in after:
            continue
        b, a = before[k], after[k]
        if direction == "low":
            if isinstance(a, _NUMBER) and isinstance(b, _NUMBER) and a > b:
                worse.append(k)
        elif bool(b) and not bool(a):
            worse.append(k)
    b, a = before.get("area"), after.get("area")
    kept_every_face = before.get("faces", 0) <= after.get("faces", 0)
    if kept_every_face and isinstance(b, _NUMBER) and isinstance(a, _NUMBER) and b > 0 and a < 0.95 * b:
        worse.append("area")                     # lost more than 5% of the surface without removing a face
    return worse


def _face_edges(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every face's edges as (face, a, b), walking (i, i+1) round the face -- the walk ``surface_checks`` uses."""
    k = corners.shape[1]
    faces = np.repeat(np.arange(len(corners)), k)
    a = corners.reshape(-1)
    b = corners[:, [(i + 1) % k for i in range(k)]].reshape(-1)
    return faces, a, b


def _edge_key(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """An order-independent key per edge, so ``np.unique`` groups both traversal directions of one edge."""
    _, inv = np.unique(np.stack([np.minimum(a, b), np.maximum(a, b)], 1), axis=0, return_inverse=True)
    return inv.ravel()


def _global_faces(mesh: Mesh, indices: Sequence[int]) -> tuple[np.ndarray, dict[int, int]]:
    """The corners of every surface face in block order, and where each block starts in that list."""
    faces, starts, start = [], {}, 0
    for k in indices:
        c = np.asarray(mesh.blocks[k].corners, int)
        faces.append(c)
        starts[k] = start
        start += len(c)
    return (np.concatenate(faces) if faces else np.zeros((0, 3), int)), starts


# ------------------------------------------------------------------------------------------ 1. merge vertices
def merge_vertices(mesh: Mesh, tol_relative: float = 1e-6) -> tuple[Mesh, RepairResult]:
    """Weld vertices closer than ``tol_relative * bbox diagonal`` and remap every block's connectivity.

    A cluster of coincident vertices becomes one point at the cluster's mean position (so no vertex snaps onto a
    neighbour's exact spot). Faces whose corners collapse onto each other are removed, and points that no block
    references any more are dropped.
    """
    op = "merge_vertices"
    before = surface_summary(mesh)
    reason = _refusal(mesh)
    if reason:
        return _without_mesh(op, mesh, reason)
    tol = float(tol_relative) * max(_diagonal(mesh), 1e-12)
    pts = mesh.points
    n0 = len(pts)
    root = np.arange(n0)
    pairs = cKDTree(pts).query_pairs(tol, output_type="ndarray") if n0 > 1 else np.zeros((0, 2), int)
    for i, j in pairs:
        ri, rj = _find(root, int(i)), _find(root, int(j))
        if ri != rj:
            root[max(ri, rj)] = min(ri, rj)
    for i in range(n0):
        root[i] = _find(root, i)
    _, point_map = np.unique(root, return_inverse=True)
    point_map = point_map.ravel()
    n1 = int(point_map.max()) + 1 if len(point_map) else 0
    points = np.zeros((n1, 3))
    np.add.at(points, point_map, pts)
    points /= np.maximum(np.bincount(point_map, minlength=n1), 1)[:, None]
    blocks: list[CellBlock] = []
    rows_of: dict[int, np.ndarray] = {}
    folded = 0
    for k, b in enumerate(mesh.blocks):
        conn = point_map[b.conn]
        blocks.append(CellBlock(type=b.type, conn=conn, ids=None if b.ids is None else np.asarray(b.ids).copy()))
        if b.dim == 2 and b.type in SURFACE_TYPES:
            kk = b.corners.shape[1]
            good = np.array([len(np.unique(r)) == kk for r in conn[:, :kk]], bool)
            folded += int((~good).sum())
            rows_of[k] = np.nonzero(good)[0]
    new, _ = _filtered(_rebuild(mesh, blocks, points, point_map=point_map), rows_of)
    after = surface_summary(new)
    merged = int(n0 - new.n_points)
    res = RepairResult(
        operation=op, before=before, after=after, changed=merged + folded,
        guarantees=[f"points within {tol:.3g} of each other are one point, and every block is remapped to match",
                    f"{merged} point(s) merged away; {folded} face(s) collapsed onto a line or a point and removed",
                    "no vertex is snapped onto a neighbour's exact position: a cluster keeps its mean"],
        cannot_fix=["welding is by distance alone, so two parts that touch (a contact pair, a lid on a box) become one: "
                    "coincidence is not the same as connection",
                    "a cluster moves by up to half the tolerance, so a fine feature thinner than the tolerance is "
                    "flattened",
                    "a hole wider than the tolerance stays open, and parts that were already fused are not separated",
                    "the merged topology is not re-validated: welding two sheets that only cross (and do not share "
                    "corners) still leaves them crossing"])
    res.regressed = _regressions(before, after)
    return new, res


def _find(root: np.ndarray, i: int) -> int:
    while root[i] != i:
        root[i] = root[root[i]]
        i = int(root[i])
    return int(i)


# ------------------------------------------------------------------------------------------ 2. degenerate faces
def drop_degenerate_faces(mesh: Mesh, rel_area: float = 1e-12) -> tuple[Mesh, RepairResult]:
    """Remove faces whose first three corners span less than ``rel_area * diagonal**2`` of area.

    The threshold is the one :func:`~pinneapple_data.cae.quality.surface_checks` calls degenerate, so the before and
    after counts are the same check.
    """
    op = "drop_degenerate_faces"
    before = surface_summary(mesh)
    reason = _refusal(mesh)
    if reason:
        return _without_mesh(op, mesh, reason)
    limit = float(rel_area) * max(_diagonal(mesh), 1e-12) ** 2
    rows_of: dict[int, np.ndarray] = {}
    for k in _surface_indices(mesh):
        c = mesh.points[mesh.blocks[k].corners[:, :3]]
        area = 0.5 * np.linalg.norm(np.cross(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0]), axis=1)
        rows_of[k] = np.nonzero(area >= limit)[0]
    new, dropped = _filtered(mesh, rows_of)
    after = surface_summary(new)
    res = RepairResult(
        operation=op, before=before, after=after, changed=dropped,
        guarantees=[f"no face is left below an area of {limit:.3g}, so every remaining face has a normal to orient by",
                    "the faces that stay are byte for byte what they were: no vertex moves and no winding changes"],
        cannot_fix=["a sliver that is thin but not zero-area stays: it has a normal, so it is a shape problem, not a "
                    "topology one",
                    "removing a face opens a hole where the sliver was; close_holes can patch it, but the patch is a "
                    "convention, not the surface the sliver stood for",
                    "nothing is retriangulated: the gap left behind is not filled with better-shaped faces"])
    res.regressed = _regressions(before, after)
    return new, res


# ------------------------------------------------------------------------------------------ 3. duplicate faces
def drop_duplicate_faces(mesh: Mesh) -> tuple[Mesh, RepairResult]:
    """Remove faces that repeat a set of vertices an earlier face already used, keeping the first one.

    Like :func:`~pinneapple_data.cae.quality.surface_checks`, faces are compared on their first three corners, so a
    quad sharing three corners with a triangle counts as the same face.
    """
    op = "drop_duplicate_faces"
    before = surface_summary(mesh)
    reason = _refusal(mesh)
    if reason:
        return _without_mesh(op, mesh, reason)
    seen: set = set()
    rows_of: dict[int, list[int]] = {k: [] for k in _surface_indices(mesh)}
    for k in _surface_indices(mesh):
        for row, f in enumerate(mesh.blocks[k].corners):
            key = tuple(sorted(int(x) for x in f[:3]))
            if key in seen:
                continue
            seen.add(key)
            rows_of[k].append(row)
    new, dropped = _filtered(mesh, {k: np.asarray(v, int) for k, v in rows_of.items()})
    after = surface_summary(new)
    res = RepairResult(
        operation=op, before=before, after=after, changed=dropped,
        guarantees=["each set of three corners appears in at most one face afterwards",
                    "the face kept is the first one in input order, so its winding is the one the report judged"],
        cannot_fix=["two faces over the same three corners with opposite normals are a thin wall, not a duplicate: one "
                    "of them goes and the wall disappears",
                    "keeping the first face is a convention, not a measurement: the duplicate may have been the "
                    "correct one",
                    "quads are compared on three corners, so two quads with the same first three corners and a "
                    "different fourth are treated as the same face"])
    res.regressed = _regressions(before, after)
    return new, res


# ------------------------------------------------------------------------------------------ 4. orientation
def orient_consistently(mesh: Mesh) -> tuple[Mesh, RepairResult]:
    """Make faces that share an edge traverse it in opposite directions, one surface shell at a time.

    The walk starts at each shell's lowest-numbered face and flips only what disagrees with it. A closed shell whose
    enclosed volume then comes out negative is turned inside out as a whole, so a closed surface ends with its normals
    pointing outwards. Separate shells are independent: each gets its own seed and its own volume test.
    """
    op = "orient_consistently"
    before = surface_summary(mesh)
    reason = _refusal(mesh)
    if reason:
        return _without_mesh(op, mesh, reason)
    indices = _surface_indices(mesh)
    corners, starts = _global_faces(mesh, indices)
    faces, a, b = _face_edges(corners)
    key = _edge_key(a, b)
    order = np.argsort(key, kind="stable")
    direction = (a < b).astype(np.int8)
    groups = np.split(order, np.nonzero(np.diff(key[order]))[0] + 1)
    adjacent: dict[int, list[tuple[int, bool]]] = defaultdict(list)
    boundary_edges = nonmanifold = 0
    for g in groups:
        if len(g) == 1:
            boundary_edges += 1
        elif len(g) == 2:
            f1, f2 = int(faces[g[0]]), int(faces[g[1]])
            same = bool(direction[g[0]]) == bool(direction[g[1]])
            adjacent[f1].append((f2, same))
            adjacent[f2].append((f1, same))
        else:
            nonmanifold += 1
    flip = np.zeros(len(corners), bool)
    seen = np.zeros(len(corners), bool)
    shells: list[list[int]] = []
    for seed in range(len(corners)):
        if seen[seed]:
            continue
        seen[seed] = True
        shell, queue = [seed], deque([seed])
        while queue:
            f = queue.popleft()
            for g, same in adjacent.get(f, ()):
                if seen[g]:
                    continue
                seen[g] = True
                flip[g] = flip[f] ^ same
                shell.append(g)
                queue.append(g)
        shells.append(sorted(shell))
    k_of_face = np.array([len(c) for c in corners])
    closed_shells = 0
    if not nonmanifold:
        for shell in shells:
            if not all(len(adjacent.get(f, ())) == k_of_face[f] for f in shell):
                continue                       # a boundary edge in the shell: it has no inside to point away from
            closed_shells += 1
            if _shell_volume(mesh.points, corners, flip, np.asarray(shell, int)) < 0:
                flip[np.asarray(shell, int)] ^= True
    blocks: list[CellBlock] = []
    changed = 0
    for k, blk in enumerate(mesh.blocks):
        if k not in starts:
            blocks.append(_copy_block(blk))
            continue
        f0 = starts[k]
        f = flip[f0:f0 + len(blk.conn)]
        conn = np.array(blk.conn, copy=True)
        conn[f] = _reverse_conn(blk)[f]
        changed += int(f.sum())
        blocks.append(CellBlock(type=blk.type, conn=conn,
                                ids=None if blk.ids is None else np.asarray(blk.ids).copy()))
    new = _rebuild(mesh, blocks, mesh.points.copy())
    after = surface_summary(new)
    res = RepairResult(
        operation=op, before=before, after=after, changed=changed,
        guarantees=[f"{changed} face(s) re-wound across {len(shells)} shell(s), so neighbours agree on which side is "
                    "outside",
                    f"{closed_shells} closed shell(s) had their enclosed volume measured and were turned inside out "
                    "when it came out negative",
                    "each shell was solved on its own: flipping one does not move another",
                    "no vertex is moved, so the shape is exactly what it was"],
        cannot_fix=["the walk picks a seed, so on an open shell the seed's winding is kept and the rest follows it: a "
                    "shell with no enclosed volume has no inside to point away from",
                    "a surface with a twist (a Moebius band) cannot be oriented consistently at all, and the edges the "
                    "walk could not resolve are left inconsistent",
                    "edges shared by more than three faces carry no orientation (the neighbour is ambiguous); with any "
                    "present, closed shells are not volume-tested",
                    "orientation says which way a surface faces; it does not fix an inverted element in a volume mesh, "
                    "and it does not change the geometry",
                    f"{boundary_edges} boundary edge(s) and {nonmanifold} non-manifold edge(s) gave no direction to "
                    "follow"])
    res.regressed = _regressions(before, after)
    return new, res


def _shell_volume(points: np.ndarray, corners: np.ndarray, flip: np.ndarray, shell: np.ndarray) -> float:
    """Signed volume the shell encloses with ``flip`` applied (a fan from each face's first corner)."""
    total = 0.0
    for f in shell:
        c = corners[f][::-1] if flip[f] else corners[f]
        p = points[c]
        tri = np.cross(p[1:-1] - p[0], p[2:] - p[0])
        total += float(np.einsum("ij,j->i", tri, p[0]).sum() / 6.0)
    return total


# ------------------------------------------------------------------------------------------ 5. close holes
def close_holes(mesh: Mesh, max_edges: int = 100) -> tuple[Mesh, RepairResult]:
    """Close every boundary loop of up to ``max_edges`` edges with a fan of triangles.

    A loop of up to eight edges is fanned from its highest-degree vertex, which adds no point. A longer loop gets a new
    vertex at its centroid and is fanned from there, because a fan from the corner of a long loop makes long thin
    triangles. Each fan walks the boundary edge the way the existing face does not, so the patch is consistent with the
    surface. Loops longer than ``max_edges``, and boundary chains that do not close, are left alone and counted.
    """
    op = "close_holes"
    before = surface_summary(mesh)
    reason = _refusal(mesh)
    if reason:
        return _without_mesh(op, mesh, reason)
    indices = _surface_indices(mesh)
    corners, _ = _global_faces(mesh, indices)
    faces, a, b = _face_edges(corners)
    key = _edge_key(a, b)
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    walked: dict[int, tuple[int, int]] = {}
    for e, x, y in zip(inv, a, b, strict=False):
        walked.setdefault(int(e), (int(x), int(y)))
    boundary = {e: d for e, d in walked.items() if cnt[e] == 1}
    loops, stuck = _trace_loops(boundary)
    triangles: list[tuple[int, int, int]] = []
    new_points: list[np.ndarray] = []
    closed = left = 0
    for loop in loops:
        if len(loop) < 3 or len(loop) > max_edges:
            left += 1
            continue
        n = len(loop)
        if n <= 8:
            v = _highest_degree(loop, boundary)
            m = loop.index(v)
            triangles.extend((v, loop[i], loop[(i + 1) % n])
                             for i in range(n) if i != m and (i + 1) % n != m)
        else:
            c = mesh.n_points + len(new_points)
            new_points.append(mesh.points[np.asarray(loop, int)].mean(0))
            triangles.extend((c, loop[i], loop[(i + 1) % n]) for i in range(n))
        closed += 1
    points = np.vstack([mesh.points, np.asarray(new_points)]) if new_points else mesh.points.copy()
    blocks = [_copy_block(b) for b in mesh.blocks]
    if triangles:
        blocks.append(CellBlock(type="triangle", conn=np.asarray(triangles, int)))
    new = _rebuild(mesh, blocks, points)
    after = surface_summary(new)
    res = RepairResult(
        operation=op, before=before, after=after, changed=len(triangles),
        guarantees=[f"{closed} boundary loop(s) closed with {len(triangles)} triangle(s), {len(new_points)} point(s) "
                    "added for the long loops",
                    "each added triangle walks its boundary edge the way the existing face does not, so the patch is "
                    "wound consistently with the surface",
                    "loops longer than max_edges, and boundary chains that do not meet their start, are left open and "
                    "counted, never closed silently"],
        cannot_fix=["a hole has no unique filling: the fan is a flat cone, not the surface the part really had, so a "
                    "curved or feature-carrying hole is approximated and its edges stay sharp",
                    "a boundary that branches, or a chain that does not return to its start, is left open",
                    "the patch is not tested against the rest of the mesh for self-intersection or for the volume it "
                    "encloses",
                    "closing a hole makes the surface watertight but changes the enclosed volume by the patches' own "
                    "contribution: the shape is a decision, not a measurement",
                    f"{left} loop(s) were left open, and {stuck} boundary chain(s) did not close"])
    res.regressed = _regressions(before, after)
    return new, res


def _trace_loops(boundary: dict[int, tuple[int, int]]) -> tuple[list[list[int]], int]:
    """Trace the boundary edges into loops, as ``(loops, chains that did not close)``.

    Each boundary edge ``(p, q)`` is the walk of the face that owns it, so the patch has to walk ``(q, p)``; the loops
    come back in that order, which is what makes the fan wind consistently with the surface.
    """
    unused = {e: (q, p) for e, (p, q) in boundary.items()}
    out: dict[int, list[int]] = defaultdict(list)
    for e, (p, _q) in unused.items():
        out[p].append(e)
    loops, stuck = [], 0
    for e0 in list(unused):
        if e0 not in unused:
            continue
        p, q = unused.pop(e0)
        loop = [p, q]
        while True:
            nxt = [e for e in out.get(q, ()) if e in unused]
            if not nxt:
                stuck += 1
                break
            e = nxt[0]
            _, q2 = unused.pop(e)
            if q2 == loop[0]:
                loops.append(loop)
                break
            if q2 == q:
                stuck += 1
                break
            loop.append(q2)
            q = q2
    return loops, stuck


def _highest_degree(loop: list[int], boundary: dict[int, tuple[int, int]]) -> int:
    """The loop vertex that the most boundary edges touch; ties go to the lowest index."""
    degree: dict[int, int] = defaultdict(int)
    for p, q in boundary.values():
        if p in loop:
            degree[p] += 1
        if q in loop:
            degree[q] += 1
    return max(loop, key=lambda v: (degree.get(v, 0), -v))


# ------------------------------------------------------------------------------------------ the pipeline
def repair_surface(mesh: Mesh, rel_area: float = 1e-12, tol_relative: float = 1e-6,
                   max_edges: int = 100) -> tuple[Mesh, RepairPlan]:
    """Run the surface pipeline in order, and report every step plus the consolidated before and after.

    Returns the repaired mesh and a :class:`RepairPlan`: the list of :class:`RepairResult` in the order they ran, with
    ``plan.summary`` holding ``before``, ``after``, ``fixed``, ``regressed``, ``steps``, ``changed``, ``no_op`` and
    ``refused``. A clean mesh comes back unchanged with ``plan.no_op`` True.
    """
    before = surface_summary(mesh)
    current = mesh
    steps: list[RepairResult] = []
    pipeline = (lambda m: merge_vertices(m, tol_relative),
                lambda m: drop_degenerate_faces(m, rel_area),
                drop_duplicate_faces,
                orient_consistently,
                lambda m: close_holes(m, max_edges))
    for fn in pipeline:
        current, res = fn(current)
        steps.append(res)
    after = surface_summary(current)
    summary = {"before": before, "after": after,
               "fixed": [k for k, d in _BETTER.items() if d != "none" and k in before and k in after
                         and _improved(before[k], after[k], d)],
               "regressed": _regressions(before, after),
               "steps": [s.as_dict() for s in steps],
               "changed": int(sum(s.changed for s in steps)),
               "no_op": all(not s.changed for s in steps),
               "refused": next((s.reason for s in steps if s.refused), "")}
    return current, RepairPlan(steps, summary)


def _improved(b: Any, a: Any, direction: str) -> bool:
    """True when ``a`` is an improvement on ``b`` in the metric's own direction."""
    if direction == "high":
        return bool(a) and not bool(b)
    return isinstance(a, _NUMBER) and isinstance(b, _NUMBER) and a < b

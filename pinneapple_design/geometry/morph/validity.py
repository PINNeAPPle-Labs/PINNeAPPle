"""Validity of a deformed surface: the checks that decide whether a design may enter the solver at all.

A morphing is only useful if the geometry it produces is still a body. The gate here answers the questions a
preprocessor asks before meshing — the surface closed, the orientation consistent, no degenerate faces, the volume
finite and positive, the volume and frontal area inside the band the design space declared, no face turned inside out,
no triangle pair intersecting that did not intersect before — and it answers them with a reason, so a rejected sample
says *why* instead of disappearing from the study.

The watertightness, orientation, degenerate-face and enclosed-volume numbers are ``pinneapple_data.cae.quality``'s
own surface checks, the same ones the mesh quality app reports, rather than a second implementation of them. The
self-intersection test and the thickness probe are here because the CAE layer has neither: the closed surface of a car
after a strong deformation is watertight and still wrong, and that is exactly the failure this gate is for.

Cost, measured on ``data/geometries/road_car`` (10 898 vertices, 21 760 triangles): the surface checks 0.27 s, the
self-intersection test about 0.15 s, so a gated sample costs a fraction of a second against the minutes of the mesh
that follows it. ``thickness=True`` costs more and is off by default.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["Reference", "ValidityReport", "check_surface", "frontal_area", "self_intersections",
           "thickness"]


# ------------------------------------------------------------------------------------------- geometry helpers
def _triangles(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    return V[F]


def _normals(tri: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit face normals and face areas."""
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    a2 = np.linalg.norm(n, axis=1)
    return n / np.maximum(a2, 1e-300)[:, None], 0.5 * a2


def frontal_area(V: np.ndarray, F: np.ndarray, direction: Any = (1.0, 0.0, 0.0)) -> float:
    """Projected area on the plane normal to ``direction``: what ``ref_area`` of a drag coefficient means.

    Half the sum of ``|A n.d|`` over the faces, which for a closed body is the area of one silhouette: the signed sum
    is zero (the divergence theorem), so the half is what turns the two sides of a body into one frontal area."""
    n, area = _normals(_triangles(V, F))
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    return float(0.5 * np.sum(area * np.abs(n @ d)))


# ------------------------------------------------------------------------------------------- self-intersection
def _candidate_pairs(V: np.ndarray, F: np.ndarray, cell: float | None) -> np.ndarray:
    """Pairs of faces whose axis-aligned boxes share a grid cell.

    Faces are binned by every cell their bounding box touches; a face too long for that (a cell span above 32) is set
    aside and paired against everything. Only the pairs that then pass the triangle-triangle test are intersections,
    so the grid may over-report freely."""
    tri = _triangles(V, F)
    lo, hi = tri.min(1), tri.max(1)
    size = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0
    if cell is None:
        diag = np.linalg.norm(hi - lo, axis=1)
        cell = max(float(np.median(diag)), 1e-9 * size) * 2.0
    origin = lo.min(0) - cell
    i0 = np.floor((lo - origin) / cell).astype(np.int64)
    i1 = np.floor((hi - origin) / cell).astype(np.int64)
    span = np.prod(i1 - i0 + 1, axis=1)
    small = np.nonzero(span <= 32)[0]
    big = np.nonzero(span > 32)[0]

    pairs: list[np.ndarray] = []
    if len(small):
        cnt = span[small]
        owner = np.repeat(small, cnt)
        starts = np.concatenate(([0], np.cumsum(cnt)[:-1]))
        pos = np.arange(int(cnt.sum())) - np.repeat(starts, cnt)
        d = (i1[owner] - i0[owner] + 1)
        nx, nxy = d[:, 0], d[:, 0] * d[:, 1]
        cx = i0[owner, 0] + pos % nx
        cy = i0[owner, 1] + (pos // nx) % d[:, 1]
        cz = i0[owner, 2] + pos // nxy
        key = (cx * 1000003 + cy) * 1000003 + cz
        order = np.argsort(key, kind="stable")
        key_s, owner_s = key[order], owner[order]
        edges = np.nonzero(np.diff(key_s))[0] + 1
        group_starts = np.concatenate(([0], edges))
        group_ends = np.concatenate((edges, [len(key_s)]))
        for s, e in zip(group_starts, group_ends, strict=True):
            if e - s < 2:
                continue
            g = owner_s[s:e]
            ia, ib = np.triu_indices(len(g), 1)
            pairs.append(np.stack([g[ia], g[ib]], 1))
    if len(big):
        allf = np.arange(len(F))
        for b in big:
            other = allf[allf != b]
            pairs.append(np.stack([np.full(len(other), b), other], 1))
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    p = np.concatenate(pairs)
    return p


def _segment_triangle(orig: np.ndarray, D: np.ndarray, tri: np.ndarray,
                      eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Möller-Trumbore for segments ``orig + t D`` with ``t`` in [0, 1] against ``tri`` (p, 3 vertices).

    Returns the hit mask and ``t`` (the parameter along the segment, in units of ``|D|``)."""
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    h = np.cross(D, e2)
    a = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(a) > eps
    inv = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
    s = orig - tri[:, 0]
    u = inv * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, e1)
    v = inv * np.einsum("ij,ij->i", D, q)
    t = inv * np.einsum("ij,ij->i", e2, q)
    hit = ok & (u >= -eps) & (v >= -eps) & (u + v <= 1 + eps) & (t >= -eps) & (t <= 1 + eps)
    return hit, t


def _cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """z component of the 3-D cross product of two (n, 2) arrays (numpy 2 deprecated ``np.cross`` on 2-D)."""
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def _segment_segment_2d(p0, p1, q0, q1, eps: float = 1e-12) -> np.ndarray:
    """Proper and improper crossing of 2-D segments (the coplanar case the ray test cannot see)."""
    d1 = _cross2(p1 - p0, q1 - q0)
    d2 = _cross2(q0 - p0, q1 - q0)
    d3 = _cross2(q0 - p0, p1 - p0)
    s = d2 / np.where(np.abs(d1) > eps, d1, np.nan)
    t = d3 / np.where(np.abs(d1) > eps, d1, np.nan)
    return (s >= -eps) & (s <= 1 + eps) & (t >= -eps) & (t <= 1 + eps)


def self_intersections(V: np.ndarray, F: np.ndarray, *, cell: float | None = None,
                       max_report: int = 64) -> dict[str, Any]:
    """Triangle pairs that intersect although they share no vertex.

    A deforming surface that folds onto itself keeps its faces' winding and its closed boundary: nothing but this test
    catches it. The result carries the ``keys`` of the pairs (``face_a * n_faces + face_b``, sorted) so a gate can ask
    which of them are *new* against a reference surface — most real STL bodies already have overlapping shells, and a
    gate that counted them again would be measuring the model, not the deformation."""
    empty = {"pairs": 0, "intersections": 0, "faces": [], "checked_pairs": 0,
             "keys": np.zeros(0, dtype=np.int64)}
    pairs = _candidate_pairs(V, F, cell)
    if not len(pairs):
        return empty
    tri = _triangles(V, F)
    lo, hi = tri.min(1), tri.max(1)
    # the grid bins by cell, which over-reports: the true bounding-box overlap is the cheap exact first pass
    pairs = pairs[np.all((lo[pairs[:, 0]] <= hi[pairs[:, 1]]) & (lo[pairs[:, 1]] <= hi[pairs[:, 0]]), axis=1)]
    # a pair that shares any vertex is a neighbour, not an intersection
    if len(pairs):
        shared = ((F[pairs[:, 0]][:, :, None] == F[pairs[:, 1]][:, None, :]).any(axis=(1, 2)))
        pairs = pairs[~shared]
    if len(pairs):                                      # a face binned in several cells proposes a pair more than once
        pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    if not len(pairs):
        return empty
    A, B = _triangles(V, F[pairs[:, 0]]), _triangles(V, F[pairs[:, 1]])
    hit = np.zeros(len(pairs), dtype=bool)
    for k in range(3):
        orig = A[:, k]
        D = A[:, (k + 1) % 3] - A[:, k]
        hit |= _segment_triangle(orig, D, B)[0]
    if not hit.all():                                   # the other direction (mutual crossing, not containment)
        rest = np.nonzero(~hit)[0]
        for k in range(3):
            orig = B[rest, k]
            D = B[rest, (k + 1) % 3] - B[rest, k]
            hit[rest] |= _segment_triangle(orig, D, A[rest])[0]
    # coplanar overlap: the ray test is blind to it, so project the flagged pairs and cross their edges in 2-D
    if not hit.all():
        cand = np.nonzero(~hit)[0]
        found = _near_coplanar(A[cand], B[cand], cand)
        if len(found):
            na = np.cross(A[found, 1] - A[found, 0], A[found, 2] - A[found, 0])
            drop = np.argmax(np.abs(na), axis=1)
            pa = _drop_axis(A[found], drop)
            pb = _drop_axis(B[found], drop)
            cross = np.zeros(len(found), dtype=bool)
            for k in range(3):
                for m in range(3):
                    cross |= _segment_segment_2d(pa[:, k], pa[:, (k + 1) % 3],
                                                 pb[:, m], pb[:, (m + 1) % 3])
            hit[found[cross]] = True
    hits = np.nonzero(hit)[0]
    hp = pairs[hits]
    faces = sorted({int(i) for i in hp.ravel()})[:2 * max_report]
    keys = np.sort(hp[:, 0].astype(np.int64) * len(F) + hp[:, 1])
    return {"pairs": int(len(pairs)), "intersections": int(len(hits)), "faces": faces,
            "checked_pairs": int(len(pairs)), "keys": keys}


def _drop_axis(tri: np.ndarray, drop: np.ndarray) -> np.ndarray:
    """Project triangles onto the plane whose axis is ``drop`` (per triangle), for the 2-D crossing test."""
    out = np.empty((len(tri), 3, 2))
    for ax in range(3):
        m = drop == ax
        if m.any():
            out[m] = np.delete(tri[m], ax, axis=2)
    return out


def _near_coplanar(A: np.ndarray, B: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """The subset of pairs whose planes nearly coincide, within a thousandth of the face size."""
    if not len(A):
        return idx
    na = np.cross(A[:, 1] - A[:, 0], A[:, 2] - A[:, 0])
    nb = np.cross(B[:, 1] - B[:, 0], B[:, 2] - B[:, 0])
    na /= np.maximum(np.linalg.norm(na, axis=1), 1e-300)[:, None]
    nb /= np.maximum(np.linalg.norm(nb, axis=1), 1e-300)[:, None]
    parallel = np.abs(np.einsum("ij,ij->i", na, nb)) > 1 - 1e-6
    dist = np.abs(np.einsum("ij,ij->i", B[:, 0] - A[:, 0], na))
    scale = np.linalg.norm(A[:, 1] - A[:, 0], axis=1) + 1e-300
    return idx[parallel & (dist < 1e-3 * scale)]


# ------------------------------------------------------------------------------------------- thickness probe
def thickness(V: np.ndarray, F: np.ndarray, *, samples: int = 128, seed: int = 0,
              max_hits: int = 4096) -> dict[str, Any]:
    """Local wall thickness from rays cast into the body: face centroid, along the inward normal, first hit.

    ``samples`` faces are taken at random (a fixed seed, so the number is a property of the geometry and not of the
    run) and the nearest hit on a face that does not share a vertex is the thickness there. Reported as the smallest
    of the sampled rays, with the number of rays that hit anything at all — a ray that leaves without hitting means
    the surface is not closed or the body has an opening, and the count says so instead of returning a large number.
    """
    tri = _triangles(V, F)
    n, _ = _normals(tri)
    rng = np.random.default_rng(seed)
    take = rng.choice(len(F), size=min(samples, len(F)), replace=False)
    if max_hits and len(F) > max_hits:                  # bound the work on a large mesh: a uniform sample of targets
        targets = rng.choice(len(F), size=max_hits, replace=False)
    else:
        targets = np.arange(len(F))
    T = tri[targets]
    reach = float(np.linalg.norm(V.max(0) - V.min(0)))
    best = np.inf
    missed = 0
    for i in take:
        orig = np.repeat(tri[i].mean(0)[None, :], len(T), 0)
        D = np.repeat((-n[i] * reach)[None, :], len(T), 0)
        hit, t = _segment_triangle(orig, D, T)
        shared = (F[targets][:, :, None] == F[i][None, None, :]).any(axis=(1, 2))
        hit &= ~shared
        if not hit.any():
            missed += 1
            continue
        best = min(best, float(np.min(t[hit] * reach)))
    return {"min": float(best) if np.isfinite(best) else None, "samples": int(len(take)),
            "rays_without_hit": int(missed)}


# ------------------------------------------------------------------------------------------- the gate
@dataclass
class ValidityReport:
    """The verdict on one deformed surface, with the reasons and the numbers behind them.

    ``ok`` is ``not reasons``: every failing check appends a string naming the quantity and its value against the
    limit, so a rejected sample is a record, not a silence. ``metrics`` carries everything measured, whether it failed
    or not, which is what a study plots next to the drag."""
    ok: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.ok

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "reasons": list(self.reasons), "metrics": dict(self.metrics)}


@dataclass
class Reference:
    """Everything the gate needs to know about the surface *before* the deformation, measured once.

    The volume, area, face normals and self-intersection count of a reference mesh do not change between samples of a
    study, and measuring them per sample doubles the cost of the gate for nothing: a design space builds one
    ``Reference`` and passes it to every check. The watertightness and orientation of the reference are not stored
    because the deformation cannot change them — the topology is fixed."""
    vertices: np.ndarray
    faces: np.ndarray
    volume: float | None
    area: float
    frontal_area: float
    normals: np.ndarray
    intersections: int
    intersection_keys: np.ndarray
    centroids: np.ndarray
    diagonals: np.ndarray
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0)

    @classmethod
    def build(cls, V: Any, F: Any, direction: Any = (1.0, 0.0, 0.0)) -> Reference:
        from pinneapple_data.cae.model import CellBlock, Mesh
        from pinneapple_data.cae.quality import surface_checks as cae_surface_checks

        V = np.ascontiguousarray(V, dtype=np.float64).reshape(-1, 3)
        F = np.ascontiguousarray(F, dtype=np.int64).reshape(-1, 3)
        sc = cae_surface_checks(Mesh(points=V, blocks=[CellBlock("triangle", F)]))
        n, _ = _normals(_triangles(V, F))
        si = self_intersections(V, F)
        tri = _triangles(V, F)
        return cls(vertices=V, faces=F, volume=sc.get("enclosed_volume"), area=float(sc.get("area", 0.0)),
                   frontal_area=frontal_area(V, F, direction), normals=n,
                   intersections=si["intersections"], intersection_keys=si["keys"],
                   centroids=tri.mean(1), diagonals=np.linalg.norm(tri.max(1) - tri.min(1), axis=1),
                   direction=tuple(float(x) for x in np.asarray(direction, dtype=float)))


def check_surface(V: np.ndarray, F: np.ndarray, *, reference: Any = None,
                  max_volume_change: float | None = 0.5, max_area_change: float | None = 0.5,
                  allow_flipped: int = 0, thickness_check: bool = False, thickness_samples: int = 128,
                  contact_tolerance: float = 0.0, direction: Any = (1.0, 0.0, 0.0)) -> ValidityReport:
    """Every check of a morphing, on one surface, against the surface the morpher was fitted to.

    Parameters
    ----------
    V, F: the deformed surface.
    reference: the surface before the deformation — a ``(vertices, faces)`` pair, or a :class:`Reference` built once
        from it. The volume and area bands and the flipped-face and self-intersection counts are relative to it, so a
        body that already intersected itself somewhere is not punished twice. Building the ``Reference`` once and
        reusing it across the samples of a study halves the cost of the gate, which is why the class exists.
    max_volume_change, max_area_change: allowed relative change (0.5 = ±50 %); ``None`` disables the band.
    allow_flipped: tolerated number of faces whose normal turned over (0: the surface must stay right side out).
    thickness_check: also cast the thickness rays (costlier; off by default).
    contact_tolerance: a pair of faces that were already touching in the reference — their centroids closer than
        this fraction of their two face diagonals — is not counted as a new intersection, because a smooth
        deformation that closes a gap of zero width has not made the body worse. 0 (the default) counts every new
        pair; 1 is the setting for an assembly of shells (a car body with its wheels), where a wheel-arch gap of a
        few millimetres closes under any body-wide deformation. The ignored pairs are reported in the metrics.
    """
    from pinneapple_data.cae.model import CellBlock, Mesh
    from pinneapple_data.cae.quality import surface_checks as cae_surface_checks

    V = np.asarray(V, dtype=np.float64).reshape(-1, 3)
    F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
    ref = reference if isinstance(reference, Reference) else (
        None if reference is None else Reference.build(reference[0], reference[1], direction=direction))

    reasons: list[str] = []
    metrics: dict[str, Any] = {}
    ok_finite = bool(np.isfinite(V).all())
    metrics["finite"] = ok_finite
    if not ok_finite:
        reasons.append("vertices are not finite (NaN or inf after the deformation)")

    mesh = Mesh(points=V, blocks=[CellBlock("triangle", F)])
    sc = cae_surface_checks(mesh)
    volume = sc.get("enclosed_volume")
    area = float(sc.get("area", 0.0))
    metrics.update({"faces": sc.get("faces"), "shells": sc.get("shells"), "open_edges": sc.get("open_edges"),
                    "nonmanifold_edges": sc.get("nonmanifold_edges"),
                    "inconsistent_orientation_edges": sc.get("inconsistent_orientation_edges"),
                    "watertight": sc.get("watertight"), "degenerate_faces": sc.get("degenerate_faces"),
                    "enclosed_volume": volume, "area": area})
    if not sc.get("watertight", False):
        reasons.append(f"surface is not watertight ({sc.get('open_edges')} open, "
                       f"{sc.get('nonmanifold_edges')} non-manifold edges)")
    if sc.get("degenerate_faces"):
        reasons.append(f"{sc['degenerate_faces']} degenerate faces")
    if volume is None:
        reasons.append("volume not measurable: the surface is not closed")
    elif not np.isfinite(volume) or volume <= 0:
        reasons.append(f"enclosed volume {volume!r} is not finite and positive")

    n, _area = _normals(_triangles(V, F))
    metrics["flipped_faces"] = 0
    metrics["frontal_area"] = frontal_area(V, F, direction)
    if ref is not None:
        if ref.vertices.shape != V.shape or ref.faces.shape != F.shape:
            raise ValueError("check_surface: the reference must have the same vertices and faces")
        flipped = int(np.sum(np.einsum("ij,ij->i", n, ref.normals) < 0.0))
        metrics["flipped_faces"] = flipped
        if flipped > allow_flipped:
            reasons.append(f"{flipped} faces turned inside out (limit {allow_flipped})")
        if max_volume_change is not None and volume and ref.volume:
            dv = (volume - ref.volume) / abs(ref.volume)
            metrics["volume_change"] = float(dv)
            metrics["volume_ratio"] = float(volume / ref.volume)
            if abs(dv) > max_volume_change:
                reasons.append(f"volume changed {100 * dv:+.1f} % (limit ±{100 * max_volume_change:.0f} %)")
        metrics["reference_frontal_area"] = ref.frontal_area
        if max_area_change is not None and ref.frontal_area > 0:
            da = (metrics["frontal_area"] - ref.frontal_area) / ref.frontal_area
            metrics["frontal_area_change"] = float(da)
            if abs(da) > max_area_change:
                reasons.append(f"frontal area changed {100 * da:+.1f} % (limit ±{100 * max_area_change:.0f} %)")
        n_ref = ref.intersections
        known = ref.intersection_keys
    else:
        n_ref, known = 0, np.zeros(0, dtype=np.int64)

    si = self_intersections(V, F)
    keys = si["keys"]
    if len(known):                                      # only the pairs that were not already intersecting count
        new = keys[~np.isin(keys, known, assume_unique=True)]
    else:
        new = keys
    metrics["contact_pairs_ignored"] = 0
    if contact_tolerance and ref is not None and len(new):
        fa, fb = new // len(F), new % len(F)
        gap = np.linalg.norm(ref.centroids[fa] - ref.centroids[fb], axis=1)
        touching = gap <= 0.5 * contact_tolerance * (ref.diagonals[fa] + ref.diagonals[fb])
        metrics["contact_pairs_ignored"] = int(touching.sum())
        new = new[~touching]
    metrics["self_intersections"] = si["intersections"]
    metrics["new_self_intersections"] = int(len(new))
    metrics["reference_self_intersections"] = n_ref
    metrics["intersection_faces"] = si["faces"]
    if len(new):
        reasons.append(f"{len(new)} face pairs intersect that did not in the reference ({n_ref} pairs already "
                       f"intersect there)")

    if thickness_check:
        th = thickness(V, F, samples=thickness_samples)
        metrics["thickness"] = th
        if th["rays_without_hit"]:
            reasons.append(f"{th['rays_without_hit']} of {th['samples']} thickness rays left the body without a hit")

    return ValidityReport(ok=not reasons, reasons=reasons, metrics=metrics)

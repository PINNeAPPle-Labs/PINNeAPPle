"""Convex rigid shapes for the granular solver: convex polyhedra and spheres.

Every shape is stored in its own body frame, centred on its centre of mass, so a rigid motion is a rotation matrix
and a translation.  Polyhedra carry closed face loops with outward winding, which the volume-overlap contact model
in :mod:`pinneapple_simulation.granular.overlap` clips directly -- that is the whole reason this module works on
``(vertices, faces)`` rather than on half-spaces or triangle soups.

    >>> from pinneapple_simulation.granular import box, faceted_sphere
    >>> cube = box(0.02, 0.02, 0.02)
    >>> round(cube.volume, 12) == round(0.02 ** 3, 12)
    True

The volumes, centroids, surface areas and inertia tensors are exact for a convex polyhedron: volume and first and
second moments come from the divergence theorem over the face loops, not from a voxel or sampling approximation.
Sphere-sphere geometry is analytic elsewhere (:mod:`pinneapple_simulation.granular.contacts`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "ConvexPolyhedron",
    "Sphere",
    "box",
    "regular_prism",
    "faceted_sphere",
    "rotation_matrix",
]

_EPS = 1.0e-12


# ------------------------------------------------------------------------------------------------- rigid motions
def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix about ``axis`` (Rodrigues). ``axis`` need not be normalised."""
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n < _EPS:
        return np.eye(3)
    k = a / n
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _orthonormal_tangents(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane normal to ``n``, with ``t1 x t2 = n``."""
    n = n / max(np.linalg.norm(n), _EPS)
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, seed)
    t1 /= max(np.linalg.norm(t1), _EPS)
    t2 = np.cross(n, t1)
    return t1, t2


# ---------------------------------------------------------------------------------------------------- polyhedron
@dataclass
class ConvexPolyhedron:
    """A closed convex polyhedron given by vertices (``(n, 3)``) and face loops (indices, outward-wound).

    Positions are measured from the centre of mass, so ``vertices.mean(0)`` is not zero in general but the volume
    centroid is.  Faces may have any number of vertices and need not be coplanar in a first guess: each face's
    normal is the area-weighted (Newell) normal of its loop.
    """

    vertices: np.ndarray
    faces: tuple[tuple[int, ...], ...]
    density: float = 1.0
    volume: float = field(init=False)
    centroid: np.ndarray = field(init=False)
    face_normals: np.ndarray = field(init=False)
    face_offsets: np.ndarray = field(init=False)
    face_areas: np.ndarray = field(init=False)
    face_centres: np.ndarray = field(init=False)
    edges: np.ndarray = field(init=False)
    bounding_radius: float = field(init=False)
    inertia: np.ndarray = field(init=False)
    area: float = field(init=False)

    def __post_init__(self) -> None:
        v = np.asarray(self.vertices, dtype=float)
        if v.ndim != 2 or v.shape[1] != 3 or len(v) < 4:
            raise ValueError("ConvexPolyhedron needs at least 4 vertices of shape (n, 3)")
        self.vertices = v
        self.faces = tuple(tuple(int(i) for i in f) for f in self.faces)
        if not self.faces:
            raise ValueError("ConvexPolyhedron needs at least one face")
        self._build()

    # ---- construction ---------------------------------------------------------------------------------------
    def _build(self) -> None:
        normals, areas, centres = [], [], []
        for face in self.faces:
            p = self.vertices[list(face)]
            n, a, c = _loop_normal_area_centroid(p)
            normals.append(n)
            areas.append(a)
            centres.append(c)
        self.face_normals = np.array(normals)
        self.face_areas = np.array(areas)
        self.face_centres = np.array(centres)
        self.face_offsets = np.einsum("ij,ij->i", self.face_normals, self.face_centres)

        volume = float(np.einsum("ij,ij->i", self.face_centres, self.face_normals) @ self.face_areas) / 3.0
        if volume <= _EPS:
            raise ValueError(
                "ConvexPolyhedron volume is not positive: the faces must wind counter-clockwise seen from outside"
            )
        self.volume = volume
        self.area = float(self.face_areas.sum())

        # first and second moments from the tetrahedra between the origin and each face fan
        first = np.zeros(3)
        second = np.zeros((3, 3))
        for face, n, c in zip(self.faces, self.face_normals, self.face_centres, strict=True):
            p = self.vertices[list(face)]
            for k in range(1, len(p) - 1):
                a, b, cc = p[0], p[k], p[k + 1]
                det = float(np.dot(a, np.cross(b, cc))) / 6.0
                if abs(det) < _EPS:
                    continue
                quad = np.array([np.zeros(3), a, b, cc])
                first += det * quad.sum(0) / 4.0
                second += det * (quad.T @ quad + np.outer(quad.sum(0), quad.sum(0))) / 20.0
                _ = n, c
        centroid = first / self.volume
        second_c = second - self.volume * np.outer(centroid, centroid)
        self.centroid = centroid
        self.inertia = (np.trace(second_c) * np.eye(3) - second_c) * self.density

        pairs = set()
        for face in self.faces:
            m = len(face)
            for k in range(m):
                i, j = face[k], face[(k + 1) % m]
                pairs.add((i, j) if i < j else (j, i))
        self.edges = np.array(sorted(pairs), dtype=int)
        self.bounding_radius = float(np.max(np.linalg.norm(self.vertices - self.centroid, axis=1)))

    @classmethod
    def from_vertices(cls, points: np.ndarray, density: float = 1.0, hull=None) -> ConvexPolyhedron:
        """Convex hull of ``points`` (triangulated faces with outward winding)."""
        from scipy.spatial import ConvexHull

        p = np.asarray(points, dtype=float)
        hull = hull if hull is not None else ConvexHull(p)
        verts = p[hull.vertices]
        remap = {int(old): new for new, old in enumerate(hull.vertices)}
        faces = []
        for simplex, plane in zip(hull.simplices, hull.equations, strict=True):
            loop = [remap[int(i)] for i in simplex]
            n = np.cross(verts[loop[1]] - verts[loop[0]], verts[loop[2]] - verts[loop[0]])
            if np.dot(n, plane[:3]) < 0.0:                      # orient against the hull's own outward plane
                loop.reverse()
            faces.append(tuple(loop))
        return cls(verts, tuple(faces), density=density)

    # ---- rigid motion ---------------------------------------------------------------------------------------
    def transformed(self, rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> ConvexPolyhedron:
        """Copy rotated by ``rotation`` (3x3, about the centre of mass) and then translated."""
        R = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
        t = np.zeros(3) if translation is None else np.asarray(translation, dtype=float)
        return ConvexPolyhedron(self.vertices @ R.T + t, self.faces, density=self.density)

    def world_vertices(self, rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Vertices in world coordinates for a body-frame shape at ``position`` with orientation ``rotation``."""
        return self.vertices @ np.asarray(rotation, dtype=float).T + np.asarray(position, dtype=float)

    def support(self, direction: np.ndarray) -> np.ndarray:
        """Farthest vertex along ``direction`` (the support mapping of the convex-hull literature)."""
        d = np.asarray(direction, dtype=float)
        return self.vertices[int(np.argmax(self.vertices @ d))]

    def aabb(self, rotation: np.ndarray, position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w = self.world_vertices(rotation, position)
        return w.min(0), w.max(0)

    @property
    def mass(self) -> float:
        return self.volume * self.density

    @property
    def is_sphere(self) -> bool:
        return False

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"ConvexPolyhedron(vertices={len(self.vertices)}, faces={len(self.faces)}, "
                f"volume={self.volume:.6g})")


def _loop_normal_area_centroid(p: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Newell normal (unit), area and centroid of a planar polygon loop."""
    nxt = np.roll(p, -1, axis=0)
    n = np.cross(p, nxt).sum(0)                                  # 2 * area * unit normal
    norm = np.linalg.norm(n)
    if norm < _EPS:
        return np.zeros(3), 0.0, p.mean(0)
    area = 0.5 * norm
    n = n / norm
    # area moments about a reference point, via fan triangles (stable for thin faces)
    ref = p.mean(0)
    centre = np.zeros(3)
    tri_area_sum = 0.0
    for k in range(len(p)):
        a = p[k] - ref
        b = p[(k + 1) % len(p)] - ref
        w = 0.5 * np.linalg.norm(np.cross(a, b))
        if w <= _EPS:
            continue
        centre += w * (ref + (a + b) / 3.0)
        tri_area_sum += w
    if tri_area_sum > _EPS:
        centre = centre / tri_area_sum
    else:  # pragma: no cover - degenerate
        centre = p.mean(0)
    return n, area, centre + np.zeros(3)


# -------------------------------------------------------------------------------------------------------- sphere
@dataclass
class Sphere:
    """A sphere of radius ``radius``; the centre of mass is the particle position."""

    radius: float
    density: float = 1.0

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("Sphere radius must be positive")

    @property
    def volume(self) -> float:
        return 4.0 / 3.0 * math.pi * self.radius ** 3

    @property
    def mass(self) -> float:
        return self.volume * self.density

    @property
    def area(self) -> float:
        return 4.0 * math.pi * self.radius ** 2

    @property
    def inertia(self) -> np.ndarray:
        i = 2.0 / 5.0 * self.mass * self.radius ** 2
        return np.diag([i, i, i])

    @property
    def bounding_radius(self) -> float:
        return self.radius

    @property
    def is_sphere(self) -> bool:
        return True

    def transformed(self, rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> Sphere:
        return Sphere(self.radius, density=self.density)

    def world_vertices(self, rotation: np.ndarray, position: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise TypeError("a sphere has no vertices: use the analytic sphere contact models")


# ------------------------------------------------------------------------------------------------------ factories
def box(lx: float, ly: float, lz: float, density: float = 1.0) -> ConvexPolyhedron:
    """Axis-aligned box centred on the origin (24 vertices, 6 quad faces, outward-wound)."""
    hx, hy, hz = lx / 2.0, ly / 2.0, lz / 2.0
    verts = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ], dtype=float)
    faces = (
        (0, 3, 2, 1),   # -z, normal (0, 0, -1)
        (4, 5, 6, 7),   # +z
        (0, 1, 5, 4),   # -y
        (2, 3, 7, 6),   # +y
        (1, 2, 6, 5),   # +x
        (0, 4, 7, 3),   # -x
    )
    poly = ConvexPolyhedron(verts, faces, density=density)
    poly.half_extents = np.array([hx, hy, hz])       # lets the contact kernel clamp analytically
    poly.is_box = True
    return poly


def regular_prism(sides: int, radius: float, height: float, density: float = 1.0) -> ConvexPolyhedron:
    """Regular ``sides``-gonal prism, axis along ``z``, centred on the origin."""
    if sides < 3:
        raise ValueError("regular_prism needs at least 3 sides")
    th = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    ring = np.c_[radius * np.cos(th), radius * np.sin(th)]
    verts = np.r_[np.c_[ring, np.full(sides, -height / 2.0)], np.c_[ring, np.full(sides, height / 2.0)]]
    faces: list[tuple[int, ...]] = [tuple(reversed(range(sides))), tuple(range(sides, 2 * sides))]
    for k in range(sides):
        k2 = (k + 1) % sides
        faces.append((k, k2, sides + k2, sides + k))
    return ConvexPolyhedron(verts, faces, density=density)


def faceted_sphere(radius: float, n_points: int = 42, density: float = 1.0) -> ConvexPolyhedron:
    """A sphere approximated by the convex hull of ``n_points`` points spread over its surface.

    Used where a rounded particle has to meet the volume-overlap contact model (the analytic sphere models are
    exact and cheaper, so prefer :class:`Sphere` when the other body is a sphere or a plane).
    """
    n = int(n_points)
    if n < 6:
        raise ValueError("faceted_sphere needs at least 6 points")
    k = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * k / n)                            # Fibonacci spherical lattice: near-uniform
    theta = math.pi * (1.0 + 5.0 ** 0.5) * k
    pts = radius * np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]
    return ConvexPolyhedron.from_vertices(pts, density=density)


_ = _skew  # kept for the integrator's moment bookkeeping imports
_ = _orthonormal_tangents

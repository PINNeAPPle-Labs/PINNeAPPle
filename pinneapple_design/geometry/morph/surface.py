"""One surface type for the morphing ops: a triangle soup ``(vertices, faces)`` lifted from whatever the caller has.

The morphing ops never touch the topology. Faces are carried along untouched, so a deformation only moves vertices,
which is what keeps a field defined on the mesh comparable across designs: vertex ``i`` is the same physical point
before and after. The loaders below accept what the repository already produces — an STL path, a
``pinneapple_design.geometry.core.mesh.MeshData``, a ``pinneapple_data.cae.Mesh``, or a bare ``(V, F)`` pair — and
return arrays, so a morpher built on one mesh can be fed the many.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["as_surface", "Surface"]


def as_surface(obj: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(vertices, faces)`` as float64 (n, 3) and int64 (m, 3) from a path, a MeshData, a cae Mesh or a pair."""
    if isinstance(obj, (str, Path)):
        return _from_stl(Path(obj))
    if isinstance(obj, tuple) and len(obj) == 2:
        v, f = obj
        return _clean(v, f)
    # MeshData and cae Mesh both carry vertices-or-points plus faces-or-a-triangle block
    v = getattr(obj, "vertices", None)
    if v is None:
        v = getattr(obj, "points", None)
    f = getattr(obj, "faces", None)
    if f is None and hasattr(obj, "blocks"):
        tris = [b.corners for b in obj.blocks if getattr(b, "dim", 3) == 2 and b.corners.shape[1] == 3]
        f = tris[0] if tris else None
    if v is None or f is None:
        raise TypeError(f"as_surface: cannot read (vertices, faces) from {type(obj).__name__}")
    return _clean(v, f)


def _from_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    from pinneapple_data.cae import io as cae_io

    mesh = cae_io.read_stl(path.read_bytes())
    return as_surface(mesh)


def _clean(v: Any, f: Any) -> tuple[np.ndarray, np.ndarray]:
    V = np.ascontiguousarray(v, dtype=np.float64).reshape(-1, 3)
    F = np.ascontiguousarray(f, dtype=np.int64)
    F = F[:, :3] if F.ndim == 2 and F.shape[1] > 3 else F
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError("as_surface: faces must be a triangle list (m, 3)")
    if F.size and int(F.max()) >= len(V):
        raise ValueError("as_surface: a face indexes a vertex that does not exist")
    return V, F


class Surface:
    """A surface held by the morphers: the reference vertices, the faces, and the deformation applied so far.

    ``Surface`` is what a morpher is built on and what a deformation returns; ``.vertices`` is always the deformed
    configuration and ``.reference`` the one the morpher was fitted to, so ``.displacement`` is the field that moved
    it — the quantity a validity check compares against."""

    __slots__ = ("reference", "vertices", "faces")

    def __init__(self, vertices: Any, faces: Any, *, reference: Any = None):
        self.vertices, self.faces = as_surface((vertices, faces))
        self.reference = self.vertices.copy() if reference is None else np.asarray(reference, dtype=np.float64)

    @classmethod
    def load(cls, obj: Any) -> Surface:
        V, F = as_surface(obj)
        return cls(V, F)

    @property
    def displacement(self) -> np.ndarray:
        return self.vertices - self.reference

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    def moved(self, vertices: np.ndarray) -> Surface:
        return Surface(vertices, self.faces, reference=self.reference)

    def as_tuple(self) -> tuple[np.ndarray, np.ndarray]:
        """The ``(vertices, faces)`` pair ``pp.cfd.ExternalFlow`` and the studio viewers take."""
        return self.vertices, self.faces

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(0), self.vertices.max(0)

    def __repr__(self) -> str:
        return f"Surface({self.n_vertices} vertices, {self.n_faces} faces)"

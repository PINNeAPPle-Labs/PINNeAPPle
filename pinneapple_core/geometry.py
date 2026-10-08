"""Geometry: a Domain plus named boundaries and a way to mesh it.

>>> from pinneapple_core import Geometry
>>> g = Geometry.box([0, 0], [2, 1])
>>> g.boundary_names
['x_min', 'x_max', 'y_min', 'y_max']
>>> g.mesh(h=0.25).dim
2
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .domain import Domain
from .mesh import Mesh

Predicate = Callable[[np.ndarray], np.ndarray]


class Geometry:
    """A :class:`Domain` with named boundary parts.

    ``boundaries`` maps a name to a predicate ``points -> bool mask`` that selects
    the part of the boundary it names. Boxes get ``x_min, x_max, y_min, ...`` for free.
    """

    def __init__(self, domain: Domain, boundaries: Optional[Dict[str, Predicate]] = None) -> None:
        self.domain = domain
        self._boundaries: Dict[str, Predicate] = dict(boundaries or {})
        if domain.is_box and not self._boundaries:
            self._boundaries = self._box_boundaries(domain)

    @staticmethod
    def _box_boundaries(domain: Domain) -> Dict[str, Predicate]:
        axis_names = "xyz"
        out: Dict[str, Predicate] = {}
        for a in range(domain.dim):
            tol = 1e-9 * (domain.hi[a] - domain.lo[a])
            out[f"{axis_names[a]}_min"] = lambda p, a=a, t=tol: np.abs(p[:, a] - domain.lo[a]) <= t
            out[f"{axis_names[a]}_max"] = lambda p, a=a, t=tol: np.abs(p[:, a] - domain.hi[a]) <= t
        return out

    # -- constructors -------------------------------------------------------
    @classmethod
    def box(cls, lo: Sequence[float], hi: Sequence[float]) -> "Geometry":
        return cls(Domain.box(lo, hi))

    @classmethod
    def from_csg(cls, shape, boundaries: Optional[Dict[str, Predicate]] = None) -> "Geometry":
        """From a 2D CSG shape of ``pinneapple_design.geometry.csg`` (L-shapes, annuli, ...)."""
        return cls(Domain.from_sdf(shape), boundaries)

    # -- queries ------------------------------------------------------------
    @property
    def dim(self) -> int:
        return self.domain.dim

    @property
    def boundary_names(self) -> List[str]:
        return list(self._boundaries)

    def tag(self, points: np.ndarray) -> np.ndarray:
        """Name of the first boundary each point lies on, ``""`` for none."""
        p = self.domain._as_points(points)
        out = np.full(p.shape[0], "", dtype=object)
        for name, pred in reversed(list(self._boundaries.items())):
            out[np.asarray(pred(p), dtype=bool)] = name
        return out

    def sample_interior(self, n: int, *, seed: int = 0) -> np.ndarray:
        return self.domain.sample_interior(n, seed=seed)

    def sample_boundary(self, n: int, name: Optional[str] = None, *, seed: int = 0) -> np.ndarray:
        """``n`` boundary points, all of the boundary or only the part called ``name``."""
        if name is None:
            return self.domain.sample_boundary(n, seed=seed)
        if name not in self._boundaries:
            raise KeyError(f"unknown boundary {name!r}; have {self.boundary_names}")
        pred = self._boundaries[name]
        out = np.empty((0, self.dim))
        trial = 0
        while out.shape[0] < n:
            cand = self.domain.sample_boundary(max(8 * n, 1024), seed=seed + trial)
            out = np.concatenate([out, cand[np.asarray(pred(cand), dtype=bool)]])
            trial += 1
            if trial > 200:
                raise RuntimeError(f"could not sample boundary {name!r}; does the predicate select anything?")
        return out[:n]

    # -- meshing ------------------------------------------------------------
    def mesh(self, h: float) -> Mesh:
        """Simplex mesh with target edge length ``h``.

        Boxes are cut into a conforming structured mesh. Other domains get a
        Delaunay mesh of a lattice plus boundary points, with cells outside the
        domain removed: fine for PINN/FEM prototyping; use gmsh (see
        ``pinneapple_design.geometry``) when mesh quality matters.
        """
        dom = self.domain
        if dom.is_box:
            shape = np.maximum(1, np.ceil((dom.hi - dom.lo) / h)).astype(int)
            return Mesh.structured(dom.lo, dom.hi, shape)
        if dom.dim != 2:
            raise NotImplementedError("meshing non-box domains is only supported in 2D")
        axes = [np.arange(lo + h / 2, hi, h) for lo, hi in zip(dom.lo, dom.hi)]
        lattice = np.stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")], axis=1)
        lattice = lattice[dom.sdf(lattice) < -0.5 * h]
        n_bnd = int(dom.boundary_measure() / h)
        bnd = dom.sample_boundary(max(n_bnd, 8), seed=0)
        pts = np.concatenate([lattice, bnd])
        return Mesh.from_points(pts, keep=lambda c: dom.sdf(c) <= 0.0)

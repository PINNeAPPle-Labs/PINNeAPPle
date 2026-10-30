"""The background grid: a uniform Cartesian box of cells carrying Q1 or Q2 Lagrange elements.

Nodes are numbered lexicographically (C order) on the full ``(p n_0 + 1) x ... `` lattice; only the nodes of active
cells become unknowns (:class:`DofMap`). Cells never conform to the part: the geometry enters through quadrature.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import basis

# a fixed, irrational fraction of a cell: shifts the grid off the planes a CAD part is usually aligned with, so a face
# of the part does not coincide with a grid plane (that only creates zero-volume cuts); the shift is reported
_JITTER = (np.sqrt(2) - 1) * 1e-3


@dataclass
class BackgroundGrid:
    lo: np.ndarray               # (d,) corner of the grid
    h: np.ndarray                # (d,) cell size per axis
    n: tuple[int, ...]           # cells per axis
    order: int = 1               # 1 = Q1 (multilinear), 2 = Q2 (multiquadratic)

    @property
    def d(self) -> int:
        return len(self.n)

    @property
    def n_cells(self) -> int:
        return int(np.prod(self.n))

    @property
    def node_shape(self) -> tuple[int, ...]:
        return tuple(self.order * k + 1 for k in self.n)

    @property
    def n_local(self) -> int:
        return basis.n_local(self.order, self.d)

    @property
    def size(self) -> float:
        """Characteristic cell size (the largest edge)."""
        return float(np.max(self.h))

    @classmethod
    def around(cls, bounds, h: float, order: int = 1, pad: float = 1.0, jitter: bool = True) -> "BackgroundGrid":
        """Grid of cells of size ``h`` covering ``bounds`` with ``pad`` cells of margin on every side."""
        lo, hi = (np.asarray(b, float) for b in bounds)
        hv = np.full(len(lo), float(h))
        start = lo - pad * hv - (_JITTER * hv if jitter else 0.0)
        n = tuple(int(k) for k in np.ceil((hi + pad * hv - start) / hv))
        return cls(start, hv, n, int(order))

    def cell_multi(self, cells: np.ndarray) -> np.ndarray:
        return np.stack(np.unravel_index(np.asarray(cells), self.n), axis=1)

    def cell_origin(self, cells: np.ndarray) -> np.ndarray:
        return self.lo + self.cell_multi(cells) * self.h

    def to_physical(self, cells: np.ndarray, xi: np.ndarray) -> np.ndarray:
        return self.lo + (self.cell_multi(cells) + xi) * self.h

    def locate(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cell of each point, its unit-cell coordinates and whether it lies in the grid."""
        r = (np.asarray(X, float) - self.lo) / self.h
        ij = np.floor(r).astype(np.int64)
        ok = np.all((ij >= 0) & (ij < np.asarray(self.n)), axis=1)
        ij = np.clip(ij, 0, np.asarray(self.n) - 1)
        return np.ravel_multi_index(ij.T, self.n), r - ij, ok

    def cell_nodes(self, cells: np.ndarray) -> np.ndarray:
        """Global lattice node numbers of each cell's local nodes: (m, nb)."""
        base = self.cell_multi(cells) * self.order
        off = basis.local_offsets(self.order, self.d)
        idx = base[:, None, :] + off[None, :, :]
        return np.ravel_multi_index(np.moveaxis(idx, -1, 0), self.node_shape)

    def node_coords(self, nodes: np.ndarray) -> np.ndarray:
        ij = np.stack(np.unravel_index(np.asarray(nodes), self.node_shape), axis=1)
        return self.lo + ij * self.h / self.order


@dataclass
class DofMap:
    """Unknowns of the active cells: ``cell_dofs[i]`` are the dof numbers of active cell ``cells[i]``."""

    grid: BackgroundGrid
    cells: np.ndarray            # active cell ids (sorted)
    cell_dofs: np.ndarray        # (n_active, nb) compact dof numbers
    nodes: np.ndarray            # lattice node of each dof
    index: np.ndarray            # cell id -> row in ``cells`` (or -1)

    @classmethod
    def build(cls, grid: BackgroundGrid, active: np.ndarray) -> "DofMap":
        cells = np.sort(np.asarray(active, np.int64))
        lat = grid.cell_nodes(cells)
        nodes, inv = np.unique(lat, return_inverse=True)
        index = np.full(grid.n_cells, -1, np.int64)
        index[cells] = np.arange(len(cells))
        return cls(grid, cells, inv.reshape(lat.shape), nodes, index)

    @property
    def n_dofs(self) -> int:
        return len(self.nodes)

    def dofs_of(self, cells: np.ndarray) -> np.ndarray:
        return self.cell_dofs[self.index[np.asarray(cells)]]

    def coords(self) -> np.ndarray:
        return self.grid.node_coords(self.nodes)

"""The mesh/result container shared by the CAE readers, the mesh-quality checks, the comparator and the interop export.

A ``Mesh`` holds either finite-element style cells (blocks of one element type with node connectivity, VTK node
ordering, corner nodes first) or an OpenFOAM-style polyhedral mesh (faces with owner/neighbour cells). Both give cell
centres and volumes, a face list for finite-volume metrics and the fields defined on points or cells.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# corner-node count and dimension of each supported element type (VTK names, as meshio uses)
ELEMENTS = {
    "vertex": (1, 0), "line": (2, 1), "line3": (2, 1),
    "triangle": (3, 2), "triangle6": (3, 2), "quad": (4, 2), "quad8": (4, 2), "quad9": (4, 2),
    "tetra": (4, 3), "tetra10": (4, 3), "hexahedron": (8, 3), "hexahedron20": (8, 3), "hexahedron27": (8, 3),
    "wedge": (6, 3), "wedge15": (6, 3), "pyramid": (5, 3), "pyramid13": (5, 3), "polygon": (0, 2),
}
LINEAR = {"line3": "line", "triangle6": "triangle", "quad8": "quad", "quad9": "quad", "tetra10": "tetra",
          "hexahedron20": "hexahedron", "hexahedron27": "hexahedron", "wedge15": "wedge", "pyramid13": "pyramid"}

# faces of each 3D element in VTK corner numbering, ordered so that the right-hand normal points outwards
CELL_FACES = {
    "tetra": [(0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)],
    "hexahedron": [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)],
    "wedge": [(0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)],
    "pyramid": [(0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)],
}
CELL_EDGES = {
    "triangle": [(0, 1), (1, 2), (2, 0)], "quad": [(0, 1), (1, 2), (2, 3), (3, 0)],
    "tetra": [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)],
    "hexahedron": [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)],
    "wedge": [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)],
    "pyramid": [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)],
}


@dataclass
class CellBlock:
    type: str                  # VTK element name (linear or higher order)
    conn: np.ndarray           # (n, nodes) int, VTK ordering
    ids: Optional[np.ndarray] = None   # element numbers in the source file (for messages)

    @property
    def linear(self) -> str:
        return LINEAR.get(self.type, self.type)

    @property
    def corners(self) -> np.ndarray:
        return self.conn[:, : ELEMENTS[self.type][0]] if ELEMENTS.get(self.type, (0,))[0] else self.conn

    @property
    def dim(self) -> int:
        return ELEMENTS.get(self.type, (0, 2))[1]


@dataclass
class PolyMesh:
    """OpenFOAM polyMesh: faces (flat node list + offsets), owner/neighbour, boundary patches."""
    face_nodes: np.ndarray     # flat int array
    face_offsets: np.ndarray   # (n_faces + 1,)
    owner: np.ndarray          # (n_faces,)
    neighbour: np.ndarray      # (n_internal,)
    n_cells: int
    patches: List[Dict[str, Any]]   # {name, type, startFace, nFaces}

    @property
    def n_faces(self) -> int:
        return len(self.owner)

    @property
    def n_internal(self) -> int:
        return len(self.neighbour)


@dataclass
class Mesh:
    points: np.ndarray                                  # (n, 3)
    blocks: List[CellBlock] = field(default_factory=list)   # FE-style cells (volume or surface)
    poly: Optional[PolyMesh] = None                     # finite-volume polyhedral mesh (OpenFOAM)
    boundary_blocks: List[Tuple[str, CellBlock]] = field(default_factory=list)   # named lower-dim groups
    cell_sets: Dict[str, np.ndarray] = field(default_factory=dict)    # name -> global cell indices
    node_sets: Dict[str, np.ndarray] = field(default_factory=dict)
    point_data: Dict[str, np.ndarray] = field(default_factory=dict)
    cell_data: Dict[str, np.ndarray] = field(default_factory=dict)    # one value (row) per cell, global order
    units: Dict[str, str] = field(default_factory=dict)              # field name -> unit
    boundary_values: Dict[str, np.ndarray] = field(default_factory=dict)  # OpenFOAM: value on each boundary face
    source: Dict[str, Any] = field(default_factory=dict)             # format, files, solver, time, notes
    _cache: Dict[str, Any] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ basic properties
    @property
    def kind(self) -> str:
        if self.poly is not None:
            return "polyhedral"
        d = self.dim
        return {3: "volume", 2: "surface" if self.is_surface else "planar", 1: "line"}.get(d, "points")

    @property
    def dim(self) -> int:
        if self.poly is not None:
            return 3
        return max((b.dim for b in self.blocks), default=0)

    @property
    def is_surface(self) -> bool:
        """2D elements that are not all in one plane (a shell or an STL surface)."""
        if self.poly is not None or self.dim != 2:
            return False
        p = self.points - self.points.mean(0)
        if len(p) < 4:
            return False
        s = np.linalg.svd(p, compute_uv=False)
        return bool(s[-1] > 1e-6 * s[0])

    @property
    def n_cells(self) -> int:
        if self.poly is not None:
            return self.poly.n_cells
        return int(sum(len(b.conn) for b in self.blocks))

    @property
    def n_points(self) -> int:
        return len(self.points)

    def bbox(self) -> Dict[str, List[float]]:
        mn, mx = self.points.min(0), self.points.max(0)
        return {"min": mn.tolist(), "max": mx.tolist(), "size": (mx - mn).tolist()}

    def element_counts(self) -> Dict[str, int]:
        if self.poly is not None:
            return self._cache.setdefault("poly_counts", _poly_cell_types(self))
        out: Dict[str, int] = {}
        for b in self.blocks:
            out[b.type] = out.get(b.type, 0) + len(b.conn)
        return out

    def cell_block_index(self, cell: int) -> Tuple[int, int]:
        """(block number, row in block) of a global cell index (FE meshes)."""
        start = 0
        for k, b in enumerate(self.blocks):
            if cell < start + len(b.conn):
                return k, cell - start
            start += len(b.conn)
        raise IndexError(cell)

    def cell_label(self, cell: int) -> str:
        """How the source file names this cell (element number for FE decks, cell index for OpenFOAM)."""
        if self.poly is not None:
            return f"cell {cell}"
        k, r = self.cell_block_index(cell)
        b = self.blocks[k]
        return f"element {int(b.ids[r])}" if b.ids is not None else f"{b.linear} {cell}"


def _poly_cell_types(m: Mesh) -> Dict[str, int]:
    """Classify polyhedral cells by face/node counts like checkMesh (hexahedra, prisms, tets, polyhedra...)."""
    p = m.poly
    nf = np.diff(p.face_offsets)
    faces_per_cell = np.bincount(p.owner, minlength=p.n_cells) + np.bincount(p.neighbour, minlength=p.n_cells)
    tri = np.bincount(p.owner, weights=(nf == 3), minlength=p.n_cells) + \
        np.bincount(p.neighbour, weights=(nf[: p.n_internal] == 3), minlength=p.n_cells)
    quad = np.bincount(p.owner, weights=(nf == 4), minlength=p.n_cells) + \
        np.bincount(p.neighbour, weights=(nf[: p.n_internal] == 4), minlength=p.n_cells)
    out = {"hexahedron": int(np.sum((faces_per_cell == 6) & (quad == 6))),
           "wedge": int(np.sum((faces_per_cell == 5) & (tri == 2) & (quad == 3))),
           "pyramid": int(np.sum((faces_per_cell == 5) & (tri == 4) & (quad == 1))),
           "tetra": int(np.sum((faces_per_cell == 4) & (tri == 4)))}
    out["polyhedron"] = int(p.n_cells - sum(out.values()))
    return {k: v for k, v in out.items() if v}

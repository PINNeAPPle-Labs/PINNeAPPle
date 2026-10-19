"""Export of machine meshes and fields: VTK unstructured grids (ParaView), CSV, HDF5, MATLAB, DXF.

* :func:`write_vtu` -- XML VTK unstructured grid (ASCII) of triangles (2-D) or wedges (3-D) with cell data.
* :func:`solution_to_vtu` -- an :class:`~.pmsm.IPMMachine` solution: regions, A_z (point data), B and |B| (cells).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["write_vtu", "solution_to_vtu"]

_VTK_TYPES = {3: 5, 6: 13}  # triangle, wedge


def _array(name, a, ncomp=1, dtype="Float64"):
    a = np.asarray(a)
    body = " ".join(f"{v:.9g}" for v in a.ravel()) if dtype.startswith("Float") else " ".join(str(int(v)) for v in a.ravel())
    return (f'<DataArray type="{dtype}" Name="{name}" NumberOfComponents="{ncomp}" format="ascii">{body}'
            f"</DataArray>")


def write_vtu(path, points: np.ndarray, cells: np.ndarray, cell_data: dict | None = None,
              point_data: dict | None = None) -> Path:
    """Write an unstructured grid; ``points`` (N, 2 or 3), ``cells`` (M, 3) triangles or (M, 6) wedges; data arrays
    of shape (M,) / (M, c) and (N,) / (N, c)."""
    P = np.asarray(points, float)
    if P.shape[1] == 2:
        P = np.column_stack([P, np.zeros(len(P))])
    C = np.asarray(cells, np.int64)
    vtype = _VTK_TYPES[C.shape[1]]
    out = ['<?xml version="1.0"?>', '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">',
           "<UnstructuredGrid>", f'<Piece NumberOfPoints="{len(P)}" NumberOfCells="{len(C)}">',
           "<Points>", _array("Points", P, 3), "</Points>", "<Cells>",
           _array("connectivity", C, 1, "Int64"), _array("offsets", np.arange(1, len(C) + 1) * C.shape[1], 1, "Int64"),
           _array("types", np.full(len(C), vtype), 1, "UInt8"), "</Cells>"]
    for tag, data in (("CellData", cell_data), ("PointData", point_data)):
        if data:
            out.append(f"<{tag}>")
            for k, v in data.items():
                v = np.asarray(v)
                out.append(_array(k, v, 1 if v.ndim == 1 else v.shape[1],
                                  "Int64" if np.issubdtype(v.dtype, np.integer) else "Float64"))
            out.append(f"</{tag}>")
    out += ["</Piece>", "</UnstructuredGrid>", "</VTKFile>"]
    path = Path(path)
    path.write_text("\n".join(out))
    return path


def solution_to_vtu(path, ms) -> Path:
    """A 2-D machine solution (``MachineSolution``): region index and B, |B| per cell, A_z per node."""
    B = np.column_stack([ms.B, np.zeros(len(ms.B))])
    return write_vtu(path, ms.mesh.points, ms.mesh.tris,
                     {"region": ms.mesh.region, "B": B, "Bmag": np.hypot(ms.B[:, 0], ms.B[:, 1])},
                     {"A_z": ms.A})

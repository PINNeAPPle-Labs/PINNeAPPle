"""Optional meshio bridge for solver-native mesh files.

This handles mesh geometry/connectivity; it does not claim to decode
OpenFOAM time directories or CalculiX FRD solution data. Those require
separate native-field readers and scientific validation.
"""
from __future__ import annotations

from pathlib import Path
import math

from .mesh_export import MeshDataset, MeshField

SUPPORTED = {"vertex", "line", "triangle", "quad", "tetra", "hexahedron",
             "wedge", "pyramid"}


def read_meshio(path: str | Path, *, coordinate_scale_to_m: float = 1.0,
                field_units: dict[str, str] | None = None,
                provenance: dict[str, str] | None = None) -> MeshDataset:
    if not (math.isfinite(coordinate_scale_to_m) and coordinate_scale_to_m > 0):
        raise ValueError("Explicit finite positive coordinate scale required")
    try:
        import meshio
    except ImportError as exc:
        raise RuntimeError("Install optional dependency meshio to read solver-native mesh files") from exc
    mesh = meshio.read(str(path))
    points = tuple(tuple(float(x)*coordinate_scale_to_m for x in row)
                   for row in mesh.points)
    cells = []
    for block in mesh.cells:
        if block.type not in SUPPORTED:
            raise ValueError(f"Unsupported topology {block.type}; refusing to drop cells")
        cells.extend((block.type, tuple(int(i) for i in row)) for row in block.data)
    units = field_units or {}
    fields = {}
    for name, values in mesh.point_data.items():
        if name not in units:
            raise ValueError(f"Missing explicit physical units for point field {name}")
        rows = values.reshape((len(points), -1))
        fields[name] = MeshField(tuple(tuple(float(v) for v in row) for row in rows),
                                 units[name], "point", tuple(f"c{i}" for i in range(rows.shape[1])))
    for name, blocks in mesh.cell_data.items():
        if name not in units:
            raise ValueError(f"Missing explicit physical units for cell field {name}")
        rows = [row for block in blocks for row in block.reshape((len(block), -1))]
        if name in fields:
            raise ValueError(f"Field {name} exists in both point and cell data; rename explicitly")
        if rows:
            fields[name] = MeshField(tuple(tuple(float(v) for v in row) for row in rows),
                                     units[name], "cell", tuple(f"c{i}" for i in range(len(rows[0]))))
    return MeshDataset(points_m=points, cells=tuple(cells), fields=fields,
                       provenance=dict(provenance or {}, source_path=str(path)))

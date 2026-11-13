"""Topology-preserving mesh field export (VTK legacy UNSTRUCTURED_GRID and NPZ).

This module only exports already normalized geometry/fields. It never
silently infers connectivity, translates solver units, or interpolates data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math

VTK_CELL_TYPES = {"vertex": (1, 1), "line": (3, 2), "triangle": (5, 3),
                  "quad": (9, 4), "tetra": (10, 4), "hexahedron": (12, 8),
                  "wedge": (13, 6), "pyramid": (14, 5)}


@dataclass(frozen=True)
class MeshField:
    values: tuple[tuple[float, ...], ...]
    units: str
    location: str  # point or cell
    components: tuple[str, ...]


@dataclass(frozen=True)
class MeshDataset:
    points_m: tuple[tuple[float, ...], ...]
    cells: tuple[tuple[str, tuple[int, ...]], ...]
    fields: dict[str, MeshField]
    provenance: dict[str, str]

    def __post_init__(self):
        if not self.points_m or any(len(p) not in (2, 3) for p in self.points_m):
            raise ValueError("Mesh requires 2D or 3D points")
        if len({len(p) for p in self.points_m}) != 1:
            raise ValueError("Mixed point dimensions")
        if any(not math.isfinite(x) for p in self.points_m for x in p):
            raise ValueError("Nonfinite coordinates")
        for kind, nodes in self.cells:
            if kind not in VTK_CELL_TYPES or len(nodes) != VTK_CELL_TYPES[kind][1]:
                raise ValueError("Unsupported cell type or connectivity")
            if any(i < 0 or i >= len(self.points_m) for i in nodes):
                raise ValueError("Cell index outside mesh")
        for name, field in self.fields.items():
            if not name or not field.units or not field.components:
                raise ValueError("Missing field metadata")
            if field.location not in ("point", "cell"):
                raise ValueError("Field location must be point or cell")
            n = len(self.points_m) if field.location == "point" else len(self.cells)
            if len(field.values) != n:
                raise ValueError("Field sample count differs from mesh topology")
            if any(len(v) != len(field.components) for v in field.values):
                raise ValueError("Field component count mismatch")
            if any(not math.isfinite(x) for row in field.values for x in row):
                raise ValueError("Nonfinite field values")


def export_mesh(dataset: MeshDataset, path: str | Path, *, format: str = "vtk") -> dict[str, str]:
    if format not in ("vtk", "structured_array"):
        raise ValueError("Supported formats: vtk, structured_array")
    path = Path(path).with_suffix(".vtk" if format == "vtk" else ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    if format == "vtk":
        lines = ["# vtk DataFile Version 3.0", "PINNeAPPle mesh", "ASCII",
                 "DATASET UNSTRUCTURED_GRID", f"POINTS {len(dataset.points_m)} double"]
        for p in dataset.points_m:
            lines.append(" ".join(f"{x:.17g}" for x in (*p, *((0.,) * (3-len(p))))))
        lines.append(f"CELLS {len(dataset.cells)} {sum(len(c)+1 for _, c in dataset.cells)}")
        lines.extend(" ".join(map(str, (len(nodes), *nodes))) for _, nodes in dataset.cells)
        lines.append(f"CELL_TYPES {len(dataset.cells)}")
        lines.extend(str(VTK_CELL_TYPES[kind][0]) for kind, _ in dataset.cells)
        for location, count in (("point", len(dataset.points_m)), ("cell", len(dataset.cells))):
            selected = [(name, f) for name, f in dataset.fields.items() if f.location == location]
            if not selected:
                continue
            lines.extend((f"{'POINT' if location == 'point' else 'CELL'}_DATA {count}",
                          f"FIELD FieldData {len(selected)}"))
            for name, f in selected:
                if not name.replace("_", "").isalnum() or name[0].isdigit():
                    raise ValueError("VTK field name must be alphanumeric or underscore")
                lines.append(f"{name} {len(f.components)} {count} double")
                lines.extend(" ".join(f"{v:.17g}" for v in row) for row in f.values)
        path.write_text("\n".join(lines)+"\n", encoding="ascii")
    else:
        import numpy as np
        arrays = {"points_m": np.asarray(dataset.points_m, dtype="float64"),
                  "cell_offsets": np.cumsum([0]+[len(c) for _, c in dataset.cells], dtype="int64"),
                  "cell_connectivity": np.asarray([i for _, c in dataset.cells for i in c], dtype="int64"),
                  "cell_types": np.asarray([VTK_CELL_TYPES[k][0] for k, _ in dataset.cells], dtype="uint8")}
        for name, f in dataset.fields.items():
            if not name.replace("_", "").isalnum() or name[0].isdigit():
                raise ValueError("Unsafe field name")
            arrays[f"field_{name}"] = np.asarray(f.values, dtype="float64")
        np.savez(path, **arrays)
    metadata = {"schema": "pinneapple.mesh_dataset.v1", "coordinate_units": "m",
                "provenance": dataset.provenance,
                "fields": {name: {"units": f.units, "location": f.location,
                                  "components": list(f.components)} for name, f in dataset.fields.items()}}
    sidecar = path.with_suffix(path.suffix+".metadata.json")
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    return {"data": str(path), "metadata": str(sidecar), "format": format}

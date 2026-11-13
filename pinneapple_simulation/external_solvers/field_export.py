"""Portable exports of solver-neutral sampled physical fields.

Structured NumPy .npy uses a record dtype; legacy VTK is ASCII POLYDATA
with VERTICES (a point cloud, not an inferred CFD/FEM volume mesh).
No interpolation, unit conversion or invented connectivity is performed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .field_contract import PhysicalField


def _safe_identifier(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError("Field/component identifiers must be VTK-safe")
    return name


def _metadata(field: PhysicalField) -> dict:
    protocol = field.protocol
    return {
        "schema": "pinneapple.physical_field.v1",
        "name": field.name,
        "units": field.units,
        "components": list(field.components),
        "reference": {
            "frame": protocol.reference.frame,
            "origin_m": list(protocol.reference.origin_m),
            "axes": list(protocol.reference.axes),
            "location": protocol.reference.location,
            "units": protocol.reference.units,
        },
        "sampling": {
            "field": protocol.field,
            "field_units": protocol.field_units,
            "statistic": protocol.statistic,
            "window_s": list(protocol.window_s),
            "spatial_interval_m": protocol.spatial_interval_m,
            "phase": protocol.phase,
            "phase_alpha_min": protocol.phase_alpha_min,
        },
        "provenance": dict(field.provenance),
    }


def export_field(field: PhysicalField, path: str | Path, *,
                 format: str = "structured_array") -> dict[str, str]:
    """Export an already normalized field, without silently shifting its origin.

    Returns paths to data and sidecar metadata. Structured arrays require
    numpy; legacy VTK needs no additional dependencies.
    """
    path = Path(path)
    if format not in {"structured_array", "vtk"}:
        raise ValueError("Supported formats: structured_array, vtk")
    _safe_identifier(field.name)
    for component in field.components:
        _safe_identifier(component)
    path = path.with_suffix(".npy" if format == "structured_array" else ".vtk")
    path.parent.mkdir(parents=True, exist_ok=True)
    if format == "structured_array":
        import numpy as np

        dtype = [(axis, "<f8") for axis in field.protocol.reference.axes]
        dtype += [(component, "<f8") for component in field.components]
        names = [name for name, _ in dtype]
        if len(set(names)) != len(names):
            raise ValueError("Coordinate axes and component names must be distinct")
        records = np.empty(len(field.values), dtype=dtype)
        for i, (point, values) in enumerate(zip(field.coordinates_m, field.values)):
            records[i] = (*point, *values)
        np.save(path, records, allow_pickle=False)
    else:
        ndim = len(field.protocol.reference.axes)
        if ndim > 3:
            raise ValueError("Legacy VTK supports at most 3 spatial axes")
        n = len(field.values)
        lines = [
            "# vtk DataFile Version 3.0",
            "PINNeAPPle sampled physical field",
            "ASCII",
            "DATASET POLYDATA",
            f"POINTS {n} double",
        ]
        for point in field.coordinates_m:
            lines.append(" ".join(format(v, ".17g") for v in (*point, *((0.0,) * (3 - ndim)))))
        lines.append(f"VERTICES {n} {2 * n}")
        lines.extend(f"1 {i}" for i in range(n))
        lines.append(f"POINT_DATA {n}")
        lines.append(f"FIELD FieldData {len(field.components)}")
        for j, component in enumerate(field.components):
            lines.append(f"{component} 1 {n} double")
            lines.extend(format(values[j], ".17g") for values in field.values)
        path.write_text("\n".join(lines) + "\n", encoding="ascii")
    sidecar = path.with_suffix(path.suffix + ".metadata.json")
    sidecar.write_text(json.dumps(_metadata(field), indent=2, sort_keys=True, allow_nan=False) + "\n",
                       encoding="utf-8")
    return {"data": str(path), "metadata": str(sidecar), "format": format}

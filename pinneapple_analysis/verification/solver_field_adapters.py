"""Read-only adapters for real solver-produced tabular field artifacts.

No solver is executed. CSV columns are explicitly mapped, so units and
physical meaning must be declared by the caller. Never deserialize pickle.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .pipeline_assurance import Check, Status
from .physics_ai_checks import field_parity


@dataclass(frozen=True)
class CSVFieldSpec:
    path: str
    coordinate_columns: tuple[str, ...]
    value_column: str
    unit: str
    coordinate_unit: str
    max_rows: int = 200_000


@dataclass(frozen=True)
class FieldSamples:
    coordinates: tuple[tuple[float, ...], ...]
    values: tuple[float, ...]
    unit: str
    coordinate_unit: str


def read_csv_field(spec: CSVFieldSpec) -> FieldSamples:
    """Read only finite scalar samples; reject ambiguous/missing columns."""
    import math
    if not spec.coordinate_columns or len(set(spec.coordinate_columns)) != len(spec.coordinate_columns):
        raise ValueError("Coordinate columns must be nonempty and unique")
    if spec.value_column in spec.coordinate_columns:
        raise ValueError("Value column must differ from coordinate columns")
    if not spec.unit.strip() or not spec.coordinate_unit.strip():
        raise ValueError("Physical units must be declared")
    if spec.max_rows < 1:
        raise ValueError("max_rows must be positive")
    coordinates, values = [], []
    with Path(spec.path).open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        required = (*spec.coordinate_columns, spec.value_column)
        if len(fields) != len(set(fields)) or any(k not in fields for k in required):
            raise ValueError("Missing or duplicate required CSV columns")
        for row in reader:
            if len(values) >= spec.max_rows:
                raise ValueError("CSV row limit exceeded")
            point = tuple(float(row[k]) for k in spec.coordinate_columns)
            value = float(row[spec.value_column])
            if not all(math.isfinite(v) for v in (*point, value)):
                raise ValueError("Nonfinite coordinate or value")
            coordinates.append(point)
            values.append(value)
    if not values:
        raise ValueError("CSV field is empty")
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("Duplicate coordinates require explicit reduction policy")
    return FieldSamples(tuple(coordinates), tuple(values), spec.unit, spec.coordinate_unit)


def compare_solver_csv(
    reference: CSVFieldSpec, candidate: CSVFieldSpec,
    *, name: str = "solver_field_parity", stage: str = "field_extraction",
    atol: float = 1e-8, rtol: float = 1e-5,
) -> Check:
    """Compare independent CSV exports at identical coordinates.

    No interpolation, unit conversion, or reordering is performed implicitly.
    """
    try:
        a, b = read_csv_field(reference), read_csv_field(candidate)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return Check(name, stage, Status.INCONCLUSIVE, reason=f"CSV input invalid: {exc}")
    if a.unit != b.unit or a.coordinate_unit != b.coordinate_unit:
        return Check(name, stage, Status.FAIL, reason="Undocumented field/coordinate unit mismatch",
                     evidence={"reference_unit": a.unit, "candidate_unit": b.unit,
                               "reference_coordinate_unit": a.coordinate_unit,
                               "candidate_coordinate_unit": b.coordinate_unit})
    if a.coordinates != b.coordinates:
        return Check(name, stage, Status.INCONCLUSIVE,
                     reason="Coordinates/order differ; independent spatial alignment is required",
                     evidence={"reference_count": len(a.values), "candidate_count": len(b.values)})
    return field_parity(name, stage, a.values, b.values, atol=atol, rtol=rtol)

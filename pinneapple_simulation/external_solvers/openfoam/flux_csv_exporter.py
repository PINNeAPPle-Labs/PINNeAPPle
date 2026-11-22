"""Export complete OpenFOAM face fluxes from an explicit solver-produced CSV.

This adapter deliberately does not guess boundary fluxes from volFields.
A separate OpenFOAM-side utility must export the surfaceScalarField with
one row per face. The CSV format is face_id,integrated_flux, where face_id
is the zero-based polyMesh face index. Missing, duplicated and unordered
face IDs are rejected rather than silently repaired.

No solver is executed. No source terms are inferred. All source integrals
must be supplied by the operator as a separate, dimensionally consistent CSV.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from . import binary_reader
from .flux_export_contract import SCHEMA, _digest


def _rows(path: str | Path, header: tuple[str, str], expected: int) -> list[float]:
    p = Path(path)
    if p.is_symlink() or not p.is_file() or p.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("CSV must be a bounded regular file")
    if expected < 1 or expected > 8_000_000:
        raise ValueError("invalid expected row count")
    values = []
    with p.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream, strict=True)
        if tuple(next(reader, ())) != header:
            raise ValueError(f"expected CSV header {header!r}")
        for index, row in enumerate(reader):
            if index >= expected or len(row) != 2 or row[0] != str(index):
                raise ValueError("missing, duplicate, extra or out-of-order face/cell ID")
            value = float(row[1])
            if not math.isfinite(value):
                raise ValueError("nonfinite CSV value")
            values.append(value)
    if len(values) != expected:
        raise ValueError(f"expected {expected} rows, got {len(values)}")
    return values


def export_openfoam_flux_csv(
    case_dir: str | Path, *,
    case_time: str,
    face_flux_csv: str | Path,
    cell_source_csv: str | Path,
    output_json: str | Path,
    unit: str,
    exporter_name: str,
) -> dict:
    """Produce the strict flux contract from complete, explicitly ordered CSVs.

    The face CSV is expected to be produced by a solver-side extraction
    utility, not by this Python function. Its identity remains self-declared.
    """
    root = Path(case_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("case_dir must be a real directory")
    mesh = root / "constant" / "polyMesh"
    owner_path = mesh / "owner"
    neighbour_path = mesh / "neighbour"
    if owner_path.is_symlink() or neighbour_path.is_symlink():
        raise ValueError("symlinked mesh inputs not allowed")
    owner = binary_reader.read_label_list(owner_path.read_bytes(), str(owner_path))
    neighbour = binary_reader.read_label_list(neighbour_path.read_bytes(), str(neighbour_path))
    if owner.ndim != 1 or owner.size == 0 or neighbour.size > owner.size:
        raise ValueError("invalid mesh connectivity")
    n_cells = int(max(owner.max(), neighbour.max() if neighbour.size else -1)) + 1
    if not case_time or not isinstance(case_time, str):
        raise ValueError("case_time required")
    if unit not in ("m3/s", "kg/s", "mol/s", "W", "N", "dimensionless/s"):
        raise ValueError("unsupported physical unit")
    if not exporter_name or not isinstance(exporter_name, str):
        raise ValueError("exporter_name required")
    flux = _rows(face_flux_csv, ("face_id", "integrated_flux"), int(owner.size))
    source = _rows(cell_source_csv, ("cell_id", "integrated_source"), n_cells)
    body = {
        "schema": SCHEMA,
        "case_time": case_time,
        "face_flux": flux,
        "cell_source_integral": source,
        "flux_unit": unit,
        "source_unit": unit,
        "flux_orientation": "owner_to_neighbour_boundary_outward",
        "provenance": {"solver": "OpenFOAM", "exporter": exporter_name,
                       "face_csv": Path(face_flux_csv).name,
                       "source_csv": Path(cell_source_csv).name},
    }
    payload = {**body, "payload_sha256": _digest(body)}
    out = Path(output_json)
    if out.is_symlink() or (out.exists() and not out.is_file()):
        raise ValueError("invalid output path")
    out.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {"path": str(out), "payload_sha256": payload["payload_sha256"],
            "face_count": len(flux), "cell_count": len(source),
            "authentication_status": "unsigned_self_declared_provenance"}

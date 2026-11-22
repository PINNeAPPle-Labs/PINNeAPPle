"""Strict offline contract for OpenFOAM face-integrated flux evidence exports.

This is a data adapter, not a flux extractor. A separate trusted exporter
must provide all faces in polyMesh owner order, including boundary faces.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .conservation_evidence import openfoam_fv_conservation

SCHEMA = "pinneapple.openfoam.fv-flux-export/1"
MAX_BYTES = 64 * 1024 * 1024


def _digest(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def check_openfoam_flux_export(
    case_dir: str | Path,
    export_path: str | Path,
    *,
    absolute_tolerance: float = 1e-8,
    relative_tolerance: float = 1e-6,
) -> dict:
    """Validate a signed-convention, dimensioned, complete offline flux export.

    Required JSON keys: schema, case_time, face_flux, cell_source_integral,
    flux_unit, source_unit, flux_orientation, provenance, payload_sha256.
    The digest detects accidental alteration but is NOT an authenticated
    signature and does not attest the identity of the solver or exporter.
    """
    path = Path(export_path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError("flux export must be a bounded regular JSON file")
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("flux export must be an object")
    required = {"schema", "case_time", "face_flux", "cell_source_integral",
                "flux_unit", "source_unit", "flux_orientation", "provenance",
                "payload_sha256"}
    if set(data) != required or data["schema"] != SCHEMA:
        raise ValueError("unexpected flux export schema or fields")
    if data["flux_orientation"] != "owner_to_neighbour_boundary_outward":
        raise ValueError("face orientation must match owner/neighbour topology")
    if data["flux_unit"] != data["source_unit"] or data["flux_unit"] not in (
        "m3/s", "kg/s", "mol/s", "W", "N", "dimensionless/s"):
        raise ValueError("flux and integrated source units must match an allowed unit")
    if not isinstance(data["case_time"], str) or not data["case_time"]:
        raise ValueError("case_time must be nonempty")
    provenance = data["provenance"]
    if (not isinstance(provenance, dict)
            or not isinstance(provenance.get("solver"), str)
            or not provenance["solver"]
            or not isinstance(provenance.get("exporter"), str)
            or not provenance["exporter"]):
        raise ValueError("solver and exporter provenance required")
    if not isinstance(data["face_flux"], list) or not isinstance(data["cell_source_integral"], list):
        raise ValueError("flux and source arrays must be lists")
    flux = np.asarray(data["face_flux"], dtype=np.float64)
    source = np.asarray(data["cell_source_integral"], dtype=np.float64)
    if (flux.ndim != 1 or source.ndim != 1
            or not np.isfinite(flux).all() or not np.isfinite(source).all()):
        raise ValueError("flux and source must be finite 1D arrays")
    body = {k: v for k, v in data.items() if k != "payload_sha256"}
    expected = _digest(body)
    import hmac
    if (not isinstance(data["payload_sha256"], str)
            or not hmac.compare_digest(data["payload_sha256"], expected)):
        raise ValueError("flux export digest mismatch")
    result = openfoam_fv_conservation(
        case_dir, integrated_face_flux=flux,
        integrated_cell_sources=source,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance)
    return {**result, "flux_export_schema": SCHEMA,
            "flux_export_sha256": expected,
            "flux_unit": data["flux_unit"],
            "case_time": data["case_time"],
            "provenance": provenance,
            "authentication_status": "unsigned_self_declared_provenance"}

"""Fail-closed offline OpenFOAM ASCII phi -> FV balance evidence pipeline.

The source CSV is an independent input. Neither cell sources nor solver
validation are inferred from phi. OpenFOAM is never executed here.
"""
from __future__ import annotations

import json
from pathlib import Path

from .ascii_surface_flux import extract_ascii_surface_scalar_flux
from .flux_csv_exporter import export_openfoam_flux_csv
from .flux_export_contract import check_openfoam_flux_export

# OpenFOAM SI dimension exponents: [mass length time temperature moles current luminous]
FLUX_DIMENSIONS = {
    "m3/s": (0, 3, -1, 0, 0, 0, 0),
    "kg/s": (1, 0, -1, 0, 0, 0, 0),
    "mol/s": (0, 0, -1, 0, 1, 0, 0),
    "W": (1, 2, -3, 0, 0, 0, 0),
    "N": (1, 1, -2, 0, 0, 0, 0),
    "dimensionless/s": (0, 0, -1, 0, 0, 0, 0),
}


def evaluate_ascii_openfoam_conservation(
    case_dir: str | Path, *,
    time: str,
    cell_source_csv: str | Path,
    output_dir: str | Path,
    unit: str,
    field: str = "phi",
    absolute_tolerance: float = 1e-8,
    relative_tolerance: float = 1e-6,
) -> dict:
    """Extract face flux, construct contract and calculate FV numeric balance.

    The caller must supply an independently derived, face-consistent cell
    source integral CSV; zeros are never assumed.
    """
    if unit not in FLUX_DIMENSIONS:
        raise ValueError("unsupported flux unit")
    out = Path(output_dir)
    if out.is_symlink() or not out.is_dir():
        raise ValueError("output_dir must be an existing real directory")
    flux_csv = out / "openfoam_face_flux.csv"
    evidence_json = out / "openfoam_flux_evidence.json"
    if flux_csv.is_symlink() or evidence_json.is_symlink():
        raise ValueError("symlinked output files are forbidden")
    extraction = extract_ascii_surface_scalar_flux(
        case_dir, time=time, field=field, output_csv=flux_csv)
    if tuple(extraction["dimensions"]) != FLUX_DIMENSIONS[unit]:
        raise ValueError(
            f"OpenFOAM dimensions {extraction['dimensions']} incompatible with {unit}")
    exported = export_openfoam_flux_csv(
        case_dir, case_time=time, face_flux_csv=flux_csv,
        cell_source_csv=cell_source_csv, output_json=evidence_json,
        unit=unit, exporter_name="PINNeAPPle ASCII surfaceScalarField reader")
    balance = check_openfoam_flux_export(
        case_dir, evidence_json,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance)
    return {
        "extraction": extraction,
        "export": exported,
        "conservation": balance,
        "scientific_status": "numeric_balance_checked_not_independently_validated",
        "solver_executed": False,
        "independent_validation": False,
    }

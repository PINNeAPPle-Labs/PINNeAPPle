"""Generate an OpenFOAM-native surfaceScalarField extraction script.

The generated script runs inside an OpenFOAM Python environment with
PyFoam? No: it uses the OpenFOAM command-line post-processing tool to
write face flux values to an explicit ASCII CSV via a user-installed
utility. This module does not assume that such a utility is installed.
"""
from __future__ import annotations

import json
from pathlib import Path


def write_openfoam_flux_extraction_plan(
    case_dir: str | Path,
    *,
    time: str,
    surface_field: str = "phi",
    output_path: str | Path,
    flux_unit: str = "m3/s",
) -> dict:
    """Write a reproducible, non-executing solver-side extraction plan.

    A plan alone is not a flux export: its commands must be adapted to the
    installed OpenFOAM version and validated against boundary patch order.
    """
    root = Path(case_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("case_dir must be an existing directory")
    if not isinstance(time, str) or not time or "/" in time or "\\" in time:
        raise ValueError("time must be a single OpenFOAM time directory name")
    if not isinstance(surface_field, str) or not surface_field.isidentifier():
        raise ValueError("surface_field must be a valid field identifier")
    if flux_unit not in ("m3/s", "kg/s", "mol/s", "W", "N", "dimensionless/s"):
        raise ValueError("unsupported flux unit")
    out = Path(output_path)
    if out.is_symlink() or (out.exists() and not out.is_file()):
        raise ValueError("invalid output path")
    plan = {
        "schema": "pinneapple.openfoam.face-flux-extraction-plan/1",
        "case_directory": str(root.resolve()),
        "time": time,
        "surface_field": surface_field,
        "flux_unit": flux_unit,
        "required_inputs": [
            "constant/polyMesh/owner",
            "constant/polyMesh/neighbour",
            f"{time}/{surface_field}",
        ],
        "required_export": {
            "face_csv_header": ["face_id", "integrated_flux"],
            "face_order": "polyMesh_internal_faces_then_boundary_patches_in_mesh_order",
            "boundary_faces": "mandatory",
            "cell_source_csv_header": ["cell_id", "integrated_source"],
            "cell_sources": "must be obtained independently from solver equation terms",
        },
        "execution_status": "not_executed",
        "scientific_status": "plan_only_no_solver_evidence",
    }
    out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan

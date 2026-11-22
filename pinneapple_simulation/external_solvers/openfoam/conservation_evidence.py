"""Read OpenFOAM polyMesh incidence for independent FV balance checks.

The caller MUST supply all integrated face fluxes, including boundary
patch faces, and all integrated cell source terms in OpenFOAM face order.
OpenFOAM volField internal values are not interchangeable with face fluxes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from pinneapple_physics.validation import check_unstructured_fv_conservation

from . import binary_reader


def openfoam_fv_conservation(
    case_dir: str | Path,
    *,
    integrated_face_flux,
    integrated_cell_sources,
    boundary_flux_targets=None,
    absolute_tolerance: float = 1e-8,
    relative_tolerance: float = 1e-6,
) -> dict:
    """Validate owner/neighbour incidence against a complete solver face-flux export.

    polyMesh/neighbour lists *internal* faces only. The remaining owner
    entries are boundary faces, with neighbour=-1 in this adapter.
    This function does not execute OpenFOAM or parse boundaryField values.
    """
    root = Path(case_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("case_dir must be an existing real directory")
    mesh = root / "constant" / "polyMesh"
    for name in ("owner", "neighbour"):
        if (mesh / name).is_symlink():
            raise ValueError("symlinked mesh files are not accepted")
    owner = np.asarray(binary_reader.read_label_list(
        (mesh / "owner").read_bytes(), str(mesh / "owner")), dtype=np.int64)
    interior = np.asarray(binary_reader.read_label_list(
        (mesh / "neighbour").read_bytes(), str(mesh / "neighbour")), dtype=np.int64)
    if owner.ndim != 1 or interior.ndim != 1 or interior.size > owner.size:
        raise ValueError("invalid OpenFOAM polyMesh owner/neighbour lengths")
    if interior.size and (np.any(interior < 0) or np.any(interior == owner[:interior.size])):
        raise ValueError("invalid internal face neighbours")
    neighbour = np.full(owner.size, -1, dtype=np.int64)
    neighbour[:interior.size] = interior
    result = check_unstructured_fv_conservation(
        owner, neighbour, integrated_face_flux, integrated_cell_sources,
        boundary_flux_targets=boundary_flux_targets,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance)
    return {**result, "source": "openfoam.polyMesh.owner-neighbour",
            "flux_provenance": "caller_supplied_integrated_face_flux",
            "boundary_flux_coverage": "required_all_faces",
            "solver_executed": False}

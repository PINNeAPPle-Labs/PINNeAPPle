"""ASCII OpenFOAM surface field extraction tests (not executed)."""
from unittest.mock import patch

import numpy as np
import pytest

from pinneapple_simulation.external_solvers.openfoam.ascii_surface_flux import (
    extract_ascii_surface_scalar_flux,
)


def _case(tmp_path, *, missing_patch=False):
    mesh = tmp_path / "constant" / "polyMesh"
    mesh.mkdir(parents=True)
    (mesh / "owner").write_bytes(b"owner")
    (mesh / "neighbour").write_bytes(b"neighbour")
    (mesh / "boundary").write_text(
        'FoamFile { class polyBoundaryMesh; format ascii; }\n'
        '2\n(\ninlet { type patch; nFaces 1; startFace 1; }\n'
        'outlet { type patch; nFaces 1; startFace 2; }\n)\n')
    tdir = tmp_path / "10"
    tdir.mkdir()
    outlet = "" if missing_patch else "outlet { type patch; value uniform 2; }"
    (tdir / "phi").write_text(
        'FoamFile { class surfaceScalarField; format ascii; }\n'
        'dimensions [0 3 -1 0 0 0 0];\n'
        'internalField nonuniform List<scalar> 1(2);\n'
        'boundaryField { inlet { type patch; value uniform -2; } '
        + outlet + ' }\n')


def test_extract_complete_face_order(tmp_path):
    _case(tmp_path)
    with patch(
        "pinneapple_simulation.external_solvers.openfoam.ascii_surface_flux.binary_reader.read_label_list",
        side_effect=[np.array([0, 0, 1]), np.array([1])],
    ):
        result = extract_ascii_surface_scalar_flux(
            tmp_path, time="10", output_csv=tmp_path / "phi.csv")
    assert result["face_count"] == 3
    assert (tmp_path / "phi.csv").read_text().splitlines()[1:] == [
        "0,2", "1,-2", "2,2"]


def test_missing_boundary_patch_rejected(tmp_path):
    _case(tmp_path, missing_patch=True)
    with patch(
        "pinneapple_simulation.external_solvers.openfoam.ascii_surface_flux.binary_reader.read_label_list",
        side_effect=[np.array([0, 0, 1]), np.array([1])],
    ):
        with pytest.raises(ValueError, match="patch names mismatch"):
            extract_ascii_surface_scalar_flux(
                tmp_path, time="10", output_csv=tmp_path / "phi.csv")

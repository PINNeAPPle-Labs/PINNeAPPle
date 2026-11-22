"""OpenFOAM conservation adapter contracts; tests are not executed."""
from unittest.mock import patch

import numpy as np
import pytest

from pinneapple_simulation.external_solvers.openfoam.conservation_evidence import (
    openfoam_fv_conservation,
)


def test_reads_owner_neighbour_and_checks_balance(tmp_path):
    mesh = tmp_path / "constant" / "polyMesh"
    mesh.mkdir(parents=True)
    (mesh / "owner").write_bytes(b"owner")
    (mesh / "neighbour").write_bytes(b"neighbour")

    def fake_read(data, _hint):
        return np.array([0, 0, 1], dtype=np.int32) if data == b"owner" else np.array([1], dtype=np.int32)

    with patch("pinneapple_simulation.external_solvers.openfoam.conservation_evidence.binary_reader.read_label_list",
               side_effect=fake_read):
        # Interior +2 owner0->cell1, boundary -2 owner0, boundary +2 owner1
        result = openfoam_fv_conservation(
            tmp_path, integrated_face_flux=[2., -2., 2.],
            integrated_cell_sources=[0., 0.],
            boundary_flux_targets=[-2., 2.])
    assert result["passed"]
    assert result["boundary_face_count"] == 2
    assert not result["solver_executed"]


def test_missing_boundary_flux_is_not_inferred(tmp_path):
    mesh = tmp_path / "constant" / "polyMesh"
    mesh.mkdir(parents=True)
    (mesh / "owner").write_bytes(b"owner")
    (mesh / "neighbour").write_bytes(b"neighbour")
    with patch("pinneapple_simulation.external_solvers.openfoam.conservation_evidence.binary_reader.read_label_list",
               side_effect=[np.array([0, 0]), np.array([1])]):
        with pytest.raises(ValueError):
            openfoam_fv_conservation(
                tmp_path, integrated_face_flux=[1.],
                integrated_cell_sources=[0., 0.])

"""Offline CSV exporter contracts; tests written but not executed."""
from unittest.mock import patch

import numpy as np
import pytest

from pinneapple_simulation.external_solvers.openfoam.flux_csv_exporter import (
    export_openfoam_flux_csv,
)


def _fixture(tmp_path):
    mesh = tmp_path / "constant" / "polyMesh"
    mesh.mkdir(parents=True)
    (mesh / "owner").write_bytes(b"owner")
    (mesh / "neighbour").write_bytes(b"neighbour")
    flux = tmp_path / "flux.csv"
    flux.write_text("face_id,integrated_flux\n0,2\n1,-2\n2,2\n")
    source = tmp_path / "source.csv"
    source.write_text("cell_id,integrated_source\n0,0\n1,0\n")
    return flux, source


def test_export_all_faces_and_cells(tmp_path):
    flux, source = _fixture(tmp_path)
    with patch(
        "pinneapple_simulation.external_solvers.openfoam.flux_csv_exporter.binary_reader.read_label_list",
        side_effect=[np.array([0, 0, 1]), np.array([1])],
    ):
        result = export_openfoam_flux_csv(
            tmp_path, case_time="10", face_flux_csv=flux,
            cell_source_csv=source, output_json=tmp_path / "evidence.json",
            unit="kg/s", exporter_name="OpenFOAM-face-export-v1")
    assert result["face_count"] == 3
    assert result["cell_count"] == 2


def test_missing_boundary_face_fails_closed(tmp_path):
    flux, source = _fixture(tmp_path)
    flux.write_text("face_id,integrated_flux\n0,2\n1,-2\n")
    with patch(
        "pinneapple_simulation.external_solvers.openfoam.flux_csv_exporter.binary_reader.read_label_list",
        side_effect=[np.array([0, 0, 1]), np.array([1])],
    ):
        with pytest.raises(ValueError, match="expected 3"):
            export_openfoam_flux_csv(
                tmp_path, case_time="10", face_flux_csv=flux,
                cell_source_csv=source, output_json=tmp_path / "evidence.json",
                unit="kg/s", exporter_name="test")


def test_duplicate_face_fails_closed(tmp_path):
    flux, source = _fixture(tmp_path)
    flux.write_text("face_id,integrated_flux\n0,2\n0,-2\n2,2\n")
    with patch(
        "pinneapple_simulation.external_solvers.openfoam.flux_csv_exporter.binary_reader.read_label_list",
        side_effect=[np.array([0, 0, 1]), np.array([1])],
    ):
        with pytest.raises(ValueError, match="out-of-order"):
            export_openfoam_flux_csv(
                tmp_path, case_time="10", face_flux_csv=flux,
                cell_source_csv=source, output_json=tmp_path / "evidence.json",
                unit="kg/s", exporter_name="test")

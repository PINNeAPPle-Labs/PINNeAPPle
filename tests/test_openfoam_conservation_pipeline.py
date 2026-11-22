"""Offline OpenFOAM evidence pipeline tests; not executed."""
from unittest.mock import patch

import pytest

from pinneapple_simulation.external_solvers.openfoam.conservation_pipeline import (
    evaluate_ascii_openfoam_conservation,
)


def test_end_to_end_contract_wiring(tmp_path):
    with patch(
        "pinneapple_simulation.external_solvers.openfoam.conservation_pipeline.extract_ascii_surface_scalar_flux",
        return_value={"dimensions": (0, 3, -1, 0, 0, 0, 0), "face_count": 3},
    ), patch(
        "pinneapple_simulation.external_solvers.openfoam.conservation_pipeline.export_openfoam_flux_csv",
        return_value={"payload_sha256": "abc"},
    ) as exporter, patch(
        "pinneapple_simulation.external_solvers.openfoam.conservation_pipeline.check_openfoam_flux_export",
        return_value={"passed": True},
    ) as checker:
        result = evaluate_ascii_openfoam_conservation(
            tmp_path, time="10", cell_source_csv=tmp_path / "sources.csv",
            output_dir=tmp_path, unit="m3/s")
    assert result["conservation"]["passed"]
    assert result["solver_executed"] is False
    assert result["independent_validation"] is False
    assert exporter.called and checker.called


def test_dimensions_mismatch_fails_before_evidence_export(tmp_path):
    with patch(
        "pinneapple_simulation.external_solvers.openfoam.conservation_pipeline.extract_ascii_surface_scalar_flux",
        return_value={"dimensions": (1, 0, -1, 0, 0, 0, 0)},
    ), patch(
        "pinneapple_simulation.external_solvers.openfoam.conservation_pipeline.export_openfoam_flux_csv",
    ) as exporter:
        with pytest.raises(ValueError, match="incompatible"):
            evaluate_ascii_openfoam_conservation(
                tmp_path, time="10", cell_source_csv=tmp_path / "sources.csv",
                output_dir=tmp_path, unit="m3/s")
    exporter.assert_not_called()

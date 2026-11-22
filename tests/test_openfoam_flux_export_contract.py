"""Offline OpenFOAM flux contract tests; not executed."""
import hashlib
import json
from unittest.mock import patch

import numpy as np
import pytest

from pinneapple_simulation.external_solvers.openfoam.flux_export_contract import (
    SCHEMA, check_openfoam_flux_export,
)


def _write(tmp_path, **updates):
    payload = {
        "schema": SCHEMA, "case_time": "10",
        "face_flux": [2., -2., 2.],
        "cell_source_integral": [0., 0.],
        "flux_unit": "kg/s", "source_unit": "kg/s",
        "flux_orientation": "owner_to_neighbour_boundary_outward",
        "provenance": {"solver": "OpenFOAM", "exporter": "trusted-export-script"},
    }
    payload.update(updates)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()
    payload["payload_sha256"] = digest
    path = tmp_path / "flux.json"
    path.write_text(json.dumps(payload))
    return path


def test_complete_export_passes_numeric_balance(tmp_path):
    path = _write(tmp_path)
    with patch(
        "pinneapple_simulation.external_solvers.openfoam.flux_export_contract.openfoam_fv_conservation",
        return_value={"passed": True, "input_sha256": "fixture"},
    ) as checker:
        result = check_openfoam_flux_export(tmp_path, path)
    assert result["passed"]
    assert result["authentication_status"] == "unsigned_self_declared_provenance"
    assert checker.call_args.kwargs["integrated_face_flux"].shape == (3,)


def test_tampering_is_rejected(tmp_path):
    path = _write(tmp_path)
    payload = json.loads(path.read_text())
    payload["face_flux"][0] = 100.
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest mismatch"):
        check_openfoam_flux_export(tmp_path, path)


def test_mismatched_units_rejected(tmp_path):
    path = _write(tmp_path, source_unit="W")
    with pytest.raises(ValueError, match="units"):
        check_openfoam_flux_export(tmp_path, path)


def test_wrong_orientation_rejected(tmp_path):
    path = _write(tmp_path, flux_orientation="neighbour_to_owner")
    with pytest.raises(ValueError, match="orientation"):
        check_openfoam_flux_export(tmp_path, path)

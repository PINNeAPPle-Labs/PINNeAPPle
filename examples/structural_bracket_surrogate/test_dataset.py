"""Integration tests for NPZ dataset schema and FEM provenance."""
import json
import numpy as np
from generate_dataset import generate


def test_dataset_schema_and_units(tmp_path):
    path = tmp_path / "sample.npz"
    result = generate(path)
    assert path.exists()
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        assert metadata["source_reproduction"] is False
        assert metadata["units"]["stress"] == "Pa"
        assert data["nodes_m"].shape[1] == 3
        assert data["tetrahedra"].shape[1] == 4
        assert data["displacement_m"].shape == data["nodes_m"].shape
        assert data["stress_pa"].shape == (len(data["tetrahedra"]), 6)
        assert data["von_mises_pa"].shape == (len(data["tetrahedra"]),)
        assert np.isfinite(data["von_mises_pa"]).all()
        assert result["force_balance_n"] < 1e-5

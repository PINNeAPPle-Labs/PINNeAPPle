"""The Well adapter against a file written to the layout its own loader reads.

The fixture reproduces that layout (``dimensions`` with ``spatial_dims``, ``t0/t1_fields`` with ``field_names``,
``boundary_conditions`` with ``bc_type``, ``scalars``) -- downloading a 2.8-21 GB scenario is exactly what the
range-request reader exists to avoid, so the layout is what is tested here.
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest

from pinneapple_data.datasets import the_well
from pinneapple_data.schema import run_gates


def write_well_file(path, *, n=3, nt=9, nx=10, ny=6):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.attrs.update({"dataset_name": "rayleigh_benard", "grid_type": "cartesian", "n_spatial_dims": 2,
                        "n_trajectories": n})
        dims = f.create_group("dimensions")
        dims.attrs["spatial_dims"] = np.array([b"x", b"y"])
        for name, values in (("time", np.linspace(0, 1, nt)), ("x", np.linspace(0, 1, nx)),
                             ("y", np.linspace(0, 1, ny))):
            ds = dims.create_dataset(name, data=np.asarray(values, dtype=np.float32))
            ds.attrs["sample_varying"] = False
        t0 = f.create_group("t0_fields")
        t0.attrs["field_names"] = np.array([b"pressure"])
        t0.create_dataset("pressure", data=rng.random((n, nt, nx, ny), dtype=np.float32)).attrs["time_varying"] = True
        t1 = f.create_group("t1_fields")
        t1.attrs["field_names"] = np.array([b"velocity"])
        t1.create_dataset("velocity", data=rng.random((n, nt, 2, nx, ny), dtype=np.float32)).attrs["time_varying"] = True
        bcs = f.create_group("boundary_conditions")
        for axis, kind in (("x", "periodic"), ("y", "no-slip")):
            ds = bcs.create_dataset(axis, data=np.zeros((n, 2), dtype=np.int32))
            ds.attrs["bc_type"] = kind
        scalars = f.create_group("scalars")
        scalars.attrs["field_names"] = np.array([b"Rayleigh", b"Prandtl"])
        scalars.create_dataset("Rayleigh", data=np.full((n,), 1e10, dtype=np.float32)).attrs["time_varying"] = False
        scalars.create_dataset("Prandtl", data=np.full((n,), 1.0, dtype=np.float32)).attrs["time_varying"] = False
    return path


def test_metadata_is_read_from_the_file(tmp_path):
    path = write_well_file(tmp_path / "rayleigh_benard_Rayleigh_1e10_Prandtl_1.hdf5")
    with the_well.open_the_well(path) as handle:
        assert handle.dataset_name == "rayleigh_benard"
        assert handle.grid_type == "cartesian"
        assert handle.spatial_dims == ["x", "y"]
        assert handle.n_trajectories == 3
        assert [name for name, _ in handle.fields] == ["pressure", "velocity"]
        assert handle.scalar_names == ["Rayleigh", "Prandtl"]


def test_a_trajectory_is_read_with_its_axes_bcs_and_scalars(tmp_path):
    path = write_well_file(tmp_path / "rayleigh_benard_Rayleigh_1e10_Prandtl_1.hdf5")
    with the_well.open_the_well(path) as handle:
        sample = handle.sample(1)
    assert sample.fields["pressure"].shape == (9, 10, 6)
    assert sample.fields["velocity"].shape == (9, 10, 6, 2)        # t1_fields: components to the last axis
    assert set(sample.coordinates) == {"t", "x", "y"}
    assert sample.scalars["Rayleigh"] == pytest.approx(1e10)
    kinds = {(bc["axis"], bc["side"]): bc["bc_type"] for bc in sample.boundary_conditions}
    assert kinds[("x", "low")] == "periodic" and kinds[("y", "high")] == "no-slip"


def test_a_bundle_needs_a_licence_and_says_where_to_find_it(tmp_path):
    path = write_well_file(tmp_path / "rayleigh_benard_Rayleigh_1e10_Prandtl_1.hdf5")
    with the_well.open_the_well(path) as handle:
        with pytest.raises(ValueError, match=r"huggingface\.co/datasets/polymathic-ai/rayleigh_benard"):
            handle.to_bundle(0)
        bundle = handle.to_bundle(0, license="CC BY 4.0")
    assert bundle.manifest.provenance.license == "CC BY 4.0"
    assert bundle.manifest.physics.boundary_conditions
    assert {p.name for p in bundle.manifest.physics.parameters} == {"Rayleigh", "Prandtl"}
    report = run_gates(bundle)
    assert report.get("integrity").passed and report.get("units").passed and report.get("location").passed


def test_the_scenario_card_gives_the_physics(tmp_path):
    path = write_well_file(tmp_path / "rayleigh_benard_Rayleigh_1e10_Prandtl_1.hdf5")
    with the_well.open_the_well(path) as handle:
        bundle = handle.to_bundle(0, license="CC BY 4.0")
    assert "Boussinesq" in bundle.manifest.physics.governing
    assert bundle.manifest.metadata["the_well"]["dataset_name"] == "rayleigh_benard"


def test_scenario_lookup_and_url():
    assert the_well.scenario_hint(".../polymathic-ai/shear_flow/resolve/main/data/train/x.hdf5") == "shear_flow"
    url = the_well.scenario_url("helmholtz_staircase")
    assert url.startswith("https://huggingface.co/datasets/polymathic-ai/helmholtz_staircase/resolve/main/")
    assert the_well.SCENARIOS["rayleigh_benard"].physics


def test_a_trajectory_out_of_range_is_an_index_error(tmp_path):
    path = write_well_file(tmp_path / "rayleigh_benard_Rayleigh_1e10_Prandtl_1.hdf5")
    with the_well.open_the_well(path) as handle:
        with pytest.raises(IndexError):
            handle.sample(7)

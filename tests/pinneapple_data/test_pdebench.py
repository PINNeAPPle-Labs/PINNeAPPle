"""The PDEBench adapter against files written to both of the archive's layouts, plus one live range read.

The local fixtures reproduce the layouts of the published files (root ``tensor``-style datasets, and one group per
sample); :func:`test_reads_a_real_file_over_http` reads an actual DaRUS file without downloading it and is skipped
unless ``PINNEAPPLE_NETWORK_TESTS=1``.
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import pytest

from pinneapple_data.datasets import pdebench
from pinneapple_data.schema import run_gates

NETWORK = os.environ.get("PINNEAPPLE_NETWORK_TESTS") == "1"
needs_network = pytest.mark.skipif(not NETWORK, reason="set PINNEAPPLE_NETWORK_TESTS=1 for live archive reads")


def write_tensor_layout(path, *, n=4, nt=7, nx=9, t_extra=1):
    """The layout of the ``*_Sols_*`` and ``Sod*`` files: fields and coordinates as root datasets."""
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    t = np.linspace(0.0, 1.0, nt + t_extra, dtype=np.float32)
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.create_dataset("density", data=rng.random((n, nt, nx), dtype=np.float32))
        f.create_dataset("pressure", data=rng.random((n, nt, nx), dtype=np.float32))
        f.create_dataset("Vx", data=rng.random((n, nt, nx), dtype=np.float32))
        f.create_dataset("Vy", data=rng.random((n, nt, nx), dtype=np.float32))
        f.create_dataset("x-coordinate", data=x)
        f.create_dataset("t-coordinate", data=t)
        f.create_dataset("nu", data=np.linspace(0.001, 0.01, n, dtype=np.float32))
        f.attrs["case"] = "synthetic 1D"
    return path


def write_group_layout(path, *, n=3, nt=6, nx=8):
    """The layout the paper's own ``data_gen`` writes: one group per sample."""
    rng = np.random.default_rng(1)
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    t = np.linspace(0.0, 1.0, nt, dtype=np.float32)
    with h5py.File(path, "w") as f:
        for i in range(n):
            g = f.create_group(f"{i:04d}")
            g.create_dataset("density", data=rng.random((nt, nx), dtype=np.float32))
            g.create_dataset("velocity", data=rng.random((nt, 1, nx), dtype=np.float32))
            g.create_dataset("x-coordinate", data=x)
            g.create_dataset("t-coordinate", data=t)
    return path


# ── the tensor layout ──────────────────────────────────────────────────────────────────────────────────────
def test_tensor_layout_is_detected_and_read(tmp_path):
    path = write_tensor_layout(tmp_path / "1D_Burgers_Sols_Nu0.01.hdf5")
    with pdebench.open_pdebench(path) as handle:
        assert handle.layout == "tensor"
        assert handle.axis_flags == {"samples": True, "time": True, "component": False}
        assert handle.n_samples == 4
        assert handle.spatial_dims == ["x"]
        assert set(handle.fields) == {"density", "pressure", "Vx", "Vy"}
        sample = handle.sample(2)
    assert set(sample.fields) == {"density", "pressure", "velocity"}
    assert sample.fields["density"].shape == (7, 9)
    assert sample.fields["velocity"].shape == (7, 9, 2)          # Vx and Vy assembled into one vector
    assert sample.parameters["nu"] == pytest.approx(0.007, rel=1e-4)      # linspace(0.001, 0.01, 4)[2]
    assert sample.raw_fields["velocity"] in ("Vx", "Vy")


def test_a_time_coordinate_longer_than_the_data_is_aligned_and_recorded(tmp_path):
    path = write_tensor_layout(tmp_path / "1D_Burgers_Sols_Nu0.01.hdf5", t_extra=1)
    with pdebench.open_pdebench(path) as handle:
        sample = handle.sample(0)
    assert sample.coordinates["t"].shape == (7,)
    assert any("t-coordinate has 8 entries" in note for note in sample.notes)


def test_a_single_trajectory_file_has_no_sample_axis(tmp_path):
    x = np.linspace(-1.0, 1.0, 11, dtype=np.float32)
    t = np.linspace(0.0, 2.0, 6, dtype=np.float32)
    with h5py.File(tmp_path / "Sod6.hdf5", "w") as f:
        f.create_dataset("density", data=np.ones((5, 11), dtype=np.float32))
        f.create_dataset("pressure", data=np.ones((5, 11), dtype=np.float32))
        f.create_dataset("x-coordinate", data=x)
        f.create_dataset("t-coordinate", data=t)
    with pdebench.open_pdebench(tmp_path / "Sod6.hdf5") as handle:
        assert handle.axis_flags == {"samples": False, "time": True, "component": False}
        assert handle.n_samples == 1
        assert handle.sample(0).fields["density"].shape == (5, 11)
        with pytest.raises(IndexError):
            handle.sample(3)


def test_a_bundle_from_a_local_file_passes_the_gates(tmp_path):
    path = write_tensor_layout(tmp_path / "1D_Burgers_Sols_Nu0.01.hdf5")
    bundle = pdebench.load_bundle(path)
    assert bundle.manifest.provenance.license == "CC BY 4.0"
    assert bundle.manifest.provenance.inputs[0].sha256            # the file was hashed
    assert bundle.manifest.physics.governing == "u_t + u u_x = nu u_xx"
    assert "nu" in bundle.manifest.physics.symbols
    report = run_gates(bundle, conserved="density")
    assert report.get("integrity").passed
    assert report.get("provenance").passed
    assert "time_alignment" in [c.check_id for c in bundle.manifest.verification.checks]


def test_the_bundle_keeps_the_raw_field_names_and_the_case(tmp_path):
    path = write_tensor_layout(tmp_path / "1D_Burgers_Sols_Nu0.01.hdf5")
    bundle = pdebench.load_bundle(path)
    meta = bundle.manifest.metadata["pdebench"]
    assert meta["case"] == "1D_Burgers"
    assert meta["raw_fields"]["velocity"] in ("Vx", "Vy")
    assert meta["axis_flags"]["samples"] is True


# ── the group layout ──────────────────────────────────────────────────────────────────────────────────────
def test_group_layout_is_detected_and_read(tmp_path):
    path = write_group_layout(tmp_path / "2D_diff-react_NA_NA.h5")
    with pdebench.open_pdebench(path) as handle:
        assert handle.layout == "groups"
        assert handle.n_samples == 3
        sample = handle.sample(1)
    assert sample.fields["density"].shape == (6, 8)
    assert sample.fields["velocity"].shape == (6, 8)              # the single component is squeezed


def test_a_group_sample_out_of_range_is_an_index_error(tmp_path):
    path = write_group_layout(tmp_path / "2D_diff-react_NA_NA.h5")
    with pdebench.open_pdebench(path) as handle:
        with pytest.raises(IndexError):
            handle.sample(9)


# ── the curated table ─────────────────────────────────────────────────────────────────────────────────────
def test_the_curated_files_carry_a_card():
    for entry in pdebench.DATASETS.values():
        card = pdebench.case_card(entry.filename)
        assert card.family == entry.family, entry.key
        assert card.governing, entry.key


def test_a_url_is_matched_by_its_archive_id_and_a_name_by_its_file_name():
    assert pdebench.case_card(pdebench.darus_url(133150)).family == "Sod"
    assert pdebench.case_card("1D_Burgers_Sols_Nu0.001.hdf5").family == "1D_Burgers"
    assert pdebench.case_card("something_else.h5").family == "unknown"


def test_dataset_lookup_accepts_a_key_a_name_or_a_family():
    assert pdebench.dataset("Sod6").darus_id == 133150
    assert pdebench.dataset("Sod6.hdf5").darus_id == 133150
    assert pdebench.dataset("2D_DarcyFlow").family == "2D_DarcyFlow"
    with pytest.raises(KeyError):
        pdebench.dataset("nope")


def test_the_smallest_curated_file_is_small():
    smallest = min(pdebench.DATASETS.values(), key=lambda d: d.size_mb)
    assert smallest.key == "Sod6" and smallest.size_mb < 10


# ── live archive ──────────────────────────────────────────────────────────────────────────────────────────
@needs_network
def test_reads_a_real_file_over_http():
    """Sod6.hdf5 (4.9 MB) read through range requests: the reader and the real archive, no download."""
    with pdebench.open_pdebench(pdebench.darus_url(133150), max_cache_bytes=8 << 20) as handle:
        assert handle.axis_flags == {"samples": False, "time": True, "component": False}
        sample = handle.sample(0)
        assert sample.fields["density"].shape == (201, 1024)
        assert sample.coordinates["x"][0] == pytest.approx(-0.999, abs=1e-3)
        bundle = handle.to_bundle(0)
    assert bundle.manifest.provenance.inputs[0].role == "remote-sample"
    assert bundle.manifest.provenance.inputs[0].sha256
    assert run_gates(bundle).get("integrity").passed


@needs_network
def test_reads_one_sample_of_a_large_file_without_downloading_it():
    """1.3 GB file, one sample: what crosses the network is what the sample needs."""
    from pinneapple_data.datasets.remote_hdf5 import HTTPRangeFile

    handle = HTTPRangeFile(pdebench.darus_url(133219))            # 2D Darcy, 1.3 GB
    try:
        import h5py
        f = h5py.File(handle, "r", driver="fileobj")
        one = np.asarray(f["tensor"][0])
        f.close()
        assert one.shape == (1, 128, 128)
        assert handle.bytes_read < 200 << 20                       # far below the file size
    finally:
        handle.close()

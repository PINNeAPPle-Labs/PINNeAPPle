"""Compressed persistence for PhysicalSample datasets (opt-in, lossless)."""
import os

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from pinneapple_data.physical_sample import PhysicalSample
from pinneapple_data.serialization import load_pt, save_pt
from pinneapple_data.upd_dataset import UPDDataset


def _samples(n=3, side=32):
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        ds = xr.Dataset(
            {"u": (("x", "y"), rng.random((side, side)).astype("float32"))},
            coords={"x": np.arange(side, dtype=float), "y": np.arange(side, dtype=float)},
        )
        out.append(PhysicalSample(state=ds, schema={"units": {"u": "-"}}, domain={"type": "grid"}))
    return out


def test_compressed_pt_roundtrip_preserves_values(tmp_path):
    samples = _samples()
    plain = str(tmp_path / "plain.pt")
    packed = str(tmp_path / "packed.pt.gz")

    save_pt(samples, plain)
    save_pt(samples, packed, compress=True)

    assert os.path.getsize(packed) > 0
    for path in (plain, packed):
        loaded = load_pt(path)
        assert len(loaded) == len(samples)
        for got, want in zip(loaded, samples):
            np.testing.assert_array_equal(got.state["u"].values, want.state["u"].values)


def test_compressed_is_smaller_for_field_data(tmp_path):
    samples = _samples(n=2, side=128)
    plain = str(tmp_path / "plain.pt")
    packed = str(tmp_path / "packed.pt.gz")

    save_pt(samples, plain)
    save_pt(samples, packed, compress=True)

    assert os.path.getsize(packed) < os.path.getsize(plain)


def test_load_autodetects_compression(tmp_path):
    samples = _samples()
    packed = str(tmp_path / "packed.pt.gz")
    save_pt(samples, packed, compress=True)

    # No format hint: gzip magic bytes drive the decision.
    assert len(load_pt(packed)) == len(samples)


def test_upd_dataset_save_compress_roundtrip(tmp_path):
    ds = UPDDataset(samples=_samples())
    path = str(tmp_path / "ds.pt.gz")

    ds.save(path, compress=True)
    assert os.path.exists(path + ".manifest.json")

    back = UPDDataset.load(path)
    assert len(back) == len(ds)
    np.testing.assert_array_equal(back[0].state["u"].values, ds[0].state["u"].values)


def test_default_save_stays_uncompressed(tmp_path):
    samples = _samples()
    path = str(tmp_path / "plain.pt")
    save_pt(samples, path)

    with open(path, "rb") as fh:
        assert fh.read(2) != b"\x1f\x8b"

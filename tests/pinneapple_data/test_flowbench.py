"""FlowBench: the X/Y tensors of the LDC families and the per-sample FPO files, with their channels named by count.

The fixtures write the layouts the archive stores -- ``(n_samples, n_channels, height, width)`` for X/Y and
``(n_time, height, width, 3)`` for an FPO sample -- because the real files are 1.4 GB each. The channel *counts* are
the ones the published files have (3 inputs and 4 outputs at 128 x 128, checked on the archive while writing this
reader), and the reader names the channels from the paper's Table 12 by that count, refusing a count it cannot name.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_data.datasets import flowbench


def write_xy(root, *, samples=4, height=8, width=10, outputs=4):
    rng = np.random.default_rng(0)
    x = np.zeros((samples, 3, height, width))
    x[:, 0] = 100.0 + np.arange(samples)[:, None, None]              # Re, constant over the grid
    x[:, 1] = rng.random((samples, height, width))                   # s
    x[:, 2] = rng.choice([0.0, 255.0], size=(samples, height, width))  # g, a mask
    y = rng.random((samples, outputs, height, width))
    np.savez(root / "X.npz", data=x)
    np.savez(root / "Y.npz", data=y)
    return root / "X.npz", root / "Y.npz"


def test_the_channels_are_named_from_the_paper_table_by_their_count(tmp_path):
    x_path, y_path = write_xy(tmp_path)
    bundle = flowbench.load_tensors(x_path, y_path, index=0)
    assert set(bundle.manifest.fields.names()) == {"g", "s", "u", "v", "p", "theta"}
    assert [(p.name, p.value) for p in bundle.manifest.physics.parameters] == [("Re", 100.0)]
    assert bundle.field_array("u").shape == (8, 10)
    assert bundle.manifest.metadata["flowbench"]["output_channels"] == ["u", "v", "p", "theta"]


def test_a_three_channel_output_is_the_navier_stokes_family(tmp_path):
    x_path, y_path = write_xy(tmp_path, outputs=3)
    bundle = flowbench.load_tensors(x_path, y_path, index=0)
    assert set(bundle.manifest.fields.names()) == {"g", "s", "u", "v", "p"}
    assert bundle.manifest.metadata["flowbench"]["output_channels"] == ["u", "v", "p"]


def test_a_channel_count_the_paper_does_not_document_is_refused(tmp_path):
    x_path, y_path = write_xy(tmp_path, outputs=5)
    with pytest.raises(ValueError, match="Table 12"):
        flowbench.load_tensors(x_path, y_path, index=0)


def test_the_whole_family_can_be_kept_on_a_sample_axis(tmp_path):
    x_path, y_path = write_xy(tmp_path, samples=4)
    bundle = flowbench.load_tensors(x_path, y_path, index=None, sample_axis=True)
    assert bundle.field_array("u").shape == (4, 8, 10)
    assert bundle.axes == ["sample", "y", "x"]
    assert bundle.manifest.metadata["flowbench"]["samples"] == 4


def test_an_index_outside_the_samples_is_refused(tmp_path):
    x_path, y_path = write_xy(tmp_path, samples=2)
    with pytest.raises(IndexError, match="outside the 2 samples"):
        flowbench.load_tensors(x_path, y_path, index=5)


def test_an_input_channel_that_is_not_constant_cannot_be_the_reynolds_number(tmp_path):
    x_path, y_path = write_xy(tmp_path)
    with np.load(x_path) as archive:
        x = np.asarray(archive["data"])
    x[0, 0, 0, 0] += 1.0                                              # break the "Re is a scalar" contract
    np.savez(x_path, data=x)
    with pytest.raises(ValueError, match="not constant over the grid"):
        flowbench.load_tensors(x_path, y_path, index=0)


def test_the_units_are_unknown_until_the_caller_declares_them(tmp_path):
    from pinneapple_data.schema import run_gates

    x_path, y_path = write_xy(tmp_path)
    assert run_gates(flowbench.load_tensors(x_path, y_path, index=0)).get("units").passed is False
    declared = flowbench.load_tensors(x_path, y_path, index=0,
                                      units={"u": "m/s", "v": "m/s", "p": "Pa", "theta": "K", "g": "1", "s": "m"})
    assert run_gates(declared).get("units").passed is True


def test_an_fpo_sample_keeps_the_time_axis_and_the_channels_last(tmp_path):
    rng = np.random.default_rng(1)
    data = rng.random((6, 12, 20, 3))
    path = tmp_path / "Re_382.npz"
    np.savez(path, data=data)
    bundle = flowbench.load_fpo(path)
    assert bundle.axes == ["t", "y", "x"]
    assert bundle.field_array("velocity").shape == (6, 12, 20, 2)
    assert bundle.field_array("pressure").shape == (6, 12, 20)
    assert [(p.name, p.value) for p in bundle.manifest.physics.parameters] == [("Re", 382.0)]


def test_one_snapshot_of_an_fpo_sample_drops_the_time_axis(tmp_path):
    rng = np.random.default_rng(2)
    path = tmp_path / "Re_100.npz"
    np.savez(path, data=rng.random((6, 12, 20, 3)))
    bundle = flowbench.load_fpo(path, index=2)
    assert bundle.axes == ["y", "x"]
    assert bundle.field_array("pressure").shape == (12, 20)
    with pytest.raises(IndexError, match="outside the 6 snapshots"):
        flowbench.load_fpo(path, index=9)


def test_an_fpo_file_with_the_wrong_channel_position_is_refused(tmp_path):
    path = tmp_path / "Re_1.npz"
    np.savez(path, data=np.zeros((6, 3, 12, 20)))              # channels second: the X/Y layout, not the FPO one
    with pytest.raises(ValueError, match="channels last"):
        flowbench.load_fpo(path)


def test_the_geometry_file_gives_the_obstacle(tmp_path):
    rng = np.random.default_rng(3)
    path = tmp_path / "input_geometry.npz"
    np.savez(path, data=rng.random((12, 20)), mask=rng.integers(0, 2, (12, 20)).astype(np.float32))
    bundle = flowbench.load_geometry(path)
    assert set(bundle.manifest.fields.names()) == {"distance", "mask"}
    assert bundle.field_array("mask").shape == (12, 20)
    assert bundle.manifest.fields.get("mask").unit == "1"


def test_the_licence_is_the_archives_and_it_is_non_commercial():
    assert flowbench.LICENSE == "CC-BY-NC-4.0"
    assert "non-commercial" in flowbench.LICENSE_NOTE
    assert flowbench.TENSORS["ns"] == (("Re", "g", "s"), ("u", "v", "p"))

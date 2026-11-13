"""Operator datasets (INT-008): initial condition -> trajectory, split by trajectory and never by time step."""
from __future__ import annotations

import h5py
import numpy as np
import pytest

from pinneapple_data.datasets.operator import from_bundles, from_pdebench
from pinneapple_data.schema import AxisSpec, Field, Physics, PhysicsDataBundle
from pinneapple_data.schema.models import Parameter

NT, NX = 12, 16


def trajectory(speed: float, label: str = "") -> PhysicsDataBundle:
    x = np.linspace(0.0, 1.0, NX)
    t = np.linspace(0.0, 1.0, NT)
    u = (1.0 + 0.2 * np.sin(2 * np.pi * (x[None, :] - speed * t[:, None]))).astype(np.float32)
    return PhysicsDataBundle.from_arrays(
        {"field:u": u, "axis:t": t, "axis:x": x},
        fields=[Field.from_array("u", u, axes=("t", "x"))],
        axes=[AxisSpec("t", "s"), AxisSpec("x", "m")],
        physics=Physics(parameters=[Parameter(name="speed", value=speed, unit="1")]),
        source=f"trajectory {label or speed}", license="CC0-1.0", kind="synthetic")


def test_the_dataset_is_built_from_the_first_snapshot_and_the_parameters():
    op = from_bundles([trajectory(s, str(i)) for i, s in enumerate((0.1, 0.2, 0.3))], params=("speed",))
    assert op.ic.shape == (3, NX) and op.targets.shape == (3, NT, NX)
    assert op.param_names == ("speed",)
    assert np.allclose(op.params[:, 0], [0.1, 0.2, 0.3])
    assert np.allclose(op.ic[1], op.targets[1, 0])            # the input *is* the first snapshot
    assert "3 trajectories" in op.summary()


def test_channels_are_the_ic_plus_one_constant_channel_per_parameter():
    op = from_bundles([trajectory(s, str(i)) for i, s in enumerate((0.1, 0.2))], params=("speed",))
    channels = op.channels(0)
    assert channels.shape == (2, NX)                          # IC + speed
    assert np.allclose(channels[0], op.ic[0])
    assert np.allclose(channels[1], 0.1)


def test_a_missing_parameter_is_nan_not_zero():
    bundle = trajectory(0.1)
    bundle.manifest.physics.parameters = []
    op = from_bundles([bundle, trajectory(0.2)], params=("speed",))
    assert np.isnan(op.params[0, 0]) and not np.isnan(op.params[1, 0])


def test_the_split_is_by_trajectory_and_says_there_is_no_leakage():
    op = from_bundles([trajectory(0.1 * i, str(i)) for i in range(10)])
    splits = op.split(0.6, 0.2, seed=3)
    assert (len(splits.train), len(splits.val), len(splits.test)) == (6, 2, 2)
    sides = [set(s.trajectories) for s in (splits.train, splits.val, splits.test)]
    assert not (sides[0] & sides[1]) and not (sides[0] & sides[2]) and not (sides[1] & sides[2])
    assert op.leakage(splits)["ok"] is True
    assert "6 train / 2 validation / 2 test" in splits.summary()


def test_a_bad_split_fraction_is_refused():
    op = from_bundles([trajectory(0.1)])
    with pytest.raises(ValueError, match="train \\+ val < 1"):
        op.split(0.9, 0.2)


def test_bundles_of_different_shapes_are_refused():
    small = trajectory(0.1)
    big = PhysicsDataBundle.from_arrays(
        {"field:u": np.zeros((5, 5), np.float32), "axis:t": np.arange(5.0), "axis:x": np.arange(5.0)},
        fields=[Field.from_array("u", np.zeros((5, 5), np.float32), axes=("t", "x"))],
        axes=[AxisSpec("t", "s"), AxisSpec("x", "m")], source="other", license="CC0-1.0", kind="synthetic")
    with pytest.raises(ValueError, match="do not share a shape"):
        from_bundles([small, big])


def test_a_pdebench_file_becomes_an_operator_dataset(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "1D_Burgers_Sols_Nu0.01.hdf5"
    with h5py.File(path, "w") as f:
        f.create_dataset("tensor", data=rng.random((4, 6, 8), dtype=np.float32))
        f.create_dataset("x-coordinate", data=np.linspace(0, 1, 8, dtype=np.float32))
        f.create_dataset("t-coordinate", data=np.linspace(0, 1, 6, dtype=np.float32))
        f.create_dataset("Nu", data=np.full((4,), 0.01, dtype=np.float32))
    op = from_pdebench(path, samples=range(4), params=("Nu",))
    assert op.n == 4 and op.field == "u"
    assert op.targets.shape == (4, 6, 8)
    assert np.allclose(op.params[:, 0], 0.01)
    assert op.license == "CC BY 4.0"

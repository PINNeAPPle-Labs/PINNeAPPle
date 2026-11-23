"""Gates for the reservoir surrogate workflow.

The expensive part of this workflow is a simulation campaign, which no test suite can
run. What is tested here is everything that decides whether the workflow is usable on a
campaign's output: the reader refuses a dataset that cannot be trusted, every
registered operator trains a couple of epochs on a tiny grid without a shape or
convention error (each has its own calling convention, and that is where this class of
bug lives), the metrics are computed on the active cells only, and a checkpoint
round-trips into a model that predicts the same field.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pinneapple_neural.workflows.reservoir_surrogate import (
    ReservoirSurrogateConfig,
    breakthrough_time,
    load_reservoir_surrogate,
    predict_reservoir_field,
    read_reservoir_dataset,
    relative_l2,
    train_reservoir_surrogate,
)

NX, NY, N_TIMES, N_SCENARIOS = 9, 7, 5, 4


def _write_dataset(path, scenarios=N_SCENARIOS, nan_inside=False):
    h5py = pytest.importorskip("h5py")
    generator = np.random.default_rng(3)
    actnum = np.zeros((NX, NY), dtype=np.uint8)
    actnum[1:-1, 1:-1] = 1
    mask = actnum == 1
    params = np.stack([
        np.linspace(2200.0, 5600.0, scenarios),
        np.linspace(1700.0, 4200.0, scenarios),
    ], axis=1)
    times = np.linspace(0.0, 1095.0, N_TIMES)
    swat = np.full((scenarios, N_TIMES, NX, NY), np.nan, dtype=np.float32)
    pressure = np.full_like(swat, np.nan)
    ii, jj = np.where(mask)
    for index in range(scenarios):
        front = (ii / NX) * (0.3 + 0.15 * index)
        for step in range(N_TIMES):
            fraction = times[step] / times[-1]
            swat[index, step][mask] = np.clip(0.2 + front * fraction, 0, 1)
            pressure[index, step][mask] = 300.0 + 40.0 * fraction + 5.0 * index
    if nan_inside:
        swat[0, 0, ii[0], jj[0]] = np.nan
    with h5py.File(path, "w") as handle:
        static = handle.create_group("static")
        static.create_dataset("actnum", data=actnum)
        static.create_dataset("permx", data=np.where(mask, 120.0, 0.0).astype(np.float32))
        static.create_dataset("permy", data=np.where(mask, 90.0, 0.0).astype(np.float32))
        group = handle.create_group("scenarios")
        group.create_dataset("params", data=params, dtype=np.float32)
        group.attrs["params_columns"] = np.array(["RATE_INJ1", "LRAT_NA2"], dtype=object)
        group.create_dataset("tag", data=np.array([f"S{index:03d}" for index in range(scenarios)],
                                                  dtype="S"))
        group.create_dataset("group", data=np.array(["train"] * scenarios, dtype="S"))
        group.create_dataset("inside_design", data=np.ones(scenarios, dtype=bool))
        group.create_dataset("time_days", data=times)
        group.create_dataset("swat", data=swat)
        group.create_dataset("pressure", data=pressure)
    return path


def test_reader_recovers_the_documented_layout(tmp_path):
    path = _write_dataset(tmp_path / "dataset.h5")
    data = read_reservoir_dataset(path)
    assert data.n_scenarios == N_SCENARIOS
    assert data.grid == (NX, NY)
    assert int(data.mask.sum()) == (NX - 2) * (NY - 2)
    assert data.parameter_names == ["RATE_INJ1", "LRAT_NA2"]
    assert data.summary()["groups"] == {"train": N_SCENARIOS}


def test_reader_refuses_nan_inside_the_active_cells(tmp_path):
    path = _write_dataset(tmp_path / "broken.h5", nan_inside=True)
    with pytest.raises(ValueError, match="non-finite values inside the active cells"):
        read_reservoir_dataset(path)


def test_reader_honours_the_requested_horizon(tmp_path):
    path = _write_dataset(tmp_path / "dataset.h5")
    data = read_reservoir_dataset(path, n_times=3)
    assert data.time_days.size == 3
    assert data.swat.shape[1] == 3


@pytest.mark.parametrize(
    "architecture,kind",
    [
        ("hadamard_deeponet", "hadamard"),
        ("chebyshev_branch_trunk", "chebyshev_branch"),
        ("cpno", "point_operator"),
    ],
)
def test_every_operator_trains_and_predicts_a_field(tmp_path, architecture, kind):
    """One convention per architecture; a shape bug here would only show on real data."""
    path = _write_dataset(tmp_path / "dataset.h5")
    config = ReservoirSurrogateConfig(
        architecture=architecture, out_dir=str(tmp_path / "run"), epochs=2, folds=2,
        batch_scenarios=2, max_points=512, learning_rate=1e-3, log_every=0,
    )
    summary = train_reservoir_surrogate(config, path)
    assert summary["kind"] == kind
    assert summary["parameters"] > 0
    assert len(summary["folds"]) == 2
    for fold in summary["folds"]:
        assert fold["scenarios"] == N_SCENARIOS // 2
        assert np.isfinite(fold["rel_l2_saturation"])
        assert np.isfinite(fold["rel_l2_pressure"])
    assert np.isfinite(summary["train"]["rel_l2_saturation"])
    assert summary["history"]["final"][-1] < summary["history"]["final"][0] or len(summary["history"]["final"]) == 1

    model, context = load_reservoir_surrogate(summary["checkpoint"], config, path)
    prediction = predict_reservoir_field(model, context, read_reservoir_dataset(path))
    assert prediction["swat"].shape == (N_SCENARIOS, N_TIMES, NX, NY)
    assert np.isnan(prediction["swat"][:, :, 0, 0]).all(), "inactive cells must stay NaN"
    assert np.isfinite(prediction["swat"][:, :, 2, 2]).all()


def test_holdout_file_is_evaluated_separately(tmp_path):
    path = _write_dataset(tmp_path / "dataset.h5")
    holdout = _write_dataset(tmp_path / "holdout.h5", scenarios=2)
    config = ReservoirSurrogateConfig(architecture="hadamard_deeponet", out_dir=str(tmp_path / "run"),
                                      epochs=1, folds=2, batch_scenarios=1, log_every=0)
    summary = train_reservoir_surrogate(config, path, holdout)
    assert summary["holdout"]["scenarios"] == 2
    assert np.isfinite(summary["holdout"]["rel_l2_saturation"])


def test_unknown_architecture_is_refused_with_the_available_list(tmp_path):
    path = _write_dataset(tmp_path / "dataset.h5")
    config = ReservoirSurrogateConfig(architecture="not_an_operator", out_dir=str(tmp_path / "run"))
    with pytest.raises(ValueError, match="not in the neural-operator registry"):
        train_reservoir_surrogate(config, path)


def test_relative_l2_ignores_padding():
    truth = np.array([[[1.0, 2.0]], [[3.0, np.nan]]])
    prediction = np.array([[[1.0, 2.0]], [[0.0, 5.0]]])
    mask = np.ones_like(truth, dtype=bool)
    mask[1, :, 1] = False
    assert relative_l2(prediction, truth, mask) == pytest.approx(3.0 / np.sqrt(14.0))


def test_breakthrough_is_measured_against_the_cells_own_start():
    times = np.arange(0, 100, 10)
    series = np.array([0.30, 0.30, 0.31, 0.33, 0.40, 0.5, 0.6, 0.7, 0.8, 0.9])
    assert breakthrough_time(series, times) == pytest.approx(30.0 / 365.25)
    assert np.isnan(breakthrough_time(np.full_like(series, 0.5), times))


def test_loss_is_finite_and_masked_on_a_degenerate_field():
    """A target that is NaN almost everywhere must not become NaN or zero-information."""
    from pinneapple_neural.workflows.reservoir_surrogate import _masked_loss

    prediction = torch.zeros(1, 2, 3, 4)
    target = torch.full((1, 2, 3, 4), float("nan"))
    target[0, 0, 0, 0] = 1.0
    assert float(_masked_loss(prediction, target)) == pytest.approx(1.0)

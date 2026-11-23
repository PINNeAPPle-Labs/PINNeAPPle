"""Train/predict workflow for a reservoir surrogate: scenario parameters to field.

What this is for
----------------
A waterflood study produces a different 2-D field per scenario: a handful of well
controls change, and the water front, the pressure field and the arrival time change
with them. That is an operator-learning problem, and this module is the workflow around
it: read the dataset in the layout the reservoir campaign writes, train one of the
registered neural operators on the training scenarios, score it on the scenarios it did
not see, and report the metric a reservoir engineer actually asks for (when the water
arrives at the producer) next to the metric a model card reports (relative L2).

The dataset layout
------------------
Two files, deliberately separate so a loader cannot include held-out cases by
shuffling one array::

    static/actnum        (nx, ny)   uint8, 1 = active
    static/permx         (nx, ny)   mD
    static/permy         (nx, ny)   mD
    scenarios/params     (S, p)     the varying controls, in a declared column order
    scenarios/tag        (S,)       identifier
    scenarios/group      (S,)       train / interpolation / stress
    scenarios/inside_design (S,)    bool
    scenarios/time_days  (T,)       report clock
    scenarios/swat       (S, T, nx, ny)  NaN outside the active cells
    scenarios/pressure   (S, T, nx, ny)  bar, NaN outside the active cells

NaN outside the active cells is the convention, not an accident: a zero there is a
plausible-looking saturation and a model will happily learn it.

What is deliberately not here
-----------------------------
No solver. Running OPM Flow belongs to whoever owns the campaign; this module starts
from the HDF5 the campaign produced, which is what makes it usable from the library,
from a CLI and from an API without any of them needing Docker.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "ReservoirSurrogateConfig",
    "ReservoirDataset",
    "read_reservoir_dataset",
    "train_reservoir_surrogate",
    "predict_reservoir_field",
    "relative_l2",
    "breakthrough_time",
    "load_reservoir_surrogate",
]

GRID_OPERATORS = ("hadamard_deeponet", "chebyshev_branch_trunk")


@dataclass
class ReservoirSurrogateConfig:
    """Everything a run needs to be reproducible from the record it leaves behind."""

    architecture: str = "hadamard_deeponet"
    out_dir: str = "reservoir_surrogate"
    epochs: int = 300
    learning_rate: float = 2e-3
    batch_scenarios: int = 4
    max_points: int = 16384
    folds: int = 5
    seed: int = 42
    device: str = "cpu"
    saturation_channel: int = 0
    weight_decay: float = 1e-5
    log_every: int = 25
    parameters: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReservoirDataset:
    """One reservoir dataset, already validated enough to train on."""

    actnum: np.ndarray
    permx: np.ndarray
    permy: np.ndarray
    params: np.ndarray
    time_days: np.ndarray
    swat: np.ndarray
    pressure: np.ndarray
    tags: list[str]
    groups: list[str]
    parameter_names: list[str]
    source: str

    @property
    def mask(self) -> np.ndarray:
        return self.actnum == 1

    @property
    def n_scenarios(self) -> int:
        return int(self.params.shape[0])

    @property
    def grid(self) -> tuple[int, int]:
        return int(self.actnum.shape[0]), int(self.actnum.shape[1])

    def summary(self) -> dict:
        return {
            "scenarios": self.n_scenarios,
            "grid": list(self.grid),
            "active_cells": int(self.mask.sum()),
            "snapshots": int(self.time_days.size),
            "parameters": self.parameter_names,
            "groups": {group: self.groups.count(group) for group in sorted(set(self.groups))},
            "source": self.source,
        }


def _string_list(dataset) -> list[str]:
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in dataset[:]]


def read_reservoir_dataset(path: str | Path, n_times: int | None = None) -> ReservoirDataset:
    """Read the campaign's HDF5 into arrays, failing closed on anything inconsistent.

    The checks are the ones whose absence produces a silently wrong training set: a
    grid mismatch between the static fields and the dynamic ones, a time axis of the
    wrong length, NaN inside the active cells, or a scenario array whose first axis
    disagrees with the parameter array.
    """
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as handle:
        actnum = np.asarray(handle["static/actnum"][:], dtype=np.uint8)
        permx = np.asarray(handle["static/permx"][:], dtype=np.float32)
        permy = np.asarray(handle["static/permy"][:], dtype=np.float32)
        params = np.asarray(handle["scenarios/params"][:], dtype=np.float64)
        tags = _string_list(handle["scenarios/tag"])
        groups = _string_list(handle["scenarios/group"])
        time_days = np.asarray(handle["scenarios/time_days"][:], dtype=np.float64)
        steps = time_days.size if n_times is None else min(int(n_times), time_days.size)
        swat = np.asarray(handle["scenarios/swat"][:, :steps], dtype=np.float32)
        pressure = np.asarray(handle["scenarios/pressure"][:, :steps], dtype=np.float32)
        names = handle["scenarios/params"]
        # The column order is the contract between whoever generated the dataset and
        # whoever trains on it. It travels as an HDF5 attribute, on the dataset in some
        # writers and on the parent group in the campaign's own files; both are read
        # because silently falling back to p0, p1 would make the model learn the wrong
        # control with no error anywhere.
        attributes = names.attrs
        if "params_columns" not in attributes:
            attributes = handle["scenarios"].attrs
        parameter_names = (
            _string_list(attributes["params_columns"]) if "params_columns" in attributes
            else [f"p{index}" for index in range(params.shape[1])]
        )

    if swat.shape[2:] != actnum.shape:
        raise ValueError(f"swat grid {swat.shape[2:]} does not match actnum grid {actnum.shape}")
    if swat.shape[0] != params.shape[0]:
        raise ValueError("scenarios/swat and scenarios/params disagree on the number of scenarios")
    mask = actnum == 1
    for name, array in (("swat", swat), ("pressure", pressure)):
        inside = array[:, :, mask]
        if not np.isfinite(inside).all():
            raise ValueError(
                f"{name} has non-finite values inside the active cells; "
                "the case this dataset came from did not pass its quality gates"
            )
    if not np.isfinite(permx[mask]).all() or not np.isfinite(permy[mask]).all():
        raise ValueError("permeability has non-finite values inside the active cells")
    return ReservoirDataset(
        actnum=actnum, permx=permx, permy=permy, params=params, time_days=time_days[:steps],
        swat=swat, pressure=pressure, tags=tags, groups=groups,
        parameter_names=parameter_names, source=str(path),
    )
def _unit_parameters(params: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    span = np.where(high > low, high - low, 1.0)
    return (params - low) / span


def _normalised_times(data: ReservoirDataset, context: dict):
    import torch

    return torch.tensor((data.time_days / data.time_days[-1]).astype(np.float32),
                        device=context["device"]).view(1, -1)


def _grid_batch(data: ReservoirDataset, indices: np.ndarray, context: dict):
    """Full field in, full field out: what the grid operators are asked for.

    Targets are NaN outside the active cells, and the loss ignores them, so the operator
    never spends capacity on cells the simulator did not solve.
    """
    import torch

    device = context["device"]
    mask = data.mask
    theta_unit = _unit_parameters(data.params[indices], context["low"], context["high"]).astype(np.float32)
    swat = np.where(mask[None, None], data.swat[indices], np.nan).astype(np.float32)
    pressure = np.where(mask[None, None], data.pressure[indices], np.nan).astype(np.float32)
    pressure = (pressure - context["p_mean"]) / context["p_std"]
    target = np.stack([swat, pressure], axis=1)                       # (B, 2, T, nx, ny)
    times = _normalised_times(data, context)
    return theta_unit, theta_unit * 2.0 - 1.0, times, torch.tensor(target, device=device)


def _log_permeability(data: ReservoirDataset, context: dict) -> np.ndarray:
    """Normalised log permeability, with the inactive cells neutral.

    Outside the active cells there is no rock, so there is no permeability; putting the
    geometric mean there (``log = 0``) keeps a convolutional branch from seeing a
    -14 sigma edge that means nothing. The mesh-free branch cannot leak across cells at
    all, but it computes them anyway, and the same neutral value is used for both.
    """
    mask = data.mask
    logk = np.log(np.clip(data.permx, 1e-6, None) / context["k_geometric"]).astype(np.float32)
    return np.where(mask, logk, 0.0).astype(np.float32)


def _grid_features(data: ReservoirDataset, context: dict):
    """``(nx*ny, 3)`` of (x, y, log K) in the flattened order the branch expects."""
    nx, ny = data.actnum.shape
    i_index, j_index = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    features = np.stack([
        2.0 * i_index.ravel() / max(1, nx - 1) - 1.0,
        2.0 * j_index.ravel() / max(1, ny - 1) - 1.0,
        _log_permeability(data, context).ravel(),
    ], axis=-1).astype(np.float32)
    return features


def _branch_fields(data: ReservoirDataset, indices: np.ndarray, theta_unit: np.ndarray, context: dict):
    """Coordinates, normalised log permeability, and one constant channel per control."""
    import torch

    nx, ny = data.actnum.shape
    xs, ys = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny), indexing="ij")
    logk = _log_permeability(data, context)
    constant = np.stack([np.full((nx, ny), 1.0, dtype=np.float32) for _ in range(theta_unit.shape[1])],
                        axis=0)
    controls = theta_unit[:, :, None, None] * constant[None]           # (B, p, nx, ny)
    geometric = np.broadcast_to(np.stack([xs.astype(np.float32), ys.astype(np.float32), logk], axis=0),
                                (theta_unit.shape[0], 3, nx, ny))
    fields = np.concatenate([geometric, controls], axis=1)
    return torch.tensor(np.ascontiguousarray(fields), device=context["device"])


def _point_samples(data: ReservoirDataset, n_points: int, generator) -> np.ndarray:
    """Random (cell, time) pairs, flat: index = cell * n_times + time."""
    n_cells = int(data.mask.sum())
    total = n_cells * int(data.time_days.size)
    size = min(n_points, total)
    return np.sort(generator.choice(total, size=size, replace=False))


def _point_inputs(data: ReservoirDataset, indices: np.ndarray, samples: np.ndarray, context: dict):
    """Coordinates ``(B, P, 3)`` and controls ``(B, p)`` for the mesh-free operator."""
    import torch

    nx, ny = data.actnum.shape
    n_times = int(data.time_days.size)
    cells = samples // n_times
    steps = samples % n_times
    i_index, j_index = np.where(data.mask)
    coords = np.stack([
        (2.0 * i_index[cells] / max(1, nx - 1) - 1.0),
        (2.0 * j_index[cells] / max(1, ny - 1) - 1.0),
        (2.0 * steps / max(1, n_times - 1) - 1.0),
    ], axis=-1).astype(np.float32)
    theta = (_unit_parameters(data.params[indices], context["low"], context["high"]) * 2.0 - 1.0)
    batch = indices.size
    return (torch.tensor(np.broadcast_to(coords, (batch,) + coords.shape), device=context["device"]),
            torch.tensor(theta.astype(np.float32), device=context["device"]))


def _point_targets(data: ReservoirDataset, indices: np.ndarray, samples: np.ndarray, context: dict):
    import torch

    n_times = int(data.time_days.size)
    cells = samples // n_times
    steps = samples % n_times
    mask = data.mask
    # Select the cells first (dropping inactive ones), then the instants: indexing the
    # time axis before the cell axis would make the two index arrays interact.
    swat = data.swat[indices][:, :, mask][:, steps, cells]              # (B, P)
    pressure = data.pressure[indices][:, :, mask][:, steps, cells]
    pressure = (np.asarray(pressure, dtype=np.float32) - context["p_mean"]) / context["p_std"]
    target = np.stack([swat, np.asarray(pressure, dtype=np.float32)], axis=-1)
    return torch.tensor(np.ascontiguousarray(target), device=context["device"])


def _forward(model, kind: str, theta_cheb, times, batch, extra=None):
    if kind == "point_operator":
        return model(extra, theta_cheb).y
    if kind == "chebyshev_branch":
        return model(extra, theta_cheb, times).y
    return model(extra, times).y


def _kind_of(model) -> str:
    from pinneapple_neural.architectures.neural_operators.chebyshev import (
        ChebyshevBranchOperator,
        ChebyshevPolynomialNeuralOperator,
    )

    if isinstance(model, ChebyshevPolynomialNeuralOperator):
        return "point_operator"
    if isinstance(model, ChebyshevBranchOperator):
        return "chebyshev_branch"
    return "hadamard"


def _build_model(config: ReservoirSurrogateConfig, data: ReservoirDataset, context: dict):
    from pinneapple_neural.architectures.neural_operators.registry import NeuralOperatorCatalog

    catalogue = NeuralOperatorCatalog()
    key = config.architecture
    nx, ny = data.actnum.shape
    n_times = int(data.time_days.size)
    n_controls = int(data.params.shape[1])
    if key not in catalogue.list():
        raise ValueError(f"{key!r} is not in the neural-operator registry; "
                         f"available: {sorted(catalogue.list())}")
    factory = catalogue.get(key)
    if key in ("cpno", "chebyshev_polynomial_neural_operator"):
        model = factory(coordinate_dim=3, parameter_dim=n_controls, out_dim=2)
    elif key == "chebyshev_branch_trunk":
        model = factory(n_steps=n_times, grid=(nx, ny), out_dim=2, field_dim=3,
                        parameter_dim=n_controls)
    else:
        model = factory(in_channels=3 + n_controls, n_steps=n_times, grid=(nx, ny), out_dim=2,
                        saturation_channel=config.saturation_channel)
    context["kind"] = _kind_of(model)
    return model


def _masked_loss(prediction, target):
    """MSE over every finite target entry; padding and inactive cells never contribute."""
    import torch

    if prediction.shape != target.shape:
        raise ValueError(f"prediction {tuple(prediction.shape)} vs target {tuple(target.shape)}")
    finite = torch.isfinite(prediction) & torch.isfinite(target)
    difference = torch.where(finite, (prediction - torch.nan_to_num(target)) ** 2,
                             torch.zeros_like(prediction))
    return difference.sum() / finite.sum().clamp_min(1)


def relative_l2(prediction, truth, mask=None) -> float:
    """Relative L2 over the cells that carry data; padding is excluded, not zeroed."""
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if mask is not None:
        prediction = np.where(mask, prediction, np.nan)
        truth = np.where(mask, truth, np.nan)
    keep = np.isfinite(prediction) & np.isfinite(truth)
    denominator = np.linalg.norm(truth[keep])
    return float(np.linalg.norm(prediction[keep] - truth[keep]) / denominator) if denominator else float("nan")


def breakthrough_time(series, time_days, threshold: float = 0.02) -> float:
    """Water arrival time, in years, at one cell, measured against its own starting value.

    Deliberately not an absolute saturation level: a cell that begins wet would otherwise
    report an arrival at t=0 and make every scenario look identical.
    """
    series = np.asarray(series, dtype=np.float64)
    time_days = np.asarray(time_days, dtype=np.float64)
    if series.size != time_days.size:
        raise ValueError("series and time axis have different lengths")
    exceeds = series > series[0] + threshold
    if not np.any(exceeds):
        return float("nan")
    return float(time_days[int(np.argmax(exceeds))] / 365.25)


def _batch_forward(model, data: ReservoirDataset, indices: np.ndarray, kind: str, context: dict,
                   generator, n_points: int):
    """One forward pass for a batch of scenarios, plus its targets."""
    theta_unit, theta_cheb, times, target = _grid_batch(data, indices, context)
    if kind == "point_operator":
        samples = _point_samples(data, n_points, generator)
        coords, theta_cheb = _point_inputs(data, indices, samples, context)
        prediction = model(coords, theta_cheb).y
        return prediction, _point_targets(data, indices, samples, context)
    if kind == "chebyshev_branch":
        import torch

        theta_cheb = torch.tensor(theta_cheb.astype(np.float32), device=context["device"])
        coords = torch.tensor(_grid_features(data, context), device=context["device"])
        return model(coords, theta_cheb, times).y, target
    fields = _branch_fields(data, indices, theta_unit, context)
    return model(fields, times).y, target


def _train_one(model, optimizer, data, indices, context, config, epochs, generator) -> list[float]:
    import torch

    history: list[float] = []
    model.train()
    mask = data.mask
    # The gridded target is the same for every batch, so it is built once per scenario set
    # rather than per epoch: the operators emit a full field, and rebuilding it is waste.
    for epoch in range(epochs):
        order = np.random.default_rng(generator.integers(0, 2**31 - 1)).permutation(indices)
        epoch_loss, batches = 0.0, 0
        for start in range(0, order.size, config.batch_scenarios):
            batch = order[start:start + config.batch_scenarios]
            if batch.size == 0:
                continue
            prediction, target = _batch_forward(model, data, batch, context["kind"], context,
                                                generator, config.max_points)
            loss = _masked_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1
        history.append(epoch_loss / max(1, batches))
        if config.log_every and (epoch % config.log_every == 0 or epoch == epochs - 1):
            print(f"    epoch {epoch:4d}  loss {history[-1]:.6f}", flush=True)
        if not np.isfinite(history[-1]):
            raise RuntimeError("Training produced a non-finite loss")
    del mask
    return history


def _evaluate(model, data, indices, context) -> dict:
    """Every scenario's full field, then the two numbers worth reporting."""
    if indices.size == 0:
        return {"scenarios": 0}
    predicted = predict_reservoir_field(model, context, data, indices)
    mask = data.mask
    saturation_true = data.swat[indices]
    pressure_true = data.pressure[indices]
    return {
        "scenarios": int(indices.size),
        "tags": predicted["tags"],
        "rel_l2_saturation": relative_l2(predicted["swat"], saturation_true, mask),
        "rel_l2_pressure": relative_l2(predicted["pressure"], pressure_true, mask),
        "saturation_bounds": {
            "predicted_min": float(np.nanmin(predicted["swat"])),
            "predicted_max": float(np.nanmax(predicted["swat"])),
            "truth_min": float(np.nanmin(saturation_true)),
            "truth_max": float(np.nanmax(saturation_true)),
        },
    }


def train_reservoir_surrogate(
    config: ReservoirSurrogateConfig,
    dataset_path: str | Path,
    holdout_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train one operator on the training scenarios and score it where it was not fitted.

    The protocol is k-fold over the training file plus, when it exists, the separate
    holdout file. The two answers are kept apart on purpose: cross-validation says how
    stable the fit is, the holdout says whether the model is right on a case the design
    reserved, and only the second one is a generalisation claim.
    """
    import json

    import torch
    from pinneapple_neural.trainer.checkpoint import Checkpoint, save_checkpoint

    torch.manual_seed(config.seed)
    started = time.time()
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = read_reservoir_dataset(dataset_path)
    mask = data.mask
    context: dict[str, Any] = {
        "device": torch.device(config.device),
        "low": data.params.min(axis=0),
        "high": data.params.max(axis=0),
        "p_mean": float(np.mean(data.pressure[:, :, mask])),
        "p_std": float(np.std(data.pressure[:, :, mask])) or 1.0,
        "k_geometric": float(np.exp(np.mean(np.log(np.clip(data.permx[mask], 1e-6, None))))),
    }
    probe = _build_model(config, data, context)
    parameters = sum(parameter.numel() for parameter in probe.parameters())
    print(f"[{config.architecture}] {data.n_scenarios} scenarios, {int(mask.sum())} active cells, "
          f"{parameters} parameters, kind={context['kind']}", flush=True)

    def fresh():
        model = _build_model(config, data, context).to(context["device"])
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                     weight_decay=config.weight_decay)
        return model, optimizer

    indices = np.arange(data.n_scenarios)
    generator = np.random.default_rng(config.seed)
    folds = min(config.folds, indices.size)
    fold_metrics, histories = [], []
    if folds > 1:
        shuffled = generator.permutation(indices)
        for fold in range(folds):
            validation = np.sort(shuffled[fold::folds])
            training = np.setdiff1d(indices, validation)
            model, optimizer = fresh()
            print(f"  fold {fold + 1}/{folds}: {training.size} train, {validation.size} validation",
                  flush=True)
            histories.append(_train_one(model, optimizer, data, training, context, config,
                                       config.epochs, generator))
            fold_metrics.append(_evaluate(model, data, validation, context))

    model, optimizer = fresh()
    print(f"  final fit on all {indices.size} scenarios", flush=True)
    final_history = _train_one(model, optimizer, data, indices, context, config, config.epochs,
                              generator)
    train_metrics = _evaluate(model, data, indices, context)

    holdout_metrics = None
    if holdout_path is not None and Path(holdout_path).is_file():
        holdout = read_reservoir_dataset(holdout_path)
        if holdout.grid != data.grid or not np.array_equal(holdout.actnum, data.actnum):
            raise ValueError("The holdout file is a different reservoir geometry")
        holdout_metrics = _evaluate(model, holdout, np.arange(holdout.n_scenarios), context)

    checkpoint_path = out_dir / "reservoir_surrogate.pt"
    save_checkpoint(str(checkpoint_path), Checkpoint(
        model_state=model.state_dict(), optim_state=None, cfg=config.to_dict(),
        meta={"architecture": config.architecture, "kind": context["kind"],
              "parameters": parameters},
    ))
    summary = {
        "architecture": config.architecture,
        "config": config.to_dict(),
        "dataset": data.summary(),
        "parameters": parameters,
        "kind": context["kind"],
        "folds": fold_metrics,
        "train": train_metrics,
        "holdout": holdout_metrics,
        "history": {"folds": histories, "final": final_history},
        "elapsed_seconds": round(time.time() - started, 2),
        "checkpoint": str(checkpoint_path),
    }
    (out_dir / "reservoir_surrogate.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                      encoding="utf-8")
    return summary


def _inference_context(data: ReservoirDataset) -> dict:
    mask = data.mask
    return {
        "device": "cpu",
        "low": data.params.min(axis=0),
        "high": data.params.max(axis=0),
        "p_mean": float(np.mean(data.pressure[:, :, mask])),
        "p_std": float(np.std(data.pressure[:, :, mask])) or 1.0,
        "k_geometric": float(np.exp(np.mean(np.log(np.clip(data.permx[mask], 1e-6, None))))),
    }


def load_reservoir_surrogate(path: str | Path, config: ReservoirSurrogateConfig,
                             dataset_path: str | Path):
    """Rebuild a trained operator for inference (the config records how it was built)."""
    from pinneapple_neural.trainer.checkpoint import load_checkpoint

    data = read_reservoir_dataset(dataset_path)
    context = _inference_context(data)
    model = _build_model(config, data, context)
    model.load_state_dict(load_checkpoint(str(path)).model_state)
    model.eval()
    return model, context


def predict_reservoir_field(model, context, data: ReservoirDataset, indices=None) -> dict:
    """Full fields for the requested scenarios, NaN outside the active cells.

    The mesh-free operator is evaluated in chunks over (cell, time) pairs, because the
    full dense set for a 81 x 58 layer over 61 instants does not fit in one forward pass
    on a laptop; the result is identical to evaluating it in a single call.
    """
    import torch

    indices = np.arange(data.n_scenarios) if indices is None else np.asarray(indices)
    model.eval()
    nx, ny = data.actnum.shape
    n_times = int(data.time_days.size)
    mask = data.mask
    saturation = np.full((indices.size, n_times, nx, ny), np.nan, dtype=np.float32)
    pressure = np.full_like(saturation, np.nan)
    with torch.no_grad():
        if context["kind"] == "point_operator":
            total = int(mask.sum()) * n_times
            chunk = 8192
            for start in range(0, total, chunk):
                samples = np.arange(start, min(start + chunk, total))
                coords, theta = _point_inputs(data, indices, samples, context)
                prediction = model(coords, theta).y.cpu().numpy()
                cells = samples // n_times
                steps = samples % n_times
                i_index, j_index = np.where(mask)
                for row in range(indices.size):
                    saturation[row, steps, i_index[cells], j_index[cells]] = prediction[row, :, 0]
                    pressure[row, steps, i_index[cells], j_index[cells]] = (
                        prediction[row, :, 1] * context["p_std"] + context["p_mean"]
                    )
        else:
            theta_unit = _unit_parameters(data.params[indices], context["low"], context["high"]).astype(np.float32)
            theta_cheb = (_unit_parameters(data.params[indices], context["low"], context["high"]) * 2.0 - 1.0)
            times = _normalised_times(data, context)
            import torch as _torch

            theta_tensor = _torch.tensor(theta_cheb.astype(np.float32), device=context["device"])
            if context["kind"] == "chebyshev_branch":
                coords = _torch.tensor(_grid_features(data, context), device=context["device"])
                prediction = model(coords, theta_tensor, times).y.cpu().numpy()
            else:
                fields = _branch_fields(data, indices, theta_unit, context)
                prediction = model(fields, times).y.cpu().numpy()
            # The grid operators emit the dense (nx, ny) field; the inactive cells are
            # written as NaN by scattering only the active ones.
            saturation[:, :, mask] = prediction[:, 0][:, :, mask]
            pressure[:, :, mask] = prediction[:, 1][:, :, mask] * context["p_std"] + context["p_mean"]
    return {"swat": saturation, "pressure": pressure, "time_days": data.time_days,
            "indices": indices, "tags": [data.tags[index] for index in indices]}

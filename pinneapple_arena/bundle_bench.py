"""Equal-budget PINN vs FNO on an external dataset, with the verification gates attached to the result.

The task this harness poses is the one an imported benchmark is actually for: reproduce a field ``u(t, x)`` over
a trajectory, and be scored where the model was not trained -- the last ``1 - train_fraction`` of the time steps.
Two families see the same field, the same split, the same optimiser budget and the same seed, are scored with the
same metric (relative L2) and are gated by the same checks:

``fno``
    a 1-D Fourier neural operator over the spatial axis, with the normalised time as a constant input channel and
    a position grid (plus any static fields named in ``config.inputs``, e.g. a permeability): a spectral model of
    the same map.
``pinn``
    a coordinate network ``(t, x) -> u`` trained on the same grid points (a data-supervised PINN). Pass
    ``residual=`` with a callable ``residual(model, t, x, u, du_dt, du_dx, du_dxx) -> Tensor`` to add a PDE term
    and make it a physics-informed one: this library cannot turn the bundle's ``physics.governing`` prose into an
    equation, and the module says so rather than pretending the residual is there.

Both are the smallest honest version of their family (a reduced budget on a CPU), so the numbers here compare the
*families on this data*, not either family against its best published result -- and the report says so.

    >>> from pinneapple_arena.bundle_bench import BenchConfig, run_benchmark
    >>> report = run_benchmark(bundle, BenchConfig(field="density", epochs=200))
    >>> print(report.to_markdown())          # rel-L2 per model, the leader, and the gates that ran
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from dataclasses import field as _dc_field
from typing import Any

import numpy as np

from pinneapple_data.schema import PhysicsDataBundle, run_gates
from pinneapple_data.schema.gates import relative_l2

__all__ = ["BenchConfig", "ModelResult", "BenchReport", "prepare", "run_benchmark", "run_model"]


@dataclass(frozen=True)
class BenchConfig:
    """Everything that has to be equal between the two models, plus the model-specific size knobs."""

    field: str = "density"
    inputs: tuple[str, ...] = ()          # static fields fed to the FNO as extra channels
    train_fraction: float = 0.8
    epochs: int = 200
    lr: float = 1e-3
    batch_size: int = 16
    seed: int = 0
    device: str = "cpu"
    width: int = 32                       # FNO
    modes: int = 8
    layers: int = 4
    hidden: tuple[int, ...] = (64, 64, 64)   # PINN MLP
    tolerance: float = 5e-2               # rel-L2 a model has to stay under to count as reproducing the data

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        d["inputs"] = list(self.inputs)
        return d


@dataclass
class ModelResult:
    """One model's outcome: what it was, what it cost and how far its field is from the data."""

    model: str
    parameters: int = 0
    seconds: float = 0.0
    rel_l2: float = float("nan")
    per_sample: list[float] = _dc_field(default_factory=list)
    train_loss: list[float] = _dc_field(default_factory=list)
    epochs_run: int = 0
    note: str = ""

    @property
    def passed(self) -> bool:
        return bool(np.isfinite(self.rel_l2))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchReport:
    """The comparison and the verification of it."""

    bundle_id: str = ""
    field: str = ""
    config: BenchConfig = _dc_field(default_factory=BenchConfig)
    results: dict[str, ModelResult] = _dc_field(default_factory=dict)
    gates: dict[str, Any] = _dc_field(default_factory=dict)
    n_train: int = 0
    n_test: int = 0
    summary: str = ""

    def leader(self) -> str:
        """The model with the lowest held-out relative L2 (``""`` when none produced a prediction)."""
        scored = {k: r for k, r in self.results.items() if np.isfinite(r.rel_l2)}
        return min(scored, key=lambda k: scored[k].rel_l2) if scored else ""

    @property
    def ok(self) -> bool:
        """True when every model stayed under the configured tolerance."""
        return bool(self.results) and all(
            np.isfinite(r.rel_l2) and r.rel_l2 <= self.config.tolerance for r in self.results.values())

    def to_dict(self) -> dict[str, Any]:
        return {"bundle_id": self.bundle_id, "field": self.field, "config": self.config.to_dict(),
                "n_train": self.n_train, "n_test": self.n_test, "leader": self.leader(), "ok": self.ok,
                "results": {k: r.to_dict() for k, r in self.results.items()}, "gates": self.gates,
                "summary": self.summary}

    def to_markdown(self) -> str:
        cfg = self.config
        lines = [
            f"# PINN vs FNO on `{self.bundle_id}` (field `{self.field}`)",
            "",
            f"- split: {self.n_train} train / {self.n_test} test time steps (last {1 - cfg.train_fraction:.0%} held out), "
            f"{cfg.epochs} epochs, "
            f"lr {cfg.lr}, seed {cfg.seed}",
            "- both models get the same data, optimiser budget and seed; a CPU-sized budget, not a paper "
            "reproduction",
            "",
            "| model | parameters | rel-L2 (held out) | wall time |",
            "|---|---:|---:|---:|",
        ]
        for name, r in self.results.items():
            lines.append(f"| {name} | {r.parameters:,} | {r.rel_l2:.4e} | {r.seconds:.1f} s |")
        leader = self.leader()
        if leader:
            lines += ["", f"Leader: **{leader}** (lowest held-out relative L2, tolerance {cfg.tolerance:.1e})."]
        if self.gates:
            lines += ["", "## Verification gates", "", "| gate | status | value | detail |", "|---|---|---:|---|"]
            for gate in self.gates.get("gates", []):
                lines.append(f"| {gate['check_id']} | {gate['status']} | {gate.get('value')} | {gate.get('detail','')} |")
        if self.summary:
            lines += ["", self.summary]
        return "\n".join(lines) + "\n"


# ── data ───────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class _Data:
    """The tensors both models share: one field ``(nt, nx)``, its coordinates, and the time split.

    The split is along *time*: the models see the first ``train_fraction`` of the trajectory and are scored on the
    rest, which is the question a surrogate is asked ("continue this record"), not a shuffle that a smooth field
    would pass by interpolation alone."""

    y: Any                     # torch.Tensor (nt, nx) float32
    extra: Any                 # torch.Tensor (E, nx) or None: static channels, broadcast along time
    t: np.ndarray              # (nt,) normalised
    x: np.ndarray              # (nx,) normalised
    train: np.ndarray
    test: np.ndarray

    @property
    def nt(self) -> int:
        return int(self.y.shape[0])

    @property
    def nx(self) -> int:
        return int(self.y.shape[1])


def prepare(bundle: PhysicsDataBundle, config: BenchConfig) -> _Data:
    """Turn a bundle into the tensors the harness trains on (time x space fields only)."""
    import torch

    if config.field not in bundle:
        raise KeyError(f"field '{config.field}' is not in bundle {bundle.manifest.id}: {bundle.manifest.fields.names()}")
    y = np.asarray(bundle.field_array(config.field), dtype=np.float32)
    if y.ndim != 2:
        raise NotImplementedError(
            f"field '{config.field}' has shape {y.shape}: the harness handles (time, space) fields. A vector or "
            "multi-axis field needs its own preparation, and averaging it away would be a silent assumption.")
    if "t" not in bundle.axes or "x" not in bundle.axes:
        raise ValueError(f"bundle {bundle.manifest.id} needs a time axis and one spatial axis, has {bundle.axes}")
    t = np.asarray(bundle.axis_array("t"), dtype=np.float64)
    x = np.asarray(bundle.axis_array("x"), dtype=np.float64)
    t = (t - t.min()) / (np.ptp(t) or 1.0)
    x = (x - x.min()) / (np.ptp(x) or 1.0)
    extra = None
    if config.inputs:
        channels = []
        for name in config.inputs:
            arr = np.asarray(bundle.field_array(name), dtype=np.float32)
            if arr.ndim == 2:
                if arr.shape != y.shape:
                    raise ValueError(f"input field '{name}' has shape {arr.shape}, expected {y.shape}")
                channels.append(arr)
            elif arr.shape[0] == y.shape[1]:
                channels.append(np.broadcast_to(arr, y.shape))
            else:
                raise ValueError(f"input field '{name}' has shape {arr.shape}, which fits neither (nt, nx) nor (nx,)")
        extra = torch.tensor(np.stack(channels))                  # (E, nt, nx)
    idx = np.arange(len(t))
    n_train = max(1, min(len(t) - 1, int(round(len(t) * config.train_fraction))))
    return _Data(y=torch.tensor(y), extra=extra, t=t, x=x, train=idx[:n_train], test=idx[n_train:])


def _fno_inputs(data: _Data, config: BenchConfig):
    """The FNO's input: the normalised time as a constant field, plus the static channels and a grid channel."""
    import torch
    t_field = torch.tensor(data.t, dtype=torch.float32).view(-1, 1, 1).expand(-1, 1, data.nx)
    parts = [t_field]
    if data.extra is not None:
        parts.append(data.extra.permute(1, 0, 2))                 # (nt, E, nx)
    return torch.cat(parts, dim=1)


# ── the two models ─────────────────────────────────────────────────────────────────────────────────────────
def _train_fno(data: _Data, config: BenchConfig) -> tuple[Any, list[float], int]:
    import torch

    from pinneapple_neural.architectures.neural_operators.fno import FourierNeuralOperator

    torch.manual_seed(config.seed)
    X = _fno_inputs(data, config)
    in_channels = int(X.shape[1])
    model = FourierNeuralOperator(in_channels=in_channels, out_channels=1, width=config.width,
                                  modes=config.modes, layers=config.layers, use_grid=True).to(config.device)
    scale = float(data.y[data.train].std()) or 1.0
    mean = float(data.y[data.train].mean())
    Y = ((data.y - mean) / scale).unsqueeze(1).to(config.device)
    X = X.to(config.device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr)
    losses: list[float] = []
    train_idx = torch.as_tensor(data.train)
    for _ in range(config.epochs):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx))]
        epoch_loss = 0.0
        for start in range(0, len(perm), config.batch_size):
            batch = perm[start:start + config.batch_size]
            optimiser.zero_grad()
            out = model(X[batch])
            pred = out.y if hasattr(out, "y") else out
            loss = torch.mean((pred - Y[batch]) ** 2)
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss.detach()) * len(batch)
        losses.append(epoch_loss / max(1, len(perm)))
    return model, losses, sum(p.numel() for p in model.parameters())


def _train_pinn(data: _Data, config: BenchConfig,
                residual: Callable[..., Any] | None = None) -> tuple[Any, list[float], int]:
    import torch
    from torch import nn

    torch.manual_seed(config.seed)
    nt, nx = data.nt, data.nx
    layers: list[nn.Module] = []
    last = 2
    for h in config.hidden:
        layers += [nn.Linear(last, h), nn.Tanh()]
        last = h
    layers.append(nn.Linear(last, 1))
    net = nn.Sequential(*layers).to(config.device)
    optimiser = torch.optim.Adam(net.parameters(), lr=config.lr)
    scale = float(data.y[data.train].std()) or 1.0
    mean = float(data.y[data.train].mean())
    target = ((data.y - mean) / scale).to(config.device)
    tt = torch.tensor(data.t, dtype=torch.float32).view(-1, 1).expand(-1, nx).reshape(-1, 1).to(config.device)
    xx = torch.tensor(data.x, dtype=torch.float32).view(1, -1).expand(nt, -1).reshape(-1, 1).to(config.device)
    per_t = nt * nx
    train_rows = torch.as_tensor(np.concatenate([np.arange(i * nx, (i + 1) * nx) for i in data.train]))
    losses: list[float] = []
    for _ in range(config.epochs):
        perm = train_rows[torch.randperm(len(train_rows))]
        epoch_loss = 0.0
        for start in range(0, len(perm), config.batch_size * nx):
            rows = perm[start:start + config.batch_size * nx]
            optimiser.zero_grad()
            t = tt[rows].clone().requires_grad_(residual is not None)
            x = xx[rows].clone().requires_grad_(residual is not None)
            u = net(torch.cat([t, x], dim=1))
            loss = torch.mean((u.reshape(-1) - target.reshape(-1)[rows]) ** 2)
            if residual is not None:
                du_dt = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
                du_dx = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
                du_dxx = torch.autograd.grad(du_dx.sum(), x, create_graph=True)[0]
                loss = loss + 1e-3 * torch.mean(residual(net, t, x, u, du_dt, du_dx, du_dxx) ** 2)
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss.detach()) * len(rows)
        losses.append(epoch_loss / max(1, per_t))
    return net, losses, sum(p.numel() for p in net.parameters())


# ── scoring ────────────────────────────────────────────────────────────────────────────────────────────────
def _predict_fno(model, data: _Data, config: BenchConfig, mean: float, scale: float) -> np.ndarray:
    import torch
    X = _fno_inputs(data, config).to(config.device)
    model.eval()
    with torch.no_grad():
        out = model(X[data.test])
        pred = out.y if hasattr(out, "y") else out
    return pred.squeeze(1).cpu().numpy() * scale + mean


def _predict_pinn(net, data: _Data, config: BenchConfig, mean: float, scale: float) -> np.ndarray:
    import torch
    nt, nx = data.nt, data.nx
    tt = torch.tensor(data.t, dtype=torch.float32).view(-1, 1).expand(-1, nx).reshape(-1, 1)
    xx = torch.tensor(data.x, dtype=torch.float32).view(1, -1).expand(nt, -1).reshape(-1, 1)
    rows = torch.as_tensor(np.concatenate([np.arange(i * nx, (i + 1) * nx) for i in data.test]))
    with torch.no_grad():
        u = net(torch.cat([tt[rows], xx[rows]], dim=1).to(config.device))
    return u.reshape(len(data.test), nx).cpu().numpy() * scale + mean


def run_model(bundle: PhysicsDataBundle, config: BenchConfig, model: str, *,
              residual: Callable[..., Any] | None = None) -> tuple[ModelResult, np.ndarray, np.ndarray]:
    """Train one model and return its result, its predictions on the held-out block and the reference block."""
    import torch

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    data = prepare(bundle, config)
    started = time.time()
    if model == "fno":
        net, losses, params = _train_fno(data, config)
    elif model == "pinn":
        net, losses, params = _train_pinn(data, config, residual=residual)
    else:
        raise ValueError(f"unknown model '{model}'; use 'fno' or 'pinn'")
    seconds = time.time() - started
    y = data.y.numpy()
    mean, scale = float(y[data.train].mean()), float(y[data.train].std()) or 1.0
    if model == "fno":
        pred = _predict_fno(net, data, config, mean, scale)
    else:
        pred = _predict_pinn(net, data, config, mean, scale)
    ref = y[data.test]
    per_sample = [relative_l2(p, r) if float(np.linalg.norm(r)) > 0 else float("nan") for p, r in zip(pred, ref, strict=True)]
    result = ModelResult(model=model, parameters=params, seconds=seconds,
                         rel_l2=float(np.nanmean(per_sample)) if per_sample else float("nan"),
                         per_sample=[float(v) for v in per_sample], train_loss=losses, epochs_run=config.epochs,
                         note="data-supervised PINN (no PDE residual supplied)" if (model == "pinn" and residual is None)
                         else "")
    return result, pred, ref


def run_benchmark(bundle: PhysicsDataBundle, config: BenchConfig | None = None, *,
                  models: Sequence[str] = ("pinn", "fno"), residual: Callable[..., Any] | None = None,
                  ledger=None) -> BenchReport:
    """Train every model on the same split, score them the same way, and gate the best prediction.

    With ``ledger`` (an ``pinneapple_veriphysics.ledger.EvidenceLedger``) each model's held-out score is sealed
    with the bundle digest as its input hash, so the words "the FNO reached 3e-2" have a record behind them."""
    cfg = config or BenchConfig()
    report = BenchReport(bundle_id=bundle.manifest.id, field=cfg.field, config=cfg)
    data = prepare(bundle, cfg)
    report.n_train, report.n_test = len(data.train), len(data.test)
    best: tuple[str, np.ndarray, np.ndarray] | None = None
    for name in models:
        result, pred, ref = run_model(bundle, cfg, name, residual=residual)
        report.results[name] = result
        if best is None or (np.isfinite(result.rel_l2) and result.rel_l2 < report.results[best[0]].rel_l2):
            best = (name, pred, ref)
        if ledger is not None:
            ledger.record(f"benchmark/{name}", passed=bool(result.rel_l2 <= cfg.tolerance),
                          inputs_sha256=bundle.digest(),
                          outputs={"rel_l2": result.rel_l2, "parameters": result.parameters,
                                   "epochs": result.epochs_run, "field": cfg.field},
                          started_at=_iso(time.time() - result.seconds), finished_at=_iso(time.time()),
                          name=f"{name} on {bundle.manifest.id}", category="validation", tool="pinneapple_arena.bundle_bench")
    if best is not None and np.isfinite(report.results[best[0]].rel_l2):
        _, pred, ref = best
        # the gates check the prediction, not the data it came from: a bundle whose field is the model's answer
        # on the held-out block, so `reference` measures the model and every other gate measures the same
        # declarations the data passed
        predicted = _bundle_with_prediction(bundle, cfg.field, pred, slice(len(data.train), None))
        gates = run_gates(predicted, gates=("units", "location", "frame", "provenance", "reference"),
                          reference=ref, reference_field=cfg.field,
                          reference_slice=slice(len(data.train), None),
                          tolerances={"reference": cfg.tolerance})
        report.gates = gates.to_dict()
        report.gates["gates"].insert(0, {**run_gates(bundle, gates=("integrity",)).to_dict()["gates"][0],
                                         "detail": "the data bundle's own digest"})
    report.summary = (
        f"Leader {report.leader() or 'n/a'} on {report.n_test} held-out time steps; "
        f"tolerance {cfg.tolerance:.1e} -> {'within' if report.ok else 'outside'}. "
        "Reduced CPU budget: this compares the two families on this data, not against published numbers.")
    return report


def _bundle_with_prediction(bundle: PhysicsDataBundle, field: str, pred: np.ndarray,
                            where: slice) -> PhysicsDataBundle:
    """A copy of the bundle whose ``field`` holds the model's prediction on ``where`` and the data elsewhere."""
    import copy

    predicted = copy.deepcopy(bundle)
    key = bundle.manifest.fields.get(field).data.key
    values = np.array(predicted.arrays[key], copy=True)
    values[where] = pred
    predicted.arrays[key] = values
    predicted.manifest.digest = ""
    return predicted


def _iso(ts: float) -> str:
    import datetime as dt
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

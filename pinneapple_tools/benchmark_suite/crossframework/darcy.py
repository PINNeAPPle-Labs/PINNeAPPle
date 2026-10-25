"""Track B of the cross-framework benchmark: the Darcy data, its normalisation and the one training loop.

The three FNO implementations (PINNeAPPle, PhysicsNeMo, neuraloperator) are trained by the same loop on the same
data in the same order, so the comparison measures the model implementations and their speed, not three different
trainers. Protocol (``benchmarks/vs_frameworks/PROTOCOL.md``, track B):

* data: zoo dataset ``darcy_2d_li2021`` (Li et al. 2021 problem, 128 x 128), cell-averaged to 64 x 64
  (:func:`pinneapple_simulation.numerical_solvers.operator_datasets.block_average`);
* split: train rows 0-899, validation 900-999, the 200 test rows;
* normalisation: input scaled by the training mean / std (scalars), output per-pixel unit Gaussian from the training
  rows, decoded before the loss;
* loss: per-sample relative L2 averaged over the batch; Adam with weight decay, cosine schedule over all steps;
  batch 20; a fixed number of epochs; the data order of epoch e is ``default_rng(seed * 1000 + e).permutation``.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = ["load_split", "DarcyData", "load_darcy", "train_operator", "inference_latency", "run_track_b"]

N_TRAIN, N_VAL = 900, 100


def load_split(split: str, resolution: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """``(a, u)`` of ``split`` in ``("train", "val", "test")`` at ``resolution`` (float32, (n, r, r))."""
    import pinneapple as pp
    from pinneapple_simulation.numerical_solvers.operator_datasets import block_average

    src = "test" if split == "test" else "train"
    d = pp.datasets.load("darcy_2d_li2021", split=src)
    a, u = np.asarray(d["a"], dtype=np.float64), np.asarray(d["u"], dtype=np.float64)
    if split == "train":
        a, u = a[:N_TRAIN], u[:N_TRAIN]
    elif split == "val":
        a, u = a[N_TRAIN:N_TRAIN + N_VAL], u[N_TRAIN:N_TRAIN + N_VAL]
    f = a.shape[-1] // resolution
    if f * resolution != a.shape[-1]:
        raise ValueError(f"resolution {resolution} does not divide {a.shape[-1]}")
    if f > 1:
        a, u = block_average(a, f), block_average(u, f)
    return a.astype(np.float32), u.astype(np.float32)


@dataclass
class DarcyData:
    """The three splits and the training-split normalisation."""
    a: dict[str, np.ndarray]
    u: dict[str, np.ndarray]
    a_mean: float
    a_std: float
    u_mean: np.ndarray
    u_std: np.ndarray

    def x(self, split: str):
        import torch
        return torch.from_numpy(((self.a[split] - self.a_mean) / self.a_std)[:, None].astype(np.float32))

    def decode(self, y):
        """Model output (normalised, (B, 1, r, r)) -> u (B, r, r)."""
        import torch
        return y[:, 0] * torch.from_numpy(self.u_std) + torch.from_numpy(self.u_mean)


def load_darcy(resolution: int = 64, n_train: int = N_TRAIN) -> DarcyData:
    a, u = {}, {}
    for s in ("train", "val", "test"):
        a[s], u[s] = load_split(s, resolution)
    a["train"], u["train"] = a["train"][:n_train], u["train"][:n_train]
    return DarcyData(a, u, float(a["train"].mean()), float(a["train"].std()),
                     u["train"].mean(0).astype(np.float32), (u["train"].std(0) + 1e-6).astype(np.float32))


def _rel_l2(pred, true):
    import torch
    b = pred.shape[0]
    return (torch.linalg.vector_norm((pred - true).reshape(b, -1), dim=1)
            / torch.linalg.vector_norm(true.reshape(b, -1), dim=1)).mean()


def _forward(model, x):
    y = model(x)
    return getattr(y, "y", y)                     # PINNeAPPle operators return an OperatorOutput with .y


def train_operator(model, data: DarcyData, *, epochs: int, seed: int, batch_size: int = 20, lr: float = 1e-3,
                   weight_decay: float = 1e-4, log: Callable[[str], None] | None = None) -> dict[str, object]:
    """Train ``model`` ((B, 1, r, r) -> (B, 1, r, r), normalised) with the protocol's loop. Returns the history
    (loss per epoch, wall and CPU time per epoch) and the decoded predictions on the validation and test splits."""
    import torch

    torch.manual_seed(seed)
    X, U = data.x("train"), torch.from_numpy(data.u["train"])
    n = len(X)
    steps_per_epoch = n // batch_size
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * steps_per_epoch))
    hist = {"loss": [], "epoch_wall_s": [], "epoch_cpu_s": []}
    for e in range(epochs):
        model.train()
        w0, c0 = time.perf_counter(), time.process_time()
        perm = np.random.default_rng(seed * 1000 + e).permutation(n)
        tot = 0.0
        for i in range(steps_per_epoch):
            idx = torch.from_numpy(perm[i * batch_size:(i + 1) * batch_size])
            loss = _rel_l2(data.decode(_forward(model, X[idx])), U[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach())
        hist["loss"].append(tot / steps_per_epoch)
        hist["epoch_wall_s"].append(time.perf_counter() - w0)
        hist["epoch_cpu_s"].append(time.process_time() - c0)
        if log:
            log(f"epoch {e + 1}/{epochs} loss {hist['loss'][-1]:.4f} {hist['epoch_wall_s'][-1]:.1f} s")
    model.eval()
    out = {"history": hist}
    with torch.no_grad():
        for s in ("val", "test"):
            xs = data.x(s)
            out[f"pred_{s}"] = torch.cat([data.decode(_forward(model, xs[i:i + 50]))
                                          for i in range(0, len(xs), 50)]).numpy()
    return out


def run_track_b(argv, *, framework: str, source: str, build: Callable[[], object], model_config: dict) -> None:
    """Shared body of the three track-B runners: the runner only says how its library builds the FNO."""
    from . import protocol as proto
    from . import records as R

    args = proto.runner_parser(f"{framework}, track B").parse_args(argv)
    R.setup_threads(args.threads)
    cfg = {**proto.B, "epochs": args.epochs, "n_train": args.n_train, "model": model_config}
    meta = proto.base_meta(args, track="B", framework=framework, source=source, config=cfg)
    t0 = time.perf_counter()
    arrays: dict[str, np.ndarray] = {}
    try:
        import torch
        data = load_darcy(proto.B["resolution"], args.n_train)
        torch.manual_seed(args.seed)
        model = build()
        n_par = int(sum(p.numel() for p in model.parameters()))
        n_real = int(sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters()))
        with R.Clock() as clk:
            out = train_operator(model, data, epochs=args.epochs, seed=args.seed, batch_size=proto.B["batch_size"],
                                 lr=proto.B["lr"], weight_decay=proto.B["weight_decay"])
        hist = out["history"]
        meta.update(status="ok", wall_s=clk.wall_s, cpu_s=clk.cpu_s, loadavg_before=clk.loadavg_before,
                    loadavg_after=clk.loadavg_after, iterations=args.epochs * (len(data.a["train"]) //
                                                                              proto.B["batch_size"]),
                    epochs=args.epochs, n_parameters=n_par, n_parameters_real=n_real, loss_history=hist["loss"],
                    epoch_wall_s=hist["epoch_wall_s"], epoch_cpu_s=hist["epoch_cpu_s"],
                    latency_batch1_s=inference_latency(model, data, 1),
                    latency_batch20_s=inference_latency(model, data, 20))
        from .evaluate import evaluate_operator
        meta["val_metrics"] = evaluate_operator(out["pred_val"], data.u["val"])
        arrays = {"pred_test": out["pred_test"].astype(np.float32)}
    except Exception as exc:  # noqa: BLE001 - a failure is a result
        import traceback
        meta.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-4000:])
    meta["process_wall_s"] = time.perf_counter() - t0
    R.write_run(args.out, args.run_id, meta, arrays)


def inference_latency(model, data: DarcyData, batch: int, repeats: int = 20, warmup: int = 3) -> float:
    """Median wall time [s] of one forward pass on ``batch`` test inputs (no gradients)."""
    import torch
    x = data.x("test")[:batch]
    model.eval()
    ts = []
    with torch.no_grad():
        for i in range(warmup + repeats):
            t0 = time.perf_counter()
            _forward(model, x)
            if i >= warmup:
                ts.append(time.perf_counter() - t0)
    return float(np.median(ts))

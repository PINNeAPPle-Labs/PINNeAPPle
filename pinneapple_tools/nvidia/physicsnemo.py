"""Small PhysicsNeMo surrogates for tabular physics (a few inputs -> a few outputs), trained and checked the
PINNeAPPle way. Grid operators (FNO, MeshGraphNet) are in :mod:`pinneapple_neural.physicsnemo_bridge`.

``MLPSurrogate.fit(X, Y, budget)``
    ``physicsnemo.models.mlp.FullyConnected`` with inputs and outputs scaled to [-1, 1] by the training box, Adam
    with a cosine learning rate, mini-batches, run until a :class:`~pinneapple_analysis.cost.equal_budget.ComputeBudget`
    is spent (``train_within_budget``, so two models with the same budget got the same compute).
``predict(X)``, ``in_box(X)``, ``save`` / ``load``
    NumPy in and out; the training box is kept so a caller can count extrapolated queries.
``sign_violations(f, grid, axis, output, sign)``
    a physics check on any model: the fraction of grid cells where ``d output / d input[axis]`` has the wrong sign
    (e.g. a wheel's load must grow with sinkage), by finite differences on the grid.
``relative_errors(pred, ref)``
    per-output relative RMS errors.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["MLPSurrogate", "sign_violations", "relative_errors"]


def _require():
    from pinneapple_neural.physicsnemo_bridge.adapters import _require_physicsnemo
    return _require_physicsnemo()


@dataclass
class MLPSurrogate:
    in_features: int
    out_features: int
    layer_size: int = 64
    num_layers: int = 4
    activation_fn: str = "silu"
    seed: int = 0
    x_lo: np.ndarray | None = None
    x_hi: np.ndarray | None = None
    y_lo: np.ndarray | None = None
    y_hi: np.ndarray | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    net: Any = None

    def __post_init__(self):
        _require()
        import torch
        from physicsnemo.models.mlp import FullyConnected
        torch.manual_seed(self.seed)
        if self.net is None:
            self.net = FullyConnected(in_features=self.in_features, layer_size=self.layer_size,
                                      out_features=self.out_features, num_layers=self.num_layers,
                                      activation_fn=self.activation_fn)

    @property
    def physicsnemo_model(self):
        return self.net

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))

    def _xn(self, X):
        return 2.0 * (np.asarray(X, float) - self.x_lo) / np.maximum(self.x_hi - self.x_lo, 1e-12) - 1.0

    def fit(self, X, Y, budget, *, batch_size: int = 256, lr: float = 3e-3, evaluate=None, eval_at=()):
        """Train on ``X (n, in)``, ``Y (n, out)`` until ``budget`` is spent; returns the ``BudgetRun``."""
        import torch

        from pinneapple_analysis.cost.equal_budget import cosine_lr, set_lr, train_within_budget
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        self.x_lo, self.x_hi = X.min(0), X.max(0)
        self.y_lo, self.y_hi = Y.min(0), Y.max(0)
        Xt = torch.tensor(self._xn(X), dtype=torch.float32)
        Yt = torch.tensor(2.0 * (Y - self.y_lo) / np.maximum(self.y_hi - self.y_lo, 1e-12) - 1.0, dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        rng = np.random.default_rng(self.seed)

        def step(progress):
            self.net.train()
            idx = torch.from_numpy(rng.integers(0, len(Xt), min(batch_size, len(Xt))))
            set_lr(opt, cosine_lr(progress, lr))
            loss = torch.mean((self.net(Xt[idx]) - Yt[idx]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            return float(loss.detach())

        run = train_within_budget(step, budget, count_first_step=False, evaluate=evaluate, eval_at=eval_at)
        self.net.eval()
        self.history = run.history
        return run

    def predict(self, X) -> np.ndarray:
        import torch
        X = np.atleast_2d(np.asarray(X, float))
        with torch.no_grad():
            yn = self.net(torch.tensor(self._xn(X), dtype=torch.float32)).double().numpy()
        return (yn + 1.0) * (self.y_hi - self.y_lo) / 2.0 + self.y_lo

    __call__ = predict

    def in_box(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        return np.all((X >= self.x_lo) & (X <= self.x_hi), axis=1)

    def save(self, path) -> Path:
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path.with_suffix(".pt"))
        meta = {k: getattr(self, k) for k in ("in_features", "out_features", "layer_size", "num_layers",
                                              "activation_fn", "seed")}
        meta.update({k: np.asarray(getattr(self, k)).tolist() for k in ("x_lo", "x_hi", "y_lo", "y_hi")})
        path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
        return path.with_suffix(".pt")

    @classmethod
    def load(cls, path) -> MLPSurrogate:
        import torch
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        box = {k: np.asarray(meta.pop(k)) for k in ("x_lo", "x_hi", "y_lo", "y_hi")}
        m = cls(**meta, **box)
        m.net.load_state_dict(torch.load(path.with_suffix(".pt"), weights_only=True))
        m.net.eval()
        return m


def sign_violations(f, axes: list[np.ndarray], axis: int, output: int, sign: int = 1, *, tol: float = 0.0) -> float:
    """Fraction of cells of the tensor grid ``axes`` (one 1-D array per input) where the forward difference of
    ``f(X)[:, output]`` along input ``axis`` has the sign opposite to ``sign`` by more than ``tol``."""
    mesh = np.meshgrid(*axes, indexing="ij")
    X = np.stack([m.ravel() for m in mesh], 1)
    y = np.asarray(f(X))[:, output].reshape(mesh[0].shape)
    d = np.diff(y, axis=axis) * sign
    return float(np.mean(d < -tol))


def relative_errors(pred, ref) -> np.ndarray:
    """Per-output RMS error relative to the RMS of the reference output."""
    pred, ref = np.asarray(pred, float), np.asarray(ref, float)
    return np.sqrt(np.mean((pred - ref) ** 2, 0)) / np.maximum(np.sqrt(np.mean(ref ** 2, 0)), 1e-300)

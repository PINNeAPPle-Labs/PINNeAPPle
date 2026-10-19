"""Data-driven direct forecasters on lagged windows: gradient boosting and neural networks (MLP, LSTM, TCN,
transformer), with optional physics-derived input features and, for the MLP, a physics residual penalty.

All of them predict the change of the target from the origin, y_{t+h} - y_t, standardised per horizon, from a
window of the last ``n_lags`` observations (every ``lag_step`` samples) of the observed columns. Predicting the
change keeps persistence as the zero output, so an untrained or over-regularised model falls back to the random-
walk forecast instead of to a constant.

Neural networks are trained with :class:`pinneapple_neural.trainer.Trainer` (Adam, weight decay), early stopping on
the validation block with the best weights restored, and an optional ensemble of ``n_members`` seeds whose mean
is the forecast.
"""
from __future__ import annotations

import tempfile
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

from .base import Forecaster, SeriesData, targets_at

__all__ = ["LagFeatures", "GBMForecaster", "NeuralForecaster", "PhysicsLossMLP", "train_torch_regressor"]


class LagFeatures:
    """Lagged windows of the observed columns, plus optional extra features ``extra(data, origins) -> (n, k)``.

    An ``extra`` object with a ``fit(data, train_origins, horizons)`` method is fitted on the training origins
    first (physics features whose parameters are estimated from the training data only).
    """

    def __init__(self, n_lags: int = 8, lag_step: int = 1, cols: list[int] | None = None,
                 extra: Callable | None = None):
        self.n_lags, self.lag_step, self.cols, self.extra = int(n_lags), int(lag_step), cols, extra

    @property
    def min_history(self) -> int:
        return (self.n_lags - 1) * self.lag_step + 1

    def fit(self, data: SeriesData, train_origins, horizons) -> LagFeatures:
        if self.extra is not None and hasattr(self.extra, "fit"):
            self.extra.fit(data, train_origins, horizons)
        return self

    def window(self, data: SeriesData, origins) -> np.ndarray:
        cols = self.cols if self.cols is not None else list(range(data.obs.shape[1]))
        lags = np.arange(self.n_lags - 1, -1, -1) * self.lag_step          # oldest first
        idx = np.asarray(origins)[:, None] - lags[None, :]
        if idx.min() < 0:
            raise ValueError("an origin has less history than n_lags * lag_step")
        return data.obs[idx][:, :, cols]                                    # (n, L, d)

    def flat(self, data: SeriesData, origins) -> np.ndarray:
        w = self.window(data, origins)
        X = w.reshape(len(w), -1)
        if self.extra is not None:
            X = np.concatenate([X, np.asarray(self.extra(data, origins), float)], axis=1)
        return X


class _Scaler:
    def fit(self, X, axis=0):
        self.mu = X.mean(axis=axis, keepdims=True)
        self.sd = X.std(axis=axis, keepdims=True) + 1e-8
        return self

    def __call__(self, X):
        return (X - self.mu) / self.sd

    def inverse(self, Z):
        return Z * self.sd + self.mu


def _delta_targets(data, origins, horizons):
    return targets_at(data, origins, horizons) - data.y[np.asarray(origins)][:, None]


class GBMForecaster(Forecaster):
    """Histogram gradient boosting (scikit-learn), one regressor per horizon, on lag features.

    scikit-learn's own early stopping holds out a *random* fraction of the training rows, which leaks across time
    on serial data, so it is switched off; the number of boosting rounds is a hyperparameter for the purged
    walk-forward search instead.
    """
    name, family = "gbm", "data"

    def __init__(self, n_lags: int = 8, lag_step: int = 1, max_iter: int = 200, learning_rate: float = 0.05,
                 max_leaf_nodes: int = 15, min_samples_leaf: int = 20, l2: float = 0.0,
                 features: Callable | None = None, cols=None, name: str | None = None,
                 family: str | None = None, seed: int = 0):
        super().__init__()
        self.lags = LagFeatures(n_lags, lag_step, cols, features)
        self.kw = dict(max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
                       min_samples_leaf=min_samples_leaf, l2_regularization=l2, early_stopping=False,
                       random_state=seed)
        self.name = name or self.name
        self.family = family or self.family

    def _fit(self, data, train_origins, val_origins):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.lags.fit(data, train_origins, self.horizons)
        X = self.lags.flat(data, train_origins)
        Y = _delta_targets(data, train_origins, self.horizons)
        self.models_ = [HistGradientBoostingRegressor(**self.kw).fit(X, Y[:, j]) for j in range(Y.shape[1])]

    def _predict_mean(self, data, origins):
        X = self.lags.flat(data, origins)
        d = np.column_stack([m.predict(X) for m in self.models_])
        return data.y[origins][:, None] + d

    def n_params(self):
        # one threshold per split node and one value per leaf
        return int(sum(sum(p.get_n_leaf_nodes() * 2 - 1 for p in it) for m in self.models_
                       for it in m._predictors))


# ---------------------------------------------------------------------------
# Neural networks
# ---------------------------------------------------------------------------

def _mlp(n_in, n_out, hidden, layers, dropout):
    import torch.nn as nn
    mods, w = [], n_in
    for _ in range(int(layers)):
        mods += [nn.Linear(w, hidden), nn.GELU()]
        if dropout > 0:
            mods.append(nn.Dropout(dropout))
        w = hidden
    mods.append(nn.Linear(w, n_out))
    return nn.Sequential(*mods)


def _sequence_net(arch, L, d, n_out, hidden, layers, dropout):
    import torch.nn as nn
    if arch == "lstm":
        from ..models.recurrent import LSTMForecaster, RecurrentConfig
        core = LSTMForecaster(RecurrentConfig(input_len=L, horizon=n_out, n_features=d, n_targets=1,
                                              hidden_size=hidden, num_layers=layers, dropout=dropout))
    elif arch == "tcn":
        from ..models.tcn import TCNConfig, TCNForecaster
        core = TCNForecaster(TCNConfig(input_len=L, horizon=n_out, n_features=d, n_targets=1, n_channels=hidden,
                                       n_layers=layers, kernel_size=3, dropout=dropout))
    elif arch == "transformer":
        from pinneapple_neural.architectures.transformers.transformer import VanillaTransformer
        heads = 4 if hidden % 4 == 0 else 1
        return VanillaTransformer(in_dim=d, out_dim=n_out, d_model=hidden, nhead=heads, num_layers=layers,
                                  dim_feedforward=2 * hidden, dropout=dropout, pool="last", max_len=max(L, 8))
    else:
        raise ValueError(f"unknown architecture {arch!r}")

    class _Squeeze(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x).y_hat[..., 0]

    return _Squeeze(core)


def train_torch_regressor(build: Callable[[], Any], X_tr, Y_tr, X_va=None, Y_va=None, *, epochs: int = 100,
                          lr: float = 1e-3, weight_decay: float = 0.0, batch_size: int = 128,
                          patience: int | None = 10, seed: int = 0,
                          loss_fn: Callable | None = None) -> dict[str, Any]:
    """Train ``build()`` on (X, Y) with :class:`pinneapple_neural.trainer.Trainer`.

    With ``patience`` (and a validation set) training stops when the validation loss has not improved for that
    many epochs and the best weights are restored; ``patience=None`` trains the full ``epochs`` and keeps the last
    weights. Returns the model and the history (train / validation loss per epoch).
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from pinneapple_neural.trainer.callbacks import EarlyStopping
    from pinneapple_neural.trainer.checkpoint import load_checkpoint
    from pinneapple_neural.trainer.trainer import TrainConfig, Trainer

    torch.manual_seed(seed)
    model = build()
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
    g = torch.Generator().manual_seed(seed)
    tr = DataLoader(TensorDataset(t(X_tr), t(Y_tr)), batch_size=batch_size, shuffle=True, generator=g)
    has_val = X_va is not None and len(X_va) > 0
    va = DataLoader(TensorDataset(t(X_va), t(Y_va)) if has_val else TensorDataset(t(X_tr[:256]), t(Y_tr[:256])),
                    batch_size=1024)
    mse = loss_fn or (lambda m, yh, b: torch.mean((yh - b["y"]) ** 2))
    use_es = patience is not None and has_val
    trainer = Trainer(model, mse, metrics=[], early_stopping=EarlyStopping(patience=int(patience)) if use_es else None)
    with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)     # torch.cuda.amp deprecation inside the Trainer
        cfg = TrainConfig(epochs=int(epochs), lr=lr, weight_decay=weight_decay, device="cpu", seed=seed,
                          log_dir=tmp, run_name="fit", save_best=use_es, physics_aware_validation=False)
        out = trainer.fit(tr, va, cfg)
        if use_es and out["best_path"]:
            model.load_state_dict(load_checkpoint(out["best_path"]).model_state)
    model.eval()
    hist = out["history"]
    best = int(np.argmin([h["val_total"] for h in hist])) if hist else 0
    return {"model": model, "history": hist, "epochs_run": len(hist), "best_epoch": best}


class NeuralForecaster(Forecaster):
    """Direct multi-horizon neural forecaster: ``arch`` in {"mlp", "lstm", "tcn", "transformer"}.

    The MLP sees the flattened standardised window (plus any extra features); the sequence models see the window
    as a sequence (L, d). ``n_members`` > 1 trains an ensemble from different seeds and averages it.
    """
    name, family = "mlp", "data"

    def __init__(self, arch: str = "mlp", n_lags: int = 16, lag_step: int = 1, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.0, weight_decay: float = 0.0, lr: float = 1e-3, epochs: int = 100,
                 patience: int | None = 10, batch_size: int = 128, n_members: int = 1, seed: int = 0,
                 features: Callable | None = None, cols=None, max_train: int | None = None,
                 name: str | None = None, family: str | None = None):
        super().__init__()
        self.arch = arch
        self.lags = LagFeatures(n_lags, lag_step, cols, features if arch == "mlp" else None)
        self.hp = dict(hidden=hidden, layers=layers, dropout=dropout, weight_decay=weight_decay, lr=lr,
                       epochs=epochs, patience=patience, batch_size=batch_size)
        self.n_members, self.seed, self.max_train = int(n_members), int(seed), max_train
        self.name = name or (arch if arch != "mlp" else self.name)
        self.family = family or self.family

    # inputs ---------------------------------------------------------------
    def _inputs(self, data, origins):
        if self.arch == "mlp":
            return self.xs_(self.lags.flat(data, origins))
        return self.xs_(self.lags.window(data, origins))

    def _n_out(self):
        return len(self.horizons)

    def _targets(self, data, origins):
        return _delta_targets(data, origins, self.horizons)

    def _build(self, sample_x):
        hp = self.hp
        if self.arch == "mlp":
            return lambda: _mlp(sample_x.shape[1], self._n_out(), hp["hidden"], hp["layers"], hp["dropout"])
        L, d = sample_x.shape[1], sample_x.shape[2]
        return lambda: _sequence_net(self.arch, L, d, self._n_out(), hp["hidden"], hp["layers"], hp["dropout"])

    def _loss(self):
        return None

    def _fit(self, data, train_origins, val_origins):
        if self.max_train is not None and len(train_origins) > self.max_train:
            train_origins = train_origins[-int(self.max_train):]
        self.lags.fit(data, train_origins, self.horizons)
        raw = self.lags.flat(data, train_origins) if self.arch == "mlp" else self.lags.window(data, train_origins)
        self.xs_ = _Scaler().fit(raw, axis=0 if self.arch == "mlp" else (0, 1))
        X = self.xs_(raw)
        Yd = self._targets(data, train_origins)
        self.ys_ = _Scaler().fit(Yd)
        Y = self.ys_(Yd)
        Xv = Yv = None
        if val_origins is not None:
            Xv, Yv = self._inputs(data, val_origins), self.ys_(self._targets(data, val_origins))
        hp = self.hp
        self.members_, self.histories_ = [], []
        for m in range(self.n_members):
            res = train_torch_regressor(self._build(X), X, Y, Xv, Yv, epochs=hp["epochs"], lr=hp["lr"],
                                        weight_decay=hp["weight_decay"], batch_size=hp["batch_size"],
                                        patience=hp["patience"], seed=self.seed + 1000 * m, loss_fn=self._loss())
            self.members_.append(res["model"])
            self.histories_.append({"epochs_run": res["epochs_run"], "best_epoch": res["best_epoch"],
                                    "train": [h["train_total"] for h in res["history"]],
                                    "val": [h["val_total"] for h in res["history"]]})
        self.info = {"arch": self.arch, "n_members": self.n_members,
                     "epochs_run": [h["epochs_run"] for h in self.histories_],
                     "best_epoch": [h["best_epoch"] for h in self.histories_], **{k: v for k, v in hp.items()},
                     "n_lags": self.lags.n_lags, "lag_step": self.lags.lag_step}

    def _raw_outputs(self, data, origins):
        import torch
        X = torch.as_tensor(self._inputs(data, origins), dtype=torch.float32)
        with torch.no_grad():
            outs = [np.concatenate([m(X[i:i + 4096]).numpy() for i in range(0, len(X), 4096)])
                    for m in self.members_]
        return np.mean(outs, axis=0)

    def _predict_mean(self, data, origins):
        d = self.ys_.inverse(self._raw_outputs(data, origins))
        return data.y[origins][:, None] + d

    def n_params(self):
        return int(sum(sum(p.numel() for p in m.parameters()) for m in self.members_))


class PhysicsLossMLP(NeuralForecaster):
    """MLP that predicts the whole observed state at every step 1..max(h) and is penalised for violating a known
    vector field: loss = MSE + ``lam`` * mean |(X_{k+1} - X_k) / dt - f((X_{k+1} + X_k) / 2)|^2 / scale^2 along the
    predicted path (implicit-midpoint residual, second order in dt; X_0 is the observed state at the origin).

    ``physical`` is an :class:`~pinneapple_systems.time_series.forecast_lab.physical.ODEForecaster` (not yet
    fitted): its parameters are estimated on the same training origins and its vector field (torch) is the
    constraint. Requires every state variable to be observed. ``lam = 0`` is the same network without the penalty.
    """
    name, family = "mlp_physics_loss", "data+physics"

    def __init__(self, physical, lam: float = 0.1, **kw):
        kw.setdefault("name", "mlp_physics_loss")
        kw.setdefault("family", "data+physics")
        super().__init__(arch="mlp", **kw)
        self.physical, self.lam = physical, float(lam)

    def _n_out(self):
        return int(self.horizons.max()) * self.d_

    def _targets(self, data, origins):
        o = np.asarray(origins)
        steps = np.arange(1, int(self.horizons.max()) + 1)
        return (data.obs[o[:, None] + steps[None]] - data.obs[o][:, None, :]).reshape(len(o), -1)

    def _loss(self):
        import torch
        rhs = self.physical.torch_rhs()
        H, d, dt = int(self.horizons.max()), self.d_, self.dt_
        ys_mu = torch.as_tensor(self.ys_.mu, dtype=torch.float32).view(1, H, d)
        ys_sd = torch.as_tensor(self.ys_.sd, dtype=torch.float32).view(1, H, d)
        xs_mu = torch.as_tensor(self.xs_.mu, dtype=torch.float32).view(-1)[-d:]
        xs_sd = torch.as_tensor(self.xs_.sd, dtype=torch.float32).view(-1)[-d:]
        scale = torch.as_tensor(self.f_scale_, dtype=torch.float32)
        lam = self.lam

        def loss(model, yh, batch):
            mse = torch.mean((yh - batch["y"]) ** 2)
            if lam == 0:
                return {"total": mse, "mse": mse}
            x0 = batch["x"][:, -d:] * xs_sd + xs_mu                         # observed state at the origin
            path = x0[:, None, :] + yh.view(-1, H, d) * ys_sd + ys_mu       # (B, H, d)
            path = torch.cat([x0[:, None, :], path], dim=1)
            res = (path[:, 1:] - path[:, :-1]) / dt - rhs(0.5 * (path[:, 1:] + path[:, :-1]))
            phys = torch.mean((res / scale) ** 2)
            return {"total": mse + lam * phys, "mse": mse, "physics": phys}
        return loss

    def _fit(self, data, train_origins, val_origins):
        self.d_, self.dt_ = data.obs.shape[1], float(data.dt)
        if self.lags.cols is not None or self.lags.extra is not None:
            raise ValueError("PhysicsLossMLP takes all observed columns (the full state) and no extra features")
        self.physical.fit(data, train_origins, None, self.horizons)
        seg = data.obs[: int(np.max(train_origins)) + 1]
        self.f_scale_ = np.std(self.physical.rhs_numpy(seg), axis=0) + 1e-8
        super()._fit(data, train_origins, val_origins)
        self.info.update({"lam": self.lam, "physics_params": self.physical.params_})

    def _predict_mean(self, data, origins):
        H, d = int(self.horizons.max()), self.d_
        dpath = self.ys_.inverse(self._raw_outputs(data, origins)).reshape(len(origins), H, d)
        return data.y[origins][:, None] + dpath[:, self.horizons - 1, data.target]

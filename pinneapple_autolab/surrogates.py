"""Surrogates of the reference solver, compared on data they did not see, each with an uncertainty.

Scalar models (one output at a time, inputs in the unit cube):

* :class:`GPModel`: Gaussian process, Matern 5/2 with ARD length scales (scikit-learn); predictive std.
* :class:`MLPEnsemble`: ``n`` small MLPs on bootstrap resamples (scikit-learn); std = spread of the members.
* :class:`QuadraticRSM`: ridge regression on a full quadratic basis (the classical response surface); std = the
  residual std of its own fit (homoscedastic), the honest baseline a learned model has to beat.

:func:`compare_models` scores every family by k-fold cross-validation (each fold's points held out of its fit):
RMSE, relative error, R², the fraction of held-out points inside the 95 % interval (calibration) and the fit time.
:class:`ModelBoard` keeps the table and the chosen family per output (lowest held-out RMSE).

Fields on a fixed set of points (surface pressure on a parametric shape): :class:`FieldSurrogate` = POD (SVD of the
centred snapshots) + one scalar model per retained mode; the std of the field comes from the modal stds.
Errors per region (:func:`field_errors`) say where on the shape the model is wrong.
"""
from __future__ import annotations

import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["GPModel", "MLPEnsemble", "QuadraticRSM", "FAMILIES", "make_model", "cross_validate", "compare_models",
           "ModelBoard", "FieldSurrogate", "field_errors"]


class _Model:
    name = "model"
    fit_s = 0.0

    def fit(self, U: np.ndarray, y: np.ndarray) -> _Model:
        raise NotImplementedError

    def predict(self, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class GPModel(_Model):
    name = "GP (Matern 5/2, ARD)"

    def __init__(self, restarts: int = 1, seed: int = 0):
        self.restarts, self.seed = restarts, seed

    def fit(self, U, y):
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        t0 = time.perf_counter()
        d = U.shape[1]
        k = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(np.full(d, 0.5), (1e-2, 1e2), nu=2.5) \
            + WhiteKernel(1e-4, (1e-8, 1e-1))
        self.gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=self.restarts,
                                           random_state=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.gp.fit(U, y)
        self.fit_s = time.perf_counter() - t0
        return self

    def predict(self, U):
        m, s = self.gp.predict(np.atleast_2d(U), return_std=True)
        return m, s

    def length_scales(self) -> np.ndarray:
        return np.atleast_1d(self.gp.kernel_.k1.k2.length_scale)


class MLPEnsemble(_Model):
    name = "MLP ensemble (5 x 32-32, bootstrap)"

    def __init__(self, n: int = 5, hidden: tuple[int, ...] = (32, 32), seed: int = 0, max_iter: int = 600):
        self.n, self.hidden, self.seed, self.max_iter = n, hidden, seed, max_iter

    def fit(self, U, y):
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.neural_network import MLPRegressor
        t0 = time.perf_counter()
        self.mu, self.sd = float(np.mean(y)), float(np.std(y) or 1.0)
        t = (y - self.mu) / self.sd
        rng = np.random.default_rng(self.seed)
        self.members = []
        for i in range(self.n):
            idx = rng.integers(0, len(U), len(U)) if self.n > 1 else np.arange(len(U))
            net = MLPRegressor(hidden_layer_sizes=self.hidden, activation="tanh", solver="lbfgs", alpha=1e-3, max_iter=self.max_iter,
                               random_state=self.seed + i)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                net.fit(U[idx], t[idx])
            self.members.append(net)
        self.fit_s = time.perf_counter() - t0
        return self

    def predict(self, U):
        P = np.stack([m.predict(np.atleast_2d(U)) for m in self.members]) * self.sd + self.mu
        return P.mean(0), P.std(0)


class QuadraticRSM(_Model):
    name = "Quadratic response surface (ridge)"

    def __init__(self, alpha: float = 1e-6):
        self.alpha = alpha

    @staticmethod
    def _basis(U):
        U = np.atleast_2d(U)
        n, d = U.shape
        cols = [np.ones(n)] + [U[:, i] for i in range(d)]
        cols += [U[:, i] * U[:, j] for i in range(d) for j in range(i, d)]
        return np.stack(cols, 1)

    def fit(self, U, y):
        t0 = time.perf_counter()
        B = self._basis(U)
        self.w = np.linalg.solve(B.T @ B + self.alpha * np.eye(B.shape[1]), B.T @ y)
        r = y - B @ self.w
        dof = max(len(y) - B.shape[1], 1)
        self.sigma = float(np.sqrt(np.sum(r ** 2) / dof))
        self.fit_s = time.perf_counter() - t0
        return self

    def predict(self, U):
        m = self._basis(U) @ self.w
        return m, np.full(len(m), self.sigma)


FAMILIES = {"gp": GPModel, "mlp_ensemble": MLPEnsemble, "quadratic": QuadraticRSM}


def make_model(family: str, **kw) -> _Model:
    if family not in FAMILIES:
        raise ValueError(f"unknown surrogate family {family!r}; use one of {sorted(FAMILIES)}")
    return FAMILIES[family](**kw)


def _metrics(y: np.ndarray, m: np.ndarray, s: np.ndarray) -> dict[str, float]:
    e = m - y
    ss = float(np.sum((y - y.mean()) ** 2)) or 1.0
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
            "max_abs": float(np.max(np.abs(e))), "rel_mae": float(np.mean(np.abs(e) / np.maximum(np.abs(y), 1e-12))),
            "r2": float(1 - np.sum(e ** 2) / ss), "coverage95": float(np.mean(np.abs(e) <= 1.96 * np.maximum(s, 1e-12))),
            "mean_std": float(np.mean(s)), "n": int(len(y))}


def cross_validate(family: str, U: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 0) -> dict[str, Any]:
    """k-fold cross-validation of one family on one output; returns the metrics and the held-out predictions."""
    n = len(y)
    k = max(2, min(k, n))
    perm = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(perm, k)
    m, s = np.empty(n), np.empty(n)
    t = 0.0
    for f in folds:
        tr = np.setdiff1d(perm, f)
        mod = make_model(family).fit(U[tr], y[tr])
        t += mod.fit_s
        m[f], s[f] = mod.predict(U[f])
    out = _metrics(y, m, s)
    out.update(fit_s=t / k, k=k)
    return {"metrics": out, "pred": m, "std": s}


@dataclass
class ModelBoard:
    """Cross-validated scores of every family on every output, the choice per output and the fitted models."""
    table: dict[str, dict[str, dict[str, float]]]            # output -> family -> metrics
    chosen: dict[str, str]                                   # output -> family
    models: dict[str, _Model] = field(default_factory=dict)  # output -> model refitted on all data
    cv_pred: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)   # output -> family -> (n, 2) mean, std
    n_train: int = 0
    cv_y: dict[str, np.ndarray] = field(default_factory=dict)                 # output -> the reference values

    def predict(self, U: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {k: m.predict(U) for k, m in self.models.items()}

    def to_dict(self) -> dict[str, Any]:
        return {"n_train": self.n_train, "chosen": dict(self.chosen),
                "names": {f: FAMILIES[f].name for f in FAMILIES},
                "table": {o: {f: dict(v) for f, v in d.items()} for o, d in self.table.items()}}


def compare_models(U: np.ndarray, Y: Mapping[str, np.ndarray], families: Sequence[str] = ("gp", "mlp_ensemble",
                   "quadratic"), k: int = 5, seed: int = 0, select: str = "rmse") -> ModelBoard:
    """Cross-validate every family on every output, choose the lowest held-out ``select`` metric per output (a
    family whose 95 % interval holds fewer than half the held-out points is only chosen if nothing else is
    available), refit the chosen family on all data."""
    table: dict[str, dict[str, dict[str, float]]] = {}
    chosen: dict[str, str] = {}
    models: dict[str, _Model] = {}
    cvp: dict[str, dict[str, np.ndarray]] = {}
    cvy: dict[str, np.ndarray] = {}
    for out, y in Y.items():
        y = np.asarray(y, float)
        ok = np.isfinite(y)
        table[out], cvp[out], cvy[out] = {}, {}, y[ok]
        for fam in families:
            if fam == "quadratic" and ok.sum() <= (U.shape[1] + 1) * (U.shape[1] + 2) // 2:
                table[out][fam] = {"skipped": 1.0}
                continue
            r = cross_validate(fam, U[ok], y[ok], k=k, seed=seed)
            table[out][fam] = r["metrics"]
            cvp[out][fam] = np.c_[r["pred"], r["std"]]
        scored = {f: v for f, v in table[out].items() if "skipped" not in v}
        calibrated = {f: v for f, v in scored.items() if v["coverage95"] >= 0.5} or scored
        chosen[out] = min(calibrated, key=lambda f: calibrated[f][select])
        models[out] = make_model(chosen[out]).fit(U[ok], y[ok])
    return ModelBoard(table, chosen, models, cvp, int(len(U)), cvy)


# ---------------------------------------------------------------------------------------------------- fields
class FieldSurrogate:
    """POD + per-mode regression of fields sampled on fixed points. ``energy``: fraction of the snapshot variance
    the retained modes keep (capped at ``max_modes`` and at n_snapshots - 1)."""

    def __init__(self, family: str = "gp", energy: float = 0.99, max_modes: int = 12):
        self.family, self.energy, self.max_modes = family, energy, max_modes

    def fit(self, U: np.ndarray, S: np.ndarray) -> FieldSurrogate:
        t0 = time.perf_counter()
        self.mean = S.mean(0)
        A = S - self.mean
        Uu, sv, Vt = np.linalg.svd(A, full_matrices=False)
        e = np.cumsum(sv ** 2) / max(np.sum(sv ** 2), 1e-30)
        r = int(min(np.searchsorted(e, self.energy) + 1, self.max_modes, max(len(S) - 1, 1)))
        self.n_modes, self.energy_kept = r, float(e[r - 1])
        self.modes = Vt[:r]
        C = A @ self.modes.T
        self.models = [make_model(self.family).fit(U, C[:, i]) for i in range(r)]
        self.fit_s = time.perf_counter() - t0
        return self

    def predict(self, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        U = np.atleast_2d(U)
        ms = [m.predict(U) for m in self.models]
        Cm = np.stack([m for m, _ in ms], 1)
        Cs = np.stack([s for _, s in ms], 1)
        mean = self.mean + Cm @ self.modes
        std = np.sqrt((Cs ** 2) @ (self.modes ** 2))
        return mean, std


def field_errors(pred: np.ndarray, ref: np.ndarray, regions: Mapping[str, np.ndarray] | None = None,
                 std: np.ndarray | None = None) -> dict[str, dict[str, float]]:
    """RMSE, max error and (with ``std``) the 95 % coverage of a predicted field, overall and per region."""
    regs = {"all": np.ones(len(ref), bool), **{k: np.asarray(v) for k, v in (regions or {}).items()}}
    out = {}
    for k, msk in regs.items():
        e = pred[msk] - ref[msk]
        d = {"rmse": float(np.sqrt(np.mean(e ** 2))), "max_abs": float(np.max(np.abs(e))), "n": int(np.size(e))}
        if std is not None:
            d["coverage95"] = float(np.mean(np.abs(e) <= 1.96 * np.maximum(std[msk], 1e-12)))
            d["mean_std"] = float(np.mean(std[msk]))
        out[k] = d
    return out

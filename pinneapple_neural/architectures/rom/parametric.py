"""Parametric reduced-order models: design parameters -> full field, without running the solver.

A POD basis of the snapshot fields (one snapshot per parameter point) and a regression of the POD coefficients on
the parameters: radial basis functions (POD-RBF), a Gaussian process with predictive uncertainty (POD-GPR), or the
nearest training snapshot (the baseline every surrogate must beat). Works with any solver whose fields share one
mesh (same node or cell numbering across the parameter points).

    rom = ParametricPOD(energy=0.9999, regressor="gpr").fit(P_train, X_train)   # P: (n, p), X: (n, D)
    X_hat, X_std = rom.predict(P_test, return_std=True)
    rom.error(P_test, X_test)                                                      # relative L2 per sample
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Regressor = Literal["rbf", "gpr", "nearest", "linear"]


@dataclass
class ParametricPOD:
    """POD basis + coefficient regression. ``r``: largest number of modes; ``energy``: keep the modes holding this
    fraction of the snapshot variance (whichever is smaller). Parameters are scaled to [0, 1] on the training box.
    ``normalize``: split each snapshot into its amplitude (its L2 norm, regressed as log by RBF) and its shape (unit
    norm, compressed by POD); much better when the response spans orders of magnitude (a stiffness ~ W H^3)."""
    r: int = 32
    energy: float | None = 0.99999
    regressor: Regressor = "rbf"
    kernel: str = "thin_plate_spline"
    smoothing: float = 0.0
    normalize: bool = False
    seed: int = 0
    mean_: np.ndarray = field(default=None, repr=False)
    basis_: np.ndarray = field(default=None, repr=False)
    sv_: np.ndarray = field(default=None, repr=False)

    def _scale(self, P):
        return (np.asarray(P, float) - self._lo) / self._span

    def fit(self, P: np.ndarray, X: np.ndarray) -> ParametricPOD:
        P = np.atleast_2d(np.asarray(P, float))
        X = np.asarray(X, float).reshape(len(P), -1)
        self._lo, self._span = P.min(0), np.maximum(np.ptp(P, axis=0), 1e-30)
        if self.normalize:
            from scipy.interpolate import RBFInterpolator
            amp = np.maximum(np.linalg.norm(X, axis=1), 1e-300)
            self._amp = RBFInterpolator(self._scale(P), np.log(amp), kernel=self.kernel)
            X = X / amp[:, None]
        self.mean_ = X.mean(0)
        U, S, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        e = np.cumsum(S ** 2) / max((S ** 2).sum(), 1e-300)
        r = min(self.r, len(S))
        if self.energy is not None:
            r = min(r, int(np.searchsorted(e, self.energy) + 1))
        self.sv_ = S
        self.basis_ = Vt[:r].T                                       # (D, r)
        A = (X - self.mean_) @ self.basis_                            # (n, r) coefficients
        Ps = self._scale(P)
        self._P, self._A = Ps, A
        if self.regressor == "rbf":
            from scipy.interpolate import RBFInterpolator
            self._model = RBFInterpolator(Ps, A, kernel=self.kernel, smoothing=self.smoothing)
        elif self.regressor == "gpr":
            import warnings

            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
            self._model = []
            for j in range(A.shape[1]):                               # one GP per coefficient, own length scales
                k = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(np.full(P.shape[1], 0.3), (1e-2, 1e2)) \
                    + WhiteKernel(1e-8, (1e-12, 1e-3))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._model.append(GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=2,
                                                                random_state=self.seed).fit(Ps, A[:, j]))
        elif self.regressor == "linear":
            from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
            self._model = (LinearNDInterpolator(Ps, A), NearestNDInterpolator(Ps, A))
        elif self.regressor != "nearest":
            raise ValueError(f"unknown regressor {self.regressor}")
        return self

    @property
    def rank(self) -> int:
        return int(self.basis_.shape[1])

    @property
    def energy_captured(self) -> float:
        s2 = self.sv_ ** 2
        return float(s2[: self.rank].sum() / s2.sum())

    def coefficients(self, P: np.ndarray, return_std: bool = False):
        Ps = self._scale(np.atleast_2d(P))
        std = None
        if self.regressor == "rbf":
            A = self._model(Ps)
        elif self.regressor == "gpr":
            out = [g.predict(Ps, return_std=True) for g in self._model]
            A = np.stack([o[0] for o in out], 1)
            std = np.stack([o[1] for o in out], 1)
        elif self.regressor == "linear":
            A = self._model[0](Ps)
            bad = np.isnan(A).any(1)
            if bad.any():
                A[bad] = self._model[1](Ps[bad])
        else:
            d = ((Ps[:, None, :] - self._P[None]) ** 2).sum(-1)
            A = self._A[np.argmin(d, 1)]
        return (A, std) if return_std else A

    def predict(self, P: np.ndarray, return_std: bool = False):
        """Fields at new parameter points (n, D); with ``return_std`` (GPR) also the pointwise standard deviation
        propagated through the orthonormal basis (independent coefficients)."""
        A, std = self.coefficients(P, return_std=True)
        X = A @ self.basis_.T + self.mean_
        amp = np.exp(self._amp(self._scale(np.atleast_2d(P))))[:, None] if self.normalize else 1.0
        X = X * amp
        if not return_std:
            return X
        if std is None:
            return X, None
        return X, np.sqrt((std ** 2) @ (self.basis_ ** 2).T) * amp

    def projection_error(self, X: np.ndarray) -> np.ndarray:
        """Relative L2 error of the best approximation in the basis (the floor any regressor can reach)."""
        X = np.asarray(X, float).reshape(len(X), -1)
        if self.normalize:
            X = X / np.maximum(np.linalg.norm(X, axis=1), 1e-300)[:, None]
        Xc = X - self.mean_
        R = Xc - (Xc @ self.basis_) @ self.basis_.T
        return np.linalg.norm(R, axis=1) / np.maximum(np.linalg.norm(X, axis=1), 1e-300)

    def error(self, P: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Relative L2 error of the prediction per sample."""
        X = np.asarray(X, float).reshape(len(X), -1)
        return np.linalg.norm(self.predict(P) - X, axis=1) / np.maximum(np.linalg.norm(X, axis=1), 1e-300)


def latin_hypercube(n: int, bounds: dict[str, tuple[float, float]], seed: int = 0) -> tuple[np.ndarray, list[str]]:
    """``n`` points of a Latin hypercube in the box ``bounds`` (name -> (lo, hi)). Returns (points, names)."""
    rng = np.random.default_rng(seed)
    names = list(bounds)
    U = (np.stack([rng.permutation(n) for _ in names], 1) + rng.random((n, len(names)))) / n
    lo = np.array([bounds[k][0] for k in names])
    hi = np.array([bounds[k][1] for k in names])
    return lo + U * (hi - lo), names

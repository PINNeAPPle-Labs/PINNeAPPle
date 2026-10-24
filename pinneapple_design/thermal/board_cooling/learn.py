"""Fast approximation of the coupled reference: junction temperatures and the board-top temperature field.

Two parametric reduced-order models (:class:`pinneapple_neural.architectures.rom.ParametricPOD`, POD + one
Gaussian process per coefficient) on ``log(T - T_air)``:

* the junction temperatures of every part, nominal and pessimistic (:func:`.simulate.response`);
* the board-top map (POD of the snapshots, ``MAP_SHAPE`` grid).

The Gaussian processes give a predictive standard deviation, carried back to °C (delta method) so the optimiser can
ask for the limit to hold at ``mean + k std`` and the trust report can check that the stated spread covers the
measured hold-out errors. :func:`holdout` measures it on designs the model never saw, against the nearest-snapshot
baseline every surrogate must beat.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pinneapple_neural.architectures.rom.parametric import ParametricPOD


def _features(X: np.ndarray) -> np.ndarray:
    Z = np.array(X, float, copy=True)
    Z[:, 6] = np.log(Z[:, 6])                       # air speed acts on h roughly as a power law
    return Z


@dataclass
class CoolingSurrogate:
    t_air_c: float
    modes: int = 16
    tj_: ParametricPOD = field(default=None, repr=False)
    map_: ParametricPOD = field(default=None, repr=False)
    fit_seconds: float = 0.0
    n_train: int = 0

    def fit(self, X: np.ndarray, Y: np.ndarray, maps: np.ndarray) -> CoolingSurrogate:
        t0 = time.perf_counter()
        Z = _features(X)
        self.tj_ = ParametricPOD(r=Y.shape[1], energy=None, regressor="gpr").fit(Z, np.log(Y - self.t_air_c))
        M = np.log(np.maximum(maps.reshape(len(maps), -1) - self.t_air_c, 1e-3))
        self.map_ = ParametricPOD(r=self.modes, energy=0.999999, regressor="gpr").fit(Z, M)
        self._shape = maps.shape[1:]
        self.fit_seconds, self.n_train = time.perf_counter() - t0, len(X)
        return self

    def predict(self, X: np.ndarray, maps: bool = True) -> dict[str, np.ndarray]:
        """Junction temperatures (n, n_out) with their std [K]; with ``maps`` the board-top map (n, ny, nx) too."""
        Z = _features(np.atleast_2d(X))
        ly, sy = self.tj_.predict(Z, return_std=True)
        rise = np.exp(ly)
        out = {"tj": self.t_air_c + rise, "tj_std": rise * sy}
        if maps:
            lm, sm = self.map_.predict(Z, return_std=True)
            r = np.exp(lm)
            out["map"] = (self.t_air_c + r).reshape(-1, *self._shape)
            out["map_std"] = (r * sm).reshape(-1, *self._shape)
        return out

    @property
    def map_rank(self) -> int:
        return self.map_.rank


def _nearest(Xtr, Ytr, Mtr, Xte):
    lo, span = Xtr.min(0), np.maximum(np.ptp(Xtr, 0), 1e-12)
    a, b = (_features(Xtr) - _features(lo[None])) / span, (_features(Xte) - _features(lo[None])) / span
    i = np.argmin(((b[:, None, :] - a[None]) ** 2).sum(-1), 1)
    return Ytr[i], Mtr[i]


def errors(pred_tj, pred_map, Y, maps) -> dict[str, Any]:
    e = pred_tj - Y
    em = (pred_map - maps).reshape(len(maps), -1)
    return {"tj_mae_k": float(np.abs(e).mean()), "tj_max_abs_k": float(np.abs(e).max()),
            "tj_mae_by_output_k": np.abs(e).mean(0).tolist(),
            "map_rmse_k": float(np.sqrt((em ** 2).mean())), "map_max_abs_k": float(np.abs(em).max()),
            "map_rel_l2": float(np.mean(np.linalg.norm(em, axis=1)
                                        / np.linalg.norm(maps.reshape(len(maps), -1), axis=1)))}


def holdout(data: dict[str, np.ndarray], t_air_c: float, *, test_fraction: float = 0.2, seed: int = 0,
            modes: int = 16) -> dict[str, Any]:
    """Train on part of the reference solves, measure on the rest: errors of the surrogate and of the nearest
    snapshot, coverage of the +-2 std band, prediction time against the reference time."""
    n = len(data["X"])
    idx = np.random.default_rng(seed).permutation(n)
    nte = max(2, int(round(test_fraction * n)))
    te, tr = idx[:nte], idx[nte:]
    sur = CoolingSurrogate(t_air_c, modes).fit(data["X"][tr], data["Y"][tr], data["maps"][tr])
    t0 = time.perf_counter()
    p = sur.predict(data["X"][te])
    t_pred = (time.perf_counter() - t0) / len(te)
    nn_y, nn_m = _nearest(data["X"][tr], data["Y"][tr], data["maps"][tr], data["X"][te])
    z = np.abs(p["tj"] - data["Y"][te]) / np.maximum(p["tj_std"], 1e-9)
    return {"n_train": int(len(tr)), "n_test": int(len(te)), "test_index": te.tolist(),
            "surrogate": errors(p["tj"], p["map"], data["Y"][te], data["maps"][te]),
            "nearest_snapshot": errors(nn_y, nn_m, data["Y"][te], data["maps"][te]),
            "coverage_2std": float(np.mean(z <= 2.0)),
            "map_modes": sur.map_rank, "fit_seconds": sur.fit_seconds,
            "predict_seconds_per_design": t_pred,
            "reference_seconds_per_design": float(np.median(data["seconds"])),
            "speedup": float(np.median(data["seconds"]) / max(t_pred, 1e-12))}

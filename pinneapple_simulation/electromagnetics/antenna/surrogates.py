"""Surrogates of an antenna quantity over (design, frequency), and honest validation of them.

Models (all ``fit(table, target)`` / ``predict(params_list, f_ghz) -> (mean, std)``):

* :class:`FrequencyMean` -- the mean of the target over the training designs at each frequency (interpolated in
  frequency): the baseline any model must beat;
* :class:`HWOffset` -- the Hansen-Woodyard limit ``10 log10(7.28 L / lambda)`` plus one fitted offset (one
  parameter: physics carries the length and frequency dependence);
* :class:`LinearPhysics` -- ridge regression on physical features (:func:`physical_features`: ``log(L/lambda)``,
  ``W/lambda``, ``Wa/lambda``, slots on/off, director on/off);
* :class:`GPModel` -- Gaussian process (squared-exponential, one length scale per feature, hyper-parameters by
  maximum marginal likelihood) on the same features plus the HW limit as its mean; returns a predictive std.

Validation: :func:`lodo` (leave one *design* out: every frequency of the held-out design leaves together, so the
model never sees the design it predicts) and :func:`holdout` (a fixed set of test designs). Leaving one row out
would leak the design through its other frequencies and flatter the model.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import bounds
from .runner import ResultTable

__all__ = ["physical_features", "columns", "FEATURES", "FrequencyMean", "HWOffset", "LinearPhysics", "GPModel",
           "default_models", "lodo", "holdout", "ValidationResult"]

FEATURES = ("log_L_lambda", "W_lambda", "Wa_lambda", "slots", "director")


def _get(p: Mapping[str, Any], k: str, default: float = 0.0) -> float:
    try:
        v = float(p.get(k, default))
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def physical_features(params: Sequence[Mapping[str, Any]], f_ghz) -> np.ndarray:
    """``(n, 5)`` features of each (design, frequency): ``log(L/lambda)``, ``W/lambda``, ``Wa/lambda``,
    ``nslot > 0`` and ``dir_a > 0``. Lengths in mm (``Lb``, ``W``, ``Wa``)."""
    f = np.broadcast_to(np.asarray(f_ghz, dtype=float), (len(params),))
    lam = 299.792458 / f                                                    # mm
    out = np.empty((len(params), len(FEATURES)))
    for i, p in enumerate(params):
        out[i] = [np.log(_get(p, "Lb") / lam[i]), _get(p, "W") / lam[i], _get(p, "Wa") / lam[i],
                  float(_get(p, "nslot") > 0), float(_get(p, "dir_a") > 0)]
    return out


def _hw_dbi(params: Sequence[Mapping[str, Any]], f_ghz) -> np.ndarray:
    L = np.array([_get(p, "Lb") for p in params]) * 1e-3
    f = np.broadcast_to(np.asarray(f_ghz, dtype=float), (len(params),)) * 1e9
    return np.asarray(bounds.endfire_directivity_dbi(L, f))


def _xy(table: ResultTable, target: str):
    params = [{k: r.get(k) for k in r} for r in table.rows]
    f = table.col("f_GHz")
    y = table.col(target)
    ok = np.isfinite(y) & np.isfinite(f)
    return [p for p, o in zip(params, ok, strict=True) if o], f[ok], y[ok]


class _Model:
    name = "model"

    def fit(self, table: ResultTable, target: str) -> _Model:
        params, f, y = _xy(table, target)
        if len(y) == 0:
            raise ValueError(f"no finite {target!r} values to fit")
        self._fit(params, f, y)
        return self

    def _fit(self, params, f, y):
        raise NotImplementedError

    def predict(self, params: Sequence[Mapping[str, Any]], f_ghz) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class FrequencyMean(_Model):
    """Mean of the training targets at each frequency (linear interpolation between frequencies)."""
    name = "frequency mean"

    def _fit(self, params, f, y):
        self.f = np.unique(f)
        self.m = np.array([y[f == fi].mean() for fi in self.f])
        self.s = np.array([y[f == fi].std(ddof=1) if (f == fi).sum() > 1 else np.nan for fi in self.f])

    def predict(self, params, f_ghz):
        f = np.broadcast_to(np.asarray(f_ghz, dtype=float), (len(params),))
        return np.interp(f, self.f, self.m), np.interp(f, self.f, np.nan_to_num(self.s, nan=0.0))


class HWOffset(_Model):
    """Hansen-Woodyard limit plus one fitted offset (dB)."""
    name = "HW limit + offset"

    def _fit(self, params, f, y):
        r = y - _hw_dbi(params, f)
        self.offset = float(r.mean())
        self.sd = float(r.std(ddof=1)) if len(r) > 1 else 0.0

    def predict(self, params, f_ghz):
        hw = _hw_dbi(params, f_ghz)
        return hw + self.offset, np.full(len(hw), self.sd)


class LinearPhysics(_Model):
    """Ridge regression on :func:`physical_features` (standardised; constant features dropped)."""
    name = "linear physical features"

    def __init__(self, ridge: float = 1e-3):
        self.ridge = ridge

    def _fit(self, params, f, y):
        X = physical_features(params, f)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.keep = self.sd > 1e-12
        Z = np.c_[np.ones(len(X)), (X[:, self.keep] - self.mu[self.keep]) / self.sd[self.keep]]
        A = Z.T @ Z + self.ridge * len(y) * np.diag([0] + [1] * (Z.shape[1] - 1))
        self.w = np.linalg.lstsq(A, Z.T @ y, rcond=None)[0]
        res = y - Z @ self.w
        dof = max(len(y) - Z.shape[1], 1)
        self.sigma = float(np.sqrt(res @ res / dof))

    def predict(self, params, f_ghz):
        X = physical_features(params, f_ghz)
        Z = np.c_[np.ones(len(X)), (X[:, self.keep] - self.mu[self.keep]) / self.sd[self.keep]]
        return Z @ self.w, np.full(len(X), self.sigma)


class GPModel(_Model):
    """GP on :func:`physical_features` with the HW limit as prior mean (``prior="hw"``) or a constant.

    Squared-exponential kernel with one length scale per feature, signal and noise variances; hyper-parameters by
    maximum log marginal likelihood (L-BFGS-B from a few starts). ``predict`` returns the mean and the predictive
    std including the fitted noise. ``features(params, f_ghz) -> (n, d)`` replaces :func:`physical_features` for
    any other design space (then use ``prior="const"``), e.g. ``columns("feed_gain_dBi", "side")``."""
    name = "Gaussian process"

    def __init__(self, prior: str = "hw", restarts: int = 4, seed: int = 0, features=None):
        self.prior, self.restarts, self.seed = prior, restarts, seed
        self.features = features or physical_features

    def _mean(self, params, f):
        return _hw_dbi(params, f) + self.offset if self.prior == "hw" else np.full(len(params), self.offset)

    def _fit(self, params, f, y):
        from scipy.optimize import minimize

        X = self.features(params, f)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.keep = self.sd > 1e-12
        U = (X[:, self.keep] - self.mu[self.keep]) / self.sd[self.keep]
        base = _hw_dbi(params, f) if self.prior == "hw" else np.zeros(len(y))
        self.offset = float((y - base).mean())
        r = y - base - self.offset
        self.ysd = float(r.std()) or 1.0
        t = r / self.ysd
        d = U.shape[1]

        def nll(th):
            ell, s2, n2 = np.exp(th[:d]), np.exp(th[d]), np.exp(th[d + 1])
            K = s2 * np.exp(-0.5 * _sqd(U / ell, U / ell)) + (n2 + 1e-8) * np.eye(len(t))
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                return 1e10
            a = np.linalg.solve(L.T, np.linalg.solve(L, t))
            return 0.5 * t @ a + np.log(np.diag(L)).sum()

        rng = np.random.default_rng(self.seed)
        best = None
        lo = [np.log(0.05)] * d + [np.log(1e-2), np.log(1e-4)]
        hi = [np.log(20.0)] * d + [np.log(10.0), np.log(1.0)]
        for k in range(max(1, self.restarts)):
            x0 = np.r_[np.zeros(d), 0.0, np.log(0.1)] if k == 0 else rng.uniform(lo, hi)
            res = minimize(nll, x0, method="L-BFGS-B", bounds=list(zip(lo, hi, strict=True)))
            if best is None or res.fun < best.fun:
                best = res
        th = best.x
        self.ell, self.s2, self.n2 = np.exp(th[:d]), float(np.exp(th[d])), float(np.exp(th[d + 1]))
        self.U, self.t = U, t
        K = self.s2 * np.exp(-0.5 * _sqd(U / self.ell, U / self.ell)) + (self.n2 + 1e-8) * np.eye(len(t))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, t))

    def predict(self, params, f_ghz):
        X = self.features(params, f_ghz)
        U = (X[:, self.keep] - self.mu[self.keep]) / self.sd[self.keep]
        Ks = self.s2 * np.exp(-0.5 * _sqd(U / self.ell, self.U / self.ell))
        m = Ks @ self.alpha
        v = np.linalg.solve(self.L, Ks.T)
        var = np.maximum(self.s2 - (v ** 2).sum(0), 0) + self.n2
        return self._mean(params, f_ghz) + self.ysd * m, self.ysd * np.sqrt(var)


def _sqd(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T, 0.0)


def columns(*names: str):
    """A ``features`` function for :class:`GPModel`: the named design parameters as they are."""
    def features(params, f_ghz):
        return np.array([[_get(p, n) for n in names] for p in params], dtype=float)
    return features


def default_models() -> dict[str, Callable[[], _Model]]:
    return {"freq_mean": FrequencyMean, "hw_offset": HWOffset, "linear": LinearPhysics, "gp": GPModel}


# -------------------------------------------------------------------------------------------------- validation
@dataclass
class ValidationResult:
    """Out-of-sample predictions of each model; ``rmse[model]`` in the target's unit."""
    target: str
    scheme: str
    rows: list[dict[str, Any]] = field(default_factory=list)      # tag, f_GHz, y, pred[model], std[model]
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def models(self) -> list[str]:
        return [k[5:] for k in (self.rows[0] if self.rows else {}) if k.startswith("pred_")]

    def errors(self, model: str) -> np.ndarray:
        return np.array([r[f"pred_{model}"] - r["y"] for r in self.rows], dtype=float)

    @property
    def rmse(self) -> dict[str, float]:
        return {m: float(np.sqrt(np.nanmean(self.errors(m) ** 2))) for m in self.models}

    @property
    def coverage95(self) -> dict[str, float]:
        """Fraction of held-out values inside mean +- 1.96 std (calibration of the stated uncertainty)."""
        out = {}
        for m in self.models:
            e = self.errors(m)
            s = np.array([r[f"std_{m}"] for r in self.rows], dtype=float)
            ok = np.isfinite(e) & np.isfinite(s)
            out[m] = float(np.mean(np.abs(e[ok]) <= 1.96 * s[ok])) if ok.any() else float("nan")
        return out

    def best(self) -> str:
        r = self.rmse
        return min(r, key=r.get)

    def to_dict(self) -> dict[str, Any]:
        return {"target": self.target, "scheme": self.scheme, "n": len(self.rows), "rmse": self.rmse,
                "coverage95": self.coverage95, "failed": self.failed}


def _evaluate(train: ResultTable, test: ResultTable, target: str, models, out: ValidationResult) -> None:
    params, f, y = _xy(test, target)
    if not len(y):
        return
    tags = [str(p.get("tag")) for p in params]
    preds = {}
    for name, make in models.items():
        try:
            m = make().fit(train, target)
            preds[name] = m.predict(params, f)
        except (ValueError, np.linalg.LinAlgError) as e:
            out.failed[name] = f"{type(e).__name__}: {e}"
            preds[name] = (np.full(len(y), np.nan), np.full(len(y), np.nan))
    for i in range(len(y)):
        row = {"tag": tags[i], "f_GHz": float(f[i]), "y": float(y[i])}
        for name, (m, s) in preds.items():
            row[f"pred_{name}"], row[f"std_{name}"] = float(m[i]), float(s[i])
        out.rows.append(row)


def lodo(table: ResultTable, target: str, models: Mapping[str, Callable[[], Any]] | None = None) -> ValidationResult:
    """Leave-one-design-out: each design (all its frequencies) predicted by models trained on the others."""
    models = models or default_models()
    out = ValidationResult(target, "leave-one-design-out")
    for tag in table.tags:
        train = table.where(lambda r, tag=tag: str(r.get("tag")) != tag)
        test = table.where(lambda r, tag=tag: str(r.get("tag")) == tag)
        if len(train.tags) < 2:
            continue
        _evaluate(train, test, target, models, out)
    return out


def holdout(table: ResultTable, target: str, test_tags: Sequence[str],
            models: Mapping[str, Callable[[], Any]] | None = None) -> ValidationResult:
    """Train on every design not in ``test_tags``; predict those."""
    models = models or default_models()
    test_tags = {str(t) for t in test_tags}
    out = ValidationResult(target, f"held-out ({len(test_tags)} designs)")
    _evaluate(table.where(lambda r: str(r.get("tag")) not in test_tags),
              table.where(lambda r: str(r.get("tag")) in test_tags), target, models, out)
    return out

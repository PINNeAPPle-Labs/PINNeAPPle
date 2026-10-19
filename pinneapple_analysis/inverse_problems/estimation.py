"""Estimators shared by identifiability, calibration and the inverse studio.

* :func:`least_squares_fit`: weighted nonlinear least squares in the scaled parameters ``z`` (trust-region
  reflective, bounds respected), optionally with a Gaussian prior and a smoothness penalty (MAP / Tikhonov), and
  with any subset of parameters held fixed (used by the profile likelihood);
* :func:`gradient_fit`: L-BFGS-B on the same misfit with an exact gradient (the model's adjoint solve, or autograd);
* :func:`metropolis`: adaptive random-walk Metropolis (Haario et al. 2001) on the Gaussian likelihood with flat
  priors inside the bounds (plus the optional Gaussian prior), for posterior means, intervals and correlations.

Misfit convention everywhere: ``chi2 = sum(((pred - values) / sigma) ** 2)``.
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .forward import ForwardModel, Observations, ParameterSet, _predict, jacobian

__all__ = ["FitResult", "Prior", "least_squares_fit", "gradient_fit", "metropolis", "PosteriorSamples"]


@dataclass
class Prior:
    """Gaussian prior on the scaled parameters (``mean``/``std`` in ``z``: for a log parameter ``std=0.5`` means a
    factor of about 1.6) and an optional smoothness penalty ``weight * |L z|^2`` (e.g. first differences of the
    zones of a field)."""
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    L: np.ndarray | None = None
    weight: float = 0.0

    def residuals(self, z: np.ndarray) -> np.ndarray:
        parts = []
        if self.mean is not None:
            parts.append((z - self.mean) / self.std)
        if self.L is not None and self.weight > 0:
            parts.append(np.sqrt(self.weight) * (self.L @ z))
        return np.concatenate(parts) if parts else np.zeros(0)

    def jac(self, p: int) -> np.ndarray:
        parts = []
        if self.mean is not None:
            parts.append(np.diag(1.0 / np.broadcast_to(self.std, (p,))))
        if self.L is not None and self.weight > 0:
            parts.append(np.sqrt(self.weight) * self.L)
        return np.vstack(parts) if parts else np.zeros((0, p))

    def precision(self, p: int) -> np.ndarray:
        J = self.jac(p)
        return J.T @ J


@dataclass
class FitResult:
    z: np.ndarray
    theta: dict[str, float]
    pred: np.ndarray
    chi2: float
    prior_penalty: float
    jac: np.ndarray                    # d pred / d z at the optimum (n, p), unweighted
    nfev: int
    success: bool
    message: str
    seconds: float
    derivatives: str
    at_bound: list[str] = field(default_factory=list)
    history: list[float] = field(default_factory=list)

    @property
    def dof(self) -> int:
        return max(1, len(self.pred) - len(self.z))


def _weighted(obs: Observations, pred: np.ndarray) -> np.ndarray:
    return (pred - obs.values) / obs.sigma


def least_squares_fit(model: ForwardModel, params: ParameterSet, obs: Observations, *,
                      z0: np.ndarray | None = None, free: Sequence[int] | None = None,
                      prior: Prior | None = None, loss: str = "linear", max_nfev: int = 200,
                      derivatives: str = "auto") -> FitResult:
    """Minimise ``chi2 (+ prior)`` over the scaled parameters. ``free``: indices of ``z`` that move (the others stay
    at ``z0``). ``loss``: ``"linear"`` (Gaussian) or a robust scipy loss (``"soft_l1"``, ``"huber"``, ``"cauchy"``).
    """
    from scipy.optimize import least_squares

    if not obs.has_values:
        raise ValueError("the observations have no measured values to fit")
    t0 = time.time()
    n_eval0 = getattr(model, "_n_eval", 0)
    z_full = params.clip(params.z0() if z0 is None else np.asarray(z0, float).copy())
    idx = np.arange(len(params)) if free is None else np.asarray(list(free), int)
    lo, hi = params.bounds()
    how = {"d": derivatives}
    history: list[float] = []

    def full(zf):
        z = z_full.copy()
        z[idx] = zf
        return z

    cache: dict[str, Any] = {}

    def evaluate(zf):
        key = zf.tobytes()
        if cache.get("key") != key:
            pred, J, h = jacobian(model, params, full(zf), obs, method=derivatives)
            cache.update(key=key, pred=pred, J=J)
            how["d"] = h
        return cache["pred"], cache["J"]

    def fun(zf):
        pred, _ = evaluate(zf)
        r = _weighted(obs, pred)
        if prior is not None:
            r = np.concatenate([r, prior.residuals(full(zf))])
        history.append(float(np.sum(r**2)))
        return np.where(np.isfinite(r), r, 1e6)

    def jac(zf):
        _, J = evaluate(zf)
        Jw = J[:, idx] / obs.sigma[:, None]
        if prior is not None:
            Jp = prior.jac(len(z_full))
            Jw = np.vstack([Jw, Jp[:, idx]]) if len(Jp) else Jw
        return np.where(np.isfinite(Jw), Jw, 0.0)

    x0 = np.clip(z_full[idx], lo[idx] + 1e-12 * (np.abs(lo[idx]) + 1), hi[idx] - 1e-12 * (np.abs(hi[idx]) + 1))
    x0 = np.where(np.isfinite(x0), x0, z_full[idx])
    res = least_squares(fun, x0, jac=jac, bounds=(lo[idx], hi[idx]), method="trf", loss=loss, max_nfev=max_nfev,
                        x_scale=1.0)
    z = full(res.x)
    pred, J = evaluate(res.x)
    chi2 = float(np.sum(_weighted(obs, pred) ** 2))
    pen = float(np.sum(prior.residuals(z) ** 2)) if prior is not None else 0.0
    tol = 1e-6
    at_bound = [params.free[i].name for i in idx
                if (np.isfinite(lo[i]) and z[i] - lo[i] < tol * (1 + abs(lo[i])))
                or (np.isfinite(hi[i]) and hi[i] - z[i] < tol * (1 + abs(hi[i])))]
    return FitResult(z, params.theta(z), pred, chi2, pen, J, getattr(model, "_n_eval", 0) - n_eval0,
                     bool(res.success), str(res.message), time.time() - t0, how["d"], at_bound, history)


def gradient_fit(model: ForwardModel, params: ParameterSet, obs: Observations, *, z0: np.ndarray | None = None,
                 prior: Prior | None = None, gradient: str = "adjoint", max_iter: int = 300) -> FitResult:
    """L-BFGS-B on ``0.5 * (chi2 + prior)`` with an exact gradient: ``"adjoint"`` (``model.adjoint_gradient``: one
    forward and one adjoint solve per iteration, whatever the number of parameters) or ``"autograd"``
    (``model.predict_torch``)."""
    from scipy.optimize import minimize

    if not obs.has_values:
        raise ValueError("the observations have no measured values to fit")
    if gradient == "adjoint" and not callable(getattr(model, "adjoint_gradient", None)):
        raise ValueError(f"model '{model.name}' has no adjoint solver")
    if gradient == "autograd" and not callable(getattr(model, "predict_torch", None)):
        raise ValueError(f"model '{model.name}' has no differentiable (torch) implementation")
    t0 = time.time()
    n_eval0 = getattr(model, "_n_eval", 0)
    z0 = params.clip(params.z0() if z0 is None else np.asarray(z0, float))
    lo, hi = params.bounds()
    history: list[float] = []

    def obj(z):
        theta = params.theta(z)
        dtdz = np.array([theta[p.name] if p.log else p.scale for p in params.free])
        if gradient == "adjoint":
            phi, g = model.adjoint_gradient(theta, obs, params.names)
            model._n_eval = getattr(model, "_n_eval", 0) + 1
            g = np.asarray(g, float) * dtdz
        else:
            import torch

            zt = torch.tensor(z, dtype=torch.float64, requires_grad=True)
            th = {k: torch.tensor(v, dtype=torch.float64) for k, v in theta.items()}
            for i, p in enumerate(params.free):
                th[p.name] = torch.exp(zt[i]) if p.log else zt[i] * p.scale
            pred = model.predict_torch(th, obs.inputs, obs.observable)
            r = (pred - torch.as_tensor(obs.values)) / torch.as_tensor(obs.sigma)
            phi_t = 0.5 * torch.sum(r**2)
            phi_t.backward()
            model._n_eval = getattr(model, "_n_eval", 0) + 1
            phi, g = float(phi_t.detach()), zt.grad.numpy().copy()
        if prior is not None:
            rp = prior.residuals(z)
            phi += 0.5 * float(rp @ rp)
            g = g + prior.jac(len(z)).T @ rp
        history.append(2 * phi)
        return phi, g

    bounds = [(None if not np.isfinite(a) else a, None if not np.isfinite(b) else b) for a, b in zip(lo, hi, strict=False)]
    res = minimize(obj, z0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8})
    pred, J, how = jacobian(model, params, res.x, obs)
    chi2 = float(np.sum(_weighted(obs, pred) ** 2))
    pen = float(np.sum(prior.residuals(res.x) ** 2)) if prior is not None else 0.0
    at_bound = [params.free[i].name for i in range(len(res.x))
                if (np.isfinite(lo[i]) and res.x[i] - lo[i] < 1e-6) or (np.isfinite(hi[i]) and hi[i] - res.x[i] < 1e-6)]
    label = "adjoint gradient" if gradient == "adjoint" else "autograd gradient"
    return FitResult(res.x, params.theta(res.x), pred, chi2, pen, J, getattr(model, "_n_eval", 0) - n_eval0,
                     bool(res.success), str(res.message), time.time() - t0, label, at_bound, history)


@dataclass
class PosteriorSamples:
    names: list[str]
    samples_z: np.ndarray               # (m, p) after burn-in and thinning
    samples: np.ndarray                 # (m, p) physical units
    acceptance: float
    n_evaluations: int
    seconds: float

    def summary(self, level: float = 0.95) -> dict[str, Any]:
        a = (1 - level) / 2
        corr = np.corrcoef(self.samples_z.T) if self.samples_z.shape[1] > 1 else np.ones((1, 1))
        return {"names": self.names, "mean": self.samples.mean(axis=0).tolist(),
                "median": np.median(self.samples, axis=0).tolist(), "std": self.samples.std(axis=0, ddof=1).tolist(),
                "lo": np.quantile(self.samples, a, axis=0).tolist(),
                "hi": np.quantile(self.samples, 1 - a, axis=0).tolist(), "level": level,
                "correlation": np.nan_to_num(corr).round(4).tolist(), "acceptance": round(self.acceptance, 3),
                "n_samples": int(len(self.samples)), "n_evaluations": self.n_evaluations,
                "seconds": round(self.seconds, 2)}


def metropolis(model: ForwardModel, params: ParameterSet, obs: Observations, *, z0: np.ndarray,
               cov0: np.ndarray | None = None, n_steps: int = 4000, burn: int | None = None,
               prior: Prior | None = None, seed: int = 0, thin: int = 2, max_seconds: float = 30.0
               ) -> PosteriorSamples:
    """Adaptive random-walk Metropolis in ``z``: flat prior inside the bounds (times ``prior`` if given), Gaussian
    likelihood with the observations' sigma. The proposal covariance starts at ``cov0`` (e.g. the inverse Fisher
    information, regularised) and adapts to the chain (Haario et al. 2001) after the first quarter. Stops early at
    ``max_seconds`` (the summary reports the number of samples actually drawn)."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    p = len(z0)
    lo, hi = params.bounds()
    n_eval0 = getattr(model, "_n_eval", 0)

    def logpost(z):
        if np.any(z < lo) or np.any(z > hi):
            return -np.inf
        pred = _predict(model, params.theta(z), obs)
        if not np.all(np.isfinite(pred)):
            return -np.inf
        lp = -0.5 * float(np.sum(_weighted(obs, pred) ** 2))
        if prior is not None:
            rp = prior.residuals(z)
            lp -= 0.5 * float(rp @ rp)
        return lp

    sd = 2.38**2 / p
    cov = np.eye(p) * 0.01 if cov0 is None else np.array(cov0, float)
    cov = 0.5 * (cov + cov.T) + 1e-10 * np.eye(p)
    try:
        chol = np.linalg.cholesky(sd * cov)
    except np.linalg.LinAlgError:
        chol = np.linalg.cholesky(sd * (np.diag(np.diag(cov)) + 1e-8 * np.eye(p)))
    z, lp = np.asarray(z0, float).copy(), logpost(np.asarray(z0, float))
    chain, acc = [], 0
    burn = n_steps // 4 if burn is None else burn
    for it in range(n_steps):
        prop = z + chol @ rng.standard_normal(p)
        lpp = logpost(prop)
        if np.log(rng.random()) < lpp - lp:
            z, lp = prop, lpp
            acc += 1
        chain.append(z.copy())
        if it >= burn // 2 and it % 100 == 0 and it > 0:
            c = np.cov(np.array(chain[len(chain) // 2:]).T).reshape(p, p) + 1e-10 * np.eye(p)
            try:
                chol = np.linalg.cholesky(sd * c)
            except np.linalg.LinAlgError:
                pass
        if time.time() - t0 > max_seconds and it > 200:
            break
    arr = np.array(chain)
    kept = arr[min(burn, len(arr) // 2)::thin]
    phys = np.array([[params.free[i].from_z(v) for i, v in enumerate(row)] for row in kept])
    return PosteriorSamples(params.names, kept, phys, acc / max(1, len(arr)),
                            getattr(model, "_n_eval", 0) - n_eval0, time.time() - t0)

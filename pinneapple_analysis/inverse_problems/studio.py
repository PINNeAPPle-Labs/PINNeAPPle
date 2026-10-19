"""One interface to formulate an inverse problem, solve it with several methods and validate the answers.

An :class:`InverseProblem` is a forward model, the unknown parameters (scalars or the zones of a field), the
measurements, optional held-out measurements and, in synthetic studies, the known truth. :func:`solve_inverse`
runs one method; :func:`compare_methods` runs several on the *same* data and ranks them by held-out error.

=================  ===============================================================================================
``least_squares``  classical weighted least squares (trust-region reflective in scaled parameters); Jacobian from
                   the model's own sensitivities, autograd or central differences; Gauss-Newton intervals.
``adjoint``        L-BFGS-B on the misfit with the gradient from one adjoint solve (models with
                   ``adjoint_gradient``, e.g. :class:`~.models.RodModel`): the cost does not grow with the number
                   of unknowns, which is what makes field reconstruction affordable.
``autodiff``       L-BFGS-B with the gradient by PyTorch automatic differentiation through the model
                   (``predict_torch``).
``bayesian``       MAP with a Gaussian prior on the scaled parameters (and a smoothness prior on field zones:
                   Tikhonov regularisation), Laplace covariance, and an adaptive-Metropolis posterior when the
                   model is cheap.
``pinn``           a physics-informed network that learns the field and the unknown together (models with
                   ``pinn``: the Inverse Heat Lab fin and plate).
=================  ===============================================================================================

Every solution carries the estimates, their uncertainty when the method gives one, the reconstructed fields, the
residuals, the fit and held-out metrics, and, when the truth is known, the parameter and field errors and whether
the truth lies inside the stated interval. The identifiability report (``identifiability.py``) is computed once per
problem (at the least-squares fit: the Fisher information is local) and attached, so a method's answer for a
non-identifiable parameter is flagged instead of trusted.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .calibration import residual_metrics
from .estimation import Prior, gradient_fit, least_squares_fit, metropolis
from .forward import ForwardModel, Observations, Parameter, ParameterSet, _predict
from .identifiability import IdentifiabilityReport, analyze_identifiability

__all__ = ["InverseProblem", "METHODS", "available_methods", "solve_inverse", "compare_methods"]

METHODS = {
    "least_squares": "Classical least squares (Gauss-Newton / trust region)",
    "adjoint": "Adjoint-gradient optimisation (L-BFGS-B)",
    "autodiff": "Automatic differentiation (PyTorch) + L-BFGS-B",
    "bayesian": "Bayesian regularisation (MAP + Laplace / MCMC posterior)",
    "pinn": "Physics-informed neural network",
}


@dataclass
class InverseProblem:
    model: ForwardModel
    observations: Observations
    parameters: list[Parameter]
    validation: Observations | None = None
    name: str = "inverse problem"
    description: str = ""
    prior_std: float = 1.0                     # Bayesian prior std in z (ln units for log parameters)
    smoothness: float = 0.0                    # weight of the first-difference prior on ``field_groups``
    field_groups: list[list[str]] = field(default_factory=list)   # parameters forming one field (zones in order)

    @property
    def params(self) -> ParameterSet:
        return ParameterSet(self.parameters)

    @property
    def truth(self) -> dict[str, float] | None:
        return self.params.truth()

    def groups(self) -> list[list[str]]:
        """``field_groups``, or the model's own (``model.field_groups()``) when none were given."""
        if self.field_groups:
            return self.field_groups
        fn = getattr(self.model, "field_groups", None)
        return fn() if callable(fn) else []

    def prior(self) -> Prior:
        ps = self.params
        L = None
        rows = []
        for grp in self.groups():
            idx = [ps.names.index(n) for n in grp if n in ps.names]
            for a, b in zip(idx[:-1], idx[1:], strict=False):
                r = np.zeros(len(ps))
                r[a], r[b] = -1.0, 1.0
                rows.append(r)
        if rows:
            L = np.array(rows)
        return Prior(mean=ps.z0(), std=np.full(len(ps), self.prior_std), L=L, weight=self.smoothness)


def available_methods(problem: InverseProblem) -> dict[str, dict[str, Any]]:
    """Which methods apply to this problem, and why not when they do not."""
    m = problem.model
    out = {k: {"label": v, "available": True, "reason": ""} for k, v in METHODS.items()}
    if not callable(getattr(m, "adjoint_gradient", None)):
        out["adjoint"].update(available=False, reason="the model has no adjoint solver (only discretised PDE models "
                                                      "such as the rod do)")
    if not callable(getattr(m, "predict_torch", None)):
        out["autodiff"].update(available=False, reason="the model has no differentiable (PyTorch) implementation")
    if not callable(getattr(m, "pinn", None)):
        out["pinn"].update(available=False, reason="no physics-informed network formulation for this model")
    else:
        ps = {p.name: p for p in problem.parameters}
        free = [p.name for p in problem.parameters if not p.fixed]
        if free != ["h"]:
            out["pinn"].update(available=False, reason="the PINN formulation estimates h alone; fix the other "
                                                       "parameters")
        elif "k" in ps and not ps["k"].fixed:
            out["pinn"].update(available=False, reason="fix k for the PINN")
    return out


def _gn_std(params: ParameterSet, J: np.ndarray, obs: Observations, chi2: float, extra_prec=None) -> np.ndarray:
    """Gauss-Newton standard deviations in z with the stated noise, widened by sqrt(chi2/dof) when the residuals
    are larger than it (never narrowed: a lucky small residual is not evidence of a smaller noise)."""
    Jw = J / obs.sigma[:, None]
    H = Jw.T @ Jw
    dof = max(1, obs.n - len(params))
    s2 = max(chi2 / dof, 1.0) if extra_prec is None else 1.0
    if extra_prec is not None:
        H = H + extra_prec
    # pseudo-inverse: parameters outside a non-identifiable direction keep their (conditional) variance
    cov = s2 * np.linalg.pinv(H, rcond=1e-10, hermitian=True)
    return np.sqrt(np.maximum(np.diag(cov), 0))


def _intervals(params: ParameterSet, z: np.ndarray, sd: np.ndarray, level: float = 0.95) -> list[list | None]:
    q = 1.959963984540054 if abs(level - 0.95) < 1e-9 else float(__import__("scipy").stats.norm.ppf(0.5 + level / 2))
    out = []
    for i, p in enumerate(params.free):
        if not np.isfinite(sd[i]) or sd[i] > 30:
            out.append(None)
        elif p.log:
            out.append([math.exp(z[i] - q * sd[i]), math.exp(z[i] + q * sd[i])])
        else:
            out.append([(z[i] - q * sd[i]) * p.scale, (z[i] + q * sd[i]) * p.scale])
    return out


def _fields(model: ForwardModel, theta) -> dict[str, Any] | None:
    fn = getattr(model, "fields", None)
    return fn(theta) if callable(fn) else None


def _field_error(est: dict | None, ref: dict | None) -> dict[str, dict[str, float]] | None:
    if not est or not ref:
        return None
    out = {}
    for k, v in est.items():
        if k in ref:
            a, b = np.asarray(v["values"], float), np.asarray(ref[k]["values"], float)
            if a.shape == b.shape:
                out[k] = {"rmse": float(np.sqrt(np.mean((a - b) ** 2))), "max_abs": float(np.max(np.abs(a - b))),
                          "unit": v.get("unit", "")}
    return out or None


def solve_inverse(problem: InverseProblem, method: str, *, identifiability: IdentifiabilityReport | None = None,
                  pinn_steps: int = 1500, mcmc_steps: int = 3000, seed: int = 0,
                  progress: Callable[[int, int, float], None] | None = None) -> dict[str, Any]:
    """Solve ``problem`` with ``method`` (a key of :data:`METHODS`); returns a JSON-ready dict (see module doc)."""
    if method not in METHODS:
        raise ValueError(f"unknown method '{method}'; use one of {list(METHODS)}")
    avail = available_methods(problem)[method]
    if not avail["available"]:
        return {"method": method, "label": METHODS[method], "status": "not applicable", "reason": avail["reason"]}
    model, obs, params = problem.model, problem.observations, problem.params
    t0 = time.time()
    n0 = getattr(model, "_n_eval", 0)
    notes: list[str] = []
    std = None
    post = None
    history: list = []
    pinn_field = None
    if method == "least_squares":
        r = least_squares_fit(model, params, obs)
        std = _gn_std(params, r.jac, obs, r.chi2)
        z, history, deriv, converged = r.z, r.history, r.derivatives, r.success
    elif method in ("adjoint", "autodiff"):
        r = gradient_fit(model, params, obs, gradient="adjoint" if method == "adjoint" else "autograd")
        std = _gn_std(params, r.jac, obs, r.chi2)
        z, history, deriv, converged = r.z, r.history, r.derivatives, r.success
        if not r.success:
            notes.append(f"the optimiser did not converge: {r.message}")
    elif method == "bayesian":
        prior = problem.prior()
        r = least_squares_fit(model, params, obs, prior=prior)
        prec = prior.precision(len(params))
        std = _gn_std(params, r.jac, obs, r.chi2, extra_prec=prec)
        z, history, deriv, converged = r.z, r.history, r.derivatives, r.success
        notes.append(f"Gaussian prior: std {problem.prior_std:g} in ln units around the initial guesses"
                     + (f"; smoothness weight {problem.smoothness:g} on {problem.groups()}"
                        if problem.smoothness > 0 and problem.groups() else ""))
        lo, hi = params.bounds()
        if getattr(model, "cost", "cheap") == "cheap" and len(params) <= 8 and np.all(np.isfinite(lo)) \
                and np.all(np.isfinite(hi)):
            Jw = r.jac / obs.sigma[:, None]
            cov0 = np.linalg.inv(Jw.T @ Jw + prec + 1e-9 * np.eye(len(params)))
            ps = metropolis(model, params, obs, z0=r.z, cov0=cov0, n_steps=mcmc_steps, prior=prior, seed=seed,
                            max_seconds=25.0)
            post = ps.summary()
            post["note"] = "adaptive Metropolis on the same prior and likelihood"
    else:  # pinn
        pmap = {p.name: p for p in problem.parameters}
        truth_h = (problem.truth or {}).get("h")
        out = model.pinn(obs, pmap, steps=pinn_steps, seed=seed, progress=progress, h_true=truth_h)
        theta = params.theta()
        theta.update(out["theta"])
        z = params.z_of(theta)
        history = out.get("history", [])
        pinn_field = out.get("field")
        deriv = "network autograd (inside the PINN loss)"
        converged = True
        notes.append("A single PINN run gives no uncertainty; the readings are fitted by the network, the model "
                     "predictions below use the PINN's h in the reference model.")
    theta = params.theta(z)
    pred = _predict(model, theta, obs)
    fit_metrics = residual_metrics(pred, obs)
    val_metrics = None
    val_pred = None
    if problem.validation is not None and problem.validation.n:
        val_pred = _predict(model, theta, problem.validation)
        val_metrics = residual_metrics(val_pred, problem.validation)
    ci = _intervals(params, z, std) if std is not None else [None] * len(params)
    if post is not None:
        ci = [[post["lo"][i], post["hi"][i]] for i in range(len(params))]
    rows = []
    truth = problem.truth
    ident = {q["name"]: q for q in identifiability.parameters} if identifiability else {}
    for i, p in enumerate(params.free):
        status = ident.get(p.name, {}).get("status")
        row = {"name": p.name, "unit": p.unit, "initial": p.value, "estimate": theta[p.name],
               "interval95": ci[i], "identifiability": status}
        if status not in (None, "identifiable"):
            if method == "bayesian":
                row["note"] = "not identifiable from the data: this interval is set by the prior"
            else:
                row["interval95"] = None
                row["note"] = "not identifiable from the data: no interval, and the estimate is one of many"
        if post is not None:
            row["posterior_mean"] = post["mean"][i]
            row["posterior_std"] = post["std"][i]
        if truth is not None:
            row["truth"] = truth[p.name]
            row["error_pct"] = 100 * (theta[p.name] - truth[p.name]) / max(abs(truth[p.name]), 1e-300)
            iv = row["interval95"]
            row["truth_in_interval"] = None if iv is None else bool(iv[0] <= truth[p.name] <= iv[1])
        rows.append(row)
    fields = _fields(model, theta)
    truth_fields = _fields(model, {**params.theta(), **truth}) if truth is not None else None
    sol = {"method": method, "label": METHODS[method], "status": "done", "theta": theta, "parameters": rows,
           "fit": fit_metrics, "validation": val_metrics, "fields": fields, "pinn_field": pinn_field,
           "field_error": _field_error(fields, truth_fields), "posterior": post,
           "residuals": (pred - obs.values).tolist(), "prediction": pred.tolist(),
           "validation_prediction": None if val_pred is None else val_pred.tolist(),
           "history": history[-300:] if history else [], "derivatives": deriv, "notes": notes,
           "converged": bool(converged),
           "seconds": time.time() - t0, "forward_evaluations": getattr(model, "_n_eval", 0) - n0}
    if truth is not None:
        recovered = []
        for row in rows:
            if row["identifiability"] not in (None, "identifiable"):
                continue
            inside = row["truth_in_interval"]
            recovered.append(inside if inside is not None else abs(row["error_pct"]) < 5)
        sol["truth_recovered"] = all(recovered) if recovered else None
        sol["truth_check"] = ("truth inside the 95 % interval of every identifiable parameter (or within 5 % when "
                              "the method gives no interval)")
    return sol


def compare_methods(problem: InverseProblem, methods: Sequence[str] = ("least_squares", "adjoint", "autodiff",
                                                                         "bayesian"), *,
                    identifiability: bool | IdentifiabilityReport = True, pinn_steps: int = 1500, seed: int = 0,
                    progress: Callable[[str, int, int, float], None] | None = None) -> dict[str, Any]:
    """Run ``methods`` on the same problem; returns the identifiability report, one solution per method and a
    ranking by held-out RMSE (fit RMSE when there are no held-out measurements)."""
    from pinneapple_core.jsonable import to_jsonable

    t0 = time.time()
    rep = identifiability if isinstance(identifiability, IdentifiabilityReport) else (
        analyze_identifiability(problem.model, problem.observations, problem.parameters, profile=False,
                                posterior=False, fit_first=True) if identifiability else None)
    sols = []
    for m in methods:
        cb = (lambda s, t, v, _m=m: progress(_m, s, t, v)) if progress else None
        try:
            sols.append(solve_inverse(problem, m, identifiability=rep, pinn_steps=pinn_steps, seed=seed,
                                      progress=cb))
        except (ValueError, RuntimeError) as exc:
            sols.append({"method": m, "label": METHODS.get(m, m), "status": "error", "reason": str(exc)})
    done = [s for s in sols if s["status"] == "done"]
    key = "validation" if problem.validation is not None and problem.validation.n else "fit"
    def rank(s):                     # RMSE to 3 significant digits; a tie goes to the cheaper method
        r = s[key]["rmse"] if s.get(key) else math.inf
        return (float(f"{r:.3g}") if math.isfinite(r) else math.inf, s.get("forward_evaluations") or 0)

    ranking = sorted(done, key=rank)
    table = [{"method": s["method"], "label": s["label"], "status": s["status"],
              "fit_rmse": s.get("fit", {}).get("rmse") if s["status"] == "done" else None,
              "validation_rmse": (s.get("validation") or {}).get("rmse") if s["status"] == "done" else None,
              "seconds": s.get("seconds"), "forward_evaluations": s.get("forward_evaluations"),
              "truth_recovered": s.get("truth_recovered"), "reason": s.get("reason", "")} for s in sols]
    return to_jsonable({"problem": {"name": problem.name, "description": problem.description,
                                    "model": problem.model.describe(),
                                    "parameters": [p.to_dict() for p in problem.parameters],
                                    "observations": problem.observations.to_dict(),
                                    "validation": None if problem.validation is None else problem.validation.to_dict()},
                        "identifiability": None if rep is None else rep.to_dict(), "solutions": sols, "table": table,
                        "ranked_by": f"{key} RMSE (ties to 3 significant digits go to fewer model runs)", "best": ranking[0]["method"] if ranking else None,
                        "available": available_methods(problem), "seconds": time.time() - t0})

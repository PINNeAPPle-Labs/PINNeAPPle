"""Model calibration: fit model parameters to measurements, with honest uncertainty and a held-out check.

:func:`calibrate` does, on one model and one set of measurements:

1. splits the measurements into a fit set and a held-out set (given, a random fraction, or the last rows by an input
   such as time: an extrapolation test);
2. weighted least squares in the scaled parameters (trust-region reflective inside the bounds; robust losses on
   request);
3. **identifiability** (:func:`~.identifiability.analyze_identifiability` of the fit design at the calibrated
   values): a parameter the data cannot determine gets *no* confidence interval: it is reported "not estimable"
   with the reason and the combination the data do fix;
4. **confidence intervals** for the estimable parameters, two ways: from the Jacobian at the optimum
   (``cov = s^2 (J^T W J)^-1`` with ``s^2 = chi2/dof``, Student t quantile; symmetric in ln for log parameters) and
   by a residual bootstrap (refit on ``y_hat + resampled standardised residuals * sigma``, percentile interval);
5. residuals, metrics before (starting values) and after on the fit set and on the held-out measurements, and the
   prediction of every measurement before and after.

Every number comes from these runs; ``warnings`` lists what makes them doubtful (a parameter on its bound, a reduced
chi-square far from 1, held-out error much larger than the fit error, too few points for the bootstrap).
"""
from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .estimation import least_squares_fit
from .forward import ForwardModel, Observations, Parameter, ParameterSet, _predict
from .identifiability import IdentifiabilityReport, analyze_identifiability

__all__ = ["CalibrationResult", "calibrate", "residual_metrics"]


def residual_metrics(pred: np.ndarray, obs: Observations) -> dict[str, Any]:
    """RMSE, MAE, max |error|, bias, chi2 and chi2 per point of ``pred`` against ``obs.values`` (all rows, then per
    observable when there are several)."""
    if obs.n == 0:
        return {"n": 0}
    e = np.asarray(pred) - obs.values
    out = {"n": int(obs.n), "rmse": float(np.sqrt(np.mean(e**2))), "mae": float(np.mean(np.abs(e))),
           "max_abs": float(np.max(np.abs(e))), "bias": float(np.mean(e)),
           "chi2": float(np.sum((e / obs.sigma) ** 2)), "chi2_per_point": float(np.mean((e / obs.sigma) ** 2))}
    labels = sorted({str(o) for o in obs.observable})
    if len(labels) > 1:
        out["by_observable"] = {}
        for lab in labels:
            m = np.array([str(o) == lab for o in obs.observable])
            out["by_observable"][lab] = {"n": int(m.sum()), "rmse": float(np.sqrt(np.mean(e[m] ** 2))),
                                         "max_abs": float(np.max(np.abs(e[m])))}
    return out


@dataclass
class CalibrationResult:
    parameters: list[dict[str, Any]]
    fit: dict[str, Any]
    metrics: dict[str, dict[str, Any]]            # {"fit": {"before", "after"}, "holdout": {"before", "after"}}
    rows: dict[str, Any]                          # per measurement: inputs, observed, before, after, set
    identifiability: dict[str, Any]
    bootstrap: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def theta(self) -> dict[str, float]:
        return {p["name"]: p["calibrated"] for p in self.parameters}

    def to_dict(self) -> dict[str, Any]:
        from pinneapple_core.jsonable import to_jsonable

        return to_jsonable({k: getattr(self, k) for k in self.__dataclass_fields__})


def calibrate(model: ForwardModel, obs: Observations, parameters: Sequence[Parameter] | None = None, *,
              holdout: Observations | None = None, holdout_fraction: float = 0.0, holdout_by: str | None = None,
              loss: str = "linear", bootstrap: int = 100, confidence: float = 0.95, seed: int = 0,
              identifiability: IdentifiabilityReport | None = None, max_nfev: int = 300,
              absolute_sigma: bool = True, time_budget: float = 60.0) -> CalibrationResult:
    """Calibrate ``parameters`` (default the model's; ``value`` is the starting guess) against ``obs``.

    ``holdout`` (explicit validation measurements) or ``holdout_fraction`` (random rows, or the rows with the largest
    ``holdout_by`` input) are never used in the fit. ``bootstrap``: number of refits (0 to skip).
    ``absolute_sigma``: the sigma given is the real measurement noise (a sensor specification): Jacobian intervals
    use it, widened by ``sqrt(chi2/dof)`` when the residuals are larger, never narrowed. ``False`` when sigma is only
    a relative weight: the noise level is then estimated from the residuals (``s^2 = chi2/dof``)."""
    from scipy import stats

    t0 = time.time()
    if not obs.has_values:
        raise ValueError("calibration needs measured values")
    params = ParameterSet(parameters if parameters is not None else model.default_parameters())
    warnings: list[str] = []
    if holdout is None and holdout_fraction > 0:
        fit_obs, hold_obs = obs.split(holdout_fraction, seed=seed, by=holdout_by)
    else:
        fit_obs, hold_obs = obs, holdout
    p = len(params)
    if fit_obs.n <= p:
        raise ValueError(f"{fit_obs.n} measurements in the fit set for {p} free parameters: need more measurements "
                         f"than parameters")

    # 2. before
    z0 = params.z0()
    before_fit = _predict(model, params.theta(z0), fit_obs)
    before_hold = _predict(model, params.theta(z0), hold_obs) if hold_obs is not None else None

    # 3. fit
    res = least_squares_fit(model, params, fit_obs, loss=loss, max_nfev=max_nfev)
    after_fit = res.pred
    # identifiability of the fit design at the calibrated values (the Fisher information is local)
    if identifiability is None:
        at_fit = [Parameter.from_dict({**q.to_dict(), "value": res.theta[q.name]}) for q in params.all]
        identifiability = analyze_identifiability(model, fit_obs.with_values(np.full(fit_obs.n, np.nan)), at_fit,
                                                  profile=False, posterior=False, fit_first=False,
                                                  time_budget=min(20.0, time_budget / 3))
    ident = {q["name"]: q for q in identifiability.parameters}
    after_hold = _predict(model, res.theta, hold_obs) if hold_obs is not None else None
    dof = fit_obs.n - p
    s2 = res.chi2 / dof
    s2_used = max(s2, 1.0) if absolute_sigma else s2
    if res.at_bound:
        warnings.append(f"{', '.join(res.at_bound)} ended on a bound: widen the bounds or treat the value as a limit, "
                        f"not an estimate.")
    if s2 > 4:
        warnings.append(f"reduced chi-square {s2:.1f} (expected about 1): the residuals are larger than the stated "
                        f"noise, so the model misses something (or sigma is too small); intervals are widened by "
                        f"sqrt(chi2/dof).")
    elif s2 < 0.25:
        warnings.append(f"reduced chi-square {s2:.2f}: the residuals are smaller than the stated noise (sigma may be "
                        f"overstated)" + ("; intervals keep the stated sigma." if absolute_sigma else
                                          "; intervals are narrowed by sqrt(chi2/dof)."))

    # 4a. Jacobian intervals
    Jw = res.jac / fit_obs.sigma[:, None]
    tq = float(stats.t.ppf(0.5 + confidence / 2, max(dof, 1)))
    try:
        cov = s2_used * np.linalg.pinv(Jw.T @ Jw, rcond=1e-10, hermitian=True)
        sd = np.sqrt(np.maximum(np.diag(cov), 0))
        # correlations from a slightly regularised inverse: in a non-identifiable direction the pseudo-inverse
        # drops the direction (and can flip the sign); the ridge keeps it, giving +-1 along the ambiguity
        H = Jw.T @ Jw
        C = np.linalg.inv(H + 1e-10 * max(float(np.max(np.linalg.eigvalsh(H))), 1e-300) * np.eye(p))
        dd = np.sqrt(np.maximum(np.diag(C), 1e-300))
        corr = np.clip(C / np.outer(dd, dd), -1, 1)
    except np.linalg.LinAlgError:
        sd, corr = np.full(p, np.inf), np.full((p, p), np.nan)

    # 4b. bootstrap
    boot = {"n_requested": int(bootstrap), "n_done": 0}
    samples = []
    if bootstrap > 0:
        rng = np.random.default_rng(seed)
        std_res = (fit_obs.values - after_fit) / fit_obs.sigma
        std_res = std_res - std_res.mean()
        for b in range(bootstrap):
            if time.time() - t0 > time_budget:
                warnings.append(f"bootstrap stopped after {b} of {bootstrap} refits (time budget {time_budget:g} s).")
                break
            yb = after_fit + rng.choice(std_res, size=fit_obs.n, replace=True) * fit_obs.sigma
            rb = least_squares_fit(model, params, fit_obs.with_values(yb), z0=res.z, loss=loss, max_nfev=100)
            samples.append([rb.theta[n] for n in params.names])
        boot["n_done"] = len(samples)
    samples_a = np.array(samples) if samples else np.zeros((0, p))
    alpha = (1 - confidence) / 2

    out_params = []
    for i, par in enumerate(params.free):
        name = par.name
        st = ident[name]["status"] if name in ident else "identifiable"
        estimable = st == "identifiable"
        val = res.theta[name]
        row: dict[str, Any] = {"name": name, "unit": par.unit, "initial": par.value, "calibrated": val,
                               "lower": par.lower, "upper": par.upper, "identifiability": st,
                               "truth": par.truth, "at_bound": name in res.at_bound}
        if estimable and np.isfinite(sd[i]):
            if par.log:
                lo, hi = math.exp(res.z[i] - tq * sd[i]), math.exp(res.z[i] + tq * sd[i])
                se = val * sd[i]
            else:
                lo, hi = val - tq * sd[i] * par.scale, val + tq * sd[i] * par.scale
                se = sd[i] * par.scale
            row["ci_jacobian"] = [lo, hi]
            row["std_error"] = se
        else:
            row["ci_jacobian"] = None
            row["std_error"] = None
        if estimable and len(samples_a) >= 20:
            row["ci_bootstrap"] = [float(np.quantile(samples_a[:, i], alpha)),
                                   float(np.quantile(samples_a[:, i], 1 - alpha))]
        else:
            row["ci_bootstrap"] = None
        if len(samples_a):
            row["bootstrap_range"] = [float(samples_a[:, i].min()), float(samples_a[:, i].max())]
        if not estimable:
            amb = next((a for a in identifiability.ambiguities if name in a["parameters"]), None)
            row["not_estimable"] = ident[name]["reason"] + (f"; {amb['text']}" if amb else "")
        elif row["ci_bootstrap"] is None and bootstrap > 0:
            row["note"] = "too few bootstrap refits for an interval"
        out_params.append(row)
    if any(r["ci_jacobian"] is None for r in out_params):
        warnings.append("Parameters marked 'not estimable' have no confidence interval: the measurements cannot "
                        "determine them (see identifiability). Fix them, add the suggested measurements, or "
                        "calibrate only the combination the data determine.")

    m_fit = {"before": residual_metrics(before_fit, fit_obs), "after": residual_metrics(after_fit, fit_obs)}
    metrics = {"fit": m_fit}
    if hold_obs is not None and hold_obs.n:
        metrics["holdout"] = {"before": residual_metrics(before_hold, hold_obs),
                              "after": residual_metrics(after_hold, hold_obs)}
        rh, rf = metrics["holdout"]["after"]["rmse"], m_fit["after"]["rmse"]
        if rh > 3 * max(rf, 1e-300) and rh > 2 * float(np.median(hold_obs.sigma)):
            warnings.append(f"held-out RMSE {rh:.4g} is {rh / max(rf, 1e-300):.1f}x the fit RMSE: the calibrated "
                            f"model does not generalise to the held-out measurements.")
    else:
        warnings.append("No held-out measurements: the fit was not validated on data it did not see.")

    def rows_of(o: Observations, before, after, label):
        return {"inputs": {k: v.tolist() for k, v in o.inputs.items()}, "observable": [str(x) for x in o.observable],
                "observed": o.values.tolist(), "sigma": o.sigma.tolist(), "before": np.asarray(before).tolist(),
                "after": np.asarray(after).tolist(), "set": [label] * o.n}

    rows = rows_of(fit_obs, before_fit, after_fit, "fit")
    if hold_obs is not None and hold_obs.n:
        h = rows_of(hold_obs, before_hold, after_hold, "holdout")
        for k in ("observable", "observed", "sigma", "before", "after", "set"):
            rows[k] = rows[k] + h[k]
        for k in set(rows["inputs"]) | set(h["inputs"]):
            rows["inputs"][k] = rows["inputs"].get(k, [None] * fit_obs.n) + h["inputs"].get(k, [None] * hold_obs.n)
    return CalibrationResult(
        parameters=out_params,
        fit={"chi2": res.chi2, "dof": dof, "reduced_chi2": s2, "nfev": res.nfev, "success": res.success,
             "message": res.message, "loss": loss, "absolute_sigma": absolute_sigma, "noise_scale": math.sqrt(s2_used), "derivatives": res.derivatives, "seconds": res.seconds,
             "correlation": np.nan_to_num(corr).round(4).tolist(), "names": params.names,
             "confidence": confidence, "history": res.history[-200:]},
        metrics=metrics, rows=rows, identifiability=identifiability.to_dict(), bootstrap=boot, warnings=warnings,
        seconds=time.time() - t0)

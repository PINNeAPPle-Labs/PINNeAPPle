"""The standard inverse-modeling / parameter-discovery workflow (#1189): identifiability before the fit, estimation,
intervals, and the checks that say whether the estimate can be trusted.

The forward model is a library tool (or a callable) with keyword inputs: the measured conditions (columns of the
measurements), the parameters to find (``parameters``: name -> ``{lower, upper, initial}``) and fixed values.

1. **Identifiability first** (:func:`.identifiability.analyze_identifiability` at the initial guess, on the
   conditions of the measurements): the Fisher information's rank and conditioning, the parameter combinations the
   data cannot separate (named), and the D-optimal next measurements among ``candidates``. A structurally
   non-identifiable parameter is reported, never estimated.
2. **Estimation** (:func:`.calibration.calibrate`): least squares in log space for positive parameters, with
   ``holdout_fraction`` rows never used in the fit, bootstrap and Jacobian intervals.
3. **Checks**: reduced chi-square against the stated noise; the held-out rows; residual whiteness (Wald-Wolfowitz
   runs test of the residual signs, in the order of the first condition); no estimate at its bound; and a
   **synthetic recovery**: data generated at the estimate with the stated noise, refitted ``n_recovery`` times,
   must give intervals that cover the estimate at about their nominal rate.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np


def _runs_test(signs: np.ndarray) -> float:
    """Two-sided p-value of the Wald-Wolfowitz runs test on a sequence of +/- signs."""
    s = signs[signs != 0]
    n1, n2 = int((s > 0).sum()), int((s < 0).sum())
    n = n1 + n2
    if n1 == 0 or n2 == 0:
        return 0.0
    runs = 1 + int(np.sum(s[1:] != s[:-1]))
    mu = 2 * n1 * n2 / n + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n) / (n * n * (n - 1))
    z = (runs - mu) / math.sqrt(max(var, 1e-300))
    return float(math.erfc(abs(z) / math.sqrt(2)))


def inverse_workflow(*, tool: str | Callable[..., Any], measurements: Mapping[str, Any], output: str,
                     parameters: Mapping[str, Mapping[str, float]], fixed: Mapping[str, Any] | None = None,
                     candidates: Mapping[str, Sequence[float]] | None = None, holdout_fraction: float = 0.2,
                     bootstrap: int = 50, n_recovery: int = 10, seed: int = 0) -> dict[str, Any]:
    """``measurements``: ``{"inputs": {column: [...]}, "values": [...], "sigma": float | [...]}``."""
    from .calibration import calibrate
    from .forward import CallableModel, Observations, Parameter
    from .identifiability import analyze_identifiability
    if isinstance(tool, str):
        from pinneapple_autolab.study.verify import resolve_model
        fn = resolve_model(tool)
    else:
        fn = tool
    fixed = dict(fixed or {})
    cols = {k: np.asarray(v, float) for k, v in measurements["inputs"].items()}
    values = np.asarray(measurements["values"], float)
    sigma = np.broadcast_to(np.asarray(measurements.get("sigma", 1.0), float), values.shape).copy()

    def predict(theta, inputs):
        n = len(next(iter(inputs.values())))
        out = []
        for i in range(n):
            r = fn(**fixed, **{k: float(v[i]) for k, v in inputs.items()}, **{k: float(v) for k, v in theta.items()})
            out.append(float(r[output]) if isinstance(r, Mapping) else float(r))
        return np.array(out)

    def params_at(start: Mapping[str, float] | None = None):
        return [Parameter(name=k, value=float((start or {}).get(k, p.get("initial", 0.5 * (p["lower"] + p["upper"])))),
                          lower=float(p["lower"]), upper=float(p["upper"]), unit=str(p.get("unit", "")))
                for k, p in parameters.items()]
    P0 = params_at()
    model = CallableModel(predict, inputs=list(cols), outputs=[output], parameters=P0, name=str(tool))
    obs = Observations(inputs=cols, values=values, sigma=sigma)
    cand = Observations(inputs={k: np.asarray(v, float) for k, v in candidates.items()},
                        sigma=np.ones(len(next(iter(candidates.values()))))) if candidates else None
    ident = analyze_identifiability(model, Observations(inputs=cols, sigma=sigma), P0, candidates=cand,
                                    profile=False, posterior=False, fit_first=False, seed=seed)
    status = {p["name"]: p.get("status") for p in ident.parameters}
    checks: list[dict[str, Any]] = []

    def chk(code, ok, detail, value=None):
        ok = None if ok is None else bool(ok)
        checks.append({"code": code, "status": "pass" if ok is True else ("fail" if ok is False else "inconclusive"),
                       "detail": detail, "value": value})
    structural = ident.rank < ident.n_parameters
    chk("identify.structural", not structural, f"Fisher information rank {ident.rank} of {ident.n_parameters} at the "
        f"initial guess (condition number {ident.condition_number:.3g})" + (
            "; not separable: " + "; ".join(str(a.get("text") or a.get("fixed_combination") or a)
                                            for a in ident.ambiguities)
            if ident.ambiguities else ""), ident.rank)
    out: dict[str, Any] = {"identifiability": {"rank": ident.rank, "n_parameters": ident.n_parameters,
                                               "condition_number": ident.condition_number, "status": status,
                                               "ambiguities": ident.ambiguities,
                                               "suggestions": ident.suggestions[:3]}}
    if structural:
        out.update(estimated=False, checks=checks)
        return out
    cal = calibrate(model, obs, P0, holdout_fraction=holdout_fraction, bootstrap=bootstrap, seed=seed).to_dict()
    rows = {p["name"]: p for p in cal["parameters"]}
    est = {k: float(r["calibrated"]) for k, r in rows.items()}
    ci = {k: r.get("ci_jacobian") for k, r in rows.items()}
    chi2 = cal["fit"].get("reduced_chi2")
    chk("identify.fits_data", None if chi2 is None else chi2 <= 2.0, f"reduced chi-square {chi2:.3g} against the "
                                                                      "stated noise (<= 2)", chi2)
    hold = (cal.get("metrics") or {}).get("holdout", {}).get("after") or {}
    hz = hold.get("chi2_per_point")
    chk("identify.held_out", None if hz is None else hz <= 4.0, f"held-out rows: mean (residual/sigma)^2 {hz:.3g} "
                                                                "(<= 4)" if hz is not None else "no held-out rows", hz)
    at_bound = [k for k, r in rows.items() if r.get("at_bound")]
    chk("identify.inside_bounds", not at_bound, "every estimate inside its bounds" if not at_bound else
        f"{at_bound} at a bound: widen it or the data do not determine it")
    practical = [k for k, r in rows.items() if r.get("identifiability") not in (None, "identifiable")]
    chk("identify.practical", not practical, "every parameter practically identifiable" if not practical else
        f"practically non-identifiable: {practical} (intervals too wide or profile open)")
    res = values - predict(est, cols)
    order = np.argsort(next(iter(cols.values())), kind="stable")
    pv = _runs_test(np.sign(res[order]))
    chk("identify.white_residuals", pv >= 0.05, f"runs test of the residual signs along "
                                                f"{next(iter(cols))}: p = {pv:.3g} (>= 0.05: no structure left)", pv)
    # synthetic recovery at the estimate
    rng = np.random.default_rng(seed + 1)
    mean = predict(est, cols)
    cover = {k: 0 for k in est}
    done = 0
    for _ in range(n_recovery):
        yk = mean + sigma * rng.standard_normal(len(mean))
        try:
            r = calibrate(model, Observations(inputs=cols, values=yk, sigma=sigma), params_at(est),
                          bootstrap=0, seed=seed).to_dict()
        except Exception:  # noqa: BLE001 - a failed refit counts as a miss, not a crash
            continue
        done += 1
        for p in r["parameters"]:
            lo, hi = (p.get("ci_jacobian") or [np.nan, np.nan])
            cover[p["name"]] += int(lo <= est[p["name"]] <= hi)
    rate = {k: v / max(done, 1) for k, v in cover.items()}
    worst = min(rate.values()) if rate else 0.0
    nominal = 0.95
    floor = nominal - 2 * math.sqrt(nominal * (1 - nominal) / max(done, 1))
    chk("identify.recovery", done >= max(3, n_recovery // 2) and worst >= floor,
        f"{done} synthetic refits at the estimate: the 95 % intervals cover it in "
        + ", ".join(f"{k} {v:.0%}" for k, v in rate.items()) + f" (>= {floor:.0%} at that count)", rate)
    out.update(estimated=True, estimate=est, intervals=ci, reduced_chi2=chi2, holdout=hold, checks=checks,
               recovery=rate, n_measurements=int(len(values)))
    return out

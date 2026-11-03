"""The standard extreme-climate workflow (#1198): a station or gridded series -> extremes (block maxima, or peaks
over a threshold with declustering) -> GEV / GPD by maximum likelihood with profile-likelihood intervals ->
non-stationary location / scale on covariates selected by likelihood ratio -> return levels and periods, compound
events through a copula, a probability-ratio attribution -> diagnostics (QQ, return-level curve, Anderson-Darling)
and a coverage study on synthetic samples.

Protocol
--------
1. **Data**: a bundled case (``portpirie``, ``fremantle``, ``rain`` from Coles 2001, or ``synthetic``: a
   non-stationary GEV driven by a smooth warming covariate, with a second variable joined by a Gumbel copula of
   known parameter), or a CSV (``value_column``, optional ``time_column`` (years, or ISO dates for daily data),
   ``covariates`` and a ``second_column`` for compound events).
2. **Extremes**: ``block`` (the values are block maxima, or the maximum per year when the times are dates or the
   series is daily) or ``pot`` (exceedances of ``threshold``, chosen when 0 by the threshold-stability rule of
   :func:`.extraction.choose_threshold`, declustered by runs of length ``run``).
3. **Fit**: GEV (block) or GPD (POT) by maximum likelihood. For block maxima with covariates, each covariate is
   tried in the location, then in the log-scale, and kept when the likelihood-ratio test against the nested model
   has ``p < lr_alpha`` (forward selection).
4. **Return levels** for each period with the delta-method and the profile-likelihood intervals; for a
   non-stationary fit they are the effective levels at the ``reference`` covariates (default: the last
   observation's, the current climate).
5. **Compound events** (when there is a second variable): copula on ranks, the joint probability that both
   variables exceed their ``T``-year marginal levels, the joint return period against independence, and a
   bootstrap interval.
6. **Attribution** (when a covariate is selected and ``factual`` / ``counterfactual`` covariate values are given):
   ``PR = p_factual / p_counterfactual`` of exceeding ``event`` (default: the largest observation), and the change
   in the ``T``-year level, with parametric-bootstrap intervals (refits of samples drawn from the fitted model).
7. **Diagnostics and checks**: QQ on the Gumbel (GEV) or exponential (GPD) scale, Anderson-Darling with a
   parametric-bootstrap p-value, the bundled cases against the values published in Coles (2001), the stationary
   GEV against SciPy's independent fit, and a coverage study: ``calibration_replicates`` samples of the same size
   drawn from the fitted stationary model, refitted, counting how often the 95 % intervals hold the true
   parameters and the 100-year level (passes when the rate is inside the binomial 99.7 % band of the nominal).
"""
from __future__ import annotations

import csv
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy import stats

from .copula import fit_copula, gumbel_copula_sample, joint_exceedance
from .extraction import block_maxima, choose_threshold, decluster_runs, mean_excess
from .models import (
    GEVFit,
    GPDFit,
    anderson_darling,
    fit_gev,
    fit_gpd,
    gev_sample,
    gpd_sample,
    likelihood_ratio_test,
)

__all__ = ["extreme_climate_workflow", "load_case", "coverage_study", "CASES", "PUBLISHED", "as_tables"]

DATA = os.path.join(os.path.dirname(__file__), "data")

CASES = {
    "portpirie": "annual maximum sea level at Port Pirie, 1923-1987 (Coles 2001, §3.4.1)",
    "fremantle": "annual maximum sea level at Fremantle, 1897-1989, with the SOI (Coles 2001, §6.3.1)",
    "rain": "daily rainfall in south-west England, 1914-1962 (Coles 2001, §4.4.1)",
    "synthetic": "non-stationary GEV driven by a smooth warming covariate, with a Gumbel-copula partner variable",
}

# Values printed in Coles (2001); the tolerance is half a unit of the last printed digit plus the same again for
# the optimiser, so a rounding of the book's value passes and a different optimum does not.
PUBLISHED: dict[str, dict[str, tuple[float, float]]] = {
    "portpirie": {"mu0": (3.87, 0.01), "sigma": (0.198, 0.001), "xi": (-0.050, 0.001),
                  "return_level_10": (4.30, 0.01), "return_level_100": (4.69, 0.01),
                  "delta_lower_100": (4.38, 0.01), "delta_upper_100": (5.00, 0.01)},
    "fremantle": {"mu_year": (0.00203, 0.00001), "nll_stationary": (-43.6, 0.1), "nll_trend": (-49.9, 0.1)},
    "rain": {"sigma": (7.44, 0.01), "xi": (0.184, 0.001), "return_level_100": (106.3, 0.1)},
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _read_csv(path: str) -> dict[str, list[str]]:
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        raise ValueError(f"{path}: no data rows")
    head = [h.strip() for h in rows[0]]
    cols: dict[str, list[str]] = {h: [] for h in head}
    for r in rows[1:]:
        if not r:
            continue
        for h, v in zip(head, r, strict=False):
            cols[h].append(v.strip())
    return cols


def _floats(vals: Sequence[str]) -> np.ndarray:
    out = np.empty(len(vals))
    for i, v in enumerate(vals):
        try:
            out[i] = float(v)
        except ValueError:
            out[i] = np.nan
    return out


def _synthetic(seed: int) -> dict[str, Any]:
    """1950-2023 annual maxima: GMST-like covariate g(t) (0 -> about 1.2, smooth plus noise), maxima
    GEV(mu = 30 + 2 g, sigma = 2, xi = 0.1), and a partner variable (a drought index) with a Gumbel copula
    theta = 2 (Kendall tau 0.5) and a GEV(10, 1.5, -0.1) margin."""
    rng = np.random.default_rng(seed)
    years = np.arange(1950, 2024)
    g = 1.2 * ((years - 1950) / 73.0) ** 1.6 + rng.normal(0, 0.05, len(years))
    uv = gumbel_copula_sample(len(years), 2.0, rng)
    from .models import _gev_ppf
    x = _gev_ppf(uv[:, 0], 30.0 + 2.0 * g, 2.0, 0.1)
    y = _gev_ppf(uv[:, 1], 10.0, 1.5, -0.1)
    return {"values": x, "time": years.astype(float), "covariates": {"year": years.astype(float), "gmst": g},
            "second": y, "kind": "block", "units": "", "name": "synthetic",
            "truth": {"mu0": 30.0, "mu_gmst": 2.0, "sigma": 2.0, "xi": 0.1, "copula_theta": 2.0}}


def load_case(name: str, seed: int = 0) -> dict[str, Any]:
    """A bundled case as ``values``, ``time``, ``covariates``, ``kind`` (``block`` or ``pot``) and metadata."""
    if name == "synthetic":
        return _synthetic(seed)
    if name not in CASES:
        raise ValueError(f"unknown case {name!r}; choose one of {sorted(CASES)} or give csv_path")
    c = _read_csv(os.path.join(DATA, f"{name}.csv"))
    if name == "rain":
        v = _floats(c["rain_mm"])
        return {"values": v, "time": np.arange(len(v)) / 365.0 + 1914.0, "covariates": {}, "kind": "pot",
                "obs_per_year": 365.0, "units": "mm", "name": name, "daily": True}
    yr = _floats(c["year"])
    cov = {"year": yr}
    if "soi" in c:
        cov["soi"] = _floats(c["soi"])
    return {"values": _floats(c["sea_level_m"]), "time": yr, "covariates": cov, "kind": "block", "units": "m",
            "name": name}


def _load_csv(path: str, value_column: str, time_column: str, covariates: Sequence[str],
              second_column: str) -> dict[str, Any]:
    c = _read_csv(path)
    if value_column not in c:
        raise ValueError(f"{path}: no column {value_column!r} (columns: {list(c)})")
    v = _floats(c[value_column])
    out: dict[str, Any] = {"values": v, "covariates": {}, "kind": "block", "units": "", "name": os.path.basename(path)}
    if time_column:
        raw = c[time_column]
        t = _floats(raw)
        if np.all(np.isfinite(t)):
            out["time"] = t
        else:                                          # ISO dates: daily data, blocks are years
            yrs = np.array([float(s[:4]) for s in raw])
            out["time"], out["block_labels"], out["daily"] = yrs, yrs, True
    else:
        out["time"] = np.arange(len(v), dtype=float)
    for nm in covariates:
        if nm not in c:
            raise ValueError(f"{path}: no covariate column {nm!r}")
        out["covariates"][nm] = _floats(c[nm])
    if second_column:
        out["second"] = _floats(c[second_column])
    return out


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------

def _select_nonstationary(z, cov: Mapping[str, np.ndarray], names: Sequence[str], alpha: float
                          ) -> tuple[GEVFit, list[dict[str, Any]]]:
    """Forward selection: each covariate in the location, then in the log-scale, kept when the LR test against
    the current model has p < alpha."""
    cur = fit_gev(z)
    steps = [{"model": "stationary", "nll": cur.nll, "n_params": cur.n_params, "kept": True}]
    mu_c: list[str] = []
    sg_c: list[str] = []
    for where in ("mu", "sigma"):
        remaining = [c for c in names if c not in (mu_c if where == "mu" else sg_c)]
        while remaining:
            trials = []
            for c in remaining:
                m = mu_c + [c] if where == "mu" else mu_c
                s = sg_c + [c] if where == "sigma" else sg_c
                f = fit_gev(z, cov, mu=m, sigma=s, start=cur)
                trials.append((f.nll, c, f))
            nll, c, f = min(trials, key=lambda t: t[0])
            lr = likelihood_ratio_test(cur.nll, nll, 1)
            kept = lr["p_value"] < alpha
            label = f"{where} ~ " + " + ".join((mu_c if where == "mu" else sg_c) + [c])
            steps.append({"model": label, "nll": nll, "n_params": f.n_params, "kept": kept, **lr})
            if not kept:
                break
            cur = f
            (mu_c if where == "mu" else sg_c).append(c)
            remaining.remove(c)
    return cur, steps


def _gof_gev(fit: GEVFit, n_boot: int, rng) -> dict[str, Any]:
    n = fit.n
    r = np.sort(fit.gumbel_residuals())
    q = -np.log(-np.log(np.arange(1, n + 1) / (n + 1.0)))
    a2 = anderson_darling(fit.cdf_of_data())
    boot = []
    mu = fit._design(fit.mu_covariates, fit.covariates, n) @ fit.theta[: fit.k_mu]
    sg = np.exp(fit._design(fit.sigma_covariates, fit.covariates, n) @ fit.theta[fit.k_mu: fit.k_mu + fit.k_sigma])
    for _ in range(n_boot):
        zb = gev_sample(n, mu, sg, fit.xi, rng)
        try:
            fb = fit_gev(zb, fit.covariates, mu=fit.mu_covariates, sigma=fit.sigma_covariates)
        except ValueError:
            continue
        boot.append(anderson_darling(fb.cdf_of_data()))
    p = (1 + np.sum(np.asarray(boot) >= a2)) / (1 + len(boot)) if boot else float("nan")
    return {"scale": "gumbel", "empirical": r.tolist(), "model": q.tolist(),
            "qq_correlation": float(np.corrcoef(q, r)[0, 1]), "qq_max_abs": float(np.max(np.abs(q - r))),
            "anderson_darling": a2, "ad_p_value": float(p), "ad_bootstrap": len(boot)}


def _gof_gpd(fit: GPDFit, n_boot: int, rng) -> dict[str, Any]:
    n = fit.n_exceed
    u = np.sort(fit.cdf_of_data())
    e = -np.log1p(-np.clip(u, 0, 1 - 1e-16))                    # standard exponential scale
    q = -np.log1p(-np.arange(1, n + 1) / (n + 1.0))
    a2 = anderson_darling(u)
    boot = []
    for _ in range(n_boot):
        yb = gpd_sample(n, fit.sigma, fit.xi, rng)
        try:
            fb = fit_gpd(yb, 0.0, already_excesses=True)
        except ValueError:
            continue
        boot.append(anderson_darling(fb.cdf_of_data()))
    p = (1 + np.sum(np.asarray(boot) >= a2)) / (1 + len(boot)) if boot else float("nan")
    return {"scale": "exponential", "empirical": e.tolist(), "model": q.tolist(),
            "qq_correlation": float(np.corrcoef(q, e)[0, 1]), "qq_max_abs": float(np.max(np.abs(q - e))),
            "anderson_darling": a2, "ad_p_value": float(p), "ad_bootstrap": len(boot)}


def _binomial_band(p: float, n: int) -> tuple[float, float]:
    s = math.sqrt(p * (1 - p) / n)
    return p - 3 * s, min(1.0, p + 3 * s)


def coverage_study(mu: float, sigma: float, xi: float, n: int, replicates: int = 200, level: float = 0.95,
                   T: float = 100.0, seed: int = 0) -> dict[str, Any]:
    """Coverage of the Wald intervals of (mu, sigma, xi) and of the profile interval of the ``T``-block return
    level, over ``replicates`` GEV samples of size ``n`` with known parameters; ``band`` is the binomial 99.7 %
    band of the nominal rate for that many replicates."""
    from .models import gev_return_level
    rng = np.random.default_rng(seed)
    true = {"mu0": float(mu), "sigma": float(sigma), "xi": float(xi),
            f"return_level_{int(T)}": float(gev_return_level(T, mu, sigma, xi))}
    z = stats.norm.ppf(0.5 + level / 2)
    hits = {k: 0 for k in true}
    done = 0
    for _ in range(replicates):
        f = fit_gev(gev_sample(n, mu, sigma, xi, rng))
        c, se = f.coefficients(), f.standard_errors()
        for k in ("mu0", "sigma", "xi"):
            hits[k] += int(abs(c[k] - true[k]) <= z * se[k])
        lo, hi = f.profile_return_level(T, level=level)
        hits[f"return_level_{int(T)}"] += int(lo <= true[f"return_level_{int(T)}"] <= hi)
        done += 1
    band = _binomial_band(level, done)
    return {"replicates": done, "n": int(n), "truth": true, "nominal": level, "band": list(band),
            "coverage": {k: v / done for k, v in hits.items()},
            "interval": {"mu0": "wald", "sigma": "wald", "xi": "wald", f"return_level_{int(T)}": "profile"}}


def _calibration_gev(fit: GEVFit, replicates: int, level: float, rng) -> dict[str, Any]:
    """:func:`coverage_study` at the fit's stationary parameters (covariates at their means) and size."""
    mu = float(fit.mu({c: np.mean(fit.covariates[c]) for c in fit.mu_covariates})[0])
    sg = float(fit.sigma({c: np.mean(fit.covariates[c]) for c in fit.sigma_covariates})[0])
    return coverage_study(mu, sg, fit.xi, fit.n, replicates, level, seed=int(rng.integers(2 ** 31)))


def _calibration_gpd(fit: GPDFit, m: float, replicates: int, level: float, rng) -> dict[str, Any]:
    true = {"sigma": fit.sigma, "xi": fit.xi, "return_level": float(fit.return_level(m))}
    z = stats.norm.ppf(0.5 + level / 2)
    hits = {k: 0 for k in true}
    done = 0
    for _ in range(replicates):
        y = gpd_sample(fit.n_exceed, fit.sigma, fit.xi, rng)
        f = fit_gpd(y + fit.threshold, fit.threshold, n_obs=fit.n_obs)
        se = np.sqrt(np.diag(f.cov))
        hits["sigma"] += int(abs(f.sigma - true["sigma"]) <= z * se[0])
        hits["xi"] += int(abs(f.xi - true["xi"]) <= z * se[1])
        f.rate = fit.rate                           # the rate is held at the data's (its variance is separate)
        lo, hi = f.profile_return_level(m, level=level)
        hits["return_level"] += int(lo <= true["return_level"] <= hi)
        done += 1
    band = _binomial_band(level, done)
    return {"replicates": done, "n": fit.n_exceed, "truth": true, "nominal": level, "band": list(band),
            "coverage": {k: v / done for k, v in hits.items()},
            "interval": {"sigma": "wald", "xi": "wald", "return_level": "profile"}}


def _attribution(fit: GEVFit, factual: Mapping[str, float], counterfactual: Mapping[str, float], event: float,
                 T: float, n_boot: int, level: float, rng) -> dict[str, Any]:
    def measure(f: GEVFit):
        p1, p0 = f.exceedance_probability(event, factual), f.exceedance_probability(event, counterfactual)
        pr = p1 / p0 if p0 > 0 else np.inf
        dI = float(f.return_level(T, factual) - f.return_level(T, counterfactual))
        return p1, p0, pr, dI
    p1, p0, pr, dI = measure(fit)
    n = fit.n
    mu = fit._design(fit.mu_covariates, fit.covariates, n) @ fit.theta[: fit.k_mu]
    sg = np.exp(fit._design(fit.sigma_covariates, fit.covariates, n) @ fit.theta[fit.k_mu: fit.k_mu + fit.k_sigma])
    B = []
    for _ in range(n_boot):
        zb = gev_sample(n, mu, sg, fit.xi, rng)
        try:
            fb = fit_gev(zb, fit.covariates, mu=fit.mu_covariates, sigma=fit.sigma_covariates, start=fit)
        except ValueError:
            continue
        B.append(measure(fb))
    B = np.asarray(B)
    a = (1 - level) / 2
    lpr = np.log(np.clip(B[:, 2], 1e-300, 1e300))
    ci_pr = np.exp(np.quantile(lpr, [a, 1 - a])) if len(B) else [np.nan, np.nan]
    ci_di = np.quantile(B[:, 3], [a, 1 - a]) if len(B) else [np.nan, np.nan]
    ci_p1 = np.quantile(B[:, 0], [a, 1 - a]) if len(B) else [np.nan, np.nan]
    ci_p0 = np.quantile(B[:, 1], [a, 1 - a]) if len(B) else [np.nan, np.nan]
    return {"event": float(event), "factual": dict(factual), "counterfactual": dict(counterfactual),
            "p_factual": p1, "p_counterfactual": p0, "probability_ratio": float(pr),
            "probability_ratio_interval": [float(v) for v in ci_pr], "p_factual_interval": [float(v) for v in ci_p1],
            "p_counterfactual_interval": [float(v) for v in ci_p0], "return_period": T,
            "intensity_change": dI, "intensity_change_interval": [float(v) for v in ci_di], "bootstrap": int(len(B))}


def _compound(x, y, return_periods, n_boot: int, level: float, rng, events_per_year: float = 1.0
              ) -> dict[str, Any]:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    cf = fit_copula(x, y)
    rows = []
    boots = []
    n = len(x)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        try:
            boots.append(fit_copula(x[i], y[i]))
        except (ValueError, FloatingPointError):
            continue
    a = (1 - level) / 2
    for T in return_periods:
        u = 1.0 - 1.0 / T
        p = float(np.ravel(joint_exceedance(cf, u, u))[0])
        pb = np.array([float(np.ravel(joint_exceedance(b, u, u))[0]) for b in boots]) if boots else np.array([np.nan])
        lo, hi = np.quantile(pb, [a, 1 - a])
        rows.append({"T": float(T), "p_and": p, "p_and_interval": [float(lo), float(hi)],
                     "joint_return_period": float(1.0 / (events_per_year * p)) if p > 0 else float("inf"),
                     "joint_return_period_interval": [float(1 / (events_per_year * hi)) if hi > 0 else float("inf"),
                                                      float(1 / (events_per_year * lo)) if lo > 0 else float("inf")],
                     "independent_return_period": float(T * T / events_per_year)})
    fams = [b.family for b in boots]
    th = np.array([b.params.get(cf.family, np.nan) for b in boots]) if cf.family != "independence" else np.array([])
    th = th[np.isfinite(th)]
    ci = [float(v) for v in np.quantile(th, [a, 1 - a])] if len(th) else [float("nan")] * 2
    return {"copula": cf.to_dict(), "param_interval": ci, "joint": rows, "bootstrap": len(boots),
            "bootstrap_family_share": {f: fams.count(f) / len(fams) for f in set(fams)} if fams else {}}


def _check_published(name: str, measured: Mapping[str, float]) -> list[dict[str, Any]]:
    out = []
    for k, (val, tol) in PUBLISHED.get(name, {}).items():
        m = measured.get(k)
        if m is None:
            continue
        ok = abs(m - val) <= tol
        out.append({"code": f"published.{k}", "status": "pass" if ok else "fail",
                    "detail": f"{m:.6g} against {val:g} printed in Coles (2001) (allowed {tol:g})"})
    return out


# ---------------------------------------------------------------------------
# The workflow
# ---------------------------------------------------------------------------

def extreme_climate_workflow(*, dataset: str = "fremantle", csv_path: str = "", value_column: str = "",
                             time_column: str = "", covariates: Sequence[str] = (), second_column: str = "",
                             method: str = "auto", threshold: float = 0.0, run: int = 1,
                             obs_per_year: float = 0.0, return_periods: Sequence[float] = (2, 5, 10, 20, 50, 100),
                             level: float = 0.95, lr_alpha: float = 0.05,
                             reference: Mapping[str, float] | None = None,
                             factual: Mapping[str, float] | None = None,
                             counterfactual: Mapping[str, float] | None = None, event: float = 0.0,
                             gof_bootstrap: int = 199, attribution_bootstrap: int = 300, compound_bootstrap: int = 200,
                             calibration_replicates: int = 100, seed: int = 0) -> dict[str, Any]:
    """Run the extreme-value pipeline of the module docstring and return every result and check."""
    rng = np.random.default_rng(seed)
    if csv_path:
        d = _load_csv(csv_path, value_column, time_column, covariates, second_column)
    else:
        d = load_case(dataset, seed)
        if covariates:
            d["covariates"] = {k: v for k, v in d["covariates"].items() if k in covariates}
    name = d["name"]
    kind = d["kind"] if method == "auto" else method
    if kind not in ("block", "pot"):
        raise ValueError("method must be 'auto', 'block' or 'pot'")
    values = np.asarray(d["values"], float)
    checks: list[dict[str, Any]] = []
    out: dict[str, Any] = {"data": {"name": name, "description": CASES.get(name, csv_path), "n": int(len(values)),
                                    "missing": int(np.sum(~np.isfinite(values))), "units": d.get("units", "")},
                           "method": kind, "level": level}
    T = [float(t) for t in return_periods]
    if any(t <= 1 for t in T):
        raise ValueError("return periods must exceed 1 block")
    measured: dict[str, float] = {}

    if kind == "block":
        cov = {k: np.asarray(v, float) for k, v in d["covariates"].items()}
        if d.get("daily"):
            labels = d.get("block_labels", np.floor(d["time"]))
            bm = block_maxima(values, labels, min_fraction=0.5)
            z = bm["maximum"]
            cov = {k: np.array([np.nanmean(v[labels == b]) for b in bm["block"]]) for k, v in cov.items()}
            cov.setdefault("year", bm["block"].astype(float))
            out["extraction"] = {"blocks": int(len(z)), "rule": "maximum per year (years with >= 50 % of the "
                                                                "largest year's valid days)"}
        else:
            ok = np.isfinite(values) & np.all([np.isfinite(v) for v in cov.values()], axis=0) if cov else \
                np.isfinite(values)
            z = values[ok]
            cov = {k: v[ok] for k, v in cov.items()}
            out["extraction"] = {"blocks": int(len(z)), "rule": "the values are block maxima"}
        names = list(cov)
        fit, steps = _select_nonstationary(z, cov, names, lr_alpha) if names else (fit_gev(z), [])
        stationary = fit_gev(z)
        out["selection"] = steps
        coefs, ses = fit.coefficients(), fit.standard_errors()
        out["fit"] = {"distribution": "GEV", "coefficients": coefs, "standard_errors": ses, "nll": fit.nll,
                      "n_params": fit.n_params, "mu_covariates": list(fit.mu_covariates),
                      "sigma_covariates": list(fit.sigma_covariates), "converged": fit.converged,
                      "shape_profile_interval": list(fit.profile_shape(level))}
        if reference is None:
            ref = {c: float(cov[c][-1]) for c in names}
        else:
            ref = {c: float(reference.get(c, cov[c][-1])) for c in names}
        out["reference"] = ref
        zq = stats.norm.ppf(0.5 + level / 2)
        rl = []
        for t in T:
            zt = float(fit.return_level(t, ref))
            se = fit.return_level_se(t, ref)
            plo, phi = fit.profile_return_level(t, ref, level)
            rl.append({"T": t, "level": zt, "delta": [zt - zq * se, zt + zq * se], "profile": [plo, phi]})
        out["return_levels"] = rl
        grid = np.geomspace(1.05, 1000.0, 60)
        lv = np.array([float(fit.return_level(t, ref)) for t in grid])
        se_g = np.array([fit.return_level_se(t, ref) for t in grid])
        curve = {"T": grid.tolist(), "level": lv.tolist(), "lower": (lv - zq * se_g).tolist(),
                 "upper": (lv + zq * se_g).tolist(), "band": "delta method"}
        if fit.stationary:
            zs = np.sort(z)
            curve["empirical_T"] = (1.0 / (1.0 - np.arange(1, len(z) + 1) / (len(z) + 1.0))).tolist()
            curve["empirical_level"] = zs.tolist()
        out["return_level_curve"] = curve
        gof = _gof_gev(fit, gof_bootstrap, rng)
        out["diagnostics"] = gof
        sc = stationary.coefficients()
        c_, loc, sc_ = stats.genextreme.fit(z)
        diffs = {"mu0": abs(loc - sc["mu0"]) / abs(sc["sigma"]), "sigma": abs(sc_ - sc["sigma"]) / sc["sigma"],
                 "xi": abs(-c_ - sc["xi"])}
        nll_scipy = -np.sum(stats.genextreme.logpdf(z, c_, loc, sc_))
        ok = stationary.nll <= nll_scipy + 1e-6 or max(diffs.values()) < 1e-3
        checks.append({"code": "fit.scipy_cross_check", "status": "pass" if ok else "fail",
                       "detail": f"stationary GEV against scipy.stats.genextreme.fit: nll {stationary.nll:.6f} vs "
                                 f"{nll_scipy:.6f}; |d mu|/sigma {diffs['mu0']:.1e}, |d sigma|/sigma "
                                 f"{diffs['sigma']:.1e}, |d xi| {diffs['xi']:.1e}"})
        measured.update(sc)
        for r in rl:
            measured[f"return_level_{int(r['T'])}"] = r["level"] if fit.stationary else \
                float(stationary.return_level(r["T"]))
        if fit.stationary:
            for r in rl:
                measured[f"delta_lower_{int(r['T'])}"], measured[f"delta_upper_{int(r['T'])}"] = r["delta"]
        else:
            for r in T:
                s100 = stationary.return_level_se(r)
                measured[f"delta_lower_{int(r)}"] = float(stationary.return_level(r)) - zq * s100
                measured[f"delta_upper_{int(r)}"] = float(stationary.return_level(r)) + zq * s100
        measured["nll_stationary"] = stationary.nll
        for s_ in steps:
            if s_["model"] == "mu ~ year":
                tr = fit_gev(z, cov, mu=["year"], start=stationary)
                measured["mu_year"], measured["nll_trend"] = tr.coefficients()["mu_year"], tr.nll
        # attribution
        sel = list(dict.fromkeys(list(fit.mu_covariates) + list(fit.sigma_covariates)))
        if sel and (factual or counterfactual or name == "synthetic"):
            if name == "synthetic" and not factual:
                factual, counterfactual = {"gmst": 1.2, "year": 2023.0}, {"gmst": 0.0, "year": 1950.0}
            fa = {c: float((factual or {}).get(c, ref[c])) for c in sel}
            cf = {c: float((counterfactual or {}).get(c, ref[c])) for c in sel}
            ev = float(event) if event else float(np.max(z))
            out["attribution"] = _attribution(fit, fa, cf, ev, max(T), attribution_bootstrap, level, rng)
        elif factual or counterfactual:
            out["attribution"] = {"skipped": "no covariate was selected, so the two climate states have the same "
                                             "distribution (probability ratio 1 by construction)"}
        if calibration_replicates > 0:
            out["calibration"] = _calibration_gev(fit, calibration_replicates, level, rng)
    else:
        good = np.isfinite(values)
        x = np.where(good, values, -np.inf)
        if threshold and threshold > 0:
            u = float(threshold)
            thr = {"threshold": u, "rule": "given"}
        else:
            ch = choose_threshold(values, run=run)
            u = ch["threshold"]
            thr = {"threshold": u, "rule": ch["rule"], "stability": ch["stability"]}
        me = mean_excess(values, np.quantile(values[good], np.linspace(0.8, 0.995, 25)))
        thr["mean_excess"] = {k: v.tolist() for k, v in me.items()}
        dc = decluster_runs(x, u, run)
        thr.update({"n_exceed": dc["n_exceed"], "n_clusters": dc["n_clusters"], "extremal_index": dc["extremal_index"],
                    "run": run})
        out["extraction"] = thr
        n_obs = int(good.sum())
        fit = fit_gpd(dc["peaks"], u, n_obs=n_obs)
        npy = float(obs_per_year or d.get("obs_per_year") or 365.25)
        se = np.sqrt(np.diag(fit.cov))
        out["fit"] = {"distribution": "GPD", "threshold": u, "coefficients": {"sigma": fit.sigma, "xi": fit.xi},
                      "standard_errors": {"sigma": float(se[0]), "xi": float(se[1])}, "nll": fit.nll,
                      "rate": fit.rate, "obs_per_year": npy, "converged": fit.converged,
                      "modified_scale": fit.modified_scale, "shape_profile_interval": list(fit.profile_shape(level))}
        zq = stats.norm.ppf(0.5 + level / 2)
        rl = []
        for t in T:
            m = t * npy
            xm = float(fit.return_level(m))
            s = fit.return_level_se(m)
            plo, phi = fit.profile_return_level(m, level)
            rl.append({"T": t, "level": xm, "delta": [xm - zq * s, xm + zq * s], "profile": [plo, phi]})
        out["return_levels"] = rl
        grid = np.geomspace(max(1.0 / (fit.rate * npy), 0.1) * 1.05, 1000.0, 60)
        lv = np.array([float(fit.return_level(t * npy)) for t in grid])
        se_g = np.array([fit.return_level_se(t * npy) for t in grid])
        years = n_obs / npy
        pk = np.sort(dc["peaks"])[::-1]
        out["return_level_curve"] = {"T": grid.tolist(), "level": lv.tolist(), "lower": (lv - zq * se_g).tolist(),
                                     "upper": (lv + zq * se_g).tolist(), "band": "delta method",
                                     "empirical_T": (years / np.arange(1, len(pk) + 1)).tolist(),
                                     "empirical_level": pk.tolist()}
        out["diagnostics"] = _gof_gpd(fit, gof_bootstrap, rng)
        if name == "rain":                 # the book's set-up: every exceedance of 30 mm, no declustering
            book = fit_gpd(values[good], 30.0, n_obs=n_obs)
            measured.update({"sigma": book.sigma, "xi": book.xi,
                             "return_level_100": float(book.return_level(100 * 365.0))})
        if factual or counterfactual:
            out["attribution"] = {"skipped": "the POT fit is stationary here: give block maxima with covariates "
                                             "for an attribution"}
        if calibration_replicates > 0:
            out["calibration"] = _calibration_gpd(fit, 100 * npy, calibration_replicates, level, rng)

    # compound events
    if d.get("second") is not None:
        second = np.asarray(d["second"], float)
        # the first variable enters through its fitted distribution (G_i(z_i)): with covariates this removes the
        # trend, which would otherwise dilute the dependence measured on the ranks
        first = fit.cdf_of_data() if kind == "block" and not d.get("daily") and len(fit.data) == len(values) \
            else None
        if first is not None and len(first) == len(second):
            out["compound"] = _compound(first, second, T, compound_bootstrap, level, rng)
        else:
            out["compound"] = {"skipped": "a compound analysis needs one pair per block (annual values)"}

    # ---- checks ---------------------------------------------------------------------------------------------
    checks.append({"code": "fit.converged", "status": "pass" if out["fit"]["converged"] else "fail",
                   "detail": f"maximum likelihood, nll {out['fit']['nll']:.6g}"})
    xi = out["fit"]["coefficients"]["xi"]
    checks.append({"code": "fit.shape_range", "status": "pass" if -0.5 < xi < 0.5 else "warn",
                   "detail": f"xi = {xi:.3f}: the MLE's usual asymptotics hold for xi > -0.5; |xi| >= 0.5 is "
                             "unusual for climate maxima"})
    mono = all(a["level"] < b["level"] for a, b in zip(rl, rl[1:], strict=False)) if len(rl) > 1 else True
    inside = all(r["profile"][0] <= r["level"] <= r["profile"][1] and r["delta"][0] <= r["level"] <= r["delta"][1]
                 for r in rl)
    checks.append({"code": "return_levels.consistent", "status": "pass" if mono and inside else "fail",
                   "detail": "levels increase with the period and every interval holds its estimate"
                   if mono and inside else "levels not increasing or an interval misses its estimate"})
    g = out["diagnostics"]
    checks.append({"code": "gof.anderson_darling", "status": "pass" if g["ad_p_value"] > 0.05 else "fail",
                   "detail": f"A^2 = {g['anderson_darling']:.3f}, parametric-bootstrap p = {g['ad_p_value']:.3f} "
                             f"({g['ad_bootstrap']} refits)"})
    checks.append({"code": "gof.qq", "status": "pass" if g["qq_correlation"] > 0.97 else "warn",
                   "detail": f"QQ correlation on the {g['scale']} scale {g['qq_correlation']:.4f}, largest gap "
                             f"{g['qq_max_abs']:.3f}"})
    if "calibration" in out:
        c = out["calibration"]
        lo, hi = c["band"]
        for k, v in c["coverage"].items():
            checks.append({"code": f"calibration.{k}", "status": "pass" if lo <= v <= hi else "fail",
                           "detail": f"{v:.3f} of {c['replicates']} synthetic samples (n = {c['n']}) inside the "
                                     f"{c['interval'][k]} {level:.0%} interval; binomial band [{lo:.3f}, {hi:.3f}]"})
    if "selection" in out and len(out["selection"]) > 1:
        kept = [s["model"] for s in out["selection"][1:] if s["kept"]]
        detail = "; ".join(f"{s['model']}: D = {s['deviance']:.2f}, p = {s['p_value']:.2g}"
                           f"{' kept' if s['kept'] else ' rejected'}" for s in out["selection"][1:])
        checks.append({"code": "nonstationarity.selection", "status": "pass", "detail":
                       (f"kept {', '.join(kept)}; " if kept else "stationary kept; ") + detail})
    if name == "synthetic":
        tr = d["truth"]
        cf_ = out["fit"]["coefficients"]
        se_ = out["fit"]["standard_errors"]
        for k in ("mu_gmst", "xi"):
            if k in cf_:
                ok = abs(cf_[k] - tr[k]) <= 3 * se_[k]
                checks.append({"code": f"synthetic.{k}", "status": "pass" if ok else "fail",
                               "detail": f"{cf_[k]:.3f} +- {se_[k]:.3f} against the true {tr[k]:g}"})
        if "gmst" not in out["fit"]["mu_covariates"]:
            checks.append({"code": "synthetic.trend_found", "status": "fail",
                           "detail": "the selection did not keep the warming covariate in the location"})
        if "attribution" in out and "probability_ratio" in out["attribution"]:
            a = out["attribution"]
            from .models import _gev_cdf
            p1 = 1 - _gev_cdf(a["event"], tr["mu0"] + tr["mu_gmst"] * a["factual"]["gmst"], tr["sigma"], tr["xi"])
            p0 = 1 - _gev_cdf(a["event"], tr["mu0"] + tr["mu_gmst"] * a["counterfactual"]["gmst"], tr["sigma"],
                              tr["xi"])
            pr_true = float(p1 / p0)
            a["probability_ratio_true"] = pr_true
            lo_, hi_ = a["probability_ratio_interval"]
            checks.append({"code": "synthetic.probability_ratio", "status": "pass" if lo_ <= pr_true <= hi_ else
                           "fail", "detail": f"PR {a['probability_ratio']:.2f} [{lo_:.2f}, {hi_:.2f}] against the "
                                            f"true {pr_true:.2f}"})
        if "compound" in out and "copula" in out["compound"]:
            cp = out["compound"]["copula"]
            th = cp["params"]["gumbel"]
            lo_, hi_ = out["compound"]["param_interval"]
            tau = cp["kendall_tau"]
            ok = cp["family"] == "gumbel" and lo_ <= tr["copula_theta"] <= hi_
            checks.append({"code": "synthetic.copula", "status": "pass" if ok else "fail",
                           "detail": f"{cp['family']} chosen by AIC; Gumbel theta {th:.2f} [{lo_:.2f}, {hi_:.2f}] "
                                     f"against the true 2 (Kendall tau {tau:.2f} against 0.5)"})
    checks.extend(_check_published(name, measured))
    if "attribution" in out and "probability_ratio" in out["attribution"]:
        a = out["attribution"]
        checks.append({"code": "attribution.interval", "status": "pass" if a["bootstrap"] >= 50 else "warn",
                       "detail": f"PR {a['probability_ratio']:.3g} with a {level:.0%} interval from "
                                 f"{a['bootstrap']} parametric-bootstrap refits"})
    out["checks"] = checks
    out["passed"] = all(c["status"] != "fail" for c in checks)
    last = rl[-1]
    unit = d.get("units", "")
    s = (f"{name}: {out['fit']['distribution']}"
         + (f" with {' + '.join(out['fit'].get('mu_covariates', []))} in the location"
            if out["fit"].get("mu_covariates") else "")
         + f", xi = {xi:.3f}; {last['T']:g}-year level {last['level']:.4g} {unit} "
         f"(profile [{last['profile'][0]:.4g}, {last['profile'][1]:.4g}])")
    if "attribution" in out and "probability_ratio" in out["attribution"]:
        a = out["attribution"]
        s += (f"; probability ratio {a['probability_ratio']:.3g} [{a['probability_ratio_interval'][0]:.3g}, "
              f"{a['probability_ratio_interval'][1]:.3g}]")
    n_fail = sum(c["status"] == "fail" for c in checks)
    s += f"; {len(checks) - n_fail}/{len(checks)} checks pass"
    out["summary"] = s
    return out


def as_tables(result: Mapping[str, Any]) -> dict[str, list[list[Any]]]:
    """Rows for the return-level, selection, compound and check tables of a workflow result."""
    t = {"return_levels": [[r["T"], round(r["level"], 4), f"[{r['delta'][0]:.4g}, {r['delta'][1]:.4g}]",
                            f"[{r['profile'][0]:.4g}, {r['profile'][1]:.4g}]"] for r in result["return_levels"]],
         "checks": [[c["code"], c["status"], c["detail"]] for c in result["checks"]]}
    if result.get("selection"):
        t["selection"] = [[s["model"], round(s["nll"], 4), s["n_params"], round(s.get("deviance", 0.0), 3),
                           s.get("p_value", ""), "yes" if s["kept"] else "no"] for s in result["selection"]]
    if result.get("compound", {}).get("joint"):
        t["compound"] = [[r["T"], f"{r['p_and']:.3g}", f"{r['joint_return_period']:.4g}",
                          f"[{r['joint_return_period_interval'][0]:.4g}, {r['joint_return_period_interval'][1]:.4g}]",
                          r["independent_return_period"]] for r in result["compound"]["joint"]]
    return t

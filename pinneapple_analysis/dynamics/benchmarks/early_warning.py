"""Early-warning benchmarks: estimator accuracy, detection rates near bifurcations, false-positive rates of controls.

| Case | Reference | Source |
|---|---|---|
| ``ou_ac1`` / ``ou_variance`` | ``exp(-theta dt)``, ``sigma^2 / (2 theta)`` | stationary Ornstein-Uhlenbeck, exact |
| ``detect_<kind>`` | detection rate (variance and AC1 both rising significantly) | stochastic normal forms driven towards, not past, the bifurcation |
| ``fp_stable`` | the nominal level ``alpha`` | constant stability: a valid test rejects about ``alpha`` of the time |
| ``fp_trend_raw`` / ``fp_trend_loess`` | – / ``alpha`` | accelerating trend in the mean, without and with detrending |
| ``fp_growing_noise_variance`` / ``_ac1`` | – / ``alpha`` | constant stability, noise amplitude x3: variance rises, AC1 should not |
| ``detect_ou_obs_noise_*`` | – | the decreasing-stability OU with added white measurement noise |
| ``detect_ou_subsampled_*`` | – | the same series sampled every 4th / 16th point |

Rows marked "–" are measurements without a pass/fail reference: they show what happens, they are not a claim.
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..early_warning import EarlyWarningAnalyzer, ramp_series
from ._case import BenchmarkCase

__all__ = ["run"]
SIGMA = {"saddle_node": 0.002, "transcritical": 0.003, "pitchfork": 0.05, "ou_decreasing": 0.05}


def _rate(kind: str, reps: int, n_sur: int, seed: int, *, detrend: str = "loess", which=("variance", "ac1"),
          obs_noise: float = 0.0, every: int = 1, **kw) -> tuple[float, int, float]:
    an = EarlyWarningAnalyzer(detrend=detrend, n_surrogates=n_sur)
    hits = 0
    t0 = time.perf_counter()
    for r in range(reps):
        s = ramp_series(kind, seed=seed + 1000 * r, sigma=SIGMA.get(kind, 0.05), end=0.1, **kw)
        x = s["x"]
        if obs_noise:
            sd = float(np.sqrt(s["variance_theory"][0]))
            x = x + obs_noise * sd * np.random.default_rng(seed + 7 + r).standard_normal(x.size)
        res = an.analyze((s["t"][::every], x[::every]), seed=seed + r)
        ind = res.details["indicators"] if res.applicable else {}
        hits += all(k in ind and ind[k].significant(0.05) for k in which)
    return hits / reps, reps, time.perf_counter() - t0


def run(quick: bool = False, seed: int = 0) -> list[BenchmarkCase]:
    reps = 10 if quick else 40
    n_sur = 49 if quick else 99
    cases: list[BenchmarkCase] = []

    # ---- estimator accuracy on a stationary OU
    theta, sig = 0.5, 0.1
    ac, va = [], []
    t0 = time.perf_counter()
    n_rep = 10 if quick else 40
    for r in range(n_rep):
        s = ramp_series("stable", seed=seed + r, n=2000, t_end=1000.0, sigma=sig, start=theta)
        res = EarlyWarningAnalyzer(detrend="linear", surrogate="none").analyze((s["t"], s["x"]))
        ind = res.details["indicators"]
        ac.append(float(np.nanmean(ind["ac1"].values)))
        va.append(float(np.nanmean(ind["variance"].values)))
    cost = time.perf_counter() - t0
    dt = 1000.0 / 1999
    cases.append(BenchmarkCase("ou_ac1", "mean rolling lag-1 autocorrelation", math.exp(-theta * dt),
                               "exp(-theta dt)", float(np.mean(ac)), float(np.std(ac, ddof=1) / math.sqrt(n_rep)),
                               cost_s=cost, config={"theta": theta, "sigma": sig, "n": 2000, "replicates": n_rep},
                               limitations="windows of half the record; the lag-1 estimator is biased low by about "
                                           "(1 + 4 phi)/n_window"))
    cases.append(BenchmarkCase("ou_variance", "mean rolling variance", sig ** 2 / (2 * theta), "sigma^2 / (2 theta)",
                               float(np.mean(va)), float(np.std(va, ddof=1) / math.sqrt(n_rep)), cost_s=0.0))

    # ---- detection near bifurcations
    for kind in ("saddle_node", "transcritical", "pitchfork", "ou_decreasing"):
        rate, n, c = _rate(kind, reps, n_sur, seed)
        cases.append(BenchmarkCase(f"detect_{kind}", "rate of significant rise of variance and AC1", 1.0,
                                   "the system approaches the bifurcation", rate, tolerance=0.2, cost_s=c,
                                   config={"replicates": n, "surrogates": n_sur, "sigma": SIGMA[kind],
                                           "rate": "1 -> 0.1", "n": 1000},
                                   limitations="pass means a detection rate of at least 0.8 for this noise and record "
                                               "length; it does not say how early the signal appears"))

    # ---- false positives of controls
    def fp_case(name, kind, reference, **kw):
        rate, n, c = _rate(kind, reps, n_sur, seed + 11, **kw)
        se = math.sqrt(0.05 * 0.95 / n)
        if reference is None:
            return BenchmarkCase(name, "rate of significant rise", 0.05, "nominal level", rate, tolerance=math.inf,
                                 cost_s=c, config={"replicates": n, **{k: str(v) for k, v in kw.items()}},
                                 limitations="no pass/fail: shows a known false positive")
        return BenchmarkCase(name, "rate of significant rise", 0.05, "nominal level alpha = 0.05", rate,
                             tolerance=max(3 * se, 0.15 if quick else 0.1), cost_s=c,
                             config={"replicates": n, **{k: str(v) for k, v in kw.items()}},
                             limitations="the tolerance allows sampling error of the rate; a conservative test (rate "
                                         "below 0.05) also passes")

    cases.append(fp_case("fp_stable", "stable", 0.05))
    cases.append(fp_case("fp_trend_raw", "stable_trend", None, detrend="none"))
    cases.append(fp_case("fp_trend_loess", "stable_trend", 0.05))
    cases.append(fp_case("fp_growing_noise_variance", "stable_growing_noise", None, which=("variance",)))
    cases.append(fp_case("fp_growing_noise_ac1", "stable_growing_noise", 0.05, which=("ac1",)))

    # ---- robustness: measurement noise and coarser sampling
    for level in (1.0, 3.0):
        rate, n, c = _rate("ou_decreasing", reps, n_sur, seed + 21, obs_noise=level)
        cases.append(BenchmarkCase(f"detect_ou_obs_noise_{level:g}", "detection rate with white measurement noise "
                                   f"({level:g} x the initial std)", 1.0, "no reference", rate, tolerance=math.inf,
                                   cost_s=c, config={"replicates": n}, limitations="measurement"))
    for every in (4, 16):
        rate, n, c = _rate("ou_decreasing", reps, n_sur, seed + 31, every=every)
        cases.append(BenchmarkCase(f"detect_ou_subsampled_{every}", f"detection rate keeping every {every}th sample",
                                   1.0, "no reference", rate, tolerance=math.inf, cost_s=c,
                                   config={"replicates": n, "n_used": 1000 // every}, limitations="measurement"))
    return cases

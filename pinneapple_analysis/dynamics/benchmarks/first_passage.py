"""First-passage benchmarks against closed forms and an independent numerical solution.

| Case | Reference | Source |
|---|---|---|
| ``bm_drift_*`` | inverse-Gaussian mean ``a/mu`` and CDF | closed form, checked against ``scipy.stats.invgauss`` in the tests |
| ``bm_interval_*`` | exit mean, exit probability, survival series | scale function / backward equation (Cox & Miller 1965) |
| ``ou_mfpt`` | mean first-passage time of an OU process | quadrature of the backward equation, cross-checked in the tests against a finite-difference BVP solve |
| ``reflect_absorb`` | ``(b^2 - x0^2) / sigma^2`` | Brownian motion reflected at 0, absorbed at ``b`` |
| ``crossing_order`` | ``ln(1/c)/k`` for ``x = e^{-kt}`` | exact; the interpolation error must fall as ``h^2`` |
| ``km_censored`` | ``S(t) = e^{-t}`` | exponential event times under independent exponential censoring |
| ``bm_drift_mean_coarse_*`` | ``a/mu``; without the bridge, ``a/mu`` with the barrier shifted by ``0.5826 sigma sqrt(dt)`` | Broadie, Glasserman & Kou 1997 (Math. Finance 7:325) continuity correction |

``quick=True`` uses smaller samples (for CI); the default sizes are the ones whose numbers are quoted in the guide.
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..first_passage import (
    SDE,
    BoundaryCondition,
    FirstPassageAnalyzer,
    HittingTimeDistribution,
    brownian_exit_mean,
    brownian_exit_probability,
    brownian_exit_survival,
    brownian_first_passage_cdf,
    brownian_first_passage_mean,
    ou_mean_first_passage,
)
from ..trajectories import from_array
from ._case import BenchmarkCase

__all__ = ["run"]


def _dkw(n: int, alpha: float = 0.01) -> float:
    """Dvoretzky-Kiefer-Wolfowitz bound (Massart 1990): sup |F_n - F| <= eps with probability 1 - alpha."""
    return math.sqrt(math.log(2 / alpha) / (2 * n))


def run(quick: bool = False, seed: int = 0) -> list[BenchmarkCase]:
    n = 4_000 if quick else 20_000
    cases: list[BenchmarkCase] = []

    # ---- Brownian motion with drift, one barrier (inverse Gaussian)
    mu, sig, x0, b = 0.5, 1.0, 0.0, 1.0
    an = FirstPassageAnalyzer(BoundaryCondition.threshold(b))
    t0 = time.perf_counter()
    r = an.monte_carlo(SDE.brownian(mu, sig), x0, t_max=60.0, dt=2e-3, n_paths=n, seed=seed)
    cost = time.perf_counter() - t0
    d = r.distribution
    m = d.mean()
    cfg = {"mu": mu, "sigma": sig, "x0": x0, "b": b, "dt": 2e-3, "n_paths": n, "t_max": 60.0, "bridge": True}
    lim = "t_max = 60 censors a small tail, so the mean is the restricted mean to 60 (true restricted mean differs " \
          "from a/mu by less than 1e-6 for these parameters)"
    cases.append(BenchmarkCase("bm_drift_mean", "E[tau]", brownian_first_passage_mean(x0, b, mu, sig),
                               "a/mu (inverse Gaussian)", m["value"], m["se"], cost_s=cost, config=cfg,
                               limitations=lim))
    tt = np.linspace(0.05, 15, 300)
    sup = float(np.max(np.abs(d.cdf(tt) - brownian_first_passage_cdf(tt, x0, b, mu, sig))))
    cases.append(BenchmarkCase("bm_drift_cdf", "sup_t |F_MC - F|", 0.0, "inverse-Gaussian CDF", sup,
                               tolerance=_dkw(n), cost_s=0.0, config=cfg,
                               limitations="tolerance is the 99 % DKW bound of the Monte Carlo error alone"))

    # ---- coarse step: with the bridge, and without it (discrete monitoring, biased late)
    h = 2e-2
    t0 = time.perf_counter()
    r_b = an.monte_carlo(SDE.brownian(mu, sig), x0, t_max=60.0, dt=h, n_paths=n, seed=seed + 5, bridge=True)
    cost_b = time.perf_counter() - t0
    t0 = time.perf_counter()
    r_nb = an.monte_carlo(SDE.brownian(mu, sig), x0, t_max=60.0, dt=h, n_paths=n, seed=seed + 6, bridge=False)
    cost_nb = time.perf_counter() - t0
    wb, nb = r_b.distribution.mean(), r_nb.distribution.mean()
    cases.append(BenchmarkCase("bm_drift_mean_coarse_bridge", f"E[tau], dt = {h:g}", 2.0, "a/mu", wb["value"],
                               wb["se"], cost_s=cost_b, config={**cfg, "dt": h}))
    beta = 0.5826                                      # -zeta(1/2)/sqrt(2 pi)
    shifted = (b - x0 + beta * sig * math.sqrt(h)) / mu
    cases.append(BenchmarkCase("bm_drift_mean_coarse_no_bridge", f"E[tau], dt = {h:g}, discrete monitoring",
                               shifted, "a/mu with the barrier shifted by 0.5826 sigma sqrt(dt) (Broadie, "
                               "Glasserman & Kou 1997)", nb["value"], nb["se"], cost_s=cost_nb,
                               config={**cfg, "dt": h, "bridge": False},
                               convergence={"unshifted_truth": 2.0, "bias_vs_unshifted": nb["value"] - 2.0},
                               limitations="the shift is an asymptotic (small dt) correction, not exact: this "
                                           "case shows the size of the discrete-monitoring bias"))

    # ---- interval exit, with drift
    mu2, sig2, x02 = 0.7, 0.8, 0.3
    an2 = FirstPassageAnalyzer(BoundaryCondition.interval(0.0, 1.0))
    t0 = time.perf_counter()
    r2 = an2.monte_carlo(SDE.brownian(mu2, sig2), x02, t_max=10.0, dt=1e-3, n_paths=n, seed=seed + 1,
                         keep_paths=0)
    cost = time.perf_counter() - t0
    d2 = r2.distribution
    m2 = d2.mean()
    cfg2 = {"mu": mu2, "sigma": sig2, "x0": x02, "interval": (0.0, 1.0), "dt": 1e-3, "n_paths": n}
    cases.append(BenchmarkCase("bm_interval_mean", "E[exit time]", brownian_exit_mean(x02, 0, 1, mu2, sig2),
                               "(L P_up - y)/mu", m2["value"], m2["se"], cost_s=cost, config=cfg2))
    # exit side from a separate run with the two sides as two boundaries
    an_sides = FirstPassageAnalyzer([BoundaryCondition.threshold(1.0, "up", name="upper"),
                                     BoundaryCondition.threshold(0.0, "down", name="lower")])
    sim = an_sides.simulate(SDE.brownian(mu2, sig2), x02, t_max=10.0, dt=1e-3, n_paths=n, seed=seed + 1)
    up_first = float(np.mean(sim["tau"][0] < sim["tau"][1]))
    cases.append(BenchmarkCase("bm_interval_p_up", "P(exit through upper)",
                               brownian_exit_probability(x02, 0, 1, mu2, sig2), "scale function", up_first,
                               math.sqrt(up_first * (1 - up_first) / n), config=cfg2))
    tt2 = np.linspace(0.02, 2.0, 200)
    sup2 = float(np.max(np.abs(d2.survival(tt2) - brownian_exit_survival(tt2, x02, 0, 1, mu2, sig2))))
    cases.append(BenchmarkCase("bm_interval_survival", "sup_t |S_MC - S|", 0.0, "eigenfunction series", sup2,
                               tolerance=_dkw(n), config=cfg2,
                               limitations="tolerance is the 99 % DKW bound of the Monte Carlo error alone"))

    # ---- Ornstein-Uhlenbeck mean first-passage time
    th, mo, so, x0o, lv = 1.0, 0.0, 1.0, 0.0, 1.5
    an3 = FirstPassageAnalyzer(BoundaryCondition.threshold(lv))
    no = n // 2
    t0 = time.perf_counter()
    r3 = an3.monte_carlo(SDE.ornstein_uhlenbeck(th, mo, so), x0o, t_max=200.0, dt=5e-3, n_paths=no, seed=seed + 2)
    cost = time.perf_counter() - t0
    m3 = r3.distribution.mean()
    cases.append(BenchmarkCase("ou_mfpt", "E[tau]", ou_mean_first_passage(x0o, lv, th, mo, so),
                               "quadrature of the backward equation", m3["value"], m3["se"], cost_s=cost,
                               config={"theta": th, "mean": mo, "sigma": so, "x0": x0o, "level": lv, "dt": 5e-3,
                                       "n_paths": no, "t_max": 200.0}))

    # ---- reflecting at 0, absorbing at 1
    an4 = FirstPassageAnalyzer(BoundaryCondition.threshold(1.0))
    t0 = time.perf_counter()
    r4 = an4.monte_carlo(SDE.brownian(0.0, 1.0), 0.3, t_max=30.0, dt=1e-3, n_paths=n, seed=seed + 3,
                         reflecting=[BoundaryCondition.threshold(0.0, "down", behaviour="reflecting")])
    cost = time.perf_counter() - t0
    m4 = r4.distribution.mean()
    cases.append(BenchmarkCase("reflect_absorb", "E[tau]", (1.0 - 0.09) / 1.0, "(b^2 - x0^2)/sigma^2",
                               m4["value"], m4["se"], cost_s=cost,
                               config={"x0": 0.3, "b": 1.0, "sigma": 1.0, "dt": 1e-3, "n_paths": n},
                               limitations="mirror reflection after the Euler step; no bridge correction at the "
                                           "reflecting wall"))

    # ---- deterministic crossing: second-order convergence of the interpolated time
    k, c = 1.3, 0.2
    truth = math.log(1 / c) / k
    errs, hs = [], [0.1, 0.05, 0.025, 0.0125]
    t0 = time.perf_counter()
    for h in hs:                                       # worst case over 16 grid offsets: the crossing moves inside
        worst = 0.0                                    # its bracket, so a single grid gives a noisy order
        for u in np.arange(1, 17) / 17:
            t = np.concatenate([[0.0], np.arange(u * h, 3.0 + h, h)])
            res = FirstPassageAnalyzer(BoundaryCondition.threshold(c, "down")).crossing(from_array(t, np.exp(-k * t)))
            worst = max(worst, abs(res.estimate[f"0<={c:g}"]["tau"] - truth))
        errs.append(worst)
    cost = time.perf_counter() - t0
    order = float(np.polyfit(np.log(hs), np.log(errs), 1)[0])
    cases.append(BenchmarkCase("crossing_order", "observed order of the interpolation error", 2.0,
                               "linear interpolation of a C^2 function", order, tolerance=0.3, cost_s=cost,
                               convergence={"h": hs, "error": errs},
                               limitations="smooth trajectory; a rough (e.g. noisy) trajectory has no such order"))

    # ---- Kaplan-Meier under censoring
    rng = np.random.default_rng(seed + 4)
    ev = rng.exponential(1.0, n)
    cens = rng.exponential(2.0, n)
    tobs, obs = np.minimum(ev, cens), ev <= cens
    t0 = time.perf_counter()
    dist = HittingTimeDistribution(tobs, obs)
    tt3 = np.linspace(0.0, 2.0, 201)
    sup3 = float(np.max(np.abs(dist.survival(tt3) - np.exp(-tt3))))
    lo, hi = dist.survival_function().band(tt3)
    coverage = float(np.mean((lo <= np.exp(-tt3)) & (np.exp(-tt3) <= hi)))
    naive = float(np.max(np.abs(np.mean(tobs[:, None] > tt3[None, :], axis=0) - np.exp(-tt3))))
    cost = time.perf_counter() - t0
    cases.append(BenchmarkCase("km_censored", "sup_t<=2 |S_KM - S|", 0.0, "S(t) = exp(-t)", sup3,
                               tolerance=4 * math.sqrt(math.e ** 2 / n), cost_s=cost,
                               config={"n": n, "event_rate": 1.0, "censor_rate": 0.5,
                                       "censored_fraction": float(1 - obs.mean())},
                               convergence={"pointwise_band_coverage": coverage, "naive_sup_error": naive},
                               limitations="tolerance from the Greenwood variance bound at t = 2; "
                                           "the naive estimate that ignores censoring is reported for contrast"))
    return cases

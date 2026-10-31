"""Fisher-information benchmarks against exact results.

| Case | Reference | Source |
|---|---|---|
| ``linear_fim`` | ``X^T X / sigma^2`` | Gaussian linear regression, exact |
| ``ols_efficiency_*`` | efficiency 1 | OLS attains the bound for linear Gaussian models (Gauss-Markov + normality) |
| ``decay_k_optimal_time`` | ``1/k`` | maximiser of ``(A t e^{-kt})^2`` |
| ``decay_d_optimal_second_time`` | ``1/k`` (first at 0) | Box & Lucas 1959, Biometrika 46:77 |
| ``tau_c_optimal_std`` | ``sigma / (k c sqrt(n))`` | delta method for the time ``y`` reaches ``c`` |
| ``poisson_score_mc`` | ``sum t / theta`` | Poisson rate ``theta t`` |
| ``heteroscedastic_score_mc`` | ``sum t^2/rel^2 + 2 t^2`` | relative Gaussian error, exact information |
| ``conjugate_posterior`` | ``(X^T X/sigma^2 + P)^-1`` | Gaussian conjugate prior |
| ``nonlinear_efficiency_small_n`` | the bound | least squares on 4 noisy points of a decay: shows the bound need not be attained |
"""
from __future__ import annotations

import math
import time

import numpy as np

from ...inverse_problems.forward import CallableModel, Observations, Parameter
from ..fisher import FisherInformation, ObservationScheduler, estimator_performance
from ._case import BenchmarkCase

__all__ = ["run"]
A0, K0, SIG = 2.0, 0.5, 0.05


def _decay(free=("A", "k")):
    ps = {"A": Parameter("A", A0, 0.1, 10.0), "k": Parameter("k", K0, 0.01, 5.0)}
    return CallableModel(lambda th, inp: th.get("A", A0) * np.exp(-th.get("k", K0) * inp["t"]), inputs=["t"],
                         parameters=[ps[n] for n in free])


def _line():
    return CallableModel(lambda th, inp: th["b0"] + th["b1"] * inp["t"], inputs=["t"],
                         parameters=[Parameter("b0", 1.0, -10, 10, log=False),
                                     Parameter("b1", 2.0, -10, 10, log=False)])


def run(quick: bool = False, seed: int = 0) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    t = np.linspace(0, 1, 20)
    X = np.column_stack([np.ones_like(t), t])
    obs = Observations({"t": t}, 0.1)
    t0 = time.perf_counter()
    fim = FisherInformation.gaussian(_line(), obs)
    rel = float(np.max(np.abs(fim.F - X.T @ X / 0.01) / np.abs(X.T @ X / 0.01)))
    cases.append(BenchmarkCase("linear_fim", "max relative error of F", 0.0, "X^T X / sigma^2", rel, tolerance=1e-6,
                               cost_s=time.perf_counter() - t0, config={"derivatives": fim.how}))
    n_rep = 200 if quick else 1000
    t0 = time.perf_counter()
    perf = estimator_performance(_line(), obs, n_rep=n_rep, seed=seed)
    cost = time.perf_counter() - t0
    for name in ("b0", "b1"):
        e = perf.estimate[name]["efficiency"]
        cases.append(BenchmarkCase(f"ols_efficiency_{name}", "CRLB variance / empirical variance", 1.0,
                                   "OLS is efficient here", e, mc_se=e * math.sqrt(2 / (n_rep - 1)), cost_s=cost,
                                   config={"n_rep": n_rep, "seed": seed}))

    t0 = time.perf_counter()
    r = ObservationScheduler(_decay(("k",)), (0, 10), n_grid=401, sigma=SIG).schedule(1, "D")
    cases.append(BenchmarkCase("decay_k_optimal_time", "best single time for k", 1 / K0, "maximiser of (A t e^-kt)^2",
                               r.estimate["times"][0], tolerance=10 / 400, cost_s=time.perf_counter() - t0,
                               limitations="grid of 401 candidate times on [0, 10]: resolution 0.025"))
    t0 = time.perf_counter()
    sch = ObservationScheduler(_decay(), (0, 10), n_grid=401, sigma=SIG)
    r2 = sch.schedule(2, "D")
    times = sorted(r2.estimate["times"])
    cases.append(BenchmarkCase("decay_d_optimal_second_time", "second D-optimal time (first at 0)", 1 / K0,
                               "Box & Lucas 1959", times[1], tolerance=10 / 400, cost_s=time.perf_counter() - t0,
                               convergence={"first_time": times[0]}))
    c, n = 0.5, 3
    t0 = time.perf_counter()
    rc = sch.schedule(n, "c", target=lambda th: math.log(th["A"] / c) / th["k"])
    exact = SIG / (K0 * c * math.sqrt(n))
    cases.append(BenchmarkCase("tau_c_optimal_std", "c-optimal bound on the crossing time", exact,
                               "sigma / (k c sqrt(n))", rc.uncertainty["target_std"]["target"],
                               tolerance=0.02 * exact, cost_s=time.perf_counter() - t0,
                               convergence={"times": rc.estimate["times"],
                                            "d_optimal_target_std": sch.schedule(n, "D", target=lambda th: math.log(
                                                th["A"] / c) / th["k"]).uncertainty["target_std"]["target"]},
                               limitations="the closed form is the limit of measuring exactly at tau; the grid puts "
                                           "the points within 0.05 of it"))

    tp = np.array([0.5, 1.0, 2.0, 4.0])
    th = 3.0
    n_mc = 2000 if quick else 20000

    def ll_p(r_, y):
        lam = r_[0] * tp
        return float(np.sum(y * np.log(lam) - lam))

    t0 = time.perf_counter()
    mc = FisherInformation.from_loglik(ll_p, {"r": th}, lambda p, rng: rng.poisson(p["r"] * tp), n_mc=n_mc, seed=seed)
    cases.append(BenchmarkCase("poisson_score_mc", "F (score Monte Carlo)", float(tp.sum() / th), "sum t / theta",
                               float(mc.F[0, 0]), mc_se=float(mc.mc_se[0, 0]), cost_s=time.perf_counter() - t0,
                               config={"n_mc": n_mc}))
    th_t = np.linspace(0.2, 4, 15)
    relerr = 0.1

    def ll_h(k, y):
        mu = A0 * np.exp(-k[0] * th_t)
        s = relerr * mu
        return float(np.sum(-np.log(s) - 0.5 * ((y - mu) / s) ** 2))

    def sim_h(p, rng):
        mu = A0 * np.exp(-p["k"] * th_t)
        return mu + relerr * mu * rng.standard_normal(th_t.size)

    t0 = time.perf_counter()
    mch = FisherInformation.from_loglik(ll_h, {"k": K0}, sim_h, n_mc=n_mc, seed=seed + 1)
    cases.append(BenchmarkCase("heteroscedastic_score_mc", "F (score Monte Carlo)",
                               float(np.sum(th_t ** 2 / relerr ** 2 + 2 * th_t ** 2)), "relative Gaussian error",
                               float(mch.F[0, 0]), mc_se=float(mch.mc_se[0, 0]), cost_s=time.perf_counter() - t0,
                               config={"n_mc": n_mc}))
    t6 = np.linspace(0, 1, 6)
    X6 = np.column_stack([np.ones_like(t6), t6])
    P = np.diag([4.0, 1.0])
    t0 = time.perf_counter()
    post = FisherInformation.gaussian(_line(), Observations({"t": t6}, 0.2)).with_prior(P).crlb()
    ref = np.linalg.inv(X6.T @ X6 / 0.04 + P)
    cases.append(BenchmarkCase("conjugate_posterior", "max relative error of the posterior covariance", 0.0,
                               "(X^T X / sigma^2 + P)^-1", float(np.max(np.abs(post.cov - ref) / np.abs(ref))),
                               tolerance=1e-6, cost_s=time.perf_counter() - t0))

    obs4 = Observations({"t": np.array([0.0, 0.0, 2.0, 2.0])}, 0.3)
    t0 = time.perf_counter()
    perf4 = estimator_performance(_decay(), obs4, n_rep=n_rep, seed=seed)
    e4 = perf4.estimate["k"]
    cases.append(BenchmarkCase("nonlinear_efficiency_small_n", "efficiency of least squares for k (4 points, "
                               "sigma 0.3)", 1.0, "the bound", e4["efficiency"], tolerance=math.inf,
                               cost_s=time.perf_counter() - t0,
                               convergence={"bias": e4["bias"], "bias_se": e4["bias_se"],
                                            "empirical_std": e4["empirical_std"], "crlb_std": e4["crlb_std"],
                                            "efficiency_ci95": e4["efficiency_ci95"]},
                               limitations="no pass/fail: a non-linear model with few, noisy data need not attain "
                                           "the bound; this row measures by how much"))
    return cases

"""First-passage and hitting times (pinneapple_analysis.dynamics.first_passage).

References: the inverse-Gaussian law of Brownian first passage (checked against scipy.stats.invgauss, an
independent implementation); the exit probability, mean exit time and survival series of drifted Brownian motion in
an interval (the series is checked against the closed-form mean by integrating it); the Ornstein-Uhlenbeck mean
first-passage time (quadrature checked against an independent finite-difference solve of the backward equation);
Brownian motion reflected at 0 and absorbed at b, E[tau] = (b^2 - x0^2)/sigma^2; exact crossing times of smooth
functions; exponential event times under independent censoring for Kaplan-Meier.
Monte Carlo comparisons use |error| <= 4 standard errors, or the 99 % DKW bound for whole distributions.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest
from scipy import sparse, stats
from scipy.integrate import simpson
from scipy.sparse.linalg import spsolve

from pinneapple_analysis import dynamics as dyn
from pinneapple_analysis.dynamics import first_passage as fp
from pinneapple_analysis.dynamics.first_passage import (
    SDE,
    BoundaryCondition,
    FirstPassageAnalyzer,
    HittingTimeDistribution,
)


def _dkw(n, alpha=0.01):
    return math.sqrt(math.log(2 / alpha) / (2 * n))


# =========================================================================== closed forms
def test_inverse_gaussian_matches_scipy():
    for mu, sig, x0, b in [(0.5, 1.0, 0.0, 1.0), (2.0, 0.3, 1.0, 3.5), (-1.5, 0.7, 2.0, 0.5)]:
        a, nu = abs(b - x0), (mu if b > x0 else -mu)
        ig = stats.invgauss(mu=(a / nu) / (a ** 2 / sig ** 2), scale=a ** 2 / sig ** 2)
        t = np.linspace(0.01, 8, 200)
        assert np.allclose(fp.brownian_first_passage_pdf(t, x0, b, mu, sig), ig.pdf(t), rtol=1e-10, atol=1e-14)
        assert np.allclose(fp.brownian_first_passage_cdf(t, x0, b, mu, sig), ig.cdf(t), rtol=1e-9, atol=1e-14)
        assert fp.brownian_first_passage_mean(x0, b, mu, sig) == pytest.approx(ig.mean())


def test_drift_away_gives_a_defective_law():
    mu, sig, a = -0.4, 1.0, 1.0                       # P(tau < inf) = exp(2 nu a / sigma^2)
    assert fp.brownian_first_passage_cdf(1e6, 0.0, a, mu, sig) == pytest.approx(math.exp(2 * mu * a), rel=1e-6)
    assert fp.brownian_first_passage_mean(0.0, a, mu, sig) == math.inf
    from scipy.integrate import quad
    mass, _ = quad(lambda t: fp.brownian_first_passage_pdf(t, 0.0, a, mu, sig), 0, np.inf, limit=200)
    assert mass == pytest.approx(math.exp(2 * mu * a), rel=1e-6)


@pytest.mark.parametrize("mu", [0.0, 0.7, -1.3, 1e-9])
def test_exit_survival_series_integrates_to_the_closed_form_mean(mu):
    x0, lo, hi, sig = 0.3, 0.0, 1.0, 0.8
    t = np.concatenate([np.linspace(0, 0.05, 5001), np.linspace(0.05, 40, 40_001)[1:]])
    S = fp.brownian_exit_survival(t, x0, lo, hi, mu, sig)
    assert S[0] == 1.0 and np.all(np.diff(S) <= 1e-12)
    assert simpson(S, x=t) == pytest.approx(fp.brownian_exit_mean(x0, lo, hi, mu, sig), rel=1e-6)


def test_exit_probability_limits():
    assert fp.brownian_exit_probability(0.25, 0, 1, 0.0, 1.0) == pytest.approx(0.25)
    assert fp.brownian_exit_probability(0.25, 0, 1, 1e-14, 1.0) == pytest.approx(0.25)
    assert fp.brownian_exit_probability(0.25, 0, 1, 50.0, 1.0) == pytest.approx(1.0, abs=1e-10)
    assert fp.brownian_exit_mean(0.5, 0, 1, 0.0, 1.0) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        fp.brownian_exit_probability(1.5, 0, 1, 0.0, 1.0)


def _ou_mfpt_fd(x0, b, theta, m, sigma, xl=-7.0, n=8001):
    """Independent reference: finite differences for D T'' - theta (x - m) T' = -1, T(b) = 0, T'(xl) = 0."""
    x = np.linspace(xl, b, n)
    h = x[1] - x[0]
    D = sigma ** 2 / 2
    a = -theta * (x - m)
    A = sparse.diags([D / h ** 2 - a[1:] / (2 * h), -2 * D / h ** 2 * np.ones(n), D / h ** 2 + a[:-1] / (2 * h)],
                     [-1, 0, 1]).tolil()
    rhs = -np.ones(n)
    A[0, 1] = 2 * D / h ** 2
    A[n - 1, :] = 0
    A[n - 1, n - 1] = 1.0
    rhs[-1] = 0.0
    return float(np.interp(x0, x, spsolve(A.tocsr(), rhs)))


@pytest.mark.parametrize("x0,b,theta,m,sigma", [(0.0, 1.5, 1.0, 0.0, 1.0), (0.5, 1.0, 2.0, 0.2, 0.5),
                                                 (0.0, -1.0, 0.5, 0.0, 1.0)])
def test_ou_mfpt_quadrature_matches_finite_differences(x0, b, theta, m, sigma):
    if b < x0:
        ref = _ou_mfpt_fd(-x0, -b, theta, -m, sigma)
    else:
        ref = _ou_mfpt_fd(x0, b, theta, m, sigma)
    assert fp.ou_mean_first_passage(x0, b, theta, m, sigma) == pytest.approx(ref, rel=2e-5)
    assert fp.ou_mean_first_passage(1.0, 1.0, 1.0) == 0.0


# =========================================================================== boundaries
def test_boundary_gauges():
    X = np.array([[0.2, 0.0], [1.2, 0.0], [-0.5, 0.0]])
    start = np.zeros((3, 2))
    assert np.allclose(BoundaryCondition.threshold(1.0).gauge(X, start), [-0.8, 0.2, -1.5])
    assert np.allclose(BoundaryCondition.threshold(0.0, "down").gauge(X, start), [-0.2, -1.2, 0.5])
    assert np.allclose(BoundaryCondition.interval(-0.4, 1.0).gauge(X, start), [-0.6, 0.2, 0.1])
    circle = BoundaryCondition.level_set(lambda x: np.hypot(x[:, 0], x[:, 1]) - 1.0, "leave unit disc")
    assert np.allclose(circle.gauge(X, start), [-0.8, 0.2, -0.5])
    either = BoundaryCondition.threshold(0.5, "either")
    s = np.array([[0.0, 0], [1.0, 0], [0.5, 0]])
    assert np.allclose(either.gauge(np.array([[0.6, 0], [0.4, 0], [0.9, 0]]), s), [0.1, 0.1, 0.0])
    with pytest.raises(ValueError):
        BoundaryCondition.interval(1.0, 1.0)
    with pytest.raises(ValueError):
        BoundaryCondition.threshold(0.0, "sideways")
    with pytest.raises(ValueError, match="side"):
        BoundaryCondition.threshold(0.0, "either", behaviour="reflecting")
    with pytest.raises(ValueError, match="unique"):
        FirstPassageAnalyzer([BoundaryCondition.threshold(1.0), BoundaryCondition.threshold(1.0)])
    with pytest.raises(ValueError, match="absorbing"):
        FirstPassageAnalyzer(BoundaryCondition.threshold(0.0, "down", behaviour="reflecting"))


# =========================================================================== deterministic crossing
def test_crossing_of_exponential_decay_on_an_irregular_grid():
    k, c = 1.3, 0.2
    t = np.sort(np.concatenate([[0.0, 3.0], np.random.default_rng(1).uniform(0, 3, 300)]))
    res = FirstPassageAnalyzer(BoundaryCondition.threshold(c, "down", name="c")).crossing(
        dyn.from_array(t, np.exp(-k * t), kind="analytic"))
    tau = res.estimate["c"]["tau"]
    lo, hi = res.uncertainty["c"]["bracket"]
    assert lo <= math.log(1 / c) / k <= hi
    assert abs(tau - math.log(1 / c) / k) <= k * (hi - lo) ** 2 / 8   # |f''| h^2 / (8 |f'|) for f = e^{-kt}
    assert res.ok and res.unchecked_assumptions[0].name == "no unobserved excursion"


def test_crossing_error_is_second_order():
    k, c, truth = 1.3, 0.2, math.log(5) / 1.3
    errs = []
    hs = [0.08, 0.04, 0.02, 0.01]
    for h in hs:
        worst = 0.0
        for u in np.arange(1, 9) / 9:
            t = np.concatenate([[0.0], np.arange(u * h, 3.0, h)])
            r = FirstPassageAnalyzer(BoundaryCondition.threshold(c, "down")).crossing(dyn.from_array(t, np.exp(-k * t)))
            worst = max(worst, abs(r.estimate[f"0<={c:g}"]["tau"] - truth))
        errs.append(worst)
    order = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    assert 1.8 < order < 2.2


def test_multiple_thresholds_censoring_and_start_inside():
    t = np.linspace(0, 2, 201)
    x = np.sin(np.pi * t)                              # max 1 at t = 0.5
    bcs = [BoundaryCondition.threshold(0.5, name="half"), BoundaryCondition.threshold(0.9, name="most"),
           BoundaryCondition.threshold(1.5, name="never"), BoundaryCondition.threshold(0.0, "down", name="at_start")]
    r = FirstPassageAnalyzer(bcs).crossing(dyn.from_array(t, x, observed_until=2.5))
    assert r.estimate["half"]["tau"] == pytest.approx(1 / 6, abs=1e-4)
    assert r.estimate["most"]["tau"] == pytest.approx(math.asin(0.9) / math.pi, abs=1e-4)
    assert r.estimate["never"] == {"tau": None, "crossed": False, "censored_at": 2.5}
    assert r.estimate["at_start"]["tau"] == 0.0
    assert r.diagnostic("at_start: started_on_boundary").status == "warn"


def test_crossing_across_missing_samples_is_reported():
    t = np.linspace(0, 1, 11)
    x = t.copy()
    x[4:8] = np.nan                                    # the crossing of 0.55 falls in the gap
    r = FirstPassageAnalyzer(BoundaryCondition.threshold(0.55, name="b")).crossing(dyn.from_array(t, x))
    assert r.estimate["b"]["tau"] == pytest.approx(0.55)          # linear data: interpolation still exact
    assert r.uncertainty["b"]["bracket"] == pytest.approx((0.3, 0.8))
    assert r.diagnostic("b: crossing_across_gap").status == "warn"


def test_level_set_exit_of_a_disc_in_2d():
    t = np.linspace(0, 2, 41)
    x = np.column_stack([t * np.cos(3 * t), t * np.sin(3 * t)])   # radius = t: leaves the unit disc at t = 1
    bc = BoundaryCondition.level_set(lambda X: np.hypot(X[:, 0], X[:, 1]) - 1.0, "disc")
    r = FirstPassageAnalyzer(bc).crossing(dyn.from_array(t, x))
    assert r.estimate["disc"]["tau"] == pytest.approx(1.0, abs=1e-12)


def test_crossing_on_named_component_of_a_csv(tmp_path):
    p = tmp_path / "cool.csv"
    rows = "\n".join(f"{ti:.3f},{80 * math.exp(-ti / 50) + 20:.6f},{1.0}" for ti in np.arange(0, 200, 7.3))
    p.write_text("time [s],T [°C],q\n" + rows + "\n")
    tr = dyn.from_csv(str(p), "time")
    r = FirstPassageAnalyzer(BoundaryCondition.threshold(50.0, "down", component="T", name="T50")).crossing(tr)
    assert r.estimate["T50"]["tau"] == pytest.approx(50 * math.log(80 / 30), abs=7.3 ** 2 / (8 * 50))  # |f''|h^2/(8|f'|)
    with pytest.raises(KeyError):
        FirstPassageAnalyzer(BoundaryCondition.threshold(1.0, component="p")).crossing(tr)


def test_crossing_with_too_few_samples_is_not_applicable():
    t = np.array([0.0, 1.0, 2.0])
    r = FirstPassageAnalyzer(BoundaryCondition.threshold(1.0)).crossing(dyn.from_array(t, [np.nan, 0.5, np.nan]))
    assert not r.applicable and "usable samples" in r.failures[0]


# =========================================================================== Kaplan-Meier and distributions
def test_kaplan_meier_hand_computed():
    # times 1, 2+, 3, 3, 4+ (+ = censored): S(1) = 4/5, S(3) = 4/5 * (1 - 2/3) = 4/15
    km = fp.kaplan_meier([1, 2, 3, 3, 4], [True, False, True, True, False])
    assert np.allclose(km.t, [1, 3]) and np.allclose(km.S, [0.8, 4 / 15])
    assert km(0.5) == 1.0 and km(2.5) == pytest.approx(0.8) and km(3.0) == pytest.approx(4 / 15)
    assert np.isnan(km(5.0))                           # beyond what was observed
    var1 = 0.8 ** 2 * (1 / (5 * 4))                    # Greenwood at t = 1
    lo, hi = km.band(1.0)
    se_ll = math.sqrt(var1) / (0.8 * abs(math.log(0.8)))
    z = stats.norm.ppf(0.975)
    assert lo == pytest.approx(math.exp(-math.exp(math.log(-math.log(0.8)) + z * se_ll)))
    assert hi == pytest.approx(math.exp(-math.exp(math.log(-math.log(0.8)) - z * se_ll)))


def test_kaplan_meier_recovers_survival_under_censoring_where_naive_fails():
    rng = np.random.default_rng(4)
    n = 5000
    ev, cens = rng.exponential(1.0, n), rng.exponential(2.0, n)
    d = HittingTimeDistribution(np.minimum(ev, cens), ev <= cens)
    t = np.linspace(0, 2, 101)
    assert np.max(np.abs(d.survival(t) - np.exp(-t))) < 0.03
    naive = np.mean(np.minimum(ev, cens)[:, None] > t[None, :], axis=0)
    assert np.max(np.abs(naive - np.exp(-t))) > 0.1   # ignoring censoring is badly biased (about 0.15 here)
    p = d.hitting_probability(1.0)
    assert p.method.startswith("Kaplan-Meier") and p.ci[0] < 1 - math.exp(-1) < p.ci[1]
    assert d.median == pytest.approx(math.log(2), abs=0.05)
    lo, hi = d.quantile_ci(0.5)
    assert lo <= math.log(2) <= hi
    m = d.mean()
    assert m["restricted"] and "restricted" in m["kind"]


def test_wilson_interval_and_uncensored_probability():
    assert fp.wilson_interval(0, 10)[0] == 0.0 and fp.wilson_interval(10, 10)[1] == pytest.approx(1.0)
    lo, hi = fp.wilson_interval(30, 100)
    assert lo == pytest.approx(0.2189, abs=1e-3) and hi == pytest.approx(0.3958, abs=1e-3)
    d = HittingTimeDistribution([0.5, 1.0, 1.5, 3.0, 3.0], [True, True, True, False, False], horizon=3.0)
    p = d.hitting_probability(2.0)
    assert p.p == 0.6 and p.method.startswith("binomial") and p.no_crossing[0] == pytest.approx(0.4)
    assert d.uniformly_censored and d.quantile(0.9) == math.inf
    assert d.prediction_interval(0.5)[1] == math.inf      # the upper quantile lies beyond the horizon
    with pytest.raises(ValueError, match="beyond the horizon"):
        d.hitting_probability(4.0)


# =========================================================================== Monte Carlo against closed forms
def test_mc_brownian_drift_matches_inverse_gaussian():
    mu, sig, b, n = 0.5, 1.0, 1.0, 6000
    r = FirstPassageAnalyzer(BoundaryCondition.threshold(b, name="b")).monte_carlo(
        SDE.brownian(mu, sig), 0.0, t_max=60.0, dt=2e-3, n_paths=n, seed=11)
    d = r.distribution
    t = np.linspace(0.05, 12, 200)
    assert np.max(np.abs(d.cdf(t) - fp.brownian_first_passage_cdf(t, 0, b, mu, sig))) < _dkw(n)
    m = d.mean()
    assert abs(m["value"] - 2.0) < 4 * m["se"]
    assert r.ok and r.reproducibility.seed == 11 and r.config["bridge"] is True


def test_mc_interval_exit_mean_side_and_survival():
    mu, sig, x0, n = 0.7, 0.8, 0.3, 6000
    an = FirstPassageAnalyzer([BoundaryCondition.threshold(1.0, name="up"),
                               BoundaryCondition.threshold(0.0, "down", name="down"),
                               BoundaryCondition.interval(0.0, 1.0, name="exit")])
    sim = an.simulate(SDE.brownian(mu, sig), x0, t_max=10.0, dt=1e-3, n_paths=n, seed=5)
    p_up = np.mean(sim["tau"][0] < sim["tau"][1])
    p_true = fp.brownian_exit_probability(x0, 0, 1, mu, sig)
    assert abs(p_up - p_true) < 4 * math.sqrt(p_true * (1 - p_true) / n)
    r = an.monte_carlo(SDE.brownian(mu, sig), x0, t_max=10.0, dt=1e-3, n_paths=n, seed=5)
    d = r.distributions["exit"]
    t = np.linspace(0.02, 2, 100)
    assert np.max(np.abs(d.survival(t) - fp.brownian_exit_survival(t, x0, 0, 1, mu, sig))) < _dkw(n)
    m = d.mean()
    assert abs(m["value"] - fp.brownian_exit_mean(x0, 0, 1, mu, sig)) < 4 * m["se"]
    with pytest.raises(ValueError, match="boundaries"):
        _ = r.distribution


def test_mc_ou_mean_first_passage():
    th, n = 1.0, 3000
    r = FirstPassageAnalyzer(BoundaryCondition.threshold(1.5)).monte_carlo(
        SDE.ornstein_uhlenbeck(th, 0.0, 1.0), 0.0, t_max=200.0, dt=5e-3, n_paths=n, seed=2)
    m = r.distribution.mean()
    assert not m["restricted"] and abs(m["value"] - fp.ou_mean_first_passage(0.0, 1.5, th)) < 4 * m["se"]


def test_mc_reflecting_wall():
    n = 6000
    wall = BoundaryCondition.threshold(0.0, "down", behaviour="reflecting")
    r = FirstPassageAnalyzer(BoundaryCondition.threshold(1.0)).monte_carlo(
        SDE.brownian(0.0, 1.0), 0.3, t_max=30.0, dt=1e-3, n_paths=n, seed=3, reflecting=[wall], keep_paths=50)
    m = r.distribution.mean()
    assert abs(m["value"] - 0.91) < 4 * m["se"]
    paths = r.details["paths"]
    assert np.nanmin(paths.stack()) >= 0.0             # the wall holds on every kept path
    with pytest.raises(ValueError, match="reflecting"):
        FirstPassageAnalyzer(BoundaryCondition.threshold(1.0)).monte_carlo(
            SDE.brownian(), 0.0, t_max=1, dt=0.1, n_paths=2, reflecting=[BoundaryCondition.threshold(0.0, "down")])


def test_mc_level_set_in_2d_is_flagged_as_discretely_monitored():
    # 2-D standard Brownian motion leaving the unit disc from the origin: E[tau] = R^2 / (d sigma^2) = 1/2
    bm2 = SDE(lambda t, x: np.zeros_like(x), lambda t, x: np.ones_like(x), dim=2, name="bm2")
    r = FirstPassageAnalyzer(BoundaryCondition.level_set(lambda X: np.hypot(X[:, 0], X[:, 1]) - 1.0, "disc")
                             ).monte_carlo(bm2, [0.0, 0.0], t_max=20.0, dt=1e-4, n_paths=3000, seed=8)
    m = r.distribution.mean()
    assert r.diagnostic("disc: monitoring").status == "warn"
    assert m["value"] > 0.5 - 4 * m["se"]              # discrete monitoring can only be late
    assert abs(m["value"] - 0.5) < 4 * m["se"] + 0.6 * math.sqrt(1e-4) * 2   # bias O(sqrt dt) stated in the docs


# =========================================================================== convergence and the bridge
def test_bridge_removes_most_of_the_discrete_monitoring_bias():
    mu, sig, n, h = 0.5, 1.0, 8000, 0.04
    an = FirstPassageAnalyzer(BoundaryCondition.threshold(1.0))
    nb = an.monte_carlo(SDE.brownian(mu, sig), 0.0, t_max=60, dt=h, n_paths=n, seed=1, bridge=False)
    wb = an.monte_carlo(SDE.brownian(mu, sig), 0.0, t_max=60, dt=h, n_paths=n, seed=1, bridge=True)
    e_nb = nb.distribution.mean()["value"] - 2.0
    e_wb = wb.distribution.mean()["value"] - 2.0
    se = wb.distribution.mean()["se"]
    assert e_nb > 0.5826 * sig * math.sqrt(h) / mu - 4 * se   # late by about the continuity correction
    assert abs(e_wb) < 4 * se                          # the bridge brings it back within MC noise
    assert nb.diagnostic("0>=1: monitoring").status == "warn"


def test_dt_convergence_of_the_discretely_monitored_estimate():
    an = FirstPassageAnalyzer(BoundaryCondition.threshold(1.0))
    r = an.dt_convergence(SDE.brownian(0.5, 1.0), 0.0, t_max=60, dts=[0.16, 0.04, 0.01], n_paths=8000, seed=0,
                          statistic="mean", bridge=False)
    v, se = r.convergence["value"], r.convergence["mc_se"]
    assert v[0] > v[1] > v[2]                          # the late bias shrinks with dt
    predicted = 2.0 + 0.5826 * np.sqrt(r.convergence["dt"]) / 0.5   # continuity correction, O(sqrt dt)
    assert np.all(np.abs(v - predicted) < 4 * se)
    assert np.isfinite(r.convergence["observed_order"])  # reported; too noisy at this size to test its value
    with pytest.raises(ValueError):
        an.dt_convergence(SDE.brownian(), 0.0, t_max=1, dts=[0.1])


# =========================================================================== sensitivity
def test_sensitivity_to_drift_matches_the_analytic_derivative():
    T, x0, b, sig = 2.0, 0.0, 1.0, 1.0
    mus = [0.4, 0.5, 0.6]
    an = FirstPassageAnalyzer(BoundaryCondition.threshold(b), horizon=T)
    r = an.sensitivity(lambda m: SDE.brownian(m, sig), mus, x0=x0, t_max=T, dt=2e-3, n_paths=20_000, seed=7)
    exact = (fp.brownian_first_passage_cdf(T, x0, b, 0.6, sig) - fp.brownian_first_passage_cdf(T, x0, b, 0.4, sig)) / 0.2
    dP = float(r.estimate["derivative"][1])
    assert abs(dP - exact) < 0.1                       # common random numbers keep the difference precise
    assert np.allclose(r.estimate["value"], fp.brownian_first_passage_cdf(T, x0, b, np.array(mus), sig), atol=0.015)


def test_sensitivity_to_the_noise_hypothesis():
    an = FirstPassageAnalyzer(BoundaryCondition.threshold(2.0), horizon=3.0)
    models = {"additive": SDE.brownian(0.0, 0.5), "multiplicative": SDE.geometric_brownian(0.0, 0.5)}
    r = an.sensitivity(models, x0=1.0, t_max=3.0, dt=5e-3, n_paths=4000, seed=3)
    p_add, p_mult = r.estimate["value"]
    assert r.estimate["labels"] == ["additive", "multiplicative"] and "derivative" not in r.estimate
    assert p_add == pytest.approx(fp.brownian_first_passage_cdf(3.0, 1.0, 2.0, 0.0, 0.5), abs=0.03)
    # GBM: ln X is Brownian with drift -sigma^2/2; same law gives the reference
    assert p_mult == pytest.approx(fp.brownian_first_passage_cdf(3.0, 0.0, math.log(2), -0.125, 0.5), abs=0.03)


# =========================================================================== empirical, from observed trajectories
def test_from_trajectories_with_irregular_windows_matches_inverse_gaussian():
    mu, sig, n, h = 0.5, 1.0, 2000, 2e-3
    rng = np.random.default_rng(12)
    t = np.arange(0, 6.0 + h / 2, h)
    W = np.concatenate([np.zeros((n, 1)), np.cumsum(rng.normal(0, math.sqrt(h), (n, t.size - 1)), axis=1)], axis=1)
    paths = mu * t + sig * W                           # exact Brownian paths on the grid (independent generator)
    trajs = []
    for k in range(n):                                 # each member observed until its own time (censoring)
        end = rng.uniform(2.0, 6.0)
        keep = t <= end
        stride = rng.integers(1, 4)                    # and on its own (coarser) grid
        idx = np.nonzero(keep)[0][::stride]
        trajs.append(dyn.from_array(t[idx], paths[k, idx], ("X",), observed_until=end, kind="stochastic"))
    r = FirstPassageAnalyzer(BoundaryCondition.threshold(1.0, name="b")).from_trajectories(trajs)
    d = r.distributions["b"]
    assert d.n_censored > 0 and not d.uniformly_censored
    tt = np.linspace(0.2, 2.0, 30)
    assert np.max(np.abs(d.cdf(tt) - fp.brownian_first_passage_cdf(tt, 0, 1, mu, sig))) < 0.05   # DKW 99 %: 0.036
    assert r.diagnostic("b: max_bracket_width").value > 0  # interval censoring by the sampling is reported


def test_from_trajectories_needs_events_and_members():
    t = np.linspace(0, 1, 5)
    r = FirstPassageAnalyzer(BoundaryCondition.threshold(5.0)).from_trajectories([dyn.from_array(t, t)])
    assert not r.applicable and len(r.failures) == 2


# =========================================================================== failure, reproducibility, performance
def test_monte_carlo_input_errors_and_all_censored():
    an = FirstPassageAnalyzer(BoundaryCondition.threshold(10.0))
    with pytest.raises(ValueError):
        an.monte_carlo(SDE.brownian(), 0.0, t_max=1.0, dt=2.0, n_paths=10)
    with pytest.raises(ValueError):
        an.monte_carlo(SDE.brownian(), np.zeros((3, 1)), t_max=1.0, dt=0.1, n_paths=10)
    with pytest.raises(ValueError):
        SDE.brownian(0.0, 0.0)
    r = an.monte_carlo(SDE.brownian(), 0.0, t_max=0.5, dt=0.01, n_paths=200, seed=0)
    assert not r.applicable and r.diagnostic("0>=10: censored_fraction").status == "fail"
    assert r.estimate["0>=10"]["p_hit"] == 0.0 and r.uncertainty["0>=10"]["p_hit_ci"][1] < 0.02


def test_monte_carlo_is_reproducible_and_seed_dependent():
    an = FirstPassageAnalyzer(BoundaryCondition.threshold(1.0))
    kw = dict(t_max=5.0, dt=0.01, n_paths=500)
    a = an.monte_carlo(SDE.brownian(0.5), 0.0, seed=4, **kw)
    b = an.monte_carlo(SDE.brownian(0.5), 0.0, seed=4, **kw)
    c = an.monte_carlo(SDE.brownian(0.5), 0.0, seed=5, **kw)
    assert np.array_equal(a.distribution.times, b.distribution.times)
    assert a.reproducibility.input_hash == b.reproducibility.input_hash
    assert not np.array_equal(a.distribution.times, c.distribution.times)
    u = an.monte_carlo(SDE.brownian(0.5), 0.0, seed=None, **kw)
    again = an.monte_carlo(SDE.brownian(0.5), 0.0, seed=u.reproducibility.seed, **kw)
    assert np.array_equal(u.distribution.times, again.distribution.times)
    s = a.to_study_result(study_type="uncertainty")
    assert s.metrics["estimate.0>=1.p_hit"] == a.estimate["0>=1"]["p_hit"]


def test_distribution_from_x0_samples():
    # random initial states: the mixture of inverse Gaussians
    rng = np.random.default_rng(0)
    x0 = rng.uniform(-1.0, 0.0, size=(4000, 1))
    r = FirstPassageAnalyzer(BoundaryCondition.threshold(1.0)).monte_carlo(
        SDE.brownian(1.0, 1.0), x0, t_max=40.0, dt=2e-3, n_paths=4000, seed=1)
    m = r.distribution.mean()
    assert abs(m["value"] - 1.5) < 4 * m["se"]          # E[1 - x0] / mu = 1.5


def test_monte_carlo_cost():
    an = FirstPassageAnalyzer(BoundaryCondition.threshold(1.0))
    t0 = time.perf_counter()
    an.monte_carlo(SDE.brownian(0.5), 0.0, t_max=10.0, dt=1e-3, n_paths=10_000, seed=0)
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0, f"10 000 paths x 10 000 steps took {elapsed:.1f} s"


def test_method_cards():
    for card in FirstPassageAnalyzer.cards.values():
        assert card.role == "trajectory_operator" and card.failure_modes
        assert "fails or does not apply" in str(card)


def test_benchmarks_quick_all_pass():
    from pinneapple_analysis.dynamics.benchmarks import first_passage as bench
    from pinneapple_analysis.dynamics.benchmarks import to_markdown

    cases = bench.run(quick=True)
    failed = [c.to_dict() for c in cases if not c.passed]
    assert not failed, failed
    assert "| bm_drift_mean |" in to_markdown(cases)

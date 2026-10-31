"""Fisher information, Cramér-Rao bounds and experimental design (pinneapple_analysis.dynamics.fisher).

References: Gaussian linear regression (F = X^T X / sigma^2 exactly, and OLS attains the bound); exponential decay
y = A exp(-k t) (one-parameter optimum at t = 1/k; the two-parameter D-optimal design puts equal weight at t = 0 and
t = 1/k, Box & Lucas 1959); the delta-method bound on the time tau at which y reaches c, sigma / (k c sqrt(n)) for n
measurements at t = tau; Poisson counts with rate theta t (F = sum t / theta); the parameter-dependent Gaussian
noise term 2 (dsigma)^2 / sigma^2 checked against a Monte Carlo score estimate; the conjugate Gaussian posterior.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pinneapple_analysis.dynamics import first_passage as fp
from pinneapple_analysis.dynamics import from_model
from pinneapple_analysis.dynamics.fisher import (
    ExperimentalDesign,
    FisherInformation,
    FisherInformationMatrix,
    InformationGain,
    ObservationScheduler,
    ParameterIdentifiability,
    SamplingStrategy,
    SensitivityAnalyzer,
    estimator_performance,
)
from pinneapple_analysis.inverse_problems.forward import CallableModel, Observations, Parameter
from pinneapple_analysis.inverse_problems.identifiability import d_optimal_design

A0, K0, SIG = 2.0, 0.5, 0.05


def decay(free=("A", "k")):
    params = {"A": Parameter("A", A0, 0.1, 10.0), "k": Parameter("k", K0, 0.01, 5.0)}

    def f(th, inp):
        return th.get("A", A0) * np.exp(-th.get("k", K0) * inp["t"])

    return CallableModel(f, inputs=["t"], parameters=[params[n] for n in free], name="decay")


def line():
    return CallableModel(lambda th, inp: th["b0"] + th["b1"] * inp["t"], inputs=["t"],
                         parameters=[Parameter("b0", 1.0, -10, 10, log=False), Parameter("b1", 2.0, -10, 10, log=False)])


# =========================================================================== exact references
def test_linear_regression_information_is_exact_and_ols_attains_the_bound():
    t = np.linspace(0, 1, 20)
    X = np.column_stack([np.ones_like(t), t])
    obs = Observations({"t": t}, 0.1)
    fim = FisherInformation.gaussian(line(), obs)
    assert np.allclose(fim.F, X.T @ X / 0.01, rtol=1e-7)
    assert np.allclose(fim.F, FisherInformation.linear_gaussian(X, 0.1, ["b0", "b1"]).F, rtol=1e-7)
    crlb = fim.crlb()
    assert np.allclose(crlb.cov, 0.01 * np.linalg.inv(X.T @ X), rtol=1e-6)
    perf = estimator_performance(line(), obs, n_rep=400, seed=1)
    for name in ("b0", "b1"):
        row = perf.estimate[name]
        lo, hi = row["efficiency_ci95"]
        assert lo < 1.0 < hi                           # OLS is efficient for linear Gaussian models
        assert abs(row["bias"]) < 4 * row["bias_se"]
    assert "empirical" in perf.uncertainty["kind"] and "theoretical" in perf.uncertainty["kind"]


def test_one_parameter_decay_is_best_measured_at_one_over_k():
    sch = ObservationScheduler(decay(("k",)), (0, 10), n_grid=401, sigma=SIG)
    r = sch.schedule(1, "D")
    assert r.estimate["times"] == [pytest.approx(1 / K0, abs=10 / 400)]
    t = sch.times
    info = (A0 * t * np.exp(-K0 * t) / SIG) ** 2         # closed-form information per measurement
    assert t[np.argmax(info)] == pytest.approx(1 / K0, abs=10 / 400)


def test_two_parameter_decay_d_optimal_design_is_zero_and_one_over_k():
    sch = ObservationScheduler(decay(), (0, 10), n_grid=401, sigma=SIG)
    r = sch.schedule(2, "D")
    assert sorted(r.estimate["times"]) == [pytest.approx(0.0), pytest.approx(1 / K0, abs=10 / 400)]
    r6 = sch.schedule(6, "D")
    assert sorted(r6.estimate["times"]) == pytest.approx([0, 0, 0, 2, 2, 2], abs=10 / 400)   # equal weights
    assert r6.ok and r6.estimate["criteria"]["D"] > r.estimate["criteria"]["D"]


def test_c_optimal_design_for_the_crossing_time_matches_the_delta_method():
    c = 0.5
    tau = lambda th: math.log(th["A"] / c) / th["k"]           # noqa: E731 - time at which y reaches c
    sch = ObservationScheduler(decay(), (0, 10), n_grid=401, sigma=SIG)
    n = 3
    r = sch.schedule(n, "c", target=tau)
    assert all(abs(t - tau({"A": A0, "k": K0})) < 0.1 for t in r.estimate["times"])
    exact = SIG / (K0 * c * math.sqrt(n))                      # sigma / (|dy/dt| at tau) / sqrt(n)
    assert r.uncertainty["target_std"]["target"] == pytest.approx(exact, rel=0.02)
    d = sch.schedule(n, "D", target=tau)
    assert d.uncertainty["target_std"]["target"] > 1.5 * r.uncertainty["target_std"]["target"]


def test_first_passage_time_as_the_design_target():
    """Compose with first_passage: the target is the crossing time computed by FirstPassageAnalyzer.crossing on the
    model's own predicted trajectory."""
    model = decay()
    analyzer = fp.FirstPassageAnalyzer(fp.BoundaryCondition.threshold(0.5, "down", name="c"))
    grid = np.linspace(0, 10, 4001)

    def crossing_time(th):
        tr = from_model(lambda t: model.predict(th, {"t": t}), grid, ("y",), kind="model")
        return analyzer.crossing(tr).estimate["c"]["tau"]

    fim = FisherInformation.gaussian(model, Observations({"t": np.linspace(0, 6, 25)}, SIG))
    via_fp = fim.crlb(crossing_time, rel_step=1e-3).target_std["target"]
    closed = fim.crlb(lambda th: math.log(th["A"] / 0.5) / th["k"]).target_std["target"]
    assert via_fp == pytest.approx(closed, rel=1e-3)


# =========================================================================== other likelihoods
def test_poisson_information():
    th = 3.0
    t = np.array([0.5, 1.0, 2.0, 4.0])
    m = CallableModel(lambda p, inp: p["r"] * inp["t"], inputs=["t"], parameters=[Parameter("r", th, 0.1, 100)])
    fim = FisherInformation.poisson(m, Observations({"t": t}, 1.0))
    assert fim.F[0, 0] == pytest.approx(t.sum() / th, rel=1e-6)

    def loglik(r, y):
        lam = r[0] * t
        return float(np.sum(y * np.log(lam) - lam))

    mc = FisherInformation.from_loglik(loglik, {"r": th}, lambda p, rng: rng.poisson(p["r"] * t), n_mc=4000, seed=0)
    assert abs(mc.F[0, 0] - t.sum() / th) < 4 * mc.mc_se[0, 0]


def test_parameter_dependent_noise_adds_information():
    t = np.linspace(0.2, 4, 15)
    rel = 0.1
    model = decay(("k",))
    obs = Observations({"t": t}, 1.0)
    sig = lambda th, o: rel * A0 * np.exp(-th["k"] * o.inputs["t"])    # noqa: E731 - relative error
    fixed = FisherInformation.gaussian(model, Observations({"t": t}, sig({"k": K0}, obs)))
    het = FisherInformation.gaussian(model, obs, sigma_model=sig)
    assert het.F[0, 0] > fixed.F[0, 0]
    # exact: each point contributes (dmu/dk)^2/s^2 + 2 (ds/dk)^2/s^2 = t^2/rel^2 + 2 t^2
    assert het.F[0, 0] == pytest.approx(np.sum(t ** 2 / rel ** 2 + 2 * t ** 2), rel=1e-6)

    def loglik(k, y):
        mu = A0 * np.exp(-k[0] * t)
        s = rel * mu
        return float(np.sum(-np.log(s) - 0.5 * ((y - mu) / s) ** 2))

    def sim(p, rng):
        mu = A0 * np.exp(-p["k"] * t)
        return mu + rel * mu * rng.standard_normal(t.size)

    mc = FisherInformation.from_loglik(loglik, {"k": K0}, sim, n_mc=4000, seed=1)
    assert abs(mc.F[0, 0] - het.F[0, 0]) < 4 * mc.mc_se[0, 0]


def test_torch_score_matches_finite_differences():
    torch = pytest.importorskip("torch")
    t = np.linspace(0, 3, 10)
    tt = torch.tensor(t)

    def ll_torch(th, y):
        mu = th[0] * torch.exp(-th[1] * tt)
        return -0.5 * torch.sum(((y - mu) / SIG) ** 2)

    def ll_np(th, y):
        mu = th[0] * np.exp(-th[1] * t)
        return -0.5 * np.sum(((y - mu) / SIG) ** 2)

    sim = lambda p, rng: p["A"] * np.exp(-p["k"] * t) + SIG * rng.standard_normal(t.size)   # noqa: E731
    a = FisherInformation.from_loglik(ll_torch, {"A": A0, "k": K0}, sim, n_mc=500, seed=2, backend="torch")
    b = FisherInformation.from_loglik(ll_np, {"A": A0, "k": K0}, sim, n_mc=500, seed=2, backend="fd")
    assert np.allclose(a.F, b.F, rtol=1e-5)
    exact = FisherInformation.gaussian(decay(), Observations({"t": t}, SIG)).F
    assert np.all(np.abs(a.F - exact) < 4 * a.mc_se + 1e-9)


def test_jax_backend_is_optional():
    pytest.importorskip("jax")
    t = np.linspace(0, 3, 5)

    def ll(th, y):
        import jax.numpy as jnp
        return -0.5 * jnp.sum(((y - th[0] * jnp.exp(-th[1] * t)) / SIG) ** 2)

    sim = lambda p, rng: p["A"] * np.exp(-p["k"] * t) + SIG * rng.standard_normal(t.size)   # noqa: E731
    a = FisherInformation.from_loglik(ll, {"A": A0, "k": K0}, sim, n_mc=200, seed=0, backend="jax")
    assert a.F.shape == (2, 2)


def test_missing_backend_and_bad_backend(monkeypatch):
    import importlib.util

    from pinneapple_analysis.dynamics import BackendUnavailable

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: None if n == "jax" else real(n, *a))
    with pytest.raises(BackendUnavailable):
        FisherInformation.from_loglik(lambda th, y: 0.0, {"a": 1.0}, lambda p, r: 0.0, n_mc=2, backend="jax")
    with pytest.raises(ValueError):
        FisherInformation.from_loglik(lambda th, y: 0.0, {"a": 1.0}, lambda p, r: 0.0, n_mc=2, backend="tf")


# =========================================================================== bayesian vs bound
def test_prior_gives_the_conjugate_posterior_and_is_labelled():
    t = np.linspace(0, 1, 6)
    X = np.column_stack([np.ones_like(t), t])
    fim = FisherInformation.gaussian(line(), Observations({"t": t}, 0.2))
    P = np.diag([4.0, 1.0])
    post = fim.with_prior(P).crlb()
    assert post.bayesian and "posterior" in post.kind
    assert np.allclose(post.cov, np.linalg.inv(X.T @ X / 0.04 + P), rtol=1e-6)
    plain = fim.crlb()
    assert not plain.bayesian and "Cramér-Rao" in plain.kind
    assert all(post.std[n] < plain.std[n] for n in ("b0", "b1"))


def test_bayes_d_design_needs_a_prior_and_uses_it():
    cand = Observations({"t": np.linspace(0, 10, 101)}, SIG)
    with pytest.raises(ValueError, match="prior"):
        ExperimentalDesign(decay(), cand).select(1, "bayes-D")
    d = ExperimentalDesign(decay(), cand, prior_precision=[1e4, 0.0])    # A essentially known
    r = d.select(1, "bayes-D")
    assert r.estimate["inputs"]["t"] == [pytest.approx(2.0, abs=0.1)]  # then measure where k is best seen: 1/k


# =========================================================================== identifiability
def test_identifiability_depends_on_the_schedule():
    B = Parameter("B", 0.3, 0.01, 5)
    m = CallableModel(lambda th, inp: th["A"] * np.exp(-th["k"] * inp["t"]) + th["B"], inputs=["t"],
                      parameters=[Parameter("A", A0, 0.1, 10), Parameter("k", K0, 0.01, 5), B])
    one_time = ParameterIdentifiability(m, Observations({"t": np.full(10, 1.0)}, SIG)).local()
    assert one_time.estimate["rank"] == 1 and not one_time.ok
    two_times = ParameterIdentifiability(m, Observations({"t": np.r_[np.full(5, 0.5), np.full(5, 3.0)]}, SIG)).local()
    assert two_times.estimate["rank"] == 2
    spread = ParameterIdentifiability(m, Observations({"t": np.linspace(0, 8, 20)}, SIG)).local()
    assert spread.estimate["rank"] == 3 and set(spread.estimate["status"].values()) == {"identifiable"}
    short = ParameterIdentifiability(m, Observations({"t": np.linspace(0, 0.3, 20)}, SIG)).local()
    assert "practically non-identifiable" in short.estimate["status"].values()


def test_structural_non_identifiability_is_found_and_named():
    m = CallableModel(lambda th, inp: th["a"] * th["b"] * inp["t"], inputs=["t"],
                      parameters=[Parameter("a", 2.0, 0.1, 10), Parameter("b", 3.0, 0.1, 10)])
    r = ParameterIdentifiability(m, Observations({"t": np.linspace(0, 1, 10)}, 0.1)).local()
    assert r.estimate["rank"] == 1
    (direction,) = r.estimate["null_directions"]
    assert abs(direction["a"] / direction["b"]) == pytest.approx(2.0 / 3.0, rel=1e-4)   # d(ab) = 0: da/a = -db/b
    assert set(r.estimate["status"].values()) == {"structurally non-identifiable"}
    full = ParameterIdentifiability(m, Observations({"t": np.linspace(0, 1, 10)}, 0.1)).full(profile=False,
                                                                                              posterior=False)
    assert full.estimate["rank"] == 1 and "report" in full.details


# =========================================================================== design machinery
def test_greedy_d_choice_matches_the_existing_d_optimal_design():
    cand = Observations({"t": np.linspace(0, 10, 41)}, SIG)
    base = Observations({"t": np.array([1.0, 1.5])}, SIG)
    d = ExperimentalDesign(decay(), cand, base=base)
    r = d.select(1, "D", exchange=False)
    ref = d_optimal_design(d.F0, d.Jw, n_pick=1)
    assert r.estimate["chosen"] == [ref[0]["index"]]


def test_a_and_e_criteria_and_failures():
    cand = Observations({"t": np.linspace(0, 10, 101)}, SIG)
    d = ExperimentalDesign(decay(), cand)
    for crit in ("A", "E"):
        r = d.select(4, crit)
        assert r.ok and np.isfinite(r.estimate["value"])
    flat = Observations({"t": np.zeros(5)}, SIG)                         # only t = 0: k is invisible
    bad = ExperimentalDesign(decay(), flat).select(3, "D")
    assert not bad.applicable and "singular" in bad.failures[0]
    with pytest.raises(ValueError):
        d.select(2, "c")
    with pytest.raises(ValueError):
        ExperimentalDesign(decay(), Observations({"t": np.array([0.0, 1.0])}, SIG)).select(3, "D", replicates=False)


def test_information_gain_and_next_best_time():
    base = Observations({"t": np.array([0.0, 0.0, 0.0])}, SIG)
    fim = FisherInformation.gaussian(decay(), base)
    assert not np.isfinite(fim.criteria()["D"])                          # only A is informed so far
    gain = InformationGain(fim, decay()).of(Observations({"t": np.array([2.0])}, SIG))
    assert gain["delta_logdet"] == math.inf                             # the new point makes F nonsingular
    fim2 = FisherInformation.gaussian(decay(), Observations({"t": np.array([0.0, 2.0])}, SIG))
    g2 = InformationGain(fim2, decay()).of(Observations({"t": np.array([2.0])}, SIG),
                                           target=lambda th: math.log(th["A"] / 0.5) / th["k"])
    assert g2["delta_logdet"] > 0 and g2["target_std_after"] < g2["target_std_before"]
    assert g2["expected_information_gain_nats"] == pytest.approx(0.5 * g2["delta_logdet"])
    nb = ObservationScheduler(decay(), (0, 10), n_grid=201, sigma=SIG).next_best(Observations({"t": np.zeros(4)}, SIG))
    assert nb["time"] == pytest.approx(2.0, abs=0.1)


def test_sampling_strategies_are_ranked_by_information():
    s = {"uniform": SamplingStrategy.uniform(0, 10, 12, SIG), "log": SamplingStrategy.logspaced(0.05, 10, 12, SIG),
         "random": SamplingStrategy.random(0, 10, 12, SIG, seed=3), "late": SamplingStrategy.uniform(6, 10, 12, SIG),
         "optimal": Observations({"t": np.r_[np.zeros(6), np.full(6, 2.0)]}, SIG)}
    r = SamplingStrategy.compare(decay(), s)
    assert r.estimate["ranking_by_D"][0] == "optimal" and r.estimate["ranking_by_D"][-1] == "late"
    assert all(v["n"] == 12 for v in r.estimate["strategies"].values())
    with pytest.raises(ValueError):
        SamplingStrategy.logspaced(0, 1, 3)


def test_sensitivities():
    sa = SensitivityAnalyzer(decay())
    out = sa.local(Observations({"t": np.array([0.0, 1.0, 2.0])}, SIG))
    assert np.allclose(out["relative"][:, 0], 1.0)                       # dlog y / dlog A = 1
    assert np.allclose(out["relative"][:, 1], -K0 * np.array([0.0, 1.0, 2.0]), atol=1e-6)
    tsens = sa.target(lambda th: math.log(th["A"] / 0.5) / th["k"])
    assert tsens["k"] == pytest.approx(-1.0, rel=1e-4)                   # tau ~ 1/k


def test_fim_algebra_and_reproducibility():
    o1, o2 = Observations({"t": np.array([0.0])}, SIG), Observations({"t": np.array([2.0])}, SIG)
    f1, f2 = FisherInformation.gaussian(decay(), o1), FisherInformation.gaussian(decay(), o2)
    both = FisherInformation.gaussian(decay(), o1.concat(o2))
    assert np.allclose((f1 + f2).F, both.F)
    with pytest.raises(ValueError):
        FisherInformationMatrix(np.eye(3), ["a", "b"], {}, "x")
    d = ExperimentalDesign(decay(), Observations({"t": np.linspace(0, 10, 51)}, SIG))
    a, b = d.select(3, "D"), d.select(3, "D")
    assert a.estimate["chosen"] == b.estimate["chosen"] and a.reproducibility.input_hash == b.reproducibility.input_hash


def test_benchmarks_quick_all_pass():
    from pinneapple_analysis.dynamics.benchmarks import fisher as bench

    cases = bench.run(quick=True)
    failed = [c.to_dict() for c in cases if not c.passed]
    assert not failed, failed

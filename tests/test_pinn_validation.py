"""PINN validation (#703). References: the Cole-Hopf solution of Burgers (cross-checked against the pseudo-spectral
solver of `operator_benchmarks`), a manufactured Poisson solution and the Kovasznay flow (closed forms). Converged
PINNs pass; a PINN stopped early, one trained without its boundary loss and one frozen in time each fail with the
failure mode named."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from pinneapple_analysis.validation.pinn import (
    PINNMonitor,
    burgers_1d,
    cole_hopf,
    kovasznay,
    poisson_2d,
    train_pinn,
    validate_pinn,
)

torch = pytest.importorskip("torch")
NU = 0.05 / math.pi


def test_cole_hopf_matches_the_spectral_solver():
    # u on [-1, 1] with u0 = -sin(pi x) is odd and 2-periodic: v(x', t) = u(2x' - 1, t) / 2 solves Burgers on the
    # periodic [0, 1) with viscosity nu / 4 (operator_benchmarks.burgers_1d)
    from pinneapple_simulation.numerical_solvers.operator_benchmarks import burgers_1d as spectral
    n = 256
    xp = np.arange(n) / n
    _, v = spectral(-np.sin(np.pi * (2 * xp - 1)) / 2, NU / 4, 0.8, dt=1e-4, n_save=2)
    for k, t in ((1, 0.4), (2, 0.8)):
        assert np.abs(2 * v[0, k] - cole_hopf(2 * xp - 1, t, NU)).max() < 1e-6


class _Closed(torch.nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f
        self.p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))

    def forward(self, X):
        return self.f(X)


def test_exact_solutions_pass_every_check():
    pi = math.pi
    poisson = _Closed(lambda X: torch.sin(pi * X[:, :1]) * torch.sin(2 * pi * X[:, 1:2]))
    lam = 20 - math.sqrt(400 + 4 * pi ** 2)

    def kov(X):
        x, y = X[:, :1], X[:, 1:2]
        e = torch.exp(lam * x)
        return torch.cat([1 - e * torch.cos(2 * pi * y), lam / (2 * pi) * e * torch.sin(2 * pi * y),
                          0.5 * (1 - torch.exp(2 * lam * x)) + 3.0], 1)            # pressure up to a constant
    for model, problem in ((poisson, poisson_2d()), (_Closed(kov), kovasznay(40.0))):
        rep = validate_pinn(model, problem)
        assert rep.passed, rep.summary()
        assert rep.evidence["residual"]["max"] < 1e-8 and rep["pinn.exact"]["value"] < 1e-12


@pytest.fixture(scope="module")
def runs():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = {"burgers": train_pinn(burgers_1d(NU)),
               "burgers_early": train_pinn(burgers_1d(NU), adam_steps=20, lbfgs_steps=0),
               "poisson": train_pinn(poisson_2d(), n_collocation=1000, adam_steps=1000, lbfgs_steps=500),
               "poisson_early": train_pinn(poisson_2d(), n_collocation=1000, adam_steps=20, lbfgs_steps=0),
               "poisson_no_bc": train_pinn(poisson_2d(), n_collocation=1000, adam_steps=1000, lbfgs_steps=500,
                                           weights={"boundary": 0.0})}
    return out


def test_converged_pinns_pass(runs):
    for name, problem in (("burgers", burgers_1d(NU)), ("poisson", poisson_2d()), ("poisson", poisson_2d(exact=False))):
        net, info = runs[name]
        rep = validate_pinn(net, problem, train_points=info["train_points"])
        assert rep.passed and not rep.failure_modes, rep.summary()
        r = rep.evidence["residual"]
        assert r["max_adaptive"] >= r["rms_random"]                     # adaptive points find the worst residuals
        assert 0.5 < r["fresh_over_train"] < 3                          # no overfitting of the collocation set
    assert validate_pinn(runs["burgers"][0], burgers_1d(NU))["pinn.exact"]["value"] < 0.01


def test_early_stopped_pinns_are_named_near_trivial(runs):
    for name, problem in (("burgers_early", burgers_1d(NU)), ("poisson_early", poisson_2d())):
        rep = validate_pinn(runs[name][0], problem)
        assert rep.failure_modes[0] == "near-trivial solution", rep.summary()
    # honest: a nearly constant field nearly solves Burgers, so its residual alone passes
    assert validate_pinn(runs["burgers_early"][0], burgers_1d(NU))["pinn.residual"]["status"] == "pass"


def test_zero_boundary_weight_is_named_a_boundary_trade_off(runs):
    for problem in (poisson_2d(), poisson_2d(exact=False)):
        rep = validate_pinn(runs["poisson_no_bc"][0], problem)
        assert rep.failure_modes[0] == "boundary condition not satisfied", rep.summary()
        assert rep["pinn.residual"]["status"] == "pass" and rep["pinn.trivial"]["status"] == "pass"
        assert "traded off against the residual" in rep["pinn.boundary"]["summary"]
    # honest: the error is harmonic (zero Laplacian), so L[u] vs f cannot see it; the exact solution's spectrum can
    assert validate_pinn(runs["poisson_no_bc"][0], poisson_2d(exact=False))["pinn.spectrum"]["status"] == "pass"
    assert validate_pinn(runs["poisson_no_bc"][0], poisson_2d())["pinn.spectrum"]["status"] == "fail"


def test_a_solution_frozen_in_time_is_a_propagation_failure(runs):
    net = runs["burgers"][0]

    class Frozen(torch.nn.Module):                                      # right until t = 0.3, then stops evolving
        def forward(self, X):
            return net(torch.cat([X[:, :1], X[:, 1:].clamp(max=0.3)], 1))
    rep = validate_pinn(Frozen(), burgers_1d(NU))
    assert rep.failure_modes[0] == "propagation failure in time", rep.summary()
    prof = rep.evidence["causality"]["residual_by_time"]
    assert prof[: len(prof) // 4].max() < 0.05 < prof[-len(prof) // 4:].min()


def test_monitor_stops_training_on_a_failure_mode(runs):
    mon = PINNMonitor(poisson_2d(), every=100, warmup=200, stop_on=("pinn.boundary",))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, info = train_pinn(poisson_2d(), n_collocation=1000, adam_steps=1000, lbfgs_steps=0,
                             weights={"boundary": 0.0}, monitor=mon)
    assert info["stopped"] and info["steps"] == 300 and "boundary" in mon.reason
    assert mon.history[0] == (100, mon.history[0][1]) and "pinn.boundary" in mon.history[0][1]
    ok = PINNMonitor(poisson_2d(), every=1, patience=1)
    ok(runs["poisson"][0], 1)
    assert not ok.stop and ok.history == [(1, [])]


def test_pp_validate_runs_pinn_checks(runs):
    import pinneapple as pp
    from pinneapple_analysis.validation.report_artifact import check_report
    net, info = runs["poisson"]
    rep = pp.validate(net, {"pinn": "poisson_2d"}, data={"train_points": info["train_points"]}, level="quick")
    assert rep["pinn.exact"].status == "pass" and rep["pinn.boundary"].status == "pass"
    bad = pp.validate(runs["poisson_no_bc"][0], {"pinn": poisson_2d()})
    assert bad.decision == "fail" and bad["pinn.boundary"].status == "fail"
    assert check_report(rep) == [] and check_report(bad) == []

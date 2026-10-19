"""Residual-to-error calibration (#699). References: the finite-volume solver (`operator_benchmarks.darcy_2d`, the
same discretisation as the residual), the a-posteriori theory ``||u_h - u||_A <= ||r||_{L^-1} / sqrt(min k)``, and
the shifts of the PhysicsNeMo trust experiment (coefficient contrast {1, 20}; three times shorter length scale)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from pinneapple_analysis.uncertainty.residual_bound import (
    ResidualErrorCalibrator,
    coverage_study,
    darcy_problem,
    energy_error_bound,
    heat_conduction_problem,
)
from pinneapple_simulation.numerical_solvers import operator_benchmarks as ob

torch = pytest.importorskip("torch")
N = 32


def darcy(n, seed, tau=3.0, high=12.0, low=3.0):
    a = ob.threshold_coefficient(ob.periodic_grf(n, (N, N), alpha=2.0, tau=tau, seed=seed), high=high, low=low)
    return a, np.stack([ob.darcy_2d(x, 1.0) for x in a])


def heat(n, seed, k):
    q = 10.0 * ob.periodic_grf(n, (N, N), alpha=2.5, tau=4.0, seed=seed) + 10.0         # heat source
    return q, np.stack([ob.darcy_2d(k, qi) for qi in q])


def train(method, cfg, X, Y, epochs=30):
    """Train an operator with the library trainer (pinneapple_neural.train_model)."""
    from pinneapple_analysis.benchmark.builtin.fields import _operator_model
    from pinneapple_neural import train_model
    xm, xs, ys = X.mean(), X.std(), Y.std()
    Xt, Yt = torch.tensor((X - xm) / xs, dtype=torch.float32), torch.tensor(Y / ys, dtype=torch.float32)
    batches = [(Xt[i:i + 20], Yt[i:i + 20]) for i in range(0, len(Xt), 20)]
    torch.manual_seed(0)
    m = _operator_model(method, cfg, (N, N))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_model(m, lambda mm, yh, b: ((yh - b["y"]) ** 2).mean(), batches, batches[:3], epochs=epochs, lr=2e-3,
                    device="cpu", save_best=False, log_dir=str(_tmp()))
    m.eval()

    def predict(x):
        with torch.no_grad():
            return m(torch.tensor((x - xm) / xs, dtype=torch.float32)).numpy() * ys
    return predict


def _tmp():
    import tempfile
    return tempfile.mkdtemp()


def features(problems, pred, true):
    """Relative energy-norm bound (the residual measure), relative L2 error, and theory violations."""
    res, viol = [], 0
    for P, p, u in zip(problems, pred, true, strict=True):
        b = energy_error_bound(P, p)
        res.append(b["relative"])
        viol += P.energy_norm(u - p) > b["bound"] * (1 + 1e-9)
    err = np.linalg.norm((pred - true).reshape(len(true), -1), axis=1) / \
        np.linalg.norm(true.reshape(len(true), -1), axis=1)
    return np.array(res), err, int(viol)


@pytest.fixture(scope="module")
def darcy_runs():
    A, U = darcy(1400, 1)
    shifts = {"contrast {1, 20}": darcy(150, 7701, high=20.0, low=1.0), "tau 9": darcy(150, 7702, tau=9.0)}
    out = {}
    for meth, cfg in (("fno", {"width": 16, "modes": 8}), ("deeponet", {"hidden": 128, "p": 64, "depth": 3})):
        f = train(meth, cfg, A[:400], U[:400])
        pool = features([darcy_problem(a) for a in A[400:]], f(A[400:]), U[400:])
        sh = {k: features([darcy_problem(a) for a in a_], f(a_), u_) for k, (a_, u_) in shifts.items()}
        out[meth] = (pool, sh)
    return out


def test_residual_of_the_reference_solution_is_round_off():
    a, u = darcy(3, 5)
    for ai, ui in zip(a, u, strict=True):
        P = darcy_problem(ai)
        r = P.residual(ui)
        assert np.abs(r).max() < 1e-9 * np.abs(P.f).max()
        assert energy_error_bound(P, ui)["relative"] < 1e-10
        # a finite-difference residual of the same field would not be zero: the scheme must match
        lap = (np.roll(ui, 1, 0) + np.roll(ui, -1, 0) + np.roll(ui, 1, 1) + np.roll(ui, -1, 1) - 4 * ui) * N ** 2
        assert np.abs(ai[1:-1, 1:-1] * lap[1:-1, 1:-1] + 1.0).max() > 1.0


def test_theoretical_bound_holds_for_arbitrary_candidates():
    rng = np.random.default_rng(0)
    for seed in range(5):
        a, u = darcy(1, seed, high=50.0, low=1.0)                   # contrast 50
        P = darcy_problem(a[0])
        for scale in (1e-3, 1e-1, 1.0):
            cand = u[0] + scale * np.abs(u[0]).max() * rng.standard_normal((N, N))
            b = energy_error_bound(P, cand)
            assert P.energy_norm(u[0] - cand) <= b["bound"] * (1 + 1e-10)


@pytest.mark.parametrize("method", ["fno", "deeponet"])
def test_darcy_ninety_percent_bound_covers_ninety_percent(darcy_runs, method):
    (res, err, viol), shifts = darcy_runs[method]
    assert viol == 0                                                # the theory is never violated
    study = coverage_study(res, err, alpha=0.1, n_cal=500, n_splits=200)
    assert 0.88 <= study["mean"] <= 0.92, study
    cal = ResidualErrorCalibrator(0.1).calibrate(res[:500], err[:500])
    rep = cal.evaluate(res[500:], err[500:], "held-out")
    assert rep.holds, str(rep)
    for name, (r, e, v) in shifts.items():                          # reported honestly; the theory still holds
        assert v == 0
        shifted = cal.evaluate(r, e, name)
        assert shifted.n == 150 and (shifted.holds or "input shift" in shifted.note)


def test_power_law_map_breaks_under_shift_and_says_so(darcy_runs):
    (res, err, _), shifts = darcy_runs["deeponet"]
    cal = ResidualErrorCalibrator(0.1, mode="power").fit(res[:250], err[:250]).calibrate(res[250:500], err[250:500])
    r, e, _ = shifts["contrast {1, 20}"]
    rep = cal.evaluate(r, e, "contrast {1, 20}")
    assert not rep.holds and rep.coverage < 0.5 and "input shift" in rep.note
    ratio = ResidualErrorCalibrator(0.1).calibrate(res[:500], err[:500]).evaluate(r, e, "contrast {1, 20}")
    assert ratio.holds                                             # the ratio map still covers this shift


def test_heat_conduction_coverage_and_theory():
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    k = np.exp(0.4 * np.sin(2 * np.pi * X) * np.cos(np.pi * Y))      # fixed conductivity, contrast ~2.2
    Q, T = heat(1300, 3, k)
    f = train("fno", {"width": 16, "modes": 8}, Q[:300], T[:300], epochs=25)
    res, err, viol = features([heat_conduction_problem(k, q) for q in Q[300:]], f(Q[300:]), T[300:])
    assert viol == 0
    study = coverage_study(res, err, alpha=0.1, n_cal=500, n_splits=200)
    assert 0.88 <= study["mean"] <= 0.92, study


def test_trust_check_and_report_text():
    cal = ResidualErrorCalibrator(0.1).calibrate(np.ones(50), np.linspace(0, 0.1, 50))
    assert cal.to_check(1.0, limit=0.2).status == "supports"
    assert cal.to_check(10.0, limit=0.2).status == "contradicts"
    assert cal.to_check(1.0).status == "not_run"
    rep = cal.evaluate(np.ones(10), np.full(10, 1.0), "shifted")
    assert "BROKEN" in str(rep)
    with pytest.raises(RuntimeError):
        ResidualErrorCalibrator().bound(np.ones(2))

"""Koopman operator and spectral analysis (pinneapple_analysis.dynamics.koopman).

References: linear maps with known eigenvalues; the harmonic oscillator (continuous eigenvalues +-i omega); the
nonlinear system x1' = mu x1, x2' = lam (x2 - x1^2), whose dictionary {x1, x2, x1^2} spans an invariant subspace with
continuous eigenvalues {mu, lam, 2 mu} (exact flow map used for the data); the 1-D heat equation (modal decay
exp(-nu k^2 t)); a low-rank signal plus noise for rank selection; the existing PyTorch DynamicModeDecomposition as a
cross-check of DMD.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import expm

from pinneapple_analysis import dynamics as dyn
from pinneapple_analysis.dynamics.koopman import (
    DMD,
    EDMD,
    KoopmanReconstruction,
    ObservableDictionary,
    SpectralAnalyzer,
    gavish_donoho_rank,
    snapshot_pairs,
)

MU, LAM = -0.05, -1.0
B = LAM / (LAM - 2 * MU)


def _flow(x1, x2, t):
    return x1 * np.exp(MU * t), (x2 - B * x1 ** 2) * np.exp(LAM * t) + B * x1 ** 2 * np.exp(2 * MU * t)


def _nonlinear_ensemble(n=10, steps=30, dt=0.1, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(steps) * dt
    return dyn.TrajectoryEnsemble([dyn.from_array(t, np.column_stack(_flow(*rng.uniform(-1, 1, 2), t)), ("x1", "x2"))
                                   for _ in range(n)])


def _linear(A, x0, n):
    x = np.zeros((n, len(x0)))
    x[0] = x0
    for i in range(n - 1):
        x[i + 1] = A @ x[i]
    return x


# =========================================================================== exact references
def test_dmd_recovers_the_eigenvalues_of_a_linear_map():
    A = np.array([[0.9, -0.2], [0.2, 0.9]])
    r = DMD(rank=2).fit(_linear(A, [1.0, 0.5], 50))
    assert np.max(r.details["operator"].eigenvalues.match(np.linalg.eigvals(A))) < 1e-12
    assert np.max(r.uncertainty["residuals"]) < 1e-10
    assert r.details["operator"].stability()["stable"]


def test_harmonic_oscillator_continuous_eigenvalues():
    w, dt = 2.0, 0.1
    x = _linear(expm(np.array([[0, 1], [-w * w, 0]]) * dt), [1.0, 0.0], 200)
    op = DMD(rank=2).fit(dyn.from_array(np.arange(200) * dt, x)).details["operator"]
    assert np.max(op.eigenvalues.match([1j * w, -1j * w], continuous=True)) < 1e-10
    modes = op.modes(x)
    assert all(m.frequency is not None and abs(abs(m.frequency) - w / (2 * np.pi)) < 1e-10 for m in modes)
    st = op.stability(tol=1e-8)
    assert st["n_neutral"] == 2 and st["stable"]
    pred = op.predict(x[0], 199)
    assert np.max(np.abs(pred - x)) < 1e-9


def test_edmd_on_the_exact_invariant_subspace():
    ens = _nonlinear_ensemble()
    d = ObservableDictionary.monomials(2, 2, constant=False, only=[(1, 0), (0, 1), (2, 0)])
    r = EDMD(d).fit(ens)
    op = r.details["operator"]
    assert np.max(op.eigenvalues.match([MU, LAM, 2 * MU], continuous=True)) < 1e-10
    assert np.max(r.uncertainty["residuals"]) < 1e-6
    # eigenfunctions: phi(F(x)) = lambda phi(x) on new points
    x = np.random.default_rng(9).uniform(-1, 1, (20, 2))
    fx = np.column_stack(_flow(x[:, 0], x[:, 1], 0.1))
    phi = op.eigenfunctions
    assert np.allclose(phi(fx), phi(x) * op.eigenvalues.discrete[None, :], atol=1e-10)
    rec = KoopmanReconstruction.evaluate(op, [t.x for t in list(ens)[:3]], 25)
    assert rec.one_step < 1e-12 and rec.multi_step[-1] < 1e-10


def test_residuals_flag_spurious_eigenvalues_of_a_larger_dictionary():
    ens = _nonlinear_ensemble()
    d = ObservableDictionary.monomials(2, 2, constant=True)                 # adds x1 x2 and x2^2: not invariant
    r = EDMD(d).fit(ens)
    op = r.details["operator"]
    om, res = op.eigenvalues.continuous, r.uncertainty["residuals"]
    true = [0.0, MU, 2 * MU, LAM]
    for ref in true:
        k = int(np.argmin(np.abs(om - ref)))
        assert abs(om[k] - ref) < 1e-8 and res[k] < 1e-6
    spurious = [k for k in range(om.size) if min(abs(om[k] - v) for v in true) > 1e-3]
    assert len(spurious) == 2 and all(res[k] > 1e-3 for k in spurious)
    rows = SpectralAnalyzer.spectrum(r, residual_tol=1e-6)
    assert sum(not row["trusted"] for row in rows) == 2


def test_heat_equation_snapshots_and_rank_selection():
    nu, xg, tt = 0.1, np.linspace(0, np.pi, 101), np.arange(40) * 0.05
    U = sum(a * np.exp(-nu * k * k * tt)[:, None] * np.sin(k * xg)[None, :] for k, a in ((1, 1.0), (2, 0.5), (3, 0.25)))
    tr = dyn.from_snapshots(tt, U, kind="simulation", coords={"x": xg})
    r = DMD(rank="gavish-donoho").fit(tr)
    assert r.estimate["rank"] == 3
    op = r.details["operator"]
    assert np.max(op.eigenvalues.match([-nu, -4 * nu, -9 * nu], continuous=True)) < 1e-9
    pred = op.predict(U[0], 39)
    assert np.max(np.abs(pred - U)) < 1e-9
    assert DMD(rank=0.999).fit(tr).estimate["rank"] <= 3


def test_gavish_donoho_finds_the_signal_rank_under_noise():
    rng = np.random.default_rng(0)
    signal = (rng.standard_normal((100, 4)) @ (rng.standard_normal((4, 200)) * np.array([[10], [8], [6], [4]]))).T
    Y = signal + 0.5 * rng.standard_normal(signal.shape)
    s = np.linalg.svd(Y.T, compute_uv=False)
    assert gavish_donoho_rank(s, Y.T.shape) == 4


def test_generator_edmd_from_derivatives():
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, (200, 2))
    F = np.column_stack([MU * X[:, 0], LAM * (X[:, 1] - X[:, 0] ** 2)])
    d = ObservableDictionary.monomials(2, 2, constant=False, only=[(1, 0), (0, 1), (2, 0)])
    g = EDMD(d).fit_generator(X, F, dt=0.1)
    ev = np.sort(g.estimate["generator_eigenvalues"].real)
    assert np.allclose(ev, sorted([MU, LAM, 2 * MU]), atol=1e-6)
    assert g.uncertainty["residuals"] is None
    op = g.details["operator"]                                             # expm(L dt) propagates like a map
    assert np.allclose(np.sort(np.abs(op.eigenvalues.discrete)), np.sort(np.exp(np.array([MU, LAM, 2 * MU]) * 0.1)),
                       atol=1e-6)


# =========================================================================== cross-check and noise
def test_dmd_matches_the_existing_torch_dmd():
    torch = pytest.importorskip("torch")
    from pinneapple_neural.architectures.rom.dmd import DynamicModeDecomposition

    rng = np.random.default_rng(3)
    A = np.linalg.qr(rng.standard_normal((5, 5)))[0] * 0.97
    X = _linear(A, rng.standard_normal(5), 60)
    lt, _, _ = DynamicModeDecomposition(r=5, center=False).fit(torch.tensor(X)).eig()
    ours = DMD(rank=5).fit(X).details["operator"].eigenvalues
    assert np.max(ours.match(lt.numpy())) < 1e-10


def test_tls_reduces_the_noise_bias():
    A = np.array([[0.95, -0.25], [0.25, 0.95]]) * 0.99
    x = _linear(A, [1.0, 0.0], 400)
    ref = np.linalg.eigvals(A)
    plain = SpectralAnalyzer.noise_sensitivity(DMD(rank=2), x, [0.2], ref, reps=10)[0.2]["mean_max_error"]
    tls = SpectralAnalyzer.noise_sensitivity(DMD(rank=2, tls=True), x, [0.2], ref, reps=10)[0.2]["mean_max_error"]
    assert tls < plain / 5
    noisy = x + 0.2 * np.std(x) * np.random.default_rng(0).standard_normal(x.shape)
    lam = DMD(rank=2).fit(noisy).details["operator"].eigenvalues.discrete
    assert np.all(np.abs(lam) < np.abs(ref))                               # noise biases towards damping


# =========================================================================== dictionaries and comparisons
def test_dictionaries():
    X = np.array([[0.5, -1.0], [2.0, 3.0]])
    m = ObservableDictionary.monomials(2, 2)
    assert m.names == ["1", "x0", "x1", "x0^2", "x0*x1", "x1^2"] and m.state_index == [1, 2]
    assert np.allclose(m(X)[:, 4], X[:, 0] * X[:, 1])
    r = ObservableDictionary.rbf(np.zeros((1, 2)), 1.0)
    assert r.state_index == [0, 1] and np.allclose(r(X)[:, 2], np.exp(-np.sum(X ** 2, axis=1)))
    f = ObservableDictionary.fourier(1, 2, period=1.0)
    assert len(f) == 5 and np.allclose(f(np.array([[0.25]]))[0, 2], 1.0)       # sin(2 pi 0.25)
    c = ObservableDictionary.custom([lambda X: X[:, 0] ** 3], ["x0^3"])
    assert (m + c).names[-1] == "x0^3"
    with pytest.raises(ValueError):
        ObservableDictionary.rbf(np.zeros((1, 2)), 1.0, kind="cubic")


def test_neural_dictionary_from_a_torch_module():
    torch = pytest.importorskip("torch")
    net = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Tanh())
    d = ObservableDictionary.neural(net, 3, d=2)
    X = np.random.default_rng(0).uniform(-1, 1, (5, 2))
    with torch.no_grad():
        ref = net(torch.as_tensor(X, dtype=torch.float32)).numpy()
    assert np.allclose(d(X)[:, 2:], ref, atol=1e-6) and d.state_index == [0, 1]
    r = EDMD(d, ridge=1e-8).fit(_nonlinear_ensemble())
    assert r.estimate["m"] == 5


def test_comparison_with_state_space_regression_and_order_selection():
    ens = list(_nonlinear_ensemble(n=12))
    train, test = dyn.TrajectoryEnsemble(ens[:8]), [t.x for t in ens[8:]]
    d = ObservableDictionary.monomials(2, 2, constant=False, only=[(1, 0), (0, 1), (2, 0)])
    cmp = SpectralAnalyzer.compare_with_state_regression(d, train, test, horizon=25)
    assert cmp["koopman_error"][-1] < 1e-10                                 # exact lifted linear dynamics
    assert cmp["state_regression_error"][-1] < 1e-10                        # the flow map is also in the span here
    # a limit cycle has no finite invariant polynomial subspace: linear lifting loses to the nonlinear regression
    from scipy.integrate import solve_ivp

    rng = np.random.default_rng(0)
    tt = np.arange(200) * 0.05
    vdp = [solve_ivp(lambda t, y: [y[1], (1 - y[0] ** 2) * y[1] - y[0]], (0, tt[-1]), rng.uniform(-2, 2, 2),
                     t_eval=tt, rtol=1e-10, atol=1e-12).y.T for _ in range(20)]
    cmp2 = SpectralAnalyzer.compare_with_state_regression(
        ObservableDictionary.monomials(2, 3), dyn.TrajectoryEnsemble([dyn.from_array(tt, v) for v in vdp[:15]]),
        vdp[15:], horizon=100)
    assert cmp2["koopman_error"][-1] > 5 * cmp2["state_regression_error"][-1]
    sel = SpectralAnalyzer.select_order(lambda deg: EDMD(ObservableDictionary.monomials(2, deg, constant=False)),
                                        [1, 2], train, test, horizon=20)
    assert sel["best"] == "2"


def test_snapshot_pairs_and_failures():
    t = np.arange(10.0)
    ens = dyn.TrajectoryEnsemble([dyn.from_array(t, t), dyn.from_array(t, -t)])
    X0, X1, dt = snapshot_pairs(ens)
    assert X0.shape == (18, 1) and dt == 1.0
    assert np.allclose(np.abs(X1 - X0), 1.0)                                 # pairs never cross members
    with pytest.raises(ValueError, match="regular"):
        snapshot_pairs(dyn.from_array(np.array([0.0, 1.0, 3.0]), [1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="time steps"):
        snapshot_pairs(dyn.TrajectoryEnsemble([dyn.from_array(t, t), dyn.from_array(2 * t, t)]))
    x = np.ones((6, 1))
    x[2] = np.nan
    X0, X1, _ = snapshot_pairs(x)
    assert X0.shape[0] == 3                                                  # pairs touching the NaN dropped
    small = EDMD(ObservableDictionary.monomials(2, 3)).fit(np.random.default_rng(0).normal(size=(8, 2)))
    assert small.diagnostic("n_pairs").status == "warn"
    with pytest.raises(ValueError):
        DMD(rank="elbow").fit(np.random.default_rng(0).normal(size=(10, 3)))


def test_contract():
    r = DMD(rank=2).fit(_linear(np.eye(2) * 0.9, [1.0, 1.0], 20))
    assert DMD.card.role == "trajectory_operator" and EDMD.card.role == "trajectory_operator"
    assert "autonomous dynamics" in {a.name for a in r.unchecked_assumptions}
    assert r.to_study_result().metrics["estimate.rank"] == 2.0
    assert math.isfinite(r.estimate["spectral_radius"])


def test_benchmarks_quick_all_pass():
    from pinneapple_analysis.dynamics.benchmarks import koopman as bench

    failed = [c.to_dict() for c in bench.run(quick=True) if not c.passed]
    assert not failed, failed

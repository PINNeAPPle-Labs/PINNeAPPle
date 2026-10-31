"""Koopman / DMD benchmarks against exact spectra.

| Case | Reference | Source |
|---|---|---|
| ``linear_map`` | eigenvalues of ``A`` | exact |
| ``oscillator`` | ``+-i omega`` | harmonic oscillator, exact flow map sampled at ``dt`` |
| ``nonlinear_subspace`` | ``{mu, lam, 2 mu}`` | ``x1' = mu x1, x2' = lam (x2 - x1^2)``, dictionary ``{x1, x2, x1^2}`` |
| ``nonlinear_residual_*`` | residual 0 for the 4 true eigenvalues, > 0 for spurious ones | the same with all monomials of degree <= 2 |
| ``generator_edmd`` | ``{mu, lam, 2 mu}`` | the same system from derivative data |
| ``heat_modes`` | ``-nu k^2`` | 1-D heat equation, 3 Fourier modes, 101-point snapshots |
| ``gavish_donoho_rank`` | 4 | rank-4 signal plus white noise, 100 x 199 |
| ``torch_dmd_crosscheck`` | the existing ``DynamicModeDecomposition`` | same data, rank 5 |
| ``noise_bias_dmd`` / ``_tls`` | the true eigenvalues | 20 % white noise on a damped rotation |
| ``koopman_vs_state_regression`` | 0 | 25-step forecast error, same dictionary (both exact on the invariant subspace) |
| ``van_der_pol_deg*`` | – | limit cycle: Koopman with polynomial dictionaries vs the nonlinear state regression |
"""
from __future__ import annotations

import time

import numpy as np
from scipy.linalg import expm

from ..koopman import DMD, EDMD, ObservableDictionary, SpectralAnalyzer, gavish_donoho_rank
from ..trajectories import TrajectoryEnsemble, from_array, from_snapshots
from ._case import BenchmarkCase

__all__ = ["run"]
MU, LAM = -0.05, -1.0


def _lin(A, x0, n):
    x = np.zeros((n, len(x0)))
    x[0] = x0
    for i in range(n - 1):
        x[i + 1] = A @ x[i]
    return x


def _flow(x1, x2, t):
    b = LAM / (LAM - 2 * MU)
    return x1 * np.exp(MU * t), (x2 - b * x1 ** 2) * np.exp(LAM * t) + b * x1 ** 2 * np.exp(2 * MU * t)


def _timed(f):
    t0 = time.perf_counter()
    out = f()
    return out, time.perf_counter() - t0


def run(quick: bool = False, seed: int = 0) -> list[BenchmarkCase]:
    rng = np.random.default_rng(seed)
    cases: list[BenchmarkCase] = []

    A = np.array([[0.9, -0.2], [0.2, 0.9]])
    r, c = _timed(lambda: DMD(rank=2).fit(_lin(A, [1.0, 0.5], 50)))
    cases.append(BenchmarkCase("linear_map", "max eigenvalue error", 0.0, "eig(A)",
                               float(np.max(r.details["operator"].eigenvalues.match(np.linalg.eigvals(A)))),
                               tolerance=1e-10, cost_s=c))
    w, dt = 2.0, 0.1
    xo = _lin(expm(np.array([[0, 1], [-w * w, 0]]) * dt), [1.0, 0.0], 200)
    r, c = _timed(lambda: DMD(rank=2).fit(from_array(np.arange(200) * dt, xo)))
    cases.append(BenchmarkCase("oscillator", "max continuous-eigenvalue error", 0.0, "+-i omega",
                               float(np.max(r.details["operator"].eigenvalues.match([1j * w, -1j * w],
                                                                                    continuous=True))),
                               tolerance=1e-9, cost_s=c))

    t = np.arange(30) * 0.1
    ens = TrajectoryEnsemble([from_array(t, np.column_stack(_flow(*rng.uniform(-1, 1, 2), t))) for _ in range(10)])
    d3 = ObservableDictionary.monomials(2, 2, constant=False, only=[(1, 0), (0, 1), (2, 0)])
    r, c = _timed(lambda: EDMD(d3).fit(ens))
    cases.append(BenchmarkCase("nonlinear_subspace", "max continuous-eigenvalue error", 0.0, "{mu, lam, 2 mu}",
                               float(np.max(r.details["operator"].eigenvalues.match([MU, LAM, 2 * MU],
                                                                                    continuous=True))),
                               tolerance=1e-9, cost_s=c,
                               convergence={"max_residual": float(np.max(r.uncertainty["residuals"]))}))
    d6 = ObservableDictionary.monomials(2, 2, constant=True)
    r6, c = _timed(lambda: EDMD(d6).fit(ens))
    om, res = r6.details["operator"].eigenvalues.continuous, r6.uncertainty["residuals"]
    true = [0.0, MU, 2 * MU, LAM]
    is_true = np.array([min(abs(o - v) for v in true) < 1e-6 for o in om])
    cases.append(BenchmarkCase("nonlinear_residual_true", "max residual of the 4 true eigenpairs", 0.0, "invariant",
                               float(np.max(res[is_true])), tolerance=1e-6, cost_s=c,
                               convergence={"n_true_found": int(is_true.sum())}))
    cases.append(BenchmarkCase("nonlinear_residual_spurious", "min residual of the spurious eigenpairs", 0.0,
                               "non-invariant monomials (x1 x2, x2^2)", float(np.min(res[~is_true])),
                               tolerance=float("inf"), cost_s=0.0,
                               convergence={"spurious_continuous": [complex(v) for v in om[~is_true]]},
                               limitations="no pass/fail: shows the size of the residual that flags them"))

    X = rng.uniform(-1, 1, (200, 2))
    F = np.column_stack([MU * X[:, 0], LAM * (X[:, 1] - X[:, 0] ** 2)])
    g, c = _timed(lambda: EDMD(d3).fit_generator(X, F))
    ev = np.sort(g.estimate["generator_eigenvalues"].real)
    cases.append(BenchmarkCase("generator_edmd", "max generator-eigenvalue error", 0.0, "{mu, lam, 2 mu}",
                               float(np.max(np.abs(ev - np.sort([MU, LAM, 2 * MU])))), tolerance=1e-6, cost_s=c,
                               limitations="dictionary derivatives by central differences (h = 1e-5)"))

    nu, xg, tt = 0.1, np.linspace(0, np.pi, 101), np.arange(40) * 0.05
    U = sum(a * np.exp(-nu * k * k * tt)[:, None] * np.sin(k * xg)[None, :] for k, a in ((1, 1.0), (2, 0.5),
                                                                                        (3, 0.25)))
    r, c = _timed(lambda: DMD(rank="gavish-donoho").fit(from_snapshots(tt, U)))
    cases.append(BenchmarkCase("heat_modes", "max continuous-eigenvalue error", 0.0, "-nu k^2, k = 1..3",
                               float(np.max(r.details["operator"].eigenvalues.match([-nu, -4 * nu, -9 * nu],
                                                                                    continuous=True))),
                               tolerance=1e-8, cost_s=c, convergence={"rank": r.estimate["rank"]}))

    sig = (rng.standard_normal((100, 4)) @ (rng.standard_normal((4, 200)) * np.array([[10], [8], [6], [4]]))).T
    Y = sig + 0.5 * rng.standard_normal(sig.shape)
    s = np.linalg.svd(Y.T, compute_uv=False)
    cases.append(BenchmarkCase("gavish_donoho_rank", "selected rank", 4.0, "true signal rank",
                               float(gavish_donoho_rank(s, Y.T.shape)), tolerance=0.0))

    try:
        import torch

        from pinneapple_neural.architectures.rom.dmd import DynamicModeDecomposition

        Aq = np.linalg.qr(rng.standard_normal((5, 5)))[0] * 0.97
        Xq = _lin(Aq, rng.standard_normal(5), 60)
        lt, _, _ = DynamicModeDecomposition(r=5, center=False).fit(torch.tensor(Xq)).eig()
        ours = DMD(rank=5).fit(Xq).details["operator"].eigenvalues
        cases.append(BenchmarkCase("torch_dmd_crosscheck", "max eigenvalue difference", 0.0,
                                   "rom.DynamicModeDecomposition", float(np.max(ours.match(lt.numpy()))),
                                   tolerance=1e-10))
    except ImportError:
        pass

    An = np.array([[0.95, -0.25], [0.25, 0.95]]) * 0.99
    xn = _lin(An, [1.0, 0.0], 400)
    ref = np.linalg.eigvals(An)
    reps = 10 if quick else 50
    for tls in (False, True):
        out, c = _timed(lambda tls=tls: SpectralAnalyzer.noise_sensitivity(DMD(rank=2, tls=tls), xn, [0.2], ref,
                                                                           reps=reps, seed=seed)[0.2])
        cases.append(BenchmarkCase(f"noise_bias_{'tls' if tls else 'dmd'}", "mean max eigenvalue error, 20 % noise",
                                   0.0, "eig(A)", out["mean_max_error"], tolerance=float("inf"), cost_s=c,
                                   convergence={"std": out["std"], "replicates": reps},
                                   limitations="no pass/fail: the noise floor of each estimator"))

    ens12 = list(TrajectoryEnsemble([from_array(t, np.column_stack(_flow(*rng.uniform(-1, 1, 2), t)))
                                     for _ in range(12)]))
    cmp, c = _timed(lambda: SpectralAnalyzer.compare_with_state_regression(
        d3, TrajectoryEnsemble(ens12[:8]), [m.x for m in ens12[8:]], horizon=25))
    cases.append(BenchmarkCase("koopman_vs_state_regression", "25-step forecast RMS error, Koopman", 0.0,
                               "exact invariant subspace", float(cmp["koopman_error"][-1]), tolerance=1e-9, cost_s=c,
                               convergence={"state_regression_error": float(cmp["state_regression_error"][-1])},
                               limitations="here the flow map is also linear in the dictionary, so both are exact"))

    from scipy.integrate import solve_ivp

    tv = np.arange(200) * 0.05
    vdp = [from_array(tv, solve_ivp(lambda _t, y: [y[1], (1 - y[0] ** 2) * y[1] - y[0]], (0, tv[-1]),
                                    rng.uniform(-2, 2, 2), t_eval=tv, rtol=1e-10, atol=1e-12).y.T)
           for _ in range(20)]
    for deg in (3, 5):
        cv, c = _timed(lambda deg=deg: SpectralAnalyzer.compare_with_state_regression(
            ObservableDictionary.monomials(2, deg), TrajectoryEnsemble(vdp[:15]), [m.x for m in vdp[15:]],
            horizon=100))
        cases.append(BenchmarkCase(f"van_der_pol_deg{deg}", "100-step forecast RMS error, Koopman (linear lifting)",
                                   0.0, "no exact reference (limit cycle)", float(cv["koopman_error"][-1]),
                                   tolerance=float("inf"), cost_s=c,
                                   convergence={"state_regression_error": float(cv["state_regression_error"][-1]),
                                                "koopman_error_10_steps": float(cv["koopman_error"][10])},
                                   limitations="no pass/fail: a limit cycle has no finite invariant polynomial "
                                               "subspace; the nonlinear state regression forecasts it better"))
    return cases

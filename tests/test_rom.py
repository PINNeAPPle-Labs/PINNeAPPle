"""Reduced-order models on problems with known answers."""
import numpy as np
import torch

from pinneapple_neural.architectures.rom import (
    POD,
    DynamicModeDecomposition,
    OperatorInference,
    ParametricPOD,
    latin_hypercube,
)


def _travelling_wave(T=120, n=200, f=0.05):
    x = np.linspace(0, 2 * np.pi, n)
    t = np.arange(T)
    return np.sin(x[None] - 2 * np.pi * f * t[:, None]) + 0.3 * np.sin(2 * (x[None] - 2 * np.pi * f * t[:, None]))


def test_pod_finds_the_mode_pairs_of_a_travelling_wave():
    X = torch.tensor(_travelling_wave(), dtype=torch.float64)
    pod = POD(r=8, center=True).fit(X)
    evr = pod.explained_variance_ratio_.numpy()
    assert evr[:4].sum() > 0.999                       # two harmonics = two pairs of modes
    assert abs(evr[0] / evr[1] - 1) < 0.05


def test_dmd_recovers_the_frequency_and_forecasts():
    U = _travelling_wave()
    dmd = DynamicModeDecomposition(r=4).fit(torch.tensor(U[:80], dtype=torch.float64)[None])
    lam, _, _ = dmd.eig()
    f = np.sort(np.abs(np.angle(lam.numpy())) / (2 * np.pi))
    assert abs(f[0] - 0.05) < 1e-3 and abs(f[-1] - 0.10) < 1e-3
    pred = dmd.rollout(torch.tensor(U[79][None], dtype=torch.float64), 40)[0].numpy()
    assert np.abs(pred - U[79:]).max() < 1e-3


def test_opinf_quadratic_latent_dynamics():
    # a_{t+1} = A a_t + H (a_t x a_t): a damped rotation with a weak quadratic term
    th, r = 0.2, 2
    A = 0.98 * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    H = np.zeros((r, r * r))
    H[0, 3] = 0.05
    a = [np.array([1.0, 0.0])]
    for _ in range(150):
        a.append(A @ a[-1] + H @ np.kron(a[-1], a[-1]))
    a = torch.tensor(np.array(a), dtype=torch.float64)[None]
    oi = OperatorInference(r=r, use_quadratic=True, use_bias=False, l2_linear=1e-12, l2_quad=1e-12,
                           center=False).fit(a[:, :100])
    pred = oi.rollout(a[:, 99], 50)[0].numpy()
    assert np.abs(pred - a[0, 99:150].numpy()).max() < 1e-6


def test_parametric_pod_beats_nearest_design():
    x = np.linspace(0, 1, 300)
    field = lambda P: np.array([np.exp(-((x - p[0]) / (0.1 + 0.2 * p[1])) ** 2) * (1 + p[1]) for p in P])  # noqa: E731
    box = {"c": (0.2, 0.8), "w": (0.0, 1.0)}
    Ptr, _ = latin_hypercube(60, box, 0)
    Pte, _ = latin_hypercube(20, {"c": (0.3, 0.7), "w": (0.1, 0.9)}, 3)
    e_nn = ParametricPOD(regressor="nearest").fit(Ptr, field(Ptr)).error(Pte, field(Pte))
    for reg in ("rbf", "gpr"):
        rom = ParametricPOD(regressor=reg, r=20, energy=None).fit(Ptr, field(Ptr))
        e = rom.error(Pte, field(Pte))
        assert np.median(e) < 0.3 * np.median(e_nn), reg
    X, s = ParametricPOD(regressor="gpr", r=20, energy=None).fit(Ptr, field(Ptr)).predict(Pte, return_std=True)
    assert s.shape == X.shape and (s >= 0).all()
    # the amplitude / shape split keeps the error for responses spanning orders of magnitude
    big = lambda P: field(P) * (10 ** (3 * P[:, 1]))[:, None]                           # noqa: E731
    e0 = ParametricPOD(regressor="rbf", r=20, energy=None).fit(Ptr, big(Ptr)).error(Pte, big(Pte))
    e1 = ParametricPOD(regressor="rbf", r=20, energy=None, normalize=True).fit(Ptr, big(Ptr)).error(Pte, big(Pte))
    assert np.median(e1) < np.median(e0)


def test_opinf_continuous_recovers_linear_operator():
    A = np.array([[-0.1, -2.0], [2.0, -0.1]])
    dt, T = 1e-3, 4000
    a = [np.array([1.0, 0.0])]
    from scipy.linalg import expm
    M = expm(A * dt)
    for _ in range(T - 1):
        a.append(M @ a[-1])
    a = torch.tensor(np.array(a))[None]
    oi = OperatorInference(r=2, use_quadratic=False, use_bias=False, l2_linear=1e-12, center=False).fit_continuous(a, dt=dt)
    W = oi.W.numpy()                                   # features @ W = da/dt (row-vector convention)
    assert np.abs(W.T - A).max() < 1e-4

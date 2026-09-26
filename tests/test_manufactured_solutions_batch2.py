"""Exact solutions plugged into compiled residuals (item 2 of the 2026-09-24 follow-ups).

Each equation gets a closed-form solution that must give a ~zero residual, and a perturbed one
that must not. Every exact solution below is derived in its docstring so it can be checked by hand.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from pinneapple_physics.pde_environment.scales import ScaleSpec
from pinneapple_physics.pde_environment.spec import PDETermSpec, ProblemSpec
from pinneapple_physics.pinn_solver.compiler.compile import compile_problem


class _Exact(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x)


def _residual(kind, coords, fields, params, fn, x):
    pde = PDETermSpec(kind=kind, fields=tuple(fields), coords=tuple(coords), params=params)
    spec = ProblemSpec(name=f"_mms_{kind}", dim=len(coords), coords=tuple(coords), fields=tuple(fields),
                       pde=pde, conditions=(), scales=ScaleSpec())
    loss_fn = compile_problem(spec)
    nc, nf = len(coords), len(fields)
    batch = {"x_col": x.requires_grad_(True), "ctx": {},
             "x_bc": torch.zeros((0, nc)), "y_bc": torch.zeros((0, nf)),
             "x_ic": torch.zeros((0, nc)), "y_ic": torch.zeros((0, nf)),
             "x_data": torch.zeros((0, nc)), "y_data": torch.zeros((0, nf))}
    out = loss_fn(_Exact(fn), None, batch)
    return float((out["pde"] if isinstance(out, dict) and "pde" in out else out["total"]).item())


def _pts(n, lows, highs, seed=0):
    g = torch.Generator().manual_seed(seed)
    lo, hi = torch.tensor(lows, dtype=torch.float32), torch.tensor(highs, dtype=torch.float32)
    return lo + (hi - lo) * torch.rand(n, len(lows), generator=g)


def _pair(kind, coords, fields, params, exact, wrong, x, tol=1e-6):
    assert _residual(kind, coords, fields, params, exact, x.clone()) < tol
    assert _residual(kind, coords, fields, params, wrong, x.clone()) > 1e3 * tol


def test_advection_diffusion_decaying_travelling_wave():
    """c = exp(-kappa k^2 t) sin(k (x - u0 t)): c_t = -kappa k^2 c - u0 k exp(.)cos(.),
    u0 c_x = u0 k exp(.)cos(.), kappa c_xx = -kappa k^2 c  ->  c_t + u0 c_x - kappa c_xx = 0."""
    kappa, u0, k = 0.05, 0.7, 2.0
    exact = lambda X: torch.exp(-kappa * k * k * X[:, 0:1]) * torch.sin(k * (X[:, 1:2] - u0 * X[:, 0:1]))
    wrong = lambda X: torch.exp(-kappa * k * k * X[:, 0:1]) * torch.sin(k * (X[:, 1:2] + u0 * X[:, 0:1]))
    _pair("advection_diffusion", ("t", "x"), ("c",), {"kappa": kappa, "u0": u0}, exact, wrong,
          _pts(256, [0, -1], [1, 1]))


def test_wave_standing_mode():
    """u = sin(pi x) cos(c pi t): u_tt = -c^2 pi^2 u = c^2 u_xx."""
    c = 1.5
    exact = lambda X: torch.sin(math.pi * X[:, 1:2]) * torch.cos(c * math.pi * X[:, 0:1])
    wrong = lambda X: torch.sin(math.pi * X[:, 1:2]) * torch.cos(2 * c * math.pi * X[:, 0:1])
    _pair("wave_equation", ("t", "x"), ("u",), {"c": c}, exact, wrong, _pts(256, [0, 0], [1, 1]))


def test_helmholtz_eigenfunction():
    """u = sin(a x) sin(b y) with k^2 = a^2 + b^2: lap u + k^2 u = 0."""
    a, b = 2.0, 3.0
    exact = lambda X: torch.sin(a * X[:, 0:1]) * torch.sin(b * X[:, 1:2])
    wrong = lambda X: torch.sin(a * X[:, 0:1]) * torch.sin(2 * b * X[:, 1:2])
    _pair("helmholtz", ("x", "y"), ("u",), {"k": math.sqrt(a * a + b * b)}, exact, wrong, _pts(256, [0, 0], [1, 1]))


def test_sir_logistic_limit():
    """gamma = 0 (SI model): I = N / (1 + (N/I0 - 1) exp(-beta t)), S = N - I, R = 0 solves
    S' = -beta S I / N, I' = beta S I / N (the logistic equation for I)."""
    N, beta, I0 = 1000.0, 0.8, 10.0

    def exact(X):
        I = N / (1 + (N / I0 - 1) * torch.exp(-beta * X[:, 0:1]))
        return torch.cat([N - I, I, torch.zeros_like(I)], 1)

    def wrong(X):
        I = N / (1 + (N / I0 - 1) * torch.exp(-2 * beta * X[:, 0:1]))
        return torch.cat([N - I, I, torch.zeros_like(I)], 1)

    x = _pts(256, [0], [10])
    params = {"beta": beta, "gamma": 0.0, "N": N}
    assert _residual("sir_ode", ("t",), ("S", "I", "R"), params, exact, x.clone()) < 1e-3 * N
    assert _residual("sir_ode", ("t",), ("S", "I", "R"), params, wrong, x.clone()) > 1.0


def test_pk_two_compartment_eigen_solution():
    """Linear system C' = M C, M = [[-(k12+kel), k21], [k12, -k21]]: C(t) = V exp(L t) V^-1 C0."""
    k12, k21, kel = 0.5, 0.3, 0.2
    M = np.array([[-(k12 + kel), k21], [k12, -k21]])
    lam, V = np.linalg.eig(M)
    coef = np.linalg.solve(V, np.array([10.0, 0.0]))
    Vt, lt, ct = (torch.tensor(a, dtype=torch.float32) for a in (V.real, lam.real, coef.real))

    def exact(X):
        e = torch.exp(X[:, 0:1] * lt[None, :]) * ct[None, :]
        return e @ Vt.T

    wrong = lambda X: exact(0.8 * X)
    _pair("pk_two_compartment_ode", ("t",), ("C1", "C2"), {"k12": k12, "k21": k21, "kel": kel},
          exact, wrong, _pts(256, [0], [10]), tol=1e-8)


def test_opinion_dynamics_stationary_kink():
    """Allen-Cahn: u = tanh(k x), u'' = -2 k^2 u (1 - u^2); with k^2 = alpha/(2D) the diffusion
    and reaction terms cancel and u_t = 0."""
    D, alpha = 0.01, 1.0
    k = math.sqrt(alpha / (2 * D))
    exact = lambda X: torch.tanh(k * X[:, 1:2]) + 0 * X[:, 0:1]
    wrong = lambda X: torch.tanh(2 * k * X[:, 1:2]) + 0 * X[:, 0:1]
    _pair("opinion_dynamics_2d", ("t", "x", "y"), ("u",), {"D": D, "alpha": alpha}, exact, wrong,
          _pts(256, [0, -0.5, -0.5], [1, 0.5, 0.5]), tol=1e-5)


def test_maxwell_te_plane_wave():
    """Ex = 0, Ey = cos(kx - wt), Hz = (eps w / k) cos(kx - wt), w^2 = k^2 / (mu eps), sigma = 0."""
    eps, mu, k = 2.0, 0.5, 3.0
    w = k / math.sqrt(mu * eps)

    def mk(omega):
        def f(X):
            ph = k * X[:, 1:2] - omega * X[:, 0:1]
            return torch.cat([0 * ph, torch.cos(ph), eps * w / k * torch.cos(ph)], 1)
        return f

    exact = mk(w)
    _pair("maxwell_te", ("t", "x", "y"), ("Ex", "Ey", "Hz"), {"epsilon": eps, "mu": mu, "sigma": 0.0},
          exact, mk(1.3 * w), _pts(256, [0, 0, 0], [1, 1, 1]))


def test_stokes_polynomial_flow():
    """u = x^2, v = -2xy (div-free); lap u = 2, lap v = 0 -> p = 2 mu x balances momentum."""
    mu = 0.7
    exact = lambda X: torch.cat([X[:, 0:1] ** 2, -2 * X[:, 0:1] * X[:, 1:2], 2 * mu * X[:, 0:1]], 1)
    wrong = lambda X: torch.cat([X[:, 0:1] ** 2, -2 * X[:, 0:1] * X[:, 1:2], -2 * mu * X[:, 0:1]], 1)
    _pair("stokes", ("x", "y"), ("u", "v", "p"), {"mu": mu}, exact, wrong, _pts(256, [-1, -1], [1, 1]))


def test_brinkman_boundary_layer_profile():
    """v = 0, p = 0, u = cosh(lambda y): -mu_eff u'' + (mu/K) u = 0 for lambda^2 = mu / (K mu_eff)."""
    mu, mu_eff, K = 1.0, 0.5, 0.2
    lam = math.sqrt(mu / (K * mu_eff))
    exact = lambda X: torch.cat([torch.cosh(lam * X[:, 1:2]), 0 * X[:, 0:1], 0 * X[:, 0:1]], 1)
    wrong = lambda X: torch.cat([torch.cosh(0.5 * lam * X[:, 1:2]), 0 * X[:, 0:1], 0 * X[:, 0:1]], 1)
    _pair("brinkman", ("x", "y"), ("u", "v", "p"), {"mu": mu, "mu_eff": mu_eff, "K": K}, exact, wrong,
          _pts(256, [-1, -1], [1, 1]), tol=1e-5)

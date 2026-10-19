"""Synthetic time series with known ground truth, for forecasting studies.

Every generator returns a :class:`SyntheticSeries`: the clean state, the noisy observation that a forecaster sees,
the true parameters, and (for the chaotic systems) the largest Lyapunov exponent computed from the equations with
:func:`largest_lyapunov_benettin`, so a forecast horizon can be stated in Lyapunov times.

Systems
-------
* Lorenz-63 (Lorenz 1963, J. Atmos. Sci. 20:130), sigma=10, rho=28, beta=8/3.
* Lorenz-96 (Lorenz 1996, ECMWF Seminar on Predictability), cyclic, forcing F.
* Mackey-Glass delay equation (Mackey & Glass 1977, Science 197:287), dx/dt = beta x_tau / (1 + x_tau^n) - gamma x.
* Ornstein-Uhlenbeck process dx = -theta x dt + sigma dW, sampled exactly (an AR(1)).
* Stochastically forced damped oscillator x'' + 2 zeta omega x' + omega^2 x = sigma xi(t), sampled exactly
  (matrix exponential and Van Loan's integral for the process-noise covariance).
* Double-well Langevin equation dx = (x - x^3) dt + sigma dW: noise-driven switching between two regimes.
* White noise and a Gaussian random walk (controls where nothing can beat the baseline).

The vector fields (:func:`lorenz63_rhs`, :func:`lorenz96_rhs`, :func:`mackey_glass_rhs`, :func:`double_well_drift`)
accept NumPy arrays or torch tensors of shape (..., d), so the same definition drives data generation, the fitted
physical models, and the known part of a universal differential equation.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "SyntheticSeries", "rk4_step", "lorenz63_rhs", "lorenz96_rhs", "mackey_glass_rhs", "double_well_drift",
    "mackey_glass_step", "largest_lyapunov_benettin", "mackey_glass_lyapunov", "lorenz63", "lorenz96", "mackey_glass",
    "ornstein_uhlenbeck", "stochastic_oscillator", "oscillator_transition", "double_well", "white_noise",
    "random_walk", "ode_stepper", "mackey_glass_stepper", "linear_sde_stepper",
]


def _is_torch(x) -> bool:
    return type(x).__module__.startswith("torch")


def _stack(parts, x):
    if _is_torch(x):
        import torch
        return torch.stack(parts, dim=-1)
    return np.stack(parts, axis=-1)


def _roll(x, k):
    if _is_torch(x):
        import torch
        return torch.roll(x, k, dims=-1)
    return np.roll(x, k, axis=-1)


def rk4_step(f: Callable, x, dt: float):
    """One classical Runge-Kutta step of ``dx/dt = f(x)`` (NumPy or torch, batched over leading axes)."""
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def lorenz63_rhs(x, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0):
    a, b, c = x[..., 0], x[..., 1], x[..., 2]
    return _stack([sigma * (b - a), a * (rho - c) - b, a * b - beta * c], x)


def lorenz96_rhs(x, F: float = 8.0):
    return (_roll(x, -1) - _roll(x, 2)) * _roll(x, 1) - x + F


def mackey_glass_rhs(x, x_tau, beta: float = 0.2, gamma: float = 0.1, n: float = 10.0):
    return beta * x_tau / (1.0 + np.abs(x_tau) ** n) - gamma * x


def double_well_drift(x, a: float = 1.0, b: float = 1.0):
    """Drift of a particle in the quartic potential V(x) = -a x^2 / 2 + b x^4 / 4 (minima at +-sqrt(a / b))."""
    return a * x - b * x ** 3


def mackey_glass_step(hist: np.ndarray, dt: float, beta: float = 0.2, gamma: float = 0.1, n: float = 10.0
                      ) -> np.ndarray:
    """Advance Mackey-Glass histories by one RK4 step of size ``dt`` (method of steps).

    ``hist`` has shape (B, m + 1): the last m + 1 values on a grid of spacing dt, with tau = m dt; the newest value
    is ``hist[:, -1]``. The delayed value at the half step is the mean of its two grid neighbours. Returns the
    shifted history (B, m + 1).
    """
    x = hist[:, -1]
    xt0, xt1 = hist[:, 0], hist[:, 1]
    xth = 0.5 * (xt0 + xt1)
    f = lambda y, yt: mackey_glass_rhs(y, yt, beta, gamma, n)  # noqa: E731
    k1 = f(x, xt0)
    k2 = f(x + 0.5 * dt * k1, xth)
    k3 = f(x + 0.5 * dt * k2, xth)
    k4 = f(x + dt * k3, xt1)
    new = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.concatenate([hist[:, 1:], new[:, None]], axis=1)


def largest_lyapunov_benettin(step: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, *, step_time: float,
                              n_steps: int = 20000, transient: int = 2000, d0: float = 1e-8,
                              seed: int = 0) -> float:
    """Largest Lyapunov exponent (per unit time) of a deterministic map ``x -> step(x)`` by two-trajectory
    renormalisation (Benettin, Galgani, Giorgilli & Strelcyn 1980, Meccanica 15:9).

    ``step`` acts on a batch (B, d); ``step_time`` is the time one call advances. The separation is renormalised to
    ``d0`` after every step and the exponent is the mean log growth rate.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, float)[None].copy()
    for _ in range(int(transient)):
        x = step(x)
    v = rng.standard_normal(x.shape)
    z = np.concatenate([x, x + d0 * v / np.linalg.norm(v)])        # both trajectories in one batched step
    total = 0.0
    for _ in range(int(n_steps)):
        z = step(z)
        d = float(np.linalg.norm(z[1] - z[0]))
        total += np.log(d / d0)
        z[1] = z[0] + (z[1] - z[0]) * (d0 / d)
    return total / (n_steps * step_time)


def mackey_glass_lyapunov(*, dt_int: float = 0.1, total_time: float = 50000.0, transient_time: float = 2000.0,
                          tau: float = 17.0, beta: float = 0.2, gamma: float = 0.1, n: float = 10.0,
                          seed: int = 0) -> float:
    """Largest Lyapunov exponent (per unit time, natural log) of the Mackey-Glass delay equation.

    The state of a delay equation is its history on [t - tau, t]; here the tau / dt_int + 1 grid values. A random
    perturbation of the whole history is followed with :func:`largest_lyapunov_benettin` (renormalised every
    step, Euclidean norm over the grid) after a spin-up of 500 time units and a further ``transient_time``.
    """
    rng = np.random.default_rng(seed)
    m = int(round(tau / dt_int))
    hist = (1.2 + 0.05 * rng.standard_normal(m + 1))[None]
    step = lambda h: mackey_glass_step(h, dt_int, beta, gamma, n)  # noqa: E731
    for _ in range(int(500 / dt_int)):
        hist = step(hist)
    return largest_lyapunov_benettin(step, hist[0], step_time=dt_int, n_steps=int(total_time / dt_int),
                                     transient=int(transient_time / dt_int), seed=seed)


@dataclass
class SyntheticSeries:
    """A synthetic series and its ground truth.

    ``state`` is the clean state (T, d); ``obs`` the observed variables with observation noise (T, d_obs);
    ``target`` the column of ``obs`` that is forecast; ``noise_std`` the absolute observation-noise standard
    deviation (per observed column); ``lyapunov`` the largest Lyapunov exponent per unit time (None when the system
    is not chaotic); ``kind`` one of ``"chaotic"``, ``"stochastic"``, ``"noise"``.
    """

    name: str
    t: np.ndarray
    state: np.ndarray
    obs: np.ndarray
    dt: float
    target: int = 0
    observed: list[int] = field(default_factory=lambda: [0])
    params: dict[str, Any] = field(default_factory=dict)
    noise_std: Any = 0.0
    noise_level: float = 0.0
    lyapunov: float | None = None
    kind: str = "chaotic"

    @property
    def lyapunov_time(self) -> float | None:
        """1 / lambda_max in time units (None when not chaotic)."""
        return None if not self.lyapunov else 1.0 / self.lyapunov

    def horizon_in_lyapunov_times(self, steps) -> np.ndarray | None:
        if not self.lyapunov:
            return None
        return np.asarray(steps, float) * self.dt * self.lyapunov


def _observe(state: np.ndarray, cols: list[int], level: float, rng) -> tuple:
    clean = state[:, cols]
    std = level * clean.std(axis=0)
    return clean + std * rng.standard_normal(clean.shape), std


def _ode_path(f, x0, dt: float, sub: int, n: int, transient: int):
    h = dt / sub
    x = np.asarray(x0, float)[None]
    for _ in range(transient * sub):
        x = rk4_step(f, x, h)
    out = np.empty((n, x.shape[1]))
    for i in range(n):
        out[i] = x[0]
        for _ in range(sub):
            x = rk4_step(f, x, h)
    return out


def _ode_lyapunov(f, x0, h: float, n_steps: int, transient: int) -> float:
    return largest_lyapunov_benettin(lambda z: rk4_step(f, z, h), x0, step_time=h, n_steps=n_steps,
                                     transient=transient)


def lorenz63(n: int = 6000, *, dt: float = 0.05, sub: int = 5, noise: float = 0.1, seed: int = 0,
             sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0, observed=(0, 1, 2),
             lyapunov_steps: int = 50000) -> SyntheticSeries:
    """Lorenz-63 sampled every ``dt`` (RK4 with ``sub`` substeps); observation noise ``noise`` x std per column."""
    rng = np.random.default_rng(seed)
    f = lambda x: lorenz63_rhs(x, sigma, rho, beta)  # noqa: E731
    x0 = np.array([1.0, 1.0, 1.0]) + 0.1 * rng.standard_normal(3)
    state = _ode_path(f, x0, dt, sub, n, transient=400)
    obs, std = _observe(state, list(observed), noise, rng)
    lam = _ode_lyapunov(f, state[0], dt / sub, lyapunov_steps, 1000)
    return SyntheticSeries(f"lorenz63_noise{noise:g}", np.arange(n) * dt, state, obs, dt, 0, list(observed),
                           {"sigma": sigma, "rho": rho, "beta": beta}, std, noise, lam, "chaotic")


def lorenz96(n: int = 6000, *, K: int = 40, F: float = 8.0, dt: float = 0.05, sub: int = 5, noise: float = 0.1,
             seed: int = 0, lyapunov_steps: int = 20000) -> SyntheticSeries:
    """Lorenz-96 with ``K`` variables, every variable observed, target variable 0."""
    rng = np.random.default_rng(seed)
    f = lambda x: lorenz96_rhs(x, F)  # noqa: E731
    x0 = F * np.ones(K) + 0.01 * rng.standard_normal(K)
    state = _ode_path(f, x0, dt, sub, n, transient=200)
    obs, std = _observe(state, list(range(K)), noise, rng)
    lam = _ode_lyapunov(f, state[0], dt / sub, lyapunov_steps, 500)
    return SyntheticSeries(f"lorenz96_K{K}_noise{noise:g}", np.arange(n) * dt, state, obs, dt, 0, list(range(K)),
                           {"F": F, "K": K}, std, noise, lam, "chaotic")


def mackey_glass(n: int = 6000, *, dt: float = 3.0, dt_int: float = 0.1, tau: float = 17.0, beta: float = 0.2,
                 gamma: float = 0.1, n_exp: float = 10.0, noise: float = 0.1, seed: int = 0,
                 lyapunov_time: float = 50000.0) -> SyntheticSeries:
    """Mackey-Glass sampled every ``dt``, integrated with RK4 at ``dt_int`` (tau must be a multiple of dt_int)."""
    rng = np.random.default_rng(seed)
    m = int(round(tau / dt_int))
    sub = int(round(dt / dt_int))
    hist = (1.2 + 0.05 * rng.standard_normal(m + 1))[None]
    step = lambda h: mackey_glass_step(h, dt_int, beta, gamma, n_exp)  # noqa: E731
    for _ in range(int(500 / dt_int)):
        hist = step(hist)
    lam = mackey_glass_lyapunov(dt_int=dt_int, total_time=lyapunov_time, tau=tau, beta=beta, gamma=gamma, n=n_exp,
                                seed=seed)
    out = np.empty(n)
    for i in range(n):
        out[i] = hist[0, -1]
        for _ in range(sub):
            hist = step(hist)
    state = out[:, None]
    obs, std = _observe(state, [0], noise, rng)
    return SyntheticSeries(f"mackey_glass_noise{noise:g}", np.arange(n) * dt, state, obs, dt, 0, [0],
                           {"tau": tau, "beta": beta, "gamma": gamma, "n": n_exp, "dt_int": dt_int}, std, noise,
                           lam, "chaotic")


def ornstein_uhlenbeck(n: int = 6000, *, dt: float = 0.1, theta: float = 1.0, sigma: float = 1.0,
                       noise: float = 0.1, seed: int = 0) -> SyntheticSeries:
    """Exact sampling: x_{k+1} = e^{-theta dt} x_k + sqrt(sigma^2 (1 - e^{-2 theta dt}) / (2 theta)) eps."""
    rng = np.random.default_rng(seed)
    phi = np.exp(-theta * dt)
    q = np.sqrt(sigma ** 2 * (1 - phi ** 2) / (2 * theta))
    x = np.empty(n)
    x[0] = rng.standard_normal() * sigma / np.sqrt(2 * theta)
    eps = rng.standard_normal(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + q * eps[i]
    state = x[:, None]
    obs, std = _observe(state, [0], noise, rng)
    return SyntheticSeries(f"ornstein_uhlenbeck_noise{noise:g}", np.arange(n) * dt, state, obs, dt, 0, [0],
                           {"theta": theta, "sigma": sigma}, std, noise, None, "stochastic")


def oscillator_transition(omega: float, zeta: float, sigma: float, dt: float):
    """Exact discretisation of d[x, v] = A [x, v] dt + [0, sigma] dW with A = [[0, 1], [-omega^2, -2 zeta omega]].

    Returns (Phi, Q): the transition matrix exp(A dt) and the process-noise covariance (Van Loan 1978, IEEE TAC
    23:395).
    """
    from scipy.linalg import expm
    A = np.array([[0.0, 1.0], [-omega ** 2, -2 * zeta * omega]])
    G = np.array([[0.0, 0.0], [0.0, sigma ** 2]])
    M = np.zeros((4, 4))
    M[:2, :2], M[:2, 2:], M[2:, 2:] = -A, G, A.T
    E = expm(M * dt)
    Phi = E[2:, 2:].T
    Q = Phi @ E[:2, 2:]
    return Phi, 0.5 * (Q + Q.T)


def stochastic_oscillator(n: int = 6000, *, dt: float = 0.25, omega: float = 1.0, zeta: float = 0.05,
                          sigma: float = 0.3, noise: float = 0.1, seed: int = 0) -> SyntheticSeries:
    """Damped oscillator driven by white noise, position observed with noise (velocity hidden)."""
    rng = np.random.default_rng(seed)
    Phi, Q = oscillator_transition(omega, zeta, sigma, dt)
    L = np.linalg.cholesky(Q + 1e-15 * np.eye(2))
    s = np.zeros((n, 2))
    s[0] = [1.0, 0.0]
    for i in range(1, n):
        s[i] = Phi @ s[i - 1] + L @ rng.standard_normal(2)
    obs, std = _observe(s, [0], noise, rng)
    return SyntheticSeries(f"stochastic_oscillator_noise{noise:g}", np.arange(n) * dt, s, obs, dt, 0, [0],
                           {"omega": omega, "zeta": zeta, "sigma": sigma}, std, noise, None, "stochastic")


def double_well(n: int = 6000, *, dt: float = 0.1, sigma: float = 0.7, sub: int = 100, a: float = 1.0,
                b: float = 1.0, noise: float = 0.1, seed: int = 0) -> SyntheticSeries:
    """Double-well Langevin dynamics (Euler-Maruyama with ``sub`` substeps per sample): two regimes at +-1."""
    rng = np.random.default_rng(seed)
    h = dt / sub
    dw = rng.standard_normal((n, sub)) * np.sqrt(h) * sigma
    x = np.empty(n)
    cur = 1.0
    for i in range(n):
        x[i] = cur
        for j in range(sub):
            cur = cur + (a * cur - b * cur ** 3) * h + dw[i, j]
    state = x[:, None]
    obs, std = _observe(state, [0], noise, rng)
    return SyntheticSeries(f"double_well_noise{noise:g}", np.arange(n) * dt, state, obs, dt, 0, [0],
                           {"a": a, "b": b, "sigma": sigma}, std, noise, None, "stochastic")


def white_noise(n: int = 6000, *, seed: int = 0) -> SyntheticSeries:
    x = np.random.default_rng(seed).standard_normal((n, 1))
    return SyntheticSeries("white_noise", np.arange(n, dtype=float), x, x.copy(), 1.0, 0, [0], {}, 0.0, 0.0, None,
                           "noise")


def random_walk(n: int = 6000, *, seed: int = 0) -> SyntheticSeries:
    x = np.cumsum(np.random.default_rng(seed).standard_normal(n))[:, None]
    return SyntheticSeries("random_walk", np.arange(n, dtype=float), x, x.copy(), 1.0, 0, [0], {}, 0.0, 0.0, None,
                           "noise")


def ode_stepper(rhs: Callable, dt: float, sub: int = 1) -> Callable[[np.ndarray], np.ndarray]:
    """A map advancing a batch of states (m, d) of dx/dt = rhs(x) by ``dt`` (``sub`` RK4 substeps)."""
    h = dt / sub

    def step(X):
        for _ in range(sub):
            X = rk4_step(rhs, X, h)
        return X
    return step


def mackey_glass_stepper(dt_int: float = 0.1, sub: int = 10, **params) -> Callable[[np.ndarray], np.ndarray]:
    """A map advancing Mackey-Glass histories (m, tau / dt_int + 1) by ``sub`` RK4 steps of ``dt_int``."""
    def step(H):
        for _ in range(sub):
            H = mackey_glass_step(H, dt_int, **params)
        return H
    return step


def linear_sde_stepper(Phi: np.ndarray, Q: np.ndarray, seed: int = 0) -> Callable[[np.ndarray], np.ndarray]:
    """A map x -> Phi x + w for a batch of states, with one noise draw w ~ N(0, Q) per call shared by the whole
    batch: members that differ only in their initial state (sensitivity to initial conditions, same forcing)."""
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(np.asarray(Q) + 1e-15 * np.eye(len(Q)))

    def step(X):
        return X @ np.asarray(Phi).T + (L @ rng.standard_normal(len(Q)))[None]
    return step

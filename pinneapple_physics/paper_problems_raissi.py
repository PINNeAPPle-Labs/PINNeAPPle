"""The continuous-time problems of Raissi, Perdikaris & Karniadakis, "Physics Informed Deep Learning (Part I)"
(arXiv:1711.10561v1; J. Comput. Phys. 378, 2019): domains, data layouts and test grids as the paper and its official
code (maziarraissi/PINNs at 932f50a2) define them, and our own references. No number in this module is a result.

- Burgers (§2.1): ``u_t + u u_x - (0.01/pi) u_xx = 0`` on [-1, 1] x [0, 1], ``u(0, x) = -sin(pi x)``,
  ``u(t, +-1) = 0``; reference: the Cole-Hopf solution :func:`~pinneapple_physics.closed_form.burgers_sine_exact`.
- Schrodinger (§2.2): ``i h_t + 0.5 h_xx + |h|^2 h = 0`` on [-5, 5) x [0, pi/2], ``h(0, x) = 2 sech x``, periodic;
  reference: :func:`nls_reference`, a Fourier pseudo-spectral solution (the paper used Chebfun, 256 modes, RK4).
"""
from __future__ import annotations

import numpy as np

from pinneapple_physics.closed_form.burgers import burgers_sine_exact

__all__ = ["BURGERS_NU", "burgers_grid", "burgers_candidates", "nls_grid", "nls_reference"]

BURGERS_NU = 0.01 / np.pi


def burgers_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The official test grid: ``x`` (256 points on [-1, 1]), ``t`` (0, 0.01, ..., 0.99) and ``u`` (len t, len x)."""
    x = np.linspace(-1.0, 1.0, 256)
    t = np.arange(100) * 0.01
    X, T = np.meshgrid(x, t)
    return x, t, burgers_sine_exact(X, T, BURGERS_NU)


def burgers_candidates() -> tuple[np.ndarray, np.ndarray]:
    """The official data candidates, ``(X, u)`` with columns (x, t): the 256 grid points at t = 0, then the 100
    times at x = -1, then at x = +1 (``Burgers.py``: ``xx1``, ``xx2``, ``xx3``)."""
    x, t, u = burgers_grid()
    X = np.vstack([np.stack([x, np.zeros_like(x)], 1), np.stack([-np.ones_like(t), t], 1),
                   np.stack([np.ones_like(t), t], 1)])
    U = np.concatenate([u[0], u[:, 0], u[:, -1]])
    return X, U


def nls_grid() -> tuple[np.ndarray, np.ndarray]:
    """The reference grid: 256 periodic points on [-5, 5) and 201 times on [0, pi/2]."""
    return -5.0 + 10.0 * np.arange(256) / 256, np.linspace(0.0, np.pi / 2, 201)


def nls_reference(n_steps: int = 100_000) -> np.ndarray:
    """``h`` (201, 256), complex, on :func:`nls_grid`: the Fourier pseudo-spectral solution with 256 modes and the
    integrating-factor RK4 scheme (the linear part ``-i k^2 / 2`` exact), ``n_steps`` equal steps over [0, pi/2]."""
    x, t_out = nls_grid()
    k = 2.0 * np.pi * np.fft.fftfreq(x.size, d=x[1] - x[0])
    lin = -0.5j * k ** 2
    dt = (np.pi / 2) / n_steps
    e_half, e_full = np.exp(lin * dt / 2), np.exp(lin * dt)

    def nonlin(v):
        h = np.fft.ifft(v)
        return 1j * np.fft.fft(np.abs(h) ** 2 * h)

    v = np.fft.fft(2.0 / np.cosh(x))
    every = n_steps // (t_out.size - 1)
    if every * (t_out.size - 1) != n_steps:
        raise ValueError("n_steps must be a multiple of 200")
    out = [np.fft.ifft(v)]
    for step in range(1, n_steps + 1):
        a = dt * nonlin(v)
        b = dt * nonlin(e_half * (v + a / 2))
        c = dt * nonlin(e_half * v + b / 2)
        d = dt * nonlin(e_full * v + e_half * c)
        v = e_full * v + (e_full * a + 2 * e_half * (b + c) + d) / 6
        if step % every == 0:
            out.append(np.fft.ifft(v))
    return np.array(out)

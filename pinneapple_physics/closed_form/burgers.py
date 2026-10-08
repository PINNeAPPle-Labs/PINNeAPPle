"""Exact solution of the viscous Burgers equation with u(x, 0) = -sin(pi x), u(+-1, t) = 0 on x in [-1, 1].

Cole-Hopf transform (Basdevant et al., Computers & Fluids 14, 1986): with phi_0(y) = exp(-cos(pi y) / (2 pi nu)),

    u(x, t) = - int sin(pi (x - eta)) phi_0(x - eta) G(eta, t) d eta / int phi_0(x - eta) G(eta, t) d eta,

G the heat kernel with variance 2 nu t. Substituting eta = sqrt(4 nu t) z turns both integrals into
Gauss-Hermite quadratures. The constant exp(1 / (2 pi nu)) is factored out of phi_0 (it cancels) so the exponent
stays non-negative and bounded: no overflow for nu down to about 1e-3 with the default 120 nodes.
This is the reference used by the PINN literature for the nu = 0.01/pi benchmark (Raissi et al., 2019).
"""
from __future__ import annotations

import numpy as np

__all__ = ["burgers_sine_exact"]


def burgers_sine_exact(x, t, nu: float, n_nodes: int = 120) -> np.ndarray:
    """``u(x, t)`` for arrays ``x``, ``t`` of the same shape (broadcast). ``t = 0`` returns the initial condition."""
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    x, t = np.broadcast_arrays(x, t)
    if nu <= 0:
        raise ValueError("nu must be positive")
    z, w = np.polynomial.hermite.hermgauss(n_nodes)
    xf, tf = x.ravel(), t.ravel()
    s = np.sqrt(4.0 * nu * np.maximum(tf, 0.0))
    y = xf[:, None] - s[:, None] * z[None, :]
    # log phi_0 shifted by its maximum 1/(2 pi nu): exponent (1 - cos(pi y)) / (2 pi nu) is in [0, 1/(pi nu)];
    # subtract the per-point maximum before exponentiating so the largest term is exp(0) = 1.
    e = (1.0 - np.cos(np.pi * y)) / (2.0 * np.pi * nu)
    e -= e.max(axis=1, keepdims=True)
    phi = w[None, :] * np.exp(e)
    u = -np.sum(phi * np.sin(np.pi * y), axis=1) / np.sum(phi, axis=1)
    u = np.where(tf <= 0.0, -np.sin(np.pi * xf), u)
    return u.reshape(x.shape)

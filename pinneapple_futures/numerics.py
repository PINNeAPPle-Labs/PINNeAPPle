"""The few numerical pieces the Futures experiments share: a 1-D steady conductor, its fluxes and its order.

Everything here is small on purpose. The Lab's experiments are not a second solver library -- they are checks of
PINNeAPPle's own physics on systems that do not exist yet, so they need one honest finite-difference conductor, a
flux that is reported per face (so that its constancy can be checked rather than assumed) and a refinement
ladder with an observed order.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

__all__ = ["tridiag", "steady_conduction_1d", "refinement", "harmonic_mean"]


def tridiag(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a tridiagonal system (Thomas algorithm); ``lower[0]`` and ``upper[-1]`` are not used."""
    n = diag.size
    a, b, c, d = (np.array(v, dtype=float, copy=True) for v in (lower, diag, upper, rhs))
    for i in range(1, n):
        m = a[i] / b[i - 1]
        b[i] -= m * c[i - 1]
        d[i] -= m * d[i - 1]
    x = np.empty(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


def harmonic_mean(k_left: np.ndarray, k_right: np.ndarray) -> np.ndarray:
    """The conductivity of a face between two cells in series (exact for two half-cells)."""
    s = k_left + k_right
    if np.any(s == 0):
        raise ValueError("the conductivity vanishes on a face: the two cells are insulating")
    return 2.0 * k_left * k_right / s


def steady_conduction_1d(k: Callable[[np.ndarray], np.ndarray], t_left: float, t_right: float, *,
                         length: float = 1.0, n: int = 101, tol: float = 1e-11, itmax: int = 2000,
                         spatial: Callable[[np.ndarray], np.ndarray] | None = None) -> dict:
    """Steady conduction ``d/dx[k(T) dT/dx] = 0`` on ``[0, length]`` with the two temperatures imposed.

    ``k(T)`` may be any positive callable (a constant, a linear law, a switch). Solved by Picard iteration on the
    conservative flux form: the face conductivity is the harmonic mean of the two neighbouring cells, so the
    discrete heat flux is the same on every face at convergence -- which :func:`refinement` and the experiments
    check instead of assuming.

    Returns ``x`` (nodes), ``T`` (temperatures), ``q`` (the flux, positive in +x, from the last face),
    ``faces`` (the flux on every face, which must all be equal), ``iterations`` and ``converged``.
    """
    if n < 3:
        raise ValueError("a 1-D conductor needs at least three nodes")
    if length <= 0:
        raise ValueError("the length must be positive")
    h = length / (n - 1)
    x = np.linspace(0.0, length, n)
    t = t_left + (t_right - t_left) * x / length

    def _k(values: np.ndarray) -> np.ndarray:
        kk = np.asarray(k(values), dtype=float)
        kk = np.full(values.shape, float(kk)) if kk.ndim == 0 else kk
        return kk * sp

    sp = np.ones(n) if spatial is None else np.asarray(spatial(x), dtype=float)
    if not np.all(sp > 0):
        raise ValueError("the spatial factor of the conductivity must be positive")

    converged = False
    for it in range(1, itmax + 1):
        kk = _k(t)
        if not np.all(kk > 0):
            raise ValueError("the conductivity is not positive over the temperature range of this problem")
        kf = harmonic_mean(kk[:-1], kk[1:])
        lower = np.zeros(n)
        diag = np.zeros(n)
        upper = np.zeros(n)
        rhs = np.zeros(n)
        lower[1:] = -kf / h**2                      # coefficient of T_(i-1) in row i
        upper[:-1] = -kf / h**2                     # coefficient of T_(i+1) in row i
        diag[:-1] += kf / h**2
        diag[1:] += kf / h**2
        diag[0] = 1.0
        upper[0] = 0.0
        rhs[0] = t_left
        diag[-1] = 1.0
        lower[-1] = 0.0
        rhs[-1] = t_right
        new = tridiag(lower, diag, upper, rhs)
        err = float(np.max(np.abs(new - t)))
        t = new
        if err < tol * max(1.0, float(np.max(np.abs(t)))):     # a relative criterion: the plateau is ~1e-10 K
            converged = True
            break
    kk = _k(t)
    kf = harmonic_mean(kk[:-1], kk[1:])
    faces = -kf * np.diff(t) / h                    # positive in +x
    return {"x": x, "T": t, "faces": faces, "q": float(faces[-1]), "iterations": it, "converged": converged,
            "h": h}


def refinement(error_at: Callable[[int], float], sizes: list[int]) -> dict:
    """A refinement ladder: the error at each resolution and the observed order between consecutive ones.

    ``error_at(n)`` returns the error of the run at resolution ``n``; the resolutions must be chosen so that
    consecutive ones halve. The order is :func:`pinneapple_science.core.observed_order`.
    """
    from pinneapple_science.core import observed_order

    errors = [float(error_at(n)) for n in sizes]
    return {"sizes": list(sizes), "errors": errors, "orders": observed_order(errors, 2.0)}

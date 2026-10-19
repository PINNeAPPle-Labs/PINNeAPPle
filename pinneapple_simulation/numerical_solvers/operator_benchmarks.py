"""Reference solvers and input distributions of the standard operator-learning benchmarks.

The three problems of Li et al. (2021), "Fourier Neural Operator for Parametric Partial Differential Equations",
ICLR, used throughout the neural-operator literature, written from the equations (no code taken from elsewhere):

* ``burgers_1d``: viscous Burgers ``u_t + (u^2/2)_x = nu u_xx`` on the periodic interval [0, 1), pseudo-spectral
  in space (2/3-rule dealiasing) and ETDRK4 in time (``etdrk4.ETDRK4``), batched over initial conditions.
* ``navier_stokes_2d_vorticity``: 2-D incompressible Navier-Stokes in vorticity form on the periodic unit square,
  ``w_t + u . grad w = nu lap w + f``, ``u = (psi_y, -psi_x)``, ``-lap psi = w``; pseudo-spectral, dealiased,
  ETDRK4, batched. ``li_forcing`` is the forcing of the FNO paper.
* ``darcy_2d``: ``-div(a grad u) = f`` on the unit square with ``u = 0`` on the boundary, cell-centred finite
  volumes (harmonic face coefficients, two-point flux), sparse direct solve.
* ``periodic_grf``: Gaussian random fields with covariance ``sigma^2 (-lap + tau^2 I)^(-alpha)`` on the periodic
  unit box (the input distributions of the three problems; ``threshold_coefficient`` gives the piecewise-constant
  Darcy permeability).
* ``FourierInterpolant``: a differentiable (autograd) evaluation of a periodic grid field at arbitrary points, so a
  PINN can take a gridded initial condition as a hard constraint.

Checks are in ``tests/test_operator_benchmarks.py``: exact decay of a Taylor-Green vortex, mass conservation and
the Cole-Hopf solution of Burgers, a manufactured variable-coefficient Darcy solution with second-order
convergence.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .etdrk4 import ETDRK4

__all__ = ["periodic_grf", "threshold_coefficient", "burgers_1d", "navier_stokes_2d_vorticity", "li_forcing",
           "darcy_2d", "darcy_matrix", "darcy_operator", "spectral_downsample", "FourierInterpolant"]


# --------------------------------------------------------------------------- random inputs
def periodic_grf(n_samples: int, shape: Sequence[int], *, alpha: float, tau: float, sigma: float | None = None,
                 seed: int | None = None) -> np.ndarray:
    """Zero-mean Gaussian random fields on the periodic unit box with covariance ``sigma^2 (-lap + tau^2)^-alpha``.

    Spectral synthesis: each Fourier coefficient is complex standard normal times
    ``sigma (4 pi^2 |k|^2 + tau^2)^(-alpha/2)``; the mean mode is zero. ``sigma`` defaults to
    ``tau^(alpha - d/2)`` (the FNO paper's convention), ``d = len(shape)``. Returns ``(n_samples, *shape)``.
    """
    shape = tuple(int(s) for s in shape)
    d = len(shape)
    if sigma is None:
        sigma = tau ** (0.5 * (2 * alpha - d))
    rng = np.random.default_rng(seed)
    ks = np.meshgrid(*[np.fft.fftfreq(n, 1.0 / n) for n in shape], indexing="ij")
    k2 = sum(k ** 2 for k in ks)
    amp = sigma * (4 * np.pi ** 2 * k2 + tau ** 2) ** (-alpha / 2)
    amp.flat[0] = 0.0
    size = int(np.prod(shape))
    xi = rng.standard_normal((n_samples, *shape)) + 1j * rng.standard_normal((n_samples, *shape))
    axes = tuple(range(1, d + 1))
    return np.real(np.fft.ifftn(size * amp * xi, axes=axes))


def threshold_coefficient(field: np.ndarray, high: float = 12.0, low: float = 3.0) -> np.ndarray:
    """Piecewise-constant coefficient ``high`` where ``field >= 0`` and ``low`` elsewhere (FNO Darcy inputs)."""
    return np.where(field >= 0, high, low).astype(np.float64)


def spectral_downsample(u: np.ndarray, factor: int, axes: Sequence[int]) -> np.ndarray:
    """Point values on every ``factor``-th node of a periodic grid (exact for band-limited fields)."""
    sl = [slice(None)] * u.ndim
    for a in axes:
        sl[a] = slice(None, None, factor)
    return u[tuple(sl)]


# --------------------------------------------------------------------------- Burgers 1-D
def burgers_1d(u0: np.ndarray, nu: float, t_end: float, *, dt: float = 1e-4, n_save: int = 1,
               dealias: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Viscous Burgers on the periodic interval [0, 1).

    ``u0``: ``(n,)`` or ``(batch, n)`` point values on ``x_j = j / n``. Returns ``(t, u)`` with ``u`` of shape
    ``(batch, n_save + 1, n)`` at ``n_save + 1`` equally spaced times from 0 to ``t_end`` (``t_end`` is rounded to
    a whole number of steps per save interval).
    """
    u0 = np.atleast_2d(np.asarray(u0, dtype=np.float64))
    n = u0.shape[-1]
    k = np.fft.fftfreq(n, 1.0 / n)
    ik = 2j * np.pi * k
    mask = (np.abs(k) < n / 3).astype(float) if dealias else np.ones(n)

    def nonlinear(uh, t):
        u = np.fft.ifft(uh, axis=-1).real
        return -0.5 * ik * mask * np.fft.fft(u * u, axis=-1)

    steps_per_save = max(1, int(round(t_end / n_save / dt)))
    integ = ETDRK4(-nu * (2 * np.pi * k) ** 2, nonlinear, dt=t_end / n_save / steps_per_save)
    uh = np.fft.fft(u0, axis=-1)
    out = [u0.copy()]
    for _ in range(n_save):
        for _ in range(steps_per_save):
            uh = integ.step(uh, 0.0)
        out.append(np.fft.ifft(uh, axis=-1).real)
    if not np.all(np.isfinite(out[-1])):
        raise FloatingPointError("burgers_1d blew up: reduce dt")
    return np.linspace(0.0, t_end, n_save + 1), np.stack(out, axis=1)


# --------------------------------------------------------------------------- Navier-Stokes 2-D vorticity
def li_forcing(n: int) -> np.ndarray:
    """``f(x, y) = 0.1 (sin(2 pi (x + y)) + cos(2 pi (x + y)))`` on the ``n x n`` periodic grid (Li et al. 2021)."""
    x = np.arange(n) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    return 0.1 * (np.sin(2 * np.pi * (X + Y)) + np.cos(2 * np.pi * (X + Y)))


def navier_stokes_2d_vorticity(w0: np.ndarray, nu: float, t_end: float, *, forcing: np.ndarray | None = None,
                               dt: float = 1e-3, n_save: int = 1, dealias: bool = True
                               ) -> tuple[np.ndarray, np.ndarray]:
    """2-D incompressible Navier-Stokes, vorticity form, periodic unit square, grid ``x_i = i / n``.

    ``w0``: ``(n, n)`` or ``(batch, n, n)``; ``forcing``: ``(n, n)`` (time independent) or None.
    Returns ``(t, w)`` with ``w`` of shape ``(batch, n_save + 1, n, n)``.
    """
    w0 = np.asarray(w0, dtype=np.float64)
    if w0.ndim == 2:
        w0 = w0[None]
    n = w0.shape[-1]
    k = np.fft.fftfreq(n, 1.0 / n)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    kx, ky = 2 * np.pi * KX, 2 * np.pi * KY
    lap = -(kx ** 2 + ky ** 2)
    inv_lap = np.where(lap == 0, 0.0, 1.0 / np.where(lap == 0, 1.0, lap))
    mask = ((np.abs(KX) < n / 3) & (np.abs(KY) < n / 3)).astype(float) if dealias else np.ones((n, n))
    fh = np.fft.fft2(forcing) if forcing is not None else 0.0

    def nonlinear(wh, t):
        psih = -inv_lap * wh                      # -lap psi = w
        u = np.fft.ifft2(1j * ky * psih).real     # u = psi_y
        v = np.fft.ifft2(-1j * kx * psih).real    # v = -psi_x
        wx = np.fft.ifft2(1j * kx * wh).real
        wy = np.fft.ifft2(1j * ky * wh).real
        return -mask * np.fft.fft2(u * wx + v * wy) + fh

    steps_per_save = max(1, int(round(t_end / n_save / dt)))
    integ = ETDRK4(nu * lap, nonlinear, dt=t_end / n_save / steps_per_save)
    wh = np.fft.fft2(w0)
    out = [w0.copy()]
    for _ in range(n_save):
        for _ in range(steps_per_save):
            wh = integ.step(wh, 0.0)
        out.append(np.fft.ifft2(wh).real)
    if not np.all(np.isfinite(out[-1])):
        raise FloatingPointError("navier_stokes_2d_vorticity blew up: reduce dt")
    return np.linspace(0.0, t_end, n_save + 1), np.stack(out, axis=1)


# --------------------------------------------------------------------------- Darcy 2-D
def darcy_2d(a: np.ndarray, f: float | np.ndarray = 1.0, *, axis_weights: Sequence[float] = (1.0, 1.0)) -> np.ndarray:
    """``-div(a grad u) = f`` on the unit square, ``u = 0`` on the boundary; ``a`` (and ``f``) at the ``n x n``
    cell centres ``((i + 1/2) h, (j + 1/2) h)``, ``h = 1/n``. Returns ``u`` at the cell centres.

    Two-point flux finite volumes with harmonic-mean face coefficients (exact flux continuity across a
    coefficient jump aligned with the faces); the boundary face uses the cell's own ``a`` over a half cell.
    Second order for smooth coefficients. The matrix is :func:`darcy_matrix`; ``axis_weights``: see there.
    """
    a = np.asarray(a, dtype=np.float64)
    n = a.shape[0]
    rhs = np.broadcast_to(np.asarray(f, dtype=np.float64), (n, n)).ravel()
    return spla.spsolve(darcy_matrix(a, axis_weights), rhs).reshape(n, n)


def darcy_matrix(a: np.ndarray, axis_weights: Sequence[float] = (1.0, 1.0)) -> sp.csc_matrix:
    """The ``n^2 x n^2`` finite-volume matrix of :func:`darcy_2d` (``A u = f`` with ``u`` raveled in C order). It is
    symmetric positive definite, and ``A - min(a) * darcy_matrix(ones)`` is positive semi-definite (every face
    coefficient is a harmonic mean, never below ``min a``).

    ``axis_weights`` multiplies the transmissibilities of the x and y faces (an anisotropic conductivity
    ``a diag(w_x, w_y)``; ``(1, 1)`` is the isotropic problem). ``A @ u.ravel() - f`` is the conservative residual
    of a field ``u`` on that grid."""
    a = np.asarray(a, dtype=np.float64)
    n = a.shape[0]
    if a.shape != (n, n):
        raise ValueError("a must be square (n, n)")
    h = 1.0 / n
    idx = np.arange(n * n).reshape(n, n)
    rows, cols, vals = [], [], []
    diag = np.zeros((n, n))
    # interior faces in x (between i and i+1) and y
    for axis in (0, 1):
        a_l = a[:-1, :] if axis == 0 else a[:, :-1]
        a_r = a[1:, :] if axis == 0 else a[:, 1:]
        t = float(axis_weights[axis]) * 2 * a_l * a_r / (a_l + a_r) / h ** 2
        i_l = idx[:-1, :] if axis == 0 else idx[:, :-1]
        i_r = idx[1:, :] if axis == 0 else idx[:, 1:]
        rows += [i_l.ravel(), i_r.ravel()]
        cols += [i_r.ravel(), i_l.ravel()]
        vals += [-t.ravel(), -t.ravel()]
        if axis == 0:
            diag[:-1, :] += t
            diag[1:, :] += t
        else:
            diag[:, :-1] += t
            diag[:, 1:] += t
    # Dirichlet faces: flux a (u - 0) / (h/2)
    tb = 2 * a / h ** 2
    diag[0, :] += axis_weights[0] * tb[0, :]
    diag[-1, :] += axis_weights[0] * tb[-1, :]
    diag[:, 0] += axis_weights[1] * tb[:, 0]
    diag[:, -1] += axis_weights[1] * tb[:, -1]
    rows.append(idx.ravel())
    cols.append(idx.ravel())
    vals.append(diag.ravel())
    return sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n * n, n * n))


#: Name used by the validation arena and the fault-injection suite; same matrix as :func:`darcy_matrix`.
darcy_operator = darcy_matrix


# --------------------------------------------------------------------------- differentiable interpolation
class FourierInterpolant:
    """Evaluate a periodic grid field (period 1 in every axis, nodes ``j / n``) at arbitrary points, with autograd.

    ``values``: ``(n,)`` or ``(n, n)``. Calling with coordinate tensors ``x`` (1-D) or ``x, y`` (2-D), each
    ``(N, 1)``, returns ``(N, 1)``; derivatives with respect to the coordinates follow from autograd, so a PINN can
    use a gridded initial condition as a hard constraint. Exact for band-limited fields (the trigonometric
    interpolant); the Nyquist mode is dropped so the interpolant is real.
    """

    def __init__(self, values: np.ndarray, *, max_modes: int | None = None, dtype=None):
        import torch
        v = np.asarray(values, dtype=np.float64)
        if v.ndim not in (1, 2):
            raise ValueError("values must be 1-D or 2-D")
        self.ndim = v.ndim
        n = v.shape[0]
        c = np.fft.fft(v) / n if v.ndim == 1 else np.fft.fft2(v) / n ** 2
        k = np.fft.fftfreq(n, 1.0 / n)
        keep = np.abs(k) < n / 2
        if max_modes is not None:
            keep &= np.abs(k) <= max_modes
        self.k = torch.tensor(k[keep], dtype=dtype or torch.get_default_dtype())
        c = c[keep] if v.ndim == 1 else c[np.ix_(keep, keep)]
        self.c = torch.tensor(c, dtype=torch.complex128 if (dtype == torch.float64) else torch.complex64)

    def __call__(self, *coords):
        import torch
        if len(coords) != self.ndim:
            raise ValueError(f"expected {self.ndim} coordinate tensors")
        k = self.k.to(coords[0].device)
        c = self.c.to(coords[0].device)
        ex = torch.exp(2j * np.pi * coords[0].reshape(-1, 1).to(c.dtype) * k.to(c.dtype))
        if self.ndim == 1:
            return (ex @ c).real.reshape(-1, 1).to(coords[0].dtype)
        ey = torch.exp(2j * np.pi * coords[1].reshape(-1, 1).to(c.dtype) * k.to(c.dtype))
        return ((ex @ c) * ey).sum(dim=1).real.reshape(-1, 1).to(coords[0].dtype)

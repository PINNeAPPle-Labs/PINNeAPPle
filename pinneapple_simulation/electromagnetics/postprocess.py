"""Post-processing of A_z solutions: torque, flux linkage, back-EMF, iron losses, magnet eddy losses.

Torque
------
* :func:`torque_maxwell_stress` -- Maxwell stress tensor averaged over an air annulus
  ``r1 < r < r2`` (Arkkio's method, M. Arkkio, Acta Polytechnica Scandinavica EE-59, 1987)::

      T = L / (mu0 (r2 - r1)) * int_annulus r B_r B_phi dS

  Averaging the stress over the annulus instead of taking one circle removes most of the mesh noise.
* :func:`torque_coenergy` -- virtual work: ``T = dW'/dtheta`` at constant currents, by central finite
  differences of the co-energy of two solutions at ``theta +- delta`` (the caller supplies a function
  that solves at a given angle). Agreement of the two methods is a check of the whole chain.

Flux linkage and EMF
--------------------
A coil side with ``N`` turns in a region of area ``S`` and orientation ``+-1`` links
``psi = L N / S * (+-1) * int A dS``; a phase sums its coil sides (divide by parallel paths).
``e = dpsi/dt = omega_m dpsi/dtheta`` (sign convention: generator EMF is ``-e``).

Iron loss
---------
Per element, the flux density over one electrical period (stator elements in the stator frame,
rotor elements in the rotor frame) is split into Fourier harmonics; each harmonic ``n`` traces an
ellipse whose major and minor axes are treated as two alternating fields (a common engineering
approximation for rotational loss). Each alternating component of amplitude ``B`` at ``f`` adds
``k_h f^alpha B^beta + k_c f^2 B^2 + k_e f^1.5 B^1.5`` W/kg (Steinmetz hysteresis term plus Bertotti's
classical eddy and excess terms; G. Bertotti, IEEE Trans. Magn. 24(1):621-630, 1988).
Coefficients must come from fits to the material's loss data.

Magnet eddy loss (resistance limited)
-------------------------------------
In 2D each magnet piece is an infinitely long conductor without net current, so
``J = -sigma (dA/dt - <dA/dt>_piece)``; the field of these currents is neglected, valid while the
skin depth is large compared with the magnet. Use
:func:`pinneapple_simulation.electromagnetics.eddy.solve_time_harmonic` when it is not.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .fem import Geometry
from .materials import MU0
from .mesh import Mesh2D

__all__ = ["torque_maxwell_stress", "torque_maxwell_stress_gradient", "torque_coenergy", "region_integral_A", "flux_linkage",
           "back_emf", "IronLossModel", "iron_loss", "classical_eddy_coefficient",
           "magnet_eddy_loss_resistance_limited", "element_flux_density_history", "flux_density_ellipses",
           "iron_loss_components"]

# 3-point edge-midpoint rule on a triangle (exact for quadratics), barycentric coordinates
_MID = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])


def torque_maxwell_stress(mesh: Mesh2D, B: np.ndarray, r1: float, r2: float, length: float = 1.0,
                          elements: np.ndarray | None = None) -> float:
    """Arkkio torque (N m) on everything inside the air annulus ``r1 < r < r2``.

    ``elements`` selects the annulus elements (default: centroid radius in (r1, r2)); they must be air
    and tile the annulus exactly (on a polar mesh, take ``r1``, ``r2`` on ring radii).
    """
    if elements is None:
        c = mesh.centroids()
        rc = np.hypot(c[:, 0], c[:, 1])
        elements = np.where((rc > r1) & (rc < r2))[0]
    p = mesh.points[mesh.tris[elements]]  # (m, 3, 2)
    q = np.einsum("kj,mjd->mkd", _MID, p)  # quadrature points
    x, y = q[..., 0], q[..., 1]
    r = np.hypot(x, y)
    bx, by = B[elements, 0][:, None], B[elements, 1][:, None]
    # r B_r B_phi = (bx x + by y)(by x - bx y) / r
    integrand = (bx * x + by * y) * (by * x - bx * y) / r
    area = np.abs(0.5 * ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                         - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])))
    integral = float(np.sum(area * integrand.mean(axis=1)))
    return length * integral / (MU0 * (r2 - r1))


def torque_maxwell_stress_gradient(mesh: Mesh2D, geometry: Geometry, r1: float, r2: float,
                                   elements: np.ndarray, length: float = 1.0) -> np.ndarray:
    """The Arkkio torque of :func:`torque_maxwell_stress` as a quadratic form ``T = sum_e B_e^T S_e B_e`` of
    the nodal potential: returns an object with ``value(A)`` and ``gradient(A)`` (dT/dA, for adjoints)."""
    p = mesh.points[mesh.tris[elements]]
    q = np.einsum("kj,mjd->mkd", _MID, p)
    x, y = q[..., 0], q[..., 1]
    r = np.hypot(x, y)
    area = geometry.area[elements]
    # r B_r B_phi = bx by (x^2 - y^2)/r + (by^2 - bx^2) x y / r
    s_xx = -(x * y / r).mean(1) * area
    s_yy = (x * y / r).mean(1) * area
    s_xy = 0.5 * ((x * x - y * y) / r).mean(1) * area
    c = length / (MU0 * (r2 - r1))
    t = mesh.tris[elements]
    return _QuadTorque(c, s_xx, s_yy, s_xy, t, geometry.gx[elements], geometry.gy[elements], mesh.n_nodes)


class _QuadTorque:
    """Torque as a quadratic form of the nodal potential; ``value(A)`` and ``gradient(A)``."""

    def __init__(self, c, sxx, syy, sxy, tris, gx, gy, n):
        self.c, self.sxx, self.syy, self.sxy = c, sxx, syy, sxy
        self.tris, self.gx, self.gy, self.n = tris, gx, gy, n

    def _B(self, A):
        a = A[self.tris]
        return (a * self.gy).sum(1), -(a * self.gx).sum(1)

    def value(self, A):
        bx, by = self._B(A)
        return self.c * float(np.sum(self.sxx * bx * bx + self.syy * by * by + 2 * self.sxy * bx * by))

    def gradient(self, A):
        bx, by = self._B(A)
        dbx = 2 * (self.sxx * bx + self.sxy * by)
        dby = 2 * (self.syy * by + self.sxy * bx)
        ge = self.c * (dbx[:, None] * self.gy - dby[:, None] * self.gx)
        return np.bincount(self.tris.ravel(), weights=ge.ravel(), minlength=self.n)


def torque_coenergy(coenergy_at: Callable[[float], float], theta: float, delta: float,
                    length: float = 1.0, stencil: str = "central") -> float:
    """Virtual-work torque ``dW'/dtheta`` (N m) from co-energies per metre ``coenergy_at(angle)``.

    ``stencil``: ``"central"`` (theta +- delta), ``"forward"`` (theta, +delta, +2 delta) or
    ``"backward"``; one-sided stencils are second order too and keep all angles on one side of
    ``theta`` (useful when the moving-band triangulation changes exactly at ``theta``).
    """
    if stencil == "central":
        d = (coenergy_at(theta + delta) - coenergy_at(theta - delta)) / (2 * delta)
    elif stencil == "forward":
        w0, w1, w2 = (coenergy_at(theta + k * delta) for k in range(3))
        d = (-3 * w0 + 4 * w1 - w2) / (2 * delta)
    elif stencil == "backward":
        w0, w1, w2 = (coenergy_at(theta - k * delta) for k in range(3))
        d = (3 * w0 - 4 * w1 + w2) / (2 * delta)
    else:
        raise ValueError(stencil)
    return length * d


def region_integral_A(geometry: Geometry, tris: np.ndarray, A: np.ndarray, mask: np.ndarray) -> float:
    """int A dS over the elements in ``mask`` (P1: area times mean of the vertex values)."""
    return float(np.sum(geometry.area[mask] * A[tris[mask]].mean(axis=1)))


@dataclass
class CoilSide:
    """A coil side: ``region`` name, phase index, turns ``N`` and orientation ``sign`` (+1 / -1)."""

    region: str
    phase: int
    turns: float
    sign: int


def flux_linkage(mesh: Mesh2D, geometry: Geometry, A: np.ndarray, coil_sides: Sequence[CoilSide],
                 n_phases: int, length: float = 1.0, parallel_paths: int = 1) -> np.ndarray:
    """Flux linkage per phase (Wb-turn)."""
    psi = np.zeros(n_phases)
    for cs in coil_sides:
        m = mesh.mask(cs.region)
        S = geometry.area[m].sum()
        if S <= 0:
            continue
        psi[cs.phase] += cs.sign * cs.turns / S * region_integral_A(geometry, mesh.tris, A, m)
    return length * psi / parallel_paths


def back_emf(psi: np.ndarray, theta: np.ndarray, omega_mech: float) -> np.ndarray:
    """``e = omega dpsi/dtheta`` for flux linkages sampled over one full period of ``theta``
    (uniform, endpoint excluded); spectral derivative along axis 0."""
    psi = np.asarray(psi, float)
    n = psi.shape[0]
    span = (theta[1] - theta[0]) * n
    k = np.fft.rfftfreq(n, d=1.0 / n) * 2 * math.pi / span
    shape = (-1,) + (1,) * (psi.ndim - 1)
    dpsi = np.fft.irfft(1j * k.reshape(shape) * np.fft.rfft(psi, axis=0), n=n, axis=0)
    return omega_mech * dpsi


@dataclass(frozen=True)
class IronLossModel:
    """Loss per kg for an alternating component of amplitude B (T) at frequency f (Hz):
    ``k_h f^alpha B^beta + k_c f^2 B^2 + k_e f^1.5 B^1.5``; ``density`` in kg/m^3."""

    k_h: float
    alpha: float = 1.0
    beta: float = 2.0
    k_c: float = 0.0
    k_e: float = 0.0
    density: float = 7650.0

    def specific_loss(self, f, B):
        f, B = np.asarray(f, float), np.asarray(B, float)
        return (self.k_h * f ** self.alpha * B ** self.beta + self.k_c * (f * B) ** 2
                + self.k_e * (f * B) ** 1.5)


def classical_eddy_coefficient(sigma: float, thickness: float, density: float) -> float:
    """``k_c = pi^2 sigma d^2 / (6 density)`` (W/kg per (Hz T)^2) for laminations of thickness ``d`` under
    sinusoidal flux, low-frequency limit."""
    return math.pi ** 2 * sigma * thickness ** 2 / (6.0 * density)


def element_flux_density_history(B_steps: Sequence[np.ndarray], thetas: Sequence[float],
                                 rotor_mask: np.ndarray) -> np.ndarray:
    """Stack B per step to (steps, M, 2), turning rotor elements into the rotor frame."""
    out = np.array(B_steps, float)
    for k, th in enumerate(thetas):
        c, s = math.cos(th), math.sin(th)
        bx, by = out[k, rotor_mask, 0].copy(), out[k, rotor_mask, 1].copy()
        out[k, rotor_mask, 0] = c * bx + s * by
        out[k, rotor_mask, 1] = -s * bx + c * by
    return out


def flux_density_ellipses(B_hist: np.ndarray, mask: np.ndarray, n_harmonics: int | None = None):
    """Semi-axes of the ellipse traced by each harmonic of B per element: ``(b_major, b_minor)``, each of shape
    (harmonics, elements in ``mask``), from ``B_hist`` (steps, M, 2) sampled uniformly over one period."""
    Bh = B_hist[:, mask, :]
    n = Bh.shape[0]
    F = np.fft.rfft(Bh, axis=0) / n
    F[1:] *= 2.0  # one-sided amplitudes (Nyquist bin is rare for odd n; treated alike)
    nh = F.shape[0] - 1 if n_harmonics is None else min(n_harmonics, F.shape[0] - 1)
    X, Y = F[1:nh + 1, :, 0], F[1:nh + 1, :, 1]  # complex amplitudes of B(t) = Re(X e^{jwt}), Re(Y e^{jwt})
    s1 = np.abs(X) ** 2 + np.abs(Y) ** 2
    s2 = np.abs(X ** 2 + Y ** 2)
    return np.sqrt(0.5 * (s1 + s2)), np.sqrt(np.maximum(0.5 * (s1 - s2), 0.0))


def iron_loss(B_hist: np.ndarray, f_fundamental: float, model: IronLossModel, area: np.ndarray,
              mask: np.ndarray, length: float = 1.0, n_harmonics: int | None = None):
    """Iron loss (W) of the elements in ``mask`` from ``B_hist`` (steps, M, 2) sampled uniformly over one
    period of the fundamental. Returns (total W, loss density per element W/m^3)."""
    b_maj, b_min = flux_density_ellipses(B_hist, mask, n_harmonics)
    dens = np.zeros(b_maj.shape[1])
    for h in range(1, b_maj.shape[0] + 1):
        f = h * f_fundamental
        dens += model.density * (model.specific_loss(f, b_maj[h - 1]) + model.specific_loss(f, b_min[h - 1]))
    total = float(length * np.sum(area[mask] * dens))
    return total, dens


def iron_loss_components(B_hist: np.ndarray, area: np.ndarray, mask: np.ndarray, alpha: float, beta: float,
                         density: float, length: float = 1.0, n_harmonics: int | None = None) -> np.ndarray:
    """Frequency-free sums ``(S_h, S_c, S_e)`` such that the loss of :class:`IronLossModel` with exponents
    ``alpha``, ``beta`` at fundamental frequency ``f`` is ``k_h f^alpha S_h + k_c f^2 S_c + k_e f^1.5 S_e`` (W):
    one field solution period gives the iron loss at every speed (the field waveform is fixed by the
    currents in the rotor frame)."""
    b_maj, b_min = flux_density_ellipses(B_hist, mask, n_harmonics)
    h = np.arange(1, b_maj.shape[0] + 1)[:, None]
    w = density * length * area[mask][None, :]
    S_h = np.sum(w * h ** alpha * (b_maj ** beta + b_min ** beta))
    S_c = np.sum(w * h ** 2 * (b_maj ** 2 + b_min ** 2))
    S_e = np.sum(w * h ** 1.5 * (b_maj ** 1.5 + b_min ** 1.5))
    return np.array([S_h, S_c, S_e])


def magnet_eddy_loss_resistance_limited(mesh: Mesh2D, geometry: Geometry, A_hist: np.ndarray,
                                        period: float, sigma: float, pieces: Sequence[np.ndarray],
                                        length: float = 1.0):
    """Time-averaged eddy loss (W) in magnet pieces from nodal ``A_hist`` (steps, N) in the magnet's own
    frame, sampled uniformly over ``period`` seconds. ``pieces``: element masks, one per insulated piece."""
    n = A_hist.shape[0]
    k = np.fft.rfftfreq(n, d=1.0 / n) * 2 * math.pi / period
    dA = np.fft.irfft(1j * k[:, None] * np.fft.rfft(A_hist, axis=0), n=n, axis=0)
    total = 0.0
    tris = mesh.tris
    # P1 quadrature of (dA_e - c)^2 over each triangle: exact via the vertex mass matrix
    Me = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) / 12.0
    for mask in pieces:
        t = tris[mask]
        ar = geometry.area[mask]
        S = ar.sum()
        for step in range(n):
            v = dA[step][t]  # (m, 3)
            mean = np.sum(ar * v.mean(1)) / S
            w = v - mean
            total += sigma * float(np.sum(ar * np.einsum("mi,ij,mj->m", w, Me, w)))
    return length * total / n

"""Mie scattering of a plane wave by a homogeneous sphere (and a perfectly conducting one).

Implementation of the series of C. F. Bohren and D. R. Huffman, *Absorption and Scattering of Light by Small
Particles* (1983), sec. 4.4 and appendix A (BHMIE), with the logarithmic derivative ``D_n(m x)`` by downward
recurrence and the series length of W. J. Wiscombe, *Improved Mie scattering algorithms*, Appl. Opt. 19 (1980)
1505, ``n_max = x + 4 x^{1/3} + 2``.

Convention: Bohren & Huffman use ``exp(-i omega t)``, so absorbing media have ``Im(m) > 0`` there. This package
uses ``exp(+j omega t)`` (``Im(eps) < 0`` for loss); :func:`mie` accepts the relative index in this package's
convention and conjugates it internally. Efficiencies are real and convention-free.

``x = 2 pi n_medium a / lambda`` is the size parameter, ``m = n_sphere / n_medium`` the relative index.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["MieResult", "mie", "mie_pec", "rayleigh_qsca", "mie_coefficients"]


@dataclass
class MieResult:
    x: float
    m: complex
    an: np.ndarray
    bn: np.ndarray
    qext: float
    qsca: float
    qback: float
    g: float  # asymmetry parameter <cos theta>

    @property
    def qabs(self) -> float:
        return self.qext - self.qsca

    def amplitudes(self, theta):
        """Scattering amplitudes ``S1(theta), S2(theta)`` (B&H eq. 4.74, their convention)."""
        mu = np.cos(np.atleast_1d(np.asarray(theta, float)))
        n_max = len(self.an)
        pi_prev = np.zeros_like(mu)
        pi_n = np.ones_like(mu)
        S1 = np.zeros(mu.shape, complex)
        S2 = np.zeros(mu.shape, complex)
        for n in range(1, n_max + 1):
            tau_n = n * mu * pi_n - (n + 1) * pi_prev
            f = (2 * n + 1) / (n * (n + 1))
            S1 += f * (self.an[n - 1] * pi_n + self.bn[n - 1] * tau_n)
            S2 += f * (self.an[n - 1] * tau_n + self.bn[n - 1] * pi_n)
            pi_prev, pi_n = pi_n, ((2 * n + 1) * mu * pi_n - (n + 1) * pi_prev) / n
        return S1, S2


def _n_max(x: float) -> int:
    return int(round(x + 4.0 * x ** (1.0 / 3.0) + 2.0))


def _riccati(x: float, n_max: int):
    """psi_n(x) = x j_n(x) and xi_n(x) = x h_n^(1)(x) for n = 0..n_max by upward recurrence."""
    psi = np.zeros(n_max + 1)
    chi = np.zeros(n_max + 1)
    psi[0], chi[0] = math.sin(x), math.cos(x)
    if n_max >= 1:
        psi[1] = psi[0] / x - chi[0]
        chi[1] = chi[0] / x + psi[0]
    for n in range(1, n_max):
        psi[n + 1] = (2 * n + 1) / x * psi[n] - psi[n - 1]
        chi[n + 1] = (2 * n + 1) / x * chi[n] - chi[n - 1]
    return psi, psi - 1j * chi  # B&H: xi = psi - i chi with chi_n = -x y_n


def mie_coefficients(m_bh: complex, x: float, n_max: int | None = None):
    """``a_n, b_n`` (n = 1..n_max) in Bohren & Huffman's convention (``Im(m_bh) >= 0`` absorbs)."""
    n_max = n_max or _n_max(x)
    mx = m_bh * x
    n_start = max(n_max, int(abs(mx))) + 16
    D = np.zeros(n_start + 1, complex)
    for n in range(n_start, 0, -1):
        D[n - 1] = n / mx - 1.0 / (D[n] + n / mx)
    psi, xi = _riccati(x, n_max)
    n = np.arange(1, n_max + 1)
    Dn = D[1:n_max + 1]
    a = ((Dn / m_bh + n / x) * psi[1:] - psi[:-1]) / ((Dn / m_bh + n / x) * xi[1:] - xi[:-1])
    b = ((m_bh * Dn + n / x) * psi[1:] - psi[:-1]) / ((m_bh * Dn + n / x) * xi[1:] - xi[:-1])
    return a, b


def _efficiencies(x, an, bn):
    n = np.arange(1, len(an) + 1)
    qext = 2.0 / x ** 2 * float(np.sum((2 * n + 1) * (an + bn).real))
    qsca = 2.0 / x ** 2 * float(np.sum((2 * n + 1) * (np.abs(an) ** 2 + np.abs(bn) ** 2)))
    qback = float(np.abs(np.sum((2 * n + 1) * (-1.0) ** n * (an - bn))) ** 2 / x ** 2)
    # asymmetry parameter, B&H eq. 4.82
    g1 = np.sum(n[:-1] * (n[:-1] + 2) / (n[:-1] + 1) * (an[:-1] * an[1:].conj() + bn[:-1] * bn[1:].conj()).real)
    g2 = np.sum((2 * n + 1) / (n * (n + 1)) * (an * bn.conj()).real)
    g = 4.0 / (x ** 2 * qsca) * float(g1 + g2) if qsca > 0 else 0.0
    return qext, qsca, qback, g


def mie(m: complex, x: float, n_max: int | None = None) -> MieResult:
    """Efficiencies of a homogeneous sphere, ``m`` in this package's ``exp(+j omega t)`` convention."""
    m_bh = complex(m).conjugate()
    an, bn = mie_coefficients(m_bh, x, n_max)
    return MieResult(x, complex(m), an, bn, *_efficiencies(x, an, bn))


def mie_pec(x: float, n_max: int | None = None) -> MieResult:
    """Perfectly conducting sphere: ``a_n = psi_n'(x) / xi_n'(x)``, ``b_n = psi_n(x) / xi_n(x)``, the
    ``|m| -> infinity`` limit of the coefficients above; its small-sphere limit is ``Q_sca = (10/3) x^4``
    (Jackson eq. 10.18 divided by ``pi a^2``)."""
    n_max = n_max or _n_max(x)
    psi, xi = _riccati(x, n_max)
    n = np.arange(1, n_max + 1)
    dpsi = psi[:-1] - n / x * psi[1:]
    dxi = xi[:-1] - n / x * xi[1:]
    an = dpsi / dxi
    bn = psi[1:] / xi[1:]
    return MieResult(x, complex(np.inf), an, bn, *_efficiencies(x, an, bn))


def rayleigh_qsca(m: complex, x: float) -> float:
    """Small-sphere limit ``Q_sca = (8/3) x^4 |(m^2 - 1) / (m^2 + 2)|^2`` (B&H eq. 5.8)."""
    m = complex(m)
    return 8.0 / 3.0 * x ** 4 * abs((m * m - 1) / (m * m + 2)) ** 2

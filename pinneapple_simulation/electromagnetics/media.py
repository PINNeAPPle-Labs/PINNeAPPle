"""Electric and magnetic media for wave problems: linear, anisotropic and dispersive (Drude, Lorentz, Debye).

Time convention. Phasors are ``Re[X exp(+j omega t)]`` throughout this package (the engineering convention of
Taflove & Hagness, Pozar and scikit-rf, and of :mod:`.eddy`), so a lossy medium has ``Im(eps_r) < 0`` and a
wave travelling towards ``+z`` is ``exp(-j k z)``. Physics texts (Jackson, Bohren & Huffman) use
``exp(-i omega t)``: their ``eps`` is the complex conjugate of the one here.

Dispersion models, relative permittivity ``eps_r(omega) = eps_inf + sum_k chi_k(omega) + sigma / (j omega eps0)``
(Taflove & Hagness, *Computational Electrodynamics*, 3rd ed., ch. 9):

* :class:`Drude` ``chi = -omega_p^2 / (omega^2 - j omega gamma)`` -- free electrons (metals, plasmas);
* :class:`Lorentz` ``chi = delta_eps omega_0^2 / (omega_0^2 + 2 j omega delta - omega^2)`` -- bound resonances;
* :class:`Debye` ``chi = delta_eps / (1 + j omega tau)`` -- orientational relaxation (water, tissue).

Each pole carries the coefficients of its auxiliary differential equation (ADE) for the FDTD solver, a
second-order accurate discretisation of the polarisation current ``J_p = dP/dt``.

:class:`Anisotropic` holds 3x3 tensors ``eps_r``, ``mu_r`` and ``sigma`` and returns the refractive indices
of a plane wave along a direction (Fresnel's equation of wave normals as a generalised eigenproblem).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .constants import C0, EPS0

__all__ = ["Drude", "Lorentz", "Debye", "Medium", "Anisotropic", "rotation", "VACUUM", "drude_metal",
           "is_passive"]


@dataclass(frozen=True)
class Drude:
    """``chi(omega) = -omega_p^2 / (omega^2 - j omega gamma)``; plasma frequency ``omega_p`` and collision rate
    ``gamma`` in rad/s. With ``gamma = 0`` it is a collisionless plasma."""

    omega_p: float
    gamma: float = 0.0

    def chi(self, omega):
        w = np.asarray(omega, complex)
        return -self.omega_p ** 2 / (w * w - 1j * w * self.gamma)

    def ade(self, dt: float):
        """Coefficients of ``dJ/dt + gamma J = eps0 omega_p^2 E`` (Taflove & Hagness eq. 9.50, centred in time):
        ``J^{n+1} = ka J^n + kb (E^{n+1} + E^n)``."""
        g = 0.5 * self.gamma * dt
        ka = (1.0 - g) / (1.0 + g)
        kb = 0.5 * EPS0 * self.omega_p ** 2 * dt / (1.0 + g)
        return ka, kb


@dataclass(frozen=True)
class Lorentz:
    """``chi(omega) = delta_eps omega_0^2 / (omega_0^2 + 2 j omega delta - omega^2)``; resonance ``omega_0``
    (rad/s), damping ``delta`` (1/s), strength ``delta_eps = eps_s - eps_inf``."""

    delta_eps: float
    omega0: float
    delta: float = 0.0

    def chi(self, omega):
        w = np.asarray(omega, complex)
        return self.delta_eps * self.omega0 ** 2 / (self.omega0 ** 2 + 2j * w * self.delta - w * w)

    def ade(self, dt: float):
        """Coefficients of ``d2P/dt2 + 2 delta dP/dt + omega0^2 P = eps0 delta_eps omega0^2 E``, centred:
        ``P^{n+1} = a1 P^n + a2 P^{n-1} + a3 E^n``."""
        d = 1.0 + self.delta * dt
        a1 = (2.0 - (self.omega0 * dt) ** 2) / d
        a2 = -(1.0 - self.delta * dt) / d
        a3 = EPS0 * self.delta_eps * (self.omega0 * dt) ** 2 / d
        return a1, a2, a3


@dataclass(frozen=True)
class Debye:
    """``chi(omega) = delta_eps / (1 + j omega tau)``; relaxation time ``tau`` (s)."""

    delta_eps: float
    tau: float

    def chi(self, omega):
        return self.delta_eps / (1.0 + 1j * np.asarray(omega, complex) * self.tau)

    def ade(self, dt: float):
        """Coefficients of ``tau dP/dt + P = eps0 delta_eps E``, centred (trapezoidal):
        ``P^{n+1} = ka P^n + kb (E^{n+1} + E^n)``."""
        r = dt / (2.0 * self.tau)
        ka = (1.0 - r) / (1.0 + r)
        kb = EPS0 * self.delta_eps * r / (1.0 + r)
        return ka, kb


Pole = Drude | Lorentz | Debye


@dataclass(frozen=True)
class Medium:
    """Isotropic linear medium: ``eps_inf``, ``mu_r``, conductivity ``sigma`` (S/m), magnetic conductivity
    ``sigma_m`` (ohm/m, used by PMLs) and dispersive poles."""

    eps_inf: float = 1.0
    mu_r: float = 1.0
    sigma: float = 0.0
    sigma_m: float = 0.0
    poles: tuple[Pole, ...] = field(default_factory=tuple)

    def eps_r(self, omega):
        """Complex relative permittivity at angular frequency ``omega`` (``exp(+j omega t)``)."""
        w = np.asarray(omega, complex)
        out = self.eps_inf + 0j * w
        for p in self.poles:
            out = out + p.chi(w)
        if self.sigma:
            out = out + self.sigma / (1j * w * EPS0)
        return out

    def n(self, omega):
        """Complex refractive index ``sqrt(eps_r mu_r)`` with ``Im(n) <= 0`` (decaying ``exp(-j k z)``)."""
        n = np.sqrt(self.eps_r(omega) * self.mu_r + 0j)
        return np.where(n.imag > 0, -n, n) if np.ndim(n) else (-n if n.imag > 0 else n)

    def k(self, omega):
        return np.asarray(omega) / C0 * self.n(omega)

    def impedance(self, omega):
        """Wave impedance relative to free space, ``sqrt(mu_r / eps_r)`` (``Re >= 0``)."""
        z = np.sqrt(self.mu_r / self.eps_r(omega) + 0j)
        return np.where(z.real < 0, -z, z)

    @property
    def is_dispersive(self) -> bool:
        return bool(self.poles)


VACUUM = Medium()


def drude_metal(omega_p: float, gamma: float = 0.0, eps_inf: float = 1.0) -> Medium:
    return Medium(eps_inf=eps_inf, poles=(Drude(omega_p, gamma),))


def is_passive(medium: Medium, omegas: Sequence[float], tol: float = 1e-12) -> bool:
    """Passivity of a medium under ``exp(+j omega t)``: ``Im(eps_r) <= 0`` at every given frequency."""
    e = medium.eps_r(np.asarray(omegas, float))
    return bool(np.all(e.imag <= tol * np.maximum(1.0, np.abs(e))))


def rotation(axis: Sequence[float], angle: float) -> np.ndarray:
    """Rotation matrix about ``axis`` by ``angle`` (rad), Rodrigues' formula."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


@dataclass
class Anisotropic:
    """Linear anisotropic medium with relative tensors ``eps_r`` and ``mu_r`` (3x3) and conductivity ``sigma``."""

    eps_r: np.ndarray
    mu_r: np.ndarray = field(default_factory=lambda: np.eye(3))
    sigma: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))

    def __post_init__(self):
        self.eps_r = np.asarray(self.eps_r, complex) * np.ones((3, 3)) if np.ndim(self.eps_r) == 0 \
            else np.asarray(self.eps_r, complex)
        self.mu_r = np.asarray(self.mu_r, complex)
        self.sigma = np.asarray(self.sigma, float)

    @classmethod
    def uniaxial(cls, n_o: float, n_e: float, axis: Sequence[float] = (0, 0, 1)) -> Anisotropic:
        """Uniaxial crystal with ordinary and extraordinary indices and optic axis ``axis``."""
        a = np.asarray(axis, float)
        a = a / np.linalg.norm(a)
        eps = n_o ** 2 * np.eye(3) + (n_e ** 2 - n_o ** 2) * np.outer(a, a)
        return cls(eps)

    def rotated(self, R: np.ndarray) -> Anisotropic:
        return Anisotropic(R @ self.eps_r @ R.T, R @ self.mu_r @ R.T, R @ self.sigma @ R.T)

    def eps_eff(self, omega: float | None = None) -> np.ndarray:
        if omega is None or not np.any(self.sigma):
            return self.eps_r
        return self.eps_r + self.sigma / (1j * omega * EPS0)

    @property
    def is_reciprocal(self) -> bool:
        """Lorentz reciprocity holds iff eps, mu and sigma are symmetric tensors."""
        return all(np.allclose(T, T.T) for T in (self.eps_r, self.mu_r, self.sigma))

    def indices(self, direction: Sequence[float], omega: float | None = None) -> np.ndarray:
        """Refractive indices of the two plane waves along unit ``direction``.

        From ``k x (mu^-1 (k x E)) + k0^2 eps E = 0`` with ``k = k0 n s``: ``n^2 [s]x mu^-1 [s]x E = -eps E``,
        a generalised eigenproblem whose two finite eigenvalues are ``1/n^2``. Sorted ascending by real part."""
        s = np.asarray(direction, float)
        s = s / np.linalg.norm(s)
        Sx = np.array([[0, -s[2], s[1]], [s[2], 0, -s[0]], [-s[1], s[0], 0]])
        A = -Sx @ np.linalg.inv(self.mu_r) @ Sx  # positive semi-definite, rank 2
        eps = self.eps_eff(omega)
        # A E = (1/n^2) eps E restricted to the 2-D range of A: eigenvalues of eps^-1 A
        lam = np.linalg.eigvals(np.linalg.solve(eps, A))
        lam = lam[np.argsort(-np.abs(lam))][:2]
        n = np.sqrt(1.0 / lam + 0j)
        n = np.where(n.real < 0, -n, n)
        return np.sort_complex(n)

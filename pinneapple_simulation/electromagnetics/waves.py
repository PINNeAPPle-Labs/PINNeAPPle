"""Closed-form wave electromagnetics, ``exp(+j omega t)`` convention (see :mod:`.media`).

* plane waves at interfaces: :func:`fresnel`, :func:`brewster_angle`, :func:`critical_angle`, and the
  characteristic-matrix method for planar multilayers :func:`multilayer` (Born & Wolf, *Principles of Optics*,
  7th ed., sec. 1.5-1.6);
* conductors: :func:`skin_depth`, :func:`surface_impedance`;
* guided waves and resonators: rectangular and circular waveguide cut-offs and fields, rectangular and
  cylindrical cavity resonances and the conductor-loss Q of the rectangular TE101 mode (D. M. Pozar,
  *Microwave Engineering*, 4th ed., ch. 3 and 6; Jackson ch. 8);
* radiation: the Hertzian dipole's complete fields and power, the thin centre-fed dipole pattern of any length,
  the half-wave dipole's radiation resistance and directivity (C. A. Balanis, *Antenna Theory*, 4th ed., ch. 4);
* the exact TM (E along the axis) scattering of a plane wave by a dielectric or perfectly conducting circular
  cylinder as a series of cylinder functions (R. F. Harrington, *Time-Harmonic Electromagnetic Fields*, 1961,
  sec. 5-8 and 5-9; C. A. Balanis, *Advanced Engineering Electromagnetics*, 2nd ed., sec. 11.5);
* surface plasmon polaritons on a flat metal-dielectric interface (S. A. Maier, *Plasmonics*, 2007, ch. 2)
  and the bulk plasma dispersion.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .constants import C0, EPS0, ETA0, MU0

__all__ = [
    "fresnel", "brewster_angle", "critical_angle", "multilayer", "MultilayerResult", "skin_depth",
    "surface_impedance", "rect_waveguide_cutoff", "rect_waveguide_modes", "rect_te_mn_fields",
    "circular_waveguide_cutoff", "circular_waveguide_modes", "propagation_constant", "rect_cavity_resonance",
    "rect_cavity_modes", "cylindrical_cavity_resonance", "rect_cavity_te101_q", "hertzian_dipole_fields",
    "hertzian_dipole_power", "hertzian_dipole_radiation_resistance", "dipole_pattern",
    "half_wave_dipole_radiation_resistance", "half_wave_dipole_directivity", "spp_wavenumber",
    "plasma_dispersion_omega", "free_space_wavelength", "cylinder_scattering_tm", "cylinder_tm_coefficients",
]


def free_space_wavelength(f: float) -> float:
    return C0 / f


# ------------------------------------------------------------------------------------------ interfaces
def fresnel(n1, n2, theta_i):
    """Amplitude reflection and transmission coefficients ``(rs, rp, ts, tp)`` at a planar interface from
    medium ``n1`` to ``n2`` (non-magnetic, complex indices allowed, ``Im(n) <= 0`` for loss).

    s (TE): ``rs = (n1 cos ti - n2 cos tt) / (n1 cos ti + n2 cos tt)``, ``ts = 1 + rs``.
    p (TM): ratio of reflected to incident *electric* field with Born & Wolf's sign convention,
    ``rp = (n2 cos ti - n1 cos tt) / (n2 cos ti + n1 cos tt)``, ``tp = 2 n1 cos ti / (n2 cos ti + n1 cos tt)``.
    At normal incidence ``rp = -rs``. Power coefficients: ``R = |r|^2``, ``T = Re(n2 cos tt) / Re(n1 cos ti) |t|^2``
    for s (and with the same factor for p)."""
    n1 = np.asarray(n1, complex)
    n2 = np.asarray(n2, complex)
    ci = np.cos(np.asarray(theta_i, float)) + 0j
    si = np.sin(np.asarray(theta_i, float))
    st = n1 * si / n2
    ct = np.sqrt(1.0 - st * st)
    # branch: transmitted wave decays away from the interface (Im(n2 ct) <= 0 under exp(+j omega t))
    ct = np.where((n2 * ct).imag > 0, -ct, ct)
    rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)
    rp = (n2 * ci - n1 * ct) / (n2 * ci + n1 * ct)
    ts = 2 * n1 * ci / (n1 * ci + n2 * ct)
    tp = 2 * n1 * ci / (n2 * ci + n1 * ct)
    return rs, rp, ts, tp


def brewster_angle(n1: float, n2: float) -> float:
    """``atan(n2 / n1)``: p-polarised reflection vanishes."""
    return math.atan2(n2, n1)


def critical_angle(n1: float, n2: float) -> float:
    """``asin(n2 / n1)`` for ``n1 > n2`` (total internal reflection above it)."""
    if n2 >= n1:
        raise ValueError("no critical angle for n2 >= n1")
    return math.asin(n2 / n1)


@dataclass
class MultilayerResult:
    r: complex | np.ndarray
    t: complex | np.ndarray
    R: float | np.ndarray
    T: float | np.ndarray


def multilayer(n_layers, d_layers, wavelength, theta_i: float = 0.0, pol: str = "s",
               n_in: complex = 1.0, n_out: complex = 1.0) -> MultilayerResult:
    """Reflection and transmission of a stack of planar layers (indices ``n_layers``, thicknesses ``d_layers``
    in m) between half-spaces ``n_in`` and ``n_out``, by Born & Wolf's characteristic matrices. ``wavelength``
    may be an array (vacuum wavelength, m). Field coefficients refer to E for s and to H for p
    (so ``r`` for p equals Born & Wolf's ``r_TM``, ``R`` and ``T`` are unambiguous)."""
    lam = np.atleast_1d(np.asarray(wavelength, float))
    k0 = 2 * np.pi / lam
    s = np.asarray(n_in, complex) * math.sin(theta_i)

    def admittance(n):
        c = np.sqrt(1 - (s / n) ** 2 + 0j)
        c = np.where((n * c).imag > 0, -c, c)
        y = n * c if pol == "s" else n / c
        return y, c

    y0, _ = admittance(np.asarray(n_in, complex))
    ys, _ = admittance(np.asarray(n_out, complex))
    M = np.zeros((len(lam), 2, 2), complex)
    M[:, 0, 0] = M[:, 1, 1] = 1.0
    for n, d in zip(n_layers, d_layers, strict=True):
        n = np.asarray(n, complex)
        y, c = admittance(n)
        delta = k0 * n * c * d
        L = np.empty_like(M)
        L[:, 0, 0] = np.cos(delta)
        L[:, 0, 1] = 1j * np.sin(delta) / y
        L[:, 1, 0] = 1j * y * np.sin(delta)
        L[:, 1, 1] = np.cos(delta)
        M = M @ L
    B = M[:, 0, 0] + M[:, 0, 1] * ys
    Cc = M[:, 1, 0] + M[:, 1, 1] * ys
    r = (y0 * B - Cc) / (y0 * B + Cc)
    t = 2 * y0 / (y0 * B + Cc)
    R = np.abs(r) ** 2
    T = (ys.real / y0.real) * np.abs(t) ** 2
    sq = (lambda a: a[0]) if np.ndim(wavelength) == 0 else (lambda a: a)
    return MultilayerResult(sq(r), sq(t), sq(R), sq(T))


# ------------------------------------------------------------------------------------------ conductors
def skin_depth(omega, sigma: float, mu_r: float = 1.0, eps_r: float = 1.0, exact: bool = False):
    """Skin depth (m). Good-conductor form ``sqrt(2 / (omega mu sigma))``; ``exact=True`` gives ``1 / |Im k|`` with
    ``k^2 = omega^2 mu eps - j omega mu sigma``, valid for any loss tangent."""
    omega = np.asarray(omega, float)
    mu = MU0 * mu_r
    if not exact:
        return np.sqrt(2.0 / (omega * mu * sigma))
    k = np.sqrt(omega ** 2 * mu * eps_r * EPS0 - 1j * omega * mu * sigma)
    return 1.0 / np.abs(k.imag)


def surface_impedance(omega, sigma: float, mu_r: float = 1.0):
    """Surface impedance of a good conductor, ``Zs = (1 + j) / (sigma delta) = (1 + j) sqrt(omega mu / (2 sigma))``."""
    return (1 + 1j) * np.sqrt(np.asarray(omega, float) * MU0 * mu_r / (2.0 * sigma))


# ------------------------------------------------------------------------------------------ waveguides
def rect_waveguide_cutoff(m: int, n: int, a: float, b: float, eps_r: float = 1.0, mu_r: float = 1.0) -> float:
    """Cut-off frequency (Hz) of the TE_mn / TM_mn mode of an ``a x b`` rectangular guide:
    ``f_c = c / (2 pi sqrt(eps_r mu_r)) sqrt((m pi / a)^2 + (n pi / b)^2)``."""
    kc = math.hypot(m * math.pi / a, n * math.pi / b)
    return C0 * kc / (2.0 * math.pi * math.sqrt(eps_r * mu_r))


def rect_waveguide_modes(a: float, b: float, n_modes: int = 10, eps_r: float = 1.0, max_index: int = 8):
    """The lowest ``n_modes`` modes as ``(f_c, "TE"/"TM", m, n)`` sorted by cut-off. TE needs ``m + n > 0``,
    TM needs ``m, n >= 1``."""
    out = []
    for m in range(max_index + 1):
        for n in range(max_index + 1):
            if m + n == 0:
                continue
            fc = rect_waveguide_cutoff(m, n, a, b, eps_r)
            out.append((fc, "TE", m, n))
            if m >= 1 and n >= 1:
                out.append((fc, "TM", m, n))
    out.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
    return out[:n_modes]


def rect_te_mn_fields(x, y, m: int, n: int, a: float, b: float):
    """Transverse profile of TE_mn in an ``a x b`` guide (unnormalised): ``Hz = cos(m pi x/a) cos(n pi y/b)``,
    ``E_t`` proportional to ``(n/b cos(m pi x/a) sin(n pi y/b), -m/a sin(m pi x/a) cos(n pi y/b))``."""
    kx, ky = m * np.pi / a, n * np.pi / b
    Hz = np.cos(kx * x) * np.cos(ky * y)
    Ex = ky * np.cos(kx * x) * np.sin(ky * y)
    Ey = -kx * np.sin(kx * x) * np.cos(ky * y)
    return Ex, Ey, Hz


def circular_waveguide_cutoff(kind: str, m: int, n: int, radius: float, eps_r: float = 1.0) -> float:
    """Cut-off (Hz) of TE_mn (``n``-th zero of ``J_m'``) or TM_mn (``n``-th zero of ``J_m``) of a circular guide."""
    from scipy.special import jn_zeros, jnp_zeros

    if kind.upper() == "TE":
        x = jnp_zeros(m, n)[-1] if m > 0 else jnp_zeros(0, n)[-1]
    else:
        x = jn_zeros(m, n)[-1]
    return C0 * x / (2.0 * math.pi * radius * math.sqrt(eps_r))


def circular_waveguide_modes(radius: float, n_modes: int = 10, eps_r: float = 1.0, max_m: int = 6,
                             max_n: int = 4):
    out = []
    for kind in ("TE", "TM"):
        for m in range(max_m + 1):
            for n in range(1, max_n + 1):
                out.append((circular_waveguide_cutoff(kind, m, n, radius, eps_r), kind, m, n))
    out.sort()
    return out[:n_modes]


def propagation_constant(f, f_cutoff: float, eps_r: float = 1.0, mu_r: float = 1.0):
    """``beta = k sqrt(1 - (fc/f)^2)`` above cut-off; below it ``-j alpha`` (evanescent, ``exp(-j beta z)``)."""
    f = np.asarray(f, float)
    k = 2 * np.pi * f * math.sqrt(eps_r * mu_r) / C0
    return k * np.sqrt(1.0 - (f_cutoff / f) ** 2 + 0j) * np.where(f >= f_cutoff, 1.0, -1.0)


# ------------------------------------------------------------------------------------------ cavities
def rect_cavity_resonance(m: int, n: int, p: int, a: float, b: float, d: float, eps_r: float = 1.0,
                          mu_r: float = 1.0) -> float:
    """Resonance (Hz) of mode (m, n, p) of an ``a x b x d`` PEC box:
    ``f = c / (2 sqrt(eps_r mu_r)) sqrt((m/a)^2 + (n/b)^2 + (p/d)^2)``."""
    return C0 / (2.0 * math.sqrt(eps_r * mu_r)) * math.sqrt((m / a) ** 2 + (n / b) ** 2 + (p / d) ** 2)


def rect_cavity_modes(a: float, b: float, d: float, n_modes: int = 10, eps_r: float = 1.0, max_index: int = 6):
    """Lowest resonances of a PEC box with multiplicity: ``(f, [labels])``. A mode exists as TE_mnp (relative to z)
    if ``(m or n) and p >= 1`` and as TM_mnp if ``m, n >= 1``; at least two indices must be non-zero."""
    found: dict[float, list[str]] = {}
    for m in range(max_index + 1):
        for n in range(max_index + 1):
            for p in range(max_index + 1):
                labels = []
                if (m > 0 or n > 0) and p > 0:
                    labels.append(f"TE{m}{n}{p}")
                if m > 0 and n > 0:
                    labels.append(f"TM{m}{n}{p}")
                if not labels:
                    continue
                f = round(rect_cavity_resonance(m, n, p, a, b, d, eps_r), 3)
                found.setdefault(f, []).extend(labels)
    return sorted(found.items())[:n_modes]


def cylindrical_cavity_resonance(kind: str, m: int, n: int, p: int, radius: float, length: float,
                                 eps_r: float = 1.0) -> float:
    """TE_mnp (``p >= 1``) or TM_mnp (``p >= 0``) resonance of a closed circular cylinder (Pozar sec. 6.4)."""
    from scipy.special import jn_zeros, jnp_zeros

    x = (jnp_zeros(m, n) if kind.upper() == "TE" else jn_zeros(m, n))[-1]
    return C0 / (2 * math.pi * math.sqrt(eps_r)) * math.sqrt((x / radius) ** 2 + (p * math.pi / length) ** 2)


def rect_cavity_te101_q(a: float, b: float, d: float, sigma: float, eps_r: float = 1.0) -> float:
    """Conductor-loss Q of the TE101 mode (Pozar eq. 6.46):
    ``Qc = (k a d)^3 b eta / (2 pi^2 Rs) / (2 a^3 b + 2 b d^3 + a^3 d + a d^3)``."""
    f = rect_cavity_resonance(1, 0, 1, a, b, d, eps_r)
    k = 2 * math.pi * f * math.sqrt(eps_r) / C0
    Rs = math.sqrt(2 * math.pi * f * MU0 / (2.0 * sigma))
    eta = ETA0 / math.sqrt(eps_r)
    return (k * a * d) ** 3 * b * eta / (2 * math.pi ** 2 * Rs) / (2 * a ** 3 * b + 2 * b * d ** 3 + a ** 3 * d + a * d ** 3)


# ------------------------------------------------------------------------------------------ radiation
def hertzian_dipole_fields(r, theta, current_moment: complex, omega: float):
    """Complete fields ``(E_r, E_theta, H_phi)`` of a z-directed Hertzian dipole with moment ``I l`` (A m) at
    the origin (Balanis eqs. 4-8a..c, ``exp(+j omega t)``)::

        H_phi   = j k I l sin(t) / (4 pi r) [1 + 1/(j k r)] e^{-jkr}
        E_r     = eta I l cos(t) / (2 pi r^2) [1 + 1/(j k r)] e^{-jkr}
        E_theta = j eta k I l sin(t) / (4 pi r) [1 + 1/(j k r) - 1/(k r)^2] e^{-jkr}
    """
    r = np.asarray(r, float)
    th = np.asarray(theta, float)
    k = omega / C0
    e = np.exp(-1j * k * r)
    jkr = 1j * k * r
    Il = current_moment
    Hphi = 1j * k * Il * np.sin(th) / (4 * np.pi * r) * (1 + 1 / jkr) * e
    Er = ETA0 * Il * np.cos(th) / (2 * np.pi * r ** 2) * (1 + 1 / jkr) * e
    Eth = 1j * ETA0 * k * Il * np.sin(th) / (4 * np.pi * r) * (1 + 1 / jkr - 1 / (k * r) ** 2) * e
    return Er, Eth, Hphi


def hertzian_dipole_power(current_moment: complex, omega: float) -> float:
    """Time-averaged radiated power ``eta k^2 |I l|^2 / (12 pi)`` (W, peak phasor)."""
    k = omega / C0
    return ETA0 * k * k * abs(current_moment) ** 2 / (12 * math.pi)


def hertzian_dipole_radiation_resistance(length: float, wavelength: float) -> float:
    """``R = (2 pi / 3) eta (l / lambda)^2`` = ``80 pi^2 (l/lambda)^2`` ohm with ``eta = 120 pi``."""
    return 2 * math.pi / 3 * ETA0 * (length / wavelength) ** 2


def dipole_pattern(theta, length_wavelengths: float = 0.5):
    """Far-field E_theta pattern of a thin centre-fed dipole of length ``L`` with sinusoidal current (Balanis
    eq. 4-58a): ``[cos(k L/2 cos t) - cos(k L/2)] / sin t`` (unnormalised). ``L = lambda/2`` gives
    ``cos(pi/2 cos t) / sin t``."""
    th = np.asarray(theta, float)
    kl2 = math.pi * length_wavelengths
    s = np.sin(th)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(s) > 1e-12, (np.cos(kl2 * np.cos(th)) - math.cos(kl2)) / np.where(s == 0, 1, s), 0.0)


def cylinder_tm_coefficients(ka: float, eps_r: complex | None, n_max: int | None = None):
    """Coefficients ``(n, a_n, b_n)`` of the TM_z scattering by a cylinder of radius ``a`` (``ka = k0 a``):
    scattered ``Ez = E0 sum j^-n a_n H_n^(2)(k rho) e^{j n phi}``, inside ``E0 sum j^-n b_n J_n(k1 rho) e^{j n phi}``
    for the incident ``E0 exp(-j k x)``. ``eps_r=None`` is a perfect conductor (``a_n = -J_n(ka) / H_n(ka)``).
    Continuity of ``Ez`` and ``H_phi`` (non-magnetic) at ``rho = a``."""
    from scipy.special import h2vp, hankel2, jv, jvp

    n_max = n_max or int(np.ceil(ka + 4.05 * ka ** (1 / 3) + 10))
    n = np.arange(-n_max, n_max + 1)
    J, dJ = jv(n, ka), jvp(n, ka)
    H, dH = hankel2(n, ka), h2vp(n, ka)
    if eps_r is None:
        return n, -J / H, np.zeros_like(H)
    m = np.sqrt(complex(eps_r))
    m = -m if m.imag > 0 else m
    J1, dJ1 = jv(n, m * ka), jvp(n, m * ka)
    a = (m * dJ1 * J - dJ * J1) / (dH * J1 - m * dJ1 * H)
    b = (J + a * H) / J1
    return n, a, b


def cylinder_scattering_tm(x, y, k0: float, radius: float, eps_r: complex | None, E0: complex = 1.0,
                           total: bool = True):
    """Phasor ``Ez`` at points ``(x, y)`` for a plane wave ``E0 exp(-j k0 x)`` (``exp(+j omega t)``) on a circular
    cylinder centred at the origin (``eps_r=None``: PEC). ``total=False`` returns the scattered field outside
    (and the total field inside)."""
    from scipy.special import hankel2, jv

    x, y = np.asarray(x, float), np.asarray(y, float)
    rho, phi = np.hypot(x, y), np.arctan2(y, x)
    ka = k0 * radius
    n, a, b = cylinder_tm_coefficients(ka, eps_r)
    jn = (1j) ** (-n.astype(float))
    ein = np.exp(1j * np.multiply.outer(phi, n))
    out_s = (jn * a * hankel2(n, k0 * np.maximum(rho, 1e-300)[..., None]) * ein).sum(-1)
    inc = np.exp(-1j * k0 * x)
    if eps_r is None:
        inside = np.zeros_like(inc)
    else:
        m = np.sqrt(complex(eps_r))
        m = -m if m.imag > 0 else m
        inside = (jn * b * jv(n, m * k0 * rho[..., None]) * ein).sum(-1)
    outside = out_s + (inc if total else 0.0)
    return E0 * np.where(rho >= radius, outside, inside)


def half_wave_dipole_radiation_resistance(eta: float = ETA0) -> float:
    """``R = eta / (4 pi) Cin(2 pi)`` with ``Cin(x) = gamma + ln x - Ci(x)`` (Balanis eq. 4-93): 73.08 ohm in
    free space (input resistance of an infinitely thin lambda/2 dipole)."""
    from scipy.special import sici

    x = 2 * math.pi
    cin = np.euler_gamma + math.log(x) - sici(x)[1]
    return eta / (4 * math.pi) * float(cin)


def half_wave_dipole_directivity() -> float:
    """``D0 = 4 / Cin(2 pi)`` = 1.641 (2.15 dBi)."""
    from scipy.special import sici

    x = 2 * math.pi
    return 4.0 / float(np.euler_gamma + math.log(x) - sici(x)[1])


# ------------------------------------------------------------------------------------------ plasmonics
def spp_wavenumber(omega, eps_metal, eps_dielectric: float = 1.0):
    """Surface plasmon polariton on a flat interface: ``k_sp = (omega/c) sqrt(eps_m eps_d / (eps_m + eps_d))``
    (Maier eq. 2.14), with ``Im(k_sp) <= 0`` for a lossy metal under ``exp(+j omega t)``."""
    em = np.asarray(eps_metal, complex)
    k = np.asarray(omega, float) / C0 * np.sqrt(em * eps_dielectric / (em + eps_dielectric))
    return np.where(k.real < 0, -k, k)


def plasma_dispersion_omega(k, omega_p: float):
    """Transverse waves in a collisionless plasma: ``omega^2 = omega_p^2 + c^2 k^2``."""
    return np.sqrt(omega_p ** 2 + (C0 * np.asarray(k, float)) ** 2)

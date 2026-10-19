"""Atmospheric turbulence and gusts for flight simulation (MIL-F-8785C / MIL-HDBK-1797).

Intensities and scale lengths (:func:`turbulence_spec`):

* low altitude, h < 1000 ft: sigma_w = 0.1 W20, sigma_u = sigma_v = sigma_w / (0.177 + 0.000823 h)^0.4,
  L_w = h, L_u = L_v = h / (0.177 + 0.000823 h)^1.2 (h, L in ft; W20 the wind speed at 20 ft: 15 / 30 / 45 kt for
  light / moderate / severe);
* medium / high altitude, h > 2000 ft: isotropic, sigma from the probability-of-exceedance chart (light 1e-2, moderate
  1e-3, severe 1e-5), L = 1750 ft (Dryden) or 2500 ft (von Karman);
* in between, intensities and lengths are interpolated linearly between their values at 1000 ft and 2000 ft.

The scale lengths follow MIL-F-8785C (L_u = L_v at low altitude, all equal above 2000 ft). MIL-HDBK-1797 writes the
same spectra with half lengths for v and w (L_v -> 2 L_v); the two forms are the same functions of frequency.

Spectra (one-sided in the spatial frequency Omega [rad/m], integral over 0..inf = sigma^2, :func:`psd`):

    Dryden       Phi_u = s_u^2 2L_u/pi / (1 + (L_u W)^2)
                 Phi_v = s_v^2 L_v/pi (1 + 3 (L_v W)^2) / (1 + (L_v W)^2)^2            (w as v with L_w)
    von Karman   Phi_u = s_u^2 2L_u/pi / (1 + (1.339 L_u W)^2)^(5/6)
                 Phi_v = s_v^2 L_v/pi (1 + 8/3 (1.339 L_v W)^2) / (1 + (1.339 L_v W)^2)^(11/6)

Generators, all frozen in space (Taylor's hypothesis: the aircraft flies through a field that does not change while
it crosses it), validated against the target spectra and variances in ``tests/test_flight_turbulence.py``:

* :class:`FrozenLine`: one realisation along the flight path by spectral synthesis (random phases, exact spectral
  amplitudes), any of the two forms; the same gust is met by the wing and, later, by the tail;
* :class:`DrydenFilter`: the MIL-HDBK-1797 forming filters driven by white noise, discretised exactly (Van Loan), for
  streaming / real-time use at one point;
* :class:`FrozenField3D`: an isotropic, divergence-free 3-D von Karman field from random Fourier modes (Kraichnan
  1970), evaluated at any points: a large aircraft sees different gusts on each wing (rolling moment from the
  spanwise gust gradient);
* :class:`DiscreteGust`: the 1-cosine gust (MIL-F-8785C ramp form, or the full 1-cosine shape of length 2H used for
  the Pratt formula and CS-25.341);
* :class:`ShearWind`: mean wind with the MIL-F-8785C logarithmic profile near the ground.

Every field returns the wind velocity in the earth frame NED (x north, y east, z down) in m/s at points given in the
same frame, from ``velocity(t, points)``. The turbulence components u (longitudinal), v (lateral) and w (vertical) are
taken along the flight path heading ``heading_deg`` (u horizontal along it, v to its right, w down).

Sources: MIL-F-8785C (1980), Military Specification, Flying Qualities of Piloted Airplanes, section 3.7; MIL-HDBK-1797
(1997), Flying Qualities of Piloted Aircraft, section 4.9 and appendix; the high-altitude intensity chart (MIL-F-8785C
Figure 7) as tabulated in ``HIGH_ALTITUDE_SIGMA_FT_S``. Random Fourier modes: Kraichnan, R. H. (1970), Phys. Fluids 13,
22-31; Smirnov, Shi and Celik (2001), J. Fluids Eng. 123, 359-371.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm

FT = 0.3048
KT = 0.514444
VK_A = 1.339                                 # von Karman constant in the MIL spectra

W20_KT = {"light": 15.0, "moderate": 30.0, "severe": 45.0}
PROBABILITY = {"light": 1e-2, "moderate": 1e-3, "severe": 1e-5}

# MIL-F-8785C Figure 7 (MIL-HDBK-1797 Figure 266): RMS turbulence intensity [ft/s] against altitude [ft] for
# probabilities of exceedance 2e-1 ... 1e-6 (the table used by common implementations of the specification)
HIGH_ALTITUDE_H_FT = np.array([500, 1750, 3750, 7500, 15000, 25000, 35000, 45000, 55000, 65000, 75000, 80000.0])
HIGH_ALTITUDE_SIGMA_FT_S = {
    2e-1: [3.2, 2.2, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    1e-1: [4.2, 3.6, 3.3, 1.6, 0, 0, 0, 0, 0, 0, 0, 0],
    1e-2: [6.6, 6.9, 7.4, 6.7, 4.6, 2.7, 0.4, 0, 0, 0, 0, 0],
    1e-3: [8.6, 9.6, 10.6, 10.1, 8.0, 6.6, 5.0, 4.2, 2.7, 0, 0, 0],
    1e-4: [11.8, 13.0, 16.0, 15.1, 11.6, 9.7, 8.1, 8.2, 7.9, 4.9, 3.2, 2.1],
    1e-5: [15.6, 17.6, 23.0, 23.6, 22.1, 20.0, 16.0, 15.1, 12.1, 7.9, 6.2, 5.1],
    1e-6: [18.7, 21.5, 28.4, 30.2, 30.7, 31.0, 25.2, 23.1, 17.5, 10.7, 8.4, 7.2],
}


@dataclass(frozen=True)
class TurbulenceSpec:
    """RMS intensities [m/s] and scale lengths [m] of the three turbulence components, and the spectral form."""
    sigma_u: float
    sigma_v: float
    sigma_w: float
    L_u: float
    L_v: float
    L_w: float
    model: str = "von_karman"               # "dryden" | "von_karman"
    altitude: float = float("nan")          # m, informative
    severity: str = ""

    @property
    def sigma(self) -> np.ndarray:
        return np.array([self.sigma_u, self.sigma_v, self.sigma_w])

    @property
    def lengths(self) -> np.ndarray:
        return np.array([self.L_u, self.L_v, self.L_w])

    def as_dict(self):
        return {k: getattr(self, k) for k in ("sigma_u", "sigma_v", "sigma_w", "L_u", "L_v", "L_w", "model",
                                               "altitude", "severity")}


def _low(h_ft: float, w20_kt: float):
    h = max(h_ft, 10.0)
    sw = 0.1 * w20_kt * KT / FT                                  # ft/s
    f = 0.177 + 0.000823 * h
    su = sw / f ** 0.4
    Lu = h / f ** 1.2
    return np.array([su, su, sw]), np.array([Lu, Lu, h])


def _high(h_ft: float, prob: float, model: str):
    s = float(np.interp(h_ft, HIGH_ALTITUDE_H_FT, HIGH_ALTITUDE_SIGMA_FT_S[prob]))
    L = 1750.0 if model == "dryden" else 2500.0
    return np.full(3, s), np.full(3, L)


def turbulence_spec(altitude: float, severity: str = "moderate", model: str = "von_karman", *,
                    w20: float | None = None, probability: float | None = None) -> TurbulenceSpec:
    """Intensities and scale lengths at ``altitude`` [m] (above ground level for the low-altitude model) for a
    severity ``light`` / ``moderate`` / ``severe``. ``w20`` [m/s] overrides the 20 ft wind of the severity (low
    altitude), ``probability`` the exceedance probability (medium / high altitude, a key of
    ``HIGH_ALTITUDE_SIGMA_FT_S``)."""
    if model not in ("dryden", "von_karman"):
        raise ValueError("model must be 'dryden' or 'von_karman'")
    w20_kt = W20_KT[severity] if w20 is None else w20 / KT
    prob = PROBABILITY[severity] if probability is None else probability
    h = altitude / FT
    if h <= 1000:
        s, L = _low(h, w20_kt)
    elif h >= 2000:
        s, L = _high(h, prob, model)
    else:
        s1, L1 = _low(1000.0, w20_kt)
        s2, L2 = _high(2000.0, prob, model)
        f = (h - 1000.0) / 1000.0
        s, L = s1 + f * (s2 - s1), L1 + f * (L2 - L1)
    s, L = s * FT, L * FT
    return TurbulenceSpec(*s, *L, model=model, altitude=float(altitude), severity=severity)


def psd(spec: TurbulenceSpec, Omega, component: str = "w") -> np.ndarray:
    """One-sided spatial PSD [(m/s)^2 / (rad/m)] of component u, v or w at spatial frequency Omega [rad/m]."""
    i = "uvw".index(component)
    s, L = spec.sigma[i], spec.lengths[i]
    W = np.abs(np.asarray(Omega, float))
    if spec.model == "dryden":
        if i == 0:
            return s * s * 2 * L / math.pi / (1 + (L * W) ** 2)
        return s * s * L / math.pi * (1 + 3 * (L * W) ** 2) / (1 + (L * W) ** 2) ** 2
    x = (VK_A * L * W) ** 2
    if i == 0:
        return s * s * 2 * L / math.pi / (1 + x) ** (5 / 6)
    return s * s * L / math.pi * (1 + 8 / 3 * x) / (1 + x) ** (11 / 6)


def psd_temporal(spec: TurbulenceSpec, omega, V: float, component: str = "w") -> np.ndarray:
    """One-sided temporal PSD [(m/s)^2 / (rad/s)] seen at true airspeed V (frozen field: Omega = omega / V)."""
    return psd(spec, np.asarray(omega, float) / V, component) / V


def welch_psd(x: np.ndarray, d: float, nseg: int = 16):
    """One-sided PSD of a sampled signal against angular frequency (rad per unit of the sample spacing ``d``):
    Welch's method with a Hann window, 50 % overlap. Returns (Omega, Phi) with integral of Phi dOmega = variance."""
    from scipy.signal import welch
    n = len(x)
    f, P = welch(np.asarray(x, float) - np.mean(x), fs=1.0 / d, nperseg=max(64, n // nseg), detrend=False)
    return 2 * math.pi * f, P / (2 * math.pi)


# ---------------------------------------------------------------------------------------------- frames
def path_axes(heading_deg: float) -> np.ndarray:
    """Rows: the u (along the heading, horizontal), v (to the right) and w (down) unit vectors in NED."""
    c, s = math.cos(math.radians(heading_deg)), math.sin(math.radians(heading_deg))
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


class WindField:
    """A wind field: ``velocity(t, points)`` -> (M, 3) NED wind [m/s] at (M, 3) NED points [m]."""

    def velocity(self, t: float, points: np.ndarray) -> np.ndarray:            # pragma: no cover - interface
        raise NotImplementedError

    def __add__(self, other: WindField) -> WindField:
        return SumWind([self, other])


@dataclass
class SumWind(WindField):
    fields: Sequence[WindField]

    def velocity(self, t, points):
        P = np.atleast_2d(points)
        return sum((f.velocity(t, P) for f in self.fields), np.zeros_like(P, dtype=float))


@dataclass
class ConstantWind(WindField):
    """Uniform steady wind ``ned`` [m/s] (the velocity of the air, e.g. (0, -10, 0) blows towards the west)."""
    ned: Sequence[float] = (0.0, 0.0, 0.0)

    def velocity(self, t, points):
        return np.tile(np.asarray(self.ned, float), (len(np.atleast_2d(points)), 1))


@dataclass
class ShearWind(WindField):
    """Mean wind with the MIL-F-8785C logarithmic profile W(h) = W20 ln(h / z0) / ln(20 ft / z0); ``from_deg`` the
    direction the wind comes from (meteorological, degrees from north), ``z0`` the roughness length (0.15 ft = 0.046 m
    for category C flight phases, MIL-F-8785C 3.7.2.1). Below z0 the wind is zero; ``ground_z`` is the NED z of the
    ground (h = ground_z - z)."""
    w20: float = 10.0
    from_deg: float = 270.0
    z0: float = 0.15 * FT
    ground_z: float = 0.0

    def speed(self, h):
        h = np.maximum(np.asarray(h, float), self.z0)
        return self.w20 * np.log(h / self.z0) / math.log(20 * FT / self.z0)

    def velocity(self, t, points):
        P = np.atleast_2d(points)
        W = self.speed(self.ground_z - P[:, 2])
        to = math.radians(self.from_deg + 180.0)
        return np.c_[W * math.cos(to), W * math.sin(to), np.zeros(len(P))]


# ---------------------------------------------------------------------------------------------- frozen line
@dataclass
class FrozenLine(WindField):
    """One frozen realisation of the three components along a straight line through ``origin`` with heading
    ``heading_deg``, length ``length`` [m] at spacing ``dx`` [m], by spectral synthesis: each Fourier coefficient has
    the exact target amplitude sqrt(Phi dOmega) and a random phase, so the periodogram of the whole line equals the
    target spectrum and the variance equals the integral of the spectrum over the represented band. The line is
    periodic with period ``length``; take it several scale lengths long. Points off the line take the value at their
    projection on it (no lateral variation: use :class:`FrozenField3D` for that)."""
    spec: TurbulenceSpec
    length: float = 40_000.0
    dx: float = 1.0
    heading_deg: float = 0.0
    origin: Sequence[float] = (0.0, 0.0, 0.0)
    seed: int | None = 0
    series: np.ndarray = field(init=False, repr=False)        # (3, n): u, v, w

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        n = int(round(self.length / self.dx))
        n += n % 2
        self.n = n
        dW = 2 * math.pi / (n * self.dx)
        W = np.arange(n // 2 + 1) * dW
        out = []
        for c in "uvw":
            P = psd(self.spec, W, c)
            P[0] = 0.0                                              # zero mean
            P[-1] = 0.0                                             # no Nyquist term
            amp = np.sqrt(2 * P * dW)                                # x = sum amp cos(W x + phase): var = sum P dW
            ph = rng.uniform(0, 2 * math.pi, len(W))
            Z = amp * np.exp(1j * ph) * n / 2
            out.append(np.fft.irfft(Z, n))
        self.series = np.array(out)
        self.axes = path_axes(self.heading_deg)

    def coordinate(self, points) -> np.ndarray:
        P = np.atleast_2d(points) - np.asarray(self.origin, float)
        return P @ self.axes[0]

    def components(self, s) -> np.ndarray:
        """(len(s), 3) u, v, w at path coordinates s [m] (linear interpolation, periodic)."""
        x = np.mod(np.asarray(s, float) / self.dx, self.n)
        i0 = np.floor(x).astype(int) % self.n
        f = x - np.floor(x)
        i1 = (i0 + 1) % self.n
        return (self.series[:, i0] * (1 - f) + self.series[:, i1] * f).T

    def velocity(self, t, points):
        return self.components(self.coordinate(points)) @ self.axes


# ---------------------------------------------------------------------------------------------- Dryden filters
class DrydenFilter:
    """MIL-HDBK-1797 Dryden forming filters, driven by white noise, for one point flying at true airspeed V:

        H_u(s) = s_u sqrt(2 L_u / (pi V)) / (1 + L_u/V s)
        H_v(s) = s_v sqrt(L_v / (pi V)) (1 + sqrt(3) L_v/V s) / (1 + L_v/V s)^2        (w as v)

    The white noise has a two-sided spectral density pi so the output has the one-sided MIL spectrum; the state-space
    filters are discretised exactly for the step ``dt`` (Van Loan), so the variance is exact at any step. ``step()``
    returns the next (u, v, w) sample in path axes."""

    def __init__(self, spec: TurbulenceSpec, V: float, dt: float, seed: int | None = 0):
        if spec.model != "dryden":
            spec = TurbulenceSpec(spec.sigma_u, spec.sigma_v, spec.sigma_w, spec.L_u, spec.L_v, spec.L_w, "dryden",
                                  spec.altitude, spec.severity)
        self.spec, self.V, self.dt = spec, V, dt
        self.rng = np.random.default_rng(seed)
        self.blocks = []
        for i in range(3):
            s, L = spec.sigma[i], spec.lengths[i]
            T = L / V
            if i == 0:
                A = np.array([[-1 / T]])
                B = np.array([[1.0]])
                C = np.array([[s * math.sqrt(2 * L / (math.pi * V)) / T]])
            else:
                # (1 + sqrt3 T s) / (1 + T s)^2 = (sqrt3/T s + 1/T^2) / (s^2 + 2/T s + 1/T^2), controllable form
                A = np.array([[0.0, 1.0], [-1 / T ** 2, -2 / T]])
                B = np.array([[0.0], [1.0]])
                C = s * math.sqrt(L / (math.pi * V)) * np.array([[1 / T ** 2, math.sqrt(3) / T]])
            B = B * math.sqrt(math.pi)                               # two-sided white-noise density pi
            nx = A.shape[0]
            M = np.zeros((2 * nx, 2 * nx))
            M[:nx, :nx], M[:nx, nx:], M[nx:, nx:] = -A, B @ B.T, A.T
            E = expm(M * dt)
            Ad = E[nx:, nx:].T
            Q = Ad @ E[:nx, nx:]
            Q = 0.5 * (Q + Q.T)
            Lq = np.linalg.cholesky(Q + 1e-300 * np.eye(nx))
            # start in the stationary state: solve the discrete Lyapunov equation P = Ad P Ad' + Q
            from scipy.linalg import solve_discrete_lyapunov
            P0 = solve_discrete_lyapunov(Ad, Q)
            x0 = np.linalg.cholesky(P0 + 1e-300 * np.eye(nx)) @ self.rng.standard_normal(nx)
            self.blocks.append([Ad, Lq, C, x0])

    def step(self) -> np.ndarray:
        out = np.empty(3)
        for i, b in enumerate(self.blocks):
            Ad, Lq, C, x = b
            out[i] = (C @ x)[0]
            b[3] = Ad @ x + Lq @ self.rng.standard_normal(len(x))
        return out

    def series(self, n: int) -> np.ndarray:
        """(n, 3) consecutive samples (the same sequence as n calls of :meth:`step`)."""
        out = np.empty((n, 3))
        for k in range(n):
            out[k] = self.step()
        return out


# ---------------------------------------------------------------------------------------------- 3-D frozen field
def von_karman_energy(k, L: float, sigma: float = 1.0):
    """Isotropic von Karman energy spectrum E(k) = C sigma^2 L (a L k)^4 / (1 + (a L k)^2)^(17/6) [m^3/s^2],
    a = 1.339, with C fixed numerically so that the kinetic energy (integral of E) is 3 sigma^2 / 2. Its
    one-dimensional longitudinal and transverse spectra are then the MIL von Karman Phi_u and Phi_w with scale length
    L (checked by quadrature in ``tests/test_flight_turbulence.py``)."""
    x = VK_A * L * np.asarray(k, float)
    return sigma * sigma * L * _VK_E_CONST * x ** 4 / (1 + x * x) ** (17 / 6)


def _vk_const():
    from scipy.integrate import quad
    # integral of x^4 / (1 + x^2)^(17/6) dx over 0..inf, with k = x / (a L): integral E dk = C s^2 L I / (a L)
    integral = quad(lambda x: x ** 4 / (1 + x * x) ** (17 / 6), 0, np.inf, limit=400)[0]
    return 1.5 * VK_A / integral


_VK_E_CONST = _vk_const()


def isotropic_line_spectra(k1, L: float, sigma: float = 1.0):
    """One-sided longitudinal (F11) and transverse (F22) spectra [(m/s)^2 / (rad/m)] of the isotropic field with
    the energy spectrum :func:`von_karman_energy`, by quadrature of E(k) (Batchelor 1953):
    F11(k1) = int_k1^inf E(k)/k (1 - k1^2/k^2) dk, F22(k1) = 1/2 int_k1^inf E(k)/k (1 + k1^2/k^2) dk."""
    from scipy.integrate import quad
    k1 = np.atleast_1d(np.asarray(k1, float))
    F11 = np.array([quad(lambda k, q=q: von_karman_energy(k, L, sigma) / k * (1 - q * q / (k * k)), q, np.inf,
                         limit=400)[0] for q in k1])
    F22 = np.array([0.5 * quad(lambda k, q=q: von_karman_energy(k, L, sigma) / k * (1 + q * q / (k * k)), q, np.inf,
                               limit=400)[0] for q in k1])
    return F11, F22


@dataclass
class FrozenField3D(WindField):
    """Isotropic, divergence-free von Karman turbulence in 3-D from random Fourier modes:
    u(x) = sum_n a_n sigma_n cos(k_n . x + phi_n), with |k_n| on log-spaced shells between ``k_min`` and ``k_max``,
    directions uniform on the sphere (``n_dirs`` per shell), sigma_n a random unit vector normal to k_n (so
    div u = 0) and a_n = 2 sqrt(E(k_n) dk_n / n_dirs) (kinetic energy = integral of E over the band).

    The scale length is ``spec.L_w`` and the reference intensity ``spec.sigma_w``; the u, v, w components (path axes
    of ``heading_deg``) are then scaled by sigma_u / sigma_w and sigma_v / sigma_w (this keeps the low-altitude
    intensities but not their different scale lengths: an assumption, isotropy holds above 2000 ft). ``k_max``
    defaults to pi / (2 * dx_min) with ``dx_min`` the smallest length the aircraft must resolve (a chord); the
    fraction of the variance in the band is ``variance_fraction``. Evaluates at any (M, 3) points: cost M x modes."""
    spec: TurbulenceSpec
    n_shells: int = 96
    n_dirs: int = 24
    k_min: float | None = None
    k_max: float | None = None
    dx_min: float = 2.0
    heading_deg: float = 0.0
    seed: int | None = 0

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        L = self.spec.L_w
        kmin = self.k_min or 0.02 / L
        kmax = self.k_max or math.pi / (2 * self.dx_min)
        self.k_min_used, self.k_max_used = kmin, kmax          # the resolved band [rad/m]
        edges = np.geomspace(kmin, kmax, self.n_shells + 1)
        kc = np.sqrt(edges[:-1] * edges[1:])
        dk = np.diff(edges)
        E = von_karman_energy(kc, L, 1.0)
        n = self.n_shells * self.n_dirs
        # directions: uniform on the sphere (random), wave vectors and orthogonal unit amplitudes
        z = rng.uniform(-1, 1, n)
        ph = rng.uniform(0, 2 * math.pi, n)
        d = np.c_[np.sqrt(1 - z * z) * np.cos(ph), np.sqrt(1 - z * z) * np.sin(ph), z]
        kmag = np.repeat(kc, self.n_dirs) * np.exp(rng.uniform(-0.5, 0.5, n) * np.log(edges[1] / edges[0]))
        self.k = d * kmag[:, None]
        r = rng.standard_normal((n, 3))
        s = np.cross(d, r)
        self.s = s / np.linalg.norm(s, axis=1, keepdims=True)
        self.a = np.repeat(2 * np.sqrt(E * dk / self.n_dirs), self.n_dirs)
        self.phi = rng.uniform(0, 2 * math.pi, n)
        from scipy.integrate import quad
        tot = 1.5
        band = quad(lambda k: von_karman_energy(k, L, 1.0), kmin, kmax, limit=400)[0]
        self.variance_fraction = band / tot
        self.axes = path_axes(self.heading_deg)
        self.scale = np.array([self.spec.sigma_u, self.spec.sigma_v, self.spec.sigma_w])      # per path component

    def components(self, points_path) -> np.ndarray:
        """(M, 3) u, v, w at points given in path axes (u along the heading, v right, w down) [m]."""
        P = np.atleast_2d(points_path)
        arg = P @ self.k.T + self.phi                                       # (M, n)
        U = np.cos(arg) @ (self.a[:, None] * self.s)                        # unit-sigma field
        return U * self.scale

    def velocity(self, t, points):
        P = np.atleast_2d(points) @ self.axes.T
        return self.components(P) @ self.axes


# ---------------------------------------------------------------------------------------------- discrete gust
@dataclass
class DiscreteGust(WindField):
    """1-cosine gust met by the aircraft when its path coordinate s (along ``heading_deg`` from ``origin``) passes
    ``start``. ``shape="ramp"``: MIL-F-8785C 3.7.4, v = V_m/2 (1 - cos(pi x / d_m)) for 0 <= x <= d_m and V_m beyond;
    ``shape="full"``: v = V_m/2 (1 - cos(pi x / H)) for 0 <= x <= 2H and 0 elsewhere (CS-25.341, Pratt), with
    H = ``length``. ``direction``: unit vector in path axes (u, v, w), (0, 0, -1) an up-gust."""
    amplitude: float = 10.0
    length: float = 50.0
    start: float = 0.0
    shape: str = "full"
    direction: Sequence[float] = (0.0, 0.0, -1.0)
    heading_deg: float = 0.0
    origin: Sequence[float] = (0.0, 0.0, 0.0)

    def profile(self, x) -> np.ndarray:
        x = np.asarray(x, float)
        d = self.length
        if self.shape == "ramp":
            return np.where(x < 0, 0.0, np.where(x < d, 0.5 * self.amplitude * (1 - np.cos(math.pi * x / d)),
                                                 self.amplitude))
        return np.where((x >= 0) & (x <= 2 * d), 0.5 * self.amplitude * (1 - np.cos(math.pi * x / d)), 0.0)

    def velocity(self, t, points):
        ax = path_axes(self.heading_deg)
        s = (np.atleast_2d(points) - np.asarray(self.origin, float)) @ ax[0]
        g = self.profile(s - self.start)
        return g[:, None] * (np.asarray(self.direction, float) @ ax)[None]


def pratt_gust_load(wing_loading: float, cl_alpha: float, mac: float, U_de: float, V_e: float,
                    rho0: float = 1.225, rho: float = 1.225, g: float = 9.80665):
    """Pratt-Walker gust load factor increment (FAR 25.341 before amendment 25-86, CS-23.341):

        mu = 2 (W/S) / (rho c a g),  K_g = 0.88 mu / (5.3 + mu),  dn = K_g rho0 U_de V_e a / (2 W/S)

    with W/S [N/m^2], a [/rad], c the mean geometric chord [m], U_de and V_e equivalent speeds [m/s] and rho the
    density at altitude. Returns dict(mu, K_g, dn, dn_unalleviated)."""
    mu = 2 * wing_loading / (rho * mac * cl_alpha * g)
    Kg = 0.88 * mu / (5.3 + mu)
    dn0 = rho0 * U_de * V_e * cl_alpha / (2 * wing_loading)
    return {"mu": mu, "K_g": Kg, "dn": Kg * dn0, "dn_unalleviated": dn0}

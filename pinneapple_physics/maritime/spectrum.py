"""Wave spectra: JONSWAP and Pierson-Moskowitz, the sea-state numbers read back off a spectrum, and the time series a
spectrum implies.

The two shapes are the standard ones (Hasselmann et al. 1973; Pierson & Moskowitz 1964; the forms as in e.g. DNV-RP-C205
and Goda, *Random Seas and Design of Maritime Structures*, 3rd ed., 2010, ch. 2):

.. math::

    S(\\omega) = \\frac{\\alpha g^2}{\\omega^5} \\exp\\left[-\\frac{5}{4}\\left(\\frac{\\omega_p}{\\omega}\\right)^4\\right]
                 \\gamma^{r}, \\qquad
    r = \\exp\\left[-\\frac{(\\omega - \\omega_p)^2}{2 \\sigma^2 \\omega_p^2}\\right]

with ``sigma = 0.07`` below the peak and ``0.09`` above it, and ``gamma = 1`` giving Pierson-Moskowitz. The significant
height is not a label put on the spectrum: ``Hs`` is fixed by scaling ``alpha`` so that ``4 sqrt(m0)`` equals the ``Hs``
that was asked for, which is the one place where the spectrum and its own integral can disagree and the module makes
them agree to machine precision.

Two relations are worth stating because they are what the rest of the package leans on:

* a fully developed Pierson-Moskowitz sea for a wind ``U`` has ``Hs = 0.21 U^2/g`` and ``Tp = 7.16 U/g`` (the constants
  come out of ``alpha = 0.0081`` and ``omega_p = 0.877 g/U``, they are not fitted here);
* ``Hs`` of a time series is ``4`` times the standard deviation of the surface elevation, so a synthesized record and the
  spectrum it came from have to agree on it -- :func:`components` is checked against that in the tests.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np

__all__ = ["G", "wavenumber", "wavelength", "group_velocity", "phase_speed", "Jonswap", "fully_developed", "moments",
           "hs_from_spectrum", "peak_period", "components", "WaveComponents", "fit", "wave_age", "GAMMA_DEFAULT"]

G = 9.81                      # m/s^2, what every relation here uses (no local gravity is guessed)
RHO = 1025.0                  # kg/m^3, sea water, only used where a mass is asked for
GAMMA_DEFAULT = 3.3           # the JONSWAP average of Hasselmann et al. (1973)
SIGMA_LOW, SIGMA_HIGH = 0.07, 0.09


def wavenumber(omega, depth: float | None = None) -> np.ndarray:
    """``k`` from the linear dispersion relation ``omega^2 = g k tanh(k h)``, deep water when ``depth`` is ``None``.

    Newton from the deep-water value converges in five iterations over the range of sea waves; the tangent form is
    used rather than ``tanh`` because it does not lose the shallow-water limit.
    """
    omega = np.asarray(omega, dtype=float)
    if depth is None:
        return omega ** 2 / G
    k = np.where(omega > 0, omega ** 2 / G, 0.0)
    for _ in range(40):
        th = np.tanh(k * depth)
        f = G * k * th - omega ** 2
        df = G * th + G * k * depth * (1.0 - th ** 2)
        step = np.where(np.abs(df) > 0, f / np.where(df == 0, 1.0, df), 0.0)
        k = np.maximum(k - step, 1e-12)
        if np.all(np.abs(step) < 1e-12 * np.maximum(k, 1.0)):
            break
    return k


def wavelength(period, depth: float | None = None) -> np.ndarray:
    """``L = 2 pi / k`` for a period in seconds (deep water when ``depth`` is ``None``)."""
    return 2.0 * math.pi / wavenumber(2.0 * math.pi / np.asarray(period, dtype=float), depth)


def phase_speed(period, depth: float | None = None) -> np.ndarray:
    """``c = omega / k``."""
    omega = 2.0 * math.pi / np.asarray(period, dtype=float)
    return omega / wavenumber(omega, depth)


def group_velocity(period, depth: float | None = None) -> np.ndarray:
    """``c_g = domega/dk``: ``g/(2 omega)`` in deep water, ``(omega/2k)(1 + 2kh/sinh(2kh))`` otherwise.

    This is the speed at which the energy of one band travels, so it is what turns a distance into a lead time -- a
    14 s swell covers 10.9 m/s * 3600 s = 39 km in an hour, and a 6 s sea covers 4.7 m/s.
    """
    omega = 2.0 * math.pi / np.asarray(period, dtype=float)
    if depth is None:
        return G / (2.0 * omega)
    k = wavenumber(omega, depth)
    kh = k * depth
    return omega / (2.0 * k) * (1.0 + 2.0 * kh / np.sinh(2.0 * kh))


def wave_age(cp, wind_speed):
    """``c_p / U``: a fully developed Pierson-Moskowitz sea sits at 1.14, and anything above it is swell."""
    return np.asarray(cp, dtype=float) / np.maximum(np.asarray(wind_speed, dtype=float), 1e-9)


@dataclass(frozen=True)
class Jonswap:
    """A JONSWAP spectrum (``gamma = 1`` is Pierson-Moskowitz) whose own integral is exactly ``Hs``.

    ``hs`` [m] and ``tp`` [s] are what the sea state is described by; ``alpha`` is solved for, not quoted.
    """

    hs: float
    tp: float
    gamma: float = GAMMA_DEFAULT

    def __post_init__(self) -> None:
        if not (self.hs > 0 and self.tp > 0):
            raise ValueError(f"hs and tp must be positive, got hs={self.hs}, tp={self.tp}")
        if self.gamma < 1.0:
            raise ValueError(f"gamma >= 1 (1 is Pierson-Moskowitz), got {self.gamma}")

    @property
    def omega_p(self) -> float:
        return 2.0 * math.pi / self.tp

    @property
    def fp(self) -> float:
        return 1.0 / self.tp

    def shape(self, omega) -> np.ndarray:
        """The spectrum with ``alpha = 1`` -- the shape only, in s^5/rad, which the normalisation divides out."""
        omega = np.asarray(omega, dtype=float)
        safe = np.maximum(omega, 1e-12)
        ratio = self.omega_p / safe
        sigma = np.where(safe <= self.omega_p, SIGMA_LOW, SIGMA_HIGH)
        peak = np.exp(-((safe - self.omega_p) ** 2) / (2.0 * sigma ** 2 * self.omega_p ** 2))
        return G ** 2 / safe ** 5 * np.exp(-1.25 * ratio ** 4) * self.gamma ** peak

    def __call__(self, omega) -> np.ndarray:
        """``S(omega)`` in m^2 s/rad, normalised so that ``integral S domega = Hs^2/16``."""
        return self.alpha * self.shape(omega)

    @cached_property
    def alpha(self) -> float:
        """The Phillips constant that makes ``4 sqrt(m0) = Hs``."""
        omega = np.linspace(0.2 * self.omega_p, 12.0 * self.omega_p, 8001)
        m0 = float(np.trapezoid(self.shape(omega), omega))
        return (self.hs ** 2 / 16.0) / m0

    def density_hz(self, freq) -> np.ndarray:
        """``S(f)`` in m^2/Hz, which is what a buoy reports: ``S(f) = 2 pi S(omega)``."""
        return 2.0 * math.pi * self(2.0 * math.pi * np.asarray(freq, dtype=float))

    @property
    def m0(self) -> float:
        return self.hs ** 2 / 16.0

    def significant_height(self) -> float:
        """``4 sqrt(m0)`` of the discretised spectrum -- a round trip that must return ``hs``."""
        freq = np.linspace(1e-3, 6.0 / self.tp, 40001)[1:]
        return hs_from_spectrum(freq, self.density_hz(freq))

    def steepness(self) -> float:
        """``Hs / L_p``, the number that says whether the wave is about to break (0.14 is the usual limit in deep water)."""
        return self.hs / float(wavelength(self.tp))


def fully_developed(wind_speed: float, alpha: float = 0.0081) -> Jonswap:
    """A fully developed Pierson-Moskowitz sea for a 10 m wind of ``wind_speed`` [m/s].

    ``omega_p = 0.877 g/U`` and ``m0 = alpha g^2 / (5 omega_p^4)`` (Pierson & Moskowitz 1964), which give
    ``Tp = 7.16 U/g`` and ``Hs = 0.21 U^2/g``.
    """
    if not wind_speed > 0:
        raise ValueError(f"wind speed must be positive, got {wind_speed}")
    omega_p = 0.877 * G / wind_speed
    m0 = alpha * G ** 2 / (5.0 * omega_p ** 4)
    return Jonswap(hs=4.0 * math.sqrt(m0), tp=2.0 * math.pi / omega_p, gamma=1.0)


def moments(freq, density) -> dict[str, float]:
    """The spectral moments of a measured spectrum on the bands a buoy reports, plus what is read off them.

    ``m_n = integral f^n S(f) df`` with the band widths implied by the band centres, so a non-uniform grid (NDBC's is
    one) integrates the same way a uniform one would.
    """
    freq = np.asarray(freq, dtype=float)
    density = np.nan_to_num(np.asarray(density, dtype=float), nan=0.0)
    m0 = float(np.trapezoid(density, freq))
    m1 = float(np.trapezoid(density * freq, freq)) if freq.size > 1 else 0.0
    m2 = float(np.trapezoid(density * freq ** 2, freq)) if freq.size > 1 else 0.0
    return {"m0": m0, "m1": m1, "m2": m2, "hs": 4.0 * math.sqrt(max(m0, 0.0)),
            "tm01": m0 / m1 if m1 > 0 else float("nan"), "tm02": math.sqrt(m0 / m2) if m2 > 0 else float("nan"),
            "tp": 1.0 / peak_frequency(freq, density) if m0 > 0 else float("nan")}


def hs_from_spectrum(freq, density) -> float:
    """``4 sqrt(m0)`` -- the significant height of a measured or computed spectrum."""
    freq = np.asarray(freq, dtype=float)
    density = np.nan_to_num(np.asarray(density, dtype=float), nan=0.0)
    if freq.size < 2:
        return float("nan")
    return 4.0 * math.sqrt(max(float(np.trapezoid(density, freq)), 0.0))


def peak_frequency(freq, density) -> float:
    """The frequency of the largest band, refined by a parabola through it and its two neighbours.

    The band grid is coarse (0.005-0.02 Hz), so the raw argmax would quantise the peak period to 0.5 s or worse at the
    long-period end; the parabola recovers the peak between samples.
    """
    freq = np.asarray(freq, dtype=float)
    density = np.nan_to_num(np.asarray(density, dtype=float), nan=0.0)
    if density.size < 3 or not np.any(density > 0):
        return float("nan")
    i = int(np.argmax(density))
    if i == 0 or i == density.size - 1 or density[i + 1] <= 0 or density[i - 1] <= 0:
        return float(freq[i])
    y0, y1, y2 = (math.log(density[i - 1]), math.log(density[i]), math.log(density[i + 1]))
    denom = y0 - 2.0 * y1 + y2
    if denom == 0:
        return float(freq[i])
    shift = 0.5 * (y0 - y2) / denom
    left, right = freq[i] - freq[i - 1], freq[i + 1] - freq[i]
    return float(freq[i] + np.clip(shift, -1.0, 1.0) * (right if shift > 0 else left))


def peak_period(freq, density) -> float:
    """``1 / f_peak``, the period a buoy reports as the dominant period."""
    f = peak_frequency(freq, density)
    return 1.0 / f if f > 0 else float("nan")


@dataclass(frozen=True)
class WaveComponents:
    """A spectrum cut into discrete components: the amplitudes, the frequencies and the phases a record is built from."""

    freq: np.ndarray             # Hz
    amplitude: np.ndarray        # m, of the surface elevation
    phase: np.ndarray            # rad, uniform in [0, 2 pi) unless somebody set them

    def elevation(self, time) -> np.ndarray:
        """``eta(t) = sum a_i cos(2 pi f_i t + phi_i)`` for one time or an array of times."""
        time = np.atleast_1d(np.asarray(time, dtype=float))
        phase = 2.0 * math.pi * np.outer(time, self.freq) + self.phase[None, :]
        return np.cos(phase) @ self.amplitude

    def vertical_acceleration(self, time) -> np.ndarray:
        """``-sum a_i omega_i^2 cos(...)``, the acceleration a deck-mounted instrument or a cargo would feel."""
        omega = 2.0 * math.pi * self.freq
        time = np.atleast_1d(np.asarray(time, dtype=float))
        phase = 2.0 * math.pi * np.outer(time, self.freq) + self.phase[None, :]
        return -(np.cos(phase) * omega[None, :] ** 2) @ self.amplitude

    @property
    def hs(self) -> float:
        """``4 sqrt(sum a_i^2 / 2)``, which is what the record will measure."""
        return 4.0 * math.sqrt(float(np.sum(self.amplitude ** 2)) / 2.0)


def components(freq, density, *, seed: int = 0, count: int | None = None) -> WaveComponents:
    """Cut a spectrum into ``count`` equal-energy components.

    Equal-energy bins are used rather than the published band centres: the bands are coarse and unevenly spaced, and a
    component per band would put the whole of the peak's energy at one frequency, which makes a record that repeats.
    ``a_i = sqrt(2 m0 / n)``, so the sum reproduces the spectrum's ``m0`` exactly.
    """
    freq = np.asarray(freq, dtype=float)
    density = np.nan_to_num(np.asarray(density, dtype=float), nan=0.0)
    if count is None:
        count = 128
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(freq))])
    total = cdf[-1]
    if not total > 0:
        raise ValueError("the spectrum has no energy")
    targets = (np.arange(count) + 0.5) / count * total
    f_i = np.interp(targets, cdf, freq)
    amplitude = np.full(count, math.sqrt(2.0 * total / count))
    rng = np.random.default_rng(seed)
    return WaveComponents(freq=f_i, amplitude=amplitude, phase=rng.uniform(0.0, 2.0 * math.pi, count))


def fit(freq, density, *, guess: Jonswap | None = None, gamma_fixed: float | None = None) -> tuple[Jonswap, float]:
    """Fit a JONSWAP shape to a measured spectrum, in log space, over the bands that carry energy.

    Returns the spectrum and the RMS error of the fit in log space over the bands that carry the sea -- the number that
    says whether the analytic shape is a description of this sea or a caricature of it. Only bands above a tenth of the
    peak are fitted: the tail of a buoy spectrum is quantised noise, and including it would measure the instrument
    rather than the shape. ``gamma`` is fitted unless ``gamma_fixed`` is given, which is how the Pierson-Moskowitz shape
    is put against the same measurement.
    """
    from scipy.optimize import least_squares

    freq = np.asarray(freq, dtype=float)
    density = np.nan_to_num(np.asarray(density, dtype=float), nan=0.0)
    keep = (density > 0.10 * max(density.max(), 1e-12)) & (freq > 0)
    f_k, s_k = freq[keep], density[keep]
    if f_k.size < 4:
        raise ValueError("not enough energetic bands to fit a spectrum")
    start = guess or Jonswap(hs=max(hs_from_spectrum(freq, density), 0.05), tp=peak_period(freq, density))

    def residual(p: np.ndarray) -> np.ndarray:
        hs, tp = math.exp(p[0]), math.exp(p[1])
        gamma = gamma_fixed if gamma_fixed is not None else math.exp(p[2])
        model = Jonswap(hs=hs, tp=tp, gamma=max(gamma, 1.0)).density_hz(f_k)
        return np.log(np.maximum(model, 1e-12)) - np.log(s_k)

    p0 = [math.log(start.hs), math.log(start.tp)] + ([] if gamma_fixed is not None else [math.log(start.gamma)])
    lower = [math.log(1e-2), math.log(1.0)] + ([] if gamma_fixed is not None else [math.log(1.0)])
    upper = [math.log(30.0), math.log(30.0)] + ([] if gamma_fixed is not None else [math.log(12.0)])
    sol = least_squares(residual, p0, bounds=(lower, upper), xtol=1e-8, ftol=1e-8)
    hs, tp = math.exp(sol.x[0]), math.exp(sol.x[1])
    gamma = gamma_fixed if gamma_fixed is not None else math.exp(sol.x[2])
    return Jonswap(hs=hs, tp=tp, gamma=max(gamma, 1.0)), float(np.sqrt(np.mean(sol.fun ** 2)))

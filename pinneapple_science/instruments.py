"""Virtual laboratory instruments and the fits a bench experiment ends with.

An instrument reads a true value through what a real one adds: a gain error and an offset fixed for the
instrument (drawn once from its calibration uncertainty), random noise on every reading, and the resolution of its
display or converter. The bench knows only the specification (``noise``, ``gain_u``, ``offset_u``,
``resolution``), as on a data sheet, never the drawn errors, so the uncertainty it reports is the one a student or
a metrologist would compute (GUM type A from the scatter of the fit, type B from the specification).

    probe = FieldProbe(noise=2e-6, gain_u=0.005, offset_u=1e-6, resolution=1e-7, seed=1)
    ammeter = Ammeter(noise=1e-4, gain_u=0.002, resolution=1e-3, seed=2)
    B = probe.read(B_true)                     # one reading
    fit = linear_fit(I, B, sigma_y=probe.sigma(B))
    fit.slope, fit.slope_u                     # with the type-A standard uncertainty

References: JCGM 100:2008 (GUM), sec. 4.2-4.3 (type A and type B) and 5.1 (combined uncertainty); a uniform
distribution of half-width ``a`` (the resolution) has standard uncertainty ``a / sqrt(3)`` (GUM 4.3.7).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Instrument", "FieldProbe", "Ammeter", "Voltmeter", "Oscilloscope", "LinearFit", "linear_fit",
           "CurveFit", "curve_fit", "combine"]


@dataclass
class Instrument:
    """A reading instrument. ``noise``: standard deviation of the random noise per reading (in the unit read);
    ``gain_u``: relative standard uncertainty of the gain (calibration); ``offset_u``: standard uncertainty of the
    zero offset; ``resolution``: the quantisation step (0 for none). The gain and offset errors are drawn once,
    when the instrument is made."""

    name: str = "instrument"
    unit: str = ""
    noise: float = 0.0
    gain_u: float = 0.0
    offset_u: float = 0.0
    resolution: float = 0.0
    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False)
    _gain: float = field(init=False, repr=False)
    _offset: float = field(init=False, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        self._gain = 1.0 + self.gain_u * self._rng.standard_normal()
        self._offset = self.offset_u * self._rng.standard_normal()

    def read(self, true_value, n: int = 1):
        """Reading(s) of ``true_value`` (scalar or array); ``n > 1`` averages ``n`` readings per value."""
        x = np.asarray(true_value, float)
        reads = self._gain * x[..., None] + self._offset + self.noise * self._rng.standard_normal(x.shape + (n,))
        r = reads.mean(axis=-1)
        if self.resolution > 0:
            r = np.round(r / self.resolution) * self.resolution
        return float(r) if r.ndim == 0 else r

    def sigma(self, reading, n: int = 1):
        """Standard uncertainty of one (averaged) reading from the random part: noise and resolution."""
        s = math.sqrt(self.noise ** 2 / n + (self.resolution / math.sqrt(12)) ** 2)
        return np.full(np.shape(reading), s) if np.ndim(reading) else s

    def type_b(self, value: float) -> float:
        """Type-B standard uncertainty of a value read with this instrument from its calibration specification
        (gain and offset)."""
        return math.hypot(self.gain_u * value, self.offset_u)


def FieldProbe(**kw) -> Instrument:
    """A Hall magnetic-field probe (tesla)."""
    return Instrument(**{"name": "Hall probe", "unit": "T", **kw})


def Ammeter(**kw) -> Instrument:
    return Instrument(**{"name": "ammeter", "unit": "A", **kw})


def Voltmeter(**kw) -> Instrument:
    return Instrument(**{"name": "voltmeter", "unit": "V", **kw})


@dataclass
class Oscilloscope:
    """A digitising oscilloscope: samples ``signal(t)`` at ``rate`` over ``duration`` after ``t0`` with vertical
    noise, a vertical gain error, an ADC of ``bits`` over the full scale ``±v_range`` and a time-base error
    ``timebase_u`` (relative)."""

    rate: float
    duration: float
    v_range: float
    bits: int = 8
    noise: float = 0.0
    gain_u: float = 0.0
    timebase_u: float = 0.0
    seed: int | None = None

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self._rng = rng
        self._gain = 1.0 + self.gain_u * rng.standard_normal()
        self._tb = 1.0 + self.timebase_u * rng.standard_normal()

    @property
    def lsb(self) -> float:
        return 2 * self.v_range / 2 ** self.bits

    def capture(self, signal: Callable[[np.ndarray], np.ndarray], t0: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """``(t, v)``: the time axis as the scope labels it, and the quantised, clipped samples."""
        n = int(round(self.duration * self.rate))
        t_label = t0 + np.arange(n) / self.rate
        t_true = t0 + (t_label - t0) * self._tb
        v = self._gain * np.asarray(signal(t_true), float) + self.noise * self._rng.standard_normal(n)
        v = np.clip(np.round(v / self.lsb) * self.lsb, -self.v_range, self.v_range - self.lsb)
        return t_label, v

    def sigma(self) -> float:
        return math.sqrt(self.noise ** 2 + self.lsb ** 2 / 12)


@dataclass
class LinearFit:
    slope: float
    intercept: float
    slope_u: float
    intercept_u: float
    cov: np.ndarray
    chi2: float
    dof: int
    residuals: np.ndarray

    @property
    def reduced_chi2(self) -> float:
        return self.chi2 / self.dof if self.dof > 0 else float("nan")


def linear_fit(x: Sequence[float], y: Sequence[float], sigma_y=None, *, through_origin: bool = False,
               scale_by_chi2: bool = True) -> LinearFit:
    """Weighted least squares ``y = a x + b`` (``b = 0`` with ``through_origin``). Uncertainties from the
    covariance ``(A^T W A)^{-1}``, scaled by the reduced chi-square when it exceeds one (``scale_by_chi2``), so a
    scatter larger than the stated sigma is not hidden."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    s = np.ones_like(y) if sigma_y is None else np.broadcast_to(np.asarray(sigma_y, float), y.shape)
    A = x[:, None] if through_origin else np.stack([x, np.ones_like(x)], axis=1)
    W = 1.0 / s ** 2
    cov = np.linalg.inv(A.T @ (A * W[:, None]))
    beta = cov @ (A.T @ (W * y))
    res = y - A @ beta
    chi2 = float(np.sum(W * res ** 2))
    dof = len(y) - A.shape[1]
    if sigma_y is None or (scale_by_chi2 and dof > 0 and chi2 / dof > 1):
        cov = cov * (chi2 / dof if dof > 0 else 1.0)
    a = float(beta[0])
    b = 0.0 if through_origin else float(beta[1])
    au = math.sqrt(cov[0, 0])
    bu = 0.0 if through_origin else math.sqrt(cov[1, 1])
    return LinearFit(a, b, au, bu, cov, chi2, dof, res)


@dataclass
class CurveFit:
    params: np.ndarray
    errors: np.ndarray
    cov: np.ndarray
    chi2: float
    dof: int
    residuals: np.ndarray


def curve_fit(model: Callable, x, y, p0, sigma_y=None) -> CurveFit:
    """Non-linear least squares (scipy ``curve_fit``) with the same uncertainty rule as :func:`linear_fit`."""
    from scipy.optimize import curve_fit as _cf

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    s = None if sigma_y is None else np.broadcast_to(np.asarray(sigma_y, float), y.shape)
    p, cov = _cf(model, x, y, p0=p0, sigma=s, absolute_sigma=s is not None, maxfev=20000)
    res = y - model(x, *p)
    w = 1.0 if s is None else 1.0 / s ** 2
    chi2 = float(np.sum(w * res ** 2))
    dof = len(y) - len(p)
    if s is not None and dof > 0 and chi2 / dof > 1:
        cov = cov * chi2 / dof
    return CurveFit(np.asarray(p), np.sqrt(np.diag(cov)), cov, chi2, dof, res)


def combine(*relative_uncertainties: float) -> float:
    """Root-sum-square of independent relative standard uncertainties (GUM eq. 10 for a product of powers one)."""
    return math.sqrt(sum(u * u for u in relative_uncertainties))

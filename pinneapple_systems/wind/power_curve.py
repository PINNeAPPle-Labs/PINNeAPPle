"""Power curves: the physics that ties wind speed to electrical power.

Three families live here, and a report must say which one produced a number:

``ReferencePowerCurve``
    The analytic shape every turbine is specified with -- cubic in region 2,
    rated plateau in region 3, zero below cut-in and above cut-out, with an
    explicit :math:`C_p`. It is a **reference**, not a measurement, and it is the
    only curve that can be evaluated at a wind speed with no data behind it.
``BinnedPowerCurve``
    The IEC 61400-12-1 measured power curve: mean power in 0.5 m/s bins, which
    is the industry definition of "the power curve of this machine". Fitted on
    the training period only.
``LogisticPowerCurve``
    A five-parameter logistic fitted to the same points. Smooth, monotone by
    construction on the fitted range, and it extrapolates far less violently
    than an interpolated bin table.

Air-density handling follows IEC 61400-12-1: wind speed is scaled to the
reference density as :math:`u_0 = u (\\rho/\\rho_0)^{1/3}`, which is exact for
the cubic regime and an approximation once the machine is at rated power.
Density itself is estimated from the measured ambient temperature and a standard
atmosphere pressure at the site elevation -- the SCADA feed carries no pressure,
so this is a documented approximation, not a measurement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "RHO_REF",
    "air_density",
    "standard_pressure",
    "density_corrected_speed",
    "relative_density",
    "ReferencePowerCurve",
    "BinnedPowerCurve",
    "LogisticPowerCurve",
    "fit_binned_curve",
    "fit_logistic_curve",
    "betz_limit",
]

RHO_REF = 1.225          # kg/m^3, ISA sea-level density (IEC reference)
R_GAS = 287.05           # J/(kg K), dry air
T0_K = 273.15


# ---------------------------------------------------------------------------
# Air
# ---------------------------------------------------------------------------

def standard_pressure(elevation_m: float) -> float:
    """Barometric pressure [Pa] of the ISA standard atmosphere at an elevation."""
    return 101_325.0 * (1.0 - 2.25577e-5 * float(elevation_m)) ** 5.25588


def air_density(temperature_c, pressure_pa: float = 101_325.0):
    """Dry-air density [kg/m^3] from temperature [degC] and pressure [Pa]."""
    t_k = np.asarray(temperature_c, dtype=float) + T0_K
    return pressure_pa / (R_GAS * np.maximum(t_k, 1.0))


def relative_density(temperature_c, elevation_m: float) -> np.ndarray:
    """Density relative to :data:`RHO_REF` at the site's standard pressure."""
    rho = air_density(temperature_c, standard_pressure(elevation_m))
    return rho / RHO_REF


def density_corrected_speed(wind_speed, rho_relative) -> np.ndarray:
    """IEC 61400-12-1 density normalisation ``u (rho/rho_ref)^(1/3)``.

    The cube root comes from :math:`P \\propto \\rho u^3`: the speed that, at
    reference density, carries the same kinetic flux. It is exact in region 2
    and applied unchanged at rated power, where it is only an approximation.
    """
    return np.asarray(wind_speed, dtype=float) * np.cbrt(np.asarray(rho_relative, dtype=float))


def betz_limit() -> float:
    """16/27 -- the maximum fraction of the kinetic flux a rotor can extract."""
    return 16.0 / 27.0


# ---------------------------------------------------------------------------
# Reference (analytic) curve
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReferencePowerCurve:
    """Idealised turbine curve: cubic to rated, plateau, cut-out.

    This curve encodes *only* what the nameplate and momentum theory give:
    swept area, rated power, cut-in/out and a design :math:`C_p`. It is what a
    manufacturer would put in a datasheet and what an AEP estimate uses when no
    measurement exists. It is not fitted to anything and must never be reported
    as having been validated against data.
    """

    rated_power_kw: float
    rotor_diameter: float
    cut_in: float = 3.0
    cut_out: float = 25.0
    cp: float = 0.45
    rho_ref: float = RHO_REF
    name: str = "reference power curve"

    def __post_init__(self) -> None:
        if not 0 < self.cp <= betz_limit() + 1e-12:
            raise ValueError(f"Cp must be in (0, 16/27]; got {self.cp}")
        if self.rated_power_kw <= 0 or self.rotor_diameter <= 0:
            raise ValueError("rated power and rotor diameter must be positive")

    @property
    def rotor_area(self) -> float:
        return math.pi * (self.rotor_diameter / 2.0) ** 2

    @property
    def rated_speed(self) -> float:
        """Free-stream speed at which this curve reaches rated power [m/s]."""
        return (self.rated_power_kw * 1000.0 / (0.5 * self.rho_ref * self.rotor_area * self.cp)) ** (1.0 / 3.0)

    def power(self, wind_speed) -> np.ndarray:
        u = np.asarray(wind_speed, dtype=float)
        p = 0.5 * self.rho_ref * self.rotor_area * self.cp * np.maximum(u, 0.0) ** 3 / 1000.0
        p = np.minimum(p, self.rated_power_kw)
        operating = (u >= self.cut_in) & (u <= self.cut_out)
        return np.where(operating, p, 0.0)

    def cp_of(self, wind_speed) -> np.ndarray:
        """Effective :math:`C_p` including the pitch-back once rated is reached."""
        u = np.maximum(np.asarray(wind_speed, dtype=float), 1e-9)
        flux = 0.5 * self.rho_ref * self.rotor_area * u**3 / 1000.0
        return np.clip(self.power(u) / np.maximum(flux, 1e-12), 0.0, betz_limit())


# ---------------------------------------------------------------------------
# Measured (binned) curve -- IEC 61400-12-1
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BinnedPowerCurve:
    """IEC-style binned power curve fitted on a training period.

    ``counts`` is kept next to the values so every bin carries its support: a
    bin with 12 samples and a bin with 3000 samples are not the same evidence,
    and the reporting layer uses the counts to refuse extrapolation.
    """

    bin_centres: np.ndarray
    power_kw: np.ndarray
    counts: np.ndarray
    bin_width: float = 0.5
    min_count: int = 10
    cut_out: float | None = None
    name: str = "IEC binned power curve"
    meta: dict = field(default_factory=dict)

    def power(self, wind_speed) -> np.ndarray:
        """Interpolate the curve; refuses to invent values outside its support.

        Below the first supported bin the curve holds its first value (there is
        no power below cut-in anyway); above the last supported bin it returns
        zero past ``cut_out`` and holds the last value in between. Holding a
        plateau instead of extrapolating a cubic is the conservative choice.
        """
        u = np.asarray(wind_speed, dtype=float)
        supported = self.counts >= self.min_count
        if not supported.any():
            return np.zeros_like(u)
        centres = self.bin_centres[supported]
        values = self.power_kw[supported]
        out = np.interp(u, centres, values)
        out = np.where(u < centres[0], values[0], out)
        if self.cut_out is not None:
            out = np.where(u > self.cut_out, 0.0, out)
        return out

    def support(self) -> dict:
        supported = self.counts >= self.min_count
        return {
            "bins_total": int(self.counts.size),
            "bins_supported": int(supported.sum()),
            "speed_min": float(self.bin_centres[supported].min()) if supported.any() else None,
            "speed_max": float(self.bin_centres[supported].max()) if supported.any() else None,
            "samples": int(self.counts.sum()),
        }

    def to_records(self) -> list[dict]:
        return [
            {"centre_ms": float(c), "power_kw": float(p), "count": int(n),
             "supported": bool(n >= self.min_count)}
            for c, p, n in zip(self.bin_centres, self.power_kw, self.counts, strict=True)
        ]


def fit_binned_curve(
    wind_speed,
    power,
    *,
    bin_width: float = 0.5,
    min_count: int = 10,
    speed_range: tuple[float, float] = (0.0, 30.0),
    cut_out: float | None = 25.0,
    name: str = "IEC binned power curve",
    meta: dict | None = None,
) -> BinnedPowerCurve:
    """Bin a training period into the measured power curve.

    The caller is responsible for having filtered the operating mask (turbine
    running, not curtailed, not iced); this function only averages what it is
    given, exactly as IEC 61400-12-1 prescribes.
    """
    u = np.asarray(wind_speed, dtype=float)
    p = np.asarray(power, dtype=float)
    finite = np.isfinite(u) & np.isfinite(p)
    u, p = u[finite], p[finite]
    edges = np.arange(speed_range[0], speed_range[1] + bin_width, bin_width)
    idx = np.digitize(u, edges) - 1
    n_bins = edges.size - 1
    centres = 0.5 * (edges[:-1] + edges[1:])
    sums = np.bincount(idx, weights=p, minlength=n_bins)[:n_bins]
    counts = np.bincount(idx, minlength=n_bins)[:n_bins].astype(int)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return BinnedPowerCurve(
        bin_centres=centres,
        power_kw=means,
        counts=counts,
        bin_width=bin_width,
        min_count=min_count,
        cut_out=cut_out,
        name=name,
        meta=dict(meta or {}),
    )


# ---------------------------------------------------------------------------
# Logistic curve
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LogisticPowerCurve:
    """Five-parameter logistic power curve, smooth and bounded.

    ``power(u) = rated * ((1 + exp(-a (u - b)))^{-c} - (1 + exp(-a (u_in - b)))^{-c}) / normaliser``
    clipped to ``[0, rated]`` and gated by cut-in/cut-out. The shape has no
    claim to being physical inside the transition -- it is a conventional
    smooth fit, included because it is what most commercial tools use as their
    "parametric power curve".
    """

    rated_power_kw: float
    a: float
    b: float
    c: float
    cut_in: float = 3.0
    cut_out: float = 25.0
    name: str = "logistic power curve"

    def _raw(self, u):
        x = 1.0 + np.exp(-self.a * (u - self.b))
        return x ** (-self.c) - (1.0 + np.exp(-self.a * (self.cut_in - self.b))) ** (-self.c)

    def power(self, wind_speed) -> np.ndarray:
        u = np.asarray(wind_speed, dtype=float)
        top = self._raw(self.cut_out) if self.cut_out > self.b else self._raw(np.array(self.b + 50.0))
        raw = self._raw(u) / max(top, 1e-12)
        p = self.rated_power_kw * np.clip(raw, 0.0, 1.0)
        return np.where((u >= self.cut_in) & (u <= self.cut_out), p, 0.0)


def fit_logistic_curve(
    wind_speed,
    power,
    *,
    rated_power_kw: float,
    cut_in: float = 3.0,
    cut_out: float = 25.0,
    power_bins: int = 40,
) -> LogisticPowerCurve:
    """Least-squares fit of :class:`LogisticPowerCurve` to binned medians."""
    from scipy.optimize import curve_fit

    u = np.asarray(wind_speed, dtype=float)
    p = np.asarray(power, dtype=float)
    mask = (u >= cut_in) & (u <= cut_out) & np.isfinite(p)
    u, p = u[mask], p[mask]
    if u.size < 50:
        raise ValueError("not enough points inside the operating window to fit a logistic curve")

    edges = np.linspace(cut_in, cut_out, power_bins + 1)
    idx = np.digitize(u, edges) - 1
    centres, medians, weights = [], [], []
    for k in range(power_bins):
        sel = idx == k
        if sel.sum() >= 20:
            centres.append(float(np.mean(u[sel])))
            medians.append(float(np.median(p[sel])))
            weights.append(float(np.sqrt(sel.sum())))
    centres_a = np.asarray(centres)
    medians_a = np.asarray(medians)
    weights_a = np.asarray(weights)
    if centres_a.size < 5:
        raise ValueError("not enough populated speed bins to fit a logistic curve")

    def model(x, a, b, c):
        return LogisticPowerCurve(rated_power_kw, a, b, c, cut_in, cut_out).power(x)

    p0 = (1.2, 7.5, 1.5)
    bounds = ([0.05, cut_in, 0.1], [20.0, cut_out, 20.0])
    popt, _ = curve_fit(
        model, centres_a, medians_a, p0=p0, bounds=bounds,
        sigma=1.0 / np.maximum(weights_a, 1e-6), maxfev=20_000,
    )
    return LogisticPowerCurve(rated_power_kw, float(popt[0]), float(popt[1]), float(popt[2]), cut_in, cut_out)

"""A zonal energy-balance model of Mars: the insolation, the albedo of ice against regolith, the CO2 greenhouse,
and the heat the atmosphere and the ground carry poleward (#374).

The model is Budyko-Sellers in one dimension, ``x = sin(latitude)``, with a real annual integration of Mars' orbit
rather than a fitted insolation curve:

    C dT/dt = (1 - a(x,T)) S(x) - sigma T^4 + G(p) + F(t) + D d/dx[(1 - x^2) dT/dx]

* **S(x)** is the annual mean of the *daily* insolation, integrated over the orbit with Mars' eccentricity (0.0934)
  and obliquity (25.19 deg). The orbit is integrated, not fitted: the subsolar latitude follows
  ``sin(dec) = sin(obliquity) sin(Ls)`` with ``Ls`` running from the true anomaly, so the perihelion at
  ``Ls = 250.99 deg`` makes the southern summer the short hot one, and the derivative of the obliquity enters the
  answer. The global mean of ``S`` is checked against ``S0/4 (a/r)^2`` averaged over the orbit, which it matches to
  the discretisation.
* **a(x,T)** is the regolith or the frost, whichever the surface is: the Bond albedo of the fact sheet (0.250) where
  the ground is bare, and the albedo of CO2 frost where the CO2 cap survives. That contrast -- a bright cap against a
  dark surface -- is the strongest feedback in the model.
* **G(p)** is the CO2 greenhouse, in the form that has to be right for this problem: it **saturates**. The two ends
  are published (Jakosky & Edwards 2018: 20 mbar warms by *less than 10 K*, and about 1 bar would bring the surface
  close to melting, i.e. the ~60 K the paper calls necessary), and a power law through those two ends,
  ``G = G_1bar (p / 1 bar)^beta``, gives ``beta = 0.485``. Its prediction at today's 6.36 mbar is 5.2 K, which is the
  greenhouse the planet is observed to have -- a third point the fit was not given.
* **D** carries heat poleward, in the Budyko form ``D (1 - x^2) dT/dx`` with a finite-volume discretisation that
  conserves the global mean *exactly* (test: the diffusion term integrates to zero to roundoff even at the poles,
  where ``1 - x^2`` vanishes at the cell face).

The time integration is explicit in the radiation and implicit in the diffusion (a tridiagonal solve per step), so
a century costs hundreds of steps rather than hundreds of thousands.

What this model is not: it has no seasons (the insolation is annual and the seasonal cap of Mars, 2 mbar of CO2 that
comes and goes every year, is outside it), no dust, no clouds, no water cycle, and no dynamics beyond zonal diffusion.
The assumptions are listed in ``docs/guides/mars_terraform.md`` and, item by item, in :data:`ASSUMPTIONS`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .mars import MARS, Mars, perihelion_solar_longitude

__all__ = ["STEFAN_BOLTZMANN", "ASSUMPTIONS", "Grid", "Climate", "annual_insolation", "daily_insolation",
           "greenhouse_warming", "greenhouse_factor", "greenhouse_optical_depth", "ZonalEBM"]

#: Stefan-Boltzmann constant, W m^-2 K^-4.
STEFAN_BOLTZMANN = 5.670374419e-8

#: What the model assumes and cannot show, so the guide and the results file can say which is which.
ASSUMPTIONS: tuple[str, ...] = (
    "annual means only: the seasonal cycle of insolation and the 2 mbar seasonal CO2 cap are outside the model",
    "one atmosphere, well mixed: the CO2 partial pressure is uniform over the planet at every instant",
    "the greenhouse is a flux proportional to a power of the CO2 pressure, calibrated to two published ends "
    "(20 mbar -> <10 K, 1 bar -> ~60 K); its shape between and beyond them is an assumption",
    "an engineered greenhouse is applied as a flux in W/m2, with no chemistry, lifetime or source",
    "radiative equilibrium is reached locally apart from the zonal transport: no dynamics, no dust lifting, "
    "no clouds, no water cycle, no topography",
    "the heat capacity is one mixed layer of atmosphere and regolith skin, the same at every latitude",
    "the polar cap is a zonal film whose extent is set by the frost point at the current pressure; its thickness "
    "is uniform, and its total is Jakosky & Edwards' polar reservoir",
    "the regolith releases its CO2 at a single e-folding time, not through a thermal diffusion front",
)


@dataclass(frozen=True)
class Grid:
    """Equal-area latitude bands: uniform in ``x = sin(latitude)``, so each band carries the same area."""

    bands: int = 36

    @property
    def edges(self) -> np.ndarray:
        return np.linspace(-1.0, 1.0, self.bands + 1)

    @property
    def x(self) -> np.ndarray:
        e = self.edges
        return 0.5 * (e[1:] + e[:-1])

    @property
    def latitude_deg(self) -> np.ndarray:
        return np.degrees(np.arcsin(self.x))

    @property
    def weights(self) -> np.ndarray:
        """Fraction of the sphere's area in each band: ``d sin(latitude) / 2``, summing to 1."""
        e = self.edges
        return 0.5 * np.diff(e)

    @property
    def dx(self) -> float:
        return float(self.edges[1] - self.edges[0])

    def faces(self) -> np.ndarray:
        """``1 - x^2`` at the ``bands + 1`` cell faces: the poleward transport dies at both poles."""
        return 1.0 - self.edges ** 2


def daily_insolation(latitude_deg, solar_longitude_deg=0.0, distance_au=1.0) -> np.ndarray:
    """Daily mean top-of-atmosphere irradiance (W m^-2) at a latitude and a solar longitude.

    The standard box formula, ``S/pi (H0 sin(lat) sin(dec) + cos(lat) cos(dec) sin H0)`` with the hour angle
    ``H0 = arccos(-tan(lat) tan(dec))`` clipped at the terminator, the declination
    ``sin(dec) = sin(obliquity) sin(Ls)``, and the distance correction ``(a/r)^2`` folded into ``S``.
    """
    lat = np.deg2rad(np.asarray(latitude_deg, float))
    ls = np.deg2rad(np.asarray(solar_longitude_deg, float))
    dec = np.arcsin(np.clip(np.sin(np.deg2rad(MARS.obliquity_deg)) * np.sin(ls), -1.0, 1.0))
    cos_h0 = np.clip(-np.tan(lat) * np.tan(dec), -1.0, 1.0)
    h0 = np.arccos(cos_h0)
    irradiance = MARS.solar_irradiance_at(distance_au)
    return irradiance / np.pi * (h0 * np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.sin(h0))


def _orbit(samples: int = 720) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniform-in-time samples of Mars' orbit: mean anomaly, distance (au), and areocentric solar longitude."""
    mean_anomaly = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    e = MARS.eccentricity
    anomaly = mean_anomaly + 0.0
    for _ in range(60):                                       # Kepler, vectorised
        anomaly = anomaly - (anomaly - e * np.sin(anomaly) - mean_anomaly) / (1.0 - e * np.cos(anomaly))
    radius_au = MARS.semimajor_au * (1.0 - e * np.cos(anomaly))
    ls = np.degrees(anomaly) + perihelion_solar_longitude(MARS)
    return mean_anomaly, radius_au, ls


def annual_insolation(latitude_deg, samples: int = 720) -> np.ndarray:
    """Annual mean top-of-atmosphere irradiance (W m^-2) at each latitude, integrated over Mars' orbit.

    The average is taken over the mean anomaly, which is the average over time: ``(1/2 pi) int I dM``. The global
    mean of the answer is ``S0/4`` times the time average of ``(a/r)^2``, ``S0/(4 sqrt(1 - e^2))`` -- a number this
    function does not use and is checked against.
    """
    lat = np.atleast_1d(np.asarray(latitude_deg, float))
    _, radius_au, ls = _orbit(samples)
    total = np.zeros_like(lat)
    for r, s in zip(radius_au, ls, strict=True):
        total += daily_insolation(lat, s, r)
    return total / samples


#: The two published ends of the CO2 greenhouse, and the shape that joins them.
GREENHOUSE_AT_ONE_BAR_K = 60.0
GREENHOUSE_REFERENCE_PRESSURE_PA = 1.0e5
GREENHOUSE_EXPONENT = float(np.log(9.0 / GREENHOUSE_AT_ONE_BAR_K) / np.log(2_000.0 / GREENHOUSE_REFERENCE_PRESSURE_PA))


def greenhouse_warming(pressure_pa, one_bar_k: float = GREENHOUSE_AT_ONE_BAR_K,
                       exponent: float = GREENHOUSE_EXPONENT) -> np.ndarray:
    """CO2 greenhouse warming (K) at CO2 surface pressure ``pressure_pa``: ``G_1bar (p / 1 bar)^beta``.

    ``beta = 0.485`` passes through ``G(20 mbar) = 9 K`` (Jakosky & Edwards: "less than 10 K") and
    ``G(1 bar) = 60 K`` (their "~60 K warming necessary"). The third point, ``G(6.36 mbar) = 5.2 K``, is a
    prediction and it is the greenhouse Mars is observed to have.
    """
    p = np.clip(np.asarray(pressure_pa, float), 0.0, None)
    return one_bar_k * (p / GREENHOUSE_REFERENCE_PRESSURE_PA) ** exponent


def greenhouse_optical_depth(pressure_pa, one_bar_k: float = GREENHOUSE_AT_ONE_BAR_K,
                            exponent: float = GREENHOUSE_EXPONENT,
                            reference_k: float = MARS.blackbody_temperature_k) -> np.ndarray:
    """The infrared optical depth whose grey-slab warming is :func:`greenhouse_warming`.

    A grey atmosphere raises the surface above the effective temperature by ``T_s = T_eff (1 + tau)^(1/4)``, so the
    warming the literature quotes is a statement about ``(1 + tau)`` and not about a flux added at the surface:

        tau = (1 + dT_g / T_eff)^4 - 1

    Written this way the greenhouse costs the planet a *fraction* of its outgoing radiation, which is what a
    greenhouse does, and it therefore stays a fixed number of kelvin however hot the surface gets instead of
    compounding with it. The two published ends land where the paper puts them: 20 mbar gives 9 K, and 1 bar --
    ``tau = 1.73`` -- gives 60 K, close to but short of the 273 K melting point, which is exactly Jakosky & Edwards'
    "about 1 bar ... close to the melting point of ice".
    """
    warming = greenhouse_warming(pressure_pa, one_bar_k, exponent)
    return (1.0 + warming / reference_k) ** 4 - 1.0


def greenhouse_factor(pressure_pa, one_bar_k: float = GREENHOUSE_AT_ONE_BAR_K,
                      exponent: float = GREENHOUSE_EXPONENT) -> np.ndarray:
    """``1 + tau``: the factor by which CO2 pressure multiplies the outgoing longwave radiation."""
    return 1.0 + greenhouse_optical_depth(pressure_pa, one_bar_k, exponent)


@dataclass(frozen=True)
class Climate:
    """The model's physical parameters. The first four are the ones the ensemble sweats over."""

    frost_albedo: float = 0.90
    rock_albedo: float = 0.250
    transport: float = 0.004               # D, W m^-2 K^-1, in Budyko's flux form
    greenhouse_one_bar_k: float = GREENHOUSE_AT_ONE_BAR_K
    greenhouse_exponent: float = GREENHOUSE_EXPONENT
    frost_width_k: float = 4.0             # width of the frost/no-frost transition in temperature
    heat_capacity: float = 6.0e5           # J m^-2 K^-1 of the mixed atmosphere + regolith skin
    polar_latitude_deg: float = 70.0

    def to_dict(self) -> dict[str, float | list[float]]:
        """The parameters as plain numbers -- a list per parameter when they are per ensemble member."""
        out: dict[str, float | list[float]] = {}
        for k, v in self.__dict__.items():
            array = np.asarray(v, float)
            out[k] = float(array) if array.ndim == 0 else [float(x) for x in array.ravel()]
        return out

    def varied(self, **kwargs) -> Climate:
        return replace(self, **kwargs)


class ZonalEBM:
    """The insolation, the surface, and the transport, on one grid."""

    def __init__(self, climate: Climate | None = None, grid: Grid | None = None, planet: Mars | None = None):
        self.climate = climate or Climate()
        self.grid = grid or Grid()
        self.planet = planet or MARS
        self.insolation = annual_insolation(self.grid.latitude_deg)
        self._session = None

    # ------------------------------------------------------------------ the pieces of the balance
    def albedo(self, frost) -> np.ndarray:
        """Surface albedo per band: the frost albedo where there is frost, the regolith albedo where there is none."""
        f = np.clip(np.asarray(frost, float), 0.0, 1.0)
        c = self.climate
        return c.rock_albedo + (c.frost_albedo - c.rock_albedo) * f

    def absorbed(self, frost, extra_flux=None) -> np.ndarray:
        """Absorbed shortwave plus any extra forcing (W m^-2), per band."""
        flux = (1.0 - self.albedo(frost)) * self.insolation
        return flux if extra_flux is None else flux + np.asarray(extra_flux, float)

    def greenhouse(self, pressure_pa) -> np.ndarray:
        """``1 + tau`` for the CO2 pressure: the factor that multiplies the outgoing longwave radiation."""
        c = self.climate
        return greenhouse_factor(pressure_pa, one_bar_k=c.greenhouse_one_bar_k, exponent=c.greenhouse_exponent)

    def outgoing(self, temperature_k) -> np.ndarray:
        return STEFAN_BOLTZMANN * np.asarray(temperature_k, float) ** 4

    def divergence(self, temperature_k) -> np.ndarray:
        """``D d/dx[(1 - x^2) dT/dx]`` by finite volume; its area-weighted sum is zero, by construction."""
        t = np.asarray(temperature_k, float)
        g = self.grid
        faces = g.faces()
        d = self.climate.transport
        diff = np.diff(t, axis=-1) / g.dx                       # dT/dx at the interior faces
        flux = d * faces[1:-1] * diff
        padded = np.concatenate([np.zeros(flux.shape[:-1] + (1,)), flux,
                                 np.zeros(flux.shape[:-1] + (1,))], axis=-1)
        return np.diff(padded, axis=-1) / g.dx

    def balance(self, temperature_k, pressure_pa, frost, extra_flux=None) -> np.ndarray:
        """``dT/dt * C``: everything on the right-hand side of the energy balance, in W m^-2."""
        return (self.absorbed(frost, extra_flux) - self.outgoing(temperature_k) / self.greenhouse(pressure_pa)
                + self.divergence(temperature_k))

    def global_mean(self, values) -> np.ndarray:
        """The area-weighted mean over the sphere of a per-band quantity."""
        return np.sum(np.asarray(values, float) * self.grid.weights, axis=-1)

    # ------------------------------------------------------------------ integration
    def settle(self, pressure_pa, frost, extra_flux=None, years: float = 40.0, steps_per_year: int = 24,
               start: float | None = None, members: int = 1) -> np.ndarray:
        """Integrate the thermal part to equilibrium with the frost field held fixed, and return the profile.

        ``members`` broadcasts a per-member :class:`Climate` (its physical fields shaped ``(members, 1)``): every
        member is settled with its own parameters, from the same profile and the same frost field.
        """
        start_k = 210.0 if start is None else float(start)
        temperature = np.full((members, self.grid.bands), start_k)
        frost = np.asarray(frost, float)
        frost = np.broadcast_to(frost[None, :] if frost.ndim == 1 else frost, temperature.shape)
        extra = None if extra_flux is None else np.broadcast_to(np.asarray(extra_flux, float),
                                                               temperature.shape)
        out = self.march(temperature, years, steps_per_year, pressure_pa, frost, extra)
        return out[-1]

    def advance(self, temperature, dt, pressure_pa, frost, extra_flux=None) -> np.ndarray:
        """One time step of the energy balance, of ``dt`` seconds, on a ``(n_members, n_bands)`` state."""
        capacity = self.climate.heat_capacity
        pressure = np.asarray(pressure_pa, float)
        if pressure.ndim == 1 and pressure.size > 1:           # a pressure per member, not per band
            pressure = pressure[:, None]
        return self._step(np.asarray(temperature, float), dt, pressure,
                          np.asarray(frost, float), None if extra_flux is None else np.asarray(extra_flux, float),
                          capacity)

    def march(self, temperature, years: float, steps_per_year: int, pressure_pa, frost,
              extra_flux=None) -> np.ndarray:
        """Integrate ``(n_members, n_bands)`` temperatures for ``years`` and return every step (including the start).

        Explicit in the radiation, implicit in the diffusion: the diffusion operator is stiff on a coarse latitude
        grid (``C / D ~ 3000 s`` at the equator) and would otherwise force a time step of under an hour.
        """
        t = np.array(temperature, float, ndmin=2, copy=True)
        dt = 365.25 * 86_400.0 / steps_per_year
        frost = np.asarray(frost, float)
        frost = frost[None, :] if frost.ndim == 1 else frost
        extra = None if extra_flux is None else np.asarray(extra_flux, float)[None, :]
        p = np.asarray(pressure_pa, float)
        p = p[:, None] if p.ndim == 1 else p
        steps = max(1, int(round(years * steps_per_year)))
        history = [t.copy()]
        for _ in range(steps):
            t = self._step(t, dt, p, frost, extra, self.climate.heat_capacity)
            history.append(t.copy())
        return np.stack(history)

    def _step(self, t, dt, pressure, frost, extra, capacity) -> np.ndarray:
        g = self.grid
        coefficient = self.climate.transport
        faces = g.faces()
        inner = faces[1:-1]
        sub = -coefficient * inner / g.dx ** 2                  # T_{i-1}
        sup = -coefficient * inner / g.dx ** 2                  # T_{i+1}
        absorption = coefficient * (faces[:-1] + faces[1:]) / g.dx ** 2
        factor = self.greenhouse(pressure)
        four_sigma = 4.0 * STEFAN_BOLTZMANN * t ** 3 / factor
        main = capacity / dt + four_sigma + absorption
        source = (capacity / dt * t + self.absorbed(frost, extra)
                  + 3.0 * STEFAN_BOLTZMANN * t ** 4 / factor)
        return _tridiagonal(sub, main, sup, source)




def _tridiagonal(sub, main, super_diag, right) -> np.ndarray:
    """Solve a tridiagonal system with several right-hand sides, band by band (Thomas, vectorised over the rows).

    ``sub``, ``main``, ``super_diag`` are ``(bands,)`` or ``(rows, bands)``; ``right`` is ``(rows, bands)``. The
    diagonals may come one shorter than ``right``, which is how a divergence produces them; the missing end is a
    zero, because the poleward flux dies at the poles.
    """
    right = np.asarray(right, float)
    n = right.shape[-1]
    main = np.broadcast_to(np.asarray(main, float), right.shape)
    sub = np.asarray(sub, float)
    sup = np.asarray(super_diag, float)
    if sub.shape[-1] == n - 1:
        sub = np.concatenate([np.zeros(sub.shape[:-1] + (1,)), sub], axis=-1)
    if sup.shape[-1] == n - 1:
        sup = np.concatenate([sup, np.zeros(sup.shape[:-1] + (1,))], axis=-1)
    sub = np.broadcast_to(sub, right.shape)
    sup = np.broadcast_to(sup, right.shape)
    c = np.empty_like(right)
    d = np.empty_like(right)
    c[..., 0] = sup[..., 0] / main[..., 0]
    d[..., 0] = right[..., 0] / main[..., 0]
    for i in range(1, n):
        denominator = main[..., i] - sub[..., i] * c[..., i - 1]
        c[..., i] = sup[..., i] / denominator
        d[..., i] = (right[..., i] - sub[..., i] * d[..., i - 1]) / denominator
    out = np.empty_like(right)
    out[..., -1] = d[..., -1]
    for i in range(n - 2, -1, -1):
        out[..., i] = d[..., i] - c[..., i] * out[..., i + 1]
    return out

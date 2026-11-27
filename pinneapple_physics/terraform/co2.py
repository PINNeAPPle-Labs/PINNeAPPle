"""The CO2 cycle: the frost point, the reservoirs, and the zonal cap that couples them to the temperature (#374).

**The frost point is the whole coupling.** CO2 ice and CO2 gas meet at one temperature for a given pressure, from
Clausius-Clapeyron with the triple point and the enthalpy of sublimation of :mod:`.mars`:

    ln(p / p_t) = -(dH_sub / R) (1/T - 1/T_t)

so the temperature at which the cap survives *is* a function of the pressure the cap itself sets. That is the
thermostat of Leighton & Murray, and it is why warming Mars by greenhouse gases does not simply melt the caps: from
6 mbar to 20 mbar the frost point rises by 8.5 K while the CO2 greenhouse adds about 4 K, so the ice stays. Melting
it takes a forcing aimed at the cap -- which is exactly the method Jakosky & Edwards 2018 describe (raise dust onto
the cap to lower its albedo).

**The reservoirs are published, not chosen.** Every number below is in equivalent global-mean surface pressure, the
unit Jakosky & Edwards use, with the page of the paper it comes from. Nothing here is a free parameter: the free
parameters of the model are the *fractions* of these reservoirs a scenario manages to mobilize, and those live in
:mod:`.terraforming`.

The model reproduces two published anchors without being tuned to them: the present 6.36 mbar sits at a frost point
of 146 K, which is the temperature the seasonal cap is observed to form at, and mobilizing the polar CO2 takes the
total to 15.4 mbar, which is Jakosky & Edwards' "less than 15 mbar total".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mars import JAKOSKY_EDWARDS_2018, MARS

__all__ = ["R_GAS", "Reservoir", "RESERVOIRS", "reservoir", "saturation_pressure", "frost_point", "frost_temperature",
           "INVENTORIES", "inventory", "FrostCap"]

#: The gas constant, J/mol/K. Standard.
R_GAS = 8.314462618


def saturation_pressure(temperature_k, planet=None) -> np.ndarray:
    """CO2 vapour pressure (Pa) over CO2 ice, from Clausius-Clapeyron through the triple point.

    ``p = p_t exp(-(dH/R)(1/T - 1/T_t))`` with constant ``dH`` and an ideal gas. It is exact at the triple point by
    construction, and it reproduces the measured 1 atm sublimation point (194.7 K, ``1.013e5 Pa``) to ``6 %`` -- the
    test says so -- which is the accuracy this model needs.
    """
    p = planet or MARS
    t = np.asarray(temperature_k, float)
    exponent = -(p.co2_sublimation_enthalpy / R_GAS) * (1.0 / np.clip(t, 1.0, None) - 1.0 / p.co2_triple_point_k)
    return p.co2_triple_point_pa * np.exp(np.clip(exponent, -700.0, 700.0))


def frost_point(pressure_pa, planet=None) -> np.ndarray:
    """:func:`saturation_pressure` inverted: the temperature (K) at which CO2 ice is in equilibrium with ``p``."""
    planet = planet or MARS
    p = np.asarray(pressure_pa, float)
    logarithm = np.log(np.clip(p, 1e-12, None) / planet.co2_triple_point_pa)
    inverse = 1.0 / planet.co2_triple_point_k - (R_GAS / planet.co2_sublimation_enthalpy) * logarithm
    return 1.0 / inverse


#: Alias kept for callers that read better as "what is the frost temperature of the cap".
frost_temperature = frost_point


@dataclass(frozen=True)
class Reservoir:
    """One published reservoir of CO2, as the surface pressure it would add if it were all gas."""

    key: str
    name: str
    pressure_pa: float
    low_pa: float
    source: str
    note: str
    #: whether a scenario can plausibly reach it with present-day technology (Jakosky & Edwards' "accessible")
    accessible: bool = True

    @property
    def mbar(self) -> float:
        return self.pressure_pa / 100.0

    @property
    def column_kg_m2(self) -> float:
        return MARS.column_from_pressure(self.pressure_pa)

    @property
    def g_cm2(self) -> float:
        """The same reservoir as a column in g cm^-2, the unit Jakosky & Edwards use for the regolith."""
        return self.column_kg_m2 / 10.0

    def to_dict(self) -> dict[str, object]:
        return {"key": self.key, "name": self.name, "pressure_pa": self.pressure_pa, "mbar": self.mbar,
                "g_cm2": self.g_cm2, "low_pa": self.low_pa, "accessible": self.accessible, "source": self.source,
                "note": self.note}


#: The reservoirs of Jakosky & Edwards 2018, page by page. ``pressure_pa`` is their central or upper value, ``low_pa``
#: the lower end where the paper gives a range.
RESERVOIRS: dict[str, Reservoir] = {
    "atmosphere": Reservoir(
        "atmosphere", "the present atmosphere", 636.0, 636.0,
        "NASA Mars Fact Sheet: 6.36 mbar at the mean radius; Jakosky & Edwards 2018: 'about 6 mbar'",
        "the starting point of every scenario, not something a scenario can mobilize"),
    "polar_cap": Reservoir(
        "polar_cap", "CO2 ice in the polar deposits", 900.0, 600.0, JAKOSKY_EDWARDS_2018,
        "'if the whole volume of polar-cap CO2 were emplaced into the atmosphere, it would increase the pressure "
        "to less than 15 mbar total' -- 15.4 mbar total here, 9.0 mbar of it the cap"),
    "seasonal_cap": Reservoir(
        "seasonal_cap", "the seasonal frost", 200.0, 200.0, JAKOSKY_EDWARDS_2018,
        "'as much as perhaps a third of the atmosphere at any given time, or the equivalent of about 2 mbar'; it "
        "returns every year, so it is listed and not counted as a gain"),
    "regolith": Reservoir(
        "regolith", "CO2 adsorbed in the regolith", 4000.0, 0.0, JAKOSKY_EDWARDS_2018,
        "'100 g CO2 cm-2 adsorbed within the subsurface ... would provide the equivalent pressure of about 40 mbar' "
        "-- an upper bound that assumes all 100 m of weathered material releases everything"),
    "clathrate": Reservoir(
        "clathrate", "CO2 clathrate in the polar deposits", 15_000.0, 0.0, JAKOSKY_EDWARDS_2018,
        "'up to the equivalent of about 150 mbar ... 150 mbar should be considered as an extreme upper limit, and "
        "it's possible that there is no clathrate present at all'"),
    "carbonate": Reservoir(
        "carbonate", "near-surface carbonates", 1200.0, 25.0, JAKOSKY_EDWARDS_2018,
        "'we take 12 mbar as the maximum plausible amount of CO2 that could be released from carbonates'; their "
        "Nili Fossae scenarios span 0.25 to 150 mbar"),
    "carbonate_deep": Reservoir(
        "carbonate_deep", "'deep' carbonates in the crust", 100_000.0, 100_000.0, JAKOSKY_EDWARDS_2018,
        "'~1 bar' in their figure 3, listed as an inaccessible reservoir and 'not considered a legitimate source'; "
        "releasing it is planet-scale strip mining",
        accessible=False),
}

#: Jakosky & Edwards' "representative maximum atmospheric pressure that could be achieved": 20 mbar.
MOBILIZABLE_PA = 2000.0

#: In inventories that are run against the model, by name, in equivalent pressure.
INVENTORIES: dict[str, dict[str, float]] = {
    "today": {"atmosphere": 636.0},
    "mobilizable": {"atmosphere": 636.0, "polar_cap": 900.0, "regolith": 464.0},
    "accessible_top": {"atmosphere": 636.0, "polar_cap": 900.0, "regolith": 4000.0, "carbonate": 1200.0},
    "everything": {"atmosphere": 636.0, "polar_cap": 900.0, "regolith": 4000.0, "carbonate": 1200.0,
                   "clathrate": 15_000.0},
    "deep_carbonates": {"atmosphere": 636.0, "polar_cap": 900.0, "regolith": 4000.0, "carbonate": 1200.0,
                        "clathrate": 15_000.0, "carbonate_deep": 100_000.0},
}


def reservoir(key: str) -> Reservoir:
    """One reservoir from :data:`RESERVOIRS`."""
    if key not in RESERVOIRS:
        raise KeyError(f"{key!r} is not a reservoir; known: {sorted(RESERVOIRS)}")
    return RESERVOIRS[key]


def inventory(name: str) -> float:
    """The total CO2 of a named inventory, in Pa of equivalent surface pressure."""
    if name not in INVENTORIES:
        raise KeyError(f"{name!r} is not an inventory; known: {sorted(INVENTORIES)}")
    return float(sum(INVENTORIES[name].values()))


@dataclass
class FrostCap:
    """The polar CO2 cap on the model's latitude grid: how much ice each band holds, and how much it holds full.

    The cap is a zonal film, not a slab: each band holds ``capacity`` metres of CO2 ice when its surface is at or
    below the frost point and nothing when it is above, with ``width_k`` of smoothing in between. The capacities are
    normalised so that the two polar caps, fully frosted, hold exactly :data:`RESERVOIRS` ``['polar_cap']`` --
    Jakosky & Edwards' 9 mbar -- which is the only way the published reservoir enters the geometry.

    ``capacity_pa`` is in equivalent surface pressure, so the mass in a band is ``capacity * area_fraction`` and the
    sum is directly comparable with the atmosphere.
    """

    latitude_deg: np.ndarray
    weights: np.ndarray
    capacity_pa: np.ndarray
    polar_latitude_deg: float = 70.0
    width_k: float = 4.0

    @classmethod
    def on_grid(cls, latitude_deg, weights, total_pa: float = None, polar_latitude_deg: float = 70.0,
                width_k: float = 4.0) -> FrostCap:
        """Capacities proportional to the area of the bands poleward of ``polar_latitude_deg``, summing to
        ``total_pa`` (the polar reservoir); every other band holds none."""
        lat = np.asarray(latitude_deg, float)
        w = np.asarray(weights, float)
        polar = np.abs(lat) >= polar_latitude_deg
        if not polar.any():
            raise ValueError(f"no band lies poleward of {polar_latitude_deg} deg on this grid")
        capacity = np.where(polar, w, 0.0)
        capacity = capacity / capacity.sum() * (RESERVOIRS["polar_cap"].pressure_pa if total_pa is None else total_pa)
        return cls(latitude_deg=lat, weights=w, capacity_pa=capacity, polar_latitude_deg=polar_latitude_deg,
                   width_k=width_k)

    def equilibrium_fraction(self, temperature_k, pressure_pa) -> np.ndarray:
        """The frost fraction each band would settle at: 1 below the frost point, 0 above, smoothed over ``width_k``.

        The smoothed step is what makes the model integrable: a hard step makes the cap jump from band to band and
        the time integration rings.
        """
        t = np.asarray(temperature_k, float)
        frost = frost_point(pressure_pa)
        return np.clip(0.5 + (frost - t) / (2.0 * self.width_k), 0.0, 1.0)

    def mass_pa(self, fraction) -> float:
        """The whole cap as an equivalent global surface pressure (Pa): ``sum capacity * fraction``.

        The weights sum to 1 over the sphere, and the capacities were normalised with them, so a fraction of 1 in
        every band returns the reservoir.
        """
        return float(np.sum(self.capacity_pa * np.asarray(fraction, float)))

    def cover(self, fraction) -> float:
        """Fraction of the polar *area* that is frosted."""
        polar = self.weights * (self.capacity_pa > 0.0)
        return float(np.sum(polar * np.asarray(fraction, float)) / max(np.sum(polar), 1e-30))

    def to_dict(self) -> dict[str, object]:
        return {"polar_latitude_deg": self.polar_latitude_deg, "width_k": self.width_k,
                "capacity_pa": float(self.capacity_pa.sum())}

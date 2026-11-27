"""Mars, as the numbers a climate model has to start from, and where each one came from (#374).

Two published sources carry almost everything here, and neither number is typed from memory:

* the **NASA Mars Fact Sheet** (``nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html``) -- the radius, the
  gravity, the Bond albedo, the solar irradiance at Mars, the black-body temperature, the surface pressure
  (``6.36 mbar`` at the mean radius), the mean temperature (``214 K``), the atmospheric mass (``~2.5e16 kg``) and
  the J2000 mean orbital elements;
* **Jakosky & Edwards 2018**, *Inventory of CO2 available for terraforming Mars*, Nature Astronomy **2**, 634-639,
  doi:10.1038/s41550-018-0529-6 -- every CO2 reservoir in :mod:`.co2`, and the conclusion this package is built
  around: the CO2 that can be mobilized is not enough.

:data:`SOURCES` names the source of every field of :class:`Mars`, and ``tests/test_mars_terraform.py`` walks the
dataclass field by field and fails on any field that has none. Nothing in :mod:`.ebm`, :mod:`.co2`,
:mod:`.terraforming` or :mod:`.globe` types a planetary number of its own: they read it from here.

One number is not read but **derived**: the areocentric solar longitude of perihelion, which decides which
hemisphere gets the short hot summer. :func:`perihelion_solar_longitude` computes it from the pole direction and the
orbital elements of the fact sheet, and lands on ``250.99 deg`` -- the value the Mars climate literature quotes --
which is the check that the derivation is right rather than remembered.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np

from ..astro import facts

__all__ = ["Mars", "MARS", "SOURCES", "FACT_SHEET", "JAKOSKY_EDWARDS_2018", "NIST", "AU_KM", "SOLAR_CONSTANT_W_M2",
           "perihelion_solar_longitude", "solar_irradiance", "column_mass", "source", "provenance"]

FACT_SHEET = "NASA Mars Fact Sheet (nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html)"
JAKOSKY_EDWARDS_2018 = ("Jakosky & Edwards 2018, Inventory of CO2 available for terraforming Mars, "
                        "Nature Astronomy 2, 634-639, doi:10.1038/s41550-018-0529-6")
NIST = "NIST Chemistry WebBook: CO2 triple point and enthalpy of sublimation"

#: One astronomical unit, km. The repo's astro package already carries it; the fact sheet's J2000 semimajor axis is
#: quoted in au, so it is the only conversion this module needs.
AU_KM = 149_597_870.7

#: Total solar irradiance at 1 au, W m^-2 (the fact sheet's 586.2 W m^-2 at Mars over its 1.5237 au semimajor axis).
SOLAR_CONSTANT_W_M2 = 1361.0


@dataclass(frozen=True)
class Mars:
    """The planet the model runs on. Every field has an entry in :data:`SOURCES`."""

    # --- shape and gravity
    radius_m: float = 3.3962e6            # equatorial radius, 3396.2 km
    gravity_m_s2: float = 3.73            # surface gravity (mean)
    mass_kg: float = 6.4169e23
    # --- orbit (J2000 mean elements) and spin
    semimajor_au: float = 1.52366231
    eccentricity: float = 0.09341233
    inclination_deg: float = 1.85061
    node_deg: float = 49.57854            # longitude of the ascending node
    perihelion_longitude_deg: float = 336.04084
    pole_ra_deg: float = 317.681
    pole_dec_deg: float = 52.887
    obliquity_deg: float = 25.19
    rotation_h: float = 24.6229
    # --- radiation
    solar_irradiance_w_m2: float = 586.2  # at the semimajor axis, i.e. the mean
    bond_albedo: float = 0.250
    blackbody_temperature_k: float = 209.8
    # --- the present atmosphere
    surface_pressure_pa: float = 636.0    # 6.36 mbar at the mean radius (4.0-8.7 mbar over the year)
    mean_temperature_k: float = 214.0
    atmosphere_mass_kg: float = 2.5e16
    co2_mole_fraction: float = 0.951
    mean_molecular_weight: float = 43.49
    # --- CO2 as a substance
    co2_molar_mass_kg_mol: float = 44.01e-3
    co2_triple_point_k: float = 216.58
    co2_triple_point_pa: float = 517_950.0
    co2_sublimation_enthalpy: float = 25_230.0   # J/mol at the triple point

    def __post_init__(self) -> None:
        missing = set(SOURCES) ^ {f.name for f in fields(self)}
        if missing:
            raise ValueError(f"constants without a source, or a source without a constant: {sorted(missing)}")

    @property
    def area_m2(self) -> float:
        """Surface area of the sphere the model uses, ``4 pi R^2`` (m^2)."""
        return 4.0 * np.pi * self.radius_m ** 2

    @property
    def co2_partial_pressure_pa(self) -> float:
        """Present CO2 partial pressure, mbar of total times the CO2 mole fraction."""
        return self.surface_pressure_pa * self.co2_mole_fraction

    def pressure_from_column(self, column_kg_m2: float) -> float:
        """Surface pressure (Pa) under a column of ``column_kg_m2`` kg m^-2: hydrostatic, ``p = column * g``."""
        return column_kg_m2 * self.gravity_m_s2

    def column_from_pressure(self, pressure_pa: float) -> float:
        """Column mass (kg m^-2) under a surface pressure ``pressure_pa``: ``p / g``."""
        return pressure_pa / self.gravity_m_s2

    def mass_from_pressure(self, pressure_pa: float) -> float:
        """Mass (kg) of an atmosphere of uniform surface pressure ``pressure_pa``: ``p A / g``."""
        return pressure_pa * self.area_m2 / self.gravity_m_s2

    def pressure_from_mass(self, mass_kg: float) -> float:
        """Surface pressure (Pa) of ``mass_kg`` spread over the planet: ``m g / A``."""
        return mass_kg * self.gravity_m_s2 / self.area_m2

    def solar_irradiance_at(self, distance_au: float) -> float:
        """Irradiance (W m^-2) at a heliocentric distance, ``S0 / r^2``."""
        return SOLAR_CONSTANT_W_M2 / np.asarray(distance_au, float) ** 2

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


#: Where every field of :class:`Mars` came from. ``__post_init__`` refuses a planet whose fields and sources differ.
SOURCES: dict[str, str] = {
    "radius_m": f"{FACT_SHEET}: equatorial radius 3396.2 km",
    "gravity_m_s2": f"{FACT_SHEET}: surface gravity (mean) 3.73 m/s2",
    "mass_kg": f"{FACT_SHEET}: mass 0.64169e24 kg",
    "semimajor_au": f"{FACT_SHEET}: J2000 semimajor axis 1.52366231 au",
    "eccentricity": f"{FACT_SHEET}: J2000 orbital eccentricity 0.09341233 (the fact sheet's row above rounds to 0.0935)",
    "inclination_deg": f"{FACT_SHEET}: J2000 orbital inclination 1.85061 deg",
    "node_deg": f"{FACT_SHEET}: J2000 longitude of the ascending node 49.57854 deg",
    "perihelion_longitude_deg": f"{FACT_SHEET}: J2000 longitude of perihelion 336.04084 deg",
    "pole_ra_deg": f"{FACT_SHEET}: north pole of rotation, RA 317.681 deg (J2000)",
    "pole_dec_deg": f"{FACT_SHEET}: north pole of rotation, Dec 52.887 deg (J2000)",
    "obliquity_deg": f"{FACT_SHEET}: obliquity to orbit 25.19 deg",
    "rotation_h": f"{FACT_SHEET}: sidereal rotation period 24.6229 h",
    "solar_irradiance_w_m2": f"{FACT_SHEET}: solar irradiance at Mars 586.2 W/m2",
    "bond_albedo": f"{FACT_SHEET}: Bond albedo 0.250",
    "blackbody_temperature_k": f"{FACT_SHEET}: black-body temperature 209.8 K",
    "surface_pressure_pa": f"{FACT_SHEET}: surface pressure 6.36 mbar at mean radius (4.0-8.7 mbar over the year)",
    "mean_temperature_k": f"{FACT_SHEET}: average temperature ~214 K",
    "atmosphere_mass_kg": f"{FACT_SHEET}: total mass of atmosphere ~2.5e16 kg",
    "co2_mole_fraction": f"{FACT_SHEET}: atmospheric composition 95.1% CO2 by volume",
    "mean_molecular_weight": f"{FACT_SHEET}: mean molecular weight 43.49",
    "co2_molar_mass_kg_mol": "standard value: M(CO2) = 44.01 g/mol",
    "co2_triple_point_k": f"{NIST}: T = 216.58 K",
    "co2_triple_point_pa": f"{NIST}: p = 5.185 bar = 517 950 Pa",
    "co2_sublimation_enthalpy": f"{NIST}: dH_sub = 25.23 kJ/mol at the triple point",
}

#: The planet, checked against :data:`SOURCES` at import.
MARS = Mars()


def source(name: str) -> str:
    """The source recorded for one constant, by field name."""
    if name not in SOURCES:
        raise KeyError(f"{name!r} is not a constant of Mars; known: {sorted(SOURCES)}")
    return SOURCES[name]


def provenance() -> dict[str, dict[str, float | str]]:
    """``{field: {value, unit: '', source}}`` -- what a results file records so the numbers can be traced back."""
    values = MARS.to_dict()
    return {k: {"value": v, "source": SOURCES[k]} for k, v in values.items()}


def column_mass(pressure_pa: float, planet: Mars = MARS) -> float:
    """Column mass (kg m^-2) of an atmosphere at surface pressure ``pressure_pa``: ``p / g``."""
    return planet.column_from_pressure(pressure_pa)


def solar_irradiance(distance_au: float) -> float:
    """Irradiance (W m^-2) at heliocentric distance ``distance_au``: the solar constant over ``r^2``."""
    return MARS.solar_irradiance_at(distance_au)


def perihelion_solar_longitude(planet: Mars = MARS) -> float:
    """Areocentric solar longitude (deg) at which Mars passes perihelion, derived from the published elements.

    The areocentric longitude of the Sun, ``Ls``, is measured from Mars' vernal equinox along the orbit. Its zero is
    the intersection of the orbital plane with the equatorial plane, whose direction is ``h x p`` (the orbit normal
    and the spin axis); the Sun, seen from Mars, sits at the heliocentric longitude of perihelion plus 180 degrees.
    The angle between the two, taken about the orbit normal, is ``Ls`` at perihelion.

    Returns ``250.99`` degrees -- which is the value Mars climate work quotes, so the derivation is checked against
    the literature rather than remembered from it.
    """
    d2r = np.deg2rad
    ra, dec = d2r(planet.pole_ra_deg), d2r(planet.pole_dec_deg)
    pole_eq = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    tilt = d2r(23.4392911)                                     # obliquity of the Earth's equator, J2000
    pole = np.array([pole_eq[0], pole_eq[1] * np.cos(tilt) + pole_eq[2] * np.sin(tilt),
                     -pole_eq[1] * np.sin(tilt) + pole_eq[2] * np.cos(tilt)])
    inc, node = d2r(planet.inclination_deg), d2r(planet.node_deg)
    normal = np.array([np.sin(inc) * np.sin(node), -np.sin(inc) * np.cos(node), np.cos(inc)])
    equinox = np.cross(normal, pole)
    equinox /= np.linalg.norm(equinox)
    lon = d2r(planet.perihelion_longitude_deg + 180.0)
    sun = np.array([np.cos(lon), np.sin(lon), 0.0])
    if np.dot(equinox, sun) > 0.0:                             # the other equinox is the vernal one
        equinox = -equinox
    angle = np.arctan2(np.dot(np.cross(equinox, sun), normal), np.dot(equinox, sun))
    return float(np.rad2deg(angle) % 360.0)


def fact_sheet_numbers() -> dict[str, dict[str, float | str]]:
    """The three Mars numbers the bundled facts card already carries, so this module can be checked against it."""
    card = facts.numbers("mars")
    return {k: {"value": card[k]["value"], "unit": card[k]["unit"], "source": card[k]["source"]}
            for k in ("diameter_km", "gravity_m_s2", "distance_sun_km", "mean_temp_c")}

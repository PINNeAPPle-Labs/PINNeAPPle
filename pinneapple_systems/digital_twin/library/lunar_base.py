"""A lunar base: solar power through the night, a thermal balance, and the life-support loop.

What is modelled
----------------
A crewed outpost at the lunar equator, sized for one thing above all: surviving the night.
The Moon turns once in a synodic month, so the night is fourteen and three-quarter Earth days
long, and everything here follows from that number. The array area comes from the energy the
day has to hand over to the storage; the storage comes from the load the night has to carry;
the radiator comes from what the habitat has to reject; the oxygen comes from the crew and
from the regolith the crew can process.

The conservation laws it checks
-------------------------------
``energy``  what the arrays deliver over a lunar day against what the load, the storage losses
            and the night discharge take out
``oxygen``  what the regolith plant produces in a lunar day against what the crew breathes
``water``   what the recycling loop returns plus the makeup against what the crew consumes

The checks are the invariants a sizing study must respect: the storage never goes negative,
the arrays do not have to carry the night, the radiator area is finite and positive, and the
night length agrees with the synodic month it was derived from.

Limits, stated rather than discovered
-------------------------------------
Constant solar constant and a flat equatorial site: no horizon, no dust deposition, no solar
incidence beyond a cosine, no eclipse by terrain. The thermal model is a single radiative
node — no conduction into the regolith, no multilayer insulation detail, no radiator
degradation. Oxygen from regolith is stoichiometric yield only: no reactor efficiency, no
power demand of the plant. This is a sizing model for a trade study, not a design.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from .base import Check, Parameter, ParameterError, Quantity, ReferenceModel, TwinRun, close

__all__ = ["LunarBase", "LUNAR_PARAMETERS", "night_factor", "regolith_oxygen_yield"]

#: Mean synodic month, the lunar day-night cycle an outpost lives on (days).
SYNODIC_MONTH_DAYS = 29.530_589
#: Solar irradiance at one astronomical unit (W/m2) — the value the whole power budget scales with.
SOLAR_CONSTANT = 1361.0
#: Stefan-Boltzmann constant (W/m2/K4).
SIGMA = 5.670_374_419e-8
#: Oxygen a person consumes per day (kg), the usual ECLSS sizing figure.
O2_PER_PERSON_DAY = 0.84
#: Water a person uses per day before recycling (kg).
WATER_PER_PERSON_DAY = 3.6


def night_factor() -> tuple[float, float]:
    """Day and night length of the equatorial Moon (h), from the synodic month."""
    lunar_day = SYNODIC_MONTH_DAYS * 24.0
    return 0.5 * lunar_day, 0.5 * lunar_day


def regolith_oxygen_yield(oxygen_rich_fraction: float) -> float:
    """kg of oxygen extractable per kg of regolith, from the FeO it carries.

    The route assumed is the one every lunar oxygen study starts from: reduce the ferrous iron
    (FeO -> Fe + 1/2 O2) and take the oxygen. Per kilogram of a regolith whose FeO mass
    fraction is ``f``, the oxygen is ``f * M(O) / M(FeO)`` — 0.2226 kg O2 per kg FeO, so a
    regolith with 15 % FeO gives 3.3 % of its mass as oxygen. Ilmenite and pyroclastic deposits
    do better; this is the conservative number a sizing study should use.
    """
    m_o, m_feo = 15.999, 71.844
    return oxygen_rich_fraction * m_o / m_feo


LUNAR_PARAMETERS = {
    "crew": Parameter("crew", "-", 4.0, 1.0, 12.0, "people on site"),
    "array_area": Parameter("array_area", "m2", 1_200.0, 50.0, 20_000.0, "solar array area (one side)"),
    "array_efficiency": Parameter("array_efficiency", "-", 0.28, 0.05, 0.45,
                                  "module efficiency at the operating temperature given"),
    "array_degradation": Parameter("array_degradation", "-", 0.97, 0.5, 1.0,
                                   "dust, wiring and mismatch losses as one factor"),
    "load_day": Parameter("load_day", "kW", 45.0, 1.0, 5_000.0, "average electrical load during the day"),
    "load_night": Parameter("load_night", "kW", 28.0, 1.0, 5_000.0, "average electrical load during the night"),
    "storage_round_trip": Parameter("storage_round_trip", "-", 0.85, 0.3, 1.0,
                                    "regenerative fuel cell round-trip efficiency"),
    "habitat_internal_heat": Parameter("habitat_internal_heat", "kW", 12.0, 0.1, 500.0,
                                       "crew, lighting and equipment heat inside the habitat"),
    "habitat_solar_absorbed": Parameter("habitat_solar_absorbed", "kW", 8.0, 0.0, 500.0,
                                        "solar heat the habitat surface absorbs at noon"),
    "radiator_temperature": Parameter("radiator_temperature", "K", 320.0, 250.0, 400.0,
                                      "radiator surface temperature"),
    "radiator_emissivity": Parameter("radiator_emissivity", "-", 0.9, 0.2, 1.0, "radiator emissivity"),
    "regolith_oxygen_fraction": Parameter("regolith_oxygen_fraction", "-", 0.15, 0.02, 0.40,
                                          "FeO mass fraction of the regolith the plant processes"),
    "regolith_plant_uptime": Parameter("regolith_plant_uptime", "-", 0.75, 0.1, 1.0,
                                       "fraction of the lunar day the oxygen plant runs"),
    "water_recovery": Parameter("water_recovery", "-", 0.93, 0.5, 0.999,
                                "closure of the water recycling loop"),
    "oxygen_storage_days": Parameter("oxygen_storage_days", "d", 30.0, 1.0, 400.0,
                                     "oxygen buffer the base has to hold"),
}


class LunarBase(ReferenceModel):
    """A lunar outpost sized by its night."""

    slug = "lunar_base"
    title = "Lunar base: solar power, night storage, thermal rejection and life support"
    physics = "solar geometry, radiative thermal balance, energy storage and stoichiometric life support"
    parameters = LUNAR_PARAMETERS
    assumptions = (
        "Equatorial site: the Sun sweeps a full half circle per lunar day and the arrays see a cosine.",
        "Constant solar constant, no horizon shading, no dust accumulation over the mission.",
        "Regenerative fuel cells for the night, with one round-trip efficiency for charge and discharge.",
        "One radiative node for the habitat: absorbed solar plus internal heat must leave through the radiators.",
        "Oxygen is extracted by reducing the ferrous iron of the regolith, at its stoichiometric yield.",
    )
    limits = (
        "No lunar horizon, terrain or south-pole illumination geometry — those change the power budget completely.",
        "No conduction into the regolith, no multilayer insulation detail, no radiator view-factor geometry.",
        "The regolith plant's own power demand and reactor efficiency are not modelled.",
        "The thermal model has no transient: it sizes the radiator for the worst case, it does not simulate it.",
    )
    sources = (
        "Mean synodic month 29.530589 days — the lunar day-night cycle the sizing follows.",
        "Solar constant 1361 W/m2 at 1 AU (TSI), the value the array area scales from.",
        "Stefan-Boltzmann sigma = 5.670374419e-8 W/m2/K4 for the radiator.",
        "FeO reduction stoichiometry, FeO -> Fe + 1/2 O2, for the oxygen yield.",
    )
    not_verified = (
        "No measured base data: this is a sizing study on declared assumptions.",
        "Crew consumables use published per-person rates, not a flight-qualified ECLSS model.",
        "Night survival is checked for the mean synodic month; libration and eclipse seasons are not.",
        "Costs, mass of the plant and launch mass are outside the model.",
    )

    def _validate(self, p: Mapping[str, float]) -> None:
        if p["array_efficiency"] * p["array_degradation"] <= 0.0:
            raise ParameterError("the array efficiency and degradation leave no power")
        if p["radiator_temperature"] <= 0.0:
            raise ParameterError("radiator_temperature must be above absolute zero")
        if p["load_night"] > p["load_day"]:
            raise ParameterError("the night load cannot exceed the day load in this model")

    def _run(self, p: Mapping[str, float]) -> TwinRun:
        day_h, night_h = night_factor()

        # -- power --------------------------------------------------------------
        # The Sun crosses the sky once per lunar day; the cosine integrated over the day is
        # 2/pi, so the mean irradiance on a horizontal array is 0.6366 of the solar constant.
        mean_cosine = 2.0 / math.pi
        peak_power = p["array_area"] * SOLAR_CONSTANT * p["array_efficiency"] * p["array_degradation"]
        energy_day = peak_power * mean_cosine * day_h / 1e3                # kWh
        energy_night = p["load_night"] * night_h                           # kWh
        energy_day_load = p["load_day"] * day_h                            # kWh
        storage_required = energy_night / p["storage_round_trip"]          # kWh that must go in
        stored = max(energy_day - energy_day_load, 0.0)                    # kWh the day can put away
        storage_margin = stored - storage_required
        night_fraction = energy_night / max(energy_day, 1e-9)

        # -- thermal ------------------------------------------------------------
        heat_rejected = p["habitat_internal_heat"] + p["habitat_solar_absorbed"]        # kW
        radiator_area = heat_rejected * 1e3 / (
            p["radiator_emissivity"] * SIGMA * p["radiator_temperature"] ** 4)          # m2

        # -- life support -------------------------------------------------------
        oxygen_per_day = p["crew"] * O2_PER_PERSON_DAY                                  # kg/day
        oxygen_per_lunar_day = oxygen_per_day * SYNODIC_MONTH_DAYS
        yield_per_kg = regolith_oxygen_yield(p["regolith_oxygen_fraction"])
        regolith_per_lunar_day = (oxygen_per_lunar_day / yield_per_kg)                  # kg
        # The plant runs a fraction of the time, so what it must process per hour of uptime is
        # the lunar-day demand spread over the hours it is actually running.
        plant_hours = p["regolith_plant_uptime"] * SYNODIC_MONTH_DAYS * 24.0
        regolith_per_plant_hour = regolith_per_lunar_day / max(plant_hours, 1e-9)
        oxygen_produced = regolith_per_lunar_day * yield_per_kg
        oxygen_storage = oxygen_per_day * p["oxygen_storage_days"]

        water_per_day = p["crew"] * WATER_PER_PERSON_DAY
        water_makeup = water_per_day * (1.0 - p["water_recovery"]) * SYNODIC_MONTH_DAYS
        water_returned = water_per_day * p["water_recovery"] * SYNODIC_MONTH_DAYS
        water_consumed = water_per_day * SYNODIC_MONTH_DAYS

        outputs = {
            "lunar_day": Quantity("lunar_day", day_h, "h", "half a synodic month"),
            "lunar_night": Quantity("lunar_night", night_h, "h", "half a synodic month"),
            "peak_array_power": Quantity("peak_array_power", peak_power / 1e3, "kW", "at normal incidence"),
            "energy_per_lunar_day": Quantity("energy_per_lunar_day", energy_day, "kWh",
                                             "cosine-integrated over the day"),
            "energy_night_demand": Quantity("energy_night_demand", energy_night, "kWh", "night load times night"),
            "storage_required": Quantity("storage_required", storage_required, "kWh",
                                         "night demand divided by the round-trip efficiency"),
            "storage_margin": Quantity("storage_margin", storage_margin, "kWh",
                                       "day surplus minus what the night needs; negative means the base browns out"),
            "night_to_day_ratio": Quantity("night_to_day_ratio", night_fraction, "-",
                                           "fraction of the day's generation the night consumes"),
            "radiator_area": Quantity("radiator_area", radiator_area, "m2",
                                      "sigma eps T^4 for the heat the habitat must reject"),
            "oxygen_per_day": Quantity("oxygen_per_day", oxygen_per_day, "kg/d", "crew times the per-person rate"),
            "regolith_per_lunar_day": Quantity("regolith_per_lunar_day", regolith_per_lunar_day, "kg",
                                               "at the stoichiometric FeO yield"),
            "regolith_per_plant_hour": Quantity("regolith_per_plant_hour", regolith_per_plant_hour, "kg/h",
                                                "the throughput the plant has to reach"),
            "oxygen_storage": Quantity("oxygen_storage", oxygen_storage, "kg",
                                       "buffer the base has to hold"),
            "water_makeup": Quantity("water_makeup", water_makeup, "kg",
                                     "what recycling does not close, per lunar day"),
            "water_closure": Quantity("water_closure", p["water_recovery"], "-", "fraction of water recycled"),
        }

        balances = [
            close("energy", energy_day, energy_day_load + min(stored, storage_required) + max(storage_margin, 0.0),
                  "kWh", tolerance=1e-9,
                  detail="generation against day load plus what is put into storage"),
            close("oxygen", oxygen_produced, oxygen_per_lunar_day, "kg",
                  detail="what the regolith plant yields against what the crew breathes"),
            close("water", water_returned + water_makeup, water_consumed, "kg",
                  detail="recycled plus makeup against consumption"),
        ]

        checks = [
            Check("night_from_the_synodic_month",
                  abs(day_h + night_h - SYNODIC_MONTH_DAYS * 24.0) < 1e-9,
                  f"day plus night is {SYNODIC_MONTH_DAYS * 24.0:.2f} h"),
            Check("storage_covers_the_night", storage_margin >= 0.0,
                  f"the day has to put away {storage_required:.0f} kWh and can put away {stored:.0f} kWh"),
            Check("arrays_carry_only_the_day", night_fraction < 1.0,
                  "a night longer than the day's surplus is a storage problem, not an array problem"),
            Check("radiator_area_finite", 0.0 < radiator_area < 1e6,
                  "a radiator that needs a square kilometre is not a radiator, it is a redesign"),
            Check("oxygen_yield_physical", 0.0 < yield_per_kg < 0.5,
                  "no regolith gives up half its mass as oxygen"),
            Check("water_loop_closes_with_makeup", water_makeup >= 0.0,
                  "a recovery above 100 % would be a perpetual motion machine"),
        ]

        run = TwinRun(system=self.slug, title=self.title, parameters=dict(p), outputs=outputs,
                      balances=balances, checks=checks)
        # The power profile over one lunar day, sampled hourly: the shape is the point.
        hours = [h * SYNODIC_MONTH_DAYS * 24.0 / 240.0 for h in range(241)]
        sun = []
        load = []
        for hh in hours:
            phase = (hh / (SYNODIC_MONTH_DAYS * 24.0)) % 1.0
            elevation = math.sin(2.0 * math.pi * phase) if phase < 0.5 else 0.0
            sun.append(max(peak_power * elevation, 0.0) / 1e3)
            load.append(p["load_day"] if phase < 0.5 else p["load_night"])
        run.series = self._series(len(hours), time_h=hours, solar_kw=sun, load_kw=load)
        run.not_verified = list(self.not_verified)
        return run

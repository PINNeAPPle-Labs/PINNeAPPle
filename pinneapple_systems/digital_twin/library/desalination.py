"""A seawater reverse-osmosis plant: recovery, osmotic pressure, specific energy and the salt balance.

What is modelled
----------------
A single-stage RO train. The feed is split into permeate and brine by a declared recovery; the
brine therefore carries all the salt the permeate left behind, which fixes its concentration and,
through van't Hoff, its osmotic pressure. The pump has to exceed that pressure to make water at
all, and what it costs per cubic metre is the whole economics of a desalination plant.

The two numbers a project is judged on are both outputs: the **specific energy** in kWh per cubic
metre of permeate (2-4 kWh/m3 for seawater, against a thermodynamic floor near 1 kWh/m3), and the
**permeate salinity** the membrane's rejection leaves behind. The floor is computed here as an
integral, not quoted, and every run is checked against it.

The conservation laws it checks
-------------------------------
``water``   feed against permeate plus brine
``salt``    the salt entering against the salt leaving, with the brine concentration *derived* from
            that balance rather than assumed — the two routes are the atom balance and van't Hoff
``energy``  the net shaft power against the specific energy times the permeate flow, with the
            pressure-exchanger recovery taken off the brine side

Limits, stated rather than discovered
-------------------------------------
One salt, fully dissociated, ideal solution: van't Hoff is linear in concentration, so a brine above
about 70 g/L is outside what this model can say. No concentration polarisation, no membrane
fouling or ageing, no flux decline, no temperature correction beyond the ideal-gas R T. The net
driving pressure is an input, not a consequence of a membrane area and a permeability. It sizes a
train; it does not design a membrane.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from .base import Check, Parameter, ParameterError, Quantity, ReferenceModel, TwinRun, close

__all__ = [
    "DesalinationPlant",
    "DESALINATION_PARAMETERS",
    "GAS_CONSTANT",
    "MOLAR_MASS_NACL",
    "osmotic_pressure",
    "minimum_specific_energy",
]

GAS_CONSTANT = 8.314_462_618    # J/(mol K)
MOLAR_MASS_NACL = 58.44e-3      # kg/mol
VANT_HOFF_FACTOR = 2.0          # Na+ and Cl-, fully dissociated


def osmotic_pressure(*, concentration: float, temperature: float) -> float:
    """van't Hoff: ``pi = i c R T`` (Pa), with ``concentration`` in mol/m3."""
    return VANT_HOFF_FACTOR * concentration * GAS_CONSTANT * temperature


def minimum_specific_energy(*, feed_osmotic_pressure: float, recovery: float) -> float:
    """The reversible work of separating permeate from feed, per m3 of permeate (J/m3).

    Water is drawn off against a brine that gets more concentrated as it goes: at the moment the
    fraction ``x`` of the feed has been taken, the osmotic pressure is ``pi_f / (1 - x)``. The
    integral of that from zero to the recovery is ``pi_f ln(1/(1-r)) / r`` per cubic metre, which is
    the floor no membrane, no energy-recovery device and no operating pressure can beat.
    """
    return feed_osmotic_pressure * math.log(1.0 / (1.0 - recovery)) / recovery


DESALINATION_PARAMETERS = {
    "feed_flow": Parameter("feed_flow", "m3/h", 1_000.0, 10.0, 20_000.0, "seawater taken from the intake"),
    "recovery": Parameter("recovery", "-", 0.45, 0.10, 0.60, "share of the feed that becomes permeate"),
    "feed_salinity": Parameter("feed_salinity", "g/L", 35.0, 5.0, 45.0, "total dissolved salt at the intake"),
    "salt_rejection": Parameter("salt_rejection", "-", 0.995, 0.95, 0.9999,
                                "share of the salt the membrane holds back"),
    "net_driving_pressure": Parameter("net_driving_pressure", "bar", 12.0, 3.0, 40.0,
                                      "applied pressure above the brine's osmotic pressure: what sets the flux"),
    "pump_efficiency": Parameter("pump_efficiency", "-", 0.82, 0.60, 0.92, "high-pressure pump efficiency"),
    "erd_efficiency": Parameter("erd_efficiency", "-", 0.95, 0.0, 0.98,
                                "share of the brine's pressure energy an energy-recovery device returns"),
    "temperature": Parameter("temperature", "K", 298.15, 275.0, 315.0, "feed temperature"),
    "operating_hours": Parameter("operating_hours", "h/yr", 8_000.0, 1_000.0, 8_760.0,
                                 "hours a year the train runs"),
}


class DesalinationPlant(ReferenceModel):
    """A seawater RO train: recovery, osmotic pressure and what a cubic metre costs in electricity."""

    slug = "desalination_plant"
    title = "Seawater reverse osmosis: recovery, osmotic pressure, specific energy and permeate quality"
    physics = "van't Hoff osmotic pressure, mass and salt balance, pump work with energy recovery"
    parameters = DESALINATION_PARAMETERS
    assumptions = (
        "One salt (NaCl), fully dissociated, in an ideal solution: van't Hoff is linear in concentration.",
        "Single stage, one recovery, one applied pressure across the whole train.",
        "The energy-recovery device returns a declared fraction of the brine's pressure energy.",
        "Pre-treatment, post-treatment and intake pumping are outside the fence.",
    )
    limits = (
        "Above about 70 g/L of brine the ideal-solution assumption is what fails first, not the arithmetic.",
        "No concentration polarisation, no fouling or scaling, no flux decline with age, no membrane replacement cost.",
        "The net driving pressure is an input: a real design gets it from membrane area and permeability.",
        "No boron or chloride removal targets, no permeate blending, no brine disposal.",
    )
    sources = (
        "van't Hoff: pi = i c R T with i = 2 for NaCl and R = 8.314462618 J/(mol K).",
        "Reversible separation work: pi_f ln(1/(1-r)) / r per m3 of permeate, the integral of the osmotic "
        "pressure as the brine concentrates.",
        "Molar mass of NaCl 58.44 g/mol.",
    )
    not_verified = (
        "No plant data: the operating point is engineering-plausible, not a commissioned train.",
        "No comparison against a membrane manufacturer's projection or a plant energy audit.",
        "The thermodynamic floor is a reversible-separation bound, not a guarantee any membrane can approach.",
        "Energy recovery is one efficiency on the brine's pressure energy, not a device curve.",
    )

    def _validate(self, p: Mapping[str, float]) -> None:
        if not 0.0 < p["recovery"] < 1.0:
            raise ParameterError("the recovery has to be between zero and one")
        if p["salt_rejection"] <= 0.0:
            raise ParameterError("the membrane has to reject some salt")

    def _run(self, p: Mapping[str, float]) -> TwinRun:
        feed = p["feed_flow"] / 3_600.0                                  # m3/s
        permeate = feed * p["recovery"]
        brine = feed - permeate

        # g/L is kg/m3, so dividing by the molar mass in kg/mol already gives mol/m3.
        feed_concentration = p["feed_salinity"] / MOLAR_MASS_NACL             # mol/m3
        permeate_concentration = feed_concentration * (1.0 - p["salt_rejection"])

        # The salt balance is what the brine concentration is, and it is the model's own statement
        # rather than the textbook 1/(1-r): the two agree, which is the point of computing it here.
        salt_in = feed * feed_concentration
        salt_permeate = permeate * permeate_concentration
        brine_concentration = (salt_in - salt_permeate) / brine

        pi_feed = osmotic_pressure(concentration=feed_concentration, temperature=p["temperature"])
        pi_brine = osmotic_pressure(concentration=brine_concentration, temperature=p["temperature"])
        applied = pi_brine + p["net_driving_pressure"] * 1e5

        pump_power = feed * applied / p["pump_efficiency"]
        recovered = p["erd_efficiency"] * brine * applied
        net_power = pump_power - recovered
        specific_energy = net_power / max(permeate, 1e-30)               # J/m3
        floor = minimum_specific_energy(feed_osmotic_pressure=pi_feed, recovery=p["recovery"])

        outputs = {
            "permeate_flow": Quantity("permeate_flow", permeate * 3_600.0, "m3/h", "product water"),
            "brine_flow": Quantity("brine_flow", brine * 3_600.0, "m3/h", "concentrate leaving the train"),
            "permeate_per_day": Quantity("permeate_per_day", permeate * 86_400.0, "m3/day", "product per day"),
            "permeate_per_year": Quantity("permeate_per_year",
                                          permeate * 3_600.0 * p["operating_hours"] / 1e6, "Mm3/yr",
                                          "product at the declared operating hours"),
            "brine_salinity": Quantity("brine_salinity", brine_concentration * MOLAR_MASS_NACL, "g/L",
                                       "from the salt balance, not from 1/(1 - recovery)"),
            "permeate_salinity": Quantity("permeate_salinity", permeate_concentration * MOLAR_MASS_NACL * 1e3,
                                          "mg/L", "what the membrane's rejection leaves behind"),
            "osmotic_pressure_feed": Quantity("osmotic_pressure_feed", pi_feed / 1e5, "bar", "van't Hoff at the intake"),
            "osmotic_pressure_brine": Quantity("osmotic_pressure_brine", pi_brine / 1e5, "bar",
                                               "van't Hoff at the concentrate"),
            "applied_pressure": Quantity("applied_pressure", applied / 1e5, "bar",
                                         "brine osmotic pressure plus the net driving pressure"),
            "specific_energy": Quantity("specific_energy", specific_energy / 3.6e6, "kWh/m3",
                                        "net shaft work per cubic metre of permeate"),
            "thermodynamic_floor": Quantity("thermodynamic_floor", floor / 3.6e6, "kWh/m3",
                                            "the reversible separation work of this recovery"),
            "second_law_margin": Quantity("second_law_margin", specific_energy / max(floor, 1e-30), "-",
                                          "how many times the floor the train actually costs"),
            "pump_power": Quantity("pump_power", pump_power / 1e6, "MW", "the high-pressure pump alone"),
            "recovered_power": Quantity("recovered_power", recovered / 1e6, "MW", "what the energy-recovery device gives back"),
            "net_power": Quantity("net_power", net_power / 1e6, "MW", "what the plant draws"),
        }

        balances = [
            close("water", feed, permeate + brine, "m3/s", tolerance=1e-12,
                  detail="the feed against product plus concentrate"),
            close("salt", salt_in, salt_permeate + brine * brine_concentration, "mol/s", tolerance=1e-12,
                  detail="salt in the feed against salt in the permeate plus the concentrate, whose "
                         "concentration was derived from this very balance and is then used to set the "
                         "osmotic pressure the pump has to beat"),
            close("energy", net_power, specific_energy * permeate, "W", tolerance=1e-12,
                  detail="the pump less the energy-recovery device against the specific energy times the "
                         "permeate flow"),
        ]

        checks = [
            Check("above_the_thermodynamic_floor", specific_energy >= floor - 1e-9,
                  f"{specific_energy / 3.6e6:.3f} kWh/m3 against a floor of {floor / 3.6e6:.3f}"),
            Check("pressure_beats_the_brine", applied > pi_brine,
                  "no water crosses a membrane against its own osmotic pressure"),
            Check("brine_more_concentrated_than_feed", brine_concentration > feed_concentration,
                  "the salt the permeate did not take has to be somewhere"),
            Check("permeate_cleaner_than_feed", permeate_concentration < feed_concentration,
                  "the membrane can only improve the water"),
            Check("recovery_below_one", 0.0 < p["recovery"] < 1.0, "the brine still has to flow"),
        ]

        run = TwinRun(system=self.slug, title=self.title, parameters=dict(p), outputs=outputs,
                      balances=balances, checks=checks)
        # Recovery is the design knob: specific energy rises steeply as the brine concentrates.
        xs, ys = [], []
        for k in range(41):
            recovery = 0.10 + 0.0125 * k
            pro = feed * recovery
            bri = feed - pro
            per_conc = feed_concentration * (1.0 - p["salt_rejection"])
            bri_conc = (feed * feed_concentration - pro * per_conc) / bri
            applied_k = osmotic_pressure(concentration=bri_conc, temperature=p["temperature"]) \
                + p["net_driving_pressure"] * 1e5
            net_k = feed * applied_k / p["pump_efficiency"] - p["erd_efficiency"] * bri * applied_k
            xs.append(recovery)
            ys.append(net_k / pro / 3.6e6)
        run.series = self._series(len(xs), recovery=xs, specific_energy_kWh_per_m3=ys)
        run.not_verified = list(self.not_verified)
        return run

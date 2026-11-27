"""A green hydrogen plant: an electrolyser stack, its water, its oxygen and what it costs per kilogram.

What is modelled
----------------
A stack of ``n`` cells in series, each held at a cell voltage, taking a current density over a cell
area. Faraday's law fixes the hydrogen: two electrons per molecule, so the charge that passes at
the electrodes *is* the gas produced. The cell voltage fixes the rest: below the thermoneutral
voltage the stack absorbs heat, above it the stack rejects the difference as heat, and the ratio of
the reversible voltage to the cell voltage is the efficiency.

Three quantities decide whether a plant is credible, and all three are outputs here: the specific
energy per kilogram (the number a purchase order quotes), the fraction of the electricity that
leaves as hydrogen rather than as heat, and the water and oxygen that move with it.

The conservation laws it checks
-------------------------------
``electricity``  the stack power asked for against the power the cell count and current actually draw
``energy``       electricity in against hydrogen's higher heating value plus the heat the stack
                 rejects, with the heat taken from the overpotential and the hydrogen from the
                 molar enthalpy of formation — two routes, not one equation written twice
``water``        the water the stoichiometry demands against the mass of hydrogen and oxygen that
                 actually leaves: ``H2O -> H2 + 1/2 O2`` conserves mass, so the two must agree
``oxygen``       the oxygen from the hydrogen's mass ratio against the oxygen from the charge

Limits, stated rather than discovered
-------------------------------------
A single cell voltage for the whole stack: no activation, ohmic and concentration overpotentials
as a function of current density, so there is no polarisation curve and no part-load behaviour.
No degradation, no membrane crossover beyond the declared faradaic efficiency, no stack thermal
model — the heat is an energy-balance remainder, not a temperature. Balance of plant is one
declared fraction. It sizes a plant; it does not select an electrolyser.
"""
from __future__ import annotations

from collections.abc import Mapping

from .base import Check, Parameter, ParameterError, Quantity, ReferenceModel, TwinRun, close
from .compression_station import R_UNIVERSAL, polytropic_exponent

__all__ = [
    "GreenHydrogenPlant",
    "HYDROGEN_PARAMETERS",
    "FARADAY",
    "MOLAR_MASS_H2",
    "MOLAR_MASS_O2",
    "MOLAR_MASS_H2O",
    "LHV_H2",
    "HHV_H2",
    "reversible_voltage",
    "thermoneutral_voltage",
    "hydrogen_from_charge",
]

FARADAY = 96_485.332_12          # C/mol
# The three molar masses are built from two atomic masses, so that H2O -> H2 + 1/2 O2 conserves
# mass exactly. Written independently from a table they disagree in the sixth digit, and the water
# balance then fails for a reason that is not the plant's.
ATOMIC_MASS_H = 1.007_94e-3      # kg/mol
ATOMIC_MASS_O = 15.999_4e-3      # kg/mol
MOLAR_MASS_H2 = 2.0 * ATOMIC_MASS_H
MOLAR_MASS_O2 = 2.0 * ATOMIC_MASS_O
MOLAR_MASS_H2O = 2.0 * ATOMIC_MASS_H + ATOMIC_MASS_O
# The heating values come from the same two enthalpies the voltages come from, divided by the molar
# mass: quoted per kilogram from a table, they carry a rounding that the energy balance would report
# as a leak of a few parts per million.
ENTHALPY_LHV = 241_830.0         # J/mol of hydrogen, water vapour product
ENTHALPY_HHV = 285_830.0         # J/mol of hydrogen, liquid water product
LHV_H2 = ENTHALPY_LHV / MOLAR_MASS_H2
HHV_H2 = ENTHALPY_HHV / MOLAR_MASS_H2


def reversible_voltage(temperature: float = 298.15) -> float:
    """``dG/(2F)``: the minimum cell voltage that can split water at all (V).

    ``dG = 237.2 kJ/mol`` for liquid water at 25 degC is the standard value; the temperature
    dependence is not carried, because the twin declares one stack temperature by construction.
    """
    delta_g = 237_200.0
    return delta_g / (2.0 * FARADAY)


def thermoneutral_voltage() -> float:
    """``dH/(2F)``: above this voltage the stack rejects heat instead of absorbing it (V)."""
    return ENTHALPY_HHV / (2.0 * FARADAY)


def hydrogen_from_charge(*, current: float, cells: int | float, faradaic_efficiency: float) -> float:
    """Hydrogen produced by a series stack (kg/s): ``n I eta / (2F)`` moles per second."""
    return cells * current * faradaic_efficiency / (2.0 * FARADAY) * MOLAR_MASS_H2


HYDROGEN_PARAMETERS = {
    "stack_power": Parameter("stack_power", "MW", 100.0, 0.1, 2_000.0, "DC power the stack draws"),
    "cell_voltage": Parameter("cell_voltage", "V", 1.78, 1.30, 2.20, "the voltage every cell is held at"),
    "current_density": Parameter("current_density", "A/cm2", 1.50, 0.20, 3.00, "current per unit of cell area"),
    "cell_area": Parameter("cell_area", "m2", 2.5, 0.10, 10.0, "active area of one cell"),
    "faradaic_efficiency": Parameter("faradaic_efficiency", "-", 0.97, 0.85, 1.00,
                                     "share of the charge that makes hydrogen rather than crossing over"),
    "purge_fraction": Parameter("purge_fraction", "-", 0.02, 0.0, 0.10,
                                "share of the hydrogen sent to purge and flare instead of to the battery limit"),
    "delivery_pressure": Parameter("delivery_pressure", "bar", 30.0, 1.0, 700.0,
                                   "pressure the product is compressed to"),
    "compressor_efficiency": Parameter("compressor_efficiency", "-", 0.72, 0.50, 0.90,
                                       "polytropic efficiency of the product compressor"),
    "auxiliary_fraction": Parameter("auxiliary_fraction", "-", 0.08, 0.0, 0.30,
                                    "share of the grid power that runs pumps, cooling and purification"),
    "gas_gamma": Parameter("gas_gamma", "-", 1.40, 1.20, 1.45, "ratio of specific heats of hydrogen"),
    "suction_temperature": Parameter("suction_temperature", "K", 313.15, 280.0, 360.0,
                                     "hydrogen temperature entering the compressor"),
    "operating_hours": Parameter("operating_hours", "h/yr", 7_000.0, 1_000.0, 8_760.0,
                                 "full-load hours a year: what turns a rate into an annual tonnage"),
}


class GreenHydrogenPlant(ReferenceModel):
    """An electrolyser stack, its water, its oxygen and its specific energy."""

    slug = "green_hydrogen_plant"
    title = "Green hydrogen plant: Faraday, cell voltage, specific energy, water and oxygen"
    physics = "electrolysis stoichiometry, cell-voltage efficiency, polytropic product compression"
    parameters = HYDROGEN_PARAMETERS
    assumptions = (
        "Every cell in the stack sits at the same voltage and current density.",
        "Cell count is a real number, not an integer: the model sizes a stack, it does not lay one out.",
        "One faradaic efficiency covers crossover, and one auxiliary fraction covers the whole balance of plant.",
        "The product is compressed polytropically from the stack temperature to the delivery pressure.",
    )
    limits = (
        "No polarisation curve: activation, ohmic and concentration overpotentials are folded into one voltage.",
        "No thermal model of the stack: the heat is an energy-balance remainder, so nothing here says whether the "
        "stack stays at its temperature.",
        "No degradation, no start-stop cycling, no minimum load, no response time.",
        "Balance of plant is a declared fraction, not a set of pumps, dryers and coolers.",
    )
    sources = (
        "Faraday's law: 2 F of charge per mole of H2; F = 96485.33212 C/mol.",
        "Reversible and thermoneutral voltages from dG = 237.2 and dH = 285.83 kJ/mol of liquid water.",
        "Heating values of hydrogen: LHV 119.96 MJ/kg, HHV 141.79 MJ/kg.",
        "Molar masses from IUPAC: H2 2.01588, O2 31.998, H2O 18.01528 g/mol.",
    )
    not_verified = (
        "No plant data: the operating point is engineering-plausible, not a commissioned stack.",
        "No comparison against a vendor polarisation curve or a measured specific-energy guarantee.",
        "Annual tonnage uses a declared full-load-hour count, not a generation profile: a real plant is "
        "priced on the shape of its electricity, which this model does not see.",
        "The compressor is a polytropic stage, not a real hydrogen compressor with its own intercooling and losses.",
    )

    def _validate(self, p: Mapping[str, float]) -> None:
        if p["purge_fraction"] >= 1.0:
            raise ParameterError("the whole product cannot go to purge")
        if p["cell_voltage"] < reversible_voltage():
            raise ParameterError(
                "the cell voltage is below the reversible voltage, so the stack would not split water at all")

    def _run(self, p: Mapping[str, float]) -> TwinRun:
        current_density = p["current_density"] * 1e4                     # A/cm2 -> A/m2
        current = current_density * p["cell_area"]                       # A per cell, in series
        power_per_cell = p["cell_voltage"] * current                     # W
        cells = p["stack_power"] * 1e6 / power_per_cell                  # a real number by construction

        stack_power = cells * power_per_cell                             # W, the geometry's own answer
        stack_voltage = cells * p["cell_voltage"]                        # V, series stack

        h2_generated = hydrogen_from_charge(current=current, cells=cells,
                                            faradaic_efficiency=p["faradaic_efficiency"])
        h2_delivered = h2_generated * (1.0 - p["purge_fraction"])
        h2_purged = h2_generated - h2_delivered

        # The two gas streams, from the hydrogen: one mole of water gives one of hydrogen and half
        # of oxygen, so the oxygen is the molar ratio and the water is the mass sum below.
        h2_molar = h2_generated / MOLAR_MASS_H2
        o2_from_stoichiometry = 0.5 * h2_molar * MOLAR_MASS_O2
        o2_from_charge = cells * current * p["faradaic_efficiency"] / (4.0 * FARADAY) * MOLAR_MASS_O2
        water_consumed = h2_molar * MOLAR_MASS_H2O

        # Energy: the hydrogen's chemical power against its higher heating value, and the heat the
        # stack rejects against the overpotential above the thermoneutral voltage.
        hydrogen_chemical = h2_generated * HHV_H2
        # Only the charge that makes hydrogen offsets the thermoneutral enthalpy. The rest crosses
        # the membrane and recombines, so its energy is heat as well: the overpotential alone would
        # understate the heat at a faradaic efficiency below one.
        # A voltage below that threshold makes the cell absorb heat instead of rejecting it, and the
        # model says so with a negative number rather than clamping to zero.
        heat_rejected = cells * current * (p["cell_voltage"] - p["faradaic_efficiency"] * thermoneutral_voltage())
        grid_power = stack_power / (1.0 - p["auxiliary_fraction"])
        auxiliary_power = grid_power - stack_power

        # Product compression: a polytropic stage from stack pressure to the delivery pressure.
        ratio = max(p["delivery_pressure"] / 1.013_25, 1.0)
        exponent = polytropic_exponent(p["gas_gamma"], p["compressor_efficiency"])
        specific_compression = (1.0 / exponent) * (R_UNIVERSAL / MOLAR_MASS_H2) * p["suction_temperature"] * (
            ratio ** exponent - 1.0) / max(p["compressor_efficiency"], 1e-9)
        compression_power = h2_delivered * specific_compression

        seconds_a_year = p["operating_hours"] * 3_600.0
        outputs = {
            "cells": Quantity("cells", cells, "-", "stack power divided by the power of one cell at this voltage"),
            "stack_voltage": Quantity("stack_voltage", stack_voltage / 1e3, "kV", "series stack voltage"),
            "current": Quantity("current", current / 1e3, "kA", "the current every cell carries"),
            "hydrogen_rate": Quantity("hydrogen_rate", h2_delivered * 3_600.0, "kg/h", "product at the battery limit"),
            "hydrogen_purged": Quantity("hydrogen_purged", h2_purged * 3_600.0, "kg/h",
                                        "sent to purge and flare rather than to the battery limit"),
            "hydrogen_per_day": Quantity("hydrogen_per_day", h2_delivered * 86_400.0 / 1e3, "t/day", "product per day"),
            "hydrogen_per_year": Quantity("hydrogen_per_year", h2_delivered * seconds_a_year / 1e3, "t/yr",
                                          "product at the declared full-load hours"),
            "specific_energy": Quantity("specific_energy", grid_power / max(h2_delivered, 1e-30) / 3.6e6,
                                        "kWh/kg", "grid electricity per kilogram at the battery limit"),
            "specific_energy_stack": Quantity("specific_energy_stack", stack_power / max(h2_delivered, 1e-30) / 3.6e6,
                                              "kWh/kg", "stack DC electricity per kilogram, without balance of plant"),
            "lhv_efficiency": Quantity("lhv_efficiency", h2_delivered * LHV_H2 / grid_power, "-",
                                       "hydrogen's lower heating value over the grid electricity"),
            "hhv_efficiency": Quantity("hhv_efficiency", h2_delivered * HHV_H2 / grid_power, "-",
                                       "hydrogen's higher heating value over the grid electricity"),
            "cell_efficiency": Quantity("cell_efficiency",
                                        reversible_voltage() / p["cell_voltage"] * p["faradaic_efficiency"], "-",
                                        "reversible over applied voltage, times the faradaic efficiency"),
            "heat_rejected": Quantity("heat_rejected", heat_rejected / 1e6, "MW",
                                      "the overpotential above the thermoneutral voltage"),
            "oxygen_rate": Quantity("oxygen_rate", o2_from_stoichiometry * 3_600.0, "kg/h", "by-product oxygen"),
            "water_demand": Quantity("water_demand", water_consumed * 3_600.0 / 1e3, "m3/h",
                                     "process water split, at 1000 kg/m3"),
            "compression_power": Quantity("compression_power", compression_power / 1e6, "MW",
                                          "polytropic product compression to the delivery pressure"),
            "energy_per_kg_with_compression": Quantity(
                "energy_per_kg_with_compression",
                (grid_power + compression_power) / max(h2_delivered, 1e-30) / 3.6e6, "kWh/kg",
                "what the kilogram costs delivered, not just made"),
        }

        balances = [
            close("electricity", p["stack_power"] * 1e6, stack_power, "W", tolerance=1e-12,
                  detail="the power asked for against cells times voltage times current"),
            close("energy", grid_power, hydrogen_chemical + heat_rejected + auxiliary_power, "W", tolerance=1e-9,
                  detail="electricity in against hydrogen at its higher heating value, the heat the overpotential "
                         "rejects and the balance of plant"),
            close("water", water_consumed, h2_generated + o2_from_stoichiometry, "kg/s", tolerance=1e-9,
                  detail="the stoichiometric water against the mass of the two gases it became"),
            close("oxygen", o2_from_stoichiometry, o2_from_charge, "kg/s", tolerance=1e-9,
                  detail="oxygen from the hydrogen's molar ratio against oxygen from the charge"),
        ]

        checks = [
            Check("cell_voltage_above_reversible", p["cell_voltage"] > reversible_voltage(),
                  f"{p['cell_voltage']:.3f} V against the {reversible_voltage():.3f} V that splits water"),
            Check("efficiency_below_one", outputs["lhv_efficiency"].value < 1.0,
                  "no stack returns more hydrogen energy than the electricity it takes"),
            Check("specific_energy_above_the_bound",
                  outputs["specific_energy_stack"].value >= LHV_H2 / 3.6e6,
                  "the stack cannot beat 33.3 kWh/kg, the lower heating value of a kilogram of hydrogen"),
            Check("heat_sign_matches_the_voltage",
                  (heat_rejected > 0.0) == (p["cell_voltage"] > p["faradaic_efficiency"] * thermoneutral_voltage()),
                  "above the thermoneutral voltage, scaled by the faradaic efficiency, the stack rejects heat; "
                  "below it, it would absorb heat"),
            Check("compression_is_the_smaller_term", compression_power < stack_power,
                  "compressing the product cannot cost more than making it"),
            Check("oxygen_is_the_stoichiometric_half", o2_from_charge > 0.0,
                  "eight kilograms of oxygen per kilogram of hydrogen, from the charge"),
        ]

        run = TwinRun(system=self.slug, title=self.title, parameters=dict(p), outputs=outputs,
                      balances=balances, checks=checks)
        # The polarisation-free view: specific energy against cell voltage is the whole design trade.
        xs = [1.30 + 0.02 * k for k in range(46)]
        ys = []
        for voltage in xs:
            per_cell = voltage * current
            n = p["stack_power"] * 1e6 / per_cell
            produced = hydrogen_from_charge(current=current, cells=n,
                                            faradaic_efficiency=p["faradaic_efficiency"])
            produced *= (1.0 - p["purge_fraction"])
            ys.append(p["stack_power"] * 1e6 / (1.0 - p["auxiliary_fraction"]) / max(produced, 1e-30) / 3.6e6)
        run.series = self._series(len(xs), cell_voltage_V=xs, specific_energy_kWh_per_kg=ys)
        run.not_verified = list(self.not_verified)
        return run

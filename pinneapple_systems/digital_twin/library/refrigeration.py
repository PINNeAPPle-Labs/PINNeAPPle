"""An industrial refrigeration plant: a vapour-compression cycle, its COP and the Carnot wall it cannot pass.

What is modelled
----------------
An ammonia plant on the four-point cycle every refrigeration engineer draws: evaporate at the
load temperature, compress to the condensing pressure, condense against the cooling medium, and
throttle back. Saturation pressures come from Antoine's correlation for ammonia and the latent heat
from Watson's, both anchored on the normal boiling point, so the cycle is built from correlations
that can each be checked against published data instead of from a table lookahead.

Two numbers decide whether a plant is worth building, and both are outputs: the **coefficient of
performance**, and how far it sits below the **Carnot COP** between the same two temperatures. The
gap is not an opinion — it is the second-law efficiency, and it is what an energy audit asks about.

The conservation laws it checks
-------------------------------
``energy``  the compressor's work by two routes: the enthalpy rise the isentropic efficiency
            produces, and the integral of ``v dp`` along the same path, computed numerically
``duty``    the first law round the loop (evaporator plus compressor equals condenser) against the
            condenser's own enthalpy route — desuperheat plus the latent heat at the condensing
            temperature. The residual is the property model talking, not the plant: Watson's latent
            heat and a constant liquid specific heat are not exactly consistent, and the balance
            reports how much that costs (a few per cent) instead of hiding it.

The invariants are the thermodynamic ones: no plant beats Carnot, the second-law efficiency stays in
the band real machines occupy, the discharge temperature stays inside what a compressor survives, and
the evaporator has to sit below the load it serves.

Limits, stated rather than discovered
-------------------------------------
One refrigerant (ammonia), one pressure level, no subcooling, no superheat beyond the compressor's
own discharge, no oil, no pressure drops, and a single isentropic efficiency. The property model is
Antoine plus Watson plus constant specific heats, which is accurate enough to size a plant and not
accurate enough to replace a property database. It does not select a compressor.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from .base import Check, Parameter, ParameterError, Quantity, ReferenceModel, TwinRun, close

__all__ = [
    "RefrigerationPlant",
    "REFRIGERATION_PARAMETERS",
    "saturation_pressure",
    "saturation_temperature",
    "latent_heat",
    "carnot_cop",
]

# Antoine for ammonia: log10(P[mmHg]) = A - B / (t[degC] + C), valid over -50 to +60 degC.
ANTOINE_A = 7.360_50
ANTOINE_B = 926.132
ANTOINE_C = 240.17
MMHG_TO_PA = 133.322_387
# Watson's correlation, anchored on the latent heat at the normal boiling point.
T_NBP = 239.81          # K
H_FG_NBP = 1_371_200.0  # J/kg at the normal boiling point
T_CRITICAL = 405.40     # K
WATSON_EXPONENT = 0.38
# Specific heats used by the cycle. cp of saturated ammonia liquid is 4.7 kJ/(kg K) at these
# temperatures; cp of the vapour follows from gamma and the gas constant.
CP_LIQUID = 4_700.0     # J/(kg K)
GAS_CONSTANT_AMMONIA = 8.314_462_618 / 17.031e-3   # J/(kg K)


def saturation_pressure(temperature: float) -> float:
    """Antoine's saturation pressure of ammonia at ``temperature`` (K), in Pa."""
    celsius = temperature - 273.15
    return 10.0 ** (ANTOINE_A - ANTOINE_B / (celsius + ANTOINE_C)) * MMHG_TO_PA


def saturation_temperature(pressure: float) -> float:
    """The inverse of :func:`saturation_pressure`: the temperature at which ammonia boils (K)."""
    log10_mmhg = math.log10(max(pressure, 1e-30) / MMHG_TO_PA)
    celsius = ANTOINE_B / (ANTOINE_A - log10_mmhg) - ANTOINE_C
    return celsius + 273.15


def latent_heat(temperature: float) -> float:
    """Watson's latent heat of ammonia at ``temperature`` (J/kg), anchored at the normal boiling point."""
    reduced = (1.0 - temperature / T_CRITICAL) / (1.0 - T_NBP / T_CRITICAL)
    if reduced <= 0.0:
        raise ParameterError(f"{temperature} K is at or above ammonia's critical temperature")
    return H_FG_NBP * reduced ** WATSON_EXPONENT


def carnot_cop(*, evaporating_temperature: float, condensing_temperature: float) -> float:
    """``T_evap / (T_cond - T_evap)`` — the ceiling every real cycle works under."""
    return evaporating_temperature / (condensing_temperature - evaporating_temperature)


REFRIGERATION_PARAMETERS = {
    "cooling_duty": Parameter("cooling_duty", "MW", 5.0, 0.05, 100.0, "heat the evaporator takes out"),
    "load_temperature": Parameter("load_temperature", "degC", -8.0, -40.0, 10.0,
                                  "temperature of what is being cooled"),
    "evaporator_approach": Parameter("evaporator_approach", "K", 2.0, 0.5, 8.0,
                                     "how far below the load the refrigerant has to evaporate"),
    "cooling_medium_temperature": Parameter("cooling_medium_temperature", "degC", 30.0, 5.0, 45.0,
                                            "ambient wet bulb or cooling-water supply"),
    "condenser_approach": Parameter("condenser_approach", "K", 5.0, 2.0, 15.0,
                                    "how far above the medium the refrigerant has to condense"),
    "isentropic_efficiency": Parameter("isentropic_efficiency", "-", 0.75, 0.50, 0.90,
                                       "compressor isentropic efficiency"),
    "gas_gamma": Parameter("gas_gamma", "-", 1.30, 1.20, 1.40, "ratio of specific heats of ammonia vapour"),
    "motor_efficiency": Parameter("motor_efficiency", "-", 0.95, 0.85, 0.99, "motor and drive on the shaft"),
}


class RefrigerationPlant(ReferenceModel):
    """An ammonia vapour-compression plant: pressures, the cycle, COP and the Carnot gap."""

    slug = "industrial_refrigeration"
    title = "Industrial refrigeration: vapour-compression cycle, COP and the second-law gap"
    physics = "Antoine saturation, Watson latent heat, isentropic compression, Carnot bound"
    parameters = REFRIGERATION_PARAMETERS
    assumptions = (
        "The cycle is the four-point one: saturated vapour at the evaporator, isentropic compression "
        "with an efficiency, saturated liquid at the condenser, isenthalpic throttling.",
        "The refrigerant is ammonia, with the property correlations stated in the module: Antoine, "
        "Watson and constant specific heats.",
        "No subcooling, no useful superheat, no pressure drop in either heat exchanger.",
        "The load and the cooling medium are at fixed temperatures: there is no part-load or pulldown.",
    )
    limits = (
        "One refrigerant, one pressure level: a cascade or a two-stage plant is outside this model.",
        "The property model is a correlation: a design within a few per cent of a limit needs a real "
        "property database, not Watson and a constant liquid cp.",
        "The condenser duty's two routes disagree by the property model's own inconsistency (a few per cent); "
        "the balance reports that number rather than pretending it is zero.",
        "No compressor map, no oil, no capacity control, no defrost; the compressor is an efficiency.",
    )
    sources = (
        "Antoine coefficients for ammonia over -50 to +60 degC: A = 7.36050, B = 926.132, C = 240.17 "
        "(P in mmHg, t in degC), reproducing the normal boiling point of -33.34 degC at one atmosphere.",
        "Watson correlation, anchored on h_fg = 1371.2 kJ/kg at the normal boiling point, with a critical "
        "temperature of 405.4 K and an exponent of 0.38.",
        "Isentropic compression of an ideal gas: T2/T1 = (p2/p1)^((gamma-1)/gamma).",
        "Carnot's COP: T_evap / (T_cond - T_evap), both absolute.",
    )
    not_verified = (
        "No plant data: the duty, temperatures and efficiency are engineering-plausible, not a commissioned machine.",
        "No comparison against a manufacturer's selection software or a measured COP at these conditions.",
        "No part-load performance: a real plant spends most of its life away from the design point.",
        "The two-route condenser balance is limited by the property model, not by the physics: treating its "
        "residual as an error of the plant would be wrong, and the run says so.",
    )

    def _validate(self, p: Mapping[str, float]) -> None:
        if p["evaporator_approach"] <= 0.0:
            raise ParameterError("the refrigerant has to evaporate below the load: the approach must be positive")
        if p["condenser_approach"] <= 0.0:
            raise ParameterError("the refrigerant has to condense above the medium: the approach must be positive")
        evaporating = p["load_temperature"] - p["evaporator_approach"]
        condensing = p["cooling_medium_temperature"] + p["condenser_approach"]
        if condensing <= evaporating:
            raise ParameterError(
                f"the condensing temperature ({condensing:.2f} degC) has to be above the evaporating one "
                f"({evaporating:.2f} degC)")

    def _run(self, p: Mapping[str, float]) -> TwinRun:
        t_evap = p["load_temperature"] - p["evaporator_approach"] + 273.15
        t_cond = p["cooling_medium_temperature"] + p["condenser_approach"] + 273.15
        p_evap = saturation_pressure(t_evap)
        p_cond = saturation_pressure(t_cond)
        ratio = p_cond / p_evap

        heat_capacity_vapour = p["gas_gamma"] * GAS_CONSTANT_AMMONIA / (p["gas_gamma"] - 1.0)
        exponent = (p["gas_gamma"] - 1.0) / p["gas_gamma"]
        t_discharge_isentropic = t_evap * ratio ** exponent
        rise_isentropic = t_discharge_isentropic - t_evap
        t_discharge = t_evap + rise_isentropic / p["isentropic_efficiency"]

        work_specific = heat_capacity_vapour * rise_isentropic / p["isentropic_efficiency"]  # J/kg
        # The same work by the other route: the integral of v dp along the isentropic path, divided by
        # the efficiency. An ideal gas has p v = R T, so the path is v = R T1 p^(-1/gamma) / p1^(-1/gamma).
        nodes = 2_000
        integral = 0.0
        for k in range(nodes):
            p0 = p_evap * ratio ** (k / nodes)
            p1_ = p_evap * ratio ** ((k + 1) / nodes)
            t0 = t_evap * (p0 / p_evap) ** exponent
            t1 = t_evap * (p1_ / p_evap) ** exponent
            v0 = GAS_CONSTANT_AMMONIA * t0 / p0
            v1 = GAS_CONSTANT_AMMONIA * t1 / p1_
            integral += 0.5 * (v0 + v1) * (p1_ - p0)
        work_integral = integral / p["isentropic_efficiency"]

        h_fg_evap = latent_heat(t_evap)
        h_fg_cond = latent_heat(t_cond)
        # The refrigerant that leaves the condenser as saturated liquid has to be throttled down to the
        # evaporator, and the flash gas that comes off costs evaporator capacity.
        throttle_loss = CP_LIQUID * (t_cond - t_evap)
        net_refrigeration = h_fg_evap - throttle_loss
        if net_refrigeration <= 0.0:
            raise ParameterError(
                "the throttle loss is larger than the latent heat available: this cycle cannot refrigerate at "
                "these temperatures")

        mass_flow = p["cooling_duty"] * 1e6 / net_refrigeration                       # kg/s
        shaft_power = mass_flow * work_specific
        electrical_power = shaft_power / p["motor_efficiency"]
        cop = p["cooling_duty"] * 1e6 / electrical_power
        carnot = carnot_cop(evaporating_temperature=t_evap, condensing_temperature=t_cond)

        # The condenser, by its own enthalpy route: desuperheat the discharge down to saturation, then
        # condense. Compared with the first law in the balance below.
        heat_rejected_enthalpy = mass_flow * (h_fg_cond + heat_capacity_vapour * (t_discharge - t_cond))
        # The refrigerant side sees the shaft work, not the electricity: the motor's own losses leave
        # the machine into the plant room, which is a different heat path and a different problem.
        heat_rejected_first_law = p["cooling_duty"] * 1e6 + shaft_power

        outputs = {
            "evaporating_temperature": Quantity("evaporating_temperature", t_evap - 273.15, "degC",
                                                "the load less the evaporator approach"),
            "condensing_temperature": Quantity("condensing_temperature", t_cond - 273.15, "degC",
                                               "the medium plus the condenser approach"),
            "evaporating_pressure": Quantity("evaporating_pressure", p_evap / 1e5, "bar", "Antoine at that temperature"),
            "condensing_pressure": Quantity("condensing_pressure", p_cond / 1e5, "bar", "Antoine at that temperature"),
            "pressure_ratio": Quantity("pressure_ratio", ratio, "-", "what the compressor has to climb"),
            "discharge_temperature": Quantity("discharge_temperature", t_discharge - 273.15, "degC",
                                              "after the isentropic rise, divided by the efficiency"),
            "refrigerant_flow": Quantity("refrigerant_flow", mass_flow * 3_600.0, "kg/h", "ammonia round the loop"),
            "specific_work": Quantity("specific_work", work_specific / 1e3, "kJ/kg", "shaft work per kilogram"),
            "shaft_power": Quantity("shaft_power", shaft_power / 1e6, "MW", "at the compressor coupling"),
            "electrical_power": Quantity("electrical_power", electrical_power / 1e6, "MW", "at the terminals"),
            "cop": Quantity("cop", cop, "-", "cooling delivered over electricity bought"),
            "carnot_cop": Quantity("carnot_cop", carnot, "-", "the ceiling between the same two temperatures"),
            "second_law_efficiency": Quantity("second_law_efficiency", cop / carnot, "-",
                                              "how much of the Carnot ceiling the plant actually reaches"),
            "heat_rejected": Quantity("heat_rejected", heat_rejected_first_law / 1e6, "MW",
                                      "cooling duty plus the shaft work, which is what the condenser sees; the "
                                      "motor's own losses leave into the plant room"),
            "latent_heat_evaporator": Quantity("latent_heat_evaporator", h_fg_evap / 1e3, "kJ/kg", "Watson at the evaporator"),
            "latent_heat_condenser": Quantity("latent_heat_condenser", h_fg_cond / 1e3, "kJ/kg", "Watson at the condenser"),
            "throttle_loss": Quantity("throttle_loss", throttle_loss / 1e3, "kJ/kg",
                                      "capacity the flash gas costs: cp_liquid times the lift"),
        }

        balances = [
            close("energy", work_specific, work_integral, "J/kg", tolerance=1e-6,
                  detail="the isentropic enthalpy rise divided by the efficiency against the integral of v dp "
                         "along the same path"),
            close("duty", heat_rejected_first_law, heat_rejected_enthalpy, "W", tolerance=6e-2,
                  detail="first law round the loop against the condenser's own enthalpy route (desuperheat plus "
                         "Watson's latent heat). The residual is the property model: Watson and a constant liquid "
                         "cp are not exactly consistent, and a few per cent is what that costs"),
        ]

        checks = [
            Check("cop_below_carnot", cop < carnot,
                  f"{cop:.3f} against a Carnot ceiling of {carnot:.3f}"),
            Check("second_law_efficiency_in_band", 0.30 < cop / carnot < 0.90,
                  "a real ammonia plant sits between a third and nine tenths of its Carnot ceiling"),
            Check("discharge_temperature_survivable", t_discharge - 273.15 < 150.0,
                  "above about 150 degC ammonia compressors and oil stop being happy"),
            Check("evaporator_below_the_load", t_evap - 273.15 < p["load_temperature"],
                  "heat only flows into a refrigerant colder than what it cools"),
            Check("condenser_above_the_medium", t_cond - 273.15 > p["cooling_medium_temperature"],
                  "heat only flows out of a refrigerant hotter than the medium"),
            Check("latent_heat_falls_with_temperature", h_fg_cond < h_fg_evap,
                  "Watson's correlation has to fall towards the critical point"),
        ]

        run = TwinRun(system=self.slug, title=self.title, parameters=dict(p), outputs=outputs,
                      balances=balances, checks=checks)
        # The lift is the design knob: COP collapses as the condensing temperature rises.
        xs, ys = [], []
        for k in range(41):
            medium = 10.0 + 0.9 * k
            cond = medium + p["condenser_approach"] + 273.15
            if cond <= t_evap:
                continue
            ratio_k = saturation_pressure(cond) / p_evap
            rise = t_evap * (ratio_k ** exponent - 1.0)
            work = heat_capacity_vapour * rise / p["isentropic_efficiency"]
            cop_k = net_refrigeration / (work / p["motor_efficiency"])
            xs.append(medium)
            ys.append(cop_k)
        run.series = self._series(len(xs), cooling_medium_degC=xs, cop=ys)
        run.not_verified = list(self.not_verified)
        return run

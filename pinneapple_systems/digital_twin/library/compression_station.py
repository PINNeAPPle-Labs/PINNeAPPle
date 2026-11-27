"""A gas compression station: a multistage machine, its intercoolers, its power and its fuel.

What is modelled
----------------
A station that lifts gas from a suction pressure to a discharge pressure in one or more stages,
cooling it back to a set temperature between stages. Each stage is an adiabatic polytropic
compression, so the discharge temperature follows from the ratio and the polytropic efficiency,
and the shaft work follows from the temperature rise — for an adiabatic machine every joule of
shaft work ends up in the gas.

The two answers a station is actually sized on are both here: how the total ratio should be split
between stages (equal ratios, when the intercoolers bring the gas back to the same temperature,
and the model *proves* it by minimising the work numerically rather than asserting it), and what
that costs in shaft power and fuel.

The conservation laws it checks
-------------------------------
``mass``    what the station takes in against what it delivers
``energy``  the polytropic shaft work of every stage against the enthalpy the gas took, stage by
            stage, with the integral of ``v dp`` along the path computed numerically instead of
            trusting the closed form twice
``cooling`` what the intercoolers and aftercooler remove against what the gas gives up

The invariants are the ones that make the machine trustworthy: the isentropic efficiency cannot
exceed one, splitting the ratio over more stages cannot cost more work, and equal ratios are the
minimum of the split.

Limits, stated rather than discovered
-------------------------------------
Ideal gas with constant cp and gamma: no real-gas departure, so a dense or rich gas is off. One
polytropic efficiency for the whole map, no speed lines, no fouling, no driver curve beyond a
single efficiency, and no mechanical losses. Surge is a declared minimum inlet flow from the
vendor's map, not something this model predicts. It sizes a station; it does not select a machine.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from .base import Check, Parameter, ParameterError, Quantity, ReferenceModel, TwinRun, close

__all__ = [
    "CompressionStation",
    "STATION_PARAMETERS",
    "polytropic_exponent",
    "stage_discharge_temperature",
    "stage_work",
    "split_ratios",
]

R_UNIVERSAL = 8.314_462_618


def polytropic_exponent(gamma: float, polytropic_efficiency: float) -> float:
    """``m = (gamma - 1) / (gamma eta_p)`` — the exponent in ``T2 = T1 r^m``."""
    return (gamma - 1.0) / (gamma * polytropic_efficiency)


def stage_discharge_temperature(*, suction_temperature: float, ratio: float, gamma: float,
                                polytropic_efficiency: float) -> float:
    """Discharge temperature of one stage (K)."""
    return suction_temperature * ratio ** polytropic_exponent(gamma, polytropic_efficiency)


def stage_work(*, mass_flow: float, suction_temperature: float, ratio: float, gamma: float,
               heat_capacity: float, polytropic_efficiency: float) -> float:
    """Shaft power of one stage (W): ``m_dot cp (T2 - T1)``, the enthalpy the gas took.

    An adiabatic compressor has nowhere else to put its shaft work, which is why this is the
    power and not merely a proxy for it.
    """
    t2 = stage_discharge_temperature(suction_temperature=suction_temperature, ratio=ratio,
                                     gamma=gamma, polytropic_efficiency=polytropic_efficiency)
    return mass_flow * heat_capacity * (t2 - suction_temperature)


def split_ratios(total_ratio: float, stages: int, *, fractions: tuple[float, ...] | None = None) -> list[float]:
    """The per-stage ratios: equal by default, or a declared split whose product is the total."""
    if stages < 1:
        raise ValueError("a station has at least one stage")
    if fractions is None:
        return [total_ratio ** (1.0 / stages)] * stages
    if len(fractions) != stages:
        raise ValueError(f"{len(fractions)} fractions for {stages} stages")
    total = math.prod(fractions)
    return [total_ratio * f / total for f in fractions]


STATION_PARAMETERS = {
    "mass_flow": Parameter("mass_flow", "kg/s", 120.0, 0.5, 2_000.0, "what the station moves"),
    "suction_pressure": Parameter("suction_pressure", "Pa", 3.0e6, 0.1e6, 10.0e6,
                                  "pressure at the station inlet"),
    "discharge_pressure": Parameter("discharge_pressure", "Pa", 9.0e6, 0.2e6, 25.0e6,
                                    "pressure the line downstream needs"),
    "suction_temperature": Parameter("suction_temperature", "K", 303.15, 260.0, 350.0,
                                     "gas temperature entering the first stage"),
    "stages": Parameter("stages", "-", 2.0, 1.0, 4.0, "how many stages the machine has"),
    "polytropic_efficiency": Parameter("polytropic_efficiency", "-", 0.80, 0.55, 0.92,
                                       "polytropic efficiency of every stage"),
    "intercooler_outlet": Parameter("intercooler_outlet", "K", 313.15, 290.0, 340.0,
                                    "temperature the gas is cooled back to between stages"),
    "gas_gravity": Parameter("gas_gravity", "-", 0.65, 0.55, 0.95, "molecular weight over that of air"),
    "gamma": Parameter("gamma", "-", 1.30, 1.15, 1.70, "ratio of specific heats"),
    "compressibility": Parameter("compressibility", "-", 0.92, 0.70, 1.00, "mean Z of the gas"),
    "max_discharge_temperature": Parameter("max_discharge_temperature", "K", 398.15, 320.0, 450.0,
                                           "machine or coating limit at any stage discharge"),
    "min_inlet_flow_ratio": Parameter("min_inlet_flow_ratio", "-", 0.55, 0.10, 0.95,
                                      "smallest inlet flow, as a fraction of the machine's design flow"),
    "design_inlet_flow": Parameter("design_inlet_flow", "m3/s", 2.40, 0.05, 100.0,
                                   "inlet volumetric flow the machine was selected for"),
    "driver_efficiency": Parameter("driver_efficiency", "-", 0.94, 0.80, 0.99,
                                   "gas turbine or motor efficiency on the shaft"),
    "fuel_lhv": Parameter("fuel_lhv", "MJ/kg", 48.0, 20.0, 55.0, "lower heating value of the fuel gas"),
}


class CompressionStation(ReferenceModel):
    """A multistage compressor station with intercooling."""

    slug = "compression_station"
    title = "Gas compression station: stage split, discharge temperatures, power and fuel"
    physics = "polytropic compression, intercooling, equal-ratio split, shaft power and fuel gas"
    parameters = STATION_PARAMETERS
    assumptions = (
        "Ideal gas with constant cp and gamma, and one polytropic efficiency per stage.",
        "The gas leaves every intercooler and the aftercooler at the same temperature.",
        "Every stage compresses the same mass flow: no sidestreams, no recycle, no seal gas.",
        "The driver is one efficiency on the shaft; mechanical and gear losses are inside it.",
    )
    limits = (
        "No real-gas departure: a dense phase or a rich gas needs the real-gas layer and a vendor map.",
        "No speed lines, no fouling, no degradation: the efficiency is a single number.",
        "Surge is a declared minimum inlet flow, not a predicted surge line.",
        "The station is sized, not selected: driver curve, starting, and control are outside it.",
    )
    sources = (
        "Polytropic compression: T2 = T1 r^((gamma-1)/(gamma eta_p)), the standard stage relation.",
        "Adiabatic energy balance: shaft work = m_dot cp (T2 - T1), with cp = gamma R/(gamma - 1).",
        "Equal pressure ratios minimise the total work of equal-temperature stages (proved here numerically).",
    )
    not_verified = (
        "No measured machine data: the station is a concept with plausible numbers.",
        "No comparison with a vendor performance map or a plant heat balance.",
        "Fuel consumption uses a single driver efficiency and the fuel's LHV, with no part-load curve.",
        "The intercooler is an outlet temperature, not a heat exchanger: UA, approach and duty limits are absent.",
    )

    def _validate(self, p: Mapping[str, float]) -> None:
        if p["discharge_pressure"] <= p["suction_pressure"]:
            raise ParameterError("the discharge pressure has to be above the suction pressure")
        if p["intercooler_outlet"] < p["suction_temperature"] - 1e-9:
            raise ParameterError(
                "the intercooler cannot cool the gas below the suction temperature with this model: "
                "no refrigeration is represented")

    def _run(self, p: Mapping[str, float]) -> TwinRun:
        stages = int(round(p["stages"]))
        mw = p["gas_gravity"] * 28.96e-3
        r_specific = R_UNIVERSAL / mw * p["compressibility"]
        heat_capacity = p["gamma"] * r_specific / (p["gamma"] - 1.0)
        total_ratio = p["discharge_pressure"] / p["suction_pressure"]
        m = polytropic_exponent(p["gamma"], p["polytropic_efficiency"])

        ratios = split_ratios(total_ratio, stages)

        def run_split(candidate: list[float]) -> tuple[list[float], list[float], list[float], float]:
            """Discharge temperatures, powers and cooling duties for a given split of ratios."""
            t_in = p["suction_temperature"]
            temps, powers, cooling, work_total = [], [], [], 0.0
            for ratio in candidate:
                t_out = stage_discharge_temperature(suction_temperature=t_in, ratio=ratio,
                                                    gamma=p["gamma"],
                                                    polytropic_efficiency=p["polytropic_efficiency"])
                power = p["mass_flow"] * heat_capacity * (t_out - t_in)
                temps.append(t_out)
                powers.append(power)
                cooling.append(p["mass_flow"] * heat_capacity * (t_out - p["intercooler_outlet"]))
                work_total += power
                t_in = p["intercooler_outlet"]
            return temps, powers, cooling, work_total

        temperatures, powers, cooling, shaft = run_split(ratios)
        total_cooling = sum(cooling)

        # The shaft work, checked against the integral of v dp along the polytropic path rather
        # than against a second copy of the same algebra.
        def integral_work(ratio: float, t_suction: float) -> float:
            steps = 400
            total = 0.0
            for k in range(steps):
                a = (p["suction_pressure"] * ratio ** (k / steps))
                b = (p["suction_pressure"] * ratio ** ((k + 1) / steps))
                pa, pb = a, b
                ta = t_suction * (pa / p["suction_pressure"]) ** m
                tb = t_suction * (pb / p["suction_pressure"]) ** m
                va, vb = r_specific * ta / pa, r_specific * tb / pb
                total += 0.5 * (va + vb) * (pb - pa)
            return total

        work_integral = 0.0
        t_in = p["suction_temperature"]
        for ratio in ratios:
            work_integral += p["mass_flow"] * integral_work(ratio, t_in) / p["polytropic_efficiency"]
            t_in = p["intercooler_outlet"]

        # Equal ratios claim to be the minimum of the split: search the space and see.
        best_work = shaft
        for step in range(1, 40):
            first = 0.5 + 0.0125 * step            # first stage gets a larger or smaller share
            candidate = split_ratios(total_ratio, stages, fractions=(first,) + (1.0,) * (stages - 1))
            best_work = min(best_work, run_split(candidate)[3])
        split_is_optimal = abs(best_work - shaft) <= 1e-6 * max(shaft, 1e-9)

        inlet_volume = (p["mass_flow"] * r_specific * p["suction_temperature"] / p["suction_pressure"])
        isentropic_efficiency = (total_ratio ** ((p["gamma"] - 1.0) / p["gamma"]) - 1.0) / max(
            total_ratio ** m - 1.0, 1e-30)
        fuel = shaft / (p["driver_efficiency"] * p["fuel_lhv"] * 1e6)      # kg/s

        outputs = {
            "total_ratio": Quantity("total_ratio", total_ratio, "-", "discharge over suction"),
            "stage_ratio": Quantity("stage_ratio", ratios[0], "-", "the ratio every stage takes"),
            "discharge_temperature": Quantity("discharge_temperature", temperatures[0] - 273.15, "degC",
                                              "hottest stage discharge before cooling"),
            "aftercooler_temperature": Quantity("aftercooler_temperature", p["intercooler_outlet"] - 273.15,
                                                "degC", "what the gas leaves the station at"),
            "shaft_power": Quantity("shaft_power", shaft / 1e6, "MW", "sum over the stages"),
            "specific_power": Quantity("specific_power", shaft / max(p["mass_flow"], 1e-9) / 1e3,
                                       "kJ/kg", "shaft work per kilogram compressed"),
            "cooling_duty": Quantity("cooling_duty", total_cooling / 1e6, "MW", "intercoolers plus aftercooler"),
            "isentropic_efficiency": Quantity("isentropic_efficiency", isentropic_efficiency, "-",
                                              "what the polytropic efficiency buys in isentropic terms"),
            "inlet_volume_flow": Quantity("inlet_volume_flow", inlet_volume, "m3/s",
                                          "what the first stage actually sees"),
            "inlet_flow_margin": Quantity("inlet_flow_margin",
                                          inlet_volume / max(p["design_inlet_flow"], 1e-9), "-",
                                          "inlet flow over the design flow: below the minimum the machine surges"),
            "heat_capacity": Quantity("heat_capacity", heat_capacity, "J/(kg.K)",
                                      "from gamma and R, cp = gamma R/(gamma - 1)"),
            "polytropic_exponent": Quantity("polytropic_exponent", m, "-", "(gamma-1)/(gamma eta_p)"),
            "fuel_flow": Quantity("fuel_flow", fuel * 3_600.0, "kg/h", "driver fuel at its efficiency"),
            "fuel_fraction": Quantity("fuel_fraction", fuel / max(p["mass_flow"], 1e-9), "-",
                                      "fuel over process gas: the station's own consumption"),
        }

        balances = [
            close("mass", p["mass_flow"], p["mass_flow"], "kg/s",
                  detail="no sidestreams: every stage moves the same gas"),
            close("energy", shaft, work_integral, "W", tolerance=1e-6,
                  detail="cp times the temperature rise against the integral of v dp along the path, "
                         "divided by the polytropic efficiency"),
            close("cooling", total_cooling,
                  p["mass_flow"] * heat_capacity * (sum(temperatures) - stages * p["intercooler_outlet"]),
                  "W", tolerance=1e-12,
                  detail="the intercoolers remove exactly the enthalpy the previous stage added"),
        ]

        checks = [
            Check("equal_split_is_the_minimum", split_is_optimal,
                  "the equal-ratio split the model uses is the minimum found by searching the space"),
            Check("isentropic_below_one", isentropic_efficiency <= 1.0 + 1e-12,
                  "no machine beats an isentropic one"),
            Check("discharge_below_the_limit",
                  all(t <= p["max_discharge_temperature"] + 1e-9 for t in temperatures),
                  "stage discharge against the machine and coating limit"),
            Check("above_the_surge_floor", outputs["inlet_flow_margin"].value >= p["min_inlet_flow_ratio"],
                  "the first stage has to see enough gas to stay off its surge line"),
            Check("more_stages_cost_less",
                  stages == 1 or run_split(split_ratios(total_ratio, 1))[3] >= shaft - 1e-6,
                  "intercooling between stages cannot make the compression more expensive"),
        ]

        run = TwinRun(system=self.slug, title=self.title, parameters=dict(p), outputs=outputs,
                      balances=balances, checks=checks)
        sweep_x, sweep_y = [], []
        for k in range(41):
            fraction = 0.5 + 0.025 * k if stages > 1 else 1.0
            candidate = split_ratios(total_ratio, stages, fractions=(fraction,) + (1.0,) * (stages - 1))
            sweep_x.append(fraction)
            sweep_y.append(run_split(candidate)[3] / 1e6)
        run.series = self._series(len(sweep_x), first_stage_share=sweep_x, shaft_power_MW=sweep_y)
        run.not_verified = list(self.not_verified)
        return run

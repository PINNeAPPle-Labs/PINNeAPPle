"""A chemical plant: an equilibrium reactor, a recycle loop, a feed/effluent exchanger and a flash.

What is modelled
----------------
A synthesis loop of the shape every methanol, ammonia or dimethyl-ether plant has: fresh make-up
gas at the stoichiometric ratio, a recycle compressor returning what the reactor did not convert,
a feed/effluent exchanger that preheats the reactor inlet against its own outlet, an adiabatic
equilibrium reactor, a cooler and a flash that drops the product and sends the rest back.

The reaction is written generically as ``A + 2 B <-> C`` in the gas phase, with the enthalpy and
entropy of reaction as inputs. That is deliberate: the model is about the *loop*, the balances
and the coupling between equilibrium, temperature and pressure, and it must be usable for any
synthesis with that stoichiometry by changing two numbers and citing where they came from. The
defaults are an illustrative exothermic synthesis and are labelled as such.

The conservation laws it checks
-------------------------------
``atoms_A``  the A that enters in the make-up against the A in the product, the purge and the
             recycle inventory (which does not change at steady state)
``atoms_B``  the same for B
``energy``   the heat the reaction releases against the heat the exchanger recovers and the
             cooler removes

The invariants are the ones that make the loop trustworthy: the equilibrium constant is the
exponential of the Gibbs energy and follows van't Hoff, conversion rises with pressure and falls
with temperature, and the inert purge is what keeps the loop from drowning in what does not react.

Limits, stated rather than discovered
-------------------------------------
Constant enthalpy and entropy of reaction over the temperature range (no Cp integration), ideal
gas, no kinetics — the reactor is at equilibrium, which is the optimistic bound. The flash is
perfect: the product leaves pure and no gas dissolves in it. Compressor work, pressure drop and
catalyst deactivation are outside the model.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from .base import Check, Parameter, ParameterError, Quantity, ReferenceModel, TwinRun, close

__all__ = ["ChemicalPlant", "PLANT_PARAMETERS", "equilibrium_constant", "reactor_outlet"]

R_GAS = 8.314_462_618
P_STANDARD = 1.0e5        # 1 bar, the standard state the equilibrium constant refers to


def equilibrium_constant(temperature: float, delta_h: float, delta_s: float) -> float:
    """Dimensionless K at ``temperature`` from ``dG = dH - T dS``.

    ``K = exp(-dG/RT)`` with the standard state at one bar. The stoichiometry is fixed to
    ``A + 2 B -> C``, so K multiplies two bar of B: the pressure dependence of the conversion
    comes from here and nowhere else.
    """
    dg = delta_h - temperature * delta_s
    exponent = -dg / (R_GAS * temperature)
    if exponent > 700.0:
        return math.inf
    if exponent < -700.0:
        return 0.0
    return math.exp(exponent)


def _extent(moles: Mapping[str, float], temperature: float, pressure: float,
            delta_h: float, delta_s: float) -> float:
    """Extent of reaction at equilibrium, from the molar feed (A, B, C, inert) in mol/s.

    Bisection on the extent, which is monotone: every extra mole converted raises the product
    and lowers both reactants, so the ratio ``y_C / (y_A y_B^2)`` only grows with the extent.
    """
    n_a0, n_b0, n_c0 = moles["A"], moles["B"], moles["C"]
    n_inert = moles["I"]
    n_total0 = n_a0 + n_b0 + n_c0 + n_inert
    upper = min(n_a0, 0.5 * n_b0)
    if upper <= 0.0:
        return 0.0
    k_target = equilibrium_constant(temperature, delta_h, delta_s)
    if k_target <= 0.0:
        return 0.0
    if k_target == math.inf:
        return upper

    def residual(xi: float) -> float:
        n_a = max(n_a0 - xi, 1e-30)
        n_b = max(n_b0 - 2.0 * xi, 1e-30)
        n_c = n_c0 + xi
        n_tot = n_total0 - 2.0 * xi
        ratio = (n_c / n_tot) / ((n_a / n_tot) * (n_b / n_tot) ** 2)
        k_value = ratio * (P_STANDARD / pressure) ** 2
        return k_value - k_target

    lo, hi = 0.0, upper
    if residual(lo) >= 0.0:
        return 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if residual(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def reactor_outlet(moles: Mapping[str, float], *, inlet_temperature: float, pressure: float,
                   delta_h: float, delta_s: float, heat_capacity: float,
                   iterations: int = 400, damping: float = 0.35) -> dict[str, float]:
    """An adiabatic equilibrium reactor: conversion and outlet temperature solved together.

    The two are coupled the hard way — the conversion sets the heat released, the heat released
    sets the temperature, and the temperature sets the conversion through K. A damped fixed
    point finds the same answer every time; the residual is returned so a caller can see how
    well it converged rather than trusting the iteration.
    """
    temperature = inlet_temperature
    extent = 0.0
    n_total0 = sum(moles.values())
    for _ in range(iterations):
        extent_new = _extent(moles, temperature, pressure, delta_h, delta_s)
        heat = -delta_h * extent_new                      # W (J/mol * mol/s)
        capacity = heat_capacity * max(n_total0 - 2.0 * extent_new, 1e-9)   # W/K
        temperature_new = inlet_temperature + heat / capacity
        delta = abs(temperature_new - temperature) + abs(extent_new - extent)
        temperature = temperature + damping * (temperature_new - temperature)
        extent = extent + damping * (extent_new - extent)
        if delta < 1e-10:
            break
    extent_final = _extent(moles, temperature, pressure, delta_h, delta_s)
    return {
        "extent": extent_final,
        "temperature": temperature,
        "conversion": extent_final / max(moles["A"], 1e-30),
        "residual": abs(extent_final - extent),
    }


PLANT_PARAMETERS = {
    "pressure": Parameter("pressure", "Pa", 80.0e5, 1.0e5, 400.0e5, "reactor pressure"),
    "reactor_inlet_temperature": Parameter("reactor_inlet_temperature", "K", 480.0, 350.0, 900.0,
                                           "temperature of the gas entering the catalyst bed"),
    "separator_temperature": Parameter("separator_temperature", "K", 310.0, 250.0, 400.0,
                                       "temperature of the flash that drops the product"),
    "makeup_rate": Parameter("makeup_rate", "mol/s", 10.0, 0.01, 10_000.0,
                             "fresh A fed to the loop, at the stoichiometric ratio"),
    "purge_fraction": Parameter("purge_fraction", "-", 0.02, 0.0, 0.30,
                                "fraction of the recycle that leaves as purge"),
    "inert_fraction_makeup": Parameter("inert_fraction_makeup", "-", 0.02, 0.0, 0.20,
                                       "inerts in the make-up gas"),
    "delta_h_reaction": Parameter("delta_h_reaction", "J/mol", -90_000.0, -300_000.0, 100_000.0,
                                  "enthalpy of A + 2B -> C, from the reaction's own data"),
    "delta_s_reaction": Parameter("delta_s_reaction", "J/(mol.K)", -220.0, -500.0, 100.0,
                                  "entropy of A + 2B -> C, from the reaction's own data"),
    "heat_capacity": Parameter("heat_capacity", "J/(mol.K)", 42.0, 20.0, 80.0,
                               "mean molar heat capacity of the gas, assumed constant"),
    "exchanger_effectiveness": Parameter("exchanger_effectiveness", "-", 0.85, 0.0, 0.98,
                                         "feed/effluent exchanger effectiveness (NTU model)"),
    "loop_iterations": Parameter("loop_iterations", "-", 200.0, 10.0, 2000.0,
                                 "successive-substitution sweeps for the recycle composition"),
    "convergence": Parameter("convergence", "-", 1e-10, 1e-14, 1e-4,
                             "tolerance on the recycle composition"),
}


class ChemicalPlant(ReferenceModel):
    """A synthesis loop closed by a recycle compressor."""

    slug = "chemical_plant"
    title = "Chemical plant: equilibrium reactor, recycle loop, feed/effluent exchanger and flash"
    physics = "gas-phase equilibrium, adiabatic reactor, recycle convergence, heat exchanger and flash"
    parameters = PLANT_PARAMETERS
    assumptions = (
        "The reactor reaches equilibrium: no kinetics, no catalyst volume, no deactivation.",
        "Enthalpy and entropy of reaction are constant over the temperature range used.",
        "Ideal gas; the heat capacity is one mean value for the whole mixture.",
        "The flash is perfect: the product leaves pure and no gas dissolves in the liquid.",
        "The feed/effluent exchanger is described by one effectiveness, not by a geometry.",
    )
    limits = (
        "Equilibrium is the optimistic bound on conversion; a real reactor runs below it.",
        "Compressor work, loop pressure drop and the refrigeration of the flash are not modelled.",
        "No inerts beyond the one lumped fraction, and no trace species or by-products.",
        "The model sizes a loop, it does not design a reactor or a column.",
    )
    sources = (
        "Equilibrium constant from dG = dH - T dS with the standard state at 1 bar.",
        "van't Hoff relation d ln K / d(1/T) = -dH/R, used here as an internal check.",
        "Effectiveness-NTU description of the feed/effluent exchanger.",
    )
    not_verified = (
        "The default dH and dS are illustrative, not a cited catalyst system: set them from your own reaction data.",
        "No comparison with a plant or a flowsheet simulator (Aspen, UniSim).",
        "No kinetics, so residence time, space velocity and runaway behaviour are absent.",
        "Recycle convergence is checked, but the model does not diagnose a plant that cannot converge physically.",
    )

    def _validate(self, p: Mapping[str, float]) -> None:
        if p["separator_temperature"] >= p["reactor_inlet_temperature"]:
            raise ParameterError("the flash has to be colder than the reactor inlet or nothing condenses")
        if p["purge_fraction"] >= 1.0:
            raise ParameterError("purging the whole recycle is not a loop")
        if p["inert_fraction_makeup"] >= 0.5:
            raise ParameterError("a make-up gas that is half inert cannot sustain a synthesis loop")

    def _run(self, p: Mapping[str, float]) -> TwinRun:
        pressure = p["pressure"]
        dh, ds = p["delta_h_reaction"], p["delta_s_reaction"]
        cp = p["heat_capacity"]
        purge_fraction = p["purge_fraction"]

        makeup_a = p["makeup_rate"]
        makeup_b = 2.0 * makeup_a
        makeup_i = ((makeup_a + makeup_b) * p["inert_fraction_makeup"]
                    / max(1.0 - p["inert_fraction_makeup"], 1e-9))
        makeup = {"A": makeup_a, "B": makeup_b, "C": 0.0, "I": makeup_i}

        def feed_of(extent: float) -> dict[str, float]:
            """The reactor feed that makes ``extent`` a steady state of the loop.

            At steady state everything the flash did not drop goes back: ``recycle = (1-p)
            (feed - extent)``, so the feed is fixed by the extent and the purge and nothing else.
            Solving it in closed form turns a slow recycle convergence into one scalar equation.
            One share of purge is what keeps the inerts from accumulating, and it appears here as
            ``makeup / p`` for anything that does not react.
            """
            return {
                "A": (makeup_a - (1.0 - purge_fraction) * extent) / purge_fraction,
                "B": (makeup_b - 2.0 * (1.0 - purge_fraction) * extent) / purge_fraction,
                "C": 0.0,
                "I": makeup_i / purge_fraction,
            }

        def steady_residual(extent: float) -> float:
            feed = feed_of(extent)
            if min(feed.values()) < 0.0:
                return -1.0
            out = reactor_outlet(feed, inlet_temperature=p["reactor_inlet_temperature"],
                                 pressure=pressure, delta_h=dh, delta_s=ds, heat_capacity=cp)
            return out["extent"] - extent

        upper = min(makeup_a, 0.5 * makeup_b)
        lo, hi = 0.0, upper
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if steady_residual(mid) > 0.0:
                lo = mid          # the loop can convert more than this extent
            else:
                hi = mid
            if hi - lo < 1e-14 * max(upper, 1.0):
                break
        extent = 0.5 * (lo + hi)
        feed = feed_of(extent)
        reactor = reactor_outlet(feed, inlet_temperature=p["reactor_inlet_temperature"],
                                 pressure=pressure, delta_h=dh, delta_s=ds, heat_capacity=cp)
        extent = reactor["extent"]
        feed = feed_of(extent)
        effluent = {"A": feed["A"] - extent, "B": feed["B"] - 2.0 * extent,
                    "C": extent, "I": feed["I"]}
        product = extent
        purge = {k: v * purge_fraction for k, v in effluent.items()}
        recycle = {k: v - purge[k] for k, v in effluent.items()}
        converged = abs(steady_residual(extent)) < 1e-9 * max(extent, 1e-9)

        # -- heat: reaction, preheat and cooling -------------------------------------
        heat_reaction = -dh * extent                                    # W
        n_inlet = sum(feed.values())
        n_outlet = n_inlet - 2.0 * extent
        # The exchanger moves heat from the effluent to the cold feed; its duty is bounded by
        # what the effluent can give up on its way to the cold inlet temperature.
        delta_t_max = reactor["temperature"] - p["reactor_inlet_temperature"]
        preheat = p["exchanger_effectiveness"] * cp * min(n_inlet, n_outlet) * delta_t_max  # W
        cooling = max(heat_reaction - preheat, 0.0)

        outputs = {
            "per_pass_conversion": Quantity("per_pass_conversion", reactor["conversion"], "-",
                                            "extent over the A entering the reactor"),
            "reactor_outlet_temperature": Quantity("reactor_outlet_temperature", reactor["temperature"], "K",
                                                   "adiabatic rise from the heat of reaction"),
            "adiabatic_rise": Quantity("adiabatic_rise", reactor["temperature"] - p["reactor_inlet_temperature"],
                                       "K", "what the bed heats up by"),
            "product_rate": Quantity("product_rate", product * 3600.0, "mol/h", "C leaving the flash"),
            "loop_conversion": Quantity("loop_conversion", product / max(makeup_a, 1e-30), "-",
                                        "C produced per A fed: the number the loop is judged on"),
            "recycle_rate": Quantity("recycle_rate", sum(recycle.values()), "mol/s", "gas returned to the reactor"),
            "purge_rate": Quantity("purge_rate", sum(purge.values()), "mol/s", "gas leaving the loop"),
            "inert_in_loop": Quantity("inert_in_loop", recycle["I"] / max(sum(recycle.values()), 1e-30), "-",
                                      "inert fraction the purge has to hold down"),
            "heat_of_reaction": Quantity("heat_of_reaction", heat_reaction / 1e6, "MW", "released in the bed"),
            "preheat_duty": Quantity("preheat_duty", preheat / 1e6, "MW", "recovered by the exchanger"),
            "cooling_duty": Quantity("cooling_duty", cooling / 1e6, "MW", "removed before the flash"),
            "specific_energy": Quantity("specific_energy", cooling / max(product, 1e-30) / 1e3,
                                        "kJ/mol", "cooling per mole of product"),
            "recycle_ratio": Quantity("recycle_ratio", sum(recycle.values()) / max(makeup_a, 1e-30), "-",
                                      "moles recycled per mole of make-up: an outcome, not a knob"),
        }

        balances = [
            close("atoms_A", makeup_a, product + purge["A"], "mol/s",
                  detail="A fed against A leaving as product and purge"),
            close("atoms_B", makeup_b, 2.0 * product + purge["B"], "mol/s",
                  detail="B fed against the two B bound in the product plus the purge"),
            close("energy", heat_reaction, preheat + cooling, "W", tolerance=1e-9,
                  detail="reaction heat against exchanger recovery plus cooler duty"),
        ]

        k_now = equilibrium_constant(reactor["temperature"], dh, ds)
        k_298 = equilibrium_constant(298.15, dh, ds)
        # van't Hoff, checked numerically on the function that is actually used.
        t1, t2 = 400.0, 500.0
        if 0.0 < k_now < math.inf:
            slope = (math.log(max(equilibrium_constant(t2, dh, ds), 1e-300))
                     - math.log(max(equilibrium_constant(t1, dh, ds), 1e-300))) / (1.0 / t2 - 1.0 / t1)
            vant_hoff = abs(slope + dh / R_GAS) / max(abs(dh / R_GAS), 1e-9)
        else:
            vant_hoff = 0.0

        def conversion_at(press: float, temp: float) -> float:
            moles = {"A": 0.10, "B": 0.20, "C": 0.0, "I": 0.02}
            return _extent(moles, temp, press, dh, ds) / moles["A"]

        checks = [
            Check("equilibrium_constant_from_gibbs",
                  abs(k_298 - math.exp(-(dh - 298.15 * ds) / (R_GAS * 298.15))) < 1e-9 * max(k_298, 1.0),
                  "K(298.15) is exp(-dG/RT) with the numbers the model uses"),
            Check("vant_hoff_holds", vant_hoff < 1e-9,
                  "d ln K / d(1/T) equals -dH/R on the function itself"),
            Check("pressure_raises_conversion",
                  conversion_at(1.5 * pressure, p["reactor_inlet_temperature"])
                  > conversion_at(pressure, p["reactor_inlet_temperature"]),
                  "four moles of gas becoming one: squeezing the loop converts more"),
            Check("temperature_lowers_conversion",
                  conversion_at(pressure, p["reactor_inlet_temperature"] + 20.0)
                  < conversion_at(pressure, p["reactor_inlet_temperature"]),
                  "an exothermic equilibrium gives less at a higher temperature"),
            Check("loop_converged", converged,
                  "the steady state of the recycle closes on the reactor's own equilibrium"),
            Check("product_leaves_the_system", product > 0.0,
                  "a loop that produces nothing is a recycle of its own feed"),
            Check("inerts_held_down", outputs["inert_in_loop"].value <= 0.5,
                  "the purge has to keep the loop from filling with what does not react"),
        ]

        run = TwinRun(system=self.slug, title=self.title, parameters=dict(p), outputs=outputs,
                      balances=balances, checks=checks)
        # The loop converging is the model's own behaviour, so it is worth keeping as a series.
        sweep_history = []
        probe = {"A": 0.0, "B": 0.0, "C": 0.0, "I": 0.0}
        converted = []
        for _ in range(40):
            feed_probe = {k: makeup[k] + probe[k] for k in makeup}
            out = reactor_outlet(feed_probe, inlet_temperature=p["reactor_inlet_temperature"],
                                 pressure=pressure, delta_h=dh, delta_s=ds, heat_capacity=cp)
            eff = {"A": feed_probe["A"] - out["extent"], "B": feed_probe["B"] - 2.0 * out["extent"],
                   "C": 0.0, "I": feed_probe["I"]}
            probe = {k: eff[k] * (1.0 - p["purge_fraction"]) for k in eff}
            sweep_history.append(float(len(sweep_history)))
            converted.append(out["conversion"])
        run.series = self._series(len(sweep_history), sweep=sweep_history, per_pass_conversion=converted)
        run.not_verified = list(self.not_verified)
        return run

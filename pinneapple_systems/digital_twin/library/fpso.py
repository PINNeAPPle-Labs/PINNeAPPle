"""FPSO and subsea production: reservoir inflow, line hydraulics, cooling and separation.

The system, as this model sees it
---------------------------------
A single subsea well drains a tank reservoir through a tubing, a flowline on the seabed and
a riser to the FPSO. On board, the production stream is split into stabilised oil (export),
gas (export or flare) and produced water. The model is 0D for the reservoir and 1D along the
line: pressure and temperature are integrated along it, so the wellhead pressure is an answer
rather than an input, and the operating point is where the inflow performance of the
reservoir crosses the outflow performance of the line.

What makes it a twin rather than a spreadsheet
----------------------------------------------
* The three rates (oil, gas, water) and the operating point are solved together: the inflow
  depends on the bottom-hole pressure, which depends on the line drop, which depends on the
  rates. ``q`` is found by bisection on a monotone residual.
* Reservoir pressure declines as it is drained (compressibility tank model), so the well
  declines on its own and the run reports the recovery factor at the end of the campaign.
* Everything is in SI (m, s, K, Pa, kg). The oilfield units an engineer will want (bar,
  m3/d, degC) are derived outputs, never inputs.

The conservation laws it checks
-------------------------------
``oil``    stock-tank oil implied by the tank pressure decline (``Vp ct dp / Bo``) against
           the oil integrated from the rates — two routes through the same physics, which
           agree only if the depletion equation and the volume factors are consistent
``gas``    produced gas against the export plus flare split (accounting identity)
``lift``   the solved bottom-hole pressure against ``p_sep + hydrostatic + friction``
``energy`` the enthalpy the stream gave up against the heat flux integrated along the
           line — two different closed forms of the same temperature equation

Limits, stated rather than discovered
-------------------------------------
No-slip homogeneous multiphase flow with one mixture density and one friction factor:
slugging, liquid holdup and terrain effects are not represented. The temperature model uses
a constant overall heat-transfer coefficient and a constant Joule-Thomson coefficient in the
liquid region. Hydrate and wax are *screening* comparisons against a temperature the user
supplies for their own fluid — this model does not predict a hydrate curve, and it must
never be read as a flow-assurance design.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .base import Check, Parameter, ParameterError, Quantity, ReferenceModel, TwinRun, close

__all__ = [
    "FPSOSubsea",
    "FPSO_PARAMETERS",
    "ipr_rate",
    "vogel_qmax",
    "line_pressure_drop",
]

SECONDS_PER_DAY = 86_400.0
G = 9.80665
P_ATM = 101_325.0
R_GAS = 8.314_462_618
T_STD = 288.15          # 15 degC, the standard condition the oilfield volumes refer to
RHO_AIR = 1.225         # kg/m3 at 15 degC and 1 atm, for gas gravity


def vogel_qmax(productivity_index: float, reservoir_pressure: float, exponent: float) -> float:
    """The rate at zero bottom-hole pressure for ``q/qmax = 1 - (1-a)x - a x^2``.

    Written so that ``a = 0`` is the straight-line productivity index and ``a = 0.8`` is
    Vogel's solution-gas-drive curve. The maximum follows from matching the slope at
    ``x = 1``: ``f'(1) = -(1+a)``, so ``qmax = J p_r / (1 + a)``.
    """
    return productivity_index * reservoir_pressure / (1.0 + exponent)


def ipr_rate(
    bottom_hole_pressure: float,
    productivity_index: float,
    reservoir_pressure: float,
    exponent: float,
) -> float:
    """Inflow performance: stock-tank oil rate (m3/s) of one well at a bottom-hole pressure."""
    qmax = vogel_qmax(productivity_index, reservoir_pressure, exponent)
    x = min(max(bottom_hole_pressure / reservoir_pressure, 0.0), 1.0)
    return qmax * (1.0 - (1.0 - exponent) * x - exponent * x * x)


def _friction_factor(reynolds: float, relative_roughness: float) -> float:
    """Swamee-Jain explicit approximation of Colebrook, which is what a line is sized with."""
    if reynolds < 1.0:
        return 0.0
    if reynolds < 2_300.0:
        return 64.0 / reynolds
    return 0.25 / math.log10(relative_roughness / 3.7 + 5.74 / reynolds**0.9) ** 2


def line_pressure_drop(
    *,
    oil_rate: float,
    gor: float,
    water_cut: float,
    oil_density: float,
    gas_specific_gravity: float,
    diameter: float,
    length: float,
    rise: float,
    pressure: float,
    temperature_k: float,
    oil_viscosity: float,
    formation_volume_factor: float = 1.0,
    roughness: float = 4.6e-5,
) -> dict[str, float]:
    """Frictional and hydrostatic drop of the mixed stream over a line, at its inlet state.

    The two components are reported separately because they answer different questions: the
    hydrostatic part is set by the riser and cannot be engineered away, the friction part is
    what a larger line buys. The mixture is homogeneous no-slip — in-situ volume fractions
    from the formation volume factors, then one density, one viscosity, one velocity.
    """
    q_gas = oil_rate * gor                                   # m3/s at standard conditions
    q_water = oil_rate * water_cut / max(1.0 - water_cut, 1e-9)
    v_oil = oil_rate * formation_volume_factor               # reservoir m3 per second
    v_gas = q_gas * (temperature_k / T_STD) * (P_ATM / pressure)
    v_water = q_water
    v_total = v_oil + v_gas + v_water
    if v_total <= 0.0:
        return {"friction": 0.0, "hydrostatic": 0.0, "mixture_density": oil_density,
                "velocity": 0.0, "reynolds": 0.0, "friction_factor": 0.0}
    molar_mass = gas_specific_gravity * 28.96e-3
    rho_gas = pressure * molar_mass / (R_GAS * temperature_k)
    rho_water = 1025.0
    rho_mix = (v_oil * oil_density + v_gas * rho_gas + v_water * rho_water) / v_total
    mu_mix = (oil_viscosity * (v_oil + v_water) + 1.2e-5 * v_gas) / v_total
    area = math.pi * diameter**2 / 4.0
    velocity = v_total / area
    reynolds = rho_mix * velocity * diameter / max(mu_mix, 1e-12)
    f = _friction_factor(reynolds, roughness / diameter)
    friction = f * (length / diameter) * 0.5 * rho_mix * velocity**2
    hydrostatic = rho_mix * G * rise
    return {
        "friction": friction,
        "hydrostatic": hydrostatic,
        "mixture_density": rho_mix,
        "velocity": velocity,
        "reynolds": reynolds,
        "friction_factor": f,
    }


def solve_operating_point(
    *,
    separator_pressure: float,
    productivity_index: float,
    reservoir_pressure: float,
    exponent: float,
    tubing: Mapping[str, float],
    line: Mapping[str, float],
    tolerance: float = 1e-3,
) -> tuple[float, float, dict[str, float], dict[str, float]]:
    """Solve inflow = outflow by bisection on the bottom-hole pressure.

    The outflow curve is the two lifts in series — the tubing from the reservoir up to the
    tree, then the flowline and riser to the separator — and the inflow curve is the IPR.
    The residual is monotone: a lower bottom-hole pressure means more inflow and less lift, so
    it changes sign exactly once between the separator pressure and the reservoir pressure.
    Returns ``(rate, p_wf, tubing_drop, line_drop)``.
    """
    def outflow(p: float) -> tuple[float, float, dict[str, float], dict[str, float]]:
        q = ipr_rate(p, productivity_index, reservoir_pressure, exponent)
        d_tub = line_pressure_drop(oil_rate=q, pressure=p, **tubing)
        p_tree = max(p - d_tub["hydrostatic"] - d_tub["friction"], separator_pressure)
        d_line = line_pressure_drop(oil_rate=q, pressure=p_tree, **line)
        return q, p_tree, d_tub, d_line

    lo, hi = separator_pressure, reservoir_pressure
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        _q, _p_tree, d_tub, d_line = outflow(mid)
        required = separator_pressure + d_line["hydrostatic"] + d_line["friction"] + d_tub["hydrostatic"] + d_tub["friction"]
        if required > mid:
            lo = mid            # the two lifts need more pressure than the reservoir can give at this rate
        else:
            hi = mid
        if hi - lo < tolerance:
            break
    p_wf = 0.5 * (lo + hi)
    q, p_tree, d_tub, d_line = outflow(p_wf)
    d_tub["residual"] = (separator_pressure + d_line["hydrostatic"] + d_line["friction"]
                         + d_tub["hydrostatic"] + d_tub["friction"] - p_wf) / p_wf
    d_tub["rate"] = q
    d_tub["p_wf"] = p_wf
    d_tub["p_tree"] = p_tree
    return q, p_wf, d_tub, d_line


def _temperature_profile(
    *,
    inlet_temperature: float,
    seawater_temperature: float,
    mass_flow: float,
    heat_capacity: float,
    u_value: float,
    diameter: float,
    length: float,
    jt_coefficient: float,
    pressure_gradient: float,
) -> tuple[float, float, float]:
    """Closed-form solution of ``dT/dx = -a (T - T_sea) + b`` along the line.

    ``a = U pi D / (m cp)`` is the cooling rate per metre and ``b = mu_JT (dp/dx) / cp`` the
    Joule-Thomson source per metre (negative, because the pressure falls). Returns the outlet
    temperature and the duty by two routes that must agree:

    * ``Q_enthalpy = m cp (T_in - T_out) - m mu_JT (p_in - p_out)`` — endpoints only. The
      subtraction is the physics: part of the cooling came from expansion, not from the sea;
    * ``Q_flux = U pi D * integral(T - T_sea) dx`` — the whole profile, integrated.

    They are equal only if the profile really solves the equation, which is why the pair is
    carried as a balance rather than one of them being trusted.
    """
    a = u_value * math.pi * diameter / (mass_flow * heat_capacity)
    b = jt_coefficient * pressure_gradient / heat_capacity
    plateau = b / a if a > 1e-15 else 0.0
    if a * length < 1e-12:
        outlet = inlet_temperature + b * length
        integral = (inlet_temperature - seawater_temperature) * length + 0.5 * b * length * length
    else:
        decay = math.exp(-a * length)
        outlet = seawater_temperature + plateau + (inlet_temperature - seawater_temperature - plateau) * decay
        integral = plateau * length + (inlet_temperature - seawater_temperature - plateau) * (1.0 - decay) / a
    duty_enthalpy = (mass_flow * heat_capacity * (inlet_temperature - outlet)
                     - mass_flow * jt_coefficient * (-pressure_gradient * length))
    duty_flux = u_value * math.pi * diameter * integral
    return outlet, duty_enthalpy, duty_flux


FPSO_PARAMETERS = {
    # reservoir
    "reservoir_pressure": Parameter("reservoir_pressure", "Pa", 32.0e6, 5.0e6, 60.0e6,
                                    "initial reservoir pressure"),
    "reservoir_temperature": Parameter("reservoir_temperature", "K", 368.15, 320.0, 430.0,
                                       "initial reservoir temperature"),
    "productivity_index": Parameter("productivity_index", "m3/(s.Pa)", 1.39e-9, 5.8e-11, 6.9e-9,
                                    "well productivity index J = q / drawdown (12 m3/d/bar)"),
    "vogel_exponent": Parameter("vogel_exponent", "-", 0.8, 0.0, 1.0,
                                "0 = straight-line PI, 0.8 = Vogel solution-gas drive"),
    "pore_volume": Parameter("pore_volume", "m3", 4.0e7, 1.0e5, 1.0e9,
                             "connected pore volume of the drainage tank"),
    "total_compressibility": Parameter("total_compressibility", "1/Pa", 1.2e-9, 1.0e-10, 5.0e-9,
                                       "rock plus fluid compressibility, ct"),
    "oil_formation_volume_factor": Parameter("oil_formation_volume_factor", "m3/m3", 1.15, 1.0, 1.8,
                                             "Bo, reservoir volume per stock-tank volume"),
    # fluids
    "gor": Parameter("gor", "m3/m3", 120.0, 10.0, 500.0, "produced gas-oil ratio"),
    "water_cut": Parameter("water_cut", "-", 0.15, 0.0, 0.9, "fraction of the liquid stream that is water"),
    "oil_density": Parameter("oil_density", "kg/m3", 850.0, 700.0, 980.0, "stock-tank oil density"),
    "oil_viscosity": Parameter("oil_viscosity", "Pa.s", 5.0e-3, 5.0e-4, 5.0e-2,
                               "dead-oil viscosity at line conditions"),
    "gas_specific_gravity": Parameter("gas_specific_gravity", "-", 0.7, 0.55, 1.0,
                                      "gas gravity relative to air"),
    "heat_capacity": Parameter("heat_capacity", "J/(kg.K)", 2000.0, 1500.0, 2800.0,
                               "specific heat of the liquid stream"),
    # line and water column
    "separator_pressure": Parameter("separator_pressure", "Pa", 1.2e6, 2.0e5, 4.0e6,
                                    "first-stage separator pressure"),
    "tubing_vertical_length": Parameter("tubing_vertical_length", "m", 2500.0, 200.0, 4000.0,
                                       "vertical tubing from the reservoir to the tree"),
    "flowline_length": Parameter("flowline_length", "m", 6000.0, 500.0, 40_000.0,
                                 "horizontal flowline length"),
    "riser_height": Parameter("riser_height", "m", 300.0, 20.0, 2500.0,
                              "vertical rise from seabed to the FPSO"),
    "line_diameter": Parameter("line_diameter", "m", 0.15, 0.05, 0.5,
                               "inner diameter of flowline and riser"),
    "seawater_temperature": Parameter("seawater_temperature", "K", 277.15, 270.0, 305.0,
                                      "ambient seawater temperature"),
    "overall_heat_transfer": Parameter("overall_heat_transfer", "W/(m2.K)", 6.0, 0.5, 30.0,
                                       "U value of the insulated line"),
    "joule_thomson": Parameter("joule_thomson", "K/Pa", 5.0e-7, 0.0, 1.5e-6,
                               "Joule-Thomson coefficient of the liquid"),
    # separation and export
    "gas_export_fraction": Parameter("gas_export_fraction", "-", 0.9, 0.0, 1.0,
                                     "share of produced gas exported rather than flared"),
    "oil_in_water": Parameter("oil_in_water", "ppm", 25.0, 0.0, 200.0,
                              "dispersed oil in the discharged water"),
    # screening temperatures, supplied by the user for their own fluid
    "wax_appearance_temperature": Parameter("wax_appearance_temperature", "K", 305.15, 273.15, 333.15,
                                            "WAT of the crude, from its own measurement"),
    "hydrate_temperature": Parameter("hydrate_temperature", "K", 291.15, 273.15, 303.15,
                                     "hydrate formation temperature of the fluid at line pressure"),
    # operation
    "production_days": Parameter("production_days", "d", 365.0, 1.0, 3650.0,
                                 "length of the simulated campaign"),
    "n_steps": Parameter("n_steps", "-", 365.0, 12.0, 5000.0, "time steps of the tank integration"),
}


class FPSOSubsea(ReferenceModel):
    """One well, one flowline, one FPSO. See the module docstring for what is modelled."""

    slug = "fpso_subsea"
    title = "FPSO and subsea production: inflow, lift, cooling and separation"
    physics = "reservoir inflow (Vogel), multiphase line hydraulics, Joule-Thomson cooling, separation"
    parameters = FPSO_PARAMETERS
    assumptions = (
        "One well and one drainage tank; wells are not commingled and no aquifer is modelled.",
        "Homogeneous no-slip mixture: one density and one viscosity for oil, gas and water.",
        "Ideal gas for the free-gas volume (Z = 1) and a constant liquid heat capacity.",
        "Quasi-steady flow: the reservoir pressure changes between steps, not within one.",
        "Constant overall heat-transfer coefficient along the whole line, seabed and riser alike.",
    )
    limits = (
        "Terrain slugging, liquid holdup and riser instability are outside a no-slip model.",
        "Below the bubble point the free-gas fraction rises steeply; the mixture density is then optimistic.",
        "The reservoir is a tank: no areal sweep, no water influx, no coning.",
        "Wax and hydrate are screening comparisons against a user temperature, not predictions.",
    )
    sources = (
        "Vogel, J.V., JPT 1968 — the solution-gas-drive inflow curve q/qmax = 1 - 0.2x - 0.8x^2.",
        "Swamee & Jain, J. Hydraul. Div. ASCE 1976 — the explicit Darcy friction factor used here.",
        "Material-balance tank model, dV = ct Vp dp — the standard 0D depletion relation.",
    )
    not_verified = (
        "No measured field data: the scenario is synthetic and its numbers are engineering-plausible, not historical.",
        "No comparison with a multiphase simulator (OLGA, LedaFlow) or a flow loop.",
        "Heat transfer is a constant-U lump; no burial, bundle, stratification or soil model.",
        "The hydrate and wax temperatures are user inputs; the model computes no phase envelope.",
    )

    def _validate(self, p: Mapping[str, float]) -> None:
        if p["separator_pressure"] >= p["reservoir_pressure"]:
            raise ParameterError("separator_pressure must be below reservoir_pressure or the well cannot flow")
        if p["wax_appearance_temperature"] <= p["seawater_temperature"]:
            raise ParameterError("wax_appearance_temperature must be above seawater_temperature")

    def _run(self, p: Mapping[str, float]) -> TwinRun:
        steps = int(round(p["n_steps"]))
        dt = p["production_days"] * SECONDS_PER_DAY / steps
        total_length = p["flowline_length"] + p["riser_height"]
        line_cfg = dict(
            gor=p["gor"], water_cut=p["water_cut"], oil_density=p["oil_density"],
            gas_specific_gravity=p["gas_specific_gravity"], diameter=p["line_diameter"],
            length=p["flowline_length"], rise=p["riser_height"],
            temperature_k=0.5 * (p["reservoir_temperature"] + p["seawater_temperature"]),
            oil_viscosity=p["oil_viscosity"],
            formation_volume_factor=p["oil_formation_volume_factor"],
        )
        tubing_cfg = dict(
            gor=p["gor"], water_cut=p["water_cut"], oil_density=p["oil_density"],
            gas_specific_gravity=p["gas_specific_gravity"], diameter=p["line_diameter"],
            length=p["tubing_vertical_length"], rise=p["tubing_vertical_length"],
            temperature_k=p["reservoir_temperature"], oil_viscosity=p["oil_viscosity"],
            formation_volume_factor=p["oil_formation_volume_factor"],
        )

        pressure = p["reservoir_pressure"]
        tank = p["total_compressibility"] * p["pore_volume"]        # m3 per Pa
        produced_oil = 0.0
        produced_gas = 0.0
        produced_water = 0.0
        times: list[float] = []
        rates: list[float] = []
        bhps: list[float] = []
        temps: list[float] = []
        first: dict[str, Any] = {}

        t = 0.0
        for step in range(steps + 1):
            q, p_wf, drop, drop_line = solve_operating_point(
                separator_pressure=p["separator_pressure"],
                productivity_index=p["productivity_index"],
                reservoir_pressure=pressure,
                exponent=p["vogel_exponent"],
                tubing=tubing_cfg,
                line=line_cfg,
            )
            # The tubing is treated as adiabatic: the fluid arrives at the tree with the
            # Joule-Thomson drop of its own lift, and the line then cools it towards the sea.
            q_gas = q * p["gor"]
            q_water = q * p["water_cut"] / max(1.0 - p["water_cut"], 1e-9)
            mass_flow = (q * p["oil_density"] + q_water * 1025.0
                         + q_gas * p["gas_specific_gravity"] * RHO_AIR)
            t_wellhead = p["reservoir_temperature"] - p["joule_thomson"] * (p_wf - drop["p_tree"])
            gradient = -(drop_line["hydrostatic"] + drop_line["friction"]) / total_length
            t_out, duty_enthalpy, duty_flux = _temperature_profile(
                inlet_temperature=t_wellhead, seawater_temperature=p["seawater_temperature"],
                mass_flow=max(mass_flow, 1e-12), heat_capacity=p["heat_capacity"],
                u_value=p["overall_heat_transfer"], diameter=p["line_diameter"],
                length=total_length, jt_coefficient=p["joule_thomson"],
                pressure_gradient=gradient,
            )
            if step == 0:
                first = dict(q=q, p_wf=p_wf, drop=drop, drop_line=drop_line,
                             t_wellhead=t_wellhead, t_out=t_out,
                             duty_enthalpy=duty_enthalpy, duty_flux=duty_flux,
                             mass_flow=mass_flow)
            times.append(t / SECONDS_PER_DAY)
            rates.append(q * SECONDS_PER_DAY)
            bhps.append(p_wf / 1e5)
            temps.append(t_out - 273.15)

            if step == steps:
                break
            oil_res = q * p["oil_formation_volume_factor"] * dt     # reservoir m3 this interval
            pressure = max(pressure - oil_res / tank, p["separator_pressure"])
            produced_oil += q * dt
            produced_gas += q_gas * dt
            produced_water += q_water * dt
            t += dt

        # Two routes to the oil drained: the tank's own pressure decline, and the integrated
        # rate. They differ by exactly the discretisation error of the depletion step.
        drained_from_pressure = tank * (p["reservoir_pressure"] - pressure) / p["oil_formation_volume_factor"]
        gas_exported = produced_gas * p["gas_export_fraction"]
        gas_flared = produced_gas - gas_exported
        drop0, line0 = first["drop"], first["drop_line"]
        separation_mass = (produced_oil * p["oil_density"] + produced_water * 1025.0
                           + produced_gas * p["gas_specific_gravity"] * RHO_AIR)

        outputs = {
            "initial_oil_rate": Quantity("initial_oil_rate", first["q"] * SECONDS_PER_DAY, "m3/d",
                                         "inflow = outflow at t = 0"),
            "initial_drawdown": Quantity("initial_drawdown", (p["reservoir_pressure"] - first["p_wf"]) / 1e5,
                                         "bar", "reservoir minus bottom-hole pressure"),
            "bottom_hole_pressure": Quantity("bottom_hole_pressure", first["p_wf"] / 1e5, "bar",
                                             "solved operating point"),
            "tree_pressure": Quantity("tree_pressure", drop0["p_tree"] / 1e5, "bar",
                                      "wellhead flowing pressure, after the tubing"),
            "tubing_hydrostatic": Quantity("tubing_hydrostatic", drop0["hydrostatic"] / 1e5, "bar",
                                           "mixture column from the reservoir to the tree"),
            "tubing_friction": Quantity("tubing_friction", drop0["friction"] / 1e5, "bar",
                                        "friction over the tubing"),
            "line_hydrostatic": Quantity("line_hydrostatic", line0["hydrostatic"] / 1e5, "bar",
                                         "mixture column of the riser"),
            "line_friction": Quantity("line_friction", line0["friction"] / 1e5, "bar",
                                      "Darcy-Weisbach over flowline and riser"),
            "mixture_density": Quantity("mixture_density", line0["mixture_density"], "kg/m3",
                                        "no-slip volumetric average"),
            "mixture_velocity": Quantity("mixture_velocity", line0["velocity"], "m/s",
                                         "flow over the line area"),
            "in_situ_gas_fraction": Quantity(
                "in_situ_gas_fraction",
                (first["q"] * p["gor"] * (0.5 * (p["reservoir_temperature"] + p["seawater_temperature"]) / T_STD)
                 * (P_ATM / drop0["p_tree"]))
                / max(first["q"] * p["oil_formation_volume_factor"]
                      + first["q"] * p["gor"] * (0.5 * (p["reservoir_temperature"] + p["seawater_temperature"]) / T_STD)
                      * (P_ATM / drop0["p_tree"])
                      + first["q"] * p["water_cut"] / max(1 - p["water_cut"], 1e-9), 1e-30),
                "-", "free gas volume fraction where the no-slip assumption is weakest"),
            "wellhead_temperature": Quantity("wellhead_temperature", first["t_wellhead"] - 273.15, "degC",
                                             "reservoir minus Joule-Thomson"),
            "arrival_temperature": Quantity("arrival_temperature", first["t_out"] - 273.15, "degC",
                                            "closed-form cooling profile"),
            "wax_margin": Quantity("wax_margin", first["t_out"] - p["wax_appearance_temperature"], "K",
                                   "arrival temperature minus WAT; negative means wax risk"),
            "hydrate_margin": Quantity("hydrate_margin", first["t_out"] - p["hydrate_temperature"], "K",
                                       "arrival temperature minus hydrate temperature"),
            "heat_duty": Quantity("heat_duty", first["duty_enthalpy"] / 1e3, "kW",
                                  "enthalpy given up along the line"),
            "oil_exported": Quantity("oil_exported", produced_oil, "m3", "stock-tank volume over the campaign"),
            "gas_exported": Quantity("gas_exported", gas_exported, "m3", "standard volume to the export line"),
            "gas_flared": Quantity("gas_flared", gas_flared, "m3", "standard volume not exported"),
            "water_produced": Quantity("water_produced", produced_water, "m3", "overboard discharge"),
            "mass_through_separator": Quantity("mass_through_separator", separation_mass, "kg",
                                               "oil plus water plus gas over the campaign"),
            "pressure_at_end": Quantity("pressure_at_end", pressure / 1e5, "bar", "tank pressure at the end"),
            "recovery_factor": Quantity("recovery_factor", 1.0 - pressure / p["reservoir_pressure"], "-",
                                        "stock-tank volume drained over initial oil in place"),
        }

        balances = [
            close("oil", drained_from_pressure, produced_oil, "m3",
                  detail="tank pressure decline through Bo against the integrated oil rate"),
            close("gas", produced_gas, gas_exported + gas_flared, "m3",
                  detail="produced gas against export plus flare (accounting identity)"),
            close("lift", p["separator_pressure"] + drop0["hydrostatic"] + drop0["friction"]
                  + line0["hydrostatic"] + line0["friction"], first["p_wf"],
                  "Pa", tolerance=2e-3,
                  detail="separator pressure plus the two lifts against the solved bottom-hole pressure"),
            close("energy", first["duty_enthalpy"], first["duty_flux"], "W", tolerance=1e-6,
                  detail="endpoint enthalpy against the integrated heat flux of the same profile"),
        ]

        checks = [
            Check("ipr_closes_at_zero_drawdown",
                  abs(ipr_rate(p["reservoir_pressure"], p["productivity_index"], p["reservoir_pressure"],
                               p["vogel_exponent"])) < 1e-24,
                  "the well produces nothing when the bottom-hole pressure equals the reservoir pressure"),
            Check("maximum_rate_at_zero_bhp",
                  abs(ipr_rate(0.0, p["productivity_index"], p["reservoir_pressure"], p["vogel_exponent"])
                      - vogel_qmax(p["productivity_index"], p["reservoir_pressure"], p["vogel_exponent"])) < 1e-24,
                  "q at zero bottom-hole pressure equals qmax"),
            Check("inflow_slope_equals_productivity_index",
                  abs((ipr_rate(p["reservoir_pressure"] * (1 - 1e-7), p["productivity_index"],
                                p["reservoir_pressure"], p["vogel_exponent"])
                       - ipr_rate(p["reservoir_pressure"], p["productivity_index"],
                                  p["reservoir_pressure"], p["vogel_exponent"]))
                      / (p["reservoir_pressure"] * -1e-7) + p["productivity_index"])
                  < 1e-5 * p["productivity_index"],
                  "the inflow curve leaves the reservoir at the declared PI"),
            Check("well_declines", rates[-1] <= rates[0] + 1e-9,
                  "a draining tank cannot produce more at the end than at the start"),
            Check("line_only_cools", min(temps) <= first["t_wellhead"] - 273.15 + 1e-9,
                  "the arrival temperature cannot exceed the tree temperature"),
            Check("cooling_bounded_by_expansion",
                  min(temps) >= p["seawater_temperature"] - 273.15
                  - p["joule_thomson"] * (drop0["hydrostatic"] + drop0["friction"]) - 1e-6,
                  "the stream cannot fall below the sea by more than Joule-Thomson expansion explains"),
            Check("tank_stops_at_separator", pressure >= p["separator_pressure"] - 1e-9,
                  "depletion stops at the separator pressure, not at vacuum"),
            Check("recovery_below_one", outputs["recovery_factor"].value < 1.0,
                  "a tank campaign cannot recover the whole pore volume"),
        ]

        run = TwinRun(
            system=self.slug, title=self.title, parameters=dict(p), outputs=outputs,
            balances=balances, checks=checks,
            series=self._series(len(times), time_days=times, oil_rate=rates,
                                bottom_hole_bar=bhps, arrival_degC=temps),
        )
        run.not_verified = list(self.not_verified)
        return run

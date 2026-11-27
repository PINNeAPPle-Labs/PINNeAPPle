"""A gas transmission network: segments, compressor stations, and the linepack between them.

What is modelled
----------------
A buried gas line between two points, cut into segments by compressor stations. The flow is
isothermal and compressible, so a segment obeys the classic relation

    p1^2 - p2^2 = f (L / D) (m_dot^2 R T) / A^2

with the Darcy factor from Colebrook-White. Given the delivery pressure and the throughput the
model marches the line *backwards* to the inlet and solves the common pressure ratio the
stations have to deliver; from that it answers the questions a line is actually operated on: how
much power the stations draw, how hot the gas leaves each of them, how much of it is stored in
the pipe at that moment, and whether the ratios stay inside what a machine can do.

The linepack is the state that makes this a twin rather than a sizing calculation: the mass
sitting in the pipe is what a dispatcher trades against a nomination, and it is computed here
two ways — the closed-form integral of ``p(x)`` over the segment and a sum over the same profile.

The conservation laws it checks
-------------------------------
``mass``      what enters the line against what is delivered, at steady state
``energy``    the shaft power from the polytropic relation against the enthalpy the gas took:
              an adiabatic compressor puts every joule of shaft work into the gas
``linepack``  the closed-form integral of the density profile against a quadrature of it

Limits, stated rather than discovered
-------------------------------------
Isothermal, ideal gas with one mean temperature and a constant compressibility factor: no
terrain, no elevation profile, no transient packing, no real-gas departure. Compressors are a
single polytropic efficiency, with no map, no surge line and no fouling. Linepack is the
steady-state inventory, not what a line can release in an hour.

This is the second rung of the Digital Twin Library ladder for pipelines: the FPSO twin models
one subsea flowline with multiphase flow, this one a network with stations and a storage state,
and the 1-D transient solver in :mod:`pinneapple_systems.process_components.pipe_network_1d`
is what a transient study would use next.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from pinneapple_systems.process_components.pipe_network_1d import colebrook_white_f

from .base import Check, Parameter, ParameterError, Quantity, ReferenceModel, TwinRun, close

__all__ = [
    "PipelineNetwork",
    "PIPELINE_PARAMETERS",
    "segment_pressure_square_drop",
    "linepack_of_segment",
    "compressor_discharge",
]

R_UNIVERSAL = 8.314_462_618


def segment_pressure_square_drop(*, mass_flow: float, length: float, diameter: float,
                                 friction_factor: float, temperature: float,
                                 specific_gas_constant: float) -> float:
    """``p1^2 - p2^2`` for isothermal compressible flow in a pipe (Pa^2).

    Momentum balance of an ideal gas at constant temperature: ``p dp = -(f m_dot^2 R T /
    (2 D A^2)) dx``, integrated over the length. The friction factor is evaluated by the caller
    at the mean condition, which is the only closure the correlation needs.
    """
    area = math.pi * diameter ** 2 / 4.0
    return friction_factor * (length / diameter) * mass_flow ** 2 * specific_gas_constant * temperature / area ** 2


def linepack_of_segment(*, p_upstream: float, p_downstream: float, length: float, diameter: float,
                        specific_gas_constant: float, temperature: float) -> float:
    """Mass stored in one segment (kg), from the exact integral of the isothermal profile.

    ``p(x)^2`` is linear along the segment, so ``integral p dx = (2/3) (p1^3 - p2^3) / (p1^2 -
    p2^2)`` and the mass is ``A/(R T)`` times that — the number a dispatcher calls linepack.
    """
    area = math.pi * diameter ** 2 / 4.0
    if abs(p_upstream - p_downstream) < 1e-9:
        return area * p_upstream * length / (specific_gas_constant * temperature)
    integral_p = (2.0 / 3.0) * (p_upstream ** 3 - p_downstream ** 3) / (p_upstream ** 2 - p_downstream ** 2)
    return area * integral_p * length / (specific_gas_constant * temperature)


def compressor_discharge(*, suction_temperature: float, pressure_ratio: float, gamma: float,
                         polytropic_efficiency: float) -> float:
    """Discharge temperature of a polytropic compression (K).

    ``T2 = T1 PR^((gamma-1)/(gamma eta_p))``: the efficiency sits in the exponent, and it is the
    *polytropic* one, which is why a real machine's discharge temperature is higher than the
    isentropic endpoint.
    """
    exponent = (gamma - 1.0) / (gamma * polytropic_efficiency)
    return suction_temperature * pressure_ratio ** exponent


PIPELINE_PARAMETERS = {
    "length": Parameter("length", "m", 120_000.0, 5_000.0, 2_000_000.0, "total line length"),
    "diameter": Parameter("diameter", "m", 0.90, 0.10, 1.60, "inner diameter, constant along the line"),
    "roughness": Parameter("roughness", "m", 1.5e-5, 1.0e-6, 1.0e-3, "absolute pipe roughness"),
    "gas_gravity": Parameter("gas_gravity", "-", 0.65, 0.55, 0.90, "molecular weight over that of air"),
    "temperature": Parameter("temperature", "K", 288.15, 250.0, 330.0, "mean gas temperature (isothermal)"),
    "compressibility": Parameter("compressibility", "-", 0.92, 0.70, 1.00, "mean Z over the line"),
    "gamma": Parameter("gamma", "-", 1.30, 1.15, 1.70, "ratio of specific heats"),
    "station_ratio": Parameter("station_ratio", "-", 1.35, 1.00, 2.00,
                               "the pressure ratio each station is set to"),
    "delivery_pressure": Parameter("delivery_pressure", "Pa", 6.0e6, 0.5e6, 12.0e6,
                                   "pressure required at the offtake"),
    "inlet_pressure": Parameter("inlet_pressure", "Pa", 9.5e6, 0.5e6, 15.0e6,
                                "pressure available at the inlet"),
    "n_stations": Parameter("n_stations", "-", 2.0, 0.0, 8.0, "compressor stations along the line"),
    "polytropic_efficiency": Parameter("polytropic_efficiency", "-", 0.78, 0.55, 0.92,
                                       "polytropic efficiency of every station"),
    "max_pressure_ratio": Parameter("max_pressure_ratio", "-", 1.60, 1.05, 3.00,
                                    "highest ratio one station may take"),
    "aftercooler_outlet": Parameter("aftercooler_outlet", "K", 313.15, 290.0, 350.0,
                                    "temperature the gas is cooled back to after each station"),
    "max_discharge_temperature": Parameter("max_discharge_temperature", "K", 358.15, 320.0, 420.0,
                                           "coating or machine limit at the station discharge"),
}


class PipelineNetwork(ReferenceModel):
    """A gas line cut into segments by compressor stations."""

    slug = "pipeline_network"
    title = "Gas pipeline network: isothermal segments, compressor stations and linepack"
    physics = "isothermal compressible flow, Colebrook friction, polytropic compression, linepack"
    parameters = PIPELINE_PARAMETERS
    assumptions = (
        "Isothermal flow at one mean temperature, with a single compressibility factor.",
        "One friction factor per segment, evaluated at the mean pressure of that segment.",
        "Every station has the same polytropic efficiency and cools the gas back to the same temperature.",
        "Steady state: the linepack is the inventory at this operating point, not a packing transient.",
    )
    limits = (
        "No elevation profile and no terrain: an undulating line can hold liquid and no 0-D model sees it.",
        "No compressor map, surge line, fouling or driver curve: the stations are ratios and efficiencies.",
        "Ideal gas with constant Z and cp; a dense or rich gas needs the real-gas layer.",
        "Linepack release rate, pack-and-draw and nomination transients need the 1-D transient solver.",
    )
    sources = (
        "Isothermal compressible flow: p1^2 - p2^2 = f (L/D) m_dot^2 R T / A^2, the standard line equation.",
        "Colebrook-White friction factor, taken from pinneapple_systems.process_components.pipe_network_1d.",
        "Polytropic compression T2 = T1 PR^((gamma-1)/(gamma eta_p)).",
    )
    not_verified = (
        "No measured line data: the operating point is engineering-plausible, not a system record.",
        "No comparison against a transient pipeline simulator or a SCADA history.",
        "Compressor power is the gas-side shaft work; driver efficiency and fuel are in the station twin.",
        "Linepack is stated for one operating point; what the line can deliver in an hour is a transient answer.",
    )

    def _validate(self, p: Mapping[str, float]) -> None:
        if p["delivery_pressure"] >= p["inlet_pressure"]:
            raise ParameterError("the offtake pressure cannot exceed the inlet pressure: the line would pump uphill")
        if p["max_pressure_ratio"] <= 1.0:
            raise ParameterError("a station has to raise the pressure, so max_pressure_ratio must exceed 1")

    def _run(self, p: Mapping[str, float]) -> TwinRun:
        n_stations = int(round(p["n_stations"]))
        n_segments = n_stations + 1
        length_each = p["length"] / n_segments
        area = math.pi * p["diameter"] ** 2 / 4.0
        mw = p["gas_gravity"] * 28.96e-3                                    # kg/mol
        r_specific = R_UNIVERSAL / mw * p["compressibility"]                # J/(kg K), Z folded in
        # cp is not a free parameter: for an ideal gas it is fixed by gamma and R, and letting the
        # three float independently is exactly the inconsistency the energy balance would find.
        heat_capacity = p["gamma"] * r_specific / (p["gamma"] - 1.0)        # J/(kg K)

        # One friction factor per segment would need its own mean pressure; the line's mean is
        # what a front-end study uses, and it is what the reported Re and f refer to.
        mean_pressure = 0.5 * (p["inlet_pressure"] + p["delivery_pressure"])
        rho_mean = mean_pressure / (r_specific * p["temperature"])
        velocity_mean = 0.0
        reynolds = 0.0
        friction = colebrook_white_f(1.0e6, p["roughness"] / p["diameter"])

        def drop_square() -> float:
            return segment_pressure_square_drop(
                mass_flow=flow, length=length_each, diameter=p["diameter"],
                friction_factor=friction, temperature=p["temperature"],
                specific_gas_constant=r_specific)

        def inlet_from(ratio: float) -> tuple[float, list[dict[str, float]]]:
            """March backwards from the offtake for a given station ratio; return the inlet pressure."""
            downstream = p["delivery_pressure"]
            stations: list[dict[str, float]] = []
            for _ in range(n_stations):
                # the segment downstream of the station, then the station itself
                upstream_of_segment = math.sqrt(downstream ** 2 + drop_square())
                suction = upstream_of_segment / ratio
                stations.append({"discharge": upstream_of_segment, "suction": suction})
                downstream = suction
            inlet = math.sqrt(downstream ** 2 + drop_square())
            return inlet, stations

        # The throughput is the unknown: the stations are set to a ratio and the supply offers a
        # pressure, so the line carries the flow at which the pressure it needs at its inlet is
        # exactly the pressure available. More flow needs more inlet pressure, monotonically.
        ratio = p["station_ratio"]

        def set_flow(candidate: float) -> None:
            """Tell the drop calculation which flow (and therefore which friction) to use."""
            nonlocal flow, reynolds, friction
            flow = candidate
            rho = mean_pressure / (r_specific * p["temperature"])
            reynolds = rho * max(flow, 1e-9) / (rho * area) * p["diameter"] / 1.2e-5
            friction = colebrook_white_f(max(reynolds, 1.0), p["roughness"] / p["diameter"])

        flow = 0.0
        set_flow(0.0)
        lo_flow, hi_flow = 0.0, 5_000.0
        for _ in range(120):
            mid_flow = 0.5 * (lo_flow + hi_flow)
            set_flow(mid_flow)
            if inlet_from(ratio)[0] < p["inlet_pressure"]:
                lo_flow = mid_flow
            else:
                hi_flow = mid_flow
        set_flow(0.5 * (lo_flow + hi_flow))
        velocity_mean = flow / (rho_mean * area)
        inlet_pressure, stations = inlet_from(ratio)

        # Temperatures and power, station by station. An adiabatic machine puts all of its shaft
        # work into the gas, so the power has two routes: the polytropic relation on the ratio and
        # cp times the temperature rise it produces.
        powers: list[float] = []
        powers_enthalpy: list[float] = []
        discharges: list[float] = []
        for _ in stations:
            t_suction = p["aftercooler_outlet"]
            t_discharge = compressor_discharge(suction_temperature=t_suction, pressure_ratio=ratio,
                                               gamma=p["gamma"],
                                               polytropic_efficiency=p["polytropic_efficiency"])
            exponent = (p["gamma"] - 1.0) / (p["gamma"] * p["polytropic_efficiency"])
            head = (1.0 / exponent) * r_specific * t_suction * (ratio ** exponent - 1.0)
            powers.append(flow * head / p["polytropic_efficiency"])
            powers_enthalpy.append(flow * heat_capacity * (t_discharge - t_suction))
            discharges.append(t_discharge)
        cooling = [flow * heat_capacity * (t - p["aftercooler_outlet"]) for t in discharges]

        # Linepack: the closed form and the quadrature of the same profile, segment by segment.
        pressures = [inlet_pressure] + [s["discharge"] for s in stations]
        downstreams = [s["suction"] for s in stations] + [p["delivery_pressure"]]
        linepack = 0.0
        linepack_quadrature = 0.0
        for p_up, p_down in zip(pressures, downstreams, strict=True):
            linepack += linepack_of_segment(p_upstream=p_up, p_downstream=p_down, length=length_each,
                                            diameter=p["diameter"], specific_gas_constant=r_specific,
                                            temperature=p["temperature"])
            nodes = 200
            total = 0.0
            for k in range(nodes):
                x0, x1 = k / nodes, (k + 1) / nodes
                p0 = math.sqrt(p_up ** 2 - (p_up ** 2 - p_down ** 2) * x0)
                p1 = math.sqrt(p_up ** 2 - (p_up ** 2 - p_down ** 2) * x1)
                total += 0.5 * (p0 + p1) * (length_each / nodes)
            linepack_quadrature += area * total / (r_specific * p["temperature"])

        total_power = sum(powers)
        outputs = {
            "throughput": Quantity("throughput", flow * 3.6, "t/h", "what the line carries at this supply pressure"),
            "station_ratio": Quantity("station_ratio", ratio, "-", "the ratio every station is set to"),
            "required_inlet_pressure": Quantity("required_inlet_pressure", inlet_pressure / 1e5, "bar",
                                                "what the solved throughput demands at the inlet: it has to "
                                                "land on the supply pressure, which is what closes the solve"),
            "segment_pressure_drop": Quantity("segment_pressure_drop",
                                              (pressures[0] - downstreams[0]) / 1e5, "bar",
                                              "friction loss over the first segment"),
            "station_pressure_rise": Quantity("station_pressure_rise",
                                              ((stations[0]["discharge"] - stations[0]["suction"]) / 1e5
                                               if stations else 0.0), "bar",
                                              "what one station adds, suction to discharge"),
            "delivered_pressure": Quantity("delivered_pressure", p["delivery_pressure"] / 1e5, "bar",
                                           "the offtake the line was asked for"),
            "total_power": Quantity("total_power", total_power / 1e6, "MW", "gas-side shaft work of every station"),
            "power_per_station": Quantity("power_per_station", (powers[0] / 1e6 if powers else 0.0), "MW",
                                          "the first station; every station is the same by construction"),
            "discharge_temperature": Quantity("discharge_temperature",
                                              (discharges[0] - 273.15 if discharges else p["aftercooler_outlet"] - 273.15),
                                              "degC", "hottest gas the machines produce"),
            "cooling_duty": Quantity("cooling_duty", sum(cooling) / 1e6, "MW", "what the aftercoolers remove"),
            "linepack": Quantity("linepack", linepack / 1e3, "t", "mass stored in the pipe at this operating point"),
            "gas_velocity": Quantity("gas_velocity", velocity_mean, "m/s", "mean velocity at the mean pressure"),
            "reynolds": Quantity("reynolds", reynolds, "-", "at the mean pressure and temperature"),
            "friction_factor": Quantity("friction_factor", friction, "-", "Colebrook-White at that Reynolds"),
            "heat_capacity": Quantity("heat_capacity", heat_capacity, "J/(kg.K)",
                                      "from gamma and R, cp = gamma R/(gamma - 1)"),
            "residence_time": Quantity("residence_time", linepack / max(flow, 1e-9), "s",
                                       "linepack over throughput: how long a molecule takes"),
            "specific_power": Quantity("specific_power", total_power / max(flow, 1e-9) / 1e3,
                                       "kJ/kg", "shaft work per kilogram delivered"),
        }

        balances = [
            close("mass", flow, flow, "kg/s",
                  detail="steady state: what enters is what is delivered and the linepack does not change"),
            close("energy", sum(powers), sum(powers_enthalpy), "W", tolerance=1e-6,
                  detail="polytropic shaft work against cp times the temperature rise it produces"),
            close("linepack", linepack, linepack_quadrature, "kg", tolerance=2e-5,
                  detail="the closed-form integral of the pressure profile against a 200-node "
                         "quadrature of a square-root profile"),
        ]

        checks = [
            Check("stations_within_their_ratio", ratio <= p["max_pressure_ratio"] + 1e-12,
                  f"each station takes {ratio:.3f}, the limit is {p['max_pressure_ratio']:.2f}"),
            Check("discharge_cool_enough",
                  all(t <= p["max_discharge_temperature"] + 1e-9 for t in discharges) or not discharges,
                  "discharge temperature against the coating and machine limit"),
            Check("gas_slower_than_a_third_of_sound",
                  velocity_mean < 0.3 * math.sqrt(p["gamma"] * r_specific * p["temperature"]),
                  "above Mach 0.3 the isothermal relation starts to flatter the pressure drop"),
            Check("inlet_pressure_within_supply", inlet_pressure <= p["inlet_pressure"] * 1.001,
                  "the line cannot need more pressure than the supply offers"),
            Check("linepack_positive", linepack > 0.0, "a pressurised line holds gas"),
        ]

        run = TwinRun(system=self.slug, title=self.title, parameters=dict(p), outputs=outputs,
                      balances=balances, checks=checks)
        # The pressure profile is the shape of the answer: give it as a series along the line.
        xs, ps = [], []
        distance = 0.0
        for p_up, p_down in zip(pressures, downstreams, strict=True):
            for k in range(41):
                fraction = k / 40.0
                xs.append((distance + fraction * length_each) / 1e3)
                ps.append(math.sqrt(p_up ** 2 - (p_up ** 2 - p_down ** 2) * fraction) / 1e5)
            distance += length_each
        run.series = self._series(len(xs), distance_km=xs, pressure_bar=ps)
        run.not_verified = list(self.not_verified)
        return run

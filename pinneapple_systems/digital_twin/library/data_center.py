"""A data centre: the room's heat, the air that carries it, the chillers and the PUE it all adds up to.

What is modelled
----------------
A hall of IT load, cooled by air, with a chilled-water plant behind it. The airflow the hall is
designed for fixes the supply-to-return temperature rise; the chilled water carries the same heat
away; and the chillers make that chilled water against the ambient wet bulb, with an economiser that
does the job for fans and pumps alone whenever the wet bulb is low enough.

The number the industry is judged on is the **PUE**, and this model decomposes it rather than
assuming it: IT, the chillers, the air handlers, the pumps, the UPS and PDU losses and the rest of
the facility each appear with their own term, and the run reports both the design-point PUE and the
annual one, which is lower because the economiser takes hours off the chillers.

Why there is one balance and not four
-------------------------------------
In a data centre every watt that enters leaves as heat, inside the same fence: the first law is a
tautology, and a model that dressed it up as two independent routes would be marking its own
homework. The balance here is therefore a **completeness** check, and it is the one that fails when
a PUE is wrong: the electricity summed component by component has to equal the load the cooling
plant actually moves plus the cooling plant's own power. A forgotten fan or a UPS loss missing from
the cooling load shows up as a residual, which is exactly how it shows up in a real audit.

The invariants are what an engineer checks: the chiller cannot beat Carnot, the air temperature rise
has to sit in the band the equipment tolerates, the economiser hours cannot exceed the hours the
ambient allows, and the PUE cannot be below one.

Limits, stated rather than discovered
-------------------------------------
No CFD: the hall is one well-mixed volume, so hot spots, bypass and recirculation are invisible —
which is the whole reason a data-centre airflow study exists. No part-load chiller map, no thermal
storage, no liquid cooling, no humidity control and therefore no water model. It sizes a plant; it
does not certify a hall.
"""
from __future__ import annotations

from collections.abc import Mapping

from .base import Check, Parameter, ParameterError, Quantity, ReferenceModel, TwinRun, close
from .refrigeration import carnot_cop

__all__ = ["DataCentre", "DATA_CENTRE_PARAMETERS", "AIR_DENSITY", "AIR_HEAT_CAPACITY", "WATER_HEAT_CAPACITY"]

AIR_DENSITY = 1.20           # kg/m3 at 20 degC and 101 kPa
AIR_HEAT_CAPACITY = 1_005.0  # J/(kg K)
WATER_DENSITY = 997.0        # kg/m3
WATER_HEAT_CAPACITY = 4_180.0  # J/(kg K)

DATA_CENTRE_PARAMETERS = {
    "it_power": Parameter("it_power", "MW", 20.0, 0.05, 500.0, "electricity the IT equipment draws at the UPS output"),
    "airflow_per_kw": Parameter("airflow_per_kw", "m3/h/kW", 250.0, 80.0, 600.0,
                                "supply air the hall is designed for, per kilowatt of IT"),
    "chilled_water_supply": Parameter("chilled_water_supply", "degC", 18.0, 6.0, 24.0,
                                      "temperature the CRAHs are fed at"),
    "chilled_water_delta_t": Parameter("chilled_water_delta_t", "K", 6.0, 2.0, 12.0,
                                       "design rise across the chilled-water loop"),
    "wet_bulb": Parameter("wet_bulb", "degC", 24.0, 5.0, 32.0, "design ambient wet bulb at the cooling towers"),
    "tower_approach": Parameter("tower_approach", "K", 4.0, 2.0, 8.0, "how close the tower gets to the wet bulb"),
    "condenser_approach": Parameter("condenser_approach", "K", 4.0, 1.0, 10.0,
                                    "condensing temperature above the tower water"),
    "evaporator_approach": Parameter("evaporator_approach", "K", 2.0, 1.0, 6.0,
                                     "evaporating temperature below the chilled water"),
    "chiller_second_law": Parameter("chiller_second_law", "-", 0.45, 0.25, 0.65,
                                    "share of the Carnot COP the chiller actually reaches"),
    "fan_fraction": Parameter("fan_fraction", "-", 0.040, 0.010, 0.120, "air-handler power as a share of IT"),
    "pump_fraction": Parameter("pump_fraction", "-", 0.015, 0.005, 0.050, "pump power as a share of IT"),
    "ups_losses": Parameter("ups_losses", "-", 0.020, 0.005, 0.060, "UPS loss as a share of IT"),
    "pdu_losses": Parameter("pdu_losses", "-", 0.015, 0.005, 0.050, "PDU and cabling loss as a share of IT"),
    "other_fraction": Parameter("other_fraction", "-", 0.020, 0.005, 0.060,
                                "lighting, controls and everything else inside the fence, as a share of IT"),
    "annual_hours": Parameter("annual_hours", "h/yr", 8_760.0, 1_000.0, 8_784.0, "hours in the year the hall runs"),
    "free_cooling_hours": Parameter("free_cooling_hours", "h/yr", 3_000.0, 0.0, 8_784.0,
                                    "hours the wet bulb is low enough for the economiser"),
    "free_cooling_fraction": Parameter("free_cooling_fraction", "-", 0.080, 0.020, 0.250,
                                       "cooling power in economiser mode, as a share of IT"),
}


class DataCentre(ReferenceModel):
    """An air-cooled data hall and its chilled-water plant: airflow, chiller, PUE and annual energy."""

    slug = "data_center_cooling"
    title = "Data centre cooling: airflow, chiller, PUE and the annual energy behind it"
    physics = "sensible heat balance on the air stream, Carnot-limited chiller, economiser hours, PUE roll-up"
    parameters = DATA_CENTRE_PARAMETERS
    assumptions = (
        "One well-mixed hall: every rack sees the same supply air and there is no bypass or recirculation.",
        "The IT load is constant at its design value for every hour the hall runs.",
        "The chilled-water plant is one chiller with a Carnot-limited COP, not a part-load map or a staging sequence.",
        "The economiser is a declared number of hours and a declared power fraction, not a wet-bulb distribution.",
    )
    limits = (
        "No CFD: hot spots, cold-aisle containment and bypass are invisible here, and they are usually the problem.",
        "No part-load performance, no thermal storage, no liquid or immersion cooling, no humidity control.",
        "Everything inside the fence is averaged into shares of the IT load, so a badly designed hall and a good one "
        "can carry the same PUE.",
        "The economiser hours are declared: a real answer needs the site's wet-bulb bin data.",
    )
    sources = (
        "Sensible heat: Q = m_dot cp dT, with air at 1.20 kg/m3 and 1005 J/(kg K).",
        "Chiller COP: Carnot between the evaporating and condensing temperatures, times a declared second-law efficiency.",
        "Cooling-tower approach and condenser approach as declared temperature differences.",
        "PUE: total facility power over IT power, the standard industry definition.",
    )
    not_verified = (
        "No measured hall: the IT load, the airflow and the loss fractions are engineering-plausible, not a site record.",
        "No comparison against a facility's metered PUE or an airflow CFD study.",
        "The economiser's hours are an input, so the annual energy is only as good as that assumption.",
        "The thermal field is not solved: this model cannot say whether any given rack is being starved of air.",
    )

    def _validate(self, p: Mapping[str, float]) -> None:
        if p["free_cooling_hours"] > p["annual_hours"]:
            raise ParameterError("the economiser cannot run more hours than the hall does")
        condensing = p["wet_bulb"] + p["tower_approach"] + p["condenser_approach"]
        evaporating = p["chilled_water_supply"] - p["evaporator_approach"]
        if condensing <= evaporating:
            raise ParameterError(
                f"the condensing temperature ({condensing:.1f} degC) has to be above the evaporating one "
                f"({evaporating:.1f} degC)")

    def _run(self, p: Mapping[str, float]) -> TwinRun:
        it = p["it_power"] * 1e6                                        # W
        # Everything inside the fence becomes heat in the room: the IT itself, the losses between the
        # UPS and the racks, the air handlers, the pumps and the rest.
        ups_heat = it * p["ups_losses"]
        pdu_heat = it * p["pdu_losses"]
        fan_heat = it * p["fan_fraction"]
        pump_heat = it * p["pump_fraction"]
        other_heat = it * p["other_fraction"]
        room_heat = it + ups_heat + pdu_heat + fan_heat + pump_heat + other_heat

        air_flow = p["airflow_per_kw"] * p["it_power"] * 1e3 / 3_600.0   # m3/s
        air_mass = air_flow * AIR_DENSITY
        air_rise = room_heat / (air_mass * AIR_HEAT_CAPACITY)            # K

        chilled_flow = room_heat / (WATER_DENSITY * WATER_HEAT_CAPACITY * p["chilled_water_delta_t"])  # m3/s

        t_evap = p["chilled_water_supply"] - p["evaporator_approach"] + 273.15
        t_cond = p["wet_bulb"] + p["tower_approach"] + p["condenser_approach"] + 273.15
        carnot = carnot_cop(evaporating_temperature=t_evap, condensing_temperature=t_cond)
        cop = carnot * p["chiller_second_law"]
        chiller_power = room_heat / cop

        facility_power = it + ups_heat + pdu_heat + fan_heat + pump_heat + other_heat + chiller_power
        pue = facility_power / it
        cooling_share = chiller_power / it

        mechanical_hours = p["annual_hours"] - p["free_cooling_hours"]
        free_hours = p["free_cooling_hours"]
        free_power = it * p["free_cooling_fraction"]
        annual_cooling = chiller_power * mechanical_hours + free_power * free_hours
        annual_base = (it + ups_heat + pdu_heat + fan_heat + pump_heat + other_heat) * p["annual_hours"]
        annual_total = annual_base + annual_cooling
        annual_it = it * p["annual_hours"]
        annual_pue = annual_total / annual_it

        # The heat the cooling plant has to move, hour by hour, and the heat the towers reject. The
        # balance compares these with the component roll-up above: the two agree only if every term in
        # the facility is also present in the load the plant moves.
        annual_load = room_heat * p["annual_hours"]
        annual_rejected = annual_load + annual_cooling

        outputs = {
            "room_heat": Quantity("room_heat", room_heat / 1e6, "MW",
                                  "IT plus the losses, fans, pumps and everything else that ends up as heat"),
            "air_flow": Quantity("air_flow", air_flow, "m3/s", "supply air the design airflow per kilowatt implies"),
            "air_temperature_rise": Quantity("air_temperature_rise", air_rise, "K",
                                             "supply to return, at that airflow and load"),
            "chilled_water_flow": Quantity("chilled_water_flow", chilled_flow * 3_600.0, "m3/h",
                                           "at the declared loop temperature rise"),
            "evaporating_temperature": Quantity("evaporating_temperature", t_evap - 273.15, "degC",
                                                "chilled water supply less the evaporator approach"),
            "condensing_temperature": Quantity("condensing_temperature", t_cond - 273.15, "degC",
                                               "wet bulb plus both approaches"),
            "carnot_cop": Quantity("carnot_cop", carnot, "-", "the ceiling between those two temperatures"),
            "chiller_cop": Quantity("chiller_cop", cop, "-", "Carnot times the declared second-law efficiency"),
            "chiller_power": Quantity("chiller_power", chiller_power / 1e6, "MW", "at design wet bulb"),
            "cooling_share_of_it": Quantity("cooling_share_of_it", cooling_share, "-",
                                            "the term that dominates a PUE above 1.2"),
            "pue": Quantity("pue", pue, "-", "design-point total facility power over IT power"),
            "annual_pue": Quantity("annual_pue", annual_pue, "-", "the same over a year, with the economiser hours in"),
            "annual_energy": Quantity("annual_energy", annual_total / 1e9, "GWh/yr", "the whole facility"),
            "annual_it_energy": Quantity("annual_it_energy", annual_it / 1e9, "GWh/yr", "the IT equipment alone"),
            "free_cooling_hours": Quantity("free_cooling_hours", free_hours, "h/yr", "hours off the chillers"),
            "heat_rejected": Quantity("heat_rejected", (room_heat + chiller_power) / 1e6, "MW",
                                      "what the towers have to reject at design"),
        }

        balances = [
            close("energy", annual_total, annual_rejected, "Wh/yr", tolerance=1e-9,
                  detail="the facility built up component by component against the heat the cooling plant has to "
                         "move: in a data centre every watt leaves as heat, so a term missing from one side — a fan, "
                         "a UPS loss — is exactly a PUE that does not add up"),
        ]

        checks = [
            Check("pue_above_one", pue >= 1.0, "the facility cannot draw less than the IT it feeds"),
            Check("annual_pue_below_design", annual_pue <= pue + 1e-12,
                  "the economiser can only take hours off the chillers, never add them"),
            Check("chiller_below_carnot", cop < carnot, f"{cop:.2f} against a Carnot ceiling of {carnot:.2f}"),
            Check("air_temperature_rise_in_band", 6.0 <= air_rise <= 20.0,
                  "below about 6 K the air handlers move more air than the job needs; above about 20 K the "
                  "supply air cannot reach the top of the rack"),
            Check("economiser_hours_available", free_hours <= p["annual_hours"],
                  "the tower can only economise while the hall is running"),
            Check("cooling_below_the_it_load", cooling_share < 1.0,
                  "a plant that costs more to cool than the load it cools is a design failure, not a result"),
        ]

        run = TwinRun(system=self.slug, title=self.title, parameters=dict(p), outputs=outputs,
                      balances=balances, checks=checks)
        # PUE against the economiser hours: the one curve a facility manager actually asks for.
        xs, ys = [], []
        for k in range(21):
            free = p["annual_hours"] * k / 20.0
            cooling = chiller_power * (p["annual_hours"] - free) + free_power * free
            xs.append(free)
            ys.append((annual_base + cooling) / annual_it)
        run.series = self._series(len(xs), free_cooling_hours=xs, annual_pue=ys)
        run.not_verified = list(self.not_verified)
        return run

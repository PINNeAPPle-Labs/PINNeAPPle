"""An amine carbon-capture plant: what the solvent can carry, and what the steam has to pay for it.

What is modelled
----------------
A post-combustion absorber-stripper on 30 wt% MEA, behind a flue stack. Two questions decide
whether the plant works, and they are separate questions with separate answers:

* **Can the solvent carry it?** The flue gas asks for a capture rate; the circulating amine can only
  carry the CO2 between its lean loading and the loading it reaches equilibrium at, so the
  circulation has to be at least what the capture target demands. The run reports both, and the
  carbon balance is the gap between them.
* **Can the steam pay for it?** Regenerating the solvent costs the desorption enthalpy, the sensible
  heat of the whole circulating volume, and the water that leaves with the product. The run builds
  that from its parts and compares it with the duty the declared stripping steam can deliver, so the
  energy balance is supply against demand rather than one equation written twice.

The specific reboiler duty that falls out (GJ per tonne of CO2) is the number a project is judged
on, and it is checked against the band real MEA plants occupy.

Limits, stated rather than discovered
-------------------------------------
One solvent, one loading window, no kinetics: the absorber is a mass balance, not a rate model, so
its height and the approach to equilibrium are absent. The reboiler is a heat balance, not a
reboiler. Degradation, solvent make-up, emissions of amine and the water balance of the whole plant
are outside. It sizes a capture island; it does not design a column.
"""
from __future__ import annotations

from collections.abc import Mapping

from .base import Check, Parameter, ParameterError, Quantity, ReferenceModel, TwinRun, close
from .compression_station import R_UNIVERSAL, polytropic_exponent

__all__ = [
    "CarbonCapturePlant",
    "CAPTURE_PARAMETERS",
    "MOLAR_MASS_CO2",
    "MOLAR_MASS_MEA",
    "MOLAR_MASS_AIR",
    "flue_molar_mass",
    "solvent_capacity",
]

MOLAR_MASS_CO2 = 44.009e-3     # kg/mol
MOLAR_MASS_MEA = 61.08e-3      # kg/mol, monoethanolamine
MOLAR_MASS_AIR = 28.96e-3      # kg/mol
LATENT_HEAT_STEAM = 2.15e6     # J/kg at the reboiler's saturated conditions


def flue_molar_mass(co2_fraction: float) -> float:
    """Mean molar mass of the flue gas (kg/mol): CO2 plus the nitrogen and oxygen it rides in."""
    return co2_fraction * MOLAR_MASS_CO2 + (1.0 - co2_fraction) * MOLAR_MASS_AIR


def solvent_capacity(*, circulation_mass_flow: float, mea_mass_fraction: float,
                     loading_window: float) -> float:
    """CO2 the circulating solvent can carry (kg/s) over a loading window (mol CO2 per mol MEA)."""
    mea_molar = circulation_mass_flow * mea_mass_fraction / MOLAR_MASS_MEA
    return mea_molar * loading_window * MOLAR_MASS_CO2


CAPTURE_PARAMETERS = {
    "flue_gas_flow": Parameter("flue_gas_flow", "kg/s", 240.0, 1.0, 2_000.0, "mass flow up the stack"),
    "co2_fraction": Parameter("co2_fraction", "-", 0.12, 0.03, 0.30, "mole fraction of CO2 in the flue gas"),
    "capture_rate": Parameter("capture_rate", "-", 0.90, 0.50, 0.98, "share of the CO2 the plant is asked to take"),
    "solvent_concentration": Parameter("solvent_concentration", "-", 0.30, 0.20, 0.40,
                                       "MEA mass fraction in the lean solvent"),
    "solvent_density": Parameter("solvent_density", "kg/m3", 1_020.0, 950.0, 1_100.0, "lean solvent density"),
    "solvent_circulation": Parameter("solvent_circulation", "m3/h", 2_340.0, 100.0, 40_000.0,
                                     "lean solvent round the loop"),
    "lean_loading": Parameter("lean_loading", "mol/mol", 0.22, 0.10, 0.35,
                              "CO2 per MEA leaving the stripper"),
    "rich_loading_limit": Parameter("rich_loading_limit", "mol/mol", 0.48, 0.30, 0.55,
                                    "the loading the solvent reaches in equilibrium with the flue gas"),
    "desorption_enthalpy": Parameter("desorption_enthalpy", "GJ/t", 1.70, 1.30, 2.20,
                                     "heat to break the CO2 out of the amine, per tonne of CO2"),
    "solvent_heat_capacity": Parameter("solvent_heat_capacity", "kJ/kg/K", 3.80, 3.00, 4.40,
                                       "specific heat of the aqueous solvent"),
    "temperature_swing": Parameter("temperature_swing", "K", 10.0, 3.0, 60.0,
                                   "how far the cross-exchanger leaves the solvent below the stripper"),
    "stripping_steam_ratio": Parameter("stripping_steam_ratio", "kg/kg", 0.35, 0.10, 1.00,
                                       "water leaving the reboiler with the product, per kilogram of CO2"),
    "steam_flow": Parameter("steam_flow", "kg/s", 54.0, 0.1, 400.0, "low-pressure steam the plant can supply"),
    "steam_to_power": Parameter("steam_to_power", "MW/(kg/s)", 0.75, 0.40, 1.20,
                                "electricity the turbine gives up for each kilogram per second of steam"),
    "compressor_efficiency": Parameter("compressor_efficiency", "-", 0.72, 0.50, 0.90,
                                       "polytropic efficiency of each CO2 compressor stage"),
    "compression_stages": Parameter("compression_stages", "-", 4.0, 1.0, 6.0,
                                     "stages, each intercooled back to the suction temperature"),
    "product_pressure": Parameter("product_pressure", "bar", 110.0, 10.0, 200.0,
                                  "pressure the CO2 is delivered at, for transport"),
    "suction_temperature": Parameter("suction_temperature", "K", 313.15, 290.0, 350.0,
                                     "CO2 entering the compressor"),
}


class CarbonCapturePlant(ReferenceModel):
    """A post-combustion MEA capture island: solvent capacity, reboiler duty and the energy penalty."""

    slug = "carbon_capture_plant"
    title = "Carbon capture plant: solvent capacity, reboiler duty and the energy penalty"
    physics = "CO2 mass balance over a loading window, reboiler heat balance, polytropic CO2 compression"
    parameters = CAPTURE_PARAMETERS
    assumptions = (
        "The flue gas is CO2 in air: no water vapour, no sulphur, no oxygen effect on the solvent.",
        "One lean loading and one rich loading limit, held over the whole circulation.",
        "The reboiler duty is desorption plus sensible heat of the circulating solvent plus the water "
        "that leaves with the product; no heat integration beyond the cross-exchanger swing.",
        "The stripped CO2 leaves the top of the stripper saturated with water at the stripper temperature.",
    )
    limits = (
        "No absorber kinetics or column height: the absorber is a mass balance, so nothing here says how "
        "tall a column would have to be or how close to equilibrium it would run.",
        "One solvent: MEA shows the trade the whole family shows, but a different amine is a different model.",
        "No solvent degradation, no amine emissions, no make-up, no water balance, and no sulphur polishing.",
        "The CO2 compressor is an ideal-gas stage model. Above about 80 bar CO2 is a dense phase, so the last "
        "stages of a real train cost far less than this model charges; the reported compression work is an "
        "overestimate and the run says so.",
        "The energy penalty is a converted steam flow, not a re-dispatch of a specific turbine.",
    )
    sources = (
        "Molar masses: CO2 44.009 g/mol, MEA 61.08 g/mol, air 28.96 g/mol.",
        "Desorption enthalpy of MEA around 1.7 GJ/t CO2, the usual figure for 30 wt% solvent.",
        "Latent heat of steam at reboiler conditions around 2.15 MJ/kg.",
        "Polytropic compression T2 = T1 r^((gamma-1)/(gamma eta_p)), the same relation the compression "
        "station uses.",
    )
    not_verified = (
        "No plant data: the flows, loadings and duties are engineering-plausible, not a commissioned island.",
        "No comparison with a pilot campaign or a vendor's specific-duty guarantee.",
        "No kinetics: the absorber's approach to equilibrium is the assumption that hides the most.",
        "The energy penalty converts steam to electricity through one declared factor.",
    )

    def _validate(self, p: Mapping[str, float]) -> None:
        if p["rich_loading_limit"] <= p["lean_loading"]:
            raise ParameterError(
                "the rich loading limit has to sit above the lean loading: a solvent that cannot pick up "
                "any CO2 carries nothing")
        if p["capture_rate"] >= 1.0:
            raise ParameterError("capture is never complete: the stack still has to flow")

    def _run(self, p: Mapping[str, float]) -> TwinRun:
        mean_molar_mass = flue_molar_mass(p["co2_fraction"])
        flue_molar = p["flue_gas_flow"] / mean_molar_mass                       # mol/s
        co2_inlet = flue_molar * p["co2_fraction"]                             # mol/s
        co2_demanded = co2_inlet * p["capture_rate"] * MOLAR_MASS_CO2          # kg/s

        circulation_mass = p["solvent_circulation"] / 3_600.0 * p["solvent_density"]   # kg/s
        window = p["rich_loading_limit"] - p["lean_loading"]
        capacity = solvent_capacity(circulation_mass_flow=circulation_mass,
                                    mea_mass_fraction=p["solvent_concentration"],
                                    loading_window=window)

        mea_molar = circulation_mass * p["solvent_concentration"] / MOLAR_MASS_MEA      # mol/s
        # Where the solvent comes back: the lean loading plus whatever the capture target put into it.
        rich_loading = p["lean_loading"] + co2_demanded / max(mea_molar * MOLAR_MASS_CO2, 1e-30)
        circulation_required = (co2_demanded * MOLAR_MASS_MEA
                                / (p["solvent_concentration"] * window * MOLAR_MASS_CO2)
                                * 3_600.0 / p["solvent_density"])

        # The reboiler, part by part.
        desorption = co2_demanded * p["desorption_enthalpy"] * 1e9 / 1e3        # W
        sensible = circulation_mass * p["solvent_heat_capacity"] * 1e3 * p["temperature_swing"]
        vaporisation = co2_demanded * p["stripping_steam_ratio"] * LATENT_HEAT_STEAM
        duty_required = desorption + sensible + vaporisation
        duty_supplied = p["steam_flow"] * LATENT_HEAT_STEAM
        specific_duty = duty_required / max(co2_demanded, 1e-30) / 1e6          # GJ/t

        stages = int(round(p["compression_stages"]))
        ratio = max(p["product_pressure"] / 1.013_25, 1.0)
        stage_ratio = ratio ** (1.0 / stages)
        exponent = polytropic_exponent(1.30, p["compressor_efficiency"])
        per_stage = (1.0 / exponent) * (R_UNIVERSAL / MOLAR_MASS_CO2) * p["suction_temperature"] * (
            stage_ratio ** exponent - 1.0) / max(p["compressor_efficiency"], 1e-9)
        compression_work = stages * per_stage
        compression_power = co2_demanded * compression_work
        penalty = p["steam_flow"] * p["steam_to_power"]

        outputs = {
            "co2_inlet": Quantity("co2_inlet", co2_inlet * MOLAR_MASS_CO2 * 3.6, "t/h", "what the stack carries"),
            "co2_captured": Quantity("co2_captured", co2_demanded * 3.6, "t/h", "the capture target"),
            "co2_captured_per_year": Quantity("co2_captured_per_year", co2_demanded * 3.6 * 8_000.0 / 1e6,
                                              "Mt/yr", "at 8 000 full-load hours"),
            "co2_emitted": Quantity("co2_emitted", (co2_inlet * MOLAR_MASS_CO2 - co2_demanded) * 3.6, "t/h",
                                    "what goes up the stack anyway"),
            "solvent_capacity": Quantity("solvent_capacity", capacity * 3.6, "t/h",
                                         "the CO2 the circulating solvent can carry over its loading window"),
            "rich_loading": Quantity("rich_loading", rich_loading, "mol/mol", "where the solvent comes back"),
            "rich_headroom": Quantity("rich_headroom", p["rich_loading_limit"] - rich_loading, "mol/mol",
                                      "how much of the working window is left unused"),
            "circulation_required": Quantity("circulation_required", circulation_required, "m3/h",
                                             "what the capture target would take"),
            "circulation_margin": Quantity("circulation_margin", capacity / max(co2_demanded, 1e-30), "-",
                                           "capacity over demand: below one the solvent cannot carry the target"),
            "reboiler_duty": Quantity("reboiler_duty", duty_required / 1e6, "MW", "desorption, sensible and vapour"),
            "specific_reboiler_duty": Quantity("specific_reboiler_duty", specific_duty, "GJ/t",
                                               "the number a capture project is judged on"),
            "desorption_share": Quantity("desorption_share", desorption / duty_required, "-",
                                         "the irreducible part of the duty"),
            "sensible_share": Quantity("sensible_share", sensible / duty_required, "-",
                                       "what circulating more solvent than necessary costs"),
            "vapour_share": Quantity("vapour_share", vaporisation / duty_required, "-",
                                     "water leaving with the product"),
            "steam_margin": Quantity("steam_margin", duty_supplied / max(duty_required, 1e-30), "-",
                                     "what the declared steam can deliver against what the stripper needs"),
            "energy_penalty": Quantity("energy_penalty", penalty, "MW", "turbine output given up for the steam"),
            "compression_power": Quantity("compression_power", compression_power / 1e6, "MW",
                                          "CO2 to the transport pressure"),
            "total_power": Quantity("total_power", (penalty * 1e6 + compression_power) / 1e6, "MW",
                                    "what the capture island costs the plant"),
        }

        balances = [
            close("carbon", co2_demanded, capacity, "kg/s", tolerance=5e-2,
                  detail="what the capture target demands against what the circulating solvent can carry over "
                         "its loading window. The default circulation is the one the target needs, so the "
                         "residual is the design margin; a plant whose circulation is too small shows the gap"),
            close("energy", duty_supplied, duty_required, "W", tolerance=5e-2,
                  detail="the duty the declared stripping steam can deliver against the breakdown the "
                         "stripper needs: desorption enthalpy, sensible heat of the whole circulation and the "
                         "water that leaves with the product"),
        ]

        checks = [
            Check("rich_loading_inside_the_window",
                  p["lean_loading"] < rich_loading <= p["rich_loading_limit"] + 1e-12,
                  "the solvent has to come back richer than it left and no richer than equilibrium allows"),
            Check("specific_duty_in_the_mea_band", 2.5 <= specific_duty <= 4.5,
                  "30 wt% MEA lands between about 3 and 4 GJ per tonne of CO2; far outside that is a "
                  "modelling error, not a better solvent"),
            Check("desorption_dominates_the_duty", desorption < duty_required,
                  "the thermodynamic part is real; the sensible and vapour parts are what optimisation attacks"),
            Check("capture_below_complete", p["capture_rate"] < 1.0,
                  "some CO2 always goes up the stack"),
            Check("compression_is_the_smaller_cost", compression_power < duty_required,
                  "pumping the CO2 to transport pressure costs far less than getting it out of the solvent"),
        ]

        run = TwinRun(system=self.slug, title=self.title, parameters=dict(p), outputs=outputs,
                      balances=balances, checks=checks)
        # The trade the whole design turns on: more circulation buys a cooler stripper and costs steam.
        xs, ys = [], []
        for k in range(41):
            factor = 0.6 + 0.03 * k
            circulated = circulation_mass * factor
            duty_k = co2_demanded * p["desorption_enthalpy"] * 1e9 / 1e3 \
                + circulated * p["solvent_heat_capacity"] * 1e3 * p["temperature_swing"] \
                + co2_demanded * p["stripping_steam_ratio"] * LATENT_HEAT_STEAM
            xs.append(factor)
            ys.append(duty_k / max(co2_demanded, 1e-30) / 1e6)
        run.series = self._series(len(xs), circulation_factor=xs, specific_duty_GJ_per_t=ys)
        run.not_verified = list(self.not_verified)
        return run

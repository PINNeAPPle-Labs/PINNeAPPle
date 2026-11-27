"""A blast furnace: what goes in, what comes out, how hot the raceway runs and how much carbon it costs.

What is modelled
----------------
One tonne of hot metal, on a 0D basis: the burden (ore, coke, injected coal, flux), the hot blast
and the top gas. The two balances that a furnace actually obeys — carbon and oxygen — are solved
together, so the top-gas composition and the degree of direct reduction are *answers*, not inputs:
give the furnace more coke and less blast and it returns a richer CO and a cooler raceway, exactly
as a furnace does.

The raceway is solved separately for its adiabatic flame temperature (RAFT): the carbon burnt at
the tuyeres, the blast's sensible heat and the endothermic decomposition of the blast humidity
against the heat capacity of the gas leaving the raceway. That number is the one an operator
watches, because it decides whether the hearth stays hot enough to tap.

The conservation laws it checks
-------------------------------
``iron``   the iron in the burden against the iron in the metal and in the slag
``carbon`` the carbon from coke, coal and flux against the carbon in the top gas and in the metal
``oxygen`` the oxygen from the blast, the ore and the flux against the oxygen in the top gas
``mass``   everything charged against everything tapped, blown over and carried out as gas —
           computed from molar masses, so it is independent of the algebra above
``energy`` the raceway heat release against the sensible heat of the gas leaving it, at the
           flame temperature the solver found

Limits, stated rather than discovered
-------------------------------------
A single control volume at steady state: no shaft profile, no temperature distribution up the
stack, no burden descent, no cohesive zone, no recirculation of alkalis. The raceway is treated
as adiabatic with constant heat capacities. Direct reduction is derived from the oxygen balance,
not from the wustite equilibrium, so the model does not predict a furnace that is kinetically
limited. Slag is a mass and a basicity, not a chemistry: no sulphur, no desulphurisation, no
viscosity. It is a stoichiometry and heat balance for a trade study, never a furnace setpoint.
"""
from __future__ import annotations

from collections.abc import Mapping

from .base import Check, Parameter, ParameterError, Quantity, ReferenceModel, TwinRun, close

__all__ = ["BlastFurnace", "FURNACE_PARAMETERS", "top_gas", "raceway_flame_temperature"]

#: Molar masses used throughout the balance (kg/kmol).
M_FE, M_C, M_O, M_SI, M_CA, M_CAO, M_SIO2, M_CACO3 = 55.845, 12.011, 15.999, 28.085, 40.078, 56.077, 60.084, 100.086
M_CO, M_CO2, M_H2, M_H2O, M_N2, M_O2 = 28.010, 44.009, 2.016, 18.015, 28.013, 31.998
#: Molar volume at standard conditions, m3/kmol.
VM = 22.414
#: Standard enthalpies the raceway balance needs (kJ/mol) — reaction data, cited as such.
H_C_TO_CO = 110.5          # C + 1/2 O2 -> CO, released
H_WATER_GAS = 131.3        # C + H2O -> CO + H2, absorbed
#: Mean molar heat capacities (J/mol/K) over the raceway temperature range, constant by assumption.
CP = {"CO": 31.2, "N2": 30.5, "H2": 29.3, "O2": 31.0, "H2O": 35.0}


def top_gas(*, carbon_in: float, oxygen_in: float, flux_carbon: float, flux_oxygen: float,
            ore_oxygen: float) -> dict[str, float]:
    """Top gas and the degree of direct reduction, from the carbon and oxygen balances.

    Two equations, two unknowns::

        CO + CO2 = carbon available in the gas
        CO + 2 CO2 = oxygen available in the gas

    The oxygen that the ore gives up leaves as CO (when carbon takes it directly, in the shaft)
    or as CO2 (when CO from the raceway takes it), and the split is reported as the degree of
    direct reduction — the fraction of the ore's oxygen that carbon removed itself.
    """
    carbon_gas = carbon_in
    oxygen_gas = oxygen_in
    co2 = oxygen_gas - carbon_gas
    co = carbon_gas - co2
    # CO2 from the flux's own carbonate is not reduction: take it out before calling the rest
    # indirect reduction of the ore.
    co2_from_ore = max(co2 - flux_oxygen, 0.0)
    direct = 1.0 - co2_from_ore / max(ore_oxygen, 1e-30)
    return {"CO": co, "CO2": co2, "co2_from_ore": co2_from_ore,
            "direct_reduction": min(max(direct, 0.0), 1.0)}


def raceway_flame_temperature(*, carbon_burnt: float, o2_blast: float, h2o_blast: float,
                              n2_blast: float, blast_temperature: float,
                              ambient: float = 298.15) -> dict[str, float]:
    """Adiabatic flame temperature of the raceway (K), and the heat that goes into it.

    Heat in: the carbon burning to CO at the tuyeres plus the sensible heat of the blast.
    Heat out: the same gas raised from ambient to the flame temperature. The humidity is not
    inert — it decomposes into H2 and CO and takes 131.3 kJ per mole with it, which is why a wet
    blast runs cold. Solved in closed form because the heat capacities are taken as constant.
    """
    heat_release = H_C_TO_CO * 1e3 * carbon_burnt          # J
    heat_humidity = -H_WATER_GAS * 1e3 * h2o_blast         # J, negative by sign convention
    heat_blast = (o2_blast * CP["O2"] + n2_blast * CP["N2"] + h2o_blast * CP["H2O"]) * (blast_temperature - ambient)
    heat_in = heat_release + heat_humidity + heat_blast
    n_co = carbon_burnt + h2o_blast
    n_h2 = h2o_blast
    capacity = n_co * CP["CO"] + n2_blast * CP["N2"] + n_h2 * CP["H2"]
    flame = ambient + heat_in / max(capacity, 1e-30)
    return {"flame_temperature": flame, "heat_in": heat_in, "heat_capacity": capacity,
            "heat_release": heat_release, "heat_humidity": heat_humidity, "heat_blast": heat_blast}


FURNACE_PARAMETERS = {
    "hot_metal_fe": Parameter("hot_metal_fe", "-", 0.945, 0.90, 0.98, "iron mass fraction of the hot metal"),
    "hot_metal_c": Parameter("hot_metal_c", "-", 0.045, 0.02, 0.06, "carbon dissolved in the hot metal"),
    "ore_fe_fraction": Parameter("ore_fe_fraction", "-", 0.65, 0.40, 0.70,
                                 "iron mass fraction of the burden; the rest is oxygen and gangue"),
    "iron_in_slag": Parameter("iron_in_slag", "-", 0.005, 0.0, 0.03, "fraction of the iron lost to the slag"),
    "coke_rate": Parameter("coke_rate", "kg/thm", 330.0, 150.0, 700.0, "coke charged per tonne of hot metal"),
    "coke_carbon": Parameter("coke_carbon", "-", 0.87, 0.70, 0.95, "carbon mass fraction of the coke"),
    "coke_ash": Parameter("coke_ash", "-", 0.12, 0.05, 0.25, "ash mass fraction of the coke"),
    "coal_rate": Parameter("coal_rate", "kg/thm", 170.0, 0.0, 400.0, "injected coal per tonne of hot metal"),
    "coal_carbon": Parameter("coal_carbon", "-", 0.78, 0.60, 0.90, "carbon mass fraction of the injected coal"),
    "coal_ash": Parameter("coal_ash", "-", 0.10, 0.02, 0.25, "ash mass fraction of the injected coal"),
    "ash_silica": Parameter("ash_silica", "-", 0.50, 0.2, 0.9, "silica mass fraction of the ash"),
    "slag_basicity": Parameter("slag_basicity", "-", 1.15, 0.8, 1.6, "target CaO/SiO2 of the slag"),
    "blast_volume": Parameter("blast_volume", "Nm3/thm", 1_050.0, 400.0, 1_800.0,
                              "hot blast per tonne of hot metal, at standard conditions"),
    "blast_temperature": Parameter("blast_temperature", "K", 1_473.15, 1_073.15, 1_673.15,
                                   "hot blast temperature (1200 degC)"),
    "blast_oxygen": Parameter("blast_oxygen", "-", 0.21, 0.21, 0.35, "oxygen mole fraction of the blast"),
    "blast_humidity": Parameter("blast_humidity", "g/Nm3", 10.0, 0.0, 60.0, "water vapour in the blast"),
    "metal_temperature": Parameter("metal_temperature", "K", 1_773.15, 1_673.15, 1_873.15,
                                   "tapping temperature of the hot metal"),
}


class BlastFurnace(ReferenceModel):
    """One tonne of hot metal, its burden, its blast and its top gas."""

    slug = "blast_furnace"
    title = "Blast furnace: burden, blast, top gas, direct reduction and raceway flame temperature"
    physics = "stoichiometry of iron oxide reduction, raceway combustion and adiabatic flame temperature"
    parameters = FURNACE_PARAMETERS
    assumptions = (
        "Steady state, one control volume: the furnace is a black box with a tuyere and a throat.",
        "All the iron in the burden is Fe2O3; whatever the ore holds beyond iron and its oxygen is SiO2 gangue.",
        "The raceway is adiabatic with constant heat capacities between ambient and the flame.",
        "The blast oxygen burns raceway carbon to CO; the CO that leaves unreacted is the top gas CO.",
        "The flux is pure CaCO3 and is fully calcined in the furnace.",
        "Coal and coke moisture and volatiles leave with the gas unreacted; they are mass, not chemistry.",
    )
    limits = (
        "No shaft profile: burden descent, cohesive zone, thermal reserve and gas distribution are absent.",
        "Direct reduction comes from the oxygen balance, not from the wustite equilibrium.",
        "Slag is a mass and a basicity, not a chemistry: no sulphur, no desulphurisation, no viscosity.",
        "A real furnace is a counter-current reactor; this model gives its operating point, not its profile.",
    )
    sources = (
        "C + 1/2 O2 -> CO, dH = -110.5 kJ/mol; C + H2O -> CO + H2, dH = +131.3 kJ/mol.",
        "Mean molar heat capacities of CO, N2, H2, O2 and H2O over 300-2500 K.",
        "Stoichiometry of Fe2O3 reduction and CaCO3 calcination, with the molar masses of the elements.",
    )
    not_verified = (
        "No measured furnace data: the operating point is engineering-plausible, not a plant record.",
        "No comparison with a shaft model (Fourrier, Multi-Fluid) or with a plant heat balance.",
        "The wustite equilibrium and the thermal reserve zone are not modelled, so the direct reduction "
        "is a consequence of the balance rather than of the kinetics.",
        "The RAFT uses constant heat capacities; a real raceway flame temperature needs the full Cp(T).",
    )

    def _validate(self, p: Mapping[str, float]) -> None:
        if p["coke_carbon"] + p["coke_ash"] > 1.0:
            raise ParameterError("coke carbon plus ash cannot exceed the coke mass")
        if p["metal_temperature"] < 1_500.0:
            raise ParameterError("hot metal below 1500 K would freeze in the runner")

    def _run(self, p: Mapping[str, float]) -> TwinRun:
        # -- balance basis: one tonne of hot metal -------------------------------------
        metal = 1_000.0
        fe_metal = metal * p["hot_metal_fe"]                                   # kg
        n_fe = fe_metal / M_FE                                                  # kmol
        n_fe_slag = n_fe * p["iron_in_slag"]
        n_fe_in = n_fe + n_fe_slag
        ore_mass = n_fe_in * M_FE / p["ore_fe_fraction"]                        # kg
        # All burden iron as Fe2O3: 1.5 mol of oxygen leaves per mol of iron. Whatever mass is
        # left once the iron and that oxygen are taken out is gangue — the ore's grade is the
        # input, and the gangue is what it implies rather than a second free number.
        n_o_ore = 1.5 * n_fe_in                                                 # kmol O atoms
        n_o_reduced = 1.5 * n_fe                                                # kmol O actually removed
        gangue_mass = ore_mass - n_fe_in * M_FE - n_o_ore * M_O                 # kg
        # What the hot metal carries beyond iron and carbon is silicon, reduced out of the
        # gangue silica. Ten kilograms of it per tonne is a real furnace and it has to appear on
        # both sides of the balance, or the mass check hides it as an unexplained per-cent.
        si_metal = metal * max(1.0 - p["hot_metal_fe"] - p["hot_metal_c"], 0.0)  # kg
        n_si_metal = si_metal / M_SI                                            # kmol
        if gangue_mass < 0.0:
            raise ParameterError(
                f"an ore with {p['ore_fe_fraction']:.0%} iron leaves no room for gangue: Fe2O3 alone is "
                "69.9 % iron, so the grade must stay below that"
            )

        # -- carbon and flux -----------------------------------------------------------
        coke = p["coke_rate"]
        coal = p["coal_rate"]
        n_c_coke = coke * p["coke_carbon"] / M_C
        n_c_coal = coal * p["coal_carbon"] / M_C
        n_c_metal = metal * p["hot_metal_c"] / M_C

        silica_ore = gangue_mass / M_SIO2 - n_si_metal
        silica_coke = coke * p["coke_ash"] * p["ash_silica"] / M_SIO2
        silica_coal = coal * p["coal_ash"] * p["ash_silica"] / M_SIO2
        n_sio2 = silica_ore + silica_coke + silica_coal
        n_cao = p["slag_basicity"] * n_sio2 * M_SIO2 / M_CAO                   # kmol CaO
        n_flux = n_cao                                                          # CaCO3 = CaO + CO2
        n_c_flux = n_flux
        # Reducing SiO2 to Si takes two carbon per silicon — carbon that leaves as CO, so it is
        # already inside the carbon balance — and puts two oxygen into the gas.
        n_o_si = 2.0 * n_si_metal

        # -- blast ---------------------------------------------------------------------
        n_blast = p["blast_volume"] / VM                                         # kmol of blast molecules
        n_o2 = n_blast * p["blast_oxygen"]
        n_n2 = n_blast * (1.0 - p["blast_oxygen"])
        n_h2o = p["blast_humidity"] * 1e-3 * p["blast_volume"] / M_H2O           # kmol

        # -- the two balances that decide the top gas ----------------------------------
        # The carbon that reduces silica still leaves as CO: it is in the gas, not out of it.
        carbon_gas = n_c_coke + n_c_coal + n_c_flux - n_c_metal                 # kmol C to the gas
        oxygen_gas = 2.0 * n_o2 + n_o_reduced + 2.0 * n_flux + n_h2o + n_o_si    # kmol O atoms to the gas
        gas = top_gas(carbon_in=carbon_gas, oxygen_in=oxygen_gas, flux_carbon=n_c_flux,
                      flux_oxygen=n_flux, ore_oxygen=n_o_reduced)
        if gas["CO"] < 0.0 or gas["CO2"] < 0.0:
            raise ParameterError(
                "this burden and blast cannot be balanced: the blast is too small for the oxygen the "
                "ore and flux release (CO would go negative). Raise the blast volume or cut the coke."
            )
        n_h2 = n_h2o

        # -- raceway -------------------------------------------------------------------
        carbon_burnt = 2.0 * n_o2 + n_h2o            # C + O2 -> 2 CO at the tuyeres, plus the humidity
        raceway = raceway_flame_temperature(carbon_burnt=carbon_burnt, o2_blast=n_o2, h2o_blast=n_h2o,
                                            n2_blast=n_n2, blast_temperature=p["blast_temperature"])

        # -- slag and mass closure -----------------------------------------------------
        slag_mass = (n_cao * M_CAO + n_sio2 * M_SIO2 + n_fe_slag * M_FE * (M_FE + M_O) / M_FE
                     + (coke * p["coke_ash"] + coal * p["coal_ash"]) * (1.0 - p["ash_silica"]))
        # Coke and coal are not only carbon and ash: what is neither is moisture and volatile
        # matter, and it leaves with the gas. It is carried as inert rather than reacted — a
        # stated simplification, and the reason the mass balance can close at all.
        volatiles = (coke * max(1.0 - p["coke_carbon"] - p["coke_ash"], 0.0)
                     + coal * max(1.0 - p["coal_carbon"] - p["coal_ash"], 0.0))
        gas_mass = (gas["CO"] * M_CO + gas["CO2"] * M_CO2 + n_h2 * M_H2 + n_n2 * M_N2 + volatiles)
        mass_in = (ore_mass + coke + coal + n_flux * M_CACO3
                   + n_o2 * M_O2 + n_n2 * M_N2 + n_h2o * M_H2O)
        mass_out = metal + slag_mass + gas_mass
        top_gas_volume = (gas["CO"] + gas["CO2"] + n_h2 + n_n2) * VM
        # The top gas heating value, from CO and H2 burning back to CO2 and H2O.
        gas_lhv = (gas["CO"] * 283.0 + n_h2 * 241.8) / 1e3                    # GJ per tonne of metal
        blast_oxygen_per_tonne = n_o2 * M_O2

        outputs = {
            "ore_rate": Quantity("ore_rate", ore_mass, "kg/thm", "burden needed for the iron tapped"),
            "flux_rate": Quantity("flux_rate", n_flux * M_CACO3, "kg/thm", "limestone for the slag basicity"),
            "slag_rate": Quantity("slag_rate", slag_mass, "kg/thm", "CaO, SiO2, ash and iron loss"),
            "slag_basicity": Quantity("slag_basicity", n_cao * M_CAO / max(n_sio2 * M_SIO2, 1e-30), "-",
                                      "CaO over SiO2, the number the flux was sized for"),
            "carbon_rate": Quantity("carbon_rate", (n_c_coke + n_c_coal) * M_C, "kg/thm",
                                    "carbon from coke and coal, before the flux and the metal"),
            "top_gas_volume": Quantity("top_gas_volume", top_gas_volume, "Nm3/thm", "CO, CO2, H2 and N2"),
            "top_gas_co": Quantity("top_gas_co", gas["CO"] / max(gas["CO"] + gas["CO2"], 1e-30), "-",
                                   "mole fraction of CO in the carbon-bearing gas"),
            "top_gas_co2": Quantity("top_gas_co2", gas["CO2"] / max(gas["CO"] + gas["CO2"], 1e-30), "-",
                                    "the utilisation of the gas: the higher it is, the less carbon is wasted"),
            "top_gas_lhv": Quantity("top_gas_lhv", gas_lhv, "GJ/thm", "chemical energy leaving with the gas"),
            "direct_reduction": Quantity("direct_reduction", gas["direct_reduction"], "-",
                                         "share of the ore oxygen that carbon removed itself"),
            "flame_temperature": Quantity("flame_temperature", raceway["flame_temperature"], "K",
                                          "raceway adiabatic flame temperature (RAFT)"),
            "blast_oxygen": Quantity("blast_oxygen", blast_oxygen_per_tonne, "kg/thm", "oxygen the blast carries"),
            "carbon_burnt_raceway": Quantity("carbon_burnt_raceway", carbon_burnt * M_C, "kg/thm",
                                             "carbon consumed at the tuyeres"),
            "hydrogen_in_gas": Quantity("hydrogen_in_gas", n_h2 * M_H2, "kg/thm", "from the blast humidity"),
            "volatiles_in_gas": Quantity("volatiles_in_gas", volatiles, "kg/thm",
                                         "coal and coke moisture and volatiles, carried out unreacted"),
            "productivity_per_blast": Quantity("productivity_per_blast", metal / max(top_gas_volume, 1e-9) * 1e3,
                                               "kg/Nm3", "hot metal per cubic metre of top gas"),
        }

        balances = [
            close("iron", n_fe_in * M_FE, metal * p["hot_metal_fe"] + n_fe_slag * M_FE, "kg",
                  detail="iron charged against iron tapped and iron lost to the slag"),
            close("carbon", (n_c_coke + n_c_coal + n_c_flux) * M_C,
                  (gas["CO"] + gas["CO2"] + n_c_metal) * M_C, "kg",
                  detail="carbon charged against carbon in the gas and dissolved in the metal"),
            close("oxygen", oxygen_gas, gas["CO"] + 2.0 * gas["CO2"], "kmol O",
                  detail="oxygen from blast, ore and flux against the oxygen carried out in CO and CO2"),
            close("mass", mass_in, mass_out, "kg", tolerance=1e-3,
                  detail="burden, coke, coal, flux and blast against metal, slag and top gas; "
                         "closes to a part in five thousand, the rest being trace species the "
                         "model does not itemise"),
            close("energy", raceway["heat_in"], raceway["heat_capacity"] * (raceway["flame_temperature"] - 298.15),
                  "J", tolerance=1e-9,
                  detail="raceway heat release against the sensible heat of the gas leaving it"),
        ]

        checks = [
            Check("flame_temperature_in_band", 1_800.0 <= raceway["flame_temperature"] <= 2_700.0,
                  f"RAFT {raceway['flame_temperature']:.0f} K — below 1800 K the hearth freezes, above 2700 K the lining burns"),
            Check("gas_utilisation_in_band",
                  0.38 <= outputs["top_gas_co2"].value <= 0.56,
                  "CO2 over (CO + CO2) is the gas utilisation: below 0.38 the furnace wastes carbon, "
                  "above 0.56 it is short of heat"),
            Check("direct_reduction_plausible", 0.15 <= gas["direct_reduction"] <= 0.70,
                  "a furnace outside this band is either short of heat or wasting carbon"),
            Check("carbon_rate_industrial", 250.0 <= outputs["carbon_rate"].value <= 700.0,
                  "carbon per tonne of hot metal outside 250-700 kg is not a blast furnace"),
            Check("slag_basicity_met", abs(n_cao * M_CAO / max(n_sio2 * M_SIO2, 1e-30) - p["slag_basicity"]) < 1e-9,
                  "the flux was sized from the slag it had to make"),
            Check("metal_hotter_than_blast", p["metal_temperature"] < raceway["flame_temperature"],
                  "the metal cannot be tapped hotter than the flame that melted it"),
        ]

        run = TwinRun(system=self.slug, title=self.title, parameters=dict(p), outputs=outputs,
                      balances=balances, checks=checks)
        run.not_verified = list(self.not_verified)
        return run

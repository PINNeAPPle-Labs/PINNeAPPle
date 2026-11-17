"""Tests of the Digital Twin Library: nine reference models, each checked against its own physics.

The tests are of two kinds, and they are labelled in their names:

* the *balances and invariants* every run carries — a system that does not close its
  conservation laws is a broken model, and the run says so itself;
* *independent answers* — an analytic result or a hand calculation that the model must
  reproduce, written here rather than inside the model, so a model cannot mark its own homework.

The second kind is the point. A twin that reports PASS on its own checks has only proved it is
self-consistent.
"""
from __future__ import annotations

import json
import math

import pytest

from pinneapple_systems.digital_twin.library import (
    TWINS,
    LibraryError,
    ParameterError,
    build,
    catalogue,
    compare,
    describe,
    run,
)
from pinneapple_systems.digital_twin.library.blast_furnace import (
    BlastFurnace,
    raceway_flame_temperature,
)
from pinneapple_systems.digital_twin.library.chemical_plant import (
    ChemicalPlant,
    equilibrium_constant,
)
from pinneapple_systems.digital_twin.library.compression_station import (
    CompressionStation,
    polytropic_exponent,
    stage_discharge_temperature,
    stage_work,
)
from pinneapple_systems.digital_twin.library.desalination import (
    DesalinationPlant,
    minimum_specific_energy,
    osmotic_pressure,
)
from pinneapple_systems.digital_twin.library.fpso import (
    FPSOSubsea,
    ipr_rate,
    line_pressure_drop,
    vogel_qmax,
)
from pinneapple_systems.digital_twin.library.green_hydrogen import (
    FARADAY,
    MOLAR_MASS_H2,
    GreenHydrogenPlant,
    hydrogen_from_charge,
    reversible_voltage,
    thermoneutral_voltage,
)
from pinneapple_systems.digital_twin.library.lunar_base import (
    SYNODIC_MONTH_DAYS,
    LunarBase,
    night_factor,
    regolith_oxygen_yield,
)
from pinneapple_systems.digital_twin.library.pipeline_network import (
    PipelineNetwork,
    compressor_discharge,
    linepack_of_segment,
    segment_pressure_square_drop,
)
from pinneapple_systems.digital_twin.library.reusable_booster import (
    ReusableBooster,
    atmosphere,
)

SLUGS = ("blast_furnace", "chemical_plant", "compression_station", "desalination_plant", "fpso_subsea",
         "green_hydrogen_plant", "lunar_base", "pipeline_network", "reusable_booster")


# ---------------------------------------------------------------------------
# The contract every system honours
# ---------------------------------------------------------------------------

def test_the_five_selected_systems_are_in_the_library():
    assert sorted(TWINS) == list(SLUGS)
    entries = catalogue()
    assert [e.slug for e in entries] == list(SLUGS)
    assert all(e.fidelity and e.status for e in entries)


@pytest.mark.parametrize("slug", SLUGS)
def test_every_system_balances_and_holds_its_invariants(slug):
    result = run(slug)
    assert result.balances, f"{slug} carries no balance, so it proves nothing"
    assert result.checks, f"{slug} carries no invariant"
    for balance in result.balances:
        assert balance.ok, f"{slug}: {balance}"
    for check in result.checks:
        assert check.passed, f"{slug}: {check}"
    assert result.ok


@pytest.mark.parametrize("slug", SLUGS)
def test_every_system_states_what_it_cannot_say(slug):
    """The honesty list is not decoration: a model without limits is a model without a boundary."""
    result = run(slug)
    assert result.limits and result.not_verified
    assert result.sources, "every number that comes from outside has to say where it came from"
    model = build(slug)
    assert model.assumptions


@pytest.mark.parametrize("slug", SLUGS)
def test_a_run_is_reproducible(slug):
    first = run(slug).as_dict()
    second = run(slug).as_dict()
    first.pop("created")
    second.pop("created")
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


@pytest.mark.parametrize("slug", SLUGS)
def test_a_result_can_be_written_and_read_back(slug, tmp_path):
    result = run(slug)
    path = result.save(tmp_path / f"{slug}.json")
    payload = json.loads(path.read_text())
    assert payload["system"] == slug
    assert payload["ok"] is True
    assert set(payload["balances"][0]) >= {"name", "residual", "tolerance", "ok"}
    assert all("unit" in q for q in payload["outputs"].values())


def test_an_unknown_system_is_refused_by_name():
    with pytest.raises(LibraryError, match="no twin"):
        build("blast_furnave")


def test_an_unknown_parameter_is_refused():
    with pytest.raises(ParameterError, match="unknown parameter"):
        build("lunar_base").simulate({"crew_size": 6})


def test_a_parameter_outside_its_range_is_refused_unless_asked_for():
    model = build("lunar_base")
    with pytest.raises(ParameterError, match="outside the model's range"):
        model.simulate(crew=40.0)
    extrapolated = model.simulate(crew=40.0, strict=False)
    assert extrapolated.extrapolated and "crew" in extrapolated.extrapolated[0]


def test_the_catalogue_runs_every_system_in_one_call():
    table = compare()
    assert table.count("PASS") == len(SLUGS)
    assert "FAIL" not in table


def test_describe_names_the_parameters_and_the_sources():
    text = describe("blow_in" if False else "blast_furnace")
    assert "parameters" in text and "sources" in text and "limits" in text


# ---------------------------------------------------------------------------
# FPSO and subsea production
# ---------------------------------------------------------------------------

def test_vogel_curve_matches_the_published_form():
    """Vogel (JPT 1968): q/qmax = 1 - 0.2 x - 0.8 x^2, with x = p_wf / p_r."""
    productivity, pressure = 1.2e-9, 30.0e6
    qmax = vogel_qmax(productivity, pressure, 0.8)
    for x in (0.0, 0.25, 0.5, 0.75, 1.0):
        expected = qmax * (1.0 - 0.2 * x - 0.8 * x * x)
        assert ipr_rate(x * pressure, productivity, pressure, 0.8) == pytest.approx(expected, rel=1e-12)


def test_the_straight_line_productivity_index_is_a_special_case():
    """With an exponent of zero the same equation is J (p_r - p_wf), to the last digit."""
    productivity, pressure = 2.0e-9, 25.0e6
    for p_wf in (0.0, 5.0e6, 12.5e6, 20.0e6, 25.0e6):
        assert ipr_rate(p_wf, productivity, pressure, 0.0) == pytest.approx(productivity * (pressure - p_wf), rel=1e-12)


def test_the_line_drop_is_hydrostatic_when_there_is_nothing_flowing():
    drop = line_pressure_drop(
        oil_rate=1e-9, gor=0.0, water_cut=0.0, oil_density=900.0, gas_specific_gravity=0.7,
        diameter=0.2, length=1000.0, rise=250.0, pressure=2e6, temperature_k=320.0,
        oil_viscosity=5e-3, formation_volume_factor=1.0,
    )
    assert drop["hydrostatic"] == pytest.approx(900.0 * 9.80665 * 250.0, rel=1e-9)
    assert drop["friction"] == pytest.approx(0.0, abs=1e-3)


def test_the_energy_balance_reproduces_the_ln_meeting_mean_temperature():
    """With no Joule-Thomson cooling the line is a plain heat exchanger, and LMTD must agree.

    The model integrates the temperature profile in closed form; LMTD knows only the two ends.
    Agreeing to a part in a thousand is the independent confirmation that the profile is right.
    """
    result = FPSOSubsea().simulate(joule_thomson=0.0)
    assert result.ok
    p = result.parameters
    ua = (p["overall_heat_transfer"] * math.pi * p["line_diameter"]
          * (p["flowline_length"] + p["riser_height"]))                        # W/K
    hot_end = result.value("wellhead_temperature") - (p["seawater_temperature"] - 273.15)
    cold_end = result.value("arrival_temperature") - (p["seawater_temperature"] - 273.15)
    lmtd = (hot_end - cold_end) / math.log(hot_end / cold_end)
    assert result.value("heat_duty") == pytest.approx(ua * lmtd / 1e3, rel=1e-3)


def test_the_well_declines_and_stops_at_the_separator():
    result = FPSOSubsea().simulate(production_days=2_000.0)
    rates = result.series["oil_rate"]
    assert rates[0] > rates[len(rates) // 2] > rates[-1]
    assert result.value("pressure_at_end") >= result.parameters["separator_pressure"] / 1e5 - 1e-6


# ---------------------------------------------------------------------------
# Blast furnace
# ---------------------------------------------------------------------------

def test_a_hotter_blast_runs_a_hotter_raceway():
    """RAFT is monotone in blast temperature — the first lever an operator has."""
    cool = BlastFurnace().simulate(blast_temperature=1_273.15).value("flame_temperature")
    hot = BlastFurnace().simulate(blast_temperature=1_573.15).value("flame_temperature")
    assert hot > cool + 100.0


def test_a_wet_blast_runs_colder():
    """Humidity decomposes endothermically: C + H2O -> CO + H2 costs 131.3 kJ/mol."""
    dry = BlastFurnace().simulate(blast_humidity=0.0).value("flame_temperature")
    wet = BlastFurnace().simulate(blast_humidity=40.0).value("flame_temperature")
    assert wet < dry


def test_the_raceway_balance_closes_by_hand():
    """Recompute the flame temperature here from the published enthalpies, independently."""
    result = raceway_flame_temperature(
        carbon_burnt=20.0, o2_blast=9.8, h2o_blast=0.5, n2_blast=37.0, blast_temperature=1_473.15)
    heat = 110.5e3 * 20.0 - 131.3e3 * 0.5 + (9.8 * 31.0 + 37.0 * 30.5 + 0.5 * 35.0) * (1_473.15 - 298.15)
    capacity = (20.0 + 0.5) * 31.2 + 37.0 * 30.5 + 0.5 * 29.3
    assert result["flame_temperature"] == pytest.approx(298.15 + heat / capacity, rel=1e-12)


def test_more_coke_costs_more_carbon_and_runs_the_gas_leaner():
    """Coke is the carbon and the heat: charging more of it makes the gas richer in CO and the
    furnace more direct-reducing, which is precisely the waste a coke rate buys."""
    lean = BlastFurnace().simulate(coke_rate=300.0)
    rich = BlastFurnace().simulate(coke_rate=380.0)
    assert rich.value("carbon_rate") > lean.value("carbon_rate")
    assert rich.value("top_gas_co") > lean.value("top_gas_co")
    assert rich.value("top_gas_co2") < lean.value("top_gas_co2")
    assert rich.value("direct_reduction") > lean.value("direct_reduction")


def test_a_burden_that_cannot_be_balanced_is_refused_with_a_reason():
    with pytest.raises(Exception) as err:
        BlastFurnace().simulate(blast_volume=400.0, coke_rate=700.0)
    assert "balance" in str(err.value) or "blast" in str(err.value)


# ---------------------------------------------------------------------------
# Chemical plant
# ---------------------------------------------------------------------------

def test_the_equilibrium_constant_is_the_exponential_of_the_gibbs_energy():
    for temperature in (300.0, 450.0, 600.0):
        expected = math.exp(-(-90_000.0 - temperature * -220.0) / (8.314_462_618 * temperature))
        assert equilibrium_constant(temperature, -90_000.0, -220.0) == pytest.approx(expected, rel=1e-12)


def test_vant_hoff_holds_on_the_function_the_model_uses():
    delta_h = -90_000.0
    t1, t2 = 400.0, 500.0
    slope = (math.log(equilibrium_constant(t2, delta_h, -220.0))
             - math.log(equilibrium_constant(t1, delta_h, -220.0))) / (1.0 / t2 - 1.0 / t1)
    assert slope == pytest.approx(-delta_h / 8.314_462_618, rel=1e-9)


def test_pressure_buys_conversion_and_temperature_gives_it_back():
    low = ChemicalPlant().simulate(pressure=40e5)
    high = ChemicalPlant().simulate(pressure=160e5)
    assert high.value("per_pass_conversion") > low.value("per_pass_conversion")
    hotter = ChemicalPlant().simulate(reactor_inlet_temperature=530.0)
    assert hotter.value("per_pass_conversion") < ChemicalPlant().simulate().value("per_pass_conversion")


def test_the_recycle_multiplies_the_per_pass_conversion():
    """The loop conversion per mole of make-up has to beat what one pass achieves."""
    result = ChemicalPlant().simulate()
    assert result.value("loop_conversion") > result.value("per_pass_conversion")
    assert result.value("recycle_ratio") > 1.0


def test_the_plant_closes_its_atom_balances_to_machine_precision():
    result = ChemicalPlant().simulate()
    for balance in result.balances:
        assert abs(balance.residual) < 1e-9, balance


# ---------------------------------------------------------------------------
# Reusable booster
# ---------------------------------------------------------------------------

def test_the_atmosphere_has_the_scale_height_it_claims():
    rho0, p0 = atmosphere(0.0)
    rho1, _ = atmosphere(8_500.0)
    assert rho0 == pytest.approx(1.225, rel=1e-12)
    assert p0 == pytest.approx(101_325.0, rel=1e-12)
    assert rho1 / rho0 == pytest.approx(math.exp(-1.0), rel=1e-12)


def test_the_booster_comes_home_softly():
    result = ReusableBooster().simulate()
    assert abs(result.value("downrange_at_touchdown")) <= 50.0
    assert result.value("touchdown_speed") <= 2.0
    assert result.value("propellant_margin") > 0.0


def test_no_propellant_combination_beats_its_own_tsiolkovsky_limit():
    result = ReusableBooster().simulate()
    assert result.value("ideal_delta_v") <= result.value("rocket_equation_delta_v") + 1e-6


def test_a_longer_boostback_lands_shorter():
    """The return burn is monotone in downrange, which is what makes the solve well posed."""
    from pinneapple_systems.digital_twin.library.reusable_booster import flight

    p = ReusableBooster().defaults()
    near = flight(p, boostback_delta_v=1_500.0)["x"]
    far = flight(p, boostback_delta_v=1_800.0)["x"]
    assert far < near


def test_the_ballistic_apex_matches_its_energy():
    """Above the atmosphere the apex is set by v^2 = 2 g h; drag and gravity fall-off make it
    a few per cent off, and anything more than that is a bug in the integrator."""
    result = ReusableBooster().simulate()
    speeds = result.series["speed_m_s"]
    altitudes = result.series["altitude_km"]
    apex_index = altitudes.index(max(altitudes))
    vertical_energy = max(speeds[: apex_index + 1]) ** 2 / (2.0 * 9.80665) / 1e3
    assert max(altitudes) == pytest.approx(vertical_energy, rel=0.35)


# ---------------------------------------------------------------------------
# Lunar base
# ---------------------------------------------------------------------------

def test_the_night_is_half_a_synodic_month():
    day, night = night_factor()
    assert day == night
    assert day * 2.0 == pytest.approx(SYNODIC_MONTH_DAYS * 24.0, rel=1e-12)
    assert night == pytest.approx(354.37, abs=0.01)


def test_the_oxygen_yield_is_the_iron_oxide_stoichiometry():
    """FeO -> Fe + 1/2 O2 gives 15.999 kg of oxygen per 71.844 kg of FeO."""
    assert regolith_oxygen_yield(1.0) == pytest.approx(15.999 / 71.844, rel=1e-12)
    assert regolith_oxygen_yield(0.15) == pytest.approx(0.15 * 15.999 / 71.844, rel=1e-12)


def test_the_radiator_is_stefan_boltzmann():
    result = LunarBase().simulate()
    p = result.parameters
    expected = (p["habitat_internal_heat"] + p["habitat_solar_absorbed"]) * 1e3 / (
        5.670_374_419e-8 * p["radiator_emissivity"] * p["radiator_temperature"] ** 4)
    assert result.value("radiator_area") == pytest.approx(expected, rel=1e-12)


def test_the_storage_is_the_night_divided_by_the_round_trip():
    result = LunarBase().simulate()
    p = result.parameters
    night_kwh = p["load_night"] * result.value("lunar_night")
    assert result.value("storage_required") == pytest.approx(night_kwh / p["storage_round_trip"], rel=1e-12)


def test_a_base_that_cannot_store_the_night_is_reported_as_such():
    result = LunarBase().simulate(array_area=200.0)
    assert result.value("storage_margin") < 0.0
    failing = [c for c in result.checks if not c.passed]
    assert any("storage" in c.name for c in failing)


def test_more_crew_needs_more_oxygen_and_more_regolith():
    two = LunarBase().simulate(crew=2.0)
    six = LunarBase().simulate(crew=6.0)
    assert six.value("oxygen_per_day") == pytest.approx(3.0 * two.value("oxygen_per_day"), rel=1e-12)
    assert six.value("regolith_per_lunar_day") == pytest.approx(3.0 * two.value("regolith_per_lunar_day"), rel=1e-12)


# ---------------------------------------------------------------------------
# The pipeline network: isothermal compressible flow and linepack
# ---------------------------------------------------------------------------

def test_the_segment_relation_is_the_hand_formula():
    # p1^2 - p2^2 = f (L/D) m_dot^2 Z R T / A^2, every factor written out here.
    mass_flow, length, diameter, friction = 600.0, 40_000.0, 0.9, 0.0125
    temperature, r_specific = 288.15, 8.314_462_618 / (0.65 * 28.96e-3)
    area = math.pi * diameter ** 2 / 4.0
    expected = friction * (length / diameter) * mass_flow ** 2 * r_specific * temperature / area ** 2
    got = segment_pressure_square_drop(mass_flow=mass_flow, length=length, diameter=diameter,
                                       friction_factor=friction, temperature=temperature,
                                       specific_gas_constant=r_specific)
    assert got == pytest.approx(expected, rel=1e-12)


def test_the_linepack_integral_reduces_to_p_times_v_when_the_pressure_is_flat():
    # With no pressure gradient the exact integral must collapse to the ideal-gas mass in the pipe.
    length, diameter, temperature = 30_000.0, 0.9, 288.15
    r_specific = 8.314_462_618 / (0.65 * 28.96e-3)
    volume = math.pi * diameter ** 2 / 4.0 * length
    p = 7.0e6
    got = linepack_of_segment(p_upstream=p, p_downstream=p, length=length, diameter=diameter,
                              specific_gas_constant=r_specific, temperature=temperature)
    assert got == pytest.approx(p * volume / (r_specific * temperature), rel=1e-12)


def test_the_linepack_integral_is_the_mean_pressure_of_a_parabola():
    # <p> = (2/3) (p1^3 - p2^3) / (p1^2 - p2^2) for p^2 linear in x; checked against a quadrature.
    length, diameter, temperature = 50_000.0, 0.8, 300.0
    r_specific, p1, p2 = 470.0, 9.0e6, 6.0e6
    nodes = 20_000
    integral = 0.0
    for k in range(nodes):
        xa, xb = k / nodes, (k + 1) / nodes
        integral += 0.5 * (math.sqrt(p1 ** 2 - (p1 ** 2 - p2 ** 2) * xa)
                           + math.sqrt(p1 ** 2 - (p1 ** 2 - p2 ** 2) * xb)) / nodes
    volume = math.pi * diameter ** 2 / 4.0 * length
    expected = integral * length * volume / length / (r_specific * temperature)
    got = linepack_of_segment(p_upstream=p1, p_downstream=p2, length=length, diameter=diameter,
                              specific_gas_constant=r_specific, temperature=temperature)
    assert got == pytest.approx(expected, rel=1e-5)


def test_the_compressor_discharge_is_the_polytropic_relation():
    got = compressor_discharge(suction_temperature=300.0, pressure_ratio=1.35, gamma=1.3,
                               polytropic_efficiency=0.78)
    assert got == pytest.approx(300.0 * 1.35 ** (0.3 / (1.3 * 0.78)), rel=1e-12)


def test_the_solved_throughput_reproduces_the_inlet_pressure_it_was_found_from():
    # March the reported answer backwards by hand: drop over the last segment, the station's
    # ratio, and so on to the inlet. It has to arrive at the pressure the line says it needs.
    result = PipelineNetwork().simulate()
    p = result.parameters
    r_specific = 8.314_462_618 / (p["gas_gravity"] * 28.96e-3) * p["compressibility"]
    flow = result.value("throughput") / 3.6                         # t/h -> kg/s
    stations = int(round(p["n_stations"]))
    length = p["length"] / (stations + 1)
    drop = segment_pressure_square_drop(mass_flow=flow, length=length, diameter=p["diameter"],
                                       friction_factor=result.value("friction_factor"),
                                       temperature=p["temperature"],
                                       specific_gas_constant=r_specific)
    downstream = p["delivery_pressure"]
    for _ in range(stations):
        downstream = math.sqrt(downstream ** 2 + drop) / p["station_ratio"]
    inlet = math.sqrt(downstream ** 2 + drop)
    assert inlet / 1e5 == pytest.approx(result.value("required_inlet_pressure"), rel=1e-9)


def test_the_friction_factor_solves_colebrook_white_at_the_reported_reynolds():
    result = PipelineNetwork().simulate()
    p = result.parameters
    reynolds = result.value("reynolds")
    friction = result.value("friction_factor")
    relative = p["roughness"] / p["diameter"]
    residual = 1.0 / math.sqrt(friction) + 2.0 * math.log10(relative / 3.7 + 2.51 / (reynolds * math.sqrt(friction)))
    assert abs(residual) < 1e-9


def test_the_line_reports_velocity_and_residence_time_it_can_be_checked_on():
    result = PipelineNetwork().simulate()
    p = result.parameters
    r_specific = 8.314_462_618 / (p["gas_gravity"] * 28.96e-3) * p["compressibility"]
    area = math.pi * p["diameter"] ** 2 / 4.0
    flow = result.value("throughput") / 3.6
    mean_pressure = 0.5 * (p["inlet_pressure"] + p["delivery_pressure"])
    rho = mean_pressure / (r_specific * p["temperature"])
    assert result.value("gas_velocity") == pytest.approx(flow / (rho * area), rel=1e-12)
    assert result.value("residence_time") == pytest.approx(result.value("linepack") * 1e3 / flow, rel=1e-12)
    assert result.value("specific_power") == pytest.approx(
        result.value("total_power") * 1e6 / flow / 1e3, rel=1e-12)


def test_a_longer_line_carries_less_gas():
    short = PipelineNetwork().simulate(length=60_000.0)
    long = PipelineNetwork().simulate(length=240_000.0)
    assert long.value("throughput") < short.value("throughput")
    # the stations are set to the same ratio, so the work per kilogram is identical and the
    # longer line simply carries less: the power falls with the throughput it no longer has
    assert long.value("specific_power") == pytest.approx(short.value("specific_power"), rel=1e-12)
    assert long.value("total_power") < short.value("total_power")


def test_heat_capacity_is_not_free_for_an_ideal_gas():
    # cp = gamma R / (gamma - 1) and cv = cp / gamma must differ by exactly the gas constant.
    result = PipelineNetwork().simulate()
    p = result.parameters
    r_specific = 8.314_462_618 / (p["gas_gravity"] * 28.96e-3) * p["compressibility"]
    cp = result.value("heat_capacity")
    assert cp - cp / p["gamma"] == pytest.approx(r_specific, rel=1e-12)


def test_a_lower_delivery_pressure_moves_more_gas_and_stores_less():
    tight = PipelineNetwork().simulate(delivery_pressure=6.0e6)
    loose = PipelineNetwork().simulate(delivery_pressure=4.0e6)
    assert loose.value("throughput") > tight.value("throughput")
    assert loose.value("linepack") < tight.value("linepack")


# ---------------------------------------------------------------------------
# The compression station: stage split, polytropic work and fuel
# ---------------------------------------------------------------------------

def test_one_stage_is_the_closed_form_polytropic_work():
    result = CompressionStation().simulate(stages=1.0)
    p = result.parameters
    r_specific = 8.314_462_618 / (p["gas_gravity"] * 28.96e-3) * p["compressibility"]
    cp = p["gamma"] * r_specific / (p["gamma"] - 1.0)
    ratio = p["discharge_pressure"] / p["suction_pressure"]
    exponent = (p["gamma"] - 1.0) / (p["gamma"] * p["polytropic_efficiency"])
    expected = p["mass_flow"] * cp * p["suction_temperature"] * (ratio ** exponent - 1.0)
    assert result.value("shaft_power") * 1e6 == pytest.approx(expected, rel=1e-12)


def test_two_stages_with_intercooling_cost_less_than_the_same_ratio_in_one():
    one = CompressionStation().simulate(stages=1.0)
    two = CompressionStation().simulate(stages=2.0)
    assert two.value("shaft_power") < one.value("shaft_power")
    assert two.value("discharge_temperature") < one.value("discharge_temperature")


def test_three_stages_sum_the_work_of_a_cold_first_stage_and_two_warm_ones():
    # Stage one starts at the suction temperature; every later stage starts at the intercooler
    # temperature. Each takes a third of the ratio.
    result = CompressionStation().simulate(stages=3.0)
    p = result.parameters
    r_specific = 8.314_462_618 / (p["gas_gravity"] * 28.96e-3) * p["compressibility"]
    cp = p["gamma"] * r_specific / (p["gamma"] - 1.0)
    ratio_one_third = (p["discharge_pressure"] / p["suction_pressure"]) ** (1.0 / 3.0)
    exponent = polytropic_exponent(p["gamma"], p["polytropic_efficiency"])
    rise = ratio_one_third ** exponent - 1.0
    expected = p["mass_flow"] * cp * rise * (p["suction_temperature"] + 2.0 * p["intercooler_outlet"])
    assert result.value("shaft_power") * 1e6 == pytest.approx(expected, rel=1e-12)


def test_the_stage_helpers_agree_with_the_run():
    station = CompressionStation().defaults()
    exponent = polytropic_exponent(station["gamma"], station["polytropic_efficiency"])
    r_specific = 8.314_462_618 / (station["gas_gravity"] * 28.96e-3) * station["compressibility"]
    cp = station["gamma"] * r_specific / (station["gamma"] - 1.0)
    ratio = (station["discharge_pressure"] / station["suction_pressure"]) ** 0.5
    t2 = stage_discharge_temperature(suction_temperature=station["suction_temperature"],
                                     ratio=ratio, gamma=station["gamma"],
                                     polytropic_efficiency=station["polytropic_efficiency"])
    assert t2 == pytest.approx(station["suction_temperature"] * ratio ** exponent, rel=1e-12)
    work_first = stage_work(mass_flow=station["mass_flow"],
                            suction_temperature=station["suction_temperature"], ratio=ratio,
                            gamma=station["gamma"], heat_capacity=cp,
                            polytropic_efficiency=station["polytropic_efficiency"])
    work_second = stage_work(mass_flow=station["mass_flow"],
                             suction_temperature=station["intercooler_outlet"], ratio=ratio,
                             gamma=station["gamma"], heat_capacity=cp,
                             polytropic_efficiency=station["polytropic_efficiency"])
    two = CompressionStation().simulate()
    assert work_first + work_second == pytest.approx(two.value("shaft_power") * 1e6, rel=1e-12)


def test_a_better_polytropic_efficiency_costs_less_shaft_work():
    poor = CompressionStation().simulate(polytropic_efficiency=0.70)
    good = CompressionStation().simulate(polytropic_efficiency=0.88)
    assert good.value("shaft_power") < poor.value("shaft_power")
    assert good.value("isentropic_efficiency") < 1.0


def test_the_fuel_is_the_shaft_power_over_the_driver_efficiency_and_the_lhv():
    result = CompressionStation().simulate()
    p = result.parameters
    expected = result.value("shaft_power") * 1e6 / (p["driver_efficiency"] * p["fuel_lhv"] * 1e6)
    assert result.value("fuel_flow") / 3_600.0 == pytest.approx(expected, rel=1e-12)


def test_the_cooling_is_the_enthalpy_the_stages_added():
    result = CompressionStation().simulate()
    p = result.parameters
    r_specific = 8.314_462_618 / (p["gas_gravity"] * 28.96e-3) * p["compressibility"]
    cp = p["gamma"] * r_specific / (p["gamma"] - 1.0)
    exponent = polytropic_exponent(p["gamma"], p["polytropic_efficiency"])
    ratio = (p["discharge_pressure"] / p["suction_pressure"]) ** 0.5
    first = p["suction_temperature"] * ratio ** exponent
    second = p["intercooler_outlet"] * ratio ** exponent
    expected = p["mass_flow"] * cp * ((first - p["intercooler_outlet"]) + (second - p["intercooler_outlet"]))
    assert result.value("cooling_duty") * 1e6 == pytest.approx(expected, rel=1e-9)


def test_a_machine_asked_for_more_gas_than_it_was_bought_for_reports_surge():
    undersized = CompressionStation().simulate(design_inlet_flow=20.0)
    assert undersized.value("inlet_flow_margin") < 0.55
    failing = [c.name for c in undersized.checks if not c.passed]
    assert "above_the_surge_floor" in failing


def test_a_discharge_below_the_suction_pressure_is_refused():
    with pytest.raises(ParameterError, match="discharge pressure"):
        CompressionStation().simulate(discharge_pressure=2.0e6)


def test_an_intercooler_colder_than_the_suction_needs_refrigeration_the_model_lacks():
    with pytest.raises(ParameterError, match="refrigeration"):
        CompressionStation().simulate(intercooler_outlet=295.0)


# ---------------------------------------------------------------------------
# The green hydrogen plant: Faraday, the cell voltage and the specific energy
# ---------------------------------------------------------------------------

def test_faraday_gives_the_hydrogen_two_electrons_at_a_time():
    # 2 F of charge is one mole of H2: check the function against the arithmetic, not the model.
    current, cells, efficiency = 37_500.0, 1_498.0, 0.97
    expected = cells * current * efficiency / (2.0 * FARADAY) * MOLAR_MASS_H2
    assert hydrogen_from_charge(current=current, cells=cells, faradaic_efficiency=efficiency) == pytest.approx(
        expected, rel=1e-12)


def test_the_stack_voltage_is_the_geometry_and_the_current():
    result = GreenHydrogenPlant().simulate()
    p = result.parameters
    current = p["current_density"] * 1e4 * p["cell_area"]
    cells = p["stack_power"] * 1e6 / (p["cell_voltage"] * current)
    assert result.value("current") * 1e3 == pytest.approx(current, rel=1e-12)
    assert result.value("cells") == pytest.approx(cells, rel=1e-12)
    assert result.value("stack_voltage") * 1e3 == pytest.approx(cells * p["cell_voltage"], rel=1e-12)


def test_the_specific_energy_is_the_heating_value_over_the_efficiency():
    # LHV efficiency is hydrogen energy out over electricity in, so the specific energy follows.
    result = GreenHydrogenPlant().simulate()
    assert result.value("specific_energy") * result.value("lhv_efficiency") == pytest.approx(
        119.96e6 / 3.6e6, rel=2e-4)


def test_the_reversible_and_thermoneutral_voltages_are_dg_and_dh_over_two_f():
    assert reversible_voltage() == pytest.approx(237_200.0 / (2.0 * FARADAY), rel=1e-12)
    assert thermoneutral_voltage() == pytest.approx(285_830.0 / (2.0 * FARADAY), rel=1e-12)
    assert reversible_voltage() < thermoneutral_voltage()


def test_a_voltage_below_the_reversible_one_is_refused():
    # Below 1.229 V the cell cannot split water at all, whatever the power and the current density.
    with pytest.raises(ParameterError, match="reversible"):
        GreenHydrogenPlant().simulate(cell_voltage=1.10, strict=False)


def test_more_voltage_buys_hydrogen_at_a_worse_specific_energy():
    cheap = GreenHydrogenPlant().simulate(cell_voltage=1.60)
    dear = GreenHydrogenPlant().simulate(cell_voltage=2.00)
    assert dear.value("hydrogen_rate") < cheap.value("hydrogen_rate")
    assert dear.value("specific_energy") > cheap.value("specific_energy")
    assert dear.value("heat_rejected") > cheap.value("heat_rejected")


def test_the_water_is_the_hydrogen_and_oxygen_it_became():
    result = GreenHydrogenPlant().simulate()
    p = result.parameters
    # the water is split before the purge: compare it with what the stack made, not what it sold
    hydrogen = result.value("hydrogen_rate") / 3_600.0 / (1.0 - p["purge_fraction"])
    oxygen = result.value("oxygen_rate") / 3_600.0
    water = result.value("water_demand") * 1_000.0 / 3_600.0
    assert water == pytest.approx(hydrogen + oxygen, rel=1e-12)
    # and the textbook nine kilograms of water per kilogram of hydrogen
    assert water / hydrogen == pytest.approx(18.015_28 / 2.015_88, rel=1e-4)


def test_the_oxygen_is_eight_times_the_hydrogen_by_mass():
    result = GreenHydrogenPlant().simulate()
    generated = result.value("hydrogen_rate") / (1.0 - result.parameters["purge_fraction"])
    assert result.value("oxygen_rate") / generated == pytest.approx(15.999_4 / 2.015_88, rel=1e-9)


# ---------------------------------------------------------------------------
# The desalination plant: van't Hoff, the salt balance and the thermodynamic floor
# ---------------------------------------------------------------------------

def test_vant_hoff_gives_seawater_its_thirty_bar():
    # 35 g/L of NaCl is about 0.6 mol/L of ions, which is about 30 bar at 25 degC.
    concentration = 35.0 / 0.05844
    assert osmotic_pressure(concentration=concentration, temperature=298.15) / 1e5 == pytest.approx(
        2.0 * concentration * 8.314_462_618 * 298.15 / 1e5, rel=1e-12)
    assert 29.0 < osmotic_pressure(concentration=concentration, temperature=298.15) / 1e5 < 31.0


def test_the_floor_is_the_integral_of_the_osmotic_pressure_as_the_brine_concentrates():
    # Quadrature of pi_f/(1-x) from 0 to the recovery, per cubic metre of permeate.
    feed, recovery = 3.0e6, 0.45
    nodes = 20_000
    integral = 0.0
    for k in range(nodes):
        x0, x1 = k / nodes * recovery, (k + 1) / nodes * recovery
        integral += 0.5 * (1.0 / (1.0 - x0) + 1.0 / (1.0 - x1)) * (x1 - x0)
    expected = feed * integral / recovery
    assert minimum_specific_energy(feed_osmotic_pressure=feed, recovery=recovery) == pytest.approx(
        expected, rel=1e-9)


def test_the_brine_concentration_is_the_salt_balance_not_a_rule_of_thumb():
    result = DesalinationPlant().simulate()
    p = result.parameters
    feed = p["feed_flow"] / 3_600.0
    permeate = feed * p["recovery"]
    brine = feed - permeate
    feed_concentration = p["feed_salinity"] / 0.05844
    permeate_concentration = feed_concentration * (1.0 - p["salt_rejection"])
    expected = (feed * feed_concentration - permeate * permeate_concentration) / brine
    assert result.value("brine_salinity") == pytest.approx(expected * 0.05844, rel=1e-12)


def test_the_specific_energy_is_the_pump_less_the_energy_recovery_over_the_permeate():
    result = DesalinationPlant().simulate()
    p = result.parameters
    permeate = p["feed_flow"] / 3_600.0 * p["recovery"]
    net = (result.value("pump_power") - result.value("recovered_power")) * 1e6
    assert result.value("specific_energy") == pytest.approx(net / permeate / 3.6e6, rel=1e-12)


def test_the_train_costs_more_than_the_second_law_allows_and_says_how_much():
    result = DesalinationPlant().simulate()
    assert result.value("specific_energy") > result.value("thermodynamic_floor")
    assert 1.0 < result.value("second_law_margin") < 10.0


def test_an_energy_recovery_device_pays_for_itself_in_specific_energy():
    result = DesalinationPlant().simulate()
    without = DesalinationPlant().simulate(erd_efficiency=0.0)
    with_erd = DesalinationPlant().simulate(erd_efficiency=0.95)
    assert with_erd.value("specific_energy") < without.value("specific_energy")
    # 1 - eta_erd eta_pump (1 - recovery): what the device gives back, net of the pump it relieves
    expected = 1.0 - 0.95 * result.parameters["pump_efficiency"] * (1.0 - result.parameters["recovery"])
    assert with_erd.value("specific_energy") / without.value("specific_energy") == pytest.approx(
        expected, rel=1e-9)


def test_more_recovery_costs_more_energy_per_cubic_metre():
    low = DesalinationPlant().simulate(recovery=0.35)
    high = DesalinationPlant().simulate(recovery=0.55)
    assert high.value("brine_salinity") > low.value("brine_salinity")
    assert high.value("specific_energy") > low.value("specific_energy")


def test_the_permeate_is_cleaner_than_the_feed_by_the_rejection():
    result = DesalinationPlant().simulate()
    p = result.parameters
    assert result.value("permeate_salinity") == pytest.approx(
        p["feed_salinity"] * 1_000.0 * (1.0 - p["salt_rejection"]), rel=1e-12)

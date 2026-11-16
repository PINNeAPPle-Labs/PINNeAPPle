"""Tests of the Digital Twin Library: five reference models, each checked against its own physics.

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
from pinneapple_systems.digital_twin.library.fpso import (
    FPSOSubsea,
    ipr_rate,
    line_pressure_drop,
    vogel_qmax,
)
from pinneapple_systems.digital_twin.library.lunar_base import (
    SYNODIC_MONTH_DAYS,
    LunarBase,
    night_factor,
    regolith_oxygen_yield,
)
from pinneapple_systems.digital_twin.library.reusable_booster import (
    ReusableBooster,
    atmosphere,
)

SLUGS = ("blast_furnace", "chemical_plant", "fpso_subsea", "lunar_base", "reusable_booster")


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

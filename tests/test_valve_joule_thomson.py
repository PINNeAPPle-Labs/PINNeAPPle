"""Joule–Thomson cooling across a valve (pinneapple_systems.process_components.joule_thomson, #1463)."""
import pytest

pytest.importorskip("CoolProp")


def test_parse_valve_request_reads_portuguese_and_english():
    from pinneapple_systems.process_components.joule_thomson import parse_valve_request

    en = parse_valve_request("inlet pressure 70 bar, inlet temperature 25 °C, outlet pressure 20 bar, Cv 20, DN 100. "
                             "My CFD outlet temperature is 24.6 °C. Methane 92 %, ethane 5 %, nitrogen 3 %")
    c = en["case"]
    assert (c["P1_bar"], c["T1_C"], c["P2_bar"], c["Cv"], c["pipe_d_mm"], c["cfd_T2_C"]) == (70, 25, 20, 20, 100, 24.6)
    assert en["gas"] == pytest.approx({"Methane": 0.92, "Ethane": 0.05, "Nitrogen": 0.03}) and en["defaulted"] == []
    pt = parse_valve_request("gás natural numa válvula globo, de 60 barg para 15 barg, temperatura de entrada 30 °C")
    assert pt["case"]["P1_bar"] == pytest.approx(61.01325) and pt["case"]["P2_bar"] == pytest.approx(16.01325)
    assert pt["case"]["T1_C"] == 30 and {"Cv", "pipe_d_mm", "composition"} <= set(pt["defaulted"])


def test_throttling_ideal_gas_has_no_joule_thomson_effect_and_real_gas_cools():
    from pinneapple_systems.process_components.joule_thomson import DEFAULT_GAS, throttle

    ideal = throttle(DEFAULT_GAS, 70e5, 298.15, 20e5, "ideal gas")
    real = throttle(DEFAULT_GAS, 70e5, 298.15, 20e5, "GERG-2008")
    assert ideal["dT_K"] == 0.0 and real["dT_K"] < -15.0 and abs(real["h_residual_J_kg"]) < 1.0


def test_valve_joule_thomson_study_literal(tmp_path):
    from pinneapple_systems.process_components.joule_thomson import valve_joule_thomson_study

    r = valve_joule_thomson_study("inlet pressure 70 bar, inlet temperature 25 °C, outlet pressure 20 bar, Cv 20, "
                                  "DN 100. CFD outlet temperature 24.6 °C", out_dir=str(tmp_path))
    st = {c["code"]: c["status"] for c in r["checks"]}
    assert st["thermo.energy_conservation"] == st["thermo.jt_integral_vs_flash"] == "pass"
    assert st["thermo.ideal_gas_limit"] == "pass" and st["cfd.outlet_temperature"] == "fail"   # no cooling in the CFD
    o = r["outputs"]
    assert o["dT_isenthalpic_K"] == pytest.approx(o["dT_mu_integral_K"], rel=0.01)
    assert o["T_jet_coldest_C"] < o["T2_C"] and o["mach_jet"] == pytest.approx(1.0, abs=1e-3)
    for f in ("throttling_path.png", "eos_comparison.png", "temperature_drop_map.png", "veriphysics_grade.png"):
        assert f in r["files"] and (tmp_path / f).stat().st_size > 0


def test_the_valve_flow_capability_matches_the_request():
    import pinneapple_flow.nodes_ext  # noqa: F401
    from pinneapple_flow.planner.extensions import match

    best = match("Estou simulando o efeito Joule-Thomson numa válvula globo com gás natural e o CFD não bate")
    assert best and best[0][1].id == "process.valve_joule_thomson"
    assert all(c.id != "process.valve_joule_thomson" for _, c, _ in match("optimise the wing planform"))


def test_pure_methane_literal_calls():
    """Literal calls (the Studio's recorded inputs): pure methane from 70 to 20 bar at 25 °C."""
    from pinneapple_systems.process_components.joule_thomson import isenthalpic_path, throttle

    r = throttle({"Methane": 1.0}, 7000000.0, 298.15, 2000000.0)
    assert r["dT_K"] < -10.0 and abs(r["h_residual_J_kg"]) < 1.0
    p = isenthalpic_path({"Methane": 1.0}, 7000000.0, 298.15, 2000000.0, 5)
    assert len(p["T"]) == 5 and abs(p["T"][-1] - r["T2_K"]) < 1e-6


def test_parser_reads_cfd_temperature_in_either_order_and_the_language():
    from pinneapple_systems.process_components.joule_thomson import parse_valve_request
    pt = parse_valve_request("Estou simulando gás natural numa válvula globo; "
                             "a temperatura de saída do meu CFD é 24,6 °C")
    assert pt["case"]["cfd_T2_C"] == 24.6 and pt["lang"] == "pt"
    en = parse_valve_request("natural gas through a globe valve; my CFD outlet temperature is 24.6 C")
    assert en["case"]["cfd_T2_C"] == 24.6 and en["lang"] == "en"

"""The multiphysics-coupling workflow (#1209): strategies, balances and closed-form coupled solutions."""
import pytest

from pinneapple_simulation.multiphysics import multiphysics_workflow


def _status(r, code):
    return next(c["status"] for c in r["checks"] if c["code"] == code)


def test_joule_wire_two_way_matches_the_closed_form_and_one_way_is_not_enough():
    r = multiphysics_workflow(case="joule_wire")
    assert r["verdict"] == "pass", r["failed"]
    assert _status(r, "closed_form") == "pass"
    assert r["recommended"] == "gauss_seidel_aitken"
    assert 0 < r["coupling_strength"]["contraction"] < 1
    assert r["one_way_error"]["T_max_K"] > 0.05
    s = r["strategies"]
    assert s["gauss_seidel_aitken"]["iterations"] < s["gauss_seidel"]["iterations"]


def test_joule_wire_at_low_current_one_way_is_enough():
    r = multiphysics_workflow(case="joule_wire", case_params={"current": 0.5})
    assert r["recommended"] == "one_way"
    assert r["verdict"] == "pass"


def test_joule_wire_beyond_runaway_is_refused():
    with pytest.raises(ValueError, match="runaway"):
        multiphysics_workflow(case="joule_wire", case_params={"current": 20.0})


def test_bimetal_strip_is_one_way_and_matches_timoshenko():
    r = multiphysics_workflow(case="bimetal_strip")
    assert r["verdict"] == "pass", r["failed"]
    assert r["recommended"] == "one_way"
    assert r["closed_form"]["field_error"] < 1e-10
    assert _status(r, "interface.axial_force") == "pass"


def test_bimetal_gradient_skips_the_closed_form_but_balances_hold():
    r = multiphysics_workflow(case="bimetal_strip", case_params={"t_top": 420.0})
    assert _status(r, "closed_form") == "skip"
    assert _status(r, "interface.heat_flux") == "pass"


def test_conjugate_channel_energy_balance_and_exact_solution():
    r = multiphysics_workflow(case="conjugate_channel")
    assert r["verdict"] == "pass", r["failed"]
    assert _status(r, "energy.global") == "pass"
    assert r["recommended"] in ("gauss_seidel_aitken", "monolithic")


def test_strong_coupling_recommends_monolithic():
    from pinneapple_simulation.multiphysics import recommend_strategy
    assert recommend_strategy(1.3, 0.5, 0.01)[0] == "monolithic"


def test_the_planner_proposes_the_workflow_in_portuguese_and_english():
    import pinneapple_flow.nodes_ext  # noqa: F401
    from pinneapple_flow.planner.extensions import match
    for text in ("acoplamento multifísica eletrotérmico com efeito joule",
                 "thermal stress in a bimetal strip, partitioned coupling with Aitken"):
        assert match(text)[0][1].id == "multiphysics.couple"

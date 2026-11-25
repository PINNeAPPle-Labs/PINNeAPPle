"""Contact with friction as a library capability: the decks, the closed forms and — where ``ccx`` exists — the run.

The verification is checked twice, as the app does it: here, against numbers written by hand (``mu*N``, ``tan(theta)``,
Lamé), and, when a ``ccx`` binary is available, by solving the real deck. Without a solver the run says so; it never
skips silently into a pass.
"""
from __future__ import annotations

import math

import pytest

from pinneapple_data import cae
from pinneapple_simulation.external_solvers.calculix import contact as K
from pinneapple_simulation.external_solvers.calculix.contact import run as R
from pinneapple_simulation.external_solvers.calculix.contact import verify as V

SOLVER = R.ccx_binary()
needs_solver = pytest.mark.skipif(SOLVER is None, reason="no ccx binary (set CCX_BIN) in this environment")
CONTACT_KEYWORDS = ("*SURFACE INTERACTION", "*FRICTION", "*CONTACT PAIR", "*SURFACE BEHAVIOR", "*CONTACT FILE")


# ----------------------------------------------------------------------------------------------------- the cases
@pytest.mark.parametrize("case_id", sorted(K.CASES))
def test_every_case_writes_a_deck_the_solver_can_read_and_prepomax_can_open(case_id):
    case, params, built = K.build(case_id, {})
    assert case_id == case.id and case.title and case.summary and case.physics
    for keyword in ("*NODE", "*ELEMENT", "*STEP", "*END STEP", "*NODE PRINT") + CONTACT_KEYWORDS:
        assert keyword in built.deck, f"{case_id} has no {keyword}"
    assert f"*FRICTION\n{params['friccao']:g}, " in built.deck
    assert built.analytic and built.interface and built.coords


@pytest.mark.parametrize("case_id", sorted(K.CASES))
def test_the_parameters_are_checked_against_the_range_the_case_is_built_for(case_id):
    case = K.get(case_id)
    assert case.resolve({}) == {p.name: p.default for p in case.params}
    with pytest.raises(ValueError, match="outside the range"):
        case.resolve({"malha": 10 ** 6})
    with pytest.raises(ValueError, match="has no parameter"):
        case.resolve({"nao_existe": 1.0})


def test_the_catalog_is_the_menu_of_the_six_cases():
    catalog = K.catalog()
    assert {c["id"] for c in catalog["cases"]} == {"bloco_plano", "junta_dupla", "cilindro_hertz", "rampa",
                                                  "parafuso_pretensao", "ajuste_forcado"}
    assert catalog["units"].startswith("mm, N, MPa")
    for entry in catalog["cases"]:
        assert entry["references"] and entry["params"] and entry["assumptions"]
        for param in entry["params"]:
            assert param["low"] <= param["default"] <= param["high"], param["name"]


def test_an_unknown_case_names_the_ones_there_are():
    with pytest.raises(KeyError, match="bloco_plano"):
        K.get("nao_existe")


# -------------------------------------------------------------------------------------------- the closed forms
def test_hertz_line_contact_gives_the_textbook_half_width_and_peak_pressure():
    out = V.hertz_line(1000.0, 100.0, 10.0, 210000.0, 0.3, rigid=True)
    e_star = 210000.0 / (1 - 0.3 ** 2)
    assert out["b"] == pytest.approx(math.sqrt(4 * 1000.0 * 100.0 / (math.pi * 10.0 * e_star)))
    assert out["p0"] == pytest.approx(2 * 1000.0 / (math.pi * out["b"] * 10.0))
    assert V.hertz_line(0.0, 100.0, 10.0, 210000.0, 0.3) == {}


def test_lame_shrink_fit_with_one_material_matches_the_closed_sum():
    pressure = V.lame_shrink_fit(0.02, 10.0, 4.0, 20.0, 210000.0, 0.3, plane_strain=True)
    e_eff = 210000.0 / (1 - 0.3 ** 2)
    ring = (20.0 ** 2 + 10.0 ** 2) / (20.0 ** 2 - 10.0 ** 2)
    shaft = (10.0 ** 2 + 4.0 ** 2) / (10.0 ** 2 - 4.0 ** 2)
    assert pressure == pytest.approx(0.02 * e_eff / (10.0 * (ring + shaft)))
    # a plane-stress formula against the same model is a different number at nu = 0.3, which is why the model is strain
    assert V.lame_shrink_fit(0.02, 10.0, 4.0, 20.0, 210000.0, 0.3, plane_strain=False) == pytest.approx(
        pressure * (1 - 0.3 ** 2))


def test_the_hertz_profile_fit_recovers_the_curve_it_was_given():
    p0, b = 40.0, 0.5
    profile = [(x, p0 * math.sqrt(1 - (x / b) ** 2)) for x in (-b, -0.75 * b, -0.5 * b, -0.25 * b, 0.0,
                                                              0.25 * b, 0.5 * b, 0.75 * b, b)]
    fit = V.hertz_fit(profile)
    assert fit is not None
    assert fit[0] == pytest.approx(p0) and fit[1] == pytest.approx(b) and fit[2] == pytest.approx(1.0)


# ------------------------------------------------------------------------------------ the checks, without a solver
def _parsed(contact: dict[int, dict[str, float]], reaction: tuple[float, float, float],
            verdict: str = "ok") -> dict:
    return {"totals": {"BFIX": cae.Totals("BFIX", reaction, 2.0)}, "contact": contact, "verdict": verdict,
            "contact_until": {}}


def _patch(interface, pressure: float, shear: float, slip: float, opening: float = 0.0) -> dict:
    return {n: {"opening": opening, "slip1": slip, "slip2": 0.0, "pressure": pressure, "shear1": shear,
                "shear2": 0.0} for n in interface}


def test_the_saturated_slider_passes_every_check_of_its_own_case():
    case, params, built = K.build("bloco_plano", {})
    friction = params["friccao"] * built.analytic["normal"]
    contact = _patch(built.interface, pressure=5.0, shear=params["friccao"] * 5.0, slip=params["deslizamento"])
    checks = {c.name: c for c in case.checks(params, built, _parsed(contact, (-friction, built.analytic["normal"], 0.0)))}
    assert set(checks) >= {"equilibrio.normal", "atrito.limite", "atrito.saturacao", "atrito.cone"}
    assert all(c.outcome == V.PASS for c in checks.values()), {n: c.detail for n, c in checks.items()}


def test_a_joint_that_transmits_more_than_mu_times_n_fails_the_check():
    """The whole point of the case: a converged run whose interface carries more than the cone allows is wrong."""
    case, params, built = K.build("bloco_plano", {})
    over = 1.5 * params["friccao"] * built.analytic["normal"]
    parsed = _parsed(_patch(built.interface, 5.0, 1.5 * params["friccao"] * 5.0, params["deslizamento"]),
                     (-over, built.analytic["normal"], 0.0))
    checks = {c.name: c for c in case.checks(params, built, parsed)}
    assert checks["atrito.limite"].outcome == V.FAIL
    assert checks["atrito.cone"].outcome == V.FAIL


def test_a_run_without_contact_results_is_inconclusive_and_never_a_pass():
    case, params, built = K.build("bloco_plano", {})
    checks = {c.name: c for c in case.checks(params, built, _parsed({}, (0.0, built.analytic["normal"], 0.0)))}
    assert checks["atrito.saturacao"].outcome == V.INCONCLUSIVE
    assert checks["atrito.cone"].outcome == V.INCONCLUSIVE


def test_the_incline_locks_below_the_friction_angle_and_slips_above_it():
    case, params, built = K.build("rampa", {"friccao": 0.5, "angulo": 15.0})
    # the support carries the vertical load: the interface reaction is F cos(theta) normal and F sin(theta) tangential
    held = _parsed(_patch(built.interface, 5.0, 0.0, 0.0), (0.0, params["carga"], 0.0))
    checks = {c.name: c for c in case.checks(params, built, held)}
    assert checks["rampa.estado"].outcome == V.PASS
    assert checks["rampa.resultante"].outcome == V.PASS
    # above atan(mu) the solver finds no equilibrium, and that is the predicted answer, not a failed run
    steep, steep_params, steep_built = K.build("rampa", {"friccao": 0.5, "angulo": 40.0})
    slipped = {c.name: c for c in steep.checks(steep_params, steep_built, _parsed({}, (0.0, 0.0, 0.0),
                                                                                   verdict="not_converged"))}
    assert slipped["rampa.estado"].outcome == V.PASS
    converged = {c.name: c for c in steep.checks(steep_params, steep_built, _parsed({}, (0.0, 0.0, 0.0),
                                                                                    verdict="ok"))}
    assert converged["rampa.estado"].outcome == V.FAIL


def test_the_shrink_fit_pressure_is_lame_and_the_interface_has_to_close():
    case, params, built = K.build("ajuste_forcado", {})
    want = built.analytic["lame_pressure"]
    contact = _patch(built.interface, pressure=want, shear=0.0, slip=0.0)
    checks = {c.name: c for c in case.checks(params, built, _parsed(contact, (0.0, 0.0, 0.0)))}
    assert checks["ajuste.pressao"].outcome == V.PASS
    assert checks["ajuste.contato_total"].outcome == V.PASS
    open_patch = _patch(built.interface, pressure=want, shear=0.0, slip=0.0)
    open_patch[next(iter(open_patch))]["pressure"] = 0.0
    broken = {c.name: c for c in case.checks(params, built, _parsed(open_patch, (0.0, 0.0, 0.0)))}
    assert broken["ajuste.contato_total"].outcome == V.FAIL


# --------------------------------------------------------------------------------------------------- the solver
def test_without_a_solver_the_deck_is_written_and_nothing_is_invented(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "ccx_binary", lambda *a, **k: None)
    result = K.run_case("bloco_plano", {}, tmp_path)
    assert result["solver"]["available"] is False and "ok" not in result
    assert "no CalculiX solver" in result["error"]
    assert (tmp_path / "bloco_plano.inp").is_file()
    assert result["checks"] == []


@needs_solver
@pytest.mark.parametrize("case_id", ["bloco_plano", "parafuso_pretensao"])
def test_the_deck_solves_and_every_closed_form_check_passes(case_id, tmp_path):
    result = K.run_case(case_id, {}, tmp_path, binary=SOLVER, timeout=600)
    assert result["ok"], result.get("error") or result.get("detail")
    outcomes = {c.name: c.outcome for c in result["checks"]}
    assert outcomes.pop("solver.convergiu") == V.PASS
    assert all(o == V.PASS for o in outcomes.values()), outcomes
    for suffix in (".inp", ".frd", ".dat", ".sta"):
        assert (tmp_path / f"{case_id}{suffix}").is_file(), suffix


@needs_solver
def test_the_transmitted_force_follows_mu_and_not_the_load_alone(tmp_path):
    low = K.run_case("bloco_plano", {"friccao": 0.2}, tmp_path / "a", binary=SOLVER, timeout=600)
    high = K.run_case("bloco_plano", {"friccao": 0.4}, tmp_path / "b", binary=SOLVER, timeout=600)
    f_low = abs(low["measured"]["reactions"]["BFIX"]["fx"])
    f_high = abs(high["measured"]["reactions"]["BFIX"]["fx"])
    assert f_low == pytest.approx(0.2 * 500.0, rel=0.02)
    assert f_high == pytest.approx(0.4 * 500.0, rel=0.02)
    assert low["measured"]["coulomb_ratio"] == pytest.approx(0.2, abs=0.01)
    assert high["measured"]["coulomb_ratio"] == pytest.approx(0.4, abs=0.01)


@needs_solver
def test_the_results_are_read_by_the_data_layer_and_not_by_a_second_parser(tmp_path):
    """The run's numbers have to be the ones :mod:`pinneapple_data.cae` reads from the same files."""
    result = K.run_case("bloco_plano", {"malha": 3}, tmp_path, binary=SOLVER, timeout=600)
    assert result["ok"]
    contact = cae.contact((tmp_path / "bloco_plano.frd").read_bytes())
    totals = cae.read_totals((tmp_path / "bloco_plano.dat").read_bytes())
    assert result["measured"]["coulomb_ratio"] == cae.coulomb_ratio(contact, result["interface"])
    assert result["measured"]["reactions"]["BFIX"]["fx"] == totals["BFIX"].values[0]

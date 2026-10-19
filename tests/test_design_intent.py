"""Design from intent (pinneapple_design.intent): parsing, units, template selection, refusal, the models of each
demo against closed forms, determinism, the package files and the entry points."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil

import numpy as np
import pytest

import pinneapple_design.intent as I
from pinneapple_design.intent import materials as M
from pinneapple_design.intent.physics import loop_track as LT

FAST = dict(n_screen=300, budget=120, robust_samples=400)
HAS_NEWTON = importlib.util.find_spec("newton") is not None
HAS_CCX = shutil.which("ccx") is not None


# ============================================================================================ parsing and units
def test_numbers_with_units_are_read_and_converted_to_si():
    ms = I.find_quantities("a bracket that holds 2 kN at 150 mm, 85 °C, 40W, 1.5 kg, NACA 2412 and 6061-T6")
    got = {(m.value, m.unit) for m in ms}
    assert {(2.0, "kN"), (150.0, "mm"), (85.0, "°C"), (40.0, "W"), (1.5, "kg")} <= got
    assert not any(m.unit for m in ms if m.value in (2412.0, 6061.0))   # part codes carry no unit
    spec = I.formulate("a steel shelf bracket holding 2 kN at 150 mm from the wall")
    assert spec.template == "bracket"
    assert spec.values["load"].value == pytest.approx(2000.0) and spec.values["load"].unit == "N"
    assert spec.values["arm_length"].value == pytest.approx(0.150)
    assert spec.values["material"].value == "steel_1018"
    assert spec.values["load"].provenance == I.STATED and spec.values["load"].quote == "2 kN"
    assert spec.values["safety_factor"].provenance == I.DEFAULTED and spec.values["safety_factor"].source
    assert spec.complete


def test_temperatures_are_absolute_and_mass_loads_are_accepted():
    spec = I.formulate("heat sink for a 40 W chip in still air, keep it below 85 °C, ambient 35 °C")
    assert spec.template == "heatsink"
    assert spec["t_limit"] == pytest.approx(358.15) and spec["t_ambient"] == pytest.approx(308.15)
    assert spec["air_velocity"] == 0.0 and spec.values["air_velocity"].quote == "still air"
    b = I.formulate("a bracket holding 20 kg at 120 mm")
    assert b["load_mass"] == pytest.approx(20.0) and not b.questions


def test_every_value_is_stated_defaulted_or_missing_and_missing_critical_values_become_questions():
    spec = I.formulate("a heat sink for a 40 W chip in still air")
    assert [q.slot for q in spec.questions] == ["t_limit"]
    assert spec.values["t_limit"].provenance == I.MISSING
    assert all(q.provenance in (I.STATED, I.DEFAULTED, I.MISSING) for q in spec.values.values())
    assert all(q.source for q in spec.values.values() if q.provenance == I.DEFAULTED)
    out = I.design("a heat sink for a 40 W chip in still air")
    assert isinstance(out, I.NeedsInput) and out.questions[0].slot == "t_limit"
    answered = I.formulate("a heat sink for a 40 W chip in still air", answers={"t_limit": (85, "°C")})
    assert answered.complete and answered.values["t_limit"].provenance == I.ANSWERED
    assert I.formulate("a bracket that holds 2 kN").questions[0].slot == "arm_length"
    assert I.formulate("a bracket 100 mm long").questions[0].slot == "load"


def test_units_are_validated_against_dimension_and_range():
    sl = I.get_template("bracket").slot_map()["load"]
    with pytest.raises(I.SpecError, match="expected"):
        sl.quantity(3, "m", I.STATED)
    with pytest.raises(I.SpecError, match="outside the range"):
        sl.quantity(900, "kN", I.STATED)
    spec = I.formulate("a toy car loop of radius 5 m")
    assert spec.issues and "outside the range" in spec.issues[0]
    assert isinstance(I.design("a toy car loop of radius 5 m"), I.NeedsInput)


def test_template_selection():
    cases = {"I want a toy car that does a loop": "loop_car", "a bracket that holds 2 kN": "bracket",
             "a heat sink for a 40 W chip in still air": "heatsink",
             "a wing for a 1.2 kg model glider": "glider_wing"}
    for text, name in cases.items():
        assert I.select_template(text)[0][0] == name, text


@pytest.mark.parametrize("text, kind", [
    ("design a gun", "refused"),
    ("an electric motor for a drone with 200 W", "needs"),
    ("a fusion reactor for my garage", "needs"),
    ("a bridge across a river that holds 40 t", "needs"),
    ("a roller coaster loop for passengers", "needs"),
    ("make me a good sandwich", "no_template"),
    ("", "no_template"),
])
def test_out_of_scope_requests_are_refused_with_what_is_needed(text, kind):
    r = I.design(text)
    assert isinstance(r, I.OutOfScope) and r.kind == kind
    assert r.reason and r.needs
    assert json.dumps(r.to_dict())


def test_llm_values_are_used_only_when_quoted_from_the_text():
    class Fake:
        def __init__(self, values):
            self.values = values

        def extract(self, text, schema):
            assert "bracket" in schema["templates"]
            return {"template": "bracket", "values": self.values}

    text = "a bracket for a 3 kN load, the arm reaches one hundred and twenty mm"
    good = Fake([{"slot": "arm_length", "value": 120, "unit": "mm", "quote": "one hundred and twenty mm"}])
    s = I.formulate(text, llm=good)
    assert s.parser == "llm+template"
    assert s["load"] == pytest.approx(3000.0)                        # deterministic reading kept
    assert "number is not in its quote" in " ".join(s.issues)       # "120" is not written in the quote
    invented = Fake([{"slot": "arm_length", "value": 0.2, "unit": "m", "quote": "200 mm arm"}])
    s2 = I.formulate("a bracket for a 3 kN load", llm=invented)
    assert "quote is not in the request" in " ".join(s2.issues) and s2.values["arm_length"].provenance == I.MISSING

    class Broken:
        def extract(self, text, schema):
            raise ConnectionError("offline")
    s3 = I.formulate("a bracket for a 3 kN load 150 mm long", llm=Broken())
    assert s3["arm_length"] == pytest.approx(0.15) and "LLM extraction failed" in " ".join(s3.issues)


def test_materials_have_cited_properties():
    for m in M.MATERIALS.values():
        assert m.props, m.key
        for k, p in m.props.items():
            assert p.value > 0 and p.unit and len(p.source) > 15, (m.key, k)
    assert {m.key for m in M.candidates("youngs_modulus", "poisson", "yield_strength", category="metal")} >= {
        "al6061_t6", "steel_1018", "ss304"}
    assert M.find("a PLA plate") == ["pla_4043d"] and M.find("an explanation") == []


# ============================================================================================ loop physics
def test_loop_1d_model_conserves_energy_without_losses():
    for Lc in (0.0, 0.15):
        r = LT.along_track(0.15, Lc, 3.2, 0.05, crr=0, cd=0, com_height=0.01, n=2000)
        cf = LT.closed_form(0.15, Lc, 3.2, 0.05, crr=0, cd=0, com_height=0.01)
        # v_top^2 = v0^2 - 2 g (z_top - 2 h) for the centre of mass
        ref = math.sqrt(3.2 ** 2 - 2 * LT.G0 * (float(cf["z_top"][0]) - 0.02))
        assert r.v_top[0] == pytest.approx(ref, rel=2e-3)
        assert cf["v_top"][0] == pytest.approx(ref, rel=1e-9)
        assert abs(r.energy_lost[0]) < 4e-3


def test_loop_threshold_matches_v_top_squared_equal_g_r():
    """Circular loop, no losses: the car completes iff v0^2 >= 5 g R_c (v_top^2 = g R_c at the top)."""
    R, h = 0.2, 0.01
    v_crit = math.sqrt(5 * LT.G0 * (R - h))
    v = np.array([0.98, 1.02]) * v_crit
    r = LT.along_track(R, 0.0, v, 0.05, crr=0, cd=0, com_height=h, n=3000)
    assert list(r.completes) == [False, True]
    assert r.top_margin[1] == pytest.approx(1.02 ** 2 * 5 - 5, abs=0.02)


def test_clothoid_entry_lowers_the_peak_load():
    v0 = 3.5
    n = [LT.along_track(0.15, c * 0.15, v0, 0.05, crr=0, cd=0).n_max[0] for c in (0.0, 1.0, 2.0)]
    assert n[0] > n[1] > n[2]
    # circle: peak at the bottom right after entry, 1 + v^2 / (g R_c)
    assert n[0] == pytest.approx(1 + v0 ** 2 / (LT.G0 * (0.15 - 0.008)), rel=0.01)


def test_loop_losses_are_monotonic():
    base = dict(crr=0.0, cd=0.0)
    v = LT.along_track(0.15, 0.1, 3.5, 0.05, **base).v_top[0]
    vr = LT.along_track(0.15, 0.1, 3.5, 0.05, crr=0.03, cd=0.0).v_top[0]
    vd = LT.along_track(0.15, 0.1, 3.5, 0.05, crr=0.0, cd=1.0).v_top[0]
    assert vr < v and vd < v


def test_track_geometry_is_closed_and_tall_as_computed():
    tr = LT.LoopTrack(0.15, 0.1)
    p = tr.sample(3000)
    assert p["theta"][-1] == pytest.approx(2 * math.pi, rel=1e-3)
    assert tr.height() == pytest.approx(float(LT.closed_form(0.15, 0.1, 3, 0.05)["z_top"][0]), rel=2e-3)
    parts = LT.track_meshes(tr, n=200)
    from pinneapple_design.geometry_library.meshops import quality
    for V, F in parts.values():
        assert quality(V, F)["watertight"]


# ============================================================================================ bracket physics
def test_bracket_models_agree_with_beam_theory():
    from pinneapple_design.intent.physics import bracket as B
    P, a, b, w, t, E = 2000.0, 0.08, 0.10, 0.04, 0.008, 205e9
    sig_b, d_b = B.beam(P, b, w, t, E)
    d_h, sig_h, res, mesh = B.hex_model(P, a, b, w, t, E, 0.29, n_t=2, fillet=t)
    assert sig_h == pytest.approx(sig_b, rel=0.03)                # nominal root stress vs M c / I
    assert d_h == pytest.approx(d_b, rel=0.05)                    # tip deflection (the base leg adds, the fillet
    react = -res.u.shape[0]                                       # stiffens)
    assert react < 0


# ============================================================================================ demos, end to end
def test_heatsink_demo_end_to_end_verified_by_finite_volume(tmp_path):
    pkg = I.design("a heat sink for a 40 W chip in still air", answers={"t_limit": (85, "°C")}, out_dir=str(tmp_path),
                   robust_samples=400)              # seven variables: the default screening and budget
    assert isinstance(pkg, I.DesignPackage)
    assert pkg.verification.ran
    row = pkg.verification.comparison[0]
    assert row["output"] == "t_source_c" and row["rel_err"] < 0.02            # FVM vs resistance network
    energy = next(c for c in pkg.verification.result.checks if c["name"] == "energy_balance")
    assert energy["passed"]
    assert pkg.candidate.outputs["margin_temp"] >= 0
    assert pkg.verification.result.outputs["t_source_c"] <= 85.0
    _check_package_files(tmp_path, pkg)


def test_glider_demo_end_to_end_with_lattice_convergence(tmp_path):
    pkg = I.design("a wing for a 1.2 kg model glider, at most 2 m span", out_dir=str(tmp_path), **FAST)
    assert isinstance(pkg, I.DesignPackage) and pkg.verification.ran
    ch = {c["name"]: c for c in pkg.verification.result.checks}
    assert all(c["passed"] for c in ch.values())
    e = pkg.verification.result.outputs["span_eff"]
    assert 0.9 < e <= 1.0                                          # Munk: e <= 1 for a planar wing
    assert pkg.design["span"] <= 2.0 + 1e-9
    assert pkg.assessment.verdict in ("will work", "likely to work")
    _check_package_files(tmp_path, pkg)


@pytest.mark.skipif(not HAS_CCX, reason="CalculiX (ccx) is not installed")
def test_bracket_demo_end_to_end_with_calculix(tmp_path):
    from pinneapple_design.intent.physics.bracket import gmsh_exe
    if gmsh_exe() is None:
        pytest.skip("gmsh is not installed")
    pkg = I.design("a bracket that holds 2 kN", answers={"arm_length": (100, "mm")}, out_dir=str(tmp_path), **FAST)
    ch = {c["name"]: c for c in pkg.verification.result.checks}
    assert ch["equilibrium (support reaction = load)"]["passed"]
    assert ch["arm bending stress vs M c / I"]["passed"]
    assert pkg.verification.result.outputs["margin_yield"] >= 0
    _check_package_files(tmp_path, pkg)


def test_bracket_and_loop_without_full_solver_report_it_as_not_run():
    pkg = I.design("a bracket that holds 2 kN", answers={"arm_length": (100, "mm")}, verify=False, **FAST)
    assert not pkg.verification.ran and pkg.assessment.verdict in ("likely to work", "unverified")
    names = [c.name for c in pkg.assessment.trust.checks]
    assert "full-fidelity verification" in names
    loop = I.design("I want a toy car that does a loop", verify=False, **FAST)
    assert loop.candidate.outputs["margin_contact"] >= 0
    assert loop.robustness.p_success >= 0.95


@pytest.mark.skipif(not HAS_NEWTON or not os.environ.get("PINNEAPPLE_SLOW"),
                    reason="Newton check of the loop takes several minutes (set PINNEAPPLE_SLOW=1)")
def test_loop_demo_newton_verification(tmp_path):
    pkg = I.design("I want a toy car that does a loop", out_dir=str(tmp_path), **FAST)
    assert pkg.verification.ran
    assert pkg.verification.result.outputs["completes"] == 1.0
    row = pkg.verification.comparison[0]
    assert row["kind"] == "at_least" and row["full"] >= row["reduced"]     # inside the worst-case envelope


def _check_package_files(folder, pkg):
    f = pkg.files
    for k in ("json", "html", "stl", "step", "glb", "check_toml"):
        assert os.path.getsize(f[k]) > 0, k
    d = json.load(open(f["json"]))
    assert d["verdict"] == pkg.verdict and d["requirements"]["template"] == pkg.plan.template
    assert all(n["fidelity"] for n in d["numbers"])
    assert open(f["step"]).read().startswith("ISO-10303-21")
    from pinneapple_check import run
    rep = run([os.path.dirname(f["check_toml"])])
    assert rep.exit_code("error") == 0, [o.message for o in rep.violations]


def test_same_seed_same_design():
    a = I.design("a wing for a 1.2 kg model glider", verify=False, seed=3, **FAST)
    b = I.design("a wing for a 1.2 kg model glider", verify=False, seed=3, **FAST)
    assert a.design == b.design and a.robustness.p_success == b.robustness.p_success
    c = I.design("I want a toy car that does a loop", verify=False, seed=1, **FAST)
    d = I.design("I want a toy car that does a loop", verify=False, seed=1, **FAST)
    assert c.design == d.design


def test_a_new_template_plugs_in():
    """The template-authoring guide's example: a register call is all it takes."""
    from pinneapple_design.intent.templates.example_spring import CoilSpringTemplate
    I.register(CoilSpringTemplate())
    try:
        assert I.select_template("a compression spring for 50 N")[0][0] == "coil_spring"
        pkg = I.design("a steel compression spring that takes 50 N with 10 mm travel", verify=False, **FAST)
        assert isinstance(pkg, I.DesignPackage) and pkg.candidate.outputs["margin_shear"] >= 0
    finally:
        from pinneapple_design.intent import parse
        parse._REGISTRY.pop("coil_spring", None)

"""The 3-D aircraft aerodynamics surrogate study (pinneapple_design.aero.aircraft_surrogate) and the lattice load
helpers it uses (pinneapple_design.aero.vlm.loads_from_circulation)."""
import math
import shutil

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("scipy")


def test_loads_from_circulation_matches_solve():
    from pinneapple_design.aero.aircraft3d import airframe_from, baseline_x
    from pinneapple_design.aero.vlm import loads_from_circulation, solve

    af = airframe_from(baseline_x())
    s = solve(af, af.mac_le_x + 0.25 * af.mac)
    for a, it in ((0.0, 0.0), (math.radians(5.0), 0.0), (math.radians(3.0), math.radians(-2.0))):
        ref = s.coefficients(a, it)
        got = loads_from_circulation(s.L, s.G.T @ np.array([1.0, a, it]), s.s_ref, s.c_ref, s.x_ref)
        for k in ("CL", "Cm", "CDi"):
            assert got[k] == pytest.approx(ref[k], rel=1e-10, abs=1e-12)


def test_panel_dcp_lift_is_kutta_joukowski():
    """CL from the panel pressure jumps (sum dCp * area / S) equals the lattice's Kutta-Joukowski lift."""
    from pinneapple_design.aero.aircraft_surrogate import NC, _case

    c = _case({"wing_area": 16.0, "aspect_ratio": 8.0, "taper": 0.7, "sweep_le": 4.0, "twist": -2.0})
    dcp = c.dcp(5.0)
    area = (c.L.chord[c.L.strip] / NC) * (c.L.P2 - c.L.P1)[:, 1]
    assert float(dcp @ area / c.s_ref) == pytest.approx(c.loads(dcp)["CL"], rel=1e-10)


def test_aircraft_surrogate_study_literal(tmp_path):
    from pinneapple_design.aero.aircraft_surrogate import aircraft_surrogate_study

    r = aircraft_surrogate_study(out_dir=str(tmp_path), n_train=8, n_test=2, n_alpha=3, epochs=60, frames=4, fps=4,
                                 width_px=480, seed=0)
    o = r["outputs"]
    assert o["panels"] == 200 and o["n_train_designs"] == 8 and o["n_test_designs"] == 2
    assert np.isfinite(o["field_error_held_out"]) and o["surrogate_ms_per_inference"] > 0
    codes = [c["code"] for c in r["checks"]]
    for code in ("grade.reference", "grade.applicability", "grade.robustness", "aero.lift_from_field",
                 "aero.linearity_alpha", "aero.symmetry", "aero.munk_bound"):
        assert code in codes
    assert all(c["status"] in ("pass", "fail", "inconclusive") for c in r["checks"])
    assert r["decision"]["verdict"] in ("APPROVED", "REVIEW", "REJECT")
    assert r["outputs"]["ood_probe_flagged"] is True         # aspect ratio 12.5 is outside the declared 6..10
    for f in ("aircraft_surrogate_poster.png", "parity.png", "errors_vs_training.png", "veriphysics_grade.png"):
        assert f in r["files"] and (tmp_path / f).stat().st_size > 0
    if shutil.which("ffmpeg"):
        assert "aircraft_surrogate.mp4" in r["files"]


def test_field_ensemble_flags_outside_the_design_space():
    from pinneapple_design.aero.aircraft_surrogate import BOUNDS, DESIGN, FieldEnsemble

    m = FieldEnsemble(4)
    lo = np.array([BOUNDS[k][0] for k in DESIGN])
    hi = np.array([BOUNDS[k][1] for k in DESIGN])
    m.Xtrain = lo + np.random.default_rng(0).random((40, len(DESIGN))) * (hi - lo)
    inside = 0.5 * (lo + hi)
    outside = inside.copy()
    outside[DESIGN.index("aspect_ratio")] = 14.0
    assert m.ood(inside)["ood"] is False
    flag = m.ood(outside)
    assert flag["ood"] is True and any("aspect_ratio" in r for r in flag["reasons"])


def test_parse_request_reads_portuguese_and_english():
    from pinneapple_design.aero.aircraft_surrogate_demo import parse_request

    pt = parse_request("Quero um surrogate de uma aeronave leve: área de asa 16 m², alongamento 12, afilamento 0,6, "
                       "enflechamento de 15°, torção de -2° e ângulo de ataque de 4 graus.")
    assert pt["lang"] == "pt" and pt["defaulted"] == [] and pt["clipped"] == []
    assert pt["design"] == {"wing_area": 16.0, "aspect_ratio": 12.0, "taper": 0.6, "sweep_le": 15.0, "twist": -2.0,
                            "alpha": 4.0}
    en = parse_request("A light aircraft with aspect ratio of 30 and washout 3 deg")
    assert en["lang"] == "en" and en["design"]["twist"] == -3.0
    assert en["clipped"] == ["aspect_ratio"] and en["design"]["aspect_ratio"] == 16.0
    assert set(en["defaulted"]) == {"wing_area", "taper", "sweep_le", "alpha"}


def test_extend_bounds_and_request_gate():
    from pinneapple_design.aero.aircraft_surrogate import BOUNDS
    from pinneapple_design.aero.aircraft_surrogate_demo import extend_bounds, request_verdict

    d = {"wing_area": 16.0, "aspect_ratio": 12.0, "taper": 0.6, "sweep_le": 15.0, "twist": -2.0, "alpha": 4.0}
    B = extend_bounds(BOUNDS, d)
    assert B["aspect_ratio"] == (6.0, pytest.approx(12.6)) and B["sweep_le"] == (0.0, pytest.approx(16.5))
    assert B["taper"] == BOUNDS["taper"]
    assert request_verdict("APPROVED", flagged=True) == "REJECT"
    assert request_verdict("REJECT", flagged=False) == "REJECT"
    assert request_verdict("APPROVED", flagged=False) == "APPROVED"


def test_aircraft_surrogate_demo_literal(tmp_path):
    from pinneapple_design.aero.aircraft_surrogate_demo import aircraft_surrogate_demo

    r = aircraft_surrogate_demo("aeronave leve, asa com alongamento 12 e enflechamento de 15°, quero treinar",
                                out_dir=str(tmp_path), n_train=(6, 8, 8), epochs=(20, 60, 60), n_test=2, n_alpha=3,
                                frames=3, width_px=480, map_n=3, cinematic="off", seed=0)
    assert [x["round"] for x in r["rounds"]] == [1, 2, 3]
    assert r["rounds"][1]["request_flagged"] is True                 # AR 12 is outside the library's 6..10
    assert r["rounds"][2]["request_flagged"] is False                # round 3 covers it
    codes = [c["code"] for c in r["checks"]]
    for code in ("demo.grade_discriminates", "demo.request_gate", "demo.extension_accepted", "grade.reference"):
        assert code in codes
    assert np.array(r["map"]["models"]["round 3"]["accepted"]).shape == (3, 3)
    assert (tmp_path / "acceptance_map.png").stat().st_size > 0
    if shutil.which("ffmpeg"):
        assert "aircraft_surrogate_demo.mp4" in r["files"]


def test_aircraft_surrogate_flow_capability():
    import pinneapple_flow.nodes_ext  # noqa: F401
    from pinneapple_flow.planner.extensions import match

    best = match("Quero um surrogate da aerodinâmica 3D de uma aeronave, alongamento 12, verificado pelo VeriPhysics")
    assert best and best[0][1].id == "aero.aircraft_surrogate"
    assert all(c.id != "aero.aircraft_surrogate" for _, c, _ in match("optimise the wing planform for induced drag"))


def test_the_helpers_on_literal_inputs():
    """Literal calls (recorded as the Studio's inputs by ``python -m pinneapple_studio inputs``)."""
    from pinneapple_design.aero.aircraft_surrogate import dataset
    from pinneapple_design.aero.aircraft_surrogate_demo import (
        extend_bounds,
        parse_request,
        request_verdict,
    )

    cases, X, Y, idx = dataset(3, 2, 0)
    assert len(cases) == 3 and X.shape == (6, 6) and Y.shape == (6, 200) and idx.tolist() == [0, 0, 1, 1, 2, 2]
    assert parse_request("aspect ratio 11, sweep 12 deg")["design"]["aspect_ratio"] == 11.0
    b = extend_bounds({"wing_area": (12.0, 20.0), "aspect_ratio": (6.0, 10.0), "taper": (0.45, 1.0),
                       "sweep_le": (0.0, 10.0), "twist": (-5.0, 0.0), "alpha": (-2.0, 8.0)},
                      {"wing_area": 16.0, "aspect_ratio": 12.0, "taper": 0.6, "sweep_le": 15.0, "twist": -2.0,
                       "alpha": 4.0})
    assert b["aspect_ratio"][1] > 12.0 and b["sweep_le"][1] > 15.0
    assert request_verdict("APPROVED", True) == "REJECT"

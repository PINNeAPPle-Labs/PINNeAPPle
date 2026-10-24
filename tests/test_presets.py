"""Industrial presets: data sheets, runs with their checks, references, surrogates, inversion and the Hub index.

Fast by default (each preset's default run and a few reference rows); PINNEAPPLE_SLOW=1 runs every preset's full
validation, optimisation, inversion and the report writer."""
from __future__ import annotations

import json
import math
import os
import warnings

import numpy as np
import pytest

import pinneapple_presets as P
from pinneapple_presets.surrogate import ARDGP, load_surrogate

SLOW = pytest.mark.skipif(not os.environ.get("PINNEAPPLE_SLOW"), reason="slow: set PINNEAPPLE_SLOW=1")
NAMES = P.names()


def test_catalogue_and_flow_node_list_agree():
    from pinneapple_flow.nodes.presets import PRESETS
    assert sorted(PRESETS) == NAMES and len(NAMES) >= 5
    import pinneapple as pp
    assert pp.presets.load is P.load


@pytest.mark.parametrize("name", NAMES)
def test_data_sheet_has_units_sources_and_capabilities(name):
    c = P.load(name).card()
    assert c["question"] and c["validity"] and c["pipeline"]
    for q in c["inputs"]:
        assert q["unit"] and q["source"], q["name"]
        if q.get("choices") is None and isinstance(q["default"], (int, float)):
            assert q["lo"] <= q["default"] <= q["hi"], q["name"]
    for k, v in c["capabilities"].items():
        if not v and k != "reason":
            assert c["missing"].get(k), f"{name}: say why there is no {k}"
    json.dumps(c, default=str)


@pytest.mark.parametrize("name", NAMES)
def test_default_run_is_trusted_and_has_provenance(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = P.load(name).run()
    assert r.trust.decisao == "APPROVED", [(c.name, c.status) for c in r.trust.checks if c.status != "supports"]
    assert r.decision["verdict"] in ("OK", "NOT OK") and r.decision["reasons"]
    assert r.meta["versions"]["numpy"] and r.meta["hardware"]["cores"] and r.meta["wall_time_s"] > 0
    json.dumps(r.to_dict())


def test_outside_the_domain_is_flagged_not_hidden():
    with pytest.warns(UserWarning, match="outside the validity domain"):
        r = P.load("water_hammer_valve_closure").run(velocity=6.0)
    assert any(c.name == "inside the validity domain" and c.status == "contradicts" for c in r.trust.checks)
    with pytest.raises(KeyError, match="unknown parameters"):
        P.load("water_hammer_valve_closure").run(speed=1.0)


def test_water_hammer_references():
    v = P.load("water_hammer_valve_closure").validate()
    assert v["all_pass"], v["references"]
    jk = v["references"][0]
    assert jk["rel_error"] < 1e-6 and "Joukowsky" in jk["source"]


def test_water_network_matches_epanet_and_finds_the_published_optimum():
    p = P.load("water_distribution_network")
    r = p.run()
    assert r.outputs["cost"] == 419000.0 and r.outputs["min_pressure"] >= 30.0
    for ref in p.references():
        out = p.run(**ref.params).outputs[ref.output]
        assert abs(out - ref.value) <= ref.tol * abs(ref.value), ref.name
    opt = p.optimize(generations=300)
    assert opt["cost_ga"] == 419000.0 and opt["feasible"]


def test_pcb_jedec_and_heatsink_independent_correlation():
    v = P.load("power_pcb_junction").validate()
    assert all(r["pass"] for r in v["references"])
    v = P.load("plate_fin_heatsink_forced").validate()
    assert v["all_pass"], [(r["case"], r["rel_error"]) for r in v["references"]]


def test_footbridge_fe_and_closed_form_agree_and_fails_ltb_as_designed():
    p = P.load("steel_footbridge_pedestrian")
    r = p.run()
    assert abs(r.outputs["f1_fe"] - r.outputs["f1"]) / r.outputs["f1"] < 1e-3
    assert r.outputs["util_ltb"] > 1 and r.decision["verdict"] == "NOT OK"      # the naive girder buckles sideways
    assert p.run(lateral_braces=5, girder_h=1.2, girder_b=0.35, girder_tf=0.015, girder_tw=0.012).decision[
        "verdict"] == "OK"


def test_beam_fe_converges_to_euler_bernoulli():
    from pinneapple_simulation.numerical_solvers.beam_fe import beam_modes, beam_static
    L, EI, m, q = 20.0, 3e8, 400.0, 2e3
    f = beam_modes(L, EI, m, 32, 3)["f"]
    for n in (1, 2, 3):
        assert f[n - 1] == pytest.approx(n ** 2 * math.pi / (2 * L ** 2) * math.sqrt(EI / m), rel=1e-5)
    s = beam_static(L, EI, q, 8)
    assert s["w_max"] == pytest.approx(5 * q * L ** 4 / (384 * EI), rel=1e-10)
    c = beam_modes(L, EI, m, 32, 1, ends=("fixed", "free"))["f"][0]
    assert c == pytest.approx(1.875104 ** 2 / (2 * math.pi * L ** 2) * math.sqrt(EI / m), rel=1e-5)


def test_pipe_network_conserves_mass_and_energy():
    from pinneapple_simulation.numerical_solvers.pipe_network import solve_network, two_loop_network
    for hl in ("HW", "DW"):
        r = solve_network(two_loop_network(headloss=hl))
        assert r["continuity_residual"] < 1e-12 and r["energy_residual"] < 1e-6
        assert r["supply"] == pytest.approx(1120 / 3600, rel=1e-12)


def test_ard_gp_interpolates_and_round_trips():
    rng = np.random.default_rng(0)
    X = rng.random((40, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2
    g = ARDGP().fit(X, y)
    Xt = rng.random((20, 2))
    m, s = g.predict(Xt)
    assert np.abs(m - (np.sin(3 * Xt[:, 0]) + Xt[:, 1] ** 2)).max() < 0.02 and np.all(s >= 0)
    g2 = ARDGP.from_dict(json.loads(json.dumps(g.to_dict())))
    assert np.allclose(g2.predict(Xt)[0], m)


@pytest.mark.parametrize("name", [n for n in NAMES if P.load(n).surrogate_spec is not None])
def test_stored_surrogate_reports_its_holdout_error_and_flags_ood(name):
    s = load_surrogate(name)
    assert s is not None, f"train it: python -m pinneapple_presets surrogate {name}"
    h = s.meta["holdout"]
    assert s.meta["n_test"] >= 15 and set(h["per_output"]) == set(s.meta["outputs"])
    p = P.load(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.run(fidelity="surrogate")
        k, (a, b) = next(iter(s.bounds.items()))
        x = next(v for v in (b + 0.1 * (b - a), a - 0.1 * (b - a)) if p.param(k).inside(v))
        bad = p.run(fidelity="surrogate", **{k: x})
    for o in s.meta["outputs"]:
        assert math.isfinite(r.outputs[o]) and r.outputs[o + "_std"] >= 0
    assert r.trust.checks[0].status == "supports"
    assert next(c for c in bad.trust.checks if c.name == "surrogate inside its training box").status == "contradicts"


def test_inversions_recover_the_synthetic_truth():
    inv = P.load("water_hammer_valve_closure").invert()
    assert inv["rel_error"] < 0.01 and inv["true_inside_ci95"]
    inv = P.load("water_distribution_network").invert()
    assert inv["rel_error"] < 0.01 and inv["true_inside_ci95"]
    inv = P.load("plate_fin_heatsink_forced").invert()
    assert inv["rel_error"] < 0.02 and inv["true_inside_ci95"]


def test_index_lists_every_preset_with_its_sheet():
    idx = P.index()
    assert [p["name"] for p in idx["presets"]] == NAMES
    assert all("results" in p and "inputs" in p for p in idx["presets"])


def test_cli_run_and_validate(capsys):
    from pinneapple_presets.__main__ import main
    assert main(["run", "water_hammer_valve_closure", "t_close=4.0"]) == 0
    assert "water_hammer_valve_closure" in capsys.readouterr().out
    assert main(["validate", "water_hammer_valve_closure"]) == 0


def test_flow_node_runs_a_preset(tmp_path, monkeypatch):
    import pinneapple_flow as pf
    monkeypatch.setenv("PINNEAPPLE_FLOW_HOME", str(tmp_path / "home"))
    f = pf.Flow("preset", nodes=[{"id": "wh", "type": "preset.run", "params": {
        "preset": "water_hammer_valve_closure", "overrides": {"t_close": 2.0}}}], edges=[])
    r = pf.run(f, workdir=tmp_path / "w")
    assert r.status == "done", {k: v["error"] for k, v in r.nodes.items()}
    assert r.report("wh")["metrics"]["p_max"]["value"] > 1.6e6          # 2 s closure exceeds PN16


@SLOW
@pytest.mark.parametrize("name", NAMES)
def test_full_validation_optimisation_and_report(name, tmp_path):
    from pinneapple_presets.report import write_report
    s = write_report(name, out_dir=str(tmp_path / name), surrogate="load", lab=False)
    assert s["validation"]["all_pass"], s["validation"]["references"]
    assert os.path.isfile(tmp_path / name / "README.md")
    if "optimization" in s:
        assert s["optimization"].get("feasible", True)


def test_solid_fem_patch_test_on_distorted_elements():
    """A linear displacement field gives the exact uniform strain on sheared and rotated hexahedra (the Jacobian
    mapping of the shape-function gradients)."""
    from pinneapple_simulation.numerical_solvers.solid_fem import HexMesh, SolidFEM, box_mesh
    m = box_mesh(1, 1, 0.2, 4, 4, 1)
    c, s = math.cos(0.5), math.sin(0.5)
    for T in (np.array([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]]), np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])):
        mm = HexMesh(m.nodes @ T.T, m.elements, m.shape, m.size)
        for inc in (True, False):
            f = SolidFEM(mm, 1.0, 0.0, incompatible=inc)
            for n in range(len(mm.nodes)):
                for d in range(3):
                    f._fixed[3 * n + d] = 1e-3 * mm.nodes[n, d]
            r = f.solve()
            assert np.allclose(r.stress[:, :3], 1e-3, rtol=1e-9) and np.allclose(r.stress[:, 3:], 0, atol=1e-12)


def test_iapws_and_standard_atmosphere_reference_values():
    from pinneapple_physics.closed_form.pump_npsh import atmosphere, iapws97_psat
    for T, ref in ((300.0, 0.353658941e-2), (500.0, 0.263889776e1), (600.0, 0.123443146e2)):   # IF97 Table 35
        assert iapws97_psat(T - 273.15) / 1e6 == pytest.approx(ref, rel=1e-8)
    assert atmosphere(1000.0) == pytest.approx(89874.6, rel=1e-5)


def test_one_two_exchanger_closed_form_equals_direct_integration():
    from pinneapple_physics.closed_form.heat_exchanger import effectiveness, one_two_exchanger_ode
    for ntu, cr in ((0.5, 0.3), (2.0, 0.8), (3.0, 1.0)):
        assert effectiveness(ntu, cr, "shell_tube") == pytest.approx(one_two_exchanger_ode(ntu, cr), rel=1e-8)


def test_insulation_kirchhoff_equals_finite_volumes():
    from pinneapple_physics.closed_form.pipe_insulation import heat_loss, radial_fd
    h = heat_loss(0.1683, 0.06, 180.0, 10.0, k0=0.036, k1=1e-4, eps=0.05, wind=3.0)
    fd = radial_fd(0.1683, 0.06, 180.0, h["t_surface"], k0=0.036, k1=1e-4)
    assert fd["q_inner"] == pytest.approx(h["q"], rel=1e-5) and fd["q_outer"] == pytest.approx(fd["q_inner"], rel=1e-9)


def test_pressure_vessel_fem_matches_lame_and_asme():
    p = P.load("pressure_vessel_asme")
    r = p.run()
    o = r.outputs
    assert o["hoop_fem"] == pytest.approx(o["hoop_lame"], rel=0.02)
    assert o["ur_fem"] == pytest.approx(o["ur_lame"], rel=0.005)
    assert o["mawp"] >= 1.0e6 and o["t_shell"] == 0.010 and r.decision["verdict"] == "OK"


def test_batch2_inversions_and_decisions():
    for name in ("shell_tube_cooler", "steam_pipe_insulation", "pressure_vessel_asme", "pump_npsh_cavitation"):
        inv = P.load(name).invert()
        assert inv["true_inside_ci95"], name
        assert inv["rel_error"] < 0.05, (name, inv["rel_error"])
    opt = P.load("pump_npsh_cavitation").optimize()
    assert opt["verified_ok"] and opt["smallest_ok_suction"] is None          # lowering the pump is the fix


def test_panel_method_converges_with_a_closed_trailing_edge():
    from pinneapple_design.aero.geometry import naca4
    from pinneapple_design.aero.panel import panel_solve
    s = naca4("2412")
    cl = [panel_solve(s, 4.0, n_per_side=n, close_te=True).cl_kutta_joukowski for n in (100, 200, 400)]
    assert abs(cl[2] - cl[1]) < abs(cl[1] - cl[0]) and abs(cl[2] - cl[1]) / cl[2] < 0.002


def test_eddy_fdm_block_solve_equals_complex_solve_and_dodd_deeds_limits():
    from scipy.sparse.linalg import spsolve

    from pinneapple_physics.closed_form.eddy_coil import MU0, coil_impedance
    from pinneapple_simulation.numerical_solvers.eddy_current_fdm import (
        annular_current_source, build_axisymmetric_helmholtz_system, solve_axisymmetric_eddy_current)
    r = np.linspace(0, 0.02, 41)
    z = np.linspace(-0.02, 0.02, 81)
    RR, ZZ = np.meshgrid(r, z, indexing="ij")
    J = annular_current_source(r, z, 0.003, 0.002, 0.002, 0.001, 100, 1.0)
    sg = np.where(ZZ < 0, 1.7e7, 0.0)
    A = solve_axisymmetric_eddy_current(r, z, 2 * np.pi * 1e4, MU0 * np.ones_like(RR), sg, J)
    M, b = build_axisymmetric_helmholtz_system(r, z, 1j * 2 * np.pi * 1e4 * MU0 * sg, -MU0 * J + 0j)
    ref = spsolve(M.tocsc(), b)
    # both solves leave round-off residuals of the same size (cond(M) ~ 1e8 with 1/h^2 ~ 4e6): compare residuals
    assert np.abs(M @ A.ravel() - b).max() <= 2 * np.abs(M @ ref - b).max() + 1e-30
    assert np.abs(A.ravel() - ref).max() <= 1e-7 * np.abs(ref).max()
    kw = dict(r1=2e-3, r2=4e-3, l1=5e-4, l2=2.5e-3, turns=200, sigma=1.74e7)
    thick = coil_impedance(1e4, thickness=0.02, **kw)
    half = coil_impedance(1e4, **kw)
    assert abs(thick["dZ"] - half["dZ"]) < 1e-6 * abs(half["dZ"])
    assert half["dR"] > 0 and half["dX"] < 0


def test_batch3_default_runs_and_inversions():
    import warnings as _w
    for name in ("airfoil_section_re4e6", "eddy_current_skin_inspection"):
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            p = P.load(name)
            inv = p.invert()
        assert inv["true_inside_ci95"] and inv["rel_error"] < 0.02, (name, inv["rel_error"])
    opt = P.load("eddy_current_skin_inspection").optimize()
    assert 0.6 < opt["t_over_delta_best"] < 1.6 and opt["verified_ok"]       # the classic t ~ delta rule


@SLOW
def test_ipm_matches_the_published_prius_torque_curve():
    v = P.load("ipm_traction_motor").validate()
    assert v["all_pass"], v["references"]

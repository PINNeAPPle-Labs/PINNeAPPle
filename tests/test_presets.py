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

"""Compact-range design (#965): both polarisations and cuts, the grid, the checks, the surrogate with UQ, the Pareto
front, the optimiser and the private entry point of the example."""
import importlib.util
import os
from dataclasses import replace

import numpy as np
import pytest

from pinneapple_simulation.electromagnetics.antenna.runner import ResultTable
from pinneapple_simulation.electromagnetics.reflector import (
    CompactRange,
    OffsetReflector,
    evaluate,
    po,
)
from pinneapple_simulation.electromagnetics.reflector import design as dz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def small(**kw) -> CompactRange:
    """A 1 GHz range of a few wavelengths: seconds per PO solve."""
    return CompactRange(freq_hz=1e9, reflector=OffsetReflector(focal=1.5, half_x=0.6, half_y=0.6, centre_y=0.9,
                                                               n=6, roll=0.15), feed_gain_dbi=12, qz_z=3.0,
                        qz_diameter=0.4, mesh_per_wavelength=6.0, qz_step_per_wavelength=3.0, eval_factor=1.6, **kw)


def test_radiate_fields_matches_radiate_and_its_h_is_the_far_field_of_e():
    k = po.wavenumber(1e9)
    rng = np.random.default_rng(0)
    C = rng.uniform(-0.2, 0.2, (40, 3))
    A = rng.uniform(1e-4, 2e-4, 40)
    Js = rng.normal(size=(2, 40, 3)) + 1j * rng.normal(size=(2, 40, 3))
    obs = np.array([[0.1, 0.2, 2.0], [0.0, 0.0, 300.0]])
    E, H = po.radiate_fields(C, A, Js, obs, k, magnetic=True)
    for p in range(2):
        assert np.allclose(E[p], po.radiate(C, A, Js[p], obs, k), rtol=1e-12, atol=1e-15)
    rh = obs[1] / np.linalg.norm(obs[1])
    assert np.allclose(H[0, 1], np.cross(rh, E[0, 1]) / po.ETA0, rtol=2e-3, atol=1e-3 * np.abs(H[0, 1]).max())   # H_r ~ 1/r^2


def test_qz_report_reproduces_evaluate_and_reports_both_polarisations_and_cuts():
    cr = small()
    rep = dz.qz_report(cr)
    old = evaluate(cr)
    for k in dz.DISK:
        assert rep[f"V_{k}"] == pytest.approx(old[k], rel=1e-12, abs=1e-12)      # same PO path as #887
    for p in ("V", "H"):
        for c in ("h", "v"):
            assert np.isfinite(rep[f"{p}_{c}cut_ripple_pp_dB"]) and rep[f"{p}_{c}cut_taper_dB"] >= 0
    assert rep["usable_diameter"] == min(rep["V_usable_diameter"], rep["H_usable_diameter"])
    assert rep["xpol_dB"] == max(rep["V_xpol_dB"], rep["H_xpol_dB"])
    assert rep["feasible"] == (rep["usable_diameter"] >= cr.qz_diameter)
    assert not dz.qz_report(cr, dz.Spec(0.01, xpol_max_db=-200.0))["feasible"]   # an impossible cross-pol limit


def test_cut_metrics_on_a_known_field():
    s = np.linspace(-0.5, 0.5, 101)
    E = 10 ** (-1.5 * (s / 0.5) ** 2 / 20) * np.exp(1j * np.deg2rad(3.0) * np.cos(2 * np.pi * s / 0.2))
    m = dz._cut_metrics(s, E, 1.0)
    assert m["taper_dB"] == pytest.approx(1.5, abs=0.02) and m["ripple_pp_dB"] < 1e-6
    assert 5.0 < m["phase_ripple_pp_deg"] < 6.5                                        # +-3 deg about the fit


def test_design_grid_and_parallel_doe():
    base = small()
    g = dz.design_grid(base, [10, 14], [1.0], focals=[1.5, 2.0])
    assert len(g) == 4
    for inp, cr in g:
        assert cr.qz_z - cr.reflector.focal == pytest.approx(base.qz_z - base.reflector.focal)
        assert 2 * cr.reflector.half_x == inp["side"] == 1.0
    one = dz.doe(base, dz.Spec(0.4), [10, 14], [1.0], workers=1)
    two = dz.doe(base, dz.Spec(0.4), [10, 14], [1.0], workers=2)
    assert one.tags == two.tags and np.allclose(one.col("ripple_pp_dB"), two.col("ripple_pp_dB"))
    assert one.param_names() == ["feed_gain_dBi", "side", "focal"]


def test_doe_resumes_from_its_checkpoint(tmp_path, monkeypatch):
    ck = str(tmp_path / "sweep.jsonl")
    base = small()
    first = dz.doe(base, dz.Spec(0.4), [10], [1.0], checkpoint=ck)
    with open(ck, "a") as fh:
        fh.write('{"tag": "cut by a kill"')                         # a half-written line is ignored
    calls = []
    real = dz._solve
    monkeypatch.setattr(dz, "_solve", lambda job: calls.append(job) or real(job))
    both = dz.doe(base, dz.Spec(0.4), [10, 14], [1.0], checkpoint=ck)
    assert len(calls) == 1 and both.tags == [first.tags[0], "g14_s1_f1_F1.5"]
    assert both.rows[0]["ripple_pp_dB"] == pytest.approx(first.rows[0]["ripple_pp_dB"])
    assert set(dz._load_jsonl(ck)) == set(both.tags)                   # the new row did not land on the cut line


def test_power_balance_and_facet_convergence():
    pb = dz.power_balance(small())
    assert 0.3 < pb["intercepted_W"] < 1.0                     # the feed radiates 1 W; part spills past the edge
    r = [b["ratio"] for b in pb["by_diameter"]]
    assert np.all(np.diff(r) > 0) and 0.8 < r[-1] < 1.05        # tends to 1 as the disk catches the diffraction
    fc = dz.facet_convergence(small(), densities=(5, 7))
    assert len(fc["runs"]) == 2 and fc["diff_to_finest"][-1]["ripple_pp_dB"] == 0.0


def synthetic(gains, sides, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in gains:
        for s in sides:
            u = 0.3 * s - 0.004 * (g - 16) ** 2 + noise * rng.normal()
            rows.append({"tag": f"g{g}s{s}", "feed_gain_dBi": g, "side": s, "focal": 4.0, "f_GHz": 2.0,
                         "usable_diameter": u, "ripple_pp_dB": 2.0 / s + 0.02 * g, "taper_dB": 0.1 * g,
                         "phase_ripple_pp_deg": 10 / s, "xpol_dB": -25 - s, "feasible": u >= 1.0})
    return ResultTable(rows)


def test_surrogate_uq_pareto_and_figures(tmp_path):
    t = synthetic([10, 13, 16, 19, 22], [2.4, 2.8, 3.2, 3.6, 4.0, 4.4], noise=0.01)
    sur = dz.fit_qz_surrogate(t)
    assert sur["inputs"] == ["feed_gain_dBi", "side"]
    u = sur["metrics"]["usable_diameter"]
    assert u["rmse"] < 0.05 and u["n"] == len(t) and 0.6 <= u["coverage95"] <= 1.0
    mu, sd = dz.predict(sur, "usable_diameter", [{"feed_gain_dBi": 16, "side": 3.0}])
    assert mu[0] == pytest.approx(0.9, abs=0.05) and sd[0] > 0
    front = dz.pareto_front(synthetic([10, 13, 16, 19, 22], [2.4, 2.8, 3.2, 3.6, 4.0, 4.4]), dz.Spec(1.0))
    assert [f["feed_gain_dBi"] for f in front] == [16] * len(front)            # the best gain at every size
    assert np.all(np.diff([f["usable_diameter"] for f in front]) > 0)
    assert [f["meets_spec"] for f in front] == [f["usable_diameter"] >= 1.0 for f in front]
    spec = dz.Spec(1.0)
    for name, fn in (("maps.png", lambda p: dz.design_maps(t, sur, spec, p)),
                     ("pareto.png", lambda p: dz.pareto_figure(t, sur, spec, p, chosen=front[-1]))):
        fn(str(tmp_path / name))
        assert (tmp_path / name).stat().st_size > 20_000


def test_smallest_reflector_finds_and_verifies_the_threshold(monkeypatch):
    """The optimiser's logic on a known answer: usable = 0.3 side - 0.004 (g - 16)^2 meets 1 m from side 3.33."""
    def fake_report(cr, spec=None, **kw):
        g, s = cr.feed_gain_dbi, 2 * cr.reflector.half_x
        u = 0.3 * s - 0.004 * (g - 16) ** 2
        return {"usable_diameter": u, "feasible": u >= spec.qz_diameter, "xpol_dB": -30.0,
                **{k: 0.0 for k in dz.DISK if k not in ("usable_diameter", "xpol_dB")}}
    monkeypatch.setattr(dz, "qz_report", fake_report)
    t = synthetic([10, 13, 16, 19, 22], [2.4, 3.0, 3.6, 4.2, 4.8])
    base = replace(small(), freq_hz=2e9, reflector=replace(small().reflector, focal=4.0))
    r = dz.smallest_reflector(base, dz.Spec(1.0), t, gain_bounds=(10, 22), side_bounds=(2.4, 4.8), max_iter=4)
    assert r["found"] and r["verified_by"] == "PO" and "PTD" in r["limits"]
    assert 1.0 / 0.3 <= r["design"]["side"] < 1.0 / 0.3 + 0.35
    assert abs(r["design"]["feed_gain_dBi"] - 16) < 3
    none = dz.smallest_reflector(base, dz.Spec(5.0), t, gain_bounds=(10, 22), side_bounds=(2.4, 4.8), max_iter=2)
    assert not none["found"] and len(none["history"]) == 2 and none["smallest_po_feasible_in_data"] is None


def test_config_and_private_entry_point_stay_outside_the_repo(tmp_path):
    spec_ = importlib.util.spec_from_file_location(
        "cr2", os.path.join(ROOT, "examples", "use_cases", "antennas", "compact_range_2ghz.py"))
    ex = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(ex)
    c = dz.from_config(ex.GENERIC)
    assert c["base"].freq_hz == 2e9 and c["spec"].xpol_max_db == -25.0 and c["grid"]["focals"] == [3.0, 4.0, 5.0]
    with pytest.raises(ValueError):
        dz.from_config({**ex.GENERIC, "client_secret": 1})
    assert dz.outside(str(tmp_path), ROOT) and not dz.outside(os.path.join(ROOT, "examples"), ROOT)
    inside = os.path.join(ROOT, "examples", "use_cases", "antennas", "results", "x.json")
    with pytest.raises(ValueError, match="inside the repository"):
        ex.run_private(inside)
    cfg = tmp_path / "range.json"
    cfg.write_text("{}")
    with pytest.raises(ValueError, match="inside the repository"):
        ex.run_private(str(cfg), os.path.join(ROOT, "examples", "private_out"))
    assert not os.path.exists(os.path.join(ROOT, "examples", "private_out"))

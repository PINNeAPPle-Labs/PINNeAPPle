"""End-fire antenna toolkit: limits, checks, AVA layout, runner (fake simulator), surrogates, design loop.

The EMerge test needs ``pip install emerge`` and ``PINNEAPPLE_SLOW=1`` (about a minute after Numba's first-run
compilation, which takes a few minutes once)."""
import json
import math
import os
import sys
import textwrap

import numpy as np
import pytest

from pinneapple_simulation.electromagnetics.antenna import bounds, checks
from pinneapple_simulation.electromagnetics.antenna import design as dz
from pinneapple_simulation.electromagnetics.antenna import surrogates as sg
from pinneapple_simulation.electromagnetics.antenna.ava import (
    AVAParams,
    InvalidGeometry,
    layout,
    microstrip_width,
)
from pinneapple_simulation.electromagnetics.antenna.package import load_package
from pinneapple_simulation.electromagnetics.antenna.runner import (
    CLISimulator,
    ResultTable,
    read_table,
    write_table,
)

C0 = 299_792_458.0


# ------------------------------------------------------------------------------------------------------- bounds
def test_hw_min_length_10dbi_at_2ghz():
    # 10 dBi = 10 (linear); lambda(2 GHz) = 149.896 mm; L = 10 / 7.28 * lambda = 205.9 mm
    L = bounds.min_length(10.0, 2e9)
    assert L == pytest.approx(10.0 / 7.28 * C0 / 2e9, rel=1e-12)
    assert L * 1e3 == pytest.approx(205.9, abs=0.05)
    assert bounds.min_length(10.0, 2e9, "ordinary") * 1e3 == pytest.approx(374.7, abs=0.05)   # 10 / 4 * lambda


def test_endfire_directivity_closed_form_and_inverse():
    assert bounds.endfire_directivity(0.15, 2e9) == pytest.approx(7.28 * 0.15 * 2e9 / C0)
    assert bounds.endfire_directivity(0.15, 2e9, "ordinary") == pytest.approx(4 * 0.15 * 2e9 / C0)
    # the feasibility frontier is the inverse of the limit: at f_min the limit equals the target
    f = bounds.feasibility_frequency(10.0, 0.206)
    assert bounds.endfire_directivity_dbi(0.206, f) == pytest.approx(10.0, abs=1e-12)
    assert f / 1e9 == pytest.approx(10 * C0 / (7.28 * 0.206) / 1e9)
    with pytest.raises(ValueError):
        bounds.endfire_directivity(0.1, 1e9, "broadside")


def test_aperture_gain_closed_form():
    # 4 pi A / lambda^2 with A = lambda^2: 4 pi (10.99 dBi); efficiency scales linearly
    lam = C0 / 3e9
    assert bounds.aperture_gain(lam ** 2, 3e9) == pytest.approx(4 * math.pi)
    assert bounds.aperture_gain_dbi(lam ** 2, 3e9, 0.5) == pytest.approx(10 * math.log10(2 * math.pi))


def test_microstrip_width_50_ohm():
    # Pozar eq. 3.197; er = 3.55, h = 1.524 mm -> W/d = 2.24
    w = microstrip_width(50.0, 3.55, 1.524)
    assert w / 1.524 == pytest.approx(2.237, abs=0.005)


# ------------------------------------------------------------------------------------------------------- checks
def _row(**kw):
    s11_db = -15.0
    G, D = 7.0, 7.5
    r = {"tag": "x", "Lb": 210.0, "f_GHz": 3.0, "S11_dB": s11_db, "D_bore_dBi": D, "G_bore_dBi": G,
         "Greal_bore_dBi": G + 10 * math.log10(1 - 10 ** (s11_db / 10)), "D_peak_dBi": 8.0,
         "Greal_peak_dBi": 7.3, "rad_eff": 10 ** ((G - D) / 10), "P_rad_pattern": 0.88, "P_rad_surface": 0.9}
    r.update(kw)
    return r


def test_consistent_row_is_approved():
    rep = checks.antenna_checks(_row())
    assert rep.decisao == "APPROVED", rep.motivos
    assert {c.name for c in rep.checks} == {"realized_gain", "efficiency_is_G_over_D", "efficiency_le_1",
                                             "peak_ge_boresight", "directivity_le_hw", "s11_resolved",
                                             "power_closure"}


def test_each_check_catches_its_error():
    st = lambda rep, name: next(c for c in rep.checks if c.name == name).status  # noqa: E731
    assert st(checks.antenna_checks(_row(Greal_bore_dBi=7.0)), "realized_gain") == "contradicts"
    assert st(checks.antenna_checks(_row(rad_eff=0.5)), "efficiency_is_G_over_D") == "contradicts"
    bad = checks.antenna_checks(_row(rad_eff=1.2))
    assert st(bad, "efficiency_le_1") == "contradicts" and bad.decisao == "REJECT"     # critical
    assert st(checks.antenna_checks(_row(D_peak_dBi=7.0)), "peak_ge_boresight") == "contradicts"
    # 7.28 * 0.21 / lambda(3 GHz) = 15.3 -> 11.8 dBi; 12.5 dBi is above it + 0.5 dB
    assert st(checks.antenna_checks(_row(D_peak_dBi=12.5)), "directivity_le_hw") == "contradicts"
    assert st(checks.antenna_checks(_row(D_peak_dBi=12.2)), "directivity_le_hw") == "supports"
    assert st(checks.antenna_checks(_row(P_rad_pattern=0.7)), "power_closure") == "contradicts"


def test_interpolated_s11_with_few_points_is_not_trusted():
    assert checks.check_s11_resolved(3, interpolated=True).status == "contradicts"
    assert checks.check_s11_resolved(9, interpolated=True).status == "supports"
    assert checks.check_s11_resolved(None, interpolated=True).status == "not_run"
    assert checks.check_s11_resolved(1, interpolated=False).status == "supports"


def test_missing_inputs_are_not_run():
    rep = checks.antenna_checks({"tag": "y", "f_GHz": 2.0})
    assert all(c.status in ("not_run", "supports") for c in rep.checks)
    assert next(c for c in rep.checks if c.name == "efficiency_le_1").status == "not_run"


# ------------------------------------------------------------------------------------------------------- layout
def test_default_layout_is_buildable_and_mirrored():
    lay = layout(AVAParams(nslot=3, dir_a=12, dir_b=6, dir_x=185))
    assert lay.wf == pytest.approx(microstrip_width(50, 3.55, 1.524))
    assert lay.min_metal > 0.5
    # the bottom flare is the mirror image of the top flare
    top_flare = lay.top[lay.top[:, 1] > 50]
    bot_flare = lay.bottom[lay.bottom[:, 1] > 50]
    assert np.allclose(np.sort(top_flare[:, 0]), np.sort(-bot_flare[:, 0]))
    assert lay.director is not None and np.abs(lay.director[:, 0]).max() == pytest.approx(6.0)


@pytest.mark.parametrize("kw", [dict(Wa=130.0), dict(xs=15.0, Lo=20.0), dict(nslot=30), dict(Ls=58.0, nslot=2),
                                dict(dir_a=10, dir_b=40, dir_x=150), dict(res=0.6)])
def test_invalid_geometry_raises(kw):
    with pytest.raises(InvalidGeometry):
        layout(AVAParams(**kw))


def test_doe_skips_unbuildable_designs():
    designs, info = dz.doe(6, seed=1)
    assert len(designs) == 6 and all(dz.is_buildable(d) for d in designs)
    assert info["drawn"] == 6 + info["invalid"]
    sob, _ = dz.doe(4, method="sobol", seed=2)
    assert len(sob) == 4


# ------------------------------------------------------------------------------------------------------- runner
FAKE = textwrap.dedent('''
    import argparse, json, math, os, signal
    ap = argparse.ArgumentParser()
    ap.add_argument("--params"); ap.add_argument("--freq", type=float, nargs="+"); ap.add_argument("--out")
    ap.add_argument("--tag"); ap.add_argument("--safe", action="store_true")
    a = ap.parse_args()
    p = json.loads(a.params)
    if p.get("crash") == "sigbus" and not a.safe:
        os.kill(os.getpid(), signal.SIGBUS)
    if p.get("crash") == "hang":
        import time; time.sleep(60)
    if p["Lb"] < 0:
        json.dump({"status": "invalid", "error": "negative length", "rows": []}, open(os.path.join(a.out, "result.json"), "w"))
        raise SystemExit(3)
    rows = [{"f_GHz": f, "S11_dB": -12.0, "D_bore_dBi": 10 * math.log10(4 * p["Lb"] * f / 299.792458)}
            for f in a.freq]
    json.dump({"status": "ok", "rows": rows}, open(os.path.join(a.out, "result.json"), "w"))
''')


@pytest.fixture
def fake(tmp_path):
    script = tmp_path / "fake_sim.py"
    script.write_text(FAKE)
    return [sys.executable, str(script)]


def test_runner_one_process_per_frequency(fake, tmp_path):
    sim = CLISimulator(fake, str(tmp_path / "runs"), timeout=30, fallback_args=["--safe"])
    t = sim.run([{"Lb": 200.0}, {"Lb": -1.0}, {"Lb": 150.0, "crash": "sigbus"}], [2.0, 3.0], tags=["a", "b", "c"])
    st = [(r.tag, r.status, r.retried) for r in sim.records]
    assert st == [("a", "ok", False), ("a", "ok", False), ("b", "invalid", False), ("c", "ok", True),
                  ("c", "ok", True)]                                  # invalid: the other frequency is skipped
    assert len(t) == 4 and t.tags == ["a", "c"]
    assert t.rows[0]["Lb"] == 200.0 and t.rows[1]["f_GHz"] == 3.0
    assert all(r.max_rss_mb > 0 for r in sim.records)
    s = sim.summary()
    assert s["status"] == {"ok": 4, "invalid": 1} and s["processes"] == 5


def test_runner_timeout_and_sigbus_without_fallback(fake, tmp_path):
    sim = CLISimulator(fake, str(tmp_path / "runs"), timeout=1.0)
    assert sim.run_one({"Lb": 1.0, "crash": "hang"}, [2.0], "h").status == "timeout"
    assert sim.run_one({"Lb": 1.0, "crash": "sigbus"}, [2.0], "s").status == "sigbus"


def test_table_roundtrip_and_formats(tmp_path):
    t = ResultTable([{"tag": "a", "Lb": 200, "f_GHz": 2.0, "S11_dB": -11.5},
                     {"tag": "a", "Lb": 200, "f_GHz": 3.0, "S11_dB": -14.0}])
    p = write_table(t, str(tmp_path / "t.csv"))
    back = read_table(p)
    assert back.rows == t.rows and back.param_names() == ["Lb"]
    (tmp_path / "t.tsv").write_text("tag\tLb\tf_GHz\tS11_dB\na\t200\t2\t-11.5\n")
    (tmp_path / "t.txt").write_text("# comment\ntag Lb f_GHz S11_dB\na 200 2 -11.5\n")
    for name in ("t.tsv", "t.txt"):
        r = read_table(str(tmp_path / name)).rows[0]
        assert r["Lb"] == 200 and r["S11_dB"] == -11.5


def test_package_reader_and_rerun(fake, tmp_path):
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    (pkg_dir / "results.csv").write_text("tag,Lb,f_GHz,S11_dB,D_bore_dBi\nv1,200,2.0,-12,7.0\n")
    (pkg_dir / "sim.py").write_text(FAKE)
    pkg = load_package(str(pkg_dir))
    assert pkg.describe()["designs"] == 1 and pkg.describe()["parameters"] == ["Lb"]
    t = pkg.simulator(str(tmp_path / "rerun"), timeout=30).run([pkg.table.design("v1")], [2.0], tags=["v1"])
    assert t.rows[0]["D_bore_dBi"] == pytest.approx(10 * math.log10(4 * 200 * 2 / 299.792458))


# ------------------------------------------------------------------------------------------------------- surrogates
def _synthetic(n=8, freqs=(2.0, 3.0, 4.0), seed=0, noise=0.2):
    """Gain = HW limit - 3 dB + 0.01 dB/mm of width + noise: the HW + offset and the linear models can learn it."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        p = {"Lb": float(np.linspace(150, 260, n)[i]), "W": rng.uniform(80, 140), "Wa": rng.uniform(40, 100), "nslot": 0,
             "dir_a": 0.0}
        for f in freqs:
            hw = float(bounds.endfire_directivity_dbi(p["Lb"] * 1e-3, f * 1e9))
            rows.append({"tag": f"s{i}", **p, "f_GHz": f,
                         "G": hw - 3 + 0.01 * (p["W"] - 110) + noise * rng.standard_normal()})
    return ResultTable(rows)


def test_lodo_holds_out_whole_designs_and_ranks_models():
    t = _synthetic()
    v = sg.lodo(t, "G")
    assert len(v.rows) == len(t) and not v.failed
    r = v.rmse
    assert r["hw_offset"] < r["freq_mean"]            # physics carries L and f
    assert r["linear"] < 0.5 and r["gp"] < 0.6        # noise is 0.2 dB
    assert 0.5 <= v.coverage95["gp"] <= 1.0


def test_holdout_and_gp_uncertainty_grows_off_data():
    t = _synthetic(n=10)
    v = sg.holdout(t, "G", ["s0", "s1"])
    assert {r["tag"] for r in v.rows} == {"s0", "s1"}
    gp = sg.GPModel().fit(t, "G")
    near = gp.predict([t.design("s2")], 3.0)[1][0]
    far = gp.predict([{"Lb": 400.0, "W": 300.0, "Wa": 250.0, "nslot": 3, "dir_a": 10.0}], 3.0)[1][0]
    assert far > near


def test_multi_fidelity_bias_correction():
    fine = _synthetic(n=4, noise=0.05, seed=3)
    coarse = ResultTable([r | {"G": r["G"] - 1.5 + 0.1 * (r["f_GHz"] - 3)} for r in fine.rows])
    mf = dz.multi_fidelity(coarse, fine, "G")
    assert mf["n_pairs"] == 12
    assert mf["rmse_coarse_vs_fine"] == pytest.approx(1.5, abs=0.15)
    assert mf["rmse_corrected_lofo"] < 0.01
    assert mf["bias_per_freq"][2.0] == pytest.approx(1.6, abs=1e-9)


def test_optimize_and_prediction_card():
    t = _synthetic(n=10)
    t = ResultTable([r | {"Greal_bore_dBi": r["G"], "S11_dB": -12.0 - 0.01 * r["Lb"]} for r in t.rows])
    models = {k: sg.LinearPhysics().fit(t, k) for k in ("Greal_bore_dBi", "S11_dB")}
    opt = dz.optimize(models, [2.0, 3.0, 4.0], budget=120, seed=0)
    assert opt.designs and all(dz.is_buildable(d) for d in opt.designs)
    assert np.all(opt.S11max <= -10.0)
    assert np.all(np.diff(opt.Gmin) <= 1e-12)             # sorted best gain first
    inside = dict(t.design("s0"), R=0.03, xs=45.0, Wg=40.0, Lo=20.0)
    card = dz.prediction_card(models["Greal_bore_dBi"], inside, 3.0, t, lodo_rmse=0.3)
    assert card.decisao in ("APPROVED", "REVIEW")
    out = dict(inside, Lb=500.0)
    assert dz.prediction_card(models["Greal_bore_dBi"], out, 3.0, t).decisao == "REJECT"   # out of domain


def test_active_learning_records_prospective_error():
    t0 = _synthetic(n=5)
    t0 = ResultTable([r | {"Greal_bore_dBi": r["G"]} for r in t0.rows])

    def simulate(designs, r):
        rows = []
        for i, d in enumerate(designs):
            for f in (2.0, 3.0, 4.0):
                g = float(bounds.endfire_directivity_dbi(d["Lb"] * 1e-3, f * 1e9)) - 3 + 0.02 * (d["W"] - 110)
                rows.append({"tag": f"al{r}_{i}", **d, "f_GHz": f, "Greal_bore_dBi": g})
        return ResultTable(rows)

    table, hist = dz.active_learning(simulate, t0, [2.0, 3.0, 4.0], rounds=2, per_round=2, n_candidates=200)
    assert len(hist) == 2 and len(table.tags) == 9
    assert all(np.isfinite(h.rmse_prospective) and h.n_train_designs >= 5 for h in hist)


# ------------------------------------------------------------------------------------------------------- EMerge
@pytest.mark.skipif(not os.environ.get("PINNEAPPLE_SLOW"), reason="EMerge FEM solve (set PINNEAPPLE_SLOW=1)")
def test_emerge_ava_one_frequency(tmp_path):
    pytest.importorskip("emerge")
    from pinneapple_simulation.electromagnetics.antenna.runner import ava_simulator
    sim = ava_simulator(str(tmp_path), timeout=1800)
    t = sim.run([{"Lb": 210.0, "res": 0.25}], [3.0], tags=["default"])
    assert sim.records[0].status == "ok", sim.records[0].message
    row = t.rows[0]
    assert os.path.exists(tmp_path / "default" / "f3" / "ff_3GHz.npz")
    rep = checks.antenna_checks(row)
    crit = {c.name: c.status for c in rep.checks}
    assert crit["efficiency_le_1"] == "supports" and crit["realized_gain"] == "supports"
    assert crit["directivity_le_hw"] == "supports" and crit["efficiency_is_G_over_D"] == "supports"
    assert -40 < row["S11_dB"] < 0 and 0 < row["rad_eff"] <= 1.0
    json.dumps(row)


def test_flow_studio_limits_flow_runs(tmp_path):
    pf = pytest.importorskip("pinneapple_flow")
    path = os.path.join(os.path.dirname(__file__), "..", "pinneapple_flow", "flows", "antenna_vivaldi_limits.json")
    with open(path) as fh:
        res = pf.run(json.load(fh), workdir=tmp_path, cache=False)
    node = res.nodes["limits"]
    assert node["status"] == "done", node["error"]
    feas = node["outputs"]["result"]["feasibility_GHz"]
    assert feas["206 mm"] == pytest.approx(bounds.feasibility_frequency(10.0, 0.206) / 1e9)

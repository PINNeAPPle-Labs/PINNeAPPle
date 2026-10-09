"""pinneapple_twin3d.wear: forecast, optimizer, geometry, dataset contract and the 4 reference use cases."""
import json
import math
import os
import shutil
import struct
import subprocess

import numpy as np
import pytest

from pinneapple_twin3d.wear import (OptimizerConfig, VesselSpec, WearDataset, ZoneSpec, backtest, explain,
                                    export_wear_twin, forecast, forecast_history, optimize, presets,
                                    scene_from_wear, synthetic_campaign)
from pinneapple_twin3d.wear.geometry import cell_mesh, cell_to_vertex
from pinneapple_twin3d.wear.optimize import predictive_policy, prepare, simulate

USE_CASES = ["steel_ladle", "pig_iron_ladle", "bof_converter", "rh_degasser", "oxyred_reactor"]


def toy_spec(e0=200, emin=100):
    return VesselSpec("toy", "Toy", (ZoneSpec("z", "Z", ((1.0, 0.0), (1.0, 1.0)), 1, 4, e0, emin),))


def toy(n, rate_a=2.0, rate_b=0.0):
    """One zone 1x4: cell 0 wears at rate_a, cell 1 at rate_b, others untouched (same toy as the JS tests)."""
    x = np.arange(n, dtype=float)
    w = np.zeros((n, 1, 4))
    w[:, 0, 0] = rate_a * x
    w[:, 0, 1] = rate_b * x
    return WearDataset(toy_spec(), x, {"z": w})


# ---------------------------------------------------------------- forecast (parity with wear-core.js)
def test_linear_remaining_life():
    f = forecast(toy(21))["z"]  # wear 40 at x=20, usable 100, 2 mm/x -> 30
    assert f.min_remaining == pytest.approx(30.0)
    assert f.rate == pytest.approx(2.0)
    assert f.confidence == "high" and f.r2 > 0.999
    assert f.critical_cell == (0, 0)


def test_no_trend_is_infinite():
    ds = toy(21, rate_a=0.0)
    assert math.isinf(forecast(ds)["z"].min_remaining)


def test_cell_at_limit_has_zero_remaining_but_rate_is_estimated():
    f = forecast(toy(60))["z"]  # wear 118 > usable 100
    assert f.min_remaining == 0.0 and f.rate == pytest.approx(2.0) and f.n_over == 1


def test_repair_resets_the_window():
    ds = toy(30)
    ds.wear["z"][15:, 0, 0] = np.arange(15)  # gunning at x=15: wear drops, then 1 mm/x
    f = forecast(ds)["z"]
    assert f.rate == pytest.approx(1.0)


def test_noise_does_not_fake_a_repair_and_interval_brackets_the_estimate():
    from pinneapple_twin3d.wear.forecast import fit_zone
    spec, hs = presets.get("bof_converter")
    ds = synthetic_campaign(spec, hs, n_readings=41, x_max=120, seed=3)  # default noise: 1 % of usable
    f = fit_zone(ds, "fundo", 40)
    assert f.n[np.isfinite(f.remaining)].min() >= 8  # windows are never truncated by noise
    ok = np.isfinite(f.remaining) & (f.remaining > 0)
    assert np.all(f.remaining_lo[ok] <= f.remaining[ok] + 1e-9) and np.all(f.remaining_hi[ok] >= f.remaining[ok] - 1e-9)
    assert np.any(f.remaining_hi[ok] > f.remaining_lo[ok])  # noisy data => a real interval
    # known limitation: at twice the noise a few cells see a >8 % dip and are truncated; keep it around 1 %
    noisy = synthetic_campaign(spec, hs, n_readings=41, x_max=120, noise=0.02, seed=3)
    g = fit_zone(noisy, "fundo", 40)
    assert np.mean(g.n[np.isfinite(g.remaining)] < 8) < 0.02  # measured ~1.2 %


def test_exact_linear_data_has_degenerate_interval():
    f = forecast(toy(21))["z"]
    assert f.remaining_lo == pytest.approx(30.0, rel=1e-6) and f.remaining_hi == pytest.approx(30.0, rel=1e-6)


def test_sparse_readings_widen_window_to_min_points():
    x = np.arange(11) * 150.0
    w = np.zeros((11, 1, 4)); w[:, 0, 0] = 0.2 * x
    ds = WearDataset(toy_spec(800, 300), x, {"z": w})
    f = forecast(ds)["z"]
    assert f.rate == pytest.approx(0.2)


def test_backtest_exact_for_linear_data():
    ds = toy(61)
    bt = backtest(ds)
    assert bt["actual_x"] == 50.0
    assert all(abs(p["error"]) < 1e-6 for p in bt["points"])
    assert backtest(toy(10))["actual_x"] is None
    h = forecast_history(ds)
    assert math.isnan(h[0]["predicted_x"])


# ---------------------------------------------------------------- optimizer (parity with optimizer.js)
CFG = OptimizerConfig(horizon=200, fixed_interval=150, windows=(0, 20, 60))


def zf(zone, deadline, rate=1.0, usable=100.0):
    from pinneapple_twin3d.wear.forecast import ZoneForecast
    return ZoneForecast(zone, 0.5, 0, deadline, rate, usable, (0, 0), "high", 1.0)


def test_reactive_breaches_predictive_does_not():
    r = optimize([zf("a", 50)], CFG)
    p = {x.id: x for x in r.policies}
    assert p["reactive"].breach > 0 and p["predictive"].breach == 0 and p["predictive"].cost < p["reactive"].cost


def test_close_deadlines_are_grouped_far_ones_are_not():
    near = optimize([zf("a", 50), zf("b", 55)], CFG).policies[-1]
    assert len(near.schedule[0]["zones"]) == 2
    cfg = OptimizerConfig(horizon=70, windows=(0, 20), fixed_interval=150)
    far = optimize([zf("a", 40), zf("b", 160)], cfg).policies[-1]
    assert [s["zones"] for s in far.schedule] == [["a"]]


def test_grouping_never_worse_than_window_zero():
    zones = [zf("a", 50), zf("b", 62), zf("c", 70)]
    zs = prepare(zones, CFG)
    p0 = simulate(zs, predictive_policy(zs, CFG, 0), CFG)
    best = optimize(zones, CFG).recommended
    assert best.cost <= p0["cost"] and best.stops <= p0["stops"]


def test_zone_already_at_limit_gets_immediate_stop_and_explanation():
    r = optimize([zf("a", 0.0)], CFG)
    assert r.recommended.schedule[0]["t"] == 0
    assert "already at the limit" in explain(r)[0]["why"]


def test_intervention_is_capped_at_design_thickness():
    cfg = OptimizerConfig(horizon=100)
    zs = prepare([zf("a", 90)], cfg)
    assert simulate(zs, [{"t": 10, "zones": ["a"]}], cfg)["breach"] == 0  # margin 80 + ext 45 capped to 100


def test_nothing_to_schedule_without_trend():
    r = optimize([zf("a", math.inf, rate=0.0)], CFG)
    assert r.recommended is None and r.notes


# ---------------------------------------------------------------- geometry
def test_cell_mesh_layout_and_orientation():
    z = ZoneSpec("c", "c", ((2.0, 0.0), (2.0, 3.0)), 3, 8, 200, 100)
    v, f = cell_mesh(z)
    assert v.shape == (3 * 8 * 4, 3) and f.shape == (3 * 8 * 2, 3) and f.max() == len(v) - 1
    r = np.hypot(v[:, 0], v[:, 2])
    assert np.allclose(r, 2.0)  # on the profile radius
    assert v[:, 1].min() == pytest.approx(0.0) and v[:, 1].max() == pytest.approx(3.0)
    # cell (0,0): first sector starts at angle 0 -> x=r, z=0, and goes counter-clockwise (z = -r sin th < 0)
    assert v[0] == pytest.approx([2.0, 0.0, 0.0])
    assert v[1, 2] < 0
    # cell values map to their 4 vertices
    vals = cell_to_vertex(np.arange(24.0).reshape(3, 8))
    assert vals.shape == (96,) and vals[4 * (1 * 8 + 2)] == 10.0


def test_off_axis_zone_is_translated():
    z = ZoneSpec("leg", "leg", ((0.5, 0.0), (0.5, 1.0)), 2, 6, 200, 100, center=(1.0, 0.0))
    v, _ = cell_mesh(z)
    assert v[:, 0].mean() == pytest.approx(1.0, abs=0.05)


def test_floor_profile_rows_are_rings():
    z = ZoneSpec("f", "f", ((0.0, 0.0), (3.0, 0.0)), 3, 12, 300, 100)
    v, _ = cell_mesh(z)
    assert np.allclose(v[:, 1], 0.0) and np.hypot(v[:, 0], v[:, 2]).max() == pytest.approx(3.0)


# ---------------------------------------------------------------- dataset contract
def test_dataset_validation():
    s = toy_spec()
    with pytest.raises(ValueError, match="strictly increasing"):
        WearDataset(s, [0, 0], {"z": np.zeros((2, 1, 4))})
    with pytest.raises(ValueError, match="expected shape"):
        WearDataset(s, [0, 1], {"z": np.zeros((2, 2, 2))})
    with pytest.raises(ValueError, match="negative"):
        WearDataset(s, [0, 1], {"z": -np.ones((2, 1, 4))})
    with pytest.raises(KeyError):
        WearDataset(s, [0, 1], {"nope": np.zeros((2, 1, 4))})
    with pytest.raises(ValueError, match="e0 > emin"):
        toy_spec(100, 100)


def test_contract_round_trip_keeps_nan_and_origin():
    spec, hs = presets.get("steel_ladle")
    ds = synthetic_campaign(spec, hs, n_readings=5)
    ds.wear["fundo"][2, 0, 0] = np.nan
    d = json.loads(json.dumps(ds.to_contract()))
    assert d["origem"] == "sintético" and d["snaps"][2]["wear"]["fundo"][0][0] is None
    back = WearDataset.from_contract(spec, d)
    assert back.origin == "synthetic" and np.isnan(back.wear["fundo"][2, 0, 0])
    assert np.allclose(back.wear["borda"], ds.wear["borda"], atol=1e-3)


def test_csv_loader():
    csv_text = "# comment\nzona,fiada,setor,valor,corrida\n" + "\n".join(
        f"z,1,{j},{10 * j + c},{c}" for c in (10, 20) for j in (1, 2, 3, 4))
    ds = WearDataset.from_csv(toy_spec(), csv_text)
    assert list(ds.x) == [10, 20] and ds.wear["z"][1, 0, 3] == 60.0 and ds.origin == "measured"
    with pytest.raises(ValueError, match="outside"):
        WearDataset.from_csv(toy_spec(), "zona,fiada,setor,valor,corrida\nz,1,9,1,1")
    with pytest.raises(ValueError, match="header"):
        WearDataset.from_csv(toy_spec(), "a,b\n1,2")


# ---------------------------------------------------------------- the four use cases, end to end
@pytest.mark.parametrize("key", USE_CASES)
def test_use_case_end_to_end(key, tmp_path):
    spec, hs = presets.get(key)
    kw = dict(repairs=(60,), repair_zones=(spec.zones[1].name,)) if key == "pig_iron_ladle" else {}
    ds = synthetic_campaign(spec, hs, n_readings=31, x_max=120, **kw)
    assert ds.origin == "synthetic"

    fc = forecast(ds)
    assert set(fc) == {z.name for z in spec.zones}
    assert all(0 <= f.worst_fraction < 1.6 for f in fc.values())
    assert any(f.min_remaining < 120 for f in fc.values()), "campaign should stress at least one zone"
    res = optimize(fc, OptimizerConfig(horizon=200))
    assert res.recommended is not None and res.recommended.breach == 0
    assert res.recommended.cost < res.baseline.cost

    out = tmp_path / key
    path = export_wear_twin(ds, str(out))
    m = json.load(open(path))
    assert [p["name"] for p in m["parts"]] == [z.name for z in spec.zones]
    assert "SYNTHETIC" in m["title"]
    assert len(m["times"]) == 31
    for z in spec.zones:
        cells = z.n_rows * z.n_sectors
        part = next(p for p in m["parts"] if p["name"] == z.name)
        assert part["vertices"] == cells * 4 and part["triangles"] == cells * 2
        names = {f["name"] for f in m["fields"] if f["part"] == z.name}
        assert names == {"wear_depth", "consumed_fraction", "residual_thickness", "remaining_life"}
    raw = open(out / "geometry.glb", "rb").read()
    assert struct.unpack("<III", raw[:12]) == (0x46546C67, 2, len(raw))
    fields = np.fromfile(out / "fields.bin", dtype="<f4")
    assert np.isfinite(fields).all()

    rep = json.load(open(out / "report.json"))
    assert rep["data_origin"] == "synthetic"
    assert set(rep) >= {"observed", "predicted", "recommended"}
    assert rep["recommended"]["disclaimer"].startswith("Decision support")
    contract = json.load(open(out / "wear_dataset.json"))
    assert contract["origem"] == "sintético"


def test_scene_field_matches_dataset_values():
    spec, hs = presets.get("bof_converter")
    ds = synthetic_campaign(spec, hs, n_readings=6)
    sc = scene_from_wear(ds)
    part = next(p for p in sc.parts if p.name == "linha_escoria")
    wd = part.fields["wear_depth"]  # (T, V)
    w = ds.wear["linha_escoria"]
    assert wd.shape == (6, w.shape[1] * w.shape[2] * 4)
    assert wd[5, 4 * (3 * w.shape[2] + 7)] == pytest.approx(w[5, 3, 7], rel=1e-5)


def test_presets_cover_the_reference_geometry():
    assert presets.get("bof_converter")[0].zone("fundo").n_rows == 12
    rh = presets.get("rh_degasser")[0]
    assert {z.group for z in rh.zones} == {"wall", "floor", "leg"}
    assert presets.get("pig_iron_ladle")[0].zone("linha_escoria").e0 == 250
    with pytest.raises(KeyError):
        presets.get("nope")


# ---------------------------------------------------------------- cross-language parity (optional)
@pytest.mark.skipif(not (os.environ.get("DIGITAL_SOLUTIONS_DIR") and shutil.which("node")),
                    reason="set DIGITAL_SOLUTIONS_DIR to the web suite to compare Python and JS numbers")
def test_python_matches_javascript_core():
    root = os.environ["DIGITAL_SOLUTIONS_DIR"]
    spec, hs = presets.get("rh_degasser")
    ds = synthetic_campaign(spec, hs, n_readings=21, x_max=120)
    contract = ds.to_contract()
    js = (f"const W=require('{root}/shared/wear-core.js'),O=require('{root}/shared/optimizer.js');"
          "const ds=JSON.parse(require('fs').readFileSync(0,'utf8'));const s=ds.snaps.length-1;"
          "const sum=W.summarizeZones(W.forecastAll(ds,s),ds);"
          "const o=O.optimize(W.zonesForOptimizer(sum),{});"
          "console.log(JSON.stringify({z:Object.fromEntries(Object.values(sum).map(z=>[z.zk,[z.minRem,z.rate,z.minLo,z.minHi]])),"
          "rec:o.recommended&&[o.recommended.id,o.recommended.cost,o.recommended.stops]}))")
    out = json.loads(subprocess.run(["node", "-e", js], input=json.dumps(contract), capture_output=True, text=True, check=True).stdout)
    fc = forecast(WearDataset.from_contract(spec, contract))  # contract rounds to 1e-3 mm on both sides
    near = lambda a, b: (math.isinf(a) if b is None else a == pytest.approx(b, rel=1e-6, abs=1e-6))  # JS Infinity -> null
    for n, (rem, rate, lo, hi) in out["z"].items():
        assert near(fc[n].min_remaining, rem)
        assert fc[n].rate == pytest.approx(rate, rel=1e-6, abs=1e-9)
        assert near(fc[n].remaining_lo, lo) and near(fc[n].remaining_hi, hi), n
    res = optimize(fc, OptimizerConfig())
    ids = {"reativo": "reactive", "fixo": "fixed", "preditivo": "predictive"}  # JS ids are Portuguese
    assert [res.recommended.id, res.recommended.stops] == [ids[out["rec"][0]], out["rec"][2]]
    assert res.recommended.cost == pytest.approx(out["rec"][1], rel=1e-6)


# ---------------------------------------------------------------- the example scripts keep working
def test_example_scripts(tmp_path):
    import importlib.util
    here = os.path.join(os.path.dirname(__file__), "..", "examples", "use_cases", "refractory_wear_twin")

    def load(name):
        spec = importlib.util.spec_from_file_location(name, os.path.join(here, name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    p = load("custom_part").build(str(tmp_path / "tank"))
    assert json.load(open(p))["title"].startswith("Cylindrical tank")
    rep = json.load(open(tmp_path / "tank" / "report.json"))
    assert rep["recommended"]["policy"]["breach"] == 0
    for key in USE_CASES:
        assert os.path.exists(load("run_use_cases").build(str(tmp_path / "all"), key))


# ---------------------------------------------------------------- calibration from plant records
from pinneapple_twin3d.wear import StopRecord, calibrate  # noqa: E402


def _stops(setup=5.0, per_zone=2.5, c_h=7.0, c_z=30.0, with_cost=True):
    out = []
    for k, n in enumerate([1, 2, 3, 4, 2, 3]):
        d = setup + per_zone * n
        out.append(StopRecord(zones=[f"z{i}" for i in range(n)], duration_h=d, cost=(c_h * d + c_z * n) if with_cost else None, x=10.0 * k))
    return out


def test_calibration_recovers_known_times_and_costs():
    cal = calibrate(_stops())
    r = {p.name: p for p in cal.report}
    assert cal.cfg.setup_hours == pytest.approx(5.0) and cal.cfg.hours_per_zone == pytest.approx(2.5)
    assert cal.cfg.cost_per_hour == pytest.approx(7.0) and cal.cfg.cost_per_zone == pytest.approx(30.0)
    assert r["setup_hours"].source == "calibrated" and r["setup_hours"].n == 6 and r["setup_hours"].se is not None


def test_calibration_marks_business_inputs_and_never_infers_them():
    cal = calibrate(_stops())
    biz = {p.name for p in cal.report if p.source == "business input"}
    assert biz == {"breach_cost_per_x", "safety_margin", "horizon", "heats_per_day", "fixed_interval"}
    assert cal.cfg.breach_cost_per_x == OptimizerConfig().breach_cost_per_x


def test_calibration_keeps_defaults_when_data_is_insufficient_or_absurd():
    assert calibrate([]).cfg == OptimizerConfig()
    same = [StopRecord(zones=["a", "b"], duration_h=h) for h in (10, 11, 12)]  # one zone count: cannot separate
    cal = calibrate(same)
    assert cal.cfg.setup_hours == OptimizerConfig().setup_hours
    assert "different zone counts" in next(p for p in cal.report if p.name == "setup_hours").note
    neg = [StopRecord(zones=["a"] * n, duration_h=d) for n, d in ((1, 10), (2, 6), (3, 2))]  # shorter with more zones
    assert calibrate(neg).cfg.hours_per_zone == OptimizerConfig().hours_per_zone
    nocost = calibrate(_stops(with_cost=False))
    assert nocost.cfg.cost_per_hour == OptimizerConfig().cost_per_hour


def test_calibration_estimates_recovery_from_detected_repairs():
    spec, hs = presets.get("pig_iron_ladle")
    ds = synthetic_campaign(spec, hs, n_readings=41, x_max=120, repairs=(60,), repair_zones=("linha_escoria",), repair_gain=0.4)
    cal = calibrate([], ds)
    r = next(p for p in cal.report if p.name == "recovery_frac")
    assert r.source == "calibrated" and r.n >= 20
    # the synthetic gunning removes 40 % of the wear accumulated by x=60 (about half the campaign): a restored share of the
    # *usable* thickness well below 40 %, but clearly positive
    assert 0.05 < cal.cfg.recovery_frac < 0.4
    # a stronger gunning must be measured as a larger recovery
    strong = synthetic_campaign(spec, hs, n_readings=41, x_max=120, repairs=(60,), repair_zones=("linha_escoria",), repair_gain=0.7)
    assert calibrate([], strong).cfg.recovery_frac > cal.cfg.recovery_frac
    assert calibrate([], synthetic_campaign(spec, hs, n_readings=11)).report[4].source == "default"  # no repairs seen


@pytest.mark.skipif(not (os.environ.get("DIGITAL_SOLUTIONS_DIR") and shutil.which("node")), reason="needs node + DIGITAL_SOLUTIONS_DIR")
def test_calibration_matches_javascript():
    root = os.environ["DIGITAL_SOLUTIONS_DIR"]
    stops = _stops()
    js = (f"const C=require('{root}/shared/calibration.js');"
          "const st=JSON.parse(require('fs').readFileSync(0,'utf8'));"
          "const base={setupHours:6,hoursPerZone:3,costPerHour:1,costPerZone:4,recoveryFrac:0.45,breachCostPerHeat:50,safetyMargin:10,horizon:300,heatsPerDay:24,fixedInterval:150};"
          "console.log(JSON.stringify(C.calibrate(st,null,base).params))")
    payload = [{"zones": len(s.zones), "durationH": s.duration_h, "cost": s.cost} for s in stops]
    out = json.loads(subprocess.run(["node", "-e", js], input=json.dumps(payload), capture_output=True, text=True, check=True).stdout)
    cfg = calibrate(stops).cfg
    assert out["setupHours"] == pytest.approx(cfg.setup_hours) and out["hoursPerZone"] == pytest.approx(cfg.hours_per_zone)
    assert out["costPerHour"] == pytest.approx(cfg.cost_per_hour) and out["costPerZone"] == pytest.approx(cfg.cost_per_zone)

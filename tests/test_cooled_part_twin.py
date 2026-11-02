"""The 3-D thermal digital twin of a cooled part (pinneapple_systems.digital_twin.cooled_part) and its study step.

References:
- A plate whose dimensions fall on the grid: its solid volume is exact (box minus the channels).
- Energy: at steady state the heat the channel walls remove equals the heat put into the loaded face (conduction
  conserves energy), and two channels placed symmetrically remove the same heat.
- Twin: on a replay from a finer-grid plant with one channel blocked at a known time, the filter alarms on that
  channel after the fault and on no channel before it, and its sensor predictions stay within twice the noise.
"""
import json
import os

import numpy as np
import pytest

from pinneapple_systems.digital_twin import cooled_part as CP

L, W, H = 0.20, 0.08, 0.016
CH_W, CH_H, CH_Y = 0.008, 0.008, (-0.030, -0.010, 0.010, 0.030)


def plate(tmp_path=None):
    V, F = CP.cold_plate(L, W, H, len(CH_Y), CH_W, CH_H)
    if tmp_path is not None:
        trimesh = pytest.importorskip("trimesh")
        p = os.path.join(tmp_path, "plate.stl")
        trimesh.Trimesh(V, F).export(p)
        return p
    return V, F


PITCH = (0.02, 0.002, 0.002)


def test_voxels_and_passages_of_a_plate():
    g = CP.voxelize_part(plate(), PITCH)
    assert g.shape == (10, 40, 8) and g.n_passages == 4 and g.flow_axis == 0
    exact = L * W * H - len(CH_Y) * L * CH_W * CH_H
    assert g.solid_volume() == pytest.approx(exact, rel=1e-12)


def test_unit_and_errors():
    V, F = plate()
    g = CP.voxelize_part((V * 1000.0, F), PITCH, unit="mm")
    assert g.n_passages == 4 and g.solid_volume() == pytest.approx(L * W * H - 4 * L * CH_W * CH_H, rel=1e-6)
    with pytest.raises(ValueError, match="unit"):
        CP.voxelize_part((V, F), PITCH, unit="ft")
    with pytest.raises(ValueError, match="no coolant passage"):
        CP.voxelize_part(_solid_box(), PITCH)
    with pytest.raises(ValueError, match="do not fit"):
        CP.cold_plate(channels=4, channel_width=0.03)


def _solid_box():
    import itertools
    c = np.array(list(itertools.product((-L / 2, L / 2), (-W / 2, W / 2), (-H / 2, H / 2))))
    tri = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6],
           [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    return c, np.array(tri)


def test_the_built_in_plate_is_a_closed_surface():
    V, F = plate()
    a = (np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]) * V[F[:, 0]]).sum() / 6    # divergence theorem
    assert a == pytest.approx(L * W * H - 4 * L * CH_W * CH_H, rel=1e-12)
    g = CP.voxelize_part("cold_plate", PITCH)
    assert g.n_passages == 4 and g.solid_volume() == pytest.approx(a, rel=1e-12)


def test_steady_energy_balance_and_symmetry():
    model = CP.CooledPart(CP.voxelize_part(plate(), PITCH), 2500.0, None, 25.0, heat_load=600.0)
    assert model.heat_in == pytest.approx(600.0, rel=1e-12)
    T = model.steady(dt=5.0, tol=1e-9)
    q = model.heat_removed(T)
    assert q.sum() == pytest.approx(model.heat_in, rel=1e-6), (q, model.heat_in)
    assert q[0] == pytest.approx(q[3], rel=1e-6) and q[1] == pytest.approx(q[2], rel=1e-6)
    assert T.min() > 25.0


@pytest.fixture(scope="module")
def replay():
    m = plate()
    model = CP.CooledPart(CP.voxelize_part(m, PITCH), 2500.0, None, 25.0, heat_load=600.0)
    plant = CP.CooledPart(CP.voxelize_part(m, np.array(PITCH) / 2), 2500.0, None, 25.0, heat_load=600.0)
    sensors = [(0.05, y, H / 2 - 0.0005) for y in CH_Y]
    rec = CP.simulate_readings(plant, sensors, t_end=600, dt=1.0, noise=0.05,
                               fault={"passage": 2, "t": 300, "ramp": 30, "fraction": 0.3}, seed=1)
    return model, rec, CP.run_twin(model, rec, seed=2)


def test_the_twin_alarms_on_the_blocked_channel_only(replay):
    model, rec, out = replay
    assert rec["simulated"] and out["simulated"]
    assert [a["passage"] for a in out["alarms"]] == [2], out["alarms"]
    assert 300 < out["alarms"][0]["t"] < 420
    checks = {c["code"]: c["status"] for c in out["checks"]}
    assert checks == {"twin.tracks": "pass", "twin.alarms": "pass", "verify.replay": "pass"}, out["checks"]
    assert out["innovation_rms"] <= 2 * rec["noise"]
    h_end = out["h_mean"][-1]
    assert h_end[1] < 0.7 * 2500.0 and all(abs(h_end[i] / 2500.0 - 1) < 0.3 for i in (0, 2, 3)), h_end


def test_a_record_without_a_fault_is_not_a_check_of_the_alarms(replay):
    model, rec, _ = replay
    k = 280
    out = CP.run_twin(model, {"t": rec["t"][:k], "y": rec["y"][:k], "sensors": rec["sensors"], "noise": 0.05},
                      seed=3)
    assert out["alarms"] == []
    assert {c["code"]: c["status"] for c in out["checks"]}["twin.alarms"] == "inconclusive"


def test_twin_step_writes_the_field_view_and_the_series(tmp_path):
    from pinneapple_flow.study_steps import twin_step
    path = plate(tmp_path)
    r = twin_step(path, str(tmp_path / "out"), heat_load=600.0, pitch=PITCH, duration=600, fault_time=300,
                  fault_passage=3, seed=4)
    assert r["simulated"] and r["grid"]["passages"] == 4
    assert [a["passage"] for a in r["alarms"]] == [3]
    assert {c["code"]: c["status"] for c in r["checks"]}["verify.replay"] == "pass"
    meta = json.loads((tmp_path / "out" / "twin.json").read_text())
    assert meta["schema"] == "pinneapple-field-view/v1" and [f["name"] for f in meta["fields"]] == ["T", "T_std"]
    assert any("passage 3" in mk["name"] for mk in meta["markers"])
    assert (tmp_path / "out" / "twin.glb").read_bytes()[:4] == b"glTF"
    s = json.loads((tmp_path / "out" / "series.json").read_text())
    assert len(s["t"]) == 600 and np.array(s["y"]).shape == (600, 4) and s["simulated"] is True
    small = r["series"]
    assert len(small["t"]) == 300 and small["t"][-1] == 600.0 and set(small) >= {"h_3", "sensor_4", "sensor_4_twin"}


def test_twin_step_reads_a_recording(tmp_path):
    from pinneapple_flow.study_steps import twin_step
    path = plate(tmp_path)
    model = CP.CooledPart(CP.voxelize_part(path, np.array(PITCH) / 2), 2500.0, None, 25.0, heat_load=600.0)
    sensors = [(0.05, y, H / 2 - 0.0005) for y in CH_Y]
    rec = CP.simulate_readings(model, sensors, t_end=200, dt=1.0, noise=0.05, seed=5)
    csv = tmp_path / "rec.csv"
    np.savetxt(csv, np.c_[rec["t"], rec["y"]], delimiter=",", header="t,s1,s2,s3,s4", comments="")
    r = twin_step(path, str(tmp_path / "o2"), heat_load=600.0, pitch=PITCH, sensors=sensors, recording=str(csv))
    assert r["simulated"] is False and r["alarms"] == []
    assert {c["code"]: c["status"] for c in r["checks"]}["twin.alarms"] == "inconclusive"
    with pytest.raises(ValueError, match="one column per sensor"):
        twin_step(path, str(tmp_path / "o3"), sensors=sensors[:2], recording=str(csv), pitch=PITCH)


def test_the_twin_kind_of_a_given_part_is_the_part_twin_step():
    from pinneapple_flow import studies as ST
    assert ST.roles_for_kind("twin", spatial=True, has_geometry=True) == ["twin"]
    assert ST.STUDY_ROLES["twin"].tool == "pinneapple_flow.study_steps:twin_step"
    assert "verify" not in ST.roles_for_kind("twin")
    assert [c for c, _ in ST.role_checks("twin", "twin")] == ["twin.tracks", "twin.alarms", "verify.replay"]


def test_the_reduced_model_follows_the_full_one_through_a_blockage():
    model = CP.CooledPart(CP.voxelize_part(plate(), PITCH), 2500.0, None, 25.0, heat_load=600.0)
    rom = CP.ReducedPart(model)
    assert rom.r < model.n // 10 and rom.energy_kept > 1 - 1e-8
    T, a = np.full(model.n, 25.0), np.zeros((rom.r, 1))
    err = 0.0
    for k in range(600):
        h = CP._blocked(model.h_nominal, {"passage": 2, "t": 300, "ramp": 30, "fraction": 0.3}, k + 1.0)
        T = model.step(T, h, 1.0)
        a = rom.step(a, h[:, None], 1.0)
        err = max(err, float(np.abs(rom.field(a)[:, 0] - T).max()))
    assert err < 0.02 * (T.max() - 25.0), (err, T.max())

"""The Unity integration: the twin folder, the Unity-authored link, the jobs and the bridge.

No Unity and no socket needed for most of it: the export is re-read by the library's own glTF reader, the Unity side
is emulated with the exact files the editor menu writes, and the bridge is driven through ``Bridge.handle``. The last
two tests use a real HTTP server with a token, and ``dotnet`` when it is installed (``test_unity_csharp.py``).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

import pinneapple_unity as pu
from pinneapple_twin3d import Scene
from pinneapple_twin3d.primitives import sweep

TIMES = [0.0, 60.0, 300.0, 900.0, 1800.0, 3600.0]


def _plate(length=0.2, width=0.08, thickness=0.006, n=21):
    section = [(-width / 2, -thickness / 2), (width / 2, -thickness / 2),
               (width / 2, thickness / 2), (-width / 2, thickness / 2)]
    s = np.linspace(0.0, length, n)
    path = np.stack([s, np.zeros(n), np.zeros(n)], axis=1)
    return sweep(section, path)


def _scene(times=TIMES) -> Scene:
    sc = Scene("Test plate", length_unit="m", times=times, time_unit="s", source="tests/test_unity_integration.py")
    v, f = _plate()
    sc.add_part("plate", v, f, group="structure", color=(0.7, 0.75, 0.8))
    sc.add_sensor("TC-01", (0.1, 0.0, 0.003), label="TC-01", unit="C")
    return sc


def _slab_params(**kw):
    p = {"alpha": 1e-5, "t_left": 120.0, "t_right": 20.0, "t0": 20.0, "axis": "x", "length": 0.2, "sensor": "TC-01"}
    p.update(kw)
    return p


# ------------------------------------------------------------------------------------------- the file lane
def test_export_writes_a_unity_project_that_checks_out(tmp_path):
    exp = pu.export_for_unity(_scene(), tmp_path / "twin")
    assert exp.verify() == [], exp.problems
    root = tmp_path / "twin"
    for rel in ("Assets/TwinData/scene.json", "Assets/TwinData/geometry.glb", "Assets/TwinData/fields.bin",
                "Assets/TwinData/pinneapple_unity.json", "Assets/PinNeAPPleTwin/Runtime/PinNeappleTwinView.cs",
                "Assets/PinNeAPPleTwin/Editor/PinNeappleMenu.cs", "README.md"):
        assert (root / rel).exists(), rel
    meta = json.loads((root / "Assets/TwinData/pinneapple_unity.json").read_text())
    assert meta["protocol"] == pu.PROTOCOL and meta["up_axis"] == "z" and meta["counts"]["steps"] == len(TIMES)
    assert "Unity" in (root / "README.md").read_text()


def test_export_round_trip_reopens_the_same_twin(tmp_path):
    sc = _scene()
    v = sc.parts[0].vertices
    shape = np.linspace(0.0, 1.0, len(v))
    sc.add_field("plate", "temperature", shape[None, :] * np.ones((len(TIMES), 1)), unit="C")
    pu.export_for_unity(sc, tmp_path / "twin")
    man, verts, fields = pu.read_unity_twin(str(tmp_path / "twin"))
    assert len(man["parts"]) == 1 and verts.shape == (len(v), 3)
    assert fields[("plate", "temperature")].shape == (len(TIMES), len(v))
    assert np.allclose(fields[("plate", "temperature")][0], np.linspace(0.0, 1.0, len(v)), atol=1e-6)
    again = pu.scene_from_twin(str(tmp_path / "twin"))
    assert [p.name for p in again.parts] == ["plate"]
    assert again.times.tolist() == TIMES
    assert again.field_units["temperature"] == "C"
    assert [s.id for s in again.sensors] == ["TC-01"]


def test_verify_reports_a_broken_field_file(tmp_path):
    sc = _scene()
    sc.add_field("plate", "temperature", np.full((len(TIMES), len(sc.parts[0].vertices)), 20.0), unit="C")
    pu.export_for_unity(sc, tmp_path / "twin")
    assert pu.verify_unity_folder(str(tmp_path / "twin")) == []
    (tmp_path / "twin" / "Assets/TwinData/fields.bin").write_bytes(b"\0\0\0\0")
    problems = pu.verify_unity_folder(str(tmp_path / "twin"))
    assert any("fields.bin" in p for p in problems), problems


# ------------------------------------------------------------------------------------------- the link lane
def _unity_folder(tmp_path, jobs=None, physics=None):
    folder = tmp_path / "unity_export"
    pu.emulate_unity_export(_scene(), folder, jobs=jobs if jobs is not None else
                            [{"job": "slab_transient", "part": "plate", "params": _slab_params()}],
                            physics=physics)
    return folder


def test_unity_export_is_read_back_with_the_axis_convention(tmp_path):
    folder = _unity_folder(tmp_path)
    link = pu.read_link(str(folder))
    assert link["protocol"] == pu.PROTOCOL and link["up_axis"] == "y"
    assert link["parts"][0]["name"] == "plate" and link["jobs"][0]["job"] == "slab_transient"
    from pinneapple_tools.visualization.studio.scene import read_gltf

    # the file Unity writes is Y-up: the plate sits in x, its 0.08 width along z and its 0.006 thickness along y
    surface = read_gltf(str(folder / "geometry.glb"), axes="as_is")[0]
    assert np.allclose(np.ptp(surface.vertices, axis=0), [0.2, 0.006, 0.08], atol=1e-5)
    # and reading it back gives the twin's Z-up frame again: 0.2 x 0.08 x 0.006
    scene, parsed = pu.from_unity(str(folder))
    assert np.allclose(np.ptp(scene.parts[0].vertices, axis=0), [0.2, 0.08, 0.006], atol=1e-5)
    assert parsed.up_axis == "y" and scene.sensors[0].unit == "C"


def test_apply_link_runs_the_jobs_with_their_checks_and_fills_the_sensor(tmp_path):
    folder = _unity_folder(tmp_path)
    scene, link = pu.from_unity(str(folder))
    reports = pu.apply_link(scene, link)
    assert [r["job"] for r in reports] == ["slab_transient"]
    assert reports[0]["decision"] == "APPROVED", reports[0]
    names = {c["name"] for c in reports[0]["checks"]}
    assert "PDE residual" in names and "boundary condition on x_min" in names
    series = scene.sensors[0].series
    assert series is not None and len(series) == len(TIMES)
    assert series[0] == pytest.approx(20.0, abs=1.0)          # starts at the initial temperature
    assert series[-1] > series[1]                             # and heats up


def test_unity_to_usd_writes_time_samples_and_usd_physics(tmp_path):
    folder = _unity_folder(tmp_path, physics={"gravity": [0.0, -9.81, 0.0],
                                              "parts": {"plate": {"collider": True, "rigid_body": False,
                                                                  "material": {"static_friction": 0.4}}}})
    out = pu.unity_to_usd(str(folder), str(tmp_path / "twin"))
    assert out["decision"] == "APPROVED"
    usda = open(out["usd"]).read()
    assert usda.startswith("#usda 1.0") and 'upAxis = "Z"' in usda
    assert "PhysicsScene" in usda and "PhysicsCollisionAPI" in usda and "PhysicsMaterialAPI" in usda
    assert "primvars:temperature.timeSamples" in usda
    assert "pinneapple-twin3d/1" in usda
    assert os.path.exists(out["manifest"])


def test_a_link_job_that_cannot_run_is_reported_not_swallowed(tmp_path):
    folder = _unity_folder(tmp_path, jobs=[{"job": "no_such_job", "part": "plate"}])
    scene, link = pu.from_unity(str(folder))
    reports = pu.apply_link(scene, link)
    assert reports[0]["decision"] == "FAILED" and "no_such_job" in reports[0]["error"]


# ------------------------------------------------------------------------------------------- the physics jobs
def test_cantilever_job_passes_below_yield_and_rejects_above_it():
    verts = np.stack([np.linspace(0.0, 0.25, 41), np.full(41, 0.1), np.zeros(41)], axis=1)
    params = {"width": 0.05, "thickness": 0.01, "load": 40.0, "length": 0.25, "axis": "x", "yield_mpa": 250.0}
    ok = pu.run_job(pu.JobRequest("cantilever_stress", "bracket", verts, times=[0.0, 3600.0], params=params))
    assert ok.decision == "APPROVED"
    peak = float(ok.fields["bending_stress"][0].max())
    assert peak == pytest.approx(12.0, rel=1e-3)             # F L 6 / (w t^2) = 12 MPa
    bad = pu.run_job(pu.JobRequest("cantilever_stress", "bracket", verts, times=[0.0, 3600.0],
                                   params={**params, "load": 1000.0}))
    assert bad.decision == "REJECT"
    assert any(c["passed"] is False and "yield" in c["name"] for c in bad.checks)


def test_fin_job_ties_the_integrated_convection_to_the_closed_form():
    verts = np.stack([np.linspace(0.0, 0.06, 41), np.zeros(41), np.zeros(41)], axis=1)
    res = pu.run_job(pu.JobRequest("fin_temperature", "fin", verts, times=[0.0, 300.0],
                                   params={"h": 60.0, "k": 205.0, "D": 0.004, "t_base": 180.0, "t_inf": 25.0,
                                           "axis": "x", "flip": True}))
    assert res.decision == "APPROVED"
    energy = next(c for c in res.checks if "dissipated heat" in c["name"])
    assert energy["value"] < 1e-3
    temp = res.fields["temperature"][0][0]
    assert temp.max() == pytest.approx(180.0, abs=1e-3)        # the base is at the base temperature
    assert temp[0] < temp.max()                                # the tip (low end, flipped) is cooler


def test_decision_follows_the_checks():
    """A critical failure rejects, a non-critical one asks for review, no checks at all is not an approval."""
    def check(name, passed, critical=True):
        return {"name": name, "passed": passed, "kind": "physics", "value": None, "threshold": None,
                "detail": "", "critical": critical}

    res = pu.JobResult("p", "j", checks=[check("residual", True), check("yield", False)])
    assert res.decision == "REJECT"
    res = pu.JobResult("p", "j", checks=[check("residual", True), check("style", False, critical=False)])
    assert res.decision == "REVIEW"
    assert pu.JobResult("p", "j").decision == "REVIEW"
    assert pu.JobResult("p", "j", checks=[check("residual", None)]).decision == "REVIEW"
    assert pu.JobResult("p", "j", checks=[check("residual", True)]).decision == "APPROVED"


# ------------------------------------------------------------------------------------------- the live lane
def test_bridge_handle_serves_a_run_without_a_socket(tmp_path):
    bridge = pu.Bridge()
    status, body = bridge.handle("/v1/health", {})
    assert status == 200 and body["ok"] and "slab_transient" in body["jobs"]
    verts = [[float(x), 0.0, 0.0] for x in np.linspace(0.0, 0.2, 21)]
    status, body = bridge.handle("/v1/run", {"job": "slab_transient", "part": "bar", "up_axis": "z",
                                             "times": TIMES, "vertices": verts, "params": _slab_params()})
    assert status == 200 and body["decision"] == "APPROVED"
    field = body["fields"]["temperature"]
    assert field["encoding"] == "base64" and field["steps"] == len(TIMES) and field["count"] == 21
    values = pu.decode_field(field)
    assert values.shape == (len(TIMES), 21)
    assert values[0, 0] == pytest.approx(120.0) and values[0, -1] == pytest.approx(20.0)
    assert values[:, 0].max() == pytest.approx(120.0)    # the hot wall stays at 120
    assert values[-1, 10] > values[0, 10] + 20.0         # the interior heats up towards the steady profile


def test_bridge_reports_bad_requests_and_refuses_folders_when_asked(tmp_path):
    bridge = pu.Bridge(allow_folders=False)
    assert bridge.handle("/v1/run", {"job": "nope", "vertices": [[0, 0, 0]]})[0] == 404
    assert bridge.handle("/v1/run", {"job": "slab_transient"})[0] == 400
    assert bridge.handle("/v1/nothing", {})[0] == 404
    status, body = bridge.handle("/v1/run", {"folder": str(tmp_path), "job": "slab_transient"})
    assert status == 400 and "folder" in body["error"]


def test_bridge_over_http_with_a_token(tmp_path):
    import urllib.error
    import urllib.request

    srv = pu.serve_bridge(port=0, block=False, token="s3cret")
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(srv.url + "v1/health")
        assert exc.value.code == 401
        health = json.loads(urllib.request.urlopen(urllib.request.Request(
            srv.url + "v1/health", headers={"X-Pinneapple-Token": "s3cret"})).read())
        assert health["ok"] and "fin_temperature" in health["jobs"]
        run = srv.post("/v1/run", {"job": "fin_temperature", "part": "fin", "up_axis": "z", "times": [0.0, 60.0],
                                   "vertices": [[0.0, 0, 0], [0.03, 0, 0], [0.06, 0, 0]],
                                   "params": {"h": 60.0, "k": 205.0, "D": 0.004, "t_base": 180.0, "t_inf": 25.0}})
        assert run["decision"] == "APPROVED" and run["seconds"] >= 0.0
        assert pu.decode_field(run["fields"]["temperature"]).shape == (2, 3)
    finally:
        srv.stop()


def test_bridge_scene_verb_runs_a_folder_and_a_twin(tmp_path):
    unity = _unity_folder(tmp_path)
    twin = tmp_path / "twin"
    pu.export_for_unity(_scene(), twin)
    srv = pu.Bridge()
    status, body = srv.handle("/v1/scene", {"folder": str(unity)})
    assert status == 200 and body["decision"] == "APPROVED" and "plate" in body["parts"]
    assert body["sensors"][0]["series"] is not None
    status, body = srv.handle("/v1/scene", {"folder": str(twin),
                                            "jobs": [{"job": "slab_transient", "part": "plate",
                                                      "params": _slab_params()}]})
    assert status == 200 and "temperature" in body["parts"]["plate"]


def test_bridge_scene_can_save_the_computed_twin_and_refuses_without_folders(tmp_path):
    twin = tmp_path / "twin"
    pu.export_for_unity(_scene(), twin)
    saved = tmp_path / "saved"
    srv = pu.Bridge()
    status, body = srv.handle("/v1/scene", {"folder": str(twin),
                                            "jobs": [{"job": "slab_transient", "part": "plate",
                                                      "params": _slab_params()}],
                                            "save": str(saved), "usd": True})
    assert status == 200 and body["decision"] == "APPROVED", body.get("error")
    written = body["twin"]
    assert pu.verify_unity_folder(written["folder"]) == [], written["problems"]
    assert os.path.exists(written["usd"])
    _man, _verts, fields = pu.read_unity_twin(written["folder"])
    assert ("plate", "temperature") in fields                  # the computed field went into the folder
    assert fields[("plate", "temperature")].shape == (len(TIMES), len(_scene().parts[0].vertices))
    status, body = pu.Bridge(allow_folders=False).handle("/v1/scene", {"folder": str(twin), "save": str(saved)})
    assert status == 400 and "folder" in body["error"]


# ------------------------------------------------------------------------------------------- the wire
def test_axis_conversion_is_its_own_inverse():
    v = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]])
    assert np.allclose(pu.y_up_to_z_up(pu.z_up_to_y_up(v)), v)
    assert np.allclose(pu.z_up_to_y_up(v), [[1.0, 3.0, -2.0], [-1.0, 4.0, -0.5]])


def test_local_coordinate_flip_and_axis():
    verts = np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 4.0, 0.0]])
    u, L = pu.protocol.local_coordinate(verts)
    assert (L, u.tolist()) == (4.0, [0.0, 2.0, 4.0])
    u, _ = pu.protocol.local_coordinate(verts, flip=True)
    assert u.tolist() == [4.0, 2.0, 0.0]


def test_job_registry_round_trip():
    def my_job(req):
        return pu.JobResult(req.part, req.job, {"dummy": (np.zeros((1, len(req.vertices)), dtype=np.float32), "")},
                            checks=[{"name": "trivial", "passed": True, "kind": "sanity", "value": 0.0,
                                     "threshold": None, "detail": "", "critical": True}])

    pu.register_job("test_dummy", my_job, "a job that returns zeros")
    try:
        assert "test_dummy" in pu.list_jobs()
        res = pu.run_job(pu.JobRequest("test_dummy", "p", np.zeros((3, 3)), times=[0.0]))
        assert res.decision == "APPROVED" and res.fields["dummy"][0].shape == (1, 3)
        with pytest.raises(KeyError):
            pu.run_job(pu.JobRequest("not_registered", "p", np.zeros((3, 3))))
    finally:
        pu.unregister_job("test_dummy")

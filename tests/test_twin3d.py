"""pinneapple_twin3d: exported glTF is valid, fields round-trip, the folder is servable."""
import json
import os
import struct
import urllib.request

import numpy as np
import pytest

from pinneapple_twin3d import Scene, serve


def _tube(n_theta=24, n_z=10, r=0.05, length=0.5):
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    z = np.linspace(0, length, n_z)
    T, Z = np.meshgrid(th, z)
    v = np.stack([r * np.cos(T).ravel(), r * np.sin(T).ravel(), Z.ravel()], 1)
    faces = []
    for j in range(n_z - 1):
        for i in range(n_theta):
            a, b = j * n_theta + i, j * n_theta + (i + 1) % n_theta
            c, d = a + n_theta, b + n_theta
            faces += [[a, b, c], [b, d, c]]
    return v, np.array(faces)


def _scene():
    v, f = _tube()
    sc = Scene("pipe test", times=[0.0, 1.0, 2.0], time_unit="year", source="unit test")
    sc.add_part("pipe", v, f, group="piping")
    sc.add_part("flange", v + [0.2, 0, 0], f, group="piping", color=(0.5, 0.5, 0.6))
    wear = np.stack([k * v[:, 2] for k in range(3)])  # (T, V), grows in time
    sc.add_field("pipe", "wear_depth", wear, unit="mm")
    sc.add_field("flange", "pressure", v[:, 2] * 1e5, unit="Pa")
    sc.add_sensor("PT-101", (0, 0, 0.25), unit="bar", series=[4.1, 4.0, 3.2], envelope=(3.5, 5.0))
    return sc, v, f, wear


def test_export_writes_valid_glb_and_manifest(tmp_path):
    sc, v, f, wear = _scene()
    path = sc.export(str(tmp_path))
    m = json.load(open(path))
    assert m["format"] == "pinneapple-twin3d/1"
    assert [p["name"] for p in m["parts"]] == ["pipe", "flange"]
    assert {"index.html", "viewer.js", "geometry.glb", "fields.bin"} <= set(os.listdir(tmp_path))

    raw = open(tmp_path / "geometry.glb", "rb").read()
    magic, version, total = struct.unpack("<III", raw[:12])
    assert (magic, version, total) == (0x46546C67, 2, len(raw))

    trimesh = pytest.importorskip("trimesh")
    loaded = trimesh.load(str(tmp_path / "geometry.glb"), force="scene")
    meshes = list(loaded.geometry.values())
    assert sorted(len(g.vertices) for g in meshes) == [len(v), len(v)]
    assert sorted(len(g.faces) for g in meshes) == [len(f), len(f)]


def test_transient_field_round_trips_through_fields_bin(tmp_path):
    sc, v, f, wear = _scene()
    m = json.load(open(sc.export(str(tmp_path), with_viewer=False)))
    meta = next(x for x in m["fields"] if x["name"] == "wear_depth")
    assert (meta["steps"], meta["count"]) == (3, len(v))
    buf = open(tmp_path / "fields.bin", "rb").read()
    got = np.frombuffer(buf, dtype="<f4", count=meta["steps"] * meta["count"], offset=meta["offset"])
    np.testing.assert_allclose(got.reshape(3, -1), wear, rtol=1e-6)
    assert meta["min"] == pytest.approx(wear.min()) and meta["max"] == pytest.approx(wear.max(), rel=1e-6)


def test_input_validation():
    sc, v, f, _ = _scene()
    with pytest.raises(ValueError, match="time steps"):
        sc.add_field("pipe", "bad", np.zeros((5, len(v))))
    with pytest.raises(ValueError, match="one value per scene time step"):
        sc.add_sensor("X", (0, 0, 0), series=[1.0])
    with pytest.raises(ValueError, match="out of range"):
        sc.add_part("broken", v, f + len(v))


def test_serve_exposes_the_scene(tmp_path):
    sc, *_ = _scene()
    sc.export(str(tmp_path))
    httpd = serve(str(tmp_path), port=0, block=False)
    try:
        body = urllib.request.urlopen(httpd.url + "scene.json", timeout=5).read()
        assert json.loads(body)["title"] == "pipe test"
        assert b"viewer.js" in urllib.request.urlopen(httpd.url, timeout=5).read()
    finally:
        httpd.shutdown()

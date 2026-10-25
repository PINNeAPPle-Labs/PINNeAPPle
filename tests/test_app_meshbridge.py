"""MeshBridge app (apps/meshbridge): the source app's examples/repair_case.py through POST /v1/convert, an Abaqus
deck to an NPZ bundle and to a surface, and the refusals."""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
trimesh = pytest.importorskip("trimesh")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INP = """*NODE
101, 0.0, 0.0, 0.0
102, 1.0, 0.0, 0.0
103, 0.0, 1.0, 0.0
104, 0.0, 0.0, 1.0
*ELEMENT, TYPE=C3D4
1, 101, 102, 103, 104
"""


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["MBR_DATA_DIR"] = str(tmp_path_factory.mktemp("mbr"))
    sys.path.insert(0, str(ROOT / "apps" / "meshbridge"))
    from meshbridge.api import app
    return TestClient(app)


def _damaged_stl() -> bytes:
    base = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    rng = np.random.default_rng(7)
    faces = np.vstack([base.faces, base.faces[rng.choice(len(base.faces), 15, replace=False)]])
    faces = np.vstack([faces, [[i, i, (i + 1) % len(base.vertices)] for i in rng.choice(len(base.vertices), 10,
                                                                                       replace=False)]])
    faces[:20] = faces[:20][:, ::-1]
    buf = io.BytesIO()
    trimesh.Trimesh(base.vertices, faces, process=False).export(buf, file_type="stl")
    return buf.getvalue()


def test_health_and_formats(client):
    assert client.get("/health").json()["status"] == "ok"
    f = client.get("/v1/formats").json()
    assert "abaqus" in f["surface_from"] and "cgns" in f["npz_bundle_from"] and "fluent" not in f["surface_from"]


def test_repair_case_through_the_endpoint(client):
    r = client.post("/v1/convert", files={"file": ("damaged.stl", _damaged_stl(), "application/octet-stream")},
                    data={"output": "stl", "repair": "true"})
    assert r.status_code == 200, r.text
    fixed = trimesh.load(io.BytesIO(r.content), file_type="stl", force="mesh", process=True)
    assert len(fixed.faces) == int(fixed.unique_faces().sum()) == int(fixed.nondegenerate_faces().sum()) == 320
    assert fixed.is_winding_consistent and "winding_consistent=True" in r.headers["x-report"]


def test_abaqus_deck_to_bundle_and_surface(client):
    r = client.post("/v1/convert", files={"file": ("tet.inp", INP.encode(), "text/plain")}, data={"output": "npz"})
    assert r.status_code == 200, r.text
    with np.load(io.BytesIO(r.content)) as z:
        assert z["coords"].shape == (4, 3)
    r = client.post("/v1/convert", files={"file": ("tet.inp", INP.encode(), "text/plain")},
                    data={"output": "obj", "repair": "true"})
    assert r.status_code == 200 and "faces_out=4" in r.headers["x-report"] and "watertight=True" in r.headers["x-report"]


def test_refusals(client):
    assert client.post("/v1/convert", files={"file": ("x.xyz", b"1 2 3", "text/plain")}).status_code == 415
    r = client.post("/v1/convert", files={"file": ("a.stl", _damaged_stl(), "application/octet-stream")},
                    data={"output": "npz"})
    assert r.status_code == 422                                  # an STL has no point bundle
    assert client.post("/v1/convert", files={"file": ("a.stl", b"", "x")}, data={"output": "step"}).status_code == 422

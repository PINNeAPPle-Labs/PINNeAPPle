"""pinneapple_data.cae and the Mesh Quality app: readers, checkMesh-equivalent finite-volume metrics (reference values
from OpenFOAM v1912 checkMesh on the same meshes), element shape metrics, connectivity and surface checks."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from pinneapple_data.cae import mesh_report, read_any
from pinneapple_data.cae import quality as Q
from pinneapple_data.cae.model import CellBlock, Mesh
from pinneapple_data.cae.upload import expand

APP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "apps", "mesh_quality"))
EX = os.path.join(APP, "examples")


def _mesh(name):
    with open(os.path.join(EX, name), "rb") as f:
        return read_any(expand([(name, f.read())]), fields=False)


def test_pitzdaily_matches_checkmesh():
    m = _mesh("openfoam_pitzDaily.zip")
    assert m.n_cells == 12225 and m.element_counts() == {"hexahedron": 12225}
    r = mesh_report(m)
    mt = r["metrics"]
    assert mt["aspect_ratio"]["worst"] == pytest.approx(8.1407, rel=1e-4)
    assert mt["non_orthogonality"]["worst"] == pytest.approx(5.95045, rel=1e-5)
    assert mt["non_orthogonality"]["mean"] == pytest.approx(1.63034, rel=1e-5)
    assert mt["skewness"]["worst"] == pytest.approx(0.260575, rel=1e-5)
    assert r["summary"]["total_volume"] == pytest.approx(1.4516e-05, rel=1e-4)
    assert r["status"] == "PASS" and r["summary"]["regions"] == 1


def test_sheared_channel_matches_checkmesh_failure():
    r = mesh_report(_mesh("openfoam_sheared_channel.zip"))
    sk = r["metrics"]["skewness"]
    assert sk["worst"] == pytest.approx(6.29531, rel=1e-5) and sk["bad"] == 22   # checkMesh: 22 highly skew faces
    assert r["metrics"]["non_orthogonality"]["worst"] == pytest.approx(68.3409, rel=1e-5)
    assert r["status"] == "WARNING"
    assert sum(g["cells"] for g in r["regions"]) == 22 and all(g["nearest_patch"] == "walls" for g in r["regions"])


def test_gmsh_tets_read_directly_match_checkmesh_after_gmshToFoam():
    m = _mesh("gmsh_bracket_tet.msh")
    r = mesh_report(m, target="cfd")
    mt = r["metrics"]
    assert m.n_cells == 8828
    assert mt["non_orthogonality"]["worst"] == pytest.approx(69.027, rel=1e-5)
    assert mt["non_orthogonality"]["mean"] == pytest.approx(21.8577, rel=1e-5)
    assert mt["skewness"]["worst"] == pytest.approx(0.816741, rel=1e-5)
    assert mt["aspect_ratio"]["worst"] == pytest.approx(7.91293, rel=1e-5)
    assert r["summary"]["total_volume"] == pytest.approx(0.000111077, rel=1e-5)


@pytest.mark.parametrize("t", ["tetra", "hexahedron", "wedge", "pyramid"])
def test_scaled_jacobian_ideal_elements_both_conventions(t):
    flip = {"tetra": (0, 2, 1, 3), "hexahedron": (0, 3, 2, 1, 4, 7, 6, 5), "wedge": (0, 2, 1, 3, 5, 4),
            "pyramid": (0, 3, 2, 1, 4)}
    p = np.array(Q._IDEAL[t], float)
    for conn in (np.arange(len(p)), np.array(flip[t])):
        m = Mesh(points=p, blocks=[CellBlock(t, conn[None])])
        assert Q.element_metrics(m)["scaled_jacobian"][0] == pytest.approx(1.0)


def test_inverted_element_stray_part_and_hinge():
    r = mesh_report(_mesh("assembly_with_errors.vtu"))
    assert r["status"] == "FAIL"
    st = {c["key"]: c for c in r["checks"]}
    assert st["scaled_jacobian"]["status"] == "fail" and r["metrics"]["scaled_jacobian"]["worst"] < 0
    assert r["summary"]["regions"] == 3 and st["hinge"]["status"] == "warn"
    assert r["regions"][0]["status"] == "fail"


def test_damaged_stl_surface():
    r = mesh_report(_mesh("bracket_surface_damaged.stl"))
    s = r["surface"]
    assert (s["open_edges"], s["inconsistent_orientation_edges"], s["shells"]) == (9, 13, 2)
    assert r["status"] == "FAIL"


def test_calculix_deck():
    m = _mesh("calculix_cantilever_hex.inp")
    assert (m.n_cells, m.n_points) == (640, 1025) and set(m.node_sets) >= {"FIXED", "TIP"}
    assert mesh_report(m)["status"] == "PASS"


def test_api_examples():
    pytest.importorskip("fastapi")
    sys.path.insert(0, APP)
    from fastapi.testclient import TestClient
    from mesh_quality.api import app
    want = {"openfoam_pitzDaily.zip": "PASS", "openfoam_sheared_channel.zip": "WARNING", "gmsh_bracket_tet.msh": "WARNING",
            "assembly_with_errors.vtu": "FAIL", "calculix_cantilever_hex.inp": "PASS", "bracket_surface_damaged.stl": "FAIL"}
    with TestClient(app) as c:
        for name, status in want.items():
            j = c.get(f"/api/example/{name}").json()
            assert j["status"] == status, name
            assert j["view"]["tris"] and j["checks"]
        with open(os.path.join(EX, "gmsh_bracket_tet.msh"), "rb") as f:
            j = c.post("/api/check", files={"files": ("b.msh", f.read())}, data={"target": "cfd"}).json()
        assert j["summary"]["target"] == "cfd"
        assert c.post("/api/check", files={"files": ("x.txt", b"hello")}).status_code == 422

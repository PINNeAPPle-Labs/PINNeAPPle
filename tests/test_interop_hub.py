"""Interoperability Hub: the neutral dataset and every export, reread and compared with the source."""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import zipfile

import numpy as np
import pytest

from pinneapple_data.cae import read_any
from pinneapple_data.cae.dataset import build_dataset, export, poly_cells
from pinneapple_data.cae.model import Mesh, PolyMesh
from pinneapple_data.cae.upload import expand

APP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "apps", "interop_hub"))
EX = os.path.join(APP, "examples")


def _m(name):
    with open(os.path.join(EX, name), "rb") as f:
        return read_any(expand([(name, f.read())]))


@pytest.fixture(scope="module")
def pitz():
    m = _m("openfoam_pitzDaily_result.zip")
    return m, build_dataset(m)


def _tmp(blob, ext):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "out." + ext)
    with open(p, "wb") as f:
        f.write(blob)
    return p


def test_vtu_reread_with_vtk_and_meshio(pitz):
    m, ds = pitz
    blocks, polys, order = poly_cells(m)
    assert {t: len(c) for t, c in blocks} == {"hexahedron": 12225} and not polys
    p = _tmp(export(ds, "vtu")[0], "vtu")
    import meshio
    mm = meshio.read(p)
    assert sum(len(c.data) for c in mm.cells) == 12225
    u = np.concatenate(mm.cell_data["U"])
    assert np.allclose(u, ds["fields"]["U"]["values"][order])
    pv = pytest.importorskip("pyvista")
    g = pv.read(p)
    assert g.volume == pytest.approx(1.4516e-05, rel=1e-4)
    assert g.compute_cell_sizes()["Volume"].min() == pytest.approx(1.6902e-10, rel=1e-4)   # checkMesh "Min volume"


def test_polyhedron_cell_written_as_vtk_polyhedron():
    # a unit cube whose bottom face is split in two triangles: 7 faces, not a standard shape
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    faces = [[0, 2, 1], [0, 3, 2], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    flat = np.array([n for f in faces for n in f])
    offs = np.concatenate([[0], np.cumsum([len(f) for f in faces])])
    poly = PolyMesh(flat, offs, np.zeros(7, np.int64), np.zeros(0, np.int64), 1,
                    [{"name": "walls", "type": "wall", "startFace": 0, "nFaces": 7}])
    m = Mesh(points=pts, poly=poly, source={"format": "test"})
    m.cell_data["T"] = np.array([42.0])
    blocks, polys, _ = poly_cells(m)
    assert not blocks and len(polys) == 1
    p = _tmp(export(build_dataset(m), "vtu")[0], "vtu")
    pv = pytest.importorskip("pyvista")
    g = pv.read(p)
    assert g.n_cells == 1 and g.celltypes[0] == 42 and g.volume == pytest.approx(1.0)
    assert g.cell_data["T"][0] == 42.0


def test_tabular_and_array_exports(pitz):
    m, ds = pitz
    import pandas as pd
    import pyarrow.parquet as pq
    U = ds["fields"]["U"]["values"]
    t = pq.read_table(io.BytesIO(export(ds, "parquet")[0]))
    df = t.to_pandas()
    assert np.allclose(df["U_x"], U[:, 0]) and len(df) == 12225
    assert json.loads(t.schema.metadata[b"pinneapple"])["column_units"]["U_x"] == "m/s"
    csv = pd.read_csv(io.BytesIO(export(ds, "csv")[0]), comment="#")
    assert "U_x [m/s]" in csv.columns and np.allclose(csv["U_x [m/s]"], U[:, 0], rtol=1e-6)
    z = np.load(io.BytesIO(export(ds, "npz")[0]))
    assert np.array_equal(z["cell/U"], U)
    import h5py
    with h5py.File(io.BytesIO(export(ds, "hdf5")[0])) as h:
        assert np.array_equal(h["fields/U"][()], U) and h["fields/U"].attrs["unit"] == "m/s"
        assert h["mesh/polyMesh/owner"].shape[0] == m.poly.n_faces
    j = json.loads(export(ds, "json")[0])
    assert j["schema"] == "pinneapple.physical_dataset/1" and len(j["fields"]["U"]["values"]) == 12225


def test_pinneapple_dataset_round_trip_with_the_library(pitz):
    m, ds = pitz
    from pinneapple_data.serialization import load_zarr
    with tempfile.TemporaryDirectory() as d:
        zipfile.ZipFile(io.BytesIO(export(ds, "pinneapple")[0])).extractall(d)
        s = load_zarr(os.path.join(d, "dataset.zarr"))[0]
        assert np.allclose(np.asarray(s.state["U"]), ds["fields"]["U"]["values"])
        assert s.provenance["fields"]["U"]["unit"] == "m/s" and s.provenance["solver"] == "OpenFOAM simpleFoam"


def test_units_calculix_to_si_and_kinematic_pressure():
    m = _m("calculix_cantilever.frd")
    a = build_dataset(m, unit_system="N-mm-t-s")
    b = build_dataset(m, unit_system="N-mm-t-s", to_si=True)
    assert a["fields"]["DISP"]["unit"] == "mm" and b["fields"]["DISP"]["unit"] == "m"
    assert np.allclose(b["fields"]["DISP"]["values"], a["fields"]["DISP"]["values"] * 1e-3)
    assert b["fields"]["STRESS"]["unit"] == "Pa" and np.allclose(b["fields"]["STRESS"]["values"], a["fields"]["STRESS"]["values"] * 1e6)
    assert b["geometry"]["bounding_box"]["size"][0] == pytest.approx(0.2)
    c = _m("openfoam_cavity_Re100_result.zip")
    k = build_dataset(c)
    p = build_dataset(c, rho=998.0)
    assert k["fields"]["p"]["unit"] == "m2/s2" and p["fields"]["p"]["unit"] == "Pa"
    assert np.allclose(p["fields"]["p"]["values"], 998.0 * k["fields"]["p"]["values"])


def test_api_inspect_and_exports():
    pytest.importorskip("fastapi")
    sys.path.insert(0, APP)
    from fastapi.testclient import TestClient
    from interop_hub.api import app
    with TestClient(app) as c:
        j = c.get("/api/example/calculix_cantilever.frd").json()
        F = j["dataset"]["fields"]
        assert F["STRESS"]["unit"] == "MPa" and F["STRESS"]["scalar"]["kind"] == "von Mises"
        assert F["DISP"]["scalar"]["min"] >= 0 and abs(F["DISP"]["scalar"]["max"] - 7.58) < 0.01   # tip deflection
        assert F["ERROR"]["quantity"] == "stress error estimate" and F["ERROR"]["unit"] == "%"
        g = c.get("/api/example/gmsh_bracket_mesh.msh").json()
        assert not g["dataset"]["fields"] and len(g["preview"]["tris"]) > 0      # a bare mesh still previews
        for fmt in ("vtu", "hdf5", "parquet", "csv", "npz", "json", "pinneapple"):
            assert c.get(f"/api/export/{j['id']}/{fmt}").status_code == 200
        with open(os.path.join(EX, "openfoam_cavity_Re100_result.zip"), "rb") as f:
            r = c.post("/api/convert/csv", files={"files": ("cav.zip", f.read())}, data={"rho": "1.2"})
        assert r.status_code == 200 and b"p [Pa]" in r.content[:400]
        assert c.get("/api/export/deadbeef/vtu").status_code == 404

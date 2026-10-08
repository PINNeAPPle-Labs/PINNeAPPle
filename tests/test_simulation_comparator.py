"""Simulation Comparator: real comparisons (OpenFOAM v1912 runs, Ghia et al. 1982 benchmark, Euler-Bernoulli,
PINNeAPPle's PINN plate) and the interpolation accuracy on an analytic field."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from pinneapple_data.cae import read_any
from pinneapple_data.cae.compare import cell_centres_volumes, compare, match_fields
from pinneapple_data.cae.upload import expand

APP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "apps", "simulation_comparator"))
EX = os.path.join(APP, "examples")


def _m(name, fields=True):
    with open(os.path.join(EX, name), "rb") as f:
        return read_any(expand([(name, f.read())]), fields=fields)


def _rel(ref, cand, field=None):
    res = compare(_m(ref), _m(cand))
    f = res["fields"][0] if field is None else next(x for x in res["fields"] if x["reference"] == field)
    return f


def test_cavity_against_ghia_converges_with_the_mesh():
    e20 = _rel("ghia1982_cavity_Re100_u_centreline.csv", "cavity_Re100_20x20.zip")
    e40 = _rel("ghia1982_cavity_Re100_u_centreline.csv", "cavity_Re100_40x40.zip")
    assert e20["candidate"] == "U[x]" and e20["outside_candidate"] == 0      # lid and wall values come from the BCs
    assert e20["summary"]["rel_l2"] == pytest.approx(0.0126, abs=0.001)
    assert e40["summary"]["rel_l2"] == pytest.approx(0.0019, abs=0.0005)


def test_calculix_against_beam_theory():
    f = _rel("cantilever_euler_bernoulli.csv", "calculix_cantilever_results.frd")
    assert f["candidate"] == "DISP[z]" and f["summary"]["rel_l2"] == pytest.approx(0.0071, abs=0.001)


def test_pinn_against_finite_volumes_same_locations():
    f = _rel("plate_finite_volume_reference.csv", "plate_pinn_prediction.csv")
    assert f["interpolation"] == "same locations" and f["unit"] == "degC"
    assert f["summary"]["max_abs"] == pytest.approx(0.674, abs=0.001)


def test_pitzdaily_mesh_and_model_comparisons():
    u = _rel("pitzDaily_kEpsilon_fine_48900cells.zip", "pitzDaily_kEpsilon_12225cells.zip", "U")
    assert u["interpolation"] == "linear" and u["summary"]["rel_l2"] == pytest.approx(0.0319, abs=0.002)
    m = _rel("pitzDaily_kEpsilon_12225cells.zip", "pitzDaily_kOmegaSST_12225cells.zip", "U")
    assert m["interpolation"] == "same locations" and m["summary"]["rel_l2"] == pytest.approx(0.0451, abs=0.002)


def test_interpolation_error_on_an_analytic_field():
    ref, cand = _m("pitzDaily_kEpsilon_fine_48900cells.zip", False), _m("pitzDaily_kEpsilon_12225cells.zip", False)
    f = lambda c: 5 + np.sin(20 * c[:, 0]) * np.cos(60 * c[:, 1])  # noqa: E731
    ref.cell_data["f"] = f(cell_centres_volumes(ref)[0])
    cand.cell_data["f"] = f(cell_centres_volumes(cand)[0])
    s = compare(ref, cand)["fields"][0]["summary"]
    assert s["rel_l2"] < 0.001 and s["nrmse"] < 0.002


def test_field_matching_aliases_and_components():
    from pinneapple_data.cae.model import Mesh
    a, b = Mesh(points=np.zeros((3, 3))), Mesh(points=np.zeros((3, 3)))
    a.point_data.update(temperature=np.ones(3), Ux=np.ones(3), pressure=np.ones(3))
    b.point_data.update(T=np.ones(3), U=np.ones((3, 3)), p=np.ones(3))
    assert set(match_fields(a, b)) == {("temperature", "T"), ("Ux", "U[x]"), ("pressure", "p")}


def test_api_examples_and_upload():
    pytest.importorskip("fastapi")
    sys.path.insert(0, APP)
    from fastapi.testclient import TestClient
    from simulation_comparator.api import app
    with TestClient(app) as c:
        j = c.get("/api/example/ghia40").json()
        assert j["status"] == "PASS" and j["view"]["fields"]["Ux"]["kind"] == "line"
        j = c.get("/api/example/model?tolerance=5").json()
        assert j["status"] == "FAIL" and j["view"]["fields"]["U"]["kind"] == "surface_cell"
        rd = lambda n: open(os.path.join(EX, n), "rb").read()  # noqa: E731
        r = c.post("/api/compare", files=[("reference", ("ref.csv", rd("plate_finite_volume_reference.csv"))),
                                          ("candidate", ("pinn.csv", rd("plate_pinn_prediction.csv")))],
                   data={"mode": "sim-ai", "tolerance": "1"})
        assert r.status_code == 200 and r.json()["status"] == "PASS"
        bad = c.post("/api/compare", files=[("reference", ("a.csv", b"x,y,q\n0,0,1\n1,0,2\n")),
                                            ("candidate", ("b.csv", b"x,y,r\n0,0,1\n1,0,2\n"))])
        assert bad.status_code == 422 and "no common field" in bad.json()["detail"]

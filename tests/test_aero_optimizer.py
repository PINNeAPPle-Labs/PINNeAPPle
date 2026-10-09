"""Aircraft Design Optimizer: shapes, the O-grid, forces, the aircraft model, the trained surrogates, the API."""
from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import zipfile

import numpy as np
import pytest

from pinneapple_design.aero.aircraft import KT, Aircraft, Polar, Requirements, evaluate, performance
from pinneapple_design.aero.case import forces, write_case
from pinneapple_design.aero.geometry import BOUNDS, REFERENCE, naca4, properties, surfaces, valid
from pinneapple_design.aero.mesh import GridSpec, cell_centres, grid_points, topology, write_polymesh

APP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "apps", "aero_optimizer"))
MODEL = os.path.join(APP, "model", "surrogate.pt")

# OpenFOAM polar of the NACA 2412 on the production grid (Re 4e6, k-omega SST), from the training-set reference runs
A2412 = [-2, 0, 2, 4, 6, 8, 10, 12, 14]
CL2412 = [-0.002, 0.221, 0.443, 0.662, 0.877, 1.082, 1.275, 1.446, 1.587]
CD2412 = [0.0088, 0.0088, 0.0093, 0.0101, 0.0115, 0.0134, 0.0161, 0.0198, 0.0256]
CM2412 = [-0.047] * 9


def test_naca_fit_and_properties():
    p = properties(naca4("2412"))
    assert abs(p["thickness"] - 0.12) < 0.003 and abs(p["camber"] - 0.02) < 0.002
    assert abs(properties(naca4("0012"))["camber"]) < 1e-6
    for name, s in REFERENCE.items():
        assert valid(s)[0]
        if name in ("NACA 2412", "NACA 0012"):                           # inside the design space; 4412/4415 lie just
            assert ((s >= BOUNDS[:, 0]) & (s <= BOUNDS[:, 1])).all()    # outside it (mid camber) -> an extrapolation test
    yu, yl = surfaces(naca4("4412"), np.linspace(0, 1, 50))
    assert (yu - yl >= -1e-12).all()


def test_ogrid_topology_and_cells():
    g = GridSpec()
    t = topology(g)
    assert len(t["wall"]) == len(t["far"]) == g.ni
    assert len(t["internal"]) == g.ni * g.nj + g.ni * (g.nj - 1)
    own = [o for o, n, _ in t["internal"]]
    assert own == sorted(own)                                         # upper-triangular order for OpenFOAM
    for s in (naca4("2412"), naca4("0012"), np.mean(BOUNDS, 1)):
        P = grid_points(s, g)
        q = np.concatenate([P, P[:, :1]], 1)
        a, b, c, d = q[:-1, :-1], q[:-1, 1:], q[1:, 1:], q[1:, :-1]
        cross = lambda u, v: u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]   # noqa: E731
        area = 0.5 * (cross(c - a, d - b))
        assert (np.abs(area) > 0).all() and (np.sign(area) == np.sign(area.flat[0])).all()   # no folded cells
        assert abs(np.linalg.norm(P[-1] - [0.5, 0], axis=1).mean() - g.radius) < 1e-6


def test_polymesh_passes_checkmesh_when_available():
    tmp = tempfile.mkdtemp()
    try:
        write_case(tmp + "/c", naca4("4412"), 4.0)
        assert os.path.exists(tmp + "/c/constant/polyMesh/owner")
        bashrc = "/usr/share/openfoam/etc/bashrc"
        if not os.path.exists(bashrc):
            pytest.skip("OpenFOAM not installed")
        import subprocess
        out = subprocess.run(["bash", "-c", f'source {bashrc} >/dev/null 2>&1; cd {tmp}/c && checkMesh'],
                             capture_output=True, text=True).stdout
        # y+ < 1 wall cells stretched over the far field: aspect ratio > 1000 is expected (NASA's grids go past 1e4);
        # every other check must pass
        failed = [ln for ln in out.splitlines() if ln.strip().startswith("***")]
        assert all("aspect ratio" in ln for ln in failed), failed
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_force_integration_closed_surface():
    s = naca4("2412")
    P = grid_points(s)
    ni = P.shape[1]
    r = forces(P, np.full(ni, 3.7), np.zeros((ni, 3)), 1e-6, 4.0)     # uniform pressure, fluid at rest
    assert abs(r["Cl"]) < 1e-9 and abs(r["Cd"]) < 1e-9 and abs(r["Cm"]) < 1e-9
    cc = cell_centres(P)[0]
    U = np.zeros((ni, 3)); U[:, 0] = 1.0                               # slip velocity along x -> drag from friction only
    r = forces(P, np.zeros(ni), U, 1e-6, 0.0)
    assert r["Cd"] > 0 and r["Cd_pressure"] == 0


def test_aircraft_model_cessna_172_class():
    e = evaluate(properties(naca4("2412")), Polar(np.array(A2412, float), np.array(CL2412), np.array(CD2412), np.array(CM2412)))
    assert 118 < e["vmax_kt"] < 132                                    # C172: ~124 kt
    assert 12 < e["fuel_l_100km"] < 17                                  # C172: ~15 L/100 km at 107 kt
    assert 45 < e["v_stall_kt"] < 58 and e["feasible"]
    # a smaller wing is faster but stalls higher
    small = performance(Polar(np.array(A2412, float), np.array(CL2412), np.array(CD2412), np.array(CM2412)),
                        Aircraft(wing_area=12.0))
    assert small["vmax_kt"] > e["vmax_kt"] and small["v_stall_kt"] > e["v_stall_kt"]


@pytest.fixture(scope="module")
def engine():
    from pinneapple_design.aero.optimize import Engine
    return Engine.load(MODEL)


def test_surrogate_matches_openfoam_on_naca2412(engine):
    co = engine.mlp_coefficients(naca4("2412")[None], np.array([0.0, 4.0, 8.0]))
    assert np.allclose(co["cl"][0], [0.221, 0.662, 1.082], atol=0.05)
    assert np.allclose(co["cd"][0], [0.0088, 0.0101, 0.0134], rtol=0.08)
    g = engine.gnn_predict(naca4("2412"), [4.0])
    assert abs(g["cl"][0] - 0.662) < 0.06 and abs(g["cd"][0] - 0.0101) / 0.0101 < 0.12
    assert g["fields"].shape[-1] == 4


def test_search_finds_flyable_designs_that_beat_the_baseline(engine):
    from pinneapple_design.aero.optimize import baseline
    ac, req = Aircraft(), Requirements()
    res = engine.search(ac, req, population=48, generations=25, seed=1)
    D = res["designs"]
    assert res["pareto"], "no Pareto designs"
    for i in res["pareto"]:
        d = D[i]
        assert d["feasible"] and d["trust"] != "low"
        assert d["v_stall"] <= req.max_stall_speed + 1e-9 and d["props"]["thickness"] >= req.min_thickness
    b = baseline(engine, ac, req)
    best = max(res["pareto"], key=lambda i: D[i]["vmax_kt"])
    assert D[best]["vmax_kt"] > b["vmax_kt"]
    assert any(not d.get("feasible") for d in D)                       # it also finds designs that do not work


def test_api_end_to_end():
    sys.path.insert(0, APP)
    from fastapi.testclient import TestClient
    from aero_optimizer.api import app
    with TestClient(app) as c:
        m = c.get("/api/meta").json()
        assert m["gnn"] and "NACA 2412" in m["references"]
        r = c.post("/api/optimize", json={"population": 32, "generations": 12}).json()
        assert r["evaluations"] == 32 * 12 and r["pareto"] and set(r["picks"]) >= {"fastest", "balanced"}
        x = r["details"][str(r["picks"]["balanced"])]["x"]
        d = c.post("/api/design", json={"x": x}).json()
        assert d["flow"]["cases"][0]["p"] and len(d["flow"]["x"]) == d["flow"]["ni"] * d["flow"]["nj"]
        z = c.get("/api/openfoam-case", params={"x": ",".join(map(str, x)), "alpha": 2})
        names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
        assert any(n.endswith("constant/polyMesh/faces") for n in names) and any(n.endswith("coefficients.py") for n in names)
        assert c.post("/api/design", json={"x": [9, 9, 9, 9, 9, 9, 9]}).status_code == 422

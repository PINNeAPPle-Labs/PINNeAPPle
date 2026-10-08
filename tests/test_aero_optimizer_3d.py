"""Whole aircraft: parametric airframe and its exports, the vortex lattice against theory, the 3D aircraft model on
the Cessna 172 class, the 3D search, the 3D API."""
from __future__ import annotations

import json
import math
import os
import struct
import sys

import numpy as np
import pytest

from pinneapple_design.aero.aircraft3d import Aircraft3D, Requirements3D, airframe_from, baseline_x
from pinneapple_design.aero.airframe import Airframe, _signed_volume, to_glb, to_stl, to_usda
from pinneapple_design.aero.geometry import REFERENCE
from pinneapple_design.aero.vlm import solve, streamlines

APP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "apps", "aero_optimizer"))
MODEL = os.path.join(APP, "model", "surrogate.pt")


def test_airframe_parts_and_exports():
    af = Airframe()
    assert abs(af.span - math.sqrt(16.2 * 7.5)) < 1e-9 and abs(af.root_chord * (1 + af.taper) * af.span / 2 - 16.2) < 1e-9
    parts = af.build()
    names = {p.name for p in parts}
    assert {"fuselage", "wing_right", "wing_left", "htail_right", "vtail", "spinner"} <= names
    for p in parts:
        assert _signed_volume(p.vertices, p.faces) > 0, p.name            # every body faces outward
    wing = next(p for p in parts if p.name == "wing_right")
    assert abs(wing.vertices[:, 1].max() - af.span / 2) < 1e-6
    glb = to_glb(parts)
    magic, ver, length = struct.unpack("<III", glb[:12])
    assert magic == 0x46546C67 and ver == 2 and length == len(glb)
    jl = struct.unpack("<I", glb[12:16])[0]
    gl = json.loads(glb[20:20 + jl])
    assert len(gl["meshes"]) == len(parts) and any(n.get("name") == "propeller" for n in gl["nodes"])
    assert any("KHR_materials_clearcoat" in m.get("extensions", {}) for m in gl["materials"])
    usd = to_usda(parts)
    assert usd.startswith("#usda 1.0") and usd.count('def Mesh "') == len(parts) and "UsdPreviewSurface" in usd
    stl = to_stl([p for p in af.build("cfd") if p.aero])
    assert struct.unpack("<I", stl[80:84])[0] * 50 + 84 == len(stl)


def _flat(AR, taper):
    return Airframe(aspect_ratio=AR, taper=taper, section=REFERENCE["NACA 0012"].copy(), twist=0, incidence=0,
                    dihedral=0, sweep_le=0, wing_area=16)


def test_vortex_lattice_against_theory():
    sol = solve(_flat(6, 1.0), 2.0, nc=6, ns_wing=24, tail=False)
    c = sol.coefficients(math.radians(5), 0)
    e = c["CL"] ** 2 / (math.pi * 6 * c["CDi"])
    assert 4.1 < sol.CL[1] < 4.35 and 0.97 < e < 0.99                    # rectangular AR 6
    sol = solve(_flat(8, 0.4), 2.0, nc=6, ns_wing=24, tail=False)
    c = sol.coefficients(math.radians(5), 0)
    assert 0.99 < c["CL"] ** 2 / (math.pi * 8 * c["CDi"]) < 1.0           # near-elliptic, never above 1
    # the tail stabilises: with it the moment slope is negative about a forward CG
    af = Airframe()
    assert solve(af, 2.6).Cm[1] < 0
    lines = streamlines(solve(af, 2.6), 0.05, 0.0, np.array([[1.0, 2.0, 0.6]]), steps=40)
    assert lines.shape == (1, 41, 3) and lines[0, -1, 0] > 3.5           # carried downstream


@pytest.fixture(scope="module")
def e3():
    from pinneapple_design.aero.optimize import Engine
    from pinneapple_design.aero.optimize3d import Engine3D
    return Engine3D(Engine.load(MODEL))


def test_cessna_class_baseline(e3):
    from pinneapple_design.aero.aircraft import Aircraft
    r = e3.evaluate(baseline_x()[None], Aircraft(), Requirements3D(), fine=True)[0]
    assert 118 < r["vmax_kt"] < 130                                       # C172N: ~123 kt
    assert 12.5 < r["fuel_l_100km"] < 16.5                                # ~15 L/100 km at 107 kt
    assert 48 < r["v_stall_kt"] < 58 and r["feasible"]
    assert 0.12 < r["static_margin"] < 0.35 and r["stall_station"] < 0.4  # washout: the root stalls first
    assert 100 < r["weights"]["wing"] < 180
    cd0 = r["drag_breakdown"]["fuselage_vtail_misc"] + 0.009
    assert 0.025 < cd0 < 0.035


def test_tip_stall_and_stability_are_caught(e3):
    from pinneapple_design.aero.aircraft import Aircraft
    x = baseline_x()
    tip = x.copy(); tip[8] = 0.40; tip[10] = 0.0                          # strong taper, no washout -> stall moves out
    fwd = x.copy(); fwd[11] = 1.95                                        # wing forward -> neutral point nears the CG
    aft = x.copy(); aft[11] = 2.75                                        # wing aft -> too stable to rotate and flare
    r_base, r_tip, r_fwd, r_aft = e3.evaluate(np.stack([x, tip, fwd, aft]), Aircraft(), Requirements3D())
    assert r_tip["stall_station"] > r_base["stall_station"] + 0.2
    assert r_fwd["static_margin"] < r_base["static_margin"] - 0.15 < r_aft["static_margin"] - 0.3
    assert "Static margin" in r_aft["violations"]


def test_3d_search(e3):
    from pinneapple_design.aero.aircraft import Aircraft
    res = e3.search(Aircraft(), Requirements3D(), population=24, generations=10, seed=3)
    D = res["designs"]
    assert res["pareto"] and any(not d.get("feasible") for d in D)
    base = D[0]["vmax_kt"]
    for i in res["pareto"]:
        d = D[i]
        assert d["feasible"] and d["static_margin"] >= 0.05 and d["stall_station"] <= 0.6 and d["span"] <= 13.5
    assert max(D[i]["vmax_kt"] for i in res["pareto"]) > base


def test_api_3d():
    sys.path.insert(0, APP)
    from fastapi.testclient import TestClient
    from aero_optimizer.api import app
    with TestClient(app) as c:
        r = c.post("/api/optimize3d", json={}).json()
        assert r["status"] == "done" and r["result"]["pareto"]
        x = r["result"]["details"][str(r["result"]["picks"]["balanced"])]["x"]
        d = c.post("/api/aircraft", json={"x": x}).json()
        assert d["lines_vlm"] and len(d["loading"]["stall_ratio"]) == len(d["loading"]["eta"]) and "ground_z" in d
        q = ",".join(map(str, x))
        assert c.get(f"/api/aircraft.glb?x={q}").content[:4] == b"glTF"
        assert c.get(f"/api/aircraft.usda?x={q}").text.startswith("#usda")
        assert c.post("/api/aircraft", json={"x": [1] * 12}).status_code == 422
        j = c.post("/api/optimize3d", json={"population": 16, "generations": 5}).json()
        assert j["status"] == "running" and c.get(f"/api/job/{j['job']}").status_code == 200

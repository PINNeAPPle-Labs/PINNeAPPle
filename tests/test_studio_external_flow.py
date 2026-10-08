"""PINNeAPPle studio (scene, exports, browser viewer, Blender renders) and the generic OpenFOAM external flow."""
from __future__ import annotations

import json
import math
import os
import struct

import numpy as np
import pytest

from pinneapple_design.geometry.bodies import _signed_volume, ahmed_body, box, cylinder, sphere
from pinneapple_tools.visualization.studio import Scene, read_stl, scalar_of, web_viewer
from pinneapple_tools.visualization.studio.scene import write_stl

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRD = os.path.join(ROOT, "apps", "interop_hub", "examples", "calculix_cantilever.frd")


def _glb_json(data: bytes) -> dict:
    n = struct.unpack("<I", data[12:16])[0]
    return json.loads(data[20:20 + n])


def test_bodies_are_closed_and_outward():
    assert abs(_signed_volume(*sphere(0.5, n=64)) - 4 / 3 * math.pi * 0.125) < 0.01      # 4/3 pi r^3
    assert abs(_signed_volume(*cylinder(0.5, 2.0, n=64)) - math.pi * 0.25 * 2) < 0.01
    assert abs(_signed_volume(*box((1, 2, 3))) - 6.0) < 1e-12
    V, F = ahmed_body(25)
    assert np.allclose(V.max(0) - V.min(0), [1.044, 0.389, 0.288], atol=1e-6) and V[:, 2].min() == pytest.approx(0.05)
    assert 0.105 < _signed_volume(V, F) < 0.117                                    # box 0.117 minus nose and slant


def test_scene_fields_and_exports(tmp_path):
    V, F = sphere(1.0, n=32)
    sc = Scene.from_arrays(V, F, name="ball")
    pts = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
    sc.map_field("cp", pts, [1.0, -1.0], k=1, label="Cp")
    cp = sc.surfaces[0].fields["cp"]
    assert cp[np.argmax(V[:, 0])] == pytest.approx(1.0) and cp[np.argmin(V[:, 0])] == pytest.approx(-1.0)
    gl = _glb_json(sc.to_glb())
    assert "_CP" in gl["meshes"][0]["primitives"][0]["attributes"]
    assert "primvars:cp" in sc.to_usda()
    p = tmp_path / "ball.stl"
    p.write_bytes(write_stl(sc.surfaces))
    name, V2, F2 = read_stl(str(p))[0]
    assert len(F2) == len(F) and abs(_signed_volume(V2, F2) - _signed_volume(V, F)) < 1e-6   # vertices merged back
    sc.add_lines([np.c_[np.linspace(-2, 2, 20), np.zeros(20), np.zeros(20)]], [np.linspace(0, 1, 20)])
    g = np.ones((4, 5))
    g[1, 1] = np.nan
    sc.add_slice("plane", [-2, 0, -1], [4, 0, 0], [0, 0, 2], g)
    ex = sc.extras()
    assert ex["fields"] == ["cp"] and ex["slices"][0]["grid"][1][1] is None and len(ex["lines"][0]["points"][0]) == 20
    idx = web_viewer(sc, str(tmp_path / "viewer"))
    for f in ("index.html", "viewer.js", "scene.glb", "scene.json", "vendor/three/three.module.min.js"):
        assert (tmp_path / "viewer" / f).exists(), f


def test_scene_from_calculix_result_is_von_mises():
    sc = Scene.from_file(FRD)
    s = sc.surfaces[0]
    assert {"DISP", "STRESS"} <= set(s.fields)
    assert s.fields["DISP"].min() >= 0 and s.fields["DISP"].max() == pytest.approx(7.58, abs=0.01)   # tip deflection
    t = np.array([[100.0, 0, 0, 0, 0, 0]])
    assert scalar_of(t)[0] == pytest.approx(100.0)                                   # uniaxial: von Mises = sigma


def test_aero_exports_use_the_generic_writer():
    from pinneapple_design.aero.airframe import Airframe, to_glb
    gl = _glb_json(to_glb(Airframe().build("low")))
    assert any(n.get("name") == "propeller" and n.get("children") for n in gl["nodes"])


def test_external_flow_case_files(tmp_path):
    from pinneapple_simulation.numerical_solvers.external_flow import ExternalFlow, flow_directions
    flow = ExternalFlow({"ahmed": ahmed_body()}, speed=40.0, ground=0.0, half_model=True, resolution="coarse")
    info = flow.write(str(tmp_path / "case"), procs=2)
    c = tmp_path / "case"
    assert (c / "constant/triSurface/ahmed.stl").exists()
    bm = (c / "system/blockMeshDict").read_text()
    assert "symmetryPlane" in bm and "ground { type wall" in bm
    u = (c / "0/U").read_text()
    assert "ground" in u and "(40.0 0.0 0.0)" in u                                 # the road moves with the flow
    assert "locationInMesh" in (c / "system/snappyHexMeshDict").read_text()
    assert info["ref_area"] == pytest.approx(0.389 * 0.288) and info["Re"] == pytest.approx(40 * 1.044 / 1.5e-5)
    d, lift, side = flow_directions(5.0, 0.0)
    assert d @ lift == pytest.approx(0) and np.allclose(np.cross(lift, d), side)


@pytest.mark.skipif(os.environ.get("PINNEAPPLE_RUN_OPENFOAM") != "1", reason="set PINNEAPPLE_RUN_OPENFOAM=1 (needs OpenFOAM, ~3 min)")
def test_external_flow_runs_on_a_sphere(tmp_path):
    from pinneapple_simulation.numerical_solvers.external_flow import ExternalFlow, openfoam_available
    if not openfoam_available():
        pytest.skip("OpenFOAM not found")
    res = ExternalFlow({"sphere": sphere(0.5, n=48)}, speed=10.0, resolution="coarse", iterations=300,
                       ref_area=math.pi * 0.25).solve(str(tmp_path / "s"), procs=2, log=lambda s: None)
    assert 0.05 < res.coefficients["CD"] < 0.8 and abs(res.coefficients["CL"]) < 0.1
    sc = res.to_scene()
    assert {"cp", "cf"} <= set(sc.field_names()) and sc.lines and sc.slices
    assert sc.field_range("cp")[1] > 0.8                                            # stagnation, Cp -> 1


def test_blender_render(tmp_path):
    pytest.importorskip("bpy")
    from pinneapple_tools.visualization.studio import render
    V, F = sphere(1.0, n=24)
    sc = Scene.from_arrays(V, F)
    sc.map_field("z", V, V[:, 2])
    out = render(sc, str(tmp_path / "s.png"), field="z", samples=1, size=(96, 54))
    assert os.path.getsize(out) > 1000


def test_gltf_round_trip_keeps_fields_and_transforms(tmp_path):
    V, F = sphere(1.0, n=24)
    sc = Scene.from_arrays(V, F, name="ball")
    sc.surfaces[0].fields["Cp_wall"] = V[:, 2].copy()
    p = tmp_path / "b.glb"
    sc.save(str(p))
    s2 = Scene.from_file(str(p)).surfaces[0]
    assert s2.name == "ball" and np.allclose(s2.vertices, V, atol=1e-6) and len(s2.faces) == len(F)
    assert np.allclose(s2.fields["Cp_wall"], V[:, 2], atol=1e-6)                 # original name, not "_CP_WALL"
    gl = _glb_json(p.read_bytes())                                              # a node translation is applied
    gl["nodes"][0]["translation"] = [1.0, 2.0, 3.0]
    js = json.dumps(gl).encode()
    js += b" " * (-len(js) % 4)
    raw = p.read_bytes()
    rest = raw[20 + struct.unpack("<I", raw[12:16])[0]:]
    q = tmp_path / "moved.glb"
    q.write_bytes(b"glTF" + struct.pack("<II", 2, 20 + len(js) + len(rest)) + struct.pack("<I", len(js)) + b"JSON"
                  + js + rest)
    s3 = Scene.from_file(str(q)).surfaces[0]
    assert np.allclose(s3.vertices - V, [1.0, -3.0, 2.0], atol=1e-6)            # glTF y-up -> scene z-up


def test_vtp_input(tmp_path):
    pytest.importorskip("vtk")
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter
    V, F = box((1, 2, 3))
    pd = vtkPolyData()
    pts = vtkPoints()
    pts.SetData(numpy_to_vtk(V))
    pd.SetPoints(pts)
    cells = vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0, 3 * len(F) + 1, 3)), numpy_to_vtkIdTypeArray(F.ravel()))
    pd.SetPolys(cells)
    u = numpy_to_vtk(np.c_[V[:, 0], np.zeros(len(V)), np.zeros(len(V))])
    u.SetName("U")
    pd.GetPointData().AddArray(u)
    w = vtkXMLPolyDataWriter()
    w.SetFileName(str(tmp_path / "b.vtp"))
    w.SetInputData(pd)
    w.Write()
    s = Scene.from_file(str(tmp_path / "b.vtp")).surfaces[0]
    assert len(s.faces) == len(F) and np.allclose(s.fields["U"], np.abs(V[:, 0]))    # vectors -> magnitude


def test_colorbar_ticks_and_pixels(tmp_path):
    from PIL import Image
    from pinneapple_tools.visualization.studio.colormap import colorbar, decimals, jet, ticks
    assert ticks(-0.5, 0.597) == ["-0.50", "-0.23", "0.05", "0.32", "0.60"]
    assert ticks(0, 2e5) == ["0", "5.00e+4", "1.00e+5", "1.50e+5", "2.00e+5"]
    assert decimals(0, 1) == 2 and decimals(0, 1000) == 0 and ticks(0, 1, nd=1)[2] == "0.5"
    assert [round(255 * c) for c in jet(0)] == [0, 0, 143] and [round(255 * c) for c in jet(2)] == [128, 0, 0]
    p = tmp_path / "white.png"
    Image.new("RGB", (1920, 1080), "white").save(p)
    colorbar(str(p), "Cp", -1, 1, lo_txt="suction", hi_txt="stagnation")
    im = np.asarray(Image.open(p).convert("RGB")).astype(int)
    row = im[1080 - 150 - 22 + 40 + 11]                            # middle of the bar (layout of colorbar at s = 1)
    lit = np.where(np.abs(row - 255).sum(1) > 60)[0]
    assert len(lit) > 500                                            # a 520 px bar in the bottom-right corner
    assert lit.min() > 1920 / 2 and abs(row[lit.min()] - [0, 0, 143]).max() < 12 and abs(row[lit.max()] - [128, 0, 0]).max() < 12


def test_studio_core_js_matches_python(tmp_path):
    """Runs the node tests of studio-core.js and checks its ticks against the Python colour bar's."""
    import shutil
    import subprocess
    from pinneapple_tools.visualization.studio.colormap import ticks
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    core = os.path.join(ROOT, "pinneapple_tools", "visualization", "studio", "web", "studio-core.js")
    shutil.copy(core, tmp_path / "studio-core.mjs")                 # .mjs: ES module on every node version
    src = open(os.path.join(ROOT, "tests", "js", "studio_core.test.mjs")).read()
    (tmp_path / "t.test.mjs").write_text(src.replace("../../pinneapple_tools/visualization/studio/web/studio-core.js", "./studio-core.mjs"))
    r = subprocess.run([node, "--test", str(tmp_path / "t.test.mjs")], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    cases = [(-0.5, 0.597), (0, 2e5), (0, 1e-4), (-3.2, 41.0), (101325, 101400), (0.001, 0.0042)]
    js = "import * as c from './studio-core.mjs'; console.log(JSON.stringify(%s.map(([a, b]) => c.ticks(a, b))))" % json.dumps(cases)
    (tmp_path / "t.mjs").write_text(js)
    out = subprocess.run([node, str(tmp_path / "t.mjs")], capture_output=True, text=True, timeout=60, cwd=tmp_path)
    assert json.loads(out.stdout) == [ticks(a, b) for a, b in cases]


def test_decimate_keeps_shape_and_fields(tmp_path):
    V, F = sphere(1.0, n=200)
    sc = Scene.from_arrays(V, F, name="ball")
    sc.surfaces[0].fields["z"] = V[:, 2].copy()
    web_viewer(sc, str(tmp_path / "v"), max_faces=5000)
    assert len(sc.surfaces[0].faces) == len(F)                         # the scene itself is untouched
    s2 = Scene.from_file(str(tmp_path / "v" / "scene.glb")).surfaces[0]
    assert 4000 < len(s2.faces) <= 5500
    assert abs(_signed_volume(s2.vertices, s2.faces) - 4 / 3 * math.pi) < 0.1
    assert np.allclose(s2.fields["z"], s2.vertices[:, 2], atol=1e-5)  # linear fields survive averaging exactly
    sc.decimate(2000)
    assert sc.n_faces() <= 2200


def test_slice_groups_in_viewer_json():
    sc = Scene.from_arrays(*box((1, 1, 1)))
    for x in (1.0, 2.0):
        sc.add_slice(f"wake {x}: speed", [x, 0, 0], [0, 1, 0], [0, 0, 1], np.ones((2, 2)), group="wake: speed")
    sc.add_slice("mid", [0, 0, 0], [1, 0, 0], [0, 0, 1], np.ones((2, 2)))
    assert [s["group"] for s in sc.extras()["slices"]] == ["wake: speed", "wake: speed", "mid"]

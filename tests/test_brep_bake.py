"""Bake exact bodies to mesh files, MeshData and a quality report (#1118)."""
import dataclasses

import numpy as np
import pytest

sm = pytest.importorskip("smlibpy")
pytest.importorskip("mapbox_earcut")

from pinneapple_design.geometry.brep.bake import (  # noqa: E402
    bake,
    mesh_quality,
    self_intersections,
    to_meshdata,
    volume_mesh,
)
from pinneapple_design.geometry.brep.step import _body  # noqa: E402
from pinneapple_design.geometry.brep.tessellate import tessellate  # noqa: E402


def body(shape, name=""):
    b = _body(shape, 0, "smlibpy")
    return dataclasses.replace(b, name=name) if name else b


def plate(nx=5, ny=3, X=120.0, Y=70.0, r=3.0):
    p = sm.Box(X, Y, 5)
    for i in range(nx):
        for j in range(ny):
            p = p - sm.Cylinder(r, 5).moved(sm.Pos(-X / 2 + X / (nx + 1) * (i + 1), -Y / 2 + Y / (ny + 1) * (j + 1), 0))
    return body(p, "plate")


def _volume(V, F):
    V = np.asarray(V, float)
    return float(np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6)


@pytest.mark.parametrize("ext", [".stl", ".obj", ".glb", ".gltf"])
def test_round_trip_keeps_counts_and_volume(tmp_path, ext):
    trimesh = pytest.importorskip("trimesh")
    b = plate()
    m = tessellate(b, "normal")
    info = bake(m, str(tmp_path / f"p{ext}"))
    unit = 1e3 if ext in (".stl", ".obj") else 1.0                        # STL / OBJ default to millimetres
    t = trimesh.load(info["path"], force="mesh", process=False)
    assert len(t.faces) == m.n_triangles == info["n_triangles"]
    if ext != ".stl":                                                      # STL stores triangles, not shared vertices
        assert len(t.vertices) == len(m.vertices)
    assert abs(_volume(t.vertices, np.asarray(t.faces)) / unit**3 / m.volume() - 1) < 1e-6
    assert info["unit"] == ("mm" if unit == 1e3 else "m")


def test_usd_text_and_binary(tmp_path):
    pxr = pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom
    assert pxr
    sp, cy = body(sm.Sphere(10), "ball"), body(sm.Cylinder(5, 20).moved(sm.Pos(30, 0, 0)), "rod")
    for ext in (".usda", ".usdc"):
        info = bake([sp, cy], str(tmp_path / f"parts{ext}"), quality="normal")
        st = Usd.Stage.Open(info["path"])
        assert UsdGeom.GetStageMetersPerUnit(st) == 1.0
        prims = {p.GetName(): UsdGeom.Mesh(p) for p in st.Traverse() if p.IsA(UsdGeom.Mesh)}
        assert set(prims) == {"ball", "rod"}
        ball = prims["ball"]
        P = np.asarray(ball.GetPointsAttr().Get())
        F = np.asarray(ball.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        assert abs(_volume(P, F) / tessellate(sp, "normal").volume() - 1) < 1e-6      # point3f is float32


def test_names_colours_and_units(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    a, c = body(sm.Box(10, 10, 10), "cube"), body(sm.Sphere(5).moved(sm.Pos(20, 0, 0)), "ball")
    ma = tessellate(a, "normal")
    h = np.linspace(0, 1, len(ma.vertices))
    info = bake([a, c], str(tmp_path / "two.glb"), colors=[h, None])
    scene = trimesh.load(info["path"], process=False)
    assert set(scene.geometry) == {"cube", "ball"} and info["bodies"] == ["cube", "ball"]
    assert scene.geometry["cube"].visual.vertex_colors.shape[0] == len(ma.vertices)
    obj = (tmp_path / "two.obj")
    bake([a, c], str(obj), colors=[h, None], unit="inch")
    text = obj.read_text()
    assert "o cube" in text and "o ball" in text
    first = next(ln for ln in text.splitlines() if ln.startswith("v "))
    assert len(first.split()) == 7                                         # x y z r g b
    xs = [float(ln.split()[1]) for ln in text.splitlines() if ln.startswith("v ")]
    assert abs(min(xs) + 5e-3 / 0.0254) < 1e-8                             # the cube's corner, x = -5 mm, in inch
    with pytest.raises(ValueError, match="no colours"):
        bake(a, str(tmp_path / "x.stl"), colors=h)
    with pytest.raises(ValueError, match="metres"):
        bake(a, str(tmp_path / "x.glb"), unit="mm")
    with pytest.raises(ValueError, match="unknown mesh format"):
        bake(a, str(tmp_path / "x.ply"))


def test_baked_volume_converges_to_the_exact_volume(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    b = body(sm.Cylinder(20, 40) - sm.Cylinder(8, 40) + sm.Sphere(12).moved(sm.Pos(0, 0, 40)))
    errs = []
    for q in ("draft", "normal", "fine"):
        t = trimesh.load(bake(b, str(tmp_path / f"{q}.stl"), quality=q, unit="m")["path"], force="mesh")
        errs.append(abs(abs(t.volume) / b.volume_m3 - 1))
    assert errs[0] > errs[1] > errs[2] and errs[2] < 2e-3


def test_meshdata_bridge_samples_the_surface():
    b = plate()
    md = to_meshdata(b, quality="fine")
    assert md.n_faces == tessellate(b, "fine").n_triangles and md.normals.shape == (md.n_faces, 3)
    lo, hi = md.bounds()
    assert np.allclose(hi - lo, b.size_m, atol=1e-9)


def test_quality_report_on_a_plate():
    b = plate(10, 5, 200, 100)
    for mesher in ("fast", "kernel"):
        m = tessellate(b, "normal", mesher)
        r = mesh_quality(m, b)
        assert r["watertight"] and r["euler_characteristic"] == 2 - 2 * 50 and r["self_intersections"] == 0
        assert r["min_angle_deg"] > 0                       # no zero-area triangle (holes in a row are collinear)
        assert r["chordal_error_over_tolerance"] <= 1.05    # measured against the face each triangle belongs to
        assert abs(r["volume_error_rel"]) < 2e-3 and abs(r["area_error_rel"]) < 2e-3


def test_self_intersections_of_two_overlapping_spheres():
    s = tessellate(body(sm.Sphere(10)), "normal", "fast")
    assert self_intersections(s) == 0
    V = np.vstack([s.vertices, s.vertices + [5e-3, 0, 0]])
    T = np.vstack([s.triangles, s.triangles + len(s.vertices)])
    two = dataclasses.replace(s, vertices=V, triangles=T, face_id=np.r_[s.face_id, s.face_id])
    assert self_intersections(two) > 0


def test_a_baked_part_runs_through_the_immersed_heat_solver(tmp_path):
    """A STEP sphere, baked to STL and read by the immersed solver, against the closed form of uniform generation with
    convection (Incropera 3.5), as tests/test_immersed.py does for an icosphere."""
    from pinneapple_simulation import immersed as im
    from pinneapple_simulation.immersed import verification as V
    from pinneapple_simulation.immersed.geometry import read_stl

    case = V.sphere_convection()
    c = V._shift(3) * 1e3                                                   # to mm
    b = body(sm.Sphere(500).moved(sm.Pos(*c)))
    stl = bake(b, str(tmp_path / "ball.stl"), quality="fine", unit="m")["path"]
    geo = im.STLGeometry(*read_stl(stl))
    sol = im.solve_heat(im.Discretization(geo, 0.125, order=1), case.conductivity, case.source, case.bcs)
    e = sol.errors(case.exact, case.grad)
    assert e["L2_rel"] < 3e-3 and abs(sol.balance["imbalance"]) < 1e-10


def test_volume_mesh_with_gmsh():
    try:
        import gmsh  # noqa: F401
    except (ImportError, OSError):                       # not installed, or its system libraries (libGLU) are missing
        pytest.skip("gmsh is not usable here")
    b = body(sm.Box(40, 20, 10) - sm.Cylinder(4, 10))
    r = volume_mesh(b, h=3e-3, quality="normal")
    assert r["dim"] == 3 and r["n_elements"] > 0
    assert abs(r["volume"] / b.volume_m3 - 1) < 0.02

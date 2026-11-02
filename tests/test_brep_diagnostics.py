"""Surface-quality diagnostics (#1117): exact and discrete curvature, G0/G1/G2 edges, zebra, highlight lines."""
from collections import Counter

import numpy as np
import pytest

sm = pytest.importorskip("smlibpy")
pytest.importorskip("mapbox_earcut")

from pinneapple_design.geometry.brep import diagnostics as D  # noqa: E402
from pinneapple_design.geometry.brep.diagnostics import view_direction  # noqa: E402
from pinneapple_design.geometry.brep.step import _body  # noqa: E402
from pinneapple_design.geometry.brep.tessellate import tessellate  # noqa: E402


def body(shape):
    return _body(shape, 0, "smlibpy")


def vertical_edges(box, h):
    return [e for e in box.edges() if abs(e.bbox().max.z - e.bbox().min.z - h) < 1e-9]


def test_sphere_curvature_exact_and_discrete():
    R = 10e-3
    b = body(sm.Sphere(10))
    m = tessellate(b, "fine", "kernel")
    c = D.exact_curvature(b, m)
    assert np.allclose(c["K"], 1 / R**2, rtol=1e-9) and np.allclose(c["H"], 1 / R, rtol=1e-9)
    radial = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    assert np.allclose((c["normal"] * radial).sum(1), 1.0, atol=1e-9)               # outward
    d = D.discrete_curvature(m)
    assert abs(np.median(d["K"]) * R**2 - 1) < 0.01 and abs(np.median(d["H"]) * R - 1) < 0.01


def test_cylinder_and_plane_curvature():
    r = 5e-3
    b = body(sm.Cylinder(5, 30))
    m = D.refine(tessellate(b, "fine", "fast"), 2e-3)          # caps get interior vertices
    c = D.exact_curvature(b, m)
    kind = np.array([b.faces[i].kind for i in c["vertex_face"]])
    side = kind == "cylinder"
    assert side.any() and (~side).any()
    assert np.allclose(c["H"][side], 1 / (2 * r), rtol=1e-9) and np.allclose(c["K"][side], 0, atol=1e-6)
    assert np.allclose(c["H"][~side], 0, atol=1e-9) and np.allclose(c["K"][~side], 0, atol=1e-9)


def test_torus_curvature_matches_the_closed_form():
    Rm, r = 20e-3, 5e-3
    b = body(sm.Torus(20, 5))
    m = tessellate(b, "normal", "kernel")
    c = D.exact_curvature(b, m)
    x, y, z = m.vertices.T
    th = np.arctan2(z, np.hypot(x, y) - Rm)                  # angle around the tube, 0 on the outer equator
    K = np.cos(th) / (r * (Rm + r * np.cos(th)))
    H = (Rm + 2 * r * np.cos(th)) / (2 * r * (Rm + r * np.cos(th)))
    assert np.allclose(c["K"], K, rtol=1e-6, atol=1e-6) and np.allclose(c["H"], H, rtol=1e-6)
    assert c["K"].min() < 0 < c["K"].max()                    # saddle inside, convex outside


def test_edge_continuity_g0_g1_g2():
    box = sm.Box(40, 30, 20)
    sharp = Counter(e["level"] for e in D.edge_continuity(body(box)))
    assert sharp == {"G0": 12}
    v = vertical_edges(box, 20)
    g1 = D.edge_continuity(body(box.fillet(v, 5)))
    g2 = D.edge_continuity(body(box.fillet(v, 5, continuity=2)))
    assert Counter(e["level"] for e in g1) == {"G0": 16, "G1": 8}
    assert Counter(e["level"] for e in g2) == {"G0": 16, "G2": 8}
    t1 = [e for e in g1 if e["level"] == "G1"]
    assert all(e["angle_deg"] < 0.5 and e["gap_m"] < 1e-9 for e in t1)
    # a circular fillet meets the flat side with a curvature jump of 1/r = 200 1/m
    assert all(abs(e["curvature_jump_per_m"] - 200.0) < 2.0 for e in t1)
    assert all(e["curvature_jump_per_m"] < 20.0 for e in g2 if e["level"] == "G2")
    assert all(len(e["faces"]) == 2 for e in g1)


def test_refine_keeps_the_mesh_and_its_face_ids():
    b = body(sm.Box(30, 20, 10) - sm.Cylinder(4, 10))
    m = tessellate(b, "normal", "fast")
    f = D.refine(m, 2e-3)
    assert f.is_watertight() and f.euler_characteristic() == m.euler_characteristic()
    assert np.allclose(f.face_areas(), m.face_areas(), rtol=1e-12)
    L = np.linalg.norm(f.vertices[f.triangles[:, [1, 2, 0]]] - f.vertices[f.triangles], axis=2)
    assert L.max() <= 2e-3 and f.n_triangles > m.n_triangles
    assert D.refine(m, 1e-9, max_triangles=4 * m.n_triangles).n_triangles == 4 * m.n_triangles


def test_highlight_lines_are_straight_on_a_plane_parallel_to_the_light_plane():
    b = body(sm.Box(40, 40, 2))
    m = D.refine(tessellate(b, "normal", "fast"), 1e-3)
    c = D.exact_curvature(b, m)
    h = D.highlight_lines(m, c["normal"], direction=(0, 1, 0), plane_normal=(0, 0, 1))
    top = c["normal"][:, 2] > 0.999
    xs = np.round(m.vertices[top, 0], 9)
    groups = [h[top][xs == x] for x in np.unique(xs)]
    assert max(g.std() for g in groups) < 1e-12                  # lines along y: lit value depends on x only
    assert h[top].max() > 0.9 and h[top].min() < 0.1             # and both lit and dark lines are on the face


def test_zebra_on_a_plane_with_a_distant_eye_is_uniform_and_on_a_sphere_varies():
    b = body(sm.Box(40, 40, 2))
    m = D.refine(tessellate(b, "normal", "fast"), 2e-3)
    c = D.exact_curvature(b, m)
    top = c["normal"][:, 2] > 0.999
    z = D.zebra(m, c["normal"], eye=(1e5, 2e5, 1e6), axis=(0, 1, 0))
    assert np.ptp(z[top]) < 1e-3                                  # parallel view rays, one normal: one stripe
    assert np.ptp(D.zebra(m, c["normal"], view=(0.3, 0.2, 1.0))[top]) < 1e-12        # a view at infinity: exact
    s = body(sm.Sphere(10))
    ms = tessellate(s, "fine", "fast")
    zs = D.zebra(ms, D.exact_curvature(s, ms)["normal"], n_stripes=8)
    assert zs.min() < 0.05 and zs.max() > 0.95


def test_render_writes_a_png(tmp_path):
    b = body(sm.Cylinder(5, 10))
    m = tessellate(b, "normal", "fast")
    c = D.exact_curvature(b, m)
    p = D.render(m, c["H"], str(tmp_path / "h.png"), colorbar="H [1/m]", shade=0.3)
    assert open(p, "rb").read(8) == b"\x89PNG\r\n\x1a\n"


def test_view_direction_literal_call():
    """Plain-number call (also the Studio's smoke input)."""
    assert np.allclose(view_direction(90.0, 0.0), [0.0, 0.0, 1.0])

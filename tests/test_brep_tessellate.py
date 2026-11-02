"""Tessellation of exact bodies (#1116): presets, fast analytic mesher, kernel fallback, error bounds."""
import math

import numpy as np
import pytest

sm = pytest.importorskip("smlibpy")
pytest.importorskip("mapbox_earcut")

from pinneapple_design.geometry.brep.step import _body  # noqa: E402
from pinneapple_design.geometry.brep.tessellate import (  # noqa: E402
    QUALITY,
    circle_segments,
    tessellate,
)


def body(shape):
    return _body(shape, 0, "smlibpy")


def plate_with_holes(nx=5, ny=3, r=3.0):
    p = sm.Box(120, 70, 5)
    for i in range(nx):
        for j in range(ny):
            p = p - sm.Cylinder(r, 5).moved(sm.Pos(-48 + 24 * i, -20 + 20 * j, 0))
    return p


def test_circle_segments_meets_the_sagitta_and_angle_bounds():
    for r, tol, ang in ((10.0, 0.01, 15.0), (1.0, 0.001, 30.0), (50.0, 0.5, 7.5)):
        n = circle_segments(r, 2 * math.pi, tol, ang)
        assert r * (1 - math.cos(math.pi / n)) <= tol + 1e-15
        assert 360.0 / n <= ang + 1e-12
        assert r * (1 - math.cos(math.pi / (n - 1))) > tol or 360.0 / (n - 1) > ang    # and it is the smallest


def test_fast_mesher_is_watertight_with_the_right_genus_and_faster():
    b = body(plate_with_holes())
    fast = tessellate(b, "normal", "fast")
    kern = tessellate(b, "normal", "kernel")
    for m in (fast, kern):
        assert m.is_watertight()
        assert m.euler_characteristic() == 2 - 2 * 15             # 15 through-holes: genus 15
    assert set(np.unique(fast.face_id)) == set(range(len(b.faces)))
    assert fast.seconds < kern.seconds


def test_chord_error_of_holes_within_the_preset_tolerance():
    r = 3.0
    b = body(plate_with_holes(r=r))
    for q in QUALITY:
        m = tessellate(b, q, "fast")
        tol = m.chord_tolerance_m
        cyl = [f.index for f in b.faces if f.kind == "cylinder"]
        V, T = m.vertices, m.triangles[np.isin(m.face_id, cyl)]
        mids = np.vstack([(V[T[:, 0]] + V[T[:, 1]]) / 2, (V[T[:, 1]] + V[T[:, 2]]) / 2, (V[T[:, 2]] + V[T[:, 0]]) / 2])
        centres = np.array([[-48 + 24 * i, -20 + 20 * j] for i in range(5) for j in range(3)]) * 1e-3
        d = np.min(np.linalg.norm(mids[:, None, :2] - centres[None], axis=2), axis=1)
        assert np.max(r * 1e-3 - d) <= tol * (1 + 1e-6), q


def test_volume_and_area_converge_with_quality():
    b = body(sm.Cylinder(20, 40) - sm.Cylinder(8, 40))                  # a tube: two bands, two annuli
    errs = []
    for q in ("draft", "normal", "fine"):
        m = tessellate(b, q, "fast")
        assert m.is_watertight()
        errs.append(abs(m.volume() / b.volume_m3 - 1))
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 1e-3


def test_cone_and_sphere_by_the_fast_mesher():
    fr = body(sm.Cone(20, 8, 30))
    m = tessellate(fr, "fine", "fast")
    assert m.mesher == "fast" and m.is_watertight() and m.euler_characteristic() == 2
    assert abs(m.volume() / fr.volume_m3 - 1) < 2e-3
    sp = body(sm.Sphere(15))
    ms = tessellate(sp, "fine", "fast")
    assert ms.mesher == "fast" and ms.is_watertight() and ms.euler_characteristic() == 2
    assert abs(ms.volume() / sp.volume_m3 - 1) < 2e-3
    r = np.linalg.norm(ms.vertices, axis=1)
    assert np.allclose(r, 15e-3, rtol=1e-12)                              # vertices on the exact sphere


def test_auto_falls_back_to_the_kernel_and_says_why():
    tor = body(sm.Torus(20, 5))
    m = tessellate(tor, "normal", "auto")
    assert m.mesher == "kernel" and "torus" in m.reason and m.fallback_faces
    assert m.is_watertight() and m.euler_characteristic() == 0          # a torus: genus 1
    with pytest.raises(ValueError, match="fast mesher"):
        tessellate(tor, "normal", "fast")
    p = tessellate(body(plate_with_holes(2, 1)), "normal", "auto")
    assert p.mesher == "fast" and not p.fallback_faces


def test_face_ids_of_the_kernel_mesh_add_up_to_the_face_areas():
    b = body(sm.Box(30, 20, 10) - sm.Cylinder(4, 10))
    m = tessellate(b, "fine", "kernel")
    exact = np.array([f.area_m2 for f in b.faces])
    assert np.allclose(m.face_areas(), exact, rtol=5e-3)
    r = m.report()
    assert r["watertight"] and r["triangles"] == m.n_triangles


def test_circle_segments_literal_call():
    """Plain-number call (also the Studio's smoke input)."""
    assert circle_segments(10.0, 6.283185307179586, 0.01, 15.0) == 71

"""Exact B-rep import of STEP files (#1115). The STEP files are generated here by the kernel itself."""
import math

import pytest

sm = pytest.importorskip("smlibpy")

from pinneapple_design.geometry.brep import available_backends, read_step, write_step  # noqa: E402


def _roundtrip(shape, tmp_path, name="part", unit="mm"):
    p = tmp_path / f"{name}.step"
    write_step(shape, p, unit=unit)
    bodies = read_step(p)
    assert len(bodies) == 1
    return bodies[0]


def test_backend_is_reported():
    assert "smlibpy" in available_backends()


def test_plate_with_holes_volume_area_and_face_types(tmp_path):
    a, b, t, r = 80.0, 50.0, 6.0, 4.0                      # mm
    plate = sm.Box(a, b, t)
    for x, y in ((-25, -12), (25, -12), (-25, 12), (25, 12)):
        plate = plate - sm.Cylinder(r, t).moved(sm.Pos(x, y, 0))
    body = _roundtrip(plate, tmp_path, "plate")
    vol = (a * b * t - 4 * math.pi * r * r * t) * 1e-9
    area = (2 * (a * b - 4 * math.pi * r * r) + 2 * (a + b) * t + 4 * 2 * math.pi * r * t) * 1e-6
    assert math.isclose(body.volume_m3, vol, rel_tol=1e-9)
    assert math.isclose(body.area_m2, area, rel_tol=1e-9)
    assert body.face_counts() == {"plane": 6, "cylinder": 4}
    assert body.is_solid and body.valid
    cyl = [f for f in body.faces if f.kind == "cylinder"]
    assert all(math.isclose(f.params["radius_m"], r * 1e-3, rel_tol=1e-12) for f in cyl)
    assert all(abs(abs(f.params["axis_direction"][2]) - 1) < 1e-12 for f in cyl)


def test_analytic_solids_match_closed_forms(tmp_path):
    R, r, h = 20.0, 5.0, 30.0
    cases = {
        "sphere": (sm.Sphere(R), 4 / 3 * math.pi * R**3, 4 * math.pi * R**2, "sphere"),
        "frustum": (sm.Cone(R, r, h), math.pi * h / 3 * (R * R + R * r + r * r),
                    math.pi * (R + r) * math.hypot(R - r, h) + math.pi * (R * R + r * r), "cone"),
        "torus": (sm.Torus(R, r), 2 * math.pi**2 * R * r * r, 4 * math.pi**2 * R * r, "torus"),
    }
    for name, (shape, V, A, kind) in cases.items():
        body = _roundtrip(shape, tmp_path, name)
        assert math.isclose(body.volume_m3, V * 1e-9, rel_tol=1e-6), name
        assert math.isclose(body.area_m2, A * 1e-6, rel_tol=1e-6), name
        assert kind in body.face_counts(), name
    cone = [f for f in _roundtrip(sm.Cone(R, r, h), tmp_path, "c2").faces if f.kind == "cone"][0]
    assert math.isclose(cone.params["half_angle_deg"], math.degrees(math.atan((R - r) / h)), rel_tol=1e-9)
    tor = [f for f in _roundtrip(sm.Torus(R, r), tmp_path, "t2").faces if f.kind == "torus"][0]
    assert math.isclose(tor.params["major_radius_m"], R * 1e-3, rel_tol=1e-9)
    assert math.isclose(tor.params["minor_radius_m"], r * 1e-3, rel_tol=1e-9)


def test_filleted_block_volume_and_area(tmp_path):
    a, b, h, rf = 40.0, 30.0, 20.0, 5.0
    blk = sm.Box(a, b, h)
    vertical = [e for e in blk.edges() if abs(e.bbox().max.z - e.bbox().min.z - h) < 1e-9]
    assert len(vertical) == 4
    body = _roundtrip(blk.fillet(vertical, rf), tmp_path, "fillet")
    V = (a * b - (4 - math.pi) * rf * rf) * h
    A = 2 * (a * b - (4 - math.pi) * rf * rf) + h * (2 * (a + b) - 8 * rf + 2 * math.pi * rf)
    assert math.isclose(body.volume_m3, V * 1e-9, rel_tol=1e-9)
    assert math.isclose(body.area_m2, A * 1e-6, rel_tol=1e-9)
    assert body.face_counts() == {"plane": 6, "cylinder": 4}


def test_units_mm_and_inch_give_the_same_body_in_metres(tmp_path):
    part = sm.Box(40, 30, 10) - sm.Cylinder(5, 10)
    mm = _roundtrip(part, tmp_path, "mm", unit="mm")
    inch = _roundtrip(part, tmp_path, "inch", unit="inch")
    assert math.isclose(mm.volume_m3, inch.volume_m3, rel_tol=1e-9)
    assert math.isclose(mm.area_m2, inch.area_m2, rel_tol=1e-9)
    assert all(math.isclose(x, y, rel_tol=1e-9) for x, y in zip(mm.size_m, inch.size_m, strict=True))
    assert all(math.isclose(x, y, rel_tol=1e-12) for x, y in zip(mm.size_m, (0.04, 0.03, 0.01), strict=True))


def test_several_bodies_keep_their_names(tmp_path):
    p = tmp_path / "two.step"
    write_step([sm.Box(10, 10, 10).named("cube"), sm.Sphere(5).moved(sm.Pos(30, 0, 0)).named("ball")], p)
    bodies = read_step(p)
    assert sorted(b.name for b in bodies) == ["ball", "cube"]
    vol = {b.name: b.volume_m3 for b in bodies}
    assert math.isclose(vol["cube"], 1e-6, rel_tol=1e-9)
    assert math.isclose(vol["ball"], 4 / 3 * math.pi * 125 * 1e-9, rel_tol=1e-6)
    assert bodies[0].summary()["faces"] == len(bodies[0].faces)


def test_freeform_face_is_reported_as_such(tmp_path):
    loft = sm.loft([sm.Rectangle(20, 20), sm.Circle(8).moved(sm.Pos(0, 0, 25))])
    body = _roundtrip(loft, tmp_path, "loft")
    kinds = set(body.face_counts())
    assert kinds - {"plane", "cylinder", "cone", "sphere", "torus"}   # at least one free-form surface
    assert body.volume_m3 > 0 and body.valid

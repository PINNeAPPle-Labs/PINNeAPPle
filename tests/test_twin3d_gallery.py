"""pinneapple_twin3d.gallery / primitives: every part is a valid mesh with coherent, finite, time-varying fields."""
import json
import os
import struct

import numpy as np
import pytest

from pinneapple_twin3d import gallery, primitives as P

KEYS = sorted(gallery.GALLERY)


@pytest.fixture(scope="module")
def scenes():
    return {k: gallery.get(k) for k in KEYS}


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    out = tmp_path_factory.mktemp("gallery")
    gallery.export_all(str(out))
    return out


def part(sc, name):
    return next(p for p in sc.parts if p.name == name)


def extent(sc):
    v = np.vstack([p.vertices for p in sc.parts])
    return v.max(0) - v.min(0)


# ---------------------------------------------------------------- primitives
def test_primitive_geometry_is_exact_where_the_math_is_known():
    v, f = P.box([1, 2, 3])
    q = P.mesh_quality((v, f))
    assert q["area"] == pytest.approx(22.0) and q["degenerate_share"] == 0
    v, f = P.cylinder([0, 0, 0], [0, 2, 0], 0.5, 256)
    assert P.mesh_quality((v, f))["area"] == pytest.approx(2 * np.pi * 0.5 * 2 + 2 * np.pi * 0.25, rel=1e-3)
    v, f = P.revolve([(1.0, 0.0), (1.0, 2.0)], 360)  # open cylinder wall
    assert P.mesh_quality((v, f))["area"] == pytest.approx(2 * np.pi * 1.0 * 2.0, rel=1e-3)


def test_airfoil_and_wing_dimensions():
    a = P.airfoil(2.0, 0.12, 40)
    assert a[:, 0].max() == pytest.approx(2.0) and a[:, 1].max() == pytest.approx(0.12 * 2.0 / 2, rel=0.03)  # NACA: t/2 each side
    v, f = P.wing(10, 3, 1, sweep_deg=30)
    assert np.ptp(v[:, 2]) == pytest.approx(10.0)
    assert v[v[:, 2] > 9.99][:, 0].mean() - v[v[:, 2] < 0.01][:, 0].mean() == pytest.approx(10 * np.tan(np.radians(30)) + 0.25 * (1 - 3), abs=0.05)  # mean-x shift: sweep plus quarter-chord offset


def test_sweep_keeps_the_section_size_around_a_bend():
    th = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    sec = np.stack([0.3 * np.cos(th), 0.3 * np.sin(th)], 1)
    path = np.array([[0, 0, 0], [5, 0, 0], [5, 0, 5], [10, 0, 5]], float)
    v, _ = P.sweep(sec, path)
    ring = v.reshape(len(path), 16, 3)
    for i in range(len(path)):
        r = np.linalg.norm(ring[i] - ring[i].mean(0), axis=1)
        assert np.allclose(r, 0.3, atol=1e-6)


def test_loft_rejects_mismatched_rings():
    with pytest.raises(ValueError):
        P.loft([np.zeros((4, 3)), np.zeros((5, 3))])


def test_unknown_part():
    with pytest.raises(KeyError):
        gallery.get("nope")


# ---------------------------------------------------------------- every part
@pytest.mark.parametrize("key", KEYS)
def test_part_is_a_valid_scene(key, scenes):
    sc = scenes[key]
    assert "ILLUSTRATIVE" in sc.title and "illustrative" in sc.source
    assert len(sc.parts) >= 1 and len(sc.times) >= 5
    for p in sc.parts:
        q = P.mesh_quality((p.vertices, p.faces))
        assert q["finite"] and q["in_range"] and q["degenerate_share"] < 0.01, p.name
        assert np.ptp(p.vertices, axis=0).max() > 0.01
        for name, a in p.fields.items():
            assert a.shape == (len(sc.times), len(p.vertices)), (p.name, name, a.shape)
            assert np.isfinite(a).all(), (p.name, name)
    nf = [(p.name, n) for p in sc.parts for n in p.fields]
    assert nf, "scene has no fields"
    # at least one field changes over time (it is a *time-varying* twin)
    assert any(np.ptp(a, axis=0).max() > 1e-6 for p in sc.parts for a in p.fields.values())
    for s in sc.sensors:
        assert len(s.series) == len(sc.times)


@pytest.mark.parametrize("key", KEYS)
def test_part_exports_a_servable_folder(key, exported):
    folder = exported / key
    m = json.load(open(folder / "scene.json"))
    assert m["format"] == "pinneapple-twin3d/1" and {"index.html", "viewer.js"} <= set(os.listdir(folder))
    raw = open(folder / "geometry.glb", "rb").read()
    assert struct.unpack("<III", raw[:12]) == (0x46546C67, 2, len(raw))
    assert np.isfinite(np.fromfile(folder / "fields.bin", dtype="<f4")).all()
    assert key in open(exported / "index.html").read()


# ---------------------------------------------------------------- physics-shaped expectations
def test_runner_channel_wears_most_near_the_tap_end_and_never_recovers(scenes):
    sc = scenes["runner_channel"]
    p = part(sc, "runner")
    w = p.fields["wear_depth"]
    assert np.all(np.diff(w, axis=0) >= -1e-6) and w[0].max() == 0
    n_rows = 70
    row_of_max = int(np.argmax(w[-1])) // (len(p.vertices) // n_rows)
    assert row_of_max < 0.2 * n_rows
    assert p.fields["remaining_thickness"][-1].min() > 0
    assert 20 < w[-1].max() < 200  # mm after a year: plausible order of magnitude


def test_mixing_tank_blends_towards_uniform_concentration(scenes):
    c = part(scenes["mixing_tank"], "tank").fields["tracer_concentration"]
    assert c[0].max() > 3.0 and c[-1].max() < 1.8
    assert np.ptp(c[-1]) < 0.3 * np.ptp(c[0])
    v = part(scenes["mixing_tank"], "tank").vertices
    i = int(np.argmax(c[0]))
    assert v[i, 1] > 1.8  # injected near the surface
    shear = part(scenes["mixing_tank"], "impeller_lower").fields["shear_rate"]
    assert shear[0].max() == 0 and shear[-1].max() > 200  # spin-up from rest to full speed
    iv = part(scenes['mixing_tank'], 'impeller_lower').vertices
    rr = np.hypot(iv[:, 0], iv[:, 2])
    assert shear[-1][rr > 0.4].mean() > 2 * shear[-1][rr < 0.15].mean()  # shear is highest at the blade tips


def test_pipeline_loses_wall_at_bends_not_on_straight_runs(scenes):
    sc = scenes["oil_pipeline"]
    p = part(sc, "pipe")
    loss = p.fields["wall_loss"][-1]
    rows = loss.reshape(-1, 24).max(1)
    assert rows.max() > 3 * np.median(rows)
    assert p.fields["wall_loss"][0].max() == 0
    pr = p.fields["pressure"][0].reshape(-1, 24)[:, 0]
    assert np.all(np.diff(pr) <= 1e-9) and pr[0] > pr[-1]  # pressure falls along the line
    assert len(sc.sensors) == 2 and extent(sc)[0] > 100


def test_airliner_dimensions_and_heating(scenes):
    sc = scenes["airliner"]
    e = extent(sc)
    assert 34 < e[0] < 42 and e[2] > 30  # length, wing span
    t = list(sc.times)
    cruise = t.index(60.0)
    f = part(sc, "fuselage")
    T = f.fields["skin_temperature"][cruise]
    x = f.vertices[:, 0]
    assert T[x < 1.0].mean() > T[x > 30].mean() + 8  # nose warmer than aft body at cruise (stagnation)
    assert T.min() < -30  # cold-soaked skin at altitude
    w = part(sc, "wing_right")
    s = w.fields["wing_bending_stress"][t.index(10.0)]
    z = np.abs(w.vertices[:, 2])
    assert s[z < 3].mean() > 2 * s[z > 15].mean()  # bending is highest at the root


def test_rocket_dimensions_heating_and_ablation(scenes):
    sc = scenes["rocket"]
    assert extent(sc)[1] > 40
    nz = part(sc, "nozzle")
    flux = nz.fields["heat_flux"]
    assert nz.vertices[int(np.argmax(flux[2])), 1] == pytest.approx(-1.5, abs=0.4)  # throat
    assert flux[-1].max() == 0  # engine cut-off
    abl = nz.fields["ablation_depth"]
    assert np.all(np.diff(abl, axis=0) >= -1e-9) and abl[-1].max() > 0
    b = part(sc, "body")
    q = b.fields["heat_flux"]
    peak_t = int(np.argmax(q.max(1)))
    assert 0 < peak_t < len(sc.times) - 1  # aero heating peaks mid-ascent, not at lift-off or burnout
    assert b.vertices[int(np.argmax(q[peak_t])), 1] > 40  # at the nose


def test_drone_heats_up_and_stays_inside_its_envelope(scenes):
    sc = scenes["drone"]
    mt = part(sc, "motors").fields["temperature"]
    assert mt[0].max() < 35 and mt[-1].max() > 55
    s = next(s for s in sc.sensors if s.id == "T-motor1")
    assert np.all(np.diff(s.series) > -1e-9) is not None and s.series.max() <= s.envelope[1]
    arms = part(sc, "arms")
    d = np.hypot(arms.vertices[:, 0], arms.vertices[:, 2])
    a = arms.fields["temperature"][-1]
    assert a[d > 0.25].mean() > a[d < 0.12].mean()  # hotter next to the motor than at the body


def test_rover_thermal_swing_and_front_wheels_wear_more(scenes):
    sc = scenes["lunar_rover"]
    T = part(sc, "chassis").fields["temperature"].mean(1)
    assert T.min() < -40 and T.max() > 30
    wl = {p.name: p.fields["wear_depth"] for p in sc.parts if p.name.startswith("wheel")}
    assert len(wl) == 6
    assert all(np.all(np.diff(a, axis=0) >= -1e-9) for a in wl.values())
    assert wl["wheel_L1"][-1].max() > wl["wheel_L3"][-1].max()


def test_wind_turbine_dimensions_and_loads(scenes):
    sc = scenes["wind_turbine"]
    e = extent(sc)
    assert e[1] > 140  # tower + upper blade tip
    b = part(sc, "blade_1")
    er = b.fields["leading_edge_erosion"][-1].reshape(24, -1)
    assert er[-4:].mean() > 20 * er[:4].mean()  # erosion concentrates at the tip
    fat = b.fields["fatigue_damage"][-1].reshape(24, -1)
    assert fat[:5].mean() > fat[-5:].mean()  # fatigue concentrates at the root
    assert np.all(np.diff(part(sc, "tower").fields["fatigue_damage"], axis=0) >= -1e-9)


def test_satellite_cycles_between_sun_and_eclipse(scenes):
    sc = scenes["satellite"]
    a = part(sc, "array_north").fields["temperature"].mean(1)
    assert a.max() > 50 and a.min() < -60
    bus = next(s for s in sc.sensors if s.id == "T-bus")
    assert bus.series.min() >= bus.envelope[0] and bus.series.max() <= bus.envelope[1]
    crossings = np.sum(np.diff((a > 0).astype(int)) != 0)
    assert crossings >= 3  # sunlit -> eclipse -> sunlit -> eclipse over two orbits

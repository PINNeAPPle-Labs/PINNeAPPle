"""Slip-sinkage calibration with held-out points, rut geometry (floor, walls, berms, multi-pass), the ground running
out past the lunar horizon (no visible mesh border from the cameras used), and the readable small-size dashboard."""
import math
import os

import numpy as np
import pytest

pytest.importorskip("scipy")

from pinneapple_simulation.terramechanics import WongReeceModel, get_soil, testbed  # noqa: E402
from pinneapple_simulation.terramechanics.terrain import DeformableTerrain  # noqa: E402


class _Flat:
    def height(self, x, y):
        return np.zeros(np.broadcast(np.asarray(x, float), np.asarray(y, float)).shape)


# ---------------------------------------------------------------------- slip-sinkage
def test_holdout_split_alternates_and_keeps_the_lowest_slip_for_training():
    d = testbed.load_dataset("ding2011", "smooth")
    tr = testbed.holdout_split(d)
    assert tr.tolist() == [True, False, True, False, True, False]
    assert d.slip[tr][0] == d.slip.min()


def test_slip_sinkage_calibration_uses_only_the_training_points_and_beats_the_cited_soil():
    d = testbed.load_dataset("ding2011", "smooth")
    cal = testbed.calibrate_holdout(d)
    before, after = cal["test_before"]["sinkage_m"]["rmse"], cal["test_after"]["sinkage_m"]["rmse"]
    assert after < 0.3 * before and after < 0.0015                      # 4.9 mm -> 0.5 mm on the held-out points
    assert cal["params"]["n1"] > 0.2                                    # sinkage grows with slip
    z = cal["report"].curve["sinkage"]
    assert np.all(np.diff(z) > 0) and z[-1] > 2.5 * z[0]
    # moving the held-out points does not change the fit
    d2 = testbed.load_dataset("ding2011", "smooth")
    d2.sinkage[~cal["train"]] *= 3.0
    d2.drawbar_pull[~cal["train"]] += 20.0
    cal2 = testbed.calibrate_holdout(d2)
    for k, v in cal["params"].items():
        assert cal2["params"][k] == pytest.approx(v, rel=1e-6)


def test_sinkage_exponent_modes():
    soil = get_soil("GRC-1")
    assert soil.n1 > 0 and soil.sources["n1"].startswith("estimate")
    w = WongReeceModel(soil, testbed.load_dataset("ding2011", "smooth").wheel)
    assert w.sinkage_exponent(0.5) == pytest.approx(soil.n0 + 0.5 * soil.n1)
    assert w.sinkage_exponent(-0.5) == pytest.approx(soil.n0)            # "drive": skid does not dig
    w.slip_sinkage = "both"
    assert w.sinkage_exponent(-0.5) == pytest.approx(soil.n0 + 0.5 * soil.n1)
    w.slip_sinkage = "none"
    assert w.sinkage_exponent(0.9) == pytest.approx(soil.n0)
    with pytest.raises(ValueError):
        WongReeceModel(soil, w.wheel, slip_sinkage="sometimes")
    w = WongReeceModel(soil, w.wheel)
    s = np.array([0.0, 0.2, 0.5])
    z = w.steady_state(np.full(3, 80.0), s)["sinkage"]
    assert np.all(np.diff(z) > 0)
    # sinkage_slip decouples the exponent from the force slip
    r0 = w.forces(0.5, 0.02)
    r1 = w.forces(0.5, 0.02, sinkage_slip=0.0)
    assert r1.Fz > r0.Fz                                                 # stiffer soil without slip-sinkage


# ---------------------------------------------------------------------- ruts
def test_rut_cross_section_has_a_floor_walls_at_the_repose_angle_and_berms():
    t = DeformableTerrain(_Flat(), (-1, 1), (-1, 1), cell=0.005, wall_angle_deg=38.0, berm=0.3)
    z, b = 0.02, 0.2
    for x in np.arange(-0.4, 0.41, 0.02):
        t.record_wheel(x, 0.0, (1.0, 0.0), 0.25, b, z)
    y = np.linspace(-0.3, 0.3, 601)
    h = t.surface(np.zeros_like(y), y)
    assert h[300] == pytest.approx(-z, abs=1e-3)                        # floor under the wheel
    assert np.all(h[np.abs(y) < 0.09] < -0.9 * z)                       # as wide as the wheel
    run = z / math.tan(math.radians(38.0))
    yw = 0.1 + 0.5 * run
    assert t.surface(0.0, yw) == pytest.approx(-0.5 * z, abs=2e-3)      # half way up the wall
    side = (np.abs(y) > 0.1 + run)
    assert h[side].max() > 0.2 * z                                      # berms above the original surface
    area_rut = -np.trapezoid(np.clip(h, None, 0), y)
    area_berm = np.trapezoid(np.clip(h, 0, None), y)
    assert area_berm == pytest.approx(0.3 * z * b, rel=0.35)            # the displaced share of the cross-section
    assert area_rut > z * b
    # multi-pass: a following, shallower wheel does not raise the floor; a deeper one deepens it
    t.record_wheel(0.0, 0.0, (1.0, 0.0), 0.25, b, 0.01)
    assert t.surface(0.0, 0.0) == pytest.approx(-z, abs=1e-3)
    t.record_wheel(0.0, 0.0, (1.0, 0.0), 0.25, b, 0.03)
    assert t.surface(0.0, 0.0) == pytest.approx(-0.03, abs=1e-3)
    # a wheel crossing a berm flattens it
    t2 = DeformableTerrain(_Flat(), (-1, 1), (-1, 1), cell=0.005)
    t2.record_wheel(0.0, 0.0, (1.0, 0.0), 0.25, b, z)
    yb = y[side][np.argmax(h[side])]
    assert t2.berm_h.max() > 0
    t2.record_wheel(0.0, yb, (1.0, 0.0), 0.25, b, 0.005)
    assert t2.surface(0.0, yb) < 0


def test_grouser_marks_are_spaced_by_the_pitch_and_the_slip():
    from pinneapple_simulation.vehicle import VehicleSimulator, build_vehicle, scenarios, viper
    spec = viper()
    ter = scenarios.flat_site(x_range=(-1.5, 3.0), y_range=(-1.5, 1.5), relief=0.0)
    run = VehicleSimulator(build_vehicle(spec), ter, get_soil("GRC-1")).run(4.0, speed=0.15, settle=1.0, ramp=0.5,
                                                                            output_dt=0.1)
    g = [a for _, kind, a in run.terrain_events if kind == "grouser"]
    xs = np.array([a[0] for a in g if a[1] > 0 and abs(a[1] - 0.65) < 0.05])   # front-left track
    dx = np.diff(np.sort(xs))
    dx = dx[(dx > 0.02)]
    slip = float(np.median(run.wheel["slip"][run.t > 1.5, 0]))
    pitch = spec.wheel.radius * spec.wheel.grousers.pitch_angle()
    assert np.median(dx) == pytest.approx(pitch * (1 - slip), rel=0.15)
    assert ter.berm_h.max() > 0.002 and ter.wall.max() > 0.005


# ---------------------------------------------------------------------- horizon
def _project_all_inside(view, cam, size):
    from pinneapple_tools.visualization.studio.planetary import _project, border_points
    px, depth = _project(border_points(view), cam, size, 35.0)
    w, h = size
    return ((depth > 0) & (px[:, 0] >= 0) & (px[:, 0] <= w) & (px[:, 1] >= 0) & (px[:, 1] <= h)).sum()


def test_ground_border_is_never_visible_from_the_cameras_used():
    pytest.importorskip("pyvista")
    from pinneapple_simulation.vehicle import VehicleSimulator, build_vehicle, scenarios, viper, viz
    from pinneapple_tools.visualization.studio.planetary import border_visibility, horizon_distance
    assert horizon_distance(2.0) == pytest.approx(2636, rel=0.01)
    spec = viper()
    site = scenarios.crater_site(scenery_craters=10)
    run = VehicleSimulator(build_vehicle(spec), site, get_soil("GRC-1")).run(2.0, speed=0.2, settle=0.5, ramp=0.5,
                                                                             output_dt=0.25, record_stresses=False)
    size = (1280, 720)
    for k in (0, len(run.t) // 2, len(run.t) - 1):
        cam = viz.follow_camera(run, k)
        view = viz.frame_view(run, site, k, sun_azimuth_deg=160.0, half_window=(3.2, 3.4))
        assert view.horizon > 1.2 * horizon_distance(cam[0][2] - float(view.Z.mean()))
        res = border_visibility(view, cam, size)
        assert res["visible"] == 0
        # the old setup (no annulus) shows its edge: the check does detect a visible border
        view.horizon = 0.0
        assert _project_all_inside(view, cam, size) > 0
        assert border_visibility(view, cam, size)["visible"] > 0
    bview = viz.beauty_view(spec)
    for cam in (viz.BEAUTY_CAMERA, ((4.3, -3.9, 1.75), (0.1, 0.0, 0.95)), ((0.4, -5.6, 1.15), (0.0, 0.0, 1.05))):
        assert border_visibility(bview, cam, (1600, 900))["visible"] == 0


# ---------------------------------------------------------------------- dashboard at social-feed size
def test_dashboard_small_layout_and_compact_gif(tmp_path):
    pytest.importorskip("imageio", reason="the GIF/MP4 writer is imageio: pip install -e '.[pyvista]'")
    from pinneapple_tools.visualization.dashboard import Chart, Dashboard, Param, Series, Tile
    from pinneapple_tools.visualization.dashboard.static import write_gif, write_png
    t = np.linspace(0, 10, 12)
    rng = np.random.default_rng(0)
    frames = [np.clip(rng.normal(80, 5, (90, 160, 3)), 0, 255).astype(np.uint8) for _ in t]
    db = Dashboard("Title", t, frames, charts=[Chart("A", "a [m]", "t [s]", [Series("x", np.sin(t))], slot="left"),
                                               Chart("B", "b [m]", "t [s]", [Series("y", np.cos(t))], slot="bottom")],
                   tiles=[Tile("D", t, "m", "{:.1f}")], params=[Param(f"p{i}", i) for i in range(18)],
                   subtitle="a long subtitle " * 12)
    p = write_png(db, str(tmp_path / "small.png"), size=(1280, 720), font_scale=1.0)
    from PIL import Image
    assert Image.open(p).size == (1280, 720)
    out = db.to_video(gif=str(tmp_path / "d.gif"), fps=6, gif_size=(1280, 720), gif_font_scale=1.0)
    im = Image.open(out["gif"])
    assert im.size == (1280, 720) and im.n_frames == len(t)
    g = write_gif(str(tmp_path / "g.gif"), [np.zeros((40, 60, 3), np.uint8)] * 3, fps=5)
    assert os.path.getsize(g) < 5000


# ---------------------------------------------------------------------- zoo PINN in the dashboard, stress overlay
def test_zoo_contact_surrogate_adapter():
    pytest.importorskip("torch")
    from pinneapple_hub import zoo
    from pinneapple_simulation.terramechanics.surrogate import (
        ZooContactSurrogate,
        compare_on_states,
    )
    from pinneapple_simulation.vehicle import viper
    m = zoo.load("rover_contact_grc1_viper")
    sur = ZooContactSurrogate(m)
    assert sur.meta["test_errors"]["Fx"]["rmse_N"] < 1.0 and sur.meta["name"].startswith("rover_contact")
    s, z = np.array([0.05, 0.2, 1.5]), np.array([0.015, 0.02, 0.02])
    P = sur.predict(s, z)
    assert P.shape == (3, 3)
    np.testing.assert_allclose(P[:, 0], m.predict(np.c_[s, z], warn=False)["Fx_N"])
    assert sur.in_range(s, z).tolist() == [True, True, False]
    # against the library's current GRC-1 model (the physics it was trained on) it is accurate, slip included
    wr = WongReeceModel(get_soil("GRC-1"), viper().wheel)
    c = compare_on_states(sur, wr, np.linspace(-0.3, 0.9, 40), np.full(40, 0.02))
    assert c["errors_inside"]["Fx"]["rmse_N"] < 2.0


def test_zoo_rover_model_matches_the_library_physics():
    """Guard: the shipped rover contact model was trained on the library's current physics. A change of GRC-1, of
    the VIPER wheel or of the Wong-Reece options without retraining (and a new zoo release) fails here."""
    pytest.importorskip("torch")
    from pinneapple_hub import zoo
    from pinneapple_simulation.terramechanics.surrogate import (
        PhysicsMismatchWarning,
        ZooContactSurrogate,
    )
    from pinneapple_simulation.vehicle import viper
    sur = ZooContactSurrogate(zoo.load("rover_contact_grc1_viper"))
    wr = WongReeceModel(get_soil("GRC-1"), viper().wheel)
    assert sur.physics is not None
    assert sur.check_physics(wr) == []
    card = zoo.info("rover_contact_grc1_viper")
    assert card.data_lineage["physics"]["soil"]["n1"] == pytest.approx(get_soil("GRC-1").n1)
    # a model of other physics is flagged, with the difference named
    wr0 = WongReeceModel(get_soil("GRC-1").with_(n1=0.0), viper().wheel)
    with pytest.warns(PhysicsMismatchWarning, match="soil.n1"):
        assert sur.check_physics(wr0) == ["soil.n1: 0.8 -> 0"]
    # the superseded release stays listed, with the reason
    vs = zoo.versions("rover_contact_grc1_viper")
    assert vs[0]["status"] == "current" and vs[0]["version"] == card.version
    old = [v for v in vs if v["version"] == "1.0.0"]
    assert old and old[0]["status"] == "superseded" and "slip-sinkage" in old[0]["reason"]


def test_physics_mismatch_and_superseded_entries():
    from pinneapple_hub.zoo import _artifacts as art
    from pinneapple_simulation.terramechanics import physics_mismatch
    from pinneapple_simulation.vehicle import viper
    cur = WongReeceModel(get_soil("GRC-1"), viper().wheel).physics_config()
    assert physics_mismatch(cur, cur) == []
    assert physics_mismatch(None, cur) == ["physics configuration not recorded"]
    assert "model (not recorded)" in physics_mismatch({"R_m": 0.25}, cur)
    e = art.entry("models", "rover_contact_grc1_viper", "1.0.0")
    assert e["revision"] == "v1.0.0" and e["superseded_by"] and "bundled" not in e["files"]["weights.pt"]
    assert e["hf_repo"] == art.entry("models", "rover_contact_grc1_viper")["hf_repo"]
    with pytest.raises(KeyError):
        art.entry("models", "rover_contact_grc1_viper", "0.0.1")


def test_semi_transparent_overlay_keeps_the_shading():
    from pinneapple_tools.visualization.studio.planetary import SurfaceView, _field_blend
    from pinneapple_tools.visualization.studio.shadows import shade_terrain
    X, Y = np.meshgrid(np.linspace(0, 1, 40), np.linspace(0, 1, 40))
    Z = 0.05 * np.sin(8 * X)                                             # relief across the field
    f = np.full(X.shape, 10.0)
    view = SurfaceView(X, Y, Z, terrain_field=f, clim=(0.0, 20.0), field_alpha=0.6, field_glow=0.0)
    tint, glow = _field_blend(view, f)
    assert glow is None
    plain = shade_terrain(X, Y, Z, (0.3, 0.0, 0.2), (0, -3, 2)).mean(-1)
    tinted = shade_terrain(X, Y, Z, (0.3, 0.0, 0.2), (0, -3, 2), tint=tint).mean(-1)
    assert np.corrcoef(plain.ravel(), tinted.ravel())[0, 1] > 0.95      # the relief's shading survives the map
    assert not np.allclose(plain, tinted)                                # and the map changes the colour
    view.field_glow = 0.55
    assert _field_blend(view, f)[1] is not None
    # muted colours: saturation 0 leaves only the map's lightness (grey tint)
    view.field_saturation, view.field_glow = 0.0, 0.0
    t0, _ = _field_blend(view, f)
    col = (t0 - (1 - view.field_alpha)) * np.asarray(view.albedo)          # the map colour the tint carries
    assert np.allclose(col[..., 0], col[..., 1], atol=1e-6) and np.allclose(col[..., 1], col[..., 2], atol=1e-6)
    from pinneapple_tools.visualization.dashboard import ColorBar
    cb = ColorBar("x [kPa]", "inferno", 0.0, 1.0, saturation=0.5)
    c, c1 = cb.colors(9), ColorBar("x", "inferno", 0.0, 1.0).colors(9)
    assert c.shape == (9, 3) and np.ptp(c, axis=1).mean() < np.ptp(c1, axis=1).mean()

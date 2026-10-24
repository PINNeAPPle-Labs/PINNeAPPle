"""3-D antenna views on the studio scene: geometry, balloon (dipole closed form), near-field cuts, stepped frequency
animation with per-frame shapes, self-contained viewer, Blender (optional) and the balloon-vs-table consistency
(EMerge, PINNEAPPLE_SLOW=1)."""
import json
import os

import numpy as np
import pytest

from pinneapple_simulation.electromagnetics.antenna import scene3d
from pinneapple_simulation.electromagnetics.antenna.ava import ETA0, AVAParams, layout
from pinneapple_tools.visualization.studio import Scene, Surface, web_viewer
from pinneapple_tools.visualization.studio.animation import Animation


def _ff(path, f, pattern="dipole", n_th=91, n_ph=181, p_acc=0.8):
    th = np.linspace(0, np.pi, n_th)
    ph = np.linspace(-np.pi, np.pi, n_ph)
    T, P = np.meshgrid(th, ph)
    Et = np.sin(T) if pattern == "dipole" else (1 + np.sin(T) * np.sin(P)) / 2      # z dipole / +y lobe
    U = np.abs(Et) ** 2 / (2 * ETA0)
    prad = float(np.trapezoid(np.trapezoid(U * np.sin(T), th, axis=1), ph))
    np.savez(path, theta=th, phi=ph, E_theta=Et.astype(complex), E_phi=np.zeros_like(Et, complex), f_GHz=f,
             P_rad=prad, P_acc=p_acc, P_inc=1.0, S11=0.1 + 0j)
    return path


def _nf(path, f, p):
    from pinneapple_simulation.electromagnetics.antenna.ava import nearfield_planes, plane_points
    out = {"f_GHz": f}
    k = 2 * np.pi * f * 1e9 / 299_792_458.0 * 1e-3
    for name, pl in nearfield_planes(p).items():
        P = plane_points(pl)
        E = np.zeros((3,) + P.shape[:2], complex)
        E[0] = np.exp(-1j * k * P[..., 1])                    # a wave along +y, x-polarised
        out.update({f"{name}_E": E, f"{name}_origin": pl["origin"], f"{name}_u": pl["u"], f"{name}_v": pl["v"],
                    f"{name}_shape": np.array([pl["nv"], pl["nu"]])})
    np.savez(path, **out)
    return path


@pytest.fixture
def run_dir(tmp_path):
    p = AVAParams(nf_step=10.0)
    for f in (2.0, 3.0, 4.0):
        d = tmp_path / "runs" / "base" / f"f{f:g}"
        d.mkdir(parents=True)
        _ff(str(d / f"ff_{f:g}GHz.npz"), f, "lobe")
        _nf(str(d / f"nf_{f:g}GHz.npz"), f, p)
    return tmp_path / "runs" / "base", p


def test_antenna_surfaces_follow_the_layout():
    p = AVAParams(nslot=2, dir_a=12, dir_b=5, dir_x=190)
    S = {s.name: s for s in scene3d.antenna_surfaces(p)}
    assert set(S) == {"substrate", "copper_top", "copper_bottom", "director"}
    sub = S["substrate"].vertices
    assert np.allclose(sub.min(0), [-p.W / 2, 0, 0]) and np.allclose(sub.max(0), [p.W / 2, p.Lb, p.h])
    top, bot = S["copper_top"].vertices, S["copper_bottom"].vertices
    assert np.allclose([top[:, 2].min(), top[:, 2].max()], [p.h, p.h + 0.035])
    assert np.allclose([bot[:, 2].min(), bot[:, 2].max()], [-0.035, 0.0])
    lay = layout(p)
    assert np.allclose(np.sort(np.unique(top[:, :2], axis=0), 0), np.sort(np.unique(lay.top, axis=0), 0))
    assert S["copper_top"].material == "copper" and S["substrate"].material == "laminate"


def test_balloon_of_a_short_dipole_has_1_76_dbi(tmp_path):
    ff = scene3d.read_farfield(_ff(str(tmp_path / "ff_3GHz.npz"), 3.0, "dipole"))
    m = scene3d.farfield_metrics(ff)
    assert m["D_peak_dBi"] == pytest.approx(10 * np.log10(1.5), abs=2e-3)     # D = 1.5 (1.76 dBi), sin^2 pattern
    assert m["G_peak_dBi"] == pytest.approx(m["D_peak_dBi"] + 10 * np.log10(ff.P_rad / 0.8), abs=1e-9)
    V, F, fields = scene3d.balloon(ff, (0, 0, 0), 1.0, dyn_db=30)
    assert V.shape == (91 * 181, 3) and F.max() < len(V) and F.shape[1] == 3
    r = np.linalg.norm(V, axis=1)
    assert r.max() == pytest.approx(1.0) and r.min() == pytest.approx(0.01)   # nulls on the axis, clipped
    assert np.max(fields["directivity_dBi"]) == pytest.approx(m["D_peak_dBi"])


def test_frequency_scene_steps_shapes_and_fields(run_dir, tmp_path):
    rd, p = run_dir
    files = scene3d.solved_files(str(rd))
    assert list(files) == [2.0, 3.0, 4.0] and all("nf" in v for v in files.values())
    sc, info = scene3d.frequency_scene(p, files, rows=[{"f_GHz": 3.0, "S11_dB": -12.0, "Greal_bore_dBi": 5.0}])
    an = sc.animation
    assert an.time_label == "f" and an.time_unit == "GHz" and an.interpolate is False
    nb = len(next(s for s in sc.surfaces if s.name == "balloon").vertices)
    assert an.surface_shapes["balloon"]["vertices"].shape == (3, nb, 3)
    assert set(an.surface_fields) == {"balloon", "cut_board", "cut_axis"}
    assert an.hud_lines(1)[0] == "f = 3.00 GHz" and "S11: -12.00 dB" in an.hud_lines(1)
    ex = sc.extras()
    assert [g["name"] for g in ex["groups"]] == ["antenna", "balloon", "cuts"]
    assert ex["animation"]["shapes"][0]["surface"] == "balloon" and ex["animation"]["interpolate"] is False
    # |E| of a unit plane wave: 0 dB everywhere (clipped at the 99.5th percentile)
    assert np.allclose(an.surface_fields["cut_board"]["values"], 0.0, atol=1e-6)
    out = scene3d.build_3d(p, str(rd), str(tmp_path / "3d"), name="base", render=False)
    W = out["files"]
    for k in ("antenna_glb", "glb_2", "glb_4", "usda_3", "viewer", "phase_viewer"):
        assert os.path.getsize(W[k]) > 1000, k
    page = open(W["viewer"]).read()
    assert "window.STUDIO_EMBED" in page and 'src="viewer.js"' not in page and "pp/viewer" in page
    from pinneapple_tools.visualization.studio.scene import read_gltf
    names = [s.name for s in read_gltf(W["glb_3"])]
    assert {"substrate", "copper_top", "balloon", "cut_board"} <= set(names)


def test_phase_scene_is_one_period(run_dir):
    rd, p = run_dir
    files = scene3d.solved_files(str(rd))
    sc = scene3d.phase_scene(p, files[3.0]["nf"], n_phase=12)
    an = sc.animation
    assert an.times[0] == 0 and an.times[-1] == 330 and an.time_unit == "°"
    v = an.surface_fields["cut_board"]["values"]
    assert v.shape[0] == 12 and np.abs(v).max() <= 1.0
    # half a period later the instantaneous field changes sign
    assert np.allclose(v[6], -v[0], atol=1e-6)


def test_animation_shapes_roundtrip_and_stepped_resample():
    an = Animation([2.0, 3.0, 4.0], time_label="f", time_unit="GHz", interpolate=False)
    V = np.random.default_rng(0).normal(size=(3, 5, 3))
    an.add_surface_shape("s", V)
    with pytest.raises(ValueError):
        an.add_surface_shape("s", V[:2])
    back = Animation.from_dict(json.loads(json.dumps(an.to_dict())))
    assert back.interpolate is False and back.time_unit == "GHz"
    assert np.allclose(back.surface_shapes["s"]["vertices"], V, atol=1e-4)
    r = an.resample(5)                                    # stepped: frames between solved ones repeat, never blend
    assert np.allclose(r.surface_shapes["s"]["vertices"][1], V[0])
    assert an.time_text(2) == "f = 4.00 GHz"


def test_single_file_viewer_runs_in_chromium(run_dir, tmp_path):
    pytest.importorskip("playwright")
    from pinneapple_tools.visualization.studio.web import _chromium_path, screenshot
    if _chromium_path() is None:
        pytest.skip("no Chromium")
    rd, p = run_dir
    sc, _ = scene3d.frequency_scene(p, scene3d.solved_files(str(rd)))
    page = web_viewer(sc, str(tmp_path), single_file=True, filename="v.html")
    png = screenshot(page, str(tmp_path / "v.png"), frame=2, setup_js="studio.setLayer('cuts', false)")
    assert png and os.path.getsize(png) > 10_000


def test_blender_renders_a_shape_animation(tmp_path):
    pytest.importorskip("bpy")
    from pinneapple_tools.visualization.studio import blender
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    F = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    sc = Scene([Surface("tet", V, F, "copper")], axes="z_up")
    an = Animation([1.0, 2.0], time_label="f", time_unit="GHz", interpolate=False)
    an.add_surface_shape("tet", np.stack([V, 2 * V]))
    sc.animation = an
    r = blender.render_animation(sc, str(tmp_path), size=(96, 64), samples=1, camera="fixed", background="studio",
                                 mp4=False, hud=True)
    assert len(r["frames"]) == 2 and all(os.path.getsize(f) > 0 for f in r["frames"])


@pytest.mark.skipif(not os.environ.get("PINNEAPPLE_SLOW"), reason="EMerge FEM solve (set PINNEAPPLE_SLOW=1)")
def test_emerge_balloon_matches_the_table(tmp_path):
    pytest.importorskip("emerge")
    from pinneapple_simulation.electromagnetics.antenna import study
    r = study.render_3d({"base": {}}, str(tmp_path), freqs=(3.0,), nf_step=6.0, render=False)["base"]
    assert r["runs"]["status"] == {"ok": 1}
    assert r["balloon_vs_table_max_abs_dB"] < 1e-9
    assert os.path.exists(tmp_path / r["files"]["phase_viewer"])

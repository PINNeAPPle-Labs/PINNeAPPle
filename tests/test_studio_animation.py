"""Time animations of studio scenes: Animation data, poses, scene.json, the browser viewer and the Blender renderer."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from pinneapple_tools.visualization.studio import Scene, Surface
from pinneapple_tools.visualization.studio.animation import (
    Animation,
    matrix_from_quat,
    quat_from_matrix,
    slerp,
)


def _in_subprocess(func: str, out_dir) -> None:
    """Run ``func(out_dir)`` of this file in a child interpreter: bpy (Blender as a module) is never imported into the
    pytest process, whose exit would otherwise run Blender's teardown alongside torch / Playwright handles (it can
    crash the interpreter at exit after every test passed)."""
    import subprocess
    import sys
    code = ("import importlib.util, sys; s = importlib.util.spec_from_file_location('t', sys.argv[1]); "
            "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); getattr(m, sys.argv[2])(sys.argv[3])")
    r = subprocess.run([sys.executable, "-c", code, os.path.abspath(__file__), func, str(out_dir)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"{func} failed in the child process (rc {r.returncode}):\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"


def _cube(s=1.0):
    V = np.array([[x, y, z] for x in (-s, s) for y in (-s, s) for z in (-s, s)], float) / 2
    F = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4],
         [1, 5, 7], [1, 7, 3]]
    return V, np.array(F)


def _yaw(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def _scene(T=6):
    V, F = _cube()
    sc = Scene([Surface("floor", [[-5, -5, -1], [5, -5, -1], [5, 5, -1], [-5, 5, -1]], [[0, 1, 2], [0, 2, 3]],
                        "concrete"),
                Surface("cube", V, F, "copper", group="body")])
    t = np.linspace(0, 1, T)
    pos = np.c_[3 * np.cos(np.pi * t), 3 * np.sin(np.pi * t), 0 * t]
    q = quat_from_matrix(np.array([_yaw(np.pi * tt + np.pi / 2) for tt in t]))
    an = Animation(t).add_track("body", pos, q)
    o = np.zeros((T, 1, 3))
    v = np.tile([1.5, 0.0, 0.0], (T, 1, 1))
    an.add_vectors("thrust", o, v, values=np.linspace(0, 1, T)[:, None], attach="body", label="Thrust [kN]")
    an.add_trail("path", pos)
    an.add_surface_field("cube", np.outer(np.linspace(-1, 1, T), V[:, 0]), label="dCp")
    an.add_hud("V", 10 * t, "Speed", "m/s")
    sc.animation = an
    return sc


def test_quaternion_matrix_round_trip():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(20, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0] *= -1
    R = matrix_from_quat(q)
    assert np.allclose(R @ np.swapaxes(R, -1, -2), np.eye(3), atol=1e-12)
    assert np.allclose(np.linalg.det(R), 1)
    assert np.allclose(quat_from_matrix(R), q, atol=1e-12)
    assert np.allclose(matrix_from_quat(quat_from_matrix(_yaw(0.7))), _yaw(0.7))


def test_slerp_unit_norm_and_endpoints():
    q0 = quat_from_matrix(_yaw(0.1))
    q1 = quat_from_matrix(_yaw(2.0))
    assert np.allclose(slerp(q0, q1, 0), q0) and np.allclose(slerp(q0, q1, 1), q1)
    for f in np.linspace(0, 1, 11):
        assert abs(np.linalg.norm(slerp(q0, q1, f)) - 1) < 1e-12
    # half way is the rotation by the mean angle
    assert np.allclose(matrix_from_quat(slerp(q0, q1, 0.5)), _yaw(1.05))


def test_pose_interpolation_endpoints_exact():
    an = _scene().animation
    for k in (0, 3, 5):
        assert np.allclose(an.at(an.times[k], "body"), an.pose("body", k))
    M = an.at(0.5 * (an.times[1] + an.times[2]), "body")
    assert np.allclose(M[:3, 3], 0.5 * (an.tracks["body"]["positions"][1] + an.tracks["body"]["positions"][2]))
    assert np.allclose(M[:3, :3] @ M[:3, :3].T, np.eye(3))
    assert set(an.frame(2)) == {"body"}


def test_body_frame_vertex_lands_in_world():
    an = _scene().animation
    k = 5                                                 # t = 1: at (-3, 0, 0), yawed by 270 degrees
    p = an.transform("body", k, [[1.0, 0.0, 0.0]])[0]      # nose of the body: +x in the body frame
    assert np.allclose(p, [-3.0, 0.0, 0.0] + _yaw(1.5 * np.pi) @ [1, 0, 0], atol=1e-9)
    o, v = an.vectors_world(0, k)                          # attached arrow turns with the body
    assert np.allclose(o[0], [-3, 0, 0], atol=1e-9) and np.allclose(v[0], [0, -1.5, 0], atol=1e-9)


def test_dict_round_trip_and_resample():
    an = _scene().animation
    d = json.loads(json.dumps(an.to_dict()))
    b = Animation.from_dict(d)
    assert np.allclose(b.times, an.times)
    assert np.allclose(b.tracks["body"]["positions"], an.tracks["body"]["positions"], atol=1e-4)
    assert np.allclose(b.tracks["body"]["quaternions"], an.tracks["body"]["quaternions"], atol=1e-6)
    assert np.allclose(b.vectors[0]["values"], an.vectors[0]["values"], atol=1e-4)
    assert b.hud[0]["label"] == "Speed" and d["hud"][0]["nd"] == 1
    r = an.resample(11)
    assert r.n_frames == 11 and np.allclose(r.times[[0, -1]], an.times[[0, -1]])
    assert np.allclose(r.pose("body", 0), an.pose("body", 0)) and np.allclose(r.pose("body", 10), an.pose("body", 5))
    assert np.allclose(np.linalg.norm(r.tracks["body"]["quaternions"], axis=1), 1)
    assert r.surface_fields["cube"]["values"].shape == (11, 8)


def test_shape_errors():
    an = Animation([0.0, 1.0])
    with pytest.raises(ValueError):
        an.add_track("b", np.zeros((3, 3)))
    with pytest.raises(ValueError):
        an.add_vectors("v", np.zeros((2, 1, 3)), np.zeros((2, 1, 3)), attach="missing")
    with pytest.raises(ValueError):
        Animation([0.0, 0.0])


def test_scene_extras_and_web_viewer(tmp_path):
    from pinneapple_tools.visualization.studio.web import web_viewer
    sc = _scene()
    ex = sc.extras()
    assert "animation" in ex and ex["animation"]["tracks"][0]["surfaces"] == [1]
    assert ex["animation"]["fields"][0]["index"] == 1
    assert "animation" not in Scene(sc.surfaces).extras()
    web_viewer(sc, str(tmp_path))
    meta = json.load(open(tmp_path / "scene.json"))
    assert len(meta["animation"]["times"]) == 6 and meta["animation"]["vectors"][0]["attach"] == "body"
    js = open(tmp_path / "viewer.js").read()
    assert "setFrame" in js and "studio-core.js" in js


def _has(mod):
    """Installed? Checked without importing it (importing bpy here would load Blender into the pytest process)."""
    import importlib.util
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


MIN_BPY = (4, 2)             # pinneapple_tools.visualization.studio.blender.MIN_BPY (not imported: it is cheap, bpy is not)


def _bpy_status():
    """(usable, reason) for the installed bpy, read from its package metadata without importing it."""
    if not _has("bpy"):
        return False, "bpy (Blender) not installed"
    from importlib.metadata import PackageNotFoundError, version
    try:
        v = tuple(int(x) for x in version("bpy").split(".")[:2])
    except (PackageNotFoundError, ValueError):
        return True, ""                                         # running inside Blender: no pip metadata
    if v < MIN_BPY:
        return False, f"bpy {v[0]}.{v[1]} is older than the supported {MIN_BPY[0]}.{MIN_BPY[1]}"
    return True, ""


BPY_OK, BPY_WHY = _bpy_status()


def _chromium_ok():
    if not _has("playwright"):
        return False
    from pinneapple_tools.visualization.studio.web import _chromium_path
    return _chromium_path() is not None or bool(os.environ.get("PINNEAPPLE_PLAYWRIGHT_BUNDLED"))


@pytest.mark.skipif(not _chromium_ok(), reason="playwright / Chromium not available")
def test_viewer_screenshot_moves(tmp_path):
    from PIL import Image

    from pinneapple_tools.visualization.studio.web import screenshot, web_viewer
    d = tmp_path / "v"
    web_viewer(_scene(), str(d))
    a = screenshot(str(d), str(tmp_path / "f0.png"), frame=0, size=(480, 300))
    b = screenshot(str(d), str(tmp_path / "f5.png"), frame=5, size=(480, 300))
    A, Bm = (np.asarray(Image.open(p).convert("L"), float) for p in (a, b))
    assert A.shape == (300, 480) and np.abs(A - Bm).mean() > 0.05     # the body (and the HUD) moved


@pytest.mark.skipif(not BPY_OK, reason=BPY_WHY)
def test_blender_render_animation(tmp_path):
    _in_subprocess("_blender_render_animation", tmp_path)


def _blender_render_animation(tmp_path):
    from pinneapple_tools.visualization.studio.blender import render_animation
    sc = _scene()
    r = render_animation(sc, str(tmp_path), frames=[0, 3], size=(64, 36), samples=1, gif=True)
    assert len(r["frames"]) == 2 and all(os.path.getsize(p) > 0 for p in r["frames"])
    assert r["mp4"] and os.path.getsize(r["mp4"]) > 0
    assert r["gif"] and os.path.getsize(r["gif"]) > 0


# ---------------------------------------------------------------------- colour maps, trails, lines, cinematic renders
def _cinematic_scene(T=6):
    sc = _scene(T)
    an = sc.animation
    an.trails.clear()
    pos = an.tracks["body"]["positions"]
    an.add_trail("path", pos, values=np.linspace(0.8, 1.6, T), label="n_z [g]", radius=0.02, alpha=0.8)
    an.add_hud("nz", np.linspace(0.8, 1.6, T), "n_z", "g", "{:.2f}")
    s = np.linspace(-3, 3, 12)
    sc.add_lines([np.c_[s, 0 * s + y, 1 + 0.2 * np.sin(s)] for y in (-1, 1)], [np.sin(s), -np.sin(s)],
                 name="wisps", label="w [m/s]", cmap="coolwarm", alpha=0.5, radius=0.03)
    return sc


def test_colormaps_and_symmetric_range():
    from pinneapple_tools.visualization.studio.compose import cmap_color, default_range
    assert cmap_color("jet", 0.0) == pytest.approx((0, 0, 143 / 255))
    assert cmap_color("bwr", 0.5) == pytest.approx((1.0, 1.0, 1.0))           # neutral midpoint
    assert cmap_color("coolwarm", 0.0)[2] > cmap_color("coolwarm", 0.0)[0]     # blue end
    assert default_range([-0.2, 0.5], "coolwarm") == (-0.5, 0.5)
    assert default_range([-0.2, 0.5], "jet") == (-0.2, 0.5)
    with pytest.raises(ValueError):
        Animation([0.0, 1.0]).add_trail("p", np.zeros((2, 3)), cmap="viridis_typo")


def test_trail_values_round_trip_and_resample():
    sc = _cinematic_scene()
    an = sc.animation
    d = json.loads(json.dumps(an.to_dict()))
    tr = d["trails"][0]
    assert tr["label"] == "n_z [g]" and tr["cmap"] == "jet" and tr["alpha"] == 0.8 and tr["radius"] == 0.02
    assert tr["range"] == pytest.approx([0.8, 1.6]) and tr["values"][-1] == pytest.approx(1.6)
    b = Animation.from_dict(d)
    assert np.allclose(b.trails[0]["values"], an.trails[0]["values"])
    r = an.resample(11)
    assert r.trails[0]["values"].shape == (11,) and r.trails[0]["values"][-1] == pytest.approx(1.6)
    v = Animation(an.times).add_vectors("g", np.zeros((6, 1, 3)), np.ones((6, 1, 3)), values=np.linspace(-1, 3, 6)[:, None],
                                        cmap="bwr")
    assert v.vectors[0]["range"] == (-3.0, 3.0) and v.to_dict()["vectors"][0]["cmap"] == "bwr"
    assert an.hud_lines(5, keys=["nz"]) == ["t = 1.00 s", "n_z: 1.60 g"]


def test_lines_cmap_alpha_in_scene_json(tmp_path):
    from pinneapple_tools.visualization.studio.web import web_viewer
    web_viewer(_cinematic_scene(), str(tmp_path))
    meta = json.load(open(tmp_path / "scene.json"))
    L = meta["lines"][0]
    assert (L["cmap"], L["alpha"], L["radius"], L["range"]) == ("coolwarm", 0.5, 0.03, None)
    assert meta["animation"]["trails"][0]["values"] is not None
    assert "cmapStops" in open(tmp_path / "studio-core.js").read()


def test_compose_grid(tmp_path):
    from PIL import Image

    from pinneapple_tools.visualization.studio.compose import compose_grid
    ims = []
    for i, c in enumerate([(200, 0, 0), (0, 200, 0), (0, 0, 200), (200, 200, 0)]):
        p = tmp_path / f"p{i}.png"
        Image.new("RGB", (80, 45), c).save(p)
        ims.append(str(p))
    out = compose_grid(ims, str(tmp_path / "grid.png"), labels=list("ABCD"), colorbar=("w [m/s]", -1, 1, "coolwarm"))
    im = np.asarray(Image.open(out).convert("RGB"))
    assert im.shape[1] == 160 and im.shape[0] > 90                    # 2 x 2 cells plus the colour bar strip
    assert tuple(im[40, 5]) == (200, 0, 0) and tuple(im[85, 85]) == (200, 200, 0)


def _ffmpeg():
    import shutil
    return shutil.which("ffmpeg") and shutil.which("ffprobe")


@pytest.mark.skipif(not _ffmpeg(), reason="ffmpeg not installed")
def test_compose_grid_video(tmp_path):
    import subprocess

    from pinneapple_tools.visualization.studio.compose import compose_grid_video
    clips = []
    for i in range(4):
        p = str(tmp_path / f"c{i}.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        f"testsrc=size=64x36:rate=4:duration={0.5 + 0.25 * i}", "-pix_fmt", "yuv420p", p], check=True)
        clips.append(p)
    r = compose_grid_video(clips, str(tmp_path / "grid.mp4"), labels=list("ABCD"), colorbar=("n_z", 0, 2, "jet"))
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
                          "stream=codec_name,width,height,pix_fmt,nb_read_frames", "-of", "json", r["mp4"]],
                         capture_output=True, text=True, check=True)
    s = json.loads(out.stdout)["streams"][0]
    assert (s["codec_name"], s["pix_fmt"], s["width"]) == ("h264", "yuv420p", 128)
    assert s["height"] > 72 and s["height"] % 2 == 0 and int(s["nb_read_frames"]) >= 4   # the longest clip: 1.25 s


@pytest.mark.skipif(not BPY_OK, reason=BPY_WHY)
def test_blender_cinematic_render(tmp_path):
    _in_subprocess("_blender_cinematic_render", tmp_path)


def _blender_cinematic_render(tmp_path):
    from PIL import Image

    from pinneapple_tools.visualization.studio.blender import render_animation
    r = render_animation(_cinematic_scene(), str(tmp_path), frames=[1, 4], size=(64, 36), samples=1,
                         camera="chase_low", motion_blur=True, dof=4.0, haze=1.0, sparkline="nz",
                         hud_keys=["V", "nz"], trails={"alpha": 0.6}, colorbars={"lines": False})
    assert len(r["frames"]) == 2 and Image.open(r["frames"][0]).size == (64, 36)
    assert r["mp4"] and os.path.getsize(r["mp4"]) > 0


# ---------------------------------------------------------------------- trail window, scene materials, stills
def test_trail_window_and_fade():
    an = Animation(np.arange(20.0))
    an.add_trail("p", np.c_[np.arange(20.0), np.zeros(20), np.zeros(20)], trail_seconds=4.0, alpha=0.8)
    idx, fade = an.trail_window(0, 10)
    assert idx.tolist() == [6, 7, 8, 9, 10] and fade[-1] == 1.0 and fade[0] == 0.0     # 4 s behind, fading out
    assert np.all(np.diff(fade) > 0)
    assert an.trail_window(0, 2)[0].tolist() == [0, 1, 2]                               # clipped at the start
    assert an.trail_window(0, 10, seconds=1.0)[0].tolist() == [9, 10]
    d = an.to_dict()["trails"][0]
    assert (d["seconds"], d["fade"], d["alpha"]) == (4.0, True, 0.8)
    b = Animation.from_dict(json.loads(json.dumps(an.to_dict())))
    assert b.trail_window(0, 10)[0].tolist() == idx.tolist()
    an.add_trail("all", np.zeros((20, 3)), trail_seconds=None, fade=False)
    assert an.trail_window(1, 10)[0].tolist() == list(range(11)) and np.all(an.trail_window(1, 10)[1] == 1)


def test_scene_materials_override_and_unknown():
    V, F = _cube()
    sc = Scene([Surface("a", V, F, "my_paint"), Surface("b", V, F, "grey")])
    assert sc.unknown_materials() == ["my_paint"]
    sc.materials["my_paint"] = {"color": [0.9, 0.1, 0.1], "metallic": 0.0, "roughness": 0.3, "clearcoat": 1.0}
    assert sc.unknown_materials() == []
    import struct
    glb = sc.to_glb()
    js = json.loads(glb[20:20 + struct.unpack("<I", glb[12:16])[0]])
    m = next(x for x in js["materials"] if x["name"] == "my_paint")
    assert m["pbrMetallicRoughness"]["baseColorFactor"][0] == pytest.approx(0.9 ** 2.2, abs=1e-4)
    assert "KHR_materials_clearcoat" in m["extensions"]


@pytest.mark.skipif(not BPY_OK, reason=BPY_WHY)
def test_blender_render_still(tmp_path):
    _in_subprocess("_blender_render_still", tmp_path)


def _blender_render_still(tmp_path):
    import pathlib

    from PIL import Image
    tmp_path = pathlib.Path(tmp_path)

    from pinneapple_tools.visualization.studio.blender import render_still
    sc = _cinematic_scene()
    sc.surfaces[0].material = "asphalt"                           # procedural ground material path
    out = render_still(sc, str(tmp_path / "still.png"), 3, size=(64, 36), samples=1, camera="chase_low",
                       motion_blur=True, dof=4.0)
    assert Image.open(out).size == (64, 36)


def test_min_bpy_matches_library():
    from pinneapple_tools.visualization.studio import blender
    assert blender.MIN_BPY == MIN_BPY

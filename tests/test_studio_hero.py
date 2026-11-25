"""The studio hero assets (#920): presets per physics, the fallback said out loud, and a real still, glTF and clip.

Nothing here needs Blender: the PyVista fallback draws the pictures, which is exactly the path a CI machine takes.
The turntable test runs only when ``ffmpeg`` is on the PATH, because without it there is no MP4 to check.
"""
from __future__ import annotations

import json
import os
from importlib import import_module

import numpy as np
import pytest

from pinneapple_tools.visualization.studio import (
    Scene,  # the function, as the package exports it
)
from pinneapple_tools.visualization.studio.hero import PRESETS
from pinneapple_tools.visualization.studio.hero import hero as hero_fn
from pinneapple_tools.visualization.studio.scene import sphere_surface


def _scene(n=24) -> Scene:
    """A sphere with a scalar field and a vector displacement -- the smallest thing a hero preset can draw."""
    s = sphere_surface((0.0, 0.0, 0.0), 1.0, name="shell")
    V = s.vertices
    s.fields["cp"] = -1.5 * V[:, 0] / np.max(np.abs(V[:, 0]))          # -1.5 .. 1.5, a Cp-like range
    s.fields["STRESS"] = 1e6 * (1.0 + V[:, 2])
    s.fields["DISP"] = np.c_[np.zeros(len(V)), np.zeros(len(V)), 1e-3 * np.ones(len(V))]
    sc = Scene(surfaces=[s])
    return sc


#: the module itself: `from ...studio import hero` gives the function, which is what a caller wants
H = import_module("pinneapple_tools.visualization.studio.hero")


def _pyvista_ok() -> bool:
    from pinneapple_tools.visualization.pyvista_bridge import can_render

    return can_render()


needs_pv = pytest.mark.skipif(not _pyvista_ok(), reason="no off-screen OpenGL context here")
needs_ffmpeg = pytest.mark.skipif(H._ffmpeg() is None, reason="needs ffmpeg on the PATH")


def test_the_five_presets_of_the_issue_exist_and_alias():
    assert set(PRESETS) == {"external_aero", "internal_flow", "conjugate_heat", "structural", "electromagnetic"}
    for alias, target in (("fluids", "external_aero"), ("structures", "structural"), ("em", "electromagnetic"),
                          ("heat", "conjugate_heat"), ("pipe", "internal_flow")):
        assert H._resolve_kind(alias).kind == target
    with pytest.raises(ValueError, match="kind="):
        hero_fn(_scene(), "astrology")


def test_the_preset_picks_the_field_it_knows(tmp_path):
    sc = _scene()
    assert H._pick_field(sc, PRESETS["external_aero"], None) == "cp"
    assert H._pick_field(sc, PRESETS["structural"], None) == "STRESS"
    assert H._pick_field(sc, PRESETS["structural"], "cp") == "cp"
    with pytest.raises(KeyError, match="no field"):
        H._pick_field(sc, PRESETS["structural"], "temperature")


def test_a_deformation_is_a_stated_exaggeration():
    sc = _scene()
    out, warp = H._deform(sc, "DISP", 0.05, H._displacements(sc, sc))
    assert warp is not None and warp["scale_factor"] > 1
    assert "max |u|" in warp["statement"] and warp["exaggeration"] == 0.05
    assert len(out.surfaces) == 1                                   # the deformed body
    assert out.lines and out.lines[-1].name == "undeformed"         # and the wireframe of the undeformed one
    assert all(len(seg) == 2 for seg in out.lines[-1].points)
    peak = np.abs(out.surfaces[0].vertices - sc.surfaces[0].vertices).max()
    assert peak == pytest.approx(0.05 * warp["reference_length"], rel=1e-6)
    flat = Scene(surfaces=[sphere_surface((0, 0, 0), 1.0)])
    assert H._deform(flat, "DISP", 0.05, {})[1] is None              # no displacement, no claim


def test_the_camera_orbits_the_subject():
    sc = _scene()
    centre, eye, d = H._frame_camera(sc, 0.0, 0.0)
    assert np.allclose(centre, 0.0, atol=1e-9) and d[2] == pytest.approx(0.0)
    _c, eye90, _d = H._frame_camera(sc, 90.0, 0.0)
    assert not np.allclose(eye, eye90)


@needs_pv
def test_a_hero_of_a_real_scene_is_two_stills_a_gltf_and_the_book(tmp_path):
    r = hero_fn(_scene(), "external_aero", out_dir=str(tmp_path), backend="pyvista", stills=((320, 180),),
                turntable=False)
    assert r["backend"] == "pyvista" and r["fallback"]                 # the fallback is named, never hidden
    still = r["stills"]["320x180"]
    assert os.path.exists(still) and os.path.getsize(still) > 1000
    assert r["gltf"] and open(r["gltf"], "rb").read(4) == b"glTF"
    assert r["field"]["name"] == "cp" and r["field"]["range"] is not None
    book = json.load(open(os.path.join(tmp_path, "hero.json"), encoding="utf-8"))
    assert book["schema"] == "pinneapple.hero/1" and book["stills"] == r["stills"]
    assert book["honesty"]["deformation"].startswith("not deformed")   # nothing was scaled here


@needs_pv
@needs_ffmpeg
def test_the_turntable_is_an_mp4_of_orbiting_frames(tmp_path):
    r = hero_fn(_scene(), "external_aero", out_dir=str(tmp_path), backend="pyvista", stills=((320, 180),),
                turntable=True, seconds=1.0, fps=4.0, turntable_backend="pyvista")
    tt = r["turntable"]
    assert tt["frames"] == 4 and tt["backend"] == "pyvista" and tt["mp4"] and tt["reason"] == ""
    assert os.path.exists(tt["mp4"]) and os.path.getsize(tt["mp4"]) > 1000
    assert tt["per_frame_s"] is not None and tt["per_frame_s"] < 30


def test_without_a_renderer_the_stills_are_absent_and_the_gltf_is_not(monkeypatch, tmp_path):
    """No Blender and no OpenGL: the return says so, and nothing pretends a picture was rendered."""
    monkeypatch.setattr(H, "_blender_available", lambda: (False, "no Blender on this machine"))
    monkeypatch.setattr(H, "_pyvista_available", lambda: (False, "no off-screen OpenGL context"))
    r = hero_fn(_scene(), "structural", out_dir=str(tmp_path), stills=((320, 180),))
    assert r["backend"] is None and r["stills"] == {}
    assert "no Blender" in r["fallback"] and "OpenGL" in r["fallback"]
    assert r["gltf"] and open(r["gltf"], "rb").read(4) == b"glTF"      # geometry needs no renderer
    assert r["turntable"]["mp4"] is None and r["turntable"]["reason"]


def test_the_fallback_reason_is_carried_when_blender_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(H, "_blender_available", lambda: (False, "no Blender: pip install bpy"))
    if not _pyvista_ok():
        pytest.skip("neither back end is available here")
    r = hero_fn(_scene(), "external_aero", out_dir=str(tmp_path), stills=((200, 120),), turntable=False)
    assert r["backend"] == "pyvista" and "pip install bpy" in r["fallback"]


def test_asking_for_cycles_without_blender_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(H, "_blender_available", lambda: (False, "no Blender"))
    with pytest.raises(RuntimeError, match="Blender"):
        hero_fn(_scene(), "external_aero", out_dir=str(tmp_path), backend="cycles", turntable=False)


def test_the_studio_exports_the_function_not_the_module():
    import pinneapple_tools.visualization.studio as S

    assert callable(S.hero) and S.hero.__module__.endswith("studio.hero")
    assert S.hero is hero_fn
    assert callable(S.hero_from_experiment)


@pytest.mark.skipif(not os.path.exists(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "calculix_cantilever", "beam.frd")),
    reason="the structural example is not in this checkout")
def test_a_structural_result_keeps_its_displacements_through_the_frd_reader():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sc = H._as_scene(os.path.join(root, "examples", "calculix_cantilever", "beam.frd"))
    vec = H._displacements(os.path.join(root, "examples", "calculix_cantilever", "beam.frd"), sc)
    assert "DISP" in vec and vec["DISP"].shape == sc.surfaces[0].vertices.shape
    assert np.isfinite(vec["DISP"]).all()
    _out, warp = H._deform(sc, "DISP", 0.05, vec)
    assert warp and warp["field"] == "DISP" and warp["max_displacement"] > 0


def test_the_cycles_call_is_built_without_a_sample_count(monkeypatch, tmp_path):
    """`samples=None` means "the quality preset's own count"; it must reach the renderer as absent, not as int(None)."""
    seen: list[dict] = []

    def fake(scene, path, **kw):
        seen.append(kw)
        open(path, "wb").write(b"\x89PNG" + b"0" * 64)
        return {"path": path, "seconds": 0.0, "backend": "cycles (fake)"}

    monkeypatch.setattr(H, "_cycles_still", fake)
    monkeypatch.setattr(H, "_blender_available", lambda: (True, ""))
    monkeypatch.setattr(H, "_importable", lambda name: True)          # pretend bpy is here: no Pillow colour bar
    r = hero_fn(_scene(), "external_aero", out_dir=str(tmp_path), backend="cycles", stills=((64, 36),),
                turntable=False, gltf=False)
    assert r["backend"] == "cycles" and len(seen) == 1
    assert seen[0]["samples"] is None                                # not 0, not an exception
    assert seen[0]["field"] == "cp" and seen[0]["clim"] is not None

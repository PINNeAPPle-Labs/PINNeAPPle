"""Rendering without ``bpy``: finding the Blender binary, and handing a scene to it through the command line."""
import importlib.util
import os
import sys

import pytest

from pinneapple_tools.visualization.studio import blender as B
from pinneapple_tools.visualization.studio import scene as S


def test_blender_executable_prefers_the_environment_then_path(monkeypatch, tmp_path):
    fake = tmp_path / "blender"
    fake.write_text("#!/bin/sh\n")
    monkeypatch.setenv("BLENDER", str(fake))
    assert B.blender_executable() == str(fake)

    monkeypatch.delenv("BLENDER")
    monkeypatch.setattr("shutil.which", lambda name: sys.executable)       # a path that exists
    assert B.blender_executable() == sys.executable


def test_blender_executable_is_none_when_there_is_none(monkeypatch):
    monkeypatch.delenv("BLENDER", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr(B, "_BLENDER_PATHS", ())
    assert B.blender_executable() is None


@pytest.mark.skipif(importlib.util.find_spec("bpy") is not None, reason="bpy is importable, so it renders in-process")
def test_render_headless_says_what_to_set_when_blender_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(B, "blender_executable", lambda: None)
    sc = S.Scene(surfaces=[S.sphere_surface((0.0, 0.0, 0.0), 0.1)])
    with pytest.raises(RuntimeError, match=r"\$BLENDER"):
        B.render_headless(sc, str(tmp_path / "out.png"))


@pytest.mark.skipif(B.blender_executable() is None, reason="needs a Blender binary on this machine")
def test_render_headless_renders_a_scene_with_its_own_materials(tmp_path):
    """The whole round trip: scene -> .glb -> Blender -> png, with a material only the scene knows about."""
    sc = S.Scene(surfaces=[S.sphere_surface((0.0, 0.0, 0.0), 0.4, name="feed", material="feed")],
                 materials={"feed": {"color": [0.10, 0.42, 0.85], "emissive": [0.20, 0.55, 1.0]}})
    out = str(tmp_path / "feed.png")
    B.render_headless(sc, out, background="dark", floor=False, size=(160, 120), samples=4,
                      view=(0.0, -1.0, 0.25), lens=50, exposure=-0.5)
    with open(out, "rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n"
    assert os.path.getsize(out) > 1000


@pytest.mark.skipif(B.blender_executable() is None, reason="needs a Blender binary on this machine")
def test_render_headless_quality_preset_and_hdri_environment(tmp_path):
    """``quality="preview"`` sizes the frame, and ``background="hdri:<file>"`` loads the image as the world (a flat
    image exercises the node graph: environment texture, strength, rotation)."""
    from PIL import Image
    env = tmp_path / "hall.png"
    Image.new("RGB", (64, 32), (200, 210, 225)).save(env)
    sc = S.Scene(surfaces=[S.sphere_surface((0.0, 0.0, 0.0), 0.4, material="steel")])
    out = str(tmp_path / "hdri.png")
    B.render_headless(sc, out, quality="preview", size=(120, 90), background=f"hdri:{env}",
                      device="cpu", colorbar_on=False, hdri_strength=1.5, hdri_rotation=30.0)
    with open(out, "rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n"


def test_quality_presets_are_the_three_looks_and_production_is_the_old_default():
    assert list(B.QUALITY) == ["preview", "production", "cinematic"]
    samples = [B.QUALITY[k]["samples"] for k in B.QUALITY]
    assert samples == sorted(samples), "preview < production < cinematic"
    assert B.QUALITY["production"]["samples"] == 96 and B.QUALITY["production"]["size"] == (1600, 900)
    assert B.QUALITY["production"]["bounces"] is None, "the default look is the one the renderer always had"


def _fake_bpy(devices, backend="NONE"):
    """The little of ``bpy`` that :func:`_set_device` touches."""
    import types
    pref = types.SimpleNamespace(devices=[types.SimpleNamespace(name=n, type=t, use=False) for n, t in devices],
                                 compute_device_type=backend)
    pref.refresh_devices = lambda: None
    ctx = types.SimpleNamespace(preferences=types.SimpleNamespace(
        addons={"cycles": types.SimpleNamespace(preferences=pref)}))
    sc = types.SimpleNamespace(cycles=types.SimpleNamespace(device="CPU"))
    return types.SimpleNamespace(context=ctx), sc


def test_device_auto_picks_the_gpu_backend_this_machine_has():
    bpy, sc = _fake_bpy([("Apple M5", "CPU"), ("Apple M5 (GPU - 8 cores)", "METAL")])
    assert B._set_device(bpy, sc, "auto") == "metal"
    assert sc.cycles.device == "GPU"
    uses = {d.name: d.use for d in bpy.context.preferences.addons["cycles"].preferences.devices}
    assert uses == {"Apple M5": False, "Apple M5 (GPU - 8 cores)": True}


def test_device_falls_back_to_cpu_instead_of_failing_the_render():
    bpy, sc = _fake_bpy([("Apple M5", "CPU")])                          # no GPU at all
    assert B._set_device(bpy, sc, "auto") == "cpu"
    assert sc.cycles.device == "CPU"
    bpy, sc = _fake_bpy([("Apple M5", "CPU"), ("Apple M5 (GPU)", "METAL")])
    assert B._set_device(bpy, sc, "cuda") == "cpu", "a backend without a device falls back, it does not raise"
    assert sc.cycles.device == "CPU"


def test_hdri_path_takes_a_file_a_directory_entry_or_a_registered_name(tmp_path, monkeypatch):
    img = tmp_path / "hall.hdr"
    img.write_bytes(b"#?RADIANCE\n")
    monkeypatch.setattr(B, "HDRIS", {})
    assert B.hdri_path(str(img)) == str(img)                    # an explicit path
    monkeypatch.setenv("PINNEAPPLE_HDRI_DIR", str(tmp_path))
    assert B.hdri_path("hall") == str(img)                      # a name inside $PINNEAPPLE_HDRI_DIR
    monkeypatch.setattr(B, "HDRIS", {"hall": str(img)})
    assert B.hdri_path("hall") == str(img)                      # a name of the registry
    with pytest.raises(FileNotFoundError, match="PINNEAPPLE_HDRI_DIR"):
        B.hdri_path("no_such_hall")                                      # and it says where it looked

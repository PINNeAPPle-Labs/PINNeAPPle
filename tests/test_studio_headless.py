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

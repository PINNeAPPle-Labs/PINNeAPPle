"""The cinematic page: what it writes, that the captions and the camera beats line up, and its refusals. The page
itself is exercised in a browser, not here -- these checks cover the files and the data the page reads."""
from __future__ import annotations

import json
import re

import numpy as np
import pytest

from pinneapple_tools.visualization.studio import cinematic_page
from pinneapple_tools.visualization.studio.cinematic import beats_for
from pinneapple_tools.visualization.studio.scene import Scene

_BODY = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
_FACES = np.array([[0, 1, 2], [0, 2, 3]], int)

SECTIONS = [
    {"kicker": "PINNeAPPle", "title": "O que a simulação mostrou", "body": "Um cantiléver do CalculiX."},
    {"kicker": "O campo", "title": "Tensão de von Mises", "body": "Da base engastada à ponta livre."},
    {"kicker": "O número", "title": "621 MPa no engaste", "body": "Onde a peça trabalha mais."},
]


def _scene(field=True) -> Scene:
    sc = Scene.from_arrays(_BODY, _FACES, name="body", material="steel")
    if field:
        sc.surfaces[0].fields["STRESS"] = np.array([0.0, 10.0, 20.0, 30.0])
        sc.labels["STRESS"] = "von Mises stress (MPa)"
    return sc


def test_the_page_writes_a_self_contained_folder(tmp_path):
    page = cinematic_page(_scene(), str(tmp_path / "demo"), field="STRESS", sections=SECTIONS)
    assert page == str(tmp_path / "demo" / "index.html")
    names = {p.name for p in (tmp_path / "demo").iterdir()}
    assert names == {"index.html", "scene.glb", "scene.json", "vendor"}
    assert (tmp_path / "demo" / "vendor" / "three" / "three.module.min.js").is_file()


def test_no_template_placeholder_survives(tmp_path):
    cinematic_page(_scene(), str(tmp_path / "demo"), field="STRESS", sections=SECTIONS)
    html = (tmp_path / "demo" / "index.html").read_text()
    for marker in ("__CFG__", "__JS__", "__HUD__", "__CAPTIONS__", "__TITLE__", "__CSS__", "__HOME__", "__LANG__", "__HEAD__"):
        assert marker not in html, f"{marker} left in the page"


def test_one_caption_per_camera_beat_and_the_config_carries_them(tmp_path):
    cinematic_page(_scene(), str(tmp_path / "demo"), field="STRESS", sections=SECTIONS)
    html = (tmp_path / "demo" / "index.html").read_text()
    assert len(re.findall(r'<section class="beat', html)) == len(SECTIONS)
    cfg = json.loads((tmp_path / "demo" / "scene.json").read_text())
    assert len(cfg["beats"]) == len(cfg["sections"]) == len(SECTIONS)      # the camera lands on every caption
    assert cfg["field"] == {"name": "STRESS", "label": "von Mises stress (MPa)", "cmap": "jet", "range": [0.0, 30.0]}
    assert cfg["sections"][1]["title"] == "Tensão de von Mises"
    assert 'const CFG = {"title"' in html


def test_the_field_is_baked_as_vertex_colours_like_the_unreal_export(tmp_path):
    cinematic_page(_scene(), str(tmp_path / "demo"), field="STRESS", sections=SECTIONS)
    data = (tmp_path / "demo" / "scene.glb").read_bytes()
    head = json.loads(data[20:20 + int.from_bytes(data[12:16], "little")].decode())
    assert "COLOR_0" in head["meshes"][0]["primitives"][0]["attributes"]


def test_the_hud_shows_the_field_label_and_its_range(tmp_path):
    cinematic_page(_scene(), str(tmp_path / "demo"), field="STRESS", sections=SECTIONS)
    html = (tmp_path / "demo" / "index.html").read_text()
    assert "von Mises stress (MPa)" in html
    assert "linear-gradient(to right" in html                      # the colour bar
    assert ">0<" in html or ">0.000<" in html or ">0<" in html


def test_a_scene_without_a_field_still_makes_a_page(tmp_path):
    cinematic_page(_scene(field=False), str(tmp_path / "demo"), sections=SECTIONS)
    html = (tmp_path / "demo" / "index.html").read_text()
    assert '<div id="hud">' not in html                            # no colour bar without a field
    assert json.loads((tmp_path / "demo" / "scene.json").read_text())["field"]["name"] is None


def test_the_default_sections_describe_the_scene(tmp_path):
    cinematic_page(_scene(), str(tmp_path / "demo"), field="STRESS")
    cfg = json.loads((tmp_path / "demo" / "scene.json").read_text())
    assert cfg["sections"][0]["title"] == "PINNeAPPle scene" and len(cfg["sections"]) == 2


def test_beats_cycle_with_a_nudge_past_the_choreography():
    assert beats_for(3) == [[-2.1, 0.24, 3.7], [-0.95, 0.4, 3.1], [0.3, 0.2, 2.8]]
    sixth = beats_for(6)[5]
    first = beats_for(1)[0]
    assert sixth[0] > first[0] and sixth[2] > first[2]             # a second pass sits wider and further round


def test_refusals(tmp_path):
    with pytest.raises(KeyError):
        cinematic_page(_scene(), str(tmp_path / "a"), field="NOPE", sections=SECTIONS)
    with pytest.raises(ValueError):
        cinematic_page(Scene(), str(tmp_path / "b"), sections=SECTIONS)
    with pytest.raises(ValueError):
        cinematic_page(_scene(), str(tmp_path / "c"), field="STRESS", sections=[])


def test_a_page_can_load_the_geometry_from_elsewhere(tmp_path):
    """A second language's page shares one copy of the geometry instead of writing it twice."""
    cinematic_page(_scene(), str(tmp_path / "en"), field="STRESS", sections=SECTIONS,
                   scene_url="../../demo/scene.glb")
    assert not (tmp_path / "en" / "scene.glb").exists()
    html = (tmp_path / "en" / "index.html").read_text()
    assert '"scene": "../../demo/scene.glb"' in html
    assert "load(CFG.scene || 'scene.glb'" in html

"""The body with a manufacturing history: every operation leaves geometry and a line in the record."""
import numpy as np
import pytest

from pinneapple_tools.visualization.studio import blender as B
from pinneapple_tools.visualization.studio import manufacture as M
from pinneapple_tools.visualization.studio.scene import MATERIALS


def _names(body):
    return [s.name for s in body.scene.surfaces]


def test_a_cast_blank_is_a_drafted_shell_with_a_parting_line():
    b = M.cast_blank(radius=0.11, height=0.09, draft=2.0)
    assert [op.name for op in b.history] == ["cast"]
    assert "saída 2.0°" in b.describe(), "the draft is part of what the part is, not a decoration"
    assert set(_names(b)) == {"blank_skin", "blank_top", "blank_base", "blank_parting"}
    # the wall really tapers: the top ring is narrower than the bottom one
    skin = next(s for s in b.scene.surfaces if s.name == "blank_skin")
    r = np.linalg.norm(skin.vertices[:, :2], axis=1)
    z = skin.vertices[:, 2]
    assert r[z > 0.08].max() < r[z < 0.01].max() - 0.002
    assert b.materials()["as_cast"]["shader"] == "as_cast"


def test_machining_a_face_stands_proud_and_uses_the_cut_material():
    b = M.cast_blank().machine_face([0, 0, 0.09], (0, 0, 1), radius=0.05)
    face = b.scene.surfaces[-1]
    assert face.material == "machined"
    assert face.vertices[:, 2].min() >= 0.09, "the cut face sits on the casting, never below it"
    assert [op.name for op in b.history] == ["cast", "machine_face"]
    assert "face usinada Ø100 mm" in b.describe()


def test_a_bore_is_a_dark_throat_under_a_chamfered_mouth():
    b = M.cast_blank().machine_face([0, 0, 0.09], (0, 0, 1), radius=0.05)
    b.bore([0, 0, 0.0944], (0, 0, 1), diameter=0.03, depth=0.04)
    throat, mouth = b.scene.surfaces[-2], b.scene.surfaces[-1]
    assert throat.material == "bore" and mouth.material == "machined"
    assert "furado Ø30 mm × 40 mm" in b.describe()
    with pytest.raises(ValueError):
        b.bore([0, 0, 0], (0, 0, 1), diameter=0)


def test_a_weld_leaves_a_bead_and_the_tinted_band_beside_it():
    b = M.cast_blank()
    th = np.linspace(0, 2 * np.pi, 60)
    b.weld(np.c_[0.1 * np.cos(th), 0.1 * np.sin(th), np.full_like(th, 0.02)], radius=0.004)
    bead, tint = b.scene.surfaces[-2], b.scene.surfaces[-1]
    assert bead.material == "raw_weld" and tint.material == "heat_tint"
    assert np.linalg.norm(tint.vertices[:, :2], axis=1).max() > np.linalg.norm(bead.vertices[:, :2], axis=1).max(), \
        "the tinted band is wider than the bead, or it stays hidden under it"
    assert "solda Ø8 mm em 60 pontos" in b.describe()


def test_a_bolt_pattern_lands_on_the_face_it_was_given():
    b = M.cast_blank()
    b.bolt_pattern([0, 0, 0.09], (0, 0, 1), radius=0.08, count=6, head_radius=0.007)
    bolts = [s for s in b.scene.surfaces if s.name.startswith("bolts")]
    assert len(bolts) == 6 and all(s.material == "machined" for s in bolts)
    assert "6 parafusos Ø14 mm a 80 mm" in b.describe()


def test_the_history_reads_as_a_process():
    b = (M.cast_blank().machine_face([0, 0, 0.09], (0, 0, 1), radius=0.05)
         .bore([0, 0, 0.0944], (0, 0, 1), diameter=0.03)
         .bolt_pattern([0, 0, 0.09], (0, 0, 1), radius=0.07, count=4))
    assert [op.name for op in b.history] == ["cast", "machine_face", "bore", "bolt_pattern"]
    assert b.describe(" → ").count(" → ") == 3
    for op in b.history:
        assert op.detail and op.surfaces is not None


def test_every_material_a_body_asks_for_is_in_the_library():
    for name in ("as_cast", "machined", "heat_tint", "bore", "raw_weld"):
        assert name in MATERIALS, name
        assert "shader" in MATERIALS[name] or name == "bore"


@pytest.mark.skipif(B.blender_executable() is None, reason="needs a Blender binary on this machine")
def test_finishing_breaks_the_edges_and_keeps_the_history():
    b = M.cast_blank(radius=0.08, height=0.06).machine_face([0, 0, 0.06], (0, 0, 1), radius=0.04)
    before = len(b.scene.surfaces)
    faces_before = sum(len(s.faces) for s in b.scene.surfaces)
    b.finish(width=0.0015)
    assert [op.name for op in b.history][-1] == "finish"
    assert "arestas quebradas 1.5 mm" in b.describe()
    assert len(b.scene.surfaces) == before, "a bevel rebuilds the skin, it does not add parts"
    assert sum(len(s.faces) for s in b.scene.surfaces) > faces_before
    assert "machined" in b.scene.materials, "the materials come back with the finished body"

"""Photography presets, composition, depth of field and the render passes."""
import os

import pytest

from pinneapple_tools.visualization.studio import blender as B
from pinneapple_tools.visualization.studio import scene as S


def test_every_camera_preset_names_a_real_view_and_composition():
    for name, p in B.CAMERAS.items():
        assert p["view"] in B.VIEWS, f"{name}: {p['view']} is not a view"
        assert p["composition"] in B.COMPOSITION, f"{name}: {p['composition']} is not a composition"
        assert p["lens"] > 0 and p["distance"] > 0, name
    assert list(B.CAMERAS) == ["product", "hero", "documentary", "head_on", "wide"]


def test_the_third_lines_are_symmetric_and_centre_is_zero():
    assert B.COMPOSITION["center"] == (0.0, 0.0)
    left, right = B.COMPOSITION["thirds_left"], B.COMPOSITION["thirds_right"]
    assert left[0] == pytest.approx(-right[0]) and left[1] == 0.0
    high, low = B.COMPOSITION["thirds_high"], B.COMPOSITION["thirds_low"]
    assert high[1] == pytest.approx(-low[1]) and high[0] == 0.0


def test_the_camera_preset_fills_only_what_it_declares():
    """product carries a shallow aperture and a third line; documentary keeps everything in focus and centred."""
    assert B.CAMERAS["product"]["fstop"] == 8.0 and B.CAMERAS["product"]["composition"] == "thirds_right"
    assert B.CAMERAS["documentary"]["fstop"] is None and B.CAMERAS["documentary"]["composition"] == "center"
    assert B.CAMERAS["hero"]["lens"] < B.CAMERAS["head_on"]["lens"], "hero is the wide one, head_on the long one"


@pytest.mark.skipif(B.blender_executable() is None, reason="needs a Blender binary on this machine")
def test_a_preset_with_depth_of_field_and_a_third_line_renders(tmp_path):
    sc = S.Scene(surfaces=[S.sphere_surface((0.0, 0.0, 0.0), 0.3, material="steel"),
                           S.sphere_surface((0.0, 0.0, -0.9), 0.3, material="steel", name="behind")])
    out = str(tmp_path / "product.png")
    B.render_headless(sc, out, camera="product", composition="thirds_left", look="high", post="vignette",
                      background="dark", size=(200, 140), samples=4, device="cpu", colorbar_on=False)
    with open(out, "rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n"
    assert os.path.getsize(out) > 500


@pytest.mark.skipif(B.blender_executable() is None, reason="needs a Blender binary on this machine")
def test_render_passes_go_to_a_multilayer_exr(tmp_path):
    sc = S.Scene(surfaces=[S.sphere_surface((0.0, 0.0, 0.0), 0.3, material="steel")])
    out = str(tmp_path / "passes.exr")
    B.render_headless(sc, out, passes=True, film_transparent=True, background="dark", size=(120, 90),
                      samples=4, device="cpu", colorbar_on=False)
    with open(out, "rb") as fh:
        magic = fh.read(4)
    assert magic == b"\x76\x2f\x31\x01", "an OpenEXR file starts with 0x76 0x2f 0x31 0x01"
    assert os.path.getsize(out) > 10000, "a multilayer pass set is not a thumbnail"
    passes_dir = os.path.splitext(out)[0] + "_passes"
    files = os.listdir(passes_dir) if os.path.isdir(passes_dir) else []
    assert files, "the passes folder is written next to the beauty"
    with open(os.path.join(passes_dir, files[0]), "rb") as fh:
        assert fh.read(4) == b"\x76\x2f\x31\x01"
    with open(os.path.join(passes_dir, files[0]), "rb") as fh:
        head = fh.read(4096)
    assert b"depth" in head, "the depth pass is in the file"

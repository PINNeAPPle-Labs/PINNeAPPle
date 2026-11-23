"""Texture maps: the UVs a map needs, and the three consumers that carry it (Blender, glTF, USD)."""
import json
import os
import struct

import numpy as np
import pytest
from PIL import Image

from pinneapple_tools.visualization.studio import blender as B
from pinneapple_tools.visualization.studio import scene as S


def _checker(path, size=8):
    a = np.zeros((size, size, 3), np.uint8)
    a[::2, :] = 255
    a[:, ::2] = 255 - a[:, ::2]
    Image.fromarray(a).save(path)
    return str(path)


def test_box_uvs_stay_inside_the_square_and_use_its_extent():
    V = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    F = np.array([[0, 1, 2], [0, 2, 3]])
    uv = S.box_uvs(V, F)
    assert uv.shape == (4, 2)
    assert uv.min() >= -1e-9 and uv.max() <= 1.0 + 1e-9
    assert uv[:, 0].min() == pytest.approx(0.0) and uv[:, 0].max() == pytest.approx(1.0)
    assert uv[:, 1].min() == pytest.approx(0.0) and uv[:, 1].max() == pytest.approx(1.0)


def test_box_uvs_project_each_axis_on_its_own_plane():
    """A wall in the yz plane and a floor in the xy plane do not collapse to the same line of the map."""
    wall = S.Surface("wall", np.array([[0.0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]]), np.array([[0, 1, 2], [0, 2, 3]]))
    floor = S.Surface("floor", np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]), np.array([[0, 1, 2], [0, 2, 3]]))
    for s in (wall, floor):
        uv = S.surface_uvs(s)
        assert np.ptp(uv[:, 0]) > 0.9 and np.ptp(uv[:, 1]) > 0.9


def test_a_surface_keeps_the_uvs_it_was_given():
    uv = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    s = S.Surface("s", np.zeros((3, 3)), np.array([[0, 1, 2]]), uv=uv)
    assert np.allclose(S.surface_uvs(s), uv)


def _gltf(glb: bytes) -> dict:
    return json.loads(glb[20:20 + struct.unpack("<I", glb[12:16])[0]])


def test_gltf_embeds_the_map_and_a_texture_coordinate(tmp_path):
    png = _checker(tmp_path / "checker.png")
    srf = S.sphere_surface((0.0, 0.0, 0.0), 0.1, material="m")
    glb = S.write_glb([srf], S.AXES["z_up"], {"m": {"color": [1, 1, 1], "metallic": 0.0, "roughness": 0.5,
                                                    "base_color_map": png, "normal_map": png}})
    g = _gltf(glb)
    assert "TEXCOORD_0" in g["meshes"][0]["primitives"][0]["attributes"]
    assert len(g["images"]) == 1, "the same file is embedded once, not once per slot"
    assert g["images"][0]["mimeType"] == "image/png"
    mat = g["materials"][0]
    assert mat["pbrMetallicRoughness"]["baseColorTexture"]["index"] == 0
    assert mat["normalTexture"]["index"] == 0
    assert g["textures"][0]["source"] == 0


def test_gltf_without_maps_carries_no_texture_data(tmp_path):
    srf = S.sphere_surface((0.0, 0.0, 0.0), 0.1, material="m")
    g = _gltf(S.write_glb([srf], S.AXES["z_up"], {"m": {"color": [1, 0, 0], "metallic": 0.0, "roughness": 0.5}}))
    assert "images" not in g and "TEXCOORD_0" not in g["meshes"][0]["primitives"][0]["attributes"]


def test_usd_references_the_map_and_writes_st(tmp_path):
    png = _checker(tmp_path / "checker.png")
    srf = S.sphere_surface((0.0, 0.0, 0.0), 0.1, material="m")
    usda = S.write_usda([srf], {"m": {"color": [1, 1, 1], "metallic": 0.0, "roughness": 0.5, "base_color_map": png}})
    assert "UsdUVTexture" in usda and png in usda
    assert "primvars:st" in usda and "UsdPrimvarReader_float2" in usda


@pytest.mark.skipif(B.blender_executable() is None, reason="needs a Blender binary on this machine")
def test_a_mapped_material_renders_in_blender(tmp_path):
    png = _checker(tmp_path / "checker.png", size=16)
    sc = S.Scene(surfaces=[S.sphere_surface((0.0, 0.0, 0.0), 0.3, material="checker")],
                 materials={"checker": {"color": [1, 1, 1], "metallic": 0.0, "roughness": 0.5,
                                        "base_color_map": png, "normal_map": png, "roughness_map": png}})
    out = str(tmp_path / "mapped.png")
    B.render_headless(sc, out, background="dark", size=(120, 90), samples=4, device="cpu", colorbar_on=False)
    with open(out, "rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n"


@pytest.mark.skipif(B.blender_executable() is None, reason="needs a Blender binary on this machine")
def test_bake_writes_the_three_maps_and_they_render(tmp_path):
    """The round trip the feature exists for: a procedural material baked into maps, then rendered from them."""
    from pinneapple_tools.visualization.studio import bake as BK
    sc = S.Scene(surfaces=[S.sphere_surface((0.0, 0.0, 0.0), 0.3, material="painted_steel")],
                 materials={"painted_steel": {"color": [0.06, 0.22, 0.55], "metallic": 0.10, "roughness": 0.55,
                                              "shader": "painted_metal", "wear": 0.3}})
    maps = BK.bake_materials_headless(sc, str(tmp_path / "maps"), size=64, samples=2)
    assert set(maps["painted_steel"]) == {"base_color_map", "roughness_map", "normal_map"}
    for path in maps["painted_steel"].values():
        assert os.path.getsize(path) > 200

    sc.materials["painted_steel"].update(maps["painted_steel"])
    out = str(tmp_path / "baked.png")
    B.render_headless(sc, out, background="dark", size=(120, 90), samples=4, device="cpu", colorbar_on=False)
    with open(out, "rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n"


@pytest.mark.skipif(B.blender_executable() is None, reason="needs a Blender binary on this machine")
def test_a_baked_scene_carries_the_maps_to_gltf(tmp_path):
    """What a client opens in the browser: the GLB has to have the baked image inside it, not a path to it."""
    from pinneapple_tools.visualization.studio import bake as BK
    sc = S.Scene(surfaces=[S.sphere_surface((0.0, 0.0, 0.0), 0.3, material="cast_iron")],
                 materials={"cast_iron": {"color": [0.20, 0.20, 0.21], "metallic": 0.75, "roughness": 0.62,
                                          "shader": "cast_metal"}})
    maps = BK.bake_materials_headless(sc, str(tmp_path / "maps"), size=64, samples=2)
    sc.materials["cast_iron"].update(maps["cast_iron"])
    g = _gltf(sc.to_glb())
    assert g["images"] and g["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"] == 0

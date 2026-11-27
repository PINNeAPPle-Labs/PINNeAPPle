"""The industrial material library: the flat numbers every exporter reads, and the procedural shaders Blender runs."""
import json
import struct

import pytest

from pinneapple_tools.visualization.studio import blender as B
from pinneapple_tools.visualization.studio import pbr, scene as S


def _shaded():
    return {k: v for k, v in S.MATERIALS.items() if v.get("shader")}


def test_every_shader_named_in_the_table_is_registered():
    assert _shaded(), "the library should have materials with a procedural shader"
    unknown = {k: v["shader"] for k, v in _shaded().items() if v["shader"] not in pbr.SHADERS}
    assert unknown == {}, f"MATERIALS entries naming a shader that does not exist: {unknown}"


def test_every_registered_shader_is_used_by_a_material():
    used = {v["shader"] for v in _shaded().values()}
    assert set(pbr.SHADERS) == used, "a shader nobody names is dead code, or the table lost an entry"


def test_the_industrial_entries_keep_the_numbers_the_exporters_need():
    for name in ("brushed_steel", "stainless_steel", "galvanized_steel", "cast_iron", "painted_steel",
                 "oxidized_steel", "rubber"):
        st = S.MATERIALS[name]
        assert len(st["color"]) == 3 and all(0.0 <= c <= 1.0 for c in st["color"]), name
        assert 0.0 <= st["metallic"] <= 1.0 and 0.0 <= st["roughness"] <= 1.0, name


def test_gltf_export_ignores_the_shader_key_and_keeps_the_pbr_factors():
    """The procedural detail is Blender-only: what leaves the library as glTF is the flat material the entry
    declares, so the browser viewer and Omniverse see the same numbers they always did."""
    srf = S.sphere_surface((0.0, 0.0, 0.0), 0.1, material="painted_steel")
    glb = S.write_glb([srf], S.AXES["z_up"], S.MATERIALS)
    length = struct.unpack("<I", glb[12:16])[0]
    gltf = json.loads(glb[20:20 + length])
    mat = next(m for m in gltf["materials"] if m["name"] == "painted_steel")
    assert "shader" not in mat and "wear" not in mat
    assert mat["pbrMetallicRoughness"]["metallicFactor"] == S.MATERIALS["painted_steel"]["metallic"]
    assert mat["pbrMetallicRoughness"]["roughnessFactor"] == S.MATERIALS["painted_steel"]["roughness"]


def test_usd_export_ignores_the_shader_key():
    srf = S.sphere_surface((0.0, 0.0, 0.0), 0.1, material="brushed_steel")
    usda = S.write_usda([srf], S.MATERIALS)
    assert "brushed_steel" in usda and "shader" not in usda


@pytest.mark.skipif(B.blender_executable() is None, reason="needs a Blender binary on this machine")
def test_every_shader_builds_in_blender(tmp_path):
    """The node graphs are the part no unit test can see: render one small frame with every shader material in it.
    A socket renamed between Blender versions fails here, not in a customer's figure."""
    names = sorted(_shaded())
    surf = [S.sphere_surface((0.25 * i, 0.0, 0.0), 0.09, name=n, material=n) for i, n in enumerate(names)]
    sc = S.Scene(surfaces=surf, axes="z_up")
    out = str(tmp_path / "library.png")
    B.render_headless(sc, out, background="dark", floor=False, size=(320, 120), samples=2, device="cpu",
                      view="side", colorbar_on=False)
    with open(out, "rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n"

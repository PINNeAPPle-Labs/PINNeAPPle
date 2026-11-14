"""The Unreal bridge's export side: geometry with the field in ``COLOR_0``, the sidecar values, the manifest and the
editor script (which only runs inside a real Unreal install and is therefore compiled here, not executed)."""
from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from pinneapple_tools.visualization.studio import export_unreal
from pinneapple_tools.visualization.studio.compose import cmap_color
from pinneapple_tools.visualization.studio.scene import Scene
from pinneapple_tools.visualization.studio.unreal import MAX_VERTICES, vertex_colors

# two triangles in z = 0 and one in z = 1: enough for a field, two surfaces and a real import
_BODY = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
_LID = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1]], float)
_FACES = np.array([[0, 1, 2], [0, 2, 3]], int)
_TRI = np.array([[0, 1, 2]], int)


def _scene(field=True) -> Scene:
    sc = Scene.from_arrays(_BODY, _FACES, name="body", material="steel")
    sc.surfaces.append(type(sc.surfaces[0])("lid", _LID, _TRI, "paint"))
    if field:
        sc.surfaces[0].fields["STRESS"] = np.array([0.0, 10.0, 20.0, 30.0])
        sc.surfaces[1].fields["STRESS"] = np.array([40.0, 50.0, 60.0])
        sc.labels["STRESS"] = "von Mises stress (MPa)"
    return sc


def _glb_json(data: bytes) -> dict:
    assert data[:4] == b"glTF", "not a binary glTF"
    n = struct.unpack("<I", data[12:16])[0]
    return json.loads(data[20:20 + n].decode("utf-8"))


def _read_gltf_accessor(data: bytes, gl: dict, index: int) -> np.ndarray:
    """The floats of one accessor, straight out of the BIN chunk (no pygltflib, no trimesh)."""
    acc, view = gl["accessors"][index], gl["bufferViews"][gl["accessors"][index]["bufferView"]]
    off = 20 + struct.unpack("<I", data[12:16])[0] + 8 + view.get("byteOffset", 0)
    count, ncomp = acc["count"], {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[acc["type"]]
    return np.frombuffer(data, np.float32, count * ncomp, off).reshape(count, ncomp)


def test_the_export_writes_the_folder_unreal_reads(tmp_path):
    out = export_unreal(_scene(), str(tmp_path / "ue"), field="STRESS")
    names = {p.name for p in (tmp_path / "ue").iterdir()}
    assert names == {"geometry.glb", "scene.usda", "fields.bin", "manifest.json", "PinneAPPleImport.py",
                     "README.md"}
    assert out == str(tmp_path / "ue")


def test_the_field_goes_into_color_0_as_linear_vertex_colours(tmp_path):
    """Unreal reads ``COLOR_0`` and ignores the ``_NAME`` attributes three.js uses, so the field has to be there."""
    export_unreal(_scene(), str(tmp_path / "ue"), field="STRESS")
    data = (tmp_path / "ue" / "geometry.glb").read_bytes()
    gl = _glb_json(data)
    prim = gl["meshes"][0]["primitives"][0]
    assert "COLOR_0" in prim["attributes"], "no vertex colours: Unreal would show a grey mesh"
    assert "_STRESS" in prim["attributes"], "the three.js-readable attribute should still be written"
    assert prim["extras"]["fields"]["_STRESS"] == "STRESS"

    rgb = _read_gltf_accessor(data, gl, prim["attributes"]["COLOR_0"])
    # the range is the whole scene's 0..60, so this surface's 0, 10, 20, 30 sit at t = 0, 1/6, 1/3, 1/2
    expected = np.array([cmap_color("jet", t) for t in (0.0, 1 / 6, 1 / 3, 0.5)]) ** 2.2   # sRGB -> linear
    np.testing.assert_allclose(rgb, expected, atol=1e-5)


def test_a_surface_without_the_field_takes_its_material_colour(tmp_path):
    sc = _scene()
    del sc.surfaces[1].fields["STRESS"]
    colors = vertex_colors(sc, "STRESS")
    assert colors["lid"].shape == (3, 3)
    np.testing.assert_allclose(colors["lid"], np.tile([0.93, 0.94, 0.95], (3, 1)))   # the "paint" material


def test_the_range_comes_from_the_scene_and_can_be_overridden(tmp_path):
    export_unreal(_scene(), str(tmp_path / "a"), field="STRESS")
    m = json.loads((tmp_path / "a" / "manifest.json").read_text())
    assert m["field"] == {"name": "STRESS", "cmap": "jet", "range": [0.0, 60.0],
                          "label": "von Mises stress (MPa)"}
    export_unreal(_scene(), str(tmp_path / "b"), field="STRESS", field_range=(-5, 5))
    assert json.loads((tmp_path / "b" / "manifest.json").read_text())["field"]["range"] == [-5.0, 5.0]
    with pytest.raises(ValueError):
        export_unreal(_scene(), str(tmp_path / "c"), field="STRESS", field_range=(1, 1))


def test_the_sidecar_carries_every_field_with_its_offsets(tmp_path):
    sc = _scene()
    sc.surfaces[0].fields["PRESSURE"] = np.arange(4.0)
    export_unreal(sc, str(tmp_path / "ue"), field="STRESS")
    m = json.loads((tmp_path / "ue" / "manifest.json").read_text())
    # surface by surface, field by field: body.STRESS 0-3, body.PRESSURE 4-7, lid.STRESS 8-10
    assert m["surfaces"][0]["fields"] == {"STRESS": {"offset": 0, "count": 4},
                                          "PRESSURE": {"offset": 4, "count": 4}}
    assert m["surfaces"][1]["fields"]["STRESS"] == {"offset": 8, "count": 3}
    raw = np.fromfile(tmp_path / "ue" / "fields.bin", np.float32)
    assert raw.size == 4 + 4 + 3
    np.testing.assert_allclose(raw[4:8], np.arange(4.0))
    np.testing.assert_allclose(raw[8:11], [40.0, 50.0, 60.0])
    assert m["fields"]["PRESSURE"]["label"] == "PRESSURE"


def test_the_editor_script_is_valid_python_bound_to_its_folder(tmp_path):
    export_unreal(_scene(), str(tmp_path / "ue"), field="STRESS", cinematic=False)
    text = (tmp_path / "ue" / "PinneAPPleImport.py").read_text()
    compile(text, "PinneAPPleImport.py", "exec")                    # parses; it imports unreal, so it cannot run here
    assert str(tmp_path / "ue") in text
    assert "geometry.glb" in text and "M_PINNeAPPleVertexColor" in text
    assert '"cinematic": false' in text
    # the {{ }} of the template are format escapes: none may survive into the file
    assert "{{" not in text and "}}" not in text


def test_a_scene_without_fields_still_exports(tmp_path):
    export_unreal(_scene(field=False), str(tmp_path / "ue"))
    gl = _glb_json((tmp_path / "ue" / "geometry.glb").read_bytes())
    assert "COLOR_0" not in gl["meshes"][0]["primitives"][0]["attributes"]
    m = json.loads((tmp_path / "ue" / "manifest.json").read_text())
    assert m["field"]["name"] is None and m["surfaces"][0]["material"] == "steel"


def test_usd_and_the_readme_can_be_switched_off_and_on(tmp_path):
    export_unreal(_scene(), str(tmp_path / "ue"), field="STRESS", usd=False)
    assert not (tmp_path / "ue" / "scene.usda").exists()
    assert "scene.usda" not in json.loads((tmp_path / "ue" / "manifest.json").read_text())["files"]
    assert "von Mises stress (MPa)" in (tmp_path / "ue" / "README.md").read_text()


def test_an_unknown_field_and_an_empty_scene_are_refused(tmp_path):
    with pytest.raises(KeyError):
        export_unreal(_scene(), str(tmp_path / "x"), field="NOPE")
    with pytest.raises(ValueError):
        export_unreal(Scene(), str(tmp_path / "y"))


def test_a_scene_above_the_import_limit_is_refused_with_the_fix(tmp_path):
    sc = _scene()
    sc.surfaces[0].vertices = np.zeros((MAX_VERTICES + 1, 3))
    with pytest.raises(ValueError, match="decimate"):
        export_unreal(sc, str(tmp_path / "z"), field="STRESS")


def test_mismatched_colours_are_refused_rather_than_written(tmp_path):
    from pinneapple_tools.visualization.studio.scene import AXES, MATERIALS, write_glb
    sc = _scene()
    with pytest.raises(ValueError, match="colors has"):
        write_glb(sc.surfaces, AXES[sc.axes], MATERIALS, colors={"body": np.zeros((2, 3))})

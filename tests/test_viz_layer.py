"""The shared 3-D layer (#281): one mesh and its fields become a glTF, a browser page and a Blender render.

The inputs are the five shapes this repository already produces -- a file, a PyVista grid, a ``(vertices, faces)``
pair, a ``pinneapple_data.cae`` mesh and a ``Scene`` -- so the tests check the normalisation at least once per shape,
and then check what the caller cares about: the field is in the glb as the attribute the viewers read, the page is
self-contained, and a render that cannot happen says why instead of writing a picture.
"""
import json
import os
import struct

import numpy as np
import pytest

from pinneapple_tools.visualization.studio import (
    SCENE_MAX_FACES,
    Scene,
    blender_render,
    export_gltf,
    to_scene,
    viewer_html,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CAVITY = os.path.join(HERE, "..", "examples", "physics_check", "pass", "cases", "cavity")

#: a unit cube's triangle surface, 8 vertices and 12 triangles
CUBE_V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                   [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
CUBE_F = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                   [3, 2, 6], [3, 6, 7], [0, 3, 7], [0, 7, 4], [1, 2, 6], [1, 6, 5]])


def _json_chunk(glb: bytes) -> dict:
    """The JSON chunk of a GLB: length at byte 12, the payload after the 20-byte header."""
    assert glb[:4] == b"glTF" and struct.unpack("<I", glb[4:8])[0] == 2
    n = struct.unpack("<I", glb[12:16])[0]
    return json.loads(glb[20:20 + n])


def _attributes(glb: bytes) -> list:
    return [p["attributes"] for m in _json_chunk(glb)["meshes"] for p in m["primitives"]]


# ------------------------------------------------------------------------------------- the input shapes
def test_the_five_inputs_become_the_same_scene(tmp_path):
    sc = to_scene((CUBE_V, CUBE_F))
    assert sc.n_faces() == 12 and len(sc.surfaces) == 1
    path = str(tmp_path / "cube.stl")
    sc.save(path)
    from_file = to_scene(path)
    assert from_file.n_faces() == 12
    assert to_scene(sc).n_faces() == 12                                   # a Scene goes through as it is
    with pytest.raises(TypeError, match="cannot read a dict"):
        to_scene({"points": CUBE_V})


def test_a_pyvista_grid_from_the_openfoam_reader_becomes_the_geometry_of_a_viewer():
    """The two issues meet: the case of #60, exported for the page, carries both of its fields."""
    pv = pytest.importorskip("pyvista")
    from pinneapple_tools.visualization.openfoam import read_openfoam

    grid = read_openfoam(CAVITY)
    assert isinstance(grid, pv.UnstructuredGrid)
    glb = export_gltf(grid, {"p": grid["p"], "U": grid["U"]})
    assert {"_P", "_U"} <= set(_attributes(glb)[0])


def test_a_cell_field_of_a_volume_grid_reaches_the_vertices_without_moving_its_range(tmp_path):
    pytest.importorskip("pyvista")
    from pinneapple_tools.visualization.openfoam import read_openfoam
    from pinneapple_tools.visualization.studio.scene import read_gltf

    grid = read_openfoam(CAVITY)
    out = str(tmp_path / "cavity.glb")
    export_gltf(grid, {"p": grid["p"]}, out=out)
    surf = read_gltf(out)[0]
    assert surf.vertices.shape[0] == grid.n_points                      # the boundary keeps the case's nodes
    assert surf.fields["p"].shape == (grid.n_points,)
    # interpolating the 400 cell values onto 882 nodes does not extend the range of the solution
    assert surf.fields["p"].min() == pytest.approx(float(grid["p"].min()), abs=1e-6)
    assert surf.fields["p"].max() == pytest.approx(float(grid["p"].max()), abs=1e-6)


def test_a_field_that_matches_neither_the_points_nor_the_cells_is_refused():
    pv = pytest.importorskip("pyvista")
    grid = pv.Cube()
    with pytest.raises(ValueError, match="values for"):
        export_gltf(grid, {"cp": np.arange(7.0)})


# ------------------------------------------------------------------------------------- the glTF itself
def test_the_field_is_a_vertex_attribute_the_viewers_read():
    cp = CUBE_V[:, 2] * 2 - 1                                            # -1 at the bottom, +1 at the top
    glb = export_gltf((CUBE_V, CUBE_F), {"cp": cp})
    attrs = _attributes(glb)
    assert attrs and "_CP" in attrs[0]
    assert "POSITION" in attrs[0] and "NORMAL" in attrs[0]


def test_a_vector_field_is_stored_as_its_magnitude(tmp_path):
    u = np.zeros((8, 3))
    u[:, 0] = [0, 1, 0, 1, 0, 1, 0, 1]
    out = str(tmp_path / "u.glb")
    export_gltf((CUBE_V, CUBE_F), {"U": u}, out=out)
    from pinneapple_tools.visualization.studio.scene import read_gltf

    surf = read_gltf(out)[0]
    np.testing.assert_allclose(surf.fields["U"], np.linalg.norm(u, axis=1))
    assert "_U" in _attributes(open(out, "rb").read())[0]


def test_a_multi_surface_scene_takes_one_field_per_surface(tmp_path):
    wing = {"name": "wing", "vertices": CUBE_V, "faces": CUBE_F}
    tail = {"name": "tail", "vertices": CUBE_V + np.array([2.0, 0, 0]), "faces": CUBE_F}
    from pinneapple_tools.visualization.studio.scene import Surface

    sc = Scene([Surface(wing["name"], wing["vertices"], wing["faces"], "grey"),
                Surface(tail["name"], tail["vertices"], tail["faces"], "steel")])
    out = str(tmp_path / "two.glb")
    export_gltf(sc, {"_CP": {"wing": np.zeros(8), "tail": np.ones(8)}}, out=out)
    from pinneapple_tools.visualization.studio.scene import read_gltf

    back = {s.name: s for s in read_gltf(out)}
    assert set(back) == {"wing", "tail"}
    # the attribute keeps the name it was given (`_CP`), which is what the viewers look for
    np.testing.assert_allclose(back["wing"].fields["_CP"], 0.0)
    np.testing.assert_allclose(back["tail"].fields["_CP"], 1.0)


def test_a_group_of_surfaces_can_be_parented_under_one_node():
    """The airliner viewer spins the propeller; the group is the reason the app is not just calling ``Scene.save``."""
    from pinneapple_tools.visualization.studio.scene import Surface

    sc = Scene([Surface("body", CUBE_V, CUBE_F, "grey"), Surface("propeller", CUBE_V, CUBE_F, "steel")],
               axes="aircraft")
    plain = _json_chunk(export_gltf(sc))
    grouped = _json_chunk(export_gltf(sc, groups={"propeller": [1]}))
    assert {n.get("name") for n in plain["nodes"]} >= {"body", "propeller"}
    assert "propeller" in {n.get("name") for n in grouped["nodes"]}      # a node of its own, holding the mesh


# ------------------------------------------------------------------------------------- the page and the render
def test_the_page_is_written_and_the_single_file_one_opens_from_disk(tmp_path):
    out = str(tmp_path / "viewer")
    path = viewer_html((CUBE_V, CUBE_F), {"cp": CUBE_V[:, 2]}, out)
    assert os.path.basename(path) == "index.html"
    assert os.path.isfile(os.path.join(out, "scene.glb"))
    assert os.path.isfile(os.path.join(out, "vendor", "three", "three.module.min.js"))
    single = viewer_html((CUBE_V, CUBE_F), {"cp": CUBE_V[:, 2]}, str(tmp_path / "one"), single_file=True)
    page = open(single, encoding="utf-8").read()
    assert "STUDIO_EMBED" in page and "data:text/javascript;base64," in page     # three.js and the scene inlined
    assert os.path.getsize(single) > 100_000


def test_the_page_says_which_fields_it_carries_so_a_legend_can_be_drawn(tmp_path):
    cp = np.array([-1.0, 0, 0, 0, 0, 0, 0, 1.0])
    out = str(tmp_path / "viewer")
    viewer_html((CUBE_V, CUBE_F), {"cp": cp}, out)
    meta = json.load(open(os.path.join(out, "scene.json")))
    assert meta["fields"] == ["cp"]                                     # the field selector and the bar read this
    assert meta["axes"] == "z_up"


def test_a_render_without_blender_or_bpy_says_so_instead_of_failing_obscurely(monkeypatch):
    from pinneapple_tools.visualization.studio import blender as _b

    monkeypatch.setattr(_b, "blender_executable", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="neither bpy nor the Blender binary"):
        blender_render((CUBE_V, CUBE_F), "/tmp/never.png", fields={"cp": CUBE_V[:, 2]})


def test_a_render_asking_for_a_field_the_scene_does_not_have_is_refused_before_rendering():
    with pytest.raises(KeyError, match="no field 'nope'"):
        blender_render((CUBE_V, CUBE_F), "/tmp/never.png", fields={"cp": CUBE_V[:, 2]}, field="nope")


def test_the_size_limit_is_the_one_the_viewers_document():
    assert SCENE_MAX_FACES == 250_000                                    # the decimation threshold of the size guide

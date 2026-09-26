"""pinneapple_twin3d.openfoam: ASCII OpenFOAM case -> Twin3D scene (patches, per-face values, boundary fields)."""
import json
import os

import numpy as np
import pytest

from pinneapple_twin3d.demo import pipe_bend_scene
from pinneapple_twin3d.openfoam import read_boundary, read_boundary_field, read_faces, read_points, scene_from_case

HEADER = """FoamFile
{
    version     2.0;
    format      ascii;
    class       %s;
    object      %s;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""


def _write(path, cls, obj, body):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(HEADER % (cls, obj) + body)


@pytest.fixture
def cube_case(tmp_path):
    """One hexahedral cell: 2 faces form the 'inlet' patch, 4 the 'walls' patch."""
    c = tmp_path / "case"
    pm = c / "constant" / "polyMesh"
    pts = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    _write(str(pm / "points"), "vectorField", "points",
           "8\n(\n" + "\n".join(f"({x} {y} {z})" for x, y, z in pts) + "\n)\n")
    faces = ["4(0 3 2 1)", "4(4 5 6 7)", "4(0 1 5 4)", "4(1 2 6 5)", "4(2 3 7 6)", "4(3 0 4 7)"]
    _write(str(pm / "faces"), "faceList", "faces", "6\n(\n" + "\n".join(faces) + "\n)\n")
    _write(str(pm / "boundary"), "polyBoundaryMesh", "boundary", """2
(
    inlet
    {
        type            patch;
        nFaces          2;
        startFace       0;
    }
    walls
    {
        type            wall;
        inGroups        List<word> 1(wall);
        nFaces          4;
        startFace       2;
    }
)
""")
    for t, vals in (("0", "4(1 2 3 4)"), ("0.5", "4(2 4 6 8)")):
        _write(str(c / t / "wallShearMag"), "volScalarField", "wallShearMag", f"""dimensions [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{{
    inlet
    {{
        type            calculated;
        value           uniform 0;
    }}
    walls
    {{
        type            calculated;
        value           nonuniform List<scalar> {vals};
    }}
}}
""")
    return str(c)


def test_ascii_readers(cube_case):
    assert read_points(cube_case).shape == (8, 3)
    assert [len(f) for f in read_faces(cube_case)] == [4] * 6
    b = read_boundary(cube_case)
    assert b["walls"] == {"type": "wall", "startFace": 2, "nFaces": 4}
    v = read_boundary_field(os.path.join(cube_case, "0.5", "wallShearMag"), "walls", 4)
    np.testing.assert_allclose(v, [2, 4, 6, 8])


def test_scene_from_case_per_face_values_over_time(cube_case, tmp_path):
    steps = [{2: 1.0}, {2: 2.0, 5: 3.0}]
    sc = scene_from_case(cube_case, face_values={"erosion_rate": steps}, units={"erosion_rate": "mm/year"},
                         times=[0.0, 1.0], time_unit="year")
    assert [p.name for p in sc.parts] == ["walls"]  # default: wall patches only
    part = sc.parts[0]
    assert part.vertices.shape == (16, 3) and part.faces.shape == (8, 3)  # 4 quads, vertices per face
    field = part.fields["erosion_rate"]
    assert field.shape == (2, 16)
    np.testing.assert_allclose(field[1, 12:16], 3.0)  # global face 5 = 4th wall face
    np.testing.assert_allclose(field[1, 4:12], 0.0)  # faces without a value get `fill`
    m = json.load(open(sc.export(str(tmp_path / "out"), with_viewer=False)))
    assert m["times"] == [0.0, 1.0] and m["fields"][0]["unit"] == "mm/year"


def test_scene_from_case_boundary_fields_from_time_dirs(cube_case):
    sc = scene_from_case(cube_case, boundary_fields=["wallShearMag"])
    assert list(sc.times) == [0.0, 0.5]
    f = sc.parts[0].fields["wallShearMag"]
    np.testing.assert_allclose(f[:, 0], [1.0, 2.0]) and np.testing.assert_allclose(f[:, -1], [4.0, 8.0])


def test_unknown_patch_is_reported(cube_case):
    with pytest.raises(KeyError, match="nope"):
        scene_from_case(cube_case, patches=["nope"])


def test_demo_scene_exports(tmp_path):
    m = json.load(open(pipe_bend_scene().export(str(tmp_path), with_viewer=False)))
    assert {p["name"] for p in m["parts"]} == {"bend", "inlet", "outlet"}
    wear = next(f for f in m["fields"] if f["name"] == "wear_depth")
    assert wear["steps"] == 6 and wear["min"] == 0.0 and wear["max"] > 0


def test_usd_export_text_and_optional_pxr_roundtrip(tmp_path):
    sc = pipe_bend_scene()
    sc.export(str(tmp_path), with_viewer=False, usd=True)
    text = open(tmp_path / "scene.usda").read()
    assert text.startswith("#usda 1.0") and 'def Mesh "bend"' in text
    assert "primvars:wear_depth.timeSamples" in text and "pinneapple:envelope" in text
    pxr = pytest.importorskip("pxr")  # pip install usd-core
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(tmp_path / "scene.usda"))
    wear = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/Twin/bend")).GetPrimvar("wear_depth")
    assert wear.GetInterpolation() == "vertex" and len(wear.GetAttr().GetTimeSamples()) == 6
    assert max(wear.Get(5)) == pytest.approx(1.75, rel=1e-5)

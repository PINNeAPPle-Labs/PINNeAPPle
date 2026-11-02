"""step_study (#1120): a STEP file to the bodies, field views, MeshData and baked files the apps show."""
import json

import numpy as np
import pytest

sm = pytest.importorskip("smlibpy")
pytest.importorskip("mapbox_earcut")

from pinneapple_design.geometry.brep import write_step  # noqa: E402
from pinneapple_design.geometry.brep.study import step_study  # noqa: E402


@pytest.fixture(scope="module")
def parts(tmp_path_factory):
    d = tmp_path_factory.mktemp("study")
    bx = sm.Box(60, 40, 20)
    v = [e for e in bx.edges() if abs(e.bbox().max.z - e.bbox().min.z - 20) < 1e-9]
    block = bx.fillet(v, 6, continuity=2) - sm.Cylinder(6, 20)          # G2 corners and a bore of r = 6 mm
    p = d / "parts.step"
    write_step([block, sm.Torus(20, 5).moved(sm.Pos(0, 0, 40))], str(p))
    return p


def test_report_and_files(parts, tmp_path):
    out = tmp_path / "out"
    r = step_study(str(parts), quality="normal", bake_formats=("stl", "obj"), out_dir=str(out))
    assert r["n_bodies"] == 2 and r["units"] == "m"
    block, torus = r["bodies"]
    assert block["continuity"] == {"G0": 18, "G2": 8}
    assert abs(block["curvature"]["H_min_per_m"] + 1 / (2 * 6e-3)) < 1e-6           # the bore: H = -1/(2r)
    assert abs(torus["curvature"]["H_max_per_m"] - 120.0) < 1e-6                    # (R + 2r) / (2r (R + r))
    for b in r["bodies"]:
        assert b["mesh"]["watertight"] and b["mesh"]["chordal_error_over_tolerance"] <= 1.05
    assert set(r["files"]) == {"body_0.glb", "body_0.json", "body_0.npz", "body_1.glb", "body_1.json", "body_1.npz",
                               "parts.stl", "parts.obj"}
    assert all((out / f).stat().st_size > 0 for f in r["files"])
    meta = json.loads((out / "body_0.json").read_text())
    assert meta["schema"] == "pinneapple-field-view/v1"
    assert [f["name"] for f in meta["fields"]] == ["H", "K", "zebra", "highlight"]
    assert [m["name"] for m in meta["markers"]] == ["G0 edges (18)", "G2 edges (8)"]
    assert (out / "body_0.glb").read_bytes()[:4] == b"glTF"
    from pinneapple_design.geometry.core.mesh import MeshData
    d = np.load(out / "body_0.npz")
    md = MeshData(**{k: d[k] for k in ("vertices", "faces", "normals")})
    assert md.n_faces == block["mesh"]["n_triangles"]
    assert "o parts_1" in (out / "parts.obj").read_text() and "o parts_2" in (out / "parts.obj").read_text()


def test_name_sets_the_stem_of_the_baked_files(parts, tmp_path):
    r = step_study(str(parts), quality="draft", bake_formats=("stl",), name="my bracket", out_dir=str(tmp_path))
    assert "my_bracket.stl" in r["files"] and (tmp_path / "my_bracket.stl").exists()


def test_without_out_dir_nothing_is_written(parts, tmp_path):
    r = step_study(str(parts), quality="draft", refine_view=False)
    assert r["files"] == [] and all("view" not in b for b in r["bodies"])


def test_refuses_other_files_and_large_ones(parts, tmp_path):
    with pytest.raises(ValueError, match=".step / .stp"):
        step_study(str(tmp_path / "part.stl"))
    with pytest.raises(ValueError, match="limit"):
        step_study(str(parts), max_input_mb=1e-6)

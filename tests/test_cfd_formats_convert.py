"""external_solvers.cfd_formats.convert: suffix dispatch, Abaqus surfaces (node ids mapped, tetrahedra reduced to their
boundary), and repair of a damaged STL (the source app's examples/repair_case.py, as a test)."""
from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from pinneapple_simulation.external_solvers.cfd_formats.convert import (  # noqa: E402
    convert_surface,
    detect_format,
    read_point_bundle,
    read_surface_mesh,
    tet_boundary_faces,
)

TWO_TETS = """*HEADING
two tetrahedra sharing the face 102-103-104, node ids start at 101
*NODE
101, 0.0, 0.0, 0.0
102, 1.0, 0.0, 0.0
103, 0.0, 1.0, 0.0
104, 0.0, 0.0, 1.0
105, 1.0, 1.0, 1.0
*ELEMENT, TYPE=C3D4, ELSET=SOLID
1, 101, 102, 103, 104
2, 105, 102, 103, 104
"""

SHELL = """*NODE
7, 0.0, 0.0, 0.0
8, 2.0, 0.0, 0.0
9, 0.0, 2.0, 0.0
*ELEMENT, TYPE=S3
1, 7, 8, 9
"""


def test_detect_format():
    assert detect_format("case.CGNS") == "cgns" and detect_format("m.e") == "exodus" and detect_format("x.stp") == "step"
    with pytest.raises(ValueError):
        detect_format("mesh.xyz")


def test_tet_boundary_drops_the_shared_face():
    faces = tet_boundary_faces(np.array([[0, 1, 2, 3], [4, 1, 2, 3]]))
    assert len(faces) == 6 and not any(set(f) == {1, 2, 3} for f in faces.tolist())


def test_abaqus_tets_become_a_closed_surface(tmp_path):
    p = tmp_path / "two.inp"
    p.write_text(TWO_TETS)
    mesh = read_surface_mesh(str(p))
    assert len(mesh.faces) == 6 and mesh.faces.max() == 4            # ids 101..105 -> rows 0..4
    out = convert_surface(str(p), str(tmp_path / "two.stl"), repair=True)
    assert out["faces_out"] == 6 and out["watertight"] and out["winding_consistent"]
    back = trimesh.load(str(tmp_path / "two.stl"), force="mesh")
    assert abs(back.volume) == pytest.approx(1 / 6 + 1 / 3, rel=1e-9)   # unit corner tet + the one on (1,1,1)


def test_abaqus_shell_ids_are_mapped(tmp_path):
    p = tmp_path / "shell.inp"
    p.write_text(SHELL)
    mesh = read_surface_mesh(str(p))
    assert mesh.faces.tolist() == [[0, 1, 2]]
    bundle = read_point_bundle(str(p))
    assert bundle["coords"].shape == (3, 3) and bundle["format"] == "abaqus"


def test_volume_only_format_has_no_surface(tmp_path):
    with pytest.raises(ValueError):
        read_surface_mesh(str(tmp_path / "case.cgns"))
    with pytest.raises(ValueError):
        convert_surface(str(tmp_path / "a.stl"), str(tmp_path / "a.step"))


def _damaged_icosphere():
    base = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    faces = base.faces.copy()
    rng = np.random.default_rng(7)
    faces = np.vstack([faces, faces[rng.choice(len(faces), 15, replace=False)]])          # duplicate faces
    deg = np.array([[i, i, (i + 1) % len(base.vertices)] for i in rng.choice(len(base.vertices), 10, replace=False)])
    faces = np.vstack([faces, deg])                                                         # zero-area faces
    faces[:20] = faces[:20][:, ::-1]                                                        # flipped patch
    return trimesh.Trimesh(vertices=base.vertices, faces=faces, process=False)


def test_repair_removes_duplicates_degenerates_and_fixes_winding(tmp_path):
    damaged = _damaged_icosphere()
    src = tmp_path / "damaged.stl"
    damaged.export(str(src))
    out = convert_surface(str(src), str(tmp_path / "fixed.stl"), repair=True)
    fixed = trimesh.load(str(tmp_path / "fixed.stl"), force="mesh", process=True)   # merge the STL soup
    assert len(fixed.faces) - int(fixed.unique_faces().sum()) == 0
    assert len(fixed.faces) - int(fixed.nondegenerate_faces().sum()) == 0
    assert fixed.is_winding_consistent and out["winding_consistent"]
    assert out["faces_in"] == len(damaged.faces) == 345 and out["faces_out"] == 320

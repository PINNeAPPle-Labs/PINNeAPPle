"""Studio views of 3-D machines (``machines3d.renders``): region surfaces, cut views, colour range, turntable."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pinneapple_simulation.electromagnetics.machines3d import renders as R
from pinneapple_simulation.electromagnetics.machines3d.afpm import AFPM3D, AFPMMachine


@pytest.fixture(scope="module")
def coarse():
    model = AFPM3D(AFPMMachine(), n_theta=192, dr=0.02, dz=0.008, air_r=0.01, air_z=0.008)
    nu, Br = model.materials(0.0)
    rng = np.random.default_rng(0)
    B = rng.normal(size=(len(model.mesh.tets), 3))
    return model, R.tet_labels(nu, Br), B


def _volume(s) -> float:
    """Volume enclosed by a closed, outward-oriented triangulated surface (divergence theorem)."""
    p = s.vertices[s.faces]
    return float(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6.0)


def _tet_volume(mesh, idx) -> float:
    p = mesh.nodes[mesh.tets[idx]]
    return float(np.abs(np.einsum("ij,ij->i", p[:, 1] - p[:, 0],
                                  np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))).sum() / 6.0)


def test_labels_follow_the_materials(coarse):
    model, lab, _ = coarse
    assert set(np.unique(lab)) == {R.AIR, R.IRON, R.MAGNET}
    assert np.array_equal(lab == R.MAGNET, np.isin(np.arange(len(lab)), np.flatnonzero(
        np.linalg.norm(model.materials(0.0)[1], axis=1) > 0)))


def test_surfaces_enclose_each_region_and_carry_its_own_field(coarse):
    model, lab, B = coarse
    surfs = {s.name: s for s in R.region_surfaces(model.mesh, lab, B, scale=1.0)}
    for name, code in (("iron", R.IRON), ("magnet", R.MAGNET)):
        s = surfs[name]
        idx = np.flatnonzero(lab == code)
        assert _volume(s) == pytest.approx(_tet_volume(model.mesh, idx), rel=1e-9)   # closed and outward
        v = s.fields["B"].reshape(-1, 3)
        assert np.all(v == v[:, :1])                                                  # flat: one value per face
        assert set(np.round(v[:, 0], 12)) <= set(np.round(np.linalg.norm(B[idx], axis=1), 12))


def test_cut_removes_the_sector(coarse):
    model, lab, B = coarse
    full = {s.name: s for s in R.region_surfaces(model.mesh, lab, B, scale=1.0)}
    cut = {s.name: s for s in R.region_surfaces(model.mesh, lab, B, cut=(0.0, 90.0), scale=1.0)}
    assert _volume(cut["magnet"]) == pytest.approx(0.75 * _volume(full["magnet"]), rel=1e-9)   # 2 of 8 poles
    c = cut["iron"].vertices[cut["iron"].faces].mean(axis=1)
    ang = np.degrees(np.arctan2(c[:, 1], c[:, 0])) % 360
    assert not np.any((ang > 1.0) & (ang < 89.0))                                     # nothing left inside it
    wrap = {s.name: s for s in R.region_surfaces(model.mesh, lab, B, cut=(315.0, 45.0), scale=1.0)}
    assert _volume(wrap["magnet"]) == pytest.approx(0.75 * _volume(full["magnet"]), rel=1e-9)  # across 0 deg


def test_scene_range_and_label(coarse):
    model, lab, B = coarse
    sc, info = R.machine_scene(model.mesh, lab, B)
    assert info["range"] == (info["B_min"], info["B_max"]) and sc.labels["B"] == "|B| (T)"
    sc, info = R.machine_scene(model.mesh, lab, B, clip=99.5)
    assert info["range"][1] < info["B_max"] and 0.0 < info["above"] <= 0.006
    assert sc.labels["B"] == f"|B| (T), max {info['B_max']:.2f} T off scale"
    assert sc.surfaces[0].vertices.max() > 50                                         # millimetres


def test_turntable_turns_once_and_loops(coarse):
    from pinneapple_tools.visualization.studio.animation import matrix_from_quat

    model, lab, B = coarse
    sc, _ = R.machine_scene(model.mesh, lab, B)
    R.turntable(sc, seconds=2.0, fps=12.0)
    an = sc.animation
    assert an.n_frames == 24 and an.times[-1] == pytest.approx(23 / 12)
    q = an.tracks["machine"]["quaternions"]
    assert np.allclose(matrix_from_quat(q[0]), np.eye(3))
    a = math.atan2(matrix_from_quat(q[6])[1, 0], matrix_from_quat(q[6])[0, 0])
    assert a == pytest.approx(math.pi / 2)                                            # a quarter turn in 0.5 s
    assert all(s.group == "machine" for s in sc.surfaces)


def test_fixed_camera_frames_a_turntable(coarse, tmp_path):
    pytest.importorskip("bpy")
    from PIL import Image

    from pinneapple_tools.visualization.studio import blender

    model, lab, B = coarse
    sc, info = R.machine_scene(model.mesh, lab, B, cut=(183.0, 273.0))
    R.turntable(sc, seconds=1.0, fps=2.0)
    r = blender.render_animation(sc, str(tmp_path), size=(96, 64), samples=1, camera="fixed", background="studio",
                                 field="B", field_range=info["range"], hud=False, colorbar_on=False, mp4=False)
    im = np.asarray(Image.open(r["frames"][1]).convert("RGB"), float)
    centre = im[16:48, 24:72]
    assert centre.std() > 10.0            # the machine is in the frame, not only the plain studio background

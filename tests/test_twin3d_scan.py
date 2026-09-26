"""Scan cleaning and MuJoCo (LiteReality-style) scene import for Twin3D."""
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from pinneapple_twin3d.scan import add_scan, clean_scan, scene_from_mjcf  # noqa: E402
from pinneapple_twin3d.scene import Scene  # noqa: E402


def _tilted_room_scan():
    """Box 'room' scanned in mm, lying on its side (floor normal along +Y), plus a stray fragment."""
    room = trimesh.creation.box(extents=[4000.0, 300.0, 3000.0])  # thin along Y: floor/ceiling dominate
    room.apply_translation([0, 150.0, 0])
    junk = trimesh.creation.icosphere(radius=5.0)
    junk.apply_translation([9000, 9000, 9000])
    m = trimesh.util.concatenate([room, junk])
    return trimesh.Trimesh(np.vstack([m.vertices, m.vertices[:3]]), np.vstack([m.faces, [[0, 0, 1]]]), process=False)


def test_clean_scan_drops_fragments_aligns_floor_and_converts_units():
    mesh, rep = clean_scan(_tilted_room_scan(), unit="mm")
    assert rep.components_dropped == 1
    assert abs(abs(rep.up_axis_before[1]) - 1.0) < 1e-6  # the dominant (floor) normal was along Y
    ext = mesh.bounds[1] - mesh.bounds[0]
    np.testing.assert_allclose(sorted(ext), sorted([4.0, 0.3, 3.0]), atol=1e-6)  # metres
    assert ext[2] == pytest.approx(0.3, abs=1e-6)  # thin axis is now vertical
    assert mesh.bounds[0][2] == pytest.approx(0.0, abs=1e-9)  # floor at z = 0
    assert rep.faces_out <= rep.faces_in


def test_point_cloud_is_rejected_with_a_clear_message():
    pc = trimesh.PointCloud(np.random.default_rng(0).random((50, 3)))
    with pytest.raises(ValueError, match="point cloud"):
        clean_scan(trimesh.Trimesh(vertices=pc.vertices, faces=np.zeros((0, 3), int)))


def test_add_scan_to_scene(tmp_path):
    sc = Scene("room twin")
    rep = add_scan(sc, "room", _tilted_room_scan(), unit="mm")
    assert sc.parts[0].name == "room" and rep.vertices_out == len(sc.parts[0].vertices)


def test_mjcf_scene_with_mesh_and_primitive_geoms(tmp_path):
    trimesh.creation.box(extents=[1.0, 1.0, 1.0]).export(tmp_path / "chair.obj")
    (tmp_path / "scene.xml").write_text("""
<mujoco model="room">
  <compiler meshdir="."/>
  <asset><mesh name="chair" file="chair.obj" scale="0.5 0.5 0.5"/></asset>
  <worldbody>
    <geom name="floor" type="plane" size="2 3 0.1"/>
    <body name="chair_body" pos="1 0 0" quat="0.7071068 0 0 0.7071068">
      <geom name="chair_geom" type="mesh" mesh="chair" pos="0 1 0" rgba="0.8 0.2 0.2 1"/>
    </body>
  </worldbody>
</mujoco>""")
    sc = scene_from_mjcf(str(tmp_path / "scene.xml"))
    names = [p.name for p in sc.parts]
    assert names == ["floor", "chair_geom"]
    chair = sc.parts[1]
    # body at (1,0,0) rotated 90 deg about z; geom offset (0,1,0) -> world (1,0,0) + R*(0,1,0) = (0,0,0)
    np.testing.assert_allclose(chair.vertices.mean(0), [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(chair.vertices.max(0) - chair.vertices.min(0), [0.5, 0.5, 0.5], atol=1e-6)
    assert chair.group == "chair_body" and chair.color[0] == pytest.approx(0.8)

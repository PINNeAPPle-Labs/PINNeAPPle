"""Geometry retrieval: D2 descriptors find the same shape under rotation/scale; caption index."""
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from scipy.spatial.transform import Rotation  # noqa: E402

from pinneapple_design.geometry.retrieval import GeometryIndex, load_caption_index, shape_descriptor  # noqa: E402


def _rot(seed):
    T = np.eye(4)
    T[:3, :3] = Rotation.random(random_state=seed).as_matrix()
    return T


def _library():
    return {
        "cube": trimesh.creation.box(extents=[1, 1, 1]),
        "plate": trimesh.creation.box(extents=[4, 4, 0.3]),
        "rod": trimesh.creation.cylinder(radius=0.2, height=5.0),
        "ball": trimesh.creation.icosphere(subdivisions=3, radius=1.0),
        "ring": trimesh.creation.torus(major_radius=2.0, minor_radius=0.4),
    }


def test_descriptor_is_invariant_to_rotation_translation_and_scale():
    m = trimesh.creation.box(extents=[1, 2, 3])
    a = shape_descriptor(m)
    m2 = m.copy()
    m2.apply_transform(_rot(3))
    m2.apply_scale(7.5)
    m2.apply_translation([10, -4, 2])
    assert float(a @ shape_descriptor(m2)) > 0.995


@pytest.mark.parametrize("name", ["cube", "plate", "rod", "ball", "ring"])
def test_each_shape_retrieves_itself_after_a_random_pose_change(name):
    idx = GeometryIndex()
    for pid, mesh in _library().items():
        idx.add_mesh(pid, mesh)
    q = _library()[name]
    q.apply_transform(_rot(11))
    q.apply_scale(0.37)
    assert idx.search(shape_descriptor(q), k=1)[0][0] == name


def test_caption_index_from_finalrev_style_parquet(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(4, 8)).astype(np.float32)
    emb[1] = emb[0] + 0.01  # near-duplicate part
    table = pa.table({"abc_id": ["00000001", "00000002", "00000003", "00000004"],
                      "caption": ["flanged pipe elbow with bolt holes", "pipe elbow with flange",
                                  "spur gear with 24 teeth", "rectangular mounting plate"],
                      "embedding": [list(e) for e in emb]})
    pq.write_table(table, tmp_path / "train-00000-of-00100.parquet")
    idx = load_caption_index([str(tmp_path / "train-00000-of-00100.parquet")])
    assert idx.neighbours("00000001", k=1)[0][0] == "00000002"
    hits = [pid for pid, _ in idx.search_captions("elbow flange", k=2)]
    assert set(hits) == {"00000001", "00000002"}

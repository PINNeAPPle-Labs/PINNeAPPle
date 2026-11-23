"""The mechanical detail builders: fasteners, welds and seams, and where they land in space."""
import numpy as np
import pytest

from pinneapple_tools.visualization.studio import detail as D
from pinneapple_tools.visualization.studio.scene import Scene


def test_a_bolt_is_a_closed_hexagonal_head_along_its_normal():
    s = D.hex_bolt([0, 0, 0], (0, 0, 1), head_radius=0.01, height=0.006, washer=False)
    assert s.vertices.shape[1] == 3
    assert s.faces.max() < len(s.vertices) and s.faces.min() >= 0
    assert len({f[0] for f in s.faces} | {f[1] for f in s.faces} | {f[2] for f in s.faces}) == len(s.vertices), \
        "every vertex is used by some triangle"
    top = s.vertices[:, 2].max()
    assert top == pytest.approx(0.006), "the head stands height above the surface"
    rad = np.linalg.norm(s.vertices[:, :2], axis=1)
    assert rad.max() == pytest.approx(0.01)


def test_a_bolt_follows_its_normal_and_centre():
    s = D.hex_bolt([1.0, 2.0, 3.0], (1, 0, 0), head_radius=0.02, height=0.01, washer=False)
    assert s.vertices[:, 0].max() == pytest.approx(1.01), "the head grows along +x, from the centre"
    assert s.vertices[:, 0].min() == pytest.approx(1.0), "and starts on the surface it stands on"
    # it spans the plane perpendicular to its normal, around its centre
    assert s.vertices[:, 1].mean() == pytest.approx(2.0, abs=1e-9)
    assert s.vertices[:, 2].mean() == pytest.approx(3.0, abs=1e-9)
    assert np.linalg.norm(s.vertices[:, 1:] - [2.0, 3.0], axis=1).max() == pytest.approx(0.02)


def test_a_washer_is_wider_and_shorter_than_the_head():
    bare = D.hex_bolt([0, 0, 0], (0, 0, 1), head_radius=0.01, height=0.008, washer=False)
    with_w = D.hex_bolt([0, 0, 0], (0, 0, 1), head_radius=0.01, height=0.008, washer=True)
    assert len(with_w.vertices) > len(bare.vertices)
    assert np.linalg.norm(with_w.vertices[:, :2], axis=1).max() > np.linalg.norm(bare.vertices[:, :2], axis=1).max()


def test_a_bolt_circle_puts_every_bolt_on_the_radius_it_was_given():
    bolts = D.bolt_circle([0, 0, 0], (0, 0, 1), radius=0.25, count=8, head_radius=0.01, washer=False)
    assert len(bolts) == 8
    for b in bolts:
        # the base ring sits on the circle; its vertices straddle it by the head radius and no further
        r = np.linalg.norm(b.vertices[:6, :2], axis=1)
        assert r.min() >= 0.25 - 0.01 - 1e-9 and r.max() <= 0.25 + 0.01 + 1e-9
    centres = np.array([b.vertices[:6].mean(axis=0) for b in bolts])
    assert np.allclose(np.linalg.norm(centres[:, :2], axis=1), 0.25)
    assert np.linalg.norm(centres[1, :2] - centres[0, :2]) == pytest.approx(2 * 0.25 * np.sin(np.pi / 8))


def test_bolt_placement_is_reproducible():
    a = D.bolt_circle([0, 0, 0], (0, 1, 0), radius=0.1, count=4)
    b = D.bolt_circle([0, 0, 0], (0, 1, 0), radius=0.1, count=4)
    assert all(np.allclose(x.vertices, y.vertices) for x, y in zip(a, b))


def test_a_weld_is_one_surface_that_passes_through_every_point():
    pts = np.array([[0, 0, 0], [0.1, 0, 0], [0.2, 0.05, 0]])
    w = D.weld_bead(pts, radius=0.004)
    assert w.faces.max() < len(w.vertices)
    for p in pts:                                        # each corner of the path is inside the bead
        d = np.linalg.norm(w.vertices - p, axis=1).min()
        assert d <= 0.004 + 1e-9
    # one surface, not a list: a weld is called out as a run
    assert w.name == "weld" and w.material == "steel"


def test_a_weld_needs_two_points():
    with pytest.raises(ValueError):
        D.weld_bead([[0, 0, 0]])


def test_a_seam_ring_is_a_torus_of_the_radius_and_thickness_asked_for():
    s = D.seam_ring([0, 0, 0], (0, 0, 1), radius=0.1, thickness=0.005, n=32)
    r = np.linalg.norm(s.vertices[:, :2], axis=1)
    assert r.min() == pytest.approx(0.1 - 0.005, abs=1e-9) and r.max() == pytest.approx(0.1 + 0.005, abs=1e-9)
    assert np.abs(s.vertices[:, 2]).max() == pytest.approx(0.005, abs=1e-9)
    assert s.faces.max() < len(s.vertices)


def test_detail_survives_a_gltf_round_trip(tmp_path):
    s = D.hex_bolt([0.0, 0.0, 0.5], (0, 0, 1))
    sc = Scene(surfaces=[s], materials={"steel": {"color": [0.8, 0.8, 0.8], "metallic": 1.0, "roughness": 0.3}})
    lo, hi = sc.bounds()
    assert lo[2] == pytest.approx(0.5) and hi[2] == pytest.approx(0.506)
    path = str(tmp_path / "bolt.glb")
    sc.save(path)
    back = Scene.from_file(path)
    assert len(back.surfaces[0].vertices) == len(s.vertices)
    assert back.surfaces[0].faces.shape == s.faces.shape

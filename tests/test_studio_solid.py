"""Solids, the manifold check and the kernel: a hole that is cut is not a hole that is drawn."""
import numpy as np
import pytest

from pinneapple_tools.visualization.studio import blender as B
from pinneapple_tools.visualization.studio import solid as S
from pinneapple_tools.visualization.studio.scene import Scene, disc_surface, rod_surface


def test_a_closed_primitive_is_closed_and_an_open_shell_is_not():
    assert S.is_closed(S.solid_drum([0, 0, 0], 0.05, 0.1, n=32))
    assert S.is_closed(S.solid_box([0, 0, 0], [0.1, 0.1, 0.1]))
    assert S.is_closed(S.cutter_cylinder([0, 0, 0], [0, 0, 0.1], 0.01, n=16))
    assert not S.is_closed(disc_surface([0, 0, 0], 0.05, n=24)), "a flat disc has a boundary"
    assert S.open_edges(disc_surface([0, 0, 0], 0.05, n=24)) == 24


def test_a_capped_rod_is_a_solid_which_is_why_it_can_be_an_operand():
    assert S.is_closed(rod_surface([0, 0, 0], [0, 0, 0.1], 0.02, n=16))


def test_no_cap_vertex_sits_on_the_axis_of_a_drum():
    d = S.solid_drum([0, 0, 0], 0.05, 0.1, n=32)
    r = np.linalg.norm(d.vertices[:, :2], axis=1)
    assert r.min() > 0.04, "a cutter on the axis meets a fan centre as a degenerate corner, and the cut goes wrong"


def test_a_loft_makes_a_shape_no_primitive_gives_and_caps_it_into_a_solid():
    a = S.circle_profile([0, 0, 0], 0.05, n=24)
    b = S.rounded_rect_profile([0, 0, 0.2], 0.09, 0.06, n=24, roundness=0.55)
    duct = S.loft([a, b], name="transicao")
    assert S.is_closed(duct) and len(duct.vertices) == 48
    assert duct.vertices[:, 2].max() == pytest.approx(0.2)
    assert np.linalg.norm(duct.vertices[:24, :2], axis=1).max() == pytest.approx(0.05)
    with pytest.raises(ValueError):
        S.loft([a])
    with pytest.raises(ValueError):
        S.loft([a, S.circle_profile([0, 0, 0.1], 0.05, n=12)])


def test_a_boolean_refuses_open_operands_instead_of_making_a_mess():
    sc = Scene(surfaces=[disc_surface([0, 0, 0], 0.05)])
    with pytest.raises(ValueError, match="open"):
        S.boolean(sc, [S.cutter_cylinder([0, 0, -0.1], [0, 0, 0.1], 0.01)])
    good = Scene(surfaces=[S.solid_drum([0, 0, 0], 0.05, 0.05)])
    with pytest.raises(ValueError, match="cutter"):
        S.boolean(good, [disc_surface([0, 0, 0], 0.01)])


@pytest.mark.skipif(B.blender_executable() is None, reason="needs a Blender binary on this machine")
def test_a_cut_through_the_axis_leaves_a_solid_with_a_hole():
    body = Scene(surfaces=[S.solid_drum([0, 0, 0], 0.08, 0.06, n=64)])
    before = len(body.surfaces[0].faces)
    out = S.boolean(body, [S.cutter_cylinder([0, 0, -0.02], [0, 0, 0.08], 0.02, n=48)])
    cut = out.surfaces[0]
    assert len(cut.faces) > before, "the interior wall of the hole is new geometry"
    assert S.is_closed(cut), "a hole does not open the solid"

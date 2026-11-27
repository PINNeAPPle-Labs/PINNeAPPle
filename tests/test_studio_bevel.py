"""The bevel: the one studio operation that edits the body instead of adding a surface beside it."""
import numpy as np
import pytest

from pinneapple_tools.visualization.studio import bevel as BV
from pinneapple_tools.visualization.studio import blender as B
from pinneapple_tools.visualization.studio.scene import Surface, rod_surface, Scene


def _part():
    sc = Scene(surfaces=[rod_surface([-0.1, 0, 0], [0.1, 0, 0], 0.05, name="rod", n=16)])
    sc.materials = {"metal": {"color": [0.8, 0.8, 0.8], "metallic": 1.0, "roughness": 0.3}}
    return sc


def test_the_defaults_are_a_machined_edge_and_not_a_razor_line():
    assert BV.DEFAULTS["segments"] >= 2, "one segment is a chamfer; more reads as a rounded edge"
    assert 0 < BV.DEFAULTS["width"] < 0.01, "a chamfer is small next to the part"
    assert 0 < BV.DEFAULTS["angle"] < 90, "the limit keeps the facets of a curved wall out of the bevel"


def test_a_bevel_needs_a_width_and_a_segment():
    sc = _part()
    with pytest.raises(ValueError):
        BV.bevel_scene(sc, width=0.0)
    with pytest.raises(ValueError):
        BV.bevel_scene(sc, segments=0)
    with pytest.raises(ValueError):
        BV.bevel_scene(sc, angle=200.0)


def test_a_bevel_refuses_to_silently_drop_the_fields():
    sc = _part()
    sc.surfaces.append(Surface("with_field", *[sc.surfaces[0].vertices, sc.surfaces[0].faces], "metal",
                               fields={"T": np.zeros(len(sc.surfaces[0].vertices))}))
    with pytest.raises(ValueError, match="fields"):
        BV.bevel_scene(sc, width=0.002)


@pytest.mark.skipif(B.blender_executable() is None, reason="needs a Blender binary on this machine")
def test_the_edge_is_cut_and_the_part_stays_the_same_size(tmp_path):
    out = BV.bevel_scene(_part(), width=0.004, segments=2)
    assert [s.name for s in out.surfaces] == ["rod"], "a bevel changes the skin, not the parts"
    assert len(out.surfaces[0].faces) > len(_part().surfaces[0].faces), "the chamfer adds faces"
    lo, hi = out.bounds()
    ref_lo, ref_hi = _part().bounds()
    assert np.allclose(lo, ref_lo, atol=0.01) and np.allclose(hi, ref_hi, atol=0.01), \
        "cutting an edge removes material, it does not grow the part"
    assert list(out.materials) == ["metal"], "the materials come back with the body"


@pytest.mark.skipif(B.blender_executable() is None, reason="needs a Blender binary on this machine")
def test_a_zero_angle_limit_cuts_every_edge_and_a_flat_wall_cuts_none():
    flat = Scene(surfaces=[rod_surface([-0.1, 0, 0], [0.1, 0, 0], 0.05, name="rod", n=48)])
    flat.materials = {"metal": {"color": [1, 1, 1], "metallic": 0.0, "roughness": 0.5}}
    none_cut = BV.bevel_scene(flat, angle=179.0, width=0.004)     # only edges sharper than 179 deg: the end rims
    every_cut = BV.bevel_scene(flat, angle=1.0, width=0.004)      # everything, facets included
    assert len(none_cut.surfaces[0].faces) < len(every_cut.surfaces[0].faces)

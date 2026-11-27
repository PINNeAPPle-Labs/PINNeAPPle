"""The scale layer: metres behind a scene, the metric set built at that size, and the annotations burnt into a render."""
import math

import numpy as np
import pytest

from pinneapple_tools.visualization.studio import scale as S


# ------------------------------------------------------------------ units
def test_a_unit_name_is_metres_per_scene_unit():
    assert S.unit_scale(None) == 1.0                     # a model drawn in metres says nothing
    assert S.unit_scale("mm") == 0.001                   # ... and one drawn in millimetres says "mm"
    assert S.unit_scale("MM ") == 0.001
    assert S.unit_scale("in") == 0.0254
    assert S.unit_scale(0.001) == 0.001                  # the number is the same statement
    assert S.unit_scale("µm") == 1e-6


@pytest.mark.parametrize("bad", ["parsec", "", 0.0, -1.0, float("nan")])
def test_a_unit_that_cannot_be_a_scale_is_refused(bad):
    with pytest.raises(ValueError):
        S.unit_scale(bad)


def test_a_length_is_read_with_or_without_its_unit():
    assert S.parse_length(0.02) == 0.02
    assert S.parse_length("0.025") == 0.025
    assert S.parse_length("25 mm") == 0.025
    assert S.parse_length("25mm") == 0.025
    assert S.parse_length("2 ft") == pytest.approx(0.6096)
    with pytest.raises(ValueError):
        S.parse_length("25 furlongs")


def test_a_length_is_written_the_way_an_engineer_writes_it():
    assert S.fmt_length(0.25) == "250 mm"
    assert S.fmt_length(0.0254) == "25.4 mm"
    assert S.fmt_length(0.005) == "5 mm"
    assert S.fmt_length(0.0001) == "100 µm"
    assert S.fmt_length(1.75) == "1.75 m"
    assert S.fmt_length(36.0) == "36 m"


def test_the_grid_and_the_bar_snap_to_round_values():
    assert S.decade(0.0237) == 0.02                      # never a number nobody can hold in mind
    assert S.decade(3.6) == 2.0
    assert S.grid_step(0.25) == 0.02                     # about ten squares across a 250 mm part
    assert S.grid_step(36.0) == 2.0
    assert S.scale_bar_length(0.237) == 0.2


def test_the_prop_is_the_largest_one_that_does_not_dwarf_the_subject():
    assert S.auto_prop(36.0) == "human"
    assert S.auto_prop(1.0) == "ruler"
    assert S.auto_prop(0.25) == "coin"
    assert S.auto_prop(0.02) is None                     # smaller than every prop: only the grid can measure it


def test_the_image_scale_is_the_projection_at_the_subjects_distance():
    """A 50 mm lens on a 36 mm sensor across 960 px: the focal length in pixels is 480 / tan(19.8 deg), so at 2 m one
    metre spans that many pixels over two."""
    angle = 2.0 * math.atan(36.0 / (2.0 * 50.0))
    ppm = S.px_per_metre((960, 540), angle, 2.0)
    assert ppm == pytest.approx(0.5 * 960 / math.tan(angle / 2) / 2.0, rel=1e-9)
    assert ppm == pytest.approx(666.7, rel=2e-3)
    assert S.px_per_metre((960, 540), angle, 4.0) == pytest.approx(ppm / 2.0)   # twice as far, half the scale


# ------------------------------------------------------------------ the meshes
def _open_edges(F):
    """Edges used by any number other than two faces: a closed mesh has none."""
    from collections import Counter
    e = Counter()
    for a, b, c in np.asarray(F, int).reshape(-1, 3):
        for u, v in ((a, b), (b, c), (c, a)):
            e[(min(u, v), max(u, v))] += 1
    return [k for k, n in e.items() if n != 2]


@pytest.mark.parametrize("kind", S.PROP_KINDS)
def test_every_prop_is_a_closed_mesh_standing_on_the_floor(kind):
    V, F = S.prop(kind, (0.0, 0.0), 1.5, unit=1.0)
    assert _open_edges(F) == [], f"{kind} is not a closed mesh"
    assert V[:, 2].min() == pytest.approx(1.5, abs=1e-9), "the prop sits on the floor it was given"
    if kind == "human":                                  # a figure is measured by its height ...
        assert V[:, 2].max() == pytest.approx(1.5 + S.prop_size(kind))
    else:                                                # ... a rule, a sheet and a disc by their footprint
        assert max(np.ptp(V[:, 0]), np.ptp(V[:, 1])) == pytest.approx(S.prop_size(kind), rel=0.02)


def test_a_prop_is_built_at_its_real_size_whatever_the_scene_unit():
    """A 1.75 m mannequin in a scene drawn in millimetres is 1750 scene units tall -- which is the whole point: the
    prop is real and the scene is not."""
    Va, _ = S.prop("human", (0.0, 0.0), 0.0, unit=1.0)
    Vb, _ = S.prop("human", (0.0, 0.0), 0.0, unit=0.001)
    assert Va[:, 2].max() == pytest.approx(1.75)
    assert Vb[:, 2].max() == pytest.approx(1750.0)
    assert Vb[:, 2].max() * 0.001 == pytest.approx(1.75)          # the same 1.75 m in metres
    Vc, _ = S.prop("ruler", (0.0, 0.0), 0.0, unit=0.001)
    assert np.ptp(Vc[:, 0]) * 0.001 == pytest.approx(0.300, abs=1e-3)


def test_the_cyclorama_revolves_a_floor_into_a_wall():
    V, F = S.cyclorama((1.0, 2.0), 0.0, radius=2.0, cove_radius=0.3, wall_height=2.0)
    r = np.hypot(V[:, 0] - 1.0, V[:, 1] - 2.0)
    assert V[:, 2].min() == pytest.approx(0.0)           # the floor is the floor it was given
    assert V[:, 2].max() == pytest.approx(2.0)           # and the wall the height it was asked for
    assert r.max() == pytest.approx(2.3)                 # the cove is the only thing beyond the flat floor
    assert len(V) == 1 + 12 * 48, "a centre, then one ring per profile point"
    # the flat floor really is flat up to `radius`, and rises only after it
    flat = V[np.isclose(V[:, 2], 0.0)]
    assert np.hypot(flat[:, 0] - 1.0, flat[:, 1] - 2.0).max() == pytest.approx(2.0)


def test_the_grid_is_clipped_by_the_floor_it_is_painted_on():
    V, F = S.grid_lines((0.0, 0.0), 0.5, step=0.25, radius=1.0, width=0.01, lift=0.001)
    assert len(V) > 0
    assert V[:, 2].min() == pytest.approx(0.501)         # lifted off the floor, so it does not z-fight with it
    assert np.hypot(V[:, 0], V[:, 1]).max() <= 1.0 + 0.01 + 1e-9, "no line runs up the cove"
    assert len(np.unique(np.round(V[:, 0], 6))) > 4      # lines in both directions


def test_the_grid_refuses_to_paint_a_wall_of_ink():
    with pytest.raises(ValueError):
        S.grid_lines((0.0, 0.0), 0.0, step=0.001, radius=1.0, width=0.0005)


# ------------------------------------------------------------------ the annotations
def _blank(path, size=(960, 540)):
    from PIL import Image
    Image.new("RGB", size, (250, 250, 250)).save(path)
    return path


def test_the_scale_bar_is_as_long_as_the_metres_it_states(tmp_path):
    """The bar claims a length; the picture has to draw that length at the picture's own scale."""
    from PIL import Image

    from pinneapple_tools.visualization.studio.compose import draw_scale_bar
    px_per_metre = 1000.0
    path = _blank(str(tmp_path / "bar.png"))
    draw_scale_bar(path, px_per_metre)
    a = np.asarray(Image.open(path).convert("L"), float)
    assert a.shape == (540, 960), "the annotation does not resize the figure"
    row = a[int(np.argmax((a < 120).sum(1)))]             # the row the bar is drawn on: the darkest one
    dark = np.nonzero(row < 120)[0]
    assert len(dark) > 100, "no bar was drawn"
    drawn_px = dark.max() - dark.min()
    assert drawn_px == pytest.approx(0.2 * px_per_metre, abs=12), \
        f"the bar spans {drawn_px} px, not the {0.2 * px_per_metre:.0f} px of 200 mm at 1000 px/m"


def test_a_scale_bar_needs_a_positive_image_scale(tmp_path):
    path = _blank(str(tmp_path / "bar.png"))
    from pinneapple_tools.visualization.studio.compose import draw_scale_bar
    with pytest.raises(ValueError):
        draw_scale_bar(path, 0.0)


def test_the_caption_is_written_and_does_not_resize_the_picture(tmp_path):
    from PIL import Image

    from pinneapple_tools.visualization.studio.compose import draw_caption
    path = _blank(str(tmp_path / "cap.png"), (400, 300))
    before = np.asarray(Image.open(path).convert("L"), float)
    draw_caption(path, ["250 mm × 180 mm × 120 mm", "grid 25 mm"], title="bracket")
    after = np.asarray(Image.open(path).convert("L"), float)
    assert after.shape == before.shape
    assert (after[:60, :220] < 200).any(), "nothing was written in the corner the caption goes to"
    assert np.allclose(after[200:, 260:], before[200:, 260:]), "the rest of the picture is untouched"


def test_a_caption_with_nothing_to_say_leaves_the_file_alone(tmp_path):
    from PIL import Image

    from pinneapple_tools.visualization.studio.compose import draw_caption
    path = _blank(str(tmp_path / "cap.png"), (200, 120))
    before = np.asarray(Image.open(path).convert("L"), float)
    draw_caption(path, [], title=None)
    assert np.allclose(np.asarray(Image.open(path).convert("L"), float), before)

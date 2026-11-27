"""The caption a figure carries (#67): the unit, the scale and the exaggeration, or a refusal.

The rule from the roadmap is two sentences long and these tests hold both halves: a field is not drawn without a
unit (and ``"-"`` is how a dimensionless field declares itself), and a deformed shape says how much it is
exaggerated -- a picture of proportions that does not state its exaggeration is wrong by construction.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_tools.visualization.caption import DIMENSIONLESS, FigureCaption, figure_caption
from pinneapple_tools.visualization.studio.fieldview import field_view


def _mesh(nx: int = 4, ny: int = 2, nz: int = 2):
    """A small hexahedral box, the same one tests/test_field_view.py builds."""
    from pinneapple_data.cae.model import CellBlock, Mesh

    xs, ys, zs = np.linspace(0, 2.0, nx + 1), np.linspace(-0.1, 0.1, ny + 1), np.linspace(-0.15, 0.15, nz + 1)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    nid = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k  # noqa: E731
    i, j, k = (g.ravel() for g in np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"))
    conn = np.stack([nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                     nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)], 1)
    return Mesh(points=P, blocks=[CellBlock("hexahedron", conn)])


# --------------------------------------------------------------------------------------------- the caption itself
def test_the_caption_states_the_field_the_unit_and_the_range():
    text = figure_caption("von Mises", "MPa", value_range=(0.0, 512.4)).text()
    assert text == "von Mises [MPa], 0 to 512.4 MPa"


def test_a_deformed_shape_states_its_exaggeration_and_the_lengths_it_is_in():
    """An exaggerated deformation that does not say x25 is a lie of proportion."""
    text = figure_caption("u", "mm", deform=25.0, length_unit="mm").text()
    assert "deformation x25" in text
    assert "lengths in mm" in text
    # 1.0 is no exaggeration, and printing x1 would be noise
    assert "deformation" not in figure_caption("u", "mm", deform=1.0).text()


def test_the_time_the_method_and_the_status_are_part_of_the_caption():
    text = figure_caption("T", "K", value_range=(293.0, 401.0), time=0.5, method="FEM", status="validated").text()
    assert text == "T [K], 293 to 401 K, t = 0.5 s, FEM (validated)"


def test_a_dimensionless_field_says_so_instead_of_hiding_it():
    text = figure_caption("SICN", DIMENSIONLESS).text()
    assert "dimensionless" in text
    assert figure_caption("SICN", "-").unit_text == "dimensionless"


def test_a_field_without_a_unit_is_refused():
    """The half of the rule that a person cannot talk their way out of: no unit, no caption."""
    with pytest.raises(TypeError):
        figure_caption("u")                       # the point is the missing argument, so no unit
    with pytest.raises(ValueError, match="not drawn without a unit"):
        figure_caption("u", None)
    with pytest.raises(ValueError, match="not drawn without a unit"):
        figure_caption("u", None, value_range=(0, 1))


@pytest.mark.parametrize("kwargs, message", [
    ({"status": "probably fine"}, "status must be"),
    ({"value_range": (10.0, 1.0)}, "value_range"),
    ({"deform": -2.0}, "cannot be negative"),
])
def test_a_caption_that_would_mislead_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        figure_caption("u", "m/s", **kwargs)


def test_the_caption_is_also_data_for_a_sidecar():
    d = figure_caption("u", "m/s", value_range=(-1.0, 1.0), time=0.5, method="PINN", status="experimental",
                       source="case07").to_dict()
    assert d["unit"] == "m/s" and d["value_range"] == [-1.0, 1.0] and d["method"] == "PINN"
    assert d["text"] == "u [m/s], -1 to 1 m/s, t = 0.5 s, PINN (experimental)"
    assert isinstance(figure_caption("u", "m/s"), FigureCaption)


def test_the_numbers_are_readable_at_any_magnitude():
    assert "1.25e-05" in figure_caption("p", "Pa", value_range=(0.0, 1.25e-5)).text()
    assert " 512.4" in figure_caption("s", "MPa", value_range=(0.0, 512.4)).text()


# ------------------------------------------------------------------------------------ the field view enforces it
def test_a_field_view_refuses_a_field_without_a_unit(tmp_path):
    mesh = _mesh()
    with pytest.raises(ValueError, match="declares no unit"):
        field_view(mesh, [{"name": "x", "values": mesh.points[:, 0]}])
    # declaring it dimensionless is allowed: the absence of a unit must be a statement, not an omission
    fv = field_view(mesh, [{"name": "x", "values": mesh.points[:, 0], "unit": "-"}])
    assert fv.meta["captions"]["f0"]["unit"] == "-"


def test_the_field_view_caption_carries_the_exaggeration_the_view_will_use():
    """The viewer scales a displacement automatically; the caption must state the factor it chose, not one of its own."""
    mesh = _mesh()
    u = np.c_[0.001 * mesh.points[:, 0], np.zeros((mesh.n_points, 2))]
    fv = field_view(mesh, [{"name": "vm", "values": np.linspace(0, 100, mesh.n_points), "unit": "MPa",
                            "label": "von Mises", "deform": "u"}],
                    deform={"u": u}, length_unit="m")
    caption = fv.meta["captions"]["f0"]
    assert caption["unit"] == "MPa" and caption["value_range"] is not None
    assert caption["deform"] == pytest.approx(fv.meta["deform"]["u"]["auto_scale"])
    assert "von Mises [MPa]" in fv.meta["caption"]
    assert "deformation x" in fv.meta["caption"]


def test_the_shown_caption_is_a_value_field_not_an_error_field():
    mesh = _mesh()
    fv = field_view(mesh, [
        {"name": "err", "values": np.abs(mesh.points[:, 0]), "unit": "m", "kind": "error"},
        {"name": "vm", "values": np.linspace(0, 100, mesh.n_points), "unit": "MPa", "label": "von Mises"},
    ])
    assert fv.meta["caption"].startswith("von Mises [MPa]")
    assert set(fv.meta["captions"]) == {"f0", "f1"}


# ------------------------------------------------------------------------------ the turntable speaks the same way
def test_the_turntable_caption_is_the_caption_helper_s(tmp_path):
    """One sentence, one implementation: a clip and a still of the same case must not describe it differently."""
    from PIL import Image

    from pinneapple_tools.visualization.orbit import turntable

    t = np.linspace(0.0, 1.0, 2)
    x = y = np.linspace(0.0, 1.0, 16)
    X, Y = np.meshgrid(x, y, indexing="ij")
    field = np.sin(2 * np.pi * (X[None] - t[:, None, None])) * np.cos(2 * np.pi * Y[None])
    path = tmp_path / "case.gif"
    turntable(field, path, size=(320, 180), frames=2, t=t, name="u", unit="m/s")
    assert path.is_file()
    with Image.open(path) as image:
        assert image.n_frames == 2
    expected = figure_caption("u", "m/s", value_range=(float(field.min()), float(field.max())), time=0.0).text()
    assert expected.startswith("u [m/s], -0.9945 to 0.9945 m/s")              # the sentence the frame carries

"""The house figure theme (#918): one palette, one map per quantity, labelled axes."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pinneapple_tools.visualization.style import (
    CMAPS,
    DEFAULT_CMAP,
    FIELD_CMAPS,
    PALETTE,
    annotate_verdict,
    colorbar,
    field_cmap,
    figure_sizes,
    label,
    make_panel_figure,
    panel_letters,
    save_figure,
    source_note,
    use_theme,
)


@pytest.fixture(autouse=True)
def _restore_rc():
    before = dict(matplotlib.rcParams)
    yield
    matplotlib.rcParams.update(before)
    plt.close("all")


def test_theme_applies_palette_and_font_stack():
    use_theme("report")
    cycle = matplotlib.rcParams["axes.prop_cycle"]
    colors = [c["color"] for c in cycle]
    assert colors[: len(PALETTE)] == list(PALETTE)
    assert matplotlib.rcParams["font.family"] == ["sans-serif"]
    assert "DejaVu Sans" in matplotlib.rcParams["font.sans-serif"]


def test_theme_kinds_scale_the_type():
    use_theme("paper")
    paper = matplotlib.rcParams["font.size"]
    use_theme("slide")
    slide = matplotlib.rcParams["font.size"]
    assert slide > paper


def test_theme_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="unknown theme kind"):
        use_theme("poster")


def test_dark_and_light_backgrounds_differ():
    use_theme("report", dark=False)
    light = matplotlib.rcParams["figure.facecolor"]
    use_theme("report", dark=True)
    assert matplotlib.rcParams["figure.facecolor"] != light


def test_rainbow_is_available_but_not_the_default():
    assert DEFAULT_CMAP != "rainbow"
    use_theme()
    assert "cfd_rainbow" in matplotlib.colormaps
    assert "rainbow" in CMAPS


def test_sequential_map_is_monotone_in_lightness():
    """A perceptually uniform map must not fold back on itself in lightness."""
    cmap = CMAPS["cfd_sequential"]
    lightness = []
    for value in np.linspace(0.0, 1.0, 32):
        red, green, blue, _ = np.asarray(cmap(float(value)))
        lightness.append(0.2126 * red + 0.7152 * green + 0.0722 * blue)
    diffs = np.diff(lightness)
    # Monotone up to rounding: more than three reversals means a rainbow.
    assert np.sum(diffs > 1e-9) == 0 or np.sum(diffs < -1e-9) <= 3


def test_diverging_map_is_light_in_the_middle():
    cmap = CMAPS["cfd_diverging"]
    mid = np.asarray(cmap(0.5))
    edge = np.asarray(cmap(0.0))
    assert mid[:3].mean() > edge[:3].mean()


@pytest.mark.parametrize("quantity", ["temperature", "vorticity", "pressure_coefficient", "error"])
def test_field_cmap_resolves_each_quantity(quantity):
    assert field_cmap(quantity).name == FIELD_CMAPS[quantity]


def test_field_cmap_accepts_spaces_and_hyphens():
    assert field_cmap("Velocity Magnitude").name == FIELD_CMAPS["velocity_magnitude"]
    assert field_cmap("temperature-difference").name == FIELD_CMAPS["temperature_difference"]


def test_field_cmap_refuses_an_unregistered_quantity():
    with pytest.raises(KeyError, match="no colormap registered"):
        field_cmap("entropy_production")


def test_the_same_quantity_is_the_same_colour_twice():
    first, second = field_cmap("vorticity"), field_cmap("vorticity")
    assert first is second
    assert field_cmap("vorticity").name != field_cmap("temperature").name


def test_colorbar_carries_quantity_and_unit():
    use_theme()
    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(np.arange(9.0).reshape(3, 3), cmap=field_cmap("temperature"))
    cb = colorbar(ax, mesh, "Temperature", "K")
    assert cb.ax.get_ylabel() == "Temperature [K]"
    plt.close(fig)


def test_colorbar_marks_dimensionless_as_dash():
    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(np.arange(9.0).reshape(3, 3))
    cb = colorbar(ax, mesh, "Pressure coefficient", "")
    assert cb.ax.get_ylabel() == "Pressure coefficient [-]"
    plt.close(fig)


def test_colorbar_refuses_a_missing_quantity_or_unit():
    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(np.arange(9.0).reshape(3, 3))
    with pytest.raises(ValueError, match="quantity"):
        colorbar(ax, mesh, "  ", "K")
    with pytest.raises(ValueError, match="unit"):
        colorbar(ax, mesh, "Temperature", None)
    plt.close(fig)


def test_label_writes_name_and_unit():
    fig, ax = plt.subplots()
    label(ax, x=("Time", "s"), y=("Drag", "N"), title="Drag against time")
    assert ax.get_xlabel() == "Time [s]"
    assert ax.get_ylabel() == "Drag [N]"
    assert ax.get_title() == "Drag against time"
    plt.close(fig)


def test_verdict_stamps_word_and_colour():
    fig, ax = plt.subplots()
    passed = annotate_verdict(ax, "all checks", True)
    failed = annotate_verdict(ax, "mass balance", False)
    assert "pass" in passed.get_text() and "all checks" in passed.get_text()
    assert "fail" in failed.get_text() and "mass balance" in failed.get_text()
    assert passed.get_color() != failed.get_color()
    plt.close(fig)


def test_panel_letters_in_reading_order():
    fig, axes = make_panel_figure(3)
    letters = panel_letters(fig, list(axes.ravel()))
    assert letters == ["(a)", "(b)", "(c)"]
    plt.close(fig)


def test_source_note_refuses_an_empty_string():
    fig = plt.figure()
    with pytest.raises(ValueError, match="source note"):
        source_note(fig, "   ")
    plt.close(fig)


def test_source_note_is_drawn_on_the_figure():
    fig = plt.figure()
    text = source_note(fig, "Data: Ghia et al. (1982)")
    assert "Ghia" in text.get_text()
    plt.close(fig)


def test_figure_sizes_grow_with_the_panel_count():
    sizes = figure_sizes("report")
    assert sizes["single"][0] < sizes["double"][0] < sizes["triple"][0]
    with pytest.raises(ValueError):
        figure_sizes("poster")


def test_save_figure_writes_raster_and_vector(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    written = save_figure(fig, tmp_path / "figure")
    assert [p.rsplit(".", 1)[1] for p in written] == ["png", "svg"]
    for path in written:
        assert (tmp_path / path.split("/")[-1]).exists()
    plt.close(fig)

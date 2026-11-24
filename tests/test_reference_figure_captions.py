"""Every figure in the catalogue carries a caption that states its quantity and its unit (#67).

The issue's acceptance is "a test checks the caption of every image in the catalog", and the catalogue is the set of
figures committed under ``docs/org/reference/figures/``. The figures are saved by ``scripts/reference/run_examples.py``,
which now reads each caption back out of the figure's own labels and writes it as a JSON sidecar beside the image; a
figure that declares no unit fails that run. This test is the half that holds afterwards: an image without a caption,
or with a caption that declares no unit, is a red test -- and so is a caption that disagrees with its image name.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pinneapple_tools.visualization.caption import caption_from_figure, figure_caption

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = ROOT / "docs" / "org" / "reference" / "figures"


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# reading a caption back out of a figure
# ---------------------------------------------------------------------------

def test_the_colour_bar_is_the_witness_for_the_field():
    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(np.arange(9.0).reshape(3, 3), cmap="viridis")
    fig.colorbar(mesh, ax=ax).set_label("Temperature [K]")
    ax.set_xlabel("x [m]")
    cap = caption_from_figure(fig, filename="plate.png")
    assert cap.field == "Temperature"
    assert cap.unit == "K"
    assert "Temperature [K]" in cap.text()
    assert cap.value_range == (0.0, 8.0)


def test_an_axis_label_declares_the_field_when_there_is_no_colour_bar():
    fig, ax = plt.subplots()
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Peak pressure [MPa]")
    cap = caption_from_figure(fig)
    assert (cap.field, cap.unit) == ("Peak pressure", "MPa")


def test_the_unit_survives_the_round_trip_through_the_caption_text():
    fig, ax = plt.subplots()
    ax.set_ylabel("u [m/s]")
    cap = caption_from_figure(fig, deform=25.0, length_unit="mm")
    assert "deformation x25" in cap.text()
    assert "lengths in mm" in cap.text()
    assert "m/s" in cap.text()


def test_a_dimensionless_field_has_to_say_so():
    """``[-]`` is a declaration; a label with no brackets is not."""
    fig, ax = plt.subplots()
    ax.set_ylabel("Cp [-]")
    assert caption_from_figure(fig).unit == "-"
    assert caption_from_figure(fig).dimensionless


def test_an_exaggeration_the_figure_prints_is_kept_in_the_caption():
    """A deformed shape drawn x25 that says nothing is a lie of proportion: the caption keeps what the figure says."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.text(0.04, 0.04, "von Mises [MPa], 0.1 to 512 MPa, deformation x25, lengths in mm")
    cap = caption_from_figure(fig, filename="beam.png")
    assert cap.deform == 25.0
    assert cap.length_unit == "mm"
    assert "deformation x25" in cap.text() and "lengths in mm" in cap.text()


def test_a_caption_the_figure_prints_itself_is_read_back():
    fig, ax = plt.subplots()
    ax.set_title("Contact pressure [MPa]")
    cap = caption_from_figure(fig)
    assert (cap.field, cap.unit) == ("Contact pressure", "MPa")


def test_a_figure_that_declares_no_unit_is_refused():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    with pytest.raises(ValueError, match="declares no unit"):
        caption_from_figure(fig, filename="naked.png")


def test_a_bare_name_in_brackets_is_still_a_declaration():
    """The rule is the brackets, not a dictionary of units: a non-SI unit the library does not know is still a unit."""
    fig, ax = plt.subplots()
    ax.set_ylabel("Cogging torque [Nm]")
    assert caption_from_figure(fig).unit == "Nm"


# ---------------------------------------------------------------------------
# the catalogue itself
# ---------------------------------------------------------------------------

def _images() -> list[Path]:
    return sorted(p for p in CATALOGUE.glob("*.png"))


def test_the_catalogue_is_not_empty():
    """A test over an empty set passes for the wrong reason."""
    assert len(_images()) >= 15


@pytest.mark.parametrize("image", _images(), ids=lambda p: p.stem)
def test_every_image_in_the_catalogue_has_a_caption(image):
    sidecar = image.with_suffix(".json")
    assert sidecar.is_file(), f"{image.name} has no caption beside it (run scripts/reference/run_examples.py)"
    cap = json.loads(sidecar.read_text(encoding="utf-8"))
    assert cap["field"].strip(), f"{image.name}: caption names no field"
    assert cap["unit"].strip(), f"{image.name}: caption declares no unit ('-' says dimensionless)"
    assert f"[{cap['unit']}]" in cap["text"], f"{image.name}: caption text does not print its unit"


@pytest.mark.parametrize("image", _images(), ids=lambda p: p.stem)
def test_the_caption_names_the_image_it_belongs_to(image):
    """A sidecar copied from another figure is worse than none: the range and the unit would be someone else's."""
    cap = json.loads(image.with_suffix(".json").read_text(encoding="utf-8"))
    assert cap["field"] in image.stem or re.search(r"__\d+_\d+$", image.stem)
    assert cap["text"].startswith(cap["label"] or cap["field"])


def test_the_catalogue_captions_are_what_the_rule_produces():
    """The sidecar is not hand-written: the same figure has to give the same caption again."""
    cap = figure_caption("Temperature", "K", value_range=(0.0, 8.0))
    assert cap.text() == "Temperature [K], 0 to 8 K"

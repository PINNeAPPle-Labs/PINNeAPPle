"""A model evaluated on an arbitrary domain, as a picture and as a VTU of the same numbers (#61).

Two things have to agree: what the figure shows and what the file a solver reads contains. These tests hold both — the
grid is built from the domain, every point goes into the VTU with its field, the file is real VTK XML (read back by a
reader when one is installed), and a model that returns the wrong shape is refused instead of being written.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

import pinneapple as pp
from pinneapple_tools.visualization.fields import render_field, write_vtu


class Model:
    """A stand-in for a trained surrogate: it has ``predict`` and names its fields."""

    fields = ("u",)

    def predict(self, X):
        return np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])


def _points(path):
    root = ET.parse(path).getroot()
    piece = root.find("./UnstructuredGrid/Piece")
    return root, piece


def test_a_two_dimensional_domain_gives_a_picture_and_a_vtu_of_the_same_numbers(tmp_path):
    written = render_field(Model(), {"x": (0.0, 1.0), "y": (0.0, 1.0)}, out=str(tmp_path / "square"), n=1024)
    assert set(written) == {"image", "vtu"}
    root, piece = _points(written["vtu"])
    assert root.get("type") == "UnstructuredGrid"
    n = int(piece.get("NumberOfPoints"))
    assert n == 32 * 32 and int(piece.get("NumberOfCells")) == n          # a point cloud: one vertex per point
    arrays = {d.get("Name"): d for d in piece.iter("DataArray")}
    assert "u" in arrays and len(arrays["u"].text.split()) == n
    assert arrays["u"].get("NumberOfComponents") == "1"
    assert (tmp_path / "square.png").exists()


def test_a_plain_callable_works_as_well_as_a_model_with_predict(tmp_path):
    written = render_field(lambda X: X[:, 0] + X[:, 1], {"x": (0.0, 1.0), "y": (0.0, 1.0)},
                           out=str(tmp_path / "callable"), field_names=["f"], image=False)
    _, piece = _points(written["vtu"])
    values = np.array([float(v) for v in piece.find("./PointData/DataArray").text.split()])
    assert values.min() == pytest.approx(0.0) and values.max() == pytest.approx(2.0)


def test_the_domain_can_be_a_problem(tmp_path):
    problem = pp.PhysicalProblem.from_preset("poisson_2d")
    written = render_field(lambda X: X[:, 0], problem, out=str(tmp_path / "problem"), field_names=["p"], image=False)
    _, piece = _points(written["vtu"])
    points = np.array([float(v) for v in piece.find("./Points/DataArray").text.split()]).reshape(-1, 3)
    assert points[:, 0].min() >= problem.domain_bounds["x"][0] - 1e-12
    assert points[:, 0].max() <= problem.domain_bounds["x"][1] + 1e-12
    assert points[:, 2].max() == 0.0, "a 2-D domain is written with z = 0, because VTK is three-dimensional"


def test_three_dimensions_are_written_whole_and_pictured_as_a_mid_plane(tmp_path):
    written = render_field(Model(), {"x": (0.0, 1.0), "y": (0.0, 1.0), "z": (0.0, 1.0)},
                           out=str(tmp_path / "box"), n=1000)
    _, piece = _points(written["vtu"])
    assert int(piece.get("NumberOfPoints")) == 10 ** 3
    assert (tmp_path / "box.png").exists()


def test_explicit_points_skip_the_image_when_asked(tmp_path):
    X = np.random.default_rng(0).random((50, 2))
    written = render_field(Model(), points=X, out=str(tmp_path / "points"), image=False)
    _, piece = _points(written["vtu"])
    assert int(piece.get("NumberOfPoints")) == 50 and "image" not in written


def test_a_model_that_returns_the_wrong_shape_is_refused(tmp_path):
    with pytest.raises(ValueError, match="expected \\(n,\\) or \\(n, n_fields\\)"):
        render_field(lambda X: np.zeros((3, 2)), {"x": (0, 1), "y": (0, 1)}, out=str(tmp_path / "bad"))


def test_the_wrong_number_of_field_names_is_refused(tmp_path):
    with pytest.raises(ValueError, match="2 field names for 1 columns"):
        render_field(Model(), {"x": (0, 1), "y": (0, 1)}, out=str(tmp_path / "names"), field_names=["a", "b"])


def test_vector_fields_are_written_with_their_components(tmp_path):
    X = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    path = write_vtu(str(tmp_path / "vec.vtu"), X, {"u": np.column_stack([X[:, 0], X[:, 1]])})
    _, piece = _points(path)
    array = piece.find("./PointData/DataArray")
    assert array.get("NumberOfComponents") == "2" and len(array.text.split()) == 6


def test_a_mesh_can_pass_its_own_connectivity(tmp_path):
    X = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    path = write_vtu(str(tmp_path / "tri.vtu"), X, {"u": np.arange(4.0)}, cells=[[0, 1, 2], [0, 2, 3]])
    _, piece = _points(path)
    types = piece.find("./Cells/DataArray[@Name='types']").text.split()
    assert types == ["5", "5"], "a three-node cell is a triangle in VTK"
    assert piece.find("./Cells/DataArray[@Name='offsets']").text.split() == ["3", "6"]


def test_a_points_array_that_is_not_two_or_three_columns_is_refused(tmp_path):
    with pytest.raises(ValueError, match=r"\(n, 2\) or \(n, 3\)"):
        write_vtu(str(tmp_path / "bad.vtu"), np.zeros((4, 4)), {"u": np.zeros(4)})


def test_a_field_with_the_wrong_length_is_refused(tmp_path):
    with pytest.raises(ValueError, match="3 values for 4 points"):
        write_vtu(str(tmp_path / "bad.vtu"), np.zeros((4, 2)), {"u": np.zeros(3)})


def test_a_vtk_reader_accepts_the_file(tmp_path):
    """The file is meant for ParaView and VTK, so the check that matters is a reader's, not a string comparison."""
    pv = pytest.importorskip("pyvista")
    written = render_field(Model(), {"x": (0.0, 1.0), "y": (0.0, 1.0)}, out=str(tmp_path / "read"), n=256)
    grid = pv.read(written["vtu"])
    assert grid.n_points == 16 * 16 and "u" in grid.array_names
    assert 0.0 <= float(grid["u"].min()) and float(grid["u"].max()) <= 1.0


def test_the_namespace_exposes_it_where_a_reader_looks_for_it():
    assert callable(pp.viz.render_field) and callable(pp.viz.write_vtu)

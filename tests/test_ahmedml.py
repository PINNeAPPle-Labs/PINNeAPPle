"""The half of the AhmedML contract this machine can carry: the surface reader, both reductions, and the surrogate."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_simulation.pipelines.ahmedml import (
    CaseSurface,
    aggregate_metrics,
    fit_surface_model,
    normalise_case,
    predict_case,
    surface_field_metrics,
)
from pinneapple_simulation.pipelines.ahmedml.surface import SURFACE_FIELDS, read_surface

vtk = pytest.importorskip("vtk")


def _write_vtp(path, points, cells, pressure, shear):
    """An XML PolyData with the release's layout: polygons, then CellData ``pMean`` and ``wallShearStressMean``."""
    poly = vtk.vtkPolyData()
    coordinates = vtk.vtkPoints()
    for point in points:
        coordinates.InsertNextPoint(*point)
    poly.SetPoints(coordinates)
    polygons = vtk.vtkCellArray()
    for cell in cells:
        polygons.InsertNextCell(len(cell))
        for index in cell:
            polygons.InsertCellPoint(index)
    poly.SetPolys(polygons)
    p_array = vtk.vtkFloatArray()
    p_array.SetName("pMean")
    p_array.SetNumberOfComponents(1)
    for value in pressure:
        p_array.InsertNextValue(float(value))
    shear_array = vtk.vtkFloatArray()
    shear_array.SetName("wallShearStressMean")
    shear_array.SetNumberOfComponents(3)
    for vector in shear:
        shear_array.InsertNextTuple3(*[float(component) for component in vector])
    poly.GetCellData().AddArray(p_array)
    poly.GetCellData().AddArray(shear_array)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    writer.Write()
    return path


def test_the_reader_returns_centroids_normals_areas_and_both_fields(tmp_path):
    points = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0],
              [4.0, 0.0, 0.0], [4.0, 0.0, 2.0]]
    cells = [[0, 1, 2, 3], [1, 4, 5]]
    path = _write_vtp(tmp_path / "surface.vtp", points, cells, pressure=[-1.5, 0.25],
                      shear=[[1e-3, 0.0, 0.0], [0.0, -2e-3, 1e-4]])
    surface = read_surface(str(path))
    assert surface["entities"] == 2
    assert np.allclose(surface["centroids"][0], [1.0, 1.0, 0.0])
    assert np.allclose(surface["normals"][0], [0.0, 0.0, 1.0])
    assert np.allclose(surface["corner_area"], [4.0, 2.0])     # 2×2 quad, and a triangle of base 2 and height 2
    assert np.allclose(surface["pressure"], [-1.5, 0.25])
    assert np.allclose(surface["wall_shear"][1], [0.0, -2e-3, 1e-4])


def test_the_reductions_are_the_published_ones_and_differ_by_their_weights():
    truth = np.array([1.0, 1.0, 1.0])
    prediction = np.array([2.0, 1.0, 1.0])
    area = np.array([3.0, 1.0, 1.0])
    scores = surface_field_metrics(prediction, truth, area)
    # one entity of five of the area is wrong by 1: weighted sqrt(3/5), uniform sqrt(1/3)
    assert scores["surface_pressure_rel_l2"] == pytest.approx(100 * np.sqrt(3.0 / 5.0))
    assert scores["surface_pressure_equal_entity_rel_l2"] == pytest.approx(100 * np.sqrt(1.0 / 3.0))


def test_the_wall_shear_reduction_uses_the_squared_vector_magnitude():
    truth = np.array([[3.0, 4.0, 0.0]])            # |y| = 5
    prediction = np.array([[0.0, 0.0, 0.0]])
    scores = surface_field_metrics(prediction, truth, np.array([1.0]), "wall_shear")
    assert scores["surface_wall_shear_rel_l2"] == pytest.approx(100.0)
    half = surface_field_metrics(np.array([[1.5, 2.0, 0.0]]), truth, np.array([1.0]), "wall_shear")
    assert half["surface_wall_shear_rel_l2"] == pytest.approx(50.0)


def test_the_two_fields_are_the_two_the_contract_binds():
    assert SURFACE_FIELDS["pressure"][0] == "pMean"
    assert SURFACE_FIELDS["wall_shear"] == ("wallShearStressMean", True)
    with pytest.raises(ValueError):
        surface_field_metrics(np.zeros(2), np.zeros(2), np.ones(2), "velocity")


def test_the_aggregation_is_the_macro_average_over_geometries():
    per_case = [{"surface_pressure_rel_l2": 10.0}, {"surface_pressure_rel_l2": 30.0}]
    assert aggregate_metrics(per_case)["surface_pressure_rel_l2"] == pytest.approx(20.0)
    with pytest.raises(ValueError):
        aggregate_metrics([])


def test_the_normalisation_reads_geometry_only():
    centroids = np.array([[0.0, 0.0, 0.0], [4.0, 2.0, 1.0]])
    coordinates, transform = normalise_case(centroids, np.zeros((2, 3)), np.array([10.0, 5.0, 2.5, 1.0]))
    assert transform["scale"] == pytest.approx(4.0)
    assert np.allclose(coordinates.min(axis=0), [-0.5, -0.25, -0.125])
    assert transform["parameter_scale"] == pytest.approx(10.0)


def _synthetic_case(case_id="run_0", n=3000, seed=0):
    generator = np.random.default_rng(seed)
    centroids = generator.uniform([-1.0, -0.2, 0.0], [0.0, 0.2, 0.4], size=(n, 3))
    normals = np.zeros((n, 3))
    normals[:, 2] = 1.0
    pressure = 0.5 * centroids[:, 0]                      # a field the network can represent
    shear = np.zeros((n, 3))
    shear[:, 0] = 1e-3 * centroids[:, 2]
    parameters = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    return CaseSurface(case_id=case_id, centroids=centroids, normals=normals, area=np.full(n, 1.0 / n),
                       parameters=parameters, pressure=pressure, wall_shear=shear)


def test_the_surrogate_learns_the_field_and_beats_the_mean():
    train = [_synthetic_case("run_0", seed=0), _synthetic_case("run_1", seed=1)]
    fitted = fit_surface_model(train, epochs=40, points_per_case=None, batch=4096, width=64, depth=2, bands=2)
    case = _synthetic_case("run_2", seed=2)
    pressure, shear = predict_case(fitted["model"], case, shear_scale=fitted["shear_scale"])
    learned = surface_field_metrics(pressure, case.pressure, case.area)["surface_pressure_rel_l2"]
    mean = surface_field_metrics(np.zeros_like(case.pressure), case.pressure, case.area)["surface_pressure_rel_l2"]
    assert learned < 0.5 * mean
    assert np.abs(shear).max() > 0.0


def test_the_fit_refuses_a_case_without_truth_and_prediction_does_not_need_it():
    case = _synthetic_case()
    blind = CaseSurface(case_id="run_9", centroids=case.centroids, normals=case.normals, area=case.area,
                        parameters=case.parameters)
    fitted = fit_surface_model([case], epochs=2, points_per_case=1000, batch=512, width=32, depth=1, bands=1)
    pressure, shear = predict_case(fitted["model"], blind, shear_scale=fitted["shear_scale"])
    assert pressure.shape == (len(case.centroids),)
    assert shear.shape == (len(case.centroids), 3)
    with pytest.raises(ValueError):
        fit_surface_model([blind], epochs=1, points_per_case=100, batch=64, width=16, depth=1, bands=1)

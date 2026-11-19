"""The parts of the DrivAerNet++ contract that can be checked without the 212 GB release."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_simulation.pipelines.drivaernetplusplus import (
    aggregate_metrics,
    load_support,
    read_legacy_polydata,
    surface_pressure_metrics,
)


def _write_vtk(path, points, pressure):
    """A binary legacy POLYDATA with the layout the release uses: POINTS, then SCALARS under POINT_DATA."""
    points = np.asarray(points, ">f4")
    scalars = np.asarray(pressure, ">f4").reshape(-1, 1)
    polygons = np.array([[3, 0, 1, 2], [3, 0, 2, 3]], dtype=">i4").ravel()
    with open(path, "wb") as handle:
        handle.write(b"# vtk DataFile Version 3.0\nDrivAer surface\nBINARY\nDATASET POLYDATA\n")
        handle.write(f"POINTS {len(points)} float\n".encode())
        handle.write(points.tobytes())
        handle.write(f"POLYGONS 2 {polygons.size}\n".encode())
        handle.write(polygons.tobytes())
        handle.write(f"POINT_DATA {len(points)}\n".encode())
        handle.write(b"SCALARS p float 1\nLOOKUP_TABLE default\n")
        handle.write(scalars.tobytes())
    return path


def test_the_reader_returns_the_release_layout(tmp_path):
    points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.5]]
    pressure = [-1853.76, -100.0, 0.0, 449.888]
    path = _write_vtk(tmp_path / "case.vtk", points, pressure)
    arrays = read_legacy_polydata(str(path))
    assert arrays["points"].shape == (4, 3)
    assert np.allclose(arrays["points"][2], [1.0, 1.0, 0.0])
    assert np.allclose(arrays["p"], pressure, atol=1e-3)
    assert int(arrays["polygon_count"][0]) == 2


def test_the_reader_refuses_what_it_cannot_read(tmp_path):
    ascii_path = tmp_path / "ascii.vtk"
    ascii_path.write_text("# vtk DataFile Version 3.0\nx\nASCII\nDATASET POLYDATA\nPOINTS 1 float\n0 0 0\n")
    with pytest.raises(ValueError, match="binary"):
        read_legacy_polydata(str(ascii_path))
    not_polydata = tmp_path / "structured.vtk"
    not_polydata.write_bytes(b"# vtk DataFile Version 3.0\nx\nBINARY\nDATASET STRUCTURED_POINTS\n")
    with pytest.raises(ValueError, match="POLYDATA"):
        read_legacy_polydata(str(not_polydata))


def test_the_metrics_are_the_two_published_reductions():
    truth = np.array([-10.0, 0.0, 10.0, 20.0])
    area = np.array([1.0, 1.0, 2.0, 4.0])
    assert surface_pressure_metrics(truth, truth, area)["surface_pressure_rel_l2"] == 0.0
    half = surface_pressure_metrics(truth * 0.5, truth, area)
    assert half["surface_pressure_rel_l2"] == pytest.approx(50.0)
    assert half["surface_pressure_equal_entity_rel_l2"] == pytest.approx(50.0)
    # an error confined to the smallest-area point must barely move the weighted metric and fully move the uniform one
    shifted = truth.copy()
    shifted[0] += 10.0
    metrics = surface_pressure_metrics(shifted, truth, area)
    assert metrics["surface_pressure_rel_l2"] < metrics["surface_pressure_equal_entity_rel_l2"]


def test_the_aggregation_is_the_macro_average_over_geometries():
    per_case = [{"surface_pressure_rel_l2": 10.0}, {"surface_pressure_rel_l2": 30.0}]
    assert aggregate_metrics(per_case)["surface_pressure_rel_l2"] == pytest.approx(20.0)


def test_the_real_case_support_has_the_arrays_the_spec_binds():
    """The released support of `E_S_WWC_WM_008`, fetched from the support archive whose hash matches the manifest."""
    import os

    path = "/Users/yanbarros/fluidsbench-data/three-d/drivanet/E_S_WWC_WM_008/support.npz"
    if not os.path.exists(path):
        pytest.skip("the fetched support case is not on this machine")
    support = load_support(path)
    assert support["p"].shape == (511063,)
    assert support["dual_area_m2"].sum() == pytest.approx(32.35184259501855, rel=1e-9)
    assert support["p"].min() == pytest.approx(-1853.760009765625, rel=1e-9)
    assert support["p"].max() == pytest.approx(449.88800048828125, rel=1e-9)
    # the identity prediction scores zero on both reductions, which is the check that the weighting is wired right
    metrics = surface_pressure_metrics(support["p"], support["p"], support["dual_area_m2"])
    assert metrics["surface_pressure_rel_l2"] == 0.0

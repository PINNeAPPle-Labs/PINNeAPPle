"""Visualisation adapters (INT-011): a bundle as a pyvista dataset, as a VTU, and as a ``.pvd`` series.

The structured path is exercised in memory and on disk; the unstructured path goes through the CAE writer, which is
what a machine without PyVista still gets -- that fallback is tested with ``available()`` forced to false, because a
fallback that is never run is a fallback that does not work.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pinneapple_data.schema import AxisSpec, Field, Physics, PhysicsDataBundle, view
from pinneapple_data.schema.view import save_series, save_vtu, to_pyvista

REPO = Path(__file__).resolve().parents[2]
CAVITY = REPO / "examples" / "physics_check" / "pass" / "cases" / "cavity"

needs_pyvista = pytest.mark.skipif(not view.available(), reason="pyvista is not installed")
needs_cavity = pytest.mark.skipif(not CAVITY.is_dir(), reason="the OpenFOAM cavity example is not in this checkout")

NX, NY, NT = 9, 7, 4


def plane(n_t: int = 0) -> PhysicsDataBundle:
    """A 2-D field on a grid, optionally with a time axis (a wave travelling in x)."""
    x = np.linspace(0.0, 1.0, NX)
    y = np.linspace(0.0, 0.5, NY)
    xx, yy = np.meshgrid(x, y, indexing="ij")       # the schema indexes a field's axes in declared order
    arrays: dict[str, np.ndarray] = {"axis:x": x, "axis:y": y}
    axes = [AxisSpec("x", "m"), AxisSpec("y", "m")]
    if n_t:
        t = np.linspace(0.0, 1.0, n_t)
        u = np.array([np.sin(2 * np.pi * (xx - 0.3 * ti)) for ti in t], dtype=np.float32)
        arrays |= {"axis:t": t, "field:u": u}
        axes.insert(0, AxisSpec("t", "s"))
    else:
        u = np.sin(2 * np.pi * xx).astype(np.float32)
        arrays["field:u"] = u
    return PhysicsDataBundle.from_arrays(arrays, fields=[Field.from_array("u", u, axes=tuple(
        a.name for a in axes))], axes=axes, physics=Physics(), source="view test", license="CC0-1.0",
        kind="synthetic")


@needs_pyvista
def test_a_plane_becomes_a_structured_grid_with_its_points_in_the_right_order():
    grid = to_pyvista(plane())
    assert grid.dimensions == (NX, NY, 1)
    assert grid.n_points == NX * NY
    assert "u" in grid.point_data
    # the first point is (x_min, y_min) and the second varies x fastest, which is VTK's ordering
    assert np.allclose(grid.points[0], [0.0, 0.0, 0.0])
    assert np.allclose(grid.points[1], [1.0 / (NX - 1), 0.0, 0.0])


@needs_pyvista
def test_the_time_axis_is_a_step_and_not_a_dimension():
    bundle = plane(n_t=NT)
    last = to_pyvista(bundle)                                  # default: the last snapshot
    first = to_pyvista(bundle, step=0)
    assert last.dimensions == (NX, NY, 1)
    assert np.allclose(last.point_data["u"], first.point_data["u"]) is False
    assert np.allclose(last.point_data["u"], np.asarray(bundle.field_array("u"))[-1].reshape(-1))
    assert np.allclose(to_pyvista(bundle, step=-1).point_data["u"], last.point_data["u"])


@needs_pyvista
def test_the_provenance_travels_with_the_grid():
    grid = to_pyvista(plane())
    assert grid.field_data["bundle_id"] == [plane().manifest.id]
    assert grid.field_data["license"] == ["CC0-1.0"]


@needs_pyvista
def test_a_cell_field_that_carries_one_value_per_node_is_flagged_not_misindexed():
    """The schema's cell fields reuse the nodal axis, so on a structured grid they land as point data -- with a note."""
    bundle = plane()
    spec = bundle.manifest.fields.get("u")
    spec.location = "cell"
    grid = to_pyvista(bundle)
    assert "u" in grid.point_data
    note = " ".join(grid.field_data["location_note"])
    assert "cell" in note and "point data" in note


@needs_pyvista
def test_a_bundle_without_a_spatial_axis_is_refused():
    alpha = np.array([0.25])
    bundle = PhysicsDataBundle.from_arrays(
        {"field:alpha": alpha}, fields=[Field.from_array("alpha", alpha)], axes=[],
        physics=Physics(), source="scalar", license="CC0-1.0", kind="synthetic")
    with pytest.raises(ValueError, match="no spatial axis"):
        to_pyvista(bundle)


@needs_pyvista
def test_one_snapshot_is_written_as_a_vtu_that_reads_back(tmp_path):
    import pyvista

    path = save_vtu(plane(n_t=NT), tmp_path / "plane.vtu", step=1)
    assert path.suffix == ".vtu" and path.stat().st_size > 500
    back = pyvista.read(path)
    assert back.n_points == NX * NY
    assert np.allclose(back.point_data["u"], np.asarray(plane(n_t=NT).field_array("u"))[1].reshape(-1))


@needs_pyvista
def test_a_series_writes_the_collection_paraview_opens(tmp_path):
    bundle = plane(n_t=NT)
    # one bundle per step, which is the shape a solver writes as it runs
    steps = [PhysicsDataBundle.from_arrays(
        {"axis:x": np.asarray(bundle.axis_array("x")), "axis:y": np.asarray(bundle.axis_array("y")),
         "field:u": np.asarray(bundle.field_array("u"))[i]},
        fields=[Field.from_array("u", np.asarray(bundle.field_array("u"))[i], axes=("x", "y"))],
        axes=[AxisSpec("x", "m"), AxisSpec("y", "m")], physics=Physics(), source=f"step {i}",
        license="CC0-1.0", kind="synthetic") for i in range(NT)]
    pvd = save_series(steps, tmp_path / "run.pvd", times=np.linspace(0, 1, NT))
    assert pvd.exists() and pvd.suffix == ".pvd"
    body = pvd.read_text()
    assert body.count("<DataSet") == NT
    assert [(tmp_path / "run" / f"run_{i:04d}.vtu").exists() for i in range(NT)] == [True] * NT
    assert 'timestep="1"' in body


def test_an_unstructured_bundle_is_written_by_the_cae_writer_when_pyvista_is_absent(tmp_path, monkeypatch):
    """The fallback path: no pyvista, so the VTU comes from the CAE exporter and still has the fields in it."""
    from pinneapple_data.schema.cae import bundle_from_path

    if not CAVITY.is_dir():
        pytest.skip("the OpenFOAM cavity example is not in this checkout")
    monkeypatch.setattr(view, "available", lambda: False)
    path = save_vtu(bundle_from_path(CAVITY, license="CC BY 4.0"), tmp_path / "cavity.vtu")
    assert path.stat().st_size > 10_000
    assert b"UnstructuredGrid" in path.read_bytes()


@needs_pyvista
@needs_cavity
def test_an_openfoam_case_becomes_an_unstructured_grid_with_its_velocity():
    from pinneapple_data.schema.cae import bundle_from_path

    grid = to_pyvista(bundle_from_path(CAVITY, license="CC BY 4.0"))
    assert grid.n_cells == 400
    assert np.asarray(grid["U"]).shape == (400, 3)


@needs_pyvista
def test_a_field_that_fits_neither_the_nodes_nor_the_cells_is_refused(monkeypatch):
    bundle = plane()
    monkeypatch.setattr(bundle, "field_array", lambda name: np.zeros(NX * NY + 1))
    with pytest.raises(ValueError, match="neither"):
        to_pyvista(bundle)

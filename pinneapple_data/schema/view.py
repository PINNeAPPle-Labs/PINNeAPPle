"""A bundle in a scientific visualisation tool: PyVista in memory, VTU/PVD on disk. INT-011.

Two representations, because the two kinds of bundle are two kinds of object in every viewer:

* a **structured** bundle becomes a ``StructuredGrid`` (1-D as a line, 2-D as a plane with ``z = 0``, 3-D as a
  block), with the nodal fields as point data and the cell fields as cell data -- the same distinction the schema
  enforces on the way in, preserved on the way out;
* an **unstructured** bundle goes through the CAE bridge and becomes an ``UnstructuredGrid``, or, when PyVista is
  not installed, straight to the VTU that :func:`pinneapple_data.schema.cae.export_bundle` already writes.

``save_series`` writes one VTU per step plus the ``.pvd`` collection that ParaView opens as an animation -- the file
pair `<name>.pvd` + `<name>/<step>.vtu` is what a reviewer double-clicks, and it is why this lives in the library
rather than in a script.

    >>> from pinneapple_data.schema.view import to_pyvista, save_vtu, save_series
    >>> mesh = to_pyvista(bundle, step=-1)          # last snapshot of a (time, space) bundle
    >>> save_vtu(bundle, "case.vtu", step=-1)
    >>> save_series(bundles_per_step, "run.pvd")
"""
from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["to_pyvista", "save_vtu", "save_series", "available"]


def available() -> bool:
    """Whether PyVista is installed (the VTU export works either way, through the CAE writer)."""
    try:
        import pyvista  # noqa: F401
        return True
    except ImportError:
        return False


def _import_pyvista():
    try:
        import pyvista
    except ImportError as exc:                                  # pragma: no cover - optional dependency
        raise ImportError("to_pyvista needs PyVista (pip install pyvista); save_vtu works on an unstructured "
                          "bundle without it, through the CAE writer") from exc
    return pyvista


def to_pyvista(bundle: Any, *, step: int | None = None, fields: Sequence[str] | None = None):
    """A ``pyvista`` dataset of one snapshot of a bundle.

    ``step`` selects the index along the time axis (default: the last one). A cell-centred field goes to
    ``cell_data`` -- and for a structured grid it must have one value per cell, so a cell field with the same length
    as the axis (the "cell centres reuse the nodal axis" convention) is placed as point data with a note, rather
    than silently misindexed."""
    pv = _import_pyvista()
    spec = bundle.manifest.mesh
    names = list(fields) if fields else bundle.manifest.fields.names()

    if spec.kind == "unstructured":
        from .cae import bundle_to_mesh
        grid = _grid_from_vtu(_vtu_bytes(bundle_to_mesh(bundle)))
        for name in names:
            values = np.asarray(bundle.field_array(name))
            grid[name] = values if values.ndim == 1 else values.reshape(len(values), -1)
        return grid

    spatial = [a for a in bundle.axes if a != "t"]
    if not spatial:
        raise ValueError("this bundle has no spatial axis to plot")
    coords = [np.asarray(bundle.axis_array(a), dtype=np.float64) for a in spatial]
    index = _step_index(bundle, step)
    dims = tuple(len(c) for c in coords)
    if len(dims) > 3:
        raise ValueError(f"a {len(dims)}-D grid is not a pyvista dataset")
    points = _points(coords)
    grid = pv.StructuredGrid()
    grid.points = points.reshape(-1, 3)
    grid.dimensions = (*dims, *([1] * (3 - len(dims))))
    n_t = int(bundle.axis_array("t").shape[0]) if "t" in bundle.axes else None
    n_nodes = int(np.prod(dims))
    renoted: list[str] = []
    for name in names:
        field = bundle.manifest.fields.get(name)
        values = np.asarray(bundle.field_array(name))
        if n_t is not None and values.ndim and values.shape[0] == n_t:
            values = values[index]
        total = values.size
        if field.location == "cell" and total == grid.n_cells:
            grid.cell_data[name] = values.reshape(-1)
        elif total % n_nodes == 0:
            grid.point_data[name] = values.reshape(n_nodes, total // n_nodes)
            if field.location == "cell":
                renoted.append(name)
        else:
            raise ValueError(f"field '{name}' has {total} values, which is neither {n_nodes} nodes nor "
                             f"{grid.n_cells} cells of this grid")
    if renoted:
        # the schema's cell fields carry one value per axis entry, not one per cell, so on a structured grid they
        # are one value per node; saying so in the file is better than letting a reader assume otherwise
        grid.field_data["location_note"] = [
            f"field(s) {', '.join(renoted)} are declared cell-centred but carry one value per node: placed as point data"]
    grid.field_data["bundle_id"] = [bundle.manifest.id]
    grid.field_data["license"] = [bundle.manifest.provenance.license]
    return grid


def _step_index(bundle: Any, step: int | None) -> int | None:
    if "t" not in bundle.axes:
        return None
    n = int(bundle.axis_array("t").shape[0])
    return n - 1 if step is None else (step if step >= 0 else n + step)


def _points(coords: Sequence[np.ndarray]) -> np.ndarray:
    """Structured grid points in VTK order (x fastest), with zeros for the missing dimensions."""
    grids = np.meshgrid(*coords, indexing="ij")
    stacked = np.stack([g.ravel() for g in grids], axis=1)
    if stacked.shape[1] < 3:
        stacked = np.concatenate([stacked, np.zeros((stacked.shape[0], 3 - stacked.shape[1]))], axis=1)
    # VTK wants the first axis fastest; meshgrid already gives the last axis fastest, so reverse the axis order
    shape = tuple(len(c) for c in coords)
    out = stacked.reshape(*shape, 3)
    return np.transpose(out, axes=(*reversed(range(len(shape))), len(shape))).reshape(-1, 3)


def _grid_from_vtu(xml: str):
    """An ``UnstructuredGrid`` from VTU text read in memory.

    Passing the text to ``pyvista.UnstructuredGrid`` works only on pyvista versions that take a string as content;
    from 0.49 a ``str`` is read as a *file name*, so the text goes through the VTK reader instead, which is
    unambiguous and needs no temp file."""
    import vtk

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.ReadFromInputStringOn()
    reader.SetInputString(xml)
    reader.Update()
    return _import_pyvista().wrap(reader.GetOutput())


def _vtu_bytes(mesh: Any) -> str:
    from ..cae.dataset import build_dataset, export
    blob, _media, _ext = export(build_dataset(mesh), "vtu")
    return blob.decode("utf-8", errors="replace")


def save_vtu(bundle: Any, path: str | os.PathLike[str], *, step: int | None = None) -> Path:
    """Write one snapshot as VTU. Uses PyVista when it is installed and the CAE writer otherwise."""
    target = Path(path)
    if target.suffix.lower() not in (".vtu", ".vts", ".vtk"):
        target = target.with_suffix(".vtu")
    target.parent.mkdir(parents=True, exist_ok=True)
    if bundle.manifest.mesh.kind == "unstructured" and not available():
        from .cae import export_bundle
        return export_bundle(bundle, target, "vtu")
    pv = _import_pyvista()
    grid = to_pyvista(bundle, step=step)
    if isinstance(grid, pv.StructuredGrid) and target.suffix.lower() != ".vts":
        # a StructuredGrid can only write .vts; every other reader wants .vtu, so cast (topology kept)
        grid = grid.cast_to_unstructured_grid()
    grid.save(str(target))
    return target


def save_series(bundles: Iterable[Any], path: str | os.PathLike[str], *, times: Sequence[float] | None = None,
                stem: str | None = None) -> Path:
    """One VTU per bundle plus the ``.pvd`` collection that ParaView reads as an animation."""
    items = list(bundles)
    if not items:
        raise ValueError("no bundles to write")
    target = Path(path)
    if target.suffix.lower() != ".pvd":
        target = target.with_suffix(".pvd")
    folder = target.with_suffix("")
    folder.mkdir(parents=True, exist_ok=True)
    name = stem or folder.name
    written = []
    for i, bundle in enumerate(items):
        written.append(save_vtu(bundle, folder / f"{name}_{i:04d}.vtu"))
    stamps = list(times) if times is not None else [
        float(np.asarray(b.axis_array("t"))[-1]) if "t" in b.axes else float(i) for i, b in enumerate(items)]
    entries = "\n".join(
        f'    <DataSet timestep="{t:g}" file="{p.name}"/>' for t, p in zip(stamps, written, strict=True))
    target.write_text('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
                      f'  <Collection>\n{entries}\n  </Collection>\n</VTKFile>\n', encoding="utf-8")
    return target

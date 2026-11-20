"""The AhmedML surface contract: read a case's boundary, and score the two fields bound to it.

AhmedML is the hardest of the open 3-D datasets to *complete* and one of the easiest to *start*: the composite needs a
surface prediction (`pMean` and `wallShearStressMean`, 25 % of the weight) **and** a volume prediction (`UMean` and
`pMean`, the other 25 % of the error weight plus the profile and cut diagnostics), and the contract demands complete
entity coverage of both. The volume is 5.6 GB per case, so a full entry is 280 GB for a 50-case split; the surface is
83 MB per case, ships its own cell areas as a `.npy`, and carries the ground truth as CellData of the public `.vtp`.

So this module implements the half that fits: the reader, the two published reductions for each surface field (face-area
weighted and equal-entity), the macro average over geometries, and the case loader with its eight design parameters.
A model built on it can be scored honestly on the surface metrics — and cannot be submitted until the volume exists,
which the module says rather than hides.

Read off a real case (`run_1`): 1 101 574 boundary entities of 1.57 m² of wetted area, `pMean` in m²/s², and
`wallShearStressMean` as a three-component kinematic traction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["read_surface", "load_case", "surface_field_metrics", "aggregate_metrics",
           "DYNAMIC_HEAD", "SURFACE_FIELDS"]

#: the inlet of the release is 1 m/s at unit density, so the dynamic head the pressure is worth is ½·ρ·U²  (m²/s²)
DYNAMIC_HEAD = 0.5

#: the two quantities the surface half of the composite is bound to, and whether each one is a vector
SURFACE_FIELDS = {"pressure": ("pMean", False), "wall_shear": ("wallShearStressMean", True)}

_PREFIX = {"pressure": "surface_pressure", "wall_shear": "surface_wall_shear"}


def read_surface(path: str) -> dict[str, np.ndarray]:
    """Polygon centroids, outward normals, the surface fields and the entity count of one boundary `.vtp`.

    The file is XML PolyData written by the release's own VTK, so it is read with VTK rather than with a hand-rolled
    parser: the cell array there is a connectivity array, and a misread of 1.2 M polygons is worse than a dependency.
    The orientation of a normal comes from the polygon winding, which the spec names as the force-orientation source."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    poly = reader.GetOutput()
    points = np.array([poly.GetPoint(i) for i in range(poly.GetNumberOfPoints())], dtype=float)
    cells = poly.GetPolys()
    if cells is None or cells.GetNumberOfCells() == 0:
        raise ValueError(f"{path}: no polygons")

    offsets = vtk_to_numpy(cells.GetOffsetsArray())
    connectivity = vtk_to_numpy(cells.GetConnectivityArray())
    sizes = np.diff(offsets)
    if np.any(sizes < 3):
        raise ValueError(f"{path}: {int(np.sum(sizes < 3))} degenerate polygons")
    count = len(sizes)
    coordinates = points[connectivity]
    centroids = np.add.reduceat(coordinates, offsets[:-1], axis=0) / sizes[:, None].astype(float)

    # a polygon's normal is the sum of a fan triangulation's triangle normals and its area is half that sum's length:
    # exact for triangles and planar quads, the standard estimate for the 422 pentagons this release carries
    normal = np.zeros((count, 3))
    quads = np.flatnonzero(sizes == 4)
    starts = offsets[:-1][quads][:, None] + np.arange(4)[None, :]
    quad = points[connectivity[starts]]                        # (n_quads, 4, 3)
    normal[quads] = (np.cross(quad[:, 1] - quad[:, 0], quad[:, 2] - quad[:, 0])
                     + np.cross(quad[:, 2] - quad[:, 0], quad[:, 3] - quad[:, 0]))
    for index in np.flatnonzero(sizes != 4):
        ring = points[connectivity[offsets[index]:offsets[index + 1]]]
        fan = np.cross(ring[1:-1] - ring[0], ring[2:] - ring[0])
        normal[index] = fan.sum(axis=0)
    magnitude = np.linalg.norm(normal, axis=1)
    if np.any(magnitude == 0.0):
        raise ValueError(f"{path}: {int(np.sum(magnitude == 0.0))} polygons with no orientation")

    data = poly.GetCellData()
    arrays = {}
    for key, (name, vector) in SURFACE_FIELDS.items():
        array = data.GetArray(name)
        if array is None:
            raise ValueError(f"{path}: the surface carries no '{name}'")
        width = array.GetNumberOfComponents()
        if vector != (width > 1):
            raise ValueError(f"{path}: '{name}' has {width} components, expected {'3' if vector else '1'}")
        values = vtk_to_numpy(array).astype(float)
        arrays[key] = values.reshape(-1, width) if width > 1 else values

    # the fan of the first corner, which is the same triangulation the normal is accumulated from
    return {"centroids": centroids, "normals": normal / magnitude[:, None],
            "corner_area": 0.5 * magnitude, "entities": np.int64(count), **arrays}


def load_case(root: str | Path, case_id: str, run_id: int) -> dict[str, np.ndarray]:
    """One case as the scoring side takes it: the surface, its published cell areas and its design vector.

    The areas come from the release's own `boundary_cell_area_<id>.npy` because they are the published weights, and the
    geometric estimate is kept beside them so the two can be compared instead of assumed equal."""
    root = Path(root)
    surface = read_surface(str(root / f"run_{run_id}" / f"boundary_{run_id}.vtp"))
    area = np.load(root / f"run_{run_id}" / f"boundary_cell_area_{run_id}.npy").astype(float)
    if area.shape != (surface["entities"],):
        raise ValueError(f"{case_id}: {area.shape[0]} areas for {surface['entities']} entities")
    parameters = np.genfromtxt(root / f"run_{run_id}" / f"geo_parameters_{run_id}.csv", delimiter=",", skip_header=1)
    surface.update({"case_id": case_id, "run_id": run_id, "area": area,
                    "parameters": np.atleast_1d(parameters).astype(float)})
    return surface


def surface_field_metrics(prediction, truth, area, field: str = "pressure") -> dict[str, float]:
    """Both published reductions of one surface field, in percent, for one geometry.

    ``100·sqrt(Σ w·(ŷ−y)² / Σ w·y²)`` with ``w`` the polygon's area for the primary metric and 1 for the secondary; for
    the three-component wall shear one entity weight multiplies the squared *vector* magnitude, which is what the
    spec's `vector_rule` says and is why the two fields share this function."""
    if field not in SURFACE_FIELDS:
        raise ValueError(f"unknown surface field '{field}'; expected one of {sorted(SURFACE_FIELDS)}")
    truth = np.asarray(truth, float)
    difference = np.asarray(prediction, float) - truth
    if difference.shape != truth.shape:
        raise ValueError(f"prediction {difference.shape} does not match the truth's {truth.shape}")
    if truth.ndim == 2:
        difference = np.sum(difference ** 2, axis=1)
        truth = np.sum(truth ** 2, axis=1)
    else:
        difference = difference ** 2
        truth = truth ** 2
    area = np.asarray(area, float)
    weighted = float(np.sqrt(np.sum(area * difference) / np.sum(area * truth)))
    uniform = float(np.sqrt(np.sum(difference) / np.sum(truth)))
    return {f"{_PREFIX[field]}_rel_l2": 100.0 * weighted,
            f"{_PREFIX[field]}_equal_entity_rel_l2": 100.0 * uniform}


def aggregate_metrics(per_case: list[dict[str, float]]) -> dict[str, float]:
    """The macro average over geometries — the aggregation the spec declares for every surface metric."""
    if not per_case:
        raise ValueError("no case to aggregate")
    return {name: float(np.mean([case[name] for case in per_case])) for name in per_case[0]}


def read_identity(path: str | Path) -> dict[str, dict]:
    """The release's own case records, keyed by case id: run id, entity counts and file digests."""
    identity = json.loads(Path(path).read_text())
    return {case["case_id"]: case for case in identity["cases"]}

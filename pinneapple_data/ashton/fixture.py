"""A synthetic fixture in the Ashton layout: parametric bluff bodies, two families, and a known-shaped shift.

The three real datasets are terabytes and need the network, so they cannot be read, benchmarked or trained on
in a test. This module writes a *small* dataset in the same layout -- ``run_<i>/boundary_<i>.vtp``,
``boundary_cell_area_<i>.npy``, ``force_mom_<i>.csv``, ``geo_parameters_<i>.csv`` -- from a parametric body of
our own, so that every reader in this package is exercised on the file it claims to read, offline and in a
second. Two things follow from that:

* the geometry is real (a closed, lofted, superelliptic car-like body with a rear slant, a taper and a ground
  clearance), so the normal/area/quadrature code is doing the arithmetic it does on a real case;
* the surface pressure comes from a stated model (:func:`pressure_coefficient`, a Newtonian-like field with a
  base-suction term), so the force coefficients are known to lie where that model puts them -- which is what
  makes the cross-family experiment in :mod:`pinneapple_data.ashton.trust` a measurement rather than a
  demonstration.

**Two families, one parametric body.** Family ``alpha`` is the rounded one (superellipse exponent 2.6-3.4, rear
slant 10-22 deg, low taper); family ``beta`` is the boxy one (exponent 5.0-7.0, slant 28-42 deg, heavier
taper, taller). Every case of both families comes out of the same :func:`bluff_body`, and the difference
between the families is a difference in where its parameters are drawn from -- which is the honest shape of an
out-of-distribution test: a model trained on one region of a design space meeting another region of it.

**The written numbers are not the reader's arithmetic.** ``force_mom_<i>.csv`` is integrated by decomposing
each quad into two triangles (a different route from the polygon integral in
:func:`pinneapple_data.ashton.loader.measured_forces`), so comparing the two is a check of the integral rather
than the same arithmetic run twice. The one case with a *hand* answer is :func:`write_box_case`: a rectangular
box whose face areas and per-face Cp are known in closed form, and whose Cd is arithmetic on a piece of paper.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .catalog import dataset as _dataset
from .catalog import layout_notes
from .vtk_xml import face_area_normals, node_area_normals, read_surface, write_surface

__all__ = ["PARAMETERS", "FAMILIES", "Params", "bluff_body", "pressure_coefficient", "sample_family",
           "write_fixture", "write_case", "write_box_case", "write_volume", "surface_summary", "HAND_CHECKED"]

#: The design parameters of one fixture body. Lengths are metres, angles degrees.
PARAMETERS = ("length", "width", "height", "slant", "roundness", "taper", "clearance")

#: The two families: ``(low, high)`` of each parameter. Family ``beta`` is the shift the experiment measures.
#: The dimensions are Ahmed-body sized (1.044 x 0.389 x 0.288 m, a 0.112 m^2 frontal area), so the fixture's
#: force coefficients land where a road-vehicle coefficient does and the catalogue's reference area is the
#: right one for the geometry rather than a scale factor nobody measured.
FAMILIES: dict[str, dict[str, tuple[float, float]]] = {
    "alpha": {"length": (0.98, 1.10), "width": (0.37, 0.41), "height": (0.27, 0.30), "slant": (10.0, 22.0),
              "roundness": (2.6, 3.4), "taper": (0.02, 0.10), "clearance": (0.05, 0.08)},
    "beta": {"length": (1.20, 1.35), "width": (0.34, 0.38), "height": (0.33, 0.38), "slant": (28.0, 42.0),
             "roundness": (5.0, 7.0), "taper": (0.14, 0.24), "clearance": (0.03, 0.06)},
}

#: The one case whose force coefficient is arithmetic on paper: a 2 x 1 x 1 box at Cp = 1 on the face the flow
#: meets and Cp = -0.4 on the face it leaves, the four other faces at 0. With ``A_ref = 1 m^2`` the drag is
#: ``1*1 + 0.4*1 = 1.4`` and the side and lift are 0. :func:`write_box_case` writes exactly that case.
HAND_CHECKED = {"length": 2.0, "width": 1.0, "height": 1.0, "cp_front": 1.0, "cp_rear": -0.4,
                "cp_other": 0.0, "a_ref": 1.0, "cd": 1.4}


@dataclass(frozen=True)
class Params:
    """One fixture body's design point."""

    length: float
    width: float
    height: float
    slant: float
    roundness: float
    taper: float
    clearance: float

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in PARAMETERS}


def sample_family(family: str, n: int, *, seed: int = 0) -> list[Params]:
    """``n`` design points of one family, uniform in the box the family declares.

    Uniform rather than maximin on purpose: the experiment in :mod:`pinneapple_data.ashton.trust` asks how a
    model trained on one family behaves on another, and a space-filling sample would make the *within*-family
    error look better than any real training set achieves.
    """
    if family not in FAMILIES:
        raise KeyError(f"{family!r} is not one of {sorted(FAMILIES)}")
    box = FAMILIES[family]
    rng = np.random.default_rng(seed)
    draws = {name: rng.uniform(low, high, size=int(n)) for name, (low, high) in box.items()}
    return [Params(**{name: float(draws[name][i]) for name in PARAMETERS}) for i in range(int(n))]


# ---------------------------------------------------------------------------
# the body
# ---------------------------------------------------------------------------

def _outline(width: float, height: float, centre_z: float, exponent: float, n_ring: int,
             top_drop: float) -> np.ndarray:
    """One cross-section in (y, z): a superellipse whose upper half is lowered by ``top_drop``."""
    theta = 2.0 * np.pi * np.arange(n_ring) / n_ring
    cos, sin = np.cos(theta), np.sin(theta)
    power = 2.0 / exponent
    y = 0.5 * width * np.sign(cos) * np.abs(cos) ** power
    z = centre_z + 0.5 * height * np.sign(sin) * np.abs(sin) ** power
    z = z - top_drop * np.clip(sin, 0.0, None)             # the slant cuts the roof, not the floor
    return np.column_stack([y, z])


def bluff_body(params: Params | dict[str, float], *, n_ring: int = 28,
               n_station: int = 13) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A closed lofted body: ``(points (n, 3), connectivity, offsets)`` in metres, flow along +x, z up.

    The stations run from the nose to the tail. The first 12 % of the length is an elliptic nose, the last 35 %
    tapers by ``taper``, and the roof is cut down over the last 30 % by ``height * slant/90``: a rear slant is
    the parameter that decides whether the wake is attached or separated, which is why it is in both families.
    """
    p = params if isinstance(params, Params) else Params(**params)
    length, width, height = p.length, p.width, p.height
    nose_fraction, taper_fraction, slant_fraction = 0.12, 0.35, 0.30
    xs = np.linspace(0.0, length, n_station)
    rings = []
    for x in xs:
        f = x / length
        scale = 1.0
        if f < nose_fraction:
            scale = math.sqrt(max(0.0, 1.0 - ((nose_fraction - f) / nose_fraction) ** 2))
        if f > 1.0 - taper_fraction:
            scale *= 1.0 - p.taper * (f - (1.0 - taper_fraction)) / taper_fraction
        drop = 0.0
        if f > 1.0 - slant_fraction:
            drop = height * (p.slant / 90.0) * (f - (1.0 - slant_fraction)) / slant_fraction
        outline = _outline(width * scale, height, p.clearance + 0.5 * height, p.roundness, n_ring, drop)
        rings.append(np.column_stack([np.full(n_ring, x), outline[:, 0], outline[:, 1]]))
    rings = np.asarray(rings)                                            # (n_station, n_ring, 3)
    points = rings.reshape(-1, 3).copy()
    faces: list[list[int]] = []
    for i in range(n_station - 1):
        for j in range(n_ring):
            a = i * n_ring + j
            b = i * n_ring + (j + 1) % n_ring
            c, d = a + n_ring, b + n_ring
            faces += [[a, b, c], [b, d, c]]
    for end, neighbour in ((0, 1), (n_station - 1, n_station - 2)):
        centre_index = len(points)
        centre = rings[end].mean(axis=0)
        points = np.vstack([points, centre])
        outward = centre - rings[neighbour].mean(axis=0)
        for j in range(n_ring):
            a = end * n_ring + j
            b = end * n_ring + (j + 1) % n_ring
            normal = np.cross(points[a] - centre, points[b] - centre)
            faces.append([centre_index, a, b] if normal @ outward > 0 else [centre_index, b, a])
    connectivity = np.asarray(faces, dtype=np.int64).reshape(-1)
    offsets = np.arange(0, len(faces) * 3 + 1, 3, dtype=np.int64)
    points, connectivity = _orient_outward(points, connectivity, offsets)
    return points, connectivity, offsets


def _orient_outward(points: np.ndarray, connectivity: np.ndarray,
                    offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flip every triangle if the closed surface came out wound the wrong way (signed volume < 0).

    The winding is consistent by construction, so one sign for the whole body is a real decision: a surface
    whose normals point inward would integrate a drag of the right magnitude with the wrong sign, which is the
    kind of error that looks like a physics result.
    """
    cells = connectivity.reshape(-1, 3)
    v0, v1, v2 = points[cells[:, 0]], points[cells[:, 1]], points[cells[:, 2]]
    volume = float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)
    if volume < 0:
        connectivity = connectivity.reshape(-1, 3)[:, ::-1].reshape(-1).copy()
    return points, connectivity


def pressure_coefficient(points: np.ndarray, connectivity: np.ndarray, offsets: np.ndarray, *,
                         cp_stagnation: float = 1.0, base: float = 1.05,
                         suction: float = 1.55) -> np.ndarray:
    """Cp on every face from the face normals: ``1 - (V/Vinf)^2`` with a model speed and a base-suction term.

    ``V/Vinf = suction * sqrt(1 - cos^2)`` puts zero speed on both stagnation poles and the strongest suction
    on the surfaces parallel to the flow, and ``base * max(0, cos)`` adds the wake suction on the rear-facing
    faces (``cos`` is the outward normal projected on the flow direction). It is a surrogate for a CFD result,
    not a CFD result: it exists so that the fixture has a surface field with the shape of one -- stagnation
    positive, roof and flanks negative, base negative and sensitive to the slant -- and so that the force
    integral can be checked against something known.
    """
    surface = _as_surface(points, connectivity, offsets)
    _areas, normals = face_area_normals(surface)
    along = normals @ np.array([1.0, 0.0, 0.0])
    speed = suction * np.sqrt(np.clip(1.0 - along ** 2, 0.0, None))
    return cp_stagnation - speed ** 2 - base * np.clip(along, 0.0, None)


def _as_surface(points: np.ndarray, connectivity: np.ndarray, offsets: np.ndarray):
    from .vtk_xml import Surface
    return Surface(points=np.asarray(points, dtype=np.float64),
                   connectivity=np.asarray(connectivity, dtype=np.int64),
                   offsets=np.asarray(offsets, dtype=np.int64))


# ---------------------------------------------------------------------------
# the writer
# ---------------------------------------------------------------------------

def _forces_by_triangles(points: np.ndarray, connectivity: np.ndarray, cp: np.ndarray) -> np.ndarray:
    """``-oint p n dA`` by Heron's formula on each triangle: the route the fixture writes, not the loader's."""
    cells = connectivity.reshape(-1, 3)
    v0, v1, v2 = points[cells[:, 0]], points[cells[:, 1]], points[cells[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    normal = cross / np.maximum(2.0 * area, 1e-30)[:, None]
    return -(cp * area) @ normal


def write_case(root: str | Path, run: int, params: Params | dict[str, float], *, dataset: str = "ahmedml",
               n_ring: int = 28, n_station: int = 13, fmt: str = "ascii", volume_points: int | None = None,
               cp_kwargs: dict[str, float] | None = None) -> dict[str, Any]:
    """Write one case of the fixture dataset: the surface, the quadrature array, the forces and the parameters.

    ``dataset`` decides the layout details that differ between the real datasets: ``"ahmedml"``/``"drivaerml"``
    write ``boundary_<i>.vtp`` (PolyData, Cp per cell, an area per polygon) while ``"windsorml"`` writes
    ``boundary_<i>.vtu`` (UnstructuredGrid, Cp per point, a quadrature weight per point) -- the same body and
    the same field, so the two quadratures can be compared on identical geometry.
    """
    data = _dataset(dataset)
    p = params if isinstance(params, Params) else Params(**params)
    points, connectivity, offsets = bluff_body(p, n_ring=n_ring, n_station=n_station)
    cp = pressure_coefficient(points, connectivity, offsets, **(cp_kwargs or {}))
    surface = _as_surface(points, connectivity, offsets)
    areas, _normals = face_area_normals(surface)
    folder = Path(root) / f"run_{int(run)}"
    folder.mkdir(parents=True, exist_ok=True)
    if data.quadrature == "cell":
        surface_file = write_surface(folder / f"boundary_{int(run)}.vtp", points=points, connectivity=connectivity,
                                     offsets=offsets, cell_data={"Cp": cp, "p": cp * 0.5 * data.density
                                                                 * data.velocity ** 2},
                                     grid="polydata", fmt=fmt, time=80.0)
    else:
        node_area, _node_normal = node_area_normals(surface)
        node_cp = _to_nodes(cp, areas, connectivity, points.shape[0])
        surface_file = write_surface(folder / f"boundary_{int(run)}.vtu", points=points, connectivity=connectivity,
                                     offsets=offsets, point_data={"Cp": node_cp}, grid="unstructuredgrid",
                                     cell_types=np.full(len(offsets) - 1, 5, dtype=np.int64), fmt=fmt, time=80.0)
        np.save(folder / f"boundary_dual_area_{int(run)}.npy", node_area.astype(np.float32))
    if data.quadrature == "cell":
        np.save(folder / f"boundary_cell_area_{int(run)}.npy", areas.astype(np.float32))
    force = _forces_by_triangles(points, connectivity, cp)
    cd, cs, cl = (force / data.a_ref)
    with open(folder / f"force_mom_{int(run)}.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(data.force_columns))
        writer.writerow([f"{v:.10e}" for v in _columns(data.force_columns, cd, cs, cl)])
    with open(folder / f"geo_parameters_{int(run)}.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(PARAMETERS))
        writer.writerow([f"{p.as_dict()[name]:.6f}" for name in PARAMETERS])
    if volume_points:
        rng = np.random.default_rng(int(run))
        lower = points.min(axis=0) - 0.5
        upper = points.max(axis=0) + 0.5
        box = rng.uniform(lower, upper, size=(int(volume_points), 3))
        write_volume(folder / f"volume_{int(run)}.vtu", points=box,
                     point_data={"p": rng.normal(size=int(volume_points)),
                                 "Ux": rng.normal(size=int(volume_points))})
    return {"run": int(run), "surface": str(surface_file), "cd": float(cd), "cl": float(cl), "cs": float(cs),
            "parameters": p.as_dict(), "n_points": int(points.shape[0]), "n_cells": int(len(offsets) - 1),
            "area": float(areas.sum())}


def _columns(columns: tuple[str, ...], cd: float, cs: float, cl: float) -> list[float]:
    by_name = {"cd": cd, "cl": cl, "cs": cs, "clf": 0.5 * cl, "clr": 0.5 * cl, "cmy": -0.02 * cl}
    return [by_name.get(name, 0.0) for name in columns]


def _to_nodes(cell_values: np.ndarray, weights: np.ndarray, connectivity: np.ndarray, n_points: int) -> np.ndarray:
    """Fold a per-face value onto the nodes, weighted by the face's share of each node's area."""
    cells = connectivity.reshape(-1, 3)
    numerator = np.zeros(n_points, dtype=np.float64)
    denominator = np.zeros(n_points, dtype=np.float64)
    for k in range(3):
        np.add.at(numerator, cells[:, k], cell_values * weights / 3.0)
        np.add.at(denominator, cells[:, k], weights / 3.0)
    return numerator / np.maximum(denominator, 1e-30)


def write_volume(path: str | Path, *, points: np.ndarray, point_data: dict[str, np.ndarray],
                 fmt: str = "binary") -> Path:
    """A small ``.vtu`` standing in for the 5.6-44 GB volume field: the points and their point arrays.

    The real volume files are the reason :func:`pinneapple_data.ashton.loader.volume_subsample` exists, and they
    cannot be in a test. A file with the same *structure* -- one ``<Piece>``, a ``Points`` array and named point
    arrays in inline binary -- is enough to check that a byte-range read returns exactly the elements a full
    read would have given, which is the only thing that can be checked offline.
    """
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    return write_surface(file, points=np.asarray(points, dtype=np.float64),
                         connectivity=np.zeros(0, dtype=np.int64), offsets=np.zeros(1, dtype=np.int64),
                         point_data=point_data, grid="unstructuredgrid", fmt=fmt, time=80.0)


def write_fixture(root: str | Path, *, family: str = "alpha", n_cases: int = 40, dataset: str = "ahmedml",
                  seed: int = 0, run_base: int = 1, n_ring: int = 28, n_station: int = 13,
                  fmt: str = "ascii") -> dict[str, Any]:
    """Write a whole fixture dataset of one family: ``n_cases`` cases in the Ashton layout, plus the aggregate
    tables the cards describe (``force_mom_all.csv``, ``geo_parameters_all.csv``).

    Returns the summary a test can assert on: the root, the dataset id, the family, the runs and the measured
    drag range of the family.
    """
    designs = sample_family(family, n_cases, seed=seed)
    cases = []
    for offset, params in enumerate(designs):
        cases.append(write_case(root, run_base + offset, params, dataset=dataset, n_ring=n_ring,
                                n_station=n_station, fmt=fmt))
    data = _dataset(dataset)
    with open(Path(root) / "force_mom_all.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", *data.force_columns])
        for case in cases:
            writer.writerow([case["run"], *[f"{v:.10e}" for v in _columns(data.force_columns, case["cd"], case["cs"],
                                                                         case["cl"])]])
    with open(Path(root) / "geo_parameters_all.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", *PARAMETERS])
        for case in cases:
            writer.writerow([case["run"], *[f"{case['parameters'][name]:.6f}" for name in PARAMETERS]])
    cds = np.array([case["cd"] for case in cases], dtype=np.float64)
    return {"root": str(root), "dataset": data.id, "family": family, "runs": [c["run"] for c in cases],
            "cd_min": float(cds.min()), "cd_max": float(cds.max()), "cd_mean": float(cds.mean()),
            "n_points": cases[0]["n_points"], "n_cells": cases[0]["n_cells"], "area": cases[0]["area"],
            "cases": cases, "layout": layout_notes()}


def write_box_case(root: str | Path, run: int = 900, *, a_ref: float = 1.0, fmt: str = "ascii") -> dict[str, Any]:
    """The one fixture case with a closed-form answer: a 2 x 1 x 1 box, Cp = 1 on its front face, -0.4 on its
    rear face and 0 on the four others.

    ``Cd = (1.0 * 1.0 + 0.4 * 1.0) / 1.0 = 1.4`` at ``A_ref = 1 m^2``, ``Cl = Cs = 0`` -- the numbers asserted in
    ``tests/test_ashton_datasets.py``. The box is written in the same layout as a real case, so the test goes
    through the reader, the quadrature and the integral exactly as a benchmark row does.
    """
    length, width, height = HAND_CHECKED["length"], HAND_CHECKED["width"], HAND_CHECKED["height"]
    points = np.array([[0, 0, 0], [0, width, 0], [0, width, height], [0, 0, height],
                       [length, 0, 0], [length, width, 0], [length, width, height], [length, 0, height]],
                      dtype=np.float64)
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (3, 7, 6, 2), (0, 4, 7, 3), (1, 2, 6, 5)]
    cp_per_quad = [HAND_CHECKED["cp_front"], HAND_CHECKED["cp_rear"]] + [HAND_CHECKED["cp_other"]] * 4
    connectivity = np.array(quads, dtype=np.int64).reshape(-1)
    offsets = np.arange(0, 4 * len(quads) + 1, 4, dtype=np.int64)
    surface = _as_surface(points, connectivity, offsets)
    areas, normals = face_area_normals(surface)
    order = [0, 1, 2, 3, 4, 5]
    cp = np.array(cp_per_quad)[order]
    del normals
    folder = Path(root) / f"run_{int(run)}"
    folder.mkdir(parents=True, exist_ok=True)
    write_surface(folder / f"boundary_{int(run)}.vtp", points=points, connectivity=connectivity, offsets=offsets,
                  cell_data={"Cp": cp}, grid="polydata", fmt=fmt)
    np.save(folder / f"boundary_cell_area_{int(run)}.npy", areas.astype(np.float32))
    cd = HAND_CHECKED["cd"] * (1.0 / a_ref)
    with open(folder / f"force_mom_{int(run)}.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cd", "cl"])
        writer.writerow([f"{cd:.10e}", "0.0000000000e+00"])
    with open(folder / f"geo_parameters_{int(run)}.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Length", "Width", "Height", "Angle_of_rear_window"])
        writer.writerow([length, width, height, 25.0])
    return {"run": int(run), "cd": cd, "a_ref": a_ref, "areas": areas, "cp": cp}


def surface_summary(path: str | Path) -> dict[str, Any]:
    """What a written case holds, read back through the reader: the numbers a test asserts on."""
    surface = read_surface(path)
    areas, _normals = face_area_normals(surface)
    return {"n_points": surface.n_points, "n_cells": surface.n_cells, "area": float(areas.sum()),
            "arrays": tuple(sorted(surface.point_data)), "cell_arrays": tuple(sorted(surface.cell_data))}

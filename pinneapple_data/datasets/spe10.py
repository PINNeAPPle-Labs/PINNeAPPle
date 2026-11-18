"""SPE10: the Tenth SPE Comparative Solution Project model 2, read as the bundle a Darcy problem needs.

SPE10 model 2 (Christie & Blunt, 2001) is the heterogeneous-permeability case that reservoir and Darcy-flow work
reports on: a 60 x 220 x 85 corner-point-free Cartesian grid of 1 122 000 cells, 20 x 10 x 2 ft each, whose top 35
layers are the Tarbert formation and whose bottom 50 are the fluvial Upper Ness. The SPE distributes it as one archive
(``por_perm_case2a.zip``) holding two ASCII tables, and this module turns them into a bundle.

The layout below was read off the archive itself (downloaded from the SPE page while writing this module), not from a
description of it:

``spe_perm.dat``
    6 values per line, 561 000 lines: ``kx ky kz`` for two cells. The value for cell ``(i, j, k)`` is at
    ``index = i + 60 * (j + 220 * k)`` -- **i varies fastest**. That ordering is not stated in the file; it was checked
    against the model's own structure: read that way the per-layer mean permeability jumps by a factor of two at layer
    36, which is where the Tarbert formation ends, and read the other way (k fastest) the layers are statistically
    indistinguishable.
``spe_phi.dat``
    6 values per line, 187 000 lines: one porosity per cell, same ordering.

Two limits that are recorded in every bundle rather than smoothed over:

* **which of the three permeability values is kx, ky or kz is a convention, not a property of the file.** The three
  columns have the same median (about 3 mD), the same mean (255 mD) and the same spread, so the numbers cannot say
  which is which; the bundle names them ``kx``/``ky``/``kz`` in the order the SPE documentation is always quoted in and
  says so in its metadata.
* **the archive states no licence.** SPE's dataset page describes the model and links the archive without terms, so
  ``license`` is a required argument here: this module will not write "unknown" into a file somebody redistributes.

    >>> from pinneapple_data.datasets import spe10
    >>> bundle = spe10.load_bundle("spe10/spe_perm.dat", "spe10/spe_phi.dat", license="SPE CSP dataset terms")  # doctest: +SKIP
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["SHAPE", "CELL_SIZE_FT", "TOP_DEPTH_FT", "TARBERT_LAYERS", "ARCHIVE_URL", "PERM_FILE", "PHI_FILE",
           "read_perm", "read_phi", "load_bundle"]

#: ``(nx, ny, nz)`` of the fine grid: 60 x 220 x 85 cells, 1 122 000 in all.
SHAPE = (60, 220, 85)

#: Cell size in the model's own unit, feet (SPE's dataset description: 20 ft x 10 ft x 2 ft).
CELL_SIZE_FT = (20.0, 10.0, 2.0)

#: Depth of the top of the model, feet (SPE's dataset description: initial depth 12 000 ft).
TOP_DEPTH_FT = 12_000.0

#: Layers 1-35 are the Tarbert formation; 36-85 are Upper Ness.
TARBERT_LAYERS = 35

ARCHIVE_URL = "https://www.spe.org/web/csp/datasets/por_perm_case2a.zip"
PERM_FILE = "spe_perm.dat"
PHI_FILE = "spe_phi.dat"


def _read_columns(path: str | os.PathLike[str], columns: int, shape: Sequence[int]) -> np.ndarray:
    """The table as ``(n_cells, columns)``, refusing a file whose size does not match the grid asked for."""
    values = np.loadtxt(path, dtype=np.float64)
    if values.size % columns:
        raise ValueError(f"{path}: {values.size} values is not a whole number of {columns}-value cells")
    cells = values.size // columns
    expected = int(np.prod(shape))
    if cells != expected:
        raise ValueError(f"{path}: {cells} cells, but the grid asked for is {tuple(shape)} = {expected}; the SPE10 "
                         "fine grid archive (por_perm_case2a.zip) is 60 x 220 x 85")
    return values.reshape(cells, columns)


def read_perm(path: str | os.PathLike[str], *, shape: Sequence[int] = SHAPE) -> np.ndarray:
    """Permeability as ``(nx, ny, nz, 3)`` in mD, the three components in the order the file lists them.

    ``shape`` is the grid to reshape onto: the fine grid by default, and a coarser one when the caller is reading an
    upscaled copy laid out the same way."""
    raw = _read_columns(path, 3, shape)
    return np.stack([raw[:, c].reshape(tuple(shape), order="F") for c in range(3)], axis=-1)


def read_phi(path: str | os.PathLike[str], *, shape: Sequence[int] = SHAPE) -> np.ndarray:
    """Porosity as ``(nx, ny, nz)``, dimensionless, in the same cell order as the permeability."""
    return _read_columns(path, 1, shape)[:, 0].reshape(tuple(shape), order="F")


def _centres(n: int, size: float, *, start: float = 0.0) -> np.ndarray:
    return start + size * (np.arange(n, dtype=np.float64) + 0.5)


def load_bundle(perm: str | os.PathLike[str] | np.ndarray, phi: str | os.PathLike[str] | np.ndarray | None = None,
                *, license: str, source: str = "", units: Mapping[str, str] | None = None,
                components: tuple[str, str, str] = ("kx", "ky", "kz"),
                shape: Sequence[int] = SHAPE) -> Any:
    """The model as a bundle: three permeability components and the porosity, on the grid SPE describes.

    ``perm``/``phi`` are paths to the archive's tables or arrays already read. The axes are cell **centres** in feet
    from the top of the model (``z`` grows downwards, as a reservoir depth does) and the fields are declared
    ``cell``-centred, which is what they are."""
    from ..schema import AxisSpec, Field, Geometry, Physics, PhysicsDataBundle
    from ..schema.bundle import file_refs

    if not license or not str(license).strip():
        raise ValueError("the SPE archive states no licence: pass the one that applies to your copy")
    grid = tuple(int(v) for v in shape)
    perm_array = read_perm(perm, shape=grid) if not isinstance(perm, np.ndarray) else np.asarray(perm)
    if perm_array.shape != grid + (3,):
        raise ValueError(f"permeability is {perm_array.shape}; expected {grid + (3,)}")
    phi_array = None
    if phi is not None:
        phi_array = read_phi(phi, shape=grid) if not isinstance(phi, np.ndarray) else np.asarray(phi)
        if phi_array.shape != grid:
            raise ValueError(f"porosity is {phi_array.shape}; expected {grid}")

    inputs = [str(p) for p in (perm, phi) if isinstance(p, (str, os.PathLike)) and Path(p).is_file()]
    nx, ny, nz = grid
    x = _centres(nx, CELL_SIZE_FT[0])
    y = _centres(ny, CELL_SIZE_FT[1])
    z = _centres(nz, CELL_SIZE_FT[2], start=TOP_DEPTH_FT)
    override = dict(units or {})

    store: dict[str, np.ndarray] = {"axis:x": x, "axis:y": y, "axis:z": z}
    fields = []
    for c, name in enumerate(components):
        values = perm_array[..., c]
        store[f"field:{name}"] = values
        fields.append(Field.from_array(
            name, values, unit=override.get(name, "mD"), location="cell", axes=("x", "y", "z"),
            description=f"permeability component {name}, as the archive lists it (the file does not label which of "
                        f"the three is x, y or z)"))
    if phi_array is not None:
        store["field:porosity"] = phi_array
        fields.append(Field.from_array("porosity", phi_array, unit=override.get("porosity", "1"), location="cell",
                                       axes=("x", "y", "z"), description="porosity, dimensionless"))

    return PhysicsDataBundle.from_arrays(
        store, fields=fields,
        axes=[AxisSpec("x", "ft"), AxisSpec("y", "ft"), AxisSpec("z", "ft")],
        physics=Physics(
            governing="single-phase incompressible Darcy flow, q = -(k/mu) grad p",
            notes=f"SPE10 model 2: layers 1-{TARBERT_LAYERS} are the Tarbert formation, the rest are Upper Ness; the "
                  "archive carries the rock properties, not a flow solution"),
        source=source or "SPE10 model 2 (fine grid)", title="SPE10 model 2 permeability and porosity",
        license=license, kind="dataset",
        geometry=Geometry(kind="structured-grid", dimension=3, units="ft",
                          description=f"a {grid[0]} x {grid[1]} x {grid[2]} Cartesian grid of "
                                      f"{CELL_SIZE_FT[0]:g} x {CELL_SIZE_FT[1]:g} x {CELL_SIZE_FT[2]:g} ft cells, "
                                      f"top at {TOP_DEPTH_FT:g} ft"),
        adapter="pinneapple_data.datasets.spe10", adapter_version="1",
        inputs=file_refs(inputs),
        citation="Christie & Blunt (2001), Tenth SPE Comparative Solution Project: a comparison of upscaling "
                 "techniques, SPE Reservoir Evaluation & Engineering 4(4)",
        url=ARCHIVE_URL,
        metadata={"spe10": {"shape": list(grid), "cell_size_ft": list(CELL_SIZE_FT), "top_depth_ft": TOP_DEPTH_FT,
                            "tarbert_layers": TARBERT_LAYERS,
                            "components": list(components),
                            "ordering": "value for cell (i, j, k) is at i + 60 * (j + 220 * k): i varies fastest",
                            "component_order_note": "the file gives three values per cell in a fixed order; that "
                                                    "order is the SPE convention (kx, ky, kz) and is not "
                                                    "recoverable from the numbers, whose distributions coincide",
                            "license_note": "the SPE archive states no terms: the licence is the caller's"}})

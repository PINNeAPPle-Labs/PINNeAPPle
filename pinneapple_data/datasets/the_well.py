"""The Well: read its HDF5 trajectories into the universal schema.

The Well (Ohana et al., NeurIPS 2024 Datasets and Benchmarks, "The Well: a Large-Scale Collection of Diverse
Physics Simulations for Machine Learning"; code BSD-3-Clause, data hosted per scenario at
`huggingface.co/polymathic-ai <https://huggingface.co/polymathic-ai>`_) holds ~15 TB of physics simulations in 16+
scenarios, every file 2-22 GB. Like PDEBench, it is read here through range requests
(:mod:`pinneapple_data.datasets.remote_hdf5`), so one trajectory costs megabytes instead of a terabyte-scale
download.

The layout, taken from the layout its own loader reads (``the_well/data/datasets.py`` v1.2.0)
----------------------------------------------------------------------------------------------
``/dimensions``
    one dataset per axis (``time``, ``x``, ``y``, ``z``), ``attrs["spatial_dims"]`` naming the spatial ones; each
    axis carries ``attrs["sample_varying"]`` (and the time axis ``attrs["time_varying"]``).
``/t0_fields``, ``/t1_fields``, ``/t2_fields``
    the fields, grouped by tensor rank, ``attrs["field_names"]``; every dataset is ``(n_trajectories, n_time,
    *spatial)``, with the time axis absent when ``attrs["time_varying"]`` is false.
``/boundary_conditions``
    one dataset per axis with ``attrs["bc_type"]`` and shape ``(n_trajectories, 2)`` (low, high).
``/scalars``
    ``attrs["field_names"]``; each is per-sample, per-time or constant.
Root attributes: ``dataset_name``, ``grid_type``, ``n_spatial_dims``, ``n_trajectories``.

This reader is exercised in the tests on a file written to that layout (a reproduction of it, not a downloaded
multi-gigabyte scenario -- the download is what the reader exists to avoid). It reads: the fields with their
canonical axes, the boundary conditions as schema ``BoundaryCondition`` entries, the scalars as ``Parameter``
and ``metadata``, and the per-sample axes that vary.

    >>> from pinneapple_data.datasets.the_well import open_the_well
    >>> with open_the_well("https://huggingface.co/datasets/polymathic-ai/rayleigh_benard/"
    ...                    "resolve/main/data/train/rayleigh_benard_Rayleigh_1e10_Prandtl_1.hdf5") as f:
    ...     f.dataset_name, f.spatial_dims, f.fields
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field as _dc_field
from pathlib import Path
from typing import Any

import numpy as np

from ..schema import (
    AxisSpec,
    BoundaryCondition,
    Field,
    Geometry,
    Parameter,
    Physics,
    PhysicsDataBundle,
)
from .remote_hdf5 import is_url, open_hdf5

__all__ = ["TheWellFile", "TheWellSample", "SCENARIOS", "Scenario", "open_the_well", "load_bundle",
           "scenario_hint"]

_FIELD_GROUPS = ("t0_fields", "t1_fields", "t2_fields")


@dataclass(frozen=True)
class Scenario:
    """One published scenario: its name on Hugging Face, the physics it solves and how big one file is."""

    name: str
    title: str
    physics: str
    example_file: str
    size_gb: float
    cite: str = "Ohana et al. (2024), The Well: a Large-Scale Collection of Diverse Physics Simulations for ML"


#: Scenarios this library has a card for. Sizes are the ones Hugging Face reports for the file named in
#: ``example_file``; the whole collection is ~15 TB, which is why nothing here downloads a scenario by default.
SCENARIOS: dict[str, Scenario] = {s.name: s for s in (
    Scenario("rayleigh_benard", "Rayleigh-Benard convection",
             "Boussinesq: incompressible Navier-Stokes coupled to an advected temperature",
             "data/train/rayleigh_benard_Rayleigh_1e10_Prandtl_1.hdf5", 8.4),
    Scenario("shear_flow", "Shear flow with a passive scalar",
             "incompressible Navier-Stokes with an advected scalar at Schmidt number Sc",
             "data/train/shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5", 13.4),
    Scenario("helmholtz_staircase", "Helmholtz equation in a staircase domain",
             "time-harmonic wave equation with a heterogeneous wave speed",
             "data/train/helmholtz_staircase_omega_006.hdf5", 2.8),
    Scenario("gray_scott_reaction_diffusion", "Gray-Scott reaction-diffusion",
             "two coupled reaction-diffusion equations (F=k, k parameters)",
             "data/train/gray_scott_reaction_diffusion_bubbles_F_0.098_k_0.057.hdf5", 21.0),
    Scenario("active_matter", "Active matter",
             "Navier-Stokes with an active nematic stress",
             "data/train/active_matter_chunk_0.hdf5", 8.0),
    Scenario("acoustic_scattering_maze", "Acoustic scattering in a maze",
             "acoustic wave equation with rigid boundaries",
             "data/train/acoustic_scattering_maze_chunk_0.hdf5", 16.0),
    Scenario("euler_multi_quadrants_openBC", "Euler equations, four quadrants, open boundaries",
             "compressible Euler with a Riemann-type initial condition per quadrant",
             "data/train/euler_multi_quadrants_openBC_gamma_1.13_C3H8_16_chunk_0.hdf5", 21.2),
    Scenario("planetswe", "Planetary shallow water",
             "rotating shallow-water equations",
             "data/train/planetswe_*", 20.0),
    Scenario("MHD_64", "Magnetohydrodynamics (64^3)",
             "compressible MHD with an externally driven field",
             "data/train/MHD_64_*", 20.0),
)}

_HF_BASE = "https://huggingface.co/datasets/polymathic-ai"


def scenario_url(name: str, filename: str | None = None) -> str:
    """The ``resolve`` URL of a scenario file (the form :class:`HTTPRangeFile` can range-read)."""
    scenario = SCENARIOS[name]
    return f"{_HF_BASE}/{name}/resolve/main/{filename or scenario.example_file}"


def scenario_hint(source: str) -> str:
    """The scenario name a path or URL belongs to, from the ``polymathic-ai/<name>`` part (or the file name)."""
    text = str(source)
    for name in SCENARIOS:
        if f"/{name}/" in text or f"/{name}_" in text or Path(text).name.startswith(f"{name}_"):
            return name
    return ""


@dataclass
class TheWellSample:
    """One trajectory of a scenario, in canonical names."""

    fields: dict[str, np.ndarray]               # name -> (time, *spatial) or (time, *spatial, components)
    coordinates: dict[str, np.ndarray]          # "t", "x", ... ; absent when the axis does not vary per sample
    scalars: dict[str, float] = _dc_field(default_factory=dict)
    boundary_conditions: list[dict[str, Any]] = _dc_field(default_factory=list)
    notes: list[str] = _dc_field(default_factory=list)


class TheWellFile:
    """An open The Well HDF5 file: metadata now, trajectories on demand (see :func:`open_the_well`)."""

    def __init__(self, source: str | Path | Any, *, handle: Any = None) -> None:
        self.source = str(source)
        self._ctx = None
        if handle is not None:
            self._f = handle
        else:
            self._ctx = open_hdf5(self.source)
            self._f = self._ctx.__enter__()
        self.notes: list[str] = []

    # ── metadata ────────────────────────────────────────────────────────────────────────────────────────
    @property
    def dataset_name(self) -> str:
        return str(self._f.attrs.get("dataset_name", scenario_hint(self.source) or "the_well"))

    @property
    def grid_type(self) -> str:
        return str(self._f.attrs.get("grid_type", "cartesian"))

    @property
    def n_trajectories(self) -> int:
        if "n_trajectories" in self._f.attrs:
            return int(self._f.attrs["n_trajectories"])
        for group in _FIELD_GROUPS:
            if group in self._f and len(self._f[group]):
                first = self._f[group][self._f[group].attrs["field_names"][0]]
                return int(first.shape[0])
        return 0

    @property
    def spatial_dims(self) -> list[str]:
        if "dimensions" in self._f and "spatial_dims" in self._f["dimensions"].attrs:
            dims = self._f["dimensions"].attrs["spatial_dims"]
            return [d.decode() if isinstance(d, bytes) else str(d) for d in dims]
        return [d for d in ("x", "y", "z") if "dimensions" in self._f and d in self._f["dimensions"]]

    @property
    def fields(self) -> list[tuple[str, str]]:
        """``(raw name, tensor rank group)`` for every field."""
        out = []
        for group in _FIELD_GROUPS:
            if group not in self._f:
                continue
            names = self._f[group].attrs.get("field_names", [])
            for name in names:
                out.append((name.decode() if isinstance(name, bytes) else str(name), group))
        return out

    @property
    def scalar_names(self) -> list[str]:
        if "scalars" not in self._f:
            return []
        names = self._f["scalars"].attrs.get("field_names", [])
        return [n.decode() if isinstance(n, bytes) else str(n) for n in names]

    # ── reading ─────────────────────────────────────────────────────────────────────────────────────────
    def _axis(self, name: str, index: int) -> np.ndarray | None:
        if "dimensions" not in self._f or name not in self._f["dimensions"]:
            return None
        dataset = self._f["dimensions"][name]
        varying = bool(dataset.attrs.get("sample_varying", False))
        values = np.asarray(dataset[index] if varying else dataset[()])   # the stored dtype
        return values

    def boundary_conditions(self, index: int) -> list[dict[str, Any]]:
        """The boundary conditions of one trajectory: axis, low/high side and type, as the file states them."""
        if "boundary_conditions" not in self._f:
            return []
        out = []
        for axis in self._f["boundary_conditions"].keys():
            dataset = self._f["boundary_conditions"][axis]
            bc_type = dataset.attrs.get("bc_type", "")
            bc_type = bc_type.decode() if isinstance(bc_type, bytes) else str(bc_type)
            sides = np.asarray(dataset[index] if dataset.ndim > 1 else dataset[()])
            for side, code in zip(("low", "high"), np.atleast_1d(sides), strict=False):
                out.append({"axis": axis, "side": side, "bc_type": bc_type, "code": int(code)})
        return out

    def sample(self, index: int = 0) -> TheWellSample:
        """Read one trajectory: fields with their axes, scalar parameters and boundary conditions."""
        if not 0 <= index < max(1, self.n_trajectories):
            raise IndexError(f"trajectory {index} out of range: this file has {self.n_trajectories}")
        coordinates: dict[str, np.ndarray] = {}
        for name in ("time",) + tuple(self.spatial_dims):
            values = self._axis(name, index)
            if values is not None:
                coordinates["t" if name == "time" else name] = values
        fields: dict[str, np.ndarray] = {}
        notes: list[str] = []
        for raw_name, group in self.fields:
            dataset = self._f[group][raw_name]
            values = np.asarray(dataset[index])
            rank = int(group[1])
            if rank == 1:                                    # a vector field: components on the last axis
                values = np.moveaxis(values, 1, -1) if values.ndim == len(self.spatial_dims) + 2 else values
            fields[raw_name] = values
        if "t" in coordinates and fields:
            n = min(arr.shape[0] for arr in fields.values())
            if len(coordinates["t"]) != n:
                keep = min(len(coordinates["t"]), n)
                notes.append(f"the time axis has {len(coordinates['t'])} entries and the fields {n}: kept the "
                             f"first {keep}")
                coordinates["t"] = coordinates["t"][:keep]
                fields = {k: v[:keep] for k, v in fields.items()}
        scalars: dict[str, float] = {}
        for name in self.scalar_names:
            dataset = self._f["scalars"][name]
            values = np.asarray(dataset[index]) if dataset.ndim > 0 else np.asarray(dataset[()])
            if values.ndim > 0 and values.size > 1:
                notes.append(f"scalar '{name}' varies ({values.shape}); its first value is recorded as the "
                             "parameter and the whole series is kept in the bundle metadata")
                scalars[name] = float(values.reshape(-1)[0])
            elif values.size:
                scalars[name] = float(values.reshape(-1)[0])
        return TheWellSample(fields=fields, coordinates=coordinates, scalars=scalars,
                             boundary_conditions=self.boundary_conditions(index), notes=notes)

    def __enter__(self) -> TheWellFile:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = None

    # ── to the universal schema ─────────────────────────────────────────────────────────────────────────
    def to_bundle(self, index: int = 0, *, license: str | None = None) -> PhysicsDataBundle:
        """Build a :class:`PhysicsDataBundle` from one trajectory.

        ``license`` has no default on purpose: the_well's *code* is BSD-3-Clause and each scenario carries its own
        data terms on Hugging Face, so a bundle that names a licence it was not given would be a legal claim this
        library is not in a position to make. Pass ``license=`` explicitly once you have checked the scenario's
        page."""
        if not license:
            raise ValueError(
                "pass license=... : the_well's code licence (BSD-3-Clause) is not the licence of its data, and "
                f"each scenario states its own terms at {_HF_BASE}/{self.dataset_name}")
        sample = self.sample(index)
        axes: list[AxisSpec] = [
            AxisSpec(name, "1", description=f"The Well {name} axis") for name in
            tuple(n for n in ("t",) + tuple(self.spatial_dims) if n in sample.coordinates)]
        axis_names = tuple(a.name for a in axes)
        arrays: dict[str, np.ndarray] = {}
        for ax in axes:
            arrays[f"axis:{ax.name}"] = sample.coordinates[ax.name]
        fields = []
        for name, values in sample.fields.items():
            arrays[f"field:{name}"] = values
            components = [f"c{i}" for i in range(values.shape[-1])] if values.ndim == len(axis_names) + 1 else []
            fields.append(Field.from_array(name, values, unit="1", location="nodal", axes=axis_names,
                                           components=components))
        scenario = SCENARIOS.get(self.dataset_name)
        physics = Physics(
            governing=scenario.physics if scenario else "",
            boundary_conditions=[BoundaryCondition(field="all", region=f"{bc['axis']}_{bc['side']}",
                                                   kind=_bc_kind(bc["bc_type"]), description=f"{bc['bc_type']}")
                                 for bc in sample.boundary_conditions],
            parameters=[Parameter(name=k, value=v, unit="1", source="The Well scalar") for k, v in sample.scalars.items()],
            notes=("Values are in the scenario's own non-dimensionalisation; units are recorded as \"1\". "
                   + (scenario.cite if scenario else "")))
        metadata = {"the_well": {"dataset_name": self.dataset_name, "grid_type": self.grid_type,
                                 "trajectory": index, "n_trajectories": self.n_trajectories,
                                 "spatial_dims": self.spatial_dims, "scalars": sample.scalars,
                                 "boundary_conditions": sample.boundary_conditions, "notes": sample.notes}}
        return PhysicsDataBundle.from_arrays(
            arrays, fields=fields, axes=axes, physics=physics,
            source=f"The Well {self.dataset_name}" + (f" :: {Path(self.source).name}" if not is_url(self.source) else ""),
            license=license, title=f"The Well {self.dataset_name}",
            id=f"the-well-{self.dataset_name.lower()}-{index:04d}",
            geometry=Geometry(kind="structured-grid", dimension=max(1, len(axis_names) - 1), units="1",
                              description=(scenario.title if scenario else self.dataset_name)),
            metadata=metadata, adapter="pinneapple_data.datasets.the_well", adapter_version="1",
            url=self.source, citation=scenario.cite if scenario else "")


def _bc_kind(bc_type: str) -> str:
    """Map The Well's ``bc_type`` names onto the schema's boundary-condition vocabulary."""
    text = (bc_type or "").lower()
    for kind, words in (("periodic", ("periodic",)), ("wall", ("wall", "dirichlet", "noslip", "no-slip")),
                        ("symmetry", ("symmetric", "symmetry")), ("open", ("open", "neumann", "outflow"))):
        if any(w in text for w in words):
            return kind
    return "dirichlet"


@contextmanager
def open_the_well(source: str | Path, **kwargs: Any) -> Iterator[TheWellFile]:
    """Open a The Well scenario file (local path or Hugging Face ``resolve`` URL) and close it afterwards."""
    handle = TheWellFile(source, **kwargs)
    try:
        yield handle
    finally:
        handle.close()


def load_bundle(source: str | Path, index: int = 0, *, license: str | None = None) -> PhysicsDataBundle:
    """Read trajectory ``index`` of a The Well file into a :class:`PhysicsDataBundle`."""
    with open_the_well(source) as handle:
        return handle.to_bundle(index, license=license)

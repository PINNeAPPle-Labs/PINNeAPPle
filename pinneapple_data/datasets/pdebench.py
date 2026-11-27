"""PDEBench: read its HDF5 files into the universal schema, locally or straight off the archive.

PDEBench (Takamoto et al., NeurIPS 2022, "PDEBench: An Extensive Benchmark for Scientific Machine Learning";
data DOI `10.18419/darus-2986 <https://doi.org/10.18419/darus-2986>`_, CC BY 4.0) ships the reference PDE datasets
that almost every operator-learning paper reports on. This module reads them without a full download:
:mod:`pinneapple_data.datasets.remote_hdf5` serves the HDF5 metadata and only the chunks a sample touches, so the
one sample a benchmark needs costs megabytes, not the 1.3-88 GB of the file it lives in.

Two layouts, both of which appear in the archive
------------------------------------------------
``tensor``
    root datasets only: one or more fields shaped ``(n_samples, n_time, *spatial)`` -- or ``(n_time, *spatial)``
    when the file holds a single trajectory -- plus ``x-coordinate`` / ``y-coordinate`` / ``z-coordinate`` /
    ``t-coordinate`` and, for the parametric families, a per-sample parameter array (``nu``, ``beta``, ``M``,
    ``eta``...). ``Sod6.hdf5`` and the ``*_Sols_*`` files are this layout, and it is the one verified against the
    published files in the tests.
``groups``
    one group per sample (``0000``, ``0001``, ...), each holding its fields and coordinates. This is the layout
    the paper's own ``data_gen`` writes; the reader follows the layout as documented and is exercised on a
    reproduction of it in the tests, not on a downloaded multi-gigabyte file.

What is preserved, and what is not
----------------------------------
* field names are canonicalised (``density``, ``pressure``, ``velocity`` with components ``u``/``v``/``w``,
  ``Vx``/``Vy``/``Vz`` folded into ``velocity``) and the raw names stay in ``metadata["pdebench"]["raw_fields"]``,
  so nothing is lost by the renaming;
* every value is dimensionless in the archive's own non-dimensionalisation, and the schema records ``"1"`` rather
  than inventing SI units;
* coordinates are read from the file, never assumed -- and where a coordinate array disagrees with the data it
  indexes (``Sod6.hdf5`` carries 202 time coordinates for 201 snapshots), the adapter aligns the shorter of the two
  and records the discrepancy in ``metadata["pdebench"]["notes"]`` and in a ``time_alignment`` check. That
  disagreement is exactly the kind of thing a silent reader hides.

    >>> from pinneapple_data.datasets.pdebench import load_bundle
    >>> bundle = load_bundle("https://darus.uni-stuttgart.de/api/access/datafile/133150")   # Sod6, 4.9 MB
    >>> bundle.manifest.fields.names()
    ['density', 'pressure', 'velocity']
"""
from __future__ import annotations

import os
import re
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field as _dc_field
from pathlib import Path
from typing import Any

import numpy as np

from ..schema import AxisSpec, Field, FileRef, Geometry, Parameter, Physics, PhysicsDataBundle
from ..schema.bundle import content_digest as _content_digest
from ..schema.models import sha256_file
from .remote_hdf5 import is_url, open_hdf5

__all__ = [
    "CaseCard", "PDEBenchFile", "PDEBenchSample", "DATASETS", "DatasetFile", "case_card", "open_pdebench",
    "load_bundle", "download", "darus_url",
]

DARUS_DOI = "10.18419/darus-2986"
DARUS_LICENSE = "CC BY 4.0"
DARUS_URL = f"https://doi.org/{DARUS_DOI}"
PAPER = ("Takamoto, M., Praditia, T., Leiteritz, R., MacKinlay, D., Alesiani, F., Pflüger, D., Niepert, M. "
         "(2022). PDEBench: An Extensive Benchmark for Scientific Machine Learning. NeurIPS 2022 Datasets and "
         "Benchmarks Track. arXiv:2210.07182")

_COORD_NAMES = ("x-coordinate", "y-coordinate", "z-coordinate", "t-coordinate")
_SPATIAL = ("x", "y", "z")
_COMPONENTS = ("u", "v", "w")


def darus_url(file_id: int | str) -> str:
    """The download URL of one file of the PDEBench dataset on DaRUS."""
    return f"https://darus.uni-stuttgart.de/api/access/datafile/{file_id}"


@dataclass(frozen=True)
class DatasetFile:
    """One published file: which case, on which archive id, how big, and what it holds."""

    key: str                    # "1D_Burgers_Sols_Nu0.001"
    family: str                 # "1D_Burgers"
    filename: str
    darus_id: int
    size_mb: float
    layout: str = "tensor"

    @property
    def url(self) -> str:
        return darus_url(self.darus_id)


#: The curated subset of the archive this library reads: one representative per family, plus the small Sod case
#: that makes a real-file integration test possible. Sizes are those reported by DaRUS v8 of the dataset.
DATASETS: dict[str, DatasetFile] = {d.key: d for d in (
    DatasetFile("Sod6", "Sod", "Sod6.hdf5", 133150, 4.9),
    DatasetFile("1D_Burgers_Sols_Nu0.001", "1D_Burgers", "1D_Burgers_Sols_Nu0.001.hdf5", 268190, 8233.0),
    DatasetFile("1D_Advection_Sols_beta1.0", "1D_Advection", "1D_Advection_Sols_beta1.0.hdf5", 255675, 8233.0),
    DatasetFile("1D_diff-sorp_NA_NA", "1D_diff-sorp", "1D_diff-sorp_NA_NA.h5", 133020, 4217.0),
    DatasetFile("1D_CFD_Rand_Eta0.1_Zeta0.1_periodic_Train", "1D_CFD",
                "1D_CFD_Rand_Eta0.1_Zeta0.1_periodic_Train.hdf5", 164668, 12410.9),
    DatasetFile("2D_DarcyFlow_beta1.0_Train", "2D_DarcyFlow", "2D_DarcyFlow_beta1.0_Train.hdf5", 133219, 1310.7),
    DatasetFile("2D_diff-react_NA_NA", "2D_diff-react", "2D_diff-react_NA_NA.h5", 133017, 13243.6),
    DatasetFile("2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train", "2D_CFD",
                "2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5", 164688, 55050.2),
    DatasetFile("3D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_Train", "3D_CFD",
                "3D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_Train.hdf5", 173286, 88080.4),
)}


@dataclass(frozen=True)
class CaseCard:
    """What equation a family solves, in words and symbols -- the ``Physics`` a bundle is stamped with.

    Nothing here is read from the file: the archive stores numbers, and the equation is the published statement
    of what those numbers are a solution of. ``reference`` names the source, because a governing equation with
    no source is a guess."""

    family: str
    title: str
    governing: str
    symbols: dict[str, str] = _dc_field(default_factory=dict)
    notes: str = ""
    reference: str = PAPER


_CASES: tuple[CaseCard, ...] = (
    CaseCard("1D_Burgers", "1D viscous Burgers equation",
             "u_t + u u_x = nu u_xx",
             {"u": "conserved scalar", "nu": "kinematic viscosity", "x": "periodic domain, [0, 1]"},
             "Periodic boundary conditions; the archive's Nu attribute is the viscosity."),
    CaseCard("1D_Advection", "1D linear advection equation",
             "u_t + beta u_x = 0",
             {"u": "advected scalar", "beta": "advection speed", "x": "periodic domain, [0, 1]"},
             "Periodic boundary conditions; beta is the file's attribute."),
    CaseCard("1D_diff-sorp", "1D diffusion-sorption equation",
             "u_t = D u_xx - (rho / beta) s_t,  s = u / (1 + rho k u)  (isotherm form set by the case)",
             {"u": "solute concentration", "D": "diffusion coefficient", "rho": "bulk density",
              "beta": "partition/porosity coefficient", "s": "sorbed concentration"},
             "The exact isotherm parameters vary with the published case; read them from the file attributes."),
    CaseCard("1D_reac-diff", "1D reaction-diffusion (Fisher-KPP type)",
             "u_t = D u_xx + R(u)",
             {"u": "concentration", "D": "diffusivity", "R": "reaction rate"},
             "Reaction of Arrhenius form in the published cases; parameters are file attributes."),
    CaseCard("1D_wave", "1D wave equation",
             "u_tt = c^2 u_xx",
             {"u": "displacement", "c": "wave speed"}),
    CaseCard("1D_CFD", "1D compressible Euler equations",
             "rho_t + (rho u)_x = 0,  (rho u)_t + (rho u^2 + p)_x = 0,  E_t + (u (E + p))_x = 0",
             {"rho": "density", "u": "velocity", "p": "pressure", "E": "total energy",
              "M": "reference Mach number", "eta": "shear viscosity", "zeta": "bulk viscosity"},
             "Compressible Euler with viscosity terms set by eta and zeta; the archive's attributes are per sample."),
    CaseCard("2D_CFD", "2D compressible Euler equations",
             "rho_t + div(rho u) = 0,  (rho u)_t + div(rho u u + p I) = 0,  E_t + div(u (E + p)) = 0",
             {"rho": "density", "u": "velocity vector", "p": "pressure", "E": "total energy",
              "M": "reference Mach number", "eta": "shear viscosity", "zeta": "bulk viscosity"}),
    CaseCard("3D_CFD", "3D compressible Euler equations",
             "rho_t + div(rho u) = 0,  (rho u)_t + div(rho u u + p I) = 0,  E_t + div(u (E + p)) = 0",
             {"rho": "density", "u": "velocity vector", "p": "pressure", "E": "total energy"}),
    CaseCard("2D_DarcyFlow", "2D Darcy flow through a heterogeneous medium",
             "-div(a(x) grad u) = f",
             {"a": "permeability field", "u": "pressure head", "f": "source term", "beta": "contrast parameter"},
             "The archive's 'tensor' is the solution u; 'nu' holds the permeability contrast."),
    CaseCard("2D_diff-react", "2D reaction-diffusion (FitzHugh-Nagumo)",
             "u_t = Du lap(u) + R_u(u, v),  v_t = Dv lap(v) + R_v(u, v)",
             {"u": "activator", "v": "inhibitor", "du": "activator diffusivity", "dv": "inhibitor diffusivity",
              "k": "reaction parameter"},
             "Two coupled fields; the file carries both channels of 'tensor'."),
    CaseCard("ns_incom", "2D incompressible Navier-Stokes (homogeneous isotropic turbulence)",
             "u_t + (u . grad) u = -grad p + nu lap(u),  div(u) = 0",
             {"u": "velocity", "p": "pressure", "nu": "kinematic viscosity"}),
    CaseCard("Sod", "Sod shock tube (1D compressible Euler)",
             "rho_t + (rho u)_x = 0,  (rho u)_t + (rho u^2 + p)_x = 0,  E_t + (u (E + p))_x = 0",
             {"rho": "density", "u": "velocity", "p": "pressure", "E": "total energy"},
             "Sod's shock-tube initial condition (Sod 1978); the archive stores 201 snapshots on x in [-1, 1] for "
             "t in [0, 2.01], dimensionless with rho_L = 1 and p_L = 1.",
             "Sod, G. A. (1978). A survey of several finite difference methods for systems of nonlinear hyperbolic "
             "conservation laws. J. Comput. Phys. 27(1), 1-31."),
)

_BY_FAMILY = {c.family: c for c in _CASES}


def case_card(name: str) -> CaseCard:
    """The card of the case whose family is the longest prefix of ``name`` (a path, a file name or a DaRUS URL)."""
    entry = _dataset_from_source(name)
    if entry is not None:
        return _BY_FAMILY[entry.family]
    stem = re.sub(r"\.(hdf5|h5)$", "", Path(str(name)).name, flags=re.I)
    best = ""
    for family in _BY_FAMILY:
        if stem.startswith(family) and len(family) > len(best):
            best = family
    if best:
        return _BY_FAMILY[best]
    return CaseCard(family="unknown", title=f"PDEBench case {stem}", governing="",
                    notes="No published card for this family in this library; the governing equation was not "
                          "recorded rather than guessed.")


def _dataset_from_source(source: str) -> DatasetFile | None:
    """The curated entry a source names: a URL is matched by its archive id first, then by its file name."""
    text = str(source)
    match = re.search(r"/datafile/(\d+)", text)
    if match:
        for entry in DATASETS.values():
            if entry.darus_id == int(match.group(1)):
                return entry
    stem = re.sub(r"\.(hdf5|h5)$", "", Path(text.split("?")[0]).name, flags=re.I)
    return DATASETS.get(stem)


def dataset(name: str) -> DatasetFile:
    """The curated :class:`DatasetFile` entry for ``name`` (a key, a file name or a family)."""
    stem = re.sub(r"\.(hdf5|h5)$", "", Path(str(name)).name, flags=re.I)
    if stem in DATASETS:
        return DATASETS[stem]
    for entry in DATASETS.values():
        if stem == entry.family or stem.startswith(entry.family + "_"):
            return entry
    raise KeyError(f"'{name}' is not in the curated PDEBench subset: {sorted(DATASETS)}")


@dataclass
class PDEBenchSample:
    """One trajectory of a PDEBench file, in canonical names."""

    fields: dict[str, np.ndarray]           # canonical name -> array (T, *spatial) or (T, *spatial, components)
    coordinates: dict[str, np.ndarray]      # "t", "x", ... -> 1-D coordinates, aligned with the fields
    parameters: dict[str, float] = _dc_field(default_factory=dict)
    raw_fields: dict[str, str] = _dc_field(default_factory=dict)   # canonical -> the name in the file
    notes: list[str] = _dc_field(default_factory=list)


class PDEBenchFile:
    """An open PDEBench HDF5 file: metadata now, samples on demand.

    Use :func:`open_pdebench` (a context manager) rather than constructing this directly, unless you already hold
    an open ``h5py.File``.

    The reader classifies the leading axes of the file before it reads anything: whether the leading axis is a
    *sample* axis, a *time* axis or a *component* axis is not written anywhere in the file, and getting it wrong
    turns a (10000, 201, 1024) dataset into 10000 samples of 201 points -- the kind of mistake that looks like a
    working pipeline. The classification is arithmetic on the coordinate lengths vs the dataset shape, and the
    outcome is kept in :attr:`axis_flags` where a caller can see and disagree with it."""

    def __init__(self, source: str | Path | Any, *, handle: Any = None, keep_open: bool = True,
                 **open_kwargs: Any) -> None:
        self.source = str(source)
        self._ctx = None
        self._keep_open = keep_open
        self.open_kwargs = dict(open_kwargs)
        if handle is not None:
            self._f = handle
        else:
            self._ctx = open_hdf5(self.source, **self.open_kwargs)
            self._f = self._ctx.__enter__()
        self.layout = "groups" if self._looks_like_groups() else "tensor"
        self.notes: list[str] = []
        self.axis_flags: dict[str, bool] = {"samples": False, "time": False, "component": False}
        self._classify()

    # ── layout discovery ──────────────────────────────────────────────────────────────────────────────────
    def _dataset_names(self) -> list[str]:
        return [k for k in self._f.keys() if hasattr(self._f[k], "shape")]

    def _group_names(self) -> list[str]:
        return [k for k in self._f.keys() if not hasattr(self._f[k], "shape")]

    def _looks_like_groups(self) -> bool:
        groups = self._group_names()
        return bool(groups) and all(re.fullmatch(r"\d+", g) for g in groups)

    def _coord_datasets(self) -> dict[str, str]:
        """Coordinate name (``x``) -> where it lives in the file."""
        out: dict[str, str] = {}
        for name in self._dataset_names():
            if name in _COORD_NAMES:
                out[name.split("-")[0]] = name
        if self.layout == "groups" and self._group_names():
            sample = self._f[self._group_names()[0]]
            for name in sample.keys():
                if name in _COORD_NAMES and hasattr(sample[name], "shape"):
                    out.setdefault(name.split("-")[0], f"{sample.name}/{name}")
        return out

    def _field_datasets(self) -> list[str]:
        coords = set(_COORD_NAMES)
        names = []
        for name in self._dataset_names():
            if name in coords:
                continue
            if len(self._f[name].shape) <= 1:      # a per-sample parameter array, not a field
                continue
            names.append(name)
        return names

    def _coord_lengths(self) -> dict[str, int]:
        return {short: int(self._f[full].shape[0]) for short, full in self._coord_datasets().items()}

    def _classify_one(self, shape: Sequence[int]) -> dict[str, int | None]:
        """What the leading axes of one dataset mean: sample, time, component (positions, or ``None``).

        The rule is arithmetic against the coordinate lengths, in the order the archive is known to write them:
        ``(samples, time, *spatial)``, ``(samples, time, component, *spatial)``, ``(time, *spatial)``,
        ``(time, component, *spatial)`` -- plus the case with no time axis at all (Darcy: ``(samples, *spatial)``
        for the permeability, ``(samples, 1, *spatial)`` for the solution)."""
        n_spatial = len(self.spatial_dims)
        lead = list(int(n) for n in shape[: len(shape) - n_spatial]) if n_spatial else list(shape)
        lt = self._coord_lengths().get("t")
        out: dict[str, int | None] = {"samples": None, "time": None, "component": None}
        if len(lead) == 1:
            if _close(lead[0], lt):
                out["time"] = 0
            else:
                out["samples"] = 0
        elif len(lead) == 2:
            if _close(lead[1], lt):
                out.update(samples=0, time=1)
            elif _close(lead[0], lt):
                out.update(time=0, component=1)
            elif lead[1] in (1, 2, 3):
                out.update(samples=0, component=1)
            else:
                out.update(samples=0, time=1)
        elif len(lead) == 3:
            out.update(samples=0, time=1, component=2)
        return out

    def _classify(self) -> None:
        """The file-level summary of what the leading axes mean, taken from its widest field dataset."""
        if self.layout == "groups":
            return
        names = self._field_datasets()
        if not names:
            return
        widest = max(names, key=lambda n: len(self._f[n].shape))
        self.axis_flags = {k: v is not None for k, v in
                           self._classify_one(tuple(int(x) for x in self._f[widest].shape)).items()}

    @property
    def sample_ids(self) -> list[str]:
        return [f"{i:04d}" for i in range(self.n_samples)]

    @property
    def n_samples(self) -> int:
        if self.layout == "groups":
            return len(self._group_names())
        counts = []
        for name in self._field_datasets():
            flags = self._classify_one(tuple(int(x) for x in self._f[name].shape))
            if flags["samples"] is not None:
                counts.append(int(self._f[name].shape[flags["samples"]]))
        return min(counts) if counts else 1

    @property
    def fields(self) -> list[str]:
        """The field names as the file spells them."""
        return self._field_datasets()

    @property
    def spatial_dims(self) -> list[str]:
        return [d for d in _SPATIAL if d in self._coord_datasets()]

    @property
    def parameters(self) -> dict[str, float | np.ndarray]:
        """Per-file scalar attributes and per-sample parameter arrays (``nu``, ``beta``, ``M``, ...)."""
        out: dict[str, float | np.ndarray] = {}
        for key, value in self._f.attrs.items():
            out[str(key)] = value.item() if hasattr(value, "item") else value
        fields = set(self._field_datasets())
        for name in self._dataset_names():
            if name in _COORD_NAMES or name in fields:
                continue
            arr = np.asarray(self._f[name][()])
            if arr.ndim == 0:
                out[name] = float(arr)
            elif arr.ndim == 1:
                out[name] = arr
        return out

    def parameter_of(self, index: int) -> dict[str, float]:
        """The scalar parameters that apply to sample ``index``."""
        out: dict[str, float] = {}
        for key, value in self.parameters.items():
            if isinstance(value, np.ndarray):
                if len(value) == self.n_samples:
                    out[key] = float(value[index])
            elif isinstance(value, (int, float, np.floating)):
                out[key] = float(value)
        return out

    # ── sample reading ───────────────────────────────────────────────────────────────────────────────────
    def _read_coords(self, group: Any | None) -> dict[str, np.ndarray]:
        coords: dict[str, np.ndarray] = {}
        for short, full in self._coord_datasets().items():
            try:
                coords[short] = np.asarray(self._f[full][()])      # the stored dtype, not an upcast
            except (KeyError, ValueError):                        # pragma: no cover - malformed coordinate
                continue
        return coords

    def _read_field(self, dataset: Any, index: int) -> np.ndarray:
        """One field of one sample, with the sample and component axes resolved and removed."""
        shape = tuple(int(x) for x in dataset.shape)
        flags = self._classify_one(shape)
        samples, component = flags["samples"], flags["component"]
        if samples == 0:
            # slice before reading: a 10000 x 128 x 128 field is 655 MB, the one sample needed is 65 kB
            if not 0 <= index < shape[0]:
                raise IndexError(f"sample {index} out of range: this dataset has {shape[0]}")
            values = np.asarray(dataset[index])
            if component is not None and component > 0:
                component -= 1
        else:
            values = np.asarray(dataset[()])
            if samples is not None:
                if not 0 <= index < shape[samples]:
                    raise IndexError(f"sample {index} out of range: this dataset has {shape[samples]}")
                values = np.take(values, index, axis=samples)
                if component is not None and component > samples:
                    component -= 1
        if component is not None:
            values = np.moveaxis(values, component, -1)
            if values.shape[-1] == 1:
                values = values[..., 0]
        return values

    def sample(self, index: int = 0) -> PDEBenchSample:
        """Read one trajectory: canonical fields, their coordinates and the parameters that apply."""
        if self.layout == "groups":
            ids = self._group_names()
            if not 0 <= index < len(ids):
                raise IndexError(f"sample {index} out of range: this file has {len(ids)}")
            group = self._f[ids[index]]
            n_spatial = len(self.spatial_dims)
            raw = {}
            for name in group.keys():
                dataset = group[name]
                if name in _COORD_NAMES or getattr(dataset, "ndim", 0) <= 1:
                    continue
                values = np.asarray(dataset[()])
                if values.ndim - n_spatial == 2:                # (time, component, *spatial)
                    values = np.moveaxis(values, 1, -1)
                    if values.shape[-1] == 1:
                        values = values[..., 0]
                raw[name] = values
            coords = self._read_coords(group)
        else:
            if not -1 <= index < self.n_samples:
                raise IndexError(f"sample {index} out of range: this file has {self.n_samples}")
            raw = {name: self._read_field(self._f[name], index) for name in self._field_datasets()}
            coords = self._read_coords(None)
        fields: dict[str, np.ndarray] = {}
        raw_fields: dict[str, str] = {}
        for raw_name, values in raw.items():
            canonical = _canonical_name(raw_name)
            if canonical in fields and fields[canonical].shape == values.shape:
                # Vx and Vy as separate datasets: one vector field, components on the last axis
                fields[canonical] = np.stack([fields[canonical], values], axis=-1)
            elif canonical in fields:
                fields[f"{canonical}_{raw_name}"] = values     # never silently drop a field
                raw_fields[f"{canonical}_{raw_name}"] = raw_name
            else:
                fields[canonical] = values
                raw_fields[canonical] = raw_name
        sample = PDEBenchSample(fields=fields, coordinates=coords, parameters=self.parameter_of(index),
                                raw_fields=raw_fields, notes=list(self.notes))
        _align_time(sample)
        return sample

    def __enter__(self) -> PDEBenchFile:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = None

    # ── to the universal schema ─────────────────────────────────────────────────────────────────────────
    def to_bundle(self, index: int = 0, *, hash_inputs: bool = True,
                  provenance: Mapping[str, Any] | None = None) -> PhysicsDataBundle:
        """Build a :class:`PhysicsDataBundle` from one sample (see :func:`load_bundle`)."""
        sample = self.sample(index)
        card = case_card(self.source)
        arrays: dict[str, np.ndarray] = {}
        axes: list[AxisSpec] = []
        axis_order = tuple(n for n in ("t",) + _SPATIAL if n in sample.coordinates)
        for name in axis_order:
            arrays[f"axis:{name}"] = sample.coordinates[name]
            axes.append(AxisSpec(name, "1", description=f"PDEBench {name}-coordinate"))
        axis_names = tuple(a.name for a in axes)
        fields = []
        for name, arr in sample.fields.items():
            arrays[f"field:{name}"] = arr
            components = list(_COMPONENTS[: arr.shape[-1]]) if arr.ndim == len(axis_names) + 1 else [
                "u"] if name == "velocity" else []
            fields.append(Field.from_array(name, arr, unit="1", location="nodal", axes=axis_names,
                                           components=components))
        if is_url(self.source):
            # a remote object cannot be hashed without downloading it; the fingerprint recorded is of the bytes
            # that were actually read into this bundle, which is what a later comparison cares about
            inputs = [FileRef(path=self.source, role="remote-sample", sha256=_content_digest(arrays))]
        else:
            path = Path(self.source)
            size = path.stat().st_size
            inputs = [FileRef(path=str(path), bytes=size,
                              sha256=sha256_file(str(path)) if hash_inputs and size < 2 << 30 else "")]
        physics = Physics(
            governing=card.governing, symbols=dict(card.symbols),
            parameters=[_parameter(k, v) for k, v in sample.parameters.items()],
            notes=" ".join(x for x in (card.notes,
                                       "Values are dimensionless in the archive's own non-dimensionalisation; the "
                                       "units are recorded as \"1\" instead of being invented.") if x))
        metadata = {"pdebench": {"file": Path(self.source).name, "layout": self.layout, "sample": index,
                                 "n_samples": self.n_samples, "axis_flags": dict(self.axis_flags),
                                 "raw_fields": sample.raw_fields, "notes": sample.notes, "case": card.family}}
        bundle = PhysicsDataBundle.from_arrays(
            arrays, fields=fields, axes=axes, physics=physics,
            source=f"PDEBench {card.family} :: {Path(self.source).name}", license=DARUS_LICENSE,
            title=f"PDEBench {card.family} :: {Path(self.source).name}",
            id=f"pdebench-{card.family.lower()}-{index:04d}",
            geometry=Geometry(kind="structured-grid", dimension=max(1, len(axis_names) - 1), units="1",
                              description=f"{card.title}; domain read from the file's coordinate arrays"),
            metadata=metadata, inputs=inputs, adapter="pinneapple_data.datasets.pdebench", adapter_version="1",
            citation=f"{PAPER} Data: {DARUS_URL}", url=self.source,
            config={"sample_index": index, "layout": self.layout, **{k: v for k, v in (provenance or {}).items()}})
        for note in sample.notes:
            bundle.check("time_alignment", tool="pinneapple_data.datasets.pdebench", detail=note)
        return bundle


def _parameter(name: str, value: float) -> Parameter:
    return Parameter(name=name, value=float(value) if np.isfinite(value) else None, unit="1",
                     source="PDEBench file attribute")


def _close(a: int | None, b: int | None, tolerance: int = 1) -> bool:
    """Whether two axis lengths are the same axis, allowing the one-step offset PDEBench files carry."""
    return a is not None and b is not None and abs(int(a) - int(b)) <= tolerance


def _canonical_name(raw_name: str) -> str:
    """The canonical name of a raw field: ``Vx``/``Vy``/``Vz``/``velocity`` fold into ``velocity``."""
    base = raw_name.strip()
    if base in ("Vx", "Vy", "Vz", "velocity"):
        return "velocity"
    if base == "tensor":
        return "u"
    return base


def _align_time(sample: PDEBenchSample) -> None:
    """Make ``t`` and the fields index the same steps, recording it when the file disagrees with itself."""
    t = sample.coordinates.get("t")
    if t is None or not sample.fields:
        return
    n = min(arr.shape[0] for arr in sample.fields.values())
    if len(t) != n:
        keep = min(len(t), n)
        sample.notes.append(
            f"the t-coordinate has {len(t)} entries and the fields {n}: the shorter was kept (first {keep} "
            "steps), because a time axis longer than its data silently mislabels every snapshot")
        sample.coordinates["t"] = t[:keep]
        for name, arr in list(sample.fields.items()):
            if arr.shape[0] != keep:
                sample.fields[name] = arr[:keep]


@contextmanager
def open_pdebench(source: str | Path, **kwargs: Any) -> Iterator[PDEBenchFile]:
    """Open a PDEBench file (a local path or a DaRUS ``http(s)`` URL) and close it afterwards."""
    handle = PDEBenchFile(source, **kwargs)
    try:
        yield handle
    finally:
        handle.close()


def load_bundle(source: str | Path, index: int = 0, **kwargs: Any) -> PhysicsDataBundle:
    """Read sample ``index`` of a PDEBench file into a :class:`PhysicsDataBundle`."""
    with open_pdebench(source) as handle:
        return handle.to_bundle(index, **kwargs)


def download(case: str | DatasetFile, root: str | os.PathLike[str] = "data/pdebench",
             *, overwrite: bool = False) -> Path:
    """Download one curated PDEBench file to ``root`` (skipped when it is already there).

    A full file is 1.3-88 GB and the adapters do not need it (they read samples over range requests); this exists
    for the cases where a local copy is genuinely wanted -- an offline benchmark, or a checksum to record."""
    entry = case if isinstance(case, DatasetFile) else dataset(case)
    target = Path(root) / entry.filename
    if target.exists() and not overwrite:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(entry.url, timeout=120) as response, open(tmp, "wb") as fh:
        while chunk := response.read(1 << 22):
            fh.write(chunk)
    os.replace(tmp, target)
    return target

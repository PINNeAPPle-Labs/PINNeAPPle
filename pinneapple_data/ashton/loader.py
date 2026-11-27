"""Open an Ashton-family dataset on disk, read one case's surface and forces, and integrate the drag.

The three datasets share a layout but not an exact file set (:mod:`pinneapple_data.ashton.catalog`), so this
module does the two things a shared reader has to do well: find the file a case actually has, and say plainly
when it is not there. Everything is lazy -- opening a dataset reads nothing, a case reads only the files the
caller asks for, and the surface reader decodes only the arrays named (:mod:`pinneapple_data.ashton.vtk_xml`).

    >>> from pinneapple_data.ashton import loader
    >>> data = loader.open_dataset("ahmedml", root="~/data/ahmedml")
    >>> data.n_cases, data.licence
    (500, 'CC-BY-SA-4.0')
    >>> case = data.case(1)
    >>> case.forces["cd"]                                  # from force_mom_1.csv, the dataset's own number
    0.23848566658
    >>> measured = data.measured_forces(1)                 # integrated from boundary_1.vtp's Cp
    >>> measured.cd, measured.cd_published                 # the two routes, side by side

**Measuring drag from the surface.** ``C_D = F_x / (0.5 rho U^2 A_ref)`` (AhmedML Eq. 35, DrivAerML and
WindsorML likewise), and ``F_x`` is the pressure integrated over the closed surface: with ``n`` the outward
normal, ``F = -oint p n dA``. In coefficients that is ``C_D = -oint C_p n_x dA / A_ref`` -- no velocity and no
density, which is why the Cp route is the one to benchmark on: the three datasets normalise with different
reference quantities (1 m/s and rho = 1 for AhmedML, 38.889 m/s for DrivAerML, 40 m/s and rho = 1.31 for
WindsorML), and a coefficient integrated from Cp is free of that choice while one integrated from pressure in
pascals is not.

The quadrature is the dataset's own: an area per native polygon for the two OpenFOAM datasets, a weight per
point for WindsorML (:func:`measured_forces` takes the published array when it is there and computes the areas
from the mesh when it is not, and reports which of the two it used).

**Not read here.** The geometry (``*.stl``, ``*.step``) and the volume field (5.6 GB to 44 GB per case) are
never opened: :meth:`AshtonData.volume_subsample` reads a chosen number of points out of the volume file *by
byte range*, which is the only form in which a 44 GB file is usable in a study.
"""
from __future__ import annotations

import base64
import csv
import os
import random
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..datasets.download import fetch_cached
from .catalog import LICENCE, LICENCE_URL, AshtonDataset, layout_notes
from .catalog import dataset as _dataset
from .vtk_xml import Surface, face_area_normals, node_area_normals, read_surface

__all__ = ["CaseNotPresent", "Case", "AshtonData", "MeasuredForces", "open_dataset", "resolve_file", "fetch",
           "read_forces", "read_parameters", "list_arrays", "resolve_field", "volume_subsample",
           "case_bundle"]

#: Extensions of the pressure fields, so a name is recognised whether the writer spelled it Cp, CpMean or
#: pressure_coefficient. The catalogue holds the candidates; this is the fallback that reads them from the file.
_DIMENSIONLESS = re.compile(r"^(cp|c_p|cpmean|cpt|cpm)$", re.IGNORECASE)


class CaseNotPresent(FileNotFoundError):
    """A file of a case is not on disk. The message carries the URL and the call that fetches it.

    A ``FileNotFoundError`` subclass so that code which already handles a missing file keeps working, and its
    own name so that the message -- which is the useful part -- can be recognised in a benchmark log.
    """


@dataclass
class MeasuredForces:
    """Drag, side and lift from a case's surface field, with the route that produced them written down."""

    dataset: str
    run: int
    cd: float
    cs: float
    cl: float
    a_ref: float
    field: str                     # the surface field the integral used
    location: str                  # "point" or "cell"
    quadrature: str                # "published" (the dataset's own array) or "computed" (from the faces)
    n_points: int
    n_cells: int
    area: float                    # the weighted area the integral actually used, m^2
    face_area: float               # the surface area of the faces themselves, m^2 (the same number for any
    #                                quadrature that covers the body, so it is what two routes are compared on)
    published_area: float | None   # sum of the quadrature weights, when the dataset publishes them
    integrated_normal_ratio: float  # |oint n dA| / A: ~0 for a closed body, 1 for an open sheet
    cd_published: float | None = None   # the coefficient the dataset's own force_mom file reports
    skipped_arrays: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def cd_error(self) -> float | None:
        """``cd - cd_published``: the two routes to the same number, or None when the CSV is not there."""
        return None if self.cd_published is None else float(self.cd - self.cd_published)

    def to_dict(self) -> dict[str, Any]:
        return {"dataset": self.dataset, "run": self.run, "cd": self.cd, "cs": self.cs, "cl": self.cl,
                "cd_published": self.cd_published, "cd_error": self.cd_error, "a_ref": self.a_ref,
                "field": self.field, "location": self.location, "quadrature": self.quadrature,
                "n_points": self.n_points, "n_cells": self.n_cells, "area": self.area,
                "face_area": self.face_area, "published_area": self.published_area,
                "integrated_normal_ratio": self.integrated_normal_ratio, "notes": list(self.notes)}


@dataclass
class Case:
    """One case of one dataset: the files it has on disk, its published coefficients and its parameters."""

    dataset: AshtonDataset
    root: Path
    run: int
    files: dict[str, Path] = field(default_factory=dict)
    forces: dict[str, float] = field(default_factory=dict)
    parameters: dict[str, float] = field(default_factory=dict)
    varref: dict[str, float] = field(default_factory=dict)

    @property
    def licence(self) -> str:
        return self.dataset.licence

    @property
    def attribution(self) -> str:
        return self.dataset.attribution

    @property
    def name(self) -> str:
        return f"{self.dataset.id}/run_{self.run}"

    @property
    def surface_path(self) -> Path:
        return self.files["surface"]

    def surface(self, arrays: Iterable[str] | None = None) -> Surface:
        return read_surface(self.surface_path, arrays=arrays)

    def field_of(self, role: str = "cp") -> tuple[str, np.ndarray, str]:
        """Resolve one of the dataset's surface fields (``"cp"``, ``"pressure"``, ``"wall_shear"``, ...)."""
        names_only = self.surface(arrays=())
        name, location = resolve_field(names_only, self.dataset, role, path=self.surface_path)
        read = self.surface(arrays=[name])
        return name, np.asarray(read.point_data.get(name, read.cell_data.get(name)), dtype=np.float64), location

    def to_dict(self) -> dict[str, Any]:
        return {"case": self.name, "dataset": self.dataset.id, "run": self.run, "licence": self.licence,
                "attribution": self.attribution, "files": {k: str(v) for k, v in self.files.items()},
                "forces": self.forces, "parameters": self.parameters}


# ---------------------------------------------------------------------------
# finding files
# ---------------------------------------------------------------------------

def resolve_file(data: AshtonDataset, which: str, run: int, root: str | os.PathLike[str]) -> Path:
    """The path of one file of one case, or :class:`CaseNotPresent` naming the URL that fetches it.

    Two spellings are tried: ``<name>_<run>`` and ``<name>_1``, because the AhmedML card spells
    ``geo_parameters_1.csv`` for every run while the files themselves are per-run. A file that is genuinely
    absent (DrivAerML's 16 withdrawn runs, AhmedML's run 500 parameters, a volume that was never downloaded)
    raises rather than returning a path that does not exist.
    """
    rel = data.file(which, run)
    base, suffix = rel.rsplit(".", 1)
    stem = base.rsplit("_", 1)[0]
    candidates = [root / rel, root / f"{stem}_1.{suffix}"]
    for candidate in candidates:
        if Path(candidate).exists():
            return Path(candidate)
    raise CaseNotPresent(
        f"{rel} is not under {root}. Fetch it with "
        f"fetch({which!r}, {int(run)}, root={str(root)!r}, dataset={data.id!r}), or by hand from "
        f"{data.resolve_url}/{rel} ({data.size}). The layout is:\n{layout_notes()}")


def fetch(which: str, run: int, *, root: str | os.PathLike[str] = ".", dataset: str | AshtonDataset = "ahmedml",
          tries: int = 4, timeout: float = 600.0) -> Path:
    """Stream one file of one case into ``root`` under its dataset name, and return the path.

    For a 66 KB ``force_mom_i.csv`` this is what makes a study of all 500 cases possible without the terabytes;
    for a 44 GB volume it is the caller's decision, not a side effect of opening a case. The transfer reuses
    :func:`pinneapple_data.datasets.download.fetch_cached` (chunked, retried, written to a temporary file and
    renamed) and then puts the file under the name the dataset gives it: an interrupted download never leaves
    a truncated ``boundary_i.vtp`` that the next reader would take for the real one.
    """
    data = dataset if isinstance(dataset, AshtonDataset) else _dataset(dataset)
    rel = data.file(which, run)
    target = Path(root) / rel
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    cached = fetch_cached(f"{data.resolve_url}/{rel}", target.parent, tries=tries, timeout=timeout)
    os.replace(cached, target)
    return target


def read_force_csv(path: str | os.PathLike[str]) -> dict[str, float]:
    """Read a ``force_mom_i.csv``: one header row of coefficient names and one row of values."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.reader(handle) if row and any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError(f"{path}: empty forces file")
    if len(rows) == 1:                                   # a single row: names only, no values
        raise ValueError(f"{path}: {rows[0]} are names but the file has no values")
    names = [cell.strip() for cell in rows[0]]
    values: dict[str, float] = {}
    for name, cell in zip(names, rows[1], strict=False):
        try:
            values[name.lower()] = float(cell)
        except ValueError:
            continue                                     # a text column (a case id) is not a coefficient
    if not values:
        raise ValueError(f"{path}: no numeric coefficient under {names}")
    return values


read_forces = read_force_csv


def read_parameters(path: str | os.PathLike[str]) -> dict[str, float]:
    """Read a ``geo_parameters_i.csv``: either ``name,value`` rows or a header row plus a value row.

    The three cards do not say which, and the files differ (AhmedML's is 283 bytes, DrivAerML's 66 KB), so both
    shapes are accepted and anything else raises with what was seen rather than returning a partial dict.
    """
    with open(path, newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.reader(handle) if row and any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError(f"{path}: empty parameters file")
    two_column = all(len(row) == 2 for row in rows)
    words_then_numbers = two_column and all(not _is_number(row[0]) for row in rows) and \
        all(_is_number(row[1]) for row in rows)
    if words_then_numbers:                              # name, value
        return {_clean(name): float(value) for name, value in rows}
    names = [_clean(name) for name in rows[0]]
    if len(rows) >= 2:
        out = {}
        for name, cell in zip(names, rows[1], strict=False):
            try:
                out[name] = float(cell)
            except ValueError:
                continue
        if out:
            return out
    raise ValueError(f"{path}: {len(rows)} row(s) of {len(rows[0])} column(s) are neither 'name,value' nor "
                     "'header + values'; the first row is {rows[0][:6]}")


def _is_number(cell: str) -> bool:
    try:
        float(cell)
        return True
    except ValueError:
        return False


def _clean(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower()).strip("_")


def list_arrays(path: str | os.PathLike[str]) -> tuple[str, ...]:
    """The DataArrays a surface file holds, decoded none of them: one pass, no payload."""
    surface = read_surface(path, arrays=())
    return tuple(sorted({"Points", *surface.available_points, *surface.available_cells}))


def resolve_field(surface: Surface, data: AshtonDataset, role: str = "cp", *,
                  path: str | os.PathLike[str] | None = None) -> tuple[str, str]:
    """Find the array that plays ``role`` in this surface: ``(name, "point" | "cell")``.

    The catalogue's candidate names are tried first (case-insensitively), then a shape-based fallback: any
    point/cell array whose name reads like a pressure coefficient. When neither finds it, the error lists what
    the file does hold -- the one thing a caller can act on.
    """
    available = {"point": list(surface.available_points or surface.point_data),
                 "cell": [name for name in (surface.available_cells or surface.cell_data)
                          if name not in ("connectivity", "offsets", "types")]}
    folded = {kind: {name.lower(): name for name in names} for kind, names in available.items()}
    for candidate in data.fields.get(role, ()):
        for kind in ("point", "cell"):
            if candidate.lower() in folded[kind]:
                return folded[kind][candidate.lower()], kind
    if role == "cp":
        for kind in ("point", "cell"):
            for name in available[kind]:
                if _DIMENSIONLESS.match(name):
                    return name, kind
    raise KeyError(f"{path or data.name}: no surface array plays '{role}'. The candidate names are "
                   f"{list(data.fields.get(role, ()))} and the file holds point arrays {available['point']} and "
                   f"cell arrays {available['cell']}")


# ---------------------------------------------------------------------------
# the dataset object
# ---------------------------------------------------------------------------

@dataclass
class AshtonData:
    """One dataset on disk: the catalogue entry, where it is, and the reads a case supports."""

    dataset: AshtonDataset
    root: Path

    # -- identity, licence and credit --------------------------------------------------------------

    @property
    def id(self) -> str:
        return self.dataset.id

    @property
    def name(self) -> str:
        return self.dataset.name

    @property
    def licence(self) -> str:
        return LICENCE

    @property
    def licence_url(self) -> str:
        return LICENCE_URL

    @property
    def attribution(self) -> str:
        return self.dataset.attribution

    @property
    def citation(self) -> str:
        return self.dataset.attribution

    @property
    def n_cases(self) -> int:
        return self.dataset.n_cases

    def to_dict(self) -> dict[str, Any]:
        return {**self.dataset.to_dict(), "root": str(self.root)}

    # -- cases -------------------------------------------------------------------------------------

    def runs(self) -> list[int]:
        """The run indices present on disk (an empty list when the dataset has not been downloaded)."""
        if not self.root.is_dir():
            return []
        found = []
        for path in sorted(self.root.glob("run_*")):
            match = re.fullmatch(r"run_(\d+)", path.name)
            if match and path.is_dir():
                found.append(int(match.group(1)))
        return found

    def cases(self) -> Iterator[Case]:
        for run in self.runs():
            yield self.case(run)

    def case(self, run: int) -> Case:
        """One case, with the files it has and its published coefficients and parameters.

        The surface and the geometry are optional here -- a case can be opened with only ``force_mom_i.csv``,
        which is what a 500-case table study needs.
        """
        run = int(run)
        on_disk = (self.root / f"run_{run}").is_dir()
        if self.dataset.runs and run not in self.dataset.runs and not on_disk:
            raise CaseNotPresent(f"{self.dataset.id} has no run {run}: the card lists "
                                 f"{min(self.dataset.runs)}..{max(self.dataset.runs)}")
        if run in self.dataset.withdrawn and not on_disk:
            raise CaseNotPresent(f"{self.dataset.id} run {run} is not published: the card lists it as withheld")
        files: dict[str, Path] = {}
        for which in ("surface", "areas", "forces", "forces_varref", "parameters", "geometry", "volume"):
            try:
                files[which] = resolve_file(self.dataset, which, run, self.root)
            except CaseNotPresent:
                continue                                  # a case without a volume or a varref file is normal
        if "forces" not in files and "surface" not in files:
            forces = self.dataset.file("forces", run)
            raise CaseNotPresent(
                f"{self.dataset.id} run {run}: neither {forces} nor {self.dataset.file('surface', run)} is under "
                f"{self.root}. Fetch one with fetch('forces', {run}, root={str(self.root)!r}, "
                f"dataset={self.dataset.id!r}), or by hand from {self.dataset.resolve_url}/{forces} "
                f"({self.dataset.size})")
        forces = read_force_csv(files["forces"]) if "forces" in files else {}
        varref = read_force_csv(files["forces_varref"]) if "forces_varref" in files else {}
        parameters = read_parameters(files["parameters"]) if "parameters" in files else {}
        return Case(dataset=self.dataset, root=self.root, run=run, files=files, forces=forces,
                    parameters=parameters, varref=varref)

    # -- measurements ------------------------------------------------------------------------------

    def measured_forces(self, run: int, *, cp_field: str | None = None, use_published_areas: bool = True,
                        a_ref: float | None = None) -> MeasuredForces:
        """Drag, side and lift of one case, integrated from its own surface pressure field.

        See the module docstring for the integral. ``use_published_areas`` decides the quadrature: the
        dataset's published array (an area per polygon, or a weight per point) when it is there, and the areas
        computed from the faces when it is not -- and the result says which, because the two differ: the
        published array belongs to the *native* cells of the CFD mesh and a re-triangulated surface does not
        reproduce it.
        """
        case = self.case(run)
        name, location = self._resolve(case, cp_field)
        surface = case.surface(arrays=[name])
        values = np.asarray(surface.point_data.get(name, surface.cell_data.get(name)), dtype=np.float64)
        reference = float(a_ref if a_ref is not None else self.dataset.a_ref)
        if reference <= 0:
            raise ValueError(f"{self.dataset.id} declares no reference area; pass a_ref=")
        notes: list[str] = []
        published = None
        if use_published_areas and "areas" in case.files:
            published = np.asarray(np.load(case.files["areas"]).reshape(-1), dtype=np.float64)
            if published.size != (surface.n_points if self.dataset.quadrature == "point" else surface.n_cells):
                notes.append(f"the published quadrature has {published.size} entries against "
                             f"{surface.n_points if self.dataset.quadrature == 'point' else surface.n_cells} "
                             "of the surface; the areas computed from the faces are used instead")
                published = None
        dimensionless = bool(_DIMENSIONLESS.match(name))
        face_area = float(face_area_normals(surface)[0].sum())
        if location == "cell":
            areas, normals = face_area_normals(surface)
            weight = published if published is not None else areas
            force = -(values * weight) @ normals                      # (3,) = -oint p n dA
            area = float(weight.sum())
            normal_ratio = float(np.linalg.norm((weight[:, None] * normals).sum(axis=0)) / max(area, 1e-30))
            quadrature = "published" if published is not None else "computed"
        else:
            node_area, node_normal = node_area_normals(surface)
            weight = published if published is not None else node_area
            force = -(values * weight) @ node_normal
            area = float(weight.sum())
            normal_ratio = float(np.linalg.norm((weight[:, None] * node_normal).sum(axis=0)) / max(area, 1e-30))
            quadrature = "published" if published is not None else "computed"
        if dimensionless:
            coefficients = force / reference
        else:
            dynamic = 0.5 * self.dataset.density * self.dataset.velocity ** 2
            if dynamic <= 0:
                raise ValueError(f"{self.dataset.id}: the surface field {name!r} is a pressure, and no "
                                 "freestream is declared to normalise it; use the Cp field")
            notes.append(f"{name!r} is a pressure; it is divided by q = 0.5*rho*U^2 = {dynamic:.4g} Pa "
                         f"(rho = {self.dataset.density}, U = {self.dataset.velocity})")
            coefficients = force / (dynamic * reference)
        published_cd = case.forces.get("cd")
        return MeasuredForces(
            dataset=self.dataset.id, run=int(run), cd=float(coefficients[0]), cs=float(coefficients[1]),
            cl=float(coefficients[2]), a_ref=reference, field=name, location=location, quadrature=quadrature,
            n_points=surface.n_points, n_cells=surface.n_cells, area=area, face_area=face_area,
            published_area=None if published is None else float(published.sum()),
            integrated_normal_ratio=normal_ratio,
            cd_published=None if published_cd is None else float(published_cd),
            skipped_arrays=surface.skipped, notes=tuple(notes))

    def _resolve(self, case: Case, cp_field: str | None) -> tuple[str, str]:
        if cp_field is not None:
            surface = case.surface(arrays=())
            for kind in ("point", "cell"):
                names = surface.point_data if kind == "point" else surface.cell_data
                if cp_field in names:
                    return cp_field, kind
            raise KeyError(f"{case.surface_path} has no array {cp_field!r}")
        surface = case.surface(arrays=())
        return resolve_field(surface, self.dataset, "cp", path=case.surface_path)

    def volume_subsample(self, run: int, count: int = 1000, *, seed: int = 0,
                         fields: Iterable[str] | None = None) -> dict[str, np.ndarray]:
        """``count`` points of one case's volume field and their values, read by byte range.

        The volume file is 5.6 GB (AhmedML), 20 GB (WindsorML) or 44 GB (DrivAerML): reading it to pick a
        thousand points is not a workflow. The VTK XML file is one long text document of base64 arrays, so a
        point range maps to a character range once the block header is accounted for, and each sampled run of
        points is one seek and one read -- which is what :func:`pinneapple_data.datasets.remote_hdf5.
        HTTPRangeFile` already does for HDF5 and is what :func:`_range_read` uses here for VTK.
        """
        case = self.case(run)
        if "volume" not in case.files:
            raise CaseNotPresent(f"{case.name} has no volume file on disk; fetch('volume', {run}, root=..., "
                                 f"dataset='{self.dataset.id}') is {self.dataset.volume} (tens of GB)")
        return volume_subsample(case.files["volume"], count, seed=seed, fields=fields)

    def case_bundle(self, run: int, *, fields: Iterable[str] | None = None) -> Any:
        return case_bundle(self, run, fields=fields)


def open_dataset(name: str | AshtonDataset, *, root: str | os.PathLike[str] | None = None) -> AshtonData:
    """Open a dataset: ``open_dataset("ahmedml", root="~/data/ahmedml")``.

    With no ``root`` the conventional place is tried (``$PINNEAPPLE_DATA/ashton/<id>``, then
    ``data/ashton/<id>``); a dataset that is not there is still open -- it reports zero cases and a case read
    raises with the URL -- because a table built from published numbers must be buildable without the data.
    """
    data = name if isinstance(name, AshtonDataset) else _dataset(name)
    if root is not None:
        return AshtonData(dataset=data, root=Path(root).expanduser())
    env = os.environ.get("PINNEAPPLE_DATA")
    conventional = [Path("data/ashton") / data.id]
    if env:
        conventional.insert(0, Path(env).expanduser() / "ashton" / data.id)
    for candidate in conventional:
        if candidate.is_dir():
            return AshtonData(dataset=data, root=candidate)
    return AshtonData(dataset=data, root=conventional[-1])


# ---------------------------------------------------------------------------
# volume subsampling by byte range
# ---------------------------------------------------------------------------

def _scan_ranges(path: str | os.PathLike[str]) -> tuple[dict[str, tuple[int, int, dict[str, str], str]],
                                                       dict[str, str], str]:
    """Every DataArray of a VTK XML file as ``(char offset, char length, attributes, parent)``.

    One pass over the file that reads it but decodes nothing: the scan finds the tags and the payload extents,
    and the extents are what make a range read of the payload possible afterwards. ``<Piece>`` and the root
    attributes come back too -- ``NumberOfPoints`` is a fact the file states, so it does not have to be
    re-derived from a byte count.
    """
    file = Path(path)
    ranges: dict[str, tuple[int, int, dict[str, str], str]] = {}
    tags: dict[str, str] = {}
    header_type = "UInt64"
    stack: list[bytes] = []
    buf, pos, base = b"", 0, 0
    close_tag = b"</DataArray>"

    def fill() -> bool:
        nonlocal buf, pos, base
        chunk = file_handle.read(1 << 22)
        if not chunk:
            return False
        base += pos
        buf, pos = buf[pos:], 0
        buf += chunk
        return True

    with open(file, "rb") as handle:
        file_handle = handle
        while True:
            start = buf.find(b"<", pos)
            if start < 0:
                if not fill():
                    return ranges, tags, header_type
                continue
            end = buf.find(b">", start)
            if end < 0:
                if not fill():
                    return ranges, tags, header_type
                continue
            tag = buf[start:end + 1]
            pos = end + 1
            if tag.startswith(b"</"):
                name = tag[2:-1].strip()
                if stack and stack[-1] == name:
                    stack.pop()
                continue
            name = tag[1:].split(b" ", 1)[0].split(b">", 1)[0].split(b"/", 1)[0].strip()
            if name in (b"Piece", b"VTKFile"):
                tags[name.decode()] = tag.decode("latin-1")
                if name == b"VTKFile":
                    match = re.search(r"header_type=['\"](\w+)['\"]", tags["VTKFile"])
                    if match:
                        header_type = match.group(1)
            if tag.endswith(b"/>"):
                continue
            stack.append(name)
            if name != b"DataArray":
                continue
            while close_tag not in buf[pos:]:
                if not fill():
                    raise ValueError(f"{file}: the file ends inside a <DataArray>")
            length = buf[pos:].find(close_tag)
            attrs = {k: (a or b) for k, a, b in
                     re.findall(r"""([A-Za-z_][\w:.-]*)\s*=\s*(?:'([^']*)'|"([^"]*)")""", tag.decode("latin-1"))}
            ranges[attrs.get("Name", "")] = (base + pos, length, attrs,
                                             (stack[-2] if len(stack) > 1 else b"").decode())
            pos += length + len(close_tag)
            stack.pop()


def _read_chars(path: str | os.PathLike[str], offset: int, length: int) -> str:
    with open(path, "rb") as handle:
        handle.seek(offset)
        return handle.read(length).decode("latin-1")


def _decode_selected(file: Path, entry: tuple[int, int, dict[str, str], str], header: str,
                     index: np.ndarray) -> np.ndarray:
    """Decode the elements ``index`` of one binary DataArray, one short read per contiguous run of indices.

    base64 puts three bytes in four characters, so a byte range is reachable from any three-byte alignment:
    the encoding is re-anchored at ``(start // 3) * 3`` and the leading ``(start % 3)`` bytes are dropped.
    That is how a hundred points out of 160 million cost a hundred short reads instead of a 44 GB download --
    and it is exact, not an approximation: no interpolation, the bytes of the chosen points and no others.
    """
    offset, _length, attrs, _parent = entry
    dtype = np.dtype({"Float32": "<f4", "Float64": "<f8", "Int32": "<i4", "Int64": "<i8", "UInt8": "u1",
                      "UInt16": "<u2", "UInt32": "<u4", "UInt64": "<u8"}.get(attrs.get("type", "Float32"), "<f4"))
    header_dtype = np.dtype("<u8") if header.lower() == "uint64" else np.dtype("<u4")
    header_size = header_dtype.itemsize
    components = int(attrs.get("NumberOfComponents", "1") or 1)
    item = dtype.itemsize
    out: list[np.ndarray] = []
    for run in np.split(index, np.flatnonzero(np.diff(index) > 1) + 1):
        if run.size == 0:
            continue
        start = header_size + int(run[0]) * components * item
        stop = header_size + (int(run[-1]) + 1) * components * item
        group = (start // 3) * 3
        chars = ((stop - group) + 2) // 3 * 4 + 8
        text = _read_chars(file, offset + (group // 3) * 4, chars)
        raw = base64.b64decode(text)[start - group: start - group + (stop - start)]
        values = np.frombuffer(raw, dtype=dtype)
        if components > 1:
            values = values.reshape(-1, components)[(run - int(run[0])).astype(int)]
        out.append(values)
    if not out:
        return np.zeros((0,), dtype=dtype)
    return np.concatenate(out)


def volume_subsample(path: str | os.PathLike[str], count: int = 1000, *, seed: int = 0,
                     fields: Iterable[str] | None = None) -> dict[str, np.ndarray]:
    """Sample ``count`` points of a volume ``.vtu`` and their point fields, without reading the file whole.

    The points are drawn uniformly without replacement from the point count the file states in ``<Piece>``,
    the sample is sorted (so the reads are contiguous runs), and each field -- by default every array in
    ``<PointData>`` -- is decoded at exactly those elements. The result is
    ``{"points": (count, 3), "<field>": (count,), "_index": (count,)}``: the sampled indices come back so a
    point can be found again in the file, and the same seed gives the same points on the same file.
    """
    file = Path(path)
    ranges, tags, header = _scan_ranges(file)
    if "Points" not in ranges:
        raise ValueError(f"{file}: no Points array, so it is not a VTK XML volume file")
    piece = tags.get("Piece", "")
    match = re.search(r"NumberOfPoints=['\"](\d+)['\"]", piece)
    if match is None:
        raise ValueError(f"{file}: <Piece> states no NumberOfPoints; this reader takes the count from the file "
                         "rather than guessing it from a byte length")
    total = int(match.group(1))
    count = int(count)
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if count >= total:
        index = np.arange(total, dtype=np.int64)
    else:
        index = np.sort(np.array(random.Random(seed).sample(range(total), count), dtype=np.int64))
    out = {"points": _decode_selected(file, ranges["Points"], header, index).reshape(-1, 3), "_index": index}
    wanted = list(fields) if fields is not None else [name for name, (_o, _l, _a, parent) in ranges.items()
                                                      if parent == "PointData" and name != "Points"]
    for name in wanted:
        if name not in ranges or name == "Points":
            raise KeyError(f"{file} has no point array {name!r}; it holds "
                           f"{sorted(n for n, (_o, _l, _a, p) in ranges.items() if p == 'PointData')}")
        out[name] = _decode_selected(file, ranges[name], header, index)
    return out


# ---------------------------------------------------------------------------
# bundle
# ---------------------------------------------------------------------------

def case_bundle(data: AshtonData | str, run: int, *, fields: Iterable[str] | None = None) -> Any:
    """One case's surface as a :class:`~pinneapple_data.schema.PhysicsDataBundle`.

    A bundle has slots for the mesh, the fields, the parameters and the provenance -- licence, citation, url,
    inputs -- which is exactly the set a dataset has to keep attached to a case. The force coefficients go to
    ``metadata``: they are per-case results, not a field, and bending them into one would lose the unit.
    """
    from ..schema import Field, Geometry, Mesh, Physics, PhysicsDataBundle
    from ..schema.bundle import file_refs
    from ..schema.models import ArrayRef, Block, Parameter

    opened = data if isinstance(data, AshtonData) else open_dataset(data)
    case = opened.case(run)
    surface = read_surface(case.surface_path, arrays=list(fields) if fields is not None else None)
    shapes = surface.cells_per_shape
    if len(shapes) != 1:
        raise ValueError(f"{case.name}: a bundle block carries one element type and this surface mixes "
                         f"{shapes}; read it with open_dataset(...).case({run}).surface() instead")
    size = next(iter(shapes))
    cells = surface.cells_of(size)
    cell_type = {3: "triangle", 4: "quad"}.get(size)
    if cell_type is None:
        raise ValueError(f"{case.name}: {size} vertices per cell is not a bundle element type")
    store: dict[str, np.ndarray] = {"mesh:nodes": surface.points, f"block:{cell_type}": cells}
    declared = []
    for name, values in surface.point_data.items():
        unit = "1" if _DIMENSIONLESS.match(name) else "unknown"
        store[f"field:{name}"] = np.asarray(values)
        declared.append(Field.from_array(name, np.asarray(values), unit=unit, location="nodal", axes=["node"],
                                        description=f"{case.name}: {name} on the surface"))
    for name, values in surface.cell_data.items():
        unit = "1" if _DIMENSIONLESS.match(name) else "unknown"
        store[f"field:{name}"] = np.asarray(values)
        declared.append(Field.from_array(name, np.asarray(values), unit=unit, location="cell", axes=["cell"],
                                        description=f"{case.name}: {name} on the surface faces"))
    inputs = file_refs([str(path) for path in case.files.values()])
    return PhysicsDataBundle.from_arrays(
        store, fields=declared, axes=[],
        mesh=Mesh(kind="unstructured", dimension=3,
                  nodes=ArrayRef(key="mesh:nodes", shape=tuple(int(n) for n in surface.points.shape),
                                 dtype=str(surface.points.dtype)),
                  blocks=[Block(type=cell_type, name="surface",
                                data=ArrayRef(key=f"block:{cell_type}", shape=tuple(int(n) for n in cells.shape),
                                              dtype=str(cells.dtype)))]),
        geometry=Geometry(kind="mesh", dimension=3, units="m", files=inputs,
                          bounding_box=tuple(float(v) for v in surface.points.min(axis=0))
                          + tuple(float(v) for v in surface.points.max(axis=0)),
                          description=f"the surface of {opened.name} run {run}"),
        physics=Physics(
            governing="time-averaged incompressible external aerodynamics, as the dataset measured it",
            symbols={"Cp": "pressure coefficient", "p": "pressure"},
            parameters=[Parameter(name=name, value=float(value), unit="1", source="geo_parameters_i.csv")
                        for name, value in case.parameters.items()],
            notes=f"{opened.name} ({opened.dataset.paper}), licence {opened.licence}. Reference quantities: "
                  f"U = {opened.dataset.velocity} m/s, rho = {opened.dataset.density}, A_ref = "
                  f"{opened.dataset.a_ref} m^2. Not re-derived here: the coefficients the dataset publishes "
                  f"({sorted(case.forces)}) are the CFD's own time averages."),
        source=f"{opened.id} run {run}", title=f"{opened.name} run {run}", license=opened.licence,
        kind="dataset", adapter="pinneapple_data.ashton.loader", adapter_version="1", inputs=inputs,
        citation=opened.attribution, url=opened.dataset.url,
        metadata={opened.id: {"run": int(run), "forces": case.forces, "forces_varref": case.varref,
                              "parameters": case.parameters, "licence": opened.licence,
                              "attribution": opened.attribution,
                              "quadrature": opened.dataset.quadrature,
                              "reference": {"velocity": opened.dataset.velocity,
                                            "density": opened.dataset.density,
                                            "a_ref": opened.dataset.a_ref, "l_ref": opened.dataset.l_ref}}})

"""Reading results into one structure, :class:`FieldData`, whatever the format.

Kinds:

* ``fv``: an OpenFOAM case (cell values, boundary-face values, face geometry, ``phi`` when written), read with the
  readers of :mod:`pinneapple_data.cae.foam`.
* ``grid``: arrays on a tensor-product grid. An ``.npz``/``.h5`` file with 1-D coordinate arrays (``x``, ``y``,
  ``z``, ``t``) and fields of shape ``(len(x), len(y), ...)`` (``indexing="ij"``), or a point file whose points
  happen to form such a grid.
* ``points``: a point cloud with point data (VTU/VTK through meshio, NPZ/HDF5/CSV tables through
  :mod:`pinneapple_data.cae.io`).
* ``qoi``: a CSV of quantities of interest, either long (``name,value[,unit]``) or one row with a column per
  quantity (``Cd [1]``).
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import _reuse

__all__ = ["FieldData", "load_result", "LoadError", "velocity_components", "scalar"]

AXES = ("x", "y", "z")


class LoadError(ValueError):
    pass


@dataclass
class FieldData:
    kind: str
    fields: dict[str, np.ndarray] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    files: dict[str, Path] = field(default_factory=dict)          # field -> file it came from (for annotations)
    source: Path | None = None
    # grid
    axes: dict[str, np.ndarray] = field(default_factory=dict)     # ordered: spatial axes then "t"
    # points
    points: np.ndarray | None = None
    # fv
    mesh: Any = None
    geom: dict[str, np.ndarray] = field(default_factory=dict)
    boundary: dict[str, np.ndarray] = field(default_factory=dict)  # field -> value on each boundary face
    patches: list[dict[str, Any]] = field(default_factory=list)
    phi: tuple[np.ndarray, np.ndarray] | None = None              # (internal-face flux, boundary-face flux)
    time: float | None = None
    previous: dict[str, Any] | None = None                        # {"time", "fields"} of the time before
    first: dict[str, Any] | None = None                           # {"time", "fields"} of the first time
    snapshots: list[tuple[float, dict[str, np.ndarray]]] = field(default_factory=list)
    # qoi
    qoi: dict[str, float] = field(default_factory=dict)

    @property
    def spatial_axes(self) -> list[str]:
        return [a for a in self.axes if a != "t"]

    @property
    def dim(self) -> int:
        if self.kind == "grid":
            return len(self.spatial_axes)
        if self.kind == "fv":
            return int(self.geom.get("dim", 3))
        if self.points is not None:
            span = np.ptp(self.points, axis=0)
            return int((span > 1e-12 * max(span.max(), 1e-300)).sum())
        return 0

    def coords(self) -> np.ndarray:
        """(n, d) coordinates of the values (cell centres, grid nodes or points)."""
        if self.kind == "fv":
            return self.geom["cell_centres"]
        if self.kind == "grid":
            mg = np.meshgrid(*[self.axes[a] for a in self.axes], indexing="ij")
            return np.stack([m.ravel() for m in mg], axis=1)
        return self.points

    def flat(self, name: str) -> np.ndarray:
        """Field values as (n,) or (n, k), in the order of :meth:`coords`."""
        v = np.asarray(self.fields[name], float)
        if self.kind == "grid":
            shape = tuple(len(self.axes[a]) for a in self.axes)
            if v.shape[:len(shape)] != shape:
                return v.ravel()
            n = int(np.prod(shape))
            return v.reshape(n) if v.ndim == len(shape) else v.reshape(n, -1)
        return v


def scalar(v: np.ndarray) -> np.ndarray:
    return v if v.ndim == 1 or v.shape[-1] == 1 else np.linalg.norm(v, axis=-1)


def velocity_components(fd: FieldData) -> list[np.ndarray] | None:
    """Velocity as a list of component arrays over the spatial dimensions: a vector field ``U`` (or ``u``) or the
    scalars ``u, v[, w]`` / ``Ux, Uy[, Uz]``."""
    f = fd.fields
    nd = fd.dim if fd.kind != "fv" else 3
    for name in ("U", "u", "velocity", "vel"):
        if name in f and f[name].ndim >= 1 and _is_vector(fd, f[name]):
            v = f[name]
            return [v[..., i] for i in range(min(v.shape[-1], nd))]
    for names in (("u", "v", "w"), ("Ux", "Uy", "Uz"), ("U_x", "U_y", "U_z"), ("vx", "vy", "vz")):
        if all(n in f for n in names[:max(nd, 2)]):
            return [f[n] for n in names[:nd] if n in f]
    return None


def velocity_name(fd: FieldData) -> str | None:
    for name in ("U", "u", "velocity", "vel"):
        if name in fd.fields and _is_vector(fd, fd.fields[name]):
            return name
    for names in (("u", "v", "w"), ("Ux", "Uy", "Uz"), ("U_x", "U_y", "U_z"), ("vx", "vy", "vz")):
        if all(n in fd.fields for n in names[:2]):
            return names[0]
    return None


def _is_vector(fd: FieldData, v: np.ndarray) -> bool:
    if fd.kind == "grid":
        return v.ndim == len(fd.axes) + 1 and v.shape[-1] in (2, 3)
    return v.ndim == 2 and v.shape[1] in (2, 3)


# ---------------------------------------------------------------------------------------------- dispatch
def load_result(path: Path, kind: str | None = None, time: str | None = None,
                fields: list[str] | None = None) -> FieldData:
    path = Path(path)
    if not path.exists():
        raise LoadError(f"{path} does not exist")
    if path.is_dir():
        if (path / "constant" / "polyMesh").is_dir():
            return load_openfoam(path, time=time, only=fields)
        raise LoadError(f"{path} is a folder but not an OpenFOAM case (no constant/polyMesh)")
    ext = path.suffix.lower()
    if kind == "qoi" or (ext in (".csv", ".tsv", ".txt") and kind != "points"):
        try:
            return load_qoi(path)
        except LoadError:
            if kind == "qoi":
                raise
    if ext in (".npz", ".h5", ".hdf5"):
        return load_arrays(path, kind=kind)
    if ext in (".vtu", ".vtk", ".vtp", ".vts", ".vtr", ".xdmf", ".msh"):
        return load_meshio(path)
    if ext in (".csv", ".tsv", ".txt", ".parquet"):
        return _points_from_mesh(_reuse.cae_io().read_points(path.name, path.read_bytes()), path)
    raise LoadError(f"unsupported result format {ext or path.name!r}")


# ---------------------------------------------------------------------------------------------- OpenFOAM
def _case_fs(case: Path) -> dict[str, bytes]:
    fs: dict[str, bytes] = {}
    for sub in [case / "constant" / "polyMesh", *[d for d in case.iterdir() if _is_time(d.name) and d.is_dir()]]:
        for f in sub.iterdir():
            if f.is_file() and not f.name.startswith(".") and f.stat().st_size < 2_000_000_000:
                fs[f.relative_to(case).as_posix()] = f.read_bytes()
    if _reuse.case_root().is_case(case):
        fs["system/controlDict"] = (case / "system" / "controlDict").read_bytes()
    return fs


def _is_time(name: str) -> bool:
    try:
        float(name)
        return True
    except ValueError:
        return False


def load_openfoam(case: Path, time: str | None = None, only: list[str] | None = None) -> FieldData:
    foam = _reuse.foam()
    fs = _case_fs(case)
    try:
        mesh = foam.read_polymesh(fs, "")
    except Exception as e:
        raise LoadError(f"cannot read the polyMesh of {case}: {e}") from e
    times = foam.time_dirs(fs, "")
    if not times:
        raise LoadError(f"{case} has no time directory with fields")
    t = _pick_time(times, time)
    info = foam.read_time(fs, mesh, "", t, only=only)
    fd = FieldData(kind="fv", source=case, mesh=mesh, patches=list(mesh.poly.patches), time=info["time"])
    fd.fields = dict(mesh.cell_data)
    fd.units = dict(mesh.units)
    fd.boundary = dict(mesh.boundary_values)
    fd.files = {n: _field_file(case, t, n) for n in fd.fields}
    fd.geom = fv_geometry(mesh)
    phi = case / t / "phi"
    if not phi.is_file() and (case / t / "phi.gz").is_file():
        phi = case / t / "phi.gz"
    if phi.is_file():
        try:
            fd.phi = read_surface_scalar(phi.read_bytes(), mesh.poly)
            fd.files["phi"] = phi
        except Exception:                                             # phi is a refinement; U is used without it
            fd.phi = None
    ti = times.index(t)
    if ti > 0:
        fd.previous = _other_time(fs, case, times[ti - 1], only)
    if t != times[0]:
        fd.first = _other_time(fs, case, times[0], only)
    return fd


def _other_time(fs, case, t, only):
    foam = _reuse.foam()
    m = foam.read_polymesh(fs, "")
    foam.read_time(fs, m, "", t, only=only)
    return {"time": float(t), "fields": dict(m.cell_data), "boundary": dict(m.boundary_values),
            "files": {n: _field_file(case, t, n) for n in m.cell_data}}


def _pick_time(times: list[str], time: str | None) -> str:
    if time is None or time == "latest":
        return times[-1]
    if time == "first":
        return times[0]
    for t in times:
        if float(t) == float(time):
            return t
    raise LoadError(f"time {time} not found; the case has {', '.join(times)}")


def _field_file(case: Path, t: str, name: str) -> Path:
    p = case / t / name
    return p if p.exists() else case / t / f"{name}.gz"


def fv_geometry(mesh) -> dict[str, np.ndarray]:
    """Face centres and area vectors, cell centres and volumes (reusing :mod:`pinneapple_data.cae.geometry`), plus
    the number of solved directions (empty patches remove one)."""
    G = _reuse.cae_geometry()
    fm = G.face_mesh(mesh)
    fg = G.face_geometry(fm)
    cg = G.cell_geometry(fm, fg)
    p = mesh.poly
    empty = np.zeros(p.n_faces - p.n_internal, bool)
    for pt in p.patches:
        if pt["type"] == "empty":
            a = pt["startFace"] - p.n_internal
            empty[a:a + pt["nFaces"]] = True
    dirs = G.solution_dirs(fm, fg)
    return {"face_centres": fg["centres"], "areas": fg["areas"], "cell_centres": cg["centres"],
            "volumes": cg["volumes"], "owner": np.asarray(p.owner), "neighbour": np.asarray(p.neighbour),
            "n_internal": p.n_internal, "n_cells": p.n_cells, "empty": empty,
            "dim": int(np.sum(dirs)) if dirs is not None else 3,
            "dirs": np.asarray(dirs if dirs is not None else [True, True, True], bool)}


def read_surface_scalar(b: bytes, poly) -> tuple[np.ndarray, np.ndarray]:
    """``phi`` (surfaceScalarField): values on the internal faces and on every boundary face (0 on empty)."""
    foam = _reuse.foam()
    t = foam._text(b)
    h = foam.header(t)
    if "surfaceScalarField" not in h.get("class", ""):
        raise ValueError("not a surfaceScalarField")
    _, sca = foam._arch(h)
    binary = h.get("format", "ascii") == "binary"
    ni, nb = poly.n_internal, poly.n_faces - poly.n_internal
    m = re.search(r"internalField\s+(uniform|nonuniform)\s*", t)
    if m.group(1) == "uniform":
        internal = np.full(ni, float(t[m.end():t.index(";", m.end())]))
    else:
        lm = re.compile(r"List<\w+>\s*").match(t, m.end())
        internal, _ = foam._list_at(t, lm.end() if lm else m.end(), binary, sca, 1)
        internal = np.asarray(internal, float)
    boundary = np.zeros(nb)
    bf = re.search(r"boundaryField\s*\{", t)
    for pt in poly.patches:
        a, n = pt["startFace"] - ni, pt["nFaces"]
        if n == 0 or pt["type"] == "empty":
            continue
        mm = re.compile(r'(?:^|\s)"?' + re.escape(pt["name"]) + r'"?\s*\{', re.M).search(t, bf.end())
        if not mm:
            continue
        blk = t[mm.end():t.index("}", mm.end())]
        vm = re.search(r"\bvalue\s+(uniform|nonuniform)\s*", blk)
        if not vm:
            continue
        if vm.group(1) == "uniform":
            boundary[a:a + n] = float(blk[vm.end():blk.index(";", vm.end())])
        else:
            lm = re.compile(r"List<\w+>\s*").match(blk, vm.end())
            arr, _ = foam._list_at(blk, lm.end() if lm else vm.end(), binary, sca, 1)
            if len(arr) == n:
                boundary[a:a + n] = np.asarray(arr, float)
    if len(internal) != ni:
        raise ValueError(f"phi has {len(internal)} internal values for {ni} faces")
    return internal, boundary


# ---------------------------------------------------------------------------------------------- arrays
def _h5_arrays(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    try:
        import h5py
    except ImportError as e:
        raise LoadError("reading HDF5 needs h5py (pip install h5py)") from e
    arrs, units = {}, {}
    with h5py.File(path, "r") as h:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.dtype.kind in "fiu":
                key = name.split("/")[-1]
                arrs[key] = obj[()]
                u = obj.attrs.get("unit", obj.attrs.get("units"))
                if u is not None:
                    units[key] = u.decode() if isinstance(u, bytes) else str(u)
        h.visititems(visit)
    return arrs, units


def load_arrays(path: Path, kind: str | None = None) -> FieldData:
    """NPZ/HDF5: a grid when 1-D coordinate arrays span the field shapes, else a point cloud. An optional array
    ``units`` of strings ``"name=unit"`` gives the units in an NPZ."""
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as z:
            arrs = {k: z[k] for k in z.files}
        units = {}
        if "units" in arrs and arrs["units"].dtype.kind in "US":
            for s in np.atleast_1d(arrs.pop("units")):
                if "=" in str(s):
                    k, v = str(s).split("=", 1)
                    units[k.strip()] = v.strip()
    else:
        arrs, units = _h5_arrays(path)
    axes = {a: np.asarray(arrs[a], float) for a in (*AXES, "t") if a in arrs and np.asarray(arrs[a]).ndim == 1}
    shape = tuple(len(v) for v in axes.values())
    grid_fields = {k: np.asarray(v, float) for k, v in arrs.items()
                   if k not in axes and np.asarray(v).dtype.kind in "fiu" and len(shape) >= 1
                   and np.asarray(v).shape[:len(shape)] == shape and np.asarray(v).ndim in (len(shape), len(shape) + 1)}
    if kind != "points" and axes and grid_fields:
        fd = FieldData(kind="grid", source=path, axes=axes, fields=grid_fields)
        fd.units = {k: v for k, v in units.items() if k in grid_fields or k in axes}
        fd.files = {k: path for k in grid_fields}
        return fd
    io_ = _reuse.cae_io()
    try:
        m = io_._points_from_arrays(arrs, path.name)
    except Exception as e:
        raise LoadError(f"{path.name}: {e}") from e
    m.units.update({k: v for k, v in units.items() if k in m.point_data})
    return _points_from_mesh(m, path)


def load_meshio(path: Path) -> FieldData:
    import contextlib
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            m = _reuse.cae_io().read_meshio(path.name, path.read_bytes())
    except (Exception, SystemExit) as e:                             # meshio calls sys.exit(1) on a bad file
        detail = buf.getvalue().strip().splitlines()
        msg = re.sub(r"\S*/(upload|" + re.escape(path.stem) + r")(\.\w+)", path.name, detail[-1] if detail else repr(e))
        raise LoadError(msg.replace('"', "")) from None
    if m.point_data:
        return _points_from_mesh(m, path)
    if m.cell_data:                                                   # cell data only: values at cell centroids
        cent = np.array([m.points[c].mean(0) for b in m.blocks for c in b.conn])
        fd = FieldData(kind="points", source=path, points=cent, fields=dict(m.cell_data), units=dict(m.units))
        fd.files = {k: path for k in fd.fields}
        return _maybe_grid(fd)
    raise LoadError(f"{path.name} has no point or cell data")


def _points_from_mesh(m, path: Path) -> FieldData:
    fd = FieldData(kind="points", source=path, points=np.asarray(m.points, float), fields=dict(m.point_data),
                   units=dict(m.units))
    fd.files = {k: path for k in fd.fields}
    return _maybe_grid(fd)


def _maybe_grid(fd: FieldData) -> FieldData:
    """Points on a full tensor-product lattice become a grid (finite differences instead of least squares)."""
    P = fd.points
    n = len(P)
    keep = [i for i in range(P.shape[1]) if np.ptp(P[:, i]) > 0]
    if not keep or n < 9:
        return fd
    uniq = [np.unique(np.round(P[:, i], 12)) for i in keep]
    if int(np.prod([len(u) for u in uniq])) != n or any(len(u) < 3 for u in uniq):
        return fd
    idx = [np.searchsorted(u, np.round(P[:, i], 12)) for u, i in zip(uniq, keep, strict=False)]
    order = np.lexsort(tuple(reversed(idx)))
    shape = tuple(len(u) for u in uniq)
    names = [AXES[i] for i in keep]
    g = FieldData(kind="grid", source=fd.source, axes=dict(zip(names, uniq, strict=False)), units=dict(fd.units), files=fd.files)
    for k, v in fd.fields.items():
        v = np.asarray(v, float)[order]
        g.fields[k] = v.reshape(*shape, *v.shape[1:])
    return g


# ---------------------------------------------------------------------------------------------- QoI
def load_qoi(path: Path) -> FieldData:
    text = path.read_text(encoding="utf-8")
    rows = [r for r in csv.reader(io.StringIO("\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))))
            if any(c.strip() for c in r)]
    if not rows:
        raise LoadError(f"{path.name} is empty")
    head = [c.strip() for c in rows[0]]
    fd = FieldData(kind="qoi", source=path)
    low = [h.lower() for h in head]
    if len(head) >= 2 and low[0] in ("name", "qoi", "quantity") and low[1] == "value":
        for r in rows[1:]:
            try:
                fd.qoi[r[0].strip()] = float(r[1])
            except (ValueError, IndexError):
                raise LoadError(f"{path.name}: bad row {r}") from None
            if len(r) > 2 and r[2].strip() and low[2:3] == ["unit"]:
                fd.units[r[0].strip()] = r[2].strip()
        return fd
    if len(rows) != 2:
        raise LoadError(f"{path.name}: a QoI CSV is `name,value[,unit]` rows or a header and one row of values")
    split = _reuse.units().split_header
    for h, v in zip(head, rows[1], strict=False):
        try:
            name, unit = split(h)
        except Exception:
            name, unit = h, None
        try:
            fd.qoi[name] = float(v)
        except ValueError:
            raise LoadError(f"{path.name}: {h} = {v!r} is not a number") from None
        if unit:
            fd.units[name] = unit
    return fd

"""Snapshot matrices for reduced-order models: many results of one model (time steps or parameter points) on one mesh,
stacked into ``X`` with one row per snapshot.

``load_snapshots(files)`` reads an upload (a list of ``(name, bytes)``, zip archives expanded):

* **OpenFOAM case** (zip or files): every numeric time directory with the selected volume fields, one snapshot per
  time (``times`` from the directory names). Directories missing a field are skipped and reported.
* **A series of mesh results** (several ``.vtu`` / ``.vtk`` / ``.frd`` / ``.msh`` / other meshio files): one snapshot
  per file, ordered by the number in the file name (``flow_0010.vtu``), which also gives ``times`` unless the
  results carry a time themselves. All files must share the mesh (same number of points or cells).
* **Arrays** (``.npz`` / ``.npy`` / ``.h5``): ``snapshots`` (n, ...) or one array per field with the snapshots
  along the first axis, plus optional ``times`` (n,), ``params`` (n, p) and ``param_names``, ``points`` (m, 2|3),
  ``hf_seconds`` (wall time of the full-order model per snapshot) and ``dt``. Optional mesh for 3-D views:
  ``cells_<vtk type>`` connectivity (e.g. ``cells_hexahedron`` (k, 8), point indices), ``snapshot_points`` (n, m, 3)
  when every snapshot has its own node coordinates (designs of different size), and ``node_sets`` in the ``meta``
  JSON (name -> {"nodes", "kind": "bc" | "load" | "region", "detail"}).
* **A table** (CSV / Parquet): one row per snapshot; ``t`` or ``time`` is the time column, ``param_*`` / ``p:*``
  columns (or ``param_columns``) are parameters, every other numeric column is a value of the field.

The result keeps the layout (which columns of ``X`` belong to which field and location), so a reconstruction can be
written back as fields on the mesh. Mesh series and arrays with connectivity also keep the cells
(:meth:`SnapshotSet.mesh`), so a snapshot can be shown on its body.
"""
from __future__ import annotations

import io as _io
import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["SnapshotSet", "load_snapshots", "snapshots_from_arrays", "FormatError"]

from .cae.io import MESHIO_EXT, FormatError

_RESERVED = {"times", "time", "t", "params", "param_names", "points", "coords", "coordinates", "hf_seconds", "dt",
             "parameters", "names", "field_names", "meta", "snapshot_points"}
_SERIES_EXT = MESHIO_EXT + (".frd",)


@dataclass
class SnapshotSet:
    """``X`` (n_snapshots, n_values) plus what each row and column is."""
    X: np.ndarray
    fields: list[dict[str, Any]]                    # {name, where, components, offset, size}
    times: np.ndarray | None = None              # (n,) for time series
    params: np.ndarray | None = None             # (n, p) for parametric results
    param_names: list[str] = field(default_factory=list)
    locations: dict[str, np.ndarray] = field(default_factory=dict)   # "point" / "cell" -> (m, 3)
    source: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)              # hf_seconds, dt, units, skipped, node_sets
    cells: list[tuple[str, np.ndarray]] = field(default_factory=list)  # (VTK type, connectivity) of the point mesh
    snapshot_points: np.ndarray | None = None    # (n, m, 3): node coordinates per snapshot, if they differ

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_values(self) -> int:
        return int(self.X.shape[1])

    @property
    def kind(self) -> str:
        """``parametric`` when there are parameters, otherwise ``time`` (a sequence)."""
        return "parametric" if self.params is not None else "time"

    def field_values(self, name: str, X: np.ndarray | None = None) -> np.ndarray:
        """Values of one field for every row of ``X`` (default: the snapshots): (n, m) or (n, m, components)."""
        f = next((f for f in self.fields if f["name"] == name), None)
        if f is None:
            raise KeyError(f"no field {name!r}; fields: {[f['name'] for f in self.fields]}")
        X = self.X if X is None else np.atleast_2d(X)
        v = X[:, f["offset"]:f["offset"] + f["size"]]
        return v.reshape(len(X), -1, f["components"]) if f["components"] > 1 else v

    def select(self, idx) -> SnapshotSet:
        idx = np.asarray(idx)
        return SnapshotSet(self.X[idx], self.fields, None if self.times is None else self.times[idx],
                           None if self.params is None else self.params[idx], self.param_names, self.locations,
                           self.source, self.meta, self.cells,
                           None if self.snapshot_points is None else self.snapshot_points[idx])

    def mesh(self, i: int | None = None):
        """The point mesh (a ``pinneapple_data.cae.Mesh``) of snapshot ``i`` (its own node coordinates when the set
        has them), with the node sets; None without connectivity."""
        from .cae.model import CellBlock, Mesh
        pts = self.locations.get("point")
        if not self.cells or pts is None:
            return None
        if self.snapshot_points is not None and i is not None:
            pts = self.snapshot_points[i]
        m = Mesh(points=np.asarray(pts, float), blocks=[CellBlock(t, np.asarray(c, np.int64)) for t, c in self.cells])
        m.node_sets = {k: np.asarray(v["nodes"], np.int64) for k, v in (self.meta.get("node_sets") or {}).items()}
        return m

    def summary(self) -> dict[str, Any]:
        return {"n_snapshots": self.n, "n_values": self.n_values, "kind": self.kind,
                "fields": [{k: f[k] for k in ("name", "where", "components", "size")} for f in self.fields],
                "times": None if self.times is None else [float(self.times[0]), float(self.times[-1])],
                "param_names": self.param_names, "locations": {k: int(len(v)) for k, v in self.locations.items()},
                "source": self.source, "meta": {k: v for k, v in self.meta.items() if k not in ("skipped", "node_sets")},
                "skipped": self.meta.get("skipped", []),
                "mesh": {"cells": int(sum(len(c) for _, c in self.cells)), "types": [t for t, _ in self.cells],
                         "own_points": self.snapshot_points is not None,
                         "node_sets": sorted(self.meta.get("node_sets") or {})} if self.cells else None}

    def to_npz_bytes(self) -> bytes:
        """The set as an ``.npz`` that ``load_snapshots`` reads back (the open exchange format of the ROM tools)."""
        out: dict[str, Any] = {}
        for f in self.fields:
            out[f["name"]] = self.field_values(f["name"])
        if self.times is not None:
            out["times"] = self.times
        if self.params is not None:
            out["params"] = self.params
            out["param_names"] = np.array(self.param_names)
        for where, loc in self.locations.items():
            out["points" if where == "point" else f"{where}_locations"] = loc
        for t, c in self.cells:
            out[f"cells_{t}"] = np.asarray(c, np.int64)
        if self.snapshot_points is not None:
            out["snapshot_points"] = self.snapshot_points
        out["meta"] = np.array(json.dumps({"fields": [{k: f[k] for k in ("name", "where", "components")}
                                                      for f in self.fields],
                                          **{k: v for k, v in self.meta.items()
                                             if k in ("hf_seconds", "dt", "units", "node_sets", "full_order")}}))
        buf = _io.BytesIO()
        np.savez_compressed(buf, **out)
        return buf.getvalue()


def _layout(arrays: dict[str, np.ndarray], where: dict[str, str]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    cols, fields, off = [], [], 0
    for name, a in arrays.items():
        a = np.asarray(a, float)
        comp = int(a.shape[2]) if a.ndim == 3 else 1
        flat = a.reshape(a.shape[0], -1)
        fields.append({"name": name, "where": where.get(name, "point"), "components": comp, "offset": off,
                       "size": int(flat.shape[1])})
        off += flat.shape[1]
        cols.append(flat)
    return np.concatenate(cols, axis=1), fields


def snapshots_from_arrays(arrays: dict[str, np.ndarray], *, times=None, params=None,
                          param_names: Sequence[str] | None = None, locations: dict[str, np.ndarray] | None = None,
                          where: dict[str, str] | None = None, source: dict[str, Any] | None = None,
                          meta: dict[str, Any] | None = None, cells: Sequence[tuple[str, Any]] | None = None,
                          snapshot_points=None) -> SnapshotSet:
    """Build a set from field arrays with the snapshots on the first axis: (n, m) scalars or (n, m, c) vectors (more
    axes are flattened into m). ``cells``: (VTK type, connectivity) of the point mesh; ``snapshot_points``: (n, m, 3)
    node coordinates per snapshot."""
    if not arrays:
        raise FormatError("no field arrays")
    norm = {}
    for k, a in arrays.items():
        a = np.asarray(a, float)
        if a.ndim == 1:
            a = a[:, None]
        elif a.ndim > 3 or (a.ndim == 3 and a.shape[2] > 9):
            a = a.reshape(a.shape[0], -1)
        norm[k] = a
    n = {a.shape[0] for a in norm.values()}
    if len(n) != 1:
        raise FormatError(f"field arrays disagree on the number of snapshots: {sorted(n)}")
    X, fields = _layout(norm, where or {})
    if not np.isfinite(X).all():
        raise FormatError(f"{int((~np.isfinite(X)).sum())} values are NaN or infinite")
    t = None if times is None else np.asarray(times, float).ravel()
    p = None if params is None else np.asarray(params, float).reshape(X.shape[0], -1)
    for nm, v in (("times", t), ("params", p)):
        if v is not None and len(v) != X.shape[0]:
            raise FormatError(f"{len(v)} {nm} for {X.shape[0]} snapshots")
    names = list(param_names) if param_names is not None else ([f"p{i}" for i in range(p.shape[1])] if p is not None else [])
    sp = None if snapshot_points is None else np.asarray(snapshot_points, float).reshape(X.shape[0], -1, 3)
    cl = [(str(k), np.asarray(c, np.int64)) for k, c in (cells or [])]
    if cl:
        npts = len((locations or {}).get("point", ())) if sp is None else sp.shape[1]
        top = max(int(c.max()) for _, c in cl) if any(c.size for _, c in cl) else -1
        if top >= npts:
            raise FormatError(f"cells refer to point {top}, the mesh has {npts} points")
    if t is not None and p is None:
        order = np.argsort(t, kind="stable")
        X, t = X[order], t[order]
        sp = None if sp is None else sp[order]
    return SnapshotSet(X, fields, t, p, names, dict(locations or {}), dict(source or {}), dict(meta or {}), cl, sp)


def _number(name: str) -> float | None:
    m = re.findall(r"(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", os.path.splitext(os.path.basename(name))[0])
    return float(m[-1]) if m else None


def _arrays_file(name: str, b: bytes) -> dict[str, np.ndarray]:
    low = name.lower()
    if low.endswith(".npy"):
        return {"snapshots": np.load(_io.BytesIO(b), allow_pickle=False)}
    if low.endswith(".npz"):
        z = np.load(_io.BytesIO(b), allow_pickle=False)
        return {k: z[k] for k in z.files}
    import h5py
    out: dict[str, np.ndarray] = {}
    with h5py.File(_io.BytesIO(b), "r") as h:
        def visit(path, obj):
            if isinstance(obj, h5py.Dataset) and obj.size < 5e8:
                v = obj[()]
                out[path.split("/")[-1]] = np.array([x.decode() if isinstance(x, bytes) else x for x in v]) \
                    if getattr(v, "dtype", None) is not None and v.dtype.kind in "SO" else v
        h.visititems(visit)
    return out


def _from_arrays(arrs: dict[str, np.ndarray], name: str, fields: Sequence[str] | None) -> SnapshotSet:
    meta: dict[str, Any] = {}
    layout: dict[str, Any] = {}
    if "meta" in arrs:
        try:
            layout = json.loads(str(np.asarray(arrs["meta"]).item()))
        except (ValueError, TypeError):
            layout = {}
    for k in ("hf_seconds", "dt"):
        if k in arrs:
            meta[k] = float(np.asarray(arrs[k]).ravel()[0])
        elif k in layout:
            meta[k] = layout[k]
    for k in ("units", "node_sets", "full_order"):
        if k in layout:
            meta[k] = layout[k]
    times = next((arrs[k] for k in ("times", "time", "t") if k in arrs), None)
    params = arrs.get("params", arrs.get("parameters"))
    pn = arrs.get("param_names", arrs.get("names"))
    pn = [str(x) for x in np.asarray(pn).ravel()] if pn is not None else None
    if params is not None and np.asarray(params).ndim == 1:
        params = np.asarray(params)[:, None]
    n = len(times) if times is not None else (len(params) if params is not None else None)
    cand = {k: np.asarray(v) for k, v in arrs.items() if k not in _RESERVED and not k.endswith("_locations")
            and not k.startswith("cells_")
            and np.issubdtype(np.asarray(v).dtype, np.number) and np.asarray(v).ndim >= 2}
    if fields:
        missing = [f for f in fields if f not in cand]
        if missing:
            raise FormatError(f"fields {missing} not in {name}; arrays: {sorted(cand)}")
        cand = {f: cand[f] for f in fields}
    if n is not None:
        cand = {k: v for k, v in cand.items() if v.shape[0] == n}
    elif cand:
        n0 = max({v.shape[0] for v in cand.values()}, key=lambda m: sum(v.shape[0] == m for v in cand.values()))
        cand = {k: v for k, v in cand.items() if v.shape[0] == n0}
    if not cand:
        raise FormatError(f"{name}: no snapshot arrays (n_snapshots x values); name them 'snapshots' or after the "
                          f"fields, with optional 'times' or 'params'")
    where = {f["name"]: f["where"] for f in layout.get("fields", [])}
    locs = {}
    for k in ("points", "coords", "coordinates"):
        if k in arrs and np.asarray(arrs[k]).ndim == 2:
            locs["point"] = _xyz(arrs[k])
            break
    for k, v in arrs.items():
        if k.endswith("_locations"):
            locs[k[: -len("_locations")]] = _xyz(v)
    cells = [(k[len("cells_"):], arrs[k]) for k in sorted(arrs) if k.startswith("cells_")]
    return snapshots_from_arrays(cand, times=times, params=params, param_names=pn, locations=locs, where=where,
                                 source={"format": "arrays", "file": name}, meta=meta, cells=cells if "point" in locs else None,
                                 snapshot_points=arrs.get("snapshot_points"))


def _xyz(a) -> np.ndarray:
    a = np.asarray(a, float)
    return np.column_stack([a, np.zeros(len(a))]) if a.shape[1] == 2 else a[:, :3]


def _from_table(name: str, b: bytes, param_columns: Sequence[str] | None) -> SnapshotSet:
    import pandas as pd
    df = pd.read_parquet(_io.BytesIO(b)) if name.lower().endswith(".parquet") else \
        pd.read_csv(_io.BytesIO(b), sep=None, engine="python", comment="#")
    cols = [str(c) for c in df.columns]
    df.columns = cols
    tcol = next((c for c in cols if c.lower() in ("t", "time")), None)
    pcols = list(param_columns) if param_columns else [c for c in cols if c.lower().startswith(("param_", "p:"))]
    missing = [c for c in pcols if c not in cols]
    if missing:
        raise FormatError(f"parameter columns {missing} not in the table")
    vcols = [c for c in cols if c != tcol and c not in pcols and pd.api.types.is_numeric_dtype(df[c])]
    if not vcols:
        raise FormatError("the table has no value columns besides time / parameters")
    params = df[pcols].to_numpy(float) if pcols else None
    names = [re.sub(r"^(param_|p:)", "", c) for c in pcols]
    times = df[tcol].to_numpy(float) if tcol and params is None else None
    return snapshots_from_arrays({"values": df[vcols].to_numpy(float)}, times=times, params=params, param_names=names,
                                 source={"format": "table", "file": name, "columns": vcols[:50]},
                                 meta={"value_columns": vcols})


def _mesh_fields(m, fields: Sequence[str] | None):
    from .cae.compare import cell_centres_volumes
    data = {}
    for where, d in (("point", m.point_data), ("cell", m.cell_data)):
        for k, v in d.items():
            v = np.asarray(v, float)
            if fields and k not in fields:
                continue
            if v.ndim > 2 or not np.isfinite(v).all():
                continue
            data[k] = (where, v)
    locs = {}
    if any(w == "point" for w, _ in data.values()):
        locs["point"] = np.asarray(m.points, float)
    if any(w == "cell" for w, _ in data.values()):
        locs["cell"] = np.asarray(cell_centres_volumes(m)[0], float)
    return data, locs


def _stack(series: list[tuple[float | None, dict[str, tuple[str, np.ndarray]]]], fields, locs, source, meta,
           cells=None, points=None):
    common = set.intersection(*[set(d) for _, d in series]) if series else set()
    if fields:
        common &= set(fields)
    if not common:
        raise FormatError("no field is present in every snapshot")
    names = [k for k in series[0][1] if k in common]
    shapes = {k: series[0][1][k][1].shape for k in names}
    keep, kept_at, skipped = [], [], list(meta.get("skipped", []))
    for i, (t, d) in enumerate(series):
        bad = [k for k in names if d[k][1].shape != shapes[k]]
        if bad:
            skipped.append({"snapshot": t, "reason": f"fields {bad} have another size (different mesh?)"})
            continue
        keep.append((t, d))
        kept_at.append(i)
    if len(keep) < 2:
        raise FormatError("fewer than two snapshots share the mesh and the fields")
    arrays = {k: np.stack([d[k][1] for _, d in keep]) for k in names}
    where = {k: keep[0][1][k][0] for k in names}
    times = [t for t, _ in keep]
    meta = dict(meta, skipped=skipped)
    sp = None
    if points is not None and cells:
        kept = [points[i] for i in kept_at]
        if len({p.shape for p in kept}) == 1 and any(not np.array_equal(p, kept[0]) for p in kept):
            sp = np.stack(kept)
    return snapshots_from_arrays(arrays, times=None if any(t is None for t in times) else times, where=where,
                                 locations=locs, source=source, meta=meta, cells=cells, snapshot_points=sp)


def load_snapshots(files: Sequence[tuple[str, bytes]], *, fields: Sequence[str] | None = None,
                   param_columns: Sequence[str] | None = None) -> SnapshotSet:
    """Read an upload into a :class:`SnapshotSet` (see the module docstring for the formats)."""
    from .cae import foam
    from .cae.io import kind_of, read_any
    from .cae.upload import expand

    files = list(files)
    npz = [(n, b) for n, b in files if n.lower().endswith(".npz")]      # an .npz is a zip: never expand it
    fs = expand([f for f in files if f not in npz])
    fs.update({n.replace("\\", "/").split("/")[-1]: b for n, b in npz})
    if not fs:
        raise FormatError("nothing uploaded")
    try:
        kind, main = kind_of(fs)
    except FormatError:
        kind, main = None, None
    if kind == "openfoam":
        mesh = foam.read_polymesh(fs, main)
        series, skipped = [], []
        for t in foam.time_dirs(fs, main):
            mesh.cell_data.clear()
            info = foam.read_time(fs, mesh, main, t, only=list(fields) if fields else None)
            if not info["fields"]:
                skipped.append({"snapshot": float(t), "reason": "no volume field in this time directory"})
                continue
            series.append((float(t), {k: ("cell", np.asarray(mesh.cell_data[k], float)) for k in info["fields"]}))
        from .cae.compare import cell_centres_volumes
        data_locs = {"cell": np.asarray(cell_centres_volumes(mesh)[0], float)}
        # a field that exists only at some times (0/ holds just the initial conditions) is dropped from those times
        names = [k for k in series[-1][1]] if series else []
        series = [(t, d) for t, d in series if all(k in d for k in names)] or series
        return _stack(series, fields, data_locs, {"format": "openfoam", "root": main, "n_cells": mesh.n_cells},
                      {"skipped": skipped})
    names = sorted(fs)
    low = {n: n.lower() for n in names}
    arr = [n for n in names if low[n].endswith((".npz", ".npy", ".h5", ".hdf5"))]
    if len(arr) == 1 and len(names) == 1:
        return _from_arrays(_arrays_file(arr[0], fs[arr[0]]), arr[0], fields)
    tab = [n for n in names if low[n].endswith((".csv", ".parquet", ".txt"))]
    if len(tab) == 1 and len(names) == 1:
        return _from_table(tab[0], fs[tab[0]], param_columns)
    ser = [n for n in names if low[n].endswith(_SERIES_EXT)]
    if len(ser) >= 2:
        ser.sort(key=lambda n: (_number(n) is None, _number(n) or 0.0, n))
        series = []
        locs: dict[str, np.ndarray] = {}
        cells, points, sets = None, [], {}
        for n in ser:
            m = read_any({n: fs[n]}, fields=True)
            data, lc = _mesh_fields(m, fields)
            locs = locs or lc
            if cells is None and "point" in lc and m.blocks:
                top = max(b.dim for b in m.blocks)
                cells = [(b.type, b.conn) for b in m.blocks if b.dim == top]
                sets = {k: {"nodes": np.asarray(v).tolist(), "kind": "region"} for k, v in m.node_sets.items()}
            points.append(np.asarray(m.points, float))
            t = m.source.get("time")
            series.append((float(t) if t is not None else _number(n), data))
        return _stack(series, fields, locs, {"format": "series", "files": len(ser), "first": ser[0]},
                      {"node_sets": sets} if sets else {}, cells=cells, points=points)
    raise FormatError("upload one .npz/.npy/.h5 file, one CSV/Parquet table (a row per snapshot), an OpenFOAM case "
                      "with several time directories, or two or more result files (.vtu, .vtk, .frd, ...) of one mesh")

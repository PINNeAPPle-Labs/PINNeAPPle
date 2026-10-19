"""Surrogate training data: input parameters -> output quantities (scalars) or fields on a grid.

``SurrogateData`` holds the arrays a surrogate is trained on; ``read_table`` reads CSV, Parquet, JSON, HDF5 and NPZ
(from a path or bytes) and ``from_table`` picks the input and output columns. A field is a set of output columns
``<name>_<i>`` (or ``<name>[i]``) ordered by ``i``, reshaped to ``grid_shape``; NPZ/HDF5 can carry ``X``, ``Y`` and
``coords`` arrays directly.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["SurrogateData", "read_table", "from_table", "from_arrays", "field_columns", "inspect_table", "grid_coords",
           "from_snapshots", "snapshot_values", "lattice_order"]

_FMT = {".csv": "csv", ".txt": "csv", ".parquet": "parquet", ".pq": "parquet", ".json": "json", ".h5": "hdf5",
        ".hdf5": "hdf5", ".npz": "npz"}


@dataclass
class SurrogateData:
    """``X`` (n, d) inputs; ``Y`` (n, m) scalar outputs or (n, *grid_shape) fields.

    ``coords`` (n_points, n_dims) are the field's grid coordinates (fields only), ``coord_names`` their names;
    ``groups`` (n,) an optional group label per sample (a geometry or a case), used for group-held-out splits;
    ``meta`` free-form provenance (source, solver, cost per sample in ``solver_s_per_sample``)."""
    X: np.ndarray
    Y: np.ndarray
    input_names: list[str]
    output_names: list[str]
    coords: np.ndarray | None = None
    coord_names: list[str] = field(default_factory=list)
    groups: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=np.float64)
        self.Y = np.asarray(self.Y, dtype=np.float64)
        if self.X.ndim != 2:
            raise ValueError(f"X must be (n, d), got shape {self.X.shape}")
        if self.Y.ndim == 1:
            self.Y = self.Y[:, None]
        if len(self.X) != len(self.Y):
            raise ValueError(f"X has {len(self.X)} rows and Y {len(self.Y)}")
        if len(self.input_names) != self.X.shape[1]:
            raise ValueError(f"{len(self.input_names)} input names for {self.X.shape[1]} input columns")
        if self.groups is not None:
            self.groups = np.asarray(self.groups)
            if len(self.groups) != len(self.X):
                raise ValueError("groups needs one label per sample")
        if self.kind == "field":
            if self.coords is None:
                self.coords = _unit_grid(self.grid_shape)
            self.coords = np.asarray(self.coords, dtype=np.float64).reshape(-1, len(self.grid_shape))
            if len(self.coords) != int(np.prod(self.grid_shape)):
                raise ValueError(f"{len(self.coords)} coordinates for a grid of shape {self.grid_shape}")
            if not self.coord_names:
                self.coord_names = [f"s{i}" for i in range(len(self.grid_shape))]
        elif len(self.output_names) != self.Y.shape[1]:
            raise ValueError(f"{len(self.output_names)} output names for {self.Y.shape[1]} output columns")
        bad = ~np.isfinite(self.X).all(1) | ~np.isfinite(self.Y.reshape(len(self.Y), -1)).all(1)
        if bad.any():
            raise ValueError(f"{int(bad.sum())} samples have non-finite inputs or outputs; remove them first")

    @property
    def kind(self) -> str:
        """``"field"`` when an output is a field on a grid (``Y`` has more than two dimensions or one output name
        over many columns), else ``"scalar"``."""
        return "field" if self.Y.ndim > 2 or (len(self.output_names) == 1 and self.Y.shape[1] > 1) else "scalar"

    @property
    def grid_shape(self) -> tuple[int, ...]:
        return tuple(self.Y.shape[1:]) if self.kind == "field" else ()

    @property
    def n(self) -> int:
        return len(self.X)

    def subset(self, idx) -> SurrogateData:
        idx = np.asarray(idx)
        return SurrogateData(self.X[idx], self.Y[idx], list(self.input_names), list(self.output_names),
                             self.coords, list(self.coord_names),
                             None if self.groups is None else self.groups[idx], dict(self.meta))

    def sha256(self) -> str:
        """Content hash of the arrays and names: the dataset version a result refers to."""
        h = hashlib.sha256()
        for a in (self.X, self.Y):
            h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
        h.update(json.dumps([self.input_names, self.output_names, list(self.grid_shape)]).encode())
        return h.hexdigest()

    def describe(self) -> dict[str, Any]:
        out = {"n_samples": self.n, "kind": self.kind, "inputs": list(self.input_names),
               "outputs": list(self.output_names), "grid_shape": list(self.grid_shape),
               "coord_names": list(self.coord_names), "sha256": self.sha256(),
               "input_ranges": {k: [float(self.X[:, i].min()), float(self.X[:, i].max())]
                                for i, k in enumerate(self.input_names)},
               "n_groups": None if self.groups is None else int(len(np.unique(self.groups))), "meta": self.meta}
        return out


def grid_coords(shape: Sequence[int], ranges: Sequence[Sequence[float]] | None = None) -> np.ndarray:
    """Coordinates ``(prod(shape), len(shape))`` of a tensor-product grid in ``ij`` order (the order fields are
    flattened in), each axis spanning ``ranges[i]`` (default [0, 1])."""
    ranges = ranges or [(0.0, 1.0)] * len(shape)
    if len(ranges) != len(shape):
        raise ValueError(f"{len(ranges)} coordinate ranges for a {len(shape)}-D grid")
    axes = [np.linspace(float(a), float(b), int(n)) for n, (a, b) in zip(shape, ranges, strict=True)]
    return np.stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")], 1)


def inspect_table(df, roles: Mapping[str, str] | None = None) -> dict[str, Any]:
    """What a table offers a surrogate: numeric columns with ranges, fields (indexed column groups ``u_0 ..
    u_k``, at least 4), text columns (candidate group labels) and suggested inputs / outputs (from ``roles``,
    e.g. a DatasetSpec's column roles, else: fields and the last numeric column are outputs)."""
    cols = [str(c) for c in df.columns]
    fields: dict[str, list[str]] = {}
    for c in cols:
        m = _IDX.match(c)
        if m:
            fields.setdefault(m.group(1), [])
    fields = {k: field_columns(cols, k) for k in fields}
    fields = {k: v for k, v in fields.items() if len(v) >= 4}
    in_field = {c for v in fields.values() for c in v}
    numeric, text = [], []
    for c in cols:
        if c in in_field:
            continue
        v = df[c]
        if v.dtype.kind in "biuf":
            a = v.to_numpy(float)
            numeric.append({"name": c, "min": float(np.nanmin(a)) if np.isfinite(a).any() else None,
                            "max": float(np.nanmax(a)) if np.isfinite(a).any() else None,
                            "n_unique": int(v.nunique()), "n_missing": int(v.isna().sum())})
        else:
            text.append({"name": c, "n_unique": int(v.nunique())})
    roles = dict(roles or {})
    if roles:
        inputs = [c for c, r in roles.items() if r == "input" and c not in in_field]
        outputs = [c for c, r in roles.items() if r == "output" and c not in in_field]
        outputs += [k for k, v in fields.items() if any(roles.get(c) == "output" for c in v)]
        group = next((c for c, r in roles.items() if r == "id" and c in [t["name"] for t in text]), None)
    else:
        nums = [n["name"] for n in numeric if n["n_unique"] > 1]
        outputs = list(fields) or nums[-1:]
        inputs = [n for n in nums if n not in outputs]
        group = next((t["name"] for t in text if 1 < t["n_unique"] < len(df)), None)
    return {"n_rows": int(len(df)), "numeric": numeric, "text": text,
            "fields": {k: {"columns": len(v), "first": v[0], "last": v[-1]} for k, v in fields.items()},
            "suggested": {"inputs": inputs, "outputs": outputs, "group": group}}


def _unit_grid(shape: Sequence[int]) -> np.ndarray:
    axes = [np.linspace(0.0, 1.0, n) for n in shape]
    return np.stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")], 1)


def read_table(source: str | Path | bytes, name: str | None = None, *, fmt: str | None = None):
    """Read a table from a path or bytes into a ``pandas.DataFrame`` (CSV, Parquet, JSON records/columns, HDF5
    datasets, NPZ arrays). For HDF5 and NPZ, 1-D arrays become columns and 2-D arrays ``a`` become ``a_0..a_k``."""
    import pandas as pd
    if fmt is None:
        ext = Path(name or (source if isinstance(source, (str, Path)) else "")).suffix.lower()
        fmt = _FMT.get(ext)
        if fmt is None:
            raise ValueError(f"unknown table format {ext!r}; use one of {sorted(set(_FMT.values()))}")
    buf = io.BytesIO(source) if isinstance(source, bytes) else source
    if fmt == "csv":
        return pd.read_csv(buf, sep=None, engine="python")
    if fmt == "parquet":
        return pd.read_parquet(buf)
    if fmt == "json":
        raw = json.loads(source.decode() if isinstance(source, bytes) else Path(source).read_text())
        if isinstance(raw, Mapping) and "rows" in raw:
            raw = raw["rows"]
        return pd.DataFrame(raw)
    if fmt in ("npz", "hdf5"):
        return pd.DataFrame(_columns(_arrays(buf, fmt)))
    raise ValueError(f"unknown table format {fmt!r}")


def _arrays(buf, fmt: str) -> dict[str, np.ndarray]:
    if fmt == "npz":
        z = np.load(buf, allow_pickle=False)
        return {k: z[k] for k in z.files}
    import h5py
    out: dict[str, np.ndarray] = {}
    with h5py.File(buf, "r") as f:
        f.visititems(lambda k, v: out.__setitem__(k.replace("/", "."), v[()]) if isinstance(v, h5py.Dataset) else None)
    return out


def _columns(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    cols: dict[str, np.ndarray] = {}
    n = None
    for k, a in arrays.items():
        a = np.asarray(a)
        if a.ndim == 0 or a.dtype.kind not in "biufUSO":
            continue
        n = len(a) if n is None else n
        if len(a) != n:
            continue
        if a.ndim == 1:
            cols[k] = a
        else:
            flat = a.reshape(len(a), -1)
            for j in range(flat.shape[1]):
                cols[f"{k}_{j}"] = flat[:, j]
    return cols


_IDX = re.compile(r"^(.*?)(?:_(\d+)|\[(\d+)\])$")


def field_columns(columns: Sequence[str], name: str) -> list[str]:
    """The columns of field ``name`` (``name_0, name_1, ...`` or ``name[0], ...``) ordered by index."""
    hits = []
    for c in columns:
        m = _IDX.match(str(c))
        if m and m.group(1) == name:
            hits.append((int(m.group(2) or m.group(3)), c))
    return [c for _, c in sorted(hits)]


def from_table(df, inputs: Sequence[str], outputs: Sequence[str], *, group: str | None = None,
               grid_shape: Sequence[int] | None = None, coords: np.ndarray | None = None,
               coord_names: Sequence[str] = (), meta: Mapping[str, Any] | None = None) -> SurrogateData:
    """Pick ``inputs`` and ``outputs`` from a DataFrame. A single output name that is not a column but has indexed
    columns (``u_0, u_1, ...``) is a field; ``grid_shape`` reshapes it (default: 1-D). Rows with missing values in
    the chosen columns are dropped (their count goes to ``meta['dropped_rows']``)."""
    cols = list(df.columns)
    missing = [c for c in inputs if c not in cols]
    if missing:
        raise ValueError(f"input columns not found: {missing}")
    out_cols: list[str] = []
    is_field = False
    for o in outputs:
        if o in cols:
            out_cols.append(o)
        else:
            fc = field_columns(cols, o)
            if not fc:
                raise ValueError(f"output column {o!r} not found (nor indexed columns {o}_0, {o}_1, ...)")
            if len(outputs) > 1:
                raise ValueError("a field output must be the only output of a surrogate")
            out_cols, is_field = fc, True
    use = list(inputs) + out_cols + ([group] if group else [])
    sub = df[use].copy()
    for c in list(inputs) + out_cols:
        sub[c] = _numeric(sub[c], c)
    n0 = len(sub)
    sub = sub.dropna()
    m = dict(meta or {})
    if len(sub) < n0:
        m["dropped_rows"] = n0 - len(sub)
    X = sub[list(inputs)].to_numpy(float)
    Y = sub[out_cols].to_numpy(float)
    if is_field:
        shape = tuple(int(s) for s in (grid_shape or (Y.shape[1],)))
        if int(np.prod(shape)) != Y.shape[1]:
            raise ValueError(f"grid_shape {shape} does not match the {Y.shape[1]} field columns")
        Y = Y.reshape(len(Y), *shape)
    return SurrogateData(X, Y, list(inputs), list(outputs), coords=coords, coord_names=list(coord_names),
                         groups=sub[group].to_numpy() if group else None, meta=m)


def _numeric(s, name: str):
    import pandas as pd
    v = pd.to_numeric(s, errors="coerce")
    if v.isna().all() and not s.isna().all():
        raise ValueError(f"column {name!r} is not numeric")
    return v


def from_arrays(arrays: Mapping[str, np.ndarray], *, input_names: Sequence[str] | None = None,
                output_names: Sequence[str] | None = None, meta: Mapping[str, Any] | None = None) -> SurrogateData:
    """From ``{"X": (n, d), "Y": (n, m) or (n, *grid), "coords"?, "groups"?, "input_names"?, "output_names"?}``
    (the layout an NPZ or HDF5 export of a simulation campaign uses)."""
    X, Y = np.asarray(arrays["X"]), np.asarray(arrays["Y"])
    names = lambda k, d: [str(s) for s in np.asarray(arrays[k]).tolist()] if k in arrays else d  # noqa: E731
    inn = list(input_names or names("input_names", [f"x{i}" for i in range(X.shape[1])]))
    if Y.ndim > 2:
        default_out = ["y"]
    else:
        default_out = [f"y{i}" for i in range(Y.shape[1] if Y.ndim == 2 else 1)]
    outn = list(output_names or names("output_names", default_out))
    return SurrogateData(X, Y, inn, outn, coords=arrays.get("coords"),
                         coord_names=names("coord_names", []), groups=arrays.get("groups"), meta=dict(meta or {}))


def lattice_order(points) -> tuple[tuple[int, ...], np.ndarray] | None:
    """``(shape, order)`` when the points are a tensor-product lattice (``points[order]`` reshapes to ``shape`` with
    the first coordinate slowest), else None."""
    P = np.asarray(points, float)
    axes = [np.unique(np.round(P[:, k], 12)) for k in range(P.shape[1])]
    shape = tuple(len(a) for a in axes)
    if int(np.prod(shape)) != len(P):
        return None
    idx = np.stack([np.searchsorted(a, np.round(P[:, k], 12)) for k, a in enumerate(axes)], 1)
    flat = np.ravel_multi_index(idx.T, shape)
    if len(np.unique(flat)) != len(P):
        return None
    return shape, np.argsort(flat)


def from_snapshots(snaps, *, meta: Mapping[str, Any] | None = None) -> SurrogateData:
    """A parametric ``pinneapple_data.snapshots.SnapshotSet`` whose point fields live on a tensor-product lattice of
    nodes (a structured mesh, possibly scaled per snapshot) as one field dataset: ``Y`` (n, *lattice, channels), the
    channels being every component of every point field (``ux, uy, uz, von_mises``), ``coords`` the unit lattice plus
    the channel index. ``meta["channels"]`` and ``meta["lattice_order"]`` let :func:`snapshot_values` map a
    prediction back onto the snapshot layout."""
    if snaps.params is None:
        raise ValueError("a surrogate maps parameters to fields: these snapshots have no parameters")
    pts = snaps.locations.get("point")
    fields = [f for f in snaps.fields if f["where"] == "point"]
    if pts is None or not fields:
        raise ValueError("no point fields with node coordinates")
    lat = lattice_order(snaps.snapshot_points[0] if snaps.snapshot_points is not None else pts)
    if lat is None:
        raise ValueError("the nodes are not a tensor-product lattice (a structured mesh); give the fields as columns")
    shape, order = lat
    chans, cols = [], []
    for f in fields:
        v = snaps.field_values(f["name"])
        v = v[..., None] if v.ndim == 2 else v
        for c in range(v.shape[2]):
            chans.append(f["name"] if v.shape[2] == 1 else f"{f['name']}[{c}]")
            cols.append(v[:, order, c])
    Y = np.stack(cols, -1).reshape(snaps.n, *shape, len(chans))
    grids = np.meshgrid(*[np.linspace(0, 1, s) for s in (*shape, len(chans))], indexing="ij")
    coords = np.stack([g.ravel() for g in grids], 1)
    m = {"source": snaps.source.get("file", "snapshots"), **(snaps.meta or {}), **dict(meta or {}),
         "channels": chans, "lattice_shape": list(shape), "lattice_order": order.tolist()}
    m.pop("node_sets", None)
    return SurrogateData(snaps.params, Y, list(snaps.param_names), ["+".join(f["name"] for f in fields)],
                         coords=coords, coord_names=["i", "j", "k"][:len(shape)] + ["channel"], meta=m)


def snapshot_values(snaps, data: SurrogateData, Y) -> np.ndarray:
    """Fields predicted on the lattice of :func:`from_snapshots` (n, *lattice, channels) back to snapshot rows
    (n, n_values) in the layout of ``snaps`` (``snaps.field_values(name, rows)`` then reads them)."""
    Y = np.asarray(Y, float).reshape(-1, int(np.prod(data.meta["lattice_shape"])), len(data.meta["channels"]))
    order = np.asarray(data.meta["lattice_order"])
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    out = np.zeros((len(Y), snaps.n_values))
    c = 0
    for f in snaps.fields:
        if f["where"] != "point":
            continue
        k = f["components"]
        v = Y[:, inv, c:c + k]                                  # back to node order
        out[:, f["offset"]:f["offset"] + f["size"]] = v.reshape(len(Y), -1)
        c += k
    return out

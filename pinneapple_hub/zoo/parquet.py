"""Parquet copies of the zoo datasets, for the Hugging Face dataset viewer and ``datasets.load_dataset``.

The npz files stay the source of truth: their SHA-256 are in the manifest and ``pp.datasets.load`` reads them. The
Parquet files are derived from them, deterministically (same npz and same pyarrow version -> same bytes):

* each split becomes ``data/<config>/<split>-NNNNN.parquet``. Splits with the same columns share a config; the
  config holding ``train`` (else ``test``, else the first split) is ``default``, any other one is named after its
  first split (a grid, a convergence study, an independent reference with other columns);
* scalar columns stay plain columns; an array column becomes a fixed-size list column of the same shape (``List(
  List(float32, length=64), length=64)`` in ``datasets``; ``with_format("numpy")`` gives the arrays back);
* a sample larger than :data:`EXPLODE_ROW_BYTES` whose array columns share a leading axis (a time series of fields)
  is stored as one row per step of that axis, with ``sample`` (row of the npz split) and ``step`` columns; the
  per-sample columns are repeated on every step. This keeps every row small enough for the viewer;
* a shard holds at most :data:`MAX_SHARD_BYTES` of uncompressed data and a row group at most :data:`ROW_GROUP_BYTES`,
  so the viewer can page any split;
* each field's metadata keeps its numpy dtype and per-row shape, so :func:`read_parquet` returns the npz arrays bit
  for bit.

    from pinneapple_hub.zoo import parquet
    layout = parquet.write_dataset("darcy_2d_li2021", parquet.read_npz(paths), "out/")   # files + features
    parquet.read_parquet(layout["configs"][0]["splits"][0]["paths"])                     # {column: array}
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["MAX_SHARD_BYTES", "ROW_GROUP_BYTES", "EXPLODE_ROW_BYTES", "read_npz", "plan_split", "write_dataset",
           "read_parquet", "features_yaml", "parquet_files"]

MAX_SHARD_BYTES = 256 * 2**20          # uncompressed bytes per Parquet file (the files on the Hub stay < 300 MB)
ROW_GROUP_BYTES = 16 * 2**20           # uncompressed bytes per row group: what the viewer reads to show a page
EXPLODE_ROW_BYTES = 512 * 2**10        # a sample larger than this is stored as one row per step (if it has steps)
COMPRESSION = "zstd"
META_KEY = b"pinneapple"


def read_npz(paths: Iterable[str | Path]) -> dict[str, dict[str, np.ndarray]]:
    """``{split: {column: array}}`` from npz files whose arrays are named ``<split>__<column>``; a split spread over
    several files (shards ``<split>-NN.npz``, in the given order) is concatenated along the rows."""
    parts: dict[str, dict[str, list[np.ndarray]]] = {}
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            for key in z.files:
                s, col = key.split("__", 1)
                parts.setdefault(s, {}).setdefault(col, []).append(z[key])
    return {s: {c: v[0] if len(v) == 1 else np.concatenate(v) for c, v in cols.items()} for s, cols in parts.items()}


# ---------------------------------------------------------------------------------------------- arrow types
def _arrow_type(dtype: np.dtype, shape: Sequence[int]):
    import pyarrow as pa
    t = pa.string() if dtype.kind == "U" else pa.from_numpy_dtype(dtype)
    for d in reversed(shape):
        t = pa.list_(t, d)
    return t


def _arrow_array(a: np.ndarray):
    import pyarrow as pa
    if a.dtype.kind == "U":
        return pa.array(a.tolist(), pa.string())
    if a.dtype.kind not in "biuf":
        raise TypeError(f"cannot store numpy dtype {a.dtype} in Parquet (numbers, booleans and strings only)")
    out = pa.array(np.ascontiguousarray(a).reshape(-1))
    for d in reversed(a.shape[1:]):
        out = pa.FixedSizeListArray.from_arrays(out, d)
    return out


def _row_bytes(dtype: np.dtype, shape: Sequence[int]) -> int:
    return int(dtype.itemsize * math.prod(shape))


# ---------------------------------------------------------------------------------------------- planning
def _order(columns: Iterable[str], preferred: Sequence[str]) -> list[str]:
    cols = set(columns)
    return [c for c in preferred if c in cols] + sorted(cols - set(preferred))


def plan_split(arrays: Mapping[str, np.ndarray], *, explode_row_bytes: int = EXPLODE_ROW_BYTES,
               max_shard_bytes: int = MAX_SHARD_BYTES, row_group_bytes: int = ROW_GROUP_BYTES) -> dict[str, Any]:
    """How one split is laid out: ``samples`` (npz rows), ``steps`` (0, or the length of the exploded axis),
    ``exploded`` columns, ``rows`` (Parquet rows), ``row_bytes`` and the rows per shard and per row group."""
    shapes = {c: a.shape[1:] for c, a in arrays.items()}
    n = len(next(iter(arrays.values()))) if arrays else 0
    bad = {c: len(a) for c, a in arrays.items() if len(a) != n}
    if bad:
        raise ValueError(f"columns of one split must have the same number of rows ({n}); got {bad}")
    size = {c: _row_bytes(a.dtype, shapes[c]) for c, a in arrays.items()}
    steps, exploded = 0, []
    fields = [c for c in arrays if len(shapes[c]) >= 2]
    if sum(size.values()) > explode_row_bytes and fields:
        steps = shapes[max(fields, key=size.__getitem__)][0]
        exploded = [c for c in arrays if len(shapes[c]) >= 1 and shapes[c][0] == steps]
    row = sum(size[c] // steps if c in exploded else size[c] for c in arrays) + (12 if steps else 0)
    row = max(row, 1)
    per_shard = max(1, max_shard_bytes // row)
    return {"samples": n, "steps": steps, "exploded": exploded, "rows": n * steps if steps else n, "row_bytes": row,
            "rows_per_shard": per_shard, "rows_per_group": max(1, min(per_shard, row_group_bytes // row))}


def _signature(arrays: Mapping[str, np.ndarray], plan: Mapping[str, Any]) -> tuple:
    sig = [("sample", "int64", ()), ("step", "int32", ())] if plan["steps"] else []
    for c, a in arrays.items():
        shape = a.shape[2:] if c in plan["exploded"] else a.shape[1:]
        sig.append((c, str(_arrow_type(a.dtype, ())), tuple(shape)))
    return tuple(sig)


def _config_name(splits: Sequence[str], is_default: bool) -> str:
    return "default" if is_default else re.sub(r"[^\w-]", "_", splits[0])


# ---------------------------------------------------------------------------------------------- writing
def _table(arrays: Mapping[str, np.ndarray], plan: Mapping[str, Any], lo: int, hi: int, meta: dict[str, Any]):
    """Rows ``lo:hi`` of the Parquet layout of a split as a ``pyarrow.Table``."""
    import pyarrow as pa
    steps = plan["steps"]
    cols, fields = [], []
    if steps:
        k = np.arange(lo, hi, dtype=np.int64)
        sample, step = k // steps, (k % steps).astype(np.int32)
        cols += [pa.array(sample), pa.array(step)]
        fields += [pa.field("sample", pa.int64(), nullable=False,
                            metadata={"description": "row of the npz split this step belongs to"}),
                   pa.field("step", pa.int32(), nullable=False,
                            metadata={"description": f"index along the stored axis (0 ... {steps - 1})"})]
    for c, a in arrays.items():
        if not steps:
            part, shape = a[lo:hi], a.shape[1:]
        elif c in plan["exploded"]:
            part, shape = a.reshape(len(a) * steps, *a.shape[2:])[lo:hi], a.shape[2:]
        else:
            part, shape = a[sample], a.shape[1:]
        cols.append(_arrow_array(part))
        fields.append(pa.field(c, _arrow_type(a.dtype, shape), nullable=False,
                               metadata={"numpy_dtype": a.dtype.str, "shape": json.dumps(list(shape)),
                                         "exploded": "1" if c in plan["exploded"] else "0"}))
    return pa.Table.from_arrays(cols, schema=pa.schema(fields, metadata={META_KEY: json.dumps(meta, sort_keys=True)}))


def write_dataset(name: str, data: Mapping[str, Mapping[str, np.ndarray]], out_dir: str | Path, *,
                  split_order: Sequence[str] = (), column_order: Sequence[str] = (),
                  explode_row_bytes: int = EXPLODE_ROW_BYTES, max_shard_bytes: int = MAX_SHARD_BYTES,
                  row_group_bytes: int = ROW_GROUP_BYTES) -> dict[str, Any]:
    """Write ``{split: {column: array}}`` as Parquet under ``out_dir/data/<config>/`` and return the layout:
    ``{"configs": [{"name", "default", "columns", "features" (dataset_info YAML), "splits": [{"name", "samples",
    "steps", "exploded", "rows", "files" (relative to out_dir), "paths", "bytes"}]}]}``."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    out = Path(out_dir)
    order = _order(data, split_order)
    groups: dict[tuple, list[str]] = {}
    plans, arrays = {}, {}
    for s in order:
        arrays[s] = {c: data[s][c] for c in _order(data[s], column_order)}
        plans[s] = plan_split(arrays[s], explode_row_bytes=explode_row_bytes, max_shard_bytes=max_shard_bytes,
                              row_group_bytes=row_group_bytes)
        groups.setdefault(_signature(arrays[s], plans[s]), []).append(s)
    members = list(groups.values())
    main = next((g for key in ("train", "test") for g in members if key in g), members[0] if members else None)
    configs = []
    for splits in sorted(members, key=lambda g: g is not main):
        cfg = _config_name(splits, splits is main)
        entry: dict[str, Any] = {"name": cfg, "default": splits is main, "splits": []}
        for s in splits:
            p = plans[s]
            files, paths, nbytes = [], [], 0
            n_shards = max(1, math.ceil(p["rows"] / p["rows_per_shard"]))
            for i in range(n_shards):
                lo, hi = i * p["rows_per_shard"], min(p["rows"], (i + 1) * p["rows_per_shard"])
                meta = {"dataset": name, "config": cfg, "split": s, "samples": p["samples"], "steps": p["steps"],
                        "shard": i, "shards": n_shards}
                table = _table(arrays[s], p, lo, hi, meta)
                rel = f"data/{cfg}/{s}-{i:05d}.parquet"
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(table, out / rel, compression=COMPRESSION, row_group_size=p["rows_per_group"],
                               use_dictionary=[f.name for f in table.schema if pa.types.is_string(f.type)])
                files.append(rel)
                paths.append(str(out / rel))
                nbytes += table.nbytes
            entry["splits"].append({"name": s, "samples": p["samples"], "steps": p["steps"],
                                    "exploded": p["exploded"], "rows": p["rows"], "files": files, "paths": paths,
                                    "bytes": int(nbytes)})
        schema = pq.read_schema(paths[0])
        entry["columns"] = schema.names
        entry["types"] = {f.name: _type_str(f) for f in schema}
        entry["features"] = features_yaml(schema)
        configs.append(entry)
    return {"dataset": name, "configs": configs}


# ---------------------------------------------------------------------------------------------- reading
def _to_numpy(col, dtype: np.dtype, shape: Sequence[int]) -> np.ndarray:
    import pyarrow as pa
    arr = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    n = len(arr)
    if dtype.kind == "U":
        return np.array(arr.to_pylist(), dtype=dtype).reshape(n, *shape)
    for _ in shape:
        arr = arr.flatten()
    return arr.to_numpy(zero_copy_only=False).astype(dtype, copy=False).reshape(n, *shape)


def read_parquet(paths: Sequence[str | Path]) -> dict[str, np.ndarray]:
    """The npz arrays of one split back from its Parquet shards (in order): the inverse of :func:`write_dataset`."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    if not paths:
        return {}
    tables = [pq.read_table(p) for p in paths]
    table = pa.concat_tables(tables)
    meta = json.loads(tables[0].schema.metadata[META_KEY])
    steps, n = int(meta["steps"]), int(meta["samples"])
    if steps:
        sample = table.column("sample").to_numpy()
        step = table.column("step").to_numpy()
        expected = np.arange(n * steps)
        if len(sample) != n * steps or not (np.array_equal(sample, expected // steps)
                                            and np.array_equal(step, expected % steps)):
            raise ValueError(f"{meta['dataset']}/{meta['split']}: rows are not the complete (sample, step) grid")
    out = {}
    for f in table.schema:
        if steps and f.name in ("sample", "step"):
            continue
        md = f.metadata or {}
        dtype = np.dtype(md[b"numpy_dtype"].decode())
        shape = json.loads(md[b"shape"])
        a = _to_numpy(table.column(f.name), dtype, shape)
        if steps:
            a = a.reshape(n, steps, *shape) if md.get(b"exploded") == b"1" else a[::steps]
        out[f.name] = a
    return out


def _type_str(field) -> str:
    md = field.metadata or {}
    if b"numpy_dtype" not in md:
        return str(field.type)
    dtype = np.dtype(md[b"numpy_dtype"].decode())
    shape = json.loads(md[b"shape"])
    base = "string" if dtype.kind == "U" else dtype.name
    return base + (f"[{', '.join(map(str, shape))}]" if shape else "")


def parquet_files(layout: Mapping[str, Any]) -> list[str]:
    """Every Parquet file of a layout (paths relative to the dataset's root)."""
    return [f for c in layout["configs"] for s in c["splits"] for f in s["files"]]


# ---------------------------------------------------------------------------------------------- dataset card YAML
def _feature(t) -> Any:
    import pyarrow as pa
    if pa.types.is_fixed_size_list(t):
        return {"list": _feature(t.value_type), "length": t.list_size}
    if pa.types.is_string(t):
        return "string"
    if pa.types.is_large_string(t):
        return "large_string"
    if pa.types.is_boolean(t):
        return "bool"
    if pa.types.is_integer(t) or pa.types.is_floating(t):
        return {"halffloat": "float16", "float": "float32", "double": "float64"}.get(str(t), str(t))
    raise TypeError(f"no dataset_info feature for Arrow type {t}")


def features_yaml(schema) -> list[dict[str, Any]]:
    """The ``dataset_info.features`` list of a Parquet schema, as ``datasets.Features._to_yaml_list`` writes it."""
    out = []
    for f in schema:
        feat = _feature(f.type)
        out.append({"name": f.name, **({"dtype": feat} if isinstance(feat, str) else feat)})
    return out

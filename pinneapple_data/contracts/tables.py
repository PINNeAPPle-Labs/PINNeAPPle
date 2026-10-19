"""Tables on disk with their contract: CSV, Parquet and HDF5, each with a JSON sidecar.

``write_table(df, "cases.parquet", contract)`` writes the table and ``cases.parquet.contract.json`` next to it.
Parquet also carries the contract in its schema metadata (key ``pinneapple.contract``) and HDF5 in a root attribute
of the same name, so the file still describes itself when the sidecar is lost. ``read_table`` returns
``(DataFrame, contract or None)``, preferring the sidecar.

Parquet needs ``pyarrow`` and HDF5 needs ``h5py``; CSV needs nothing beyond pandas.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .models import Contract, from_dict

SIDECAR_SUFFIX = ".contract.json"
META_KEY = "pinneapple.contract"
HDF5_GROUP = "table"


def sidecar_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + SIDECAR_SUFFIX)


def table_format(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in (".csv", ".txt"):
        return "csv"
    if suffix in (".parquet", ".pq"):
        return "parquet"
    if suffix in (".h5", ".hdf5"):
        return "hdf5"
    if suffix == ".json":
        return "json"
    raise ValueError(f"{path}: unsupported table format {suffix!r} (csv, parquet, h5/hdf5, json)")


def write_sidecar(path: str | Path, contract: Contract) -> Path:
    """Write ``<path>.contract.json``."""
    return contract.save(sidecar_path(path))


def read_sidecar(path: str | Path) -> Contract | None:
    sc = sidecar_path(path)
    return from_dict(json.loads(sc.read_text())) if sc.is_file() else None


def write_table(df, path: str | Path, contract: Contract | None = None) -> Path:
    """Write a DataFrame (or a dict of columns) as CSV / Parquet / HDF5 / JSON records, plus the sidecar."""
    import pandas as pd
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = table_format(path)
    meta = contract.to_json(indent=None) if contract is not None else None
    if fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "json":
        path.write_text(df.to_json(orient="records", indent=1))
    elif fmt == "parquet":
        pa, pq = _pyarrow()
        table = pa.Table.from_pandas(df, preserve_index=False)
        if meta is not None:
            table = table.replace_schema_metadata({**(table.schema.metadata or {}), META_KEY.encode(): meta.encode()})
        pq.write_table(table, path)
    else:
        h5py = _h5py()
        with h5py.File(path, "w") as f:
            g = f.create_group(HDF5_GROUP)
            g.attrs["columns"] = json.dumps([str(c) for c in df.columns])
            for c in df.columns:
                col = df[c].to_numpy()
                if col.dtype == object or col.dtype.kind in "US":
                    g.create_dataset(str(c), data=np.asarray([str(v) for v in col], dtype=object),
                                     dtype=h5py.string_dtype())
                else:
                    g.create_dataset(str(c), data=col)
            if meta is not None:
                f.attrs[META_KEY] = meta
    if contract is not None:
        write_sidecar(path, contract)
    return path


def read_table(path: str | Path):
    """Read a table and its contract: ``(DataFrame, contract or None)``."""
    import pandas as pd
    path = Path(path)
    fmt = table_format(path)
    embedded = None
    if fmt == "csv":
        df = pd.read_csv(path)
    elif fmt == "json":
        df = pd.DataFrame(json.loads(path.read_text()))
    elif fmt == "parquet":
        pa, pq = _pyarrow()
        table = pq.read_table(path)
        raw = (table.schema.metadata or {}).get(META_KEY.encode())
        embedded = raw.decode() if raw else None
        df = table.to_pandas()
    else:
        h5py = _h5py()
        with h5py.File(path, "r") as f:
            if HDF5_GROUP not in f:
                raise ValueError(f"{path}: no '/{HDF5_GROUP}' group (written by pinneapple_data.contracts?)")
            g = f[HDF5_GROUP]
            names = json.loads(g.attrs["columns"]) if "columns" in g.attrs else list(g.keys())
            cols = {}
            for n in names:
                data = g[n][()]
                if data.dtype == object or data.dtype.kind == "S":
                    data = np.asarray([v.decode() if isinstance(v, bytes) else v for v in data], dtype=object)
                cols[n] = data
            df = pd.DataFrame(cols)
            embedded = f.attrs.get(META_KEY)
            if isinstance(embedded, bytes):
                embedded = embedded.decode()
    contract = read_sidecar(path)
    if contract is None and embedded:
        contract = from_dict(json.loads(embedded))
    return df, contract


def convert(src: str | Path, dst: str | Path) -> Path:
    """Copy a table to another format (by suffix), carrying its contract along."""
    df, contract = read_table(src)
    return write_table(df, dst, contract)


def _pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("Parquet needs pyarrow: pip install pyarrow") from exc
    return pa, pq


def _h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("HDF5 needs h5py: pip install h5py") from exc
    return h5py

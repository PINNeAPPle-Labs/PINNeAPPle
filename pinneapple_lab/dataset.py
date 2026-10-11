"""Datasets produced by experiment runs: samples written in shards, with a card describing them.

A sample is a dict: arrays (any shape) and scalars / strings (labels, parameters). Samples are buffered and
written as ``shard_XXXXX.npz`` (arrays stacked when shapes agree, else stored per sample) with a
``samples.jsonl`` index of the scalar fields, and ``card.json`` (schema with shapes, dtypes, units and value
ranges, sample count, provenance, licence, checksum). ``LabStore.export_dataset`` concatenates the datasets of
many runs into one training set.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from collections.abc import Iterator
from typing import Any

import numpy as np


class DatasetWriter:
    def __init__(self, folder: str, *, name: str, description: str = "", units: dict[str, str] | None = None,
                 shard_size: int = 256, license: str = "", provenance: dict[str, Any] | None = None):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)
        self.name = name
        self.description = description
        self.units = units or {}
        self.shard_size = shard_size
        self.license = license
        self.provenance = provenance or {}
        self.n = 0
        self._buf: list[dict[str, Any]] = []
        self._shard = 0
        self._schema: dict[str, dict[str, Any]] = {}
        self._index = open(os.path.join(folder, "samples.jsonl"), "w")
        self._closed = False

    def add(self, **sample: Any) -> int:
        """Add one sample; returns its index within the dataset."""
        rec: dict[str, Any] = {}
        for k, v in sample.items():
            if isinstance(v, (str, bool, int, float, np.integer, np.floating)) or v is None:
                rec[k] = v.item() if isinstance(v, (np.integer, np.floating)) else v
                self._update_schema(k, None, v)
            else:
                a = np.asarray(v)
                rec[k] = a
                self._update_schema(k, a, None)
        self._buf.append(rec)
        idx = self.n
        self.n += 1
        if len(self._buf) >= self.shard_size:
            self._flush()
        return idx

    def _update_schema(self, k, arr, scalar):
        s = self._schema.setdefault(k, {"kind": "array" if arr is not None else "scalar"})
        if arr is not None:
            shape = list(arr.shape)
            if "shape" not in s:
                s["shape"], s["dtype"] = shape, str(arr.dtype)
            elif s["shape"] != shape:
                s["shape"] = "variable"
            if arr.size and np.issubdtype(arr.dtype, np.number):
                lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
                s["min"] = min(s.get("min", lo), lo)
                s["max"] = max(s.get("max", hi), hi)
        else:
            s["type"] = type(scalar).__name__
            if isinstance(scalar, (int, float, np.integer, np.floating)) and not isinstance(scalar, bool):
                s["min"] = min(s.get("min", float(scalar)), float(scalar))
                s["max"] = max(s.get("max", float(scalar)), float(scalar))
            elif isinstance(scalar, (str, bool)):
                vals = s.setdefault("values", [])
                if scalar not in vals and len(vals) < 50:
                    vals.append(scalar)

    def _flush(self) -> None:
        if not self._buf:
            return
        arrays: dict[str, Any] = {}
        keys = sorted({k for r in self._buf for k, v in r.items() if isinstance(v, np.ndarray)})
        for k in keys:
            vals = [r.get(k) for r in self._buf]
            if all(v is not None for v in vals) and len({v.shape for v in vals}) == 1:
                arrays[k] = np.stack(vals)
            else:
                for i, v in enumerate(vals):
                    if v is not None:
                        arrays[f"{k}__{i}"] = v
        path = os.path.join(self.folder, f"shard_{self._shard:05d}.npz")
        np.savez_compressed(path, **arrays)
        base = self.n - len(self._buf)
        for i, r in enumerate(self._buf):
            meta = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
            meta.update({"_index": base + i, "_shard": self._shard, "_row": i})
            self._index.write(json.dumps(meta) + "\n")
        self._index.flush()
        self._buf = []
        self._shard += 1

    def close(self) -> dict[str, Any]:
        if self._closed:
            return self.card()
        self._flush()
        self._index.close()
        card = self.card()
        with open(os.path.join(self.folder, "card.json"), "w") as f:
            json.dump(card, f, indent=1)
        self._closed = True
        return card

    def card(self) -> dict[str, Any]:
        h = hashlib.sha1()
        for p in sorted(glob.glob(os.path.join(self.folder, "shard_*.npz"))):
            with open(p, "rb") as f:
                h.update(f.read())
        return {"name": self.name, "description": self.description, "n_samples": self.n, "shards": self._shard,
                "schema": self._schema, "units": self.units, "license": self.license, "provenance": self.provenance,
                "sha1": h.hexdigest()}


def read_dataset(folder: str) -> Iterator[dict[str, Any]]:
    """Iterate over the samples of a dataset folder written by :class:`DatasetWriter`."""
    with open(os.path.join(folder, "samples.jsonl")) as f:
        metas = [json.loads(line) for line in f]
    cache: dict[int, Any] = {}
    for m in metas:
        sh = m["_shard"]
        if sh not in cache:
            cache.clear()
            cache[sh] = np.load(os.path.join(folder, f"shard_{sh:05d}.npz"))
        z = cache[sh]
        sample = {k: v for k, v in m.items() if not k.startswith("_")}
        row = m["_row"]
        for k in z.files:
            if "__" in k:
                base, i = k.rsplit("__", 1)
                if int(i) == row:
                    sample[base] = z[k]
            else:
                sample[k] = z[k][row]
        yield sample

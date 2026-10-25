"""Numeric tables uploaded to the arena (CSV / TSV / TXT with an auto-detected separator, JSON records, NPZ):
``parse_numeric_table(filename, content) -> (columns, array, n_skipped)``.

Rows that are not all numeric, or that have a different number of fields than the header, are skipped and
counted in ``n_skipped`` (so a caller can say how many were dropped); an NPZ file gives one column per array,
flattened and cut to the shortest.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np


def _parse_csv(content: bytes, sep: str = ",") -> tuple[list[str], list[list[float]]]:
    """Return (columns, rows_as_floats). Tries comma then semicolon then tab."""
    text = content.decode("utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("File is empty.")

    # Auto-detect separator
    header_line = lines[0]
    if sep == "auto":
        for s in [",", ";", "\t", " "]:
            if s in header_line:
                sep = s
                break
        else:
            sep = ","

    columns = [c.strip().strip('"') for c in header_line.split(sep)]
    rows: list[list[float]] = []
    for line in lines[1:]:
        parts = line.split(sep)
        if len(parts) != len(columns):
            continue  # ragged row
        try:
            rows.append([float(p.strip()) for p in parts])
        except ValueError:
            continue  # skip non-numeric rows
    return columns, rows


def _parse_npz(content: bytes) -> tuple[list[str], list[list[float]]]:
    """Load a .npz file; flatten each array to a column."""
    buf = io.BytesIO(content)
    npz = np.load(buf, allow_pickle=False)
    columns: list[str] = []
    cols_data: list[np.ndarray] = []
    for key in npz.files:
        arr = npz[key].flatten()
        columns.append(key)
        cols_data.append(arr)
    if not cols_data:
        raise ValueError("NPZ file contains no arrays.")
    n = min(len(a) for a in cols_data)
    rows = [[float(cols_data[j][i]) for j in range(len(cols_data))] for i in range(n)]
    return columns, rows


def _parse_json(content: bytes) -> tuple[list[str], list[list[float]]]:
    """Load a JSON array-of-objects."""
    records = json.loads(content.decode("utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("JSON must be a non-empty array of objects.")
    columns = list(records[0].keys())
    rows: list[list[float]] = []
    for rec in records:
        try:
            rows.append([float(rec[c]) for c in columns])
        except (KeyError, TypeError, ValueError):
            continue
    return columns, rows


def _parse_file(filename: str, content: bytes) -> tuple[list[str], list[list[float]]]:
    ext = Path(filename).suffix.lower()
    if ext in (".csv", ".txt", ".tsv"):
        sep = "\t" if ext == ".tsv" else "auto"
        return _parse_csv(content, sep=sep)
    if ext == ".npz":
        return _parse_npz(content)
    if ext == ".json":
        return _parse_json(content)
    # Try CSV as fallback
    return _parse_csv(content, sep="auto")


def _n_data_rows(filename: str, content: bytes) -> int:
    ext = Path(filename).suffix.lower()
    if ext == ".npz":
        return -1
    if ext == ".json":
        try:
            data = json.loads(content.decode("utf-8"))
            return len(data) if isinstance(data, list) else -1
        except ValueError:
            return -1
    return max(0, len([ln for ln in content.decode("utf-8", errors="replace").splitlines() if ln.strip()]) - 1)


def parse_numeric_table(filename: str, content: bytes) -> tuple[list[str], np.ndarray, int]:
    """``(columns, float32 array (rows, columns), n_skipped)`` of an uploaded table (format from the suffix)."""
    columns, rows = _parse_file(filename, content)
    arr = np.asarray(rows, dtype=np.float32).reshape(len(rows), len(columns)) if rows else np.zeros((0, len(columns)), np.float32)
    total = _n_data_rows(filename, content)
    return columns, arr, (total - len(rows)) if total >= 0 else 0

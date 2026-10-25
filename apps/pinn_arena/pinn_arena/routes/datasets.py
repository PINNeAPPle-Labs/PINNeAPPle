"""
PINNeAPPle Arena — Datasets API router.

Endpoints for uploading user-provided datasets (sensor observations, measurements,
experimental data) and querying their metadata.

Supported file formats: CSV, TSV, JSON (records), NPZ (numpy).
"""
# ruff: noqa: B008  (FastAPI Depends/Query/File defaults)
from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from pinn_arena.settings import DATASET_DIR
from pinneapple_arena.pointwise.tables import parse_numeric_table

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["datasets"])

# Persistent storage directory (same volume as SQLite data)
_DATASET_DIR = Path(DATASET_DIR)
_DATASET_DIR.mkdir(parents=True, exist_ok=True)

_MAX_SIZE_MB = 50
_MAX_BYTES   = _MAX_SIZE_MB * 1024 * 1024
_PREVIEW_ROWS = 5
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Metadata persistence
# ---------------------------------------------------------------------------

def _check_id(dataset_id: str) -> str:
    if not _UUID.fullmatch(dataset_id):
        raise FileNotFoundError(dataset_id)
    return dataset_id


def _meta_path(dataset_id: str) -> Path:
    return _DATASET_DIR / f"{_check_id(dataset_id)}.meta.json"


def _data_path(dataset_id: str) -> Path:
    return _DATASET_DIR / f"{_check_id(dataset_id)}.npz"


def _save_meta(dataset_id: str, meta: dict) -> None:
    with open(_meta_path(dataset_id), "w") as f:
        json.dump(meta, f)


def _load_meta(dataset_id: str) -> dict:
    p = _meta_path(dataset_id)
    if not p.exists():
        raise FileNotFoundError(dataset_id)
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/upload", summary="Upload a dataset file")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a CSV, TSV, JSON or NPZ dataset for use in an experiment.

    The file is parsed, stored, and a dataset_id is returned.
    Columns are auto-detected from the header (CSV/JSON) or array keys (NPZ).

    Limits: 50 MB per file.
    """
    content = await file.read()

    if len(content) > _MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {_MAX_SIZE_MB} MB limit ({len(content) / 1e6:.1f} MB uploaded).",
        )

    filename = file.filename or "dataset"
    try:
        columns, arr, n_skipped = parse_numeric_table(filename, content)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to parse file: {exc}") from exc

    if not len(arr):
        raise HTTPException(status_code=422, detail="No numeric rows found in file.")

    dataset_id = str(uuid.uuid4())

    # Save as NPZ for efficient retrieval
    np.savez_compressed(str(_data_path(dataset_id)), data=arr)

    # Compute per-column stats
    col_stats = []
    for j, col in enumerate(columns):
        vals = arr[:, j]
        col_stats.append({
            "name": col,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        })

    preview = [
        {col: float(arr[i, j]) for j, col in enumerate(columns)}
        for i in range(min(_PREVIEW_ROWS, len(arr)))
    ]

    meta = {
        "dataset_id": dataset_id,
        "original_filename": filename,
        "n_rows": len(arr),
        "n_skipped_rows": n_skipped,
        "n_cols": len(columns),
        "columns": columns,
        "col_stats": col_stats,
        "preview": preview,
    }
    _save_meta(dataset_id, meta)

    logger.info("Dataset %s uploaded: %d rows × %d cols from '%s'", dataset_id, len(arr), len(columns), filename)

    return meta


@router.get("/{dataset_id}", summary="Get dataset metadata")
async def get_dataset_info(dataset_id: str):
    """Return metadata for a previously uploaded dataset."""
    try:
        return _load_meta(dataset_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.") from None


@router.get("/{dataset_id}/data", summary="Get dataset as arrays")
async def get_dataset_data(
    dataset_id: str,
    coord_cols: str = "",
    output_cols: str = "",
):
    """
    Return the dataset as X (coords) and y (outputs) numpy arrays,
    based on the provided column mapping.

    coord_cols: comma-separated column names for coordinates
    output_cols: comma-separated column names for outputs
    """
    try:
        meta = _load_meta(dataset_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.") from None

    columns = meta["columns"]
    npz = np.load(str(_data_path(dataset_id)))
    arr = npz["data"]  # (n_rows, n_cols)

    coord_names = [c.strip() for c in coord_cols.split(",") if c.strip()] if coord_cols else []
    output_names = [c.strip() for c in output_cols.split(",") if c.strip()] if output_cols else []

    def _col_idx(name: str) -> int:
        if name not in columns:
            raise HTTPException(status_code=422, detail=f"Column '{name}' not found in dataset.")
        return columns.index(name)

    result: dict = {"n_rows": meta["n_rows"]}
    if coord_names:
        idxs = [_col_idx(c) for c in coord_names]
        result["X"] = arr[:, idxs].tolist()
    if output_names:
        idxs = [_col_idx(c) for c in output_names]
        result["y"] = arr[:, idxs].tolist()

    return result

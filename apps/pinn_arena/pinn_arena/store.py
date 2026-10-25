"""Experiments of the PINNeAPPle Arena: an in-memory cache of the running ones backed by the SQLite database
(``pinn_arena.db``), and the loader of uploaded datasets.

Read path:  memory first; DB fallback on cache miss (handles server restarts).
Write path: always update memory; always persist to DB (best-effort).
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np

from pinn_arena.db import SessionLocal
from pinn_arena.orm import ExperimentRecord
from pinn_arena.settings import DATASET_DIR

logger = logging.getLogger(__name__)
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def load_uploaded_dataset(
    dataset_id: str,
    column_mapping: dict | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a previously uploaded dataset from disk and return (coords, solution)
    as numpy arrays.

    column_mapping is a dict with keys: coord_cols, output_cols, time_col (both column lists are required).
    """
    if not _UUID.fullmatch(str(dataset_id)):
        raise FileNotFoundError(f"Dataset {dataset_id!r} is not a dataset id")
    data_path = os.path.join(DATASET_DIR, f"{dataset_id}.npz")
    meta_path = os.path.join(DATASET_DIR, f"{dataset_id}.meta.json")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset {dataset_id} not found on disk")

    npz = np.load(data_path, allow_pickle=False)
    arr = npz["data"]  # (n_rows, n_cols)

    with open(meta_path) as f:
        meta = json.load(f)
    columns: list[str] = meta["columns"]

    coord_cols = (column_mapping or {}).get("coord_cols") or []
    output_cols = (column_mapping or {}).get("output_cols") or []
    time_col = (column_mapping or {}).get("time_col")

    def _col_idx(name: str) -> int:
        if name in columns:
            return columns.index(name)
        raise ValueError(f"Column '{name}' not found in dataset")

    if coord_cols and output_cols:
        c_idxs = [_col_idx(c) for c in coord_cols]
        if time_col and time_col in columns:
            c_idxs.append(_col_idx(time_col))
        o_idxs = [_col_idx(c) for c in output_cols]
        coords   = arr[:, c_idxs].astype(np.float32)
        solution = arr[:, o_idxs].astype(np.float32)
    else:
        raise ValueError("an uploaded dataset needs a column mapping (coord_cols and output_cols)")

    return coords, solution


def _record_to_dict(record: Any) -> dict:
    """Convert an ExperimentRecord ORM row to the canonical in-memory dict format."""
    config = json.loads(record.config) if record.config else {}
    return {
        "id": record.id,
        "config": config,
        "status": (record.status or "PENDING").lower(),
        "progress": float(record.progress or 0.0),
        "current_model": record.current_model,
        "winner": record.winner,
        "error": record.error,
        "results": json.loads(record.results) if record.results else [],
        "summary": json.loads(record.result_summary) if record.result_summary else {},
        "model_statuses": json.loads(record.model_statuses) if record.model_statuses else {},
        "logs": json.loads(record.logs) if record.logs else [],
        "created_at": record.created_at.isoformat() if record.created_at else "",
        "updated_at": record.updated_at.isoformat() if record.updated_at else "",
        "started_at": record.created_at.isoformat() if record.created_at else "",
        "user_id": record.user_id,
    }


def _db_upsert_experiment(exp: dict) -> None:
    """Write all experiment fields to the DB. Both authenticated and guest experiments are persisted."""
    try:
        db = SessionLocal()
        try:
            record = db.query(ExperimentRecord).filter(
                ExperimentRecord.id == exp["id"]
            ).first()
            if record is None:
                record = ExperimentRecord(id=exp["id"])
                db.add(record)
            record.user_id = exp.get("user_id")
            record.problem_id = exp["config"].get("problem_id")
            record.status = exp["status"].upper()
            record.progress = float(exp["progress"])
            record.current_model = exp.get("current_model")
            record.winner = exp.get("winner")
            record.error = exp.get("error")
            record.config = json.dumps(exp["config"])
            record.model_statuses = json.dumps(exp.get("model_statuses", {}))
            record.logs = json.dumps(exp.get("logs", []))
            results = exp.get("results", [])
            if results:
                record.results = json.dumps(results)
            summary = exp.get("summary", {})
            if summary:
                record.result_summary = json.dumps(summary)
            if exp["status"].lower() in ("completed", "failed") and record.completed_at is None:
                record.completed_at = datetime.now(timezone.utc)
            record.updated_at = datetime.now(timezone.utc)
            db.commit()
        finally:
            db.close()
    except Exception as exc:
        logger.debug("DB upsert experiment skipped: %s", exc)


def _db_update_status(
    experiment_id: str,
    status: str,
    progress: float,
    current_model: str | None = None,
    error: str | None = None,
) -> None:
    """Lightweight DB update: only status, progress, current_model, error."""
    try:
        db = SessionLocal()
        try:
            record = db.query(ExperimentRecord).filter(
                ExperimentRecord.id == experiment_id
            ).first()
            if record is not None:
                record.status = status.upper()
                record.progress = float(progress)
                if current_model is not None:
                    record.current_model = current_model
                if error is not None:
                    record.error = error
                if status.lower() in ("completed", "failed") and record.completed_at is None:
                    record.completed_at = datetime.now(timezone.utc)
                record.updated_at = datetime.now(timezone.utc)
                db.commit()
        finally:
            db.close()
    except Exception as exc:
        logger.debug("DB update status skipped: %s", exc)


def _db_load_experiment(experiment_id: str) -> dict | None:
    """Load one experiment from DB; returns None if not found."""
    try:
        db = SessionLocal()
        try:
            record = db.query(ExperimentRecord).filter(
                ExperimentRecord.id == experiment_id
            ).first()
            return _record_to_dict(record) if record is not None else None
        finally:
            db.close()
    except Exception as exc:
        logger.debug("DB load experiment failed: %s", exc)
        return None


def _db_list_all_experiments(user_id: int | None = None) -> list[dict]:
    """Return experiments from DB. Without user_id returns ALL experiments (global leaderboard view)."""
    try:
        db = SessionLocal()
        try:
            q = db.query(ExperimentRecord)
            if user_id is not None:
                q = q.filter(ExperimentRecord.user_id == user_id)
            records = q.order_by(ExperimentRecord.created_at.desc()).all()
            return [_record_to_dict(r) for r in records]
        finally:
            db.close()
    except Exception as exc:
        logger.debug("DB list experiments failed: %s", exc)
        return []


def _db_delete_experiment(experiment_id: str) -> bool:
    """Delete an experiment row from DB. Returns True if a row was deleted."""
    try:
        db = SessionLocal()
        try:
            record = db.query(ExperimentRecord).filter(
                ExperimentRecord.id == experiment_id
            ).first()
            if record is not None:
                db.delete(record)
                db.commit()
                return True
            return False
        finally:
            db.close()
    except Exception as exc:
        logger.debug("DB delete experiment failed: %s", exc)
        return False


class ExperimentStore:
    """
    In-memory cache of active experiments, backed by SQLite via the ORM.

    Read path:  memory first; DB fallback on cache miss (handles server restarts).
    Write path: always update memory; always persist to DB (best-effort).
    list_all:   DB is the authoritative source; in-memory state overrides for active runs.
    """

    _experiments: dict[str, dict] = {}

    @classmethod
    def create(cls, config: dict, user_id: int | None = None) -> str:
        """Create a new experiment, persist immediately, return its ID."""
        experiment_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        exp: dict = {
            "id": experiment_id,
            "config": config,
            "status": "pending",
            "progress": 0.0,
            "current_model": None,
            "results": [],
            "summary": {},
            "winner": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
            "model_statuses": {},
            "started_at": now,
            "logs": [],
            "user_id": user_id,
        }
        cls._experiments[experiment_id] = exp
        _db_upsert_experiment(exp)
        return experiment_id

    @classmethod
    def get(cls, experiment_id: str) -> dict | None:
        """Return experiment dict from memory, falling back to DB on cache miss."""
        if experiment_id in cls._experiments:
            return cls._experiments[experiment_id]
        exp = _db_load_experiment(experiment_id)
        if exp is not None:
            cls._experiments[experiment_id] = exp
        return exp

    @classmethod
    def list_all(cls, user_id: int | None = None) -> list[dict]:
        """
        Return experiments newest-first.

        - Authenticated users: see ALL experiments in the DB (this is a shared
          benchmarking platform — the leaderboard is global). In-memory live state
          overrides stale DB rows for currently-running experiments.
        - Guest (user_id=None): only in-memory ownerless experiments (no DB access).
        """
        if user_id is not None:
            # Authenticated: DB is authoritative for all persisted experiments
            merged: dict[str, dict] = {e["id"]: e for e in _db_list_all_experiments()}
            # Override with live in-memory state for any running experiment
            for exp_id, exp in cls._experiments.items():
                if exp.get("status") in ("pending", "running"):
                    merged[exp_id] = exp
        else:
            # Guest: only ownerless in-memory experiments
            merged = {
                exp_id: exp
                for exp_id, exp in cls._experiments.items()
                if exp.get("user_id") is None
            }
        exps = list(merged.values())
        exps.sort(key=lambda e: e.get("created_at", ""), reverse=True)
        return exps

    @classmethod
    def update_status(
        cls,
        experiment_id: str,
        status: str,
        progress: float,
        current_model: str | None = None,
        error: str | None = None,
    ) -> None:
        """Update status/progress in memory and DB."""
        exp = cls._experiments.get(experiment_id)
        if exp is None:
            return
        exp["status"] = status
        exp["progress"] = float(progress)
        exp["updated_at"] = datetime.now(timezone.utc).isoformat()
        if current_model is not None:
            exp["current_model"] = current_model
        if error is not None:
            exp["error"] = error
        _db_update_status(experiment_id, status, progress, current_model, error)

    @classmethod
    def store_result(cls, experiment_id: str, result: dict) -> None:
        """Persist final results (full upsert including model_results, logs, summary)."""
        exp = cls._experiments.get(experiment_id)
        if exp is None:
            return
        exp["results"] = result.get("model_results", [])
        exp["summary"] = result.get("summary", {})
        exp["winner"] = result.get("winner")
        exp["updated_at"] = datetime.now(timezone.utc).isoformat()
        _db_upsert_experiment(exp)

    @classmethod
    def delete(cls, experiment_id: str) -> bool:
        """Delete from memory and DB. Returns True if found in either."""
        in_memory = experiment_id in cls._experiments
        if in_memory:
            del cls._experiments[experiment_id]
        in_db = _db_delete_experiment(experiment_id)
        return in_memory or in_db

    @classmethod
    def set_model_status(
        cls,
        experiment_id: str,
        arch_id: str,
        status: str,  # "waiting" | "training" | "completed" | "failed"
        progress: float = 0.0,
        current_loss: float | None = None,
        loss_history: list | None = None,
    ) -> None:
        """Update per-model training state in memory only (high-frequency; flushed on store_result)."""
        exp = cls._experiments.get(experiment_id)
        if exp is None:
            return
        if "model_statuses" not in exp:
            exp["model_statuses"] = {}
        prev = exp["model_statuses"].get(arch_id, {})
        exp["model_statuses"][arch_id] = {
            "status": status,
            "progress": float(progress),
            "current_loss": current_loss,
            "training_history": loss_history if loss_history is not None else prev.get("training_history", []),
        }
        exp["updated_at"] = datetime.now(timezone.utc).isoformat()

    @classmethod
    def cancel(cls, experiment_id: str) -> None:
        """Mark a pending or running experiment cancelled; its training stops at the next progress report."""
        exp = cls._experiments.get(experiment_id)
        if exp is not None:
            exp["cancelled"] = True
        cls.update_status(experiment_id, "failed", exp["progress"] if exp else 0.0, error="Cancelled by user")

    @classmethod
    def is_cancelled(cls, experiment_id: str) -> bool:
        exp = cls._experiments.get(experiment_id)
        return bool(exp and exp.get("cancelled"))

    @classmethod
    def append_log(cls, experiment_id: str, message: str) -> None:
        """Append a log line in memory (flushed to DB on store_result)."""
        exp = cls._experiments.get(experiment_id)
        if exp is None:
            return
        if "logs" not in exp:
            exp["logs"] = []
        exp["logs"].append(message)
        exp["updated_at"] = datetime.now(timezone.utc).isoformat()

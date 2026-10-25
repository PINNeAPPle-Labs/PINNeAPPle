"""Experiments: create (queued as a background task), poll, results, plots, cancel, weights, delete.

Limits per experiment (environment, see the README): PINN_ARENA_MAX_EPOCHS, PINN_ARENA_MAX_ARCHITECTURES,
PINN_ARENA_MAX_COLLOCATION; PINN_ARENA_MAX_RUNNING experiments train at once (429 above); each caller may create
PINN_ARENA_FREE_EXPERIMENTS experiments per month (appkit metering, 429 above).
"""
# ruff: noqa: B008  (FastAPI Depends/Query/File defaults)
from __future__ import annotations

import io
import logging
import math
import os
import re
import threading
import zipfile

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from pinn_arena import settings
from pinn_arena.auth import get_current_user_optional
from pinn_arena.orm import User
from pinn_arena.pipeline import ExperimentCancelled, ExperimentPipeline
from pinn_arena.schemas import (
    CreateExperimentRequest,
    ExperimentStatus,
)
from pinn_arena.settings import MODELS_DIR
from pinn_arena.store import ExperimentStore
from pinneapple_arena.pointwise.architectures import ARCHITECTURES_CATALOG
from pinneapple_arena.pointwise.problems import PROBLEMS_CATALOG

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/experiments", tags=["experiments"])
_running = threading.BoundedSemaphore(max(1, settings.MAX_RUNNING))
_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,80}")


# ---------------------------------------------------------------------------
# Ownership helper
# ---------------------------------------------------------------------------

def _check_ownership(exp: dict, current_user: User | None, action: str) -> None:
    """Raise 403 if the experiment belongs to a different user.

    Ownerless experiments (created by guests) can be acted on by anyone.
    Owned experiments can only be acted on by their owner.
    """
    owner_id = exp.get("user_id")
    if owner_id is None:
        return  # guest experiment — no restrictions
    if current_user is None or current_user.id != owner_id:
        raise HTTPException(
            status_code=403,
            detail=f"Not authorized to {action} this experiment.",
        )


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

async def run_experiment_background(experiment_id: str, config: dict) -> None:
    """
    Background task that runs the full experiment pipeline.

    Updates ExperimentStore with progress, current model, and final results.
    Catches all exceptions to ensure status is always updated.
    """
    ExperimentStore.update_status(experiment_id, "running", 0.0)
    try:
        def _progress_cb(frac: float, message: str = "") -> None:
            ExperimentStore.update_status(
                experiment_id, "running", frac,
                current_model=ExperimentStore.get(experiment_id).get("current_model"),
            )

        pipeline = ExperimentPipeline(
            experiment_id=experiment_id,
            config=config,
            progress_callback=_progress_cb,
        )
        result = await pipeline.run()
        if ExperimentStore.is_cancelled(experiment_id):
            raise ExperimentCancelled(experiment_id)
        ExperimentStore.store_result(experiment_id, result)
        ExperimentStore.update_status(experiment_id, "completed", 1.0)
        logger.info("Experiment %s completed successfully.", experiment_id)
    except ExperimentCancelled:
        logger.info("Experiment %s stopped after a cancel request.", experiment_id)
    except Exception as exc:
        logger.error("Experiment %s failed: %s", experiment_id, exc, exc_info=True)
        ExperimentStore.update_status(
            experiment_id, "failed", 0.0, error=str(exc)
        )
    finally:
        _running.release()


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

@router.post("", summary="Create and start a new experiment")
async def create_experiment(
    request: Request,
    body: CreateExperimentRequest,
    background_tasks: BackgroundTasks,
    current_user: User | None = Depends(get_current_user_optional),
):
    """
    Create a new arena experiment and start it in the background.

    The experiment will:
    1. Generate a reference dataset for the specified problem
    2. Train each specified architecture (supervised + optional physics loss)
    3. Evaluate all models against the reference solution
    4. Rank models by L2 relative error

    Returns the experiment ID immediately. Poll /experiments/{id}/status
    to check progress, or /experiments/{id}/results when completed.

    Limits: see the module docstring.
    """
    config_dict = body.config.model_dump()
    if config_dict["problem_id"] not in PROBLEMS_CATALOG:
        raise HTTPException(status_code=404, detail=f"Problem '{config_dict['problem_id']}' not found.")
    archs = config_dict["architecture_ids"]
    unknown = [a for a in archs if a not in ARCHITECTURES_CATALOG]
    if unknown or not archs:
        raise HTTPException(status_code=422, detail=f"Unknown architecture(s): {unknown}" if unknown
                            else "Select at least one architecture.")
    if len(archs) > settings.MAX_ARCHITECTURES:
        raise HTTPException(status_code=422, detail=f"At most {settings.MAX_ARCHITECTURES} architectures per "
                                                    "experiment (PINN_ARENA_MAX_ARCHITECTURES).")
    if config_dict["epochs"] > settings.MAX_EPOCHS:
        raise HTTPException(status_code=422, detail=f"At most {settings.MAX_EPOCHS} epochs (PINN_ARENA_MAX_EPOCHS).")
    for key in ("n_train_col", "n_train_bc", "n_train_ic"):
        if config_dict[key] > settings.MAX_COLLOCATION:
            raise HTTPException(status_code=422, detail=f"{key} is above {settings.MAX_COLLOCATION} "
                                                        "(PINN_ARENA_MAX_COLLOCATION).")
    rt = getattr(request.app.state, "runtime", None)
    if not _running.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="The server is training other experiments; try again later "
                                                    "(PINN_ARENA_MAX_RUNNING).", headers={"Retry-After": "60"})
    try:
        if rt is not None:
            rt.consume(request, "experiments")       # 429 above the free limit of this caller
    except Exception:
        _running.release()
        raise

    # Remap ExperimentConfig fields to pipeline's expected config shape
    pipeline_config = {
        "problem_id": config_dict["problem_id"],
        "problem_type": config_dict.get("problem_type", "forward"),
        "geometry_id": config_dict.get("geometry_id"),
        "architectures": config_dict["architecture_ids"],
        "problem_parameters": config_dict.get("custom_parameters", {}),
        "training": {
            "epochs": config_dict["epochs"],
            "lr": config_dict["learning_rate"],
            "n_col": config_dict["n_train_col"],
            "n_bc": config_dict["n_train_bc"],
            "n_ic": config_dict["n_train_ic"],
            "n_eval": min(1000, config_dict["n_train_col"] // 2),
            "batch_size": config_dict["batch_size"],
            "loss_weights": config_dict["loss_weights"],
            "use_physics_loss": True,
        },
        # Optional uploaded dataset
        "dataset_id": config_dict.get("dataset_id"),
        "column_mapping": config_dict.get("column_mapping"),
        # Inverse config (only relevant when problem_type == 'inverse')
        "inverse_config": config_dict.get("inverse_config"),
    }

    user_id = current_user.id if current_user else None
    try:
        experiment_id = ExperimentStore.create(pipeline_config, user_id=user_id)
    except Exception:
        _running.release()
        raise
    background_tasks.add_task(run_experiment_background, experiment_id, pipeline_config)

    return {
        "experiment_id": experiment_id,
        "status": "pending",
        "message": "Experiment created and queued for execution.",
    }


@router.get("", summary="List all experiments")
async def list_experiments(
    current_user: User | None = Depends(get_current_user_optional),
):
    """
    List experiments for the current user, newest first.

    Authenticated users see only their own persisted experiments.
    Guest users see only their in-memory session experiments.
    """
    user_id = current_user.id if current_user else None
    all_experiments = ExperimentStore.list_all(user_id=user_id)
    return {
        "total": len(all_experiments),
        "experiments": [
            {
                "experiment_id": exp["id"],
                "id": exp["id"],
                "status": exp["status"],
                "progress": exp["progress"],
                "created_at": exp["created_at"],
                "updated_at": exp["updated_at"],
                "problem_id": exp["config"].get("problem_id"),
                "n_architectures": len(exp["config"].get("architectures", [])),
                "winner": exp.get("winner"),
                "current_model": exp.get("current_model"),
            }
            for exp in all_experiments
        ],
    }


@router.get("/{experiment_id}/status", summary="Poll experiment status")
async def get_experiment_status(experiment_id: str):
    """
    Poll the status of an in-flight or completed experiment.

    Returns:
    - status: PENDING | RUNNING | COMPLETED | FAILED
    - progress: 0.0 to 1.0
    - current_model: name of the architecture currently being trained
    - current_architecture: alias for current_model (frontend compat)
    - message: human-readable status message
    - model_statuses: per-model training status dict
    - logs: list of log messages
    - started_at: ISO timestamp when experiment was created
    """
    exp = ExperimentStore.get(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")

    status_str = exp["status"].upper()
    try:
        status_enum = ExperimentStatus(status_str)
    except ValueError:
        status_enum = ExperimentStatus.PENDING

    message = ""
    if status_enum == ExperimentStatus.RUNNING:
        current = exp.get("current_model") or ""
        progress_pct = int(exp["progress"] * 100)
        message = f"Training {current} ({progress_pct}%)" if current else f"{progress_pct}% complete"
    elif status_enum == ExperimentStatus.COMPLETED:
        winner = exp.get("winner", "N/A")
        message = f"Completed. Best model: {winner}"
    elif status_enum == ExperimentStatus.FAILED:
        message = exp.get("error") or "Experiment failed."

    return {
        "experiment_id": experiment_id,
        "status": status_enum.value,
        "progress": float(exp["progress"]),
        "current_model": exp.get("current_model") or "",
        "current_architecture": exp.get("current_model") or "",  # frontend alias
        "message": message,
        "model_statuses": exp.get("model_statuses", {}),
        "logs": exp.get("logs", []),
        "started_at": exp.get("started_at", ""),
    }


@router.get("/{experiment_id}/results", summary="Get experiment results")
async def get_results(experiment_id: str):
    """
    Get the full results of a completed experiment.

    Returns ranked model results with metrics (L2 relative error, RMSE, R2),
    training histories, and a summary comparing all architectures.
    """
    exp = ExperimentStore.get(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")

    if exp["status"] == "pending":
        raise HTTPException(status_code=202, detail="Experiment has not started yet.")

    if exp["status"] == "running":
        raise HTTPException(
            status_code=202,
            detail=f"Experiment is still running ({int(exp['progress'] * 100)}% complete).",
        )

    if exp["status"] == "failed":
        raise HTTPException(
            status_code=422,
            detail=f"Experiment failed: {exp.get('error', 'Unknown error')}",
        )

    model_results = exp.get("results", [])
    summary = exp.get("summary", {})

    # Transform pipeline result format → frontend ModelResult schema
    def _finite(value, default: float) -> float:
        # JSON can't carry inf/NaN (a diverged run produces them); fall back to
        # the same defaults used for a missing metric.
        try:
            v = float(value)
        except (TypeError, ValueError):
            return default
        return v if math.isfinite(v) else default

    normalized = []
    for r in model_results:
        arch_id = r.get("arch_id", r.get("architecture_id", "unknown"))
        arch_info = ARCHITECTURES_CATALOG.get(arch_id)
        arch_name = arch_info.name if arch_info else arch_id.replace("_", " ").title()

        raw_metrics = r.get("metrics", {})
        metrics = {
            "l2_relative_error": _finite(raw_metrics.get("l2_relative_error",
                                         raw_metrics.get("l2_relative")), 1.0),
            "l1_error": _finite(raw_metrics.get("l1_error", raw_metrics.get("l1")), 1.0),
            "rmse": _finite(raw_metrics.get("rmse"), 1.0),
            "max_error": _finite(raw_metrics.get("max_error"), 1.0),
            "r2_score": _finite(raw_metrics.get("r2_score", raw_metrics.get("r2")), 0.0),
        }

        loss_history = r.get("loss_history", [])
        training_history = [
            {"epoch": i + 1, "total_loss": float(v)}
            for i, v in enumerate(loss_history)
        ]

        raw_family = arch_info.family if arch_info else None
        arch_family = raw_family.value if hasattr(raw_family, "value") else (str(raw_family) if raw_family else "")

        normalized.append({
            "architecture_id": arch_id,
            "architecture_name": arch_name,
            "architecture_family": arch_family,
            "status": r.get("status", "completed"),
            "error": r.get("error"),
            "training_time_s": r.get("training_time_s", 0.0),
            "final_loss": _finite(loss_history[-1], 1.0) if loss_history else 1.0,
            "metrics": metrics,
            "converged": r.get("status") == "completed" and metrics["l2_relative_error"] < 0.5,
            "training_history": training_history,
            "physics_loss_enabled": r.get("physics_loss_enabled", False),
            "inverse_result": r.get("inverse_result"),
            "n_pde_evaluations": r.get("n_pde_evaluations"),
            "time_to_convergence_s": r.get("time_to_convergence_s"),
            "convergence_epoch": r.get("convergence_epoch"),
            "peak_gpu_memory_mb": r.get("peak_gpu_memory_mb"),
            "peak_cpu_memory_mb": r.get("peak_cpu_memory_mb"),
            "inference_time_ms": r.get("inference_time_ms"),
            "quality_score": r.get("quality_score"),
            "quality_level": r.get("quality_level", "unknown"),
            "quality_flags": r.get("quality_flags", []),
            "pred_std_ratio": r.get("pred_std_ratio"),
        })

    # Completed models by L2 relative error ascending; failed ones last
    normalized.sort(key=lambda x: (x["status"] != "completed", x["metrics"]["l2_relative_error"]))
    ranking = [m["architecture_id"] for m in normalized]

    return {
        "experiment_id": experiment_id,
        "problem_id": exp["config"].get("problem_id"),
        "problem_type": exp["config"].get("problem_type", "forward"),
        "geometry_id": exp["config"].get("geometry_id"),
        "status": exp["status"],
        "created_at": exp.get("created_at", ""),
        "completed_at": exp.get("updated_at", ""),
        "model_results": normalized,
        "ranking": ranking,
        "best_model_id": ranking[0] if ranking else "",
        "comparison_summary": summary or {},
    }


@router.post("/{experiment_id}/cancel", summary="Cancel a running experiment")
async def cancel_experiment(
    experiment_id: str,
    current_user: User | None = Depends(get_current_user_optional),
):
    """
    Cancel a pending or running experiment.

    Marks the experiment as failed with a 'Cancelled by user' message.
    The background task may continue briefly, but the status is updated immediately.
    Only the owner of the experiment can cancel it (guests can cancel ownerless experiments).
    """
    exp = ExperimentStore.get(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")
    _check_ownership(exp, current_user, action="cancel")
    if exp["status"] not in ("pending", "running"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel experiment with status '{exp['status']}'.",
        )
    ExperimentStore.cancel(experiment_id)
    logger.info("Experiment %s cancelled by user.", experiment_id)
    return {"message": "Experiment cancelled."}


@router.get("/{experiment_id}/plots/{model_id}", summary="Get plot data for a model")
async def get_model_plots(experiment_id: str, model_id: str):
    """
    Return JSON-serializable plot data arrays for a specific model result.

    Used by the frontend to render solution field comparisons, error maps,
    and training loss curves.

    Returns:
    - X_eval: evaluation coordinate points
    - y_true: reference solution values
    - y_pred: model prediction values
    - loss_history: training loss per epoch
    - metrics: computed error metrics
    """
    exp = ExperimentStore.get(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")

    if exp["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Experiment is not completed yet (status: {exp['status']}).",
        )

    model_result = None
    for r in exp.get("results", []):
        if r.get("arch_id") == model_id:
            model_result = r
            break

    if model_result is None:
        available = [r.get("arch_id") for r in exp.get("results", [])]
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found in experiment. Available: {available}",
        )

    return {
        "experiment_id": experiment_id,
        "model_id": model_id,
        "status": model_result.get("status"),
        "metrics": model_result.get("metrics", {}),
        "loss_history": model_result.get("loss_history", []),
        "X_eval": model_result.get("X_eval", []),
        "y_true": model_result.get("y_true", []),
        "y_pred": model_result.get("y_pred", []),
        "training_time_s": model_result.get("training_time_s", 0.0),
        "rank": model_result.get("rank"),
    }


_MODELS_ROOT = MODELS_DIR


@router.get(
    "/{experiment_id}/models/{arch_id}/download",
    summary="Download trained model weights (.pt)",
)
async def download_model(experiment_id: str, arch_id: str):
    """
    Download the trained PyTorch state-dict for a specific architecture in an experiment.
    Returns a .pt file that can be loaded with torch.load().
    """
    exp = ExperimentStore.get(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")
    if exp["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail="Model weights are only available after the experiment completes.",
        )

    if not _SAFE_ID.fullmatch(arch_id):
        raise HTTPException(status_code=404, detail="Unknown architecture id.")
    weights_path = os.path.join(_MODELS_ROOT, experiment_id, f"{arch_id}.pt")
    if not os.path.isfile(weights_path):
        raise HTTPException(
            status_code=404,
            detail=f"Weights for '{arch_id}' not found. The model may have failed to train.",
        )

    return FileResponse(
        weights_path,
        media_type="application/octet-stream",
        filename=f"{arch_id}.pt",
    )


@router.get(
    "/{experiment_id}/models/download-all",
    summary="Download all trained model weights as a ZIP archive",
)
async def download_all_models(experiment_id: str):
    """
    Download all successfully trained model weights for an experiment as a single ZIP archive.
    Each .pt file inside the ZIP is named {arch_id}.pt.
    """
    exp = ExperimentStore.get(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")
    if exp["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail="Model weights are only available after the experiment completes.",
        )

    models_dir = os.path.join(_MODELS_ROOT, experiment_id)
    if not os.path.isdir(models_dir):
        raise HTTPException(status_code=404, detail="No trained model weights found for this experiment.")

    pt_files = [f for f in os.listdir(models_dir) if f.endswith(".pt")]
    if not pt_files:
        raise HTTPException(status_code=404, detail="No .pt files found for this experiment.")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(pt_files):
            zf.write(os.path.join(models_dir, fname), arcname=fname)
    buf.seek(0)

    problem_id = exp["config"].get("problem_id", "experiment")
    zip_name = f"models_{problem_id}_{experiment_id[:8]}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_name}"'},
    )


@router.delete("/{experiment_id}", summary="Delete an experiment")
async def delete_experiment(
    experiment_id: str,
    current_user: User | None = Depends(get_current_user_optional),
):
    """
    Delete an experiment and all its data from the store and database.
    Only the owner of the experiment can delete it (guests can delete ownerless experiments).
    """
    exp = ExperimentStore.get(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")
    _check_ownership(exp, current_user, action="delete")
    ExperimentStore.delete(experiment_id)
    return {"message": f"Experiment '{experiment_id}' deleted successfully."}

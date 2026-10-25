"""Runs one PINNeAPPle Arena experiment (the protocol itself is ``pinneapple_arena.pointwise``)."""
from __future__ import annotations

import asyncio
import gc
import logging
import os
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from pinn_arena.settings import MODELS_DIR
from pinn_arena.store import ExperimentStore, load_uploaded_dataset
from pinneapple_arena.pointwise import (
    PROBLEMS_CATALOG,
    DatasetGenerator,
    PhysicsLossGenerator,
    best_device,
    field_metrics,
    instantiate_model,
    predict,
    prediction_quality,
    rank_models,
    run_inverse_solver,
    train_closed_form_model,
    train_torch_model,
)

logger = logging.getLogger(__name__)


def _library_version() -> str:
    try:
        import pinneapple
        return str(getattr(pinneapple, "__version__", "unknown"))
    except Exception:  # noqa: BLE001
        return "unknown"


class ExperimentCancelled(RuntimeError):
    """A cancel request reached a progress report: training stops there."""


class ExperimentPipeline:
    """
    One arena experiment: reference dataset -> per architecture: build, train, evaluate on the held-out split ->
    ranking. The training and scoring are the library's (``pinneapple_arena.pointwise``); this class schedules them
    and records progress in the ExperimentStore.
    """

    def __init__(
        self,
        experiment_id: str,
        config: dict,
        progress_callback: Callable[[float, str], None] | None = None,
    ):
        self.experiment_id = experiment_id
        self.config = config
        self.progress = 0.0
        self.status = "pending"
        self.current_model: str | None = None
        self.results: list[dict] = []
        self._progress_cb = progress_callback

    def _report_progress(self, frac: float, message: str = "") -> None:
        if ExperimentStore.is_cancelled(self.experiment_id):
            raise ExperimentCancelled(self.experiment_id)
        self.progress = max(0.0, min(1.0, frac))
        if self._progress_cb is not None:
            try:
                self._progress_cb(self.progress, message)
            except Exception:
                pass
        ExperimentStore.update_status(
            self.experiment_id, "running", self.progress, self.current_model
        )
        logger.debug("Experiment %s: %.1f%% - %s", self.experiment_id, self.progress * 100, message)

    async def run(self) -> dict:
        """
        Full pipeline:
        1. Generate reference dataset
        2. Prepare collocation data
        3. For each architecture: instantiate, compile physics loss, train, evaluate
        4. Rank models by L2 relative error
        5. Return structured result

        Returns
        -------
        dict with keys: experiment_id, model_results, summary, winner
        """
        problem_id: str = self.config.get("problem_id", "burgers_1d")
        problem_type: str = self.config.get("problem_type", "forward")
        arch_ids: list[str] = self.config.get("architectures", ["vanilla_pinn"])
        problem_params: dict = self.config.get("problem_parameters", {})
        training_cfg: dict = self.config.get("training", {})
        dataset_id: str | None = self.config.get("dataset_id")
        column_mapping: dict | None = self.config.get("column_mapping")
        inverse_cfg: dict | None = self.config.get("inverse_config")

        n_epochs = int(training_cfg.get("epochs", 200))
        lr = float(training_cfg.get("lr", 1e-3))
        n_col = int(training_cfg.get("n_col", 2000))
        n_bc = int(training_cfg.get("n_bc", 400))
        n_ic = int(training_cfg.get("n_ic", 400))
        n_eval = int(training_cfg.get("n_eval", 500))
        use_physics_loss = bool(training_cfg.get("use_physics_loss", True))
        device = training_cfg.get("device", best_device())

        loop = asyncio.get_running_loop()

        gen: Any | None = None        # only set when generating (not uploading)
        ref_data: dict = {}              # filled by generator; empty for uploaded datasets

        # ── Step 1: Dataset (uploaded or generated) ──────────────────────────
        if dataset_id:
            # An uploaded dataset that can not be read fails the experiment: it is never replaced by generated data.
            self._report_progress(0.02, "Loading uploaded dataset")
            try:
                coords, solution = await loop.run_in_executor(
                    None, lambda: load_uploaded_dataset(dataset_id, column_mapping)
                )
                logger.info("Loaded uploaded dataset %s: %s coords, %s solution", dataset_id, coords.shape, solution.shape)
            except Exception as exc:
                raise RuntimeError(f"Uploaded dataset {dataset_id} could not be used: {exc}") from exc

        if not dataset_id:
            self._report_progress(0.02, "Generating reference dataset")
            gen = DatasetGenerator(problem_id, problem_params)
            try:
                ref_data = await loop.run_in_executor(
                    None, lambda: gen.generate(n_samples=max(n_eval + n_col + 500, 2000))
                )
            except Exception as exc:
                logger.error("Dataset generation failed: %s", exc)
                raise RuntimeError(f"Dataset generation failed: {exc}") from exc

            coords   = ref_data["coords"]
            solution = ref_data["solution"]

        n_total  = len(coords)

        # ── Step 2: Non-overlapping train / val / test split ─────────────────
        # test  → held-out evaluation (never seen during training)
        # val   → early-stopping monitor (not used for gradient updates)
        # train → supervised training set
        self._report_progress(0.05, "Splitting data (train / val / test)")
        all_idx = np.random.permutation(n_total)
        n_test  = max(50, min(n_eval,       n_total // 5))
        n_val_  = max(50, min(n_eval // 4,  n_total // 10))
        n_train_ = min(n_col, n_total - n_test - n_val_)

        test_idx  = all_idx[:n_test]
        val_idx_  = all_idx[n_test : n_test + n_val_]
        train_idx = all_idx[n_test + n_val_ : n_test + n_val_ + n_train_]

        X_train_matched = coords[train_idx]
        y_train_matched = solution[train_idx]
        X_val_data      = coords[val_idx_]
        y_val_data      = solution[val_idx_]
        X_eval          = coords[test_idx]
        y_eval          = solution[test_idx]

        logger.info(
            "Data split — train:%d  val:%d  test:%d  (total:%d)",
            len(train_idx), len(val_idx_), len(test_idx), n_total,
        )

        # ── Step 3: Separate collocation points for physics residual ─────────
        self._report_progress(0.06, "Preparing collocation points")
        X_col = None
        if gen is not None:             # only available when generating (not uploading)
            try:
                col_data = await loop.run_in_executor(
                    None, lambda: gen.get_collocation_data(n_col=n_col, n_bc=n_bc, n_ic=n_ic)
                )
                X_col = col_data.get("X_col")
            except Exception as exc:
                logger.warning("get_collocation_data failed: %s — physics loss will be skipped", exc)

        # ── Step 4: Compile physics losses ───────────────────────────────────
        self._report_progress(0.08, "Compiling physics losses")
        loss_gen = PhysicsLossGenerator(problem_id, problem_params)
        physics_loss_fn = None
        if use_physics_loss:
            loss_weights = training_cfg.get("loss_weights", {"pde": 1.0, "bc": 10.0, "ic": 10.0})
            physics_loss_fn = await loop.run_in_executor(
                None, lambda: loss_gen.compile_losses(loss_weights)
            )

        # ── Dimensions ───────────────────────────────────────────────────────
        # The data decides the model's input and output width (an uploaded table may differ from the catalog).
        in_dim  = int(coords.shape[1])
        out_dim = int(solution.shape[1]) if solution.ndim == 2 else 1
        n_archs = len(arch_ids)

        for aid in arch_ids:
            ExperimentStore.set_model_status(self.experiment_id, aid, "waiting", 0.0)

        # ── Step 5: Train architectures ──────────────────────────────────────
        # On CPU we train all architectures concurrently (each in its own thread).
        # On GPU we train sequentially to avoid out-of-memory errors.
        run_parallel = (device == "cpu") and n_archs > 1

        async def _train_one(arch_id: str, arch_idx_: int) -> dict:
            """Instantiate, train, and evaluate a single architecture."""
            arch_start = 0.10 + arch_idx_ / n_archs * 0.85
            arch_end   = 0.10 + (arch_idx_ + 1) / n_archs * 0.85

            self.current_model = arch_id
            ExperimentStore.set_model_status(self.experiment_id, arch_id, "training", 0.0)
            ExperimentStore.append_log(self.experiment_id, f"Starting: {arch_id}")
            self._report_progress(arch_start, f"Instantiating {arch_id}")

            # Arch-scoped callback: maps per-arch epoch fraction [0,1] to global [arch_start, arch_end]
            # Also streams live training_history so the UI loss curves update during training.
            def _arch_cb(
                frac: float, msg: str, hist: list = None,
                _aid: str = arch_id, _as: float = arch_start, _ae: float = arch_end,
            ) -> None:
                live_hist = (
                    [{"epoch": i + 1, "total_loss": float(v)} for i, v in enumerate(hist)]
                    if hist else None
                )
                ExperimentStore.set_model_status(
                    self.experiment_id, _aid, "training", frac,
                    current_loss=float(hist[-1]) if hist else None,
                    loss_history=live_hist,
                )
                self._report_progress(_as + frac * (_ae - _as), msg)

            try:
                model = instantiate_model(arch_id, in_dim, out_dim)
            except Exception as exc:
                logger.warning("Cannot instantiate %s: %s", arch_id, exc)
                ExperimentStore.set_model_status(self.experiment_id, arch_id, "failed")
                ExperimentStore.append_log(self.experiment_id, f"FAILED instantiate: {arch_id} — {exc}")
                return {
                    "arch_id": arch_id, "status": "failed", "error": str(exc),
                    "metrics": {}, "loss_history": [], "y_pred": [], "y_true": [],
                    "X_eval": [], "training_time_s": 0.0,
                }

            t0 = time.time()
            try:
                if getattr(model, "closed_form", False):
                    train_result = await loop.run_in_executor(
                        None,
                        lambda m=model, cb=_arch_cb: train_closed_form_model(
                            m, X_train_matched, y_train_matched, X_val_data, y_val_data, cb,
                        ),
                    )
                elif isinstance(model, nn.Module):
                    train_result = await loop.run_in_executor(
                        None,
                        lambda m=model, cb=_arch_cb: train_torch_model(
                            m,
                            X_train_matched, y_train_matched,
                            X_val_data,      y_val_data,
                            X_col,
                            physics_loss_fn,
                            n_epochs, lr, device,
                            cb,
                        ),
                    )
                else:
                    raise TypeError(f"{type(model).__name__} is not a torch module; the arena can not train it")
            except Exception as exc:
                logger.warning("Training failed for %s: %s", arch_id, exc)
                ExperimentStore.set_model_status(self.experiment_id, arch_id, "failed")
                ExperimentStore.append_log(self.experiment_id, f"FAILED training: {arch_id} — {exc}")
                return {
                    "arch_id": arch_id, "status": "failed", "error": str(exc),
                    "metrics": {}, "loss_history": [], "y_pred": [], "y_true": [],
                    "X_eval": [], "training_time_s": time.time() - t0,
                }

            training_time = time.time() - t0

            # ── Evaluate on HELD-OUT test set (no training leakage) ──────────
            self._report_progress(
                arch_start + 0.9 * (arch_end - arch_start), f"Evaluating {arch_id}"
            )
            ExperimentStore.set_model_status(self.experiment_id, arch_id, "evaluating", 0.92)
            inference_time_ms: float | None = None
            try:
                y_pred_eval, inference_time_ms = await loop.run_in_executor(
                    None, lambda m=model: predict(m, X_eval, device))
                metrics = field_metrics(y_pred_eval, y_eval)
            except Exception as exc:
                # A model that trained but can't be evaluated has no result to rank —
                # report it as failed rather than "completed" with inf metrics.
                logger.warning("Evaluation failed for %s: %s", arch_id, exc)
                ExperimentStore.set_model_status(self.experiment_id, arch_id, "failed")
                ExperimentStore.append_log(self.experiment_id, f"FAILED evaluation: {arch_id} — {exc}")
                return {
                    "arch_id": arch_id, "status": "failed", "error": f"Evaluation failed: {exc}",
                    "metrics": {}, "loss_history": train_result.get("loss_history", []),
                    "y_pred": [], "y_true": [], "X_eval": [], "training_time_s": training_time,
                }

            # ── Inverse solver (if requested) ────────────────────────────────
            inverse_result: dict | None = None
            if problem_type == "inverse" and inverse_cfg and isinstance(model, nn.Module):
                ExperimentStore.set_model_status(self.experiment_id, arch_id, "inverting", 0.96)
                self._report_progress(
                    arch_start + 0.95 * (arch_end - arch_start), f"Inverse solver {arch_id}"
                )
                try:
                    inverse_result = await loop.run_in_executor(
                        None,
                        lambda m=model: run_inverse_solver(
                            m, problem_id, inverse_cfg,
                            coords, solution, device,
                        ),
                    )
                    if inverse_result and "identified_params" in inverse_result:
                        ExperimentStore.append_log(
                            self.experiment_id,
                            f"Inverse params [{arch_id}]: {inverse_result['identified_params']}",
                        )
                        # Attach ground-truth values (problem defaults or custom params)
                        prob_def = PROBLEMS_CATALOG.get(problem_id)
                        param_defaults = (
                            {p.name: p.default for p in prob_def.parameters} if prob_def else {}
                        )
                        inverse_result["true_params"] = {
                            k: problem_params.get(k, param_defaults.get(k))
                            for k in inverse_result["identified_params"]
                        }
                except Exception as exc:
                    logger.warning("Inverse solver failed for %s: %s", arch_id, exc)
                    inverse_result = {"error": str(exc)}

            loss_h = [float(v) for v in train_result.get("loss_history", [])]
            ExperimentStore.set_model_status(
                self.experiment_id, arch_id, "completed", 1.0,
                current_loss=float(loss_h[-1]) if loss_h else None,
                loss_history=[{"epoch": i + 1, "total_loss": v} for i, v in enumerate(loss_h)],
            )
            ExperimentStore.append_log(
                self.experiment_id,
                f"Completed: {arch_id}  loss={loss_h[-1]:.3e}  "
                f"val_L2={metrics.get('l2_relative', float('inf')):.3e}"
                if loss_h else f"Completed: {arch_id}",
            )
            self._report_progress(arch_end, f"Completed {arch_id}")

            # Persist trained weights for later download
            if isinstance(model, nn.Module):
                _models_dir = os.path.join(MODELS_DIR, self.experiment_id)
                os.makedirs(_models_dir, exist_ok=True)
                try:
                    torch.save(model.state_dict(), os.path.join(_models_dir, f"{arch_id}.pt"))
                except Exception as _save_exc:
                    logger.warning("Could not save weights for %s: %s", arch_id, _save_exc)

            validation = prediction_quality(
                y_pred_eval if y_pred_eval is not None else np.zeros_like(y_eval),
                y_eval,
                metrics,
            )

            return {
                "arch_id": arch_id,
                "status": "completed",
                "error": None,
                "metrics": metrics,
                "loss_history": loss_h,
                "y_pred": y_pred_eval.tolist() if y_pred_eval is not None else [],
                "y_true": y_eval.tolist(),
                "X_eval": X_eval.tolist(),
                "training_time_s": float(training_time),
                "n_epochs": n_epochs,
                "stopped_at_epoch": train_result.get("stopped_at_epoch", n_epochs),
                "best_val_loss": train_result.get("best_val_loss"),
                # True only when the PDE residual was really evaluated during training (a loss that raises is skipped)
                "physics_loss_enabled": train_result.get("n_pde_evaluations", 0) > 0,
                "inverse_result": inverse_result,
                "n_pde_evaluations": train_result.get("n_pde_evaluations", 0),
                "time_to_convergence_s": train_result.get("time_to_convergence_s"),
                "convergence_epoch": train_result.get("convergence_epoch"),
                "peak_gpu_memory_mb": train_result.get("peak_gpu_memory_mb"),
                "peak_cpu_memory_mb": train_result.get("peak_cpu_memory_mb"),
                "inference_time_ms": inference_time_ms,
                "converged": metrics.get("l2_relative", 1.0) < 0.1,
                "quality_score": validation["quality_score"],
                "quality_level": validation["quality_level"],
                "quality_flags": validation["quality_flags"],
                "pred_std_ratio": validation["pred_std_ratio"],
            }

        if run_parallel:
            logger.info("Training %d architectures IN PARALLEL on CPU", n_archs)
            raw_results = await asyncio.gather(
                *[_train_one(aid, i) for i, aid in enumerate(arch_ids)],
                return_exceptions=True,
            )
            model_results = [
                r if isinstance(r, dict) else {
                    "arch_id": arch_ids[i], "status": "failed",
                    "error": str(r), "metrics": {}, "loss_history": [],
                    "y_pred": [], "y_true": [], "X_eval": [], "training_time_s": 0.0,
                }
                for i, r in enumerate(raw_results)
            ]
        else:
            logger.info(
                "Training %d architectures SEQUENTIALLY on %s", n_archs, device.upper()
            )
            model_results = []
            for i, aid in enumerate(arch_ids):
                result = await _train_one(aid, i)
                model_results.append(result)
                # Free GPU/CPU memory between architectures to avoid OOM on the next model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Step 4: Rank models by L2 relative error
        ranked = rank_models(model_results)

        winner = ranked[0]["arch_id"] if ranked else None
        summary = {
            "problem_id": problem_id,
            "n_architectures": len(arch_ids),
            "n_completed": sum(1 for r in model_results if r["status"] == "completed"),
            "n_failed": sum(1 for r in model_results if r["status"] == "failed"),
            "best_l2_relative": ranked[0]["metrics"].get("l2_relative", None) if ranked else None,
            "device": device,
            "physics_loss_available": physics_loss_fn is not None,
            "pinneapple_version": _library_version(),
            "loss_description": PhysicsLossGenerator(problem_id, problem_params).get_loss_description(),
            "reference_metadata": ref_data.get("metadata", {}) if ref_data else {},
        }

        self._report_progress(1.0, "Experiment complete")
        return {
            "experiment_id": self.experiment_id,
            "model_results": ranked,
            "summary": summary,
            "winner": winner,
        }

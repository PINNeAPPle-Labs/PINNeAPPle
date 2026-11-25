"""Experiment runner: trains multiple models on the same problem and collects results."""
from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable
import numpy as np


@dataclass
class ModelRunConfig:
    name: str                              # model registry name
    extra_kwargs: dict[str, Any] = field(default_factory=dict)  # forwarded to ModelRegistry.build
    weight_override: dict[str, float] | None = None           # physics loss weights


@dataclass
class ExperimentConfig:
    """Full configuration for one experimental run (one problem, N models)."""
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    problem_name: str = ""
    models: list[ModelRunConfig] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    epochs: int = 2000
    lr: float = 1e-3
    device: str = "cpu"
    batch_size: int = 4096
    seed: int = 42
    grad_clip: float | None = None       # gradient clipping norm (None = disabled)
    # Auto-improvement
    auto_improve: bool = True
    max_retrain_rounds: int = 2
    advisor_priority_threshold: int = 3     # apply suggestions up to this priority level


@dataclass
class ModelResult:
    name: str
    metrics: dict[str, float]
    loss_history: list[float]               # per-epoch total loss
    physics_loss_history: list[float]
    bc_loss_history: list[float]
    train_time_s: float
    n_params: int
    model_state: Any | None = None       # state_dict (serializable)
    error: str | None = None             # non-None if training failed
    retrain_rounds: int = 0                 # how many advisor-driven retrains occurred
    diagnosis: dict[str, Any] | None = None  # last DiagnosticReport summary
    report: dict[str, Any] | None = None     # surrogate_report.build_report output


@dataclass
class ExperimentResult:
    experiment_id: str
    config: ExperimentConfig
    model_results: dict[str, ModelResult]
    completed_at: str = ""
    # one pp.Experiment record per model (issue #29): the same shape a saved library experiment writes, so an app
    # run and a script run are described by the same class
    experiments: list[dict[str, Any]] = field(default_factory=list)

    def leaderboard(self) -> list:
        """Rows sorted by held-out l2_relative, then by unseen-point PDE residual."""
        rows = []
        for name, r in self.model_results.items():
            if r.error:
                continue
            row = {"model": name, **r.metrics, "n_params": r.n_params,
                   "train_time_s": round(r.train_time_s, 2)}
            if r.report:
                row["verdict"] = r.report.get("verdict", {}).get("status")
                row["convergence"] = r.report.get("convergence", {}).get("status")
            rows.append(row)

        def key(row):
            l2 = row.get("l2_relative", float("nan"))
            res = row.get("pde_residual", float("nan"))
            finite = lambda v: isinstance(v, float) and v == v and v != float("inf")
            return (0 if finite(l2) else 1, l2 if finite(l2) else 0.0,
                    res if finite(res) else float("inf"))
        return sorted(rows, key=key)


class ExperimentRunner:
    """Trains all configured models and streams progress via an async generator."""

    def __init__(self, config: ExperimentConfig, data_bundle, problem):
        self.config = config
        self.data = data_bundle
        self.problem = problem

    def manifests(self) -> list[dict[str, Any]]:
        """The configured run as ``pp.Experiment`` records -- one per model, built by the library class (#29).

        The training loop stays here (a model registry, per-epoch events and advisor retrains are not what
        ``pinneapple_physics.Experiment`` abstracts), but the *record* is the library's: ``Experiment.to_dict()``,
        with the problem fingerprint and the seed, so a run started from the app and one started from a script
        can be compared and reloaded with ``pp.Experiment.load``. A custom problem has no preset for the library
        to rebuild, so its record carries no fingerprint and says why.
        """
        from pinneapple_physics.experiment import Experiment as LibraryExperiment

        records: list[dict[str, Any]] = []
        for model_cfg in self.config.models:
            options = {"epochs": self.config.epochs, "lr": self.config.lr,
                       "batch_size": self.config.batch_size, "device": self.config.device,
                       **dict(model_cfg.extra_kwargs)}
            if model_cfg.weight_override:
                options["loss_weights"] = dict(model_cfg.weight_override)
            try:
                record = LibraryExperiment(self.config.problem_name, method=model_cfg.name, options=options,
                                           seed=self.config.seed).to_dict()
            except Exception as exc:  # a custom problem ("__custom__") has no preset to rebuild
                records.append({"name": f"{self.config.problem_name}/{model_cfg.name}/seed{self.config.seed}",
                                "problem": self.config.problem_name, "method": model_cfg.name,
                                "options": options, "seed": self.config.seed, "problem_fingerprint": None,
                                "resumable": False, "unavailable": str(exc) or type(exc).__name__})
                continue
            records.append(record)
        return records

    async def run(self, progress_cb: Callable | None = None) -> ExperimentResult:
        """Run all models sequentially. Calls ``progress_cb(event_dict)`` after each epoch."""
        import asyncio
        results: dict[str, ModelResult] = {}

        for model_cfg in self.config.models:
            if self.config.auto_improve:
                result = await asyncio.to_thread(
                    self._train_one_with_autofix, model_cfg, progress_cb
                )
            else:
                result = await asyncio.to_thread(
                    self._train_one, model_cfg, progress_cb, self.config, 0
                )
            results[model_cfg.name] = result

        from datetime import datetime, timezone
        return ExperimentResult(
            experiment_id=self.config.experiment_id,
            config=self.config,
            model_results=results,
            completed_at=datetime.now(timezone.utc).isoformat(),
            experiments=self.manifests(),
        )

    # ── Auto-improvement loop ─────────────────────────────────────────────────

    def _train_one_with_autofix(
        self,
        model_cfg: ModelRunConfig,
        progress_cb,
    ) -> ModelResult:
        """Train with advisor-driven auto-retraining."""
        from pinneapple_neural.trainer.advisor import TrainingAdvisor

        advisor = TrainingAdvisor()
        current_cfg = self.config
        best_result: ModelResult | None = None
        round_num = 0

        while True:
            result = self._train_one(model_cfg, progress_cb, current_cfg, round_num)
            result.retrain_rounds = round_num

            if best_result is None or _result_is_better(result, best_result):
                best_result = result

            if result.error or round_num >= self.config.max_retrain_rounds:
                break

            # Analyse training quality
            report = advisor.analyse(
                model_name=model_cfg.name,
                loss_history=result.loss_history,
                physics_loss_history=result.physics_loss_history,
                bc_loss_history=result.bc_loss_history,
                metrics=result.metrics,
                current_lr=current_cfg.lr,
                current_epochs=current_cfg.epochs,
                current_batch_size=current_cfg.batch_size,
            )

            # Attach diagnosis to result
            result.diagnosis = {
                "signals": report.signals,
                "suggestions": [
                    {"code": s.code, "priority": s.priority,
                     "description": s.description, "patch": s.config_patch}
                    for s in report.suggestions[:5]
                ],
                "text": report.diagnosis_text,
            }

            if progress_cb:
                progress_cb({
                    "type": "advisor",
                    "model": model_cfg.name,
                    "round": round_num,
                    "signals": report.signals,
                    "top_suggestion": report.suggestions[0].description
                    if report.suggestions else "No issues found.",
                })

            high_priority = [
                s for s in report.suggestions
                if s.priority <= self.config.advisor_priority_threshold
            ]
            if not high_priority:
                break

            # Build improved config from patches
            patch = advisor.apply_suggestions(
                high_priority, _cfg_to_dict(current_cfg),
                max_priority=self.config.advisor_priority_threshold,
            )
            current_cfg = _dict_to_cfg(patch, current_cfg)

            if progress_cb:
                progress_cb({
                    "type": "retrain",
                    "model": model_cfg.name,
                    "round": round_num + 1,
                    "patches_applied": patch.get("_advisor_patches", []),
                    "new_lr": current_cfg.lr,
                    "new_epochs": current_cfg.epochs,
                })

            round_num += 1

        return best_result

    # ── Single training run ───────────────────────────────────────────────────

    def _train_one(
        self,
        model_cfg: ModelRunConfig,
        progress_cb,
        cfg: ExperimentConfig,
        round_label: int,
    ) -> ModelResult:
        """Synchronous training of a single model with a given config.

        Physics comes from :func:`physics_adapter.build_physics` (the real
        ``compile_problem`` batch for presets, SymPy-compiled equations for
        custom problems). Reference data, when present, is split into
        train / calibration / test: only the train split is fitted, the
        calibration split feeds conformal UQ and every accuracy number is
        measured on the test split.
        """
        import torch
        from .physics_adapter import build_physics, sample_interior
        from .surrogate_report import build_report

        name = model_cfg.name
        t0 = time.time()
        try:
            torch.manual_seed(cfg.seed)
            rng = np.random.default_rng(cfg.seed)
            dev = torch.device(cfg.device)

            physics = build_physics(self.problem, weight_override=model_cfg.weight_override)
            if physics is None:
                from .physics_adapter import PhysicsAdapter
                physics = PhysicsAdapter()
                physics.label, physics.pde_kind = self.problem.name, "none"
                physics.bounds = {k: tuple(map(float, v)) for k, v in self.problem.domain_bounds.items()}
                physics.coords = list(physics.bounds)
                physics.fields = list(self.problem.field_names)
                physics.field_ranges = {}
                has_physics = False
            else:
                has_physics = True
            warnings = list(physics.warnings)
            if not has_physics:
                warnings.append("Problem has no physics definition: training on reference data only.")

            coords, fields = physics.coords, physics.fields
            x_col = np.asarray(self.data.x_col, dtype=np.float32)
            if x_col.shape[1] == len(coords) - 1 and coords[-1] == "t":
                lo, hi = physics.bounds["t"]
                x_col = np.concatenate([x_col, rng.uniform(lo, hi, (len(x_col), 1)).astype(np.float32)], 1)
            if x_col.shape[1] != len(coords):
                raise ValueError(f"Collocation points have {x_col.shape[1]} columns but the problem "
                                 f"coordinates are {coords}.")

            split, ref_note = _split_reference(x_col, self.data.u_ref, len(fields), rng)
            if ref_note:
                warnings.append(ref_note)
            if not has_physics and "X_train" not in split:
                raise ValueError("Nothing to train on: the problem has no physics definition "
                                 "and no usable reference data.")

            model = _build_model(name, len(coords), len(fields), model_cfg.extra_kwargs).to(dev)

            n_bnd = max(64, len(self.data.x_bnd) if self.data.x_bnd is not None else 0)
            n_ic = max(64, len(self.data.x_ic) if self.data.x_ic is not None else 0)
            X_eval_np = sample_interior(physics.bounds, coords, min(2048, max(256, len(x_col))),
                                        np.random.default_rng(cfg.seed + 1))
            X_eval = torch.as_tensor(X_eval_np, device=dev)
            if has_physics:
                train_batch = physics.build_batch(x_col, n_bnd, n_ic, cfg.seed, dev)
                eval_batch = physics.build_batch(X_eval_np, n_bnd, n_ic, cfg.seed + 1, dev)
                raw_initial = physics.loss(model, eval_batch)[1]
            else:
                raw_initial = None

            to_t = lambda a: None if a is None else torch.as_tensor(a, dtype=torch.float32, device=dev)
            X_tr, Y_tr = to_t(split.get("X_train")), to_t(split.get("Y_train"))
            w_data = 1.0

            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
            loss_hist, phys_hist, bc_hist = [], [], []
            snap_epochs = set(np.linspace(int(0.8 * cfg.epochs), cfg.epochs - 1, 5).astype(int).tolist())
            snapshots = []
            data_loss_val = None

            for epoch in range(cfg.epochs):
                optimizer.zero_grad()
                if has_physics:
                    total, raw = physics.loss(model, train_batch)
                else:
                    total, raw = torch.zeros((), device=dev), {}
                if X_tr is not None:
                    data_loss = torch.mean((_unwrap(model(X_tr)) - Y_tr) ** 2)
                    total = total + w_data * data_loss
                    data_loss_val = float(data_loss.detach())
                total.backward()
                if cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

                loss_hist.append(float(total.detach()))
                phys_hist.append(raw.get("pde", 0.0))
                bc_hist.append(sum(v for k, v in raw.items() if k != "pde"))
                if epoch in snap_epochs:
                    snapshots.append({k: v.detach().clone() for k, v in model.state_dict().items()})

                if progress_cb and epoch % max(1, cfg.epochs // 50) == 0:
                    progress_cb({
                        "model": name,
                        "round": round_label,
                        "epoch": epoch,
                        "total_epochs": cfg.epochs,
                        "loss": loss_hist[-1],
                        "phys_loss": phys_hist[-1],
                        "bc_loss": bc_hist[-1],
                    })

            # ── Evaluation (never on training points) ─────────────────────
            if has_physics:
                final_raw_train = physics.loss(model, train_batch)[1]
                raw_unseen = physics.loss(model, eval_batch)[1]
            else:
                final_raw_train, raw_unseen = {}, {}
            res_train = float(np.sqrt(final_raw_train["pde"])) if "pde" in final_raw_train else None

            report = build_report(
                model=model, adapter=physics, loss_history=loss_hist,
                final_raw_train=final_raw_train, raw_initial=raw_initial, raw_unseen=raw_unseen,
                res_train=res_train, X_eval=X_eval,
                split={k: (to_t(v) if k.startswith("X") else v) for k, v in split.items()},
                snapshots=snapshots, reference_source=self.data.meta.get("reference_source")
                or ("numerical solver" if self.data.solver_outputs else None),
                w_data=w_data, data_loss=data_loss_val, warnings=warnings,
            )
            metrics = _compute_metrics(report, raw_unseen, physics, self.config.metrics)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            return ModelResult(
                name=name,
                metrics=metrics,
                loss_history=loss_hist,
                physics_loss_history=phys_hist,
                bc_loss_history=bc_hist,
                train_time_s=time.time() - t0,
                n_params=n_params,
                model_state={k: v.cpu().numpy().tolist()
                             for k, v in model.state_dict().items()},
                report=report,
            )

        except Exception as e:
            return ModelResult(
                name=name,
                metrics={},
                loss_history=[],
                physics_loss_history=[],
                bc_loss_history=[],
                train_time_s=time.time() - t0,
                n_params=0,
                error=f"{type(e).__name__}: {e}",
            )


# ── Helpers ───────────────────────────────────────────────────────────────

def _result_is_better(new: ModelResult, old: ModelResult) -> bool:
    """Return True if new result has lower final loss or better l2_relative."""
    if new.error:
        return False
    if old.error:
        return True
    new_l2 = new.metrics.get("l2_relative", float("inf"))
    old_l2 = old.metrics.get("l2_relative", float("inf"))
    import math
    if math.isfinite(new_l2) and math.isfinite(old_l2):
        return new_l2 < old_l2
    new_res = new.metrics.get("pde_residual", float("inf"))
    old_res = old.metrics.get("pde_residual", float("inf"))
    if math.isfinite(new_res) and math.isfinite(old_res):
        return new_res < old_res
    # Fallback to final loss
    new_loss = new.loss_history[-1] if new.loss_history else float("inf")
    old_loss = old.loss_history[-1] if old.loss_history else float("inf")
    return new_loss < old_loss


def _cfg_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    return {
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "batch_size": cfg.batch_size,
        "grad_clip": cfg.grad_clip,
        "device": cfg.device,
    }


def _dict_to_cfg(patch: dict[str, Any], base: ExperimentConfig) -> ExperimentConfig:
    """Return a copy of base ExperimentConfig with patched fields applied."""
    from dataclasses import replace
    allowed = {"epochs", "lr", "batch_size", "grad_clip"}
    kwargs = {k: v for k, v in patch.items() if k in allowed}
    return replace(base, **kwargs)


def _build_model(name: str, in_dim: int, out_dim: int, extra_kwargs: dict):
    from pinneapple_neural.architectures import ModelRegistry
    kwargs = dict(in_dim=in_dim, out_dim=out_dim)
    kwargs.update(extra_kwargs)
    return ModelRegistry.build(name, **kwargs)


def _unwrap(out):
    y = out.y if hasattr(out, "y") else out
    return y[:, None] if y.ndim == 1 else y


def _split_reference(x_col: np.ndarray, u_ref, n_fields: int, rng: np.random.Generator,
                     frac_cal: float = 0.15, frac_test: float = 0.15):
    """Split reference data into train / calibration / test.

    Returns ``(split, note)``; ``split`` is empty when there is no usable
    reference (``note`` then says why, unless there simply is none).
    """
    if u_ref is None:
        return {}, None
    Y = np.asarray(u_ref, dtype=np.float32)
    if Y.ndim == 1:
        Y = Y[:, None]
    if len(Y) != len(x_col):
        return {}, (f"Reference data ignored: {len(Y)} values for {len(x_col)} points.")
    if Y.shape[1] != n_fields:
        return {}, (f"Reference data ignored: it has {Y.shape[1]} column(s) but the model predicts "
                    f"{n_fields} field(s), so columns cannot be matched to fields.")
    if len(Y) < 20:
        return {}, "Reference data ignored: fewer than 20 points, too few to hold out a test set."
    idx = rng.permutation(len(Y))
    n_te = max(1, int(round(frac_test * len(Y))))
    n_cal = max(1, int(round(frac_cal * len(Y))))
    te, cal, tr = idx[:n_te], idx[n_te:n_te + n_cal], idx[n_te + n_cal:]
    return {
        "X_train": x_col[tr], "Y_train": Y[tr],
        "X_cal": x_col[cal], "Y_cal": Y[cal],
        "X_test": x_col[te], "Y_test": Y[te],
    }, None


def _compute_metrics(report, raw_unseen, physics, metric_names) -> dict[str, float]:
    """Leaderboard numbers, all measured on held-out data / unseen points."""
    nan = float("nan")
    err = report.get("error", {})
    metrics: dict[str, float] = {}
    rel = err.get("rel_l2_pct") if err.get("available") else None
    pf = err.get("per_field", {}) if err.get("available") else {}

    def mean_of(key):
        vals = [v[key] for v in pf.values() if v.get(key) is not None]
        return float(np.mean(vals)) if vals else nan

    candidates = {
        "l2_relative": rel / 100 if rel is not None else nan,
        "error_pct": rel if rel is not None else nan,
        "mse": (float(np.mean([v["rmse"] ** 2 for v in pf.values() if v.get("rmse") is not None]))
                if pf else nan),
        "mae": mean_of("mae"),
        "max_error": max((v["max_abs_error"] for v in pf.values()
                          if v.get("max_abs_error") is not None), default=nan),
        "r2": err.get("r2") if err.get("r2") is not None else nan,
        "pde_residual": float(np.sqrt(raw_unseen["pde"])) if "pde" in raw_unseen else nan,
        "bc_residual": (float(np.mean([np.sqrt(v) for k, v in raw_unseen.items() if k != "pde"]))
                        if any(k != "pde" for k in raw_unseen) else nan),
    }
    wanted = set(metric_names or candidates) | {"l2_relative", "error_pct", "pde_residual"}
    for k, v in candidates.items():
        if k in wanted:
            metrics[k] = float(v) if v is not None else nan
    return metrics

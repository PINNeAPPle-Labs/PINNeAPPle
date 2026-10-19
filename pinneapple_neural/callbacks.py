"""Physics checks during training in PyTorch Lightning, PhysicsNeMo and Hugging Face (#704).

Most models are trained outside PINNeAPPle and validated once, at the end. These callbacks run the checks of
``pp.validate`` on a fixed validation batch every N steps, log them to the framework's logger, warn (or stop) when a
physics metric worsens for K consecutive checks while the training loss improves, and write the validation report and
trust card at the end of training.

The core, :class:`PhysicsMonitor`, is framework-free; the adapters are thin:

* ``PhysicsValidationCallback(monitor)`` -- a ``lightning.pytorch.Callback`` (or ``pytorch_lightning``): logs to
  every logger of the Trainer (TensorBoard, W&B, MLflow, CSV) at each check, puts the metrics in
  ``trainer.callback_metrics`` (for checkpointing or early stopping on them) and sets ``trainer.should_stop``.
* ``PhysicsTrainerCallback(monitor)`` -- a ``transformers.TrainerCallback``: logs through ``trainer.log`` (its
  ``report_to`` integrations) when attached with ``.attach(trainer)``, else into ``state.log_history``; sets
  ``control.should_training_stop``.
* :class:`PhysicsNeMoHook` -- for a PhysicsNeMo training loop (PhysicsNeMo has no trainer class): call
  ``hook(model, step, loss, log=launch_logger)``; metrics go to a ``physicsnemo.utils.logging.LaunchLogger`` (which
  forwards to W&B / MLflow when initialised) or any callable.

None of the frameworks is a dependency: the Lightning and Hugging Face classes are built on first access.

>>> mon = PhysicsMonitor("darcy_2d", reference=(a_val, u_val), every=50, patience=3, action="stop",
...                      report_path="runs/trust_card.json")
>>> trainer = lightning.Trainer(callbacks=[PhysicsValidationCallback(mon)])
"""
from __future__ import annotations

import json
import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["PhysicsMonitor", "PhysicsWarning", "PhysicsNeMoHook",  # noqa: F822  (the last two: module __getattr__)
           "PhysicsValidationCallback", "PhysicsTrainerCallback"]


class PhysicsWarning(UserWarning):
    """A physics metric worsened for several consecutive checks while the training loss improved."""


def _float(x) -> float | None:
    if x is None:
        return None
    try:
        v = float(x.detach() if hasattr(x, "detach") else x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _better(cid: str) -> str:
    """``"lower"`` or ``"higher"``: the direction of the check's value (``validation.schema``)."""
    from pinneapple_analysis.validation.schema import spec_of
    try:
        return spec_of(cid).better
    except KeyError:
        return "lower"


class PhysicsMonitor:
    """The checks of ``pp.validate(model, problem, data, reference, query, level)`` as training metrics.

    ``every``: run the checks every this many steps. ``watch``: check ids whose value is tracked for the early warning
    (default: every check with a numeric value; the direction that counts as better is the check's ``better`` in
    ``validation.schema``: lower |value| for errors, higher for coverages, spread ratios and crossing times).
    ``patience``: number of consecutive checks in which a watched metric worsened while the loss improved before the
    warning fires (a step is checked once, also with gradient accumulation). ``action``: ``"warn"`` (a
    :class:`PhysicsWarning`) or ``"stop"`` (also sets ``should_stop``). ``model_fn``: maps the framework's module to
    the model to validate (e.g. ``lambda m: m.net`` for a LightningModule that wraps the network). ``report_path``:
    where :meth:`finish` writes the report (JSON) and the trust card (``*.trust.json``).

    Metrics are named ``physics/<check id>`` plus ``physics/n_fail`` and ``physics/n_warn``."""

    def __init__(self, problem, data=None, *, reference=None, query=None, level: str = "quick", every: int = 100,
                 patience: int = 3, action: str = "warn", watch: Sequence[str] | None = None,
                 model_fn: Callable[[Any], Any] | None = None, tolerances: Mapping[str, float] | None = None,
                 report_path: str | Path | None = None, seed: int = 0):
        if action not in ("warn", "stop"):
            raise ValueError("action must be 'warn' or 'stop'")
        self.problem, self.data, self.reference, self.query = problem, data, reference, query
        self.level, self.every, self.patience, self.action = level, max(int(every), 1), int(patience), action
        self.watch = tuple(watch) if watch is not None else None
        self.model_fn = model_fn or (lambda m: m)
        self.tolerances, self.report_path, self.seed = tolerances, report_path, seed
        self.history: list[dict[str, Any]] = []
        self.alerts: list[dict[str, Any]] = []
        self.should_stop = False
        self._streak: dict[str, int] = {}
        self.report = None

    # ------------------------------------------------------------------ evaluation
    def _validate(self, module):
        from pinneapple_analysis.validation.validate import validate
        model = self.model_fn(module)
        was = getattr(model, "training", None)
        try:
            return validate(model, self.problem, data=self.data, reference=self.reference, query=self.query,
                            level=self.level, tolerances=self.tolerances, seed=self.seed)
        finally:
            if was:
                model.train()

    def evaluate(self, module) -> dict[str, float]:
        """The physics metrics of ``module`` now (no step bookkeeping)."""
        rep = self._validate(module)
        out = {f"physics/{c.id}": float(c.value) for c in rep.checks
               if isinstance(c.value, (int, float)) and not isinstance(c.value, bool) and math.isfinite(c.value)}
        out["physics/n_fail"] = float(sum(c.status == "fail" for c in rep.checks))
        out["physics/n_warn"] = float(sum(c.status == "warn" for c in rep.checks))
        return out

    def due(self, step: int) -> bool:
        return step % self.every == 0

    def step(self, module, step: int, loss=None) -> dict[str, float] | None:
        """Run the checks if ``step`` is due; returns the metrics (else None) and applies the early-warning rule."""
        if not self.due(step) or (self.history and self.history[-1]["step"] == step):
            return None                                 # not due, or already checked (gradient accumulation)
        metrics = self.evaluate(module)
        loss = _float(loss)
        prev = self.history[-1] if self.history else None
        self.history.append({"step": step, "loss": loss, "metrics": metrics})
        if prev is not None:
            improving = loss is not None and prev["loss"] is not None and loss < prev["loss"]
            for key, v in metrics.items():
                cid = key.removeprefix("physics/")
                if cid in ("n_fail", "n_warn") or (self.watch is not None and cid not in self.watch):
                    continue
                old = prev["metrics"].get(key)
                if _better(cid) == "higher":            # a coverage, a spread ratio, a crossing time
                    worse = old is not None and v < old * (1 - 1e-9) - 1e-12
                else:                                   # an error: |value| grows
                    worse = old is not None and abs(v) > abs(old) * (1 + 1e-9) + 1e-12
                self._streak[cid] = self._streak.get(cid, 0) + 1 if (worse and improving) else 0
                if self._streak[cid] == self.patience:
                    self._alert(step, cid, metrics[key], loss)
        return metrics

    def _alert(self, step: int, cid: str, value: float, loss: float | None) -> None:
        msg = (f"step {step}: physics metric {cid} worsened for {self.patience} consecutive checks (now {value:.3g}) "
               f"while the training loss improved (now {loss:.3g})")
        self.alerts.append({"step": step, "check": cid, "value": value, "loss": loss, "message": msg})
        warnings.warn(msg, PhysicsWarning, stacklevel=3)
        if self.action == "stop":
            self.should_stop = True

    # ------------------------------------------------------------------ end of training
    def finish(self, module, path: str | Path | None = None):
        """The full validation report at the end of training; written with its trust card when a path is set."""
        self.report = self._validate(module)
        path = path or self.report_path
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.report.to_json(path)
            card = self.report.to_trust_report()
            card_dict = card.to_dict() if hasattr(card, "to_dict") else dict(vars(card))
            card_dict["training_alerts"] = self.alerts
            path.with_suffix(".trust.json").write_text(json.dumps(card_dict, indent=2, default=str))
        return self.report


# ---------------------------------------------------------------------------------------------- PhysicsNeMo
class PhysicsNeMoHook:
    """For a PhysicsNeMo training loop: ``hook(model, step, loss, log=...)`` after each optimiser step.

    ``log``: a ``physicsnemo.utils.logging.LaunchLogger`` (its ``log_epoch`` forwards to W&B / MLflow when
    initialised), any callable taking ``(metrics, step)``, or None (the metrics stay in ``monitor.history``).
    Returns True when training should stop."""

    def __init__(self, monitor: PhysicsMonitor):
        self.monitor = monitor

    def __call__(self, model, step: int, loss=None, log=None) -> bool:
        metrics = self.monitor.step(model, step, loss)
        if metrics is not None and log is not None:
            if hasattr(log, "log_epoch"):
                log.log_epoch(metrics)
            else:
                log(metrics, step)
        return self.monitor.should_stop

    def finish(self, model, path=None):
        return self.monitor.finish(model, path)


# ---------------------------------------------------------------------------------------------- Lightning
def _lightning_callback_class():
    try:
        from lightning.pytorch import Callback
    except ImportError:
        try:
            from pytorch_lightning import Callback
        except ImportError as exc:
            raise ImportError("PhysicsValidationCallback needs Lightning: pip install lightning") from exc

    class PhysicsValidationCallback(Callback):
        """Lightning callback: the monitor's checks every ``monitor.every`` optimiser steps (``trainer.global_step``),
        logged to ``trainer.loggers`` and ``callback_metrics``; ``trainer.should_stop`` on an alert with ``action="stop"``; the report
        and trust card at the end of training."""

        def __init__(self, monitor: PhysicsMonitor):
            super().__init__()
            self.monitor = monitor

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            loss = outputs.get("loss") if isinstance(outputs, Mapping) else outputs
            metrics = self.monitor.step(pl_module, trainer.global_step, loss)
            if metrics is not None:                                  # straight to the loggers: log_dict alone
                for lg in trainer.loggers:                               # is throttled by log_every_n_steps
                    lg.log_metrics(metrics, step=trainer.global_step)
                pl_module.log_dict(metrics, on_step=True, on_epoch=False, logger=False)   # callback_metrics
            if self.monitor.should_stop:
                trainer.should_stop = True

        def on_train_end(self, trainer, pl_module):
            self.monitor.finish(pl_module)

    return PhysicsValidationCallback


# ---------------------------------------------------------------------------------------------- Hugging Face
def _hf_callback_class():
    try:
        from transformers import TrainerCallback
    except ImportError as exc:
        raise ImportError("PhysicsTrainerCallback needs transformers: pip install transformers") from exc

    class PhysicsTrainerCallback(TrainerCallback):
        """Hugging Face ``TrainerCallback``: the monitor's checks every ``monitor.every`` steps
        (``state.global_step``). With ``.attach(trainer)`` the metrics go through ``trainer.log`` (and so to the
        ``report_to`` integrations); otherwise they are appended to ``state.log_history``. The loss is the latest
        logged ``loss`` (set ``logging_steps`` no larger than ``monitor.every``)."""

        def __init__(self, monitor: PhysicsMonitor):
            self.monitor = monitor
            self.trainer = None

        def attach(self, trainer):
            self.trainer = trainer
            trainer.add_callback(self)
            return self

        def on_step_end(self, args, state, control, model=None, **kw):
            loss = next((h["loss"] for h in reversed(state.log_history) if "loss" in h), None)
            metrics = self.monitor.step(model, state.global_step, loss)
            if metrics is not None:
                if self.trainer is not None:
                    self.trainer.log(dict(metrics))
                else:
                    state.log_history.append({**metrics, "step": state.global_step})
            if self.monitor.should_stop:
                control.should_training_stop = True
            return control

        def on_train_end(self, args, state, control, model=None, **kw):
            self.monitor.finish(model)

    return PhysicsTrainerCallback


_FACTORIES = {"PhysicsValidationCallback": _lightning_callback_class, "PhysicsTrainerCallback": _hf_callback_class}
_BUILT: dict[str, type] = {}


def __getattr__(name: str):
    if name in _FACTORIES:
        if name not in _BUILT:
            _BUILT[name] = _FACTORIES[name]()
        return _BUILT[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

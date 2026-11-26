"""The figures a run leaves next to its artefacts (``visualization`` block of the configuration).

Small and optional by design: if matplotlib is not installed the pipeline records that the figures were not produced
and carries on. Nothing here decides anything about the run — the gates do.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["write_figures"]


def write_figures(state, out_dir: Path) -> dict[str, Any]:
    """Write the requested plots; returns ``{"figures": [...], "skipped": [...]}``."""
    cfg = state.cfg
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    skipped: list[str] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:                                          # pragma: no cover - matplotlib is optional
        return {"figures": [], "skipped": ["matplotlib is not installed"]}
    dpi = int(cfg.visualization.dpi)
    for plot in cfg.visualization.plots:
        try:
            path = _PLOTS.get(plot)
            if path is None:
                skipped.append(f"unknown plot {plot!r}")
                continue
            fig = path(state, plt)
            if fig is None:
                skipped.append(f"{plot}: nothing to plot in this run")
                continue
            target = out_dir / f"{plot}.png"
            fig.savefig(target, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            written.append(str(target))
        except Exception as exc:                                 # a figure must never fail a run
            skipped.append(f"{plot}: {type(exc).__name__}: {exc}")
    return {"figures": written, "skipped": skipped}


# ------------------------------------------------------------------------------------------------------- plots
def _training_loss(state, plt):
    history = (state.metrics.get("training") or {}).get("history")
    if not history:
        return None
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.semilogy(history, lw=1.2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("total loss")
    ax.set_title(f"{state.cfg.name}: training loss")
    ax.grid(alpha=0.3)
    return fig


def _parity(state, plt):
    """Prediction against the reference the model never trained on."""
    study = state.model if hasattr(getattr(state.model, "splits", None), "test") else None
    if study is not None and getattr(study, "model", None) is not None:
        idx = np.asarray(study.splits.test)
        data = study.data
        truth = np.asarray(data.Y)[idx].reshape(len(idx), -1)
        guess = np.asarray(study.model.predict_array(np.asarray(data.X)[idx])).reshape(truth.shape)
        n = min(truth.shape[1], 3)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.6), squeeze=False)
        for i, ax in enumerate(axes[0]):
            ax.plot(truth[:, i], guess[:, i], "o", ms=4, alpha=0.8)
            lo, hi = float(np.min(truth[:, i])), float(np.max(truth[:, i]))
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
            ax.set_xlabel("reference solve")
            ax.set_ylabel("surrogate")
            ax.set_title(data.output_names[i] if i < len(data.output_names) else f"output {i}", fontsize=9)
            ax.grid(alpha=0.3)
        fig.suptitle(f"{state.cfg.name}: held-out cases", fontsize=10)
        return fig
    if state.bundle is None or state.model_object is None:
        return None
    reference = (state.reference or {}).get("validation")
    if reference is None:
        return None
    points_path = state.metrics.get("points")
    if not points_path:
        return None
    points = np.load(points_path)
    truth = np.asarray(reference.predict(points), dtype=float)
    guess = np.asarray(state.model_object.predict(points), dtype=float).reshape(truth.shape)
    n_fields = truth.shape[1]
    fig, axes = plt.subplots(1, min(n_fields, 3), figsize=(4 * min(n_fields, 3), 3.6), squeeze=False)
    for i, ax in enumerate(axes[0]):
        ax.plot(truth[:, i], guess[:, i], ".", ms=3, alpha=0.6)
        lo, hi = float(np.min(truth[:, i])), float(np.max(truth[:, i]))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel("reference")
        ax.set_ylabel("model")
        ax.set_title(f"field {i}", fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{state.cfg.name}: prediction vs reference", fontsize=10)
    return fig


def _error_map(state, plt):
    """Where the error is: per held-out case for a parametric surrogate, over the domain for a field model."""
    study = state.model if hasattr(getattr(state.model, "splits", None), "test") else None
    if study is not None and getattr(study, "model", None) is not None:
        idx = np.asarray(study.splits.test)
        data = study.data
        X = np.asarray(data.X)[idx]
        truth = np.asarray(data.Y)[idx]
        guess = np.asarray(study.model.predict_array(X))
        err = np.abs(guess.reshape(truth.shape) - truth).mean(axis=tuple(range(1, truth.ndim)))
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        if X.shape[1] == 1:
            order = np.argsort(X[:, 0])
            ax.plot(X[order, 0], err[order], "-o", ms=4)
            ax.set_xlabel(data.input_names[0])
        else:
            ax.scatter(X[:, 0], X[:, 1], c=err, cmap="viridis")
            ax.set_xlabel(data.input_names[0])
            ax.set_ylabel(data.input_names[1])
        ax.set_ylabel("mean |error| on held-out cases")
        ax.set_title(f"{state.cfg.name}: error on data the model never saw", fontsize=10)
        ax.grid(alpha=0.3)
        return fig
    points, reference = state.points, (state.reference or {}).get("validation")
    if points is None or reference is None or state.model_object is None or points.shape[1] > 2:
        return None
    truth = np.asarray(reference.predict(points), dtype=float)
    guess = np.asarray(state.model_object.predict(points), dtype=float).reshape(truth.shape)
    err = np.abs(guess - truth).mean(axis=1)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    if points.shape[1] == 1:
        ax.plot(points[:, 0], err, "-o", ms=3)
        ax.set_xlabel("coordinate")
    else:
        ax.tripcolor(points[:, 0], points[:, 1], err, shading="gouraud")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    ax.set_ylabel("mean |error|")
    ax.set_title(f"{state.cfg.name}: error over the domain", fontsize=10)
    return fig


_PLOTS = {"training_loss": _training_loss, "parity": _parity, "error_map": _error_map}

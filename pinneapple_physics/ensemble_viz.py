"""Animated view of an adaptive physics ensemble: prediction against the reference field, case by case, with the model
the ensemble chose and how its weights moved.

``animate_ensemble(run, references, "ensemble.gif")`` takes an ``EnsembleRun`` made with ``keep_predictions=True``.
Each case is shown over a few frames: for space-time fields ``(n_t, n_x)`` the frames step through time, so the waves
move; for 1-D fields ``(n_x,)`` the case is held. Panels:

* the field: reference (black), the ensemble's prediction in the colour of the chosen model with its interval, and
  every expert as a faint line;
* the current weights, interpolated between cases so the bars slide;
* the timeline: weights over all cases (revealed up to the current case), the chosen model as a coloured strip, and
  the ensemble's relative L2 error on a log axis next to the best single model on each case.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["animate_ensemble"]


def animate_ensemble(run: Any, references: Sequence[np.ndarray], path: str | Path = "ensemble.gif", *,
                     x: np.ndarray | None = None, times: np.ndarray | None = None, frames_per_case: int = 4,
                     fps: int = 10, cases: Sequence[int] | None = None,
                     regimes: Sequence[tuple[int, str]] | None = None, title: str = "Adaptive physics ensemble",
                     dpi: int = 80, colors: Sequence[str] | None = None) -> Path:
    """Write a GIF of the ensemble run. ``cases``: subset of case indices to animate (default: all). ``regimes``:
    ``[(first_case, label), ...]`` drawn as dashed lines on the timeline. Returns the path written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    if run.predictions is None:
        raise ValueError("run the ensemble with keep_predictions=True")
    names = list(run.names)
    n_cases, n_exp = len(run.predictions), len(names)
    cases = list(range(n_cases)) if cases is None else [int(c) for c in cases]
    palette = list(colors) if colors is not None else [plt.get_cmap("tab10")(i % 10) for i in range(n_exp)]
    color = dict(zip(names, palette, strict=False))
    refs = [np.asarray(r, dtype=float) for r in references]
    spacetime = refs[0].ndim == 2
    nx = refs[0].shape[-1]
    nt = refs[0].shape[0] if spacetime else 1
    x = np.arange(nx) if x is None else np.asarray(x)
    # times shown for each case (t = 0 is the given initial condition, so the frames start after it)
    t_idx = np.unique(np.linspace(0, nt - 1, max(1, frames_per_case) + 1)[1:].round().astype(int)) if spacetime else [0]
    frames = [(k, j, s) for k in cases for s, j in enumerate(t_idx)]

    lo = min(float(np.min(r)) for r in refs)
    hi = max(float(np.max(r)) for r in refs)
    pad = 0.15 * (hi - lo + 1e-12)

    fig = plt.figure(figsize=(11, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.45, 1], hspace=0.38, wspace=0.32)
    ax_f = fig.add_subplot(gs[0, :2])
    ax_w = fig.add_subplot(gs[0, 2])
    ax_t = fig.add_subplot(gs[1, :])
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # field panel
    ax_f.set(xlim=(x[0], x[-1]), ylim=(lo - pad, hi + pad), xlabel="x", ylabel="u")
    expert_lines = {n: ax_f.plot([], [], lw=1, alpha=0.35, color=color[n])[0] for n in names}
    band = [ax_f.fill_between(x, np.zeros(nx), np.zeros(nx), alpha=0.18, color="grey")]
    (ref_line,) = ax_f.plot([], [], color="black", lw=2.6, label="reference")
    (ens_line,) = ax_f.plot([], [], lw=2.2, ls="--", label="ensemble")
    ax_f.legend(loc="upper right", fontsize=8)
    head = ax_f.text(0.01, 0.97, "", transform=ax_f.transAxes, va="top", fontsize=10,
                     bbox={"boxstyle": "round", "fc": "white", "ec": "grey"})

    # weights panel
    bars = ax_w.barh(range(n_exp), np.zeros(n_exp), color=[color[n] for n in names])
    ax_w.set(xlim=(0, 1), yticks=range(n_exp), title="weights")
    ax_w.set_yticklabels(names, fontsize=8)
    ax_w.invert_yaxis()

    # timeline panel: everything drawn once, then a cover hides the cases not reached yet
    idx = np.arange(n_cases)
    ax_t.stackplot(idx, run.weights.T, colors=[color[n] for n in names], alpha=0.85)
    for k, a in enumerate(run.active):
        ax_t.axvspan(k - 0.5, k + 0.5, ymin=1.0, ymax=1.06, color=color.get(a, "grey"), clip_on=False)
    ax_t.set(xlim=(-0.5, n_cases - 0.5), ylim=(0, 1), xlabel="case (ensemble predicts with what it learned before)",
             ylabel="weight")
    ax_e = ax_t.twinx()
    best = np.nanmin(run.errors, axis=1)
    ax_e.semilogy(idx, run.ensemble_errors, color="black", lw=1.6, label="ensemble error")
    ax_e.semilogy(idx, best, color="white", lw=1, ls=":", label="best model on the case")
    ax_e.set_ylabel("relative L2 error")
    ax_e.legend(loc="upper right", fontsize=7, facecolor="lightgrey").set_zorder(10)
    for first, label in regimes or []:
        ax_t.axvline(first - 0.5, color="black", ls="--", lw=0.8)
        ax_t.text(first, 0.97, label, fontsize=7, va="top")
    cover = ax_e.axvspan(-0.5, n_cases - 0.5, ymax=1.07, color="white", alpha=1.0, zorder=5, clip_on=False)
    (cursor,) = ax_t.plot([0, 0], [0, 1], color="black", lw=1.2, zorder=6)

    def draw(i):
        k, j, s = frames[i]
        p, ref = run.predictions[k], refs[k]
        row = (lambda a: np.asarray(a)[j]) if spacetime else (lambda a: np.asarray(a))
        ref_line.set_data(x, row(ref))
        ens_line.set_data(x, row(p["prediction"]))
        ens_line.set_color(color.get(p["active"], "C3"))
        for n in names:
            f = p["expert_predictions"].get(n)
            expert_lines[n].set_data(x, row(f)) if f is not None else expert_lines[n].set_data([], [])
        band[0].remove()
        band[0] = ax_f.fill_between(x, row(p["lower"]), row(p["upper"]), alpha=0.18,
                                    color=color.get(p["active"], "grey"), lw=0)
        when = f" · t = {times[j]:.2f}" if (spacetime and times is not None) else ""
        head.set_text(f"case {k + 1}/{n_cases}{when}\nchosen: {p['active']}   error {run.ensemble_errors[k]:.2%}")
        head.get_bbox_patch().set_edgecolor(color.get(p["active"], "grey"))
        frac = (s + 1) / len(t_idx)
        prev = run.weights[k - 1] if k > 0 else run.weights[k]
        w = prev + (run.weights[k] - prev) * frac
        for b, v in zip(bars, w, strict=True):
            b.set_width(v)
        cover.set_x(k + 0.5)
        cover.set_width(n_cases - 1 - k)
        cursor.set_xdata([k, k])
        return []

    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    path = Path(path)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path

"""The four figures a fitted design space can show, drawn with the repository's own plotting.

A family that has finished has ``model/view.json`` next to its card: the loss curve, the design space with the split,
the bodies the samples came from, and — per sample — the field the reference produced against the field the model
predicted. This module turns that file into PNGs with :mod:`pinneapple_tools.visualization`, so a figure the campaign
shows and a figure a report carries are drawn by the same theme.

    from benchmarks.lpm_figures import ensure
    path = ensure(Path("~/lpm-campaign"), "pp_fin", "loss")     # builds it only if the view is newer
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

#: what a family can show. ``loss`` and ``design`` come from the fit, ``geometry`` and ``prediction`` from its samples.
KINDS = ("loss", "design", "geometry", "prediction")

_LOCK = threading.Lock()


def _style() -> None:
    """The house theme, on a backend that needs no display: this runs inside the campaign's server."""
    import matplotlib

    matplotlib.use("Agg")                           # before pyplot is imported anywhere below
    from pinneapple_tools.visualization import use_theme

    use_theme(kind="report", dark=True)


def _source(view: dict[str, Any]) -> str:
    reference = (view.get("reference") or {}).get("name") or "the family's reference"
    return f"reference: {reference} · model: bagged LPM, mean of the heads · {view['family']}"


def _title(view: dict[str, Any], kind: str) -> str:
    return {
        "loss": f"training loss — {view['family']}",
        "design": f"design space and split — {view['family']}",
        "geometry": f"the bodies behind the samples — {view['family']}",
        "prediction": f"reference against prediction — {view['family']}",
    }[kind]


# --------------------------------------------------------------------------------------------------- the four figures


def loss_figure(view: dict[str, Any]):
    """The loss curve: the one figure that says whether the fit converged or stopped early."""
    from pinneapple_tools.visualization import PALETTE, make_figure

    fig, ax = make_figure(figsize=(5.4, 2.8))
    ax.plot(view["loss"]["epoch"], view["loss"]["loss"], color=PALETTE[0], lw=1.4)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.set_title(_title(view, "loss"), fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    first, last = view["loss"]["loss"][0], view["loss"]["loss"][-1]
    ax.annotate(f"{first:.3g} → {last:.3g}", xy=(0.98, 0.92), xycoords="axes fraction", ha="right", fontsize=8)
    fig.tight_layout()
    return fig


def design_figure(view: dict[str, Any]):
    """The design space on its two widest parameters, with the held-out bodies drawn apart from the training ones.

    The split here is the pipeline's, which is by geometry: a held-out dot is a body the model never saw, at operating
    points it never saw.
    """
    from pinneapple_tools.visualization import PALETTE, make_figure

    ax_name, ay_name = (view["axes"] + ["", ""])[:2]
    fig, ax = make_figure(figsize=(4.8, 3.6))
    for held, colour, name in ((False, PALETTE[0], "training"), (True, PALETTE[1], "held out")):
        rows = [d for d in view["designs"] if d["held_out"] is held and ax_name in d["values"]]
        ax.scatter([d["values"][ax_name] for d in rows], [d["values"][ay_name] for d in rows],
                   s=12, color=colour, label=f"{name} ({len(rows)})", alpha=0.85, linewidths=0)
    ax.set_xlabel(ax_name or "parameter 1")
    ax.set_ylabel(ay_name or "parameter 2")
    ax.set_title(_title(view, "design"), fontsize=10)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def _two_widest(view: dict[str, Any], points):
    """The two axes of the body that actually vary, so a 3-D shape is readable in a static picture."""
    import numpy as np

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return 0, 1
    span = pts.max(0) - pts.min(0)
    order = list(np.argsort(span)[::-1])
    return int(order[0]), int(order[min(1, len(order) - 1)])


def geometry_figure(view: dict[str, Any]):
    """Each sample's body: the point cloud the model sees, one panel per sample, drawn from the points themselves."""
    from pinneapple_tools.visualization import make_panel_figure, panel_letters

    samples = view["samples"]
    fig, axes = make_panel_figure(ncols=min(len(samples), 4))
    axes = axes.ravel()
    for ax, sample in zip(axes, samples, strict=False):
        i, j = _two_widest(view, sample["points"])
        xs = [p[i] for p in sample["points"]]
        ys = [p[j] for p in sample["points"]]
        # the point cloud and nothing more: joining consecutive points draws the order they were sampled in, which for
        # a surface of revolution is a zig-zag between meridians, not a body
        ax.scatter(xs, ys, s=2, linewidths=0)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(f"x{i + 1}")
        ax.set_ylabel(f"x{j + 1}")
        ax.set_title(sample["label"], fontsize=8)
    for ax in axes[len(samples):]:
        ax.axis("off")
    panel_letters(fig, axes=list(axes[: len(samples)]))
    fig.suptitle(_title(view, "geometry"), fontsize=10)
    fig.tight_layout()
    return fig


def prediction_figure(view: dict[str, Any]):
    """Reference against prediction, one row per field and one column per sample, along the ordered query points."""
    from pinneapple_tools.visualization import PALETTE, make_figure, panel_letters

    samples, fields = view["samples"], view["fields"]
    fig, axes = make_figure(nrows=max(len(fields), 1), ncols=max(len(samples), 1),
                            figsize=(3.1 * max(len(samples), 1), 2.2 * max(len(fields), 1)), squeeze=False)
    for col, sample in enumerate(samples):
        for row, field in enumerate(fields):
            ax = axes[row][col]
            target, pred = sample["target"][field], sample["pred"][field]
            ax.plot(range(len(target)), target, color=PALETTE[0], lw=1.3, label="reference")
            ax.plot(range(len(pred)), pred, color=PALETTE[1], lw=1.1, ls="--", label="model")
            if row == 0:
                ax.set_title(sample["label"], fontsize=8)
            if col == 0:
                unit = (view["units"] or [""] * len(fields))[row]
                ax.set_ylabel(f"{field} [{unit}]" if unit else field)
            if row == len(fields) - 1:
                ax.set_xlabel("query point")
            if sample["relative_l2"] is not None:
                ax.annotate(f"rel-L2 {sample['relative_l2']:.3f}", xy=(0.02, 0.04), xycoords="axes fraction",
                            fontsize=7)
    axes[0][0].legend(fontsize=7, loc="upper right")
    panel_letters(fig, axes=[axes[r][c] for r in range(len(fields)) for c in range(len(samples))])
    fig.suptitle(_title(view, "prediction"), fontsize=10)
    fig.tight_layout()
    return fig


_BUILDERS = {"loss": loss_figure, "design": design_figure, "geometry": geometry_figure,
             "prediction": prediction_figure}


# ------------------------------------------------------------------------------------------------------- the cache


def view_path(out: str | Path, family: str) -> Path:
    return Path(out) / family / "model" / "view.json"


def ensure(out: str | Path, family: str, kind: str) -> Path:
    """The PNG for one family and one kind, rebuilt only when ``view.json`` is newer than it.

    A family that has not finished has no view and gets an explicit error, which the page draws as "not measured yet"
    rather than as a blank chart.
    """
    if kind not in KINDS:
        raise KeyError(f"unknown figure {kind!r}; the families show {', '.join(KINDS)}")
    source = view_path(out, family)
    target = source.parent / "figures" / f"{kind}.png"
    with _LOCK:
        if target.is_file() and source.is_file() and target.stat().st_mtime >= source.stat().st_mtime:
            return target
        if not source.is_file():
            raise FileNotFoundError(f"{family} has no view.json yet: it has not been fitted")
        view = json.loads(source.read_text(encoding="utf-8"))
        _style()
        import matplotlib.pyplot as plt

        from pinneapple_tools.visualization import save_figure, source_note

        fig = _BUILDERS[kind](view)
        source_note(fig, _source(view))
        target.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, target.with_suffix(""), formats=("png",))   # save_figure appends the suffix
        plt.close(fig)
        return target


def build_all(out: str | Path, family: str) -> dict[str, str]:
    """Every figure a finished family can show, as paths. Used by the campaign's own ``--figures`` if asked."""
    return {kind: str(ensure(out, family, kind)) for kind in KINDS}

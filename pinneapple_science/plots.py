"""Figure conventions of :mod:`pinneapple_science`: one palette (computed = slot 1, reference = slot 2, in that
fixed order), a single-hue sequential map for magnitudes, a two-hue diverging map with a neutral middle for signed
fields, thin marks, recessive axes. Every function returns a matplotlib figure; nothing is saved here."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

#: categorical slots in fixed order (blue, orange, aqua, yellow, magenta, green, violet, red)
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
COMPUTED, REFERENCE, THIRD = SERIES[0], SERIES[1], SERIES[2]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SEQUENTIAL = "Blues"
DIVERGING = "RdBu_r"

__all__ = ["figure", "SERIES", "COMPUTED", "REFERENCE", "THIRD", "SEQUENTIAL", "DIVERGING", "style", "badge", "to_rgb",
           "close", "integer_ticks"]


def style(ax) -> None:
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_fontsize(10)


def figure(ncols: int = 1, nrows: int = 1, *, width: float = 4.2, height: float = 3.3, three_d: tuple = (),
           title: str = ""):
    """``(fig, axes)``: ``axes`` is a flat list; indices in ``three_d`` get 3-D axes."""
    fig = plt.figure(figsize=(width * ncols, height * nrows), facecolor="white")
    axes = []
    for k in range(nrows * ncols):
        ax = fig.add_subplot(nrows, ncols, k + 1, projection="3d" if k in three_d else None)
        if k not in three_d:
            style(ax)
        axes.append(ax)
    if title:
        fig.suptitle(title, fontsize=11, color=INK)
    return fig, axes


def badge(ax, text: str, ok: bool | None = True, loc: str = "lower right") -> None:
    """A small text box with a measured result (the ink is the text colour; the frame carries pass / fail)."""
    edge = {True: SERIES[5], False: SERIES[7], None: INK2}[ok]
    xy = {"lower right": (0.98, 0.04, "right", "bottom"), "upper right": (0.98, 0.96, "right", "top"),
          "upper left": (0.02, 0.96, "left", "top"), "lower left": (0.02, 0.04, "left", "bottom")}[loc]
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=xy[2], va=xy[3], fontsize=7.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=edge, lw=1.2, alpha=0.92))


def close(fig) -> None:
    plt.close(fig)


def log_ticks_ok(values) -> bool:
    v = np.asarray(values, float)
    return bool(np.all(v > 0))


def to_rgb(fig) -> np.ndarray:
    """The figure as an RGB array (for animation frames)."""
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def integer_ticks(ax, values) -> None:
    """Plain integer labels at ``values`` on a log x axis (no minor labels like 3x10^1)."""
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

    ax.xaxis.set_major_locator(FixedLocator(list(values)))
    ax.set_xticklabels([str(int(v)) for v in values])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())

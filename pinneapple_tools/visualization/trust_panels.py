"""Side-by-side field panels with a trust verdict under each model: the figure (and GIF) of "which prediction can I
use, and where does it break the physics".

Each row is a case (a regime, a sample); each panel is a field on a 2-D grid. ``kind="field"`` panels of a row share
one sequential colour scale (blue, light to dark), so the reference and the predictions are directly comparable;
``kind="residual"`` panels show a magnitude on a logarithmic orange scale (the PDE residual, an error). A panel may
carry verdict badges (``verdict``, ``verdicts``), coloured by the first status word in them (APPROVED, REVIEW,
REJECT, PASS, OK, FAIL) and always shown with their text, so the colour never carries the meaning alone.

``trust_panels(rows)`` returns a matplotlib figure; ``trust_panels_gif(frames, path)`` writes one figure per frame
into an animated GIF (Pillow, which matplotlib already depends on).
"""
from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

__all__ = ["Panel", "Row", "STATUS_COLOURS", "trust_panels", "trust_panels_gif"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
#: Verdict -> colour (good / warning / critical of a validated status palette).
STATUS_COLOURS = {"APPROVED": "#0ca30c", "REVIEW": "#fab219", "REJECT": "#d03b3b", "PASS": "#0ca30c",
                  "OK": "#0ca30c", "FAIL": "#d03b3b", "NO": "#d03b3b", "CAUTION": "#fab219"}
_BLUE = LinearSegmentedColormap.from_list("pp_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab",
                                                       "#0d366b"])
_ORANGE = LinearSegmentedColormap.from_list("pp_orange", ["#fdf3ec", "#f8cfb4", "#f0a070", "#e0703a", "#b44d17",
                                                         "#6e2c08"])


@dataclass
class Panel:
    """One field. ``kind``: ``"field"`` (shared linear scale per row), ``"residual"`` (log magnitude) or ``"error"``
    (magnitude on a linear scale from 0 to ``error_max`` of :func:`trust_panels`). ``inset``: draw a dashed square
    that many cells inside the border (e.g. the edge of an interior region)."""
    title: str
    values: np.ndarray
    kind: str = "field"
    verdict: str | None = None
    verdicts: Sequence[str] = ()
    note: str = ""
    inset: int | None = None


@dataclass
class Row:
    label: str
    panels: Sequence[Panel]


def _badge(ax, text: str, k: int = 0) -> None:
    """Badge ``k`` under the panel, coloured by the first status word in it (``"fields: REJECT 9.5x limit"``)."""
    words = "".join(c if c.isalnum() else " " for c in text.upper()).split()
    key = next((w for w in words if w in STATUS_COLOURS), "")
    col = STATUS_COLOURS.get(key, MUTED)
    ink = INK if key in ("REVIEW", "CAUTION") else "white"
    ax.text(0.5, -0.04 - 0.13 * k, text, transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
            fontweight="bold", color=ink, bbox=dict(boxstyle="round,pad=0.25", fc=col, ec="none"))


def trust_panels(rows: Sequence[Row], *, title: str = "", footer: str = "", cell: float = 2.4,
                 residual_range: tuple[float, float] | None = None, error_max: float | None = None,
                 error_label: str = "", dpi: int = 110):
    """Figure with one row per :class:`Row`: its ``"field"`` panels, a colour bar, its ``"residual"`` / ``"error"``
    panels and their colour bar (labelled ``error_label``); every row has the same number of panels of each group."""
    from matplotlib.gridspec import GridSpec
    rows = list(rows)
    second = ("residual", "error")
    nf = max(sum(p.kind not in second for p in r.panels) for r in rows)
    nr = max(sum(p.kind in second for p in r.panels) for r in rows)
    widths = [1.0] * nf + [0.06, 0.18] + ([1.0] * nr + [0.06] if nr else [])
    nb = max((bool(p.verdict) + len(p.verdicts) for r in rows for p in r.panels), default=0)
    fig = plt.figure(figsize=(cell * (nf + nr) + 1.6, cell * len(rows) * (1.2 + 0.11 * nb) + 0.9), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    top = 1 - 0.55 / fig.get_figheight() if title else 0.97
    bottom = (0.35 + 0.3 * nb + (0.3 * (footer.count(chr(10)) + 1) if footer else 0)) / fig.get_figheight()
    gs = GridSpec(len(rows), len(widths), figure=fig, width_ratios=widths, left=0.045, right=0.97, top=top,
                  bottom=bottom, hspace=0.25 + 0.13 * nb, wspace=0.06)
    res = [p.values for r in rows for p in r.panels if p.kind == "residual"]
    if residual_range is None and res:
        allv = np.concatenate([np.abs(v[np.isfinite(v)]).ravel() for v in res])
        allv = allv[allv > 0]
        residual_range = (float(np.quantile(allv, 0.02)), float(np.quantile(allv, 0.998))) if allv.size else (1e-6, 1)
    errs = [np.abs(p.values) for r in rows for p in r.panels if p.kind == "error"]
    if error_max is None and errs:
        error_max = float(np.quantile(np.concatenate([e.ravel() for e in errs]), 0.99)) or 1.0
    for i, r in enumerate(rows):
        fields = [p for p in r.panels if p.kind not in second]
        resid = [p for p in r.panels if p.kind in second]
        lo = min(float(np.nanmin(p.values)) for p in fields) if fields else 0.0
        hi = max(float(np.nanmax(p.values)) for p in fields) if fields else 1.0
        im_f = im_r = None
        for j, p in enumerate(fields + resid):
            col = j if j < len(fields) else nf + 2 + (j - len(fields))
            ax = fig.add_subplot(gs[i, col])
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#e1e0d9")
            if p.kind == "residual":
                im_r = ax.imshow(np.abs(p.values).T, origin="lower", cmap=_ORANGE, norm=LogNorm(*residual_range),
                                 interpolation="nearest")
            elif p.kind == "error":
                im_r = ax.imshow(np.abs(p.values).T, origin="lower", cmap=_ORANGE, vmin=0.0, vmax=error_max,
                                 interpolation="nearest")
            else:
                im_f = ax.imshow(p.values.T, origin="lower", cmap=_BLUE, vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_title(p.title, fontsize=9.5, color=INK, pad=4)
            if p.inset:
                from matplotlib.patches import Rectangle
                n0, n1 = p.values.shape
                ax.add_patch(Rectangle((p.inset - 0.5, p.inset - 0.5), n0 - 2 * p.inset, n1 - 2 * p.inset, fill=False,
                                       ec=INK2, lw=0.9, ls=(0, (3, 2))))
            if p.note:
                ax.text(0.03, 0.97, p.note, transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=INK,
                        bbox=dict(boxstyle="round,pad=0.2", fc=(1, 1, 1, 0.8), ec="none"))
            for k, v in enumerate(([p.verdict] if p.verdict else []) + list(p.verdicts)):
                _badge(ax, v, k)
            if j == 0:
                ax.set_ylabel(r.label, fontsize=9.5, color=INK2)
        for im, col in ((im_f, nf), (im_r, len(widths) - 1)):
            if im is None:
                continue
            cb = fig.colorbar(im, cax=fig.add_subplot(gs[i, col]))
            cb.ax.tick_params(labelsize=7, colors=INK2)
            cb.outline.set_visible(False)
            if im is im_r and error_label and any(p.kind == "error" for p in resid):
                cb.set_label(error_label, fontsize=7.5, color=INK2)
    if title:
        fig.suptitle(title, fontsize=12.5, color=INK, x=0.02, y=1 - 0.12 / fig.get_figheight(), ha="left",
                     va="top", fontweight="bold")
    if footer:
        fig.text(0.02, 0.08 / fig.get_figheight(), footer, fontsize=8, color=INK2, ha="left", va="bottom")
    return fig


def trust_panels_gif(frames: Sequence[Sequence[Row]], path: str | Path, *, duration_ms: int = 1600,
                     **kwargs) -> Path:
    """One :func:`trust_panels` figure per frame, written as a looping GIF. ``kwargs`` go to ``trust_panels`` (pass
    the same ``residual_range`` to every frame for a fixed residual scale)."""
    from PIL import Image
    images = []
    for rows in frames:
        fig = trust_panels(rows, **kwargs)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).convert("RGB"))
    size = images[0].size
    images = [im if im.size == size else im.resize(size) for im in images]
    pal = [im.quantize(colors=255, method=Image.Quantize.MEDIANCUT) for im in images]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=duration_ms, loop=0, optimize=True)
    return path

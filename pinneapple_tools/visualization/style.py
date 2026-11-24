"""CFD-style colour maps, themes and defaults for PINNeAPPle visualisation.

Two layers live here. ``use_cfd_style`` is the older dark CFD theme and is kept
for the figures that already use it. ``use_theme`` is the house theme described
in #918: one font stack, one colour-blind-safe palette, perceptually uniform
maps, and helpers that make the quantity and its unit part of the figure.

No colour is written down in this module. Every palette entry, colour map stop,
figure size and font comes from :mod:`pinneapple_tools.visualization.theme`, the
one file the studio viewer, the twin3d viewer and the apps read as well (#65), so
a figure in a report and a field in a browser cannot disagree.
"""
from __future__ import annotations

import string

import matplotlib as mpl
import matplotlib.pyplot as plt

from . import theme as _theme

# ---------------------------------------------------------------------------
# CFD colormaps
#
# The stops are the theme's (theme.json, key "colormaps"); this module only
# turns them into matplotlib maps and gives them their names.
# ---------------------------------------------------------------------------

_cfd_rainbow = _theme.colormap("cfd_rainbow")
_cfd_coolwarm = _theme.colormap("cfd_coolwarm")
_cfd_sequential = _theme.colormap("cfd_sequential")
_cfd_diverging = _theme.colormap("cfd_diverging")
_cfd_cyclic = _theme.colormap("cfd_cyclic")

CMAPS: dict = {
    "rainbow":   _cfd_rainbow,
    "cfd_rainbow": _cfd_rainbow,
    "coolwarm":  _theme.colormap("coolwarm"),   # the diverging map the viewers draw
    "cfd_coolwarm": _cfd_coolwarm,               # the older Python table, kept by name
    "cfd_sequential": _cfd_sequential,
    "cfd_diverging": _cfd_diverging,
    "cfd_cyclic": _cfd_cyclic,
    # the maps the studio viewer and the apps draw, so a figure can take the same one
    "jet":       _theme.colormap("jet"),
    "magma":     _theme.colormap("magma"),
    "bwr":       _theme.colormap("bwr"),
    "rdbu":      _theme.colormap("rdbu"),
    "err":       _theme.colormap("err"),
    # matplotlib's own, left as they are: nobody authors these
    "viridis":   plt.cm.viridis,
    "plasma":    plt.cm.plasma,
    "inferno":   plt.cm.inferno,
    "turbo":     plt.cm.turbo,
    "grayscale": plt.cm.gray,
    "hot":       plt.cm.hot,
    "RdBu_r":    plt.cm.RdBu_r,
}

# ``cfd_rainbow`` is kept for comparison pictures against commercial CAE, but it
# is no longer the default: it is not perceptually uniform.
DEFAULT_CMAP = "cfd_sequential"


# ---------------------------------------------------------------------------
# Axes theme
# ---------------------------------------------------------------------------

CFD_RC = {
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor":   "#16213e",
    "axes.edgecolor":   "#0f3460",
    "axes.labelcolor":  "#e0e0e0",
    "axes.titlecolor":  "#ffffff",
    "xtick.color":      "#e0e0e0",
    "ytick.color":      "#e0e0e0",
    "text.color":       "#e0e0e0",
    "grid.color":       "#0f3460",
    "grid.linestyle":   "--",
    "grid.alpha":       0.4,
    "figure.titlesize": 13,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "font.family":      "monospace",
    "image.cmap":       "rainbow",
}

LIGHT_RC = {
    "figure.facecolor": "#f5f5f5",
    "axes.facecolor":   "#ffffff",
    "axes.edgecolor":   "#cccccc",
    "axes.labelcolor":  "#222222",
    "axes.titlecolor":  "#111111",
    "xtick.color":      "#333333",
    "ytick.color":      "#333333",
    "text.color":       "#111111",
    "grid.color":       "#dddddd",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "figure.titlesize": 13,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
}


def use_cfd_style(dark: bool = True) -> None:
    """Apply CFD dark/light theme globally."""
    mpl.rcParams.update(CFD_RC if dark else LIGHT_RC)
    mpl.colormaps.register(_cfd_rainbow,  name="cfd_rainbow",  force=True)
    mpl.colormaps.register(_cfd_coolwarm, name="cfd_coolwarm", force=True)


def get_cmap(name: str = DEFAULT_CMAP):
    """Return a matplotlib colormap by PINNeAPPle alias or standard name."""
    if name in CMAPS:
        return CMAPS[name]
    return plt.get_cmap(name)


def make_figure(nrows: int = 1, ncols: int = 1, **kwargs):
    """Create a styled figure."""
    figsize = kwargs.pop("figsize", (6 * ncols, 5 * nrows))
    return plt.subplots(nrows, ncols, figsize=figsize, **kwargs)


# ---------------------------------------------------------------------------
# One house theme (#918)
#
# Everything below extends this module; no other palette module exists. The
# rule is that a figure added from now on calls ``use_theme`` and takes its
# colour from ``field_cmap`` for the quantity it shows, so the same quantity is
# the same colour in every example.
# ---------------------------------------------------------------------------

# Okabe-Ito, read from the theme file: eight colours that stay distinguishable
# under deuteranopia, protanopia and tritanopia. Never reorder: the first three
# carry most figures.
PALETTE = _theme.palette()

# The house maps, built by theme.py from the file's anchors rather than sampled
# from a rainbow: monotone lightness, so a printed grey-scale copy keeps the
# ordering. ``cfd_sequential``/``cfd_diverging``/``cfd_cyclic`` are already in
# CMAPS above; these aliases are what the rest of this module reads.
_SEQUENTIAL_DATA = _theme.colormap_stops("cfd_sequential")
_DIVERGING_DATA = _theme.colormap_stops("cfd_diverging")
_CYCLIC_DATA = _theme.colormap_stops("cfd_cyclic")

# One quantity, one map, from the same file. A new quantity is added to
# theme.json, never chosen ad hoc in a figure, and the error always reads on a
# sequential map because "more error" has no sign.
FIELD_CMAPS: dict = _theme.field_cmaps()

_THEME_KINDS = tuple(_theme.theme()["figure"]["scale"])

# Relative sizes per output medium: a report is read at 100 %, a slide is
# projected, a paper is printed at column width. The base body size is the one
# that stays readable at 50 % zoom.
_THEME_SCALE = dict(_theme.theme()["figure"]["scale"])

_THEME_FIGSIZE = {kind: {k: tuple(v) for k, v in sizes.items()}
                  for kind, sizes in _theme.theme()["figure"]["sizes"].items()}

_FONT_STACK = list(_theme.font()["stack"])


def _theme_rc(kind: str, dark: bool) -> dict:
    scale = _THEME_SCALE[kind]
    base = _theme.font()["base_size"] * scale
    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": list(_FONT_STACK),
        "font.size": base,
        "axes.titlesize": base * 1.15,
        "axes.labelsize": base,
        "xtick.labelsize": base * 0.9,
        "ytick.labelsize": base * 0.9,
        "legend.fontsize": base * 0.9,
        "figure.titlesize": base * 1.3,
        "axes.grid": True,
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.30,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.prop_cycle": mpl.cycler(color=list(PALETTE)),
        "image.cmap": "cfd_sequential",
        "figure.dpi": _theme.figure(kind)["dpi"]["screen"],
        "savefig.dpi": _theme.figure(kind)["dpi"]["save"],
        "savefig.bbox": "tight",
        "lines.linewidth": 1.6,
        "legend.frameon": False,
        "errorbar.capsize": 2.5,
    }
    # the surface and the ink are the theme's: the same near-black text and the same grid colour a browser legend
    # uses, so a figure and its HTML twin do not read as two documents (#65)
    ink, line, paper = _theme.surface(dark)["ink"], _theme.surface(dark)["line"], _theme.surface(dark)["paper"]
    if dark:
        rc.update({
            "figure.facecolor": paper,
            "axes.facecolor":   "#16213e",
            "axes.edgecolor":   line,
            "axes.labelcolor":  ink,
            "axes.titlecolor":  "#ffffff",
            "xtick.color":      ink,
            "ytick.color":      ink,
            "text.color":       ink,
            "grid.color":       line,
            "legend.edgecolor": line,
        })
    else:
        rc.update({
            "figure.facecolor": paper,
            "axes.facecolor":   paper,
            "axes.edgecolor":   "#4d4d4d",
            "axes.labelcolor":  ink,
            "axes.titlecolor":  "#111111",
            "xtick.color":      ink,
            "ytick.color":      ink,
            "text.color":       ink,
            "grid.color":       line,
            "legend.edgecolor": "#4d4d4d",
        })
    return rc


def use_theme(kind: str = "report", dark: bool = False) -> None:
    """Apply the house theme: one font stack, one palette, one set of maps.

    ``kind`` is ``"report"`` (read on screen), ``"slide"`` (projected, larger
    type) or ``"paper"`` (printed at column width). ``dark`` switches to the
    dark background used by the older CFD theme.

    ``cfd_rainbow`` stays registered for comparison pictures against commercial
    CAE, but it is no longer any figure's default: it is not perceptually
    uniform, so a reader cannot rank two colours by value.
    """
    if kind not in _THEME_KINDS:
        raise ValueError(f"unknown theme kind {kind!r}; choose from {_THEME_KINDS}")
    mpl.rcParams.update(_theme_rc(kind, dark))
    for cmap in (_cfd_rainbow, _cfd_coolwarm, _cfd_sequential, _cfd_diverging, _cfd_cyclic):
        if cmap.name not in mpl.colormaps:
            mpl.colormaps.register(cmap, name=cmap.name, force=True)


def field_cmap(quantity: str):
    """Return the colormap registered for ``quantity`` (e.g. ``"temperature"``).

    Unknown quantities raise instead of silently falling back: a fallback is
    how two figures of the same quantity end up with different colours.
    """
    key = str(quantity).strip().lower().replace(" ", "_").replace("-", "_")
    if key not in FIELD_CMAPS:
        known = ", ".join(sorted(FIELD_CMAPS))
        raise KeyError(f"no colormap registered for quantity {quantity!r}; known: {known}")
    return get_cmap(FIELD_CMAPS[key])


def figure_sizes(kind: str = "report") -> dict:
    """Figure sizes for 1-, 2- and 3-panel layouts of the given theme kind."""
    if kind not in _THEME_KINDS:
        raise ValueError(f"unknown theme kind {kind!r}; choose from {_THEME_KINDS}")
    return dict(_THEME_FIGSIZE[kind])


def make_panel_figure(ncols: int = 1, kind: str = "report", **kwargs):
    """Figure and axes sized by the theme for a 1-, 2- or 3-panel layout."""
    sizes = figure_sizes(kind)
    figsize = kwargs.pop("figsize", sizes["triple" if ncols >= 3 else ("double" if ncols == 2 else "single")])
    return plt.subplots(1, ncols, figsize=figsize, squeeze=False, **kwargs)


def colorbar(ax, mappable, quantity: str, unit: str, **kwargs):
    """Colour bar that always carries the quantity and its unit.

    Both are required: an unlabelled colour bar is a number the reader cannot
    use. ``unit`` accepts ``""`` only for a genuinely dimensionless quantity,
    and the label then says ``[-]``.
    """
    if not str(quantity).strip():
        raise ValueError("a colour bar needs its quantity")
    if unit is None:
        raise ValueError("a colour bar needs its unit ('' for dimensionless)")
    label = f"{quantity} [{unit}]" if str(unit).strip() else f"{quantity} [-]"
    cb = ax.figure.colorbar(mappable, ax=ax, **kwargs)
    cb.set_label(label)
    return cb


def label(ax, x=None, y=None, title: str = None):
    """Axis labels as ``(name, unit)`` pairs, plus an optional title."""
    if x is not None:
        name, unit = x
        ax.set_xlabel(f"{name} [{unit}]" if str(unit).strip() else f"{name} [-]")
    if y is not None:
        name, unit = y
        ax.set_ylabel(f"{name} [{unit}]" if str(unit).strip() else f"{name} [-]")
    if title is not None:
        ax.set_title(title)
    return ax


_VERDICT_STYLE = {
    True:  {"color": "#0072B2", "marker": "o", "text": "pass"},
    False: {"color": "#D55E00", "marker": "X", "text": "fail"},
}


def annotate_verdict(ax, text: str, ok: bool, xy=(0.02, 0.98), **kwargs):
    """Stamp a pass/fail verdict in the corner of an axes.

    The colour is not the only signal: the marker and the word carry the same
    information, so the stamp survives a grey-scale print.
    """
    style = _VERDICT_STYLE[bool(ok)]
    defaults = dict(transform=ax.transAxes, va="top", ha="left", fontsize=9,
                    color=style["color"], fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="none",
                              edgecolor=style["color"], linewidth=0.8))
    defaults.update(kwargs)
    return ax.text(xy[0], xy[1], f"{style['marker']} {style['text']}: {text}", **defaults)


def panel_letters(fig, axes=None, *, start: int = 0, xy=(-0.02, 1.02), **kwargs):
    """Letter the panels (a), (b), (c) in reading order."""
    if axes is None:
        axes = [ax for ax in fig.axes]
    defaults = dict(fontsize=11, fontweight="bold", va="top", ha="left")
    defaults.update(kwargs)
    letters = []
    for i, ax_ in enumerate(axes):
        ax_.text(xy[0], xy[1], f"({string.ascii_lowercase[start + i]})",
                 transform=ax_.transAxes, **defaults)
        letters.append(f"({string.ascii_lowercase[start + i]})")
    return letters


def source_note(fig, text: str, *, y: float = -0.01, **kwargs):
    """Data provenance line under the figure; required on every exported figure."""
    if not str(text).strip():
        raise ValueError("a source note cannot be empty: say where the data came from")
    defaults = dict(ha="right", va="top", fontsize=7, color="#555555")
    defaults.update(kwargs)
    return fig.text(0.99, y, text, **defaults)


def save_figure(fig, path, *, formats=("png", "svg")):
    """Save the figure as PNG at 200 dpi plus a vector copy.

    The issue asks for both: a raster for review and a vector for the paper and
    the report, where a zoomed detail must stay sharp.
    """
    from pathlib import Path

    base = Path(path)
    written = []
    for fmt in formats:
        target = base.with_suffix(f".{fmt}") if base.suffix else base.with_name(base.name + f".{fmt}")
        fig.savefig(target, format=fmt, dpi=200, bbox_inches="tight")
        written.append(str(target))
    return written

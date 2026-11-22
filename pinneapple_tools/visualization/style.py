"""CFD-style colour maps, themes and defaults for PINNeAPPle visualisation.

Two layers live here. ``use_cfd_style`` is the older dark CFD theme and is kept
for the figures that already use it. ``use_theme`` is the house theme described
in #918: one font stack, one colour-blind-safe palette, perceptually uniform
maps, and helpers that make the quantity and its unit part of the figure.
"""
from __future__ import annotations

import string

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CFD colormaps
# ---------------------------------------------------------------------------

_RAINBOW_DATA = [
    (0.0,   (0.000, 0.000, 0.550)),
    (0.125, (0.000, 0.000, 1.000)),
    (0.250, (0.000, 0.500, 1.000)),
    (0.375, (0.000, 1.000, 0.500)),
    (0.500, (0.000, 1.000, 0.000)),
    (0.625, (0.500, 1.000, 0.000)),
    (0.750, (1.000, 1.000, 0.000)),
    (0.875, (1.000, 0.500, 0.000)),
    (1.000, (1.000, 0.000, 0.000)),
]

_cfd_rainbow = mcolors.LinearSegmentedColormap.from_list(
    "cfd_rainbow",
    [(v[0], v[1]) for v in _RAINBOW_DATA],
    N=256,
)

_COOL_WARM_DATA = [
    (0.0, (0.085, 0.532, 0.201)),
    (0.25,(0.017, 0.427, 0.969)),
    (0.5, (0.929, 0.929, 0.929)),
    (0.75,(0.969, 0.266, 0.105)),
    (1.0, (0.655, 0.047, 0.047)),
]
_cfd_coolwarm = mcolors.LinearSegmentedColormap.from_list(
    "cfd_coolwarm", [(v[0], v[1]) for v in _COOL_WARM_DATA], N=256,
)

# Paraview-esque blue→white→red (diverging, for pressure/vorticity)
CMAPS: dict = {
    "rainbow":   _cfd_rainbow,
    "coolwarm":  _cfd_coolwarm,
    "viridis":   plt.cm.viridis,
    "plasma":    plt.cm.plasma,
    "jet":       plt.cm.jet,
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

# Okabe-Ito: eight colours that stay distinguishable under deuteranopia,
# protanopia and tritanopia. Never reorder: the first three carry most figures.
PALETTE = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermilion
    "#F0E442",  # yellow
    "#000000",  # black
)

# Perceptually uniform maps, built from anchors rather than sampled from a
# rainbow: monotone lightness, so a printed grey-scale copy keeps the ordering.
_SEQUENTIAL_DATA = [
    (0.00, (0.267, 0.005, 0.329)),
    (0.25, (0.229, 0.322, 0.545)),
    (0.50, (0.128, 0.567, 0.551)),
    (0.75, (0.369, 0.789, 0.383)),
    (1.00, (0.993, 0.906, 0.144)),
]
_DIVERGING_DATA = [
    (0.00, (0.230, 0.299, 0.754)),
    (0.25, (0.552, 0.690, 0.996)),
    (0.50, (0.968, 0.968, 0.968)),
    (0.75, (0.956, 0.598, 0.477)),
    (1.00, (0.706, 0.016, 0.150)),
]
_CYCLIC_DATA = [
    (0.000, (0.128, 0.567, 0.551)),
    (0.250, (0.369, 0.789, 0.383)),
    (0.500, (0.993, 0.906, 0.144)),
    (0.750, (0.847, 0.432, 0.145)),
    (1.000, (0.128, 0.567, 0.551)),
]

_cfd_sequential = mcolors.LinearSegmentedColormap.from_list(
    "cfd_sequential", [(v[0], v[1]) for v in _SEQUENTIAL_DATA], N=256,
)
_cfd_diverging = mcolors.LinearSegmentedColormap.from_list(
    "cfd_diverging", [(v[0], v[1]) for v in _DIVERGING_DATA], N=256,
)
_cfd_cyclic = mcolors.LinearSegmentedColormap.from_list(
    "cfd_cyclic", [(v[0], v[1]) for v in _CYCLIC_DATA], N=256,
)

CMAPS.update({
    "cfd_sequential": _cfd_sequential,
    "cfd_diverging":  _cfd_diverging,
    "cfd_cyclic":     _cfd_cyclic,
})

# One quantity, one map. A new quantity is added here, never chosen ad hoc in a
# figure, and the error always reads on a sequential map because "more error"
# has no sign.
FIELD_CMAPS: dict = {
    # sequential: a magnitude, no meaningful zero
    "temperature": "cfd_sequential",
    "pressure": "cfd_sequential",
    "density": "cfd_sequential",
    "speed": "cfd_sequential",
    "velocity_magnitude": "cfd_sequential",
    "turbulent_kinetic_energy": "cfd_sequential",
    "stress": "cfd_sequential",
    "damage": "cfd_sequential",
    "concentration": "cfd_sequential",
    "error": "cfd_sequential",
    "residual": "cfd_sequential",
    # diverging: a signed quantity around a meaningful zero
    "pressure_coefficient": "cfd_diverging",
    "vorticity": "cfd_diverging",
    "displacement": "cfd_diverging",
    "temperature_difference": "cfd_diverging",
    "stress_difference": "cfd_diverging",
    "signed_error": "cfd_diverging",
    # cyclic: an angle
    "angle": "cfd_cyclic",
    "phase": "cfd_cyclic",
}

_THEME_KINDS = ("report", "slide", "paper")

# Relative sizes per output medium: a report is read at 100 %, a slide is
# projected, a paper is printed at column width. The base body size is the one
# that stays readable at 50 % zoom.
_THEME_SCALE = {"report": 1.00, "slide": 1.35, "paper": 0.90}

_THEME_FIGSIZE = {
    "report": {"single": (6.4, 4.0), "double": (9.6, 4.0), "triple": (12.8, 4.0)},
    "slide":  {"single": (9.0, 5.5), "double": (14.0, 5.5), "triple": (18.0, 5.5)},
    "paper":  {"single": (3.5, 2.6), "double": (7.16, 2.8), "triple": (7.16, 2.4)},
}

_FONT_STACK = ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"]


def _theme_rc(kind: str, dark: bool) -> dict:
    scale = _THEME_SCALE[kind]
    base = 10.0 * scale
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
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "lines.linewidth": 1.6,
        "legend.frameon": False,
        "errorbar.capsize": 2.5,
    }
    if dark:
        rc.update({
            "figure.facecolor": "#1a1a2e",
            "axes.facecolor":   "#16213e",
            "axes.edgecolor":   "#5a6b8c",
            "axes.labelcolor":  "#e8e8e8",
            "axes.titlecolor":  "#ffffff",
            "xtick.color":      "#d8d8d8",
            "ytick.color":      "#d8d8d8",
            "text.color":       "#e8e8e8",
            "grid.color":       "#3a4a6b",
            "legend.edgecolor": "#5a6b8c",
        })
    else:
        rc.update({
            "figure.facecolor": "#ffffff",
            "axes.facecolor":   "#ffffff",
            "axes.edgecolor":   "#4d4d4d",
            "axes.labelcolor":  "#222222",
            "axes.titlecolor":  "#111111",
            "xtick.color":      "#333333",
            "ytick.color":      "#333333",
            "text.color":       "#111111",
            "grid.color":       "#c8c8c8",
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

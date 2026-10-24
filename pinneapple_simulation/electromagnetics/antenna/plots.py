"""Figures of an end-fire antenna study (matplotlib; each function draws on an ``ax`` and returns it)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from . import bounds
from .runner import ResultTable

__all__ = ["directivity_vs_length", "lodo_rmse_bars", "layout_plot", "parity_plot"]


def directivity_vs_length(table: ResultTable, ax=None, *, column: str = "D_peak_dBi", target_dbi: float = 10.0,
                          length_col: str = "Lb", label: str | None = None):
    """``column`` against ``L / lambda`` with the Hansen-Woodyard and ordinary end-fire limits and the target."""
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    L = table.col(length_col) * 1e-3
    f = table.col("f_GHz") * 1e9
    y = table.col(column)
    x = L / np.asarray(bounds.wavelength(f))
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.any():
        sc = ax.scatter(x[ok], y[ok], c=f[ok] / 1e9, cmap="viridis", s=22, label=label or column, zorder=3)
        plt.colorbar(sc, ax=ax, label="f [GHz]")
    xs = np.linspace(0.5, max(3.5, float(np.nanmax(x)) * 1.1 if ok.any() else 3.5), 200)
    ax.plot(xs, 10 * np.log10(bounds.HW_FACTOR * xs), "k-", label=f"Hansen-Woodyard {bounds.HW_FACTOR:g} L/λ")
    ax.plot(xs, 10 * np.log10(bounds.ORDINARY_FACTOR * xs), "k--", label="ordinary end-fire 4 L/λ")
    ax.axhline(target_dbi, color="tab:red", lw=1, label=f"target {target_dbi:g} dBi")
    ax.axvline(10 ** (target_dbi / 10) / bounds.HW_FACTOR, color="tab:red", lw=0.8, ls=":")
    ax.set_xlabel("L / λ")
    ax.set_ylabel(f"{column.replace('_dBi', '')} [dBi]")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    return ax


def lodo_rmse_bars(results: Mapping[str, Any], ax=None, *, title: str = "leave-one-design-out RMSE"):
    """Bars of RMSE per model for one or more targets: ``{target: ValidationResult}``."""
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(6, 3.5))[1]
    targets = list(results)
    models = list(dict.fromkeys(m for v in results.values() for m in v.rmse))
    w = 0.8 / max(len(targets), 1)
    for i, t in enumerate(targets):
        r = results[t].rmse
        ax.bar(np.arange(len(models)) + i * w, [r.get(m, np.nan) for m in models], w, label=t)
    ax.set_xticks(np.arange(len(models)) + w * (len(targets) - 1) / 2)
    ax.set_xticklabels(models, rotation=15, fontsize=8)
    ax.set_ylabel("RMSE [dB]")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    return ax


def parity_plot(result, model: str, ax=None):
    """Held-out prediction against the simulated value for one model of a ``ValidationResult``."""
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(4, 4))[1]
    y = np.array([r["y"] for r in result.rows])
    p = np.array([r[f"pred_{model}"] for r in result.rows])
    s = np.array([r[f"std_{model}"] for r in result.rows])
    ax.errorbar(y, p, yerr=1.96 * s, fmt="o", ms=3, alpha=0.7)
    lo, hi = np.nanmin(np.r_[y, p]) - 0.5, np.nanmax(np.r_[y, p]) + 0.5
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.8)
    ax.set_xlabel(f"simulated {result.target}")
    ax.set_ylabel(f"predicted ({model})")
    return ax


def layout_plot(lay, ax=None, *, colors: Sequence[str] = ("tab:red", "tab:blue", "tab:green")):
    """Top (red), bottom (blue) and director (green) conductors of an :class:`.ava.Layout`."""
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(3.5, 6))[1]
    p = lay.params
    ax.add_patch(plt.Rectangle((-p.W / 2, 0), p.W, p.Lb, fc="0.92", ec="0.5"))
    for poly, c in zip((lay.top, lay.bottom, lay.director), colors, strict=False):
        if poly is not None:
            ax.fill(poly[:, 0], poly[:, 1], c, alpha=0.45)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm] (end-fire +y)")
    return ax

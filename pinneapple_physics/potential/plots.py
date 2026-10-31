"""Figures of the potential-surrogate study, in the light theme of :mod:`pinneapple_tools.visualization.style`.

Model colours come from the style's ``cfd_coolwarm`` map (direct: warm end, potential: cool end, input-convex
potential: green end), reference data in ink grey.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib as mpl
import numpy as np

from pinneapple_tools.visualization.style import LIGHT_RC, get_cmap, make_figure

__all__ = ["MODEL_COLORS", "MODEL_LABELS", "errors_figure", "identities_figure", "isotherms_figure",
           "paired_figure"]

_cw = get_cmap("coolwarm")
MODEL_COLORS = {"direct": _cw(0.85), "potential": _cw(0.25), "potential_icnn": _cw(0.0), "reference": "#52514e"}
MODEL_LABELS = {"direct": "direct multi-output", "potential": "potential (MLP)", "potential_icnn": "potential (ICNN)",
                "reference": "CoolProp (Span-Wagner)"}
PROPS = ("p", "s", "u", "cv", "cp", "w")
REGION_LABELS = {"interp": "held-out, training region", "hotter": "extrapolation: T 335-365 K",
                 "denser": "extrapolation: rho 800-950 kg/m3"}


def _save(fig, path):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def errors_figure(summary: Mapping, title: str = "", path: str | None = None):
    """Normalised RMS error per property and region, mean over seeds with 95 % intervals (log scale)."""
    models = list(summary["models"])
    regions = [r for r in REGION_LABELS if r in summary["models"][models[0]]]
    with mpl.rc_context(LIGHT_RC):
        fig, axes = make_figure(1, len(regions), figsize=(5.2 * len(regions), 4.2), sharey=True)
        axes = np.atleast_1d(axes)
        x = np.arange(len(PROPS))
        w = 0.8 / len(models)
        for ax, g in zip(axes, regions, strict=True):
            for i, m in enumerate(models):
                iv = [summary["models"][m][g]["nrmse"][p] for p in PROPS]
                mean = np.array([v["mean"] for v in iv])
                lo = np.array([v["lo"] for v in iv])
                hi = np.array([v["hi"] for v in iv])
                err = np.vstack([mean - np.maximum(lo, 1e-6), hi - mean])
                ax.bar(x + (i - (len(models) - 1) / 2) * w, mean, w, color=MODEL_COLORS.get(m),
                       label=MODEL_LABELS.get(m, m), yerr=err, capsize=2, error_kw={"lw": 0.8})
            ax.set_yscale("log")
            ax.set_xticks(x, ["p", "s", "u", "c_v", "c_p", "w"])
            ax.set_title(REGION_LABELS[g])
            ax.grid(True, axis="y", which="both")
        axes[0].set_ylabel("RMS error / std of the property")
        axes[0].legend(fontsize=8)
        if title:
            fig.suptitle(title)
        fig.tight_layout()
    return _save(fig, path)


def identities_figure(summary: Mapping, region: str = "interp", title: str = "", path: str | None = None):
    """95th percentile of the relative violation of each thermodynamic identity, per model (log scale)."""
    models = list(summary["models"])
    names = list(summary["models"][models[0]][region]["identities"])
    with mpl.rc_context(LIGHT_RC):
        fig, ax = make_figure(1, 1, figsize=(8.5, 4.2))
        y = np.arange(len(names))
        h = 0.8 / len(models)
        for i, m in enumerate(models):
            iv = [summary["models"][m][region]["identities"][n] for n in names]
            mean = np.array([v["mean"] for v in iv])
            ax.barh(y + (i - (len(models) - 1) / 2) * h, mean, h, color=MODEL_COLORS.get(m),
                    label=MODEL_LABELS.get(m, m))
        ax.set_xscale("log")
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xlabel("relative violation, 95th percentile (mean over seeds)")
        ax.grid(True, axis="x", which="both")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title(title or f"Thermodynamic identities, {REGION_LABELS[region]}")
        fig.tight_layout()
    return _save(fig, path)


def isotherms_figure(run: Mapping, reference: Mapping[str, Mapping[str, np.ndarray]], rho: np.ndarray,
                     props: Sequence[str] = ("p", "cp", "w"), title: str = "", path: str | None = None):
    """Predictions of one seed along the stored isotherms against the reference (one row per isotherm)."""
    temps = list(reference)
    units = {"p": "p [MPa]", "cp": "c_p [kJ/(kg K)]", "w": "w [m/s]", "cv": "c_v [kJ/(kg K)]", "s": "s [kJ/(kg K)]",
             "u": "u [kJ/kg]"}
    scale = {"p": 1e-6, "cp": 1e-3, "cv": 1e-3, "s": 1e-3, "u": 1e-3, "w": 1.0}
    with mpl.rc_context(LIGHT_RC):
        fig, axes = make_figure(len(temps), len(props), figsize=(4.4 * len(props), 3.3 * len(temps)))
        axes = np.atleast_2d(axes)
        for i, T in enumerate(temps):
            for j, p in enumerate(props):
                ax = axes[i, j]
                ax.plot(rho, scale[p] * np.asarray(reference[T][p]), color=MODEL_COLORS["reference"], lw=2.2,
                        label=MODEL_LABELS["reference"])
                for m, res in run["models"].items():
                    ax.plot(rho, scale[p] * np.asarray(res["isotherms"][T][p]), color=MODEL_COLORS.get(m), lw=1.2,
                            label=MODEL_LABELS.get(m, m))
                ax.axvspan(800, rho.max(), color="#e4e2dc", alpha=0.5, lw=0)
                if p == "cp":
                    ax.set_yscale("log")
                ax.set_xlabel("rho [kg/m3]")
                ax.set_ylabel(units[p])
                ax.set_title(f"T = {T} K" + (" (extrapolated)" if float(T) > 335 else ""))
                ax.grid(True)
        axes[0, 0].legend(fontsize=7)
        if title:
            fig.suptitle(title)
        fig.tight_layout()
    return _save(fig, path)


def paired_figure(summaries: Mapping[str, Mapping], model: str = "potential", path: str | None = None):
    """Paired log10 error ratio of ``model`` against the direct surrogate (mean of the six properties), with
    95 % intervals, per configuration and region: below zero the potential is more accurate."""
    with mpl.rc_context(LIGHT_RC):
        fig, ax = make_figure(1, 1, figsize=(8.0, 3.8))
        regions = list(REGION_LABELS)
        x = np.arange(len(regions))
        w = 0.8 / len(summaries)
        cmap = get_cmap("viridis")
        for i, (name, s) in enumerate(summaries.items()):
            if model not in s["paired"]:
                continue
            iv = [s["paired"][model][g]["mean"] for g in regions]
            mean = np.array([v["mean"] for v in iv])
            err = np.vstack([mean - np.array([v["lo"] for v in iv]), np.array([v["hi"] for v in iv]) - mean])
            ax.bar(x + (i - (len(summaries) - 1) / 2) * w, mean, w, yerr=err, capsize=3,
                   color=cmap(0.15 + 0.7 * i / max(1, len(summaries) - 1)), label=name)
        ax.axhline(0, color="#52514e", lw=1)
        ax.set_xticks(x, [REGION_LABELS[g] for g in regions], fontsize=8)
        ax.set_ylabel(f"log10(error {MODEL_LABELS[model]} / error direct)")
        ax.grid(True, axis="y")
        ax.legend(fontsize=8)
        fig.tight_layout()
    return _save(fig, path)

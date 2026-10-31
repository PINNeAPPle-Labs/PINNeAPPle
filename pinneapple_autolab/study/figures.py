"""Figures of a surrogate study (matplotlib, PNG): validation against measurements, grid convergence, parity and
calibration per family, sensitivity, the Pareto front with its verification, the error drop of added runs and the
posterior of an inverse problem. One colour per family in a fixed order; measurements in ink; one axis per chart.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = ["validation", "grid_convergence", "parity", "calibration", "sobol", "pareto", "error_drop", "posterior",
           "SERIES", "INK", "MUTED"]

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _ax(figsize=(6.4, 4.0)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig, path: str, title: str, ax) -> str:
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def validation(path: str, measured: Mapping[float, float], cfd: Mapping[str, Sequence[tuple[float, float]]],
               band: float | None = None, xlabel: str = "slant angle [deg]", ylabel: str = "drag coefficient CD",
               title: str = "CFD against the wind tunnel", source: str = "") -> str:
    """Measured points (ink) and one line per CFD series ``{label: [(x, y), ...]}``; ``band``: a relative
    numerical uncertainty (e.g. the GCI) drawn around the first series."""
    fig, ax = _ax()
    for k, (lab, pts) in enumerate(cfd.items()):
        x, y = np.array(pts, float).T
        o = np.argsort(x)
        ax.plot(x[o], y[o], "-o", color=SERIES[k], lw=2, ms=5, label=lab)
        if band and k == 0:
            ax.fill_between(x[o], y[o] * (1 - band), y[o] * (1 + band), color=SERIES[k], alpha=0.15, lw=0,
                            label=f"{lab}: grid uncertainty (GCI {band:.1%})")
    x, y = np.array(sorted(measured.items())).T
    ax.plot(x, y, "s", color=INK, ms=7, label="measured" + (f" ({source})" if source else ""))
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path, title, ax)


def grid_convergence(path: str, study: Mapping[str, Any], quantity: str, measured: float | None = None,
                     title: str = "Grid convergence") -> str:
    """``quantity`` against the relative cell size h (finest = 1) with the Richardson value at h = 0."""
    fig, ax = _ax()
    h = np.array([m["h_rel"] for m in study["meshes"]])
    q = study[quantity]
    ax.plot(h, q["values"], "-o", color=SERIES[0], lw=2, ms=6, label="CFD")
    for m, hh, v in zip(study["meshes"], h, q["values"], strict=False):
        ax.annotate(f"{m['name']} ({m['cells'] / 1e3:.0f} k)", (hh, v), textcoords="offset points", xytext=(6, 4),
                    fontsize=7, color=MUTED)
    if q.get("extrapolated") is not None:
        ax.plot([0], [q["extrapolated"]], "D", color=SERIES[1], ms=7,
                label=f"Richardson ({q['extrapolated']:.4f}, p = {q['p']:.2f})")
        ax.plot([0, h[-1]], [q["extrapolated"], q["values"][-1]], ":", color=SERIES[1], lw=1.5)
    if measured is not None:
        ax.axhline(measured, color=INK, lw=1, ls="--", label=f"measured {measured:g}")
    ax.set_xlim(left=-0.1)
    ax.set_xlabel("relative cell size h (finest = 1)", color=MUTED, fontsize=9)
    ax.set_ylabel(quantity, color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path, title, ax)


def parity(path: str, rows: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]], label: str,
           title: str = "Predicted against simulated") -> str:
    """``{family: (true, pred, half_width)}``: points with their 90 % intervals, y = x."""
    fig, ax = _ax((5.0, 4.6))
    allv = np.concatenate([np.r_[t, p] for t, p, _ in rows.values()])
    lo, hi = allv.min(), allv.max()
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, lw=1)
    for k, (fam, (t, p, h)) in enumerate(rows.items()):
        ax.errorbar(t, p, yerr=h, fmt="o", ms=5, color=SERIES[k], ecolor=SERIES[k], elinewidth=1, alpha=0.9,
                    label=fam, mec=SURFACE, mew=0.8)
    ax.set_xlabel(f"{label} (CFD)", color=MUTED, fontsize=9)
    ax.set_ylabel(f"{label} (surrogate, 90 % interval)", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path, title, ax)


def calibration(path: str, coverage: Mapping[str, Mapping[str, float]], target: float = 0.9,
                title: str = "Coverage of the 90 % intervals") -> str:
    """Grouped bars ``{family: {split: coverage}}`` with the nominal level."""
    fig, ax = _ax()
    fams = list(coverage)
    splits = list(next(iter(coverage.values())))
    w = 0.8 / len(splits)
    for j, s in enumerate(splits):
        ax.bar(np.arange(len(fams)) + j * w, [coverage[f][s] for f in fams], w * 0.92, color=SERIES[j], label=s)
    ax.axhline(target, color=INK, lw=1, ls="--", label=f"nominal {target:.0%}")
    ax.set_xticks(np.arange(len(fams)) + w * (len(splits) - 1) / 2, fams, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction inside the interval", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path, title, ax)


def sobol(path: str, s: Mapping[str, Any], output: str, title: str | None = None) -> str:
    fig, ax = _ax()
    names = s["inputs"]
    d = s[output]
    x = np.arange(len(names))
    for j, (k, lab) in enumerate((("S1", "first order"), ("ST", "total"))):
        v = np.array(d[k])
        ci = np.array(d[f"{k}_ci"])
        ax.bar(x + (j - 0.5) * 0.38, v, 0.36, color=SERIES[j], label=lab,
               yerr=[v - ci[0], ci[1] - v], ecolor=INK, capsize=2)
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylabel("Sobol index", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path, title or f"What drives {output}", ax)


def pareto(path: str, front: Sequence[Mapping[str, float]], objectives: Sequence[str],
           verified: Sequence[Mapping[str, Any]] = (), title: str = "Pareto front (surrogate) and CFD checks") -> str:
    """Front predicted by the surrogate; ``verified``: ``[{"predicted": {o: v}, "cfd": {o: v}}, ...]``."""
    fig, ax = _ax()
    a, b = objectives
    F = np.array([[p[a], p[b]] for p in front])
    ax.plot(F[:, 0], F[:, 1], "-", color=SERIES[0], lw=2, label="surrogate front")
    for k, v in enumerate(verified):
        pa, pb = v["predicted"][a], v["predicted"][b]
        ax.plot(pa, pb, "o", color=SERIES[0], ms=8, mec=SURFACE, label="pick (predicted)" if k == 0 else None)
        if v.get("cfd"):
            ca, cb = v["cfd"][a], v["cfd"][b]
            ax.plot(ca, cb, "s", color=INK, ms=7, label="pick (CFD)" if k == 0 else None)
            ax.annotate("", (ca, cb), (pa, pb), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax.set_xlabel(a, color=MUTED, fontsize=9)
    ax.set_ylabel(b, color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path, title, ax)


def error_drop(path: str, curves: Mapping[str, Sequence[tuple[int, float]]], ylabel: str,
               title: str = "Held-out error as runs are added") -> str:
    fig, ax = _ax()
    for k, (lab, pts) in enumerate(curves.items()):
        n, e = np.array(pts, float).T
        ax.plot(n, e, "-o", color=SERIES[k], lw=2, ms=6, label=lab)
    ax.set_xlabel("CFD runs in the training set", color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path, title, ax)


def posterior(path: str, samples: Mapping[str, np.ndarray], truth: float | None = None, xlabel: str = "",
              extra: Mapping[str, float] | None = None, title: str = "Posterior") -> str:
    """Histograms of posterior samples ``{method: samples}`` with the true value and point estimates ``extra``."""
    fig, ax = _ax()
    allv = np.concatenate(list(samples.values()))
    bins = np.linspace(allv.min(), allv.max(), 50)
    for k, (lab, s) in enumerate(samples.items()):
        ax.hist(s, bins=bins, density=True, histtype="step", lw=2, color=SERIES[k], label=lab)
    if truth is not None:
        ax.axvline(truth, color=INK, lw=1.5, ls="--", label=f"true {truth:g}")
    for k, (lab, v) in enumerate((extra or {}).items()):
        ax.axvline(v, color=SERIES[(len(samples) + k) % len(SERIES)], lw=1.5, ls=":", label=f"{lab} {v:.2f}")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel("density", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path, title, ax)

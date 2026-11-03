"""Diagnostic figure of an extreme-climate workflow result: the QQ plot on the Gumbel (GEV) or exponential (GPD)
scale and the return-level curve with its delta-method band, the profile intervals and the empirical points."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["diagnostic_figure"]


def diagnostic_figure(result: Mapping[str, Any]):
    """A two-panel matplotlib figure of :func:`.workflow.extreme_climate_workflow`'s result."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    g = result["diagnostics"]
    c = result["return_level_curve"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    q, e = np.asarray(g["model"]), np.asarray(g["empirical"])
    a.plot(q, e, "o", ms=3, color="C0")
    lim = [min(q.min(), e.min()), max(q.max(), e.max())]
    a.plot(lim, lim, "-", color="0.4", lw=1)
    a.set_xlabel(f"model quantile ({g['scale']} scale)")
    a.set_ylabel("empirical quantile")
    a.set_title(f"QQ: r = {g['qq_correlation']:.4f}, A² = {g['anderson_darling']:.3f} (p = {g['ad_p_value']:.2f})")
    T = np.asarray(c["T"])
    b.fill_between(T, c["lower"], c["upper"], color="C0", alpha=0.2, label=f"{result['level']:.0%} {c['band']}")
    b.plot(T, c["level"], "-", color="C0", label=result["fit"]["distribution"])
    rl = result["return_levels"]
    b.errorbar([r["T"] for r in rl], [r["level"] for r in rl],
               yerr=[[r["level"] - r["profile"][0] for r in rl], [r["profile"][1] - r["level"] for r in rl]],
               fmt="s", color="C3", ms=4, capsize=3, label="profile likelihood")
    if "empirical_T" in c:
        b.plot(c["empirical_T"], c["empirical_level"], "k.", ms=4, label="data")
    b.set_xscale("log")
    b.set_xlabel("return period (years)")
    unit = result["data"].get("units", "")
    b.set_ylabel(f"return level{f' ({unit})' if unit else ''}")
    ref = result.get("reference") or {}
    b.set_title("return levels" + (f" at {', '.join(f'{k} = {v:g}' for k, v in ref.items())}" if
                                   result["fit"].get("mu_covariates") or result["fit"].get("sigma_covariates") else ""))
    b.legend(fontsize=8)
    fig.tight_layout()
    return fig

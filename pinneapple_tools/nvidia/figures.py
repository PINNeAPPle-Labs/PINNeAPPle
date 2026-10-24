"""Result figures shared by the NVIDIA examples: field panels, check tables, training curves.

``field_panels``  a row of 2-D fields with their own colour bars (``imshow``, x right, y up).
``checks_figure`` the checks of a result as a horizontal bar chart of value / limit (green pass, red fail) with the
                  decision in the title; works for :class:`pinneapple_physics.darcy_checks.DarcyCheck`, the
                  ``pinneapple_analysis.trust`` ``Check`` objects and plain dicts ``{name, value, limit, passed}``.
``curves_figure`` one or more x-y series per panel (training loss, rollout error, trajectories).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = ["field_panels", "checks_figure", "curves_figure", "check_rows"]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def field_panels(fields: Sequence[np.ndarray], titles: Sequence[str], path: str, *, cmaps: Sequence[str] | None = None,
                 extent=(0, 1, 0, 1), suptitle: str = "", units: Sequence[str] | None = None,
                 shared: Sequence[Sequence[int]] = (), ncols: int | None = None, dpi: int = 130) -> str:
    """``fields[i][ix, iy]`` drawn with x to the right and y up; ``shared``: groups of panel indices on one range;
    ``ncols``: panels per row (default: one row)."""
    plt = _plt()
    n = len(fields)
    nc = ncols or n
    nr = -(-n // nc)
    fig, axs = plt.subplots(nr, nc, figsize=(3.3 * nc, 3.2 * nr), squeeze=False)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    axs = axs.ravel()
    rng: dict[int, tuple[float, float]] = {}
    for g in shared:
        lo = min(float(np.nanmin(fields[i])) for i in g)
        hi = max(float(np.nanmax(fields[i])) for i in g)
        rng.update({i: (lo, hi) for i in g})
    for i, (ax, f, t) in enumerate(zip(axs[:n], fields, titles, strict=True)):
        lo, hi = rng.get(i, (None, None))
        im = ax.imshow(np.asarray(f).T, origin="lower", extent=extent, cmap=(cmaps or ["inferno"] * n)[i],
                       vmin=lo, vmax=hi, aspect="equal")
        ax.set_title(t, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        if units:
            cb.set_label(units[i], fontsize=8)
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def check_rows(checks: Sequence[Any]) -> list[dict[str, Any]]:
    """Checks of any supported kind as ``[{name, value, limit, passed, detail}]``."""
    rows = []
    for c in checks:
        if isinstance(c, Mapping):
            d = dict(c)
        elif hasattr(c, "to_dict"):
            d = c.to_dict()
        else:
            d = {k: getattr(c, k, None) for k in ("name", "value", "limit", "passed", "status", "reason")}
        if d.get("passed") is None and d.get("status") is not None:
            d["passed"] = {"supports": True, "contradicts": False}.get(d["status"])
        rows.append({"name": d.get("name", "?"), "value": d.get("value"), "limit": d.get("limit"),
                     "passed": d.get("passed"), "detail": d.get("detail") or d.get("reason") or "",
                     "kind": d.get("kind", "max")})
    return rows


def checks_figure(checks: Sequence[Any], path: str, *, title: str = "Physics checks", decision: str = "",
                  dpi: int = 130) -> str:
    """Bar of value / limit per check (limit / value for a lower bound, ``"kind": "min"``; log scale; the dashed
    line is the limit). Checks without a limit are drawn as pass / fail markers."""
    plt = _plt()
    rows = check_rows(checks)
    fig, ax = plt.subplots(figsize=(8.2, 0.45 * len(rows) + 1.4))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows, strict=True):
        col = "#2e9d4f" if r["passed"] else ("#c0392b" if r["passed"] is False else "#999999")
        v, lim = r["value"], r["limit"]
        if v is not None and lim not in (None, 0) and np.isfinite(float(v)):
            lower = r["kind"] == "min"
            ratio = abs(float(lim)) / max(abs(float(v)), 1e-12) if lower else abs(float(v)) / abs(float(lim))
            ratio = min(max(ratio, 1e-4), 1e3)
            ax.barh(yi, ratio, color=col, height=0.6)
            ax.text(ratio * 1.15, yi, f"{float(v):.3g} {'>=' if lower else '<='} {float(lim):.3g}", va="center",
                    fontsize=7.5)
        else:
            ax.barh(yi, 1.0, color=col, height=0.6, alpha=0.5)
            txt = "pass" if r["passed"] else ("fail" if r["passed"] is False else "not run")
            ax.text(1.15, yi, txt + (f" ({float(v):.3g})" if isinstance(v, (int, float)) else ""), va="center",
                    fontsize=7.5)
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1e3)
    ax.set_yticks(y)
    ax.set_yticklabels([r["name"] for r in rows], fontsize=8)
    ax.set_xlabel("value / limit (pass when left of the dashed line)")
    ax.set_title(title + (f": {decision}" if decision else ""), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def curves_figure(panels: Sequence[Mapping[str, Any]], path: str, *, suptitle: str = "", dpi: int = 130) -> str:
    """``panels``: ``[{"title", "xlabel", "ylabel", "series": [(label, x, y, style?)], "logy"?, "hlines"?}]``."""
    plt = _plt()
    fig, axs = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 3.4))
    for ax, p in zip(np.atleast_1d(axs), panels, strict=True):
        for s in p["series"]:
            lab, x, yv = s[:3]
            ax.plot(x, yv, s[3] if len(s) > 3 else "-", label=lab, lw=1.5, ms=4)
        for h in p.get("hlines", ()):
            ax.axhline(h[0], color=h[2] if len(h) > 2 else "k", ls="--", lw=1, label=h[1])
        if p.get("logy"):
            ax.set_yscale("log")
        ax.set_title(p.get("title", ""), fontsize=9)
        ax.set_xlabel(p.get("xlabel", ""))
        ax.set_ylabel(p.get("ylabel", ""))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path

"""Figures and the engineering report of a lab session, drawn only from its snapshot (ledger runs, candidates,
model board, decisions): nothing here computes a physical number.

* :func:`fig_pareto`: every surrogate candidate (crosses) in drag x downforce, the surrogate Pareto front (circles),
  the simulated designs by purpose, the baseline and the best simulated design, the downforce limit.
* :func:`fig_models`: cross-validated prediction against the reference per output, with the 95 % intervals, and
  the score of every surrogate family.
* :func:`fig_learning`: the error of each acquisition's prediction (made before its simulation) and the best
  simulated feasible objective against the simulations spent.
* :func:`fig_field_triptych`: reference | prediction | error of a surface field (top and bottom views).
* :func:`fig_trust_cards`: three designs the lab judged differently (favourable, uncertain, out of domain), what
  was predicted, what the reference then said.
* :func:`write_report`: the Markdown report.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = ["pareto_front", "fig_pareto", "fig_models", "fig_learning", "fig_field_triptych", "fig_trust_cards",
           "write_report", "STYLE"]

STYLE = {"bg": "#0d0f14", "panel": "#151821", "ink": "#e8e6e1", "muted": "#9a988f", "grid": "#2a2e3a",
         "candidate": "#9085e9", "front": "#19c79a", "doe": "#c3c2b7", "acquisition": "#eb6834",
         "verification": "#3987e5", "baseline": "#ffffff", "best": "#eda100", "limit": "#e87ba4",
         "favourable": "#19c79a", "uncertain": "#eda100", "out_of_domain": "#e87ba4", "unlikely": "#5d5b55"}
PURPOSE_MARK = {"baseline": ("*", 260), "doe": ("o", 46), "acquisition": ("D", 56), "verification": ("P", 90),
                "probe": ("s", 40)}


def _ax_style(ax, title: str = ""):
    ax.set_facecolor(STYLE["panel"])
    for s in ax.spines.values():
        s.set_color(STYLE["grid"])
    ax.tick_params(colors=STYLE["muted"], labelsize=8)
    ax.xaxis.label.set_color(STYLE["ink"])
    ax.yaxis.label.set_color(STYLE["ink"])
    ax.grid(color=STYLE["grid"], lw=0.5, alpha=0.7)
    if title:
        ax.set_title(title, color=STYLE["ink"], fontsize=10, loc="left")


def _fig(w, h, n=1, m=1, **kw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(n, m, figsize=(w, h), **kw)
    fig.patch.set_facecolor(STYLE["bg"])
    return fig, axes


def _save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def pareto_front(F: np.ndarray) -> np.ndarray:
    """Indices of the non-dominated rows of ``F`` (all columns minimised), sorted by the first column."""
    F = np.asarray(F, float)
    idx = np.argsort(F[:, 0])
    front, best = [], np.inf
    for i in idx:
        if F[i, 1] < best:
            front.append(i)
            best = F[i, 1]
    return np.array(front, int)


def fig_pareto(snap: Mapping[str, Any], path: str, x: str = "drag_N", y: str = "downforce_N", scale: float = 1e-3,
               units: str = "kN", annotate: Mapping[str, int] | None = None) -> str:
    cand = snap["candidates"]
    fig, ax = _fig(7.6, 5.6)
    _ax_style(ax, snap.get("title", "Drag x downforce: surrogate candidates and simulated designs"))
    cx, cy = np.asarray(cand[f"{x}_mean"]) * scale, np.asarray(cand[f"{y}_mean"]) * scale
    ax.scatter(cx, cy, marker="x", s=10, lw=0.6, color=STYLE["candidate"], alpha=0.45,
               label=f"surrogate candidates ({len(cx)})")
    fr = np.asarray(snap.get("front_idx", []), int)
    if len(fr):
        ax.plot(cx[fr], cy[fr], color=STYLE["front"], lw=1.0, alpha=0.8)
        ax.scatter(cx[fr], cy[fr], s=34, facecolor="none", edgecolor=STYLE["front"], lw=1.4,
                   label="surrogate Pareto front (balance within limits)")
    runs = [r for r in snap["runs"] if r["ok"] and r["purpose"] != "probe"]
    for purpose in ("doe", "acquisition", "verification"):
        rr = [r for r in runs if r["purpose"] == purpose]
        if rr:
            mk, s = PURPOSE_MARK[purpose]
            ax.scatter([r["outputs"][x] * scale for r in rr], [r["outputs"][y] * scale for r in rr], marker=mk, s=s,
                       color=STYLE[purpose], edgecolor=STYLE["bg"], lw=0.8, zorder=5,
                       label=f"CFD: {purpose} ({len(rr)})")
    base = next((r for r in runs if r["purpose"] == "baseline"), None)
    if base:
        bx, by = base["outputs"][x] * scale, base["outputs"][y] * scale
        ax.scatter([bx], [by], marker="*", s=300, color=STYLE["baseline"], edgecolor=STYLE["bg"], zorder=6,
                   label="baseline (CFD)")
        lim = snap.get("limits", {}).get(y)
        if lim is not None:
            ax.axhline(lim * scale, color=STYLE["limit"], ls="--", lw=1.0, label=f"{y.replace('_N', '')} limit")
    best = snap.get("best")
    if best:
        ax.scatter([best["outputs"][x] * scale], [best["outputs"][y] * scale], marker="*", s=360,
                   color=STYLE["best"], edgecolor=STYLE["bg"], zorder=7, label=f"best simulated ({best['id']})")
    for lab, i in (annotate or {}).items():
        ax.annotate(lab, (cx[i], cy[i]), xytext=(8, 8), textcoords="offset points", color=STYLE["ink"], fontsize=11,
                    fontweight="bold", arrowprops={"arrowstyle": "-", "color": STYLE["ink"], "lw": 0.6})
    ax.set_xlabel(f"D [{units}]")
    ax.set_ylabel(f"-L [{units}]")
    leg = ax.legend(fontsize=7, facecolor=STYLE["panel"], edgecolor=STYLE["grid"], labelcolor=STYLE["ink"],
                    loc="lower right")
    leg.get_frame().set_alpha(0.9)
    fig.tight_layout()
    return _save(fig, path)


def fig_models(snap: Mapping[str, Any], path: str, outputs: Sequence[str], labels: Mapping[str, str] | None = None,
               scale: Mapping[str, float] | None = None) -> str:
    board, cvp = snap["board"], snap["cv_pred"]
    fig, axes = _fig(4.2 * len(outputs), 4.6, 1, len(outputs))
    axes = np.atleast_1d(axes)
    for ax, o in zip(axes, outputs, strict=True):
        sc = (scale or {}).get(o, 1.0)
        fam = board["chosen"][o]
        y = np.asarray(cvp[o]["y"]) * sc
        _ax_style(ax, f"{(labels or {}).get(o, o)}: k-fold held-out")
        lo, hi = y.min(), y.max()
        pad = 0.08 * (hi - lo + 1e-12)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=STYLE["muted"], lw=0.8, ls=":")
        colors = {"gp": STYLE["front"], "mlp_ensemble": STYLE["candidate"], "quadratic": STYLE["acquisition"]}
        for f, pr in cvp[o]["families"].items():
            pr = np.asarray(pr) * sc
            m = board["table"][o][f]
            lab = (f"{f}: RMSE {m['rmse'] * sc:.3g}, rel {100 * m['rel_mae']:.1f} %, cov95 "
                   f"{100 * m['coverage95']:.0f} %" + ("  <- chosen" if f == fam else ""))
            if f == fam:
                ax.errorbar(y, pr[:, 0], yerr=1.96 * pr[:, 1], fmt="o", ms=4, color=colors.get(f, STYLE["ink"]),
                            ecolor=colors.get(f, STYLE["ink"]), elinewidth=0.7, alpha=0.95, label=lab, zorder=4)
            else:
                ax.scatter(y, pr[:, 0], s=12, color=colors.get(f, STYLE["ink"]), alpha=0.6, label=lab)
        ax.set_xlabel("reference (CFD)")
        ax.set_ylabel("prediction (not trained on this point)")
        ax.legend(fontsize=6.5, facecolor=STYLE["panel"], edgecolor=STYLE["grid"], labelcolor=STYLE["ink"],
                  loc="upper left")
    fig.tight_layout()
    return _save(fig, path)


def fig_learning(snap: Mapping[str, Any], path: str, outputs: Sequence[str], objective: str,
                 acceptance: Mapping[str, float]) -> str:
    fig, (a1, a2) = _fig(10.5, 4.0, 1, 2)
    _ax_style(a1, "Error of each prediction, measured by the simulation that followed it")
    dec = [d for d in snap["decisions"] if d.get("rel_error")]
    cols = [STYLE["front"], STYLE["acquisition"], STYLE["candidate"], STYLE["best"]]
    for c, o in zip(cols, outputs, strict=False):
        xs = [d["n_train"] for d in dec if o in d["rel_error"]]
        ys = [100 * d["rel_error"][o] for d in dec if o in d["rel_error"]]
        a1.plot(xs, ys, "-o", ms=4, lw=1, color=c, label=o)
        if o in acceptance:
            a1.axhline(100 * acceptance[o], color=c, ls="--", lw=0.7, alpha=0.7)
    a1.set_xlabel("simulations in the training set")
    a1.set_ylabel("|prediction - CFD| / CFD [%]")
    a1.legend(fontsize=7, facecolor=STYLE["panel"], edgecolor=STYLE["grid"], labelcolor=STYLE["ink"])
    _ax_style(a2, f"Best simulated feasible {objective} vs simulations spent")
    xs, ys, best = [], [], np.inf
    lim = snap.get("limits_all", {})
    for k, r in enumerate([r for r in snap["runs"] if r["purpose"] != "probe"], 1):
        if r["ok"] and all(lo <= r["outputs"].get(n, np.nan) <= hi for n, (lo, hi) in lim.items()):
            best = min(best, r["outputs"][objective])
        xs.append(k)
        ys.append(best if np.isfinite(best) else np.nan)
    a2.step(xs, ys, where="post", color=STYLE["best"], lw=1.6)
    for k, r in enumerate([r for r in snap["runs"] if r["purpose"] != "probe"], 1):
        if not r["ok"]:
            a2.axvline(k, color=STYLE["limit"], lw=0.8, alpha=0.6)
    a2.set_xlabel("reference simulations used (red: failed run)")
    a2.set_ylabel(objective)
    fig.tight_layout()
    return _save(fig, path)


def fig_field_triptych(points: np.ndarray, ref: np.ndarray, pred: np.ndarray, path: str, title: str = "",
                       clim: tuple[float, float] = (-1.5, 1.0), err_max: float = 0.3, label: str = "Cp",
                       mirror: bool = True) -> str:
    """Reference | prediction | |error| from above and from below (half-model points mirrored), colormap jet."""
    import matplotlib.pyplot as plt
    P = np.asarray(points, float)
    vals = [np.asarray(ref, float), np.asarray(pred, float), np.abs(np.asarray(pred) - np.asarray(ref))]
    if mirror:
        P = np.vstack([P, P * [1, -1, 1]])
        vals = [np.r_[v, v] for v in vals]
    fig, axes = _fig(15.0, 4.6, 2, 3, layout="constrained")
    names = ["Numerical (CFD)", "Prediction (surrogate)", "|Error|"]
    for row, (view, order) in enumerate((("from above", np.argsort(P[:, 2])), ("from below", np.argsort(-P[:, 2])))):
        for col in range(3):
            ax = axes[row, col]
            _ax_style(ax)
            ax.grid(False)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            v = vals[col][order]
            cm, lim = ("jet", clim) if col < 2 else ("jet", (0.0, err_max))
            yy = P[order, 1] if row == 0 else -P[order, 1]
            sc = ax.scatter(P[order, 0], yy, c=v, s=3.2, cmap=cm, vmin=lim[0], vmax=lim[1], linewidths=0)
            if row == 0:
                ax.set_title(names[col], color=STYLE["ink"], fontsize=10)
            if col == 0:
                ax.set_ylabel(view, color=STYLE["ink"])
            if row == 1:
                cb = plt.colorbar(sc, ax=axes[:, col], orientation="horizontal", fraction=0.06, pad=0.02, aspect=40)
                cb.ax.tick_params(colors=STYLE["muted"], labelsize=7)
                cb.set_label(label if col < 2 else f"|{label} error|", color=STYLE["ink"], fontsize=8)
    if title:
        fig.suptitle(title, color=STYLE["ink"], fontsize=11)
    return _save(fig, path)


def fig_trust_cards(cards: Sequence[Mapping[str, Any]], path: str, outputs: Sequence[str],
                    units: Mapping[str, str] | None = None) -> str:
    """One panel per card: ``{"tag", "label", "reasons", "predicted", "std", "reference", "decision", "action"}``;
    bars = prediction +- 1.96 std, diamond = reference (when simulated)."""
    fig, axes = _fig(4.6 * len(cards), 4.8, 1, len(cards))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cards, strict=True):
        col = STYLE.get(c["label"], STYLE["ink"])
        _ax_style(ax)
        ax.grid(False)
        ax.set_title(f"{c['tag']}  {c['label'].replace('_', ' ')}", color=col, fontsize=12, loc="left",
                     fontweight="bold")
        for k, o in enumerate(outputs):
            m, s = c["predicted"][o], c["std"][o]
            ref = (c.get("reference") or {}).get(o)
            base = c.get("baseline", {}).get(o, m)
            norm = abs(base) or 1.0
            yk = len(outputs) - k
            ax.errorbar([100 * (m - base) / norm], [yk], xerr=[100 * 1.96 * s / norm], fmt="o", color=col, ms=6,
                        capsize=4, lw=1.4)
            if ref is not None:
                ax.scatter([100 * (ref - base) / norm], [yk], marker="D", s=50, color=STYLE["ink"], zorder=5)
            ax.text(0.02, yk + 0.28, f"{o}  pred {m:.4g} +- {1.96 * s:.2g}"
                    + (f"  |  CFD {ref:.4g}" if ref is not None else "  |  not simulated"),
                    transform=ax.get_yaxis_transform(), color=STYLE["ink"], fontsize=7)
        ax.axvline(0, color=STYLE["muted"], lw=0.7, ls=":")
        ax.set_ylim(0.3, len(outputs) + 0.8)
        ax.set_yticks([])
        ax.set_xlabel("change vs baseline [%]  (bar: 95 % interval, diamond: CFD)")
        txt = "\n".join(_wrap(r, 58) for r in c.get("reasons", [])[:3])
        ax.text(0.0, -0.30, txt + f"\ntrust: {c.get('decision', '')}   ->   {c.get('action', '')}",
                transform=ax.transAxes, color=STYLE["muted"], fontsize=7, va="top")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return _save(fig, path)


def _wrap(s: str, n: int) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(s, n))


def write_report(summary: Mapping[str, Any], path: str, figures: Mapping[str, str] | None = None,
                 qualification: Mapping[str, Any] | None = None, extra: str = "") -> str:
    """The engineering report in Markdown: mission, plan, solver qualification, budget, models, decisions,
    recommendation (a simulated design), environment."""
    m = summary["mission"]
    L = [f"# {m['title']}", "", f"> {m['statement']}", ""]
    L += ["## Plan", ""]
    al = summary["plan"].get("allocation", {})
    L += [f"- Objective: {', '.join(summary['plan'].get('objectives', []))}",
          f"- Limits: {'; '.join(c['text'] for c in m['constraints'])}",
          f"- Budget: {al.get('total')} reference simulations = 1 baseline + {al.get('doe')} space-filling + "
          f"{al.get('acquisition')} chosen one at a time + {al.get('verification')} final verifications",
          f"- Fixed conditions: {', '.join(f'{k} = {v}' for k, v in m['conditions'].items())}",
          "- Hypotheses to test:"] + [f"  - {h}" for h in m.get("hypotheses", [])] + [""]
    if qualification:
        L += ["## Solver qualification (baseline design, outside the budget)", "",
              "| variant | cells | wall [s] | " + " | ".join(qualification["outputs"]) + " |",
              "|---|---|---|" + "---|" * len(qualification["outputs"])]
        for n, r in qualification["variants"].items():
            vals = " | ".join((f"{r['outputs'][o]:.4g} ({100 * r['rel_diff_vs_reference'].get(o, 0):+.1f} %)"
                               if r["ok"] else "failed") for o in qualification["outputs"])
            L.append(f"| {n} | {r.get('cells') or ''} | {r['cost_s']:.0f} | {vals} |")
        L += ["", f"Differences are relative to the `{qualification['reference']}` variant; the campaign uses "
              f"`{qualification['campaign']}`.", ""]
    b = summary["budget"]
    L += ["## Budget", "", f"{b['used_runs']} of {b['max_runs']} reference simulations, {b['solver_wall_s']:.0f} s "
          f"of solver wall time, {b['failed_runs']} failed.", ""]
    if summary.get("board"):
        bd = summary["board"]
        L += ["## Surrogates (k-fold cross-validation on the final data)", "",
              "| output | family | RMSE | rel. error | R² | 95 % coverage | fit [s] |", "|---|---|---|---|---|---|---|"]
        for o, fams in bd["table"].items():
            for f, v in fams.items():
                if "skipped" in v:
                    continue
                star = " **(chosen)**" if bd["chosen"][o] == f else ""
                L.append(f"| {o} | {f}{star} | {v['rmse']:.4g} | {100 * v['rel_mae']:.2f} % | {v['r2']:.3f} | "
                         f"{100 * v['coverage95']:.0f} % | {v['fit_s']:.2f} |")
        L.append("")
    if summary.get("field_cv"):
        fc = summary["field_cv"]
        L += [f"Surface {fc['field']} surrogate: POD ({fc['n_modes']} modes, {100 * fc['energy']:.1f} % of the "
              f"variance) + {fc['family']}, {fc['k']}-fold held-out RMSE per region: "
              + ", ".join(f"{k} {v['rmse']:.3f}" for k, v in fc["per_region"].items()), ""]
    L += ["## Decisions (each prediction made before its simulation)", "",
          "| run | judged | rule | predicted -> CFD | after |", "|---|---|---|---|---|"]
    for d in summary["decisions"]:
        pv = "; ".join(f"{k} {d['predicted'][k]:.4g} -> {d['reference'][k]:.4g} ({100 * d['rel_error'][k]:.1f} %)"
                       for k in d.get("rel_error", {}) if k in m.get("acceptance", {}))
        L.append(f"| {d['run']} | {d['label_before']} | {d['rule'] or 'final verification'} | {pv} | "
                 f"{d['label_after']} |")
    L.append("")
    L += ["## Recommendation", "", summary["conclusion"], ""]
    if summary.get("improvement"):
        for k, v in summary["improvement"].items():
            L.append(f"- {k}: baseline {v['baseline']:.4g} -> {v['best']:.4g}"
                     + (f" ({v['change_pct']:+.2f} %)" if "change_pct" in v else f" (limit: {v['limit']})"))
        L.append("")
    if extra:
        L += [extra, ""]
    for k, f in (figures or {}).items():
        L += [f"![{k}]({os.path.relpath(f, os.path.dirname(os.path.abspath(path)))})", ""]
    env = summary.get("environment", {})
    L += ["## Environment", "", "```json", json.dumps(env, indent=1), "```", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(L))
    return path

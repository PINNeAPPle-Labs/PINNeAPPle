"""Summaries of a brick experiment: sensitivity to the joint model, the strategy comparison chart, a Markdown report."""
from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

from .goals import Goal
from .parts import JointModel

SERIES = {"evolution": "#2a78d6", "surrogate": "#eb6834"}     # categorical slots 1 and 2 of a validated palette
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def sensitivity(goal: Goal, asm, factors: Sequence[float] | None = None, shear_caps=(None, 1.0, 0.3)) -> list[dict]:
    """Re-evaluate ``asm`` with the clutch scaled over its assumed band and with capped stud shear."""
    base = goal.joint
    lo, hi = base.clutch_band
    factors = factors or (lo, 0.75, 1.0, 1.25, hi)
    rows = []
    for f in factors:
        for cap in shear_caps:
            j = JointModel(clutch_n=base.clutch_n * f, clutch_band=base.clutch_band, mu=base.mu, shear_n=cap)
            g = type(goal).__new__(type(goal))
            g.__dict__.update(goal.__dict__)
            g.joint = j
            ev = g.evaluate(asm)
            rows.append({"clutch_n": j.clutch_n, "clutch_factor": f, "shear_cap_n": cap, "feasible": ev.feasible,
                         "status": ev.status, "explanation": ev.explanation,
                         "load_factor": ev.metrics.get("load_factor"), "utilisation": ev.metrics.get("utilisation")})
    return rows


def plot_comparison(summaries: dict[str, dict], path: str, *, ylabels: dict[str, str] | None = None,
                    display: dict[str, dict] | None = None) -> str:
    """Best feasible score versus physics evaluations: one panel per goal, per strategy the median over the seeds (a
    seed without a feasible build counts as the worst, so the line starts once most seeds are feasible) and the range
    of the feasible seeds, with the feasible count at the end. ``display[goal]``: ``{"scale": -1, "invert": True}``
    to plot e.g. a part count stored as a negative score on an inverted axis (fewer is up)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    goals = list(summaries)
    fig, axes = plt.subplots(1, len(goals), figsize=(4.2 * len(goals), 3.7), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    n_seeds = 0
    for ax, g in zip(axes, goals, strict=True):
        ax.set_facecolor(SURFACE)
        disp = (display or {}).get(g, {})
        k = float(disp.get("scale", 1.0))
        ends = []
        for s, summ in summaries[g].items():
            C = np.asarray(summ["curves"], float)
            n_seeds = C.shape[0]
            x = np.arange(1, C.shape[1] + 1)
            ok = np.isfinite(C)
            med = np.median(np.where(ok, C, -np.inf), 0)
            med[~np.isfinite(med)] = np.nan
            with warnings.catch_warnings():                 # columns where no seed is feasible yet
                warnings.simplefilter("ignore", RuntimeWarning)
                lo = np.where(ok.any(0), np.nanmin(np.where(ok, C, np.nan), 0), np.nan)
                hi = np.where(ok.any(0), np.nanmax(np.where(ok, C, np.nan), 0), np.nan)
            col = SERIES.get(s, MUTED)
            ax.fill_between(x, k * lo, k * hi, color=col, alpha=0.16, linewidth=0)
            ax.plot(x, k * med, color=col, lw=2, label=s)
            n_ok = int(ok[:, -1].sum())
            last = np.where(np.isfinite(med))[0]
            if len(last):
                ends.append([k * med[last[-1]], x[last[-1]], f"{k * med[last[-1]]:.0f} ({n_ok}/{C.shape[0]} feasible)"])
        if disp.get("invert"):
            ax.invert_yaxis()
        top_first = disp.get("invert", False)
        ends.sort(reverse=top_first)
        span = np.ptp(ax.get_ylim())
        for j in range(1, len(ends)):                     # keep the end labels apart
            step = 0.07 * span
            ends[j][0] = (min if top_first else max)(ends[j][0], ends[j - 1][0] + (-step if top_first else step))
        for yv, xv, txt in ends:
            ax.annotate(txt, (xv, yv), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8,
                        color=INK)
        ax.set_title(g, fontsize=11, color=INK, loc="left")
        ax.set_xlabel("physics evaluations", fontsize=9, color=MUTED)
        ax.set_ylabel((ylabels or {}).get(g, "best score"), fontsize=9, color=MUTED)
        ax.grid(True, color="#e4e3df", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#c9c8c2")
        ax.tick_params(colors=MUTED, labelsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.005, f"Lines: median over {n_seeds} seeds (a seed without a feasible build counts as the worst); "
             "bands: range of the feasible seeds.", ha="center", va="bottom", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def _fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def write_report(path: str, *, validation: dict, comparisons: dict[str, dict], best: dict[str, dict],
                 sensitivities: dict[str, list[dict]], references: dict | None = None, files: dict | None = None,
                 joint: JointModel | None = None, cross_checks: dict[str, dict] | None = None,
                 notes: Sequence[str] = (), display: dict[str, dict] | None = None) -> str:
    """Markdown report: what was measured, the closed-form checks (and where they disagree), the strategy comparison
    with its spread over seeds, the best builds, the sensitivity to the joint model and its limits."""
    joint = joint or JointModel()
    L = ["# Self-modifying interlocking-brick builds", "",
         "Every candidate build is checked by a static equilibrium analysis (rigid parts, stud clutch, friction); "
         "rejected candidates are explained (which joint let go, or where the resultant left its support) and "
         "replayed with the Newton rigid-body engine.", "",
         "## Joint model", "",
         f"- Clutch per stud: {joint.clutch_n} N. Source: {joint.source}.",
         f"- Uncertainty: none published; the sensitivity study scales it by {joint.clutch_band[0]}x to "
         f"{joint.clutch_band[1]}x (an assumed band).",
         f"- Friction on smooth contacts: mu = {joint.mu} (assumed).",
         "- Limits: " + "; ".join(joint.notes) + ".", "",
         "## Checks against closed forms", "",
         f"Trust report decision: **{validation['report'].decisao}** "
         f"(score {validation['report'].score:.2f}, coverage {validation['report'].cobertura:.2f}).", "",
         "| check | simulated | closed form | rel. error | tolerance | pass | note |", "|---|---|---|---|---|---|---|"]
    for r in validation["rows"]:
        L.append(f"| {r['check']} | {_fmt(r['simulated'], 6)} {r['unit']} | {_fmt(r['reference'], 6)} | "
                 f"{r['rel_error']:.2e} | {r['tolerance']} | {'yes' if r['passed'] else '**no**'} | {r['note']} |")
    if validation.get("harmonic_newton"):
        L += ["", "Newton on the harmonic stack (offsets scaled by a factor; the closed form stands for factor <= 1):",
              "", "| factor | top-block drop [mm] | Newton | closed form | agree |", "|---|---|---|---|---|"]
        for r in validation["harmonic_newton"]:
            L.append(f"| {r['factor']} | {r['newton_top_drop_mm']:.2f} | {'stands' if r['newton_stands'] else 'falls'} "
                     f"| {'stands' if r['closed_form_stands'] else 'falls'} | {'yes' if r['agree'] else '**no**'} |")
    if cross_checks:
        L += ["", "## Statics against Newton on searched candidates", "",
              "Candidates the statics calls stable (nothing should move) or a mechanism (something should), replayed "
              "by Newton with every stud joint intact under the goal's loads. Joint failures are left out: Newton has "
              "no clutch model to judge them.", "",
              "| goal | stable, stays | stable, moves | mechanism, moves | mechanism, stays | agreement | dt [s] |",
              "|---|---|---|---|---|---|---|"]
        for g, cc in cross_checks.items():
            c = cc["confusion"]
            L.append(f"| {g} | {c['stable_and_stays']} | {c['stable_but_moves']} | {c['tips_and_moves']} | "
                     f"{c['tips_but_stays']} | {cc['agreement']:.0%} | {cc['dt']:g} |")
    if notes:
        L += ["", "Where the simulations disagree with each other or with the closed forms:", ""] + [f"- {n}" for n in notes]
    L += ["", "## Search strategies at equal evaluation budget", ""]
    L += ["Statistic: per strategy, the median of the best score over the seeds, a seed that found no feasible build "
          "counted as the worst (the same rule as the chart), and the range over the feasible seeds.", ""]
    for g, comp in comparisons.items():
        disp = (display or {}).get(g, {})
        k, unit = float(disp.get("scale", 1.0)), disp.get("unit", "")
        L += [f"### {g}", "", f"| strategy | median{unit} | range of feasible seeds | feasible seeds | per seed |",
              "|---|---|---|---|---|"]
        for s, summ in comp.items():
            per = ", ".join("-" if v is None or not np.isfinite(v) else f"{k * v:.1f}" for v in summ["per_seed"])
            med = summ.get("best_median", float("nan"))
            lo, hi = sorted((k * summ["best_min"], k * summ["best_max"]))
            n_ok = round(summ["success_rate"] * len(summ["seeds"]))
            L.append(f"| {s} | {'-' if not np.isfinite(med) else f'{k * med:.1f}'} | {lo:.1f} to {hi:.1f} | "
                     f"{n_ok}/{len(summ['seeds'])} | {per} |")
        L.append(f"\nBudget {next(iter(comp.values()))['budget']} physics evaluations per run, seeds "
                 f"{next(iter(comp.values()))['seeds']}.")
        if g in best:
            b = best[g]
            L += ["", f"Best build ({b['strategy']}, seed {b['seed']}): {b['n_bricks']} parts, "
                      f"{k * b['score']:.1f}{unit}. {b['explanation']}"]
        if references and g in references:
            L.append("Closed-form references: " + ", ".join(f"{k} {v:.1f} mm" for k, v in references[g].items()) + ".")
        if g in sensitivities:
            L += ["", "Sensitivity of the best build to the joint model:", "",
                  "| clutch [N] | shear cap [N/stud] | result | load factor | utilisation |", "|---|---|---|---|---|"]
            for r in sensitivities[g]:
                L.append(f"| {r['clutch_n']:.2f} | {_fmt(r['shear_cap_n'])} | "
                         f"{'stands' if r['feasible'] else r['status']} | {_fmt(r['load_factor'])} | "
                         f"{_fmt(r['utilisation'])} |")
        L.append("")
    if files:
        import os
        here = os.path.dirname(os.path.abspath(path))
        L += ["## Files", ""] + [f"- {k}: `{os.path.relpath(v, here)}`" for k, v in files.items()] + [""]
    with open(path, "w") as f:
        f.write("\n".join(L))
    return path

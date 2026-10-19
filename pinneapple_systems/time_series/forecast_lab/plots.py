"""Figures of a forecasting study (matplotlib, static PNG).

Families keep one colour everywhere (fixed order, never re-assigned when a family is missing from a panel); the
baseline is drawn as a neutral grey reference line at skill 0.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["FAMILY_COLORS", "plot_skill_vs_horizon", "plot_overfitting", "plot_surrogate_tests", "plot_pit",
           "plot_fan_charts", "plot_memorise_vs_learn", "calendar_grid", "plot_surprise_calendar"]

FAMILY_COLORS = {"data": "#2a78d6", "data+physics": "#eb6834", "physical": "#1baf7a", "hybrid": "#6250d6",
                 "baseline": "#52514e"}
_INK, _MUTED, _GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.grid(True, color=_GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_MUTED)
    ax.tick_params(colors=_MUTED, labelsize=8)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_skill_vs_horizon(results: Sequence[dict], path: str, *, lyapunov: dict[str, float] | None = None,
                          ncols: int = 4, floor: float = -0.5, titles: dict[str, str] | None = None,
                          reference: str = "best_baseline") -> str:
    """One panel per series: RMSE skill of each family's representative (chosen on the validation block) against
    the horizon, relative to ``reference`` -- ``"best_baseline"`` (at each horizon the baseline with the lowest
    validation RMSE) or ``"persistence"``; filled markers where the Holm-adjusted one-sided Diebold-Mariano test
    against that reference is significant at 5 %. The horizon is in Lyapunov times when ``lyapunov[series]`` (exponent per unit time) is
    given. Skill below ``floor`` is drawn at the floor."""
    plt = _plt()
    n = len(results)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.7 * nrows), squeeze=False)
    for ax, res in zip(axes.ravel(), results, strict=False):
        _style(ax)
        H = np.asarray(res["protocol"]["horizons"], float)
        lam = (lyapunov or {}).get(res["series"])
        x = H * res["dt"] * lam if lam else H
        ax.axhline(0, color=FAMILY_COLORS["baseline"], lw=1.2, ls="--")
        for fam, name in sorted(res.get("family_representative", {}).items(),
                                key=lambda kv: list(FAMILY_COLORS).index(kv[0])):
            if fam == "baseline":
                continue
            m = res["models"][name]
            key = "skill_vs_best_baseline" if reference == "best_baseline" else None
            sk = np.maximum(np.asarray(m[key] if key else m["test"]["skill_rmse"], float), floor)
            dm = m["dm_vs_best_baseline"] if key else m["dm_vs_persistence"]
            p = np.array([(d or {}).get("p_holm", np.nan) for d in dm], float)
            c = FAMILY_COLORS[fam]
            ax.plot(x, sk, color=c, lw=2, label=f"{fam}: {name}")
            sig = p < 0.05
            ax.scatter(x[sig], sk[sig], color=c, s=22, zorder=3)
            ax.scatter(x[~sig], sk[~sig], facecolor="white", edgecolor=c, s=22, zorder=3, lw=1.2)
        ax.set_xscale("log")
        ax.set_ylim(floor - 0.05, 1.02)
        ax.set_title((titles or {}).get(res["series"], res["series"]), fontsize=9, color=_INK)
        ax.set_xlabel("horizon [Lyapunov times]" if lam else "horizon [samples]", fontsize=8, color=_MUTED)
        ax.set_ylabel("RMSE skill vs " + ("best baseline" if key else "persistence"), fontsize=8, color=_MUTED)
        ax.legend(fontsize=6, frameon=False, loc="best")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_overfitting(studies: Sequence[dict], path: str) -> str:
    """Per series: train, validation and test error of every training variant (bars), and the cross-validation
    estimates of each scheme (markers) for the variants where they were computed."""
    plt = _plt()
    fig, axes = plt.subplots(len(studies), 1, figsize=(12, 3.4 * len(studies)), squeeze=False)
    cols = {"train": "#9ec5f0", "val": "#2a78d6", "test": "#0b3d7a"}
    marks = {"shuffled_kfold": ("v", "#e34948"), "walk_forward_no_purge": ("s", "#eda100"),
             "purged_walk_forward": ("o", "#1baf7a"), "purged_kfold": ("D", "#6250d6")}
    for ax, st in zip(axes[:, 0], studies, strict=False):
        _style(ax)
        names = list(st["variants"])
        xs = np.arange(len(names))
        w = 0.26
        for k, (key, c) in enumerate(cols.items()):
            ax.bar(xs + (k - 1) * w, [st["variants"][n][key] for n in names], w * 0.92, color=c, label=key)
        for sch, (mk, c) in marks.items():
            pts = [(i, st["variants"][n]["cv"][sch]["estimate"]) for i, n in enumerate(names)
                   if sch in st["variants"][n]["cv"]]
            if pts:
                i, v = zip(*pts, strict=False)
                ax.scatter(np.asarray(i) + w, v, marker=mk, color=c, edgecolor="white", s=46, zorder=4,
                           label=f"CV: {sch.replace('_', ' ')}")
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=7)
        ax.set_ylabel("RMSE / RMSE persistence", fontsize=8, color=_MUTED)
        ax.axhline(1, color=_MUTED, lw=1, ls="--")
        ax.set_title(st["series"], fontsize=9, color=_INK)
        ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_surrogate_tests(reports: dict[str, dict], path: str, *, ncols: int = 4) -> str:
    """Nonlinear prediction error of each series (vertical line) against its shuffled and IAAFT surrogates."""
    plt = _plt()
    names = list(reports)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.4 * nrows), squeeze=False)
    for ax, name in zip(axes.ravel(), names, strict=False):
        _style(ax)
        t = reports[name]["surrogate_tests"]
        for kind, c in (("shuffle", "#e34948"), ("iaaft", "#2a78d6")):
            s = np.asarray(t[f"npe_{kind}"]["surrogates"])
            ax.hist(s, bins=15, color=c, alpha=0.55, label=f"{kind} (p={t[f'npe_{kind}']['p_value']:.2f})")
        ax.axvline(t["npe_iaaft"]["observed"], color=_INK, lw=2, label="series")
        ax.set_title(name, fontsize=9, color=_INK)
        ax.set_xlabel("NN prediction error / std (h=1)", fontsize=8, color=_MUTED)
        ax.legend(fontsize=6, frameon=False)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_pit(entries: list[dict], path: str, *, ncols: int = 4) -> str:
    """PIT histograms. ``entries``: dicts with ``title``, ``freq`` (relative frequencies) and ``deviation``."""
    plt = _plt()
    nrows = int(np.ceil(len(entries) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.2 * nrows), squeeze=False)
    for ax, e in zip(axes.ravel(), entries, strict=False):
        _style(ax)
        f = np.asarray(e["freq"])
        nb = len(f)
        ax.bar((np.arange(nb) + 0.5) / nb, f * nb, 0.92 / nb, color=FAMILY_COLORS.get(e.get("family"), "#2a78d6"))
        ax.axhline(1, color=_MUTED, ls="--", lw=1)
        ax.set_ylim(0, max(2.5, float(f.max() * nb) * 1.1))
        ax.set_title(f"{e['title']}\ndeviation {e['deviation']:.2f}", fontsize=7, color=_INK)
        ax.set_xlabel("PIT", fontsize=7, color=_MUTED)
    for ax in axes.ravel()[len(entries):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_fan_charts(panels: list[dict], path: str, *, ncols: int = 2) -> str:
    """Forecast distributions on the test period, drawn only at the evaluated horizons: a marker at the median and
    error bars for the central 50 % (thick) and 90 % (thin) intervals, with no line joining the horizons (the
    forecasts are direct, made for those horizons only). Each panel: ``title``, ``t_hist``/``y_hist`` (observed up
    to the origin), ``t_future``/``y_future`` (what happened), ``horizons_t`` and ``quantiles`` (5 x n: 5, 25, 50,
    75, 95 %); a first column equal to the origin value is skipped."""
    plt = _plt()
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 2.7 * nrows), squeeze=False)
    for ax, p in zip(axes.ravel(), panels, strict=False):
        _style(ax)
        c = FAMILY_COLORS.get(p.get("family"), "#2a78d6")
        q = np.asarray(p["quantiles"])[:, 1:]
        ht = np.asarray(p["horizons_t"])[1:]
        ax.plot(p["t_hist"], p["y_hist"], color=_MUTED, lw=1.0, label="observed")
        ax.plot(p["t_future"], p["y_future"], color=_INK, lw=0.9, ls=":", label="truth (test)")
        ax.vlines(ht, q[0], q[4], color=c, lw=1.2, alpha=0.6, label="90 % interval")
        ax.vlines(ht, q[1], q[3], color=c, lw=4.0, alpha=0.8, label="50 % interval")
        ax.scatter(ht, q[2], color="white", edgecolor=c, s=26, zorder=4, lw=1.5, label="median")
        ax.set_title(p["title"], fontsize=8, color=_INK)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8, frameon=False)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_memorise_vs_learn(records: list[dict], path: str) -> str:
    """For each record of :func:`~.pipeline.memorise_vs_learn`: truth and the two variants' forecasts over the
    end of the training period (left) and the start of the untouched test period (right). One shared legend;
    each panel title carries the variants' RMSE on the whole training or test set."""
    plt = _plt()
    fig, axes = plt.subplots(len(records), 2, figsize=(11, 2.9 * len(records) + 0.5), squeeze=False, sharey="row",
                             layout="constrained")
    cols = ["#e34948", "#2a78d6"]
    for row, r in zip(axes, records, strict=False):
        for ax, part in zip(row, ("train", "test"), strict=True):
            _style(ax)
            idx = np.asarray(r[f"{part}_index"])
            ax.plot(idx, r[f"truth_{part}"], color=_INK, lw=1.0, label="truth")
            for c, (name, v) in zip(cols, r["variants"].items(), strict=False):
                ax.plot(idx, v[f"pred_{part}"], color=c, lw=1.3, alpha=0.9, label=name)
            rmse = ", ".join(f"{name.split(':')[0]} {v['rmse'][part]:.3f}" for name, v in r["variants"].items())
            ax.set_title(f"{r['title']}: {'training period (memorised?)' if part == 'train' else 'test period'}"
                         f", h = {r['horizon']}\nRMSE on the whole {part} set: {rmse}", fontsize=8, color=_INK)
            ax.set_xlabel("sample", fontsize=7, color=_MUTED)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), fontsize=8, frameon=False)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def calendar_grid(dates: list[str], values) -> dict:
    """Arrange daily ``values`` on a week (column) x weekday (row) grid; days without a value are NaN.
    Returns the grid (7, n_weeks), the Monday of each week and the day range."""
    import datetime as dt
    d = [dt.date.fromisoformat(x) for x in dates]
    start = d[0] - dt.timedelta(days=d[0].weekday())
    n_weeks = (d[-1] - start).days // 7 + 1
    grid = np.full((7, n_weeks), np.nan)
    for day, v in zip(d, values, strict=True):
        k = (day - start).days
        grid[k % 7, k // 7] = v
    return {"grid": grid, "week_starts": [str(start + dt.timedelta(weeks=w)) for w in range(n_weeks)],
            "first": str(d[0]), "last": str(d[-1])}


def plot_surprise_calendar(records: list[dict], path: str, *, vmax: float = 1.0) -> str:
    """Calendar heatmaps, one per record (``title``, ``dates``, ``abs_error_model``, ``abs_error_baseline``,
    ``model``, ``baseline``, ``horizon``): each day coloured by log2(|error of the model| / |error of the best
    baseline|) -- blue where the model was closer, red where the baseline was, grey midpoint at equal error, white
    where there is no forecast for that day."""
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("surprise", ["#2a78d6", "#e4e3df", "#e34948"])
    cmap.set_bad("white")
    fig, axes = plt.subplots(len(records), 1, figsize=(11, 2.4 * len(records) + 0.4), squeeze=False,
                             layout="constrained")
    im = None
    for ax, r in zip(axes[:, 0], records, strict=False):
        ratio = np.log2((np.asarray(r["abs_error_model"]) + 1e-12) / (np.asarray(r["abs_error_baseline"]) + 1e-12))
        cal = calendar_grid(r["dates"], np.clip(ratio, -4 * vmax, 4 * vmax))
        im = ax.imshow(np.ma.masked_invalid(cal["grid"]), aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
                       interpolation="nearest")
        ax.set_yticks(range(7))
        ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], fontsize=6)
        step = max(1, len(cal["week_starts"]) // 12)
        ax.set_xticks(range(0, len(cal["week_starts"]), step))
        ax.set_xticklabels(cal["week_starts"][::step], fontsize=6, rotation=30, ha="right")
        win = float(np.mean(ratio < 0))
        ax.set_title(f"{r['title']}\n{r['model']} vs {r['baseline']} at h = {r['horizon']} "
                     f"({cal['first']} to {cal['last']}; model closer on {win:.0%} of {len(ratio)} days)",
                     fontsize=8, color=_INK)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes[:, 0].tolist(), shrink=0.6, pad=0.01)
    cb.set_label("log2(|model error| / |baseline error|)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path

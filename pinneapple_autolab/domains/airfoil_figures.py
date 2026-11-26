"""The figures of the airfoil study, one per thing a stage measured.

Every stage of ``examples/end_to_end/airfoil_trust`` writes the picture of what it just measured into
``results/figures``, next to the JSON it writes: the plan of experiments after the campaign, the grid-convergence
study and the POD spectrum after the analysis, the parity and the pressure trace and the gate table after the physics
checks, the evidence map after the domain, the before and after after the second campaign, and the calibration after
the report. The report draws the same functions again from the same numbers, so a figure behind a stage and a figure
beside a verdict are one figure, not two implementations.

Nothing here decides anything. Every value comes from the caller, from the file the stage wrote; a function that needs
a number it was not given draws nothing rather than guessing one.
"""
from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (the backend is chosen before pyplot is imported)

BLUE, ORANGE, GREY, GREEN, RED = "#58a6ff", "#ffa657", "#8b949e", "#3fb950", "#f85149"

__all__ = ["style", "doe_plan", "mesh_convergence", "pod_spectrum", "accuracy", "error_trace",
           "physics_gates", "sensitivity", "domain_map", "extend", "uncertainty"]


def style() -> None:
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                         "axes.edgecolor": "#444", "axes.labelcolor": "#ddd", "text.color": "#ddd",
                         "xtick.color": "#bbb", "ytick.color": "#bbb", "figure.facecolor": "#0d1117",
                         "axes.facecolor": "#0d1117", "savefig.facecolor": "#0d1117"})


def _finish(fig, path: str) -> str:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def doe_plan(path: str, plan: dict) -> str:
    """The design space the campaign actually ran, and which split each design went to."""
    style_ = {"train": BLUE, "calibration": ORANGE, "validation": GREEN}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for ax, xkey, xlabel in ((axes[0], "alpha_deg", "angle of attack [deg]"),
                             (axes[1], "speed", "speed [m/s]")):
        for name, cases in plan["splits"].items():
            ax.plot([c[xkey] for c in cases], [100 * c["thickness"] for c in cases], "o", ms=4,
                    color=style_.get(name, GREY), label=f"{name} ({len(cases)})", alpha=0.9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("thickness [% chord]")
    axes[0].legend(fontsize=7, framealpha=0.2)
    n = sum(len(c) for c in plan["splits"].values())
    axes[0].set_title(f"{n} designs, split by whole designs", color="#ddd", fontsize=9)
    axes[1].set_title("the same designs against speed", color="#ddd", fontsize=9)
    return _finish(fig, path)


def mesh_convergence(path: str, mesh: dict) -> str:
    """CL and CD against the mesh family, with the Richardson estimate and the band the GCI allows."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, q, colour in ((axes[0], "CL", BLUE), (axes[1], "CD", ORANGE)):
        cells = [m["cells"] for m in mesh["levels"]]
        ax.semilogx(cells, [m["values"][q] for m in mesh["levels"]], "o-", color=colour, lw=1.6, ms=5,
                    label="OpenFOAM")
        stud = mesh["study"].get(q, {})
        ext = stud.get("extrapolated")
        if ext is not None:
            ax.axhline(ext, color=GREY, ls="--", lw=1.0, label=f"Richardson {ext:.4f}")
            gci = stud.get("gci_fine")
            if gci:
                ax.fill_between([min(cells) * 0.7, max(cells) * 1.3], ext * (1 - gci), ext * (1 + gci),
                                color=GREY, alpha=0.15, label=f"GCI {gci:.2%}")
        ax.set_xticks(cells)
        ax.set_xticklabels([f"{c / 1000:g}k" for c in cells], fontsize=8)
        ax.minorticks_off()
        ax.set_xlabel("cells")
        ax.set_ylabel(q)
        ax.legend(fontsize=7, framealpha=0.2)
        p = stud.get("p")
        ax.set_title(f"{q}: observed order {p:.2f}" if p else f"{q}: oscillatory, no order claimed",
                     color="#ddd", fontsize=9)
    return _finish(fig, path)


def pod_spectrum(path: str, basis: dict) -> str:
    """How much of the field the basis keeps, mode by mode: the analysis before any network is involved."""
    s = np.asarray(basis["singular_values"], float) ** 2
    frac = s / s.sum()
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    modes = np.arange(1, len(frac) + 1)
    ax.bar(modes, 100 * frac, color=BLUE, label="per mode")
    ax.plot(modes, 100 * np.cumsum(frac), "o-", color=ORANGE, ms=4, lw=1.3, label="cumulative")
    ax.axvline(basis["n_modes"] + 0.5, color=GREY, ls="--", lw=1)
    ax.annotate(f"{basis['n_modes']} modes kept\n{100 * basis['energy_captured']:.2f} % of the energy",
                xy=(basis["n_modes"] + 0.5, 50), xytext=(basis["n_modes"] + 1.5, 50), color="#ddd", fontsize=8,
                va="center")
    ax.set_xlabel("POD mode")
    ax.set_ylabel("share of the field energy [%]")
    ax.set_xticks(modes)
    ax.legend(fontsize=7, framealpha=0.2)
    ax.set_title("the basis, before any network is involved", color="#ddd", fontsize=9)
    return _finish(fig, path)


def accuracy(path: str, ref: dict, prd: dict, std: dict, ids: list, rmse: dict, gaps: np.ndarray | None = None,
             floor: float = 0.10) -> str:
    """What "accurate" is worth design by design: the two held-out parities and, once the physics stage has run, the
    lift the pressure integral implies against the lift the network predicts."""
    n = 3 if gaps is not None else 2
    fig, axes = plt.subplots(1, n, figsize=(1.4 + 3.0 * n, 2.8))
    for ax, q in zip(axes[:2], ("CL", "CD"), strict=False):
        r = np.asarray(ref[q], float)
        p = np.asarray(prd[q], float)
        lo, hi = float(min(r.min(), p.min())), float(max(r.max(), p.max()))
        pad = 0.08 * (hi - lo + 1e-12)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=GREY, lw=0.8, ls="--")
        ax.errorbar(r, p, yerr=std[q], fmt="o", ms=4, color=BLUE, ecolor=GREY, elinewidth=0.7, capsize=0,
                    label="held-out designs")
        ax.set_xlabel(f"{q} reference")
        ax.set_ylabel(f"{q} surrogate")
        ax.set_title(f"{q}: RMSE {rmse[q]:.4f}", color="#ddd", fontsize=9)
        ax.legend(fontsize=7, framealpha=0.2)
        if q == "CL":                                    # the design the error trace belongs to
            worst = int(np.argmax(np.abs(p - r)))
            ax.plot([r[worst]], [p[worst]], "o", ms=10, mfc="none", mec=RED, mew=1.4)
            ax.annotate(f"{ids[worst]}: {100 * abs(p[worst] - r[worst]) / max(abs(r[worst]), floor):.1f} %",
                        (r[worst], p[worst]), textcoords="offset points", xytext=(8, -16), color=RED, fontsize=7,
                        ha="left")
    if gaps is not None:
        axes[2].bar(np.arange(len(gaps)), gaps, color=ORANGE)
        axes[2].set_xlabel("held-out design")
        axes[2].set_ylabel("|CL from Cp - CL predicted|")
        axes[2].set_title("pressure integral vs predicted lift", color="#ddd", fontsize=9)
    return _finish(fig, path)


def error_trace(path: str, stations: np.ndarray, ref_cp: np.ndarray, prd_cp: np.ndarray, design: str) -> str:
    """The surface pressure of one design against its reference, and the error along the chord, point by point."""
    n_st = len(stations)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for side, label in ((0, "upper"), (1, "lower")):
        sl = slice(side * n_st, (side + 1) * n_st)
        line = "-" if side == 0 else "--"
        axes[0].plot(stations, -np.asarray(ref_cp)[sl], line, color=BLUE, lw=1.4, label=f"reference {label}")
        axes[0].plot(stations, -np.asarray(prd_cp)[sl], line, color=ORANGE, lw=1.1, label=f"surrogate {label}")
        axes[1].plot(stations, np.asarray(prd_cp)[sl] - np.asarray(ref_cp)[sl], line, color=ORANGE, lw=1.1)
    axes[0].set_xlabel("x / c")
    axes[0].set_ylabel("-Cp")
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=7, framealpha=0.2)
    axes[0].set_title(f"surface pressure, {design}", color="#ddd", fontsize=9)
    axes[1].axhline(0, color=GREY, lw=0.8)
    axes[1].set_xlabel("x / c")
    axes[1].set_ylabel("Cp error")
    axes[1].set_title("where the model is wrong", color="#ddd", fontsize=9)
    return _finish(fig, path)


def physics_gates(path: str, summary: dict, checks: tuple) -> str:
    """The checks the pipeline ran, the surrogate beside the reference on each.

    Both series are the checks' own numbers, unchanged: the point of the panel is that the surrogate departs from
    physical behaviour by about as much as the solver that produced its training data does, and that is a comparison,
    not a threshold.
    """
    rows = [(k, label) for k, label in checks if k in summary]
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [summary[k]["surrogate_mean"] for k, _ in rows], 0.38, color=ORANGE, label="surrogate")
    ax.bar(x + 0.2, [summary[k].get("reference_mean", np.nan) for k, _ in rows], 0.38, color=BLUE,
           label="reference")
    for i, (key, _) in enumerate(rows):
        s, r = summary[key]["surrogate_mean"], summary[key].get("reference_mean")
        if r and np.isfinite(r):
            ax.annotate(f"{s / r:.2f}x", (i, max(s, r)), textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=7, color="#ddd")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in rows], fontsize=8)
    ax.set_ylabel("value, as the check reports it")
    ax.set_title("is the prediction a flow, or only a fit? (label: surrogate / reference)", color="#ddd", fontsize=9)
    ax.legend(fontsize=7, framealpha=0.2)
    return _finish(fig, path)


def sensitivity(path: str, rows: dict) -> str:
    """Every input nudged by one percent of its range: does the prediction move less than the members disagree?"""
    names = list(rows)
    ratios = [rows[n]["ratio"] for n in names]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.barh(names, ratios, color=[GREEN if r < 1 else RED for r in ratios])
    ax.axvline(1.0, color=RED, ls="--", lw=1.1, label="as large as the ensemble spread")
    for i, r in enumerate(ratios):
        ax.annotate(f"{r:.2f}x", (r, i), textcoords="offset points", xytext=(4, -3), fontsize=8, color="#ddd")
    ax.set_xlabel("movement of CL / ensemble spread, for a 1 % nudge of the input")
    ax.set_title("robustness, measured rather than promised", color="#ddd", fontsize=9)
    ax.legend(fontsize=7, framealpha=0.2)
    return _finish(fig, path)


def domain_map(path: str, dom, points) -> str:
    """Where the evidence supports a prediction, one panel per speed: a shape in the design space, not a range."""
    ia = dom.names.index("alpha_deg")
    is_ = dom.names.index("speed")
    it = dom.names.index("thickness")
    lo_a, hi_a = dom.bounds[ia]
    lo_t, hi_t = dom.bounds[it]
    alphas = np.linspace(lo_a, hi_a, 121)      # fine enough to draw the boundary, not to flatter it
    thick = np.linspace(lo_t, hi_t, 81)
    speeds = (10.0, 20.0, 30.0)
    pts = [tuple(float(v) for v in p) for p in points]
    fig, axes = plt.subplots(1, len(speeds), figsize=(8.4, 2.9), sharey=True)
    for ax, speed in zip(axes, speeds, strict=True):
        grid = np.zeros((len(thick), len(alphas)))
        for a, al in enumerate(alphas):
            for t, th in enumerate(thick):
                r = dom.assess({"alpha_deg": float(al), "speed": float(speed), "thickness": float(th)})
                grid[t, a] = 1.0 if r["status"] == "EMPIRICALLY_SUPPORTED" else 0.0
        ax.contourf(alphas, thick * 100, grid, levels=[-0.5, 0.5, 1.5], colors=["#21262d", "#0b3d24"])
        if grid.max() > 0:
            ax.contour(alphas, thick * 100, grid, levels=[0.5], colors=[GREEN], linewidths=1.0)
        near = [p for p in pts if abs(p[is_] - speed) <= 5.0]
        ax.plot([p[ia] for p in near], [100 * p[it] for p in near], "o", ms=3.5, color=BLUE,
                label="independent reference" if speed == speeds[0] else None)
        ax.set_xlabel("angle of attack [deg]")
        ax.set_title(f"{speed:.0f} m/s", color="#ddd", fontsize=9)
        if speed == speeds[0]:
            ax.set_ylabel("thickness [% chord]")
            ax.plot([], [], "s", color="#0b3d24", ms=8, label="supported")
            ax.legend(fontsize=7, framealpha=0.2, loc="lower right")
    fig.suptitle("where the evidence supports a prediction, at three freestream speeds", color="#ddd", fontsize=9)
    return _finish(fig, path)


def extend(path: str, reval: dict) -> str:
    """What the second campaign did to the same weights, and where it changed nothing."""
    mid = reval["midpoints_between_new_and_nearest_old"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    before, after = reval["domain_before_extend"], reval["domain_after_extend"]
    x = np.arange(2)
    axes[0].bar(x - 0.2, [before["evidence"], before["supported"]], 0.38, color=GREY, label="before")
    axes[0].bar(x + 0.2, [after["evidence"], after["supported"]], 0.38, color=GREEN, label="after")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["references", "supported\nof 400"])
    axes[0].legend(fontsize=7, framealpha=0.2)
    axes[0].set_title(f"the same weights, {after['evidence'] - before['evidence']} more references",
                      color="#ddd", fontsize=9)
    changed = mid["status_changes"]
    axes[1].bar([0, 1], [changed, mid["candidates"] - changed], color=[GREEN, GREY])
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["became supported", "did not"], fontsize=8)
    axes[1].set_title(f"the {mid['candidates']} geometries the new references reach", color="#ddd", fontsize=9)
    return _finish(fig, path)


def uncertainty(path: str, z_scores: np.ndarray, coverage_1s: float, coverage_2s: float) -> str:
    """What the ensemble spread is worth: the held-out error in units of the spread, and the coverage it buys."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    axes[0].hist(np.clip(z_scores, 0, 6), bins=np.linspace(0, 6, 25), color=BLUE, alpha=0.85)
    for x, lab in ((1.0, f"one sigma ({coverage_1s:.0%})"), (2.0, f"two ({coverage_2s:.0%})")):
        axes[0].axvline(x, color=ORANGE if x == 1.0 else GREEN, ls="--", lw=1.2, label=lab)
    axes[0].set_xlabel("|error| / ensemble spread")
    axes[0].set_ylabel("held-out cases")
    axes[0].set_title("is the spread the right size?", color="#ddd", fontsize=9)
    axes[0].legend(fontsize=7, framealpha=0.2)
    grid = np.linspace(0.05, 3.0, 60)
    cover = [float(np.mean(z_scores <= g)) for g in grid]
    axes[1].plot(grid, cover, color=BLUE, lw=1.6, label="measured")
    axes[1].plot(grid, [1 - np.exp(-0.5 * g ** 2) for g in grid], color=GREY, ls="--", lw=1.0,
                 label="a calibrated Gaussian")
    axes[1].set_xlabel("bar width, in ensemble standard deviations")
    axes[1].set_ylabel("fraction of errors inside")
    axes[1].set_title("calibration curve", color="#ddd", fontsize=9)
    axes[1].legend(fontsize=7, framealpha=0.2)
    return _finish(fig, path)

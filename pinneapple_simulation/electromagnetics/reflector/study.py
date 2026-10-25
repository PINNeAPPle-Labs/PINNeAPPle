"""Compact-range studies: validation of the PO code, a (feed gain x section size) sweep, a surrogate with held-out
error and the trade-off map of the quiet-zone quality.

* :func:`validate` -- the PO integrator against the closed-form field of a PO disk (:func:`.analytic.disk_on_axis`)
  at several mesh densities, the PO quiet-zone amplitude against geometrical optics (:func:`.compact_range.go_check`)
  for rolled edges of growing size, and the mesh convergence of the quiet-zone metrics;
* :func:`qz_study` -- the sweep (:func:`.compact_range.sweep`), a Gaussian process per metric over (feed gain,
  side) for each roll radius, held-out errors, and :func:`tradeoff_map`.

Every number returned is computed here; nothing is taken from elsewhere.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from . import analytic, geometry, po
from .compact_range import CompactRange, evaluate, go_check, scaled, sweep

__all__ = ["METRICS", "validate", "fit_surrogates", "tradeoff_map", "qz_figure", "qz_study"]

METRICS = ("ripple_pp_dB", "amp_pp_dB", "phase_pp_deg", "taper_dB", "usable_diameter")


def validate(base: CompactRange, *, disk_radius_wl: float = 3.0, mesh_densities: Sequence[float] = (8, 12, 16),
             rolls_wl: Sequence[float] = (3.0, 6.0), log: Callable[[str], None] = print) -> dict[str, Any]:
    """The three checks of the module docstring, at the frequency of ``base``."""
    k = po.wavenumber(base.freq_hz)
    lam = base.wavelength
    out: dict[str, Any] = {"disk": [], "go": [], "mesh": []}
    a = disk_radius_wl * lam
    z = np.linspace(2 * lam, 40 * lam, 60)
    Ea = analytic.disk_on_axis(z, a, k)
    for m in mesh_densities:
        d = geometry.flat_disk(a, lam / m)
        _, H = po.plane_wave(d.centres, k, (0, 0, -1), (1, 0, 0))
        J = po.po_currents(d.normals, d.centres, H)
        Ep = po.radiate(d.centres, d.areas, J, np.c_[0 * z, 0 * z, z], k)[:, 0]
        err = float(np.max(np.abs(Ep - Ea)) / np.max(np.abs(Ea)))
        out["disk"].append({"per_wavelength": m, "triangles": int(len(d.areas)), "max_rel_error": err})
        log(f"PO disk vs closed form, lambda/{m:g}: max relative error {err:.2e}")
    for rw in rolls_wl:
        cr = scaled(base, roll=rw * lam)
        g = go_check(cr)
        out["go"].append({"roll_wl": rw, **g})
        log(f"PO vs GO taper, roll {rw:g} lambda: max {g['max_abs_dB']:.2f} dB, rms {g['rms_dB']:.2f} dB")
    for m in mesh_densities[:2]:
        e = evaluate(CompactRange(**{**base.__dict__, "mesh_per_wavelength": m}))
        out["mesh"].append({"per_wavelength": m, **{k_: e[k_] for k_ in METRICS + ("seconds", "n_triangles")}})
        log(f"mesh lambda/{m:g}: ripple {e['ripple_pp_dB']:.3f} dB, phase {e['phase_pp_deg']:.2f} deg")
    return out


def fit_surrogates(table, metrics: Sequence[str] = METRICS, *, test_frac: float = 0.25, seed: int = 0):
    """Per roll radius: a GP per metric on (feed gain, side), fitted on all rows, and two held-out RMSEs: models
    refitted without a random ``test_frac`` of the designs (interpolation inside the grid), and without every design
    of one interior side at a time (interpolation across sizes, the harder case). Returns
    ``{roll: {metric: (model, rmse_random, n_test, rmse_leave_side_out)}}``."""
    from ..antenna import surrogates as sg
    feats = sg.columns("feed_gain_dBi", "side")
    rng = np.random.default_rng(seed)
    out: dict[float, dict[str, Any]] = {}
    for roll in sorted({float(r["roll"]) for r in table.rows}):
        t = table.where(lambda r, roll=roll: abs(float(r["roll"]) - roll) < 1e-12)
        tags = t.tags
        test = list(rng.choice(tags, max(1, int(round(test_frac * len(tags)))), replace=False))
        sides = sorted({float(r["side"]) for r in t.rows})[1:-1]
        out[roll] = {}
        for m in metrics:
            make = lambda: sg.GPModel(prior="const", features=feats)  # noqa: E731
            h = sg.holdout(t, m, test, models={"gp": make})
            err = []
            for sd in sides:
                ts = [r["tag"] for r in t.rows if abs(float(r["side"]) - sd) < 1e-12]
                err += list(sg.holdout(t, m, ts, models={"gp": make}).errors("gp"))
            out[roll][m] = (make().fit(t, m), h.rmse["gp"], len(test),
                            float(np.sqrt(np.mean(np.square(err)))) if err else float("nan"))
    return out


def tradeoff_map(table, models: dict[str, Any], path: str, *, roll: float, qz_diameter: float,
                 amp_tol: float = 0.5, phase_tol: float = 5.0, mark: tuple[float, float, str] | None = None,
                 metric: str = "ripple_pp_dB", side_unit: float = 1.0, side_label: str = "section side [m]",
                 unit_name: str = "m", title: str = "") -> str:
    """Two maps over (feed gain, side) from the GPs: the predicted ``metric`` and the predicted usable quiet-zone
    diameter (within +-amp_tol dB and +-phase_tol deg), with the simulated designs, the required diameter as a
    white line, the feasible region (usable >= ``qz_diameter``) hatched (or a note when it is empty) and an optional
    marked design ``(gain_dBi, side, label)``. ``side_unit`` divides lengths for the axes (e.g. 0.3048 for feet)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = table.where(lambda r: abs(float(r["roll"]) - roll) < 1e-12)
    g, s = t.col("feed_gain_dBi"), t.col("side")
    G, S = np.meshgrid(np.linspace(g.min(), g.max(), 90), np.linspace(s.min(), s.max(), 90))
    P = [{"feed_gain_dBi": a, "side": b} for a, b in zip(G.ravel(), S.ravel(), strict=True)]
    pred = lambda m: models[m][0].predict(P, t.col("f_GHz")[0])[0].reshape(G.shape)  # noqa: E731
    U = pred("usable_diameter")
    feas = U >= qz_diameter
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, Z, lab in ((axes[0], pred(metric), f"predicted {metric}"),
                       (axes[1], U / side_unit, f"predicted usable QZ diameter [{unit_name}]")):
        cf = ax.contourf(G, S / side_unit, Z, levels=20, cmap="jet")
        plt.colorbar(cf, ax=ax, label=lab)
        if U.min() < qz_diameter < U.max():
            ax.contour(G, S / side_unit, U, levels=[qz_diameter], colors="white", linewidths=2.2)
        if feas.any():
            ax.contourf(G, S / side_unit, feas.astype(float), levels=[0.5, 1.5], colors="none", hatches=["//"])
        ax.scatter(g, s / side_unit, c="k", s=12, label="simulated (PO)")
        if mark is not None:
            ax.scatter([mark[0]], [mark[1] / side_unit], marker="*", s=260, c="white", edgecolors="k", zorder=5,
                       label=mark[2])
        ax.set_xlabel("feed gain [dBi]")
        ax.set_ylabel(side_label)
        ax.legend(fontsize=7, loc="upper left")
    if not feas.any():
        axes[1].text(0.5, 0.04, f"no design in this range reaches a {qz_diameter / side_unit:g} {unit_name} quiet zone "
                                f"within +-{amp_tol:g} dB / +-{phase_tol:g} deg", transform=axes[1].transAxes,
                     ha="center", fontsize=8, bbox={"fc": "white", "alpha": 0.85})
    r1, r2 = models[metric][1], models["usable_diameter"][1]
    axes[0].set_title(title or f"{metric} over the {qz_diameter / side_unit:g} {unit_name} quiet zone "
                               f"(held-out RMSE {r1:.2f})", fontsize=9)
    axes[1].set_title(f"usable QZ within +-{amp_tol:g} dB / +-{phase_tol:g} deg (held-out RMSE "
                      f"{r2 / side_unit:.2f} {unit_name}); hatched: >= {qz_diameter / side_unit:g} {unit_name}",
                      fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def qz_figure(cr: CompactRange, path: str, *, length_unit: float = 1.0, unit_name: str = "m",
              title: str = "") -> dict[str, Any]:
    """Co-polar amplitude (dB re centre) and phase (deg re centre) over the evaluated region of the quiet-zone plane,
    the QZ circle, and the horizontal and vertical cuts through the centre. Returns the metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    from .compact_range import qz_field
    from .qz import qz_metrics
    f = qz_field(cr)
    co = 1 if cr.pol.upper() == "V" else 0
    xy, E = f["xy"], f["E"][:, co]
    c = int(np.argmin(np.hypot(xy[:, 0], xy[:, 1])))
    A = 20 * np.log10(np.abs(E) / np.abs(E[c]))
    P = np.rad2deg(np.angle(E / E[c]))
    m = qz_metrics(xy, E, cr.qz_diameter, E_x=f["E"][:, 1 - co])
    tri = Triangulation(xy[:, 0] / length_unit, xy[:, 1] / length_unit)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    th = np.linspace(0, 2 * np.pi, 200)
    R = cr.qz_diameter / 2 / length_unit
    Ru = m["usable_diameter"] / 2 / length_unit
    for a_, Z, lab, lim in ((ax[0], A, "amplitude [dB re centre]", (-3, 1)), (ax[1], P, "phase [deg re centre]",
                                                                             (-20, 20))):
        t_ = a_.tripcolor(tri, np.clip(Z, *lim), cmap="jet", shading="gouraud", vmin=lim[0], vmax=lim[1])
        plt.colorbar(t_, ax=a_, label=lab)
        a_.plot(R * np.cos(th), R * np.sin(th), "w-", lw=2, label=f"QZ {2 * R:g} {unit_name}")
        a_.plot(Ru * np.cos(th), Ru * np.sin(th), "k--", lw=1.2, label=f"usable {2 * Ru:.2f} {unit_name}")
        a_.set_aspect("equal")
        a_.set_xlabel(f"x [{unit_name}]")
        a_.set_ylabel(f"y - y_QZ [{unit_name}]")
        a_.legend(fontsize=7, loc="upper right")
    for k, lab in ((1, "vertical cut (x = 0)"), (0, "horizontal cut (y = y_QZ)")):
        on = np.abs(xy[:, 1 - k]) < 1e-9
        o = np.argsort(xy[on, k])
        ax[2].plot(xy[on, k][o] / length_unit, A[on][o], label=f"amplitude, {lab}")
    ax[2].axvspan(-R, R, color="0.9")
    ax[2].axhspan(-0.5, 0.5, color="tab:green", alpha=0.15)
    ax[2].set_xlabel(f"position [{unit_name}]")
    ax[2].set_ylabel("amplitude [dB re centre]")
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=0.3)
    fig.suptitle(title or f"quiet zone at {cr.freq_hz / 1e9:g} GHz: ripple {m['ripple_pp_dB']:.2f} dB p-p, phase "
                          f"{m['phase_pp_deg']:.1f} deg p-p, usable {m['usable_diameter'] / length_unit:.2f} "
                          f"{unit_name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return m


def qz_study(base: CompactRange, gains: Sequence[float], sides: Sequence[float], rolls: Sequence[float | None],
             out: str, *, mark: tuple[float, float, str] | None = None, side_unit: float = 1.0,
             side_label: str = "section side [m]", unit_name: str = "m", table=None,
             log: Callable[[str], None] = print) -> dict[str, Any]:
    """Sweep, surrogates with held-out error and one trade-off map per roll radius into ``out``; ``table``: an
    existing sweep to reuse instead of solving again."""
    from ..antenna.runner import write_table
    os.makedirs(out, exist_ok=True)
    if table is None:
        table = sweep(base, gains, sides, rolls, log=log)
        write_table(table, os.path.join(out, "qz_sweep.csv"))
    models = fit_surrogates(table)
    summary: dict[str, Any] = {"n_designs": len(table), "seconds_total": float(np.sum(table.col("seconds"))),
                               "seconds_median": float(np.median(table.col("seconds"))), "rolls": {}}
    for roll, ms in models.items():
        p = tradeoff_map(table, ms, os.path.join(out, f"tradeoff_roll{roll:g}.png"), roll=roll,
                         qz_diameter=base.qz_diameter, mark=mark, side_unit=side_unit, side_label=side_label,
                         unit_name=unit_name)
        summary["rolls"][f"{roll:g}"] = {"map": os.path.basename(p),
                                         "heldout_rmse_random": {m: v[1] for m, v in ms.items()},
                                         "n_test": next(iter(ms.values()))[2],
                                         "heldout_rmse_leave_side_out": {m: v[3] for m, v in ms.items()}}
    with open(os.path.join(out, "qz_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    return {"table": table, "models": models, "summary": summary}

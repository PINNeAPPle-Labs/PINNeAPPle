"""Figures of a rover mission (matplotlib): site and risk maps with both routes, the soil belief converging, the wheel
models against the measurements, and the traverses. Every curve is computed from the mission's own runs."""
from __future__ import annotations

import numpy as np

from .estimation import PARAMS, soil_theta

ROUTE_COLORS = ("#c0392b", "#1e8449", "#2e86c1")
LABELS = {"log10_k_phi": "log10 k_phi [N/m^(n+2)]", "n0": "n0 [-]", "n1": "n1 [-]", "phi_deg": "phi [deg]"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130})
    return plt


def _unit_image(site, cell=0.1):
    X, Y, Z = site.grid(cell)
    U = site.unit_at(X, Y)
    rgb = np.array([u.color for u in site.units])[U]
    return X, Y, Z, U, rgb


def plot_site_map(site, planners, routes, runs, path, slip_limit=0.4):
    """Truth soils with relief and routes; the risk the rover saw before (orbital prior) and after learning."""
    plt = _plt()
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    X, Y, Z, U, rgb = _unit_image(site)
    ext = (*site.x_range, *site.y_range)
    ax = axs[0]
    ax.imshow(rgb, origin="lower", extent=ext, aspect="equal")
    cs = ax.contour(X, Y, Z, levels=np.arange(-1, 4, 0.25), colors="k", linewidths=0.4, alpha=0.5)
    ax.clabel(cs, cs.levels[::4], fontsize=6, fmt="%.1f m")
    ax.set_title("True soils (hidden) and relief")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=u.color, label=u.name + ("  [orbital: " + u.orbital.name + "]"))
                       for u in site.units], loc="lower right", fontsize=6.5, framealpha=0.85)
    for k, (ttl, pl) in enumerate((("Risk from the orbital map (prior)", planners[0]),
                                   ("Risk after learning the soil", planners[-1]))):
        ax = axs[k + 1]
        Xr, Yr, R = pl.risk_map(0.25)
        im = ax.pcolormesh(Xr, Yr, R, cmap="YlOrRd", vmin=0, vmax=1, shading="auto")
        ax.contour(X, Y, Z, levels=np.arange(-1, 4, 0.5), colors="k", linewidths=0.3, alpha=0.4)
        ax.set_title(ttl)
        ax.set_aspect("equal")
        if k == 1:
            fig.colorbar(im, ax=ax, shrink=0.85, label=f"P(slip > {slip_limit:g}), worst heading")
    for ax in axs:
        for i, (r, run) in enumerate(zip(routes, runs, strict=True)):
            c = ROUTE_COLORS[i % 3]
            ax.plot(*r.waypoints.T, "--", color=c, lw=1.0, alpha=0.8)
            ax.plot(run.q[:, 0], run.q[:, 1], "-", color=c, lw=2.0,
                    label=f"route {chr(65 + i)} ({r.belief_label}): {run.meta['ended'].split(':')[0]}")
            if run.meta["ended"] != "goal":
                ax.plot(*run.q[-1, :2], "x", color=c, ms=10, mew=2.5)
        ax.plot(*site.start, "o", color="k", ms=6)
        ax.plot(*site.goal, "*", color="k", ms=12)
        ax.set_xlim(site.x_range)
        ax.set_ylim(site.y_range)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    axs[2].legend(loc="upper left", fontsize=7, framealpha=0.85)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_learning(belief, unit, truth, path, title=""):
    """Mean +- 2 sd of each soil parameter of ``unit`` after every filter update, with the prior and the truth."""
    plt = _plt()
    b = belief.units[unit]
    if not b.history:
        return None
    t = np.array([h[0] for h in b.history])
    M = np.array([h[1] for h in b.history])
    S = np.array([h[2] for h in b.history])
    m0, P0 = belief.prior[unit]
    s0 = np.sqrt(np.diag(P0))
    tt = soil_theta(truth)
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.2), constrained_layout=True)
    for j, ax in enumerate(axs):
        tx = np.r_[t[0] - 1.0, t]
        mx, sx = np.r_[m0[j], M[:, j]], np.r_[s0[j], S[:, j]]
        ax.fill_between(tx, mx - 2 * sx, mx + 2 * sx, color="#2e86c1", alpha=0.2, step="post", label="belief +- 2 sd")
        ax.step(tx, mx, where="post", color="#2e86c1", lw=1.6, label="belief mean")
        ax.axhline(tt[j], color="#c0392b", lw=1.4, ls="--", label="truth (synthetic)")
        ax.axhline(m0[j], color="0.4", lw=1.0, ls=":", label="orbital prior")
        ax.set_title(LABELS[PARAMS[j]])
        ax.set_xlabel("time on the unit [s]")
    axs[0].legend(fontsize=7)
    fig.suptitle(title or f"Soil belief of '{belief.units[unit].base.name}' unit (iterated EKF)")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_model_comparison(comparison, path):
    """Sinkage and torque against slip: measured (multibody + sensor noise) vs analytic (orbital prior, learned) vs
    the zoo surrogate, per unit the rover drove on."""
    plt = _plt()
    units = [n for n, c in comparison.items() if isinstance(c, dict) and "measured" in c]
    fig, axs = plt.subplots(2, len(units), figsize=(5.2 * len(units), 6.4), constrained_layout=True, squeeze=False)
    for k, name in enumerate(units):
        c = comparison[name]
        m = c["measured"]
        o = np.argsort(m["slip"])
        for row, (q, scale, unit) in enumerate((("sinkage", 1e3, "mm"), ("torque", 1.0, "N m"))):
            ax = axs[row, k]
            ax.scatter(m["slip"], scale * m[q], s=6, color="0.55", alpha=0.6, label="measured")
            for key, col, lab in (("analytic_prior", "#c0392b", "analytic, orbital prior"),
                                  ("analytic_learned", "#1e8449", "analytic, learned soil"),
                                  ("surrogate_zoo", "#8e44ad", "zoo surrogate (GRC-1)")):
                d = c.get(key)
                if not d:
                    continue
                y = scale * np.asarray(d["pred"][q])[o]
                rm = d[f"{q}_rmse_{'mm' if q == 'sinkage' else 'Nm'}"]
                extra = (f", {d['fraction_outside_validity']:.0%} outside its box"
                         if "fraction_outside_validity" in d else "")
                ax.plot(m["slip"][o], y, ".", ms=2.5, color=col, label=f"{lab}: RMSE {rm:.1f} {unit}{extra}")
            ax.set_xlabel("measured slip [-]")
            ax.set_ylabel(f"{q} [{unit}]")
            ax.set_title(f"{name}: {q} (n = {c['n']})")
            ax.legend(fontsize=6.5, markerscale=3)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_traverse(runs, site, path, slip_limit=0.4, front=(0, 1)):
    """Per run vs distance: the traction monitor's slip, front-wheel sinkage, cumulative drive energy, elevation with
    the soil unit under the rover."""
    from .verify import monitor_slip
    plt = _plt()
    fig, axs = plt.subplots(4, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    for i, run in enumerate(runs):
        c = ROUTE_COLORS[i % 3]
        d = run.distance() + (0.0 if i == 0 else 0.0)
        lab = f"route {chr(65 + i)}: {run.meta['ended'].split(':')[0]}"
        axs[0].plot(d, monitor_slip(run, list(front)), color=c, lw=1.5, label=lab)
        axs[1].plot(d, 1e3 * run.wheel["sinkage"][:, list(front)].mean(1), color=c, lw=1.2)
        axs[2].plot(d, run.energy(), color=c, lw=1.5)
        z = site.height(run.q[:, 0], run.q[:, 1])
        axs[3].plot(d, z, color=c, lw=1.0)
        u = run.wheel["soil_idx"][:, list(front)].max(1)
        for k, unit in enumerate(site.units):
            mk = u == k
            axs[3].scatter(d[mk], z[mk], s=3, color=unit.color, zorder=3)
    axs[0].axhline(slip_limit, color="k", ls="--", lw=1, label=f"mission limit {slip_limit:g}")
    axs[0].set_ylabel("slip (2-s mean, front)")
    axs[0].legend(fontsize=8)
    axs[1].set_ylabel("sinkage, front [mm]")
    axs[2].set_ylabel("drive energy [J]")
    axs[3].set_ylabel("elevation [m]")
    axs[3].set_xlabel("distance driven on the route [m]")
    axs[3].set_title("dots: soil unit under the front wheels (colours as in the site map)", fontsize=8)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_route_risk(planners, routes, path, slip_limit=0.4, risk_max=0.05):
    """Predicted slip and risk along every route, under the orbital prior and under the learned belief."""
    plt = _plt()
    fig, axs = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, constrained_layout=True)
    for i, r in enumerate(routes):
        c = ROUTE_COLORS[i % 3]
        for pl, ls, tag in ((planners[0], ":", "prior"), (planners[-1], "-", "learned")):
            e = pl.evaluate(r.waypoints)
            axs[0].plot(e.profile["s"], e.profile["slip"], ls, color=c, lw=1.4,
                        label=f"route {chr(65 + i)}, {tag} belief")
            axs[1].plot(e.profile["s"], e.profile["risk"], ls, color=c, lw=1.4)
    axs[0].axhline(slip_limit, color="k", ls="--", lw=1)
    axs[0].set_ylim(-0.2, 1.05)
    axs[0].set_ylabel("predicted slip (belief mean)\n1 = cannot climb")
    axs[0].legend(fontsize=7, ncol=2)
    axs[1].axhline(risk_max, color="k", ls="--", lw=1, label=f"risk limit {risk_max:g}")
    axs[1].set_ylabel(f"P(slip > {slip_limit:g})")
    axs[1].set_xlabel("distance along the route [m]")
    axs[1].legend(fontsize=7)
    fig.savefig(path)
    plt.close(fig)
    return path


__all__ = ["plot_site_map", "plot_learning", "plot_model_comparison", "plot_traverse", "plot_route_risk"]

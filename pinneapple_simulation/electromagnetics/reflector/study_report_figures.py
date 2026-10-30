"""Figures for a compact-range study: geometry, illumination, validation, quiet zone, sweeps.

Every figure is drawn from the study's stored numbers (or, for the geometry panels, from the mesh of the
design); nothing is sketched by hand.
"""
from __future__ import annotations

import os
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402
from matplotlib.patches import Circle, Rectangle, Polygon  # noqa: E402

from pinneapple_tools.reporting.paperkit import CAT, INK, INK2, MUTED, SEQ, DIV, savefig, setup_mpl  # noqa: E402

FT = 0.3048

__all__ = ["build_figures"]


def _style():
    setup_mpl()


def _ax(ax, xlabel="", ylabel="", title=""):
    ax.grid(alpha=0.25, lw=0.6, color="#d9d8d2")
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8.5, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5, color=INK2)
    if title:
        ax.set_title(title, fontsize=9, color=INK, pad=5)
    ax.tick_params(labelsize=7.5, colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b9b8b2")


def _save(fig, out: str, name: str) -> str:
    path = os.path.join(out, name)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------------------
def _geometry_figures(p, out: str) -> dict[str, str]:
    import pinneapple_simulation.electromagnetics.reflector.geometry as gm

    cr = p.cr()
    r = cr.reflector
    meshm = gm.mesh_reflector(r, cr.wavelength / 6)
    V = meshm.vertices
    figs = {}

    # --- side section x = 0: paraboloid, rolled edge, rays, quiet zone
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    on = np.abs(V[:, 0]) < cr.wavelength / 4
    o = np.argsort(V[on, 1])
    ys, zs = V[on, 1][o] / FT, V[on, 2][o] / FT
    ax.plot(ys, zs, color=CAT[3], lw=1.8, solid_capstyle="round", zorder=3)
    # GO rays: focus -> surface -> collimated along +z
    fz = r.focal
    y_top = r.centre_y + r.half_y
    th = np.linspace(np.deg2rad(2.0), 2 * np.arctan(y_top / (2 * fz)) * 1.02, 15)
    yy = 2 * fz * np.tan(th / 2)                          # paraboloid: rho = 2F tan(theta/2)
    zz = yy ** 2 / (4 * fz)
    sel = yy <= y_top
    for y0, z0 in zip(yy[sel], zz[sel], strict=True):
        ax.plot([0, y0 / FT], [fz / FT, z0 / FT], color=CAT[3], lw=0.55, alpha=0.6, zorder=1)
        ax.plot([y0 / FT, y0 / FT], [z0 / FT, cr.qz_z / FT], color=CAT[3], lw=0.55, alpha=0.6, zorder=1)
    qz = Rectangle(((r.centre_y - cr.qz_diameter / 2) / FT, (cr.qz_z - 1.5 * FT) / FT),
                   cr.qz_diameter / FT, 3.0, facecolor=CAT[0], alpha=0.14, edgecolor=CAT[0], lw=0.9,
                   zorder=2, label="quiet zone")
    ax.add_patch(qz)
    ax.plot([0], [fz / FT], marker="v", ms=7, color=CAT[7], zorder=5)
    ax.annotate("feed on the focus", (0, fz / FT), textcoords="offset points", xytext=(7, -11),
                fontsize=7, color=CAT[7])
    ax.annotate("quiet zone", ((r.centre_y + cr.qz_diameter / 2) / FT, cr.qz_z / FT),
                textcoords="offset points", xytext=(4, -2), fontsize=7, color=CAT[0])
    ax.plot([0], [0], marker="|", ms=6, color=MUTED)
    ax.text(0.06, -0.9, "paraboloid axis", fontsize=6.5, color=MUTED)
    ax.set_aspect("equal")
    ax.set_xlim(-1.0, 7.6)
    _ax(ax, "y (ft)", "z (ft)", "Side section x = 0: paraboloid, rolled edge, rays, quiet zone")
    figs["side"] = _save(fig, out, "fig_side.png")

    # --- front view: junction, edge extent, quiet zone, cuts
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    phi = np.linspace(0, 2 * np.pi, 400)
    rj = gm.superellipse_radius(phi, r.half_x, r.half_y, r.n)
    ax.fill(rj * np.cos(phi) / FT, (r.centre_y + rj * np.sin(phi)) / FT,
            facecolor=CAT[3], alpha=0.22, edgecolor=CAT[3], lw=1.0, label="parabolic section (junction)")
    # outer extent of the rolled edge, from the mesh: the largest radius per 1-degree sector
    rel = V[:, :2] - np.array([0.0, r.centre_y])
    rad_v, ang_v = np.hypot(rel[:, 0], rel[:, 1]), np.arctan2(rel[:, 1], rel[:, 0])
    edges = np.linspace(-np.pi, np.pi, 361)
    idx = np.clip(np.digitize(ang_v, edges) - 1, 0, len(edges) - 2)
    rad_out = np.full(len(edges) - 1, np.nan)
    for i in range(len(edges) - 1):
        m = idx == i
        if m.any():
            rad_out[i] = rad_v[m].max()
    mid = 0.5 * (edges[:-1] + edges[1:])
    ok = np.isfinite(rad_out)
    ax.plot(np.r_[rad_out[ok] * np.cos(mid[ok]), rad_out[ok][0] * np.cos(mid[ok][0])] / FT,
            (r.centre_y + np.r_[rad_out[ok] * np.sin(mid[ok]), rad_out[ok][0] * np.sin(mid[ok][0])]) / FT,
            color="#7a6420", lw=1.3, ls="--", label="rolled edge, outer extent")
    ax.add_patch(Circle((0, r.centre_y / FT), cr.qz_diameter / 2 / FT, fill=False, ec=CAT[0], lw=1.5,
                        label="quiet zone"))
    ax.plot([-cr.qz_diameter / 2 / FT, cr.qz_diameter / 2 / FT], [r.centre_y / FT] * 2, color=CAT[7],
            lw=1.2, label="horizontal cut")
    ax.plot([0, 0], [(r.centre_y - cr.qz_diameter / 2) / FT, (r.centre_y + cr.qz_diameter / 2) / FT],
            color=CAT[2], lw=1.2, label="vertical cut")
    ap_w = 0.24 / 2 / FT
    ax.add_patch(Rectangle((-ap_w, -0.35), 2 * ap_w, 0.32, facecolor="#9a9a94", edgecolor="none", zorder=4))
    ax.text(0, -0.62, "feed aperture", fontsize=6.5, ha="center", color=INK2)
    ax.set_aspect("equal")
    ax.legend(fontsize=6.2, loc="lower left", ncol=1, frameon=True, framealpha=0.9)
    _ax(ax, "x (ft)", "y (ft)", "Front view, from the quiet zone")
    figs["front"] = _save(fig, out, "fig_front.png")

    # --- illumination on the lit surface
    from pinneapple_simulation.electromagnetics.reflector.po import cosq_feed, wavenumber

    k = wavenumber(cr.freq_hz)
    feed = r.focus
    axis = r.centre - feed
    E, _ = cosq_feed(meshm.centres, k, position=feed, axis=axis, q=cr.feed_q, pol=(0, 1, 0))
    amp = 20 * np.log10(np.maximum(np.linalg.norm(E, axis=1), 1e-30))
    rel = np.clip(np.nan_to_num(amp - amp.max(), nan=-30.0, neginf=-30.0), -14, 0)
    fig, ax = plt.subplots(figsize=(4.6, 3.7))
    sc = ax.scatter(meshm.centres[:, 0] / FT, meshm.centres[:, 1] / FT, c=rel, s=1.6, cmap=SEQ,
                    vmin=-14, vmax=0)
    ax.plot(rj * np.cos(phi) / FT, (r.centre_y + rj * np.sin(phi)) / FT, color="white", lw=1.4)
    ax.add_patch(Circle((0, r.centre_y / FT), cr.qz_diameter / 2 / FT, fill=False, ec="k",
                        lw=1.1, ls="--"))
    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("feed illumination (dB re maximum)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    ax.set_aspect("equal")
    _ax(ax, "x (ft)", "y (ft)", "Feed illumination of the lit surface; dashed: quiet-zone footprint")
    figs["illum"] = _save(fig, out, "fig_illumination.png")
    return figs


# --------------------------------------------------------------------------------------------------
def _validation_figure(p, study, out: str) -> str:
    v = study["results"].get("validation", {})
    disk = v.get("disk_densities") or v.get("disk") or []
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.9))
    ax = axes[0]
    m = np.array([d["per_wavelength"] for d in disk], float)
    e = np.array([d["max_rel_error"] for d in disk], float)
    ax.loglog(m, e, "o", ms=5, color=CAT[0], label="PO vs closed form")
    if len(m) >= 2:
        ref = e[-1] * (m[-1] / m) ** 2
        ax.loglog(m, ref, "--", lw=1, color=MUTED, label="second order")
    ax.axhline(1e-2, color=CAT[7], lw=0.9, ls=":")
    _ax(ax, "mesh [points per wavelength]", "max relative error", "PO disk on axis vs the closed form")
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    go = v.get("go") or []
    if go:
        rw = [g["roll_wl"] for g in go]
        ax.plot(rw, [g["rms_dB"] for g in go], "o-", color=CAT[0], ms=5, label="RMS")
        ax.plot(rw, [g["max_abs_dB"] for g in go], "s-", color=CAT[1], ms=5, label="max")
        ax.set_xticks(rw)
        _ax(ax, "rolled edge [wavelengths]", "PO taper vs GO taper (dB)", "Smooth part of the quiet-zone field")
        ax.legend(fontsize=7, frameon=False)
    else:
        ax.axis("off")
    return _save(fig, out, "fig_validation.png")


# --------------------------------------------------------------------------------------------------
def _cuts_figure(p, study, out: str, pol: str) -> str:
    d = (study.get("results", {}).get("qz") or {}).get("pol", {}).get(pol)
    if not d:
        return ""
    planes = list(d.keys())
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 4.4), sharex=True)
    for j, kind in enumerate(("horizontal", "vertical")):
        ax_a, ax_p, ax_x = axes[0, j], axes[1, j], None
        for i, z in enumerate(planes):
            c = d[z]["cuts"][kind]
            t = np.asarray(c["t"], float)
            lab = f"z = {z} ft"
            ax_a.plot(t / FT, c["amp_dB"], color=CAT[i % 8], lw=1.3, label=lab)
            ax_p.plot(t / FT, c["phase_deg"], color=CAT[i % 8], lw=1.3, label=lab)
        ax_a.set_title(f"{kind} cut: amplitude", fontsize=8.5)
        ax_p.set_title(f"{kind} cut: phase", fontsize=8.5)
        _ax(ax_a, "", "dB re centre")
        _ax(ax_p, f"{'x' if kind == 'horizontal' else 'y'} (ft)", "degrees")
        ax_a.legend(fontsize=6.5, frameon=False)
    # cross-pol panel: horizontal cut of every plane
    ax = axes[0, 2]
    for i, z in enumerate(planes):
        c = d[z]["cuts"]["horizontal"]
        ax.plot(np.asarray(c["t"], float) / FT, np.clip(c["xpol_curve_dB"], -60, 0), color=CAT[i % 8], lw=1.1,
                label=f"z = {z} ft")
    _ax(ax, "x (ft)", "dB re co-pol centre", "horizontal cut: cross-polar")
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_ylim(-60, 0)
    axes[1, 2].axis("off")
    fig.suptitle(f"{pol} polarisation: co-polar cuts over {p.qz_diameter / FT:.1f} ft, "
                 f"{p.cut_points} points ({p.qz_diameter / FT * 12 / (p.cut_points - 1) * 2.54:.2f} cm spacing)",
                 fontsize=8.5, color=INK2)
    return _save(fig, out, f"fig_cuts_{pol}.png")


# --------------------------------------------------------------------------------------------------
def _qz_map_figure(p, study, out: str) -> str:
    from pinneapple_simulation.electromagnetics.reflector.compact_range import qz_field
    from pinneapple_simulation.electromagnetics.reflector.qz import qz_metrics
    from matplotlib.tri import Triangulation

    cr = p.cr(p.pols[0])
    f = qz_field(cr)
    co, xp = (1, 0) if p.pols[0] == "V" else (0, 1)
    xy, E = f["xy"], f["E"][:, co]
    i = int(np.argmin(np.hypot(xy[:, 0], xy[:, 1])))
    A = 20 * np.log10(np.abs(E) / np.abs(E[i]))
    P = np.rad2deg(np.angle(E / E[i]))
    m = qz_metrics(xy, E, cr.qz_diameter, E_x=f["E"][:, xp])
    tri = Triangulation(xy[:, 0] / FT, xy[:, 1] / FT)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    th = np.linspace(0, 2 * np.pi, 200)
    R, Ru = cr.qz_diameter / 2 / FT, m["usable_diameter"] / 2 / FT
    for ax, Z, lab, lim in ((axes[0], A, "amplitude (dB re centre)", (-4, 1)),
                            (axes[1], P, "phase (deg re centre)", (-25, 25))):
        t = ax.tripcolor(tri, np.clip(Z, *lim), cmap=DIV, shading="gouraud", vmin=lim[0], vmax=lim[1])
        cb = plt.colorbar(t, ax=ax, pad=0.02)
        cb.set_label(lab, fontsize=7.5)
        cb.ax.tick_params(labelsize=7)
        ax.plot(R * np.cos(th), R * np.sin(th), color="k", lw=1.3, label=f"quiet zone {2 * R:g} ft")
        ax.plot(Ru * np.cos(th), Ru * np.sin(th), color="w", lw=1.6, ls="--",
                label=f"usable at +-0.5 dB {2 * Ru:.2f} ft")
        ax.set_aspect("equal")
        ax.legend(fontsize=6.5, loc="upper right", framealpha=0.85)
        _ax(ax, "x (ft)", "y - y_QZ (ft)")
    axes[0].set_title(f"{p.pols[0]} polarisation, z = {p.qz_z / FT:g} ft", fontsize=8.5)
    axes[1].set_title(f"ripple {m['ripple_pp_dB']:.2f} dB p-p, phase {m['phase_pp_deg']:.1f} deg p-p",
                      fontsize=8.5)
    return _save(fig, out, "fig_qz_map.png")


# --------------------------------------------------------------------------------------------------
def _axial_figure(p, study, out: str) -> str:
    a = study["results"].get("axial")
    if not a:
        return ""
    z = np.array(a["z_ft"])
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.9))
    for i, pol in enumerate(a["pol"]):
        d = a["pol"][pol]
        axes[0].plot(z, np.array(d["usable_at_0.5dB"]) / FT, "o-", ms=3.5, color=CAT[i], lw=1.2,
                     label=f"{pol}, +-0.5 dB")
        axes[0].plot(z, np.array(d["usable_at_1dB"]) / FT, "s--", ms=3.0, color=CAT[i], lw=1.0, alpha=0.7,
                     label=f"{pol}, +-1 dB")
        axes[1].plot(z, d["ripple_pp_dB"], "o-", ms=3.5, color=CAT[i], lw=1.2, label=f"{pol}")
    tgt = p.qz_diameter / FT
    axes[0].axhline(tgt, color=CAT[7], lw=1.0, ls=":")
    axes[0].annotate(f"specification {tgt:g} ft", (z[0], tgt), textcoords="offset points", xytext=(2, 4),
                     fontsize=7, color=CAT[7])
    axes[0].axvline(p.qz_z / FT, color=MUTED, lw=0.8, ls="--")
    _ax(axes[0], "z (ft)", "usable diameter (ft)", "Usable diameter along the axis")
    axes[0].legend(fontsize=6.5, frameon=False, ncol=2)
    _ax(axes[1], "z (ft)", "ripple (dB p-p)", "Ripple along the axis")
    axes[1].legend(fontsize=7, frameon=False)
    return _save(fig, out, "fig_axial.png")


# --------------------------------------------------------------------------------------------------
def _bandwidth_figure(p, study, out: str) -> str:
    b = study["results"].get("bandwidth")
    if not b:
        return ""
    f = np.array(b["freq_ghz"])
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.9))
    for i, pol in enumerate(b["pol"]):
        d = b["pol"][pol]
        axes[0].plot(f, np.array(d["usable_at_0.5dB"]) / FT, "o-", ms=3.5, color=CAT[i], lw=1.2,
                     label=f"{pol}, +-0.5 dB")
        axes[0].plot(f, np.array(d["usable_at_1dB"]) / FT, "s--", ms=3.0, color=CAT[i], lw=1.0, alpha=0.7,
                     label=f"{pol}, +-1 dB")
        axes[1].plot(f, d["ripple_pp_dB"], "o-", ms=3.5, color=CAT[i], lw=1.2, label=f"{pol}")
    tgt = p.qz_diameter / FT
    axes[0].axhline(tgt, color=CAT[7], lw=1.0, ls=":")
    axes[0].annotate(f"specification {tgt:g} ft", (f[0], tgt), textcoords="offset points", xytext=(2, 4),
                     fontsize=7, color=CAT[7])
    _ax(axes[0], "frequency (GHz)", "usable diameter (ft)", "Usable diameter across the band")
    axes[0].legend(fontsize=6.5, frameon=False, ncol=2)
    _ax(axes[1], "frequency (GHz)", "ripple (dB p-p)", "Ripple across the band")
    axes[1].legend(fontsize=7, frameon=False)
    return _save(fig, out, "fig_bandwidth.png")


# --------------------------------------------------------------------------------------------------
def _tolerance_figure(p, study, out: str) -> str:
    t = study["results"].get("tolerance")
    if not t:
        return ""
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.9))
    pol = list(t["pol"])[0]
    for i, axis_name in enumerate(("lateral", "axial", "diagonal")):
        rows = t["pol"][pol][axis_name]
        x = [r["offset_lambda"] for r in rows]
        axes[0].plot(x, [r["usable_at_0.5dB"] / FT for r in rows], "o-", ms=3.5, color=CAT[i], lw=1.2,
                     label=axis_name)
        axes[1].plot(x, [r["amp_pp_dB"] for r in rows], "o-", ms=3.5, color=CAT[i], lw=1.2, label=axis_name)
    tgt = p.qz_diameter / FT
    axes[0].axhline(tgt, color=CAT[7], lw=1.0, ls=":")
    _ax(axes[0], "feed displacement (wavelengths)", "usable diameter (ft)",
        f"Usable diameter vs feed displacement, {pol}")
    axes[0].legend(fontsize=7, frameon=False)
    _ax(axes[1], "feed displacement (wavelengths)", "amplitude p-p (dB)",
        "Ripple vs feed displacement")
    axes[1].legend(fontsize=7, frameon=False)
    return _save(fig, out, "fig_tolerance.png")


# --------------------------------------------------------------------------------------------------
def _profile_figure(p, study, out: str) -> str:
    pr = study["results"].get("profile")
    if not pr:
        return ""
    rows = pr["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.9))
    roll = np.array([r["roll"] for r in rows]) / FT
    axes[0].scatter(roll, [r["ripple_pp_dB"] for r in rows], c=[r["roll_aspect"] for r in rows], cmap=SEQ,
                    s=26, edgecolors="k", linewidths=0.3)
    _ax(axes[0], "rolled-edge radius (ft)", "ripple (dB p-p)", "Ripple over the profile family")
    blend = np.array([r["blend"] for r in rows])
    axes[1].scatter(blend, [r["usable_at_0.5dB"] / FT for r in rows], c=roll, cmap=SEQ, s=26,
                    edgecolors="k", linewidths=0.3)
    _ax(axes[1], "blend fraction of the arc", "usable diameter (ft)", "Usable diameter over the family")
    return _save(fig, out, "fig_profile.png")


# --------------------------------------------------------------------------------------------------
def _tradeoff_figure(p, study, out: str) -> str:
    sw = study["results"].get("sweep")
    if not sw:
        return ""
    rows = sw["rows"]
    cols = len(sw["rolls"])
    fig, axes = plt.subplots(1, cols + 1, figsize=(4.1 * (cols + 1), 3.3))
    if cols == 0:
        return ""
    G = np.array(sw["gains"])
    S = np.array(sw["sides"])
    for j, roll in enumerate(sw["rolls"]):
        ax = axes[j]
        sub = [r for r in rows if abs(r["roll_m"] - roll) < 1e-12]
        Z = np.full((len(S), len(G)), np.nan)
        for r in sub:
            Z[int(np.argmin(np.abs(S - r["side_m"]))), int(np.argmin(np.abs(G - r["feed_gain_dBi"])))] = \
                r["usable_at_0.5dB"] / FT
        cf = ax.contourf(G, S, Z, levels=12, cmap=SEQ)
        ax.contour(G, S, Z, levels=[p.qz_diameter / FT], colors="k", linewidths=1.2)
        ax.scatter([r["feed_gain_dBi"] for r in sub], [r["side_m"] for r in sub], s=9, c="k", alpha=0.55)
        cb = plt.colorbar(cf, ax=ax, pad=0.02)
        cb.set_label("usable (ft) at +-0.5 dB", fontsize=7.2)
        cb.ax.tick_params(labelsize=6.5)
        _ax(ax, "feed gain (dBi)", "section side (m)", f"rolled edge {roll / FT:.2f} ft "
                                                       f"({roll / p.wavelength:.1f} $\\lambda$)")
    ax = axes[-1]
    for j, roll in enumerate(sw["rolls"]):
        sub = [r for r in rows if abs(r["roll_m"] - roll) < 1e-12]
        ax.scatter([r["side_m"] for r in sub], [r["usable_at_0.5dB"] / FT for r in sub], s=12,
                   color=CAT[j], label=f"roll {roll / FT:.2f} ft")
    ax.axhline(p.qz_diameter / FT, color=CAT[7], lw=1.0, ls=":")
    _ax(ax, "section side (m)", "usable diameter (ft)", "Every design, by rolled edge")
    ax.legend(fontsize=6.5, frameon=False)
    return _save(fig, out, "fig_tradeoff.png")


# --------------------------------------------------------------------------------------------------
def _equal_budget_figure(p, study, out: str) -> str:
    o = study["results"].get("optimize")
    if not o:
        return ""
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for i, (key, lab) in enumerate((("grid", "regular grid"), ("surrogate", "DOE + GP + verify"),
                                    ("bayesian", "Bayesian on the solver"))):
        tr = np.array(o[key]["trace"]) / FT
        ax.step(np.arange(1, len(tr) + 1), tr, where="post", lw=1.5, color=CAT[i], label=lab)
        ax.scatter([len(tr)], [tr[-1]], s=26, color=CAT[i], zorder=5)
    ax.axhline(p.qz_diameter / FT, color=CAT[7], lw=1.0, ls=":")
    ax.annotate(f"specification {p.qz_diameter / FT:g} ft", (1, p.qz_diameter / FT),
                textcoords="offset points", xytext=(2, 4), fontsize=7, color=CAT[7])
    _ax(ax, "physical-optics solves", "best usable diameter so far (ft)",
        f"Equal solver budget ({o['budget']} solves)")
    ax.legend(fontsize=6.8, frameon=False, loc="lower right")
    return _save(fig, out, "fig_equal_budget.png")


# --------------------------------------------------------------------------------------------------
def build_figures(p, study: Mapping[str, Any], out: str) -> dict[str, str]:
    """Every figure of the report, as ``{key: png path}``.  A study that was not run loses its figures and
    nothing else."""
    _style()
    os.makedirs(out, exist_ok=True)
    figs: dict[str, str] = {}
    jobs = [("geometry", lambda: _geometry_figures(p, out)),
            ("validation", lambda: {"validation": _validation_figure(p, study, out)}),
            ("qz_map", lambda: {"qz_map": _qz_map_figure(p, study, out)}),
            ("axial", lambda: {"axial": _axial_figure(p, study, out)}),
            ("bandwidth", lambda: {"bandwidth": _bandwidth_figure(p, study, out)}),
            ("tolerance", lambda: {"tolerance": _tolerance_figure(p, study, out)}),
            ("profile", lambda: {"profile": _profile_figure(p, study, out)}),
            ("tradeoff", lambda: {"tradeoff": _tradeoff_figure(p, study, out)}),
            ("equal_budget", lambda: {"equal_budget": _equal_budget_figure(p, study, out)})]
    jobs += [(f"cuts_{pol}", lambda pol=pol: {f"cuts_{pol}": _cuts_figure(p, study, out, pol)})
             for pol in p.pols]
    for name, fn in jobs:
        try:
            got = fn()
            figs.update({k: v for k, v in (got or {}).items() if v})
        except Exception as exc:                        # a missing study must not lose the report
            print(f"figure {name} skipped: {type(exc).__name__}: {exc}")
    return figs

"""Figures of the condensation benchmark (matplotlib, from the JSON of :mod:`.benchmark`)."""
from __future__ import annotations

import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _ax(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def scaling(points: list[dict], path: str) -> str:
    """Wall time and peak memory against the number of blocks (two panels, one quantity each)."""
    plt = _plt()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    series = [("reference", "seconds", "fine FV reference (one solve)"),
              ("flat", "prepare_s", "flat condensation: build"),
              ("hierarchy", "prepare_s", "hierarchy: build"),
              ("hierarchy", "solve_s", "hierarchy: per new power map"),
              ("hierarchy_m6", "prepare_s", "hierarchy, m=6: build")]
    for i, (meth, key, label) in enumerate(series):
        pts = [(p["blocks"], p[meth][key]) for p in points if key in p.get(meth, {})]
        if pts:
            x, y = zip(*sorted(pts), strict=False)
            a.loglog(x, y, "-o", color=SERIES[i], lw=2, ms=5, label=label)
    _ax(a, "blocks (repeated unit cells)", "wall time [s]", "Time (m=12 unless noted)")
    a.legend(fontsize=7, frameon=False)
    for i, meth in enumerate(("reference", "flat", "hierarchy", "hierarchy_m6")):
        pts = [(p["blocks"], p[meth]["peak_rss_gb"]) for p in points if "peak_rss_gb" in p.get(meth, {})]
        if pts:
            x, y = zip(*sorted(pts), strict=False)
            b.loglog(x, y, "-o", color=SERIES[[0, 1, 2, 4][i]], lw=2, ms=5, label=meth.replace("_m6", ", m=6"))
    _ax(b, "blocks", "peak resident memory [GB]", "Memory (fresh process per method)")
    b.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def accuracy(res: dict, path: str) -> str:
    plt = _plt()
    rows = res["rows"]
    m = [r["m"] for r in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, (k, lab) in enumerate((("err_Tmax", "peak junction T"), ("err_Tj_max", "worst junction cell"),
                                  ("field_max_err", "worst cell, full field"))):
        a.semilogy(m, [max(r[k], 1e-12) for r in rows], "-o", color=SERIES[i], lw=2, ms=5, label=lab)
    _ax(a, "interface modes per face m", "error [K]", f"Temperature error (Tmax {res['Tmax_ref']:.1f} K)")
    a.legend(fontsize=7, frameon=False)
    for i, (k, lab) in enumerate((("err_q_tim", "die/TIM heat"), ("err_q_ubump", "per-microbump heat"),
                                  ("err_q_tsv", "per-TSV heat"), ("energy_err", "energy norm (sqrt(W K))"))):
        b.semilogy(m, [max(r[k], 1e-12) for r in rows], "-o", color=SERIES[i], lw=2, ms=5, label=lab)
    _ax(b, "interface modes per face m", "relative error (sum |err| / sum |ref|)", "Interface heat flows")
    b.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def refinement(res: dict, path: str) -> str:
    plt = _plt()
    fig, a = plt.subplots(figsize=(5.5, 3.8))
    ad, un = res["adaptive"], res["uniform"]
    a.loglog([r["dofs"] for r in un], [r["energy_err"] for r in un], "-o", color=SERIES[0], lw=2, label="uniform m")
    for i, (mode, lab) in enumerate((("p", "marked faces: double the modes"), ("full", "marked faces: un-reduce"))):
        rows = ad[mode]
        a.loglog([r["dofs"] for r in rows], [r["energy_err"] for r in rows], "-o", color=SERIES[i + 1], lw=2,
                 label=lab + " (true)")
        a.loglog([r["dofs"] for r in rows], [r["estimate"] for r in rows], "--", color=SERIES[i + 1], lw=1.5,
                 label=lab + " (estimate)")
    _ax(a, "interface dofs", "energy-norm error", "Refinement where the estimate says")
    a.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def keff_maps(maps: dict, path: str) -> str:
    """Fine junction temperatures, the homogenised block averages, and their difference."""
    plt = _plt()
    fine, hom = np.asarray(maps["fine"]), np.asarray(maps["hom"])
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
    vmin, vmax = fine.min(), fine.max()
    for ax, img, title in ((axs[0], fine, "fine (every cell)"), (axs[1], hom, "Keff (block averages)")):
        im = ax.imshow(img.T, origin="lower", cmap="Blues", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, label="junction T rise [K]")
    d = hom - fine
    lim = np.abs(d).max()
    im = axs[2].imshow(d.T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    axs[2].set_title("Keff - fine", fontsize=10, loc="left")
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    fig.colorbar(im, ax=axs[2], fraction=0.046, label="[K]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def transient(res: dict, path: str) -> str:
    plt = _plt()
    t = (np.arange(res["steps"]) + 1) * res["dt"] * 1e3
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    a.plot(t, res["Tmax_ref"], color=INK, lw=2, label="fine reference")
    for i, r in enumerate(res["rows"]):
        a.plot(t, r["Tmax"], "--", color=SERIES[i], lw=1.5, label=f"CB m={r['m']}, k={r['k']}")
        b.semilogy(t, np.maximum(np.abs(np.array(r["Tmax"]) - np.array(res["Tmax_ref"])), 1e-9), color=SERIES[i],
                   lw=1.5, label=f"m={r['m']}, k={r['k']}")
    _ax(a, "time [ms]", "peak junction T rise [K]", "Power map switches every 10 ms")
    _ax(b, "time [ms]", "|error| of peak junction T [K]", "Error")
    a.legend(fontsize=7, frameon=False)
    b.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def meshless_convergence(res: dict, path: str, mesh_error: float | None = None) -> str:
    """|Tmax error| and |energy balance| against the number of nodes, per method and interface treatment."""
    plt = _plt()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    combos = [("gfdm", "pairs", False), ("rbf_fd", "pairs", False), ("gfdm", "naive", False),
              ("rbf_fd", "naive", False), ("gfdm", "pairs", True)]
    for i, (m, itf, ls) in enumerate(combos):
        rr = [r for r in res["rows"] if r["method"] == m and r["interface"] == itf and r["least_squares"] == ls
              and "nodes" in r]
        if not rr:
            continue
        x = [r["nodes"] for r in rr]
        lab = f"{'RBF-FD' if m == 'rbf_fd' else 'GFDM'}, {'interface pairs' if itf == 'pairs' else 'naive'}" + \
            (", least squares" if ls else "")
        a.loglog(x, [max(abs(r["err_Tmax"]), 1e-6) for r in rr], "-o", color=SERIES[i], lw=2, ms=5, label=lab)
        b.loglog(x, [max(abs(r["energy_balance"]), 1e-6) for r in rr], "-o", color=SERIES[i], lw=2, ms=5, label=lab)
    if mesh_error is not None:
        a.axhline(mesh_error, color=MUTED, lw=1, ls="--")
        a.text(a.get_xlim()[0] * 1.1, mesh_error * 1.15, "FV mesh error of the reference (Tmax)", color=MUTED,
               fontsize=7)
    _ax(a, "meshless nodes", "|Tmax error| [K]", f"Peak junction temperature ({res['blocks']} blocks)")
    _ax(b, "meshless nodes", "|heat out / heat in - 1|", "Energy balance")
    a.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def meshless_fields(res: dict, method: str, path: str) -> str:
    """The point cloud on a vertical slice coloured by temperature, and the junction-temperature error map."""
    plt = _plt()
    f = res["fields"][method]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
    x, z, T = (np.asarray(f[k]) * s for k, s in (("slice_x", 1e6), ("slice_z", 1e6), ("slice_T", 1.0)))
    keep = (z > 150) & (z < 520)                                     # C4 top to die top: the features
    sc = a.scatter(x[keep], z[keep], c=T[keep], s=6, cmap="Blues")
    fig.colorbar(sc, ax=a, label="temperature rise [K]")
    _ax(a, "x [um]", "z [um]", f"{'RBF-FD' if method == 'rbf_fd' else 'GFDM'} nodes near y = W/2 (interposer to die)")
    d = np.asarray(f["Tj"]) - np.asarray(res["Tj_ref"])
    lim = max(np.abs(d).max(), 1e-9)
    im = b.imshow(d.T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    fig.colorbar(im, ax=b, label="meshless - FV [K]")
    b.set_xticks([])
    b.set_yticks([])
    b.set_title("Junction temperature error", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def pareto(res: dict, path: str) -> str:
    """Error against the fraction of cells solved exactly (exception patches), per selection rule and halo."""
    plt = _plt()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    combos = sorted({(r["select"], r["halo"]) for r in res["rows"]})
    for i, (sel, halo) in enumerate(combos):
        rr = sorted([r for r in res["rows"] if r["select"] == sel and r["halo"] == halo],
                    key=lambda r: r["solved_fraction"])
        x = [max(r["solved_fraction"], 1e-3) for r in rr]
        lab = f"{sel} selection, halo {halo}"
        a.semilogy(x, [max(r["err_Tmax"], 1e-5) for r in rr], "-o", color=SERIES[i], lw=2, ms=5, label=lab)
        b.semilogy(x, [max(r["err_q_ubump"], 1e-5) for r in rr], "-o", color=SERIES[i], lw=2, ms=5, label=lab)
    _ax(a, "fraction of cells solved exactly", "|Tmax error| [K]", f"Prior {res['prior']} + exception patches")
    _ax(b, "fraction of cells solved exactly", "per-bump heat error (sum |err| / sum |ref|)", "Interface heat")
    a.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

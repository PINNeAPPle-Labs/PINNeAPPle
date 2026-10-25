"""Adaptive Physics Condensation nodes: a chip package solved by hierarchical static condensation of its repeated
blocks (library: :mod:`pinneapple_condense`)."""
from __future__ import annotations

from ..registry import Param, Port, node
from ..runtime import artifact

_TAGS = ("section:solvers", "thermal", "semiconductor packaging", "reduced order", "condensation")


@node("condense.package_thermal", "Chip package thermal by condensation", category="Solvers",
      outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("blocks", "int", 64, minimum=2, maximum=512, help="blocks per side (one microbump + TSV each)"),
              Param("modes", "int", 8, minimum=1, maximum=64, help="interface modes per face"),
              Param("power_density", "float", 1.0e6, help="mean heat flux over the die [W/m^2]"),
              Param("hotspots", "int", 3, minimum=0, maximum=20),
              Param("seed", "int", 0),
              Param("reference", "bool", False, help="also solve the fine finite-volume reference (small only)")],
      calls="pinneapple_condense.hierarchy.Hierarchy / pinneapple_condense.fv.solve_package", tags=_TAGS)
def package_thermal(ctx, inputs, params):
    """Build the synthetic multilayer package, condense the repeated blocks once (identical subtrees reused), solve
    the hotspot power map, and report the peak junction temperature, the heat through the TIM and the lid and the
    energy balance; optionally against the fine reference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pinneapple_condense import fv
    from pinneapple_condense.hierarchy import Hierarchy
    from pinneapple_condense.package import uniform_package
    from pinneapple_condense.substructure import reference_observables
    nb = int(params["blocks"])
    pkg = uniform_package((nb, nb), power_density=float(params["power_density"]), seed=int(params["seed"]),
                          hotspots=int(params["hotspots"]))
    h = Hierarchy(pkg, int(params["modes"])).prepare()
    sol = h.solve()
    o = h.observables(sol)
    P = float(pkg.power.sum())
    m = {"blocks": nb * nb, "fine_cells_equivalent": pkg.n_cells, "Tmax_rise_K": float(o["Tj"].max()),
         "q_tim_W": float(o["q_tim"].sum()), "q_lid_W": float(o["q_top"].sum()), "power_W": P,
         "energy_balance": float((o["q_top"].sum() + o["q_bottom"].sum()) / P - 1),
         "build_s": h.timings["prepare"], "solve_s": sol["seconds"], "distinct_merges": h.timings["distinct_merges"]}
    if params["reference"]:
        if pkg.n_cells > 4_000_000:
            ctx.log("reference skipped: more than 4e6 cells")
        else:
            T, info = fv.solve_package(pkg)
            m["Tmax_reference_K"] = float(reference_observables(pkg, T)["Tj"].max())
            m["reference_s"] = info.seconds
    ctx.table("Result", ["quantity", "value"], [[k, v] for k, v in m.items()])
    ctx.summary(f"Tmax rise {m['Tmax_rise_K']:.2f} K over {nb * nb} blocks ({pkg.n_cells:.2e} fine cells), "
                f"build {m['build_s']:.1f} s, solve {m['solve_s']:.2f} s, energy balance {m['energy_balance']:.1e}")
    n = pkg.stack.n
    tj = o["Tj"].reshape(nb, nb, n, n).transpose(0, 2, 1, 3).reshape(nb * n, nb * n)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(tj.T, origin="lower", cmap="Blues")
    fig.colorbar(im, ax=ax, label="junction temperature rise [K]")
    ax.set_xticks([])
    ax.set_yticks([])
    path = ctx.workdir / "junction_map.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {"metrics": m, "figure": artifact(str(path), "png")}

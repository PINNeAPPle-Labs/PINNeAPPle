"""Immersed (finite cell / CutFEM) analysis nodes: steady heat on raw geometry, no conforming mesh
(library: :mod:`pinneapple_simulation.immersed`)."""
from __future__ import annotations

from ..registry import Param, Port, node
from ..runtime import artifact

_TAGS = ("section:solvers", "thermal", "immersed", "finite cell", "cutfem", "stl", "mesh-free")


@node("immersed.heat", "Immersed steady heat on raw geometry (SDF or STL)", category="Solvers",
      outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("shape", "enum", "cylinder", choices=["sphere", "cylinder", "torus", "stl"],
                    help="built-in solid, or an STL file (stl_path)"),
              Param("size", "float", 0.05, help="characteristic size of the built-in solid [m]"),
              Param("stl_path", "str", "", help="closed STL surface, used when shape = stl"),
              Param("cells", "int", 16, minimum=4, maximum=64, help="background cells across the part"),
              Param("order", "int", 1, minimum=1, maximum=2, help="1 = Q1, 2 = Q2"),
              Param("conductivity", "float", 200.0, help="k [W/(m K)]"),
              Param("generation", "float", 1.0e6, help="uniform heat generation [W/m^3]"),
              Param("h_conv", "float", 25.0, help="convection coefficient on the whole surface [W/(m^2 K)]"),
              Param("T_inf", "float", 25.0, help="ambient temperature [C]"),
              Param("stabilization", "enum", "ghost", choices=["ghost", "fcm"])],
      calls="pinneapple_simulation.immersed.Discretization / pinneapple_simulation.immersed.solve_heat",
      tags=_TAGS)
def immersed_heat(ctx, inputs, params):
    """Put the part in a background grid, integrate its cut cells, solve steady conduction with uniform generation
    and convection on the surface, and report the peak temperature, the energy balance (generated against
    convected heat) and the cost by phase; the figure is the mid-plane temperature."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from pinneapple_simulation import immersed as im
    L = float(params["size"])
    shape = params["shape"]
    if shape == "stl":
        if not params["stl_path"]:
            raise ValueError("shape = stl needs stl_path")
        geo = im.STLGeometry.from_file(params["stl_path"])
    elif shape == "sphere":
        geo = im.ball((0.0, 0.0, 0.0), L / 2)
    elif shape == "torus":
        geo = im.torus((0.0, 0.0, 0.0), L / 3, L / 8)
    else:
        geo = im.cylinder((0.0, 0.0, 0.0), L / 4, L)
    lo, hi = geo.bounds
    h = float(np.max(hi - lo)) / int(params["cells"])
    disc = im.Discretization(geo, h, order=int(params["order"]))
    bcs = [im.Convection(float(params["h_conv"]), float(params["T_inf"]))]
    sol = im.solve_heat(disc, float(params["conductivity"]), float(params["generation"]), bcs,
                        stabilization=params["stabilization"])
    s = disc.summary()
    m = {"T_max_C": sol.max(), "dofs": disc.n_dofs, "cells_cut": s["cells_cut"],
         "min_cut_fraction": s["min_cut_fraction"], "volume_m3": s["volume"], "area_m2": s["area"],
         "generated_W": sol.balance["generated_W"], "convected_W": sol.balance["out_W"],
         "energy_imbalance": sol.balance["imbalance"], "quadrature_s": disc.timings["quadrature_s"],
         "assembly_s": sol.info["assembly_s"], "solve_s": sol.info["solve_seconds"]}
    ctx.table("Result", ["quantity", "value"], [[k, v] for k, v in m.items()])
    ctx.summary(f"T max {m['T_max_C']:.2f} C on {m['dofs']} unknowns ({m['cells_cut']} cut cells), energy balance "
                f"{m['energy_imbalance']:.1e}, quadrature {m['quadrature_s']:.1f} s, solve {m['solve_s']:.2f} s")
    n = 120
    ax0 = np.linspace(lo[0], hi[0], n)
    ax2 = np.linspace(lo[2], hi[2], n)
    A, C = np.meshgrid(ax0, ax2, indexing="ij")
    P = np.column_stack([A.ravel(), np.full(A.size, (lo[1] + hi[1]) / 2), C.ravel()])
    T = np.where(geo.phi(P) < 0, sol(P), np.nan).reshape(n, n)
    fig, ax = plt.subplots(figsize=(5, 4))
    im_ = ax.imshow(T.T, origin="lower", extent=(lo[0], hi[0], lo[2], hi[2]), cmap="inferno")
    fig.colorbar(im_, ax=ax, label="temperature [C]")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("mid-plane temperature (immersed)")
    path = ctx.workdir / "immersed_heat.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return {"metrics": m, "figure": artifact(str(path), "png")}

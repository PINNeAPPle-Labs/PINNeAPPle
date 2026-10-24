"""Exact geometry for a stress analysis: NVIDIA USD BRep builds the part, PINNeAPPle meshes, solves and checks it.

A 150 x 60 x 5 mm aluminium strap with a 20 mm hole carries 10 MPa of tension. USD BRep (the SMLib NURBS kernel)
builds the exact solid (box minus cylinder), gives its exact volume, area, mass and the curvature of every face, and
tessellates it at chosen tolerances. PINNeAPPle measures how far each mesh is from the exact surfaces (chord error,
area and volume), checks that the elements resolve the hole radius, solves plane stress on the B-rep face meshes
(constant-strain triangles, patch test first) at four element sizes, and compares the converged stress
concentration with Heywood's closed form.

    PINNEAPPLE_BREP_PYTHON=/opt/brep/bin/python python examples/nvidia/05_brep_plate_with_hole/run.py   # ~30 s
"""
import argparse
import json
import os

import numpy as np

from pinneapple_design.intent import materials
from pinneapple_lab import experiment
from pinneapple_simulation.numerical_solvers.plane_stress_fem import (
    boundary_edges,
    heywood_kt_net,
    patch_test,
    solve_plane_stress,
)
from pinneapple_tools.nvidia.brep import (
    PartRecipe,
    evaluate_part,
    geometry_checks,
    planar_face_mesh,
)
from pinneapple_tools.nvidia.diagram import Block, architecture_diagram
from pinneapple_tools.nvidia.figures import checks_figure, curves_figure
from pinneapple_tools.nvidia.runinfo import RunInfo

HERE = os.path.dirname(os.path.abspath(__file__))
L, W, T, D = 0.150, 0.060, 0.005, 0.020
S0 = 10e6                                   # Pa, remote tension


def part() -> PartRecipe:
    return (PartRecipe().box("strap", (0, 0, 0), (L, W, T))
            .cylinder("hole", (L / 2, W / 2, -T), D / 2, 3 * T).difference("part", "strap", "hole"))


def kt_on(rep, al):
    """Plane stress on the top face of the B-rep tessellation; returns (Kt_net, solution, patch-test error)."""
    V2, F2 = planar_face_mesh(*rep.mesh(-1), axis=2, value=T, atol=1e-9)
    be = boundary_edges(F2)
    right = be[np.all(np.isclose(V2[be, 0], L), 1)]
    left = np.where(np.isclose(V2[:, 0], 0.0))[0]
    corner = [left[int(np.argmin(np.abs(V2[left, 1] - W / 2)))]]
    sol = solve_plane_stress(V2, F2, E=al.props["youngs_modulus"].value, nu=al.props["poisson"].value, thickness=T,
                             fixed={"x": left, "y": corner}, traction=[(right, (S0, 0.0))])
    on_hole = np.isclose(np.linalg.norm(V2 - [L / 2, W / 2], axis=1), D / 2, atol=1e-6)
    kt = sol.nodal_stress()[on_hole, 0].max() / (S0 * W / (W - D))
    return float(kt), sol, patch_test(V2, F2), -sol.reaction[:, 0].sum() / (S0 * W * T)


def main(smoke=False, base=None):
    base = base or HERE
    OUT = os.path.join(base, "results")
    os.makedirs(OUT, exist_ok=True)
    info = RunInfo("05_brep_plate_with_hole")
    architecture_diagram(
        "05 Exact geometry for a stress analysis: PINNeAPPle + NVIDIA USD BRep",
        [Block("rec", "Part recipe (data):\nstrap - cylinder", "pinneapple", 0, 0),
         Block("brep", "USD BRep (SMLib NURBS):\nexact solid, mass, curvature", "brep", 1, 0),
         Block("tess", "USD BRep tessellation at\nchord tolerances / edge sizes", "brep", 2, 0),
         Block("geo", "Geometry checks: chord error,\nvolume / area, radius resolved", "pinneapple", 1, 1),
         Block("fem", "Plane-stress FEM on the B-rep\nface mesh (patch test first)", "pinneapple", 2, 1),
         Block("conv", "Mesh convergence, Richardson,\nHeywood Kt_net", "pinneapple", 3, 1),
         Block("dec", "Stress result usable?\ntrust decision", "pinneapple", 3, 0)],
        [("rec", "brep", "ops"), ("brep", "tess", "B-rep"), ("brep", "geo", "exact values"), ("tess", "geo", "meshes"),
         ("tess", "fem", "top-face triangles"), ("fem", "conv", "Kt(h)"), ("conv", "dec", "checks"), ("geo", "dec", "checks")],
        os.path.join(OUT, "architecture.png"),
        subtitle="The exact solid is the reference for the mesh; the closed form is the reference for the stress")
    al = materials.get("al6061_t6")
    sizes = (0.004, 0.002) if smoke else (0.004, 0.002, 0.001, 0.0005)
    with experiment("examples/nvidia/05_brep_plate_with_hole", title="Strap with a hole: USD BRep geometry, PINNeAPPle "
                    "plane-stress FEM and checks", domain="structural mechanics", methods=["B-rep (SMLib)", "FEM (CST)"],
                    out_dir=base, register=False, modules=["pinneapple_tools.nvidia.brep",
                                                           "pinneapple_simulation.numerical_solvers.plane_stress_fem"],
                    params={"L": L, "W": W, "t": T, "d": D, "sigma0_Pa": S0, "element_sizes": list(sizes)}) as ex:
        with info.step("USD BRep: exact solid and tessellations"):
            rep = evaluate_part(part(), chord_tolerances=[1e-3, 1e-4, 1e-5], density=al.props["density"].value)
        exact_v = L * W * T - np.pi * (D / 2) ** 2 * T
        exact_a = 2 * (L * W - np.pi * (D / 2) ** 2) + 2 * (L + W) * T + np.pi * D * T
        g_checks = geometry_checks(rep, element_size=sizes[-1])
        runs = []
        with info.step("plane-stress FEM at four element sizes"):
            for h in sizes:
                r = evaluate_part(part(), chord_tolerances=[h / 20], max_edge_length=h, max_samples=300)
                kt, sol, pt, fb = kt_on(r, al)
                runs.append({"h": h, "kt": kt, "patch": pt, "force_balance": fb, "elements": int(len(sol.F)),
                             "chord": r.tessellations[-1]["max_deviation"], "sol": sol})
        kts = np.array([r["kt"] for r in runs])
        p = np.log((kts[-2] - kts[-3]) / (kts[-1] - kts[-2])) / np.log(2) if len(kts) >= 3 else 1.0
        kt_inf = kts[-1] + (kts[-1] - kts[-2]) / (2 ** p - 1)
        kt_ref = heywood_kt_net(D / W)
        checks = [{"name": "B-rep volume = analytic", "value": abs(rep.volume - exact_v) / exact_v, "limit": 1e-6,
                   "passed": abs(rep.volume - exact_v) / exact_v < 1e-6},
                  {"name": "B-rep area = analytic", "value": abs(rep.area - exact_a) / exact_a, "limit": 1e-6,
                   "passed": abs(rep.area - exact_a) / exact_a < 1e-6},
                  {"name": "smallest radius = hole radius", "value": abs(rep.min_radius - D / 2) / (D / 2), "limit": 1e-6,
                   "passed": abs(rep.min_radius - D / 2) / (D / 2) < 1e-6},
                  *g_checks,
                  {"name": "patch test on every B-rep mesh", "value": max(r["patch"] for r in runs), "limit": 1e-9,
                   "passed": max(r["patch"] for r in runs) < 1e-9},
                  {"name": "force balance (reaction = applied)", "value": max(abs(r["force_balance"] - 1) for r in runs),
                   "limit": 1e-6, "passed": max(abs(r["force_balance"] - 1) for r in runs) < 1e-6,
                   "detail": "round-off of the sparse direct solve"},
                  {"name": "mesh converged (last two sizes)", "value": abs(kts[-1] - kts[-2]) / kts[-1], "limit": 0.03,
                   "passed": abs(kts[-1] - kts[-2]) / kts[-1] <= 0.03},
                  {"name": "Kt_net (Richardson) vs Heywood 2 + (1 - d/W)^3", "value": abs(kt_inf - kt_ref) / kt_ref,
                   "limit": 0.03, "passed": abs(kt_inf - kt_ref) / kt_ref <= 0.03,
                   "detail": f"{kt_inf:.3f} vs {kt_ref:.3f}; observed order {p:.2f}"}]
        decision = "APPROVED" if all(c["passed"] for c in checks) else "REJECT"
        for c in checks:
            ex.check(c["name"], c["passed"], value=c["value"], max=c["limit"], detail=c.get("detail", ""))
        ex.log_metric("kt_net_richardson", kt_inf, "", "FEM on B-rep meshes, Richardson from the three finest sizes")
        ex.log_metric("kt_net_finest", kts[-1], "", f"FEM, element size {sizes[-1] * 1e3:g} mm")
        ex.log_metric("brep_volume", rep.volume * 1e9, "mm^3", "USD BRep, relative accuracy 1e-6")
        info.mark("NVIDIA USD BRep", ran=True, how="usd-brep 1.0.1 (CPython 3.12 worker)")
        _figures(OUT, rep, runs, kts, kt_ref, kt_inf, checks, decision)
        summary = {"decision": decision, "kt_net": kts.tolist(), "element_sizes": list(sizes), "kt_richardson": kt_inf,
                   "observed_order": p, "kt_heywood": kt_ref, "elements": [r["elements"] for r in runs],
                   "brep": {k: rep.raw[k] for k in ("faces", "edges", "manifold", "volume", "area", "mass")},
                   "min_radius": rep.min_radius, "tessellations": rep.tessellations, "brep_worker_s": rep.seconds,
                   "checks": checks}
        with open(os.path.join(OUT, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=float)
        info.save(os.path.join(OUT, "run_info.json"))
    print(json.dumps({k: summary[k] for k in ("decision", "kt_net", "kt_richardson", "kt_heywood")}, indent=1))
    return summary


def _figures(OUT, rep, runs, kts, kt_ref, kt_inf, checks, decision):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    tt = rep.tessellations
    curves_figure([{"title": "B-rep tessellation vs exact surface", "xlabel": "chord tolerance asked [m]",
                    "ylabel": "measured", "logy": True,
                    "series": [("max deviation [m]", [t["chord_tolerance"] for t in tt], [t["max_deviation"] for t in tt], "o-"),
                               ("|volume error| (rel.)", [t["chord_tolerance"] for t in tt],
                                [abs(t["mesh_volume"] - rep.volume) / rep.volume for t in tt], "s-"),
                               ("|area error| (rel.)", [t["chord_tolerance"] for t in tt],
                                [abs(t["mesh_area"] - rep.area) / rep.area for t in tt], "^-")]},
                   {"title": "Stress concentration vs element size", "xlabel": "element size [mm]", "ylabel": "Kt_net",
                    "series": [("FEM on B-rep meshes", [r["h"] * 1e3 for r in runs], kts, "o-"),
                               ("Richardson", [0.0], [kt_inf], "k*")],
                    "hlines": [(kt_ref, "Heywood 2 + (1 - d/W)^3", "r")]}],
                  os.path.join(OUT, "convergence.png"), suptitle="Geometry and stress, each against its exact reference")
    s = runs[-1]["sol"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [2.4, 1]})
    tri = Triangulation(s.V[:, 0] * 1e3, s.V[:, 1] * 1e3, s.F)
    for ax, lim in ((axs[0], None), (axs[1], (L / 2 * 1e3 - 15, L / 2 * 1e3 + 15, W / 2 * 1e3 - 15, W / 2 * 1e3 + 15))):
        im = ax.tripcolor(tri, s.stress[:, 0] / 1e6, cmap="turbo", shading="flat")
        if lim:
            ax.triplot(tri, lw=0.15, color="k", alpha=0.4)
            ax.set_xlim(lim[:2])
            ax.set_ylim(lim[2:])
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
    fig.colorbar(im, ax=axs, shrink=0.85, label="sigma_xx [MPa]")
    axs[0].set_title(f"sigma_xx on the B-rep face mesh ({len(s.F)} triangles, 0.5 mm)", fontsize=9)
    axs[1].set_title("near the hole (mesh shown)", fontsize=9)
    fig.savefig(os.path.join(OUT, "stress.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    V, F = rep.mesh(0)
    fig = plt.figure(figsize=(6.5, 4))
    ax = fig.add_subplot(projection="3d")
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    ax.add_collection3d(Poly3DCollection(V[F] * 1e3, facecolor="#b8c4d6", edgecolor="#333333", linewidths=0.3))
    ax.set_xlim(0, 150)
    ax.set_ylim(-45, 105)
    ax.set_zlim(-20, 25)
    ax.set_box_aspect((150, 150, 45))
    ax.view_init(35, -60)
    ax.set_title(f"USD BRep solid, tessellated at {tt[0]['chord_tolerance'] * 1e3:g} mm chord "
                 f"({tt[0]['triangles']} triangles)", fontsize=9)
    fig.savefig(os.path.join(OUT, "brep_part.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    checks_figure(checks, os.path.join(OUT, "checks.png"), title="Geometry and stress checks", decision=decision)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(ap.parse_args().smoke)

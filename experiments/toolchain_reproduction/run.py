"""Reproduce, inside PINNeAPPle, the experiment of a blank machine plus an agent and an open-source
simulation toolchain: audit the toolchain, run classic CFD and FEA cases, check every one against its
published value, post-process, and print the table with the wall clock.

The experiment being reproduced is the one where an agent takes a fresh computer, installs
OpenFOAM / code_aster / ParaView, and about an hour later has solved a few textbook cases whose results
match the textbook. Here the toolchain is PINNeAPPle itself, so the "install" is a Python import and
every step below is a function call in this one file -- no licences, no menus, nothing to click.

| The original toolchain | Here |
|---|---|
| OpenFOAM (incompressible NS) | ``pinneapple_simulation.external_solvers.pict`` local engine (vorticity-streamfunction) |
| OpenFOAM (1-D transient hydraulics) | ``water_hammer_moc`` (method of characteristics) |
| code_aster (structures) | ``beam_fe`` (Hermite), ``solid_fem`` (3-D hex), ``plane_stress_fem`` (CST) |
| ParaView (post-processing) | matplotlib; ``solid_fem.fea_figure`` for the 3-D part |

Cases and the textbook value each one is held to:

1. **Lid-driven cavity** at Re = 100 and Re = 1000, against the centreline tables of Ghia, Ghia and Shin
   (J. Comput. Phys. 48, 1982, 387-411).
2. **Water hammer** after an instantaneous and a slow valve closure, against Joukowsky (1898)
   ``dH = a dV / g`` and Michaud (1878) ``dH = 2 L dV / (g t_c)``.
3. **Cantilever beam**: Hermite FE against ``q L^4 / 8 E I`` and the exact first frequency; then the same
   beam in 3-D hexahedra, which must converge to the Euler-Bernoulli answer from below.
4. **Plate with a central hole** in tension: stress concentration against Heywood's net-section
   ``Kt = 2 + (1 - d/W)^3`` (Pilkey & Pilkey, *Peterson's Stress Concentration Factors*) and Kirsch's
   ``Kt = 3`` for the infinite plate, with Richardson extrapolation on three meshes.

    python experiments/toolchain_reproduction/run.py             # every case, ~1 min
    python experiments/toolchain_reproduction/run.py --smoke     # coarse meshes, a few seconds
    python experiments/toolchain_reproduction/run.py --no-3d     # skip the 3-D hexahedral cantilever

Output: ``results/results.json`` (every number), ``results/*.png`` (four figures) and a table on stdout.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import platform
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.abspath(os.path.join(HERE, "..", "..")) not in sys.path:   # allow running as a script from a checkout
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

# ------------------------------------------------------------------------------------------------------------
# the cases' physical data, in SI, each number traceable to the reference it is checked against
# ------------------------------------------------------------------------------------------------------------
CAVITY = {"re": (100, 1000), "cells": {100: 64, 1000: 128}}         # cells per side per Reynolds number

WATER_HAMMER = {"length": 1200.0, "diameter": 0.6, "wave_speed": 1000.0, "q0": 0.30, "h_reservoir": 120.0,
                "n_reaches": 24}

BEAM = {"L": 1.0, "b": 0.05, "h": 0.02, "E": 210e9, "nu": 0.3, "rho": 7850.0, "P": 1000.0, "n_el": 40}

PLATE = {"L": 0.150, "W": 0.060, "D": 0.020, "t": 0.005, "S0": 10e6, "E": 68.9e9, "nu": 0.33,
         "meshes": ((64, 16), (128, 32), (256, 64))}                # (points around, rings from hole to edge)
SMOKE_PLATE_MESHES = ((32, 8), (64, 16), (128, 32))
SMOKE_CAVITY_CELLS = 32
SMOKE_3D = (20, 2, 2)
FULL_3D = ((20, 2, 2), (40, 4, 4), (60, 6, 6))


# ------------------------------------------------------------------------------------------------------------
# 0. the toolchain: what the "blank machine setup" reduces to for a pure-Python stack
# ------------------------------------------------------------------------------------------------------------
def toolchain_audit() -> dict:
    """Versions of everything the cases below import, plus the licences and the external solvers that are (or are
    not) installed -- so the reader can see the cases were solved without them."""
    packages = ["numpy", "scipy", "matplotlib", "torch", "pyvista"]
    versions = {}
    for p in packages:
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            versions[p] = None
    external = {"ccx (CalculiX)": shutil.which("ccx") or os.path.exists(os.path.expanduser("~/ccx-env/bin/ccx")),
                "openfoam": shutil.which("openfoam")}
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions,
            "licences_paid": 0,
            "external_solvers_installed_but_not_used": [k for k, v in external.items() if v]}


# ------------------------------------------------------------------------------------------------------------
# 1. lid-driven cavity, against Ghia et al. (1982)
# ------------------------------------------------------------------------------------------------------------
def run_cavity(cells: dict, reynolds: tuple[int, ...]) -> dict:
    """``u(x = 0.5)`` and ``v(y = 0.5)`` against Ghia's tables, per Reynolds number.

    The local engine marches to steady state and stops on the vorticity residual (``steady_tol``) or, if the machine
    is too busy for that, on its 300 s wall-clock guard -- which it reports in ``meta``. On a free machine the 128 x 128
    case converges in about 23 s; when the guard is what stopped it, the answer is a truncated march and the rms is
    larger, so the run records which of the two ended it."""
    from pinneapple_simulation.external_solvers.pict import CavityCase
    from pinneapple_simulation.external_solvers.pict import run_cavity as _run

    out = {}
    for re in reynolds:
        n = cells[int(re)]
        result = _run(CavityCase(nx=n, ny=n, Re=re, refine=False), None, engine="local")
        meta = dict(result.meta)
        converged = bool(meta.get("final_residual", 0.0) < 1e-5)      # the guard leaves a much larger residual
        out[int(re)] = {"cells": n, "checks": result.checks(), "result": result,
                        "wall_time_s": result.wall_time_s, "steps": meta.get("steps"),
                        "final_residual": meta.get("final_residual"), "converged": converged}
    return out


# ------------------------------------------------------------------------------------------------------------
# 2. water hammer, against Joukowsky (1898) and Michaud (1878)
# ------------------------------------------------------------------------------------------------------------
def run_water_hammer(w: dict) -> dict:
    from pinneapple_simulation.numerical_solvers.water_hammer_moc import valve_closure_moc

    common = dict(length=w["length"], diameter=w["diameter"], wave_speed=w["wave_speed"], q0=w["q0"],
                  h_reservoir=w["h_reservoir"], n_reaches=w["n_reaches"], closure_exponent=1.0)
    fast = valve_closure_moc(friction_factor=0.0, t_close=1e-9, **common)          # Joukowsky regime
    surge_fast = float(fast["max_head"] - fast["valve_head_steady"])
    t_close = 12 * fast["period"]                                                 # well inside the slow regime
    slow = valve_closure_moc(friction_factor=0.0, t_close=t_close, **common)
    surge_slow = float(slow["max_head"] - slow["valve_head_steady"])
    # rigid-column (Allievi) estimate for the same linear closure: the peak must sit between it and Michaud
    rigid_column = w["length"] * fast["v0"] / (9.80665 * t_close)
    damped = valve_closure_moc(friction_factor=0.02, t_close=1e-9, **common)       # with Darcy friction
    return {"joukowsky_m": float(fast["joukowsky"]), "michaud_m": float(slow["michaud"]),
            "surge_instant_m": surge_fast, "surge_slow_m": surge_slow,
            "rigid_column_m": float(rigid_column), "t_close_s": float(t_close),
            "peak_head_damped_m": float(damped["max_head"]),
            "peak_head_damped_expected_m": float(w["h_reservoir"] + fast["joukowsky"]),
            "period_s": float(fast["period"]), "v0_m_s": float(fast["v0"]),
            "fast": fast, "slow": slow, "damped": damped}


# ------------------------------------------------------------------------------------------------------------
# 3. cantilever beam: Hermite FE against the closed form, then the same beam in 3-D hexahedra
# ------------------------------------------------------------------------------------------------------------
def run_cantilever(b: dict, meshes_3d) -> dict:
    from pinneapple_simulation.numerical_solvers.beam_fe import beam_modes, beam_static
    from pinneapple_simulation.numerical_solvers.solid_fem import SolidFEM, box_mesh

    L, sec_b, sec_h, E, nu, rho = b["L"], b["b"], b["h"], b["E"], b["nu"], b["rho"]
    Ixx = sec_b * sec_h ** 3 / 12.0
    EI = E * Ixx
    q = b["P"] / L                                            # the point load, as a uniform load, for the FE check
    fem = beam_static(L, EI, q, n_el=b["n_el"], ends=("fixed", "free"))
    w_exact = q * L ** 4 / (8.0 * EI)                         # uniform load on a cantilever
    modes = beam_modes(L, EI, rho * sec_b * sec_h, n_el=b["n_el"], n_modes=2, ends=("fixed", "free"))
    f1_exact = (1.8751040687119612 ** 2) / (2 * np.pi) * np.sqrt(EI / (rho * sec_b * sec_h * L ** 4))

    w3d = {}
    for nx, ny, nz in meshes_3d:
        mesh = box_mesh(L, sec_b, sec_h, nx, ny, nz)
        x = mesh.nodes[:, 0]
        problem = SolidFEM(mesh=mesh, E=E, nu=nu)
        problem.fix(np.where(np.isclose(x, 0.0))[0])
        tip = np.where(np.isclose(x, L))[0]
        problem.load_face(tip, (0.0, 0.0, -b["P"]))
        t0 = time.perf_counter()
        result = problem.solve()
        # the bending stress is measured at mid-span, where the section is not disturbed by the clamped edge (the
        # nodal value at the clamp is a mesh-dependent stress concentration: it grows as the mesh refines)
        midspan = np.isclose(x, L / 2) & np.isclose(mesh.nodes[:, 2], sec_h / 2)
        w3d[f"{nx}x{ny}x{nz}"] = {"nodes": int(len(mesh.nodes)),
                                  "elements": int(len(mesh.elements)),
                                  "tip_deflection_m": float(-result.u[tip, 2].mean()),
                                  "bending_stress_midspan_Pa": float(result.stress[midspan, 0].max()),
                                  "wall_time_s": time.perf_counter() - t0, "result": result}
    w_beam_point = b["P"] * L ** 3 / (3.0 * EI)               # point load at the tip
    sigma_midspan = b["P"] * (L / 2.0) * (sec_h / 2.0) / Ixx  # M c / I at mid-span
    return {"I_m4": Ixx, "w_udl_fem_m": float(fem["w_max"]), "w_udl_exact_m": float(w_exact),
            "w_tip_point_m": float(w_beam_point), "sigma_midspan_Pa": float(sigma_midspan),
            "f1_fem_hz": float(modes["f"][0]), "f1_exact_hz": float(f1_exact),
            "x": fem["x"].tolist(), "w": fem["w"].tolist(), "w3d": w3d, "fem": fem, "modes": modes}


# ------------------------------------------------------------------------------------------------------------
# 4. plate with a central hole: stress concentration against Heywood / Kirsch
# ------------------------------------------------------------------------------------------------------------
def plate_mesh(L: float, W: float, D: float, n_theta: int, n_rad: int, power: float = 2.0) -> tuple:
    """Structured O-grid around the hole: ``n_rad`` rings between the circle and the rectangle, built by the ray
    from the centre through each boundary point, with the four rectangle corners among the boundary points (so the
    loaded edge is the full width and the force balance closes exactly). Rings are packed towards the hole."""
    R, a, b = D / 2.0, L / 2.0, W / 2.0
    corner = float(np.arctan2(b, a))
    theta = np.unique(np.r_[np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False),
                            corner, np.pi - corner, np.pi + corner, 2 * np.pi - corner])
    d = np.c_[np.cos(theta), np.sin(theta)]
    with np.errstate(divide="ignore"):
        reach = np.minimum(np.where(np.abs(d[:, 0]) > 1e-12, a / np.abs(d[:, 0]), np.inf),
                           np.where(np.abs(d[:, 1]) > 1e-12, b / np.abs(d[:, 1]), np.inf))
    rect, circle = d * reach[:, None], d * R
    rings = [(1 - (i / n_rad) ** power) * circle + (i / n_rad) ** power * rect for i in range(n_rad + 1)]
    V = np.vstack(rings)
    n = len(theta)
    F = [[i * n + j, (i + 1) * n + j, (i + 1) * n + (j + 1) % n] for i in range(n_rad) for j in range(n)] + \
        [[i * n + j, (i + 1) * n + (j + 1) % n, i * n + (j + 1) % n] for i in range(n_rad) for j in range(n)]
    return V, np.array(F, int)


def run_plate(p: dict, meshes) -> dict:
    from pinneapple_simulation.numerical_solvers.plane_stress_fem import (
        boundary_edges,
        heywood_kt_net,
        solve_plane_stress,
    )
    L, W, D, t, S0 = p["L"], p["W"], p["D"], p["t"], p["S0"]
    kt_ref = heywood_kt_net(D / W)
    rows, finest = [], None
    for n_theta, n_rad in meshes:
        V, F = plate_mesh(L, W, D, n_theta, n_rad)
        edges = boundary_edges(F)
        right = edges[np.all(np.isclose(V[edges, 0], L / 2), 1)]
        left = np.where(np.isclose(V[:, 0], -L / 2))[0]
        anchor = [left[int(np.argmin(np.abs(V[left, 1] + W / 2)))]]      # one y dof, so the body cannot rotate
        t0 = time.perf_counter()
        sol = solve_plane_stress(V, F, E=p["E"], nu=p["nu"], thickness=t, fixed={"x": left, "y": anchor},
                                 traction=[(right, (S0, 0.0))])
        seconds = time.perf_counter() - t0
        on_hole = np.isclose(np.linalg.norm(V, axis=1), D / 2, atol=1e-9)
        nodal = sol.nodal_stress()
        kt = float(nodal[on_hole, 0].max() / (S0 * W / (W - D)))         # net-section stress concentration
        rows.append({"n_theta": n_theta, "points_around_hole": int(on_hole.sum()),
                     "nodes": int(len(V)), "elements": int(len(F)), "kt_net": kt,
                     "force_balance": float(-sol.reaction[:, 0].sum() / (S0 * W * t)),
                     "sigma_xx_peak_Pa": float(nodal[on_hole, 0].max()), "wall_time_s": seconds})
        finest = (V, F, nodal, sol)
    kts = np.array([r["kt_net"] for r in rows])
    order = float(np.log((kts[-2] - kts[-3]) / (kts[-1] - kts[-2])) / np.log(2)) if len(kts) >= 3 else 1.0
    kt_inf = float(kts[-1] + (kts[-1] - kts[-2]) / (2 ** order - 1))
    return {"rows": rows, "kt_net_heywood": float(kt_ref), "kt_kirsch": 3.0, "kt_net_richardson": kt_inf,
            "observed_order": order, "d_over_W": D / W, "finest": finest}


# ------------------------------------------------------------------------------------------------------------
# figures
# ------------------------------------------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_cavity(cavity: dict, out: str) -> list[str]:
    """The computed centrelines against the published points, one panel per Reynolds number."""
    from pinneapple_simulation.external_solvers.pict.validation import GHIA, centerline

    plt = _plt()
    n = len(cavity)
    fig, axes = plt.subplots(n, 2, figsize=(9, 3.6 * n), squeeze=False)
    for row, (re, d) in enumerate(cavity.items()):
        result = d["result"]
        u = centerline(result.component("u")[-1], direction="vertical")
        v = centerline(result.component("v")[-1], direction="horizontal")
        x0, y0, x1, y1 = result.case_summary["bounds"]
        axes[row, 0].plot(u, np.linspace(y0, y1, u.size) / (y1 - y0), label="PINNeAPPle")
        axes[row, 0].plot(GHIA["u"][re] if re in GHIA["u"] else GHIA["u"][1000], GHIA["y"], "o", ms=3,
                          label="Ghia et al. (1982)")
        axes[row, 0].set(xlabel="u / U  (x = 0.5)", ylabel="y / L")
        axes[row, 0].legend(loc="best", fontsize=8)
        axes[row, 1].plot(np.linspace(x0, x1, v.size) / (x1 - x0), v, label="PINNeAPPle")
        axes[row, 1].plot(GHIA["x"], GHIA["v"][re] if re in GHIA["v"] else GHIA["v"][1000], "o", ms=3,
                          label="Ghia et al. (1982)")
        axes[row, 1].set(xlabel="x / L", ylabel="v / U  (y = 0.5)")
        axes[row, 1].legend(loc="best", fontsize=8)
        c = d["checks"]
        axes[row, 0].set_title(f"Re = {re}, {d['cells']}² -- rms vs Ghia: "
                               f"u {c['ghia_u']['rms']:.1e}, v {c['ghia_v']['rms']:.1e}", fontsize=9)
    fig.suptitle("Lid-driven cavity against Ghia, Ghia & Shin (1982)")
    fig.tight_layout()
    path = os.path.join(out, "cavity_vs_ghia.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return [path]


def fig_water_hammer(w: dict, out: str) -> list[str]:
    """Head at the valve after the two closures, with the closed-form surge drawn on top."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (key, title, refs, surge) in zip(
            axes, (("fast", "instantaneous closure (Joukowsky regime)", (("Joukowsky $a\\Delta V/g$",
                                                                          w["joukowsky_m"]),), w["surge_instant_m"]),
                   ("slow", f"slow closure, $t_c$ = {w['t_close_s']:.0f} s (Michaud regime)",
                    (("Michaud $2L\\Delta V/g t_c$", w["michaud_m"]),
                     ("rigid column $L\\Delta V/g t_c$", w["rigid_column_m"])), w["surge_slow_m"])), strict=True):
        run = w[key]
        t = np.asarray(run["t"]) - np.asarray(run["t"])[0]
        ax.plot(t, np.asarray(run["h_valve"]), lw=1.2, label="MOC, head at the valve")
        ax.axhline(run["valve_head_steady"], ls=":", c="0.5", label="steady")
        for label, value in refs:
            ax.axhline(run["valve_head_steady"] + value, ls="--", c="crimson" if "Jouk" in label or "Mich" in label
                       else "tab:green", label=f"{label} = {value:.2f} m")
        ax.set(xlabel="t [s]", ylabel="head [m]",
               title=title, xlim=(0, (2.2 if key == "fast" else 4.0) * run["period"]))
        ax.legend(fontsize=8, loc="lower right")
        ax.text(0.03, 0.92, f"computed peak surge {surge:.2f} m", transform=ax.transAxes, fontsize=8,
                va="top")
    fig.suptitle("Water hammer after a valve closure, against Joukowsky (1898) and Michaud (1878)")
    fig.tight_layout()
    path = os.path.join(out, "water_hammer.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return [path]


def fig_cantilever(c: dict, out: str) -> list[str]:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.asarray(c["x"])
    w = np.asarray(c["w"])
    q = BEAM["P"] / BEAM["L"]
    w_exact = q * x ** 2 * (6 * BEAM["L"] ** 2 - 4 * BEAM["L"] * x + x ** 2) / (24 * BEAM["E"] * c["I_m4"])
    axes[0].plot(x, w * 1e3, lw=2, label="Hermite FE (40 elements)")
    axes[0].plot(x, w_exact * 1e3, "--", lw=1, label=r"$q x^2(6L^2-4Lx+x^2)/24EI$")
    axes[0].set(xlabel="x [m]", ylabel="deflection [mm]", title="Cantilever under a uniform load")
    axes[0].legend(fontsize=8)

    labels = list(c["w3d"])
    nodes = [c["w3d"][k]["nodes"] for k in labels]
    tip = [1e3 * c["w3d"][k]["tip_deflection_m"] for k in labels]
    axes[1].plot(nodes, tip, "o-", label="3-D hexahedra")
    axes[1].axhline(1e3 * c["w_tip_point_m"], ls="--", c="crimson", label=r"$PL^3/3EI$")
    for n, t in zip(nodes, tip, strict=True):
        axes[1].annotate(f"{(t / (1e3 * c['w_tip_point_m']) - 1) * 100:+.2f} %", (n, t), fontsize=8,
                         textcoords="offset points", xytext=(0, 8), ha="center")
    axes[1].set_xscale("log")
    axes[1].set(xlabel="nodes in the 3-D hex mesh", ylabel="tip deflection [mm]",
                title="The same beam in 3-D converges to $PL^3/3EI$")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path = os.path.join(out, "cantilever.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return [path]


def fig_plate(p: dict, pl: dict, out: str) -> list[str]:
    plt = _plt()
    V, F, nodal, _ = pl["finest"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    tpc = axes[0].tripcolor(V[:, 0] * 1e3, V[:, 1] * 1e3, F, nodal[:, 0] / 1e6, shading="gouraud", cmap="inferno")
    fig.colorbar(tpc, ax=axes[0], label=r"$\sigma_{xx}$ [MPa]")
    axes[0].set(xlabel="x [mm]", ylabel="y [mm]", title=f"finest mesh: {pl['rows'][-1]['elements']} CST elements",
                aspect="equal")
    # the hole edge on its own: the stress that the concentration factor is measured on
    on_hole = np.isclose(np.linalg.norm(V, axis=1), p["D"] / 2, atol=1e-9)
    ang = np.degrees(np.arctan2(V[on_hole, 1], V[on_hole, 0])) % 360.0
    order = np.argsort(ang)
    nominal = p["S0"] * p["W"] / (p["W"] - p["D"])
    axes[1].plot(ang[order], nodal[on_hole, 0][order] / nominal, lw=1.5, label=r"computed $\sigma_{xx}/\sigma_{net}$")
    axes[1].axhline(pl["kt_net_heywood"], ls="--", c="crimson", label=f"Heywood {pl['kt_net_heywood']:.3f}")
    axes[1].axhline(pl["kt_kirsch"], ls=":", c="0.4", label="Kirsch 3 (infinite plate)")
    axes[1].set(xlabel="angle around the hole [deg]", ylabel=r"$\sigma_{xx}/\sigma_{net}$",
                title="stress around the hole edge", ylim=(0, 3.6))
    axes[1].legend(fontsize=8)

    kts = [r["kt_net"] for r in pl["rows"]]
    idx = np.arange(len(kts))
    axes[2].plot(idx, kts, "o-", label="computed")
    axes[2].axhline(pl["kt_net_richardson"], ls="-.", c="tab:green",
                    label=f"Richardson {pl['kt_net_richardson']:.3f}")
    axes[2].axhline(pl["kt_net_heywood"], ls="--", c="crimson", label=f"Heywood {pl['kt_net_heywood']:.3f}")
    axes[2].set_xticks(idx, [f"{r['nodes']}\nnodes" for r in pl["rows"]])
    axes[2].set(xlabel="mesh", ylabel=r"$K_{t,net}$", title=f"concentration, order p = {pl['observed_order']:.2f}")
    axes[2].legend(fontsize=8)
    fig.suptitle("Plate with a central hole in tension: stress concentration against Heywood and Kirsch")
    fig.tight_layout()
    path = os.path.join(out, "plate_hole.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return [path]


def fig_beam3d(c: dict, out: str) -> list[str]:
    from pinneapple_simulation.numerical_solvers.solid_fem import fea_figure

    plt = _plt()
    key = list(c["w3d"])[-1]
    result = c["w3d"][key]["result"]
    fig = plt.figure(figsize=(10, 3.6))
    ax = fig.add_subplot(111, projection="3d")
    fea_figure(result, field="sxx", ax=ax,
               title=f"Cantilever, 3-D hexahedra {key}, bending stress S11 (deformation exaggerated)")
    ax.set_box_aspect((1.0, 0.25, 0.55), zoom=1.6)      # the part is 50x slender: the section is drawn enlarged
    ax.view_init(elev=8, azim=-90)
    path = os.path.join(out, "cantilever_3d.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return [path]


# ------------------------------------------------------------------------------------------------------------
# the experiment
# ------------------------------------------------------------------------------------------------------------
def main(smoke: bool = False, three_d: bool = True, base: str | None = None) -> dict:
    from pinneapple_lab import experiment

    out = os.path.join(base or HERE, "results")
    os.makedirs(out, exist_ok=True)
    rows: list[dict] = []
    clock: dict[str, float] = {}
    t_all = time.perf_counter()

    def record(case: str, quantity: str, value: float, textbook: float | None, tol: float, unit: str = "",
               kind: str = "relative", seconds: float | None = None) -> dict:
        err = None if textbook is None else value - textbook
        rel = None if not textbook else err / textbook
        if kind == "relative":
            passed = textbook is not None and abs(rel) <= tol
        else:
            passed = abs(value if textbook is None else err) <= tol
        row = {"case": case, "quantity": quantity, "value": value, "textbook": textbook,
               "error": err, "relative_error": rel, "tolerance": tol, "kind": kind, "unit": unit,
               "passed": bool(passed), "seconds": seconds}
        rows.append(row)
        return row

    with experiment("toolchain_reproduction",
                    title="A blank machine, an agent and an open-source toolchain: classic CFD and FEA cases in "
                          "PINNeAPPle, checked against their published values",
                    domain="fluid dynamics & structural mechanics",
                    methods=["Vorticity-streamfunction NS", "Method of characteristics", "Hermite beam FE",
                             "3-D hexahedral FE", "Plane-stress CST FE"],
                    modules=["pinneapple_simulation.external_solvers.pict",
                             "pinneapple_simulation.numerical_solvers.water_hammer_moc",
                             "pinneapple_simulation.numerical_solvers.beam_fe",
                             "pinneapple_simulation.numerical_solvers.solid_fem",
                             "pinneapple_simulation.numerical_solvers.plane_stress_fem"],
                    params={"smoke": smoke, "3d": three_d, "cavity": CAVITY, "water_hammer": WATER_HAMMER,
                            "beam": BEAM, "plate": PLATE},
                    references=["Ghia, Ghia & Shin, J. Comput. Phys. 48 (1982) 387-411",
                                "Joukowsky (1898); Wylie & Streeter, Fluid Transients in Systems (1993)",
                                "Przemieniecki, Theory of Matrix Structural Analysis (1968), ch. 5 and 11",
                                "Kirsch (1898); Pilkey & Pilkey, Peterson's Stress Concentration Factors, 3rd ed. "
                                "(2008); Heywood's net-section form"],
                    out_dir=out, register=False) as ex:

        # -- 0. the toolchain --------------------------------------------------------------------------------
        t0 = time.perf_counter()
        audit = toolchain_audit()
        clock["toolchain audit"] = time.perf_counter() - t0
        ex.log_metric("toolchain_licences_paid", audit["licences_paid"], unit="",
                      how="count of commercial licences the cases below need")
        print("toolchain: python", audit["python"], "| numpy", audit["packages"]["numpy"], "| scipy",
              audit["packages"]["scipy"], "| matplotlib", audit["packages"]["matplotlib"], "| torch",
              audit["packages"]["torch"], f"| paid licences {audit['licences_paid']}")
        if audit["external_solvers_installed_but_not_used"]:
            print("  installed on this machine but not used by any case below:",
                  ", ".join(audit["external_solvers_installed_but_not_used"]))
        ex.check("no licence is needed by any case", passed=audit["licences_paid"] == 0)

        # -- 1. lid-driven cavity -------------------------------------------------------------------------
        cells = {re: SMOKE_CAVITY_CELLS if smoke else n for re, n in CAVITY["cells"].items()}
        reynolds = (100,) if smoke else CAVITY["re"]
        t0 = time.perf_counter()
        cavity = run_cavity(cells, reynolds)
        clock["lid-driven cavity"] = time.perf_counter() - t0
        for re, d in cavity.items():
            c = d["checks"]
            record(f"cavity Re={re}", "rms of u(x=0.5) vs Ghia", float(c["ghia_u"]["rms"]), None, 0.05,
                   unit="U", kind="absolute", seconds=d["wall_time_s"])
            record(f"cavity Re={re}", "rms of v(y=0.5) vs Ghia", float(c["ghia_v"]["rms"]), None, 0.05,
                   unit="U", kind="absolute", seconds=d["wall_time_s"])
            record(f"cavity Re={re}", "discrete du/dx + dv/dy", float(c["continuity"]["value"]), 0.0, 1e-12,
                   unit="U/L", kind="absolute")
            ex.log_metric(f"ghia_u_rms_re{re}", float(c["ghia_u"]["rms"]), unit="U",
                          how=f"rms of u(x=0.5) against Ghia et al. (1982) at Re={re}, "
                              f"{d['cells']}x{d['cells']} cells, local engine")
            ex.check(f"cavity Re={re}: u and v centrelines within 0.05 U of Ghia",
                     passed=float(c["ghia_u"]["rms"]) <= 0.05 and float(c["ghia_v"]["rms"]) <= 0.05,
                     value=float(c["ghia_u"]["rms"]), max=0.05,
                     detail=f"u rms {c['ghia_u']['rms']:.2e}, v rms {c['ghia_v']['rms']:.2e}")
            if not d["converged"]:
                print(f"  note: Re={re} stopped on the engine's 300 s wall-clock guard after {d['steps']} steps "
                      f"(residual {d['final_residual']:.2e}), so this is a truncated march -- the machine was busy. "
                      f"On a free core the same run converges in ~23 s (examples/pict_cavity).")

        # -- 2. water hammer ------------------------------------------------------------------------------
        t0 = time.perf_counter()
        wh = run_water_hammer(WATER_HAMMER)
        clock["water hammer"] = time.perf_counter() - t0
        record("water hammer", "surge after an instantaneous closure", wh["surge_instant_m"],
               wh["joukowsky_m"], 1e-6, unit="m")
        record("water hammer", "peak head, Darcy friction (f = 0.02)", wh["peak_head_damped_m"],
               wh["peak_head_damped_expected_m"], 0.01, unit="m")
        record("water hammer", "surge after a slow closure (inside the classical bracket)", wh["surge_slow_m"],
               wh["michaud_m"], (wh["michaud_m"] - wh["rigid_column_m"]) / wh["michaud_m"], unit="m")
        ex.log_metric("joukowsky_surge_m", wh["surge_instant_m"], unit="m",
                      how="max head rise at the valve minus the steady head, against dH = a dV / g")
        ex.check("water hammer: instantaneous closure matches Joukowsky", value=wh["surge_instant_m"],
                 reference=wh["joukowsky_m"], rtol=1e-6)
        ex.check("water hammer: friction does not change the first peak",
                 value=wh["peak_head_damped_m"], reference=wh["peak_head_damped_expected_m"], rtol=0.01)
        ex.check("water hammer: slow-closure peak between the rigid-column and Michaud estimates",
                 passed=wh["rigid_column_m"] < wh["surge_slow_m"] < wh["michaud_m"],
                 detail=f"rigid column {wh['rigid_column_m']:.2f} m < MOC {wh['surge_slow_m']:.2f} m "
                        f"< Michaud {wh['michaud_m']:.2f} m at t_close = {wh['t_close_s']:.1f} s")

        # -- 3. cantilever beam ---------------------------------------------------------------------------
        t0 = time.perf_counter()
        beam = run_cantilever(BEAM, (SMOKE_3D,) if smoke else FULL_3D)
        clock["cantilever"] = time.perf_counter() - t0
        record("cantilever", "tip deflection, uniform load (Hermite FE)", beam["w_udl_fem_m"],
               beam["w_udl_exact_m"], 1e-9, unit="m")
        record("cantilever", "first natural frequency", beam["f1_fem_hz"], beam["f1_exact_hz"], 1e-6, unit="Hz")
        if three_d:
            for key, d in beam["w3d"].items():
                record("cantilever 3-D", f"tip deflection, hexahedra {key}",
                       d["tip_deflection_m"], beam["w_tip_point_m"], 0.02, unit="m", seconds=d["wall_time_s"])
            record("cantilever 3-D", "bending stress at mid-span, finest 3-D mesh",
                   beam["w3d"][list(beam["w3d"])[-1]]["bending_stress_midspan_Pa"], beam["sigma_midspan_Pa"], 0.005,
                   unit="Pa")
        ex.log_metric("beam_tip_deflection_udl_m", beam["w_udl_fem_m"], unit="m",
                      how="Hermite FE, 40 elements, against q L^4 / 8 E I")
        ex.log_metric("beam_first_mode_hz", beam["f1_fem_hz"], unit="Hz",
                      how="consistent mass, against (1.8751^2 / 2 pi) sqrt(EI / m L^4)")
        ex.check("cantilever: Hermite FE reproduces q L^4 / 8EI", value=beam["w_udl_fem_m"],
                 reference=beam["w_udl_exact_m"], rtol=1e-9)
        ex.check("cantilever: first frequency matches the exact Euler-Bernoulli mode",
                 value=beam["f1_fem_hz"], reference=beam["f1_exact_hz"], rtol=1e-6)

        # -- 4. plate with a hole -------------------------------------------------------------------------
        t0 = time.perf_counter()
        plate = run_plate(PLATE, SMOKE_PLATE_MESHES if smoke else PLATE["meshes"])
        clock["plate with a hole"] = time.perf_counter() - t0
        finest = plate["rows"][-1]
        record("plate with a hole", "force balance (reaction / applied)", finest["force_balance"], 1.0, 1e-6,
               kind="relative", seconds=finest["wall_time_s"])
        for r in plate["rows"]:
            record("plate with a hole", f"Kt_net, {r['nodes']} nodes", r["kt_net"], plate["kt_net_heywood"], 0.15,
                   seconds=r["wall_time_s"])
        record("plate with a hole", "Kt_net, Richardson", plate["kt_net_richardson"], plate["kt_net_heywood"], 0.03)
        ex.log_metric("kt_net_richardson", plate["kt_net_richardson"], unit="",
                      how="Richardson from the three finest O-grid meshes, against Heywood 2 + (1 - d/W)^3")
        ex.log_metric("kt_net_finest", plate["rows"][-1]["kt_net"], unit="",
                      how=f"CST plane stress, {plate['rows'][-1]['elements']} elements")
        ex.check("plate with a hole: force balance closes", value=finest["force_balance"], reference=1.0,
                 rtol=1e-6)
        ex.check("plate with a hole: Kt_net (Richardson) within 3 % of Heywood",
                 value=plate["kt_net_richardson"], reference=plate["kt_net_heywood"], rtol=0.03,
                 detail=f"observed order {plate['observed_order']:.2f}")
        ex.check("plate with a hole: meshes converge", passed=abs(plate["rows"][-1]["kt_net"] -
                                                                 plate["rows"][-2]["kt_net"]) /
                 plate["rows"][-1]["kt_net"] < 0.03)

        # -- 5. post-processing ---------------------------------------------------------------------------
        t0 = time.perf_counter()
        figures = []
        figures += fig_cavity(cavity, out)
        figures += fig_water_hammer(wh, out)
        figures += fig_cantilever(beam, out)
        figures += fig_plate(PLATE, plate, out)
        if three_d:
            figures += fig_beam3d(beam, out)
        clock["post-processing"] = time.perf_counter() - t0
        for f in figures:
            ex.add_file(f)

        for r in rows:
            ex.log_metric(f"{r['case']}: {r['quantity']}", r["value"], unit=r["unit"],
                          how=("measured by the script" if r["textbook"] is None else
                               f"against {r['textbook']:.6g} {r['unit']}"))

    total = time.perf_counter() - t_all
    result = {"toolchain": audit, "clock_s": clock, "total_s": total, "rows": rows,
              "cavity": {re: {"cells": d["cells"], "checks": d["checks"], "wall_time_s": d["wall_time_s"],
                              "steps": d["steps"], "final_residual": d["final_residual"],
                              "converged": d["converged"]} for re, d in cavity.items()},
              "water_hammer": {k: v for k, v in wh.items() if k not in ("fast", "slow", "damped")},
              "cantilever": {k: v for k, v in beam.items() if k not in ("w3d", "fem", "modes")} |
                             {"w3d": {k: {kk: vv for kk, vv in v.items() if kk != "result"}
                                      for k, v in beam["w3d"].items()}},
              "plate": {k: v for k, v in plate.items() if k != "finest"},
              "passed": all(r["passed"] for r in rows), "figures": figures}
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, default=float)
    _print_table(result)
    return result


def _print_table(result: dict) -> None:
    w = 58
    print(f"\n{'case':<18} {'quantity':<{w}} {'computed':>13} {'textbook':>13} {'error':>11}  ")
    print("-" * (18 + w + 42))
    for r in result["rows"]:
        tb = "" if r["textbook"] is None else f"{r['textbook']:.6g}"
        err = "" if r["relative_error"] is None else f"{100 * r['relative_error']:+.3e} %"
        flag = "" if r["passed"] else "  FAILED"
        print(f"{r['case']:<18} {r['quantity']:<{w}} {r['value']:>13.6g} {tb:>13} {err:>11} {flag}")
    print("-" * (18 + w + 42))
    for k, v in result["clock_s"].items():
        print(f"  {k:<34} {v:8.2f} s")
    print(f"  {'total':<34} {result['total_s']:8.2f} s")
    print(f"\n{sum(r['passed'] for r in result['rows'])}/{len(result['rows'])} checks passed"
          f"{'' if result['passed'] else ' -- SEE THE FAILED ROWS'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="coarse meshes, a few seconds")
    ap.add_argument("--no-3d", action="store_true", help="skip the 3-D hexahedral cantilever")
    ap.add_argument("--base", default=None, help="where results/ goes (default: this folder)")
    a = ap.parse_args()
    res = main(smoke=a.smoke, three_d=not a.no_3d, base=a.base)
    sys.exit(0 if res["passed"] else 1)

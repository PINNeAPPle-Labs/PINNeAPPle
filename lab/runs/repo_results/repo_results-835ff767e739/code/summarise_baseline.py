"""Baseline summary: grid study (GCI), comparison with XFOIL and thin-airfoil theory, figures.

    python summarise_baseline.py RUNS_DIR OUT_DIR
        RUNS_DIR holds coarse/, medium/, fine/ (airfoil_case.py ... --grid G --run) and optionally opt/optimised.json
        OUT_DIR gets baseline_summary.json, grid_study.csv and the figures

Uses ../results/reference/xfoil_summary.json and ../results/campaign/campaign_runs.csv.
Grid convergence index after Celik et al. (2008), J. Fluids Eng. 130(7), 078001, three grids, safety factor 1.25.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np

from pinneapple_lab.uq import gci

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
GRIDS = ("coarse", "medium", "fine")


def grid_study(f1: float, f2: float, f3: float, N1: int, N2: int, N3: int) -> dict:
    """GCI of the fine (1), medium (2) and coarse (3) values with pinneapple_lab.uq.gci; 2-D cell size h = 1/sqrt(N).
    Adds the medium-grid GCI and the field names of baseline_summary.json."""
    h1, h2, h3 = (N ** -0.5 for N in (N1, N2, N3))
    g = gci([f3, f2, f1], [h3, h2, h1])
    p, r21, r32 = g["order"], g["ratio"], h3 / h2
    return {"r21": r21, "r32": r32, "observed_order": p, "oscillatory": bool((f3 - f2) / (f2 - f1) < 0),
            "extrapolated": g["extrapolated"], "rel_error_fine_vs_medium": abs((f1 - f2) / f1),
            "GCI_fine_pct": 100 * g["gci"],
            "GCI_medium_pct": 100 * g["safety"] * abs((f2 - f3) / f2) / (r32 ** p - 1)}


def thin_airfoil_alpha_l0(m: float = 0.02, p: float = 0.4) -> float:
    """Zero-lift angle (deg) of a NACA 4-digit mean line from thin-airfoil theory."""
    th = np.linspace(0, np.pi, 20001)
    x = 0.5 * (1 - np.cos(th))
    dy = np.where(x < p, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    return float(np.degrees(-np.trapezoid(dy * (np.cos(th) - 1), th) / np.pi))


def main():
    runs, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    R = {g: json.load(open(os.path.join(runs, g, "result.json"))) for g in GRIDS}
    info = {g: json.load(open(os.path.join(runs, g, "case_info.json"))) for g in GRIDS}
    N = {g: info[g]["cells"] for g in GRIDS}
    study = {}
    for k in ("Cl", "Cd", "Cm", "L_over_D"):
        study[k] = grid_study(R["fine"][k], R["medium"][k], R["coarse"][k], N["fine"], N["medium"], N["coarse"])
    with open(os.path.join(out, "grid_study.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["grid", "cells", "ni", "nj", "first_cell_over_c", "iterations", "Cl", "Cd", "Cm", "L_over_D",
                    "Cl_change_last_500_it", "Cd_change_last_500_it", "y_plus_max", "seconds"])
        for g in GRIDS:
            r = R[g]
            H = np.array(r["history"])
            last = H[H[:, 0] >= H[-1, 0] - 500]
            w.writerow([g, N[g], info[g]["ni"], info[g]["nj"], info[g]["first_cell"], r["iterations"], f"{r['Cl']:.5f}",
                        f"{r['Cd']:.6f}", f"{r['Cm']:.5f}", f"{r['L_over_D']:.2f}",
                        f"{np.ptp(last[:, 1]):.2e}", f"{np.ptp(last[:, 2]):.2e}", f"{r['y_plus_max']:.2f}", r["seconds"]])
    xf = json.load(open(os.path.join(PKG, "results", "reference", "xfoil_summary.json")))
    at4 = {k: next((p for p in v if abs(p["alpha"] - 4) < 1e-9), None) for k, v in xf["polars"].items()}
    camp = [r for r in csv.DictReader(open(os.path.join(PKG, "results", "campaign", "campaign_runs.csv")))
            if r["shape_id"] == "NACA 2412"]
    a_c = np.array([float(r["alpha_deg"]) for r in camp])
    cl_c = np.array([float(r["Cl"]) for r in camp])
    cd_c = np.array([float(r["Cd"]) for r in camp])
    lin = (a_c >= -2) & (a_c <= 6)
    slope, icpt = np.polyfit(a_c[lin], cl_c[lin], 1)
    xt = xf["polars"]["naca2412_exact_tripped"]
    xa = np.array([p["alpha"] for p in xt])
    xc = np.array([p["Cl"] for p in xt])
    sel = (xa >= -2) & (xa <= 6)
    xs, xi = np.polyfit(xa[sel], xc[sel], 1)
    med = R["medium"]
    comp = {}
    for k, p in at4.items():
        if p:
            comp[k] = {"Cl": p["Cl"], "Cd": p["Cd"], "Cm": p["Cm"], "L_over_D": p["Cl"] / p["Cd"],
                       "CFD_medium_minus_ref_Cl_pct": 100 * (med["Cl"] / p["Cl"] - 1),
                       "CFD_medium_minus_ref_Cd_pct": 100 * (med["Cd"] / p["Cd"] - 1),
                       "CFD_extrapolated_minus_ref_Cd_pct": 100 * (study["Cd"]["extrapolated"] / p["Cd"] - 1)}
    summary = {
        "case": {"airfoil": "NACA 2412 (6-weight CST fit)", "alpha_deg": 4.0, "reynolds": 4e6, "mach": 0.0,
                 "solver": "OpenFOAM v1912 simpleFoam, k-omega SST (Menter 1994), fully turbulent, y+ < 1"},
        "openfoam": {g: {k: R[g][k] for k in ("Cl", "Cd", "Cm", "L_over_D", "iterations", "y_plus_max", "Cl_osc",
                                              "Cd_osc", "seconds")} | {"cells": N[g]} for g in GRIDS},
        "grid_convergence": study,
        "xfoil_at_alpha_4": comp,
        "lift_curve": {"openfoam_campaign_slope_per_deg": slope, "openfoam_campaign_alpha_L0_deg": -icpt / slope,
                       "xfoil_tripped_slope_per_deg": xs, "xfoil_tripped_alpha_L0_deg": -xi / xs,
                       "thin_airfoil_theory_alpha_L0_deg": thin_airfoil_alpha_l0(),
                       "thin_airfoil_theory_slope_per_deg": 2 * math.pi * math.pi / 180,
                       "fit_range_deg": [-2, 6]},
    }
    opt = os.path.join(runs, "opt", "optimised.json")
    if os.path.exists(opt):
        summary["optimised"] = json.load(open(opt))
        # the brief's optimum on the three grids (optA_coarse/, opt/ medium, optA_fine/)
        oc, of = (os.path.join(runs, d, "result.json") for d in ("optA_coarse", "optA_fine"))
        if os.path.exists(oc) and os.path.exists(of):
            om = summary["optimised"]["designs"]["opt_A_brief"]["openfoam"]
            G = {"coarse": json.load(open(oc)), "medium": om, "fine": json.load(open(of))}
            summary["optimum_grid_study"] = {
                "design": "opt_A_brief",
                "grids": {g: {k: G[g][k] for k in ("Cl", "Cd", "Cm", "L_over_D")} for g in GRIDS},
                "gain_over_baseline_pct": {g: 100 * (G[g]["L_over_D"] / R[g]["L_over_D"] - 1) for g in GRIDS},
                **{f"GCI_{k}": grid_study(G["fine"][k], G["medium"][k], G["coarse"][k], N["fine"], N["medium"], N["coarse"])
                   for k in ("Cd", "L_over_D")}}
    with open(os.path.join(out, "baseline_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    figures(out, R, info, xf, a_c, cl_c, cd_c)
    print(json.dumps({"grid": {g: (round(R[g]["Cl"], 4), round(R[g]["Cd"], 6)) for g in GRIDS},
                      "Cd_extrapolated": study["Cd"]["extrapolated"], "Cd_GCI_fine_pct": study["Cd"]["GCI_fine_pct"],
                      "Cl_GCI_fine_pct": study["Cl"]["GCI_fine_pct"], "p_Cd": study["Cd"]["observed_order"],
                      "xfoil": {k: (v["Cl"], v["Cd"]) for k, v in comp.items()}, "lift": summary["lift_curve"]},
                     indent=1))


def figures(out, R, info, xf, a_c, cl_c, cd_c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"coarse": "#9aa9c9", "medium": "#1f3b73", "fine": "#c2491d"}
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), dpi=140)
    for g in GRIDS:
        H = np.array(R[g]["history"])
        ax[0].plot(H[:, 0], H[:, 1], color=col[g], label=f"{g} ({info[g]['cells']} cells)")
        ax[1].plot(H[:, 0], H[:, 2], color=col[g])
    ax[0].set_ylim(0.6, 0.7)
    ax[1].set_ylim(0.0095, 0.0115)
    for a, t in zip(ax[:2], ("Cl", "Cd"), strict=False):
        a.set_xlabel("iteration")
        a.set_title(f"{t} at the last 6 writes (every 100 iterations)", fontsize=9)
    ax[0].legend(fontsize=7)
    res = R["medium"]["residuals"]
    for k in ("Ux", "Uy", "p", "k", "omega"):
        if k in res:
            v = np.array(res[k])
            n = len(res["Ux"])
            ax[2].semilogy(v[:: max(1, len(v) // n)][:n], lw=0.7, label=k)      # p: first solve per iteration
    ax[2].set_xlabel("iteration")
    ax[2].set_title("initial residuals, medium grid", fontsize=9)
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "convergence.png"))
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=140)
    sty = {"naca2412_exact_free": ("XFOIL, free transition (Ncrit 9)", "#2e8b57", "--"),
           "naca2412_exact_tripped": ("XFOIL, tripped at 1 % chord", "#7a4fa3", "--")}
    for k, (lab, c, ls) in sty.items():
        P = xf["polars"][k]
        a = [p["alpha"] for p in P]
        ax[0].plot(a, [p["Cl"] for p in P], ls, color=c, label=lab)
        ax[1].plot([p["Cd"] for p in P], [p["Cl"] for p in P], ls, color=c, label=lab)
    o = np.argsort(a_c)
    ax[0].plot(a_c[o], cl_c[o], "o-", color="#1f3b73", ms=4, label="OpenFOAM k-omega SST (campaign grid)")
    ax[1].plot(cd_c[o], cl_c[o], "o-", color="#1f3b73", ms=4, label="OpenFOAM k-omega SST (campaign grid)")
    for g in GRIDS:
        ax[0].plot(4, R[g]["Cl"], "s", color=col[g], ms=5, label=f"baseline run, {g} grid")
        ax[1].plot(R[g]["Cd"], R[g]["Cl"], "s", color=col[g], ms=5)
    ax[0].set_xlabel("alpha (deg)")
    ax[0].set_ylabel("Cl")
    ax[1].set_xlabel("Cd")
    ax[1].set_ylabel("Cl")
    ax[1].set_xlim(0.004, 0.03)
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.suptitle("NACA 2412, Re 4 million", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "polar_vs_xfoil.png"))
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
    for g in ("medium", "fine"):
        w = R[g]["wall"]
        ax.plot(w["x"], -np.array(w["Cp"]), lw=0.9, color=col[g], label=f"OpenFOAM {g}")
    cp = os.path.join(PKG, "results", "reference", "xfoil_cp_alpha4_tripped.txt")
    if os.path.exists(cp):
        A = np.loadtxt(cp, skiprows=1)
        ax.plot(A[:, 0], -A[:, -1], "--", color="#7a4fa3", lw=0.9, label="XFOIL tripped (inviscid-viscous Cp)")
    ax.set_xlabel("x / c")
    ax.set_ylabel("-Cp")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title("Surface pressure, alpha 4 deg", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "cp_alpha4.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()

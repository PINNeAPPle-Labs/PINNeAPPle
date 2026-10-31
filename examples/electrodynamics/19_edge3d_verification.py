"""Verification of the 3-D edge-element (Nedelec) solver, ``pinneapple_simulation.electromagnetics.fem3d``.

1. Permeable sphere (mu_r = 10) in a uniform field, exact exterior potential on the outer ball (no truncation
   error): mean field and L2 errors against the closed form under mesh refinement (gmsh meshes).
2. The same meshes with the uniform-field boundary condition, solved by the native code and by GetDP
   (``fem3d.getdp3d``): the same discrete problem, so both codes must agree to solver precision.
3. Manufactured magnetostatic solution on structured cube meshes: observed order of the B error.
4. Conducting slab in a uniform AC field (200 Hz, 20 mm aluminium-like slab): B and loss against the closed form.

    python examples/electrodynamics/19_edge3d_verification.py          # needs gmsh and getdp; about 3 min
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np

from pinneapple_simulation.electromagnetics.fem3d import getdp3d, read_msh22
from pinneapple_simulation.electromagnetics.fem3d.cases import manufactured_magnetostatic, slab_eddy
from pinneapple_simulation.electromagnetics.fem3d.nedelec import NU0, EdgeSpace, solve_magnetostatic
from pinneapple_simulation.electromagnetics.fem3d.sphere import (
    SphereCase,
    errors,
    mesh_sphere,
    solve_native,
    uniform_dirichlet,
)

OUT = Path(__file__).parent / "results"
H_SPHERE = (0.3, 0.2, 0.14, 0.1, 0.07)
H_GETDP = (0.3, 0.2, 0.14)


def orders(h, e):
    h, e = np.asarray(h), np.asarray(e)
    return (np.log(e[:-1] / e[1:]) / np.log(h[:-1] / h[1:])).tolist()


def main() -> dict:
    t0 = time.perf_counter()
    case = SphereCase(a=1.0, R=3.0, mu_r=10.0)
    tmp = Path(tempfile.mkdtemp(prefix="edge3d_"))
    sphere, cross = [], []
    for h in H_SPHERE:
        msh = mesh_sphere(tmp / f"s{h}.msh", h, case)
        r = solve_native(msh, case)
        r["h_target"] = h
        sphere.append(r)
        print(f"sphere h={h}: {r['unknowns']} unknowns, mean {r['mean_in']:.3e}, L2 in {r['l2_in']:.3e}, "
              f"L2 all {r['l2_all']:.3e}, {r['seconds']:.1f} s", flush=True)
        if h in H_GETDP:
            g = getdp3d.sphere(msh, case)
            m = read_msh22(msh)
            sp = EdgeSpace(m)
            nu = np.where(m.mask("sphere"), NU0 / case.mu_r, NU0)
            t1 = time.perf_counter()
            s = solve_magnetostatic(m, nu, dirichlet=uniform_dirichlet(m, sp, case), space=sp, cg_tol=1e-12)
            e = errors(m, s.B, case)
            row = {"h_target": h, "tets": int(len(m.tets)), "native_Bz_mean": e["Bz_mean_in"],
                   "getdp_Bz_mean": g["Bz_mean_in"], "native_energy": s.energy, "getdp_energy": g["energy"],
                   "rel_diff_Bz": abs(e["Bz_mean_in"] - g["Bz_mean_in"]) / abs(g["Bz_mean_in"]),
                   "rel_diff_energy": abs(s.energy - g["energy"]) / abs(g["energy"]),
                   "truncated_mean_error": e["mean_in"], "native_s": time.perf_counter() - t1,
                   "getdp_s": g["seconds"]}
            cross.append(row)
            print(f"  GetDP: Bz {g['Bz_mean_in']:.12f} native {e['Bz_mean_in']:.12f} (rel {row['rel_diff_Bz']:.1e}),"
                  f" energy rel {row['rel_diff_energy']:.1e}", flush=True)
    hs = [r["h"] for r in sphere]
    manu = []
    for n in (5, 9, 17, 33):
        r = manufactured_magnetostatic(n)
        manu.append({"h": r["h"], "error": r["error"], "unknowns": r["unknowns"]})
        print(f"manufactured n={n}: error {r['error']:.4e}", flush=True)
    slab = []
    for nz, nxy in ((4, 2), (8, 4), (16, 8)):
        r = slab_eddy(nz, nxy=nxy, gauge="tree")
        slab.append({"nz": nz, "nxy": nxy, "error_B": r["error_B"], "loss_per_area": r["loss_per_area"],
                     "loss_exact": r["loss_exact"], "loss_error": r["loss_error"], "unknowns": r["unknowns"],
                     "delta_mm": 1e3 * r["delta"]})
        print(f"slab nz={nz}: B error {r['error_B']:.3e}, loss {r['loss_per_area']:.2f} vs {r['loss_exact']:.2f} "
              f"W/m^2 ({100 * r['loss_error']:.2f} %)", flush=True)
    out = {
        "sphere": {"case": {"a": case.a, "R": case.R, "mu_r": case.mu_r, "B_in_exact": case.B_in[2]},
                   "rows": sphere,
                   "orders": {k: orders(hs, [r[k] for r in sphere]) for k in ("mean_in", "l2_in", "l2_all")}},
        "getdp_cross_check": cross,
        "manufactured": {"rows": manu, "orders": orders([r["h"] for r in manu], [r["error"] for r in manu])},
        "slab_eddy": slab,
        "seconds": time.perf_counter() - t0,
        "date": time.strftime("%Y-%m-%d"),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "19_edge3d_verification.json").write_text(json.dumps(out, indent=1))
    figure(out)
    return out


def figure(out: dict) -> None:
    import matplotlib.pyplot as plt

    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    ink, muted = "#0b0b0b", "#52514e"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.0), facecolor="#fcfcfb")
    rows = out["sphere"]["rows"]
    h = np.array([r["h"] for r in rows])
    for k, c, lab in (("mean_in", blue, "mean B in sphere"), ("l2_in", orange, "L2 of B in sphere"),
                      ("l2_all", aqua, "L2 of B, whole domain")):
        e = np.array([r[k] for r in rows])
        ax[0].loglog(h, e, "-o", color=c, lw=2, ms=6, label=lab)
    ax[0].loglog(h, 0.06 * (h / h[0]) ** 2, ":", color=muted, lw=1)
    ax[0].loglog(h, 0.05 * (h / h[0]), "--", color=muted, lw=1)
    ax[0].text(h[-1] * 0.97, 0.06 * (h[-1] / h[0]) ** 2, "slope 2 ", color=muted, ha="right", va="center", fontsize=8)
    ax[0].text(h[-1] * 0.97, 0.05 * (h[-1] / h[0]), "slope 1 ", color=muted, ha="right", va="center", fontsize=8)
    ax[0].set_xlim(h[-1] * 0.7, h[0] * 1.15)
    ax[0].set(xlabel="median edge length in the sphere / radius", ylabel="relative error",
              title="Sphere, mu_r = 10, vs closed form")
    ax[0].legend(frameon=False, fontsize=8)
    m = out["manufactured"]["rows"]
    ax[1].loglog([r["h"] for r in m], [r["error"] for r in m], "-o", color=blue, lw=2, ms=6)
    ax[1].set(xlabel="cell size", ylabel="relative L2 error of B",
              title=f"Manufactured solution, order {out['manufactured']['orders'][-1]:.2f}")
    s = out["slab_eddy"]
    ax[2].plot([r["nz"] for r in s], [r["loss_per_area"] for r in s], "-o", color=orange, lw=2, ms=6,
               label="edge elements")
    ax[2].axhline(s[0]["loss_exact"], color=muted, ls="--", lw=1, label="closed form")
    ax[2].set(xlabel="cells across the slab", ylabel="loss (W/m²)", title="Slab eddy currents, 200 Hz")
    ax[2].legend(frameon=False, fontsize=8)
    from matplotlib.ticker import NullFormatter
    for a in ax[:2]:
        a.xaxis.set_minor_formatter(NullFormatter())
    for a in ax:
        a.set_facecolor("#fcfcfb")
        a.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "19_edge3d_verification.png", dpi=150)


if __name__ == "__main__":
    print(json.dumps({k: v for k, v in main().items() if k in ("seconds",)}))


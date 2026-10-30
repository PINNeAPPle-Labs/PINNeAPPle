"""Immersed against conforming FEM on the same parts, at equal accuracy.

For each part (a sphere and a finite cylinder with convection in closed form, and a manufactured temperature on a
sphere) the immersed solver (Q1, Q2 on a background grid) and conforming tetrahedra (P1, P2 on a gmsh mesh, P2
nodes on the CAD surface; the same Python assembly and linear solver; and CalculiX on the identical P2 mesh) are run
over a sequence of resolutions. Every run records the unknowns, the error against the exact field and the wall time
by phase: the geometry phase (cut-cell quadrature for immersed, meshing for conforming), assembly and solve. The
equal-accuracy table interpolates, per method, the unknowns and the time needed to reach a target error (log-log,
between measured runs only: no extrapolation).

    from pinneapple_simulation.immersed import compare
    rows = compare.sweep(compare.parts()["sphere"], quick=True)
    compare.equal_accuracy(rows, targets=[1e-3, 1e-4])
"""
from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import verification as V
from .assembly import Discretization
from .heat import field_errors, solve_heat


@dataclass
class Part:
    name: str
    case: V.Case
    occ: Callable                       # gmsh OpenCASCADE builder of the same solid
    mantle: Callable | None = None      # faces (by centroid) that carry the convection, for CalculiX (None = all)
    calculix: bool = False              # constant source and film data: CalculiX can run it as is


def parts() -> dict[str, Part]:
    from pinneapple_simulation.numerical_solvers import tet_fem as T
    c = V._shift(3)
    sph = V.sphere_convection()
    cyl = V.cylinder_convection()
    mms = V.manufactured(3, "ball")
    return {
        "sphere": Part("sphere", sph, T.occ_sphere(c, 0.5), calculix=True),
        "cylinder": Part("cylinder", cyl, T.occ_cylinder(c, 0.4, 1.0),
                         mantle=lambda C: np.abs(C[:, 2] - c[2]) < 0.5 - 1e-9, calculix=True),
        "sphere_mms": Part("sphere_mms", mms, T.occ_sphere(c, 0.7)),
    }


def run_immersed(part: Part, order: int, n: int, **kw) -> dict:
    """One immersed run on a grid of ``n`` cells across the part; errors on the fine octree rule of the same grid
    (built separately, outside the timing)."""
    case = part.case
    t0 = time.perf_counter()
    disc = Discretization(case.geometry, case.size / n, order=order, **kw)
    sol = solve_heat(disc, case.conductivity, case.source, case.bcs)
    wall = time.perf_counter() - t0
    ref = Discretization(case.geometry, case.size / n, order=order, keep_fine=True, **kw)
    e = field_errors(ref, sol.T, case.exact, case.grad)
    return {"part": part.name, "method": f"immersed Q{order}", "level": n, "dofs": disc.n_dofs,
            "L2_rel": e["L2_rel"], "H1_rel": e["H1_rel"], "geometry_s": disc.timings["quadrature_s"],
            "assembly_s": sol.info["assembly_s"], "solve_s": sol.info["solve_seconds"], "wall_s": wall,
            "solver": sol.info["solve_solver"], "energy_imbalance": sol.balance["imbalance"]}


def run_conforming(part: Part, order: int, size: float) -> dict:
    from pinneapple_simulation.numerical_solvers import tet_fem as T
    case = part.case
    t0 = time.perf_counter()
    mesh = T.gmsh_mesh(part.occ, size, order)
    sol = T.solve_heat(mesh, case.conductivity, case.source, case.bcs)
    wall = time.perf_counter() - t0
    e = sol.errors(case.exact, case.grad)
    return {"part": part.name, "method": f"conforming P{order}", "level": size, "dofs": sol.info["dofs"],
            "elements": len(mesh.tets), "L2_rel": e["L2_rel"], "H1_rel": e["H1_rel"], "geometry_s": mesh.meshing_s,
            "assembly_s": sol.info["assembly_s"], "solve_s": sol.info["solve_seconds"], "wall_s": wall,
            "solver": sol.info["solve_solver"], "energy_imbalance": sol.balance["imbalance"]}


def run_calculix(part: Part, size: float, workdir: str | None = None) -> dict:
    """CalculiX (C3D10, steady heat with body flux and film) on the gmsh P2 mesh; the error of its nodal field is
    measured with the P2 interpolation on the same mesh. Only for parts whose data CalculiX takes as constants."""
    from pinneapple_simulation.external_solvers.calculix import study as S
    from pinneapple_simulation.numerical_solvers import tet_fem as T
    case = part.case
    hb = case.bcs[0]
    t0 = time.perf_counter()
    mesh = T.gmsh_mesh(part.occ, size, 2)
    m = mesh.to_calculix()
    m.material = S.Material("solid", conductivity=case.conductivity, density=None)
    faces = m.faces_where(part.mantle or (lambda C: np.ones(len(C), bool)))
    step = S.Heat(film=[(faces, (float(hb.T_inf), float(hb.h)))], body=float(case.source))
    work = workdir or tempfile.mkdtemp(prefix="ccx_heat_")
    t1 = time.perf_counter()
    res = S.solve(m, step, work, threads=1, keep=False)
    t_ccx = time.perf_counter() - t1
    wall = time.perf_counter() - t0
    e = T.field_errors(mesh, res.temperature, case.exact, case.grad)
    return {"part": part.name, "method": "CalculiX C3D10", "level": size, "dofs": mesh.n_nodes,
            "elements": len(mesh.tets), "L2_rel": e["L2_rel"], "H1_rel": e["H1_rel"], "geometry_s": mesh.meshing_s,
            "assembly_s": float("nan"), "solve_s": t_ccx, "wall_s": wall, "solver": res.info["solver"]}


PLAN = {"immersed Q1": [6, 12, 24, 48], "immersed Q2": [4, 8, 12, 16, 24],
        "conforming P1": [0.2, 0.1, 0.05, 0.03], "conforming P2": [0.3, 0.2, 0.1, 0.06, 0.04],
        "CalculiX C3D10": [0.3, 0.2, 0.1, 0.06, 0.04]}
QUICK = {"immersed Q1": [6, 12], "immersed Q2": [4, 8], "conforming P1": [0.2, 0.1], "conforming P2": [0.3, 0.2],
         "CalculiX C3D10": [0.3, 0.2]}


def sweep(part: Part, plan: dict | None = None, quick: bool = False, calculix: bool | None = None,
          log: Callable | None = None) -> list[dict]:
    plan = dict(plan or (QUICK if quick else PLAN))
    if calculix is None:
        import shutil
        calculix = shutil.which("ccx") is not None
    if not (calculix and part.calculix):
        plan.pop("CalculiX C3D10", None)
    rows = []
    for method, levels in plan.items():
        for lv in levels:
            if method.startswith("immersed"):
                r = run_immersed(part, int(method[-1]), int(lv))
            elif method.startswith("conforming"):
                r = run_conforming(part, int(method[-1]), float(lv))
            else:
                r = run_calculix(part, float(lv))
            rows.append(r)
            if log:
                log(r)
    return rows


def _interp(x: np.ndarray, y: np.ndarray, target: float) -> float:
    """log-log interpolation of y at x = target over measured points (x decreasing with refinement)."""
    o = np.argsort(x)
    lx, ly = np.log(x[o]), np.log(y[o])
    t = np.log(target)
    if t < lx[0] or t > lx[-1]:
        return float("nan")
    return float(np.exp(np.interp(t, lx, ly)))


def equal_accuracy(rows: list[dict], targets=(1e-3, 1e-4), key: str = "L2_rel") -> list[dict]:
    """Unknowns and wall times (total and geometry phase) each method needs to reach each target error."""
    out = []
    for method in dict.fromkeys(r["method"] for r in rows):
        R = [r for r in rows if r["method"] == method]
        err = np.array([r[key] for r in R])
        for tgt in targets:
            row = {"method": method, "target": tgt}
            for f in ("dofs", "wall_s", "geometry_s"):
                vals = np.array([r[f] for r in R], float)
                row[f] = _interp(err, np.maximum(vals, 1e-9), tgt) if np.all(np.isfinite(vals)) else float("nan")
            out.append(row)
    return out


def figure(rows: list[dict], path: str, key: str = "L2_rel", title: str = "") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for method in dict.fromkeys(r["method"] for r in rows):
        R = [r for r in rows if r["method"] == method]
        ls = "-" if method.startswith("immersed") else "--"
        ax[0].loglog([r["dofs"] for r in R], [r[key] for r in R], ls, marker="o", label=method)
        ax[1].loglog([r["wall_s"] for r in R], [r[key] for r in R], ls, marker="o", label=method)
    ax[0].set_xlabel("unknowns")
    ax[1].set_xlabel("wall time [s] (geometry + assembly + solve)")
    for a in ax:
        a.set_ylabel(f"relative {key.split('_')[0]} error")
        a.grid(True, which="both", alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    """``python -m pinneapple_simulation.immersed.compare --parts sphere_mms sphere --out DIR [--quick]``: the sweep,
    the equal-accuracy table and the figures, written to DIR (results.json, <part>.png)."""
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["sphere_mms", "sphere"])
    ap.add_argument("--out", default="immersed_compare")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--targets", nargs="+", type=float, default=[1e-3, 1e-4])
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    allp = parts()
    out = {}
    for name in a.parts:
        rows = sweep(allp[name], quick=a.quick, log=lambda r: print(json.dumps(r), flush=True))
        out[name] = {"rows": rows, "equal_accuracy": equal_accuracy(rows, a.targets)}
        figure(rows, os.path.join(a.out, f"{name}.png"), title=name)
        with open(os.path.join(a.out, "results.json"), "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

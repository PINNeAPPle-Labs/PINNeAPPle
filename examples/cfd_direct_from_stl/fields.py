"""Velocity and pressure fields of the three methods on C1 (Schafer-Turek 2D-1) and C3 (Armaly step, Re 100), at a
resolution given on the command line (not timed; the timed runs are in run.py). Writes figures/fields_C1.png and
figures/fields_C3.png.

    python examples/cfd_direct_from_stl/fields.py --c1 16 8 20 --c3 16 16 20      # n for reference, immersed, lbm
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def _reference(case: str, n: int, X: np.ndarray):
    from pinneapple_design.geometry.mesh.generate import generate_mesh
    from pinneapple_simulation.cfd_direct.cases import SCHAFER_TUREK as c
    from pinneapple_simulation.cfd_direct.cases import BackwardStep, cylinder_polygon
    from pinneapple_simulation.numerical_solvers.taylor_hood import TaylorHood, read_gmsh
    work = tempfile.mkdtemp()
    if case == "C1":
        m = generate_mesh("polygon", vertices=cylinder_polygon(512).tolist(), enclosure=[[0, 0], [c.L, c.H]],
                          h=4 * c.D / n, refine=[{"center": list(c.c), "radius": c.D, "h": c.D / n}], out_dir=work)
        th = TaylorHood(read_gmsh(m["files"]["msh"]), c.nu)
        sol = th.solve([(lambda x: x[0] < 1e-9, lambda P: np.stack([c.inflow(P[1]), 0 * P[1]])),
                        (lambda x: (x[1] < 1e-9) | (x[1] > c.H - 1e-9), 0.0),
                        (lambda x: np.hypot(x[0] - 0.2, x[1] - 0.2) < 0.051, 0.0)])
    else:
        s = BackwardStep(100)
        m = generate_mesh("polygon", vertices=s.polygon().tolist(), h=s.S / n, out_dir=work)
        th = TaylorHood(read_gmsh(m["files"]["msh"]), s.nu)
        a, b = -s.upstream, s.downstream
        sol = th.solve([(lambda x: x[0] < a + 1e-9, lambda P: np.stack([s.inflow(P[1]), 0 * P[1]])),
                        (lambda x: (x[0] > a + 1e-9) & (x[0] < b - 1e-9), 0.0)])
    return sol.velocity(X), sol.pressure(X)


def _immersed(case: str, n: int, X: np.ndarray):
    from pinneapple_simulation.cfd_direct.cases import SCHAFER_TUREK as c
    from pinneapple_simulation.cfd_direct.cases import BackwardStep, cylinder_polygon
    from pinneapple_simulation.immersed import flow
    if case == "C1":
        fd = flow.FlowDiscretization(flow.PolygonGeometry(cylinder_polygon(512)), c.box, h=c.D / n)
        bcs = {"x-": flow.Velocity(lambda P: np.column_stack([c.inflow(P[:, 1]), 0 * P[:, 1]])), "y-": "wall",
               "y+": "wall", "x+": "outflow"}
        sol = flow.solve_navier_stokes(fd, c.nu, bcs, u_ref=c.U_mean)
    else:
        s = BackwardStep(100)
        fd = flow.FlowDiscretization(None, s.box, h=s.S / n, fluid=flow.PolygonGeometry(s.polygon()))
        bcs = {"x-": flow.Velocity(lambda P: np.column_stack([s.inflow(P[:, 1]), 0 * P[:, 1]])), "y-": "wall",
               "y+": "wall", "x+": "outflow"}
        sol = flow.solve_navier_stokes(fd, s.nu, bcs, u_ref=s.U)
    inside = fd.fluid.phi(X) < 0
    u, p = sol.velocity(X), sol.pressure(X)
    u[~inside], p[~inside] = np.nan, np.nan
    return u, p


def _lbm(case: str, n: int, X: np.ndarray):
    import torch
    from scipy.interpolate import RegularGridInterpolator

    from pinneapple_simulation.cfd_direct.cases import SCHAFER_TUREK as c
    from pinneapple_simulation.cfd_direct.cases import BackwardStep, cylinder_polygon
    from pinneapple_simulation.immersed.flow import PolygonGeometry
    from pinneapple_simulation.numerical_solvers.lbm_tunnel import WindTunnelLBM
    torch.set_num_threads(4)
    if case == "C1":
        dx, L, H, a = c.D / n, c.L, c.H, 0.0
        geom, peak, nu = PolygonGeometry(cylinder_polygon(512)), c.U_max, c.nu
        prof = c.inflow
        ft = c.L / c.U_mean
    else:
        s = BackwardStep(100)
        dx, L, H, a = s.S / n, s.upstream + s.downstream, s.H, -s.upstream
        geom, peak, nu, prof = PolygonGeometry(s.polygon()), 1.5 * s.U, s.nu, s.inflow
        ft = L / s.U
    nx, ny = int(round(L / dx)) + 1, int(round(H / dx)) + 2
    x, y = a + np.arange(nx) * dx, (np.arange(ny) - 0.5) * dx
    xc = np.clip(x, a + 0.5 * dx, a + L - 0.5 * dx)
    P = np.stack(np.meshgrid(xc, y, indexing="ij"), -1).reshape(-1, 2)
    inside = (geom.phi(P) < 0).reshape(nx, ny)
    solid = inside if case == "C1" else ~inside
    solid[:, [0, -1]] = False
    scale = peak / 0.05
    lbm = WindTunnelLBM((nx, ny, 1), nu=nu / (scale * dx), collision="trt",
                        faces={"x-": "inlet", "x+": "outlet", "y-": "wall", "y+": "wall", "z-": "periodic",
                               "z+": "periodic"})
    lbm.add_solid(torch.as_tensor(solid[:, :, None]))
    u0 = np.zeros((3, nx, ny, 1), np.float32)
    u0[0] = (prof(np.clip(y, 0, H)) / scale)[None, :, None] if case == "C1" else 0.0
    u0[0, 0] = (prof(y) / scale)[:, None]
    lbm.initialize(u0)
    lbm.step(int(3 * ft / (dx / scale)))
    rho, u = lbm.macroscopic()
    ux, uy = u[0, :, :, 0].numpy() * scale, u[1, :, :, 0].numpy() * scale
    p = (rho[:, :, 0].numpy() - 1) / 3 * scale ** 2
    out_u = np.column_stack([RegularGridInterpolator((x, y), f, bounds_error=False)(X) for f in (ux, uy)])
    out_p = RegularGridInterpolator((x, y), p, bounds_error=False)(X)
    m = geom.phi(X) < 0 if case == "C1" else geom.phi(X) > 0
    out_u[m], out_p[m] = np.nan, np.nan
    return out_u, out_p


def plot(case: str, ns: tuple[int, int, int], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pinneapple_simulation.cfd_direct.cases import SCHAFER_TUREK as c
    from pinneapple_simulation.cfd_direct.cases import BackwardStep
    if case == "C1":
        xs, ys = np.linspace(0, c.L, 441), np.linspace(0, c.H, 83)
        title = "C1 Schafer-Turek 2D-1 (Re 20)"
    else:
        s = BackwardStep(100)
        xs, ys = np.linspace(-2, 12, 561), np.linspace(0, s.H, 83)
        title = "C3 Armaly step, Re 100"
    G = np.stack(np.meshgrid(xs, ys, indexing="xy"), -1).reshape(-1, 2)
    res = {"R: body-fitted + Taylor-Hood": _reference(case, ns[0], G),
           "B: immersed cut cell": _immersed(case, ns[1], G),
           "A: LBM voxelised": _lbm(case, ns[2], G)}
    fig, axs = plt.subplots(3, 2, figsize=(13, 5.6 if case == "C1" else 4.6), squeeze=False)
    umax = np.nanmax(np.linalg.norm(res["R: body-fitted + Taylor-Hood"][0], axis=1))
    pr = res["R: body-fitted + Taylor-Hood"][1]
    plo, phi_ = np.nanpercentile(pr, 1), np.nanpercentile(pr, 99)
    for i, ((name, (u, p)), n) in enumerate(zip(res.items(), ns, strict=True)):
        sp = np.linalg.norm(u, axis=1).reshape(len(ys), len(xs))
        im0 = axs[i][0].imshow(sp, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], cmap="viridis",
                               vmin=0, vmax=umax, aspect="equal")
        im1 = axs[i][1].imshow((p - np.nanmean(p) + np.nanmean(pr)).reshape(len(ys), len(xs)), origin="lower",
                               extent=[xs[0], xs[-1], ys[0], ys[-1]], cmap="RdBu_r", vmin=plo, vmax=phi_,
                               aspect="equal")
        axs[i][0].set_title(f"{name} (n = {n}): |u|", fontsize=8)
        axs[i][1].set_title(f"{name} (n = {n}): p (mean shifted to R's)", fontsize=8)
        for ax in axs[i]:
            ax.tick_params(labelsize=6)
    fig.colorbar(im0, ax=[a[0] for a in axs], shrink=0.6)
    fig.colorbar(im1, ax=[a[1] for a in axs], shrink=0.6)
    fig.suptitle(title, fontsize=10)
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1", type=int, nargs=3, default=[16, 16, 20])
    ap.add_argument("--c3", type=int, nargs=3, default=[16, 16, 20])
    a = ap.parse_args()
    (HERE / "figures").mkdir(exist_ok=True)
    plot("C1", tuple(a.c1), HERE / "figures" / "fields_C1.png")
    plot("C3", tuple(a.c3), HERE / "figures" / "fields_C3.png")

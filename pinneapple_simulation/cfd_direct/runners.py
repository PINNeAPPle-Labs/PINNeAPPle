"""One run of one method on one case at one rung of its resolution ladder, with the stage times, the peak memory and
the metrics of the pre-registration (docs/research/cfd_direct_from_stl.md).

Methods:
- ``reference``: body-fitted mesh by :func:`generate_mesh` + Taylor-Hood P2/P1 (:mod:`..numerical_solvers.taylor_hood`);
- ``lbm``: :class:`..numerical_solvers.lbm_tunnel.WindTunnelLBM` on the voxelised polygon / STL;
- ``immersed``: :mod:`..immersed.flow` (Q2/Q1 cut cells, Nitsche, ghost penalty) on the polygon / STL.

Every method reads the same geometry: the 512-gon of the cylinder, the icosphere STL of the sphere.

    python -m pinneapple_simulation.cfd_direct.runners immersed C1 8 --out run.json
"""
from __future__ import annotations

import json
import os
import resource
import sys
import tempfile
import time
from typing import Any

import numpy as np

from .cases import SCHAFER_TUREK, BackwardStep, Sphere, cylinder_polygon, reattachment, sphere_stl

__all__ = ["run", "LADDERS"]

# the ladders of the pre-registration: cells per D (or per S), per method and case
LADDERS = {
    "reference": {"C1": [8, 16, 32, 64], "C2": [8, 16, 24], "C3": [8, 16, 32]},
    "lbm": {"C1": [10, 20, 40, 80], "C2": [8, 12, 16], "C3": [10, 20, 40]},
    "immersed": {"C1": [4, 8, 16, 32], "C2": [4, 6, 8], "C3": [8, 16, 32]},
}


def _rel(a: float, b: float) -> float:
    return float(abs(a - b) / abs(b))


def _load() -> list[float]:
    return [float(x) for x in os.getloadavg()]


def _peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ------------------------------------------------------------------------------------------------ C1
def _c1_metrics(Cd: float, Cl: float, dp: float) -> dict:
    c = SCHAFER_TUREK
    return {"Cd": Cd, "Cl": Cl, "dp": dp, "err_Cd": _rel(Cd, c.Cd), "err_Cl": _rel(Cl, c.Cl), "err_dp": _rel(dp, c.dp)}


def _c1_profiles(fn) -> dict:
    return {k: fn(P).tolist() for k, P in SCHAFER_TUREK.profile_points().items()}


def reference_c1(n: int, work: str, n_poly: int = 512) -> dict:
    from pinneapple_design.geometry.mesh.generate import generate_mesh

    from ..numerical_solvers.taylor_hood import TaylorHood, read_gmsh
    c = SCHAFER_TUREK
    t = {}
    t0 = time.perf_counter()
    V = cylinder_polygon(n_poly)
    hb = c.D / n
    t["geometry_s"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    m = generate_mesh("polygon", vertices=V.tolist(), enclosure=[[0.0, 0.0], [c.L, c.H]], h=4 * hb,
                      refine=[{"center": list(c.c), "radius": c.D, "h": hb}], out_dir=work, max_elements=5_000_000)
    mesh = read_gmsh(m["files"]["msh"])
    t["mesh_s"] = time.perf_counter() - t1
    r = 0.5 * c.D

    def body(x):
        return np.hypot(x[0] - c.c[0], x[1] - c.c[1]) < 1.02 * r

    th = TaylorHood(mesh, c.nu)
    sol = th.solve([(lambda x: x[0] < 1e-9, lambda X: np.stack([c.inflow(X[1]), 0 * X[1]])),
                    (lambda x: (x[1] < 1e-9) | (x[1] > c.H - 1e-9), 0.0), (body, 0.0)])
    t["assembly_s"] = th.timings["assembly_s"]
    t["solve_s"] = sol.info["solve_s"]
    t2 = time.perf_counter()
    Cd, Cl = c.coefficients(sol.force(body))
    dp = float(sol.pressure(np.array([c.p_front]))[0] - sol.pressure(np.array([c.p_back]))[0])
    prof = _c1_profiles(lambda P: sol.velocity(P)[:, 0])
    t["post_s"] = time.perf_counter() - t2
    t["total_s"] = time.perf_counter() - t0
    return {**_c1_metrics(Cd, Cl, dp), "profiles": prof, "times": t,
            "size": {"elements": int(mesh.nelements), "unknowns": th.n_unknowns, "h_body": hb},
            "newton_iterations": sol.info["newton_iterations"], "mesh": {k: m[k] for k in ("sicn_min", "n_elements")}}


def immersed_c1(n: int, work: str, n_poly: int = 512) -> dict:
    from ..immersed import flow
    c = SCHAFER_TUREK
    t = {}
    t0 = time.perf_counter()
    body = flow.PolygonGeometry(cylinder_polygon(n_poly))
    t["geometry_s"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    fd = flow.FlowDiscretization(body, c.box, h=c.D / n)
    t["mesh_s"] = time.perf_counter() - t1                      # background grid + cut-cell quadrature
    bcs = {"x-": flow.Velocity(lambda X: np.column_stack([c.inflow(X[:, 1]), 0 * X[:, 1]])), "y-": "wall",
           "y+": "wall", "x+": "outflow"}
    sol = flow.solve_navier_stokes(fd, c.nu, bcs, u_ref=c.U_mean)
    t["assembly_s"] = sol.info["assembly_s"]
    t["solve_s"] = sol.info["solve_s"]
    t2 = time.perf_counter()
    F = sol.force()
    Cd, Cl = c.coefficients(F["variational"])
    Cd_s, Cl_s = c.coefficients(F["surface"])
    dp = float(sol.pressure(np.array([c.p_front]))[0] - sol.pressure(np.array([c.p_back]))[0])
    prof = _c1_profiles(lambda P: sol.velocity(P)[:, 0])
    t["post_s"] = time.perf_counter() - t2
    t["total_s"] = time.perf_counter() - t0
    return {**_c1_metrics(Cd, Cl, dp), "Cd_surface": Cd_s, "Cl_surface": Cl_s, "profiles": prof, "times": t,
            "size": {"cells": list(fd.grid.n), "unknowns": fd.n_unknowns, "h": fd.h,
                     "cut_cells": int(len(fd.disc.quad.cut_cells))},
            "newton_iterations": sol.info["newton_iterations"], "divergence_l2": sol.divergence_l2()}


def lbm_c1(n: int, work: str, n_poly: int = 512, u_lat: float = 0.05, check_every: int = 1000,
           rtol: float = 1e-4, max_flow_through: float = 10.0, collision: str = "trt") -> dict:
    """Quasi-2-D LBM (one cell in z, periodic): ``n`` cells per D, inflow peak ``u_lat`` (lattice units)."""
    import torch

    from ..immersed.flow import PolygonGeometry
    from ..numerical_solvers.lbm_tunnel import WindTunnelLBM
    c = SCHAFER_TUREK
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    t = {}
    t0 = time.perf_counter()
    poly = PolygonGeometry(cylinder_polygon(n_poly))
    t["geometry_s"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    dx = c.D / n
    nyf = int(round(c.H / dx))
    nx, ny = int(round(c.L / dx)) + 1, nyf + 2
    x = np.arange(nx) * dx                                        # inlet layer at x = 0, outlet at x = L
    y = (np.arange(ny) - 0.5) * dx                                # walls halfway: y = 0 and y = nyf dx = H
    Xc = np.stack(np.meshgrid(x, y, indexing="ij"), -1).reshape(-1, 2)
    solid = torch.as_tensor((poly.phi(Xc) < 0).reshape(nx, ny, 1))
    t["mesh_s"] = time.perf_counter() - t1                        # voxelisation
    scale = c.U_max / u_lat                                       # physical / lattice velocity
    nu_lat = c.nu / (scale * dx)
    lbm = WindTunnelLBM((nx, ny, 1), nu=nu_lat, collision=collision,
                        faces={"x-": "inlet", "x+": "outlet", "y-": "wall", "y+": "wall", "z-": "periodic",
                               "z+": "periodic"})
    lbm.add_solid(solid)
    u0 = np.zeros((3, nx, ny, 1), np.float32)
    yy = np.clip(y, 0.0, c.H)
    u0[0] = (c.inflow(yy) / scale)[None, :, None]
    u0[0][:, [0, -1]] = 0.0
    lbm.initialize(u0)
    q = 0.5 * (c.U_mean / scale) ** 2 * n                         # dynamic pressure x D (lattice)
    steps_ft = c.L / dx / (c.U_mean / scale)
    t2 = time.perf_counter()
    hist = []
    converged = False
    while lbm.t < max_flow_through * steps_ft:
        lbm.step(check_every)
        F = lbm.force_on_solid()
        hist.append(float(F[0] / q))
        if len(hist) > 1 and abs(hist[-1] - hist[-2]) <= rtol * abs(hist[-1]):
            converged = True
            break
    t["solve_s"] = time.perf_counter() - t2
    t3 = time.perf_counter()
    F = lbm.force_on_solid()
    Cd, Cl = float(F[0] / q), float(F[1] / q)
    rho, u = lbm.macroscopic()
    rho = rho[:, :, 0].numpy()
    ux = u[0, :, :, 0].numpy()

    def interp(field, P):
        from scipy.interpolate import RegularGridInterpolator
        return RegularGridInterpolator((x, y), field)(P)

    p_lat = (interp(rho, np.array([c.p_front, c.p_back])) - 1.0) / 3.0
    dp = float((p_lat[0] - p_lat[1]) * scale ** 2)
    prof = _c1_profiles(lambda P: interp(ux, P) * scale)
    t["post_s"] = time.perf_counter() - t3
    t["total_s"] = time.perf_counter() - t0
    return {**_c1_metrics(Cd, Cl, dp), "profiles": prof, "times": t,
            "size": {"cells": [nx, ny, 1], "fluid_cells": int((~lbm._all_solid).sum()), "dx": dx, "tau": lbm.tau},
            "steps": lbm.t, "flow_through": lbm.t / steps_ft, "converged": converged, "Cd_history": hist[-20:]}



# ------------------------------------------------------------------------------------------------ C2 sphere
def _c2_metrics(case: Sphere, Fx: float) -> dict:
    Cd = case.drag_coefficient(Fx)
    return {"Re": case.Re, "Cd": Cd, "Cd_ref": case.Cd_ref, "err_Cd": _rel(Cd, case.Cd_ref),
            "Cd_schiller_naumann": case.Cd_schiller_naumann, "err_Cd_sn": _rel(Cd, case.Cd_schiller_naumann)}


def reference_c2(n: int, work: str, Re: float = 20.0, subdivisions: int = 5) -> dict:
    from pinneapple_design.geometry.mesh.generate import generate_mesh

    from ..numerical_solvers.taylor_hood import TaylorHood, read_gmsh
    case = Sphere(Re)
    t = {}
    t0 = time.perf_counter()
    stl = sphere_stl(os.path.join(work, "sphere.stl"), case.D, subdivisions)
    t["geometry_s"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    hb = case.D / n
    lo, hi = case.box
    m = generate_mesh(stl, h=case.D, enclosure=[list(lo), list(hi)], out_dir=work, max_elements=20_000_000,
                      refine=[{"center": [0, 0, 0], "radius": case.D, "h": hb},
                              {"center": [2.5 * case.D, 0, 0], "radius": 3 * case.D, "h": min(4 * hb, case.D / 2)}])
    mesh = read_gmsh(m["files"]["msh"])
    t["mesh_s"] = time.perf_counter() - t1
    r = 0.5 * case.D

    def body(x):
        return np.linalg.norm(x, axis=0) < 1.05 * r

    def far(x):
        return (x[0] < lo[0] + 1e-9) | (np.abs(x[1]) > hi[1] - 1e-9) | (np.abs(x[2]) > hi[2] - 1e-9)

    th = TaylorHood(mesh, case.nu)
    sol = th.solve([(far, (case.U, 0.0, 0.0)), (body, 0.0)])
    t["assembly_s"] = th.timings["assembly_s"]
    t["solve_s"] = sol.info["solve_s"]
    t2 = time.perf_counter()
    F = sol.force(body)
    prof = {"axis": sol.velocity(case.axis_points())[:, 0].tolist()}
    t["post_s"] = time.perf_counter() - t2
    t["total_s"] = time.perf_counter() - t0
    return {**_c2_metrics(case, F[0]), "F": F.tolist(), "profiles": prof, "times": t,
            "size": {"elements": int(mesh.nelements), "unknowns": th.n_unknowns, "h_body": hb},
            "newton_iterations": sol.info["newton_iterations"], "mesh": {k: m[k] for k in ("sicn_min", "n_elements")}}


def immersed_c2(n: int, work: str, Re: float = 20.0, subdivisions: int = 5) -> dict:
    from ..immersed import flow
    from ..immersed.geometry import STLGeometry
    case = Sphere(Re)
    t = {}
    t0 = time.perf_counter()
    stl = sphere_stl(os.path.join(work, "sphere.stl"), case.D, subdivisions)
    body = STLGeometry.from_file(stl)
    t["geometry_s"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    fd = flow.FlowDiscretization(body, case.box, h=case.D / n)
    t["mesh_s"] = time.perf_counter() - t1
    far = flow.Velocity((case.U, 0.0, 0.0))
    bcs = {"x-": far, "y-": far, "y+": far, "z-": far, "z+": far, "x+": "outflow"}
    sol = flow.solve_navier_stokes(fd, case.nu, bcs, u_ref=case.U)
    t["assembly_s"] = sol.info["assembly_s"]
    t["solve_s"] = sol.info["solve_s"]
    t2 = time.perf_counter()
    F = sol.force()
    prof = {"axis": sol.velocity(case.axis_points())[:, 0].tolist()}
    t["post_s"] = time.perf_counter() - t2
    t["total_s"] = time.perf_counter() - t0
    out = _c2_metrics(case, F["variational"][0])
    return {**out, "Cd_surface": case.drag_coefficient(F["surface"][0]), "profiles": prof, "times": t,
            "size": {"cells": list(fd.grid.n), "unknowns": fd.n_unknowns, "h": fd.h},
            "newton_iterations": sol.info["newton_iterations"]}


def lbm_c2(n: int, work: str, Re: float = 20.0, subdivisions: int = 5, u_lat: float = 0.05, check_every: int = 500,
           rtol: float = 1e-4, max_flow_through: float = 10.0, collision: str = "trt") -> dict:
    import torch

    from ..numerical_solvers.lbm_tunnel import WindTunnelLBM, voxelize_stl
    case = Sphere(Re)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    t = {}
    t0 = time.perf_counter()
    stl = sphere_stl(os.path.join(work, "sphere.stl"), case.D, subdivisions)
    t["geometry_s"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    lo, hi = case.box
    dx = case.D / n
    shape = tuple(int(round((b - a) / dx)) + 1 for a, b in zip(lo, hi, strict=True))
    center = [(0.0 - a) / dx for a in lo]                    # cell coordinates of the sphere centre
    solid = voxelize_stl(stl, shape, center, size=n)
    t["mesh_s"] = time.perf_counter() - t1
    nu_lat = u_lat * n / Re
    lbm = WindTunnelLBM(shape, nu=nu_lat, u_in=(u_lat, 0, 0), collision=collision)
    lbm.add_solid(solid)
    q = 0.5 * u_lat ** 2 * np.pi * n ** 2 / 4
    steps_ft = shape[0] / u_lat
    t2 = time.perf_counter()
    hist, converged = [], False
    while lbm.t < max_flow_through * steps_ft:
        lbm.step(check_every)
        hist.append(float(lbm.force_on_solid()[0] / q))
        if len(hist) > 1 and abs(hist[-1] - hist[-2]) <= rtol * abs(hist[-1]):
            converged = True
            break
    t["solve_s"] = time.perf_counter() - t2
    t3 = time.perf_counter()
    F = lbm.force_on_solid()
    _, u = lbm.macroscopic()
    ux = u[0].numpy()
    from scipy.interpolate import RegularGridInterpolator
    axes = [a + dx * np.arange(k) for a, k in zip(lo, shape, strict=True)]
    prof = {"axis": (RegularGridInterpolator(axes, ux)(case.axis_points()) / u_lat * case.U).tolist()}
    t["post_s"] = time.perf_counter() - t3
    t["total_s"] = time.perf_counter() - t0
    out = _c2_metrics(case, 0.0)
    Cd = float(F[0] / q)
    out.update(Cd=Cd, err_Cd=_rel(Cd, case.Cd_ref), err_Cd_sn=_rel(Cd, case.Cd_schiller_naumann))
    return {**out, "profiles": prof, "times": t, "steps": lbm.t, "flow_through": lbm.t / steps_ft,
            "converged": converged, "Cd_history": hist[-20:],
            "size": {"cells": list(shape), "solid_cells": int(solid.sum()), "dx": dx, "tau": lbm.tau,
                     "tensor_mb": lbm.bytes_per_cell() * int(np.prod(shape)) / 1e6}}


# ------------------------------------------------------------------------------------------------ C3 step
def _c3_metrics(case: BackwardStep, x1: float) -> dict:
    return {"Re": case.Re, "x1": x1, "x1_ref": case.x1_ref, "err_x1": _rel(x1, case.x1_ref)}


def _x_line(case: BackwardStep, k: int = 6001) -> np.ndarray:
    return np.linspace(0.005 * case.S, case.downstream - 1.0, k)


def reference_c3(n: int, work: str, Re: float = 100.0) -> dict:
    from pinneapple_design.geometry.mesh.generate import generate_mesh

    from ..numerical_solvers.taylor_hood import TaylorHood, read_gmsh
    case = BackwardStep(Re)
    t = {}
    t0 = time.perf_counter()
    V = case.polygon()
    t["geometry_s"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    m = generate_mesh("polygon", vertices=V.tolist(), h=case.S / n, out_dir=work, max_elements=5_000_000)
    mesh = read_gmsh(m["files"]["msh"])
    t["mesh_s"] = time.perf_counter() - t1
    a, b = -case.upstream, case.downstream
    th = TaylorHood(mesh, case.nu)
    sol = th.solve([(lambda x: x[0] < a + 1e-9, lambda X: np.stack([case.inflow(X[1]), 0 * X[1]])),
                    (lambda x: (x[0] > a + 1e-9) & (x[0] < b - 1e-9), 0.0)])   # the other boundaries: walls
    t["assembly_s"] = th.timings["assembly_s"]
    t["solve_s"] = sol.info["solve_s"]
    t2 = time.perf_counter()
    x = _x_line(case)
    d = 0.01 * case.S
    x1 = reattachment(x, sol.velocity(np.column_stack([x, np.full_like(x, d)]))[:, 0])
    prof = {k: sol.velocity(P)[:, 0].tolist() for k, P in case.profile_points().items()}
    t["post_s"] = time.perf_counter() - t2
    t["total_s"] = time.perf_counter() - t0
    return {**_c3_metrics(case, x1), "profiles": prof, "times": t,
            "size": {"elements": int(mesh.nelements), "unknowns": th.n_unknowns, "h": case.S / n},
            "newton_iterations": sol.info["newton_iterations"], "mesh": {k: m[k] for k in ("sicn_min", "n_elements")}}


def immersed_c3(n: int, work: str, Re: float = 100.0) -> dict:
    from ..immersed import flow
    case = BackwardStep(Re)
    t = {}
    t0 = time.perf_counter()
    fluid = flow.PolygonGeometry(case.polygon())
    t["geometry_s"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    fd = flow.FlowDiscretization(None, case.box, h=case.S / n, fluid=fluid)
    t["mesh_s"] = time.perf_counter() - t1
    bcs = {"x-": flow.Velocity(lambda X: np.column_stack([case.inflow(X[:, 1]), 0 * X[:, 1]])), "y-": "wall",
           "y+": "wall", "x+": "outflow"}
    sol = flow.solve_navier_stokes(fd, case.nu, bcs, u_ref=case.U)
    t["assembly_s"] = sol.info["assembly_s"]
    t["solve_s"] = sol.info["solve_s"]
    t2 = time.perf_counter()
    x = _x_line(case)
    x1 = reattachment(x, sol.velocity(np.column_stack([x, np.full_like(x, 0.01 * case.S)]))[:, 0])
    prof = {k: sol.velocity(P)[:, 0].tolist() for k, P in case.profile_points().items()}
    t["post_s"] = time.perf_counter() - t2
    t["total_s"] = time.perf_counter() - t0
    return {**_c3_metrics(case, x1), "profiles": prof, "times": t,
            "size": {"cells": list(fd.grid.n), "unknowns": fd.n_unknowns, "h": fd.h},
            "newton_iterations": sol.info["newton_iterations"]}


def lbm_c3(n: int, work: str, Re: float = 100.0, u_lat: float = 0.05, check_every: int = 1000, rtol: float = 1e-4,
           max_flow_through: float = 10.0, collision: str = "trt") -> dict:
    """Quasi-2-D LBM, ``n`` cells per step height; the inlet channel and the total height are rounded to whole cells
    (the expansion ratio of the lattice is reported)."""
    import torch

    from ..immersed.flow import PolygonGeometry
    from ..numerical_solvers.lbm_tunnel import WindTunnelLBM
    case = BackwardStep(Re)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    t = {}
    t0 = time.perf_counter()
    poly = PolygonGeometry(case.polygon())
    t["geometry_s"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    dx = case.S / n
    nyf = int(round(case.H / dx))
    a, b = -case.upstream, case.downstream
    nx, ny = int(round((b - a) / dx)) + 1, nyf + 2
    x = a + np.arange(nx) * dx
    y = (np.arange(ny) - 0.5) * dx
    xc = np.clip(x, a + 0.5 * dx, b - 0.5 * dx)                  # the inlet / outlet layers take their neighbours'
    Xc = np.stack(np.meshgrid(xc, y, indexing="ij"), -1).reshape(-1, 2)
    inside = (poly.phi(Xc) < 0).reshape(nx, ny)
    solid = ~inside
    solid[:, [0, -1]] = False                                     # the tunnel walls carry y = 0 and y = H
    t["mesh_s"] = time.perf_counter() - t1
    scale = 1.5 * case.U / u_lat                                  # inflow peak 1.5 U -> u_lat
    nu_lat = case.nu / (scale * dx)
    lbm = WindTunnelLBM((nx, ny, 1), nu=nu_lat, collision=collision,
                        faces={"x-": "inlet", "x+": "outlet", "y-": "wall", "y+": "wall", "z-": "periodic",
                               "z+": "periodic"})
    lbm.add_solid(torch.as_tensor(solid[:, :, None]))
    u0 = np.zeros((3, nx, ny, 1), np.float32)
    u0[0, 0] = (case.inflow(y) / scale)[:, None]
    lbm.initialize(u0)
    steps_ft = (b - a) / dx / (case.U / scale)
    xs = x[x > 0]
    t2 = time.perf_counter()
    hist, converged = [], False
    while lbm.t < max_flow_through * steps_ft:
        lbm.step(check_every)
        _, u = lbm.macroscopic()
        hist.append(reattachment(xs, u[0, x > 0, 1, 0].numpy()))
        if len(hist) > 1 and abs(hist[-1] - hist[-2]) <= rtol * abs(hist[-1]):
            converged = True
            break
    t["solve_s"] = time.perf_counter() - t2
    t3 = time.perf_counter()
    _, u = lbm.macroscopic()
    ux = u[0, :, :, 0].numpy() * scale
    x1 = reattachment(xs, ux[x > 0, 1])
    from scipy.interpolate import RegularGridInterpolator
    prof = {k: RegularGridInterpolator((x, y), ux)(P).tolist() for k, P in case.profile_points().items()}
    t["post_s"] = time.perf_counter() - t3
    t["total_s"] = time.perf_counter() - t0
    h_lat = int((~solid[0, 1:-1]).sum())
    return {**_c3_metrics(case, x1), "profiles": prof, "times": t, "steps": lbm.t, "flow_through": lbm.t / steps_ft,
            "converged": converged, "x1_history": hist[-20:],
            "size": {"cells": [nx, ny, 1], "dx": dx, "tau": lbm.tau, "expansion_ratio": nyf / max(h_lat, 1)}}


# ------------------------------------------------------------------------------------------------ dispatch
RUNNERS = {("reference", "C1"): reference_c1, ("immersed", "C1"): immersed_c1, ("lbm", "C1"): lbm_c1,
           ("reference", "C2"): reference_c2, ("immersed", "C2"): immersed_c2, ("lbm", "C2"): lbm_c2,
           ("reference", "C3"): reference_c3, ("immersed", "C3"): immersed_c3, ("lbm", "C3"): lbm_c3}


def run(method: str, case: str, n: int, work: str | None = None, **kw: Any) -> dict:
    """One run in this process, with the load and the peak memory recorded."""
    fn = RUNNERS[(method, case)]
    work = work or tempfile.mkdtemp(prefix=f"cfd_{method}_{case}_")
    before = _load()
    out = fn(n, work, **kw)
    out.update(method=method, case=case, n=n, load_before=before, load_after=_load(), peak_rss_mb=_peak_mb(),
               threads=os.environ.get("OMP_NUM_THREADS"), cpu_count=os.cpu_count())
    return out


def _main(argv: list[str]) -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("method")
    ap.add_argument("case")
    ap.add_argument("n", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work")
    ap.add_argument("--kw", default="{}")
    a = ap.parse_args(argv)
    res = run(a.method, a.case, a.n, a.work, **json.loads(a.kw))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    _main(sys.argv[1:])

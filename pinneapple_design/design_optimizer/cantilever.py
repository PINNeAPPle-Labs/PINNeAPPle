"""A tapered cantilever under a tip load: a gmsh-parametrized geometry solved by plane-stress finite elements, the
same quantities from beam theory, and the CAD (STEP) of a design. It is the reference problem of the design
optimisation workflow (:mod:`.workflow`, #1188): minimum mass under a stress and a deflection limit has a known
optimum through beam theory.

    from pinneapple_design.design_optimizer.cantilever import tapered_cantilever, tapered_cantilever_beam
    tapered_cantilever(h_root=0.12, h_tip=0.04, n_through=4)      # FEM on a gmsh mesh
    tapered_cantilever_beam(h_root=0.12, h_tip=0.04)              # Euler-Bernoulli + Timoshenko shear

Geometry: a beam of length ``length`` along x, clamped at x = 0, depth varying linearly from ``h_root`` to ``h_tip``
(symmetric about y = 0), out-of-plane width ``thickness``. Load: ``load`` [N] downwards, as a uniform shear traction
on the tip edge.

Mesh: gmsh transfinite (structured) triangles, ``n_through`` across the depth and ``n_along`` along the length. The
node counts do not depend on the design, so the mesh moves with the geometry and every output is a smooth function
of the dimensions (finite-difference gradients work); ``n_through`` is the fidelity (refine it to re-verify).

Outputs: ``mass_kg``; ``stress_max_Pa``, the largest bending (fibre) stress |sigma_xx| at the top and bottom surfaces
over the stations ``x_from <= x <= 0.9 length``; ``tip_deflection_m``, the mean downward displacement of the tip
edge. On the FEM, the fibre stress at a station comes from a least-squares fit ``sigma_xx = a + b x + c y + d x y``
over the elements whose centroid is within one element length of the station (bending stress is linear in y),
evaluated at the station's surfaces.

Assumptions (stated in the outputs): linear elasticity, plane stress, small displacements; the stations start at
``x_from`` (0.1 length by default) because the stress at a fully clamped edge is singular in plane elasticity
(Saint-Venant zone); the fibre stress is sigma_xx, not the stress along the inclined surface (they differ by a
factor 1 + tan^2(alpha), 0.6 % at a taper angle of 4.4 degrees). References: Timoshenko & Gere, *Mechanics of
Materials* (1997), ch. 5 and 9 (bending stress, deflection by integration); Cowper (1966), J. Appl. Mech. 33:335
(shear coefficient 10(1+nu)/(12+11nu) of a rectangle).
"""
from __future__ import annotations

import contextlib
import math
import os
from typing import Any

import numpy as np

__all__ = ["tapered_cantilever", "tapered_cantilever_beam", "export_step", "DEFAULTS", "X_OUTPUTS"]

DEFAULTS: dict[str, float] = {"length": 1.0, "thickness": 0.02, "load": 5000.0, "E": 210e9, "nu": 0.3,
                              "rho": 7850.0, "x_from": 0.1}
_STATIONS = 17
_H_REF = 0.05          # depth that sets the element length along the beam: n_along = n_through * length / _H_REF

X_OUTPUTS = {
    "tapered_cantilever": [
        {"name": "mass_kg", "type": "float", "unit": "kg", "description": "mass of the beam"},
        {"name": "stress_max_Pa", "type": "float", "unit": "Pa",
         "description": "largest fibre bending stress |sigma_xx| over the stations x_from..0.9 L (FEM)"},
        {"name": "tip_deflection_m", "type": "float", "unit": "m", "description": "mean downward tip displacement"},
        {"name": "n_elements", "type": "int", "unit": "-", "description": "triangles of the gmsh mesh"},
        {"name": "sicn_min", "type": "float", "unit": "-", "description": "worst gmsh signed inverse condition number"},
    ],
    "tapered_cantilever_beam": [
        {"name": "mass_kg", "type": "float", "unit": "kg", "description": "mass of the beam"},
        {"name": "stress_max_Pa", "type": "float", "unit": "Pa",
         "description": "largest bending stress 6 M / (b h^2) over the stations x_from..0.9 L"},
        {"name": "tip_deflection_m", "type": "float", "unit": "m",
         "description": "tip deflection: bending (Euler-Bernoulli) plus shear (Timoshenko, Cowper coefficient)"},
    ],
}


def _p(kw: dict[str, Any]) -> dict[str, float]:
    p = dict(DEFAULTS)
    p.update({k: float(v) for k, v in kw.items() if v is not None})
    return p


def _stations(p: dict[str, float]) -> np.ndarray:
    return np.linspace(p["x_from"] * p["length"], 0.9 * p["length"], _STATIONS)


def _depth(x, h_root: float, h_tip: float, L: float):
    return h_root + (h_tip - h_root) * np.asarray(x) / L


def _check(h_root: float, h_tip: float, p: dict[str, float]) -> None:
    for k, v in (("h_root", h_root), ("h_tip", h_tip), ("thickness", p["thickness"]), ("length", p["length"]),
                 ("E", p["E"]), ("rho", p["rho"])):
        if not (math.isfinite(v) and v > 0):
            raise ValueError(f"{k} must be a positive number, got {v}")
    if not 0 < p["x_from"] < 0.9:
        raise ValueError("x_from must be in (0, 0.9) (a fraction of the length)")


def tapered_cantilever_beam(h_root: float = 0.12, h_tip: float = 0.04, thickness: float | None = None, *,
                            length: float | None = None, load: float | None = None, E: float | None = None,
                            nu: float | None = None, rho: float | None = None,
                            x_from: float | None = None) -> dict[str, Any]:
    """Beam theory of the tapered cantilever (the closed-form reference of :func:`tapered_cantilever`).

    Returns
    -------
    mass_kg : float [kg]
    stress_max_Pa : float [Pa]
        Largest ``6 F (L - x) / (b h(x)^2)`` over the stations.
    tip_deflection_m : float [m]
        ``integral F (L-x)^2 / (E I(x)) dx + integral F / (k G A(x)) dx`` (Gauss-Legendre, 64 points)."""
    p = _p(dict(thickness=thickness, length=length, load=load, E=E, nu=nu, rho=rho, x_from=x_from))
    _check(h_root, h_tip, p)
    L, b, F, Em, nu_ = p["length"], p["thickness"], p["load"], p["E"], p["nu"]
    xs = _stations(p)
    sig = 6 * F * (L - xs) / (b * _depth(xs, h_root, h_tip, L) ** 2)
    g, w = np.polynomial.legendre.leggauss(64)
    x = 0.5 * L * (g + 1)
    h = _depth(x, h_root, h_tip, L)
    G = Em / (2 * (1 + nu_))
    kappa = 10 * (1 + nu_) / (12 + 11 * nu_)
    bend = 0.5 * L * np.sum(w * F * (L - x) ** 2 / (Em * b * h ** 3 / 12))
    shear = 0.5 * L * np.sum(w * F / (kappa * G * b * h))
    return {"mass_kg": p["rho"] * b * L * 0.5 * (h_root + h_tip), "stress_max_Pa": float(sig.max()),
            "tip_deflection_m": float(bend + shear), "bending_deflection_m": float(bend),
            "shear_deflection_m": float(shear), "stations_m": xs.tolist()}


def _mesh(h_root: float, h_tip: float, L: float, n_through: int, n_along: int):
    """Structured triangles of the trapezoid through gmsh; returns nodes, triangles and the worst SICN."""
    from ..geometry.mesh.generate import _gmsh_session as gmsh_session
    with gmsh_session("cantilever") as gmsh:
        occ = gmsh.model.geo
        pts = [occ.addPoint(0, -h_root / 2, 0), occ.addPoint(L, -h_tip / 2, 0), occ.addPoint(L, h_tip / 2, 0),
               occ.addPoint(0, h_root / 2, 0)]
        lines = [occ.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
        loop = occ.addCurveLoop(lines)
        s = occ.addPlaneSurface([loop])
        for ln, n in zip(lines, (n_along, n_through, n_along, n_through), strict=True):
            occ.mesh.setTransfiniteCurve(ln, n + 1)
        occ.mesh.setTransfiniteSurface(s, "Alternate", pts)
        occ.synchronize()
        gmsh.model.mesh.generate(2)
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        idx = {int(t): i for i, t in enumerate(tags)}
        V = np.asarray(xyz).reshape(-1, 3)[:, :2]
        et, enodes = gmsh.model.mesh.getElementsByType(2)
        F = np.array([idx[int(t)] for t in enodes]).reshape(-1, 3)
        q = gmsh.model.mesh.getElementQualities(et, "minSICN")
    return V, F, float(np.min(q))


def _fibre_stress(V, F, stress, xs, h_root, h_tip, L, dx) -> np.ndarray:
    c = V[F].mean(1)
    out = []
    for x in xs:
        m = np.abs(c[:, 0] - x) <= dx * 1.01
        cx, cy = c[m, 0] - x, c[m, 1]
        A = np.c_[np.ones(m.sum()), cx, cy, cx * cy]
        coef = np.linalg.lstsq(A, stress[m, 0], rcond=None)[0]
        h = float(_depth(x, h_root, h_tip, L))
        out.append(max(abs(coef[0] + coef[2] * h / 2), abs(coef[0] - coef[2] * h / 2)))
    return np.array(out)


def tapered_cantilever(h_root: float = 0.12, h_tip: float = 0.04, thickness: float | None = None, *,
                       n_through: int = 4, length: float | None = None, load: float | None = None,
                       E: float | None = None, nu: float | None = None, rho: float | None = None,
                       x_from: float | None = None) -> dict[str, Any]:
    """Plane-stress FEM of the tapered cantilever on a gmsh mesh (``n_through`` triangles pairs across the depth).

    Returns
    -------
    mass_kg : float [kg]
    stress_max_Pa : float [Pa]
        Largest fibre |sigma_xx| over the stations (see the module docstring).
    tip_deflection_m : float [m]
    n_elements : int [-]
    sicn_min : float [-]
        Worst element of the mesh (gmsh minSICN; 1 ideal, <= 0 inverted)."""
    from pinneapple_simulation.numerical_solvers.plane_stress_fem import solve_plane_stress
    p = _p(dict(thickness=thickness, length=length, load=load, E=E, nu=nu, rho=rho, x_from=x_from))
    _check(h_root, h_tip, p)
    n_through = int(n_through)
    if n_through < 2:
        raise ValueError("n_through must be at least 2")
    L, b = p["length"], p["thickness"]
    n_along = max(4, int(round(n_through * L / _H_REF)))
    V, F, sicn = _mesh(h_root, h_tip, L, n_through, n_along)
    tol = 1e-9 * L
    root = np.where(V[:, 0] < tol)[0]
    tip_nodes = np.where(V[:, 0] > L - tol)[0]
    tip_nodes = tip_nodes[np.argsort(V[tip_nodes, 1])]
    edges = np.c_[tip_nodes[:-1], tip_nodes[1:]]
    sol = solve_plane_stress(V, F, E=p["E"], nu=p["nu"], thickness=b, fixed={"x": root, "y": root},
                             traction=[(edges, (0.0, -p["load"] / (b * h_tip)))])
    sig = _fibre_stress(V, sol.F, sol.stress, _stations(p), h_root, h_tip, L, L / n_along)
    area = float(sol.area.sum())
    return {"mass_kg": p["rho"] * b * area, "stress_max_Pa": float(sig.max()),
            "tip_deflection_m": float(-sol.u[tip_nodes, 1].mean()), "n_elements": int(len(F)),
            "n_through": n_through, "n_along": n_along, "sicn_min": sicn}


@contextlib.contextmanager
def _quiet_stdout():
    import sys
    try:
        sys.stdout.flush()
        fd = sys.stdout.fileno()
    except (AttributeError, OSError, ValueError):     # no real stdout (a captured stream): nothing to silence
        yield
        return
    saved = os.dup(fd)
    try:
        with open(os.devnull, "w") as null:
            os.dup2(null.fileno(), fd)
            yield
    finally:
        os.dup2(saved, fd)
        os.close(saved)


def export_step(h_root: float, h_tip: float, out_dir: str, thickness: float | None = None, *,
                length: float | None = None, name: str = "optimum") -> dict[str, Any]:
    """Write the 3-D solid of a design (the trapezoid extruded by ``thickness``, metres) as ``<name>.step`` with
    gmsh's OpenCASCADE kernel, read it back, and return the path with the volume read from the file and the volume
    of the design (they must agree)."""
    from ..geometry.mesh.generate import _gmsh_session as gmsh_session
    p = _p(dict(thickness=thickness, length=length))
    _check(h_root, h_tip, p)
    L, b = p["length"], p["thickness"]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.step")
    with gmsh_session("cantilever_step") as gmsh:
        occ = gmsh.model.occ
        pts = [occ.addPoint(0, -h_root / 2, 0), occ.addPoint(L, -h_tip / 2, 0), occ.addPoint(L, h_tip / 2, 0),
               occ.addPoint(0, h_root / 2, 0)]
        lines = [occ.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
        s = occ.addPlaneSurface([occ.addCurveLoop(lines)])
        occ.extrude([(2, s)], 0, 0, b)
        occ.synchronize()
        with _quiet_stdout():                  # OpenCASCADE prints its STEP transfer statistics on stdout
            gmsh.write(path)
    with gmsh_session("cantilever_step_read") as gmsh, _quiet_stdout():
        gmsh.model.occ.importShapes(path)
        gmsh.model.occ.synchronize()
        vols = gmsh.model.getEntities(3)
        v_read = float(sum(gmsh.model.occ.getMass(3, t) for _, t in vols))
    return {"path": path, "volume_read_m3": v_read, "volume_design_m3": b * L * 0.5 * (h_root + h_tip),
            "n_solids": len(vols)}

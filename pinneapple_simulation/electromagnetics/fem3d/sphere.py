"""Permeable sphere in a uniform field: closed form, gmsh mesh, native edge-element and GetDP solutions.

A sphere of radius ``a`` and relative permeability ``mu_r`` in a uniform applied field ``B0`` (no currents) has

* inside, the uniform field ``B = 3 mu_r / (mu_r + 2) B0``;
* outside, the applied field plus a dipole of moment ``m = 4 pi a^3 (mu_r - 1) / (mu_r + 2) B0 / mu0``,
  with vector potential ``A = B0 x r / 2 + mu0 / (4 pi) m x r / |r|^3``

(Jackson, *Classical Electrodynamics*, 3rd ed., section 5.11; Griffiths, *Introduction to Electrodynamics*,
4th ed., example 6.1 for the analogous magnetized sphere). The computational domain is a ball of radius ``R``
with the exact exterior ``n x A`` on its surface, so the only error is discretisation (no truncation error).
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from .mesh import TetMesh, read_msh22
from .nedelec import MU0, NU0, EdgeSpace, solve_magnetostatic

__all__ = ["SphereCase", "mesh_sphere", "solve_native", "errors", "uniform_dirichlet"]


class SphereCase:
    def __init__(self, a: float = 1.0, R: float = 3.0, mu_r: float = 10.0, B0=(0.0, 0.0, 1.0)):
        self.a, self.R, self.mu_r = float(a), float(R), float(mu_r)
        self.B0 = np.asarray(B0, float)
        self.B_in = 3.0 * mu_r / (mu_r + 2.0) * self.B0
        self.m = 4.0 * math.pi * a ** 3 * (mu_r - 1.0) / (mu_r + 2.0) * self.B0 / MU0

    def A(self, x: np.ndarray) -> np.ndarray:
        """Exact vector potential outside the sphere (used on the outer boundary)."""
        x = np.atleast_2d(x)
        r3 = np.linalg.norm(x, axis=1)[:, None] ** 3
        return 0.5 * np.cross(self.B0, x) + MU0 / (4 * math.pi) * np.cross(self.m, x) / r3

    def B(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        r = np.linalg.norm(x, axis=1)[:, None]
        n = x / r
        out = self.B0 + MU0 / (4 * math.pi) * (3 * n * (n @ self.m)[:, None] - self.m) / r ** 3
        out[r[:, 0] < self.a] = self.B_in
        return out


def mesh_sphere(path, h: float, case: SphereCase, h_out: float | None = None) -> Path:
    """gmsh (OpenCASCADE) mesh of the sphere (physical volume 1, "sphere") inside the ball of radius ``R``
    (volume 2, "air"), outer surface physical 3, element size ``h`` in the sphere growing to ``h_out`` (default
    ``2 h``) at the outer surface; written as ASCII msh 2.2."""
    import gmsh

    path = Path(path)
    if not gmsh.isInitialized():
        gmsh.initialize(["-v", "0"], interruptible=False)   # no signal handler: runs off the main thread
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.clear()
    gmsh.model.add("sphere")
    occ = gmsh.model.occ
    s_in = occ.addSphere(0, 0, 0, case.a)
    s_out = occ.addSphere(0, 0, 0, case.R)
    frag, _ = occ.fragment([(3, s_out)], [(3, s_in)])
    occ.synchronize()
    vols = [t for d, t in gmsh.model.getEntities(3)]
    inner = [v for v in vols if abs(gmsh.model.occ.getMass(3, v) - 4 / 3 * math.pi * case.a ** 3) < 1e-6 * case.a ** 3]
    outer = [v for v in vols if v not in inner]
    gmsh.model.addPhysicalGroup(3, inner, 1, "sphere")
    gmsh.model.addPhysicalGroup(3, outer, 2, "air")
    surfs = gmsh.model.getBoundary([(3, v) for v in vols], combined=True, oriented=False)
    gmsh.model.addPhysicalGroup(2, [t for _, t in surfs], 3, "outer")
    h_out = 2 * h if h_out is None else h_out
    f = gmsh.model.mesh.field
    f.add("MathEval", 1)
    f.setString(1, "F", f"{h} + ({h_out} - {h}) * Max(0, Sqrt(x*x+y*y+z*z) - {case.a}) / {case.R - case.a}")
    f.setAsBackgroundMesh(1)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(str(path))
    gmsh.finalize()
    return path


def errors(mesh: TetMesh, B: np.ndarray, case: SphereCase) -> dict:
    """Relative errors of a piecewise-constant ``B`` (M, 3): mean field in the sphere, L2 in the sphere and L2
    over the whole domain (exact field sampled at a 4-point rule per tetrahedron)."""
    from .nedelec import QUAD4, QUAD4_W

    vol = mesh.volumes
    ins = mesh.mask("sphere")
    Bmean = (B[ins] * vol[ins, None]).sum(0) / vol[ins].sum()
    x = mesh.nodes[mesh.tets]
    e2 = np.zeros(len(vol))
    n2 = np.zeros(len(vol))
    for q in range(4):
        pts = np.einsum("v,kvd->kd", QUAD4[q], x)
        Be = case.B(pts)
        Be[ins] = case.B_in                                  # the curved interface: classify by the mesh region
        e2 += QUAD4_W[q] * ((B - Be) ** 2).sum(1)
        n2 += QUAD4_W[q] * (Be ** 2).sum(1)
    return {"mean_in": float(np.linalg.norm(Bmean - case.B_in) / np.linalg.norm(case.B_in)),
            "l2_in": float(np.sqrt((e2 * vol)[ins].sum() / (n2 * vol)[ins].sum())),
            "l2_all": float(np.sqrt((e2 * vol).sum() / (n2 * vol).sum())),
            "Bz_mean_in": float(Bmean[2])}


def solve_native(msh, case: SphereCase, **kw) -> dict:
    """Edge-element solution on a gmsh file from :func:`mesh_sphere`; returns errors, sizes and times."""
    t0 = time.perf_counter()
    mesh = read_msh22(msh)
    sp = EdgeSpace(mesh)
    nu = np.where(mesh.mask("sphere"), NU0 / case.mu_r, NU0)
    de = mesh.boundary_edges
    sol = solve_magnetostatic(mesh, nu, dirichlet=(de, sp.edge_values(case.A, de)), space=sp, **kw)
    out = errors(mesh, sol.B, case)
    out.update({"tets": int(len(mesh.tets)), "edges": int(mesh.n_edges), "unknowns": sol.info["unknowns"],
                "h": float(np.median(mesh.h[mesh.mask("sphere")])), "seconds": time.perf_counter() - t0,
                "energy": sol.energy})
    return out


def uniform_dirichlet(mesh: TetMesh, space: EdgeSpace, case: SphereCase):
    """Outer-boundary edge values of ``A0 = B0 x r / 2`` (the truncated problem that GetDP solves in
    :func:`~.getdp3d.sphere`)."""
    de = mesh.boundary_edges
    return de, space.edge_values(lambda x: 0.5 * np.cross(case.B0, x), de)

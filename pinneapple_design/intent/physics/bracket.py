"""Angle bracket under a tip load: beam theory, an in-house hexahedral FE model, and the CalculiX model with the
filleted corner and the bolt holes.

Frame (the same as :func:`pinneapple_simulation.external_solvers.calculix.study.geo_l_bracket`): the mounting leg
is a plate ``a x w x t`` on ``z = 0`` (bolted: its bottom face is clamped), the arm is a plate ``b x w x t`` standing
at ``x = 0..t``; the load ``P`` acts along ``+x`` on the top face of the arm (a shelf load when the mounting leg is
on a wall). Lengths in metres, forces in newtons, stresses in pascals.
"""
from __future__ import annotations

import os

import numpy as np


# ============================================================================================ closed form
def beam(P, b, w, t, E):
    """Cantilever of length ``b - t`` (from the top of the mounting leg) under a tip load: root bending stress
    ``M c / I`` and tip deflection ``P L^3 / (3 E I)`` (vectorised)."""
    L = b - t
    I_ = w * t ** 3 / 12
    sigma = P * L * (t / 2) / I_
    delta = P * L ** 3 / (3 * E * I_)
    return sigma, delta


# ============================================================================================ hexahedral FE
def l_mesh(a, b, w, t, n_t: int = 2, h: float | None = None, n_w: int = 4, fillet: float = 0.0):
    """Structured hexahedra of the L (no holes; the fillet, if any, as a staircase of whole elements): a
    :class:`HexMesh`, the clamped nodes (bottom of the mounting leg) and the loaded nodes (top face of the arm)."""
    from pinneapple_simulation.numerical_solvers.solid_fem import HexMesh

    h = h or t / n_t
    xs = np.unique(np.r_[np.linspace(0, t, n_t + 1), np.linspace(t, a, max(2, int(round((a - t) / h))) + 1)])
    zs = np.unique(np.r_[np.linspace(0, t, n_t + 1), np.linspace(t, b, max(2, int(round((b - t) / h))) + 1)])
    ys = np.linspace(0, w, n_w + 1)
    nx, ny, nz = len(xs), len(ys), len(zs)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    nodes_all = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    nid = np.arange(nx * ny * nz).reshape(nx, ny, nz)
    els = []
    for i in range(nx - 1):
        for k in range(nz - 1):
            xc, zc = 0.5 * (xs[i] + xs[i + 1]), 0.5 * (zs[k] + zs[k + 1])
            in_fillet = (t <= xc <= t + fillet and t <= zc <= t + fillet and
                         (xc - t - fillet) ** 2 + (zc - t - fillet) ** 2 >= fillet ** 2)
            if not (xc < t or zc < t or in_fillet):
                continue
            for j in range(ny - 1):
                els.append([nid[i, j, k], nid[i + 1, j, k], nid[i + 1, j + 1, k], nid[i, j + 1, k],
                            nid[i, j, k + 1], nid[i + 1, j, k + 1], nid[i + 1, j + 1, k + 1], nid[i, j + 1, k + 1]])
    els = np.array(els)
    used = np.unique(els)
    remap = -np.ones(len(nodes_all), int)
    remap[used] = np.arange(len(used))
    mesh = HexMesh(nodes_all[used], remap[els], (nx - 1, ny - 1, nz - 1), (a, w, b))
    P = mesh.nodes
    clamp = np.flatnonzero(P[:, 2] < 1e-12)
    top = np.flatnonzero(P[:, 2] > b - 1e-12)
    return mesh, clamp, top


def hex_model(P, a, b, w, t, E, nu, n_t: int = 2, fillet: float = 0.0):
    """Solve the L with :class:`SolidFEM` (C3D8I, incompatible modes). Returns the tip deflection (mean x
    displacement of the loaded face), the nominal root stress (bending stress on the arm's outer face at the section
    ``z = 2.5 t``, scaled to the root moment arm ``b - t``), and the result."""
    from pinneapple_simulation.numerical_solvers.solid_fem import SolidFEM

    mesh, clamp, top = l_mesh(a, b, w, t, n_t=n_t, fillet=fillet)
    fem = SolidFEM(mesh, E=E, nu=nu)
    fem.fix(clamp)
    fem.load_face(top, (P, 0.0, 0.0))
    res = fem.solve()
    X = mesh.nodes
    z_s = 2.5 * t
    zc = X[np.argmin(np.abs(X[:, 2] - z_s)), 2]
    outer = (np.abs(X[:, 2] - zc) < 1e-12) & (X[:, 0] < 1e-12)
    s_zz = float(np.abs(res.stress[outer, 2]).mean())
    sigma_root = s_zz * (b - t) / (b - zc)
    delta = float(res.u[top, 0].mean())
    return delta, sigma_root, res, mesh


# ============================================================================================ CalculiX
def geo_shelf_bracket(a: float, b: float, w: float, t: float, fillet: float, hole: float) -> str:
    """gmsh (OpenCASCADE) script: the L with a filleted inner corner and two bolt holes in the mounting leg."""
    xh = t + fillet + 0.5 * (a - t - fillet)
    return f"""SetFactory("OpenCASCADE");
Box(1) = {{0, 0, 0, {a}, {w}, {t}}};
Box(2) = {{0, 0, 0, {t}, {w}, {b}}};
Box(3) = {{{t}, 0, {t}, {fillet}, {w}, {fillet}}};
Cylinder(4) = {{{t + fillet}, -1, {t + fillet}, 0, {w + 2}, 0, {fillet}}};
BooleanDifference(5) = {{ Volume{{3}}; Delete; }}{{ Volume{{4}}; Delete; }};
BooleanUnion(6) = {{ Volume{{1}}; Delete; }}{{ Volume{{2, 5}}; Delete; }};
Cylinder(7) = {{{xh}, {w * 0.28}, -1, 0, 0, {t + 2}, {hole / 2}}};
Cylinder(8) = {{{xh}, {w * 0.72}, -1, 0, 0, {t + 2}, {hole / 2}}};
BooleanDifference(9) = {{ Volume{{6}}; Delete; }}{{ Volume{{7, 8}}; Delete; }};
Physical Volume("part") = {{9}};
Mesh.CharacteristicLengthFromCurvature = 1;
Mesh.MinimumElementsPerTwoPi = 24;
"""


def gmsh_exe() -> str | None:
    import shutil
    import sys
    exe = shutil.which("gmsh")
    if exe:
        return exe
    cand = os.path.join(os.path.dirname(sys.executable), "gmsh")
    return cand if os.path.exists(cand) else None


def export_cad(geo: str, workdir: str) -> dict[str, str]:
    """STEP (the OpenCASCADE B-rep, exact holes and fillet) and STL (surface triangulation) from the gmsh script."""
    import subprocess
    exe = gmsh_exe()
    if exe is None:
        raise RuntimeError("gmsh not found")
    os.makedirs(workdir, exist_ok=True)
    gp = os.path.join(workdir, "bracket.geo")
    with open(gp, "w") as f:
        f.write(geo)
    out = {}
    step = os.path.join(workdir, "bracket.step")
    p = subprocess.run([exe, gp, "-0", "-o", step, "-format", "step"], capture_output=True, text=True, timeout=600)
    if p.returncode == 0 and os.path.exists(step):
        out["step"] = step
    stl = os.path.join(workdir, "bracket.stl")
    p = subprocess.run([exe, gp, "-2", "-clmax", "0.002", "-o", stl, "-format", "stl", "-setnumber", "Mesh.Binary",
                        "1", "-setnumber", "Mesh.SaveAll", "1"], capture_output=True,
                       text=True, timeout=600)
    if p.returncode == 0 and os.path.exists(stl):
        out["stl"] = stl
    return out


def calculix_model(P, a, b, w, t, fillet, hole, E, nu, rho, size, workdir):
    """Mesh with gmsh (C3D10 tetrahedra), clamp the mounting leg's bottom face (bolted), load the arm's top face,
    solve with CalculiX. Returns ``(model, result, peak_vm, peak_where, delta, reaction)``."""
    import shutil

    from pinneapple_simulation.external_solvers.calculix.study import (
        FEModel,
        Material,
        Static,
        solve,
    )

    exe = gmsh_exe()
    if exe and not shutil.which("gmsh"):
        os.environ["PATH"] = os.path.dirname(exe) + os.pathsep + os.environ.get("PATH", "")
    m = FEModel.from_gmsh(geo_shelf_bracket(a, b, w, t, fillet, hole), size=size, workdir=os.path.join(workdir, "g"))
    m.material = Material("bracket", E, nu, rho)
    base = m.nodes_where(lambda X: X[:, 2] < 1e-9)
    top = m.nodes_where(lambda X: X[:, 2] > b - 1e-9)
    r = solve(m, Static(fix=[(base, (1, 2, 3))], loads=[(top, (P, 0.0, 0.0))]), os.path.join(workdir, "s"))
    X = m.nodes
    vm = r.von_mises
    # peak away from the clamped face (its edge is a singularity of the idealised clamp) and the loaded face
    zone = (X[:, 2] > 0.5 * t) & (X[:, 2] < b - 2 * t)
    i = int(np.flatnonzero(zone)[np.argmax(vm[zone])])
    delta = float(r.u[top, 0].mean())
    reaction = float(-r.reactions[base, 0].sum()) if r.reactions is not None else float("nan")
    return m, r, float(vm[i]), X[i], delta, reaction


def section_stress(m, r, b, w, t, z_frac: float = 0.45):
    """Mean |sigma_zz| on the arm's outer face at ``z = z_frac b`` (away from the corner and the load), and the
    beam-theory value ``M c / I`` there."""
    X = m.nodes
    zc = z_frac * b
    sec = (np.abs(X[:, 2] - zc) < max(0.08 * b, 2e-3)) & (X[:, 0] < 1e-9) & (np.abs(X[:, 1] - w / 2) < 0.3 * w)
    return float(np.abs(r.stress[sec, 2]).mean()), zc, int(sec.sum())


def mesh_sizes(t: float) -> tuple[float, float]:
    """Two quadratic-tetrahedron sizes: about two and three elements through the thickness."""
    return (max(t * 0.5, 0.001), max(t / 3, 0.0007))


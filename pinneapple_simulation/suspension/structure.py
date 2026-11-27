"""A wishbone as CAD, as a mesh, and as a CalculiX result: mass, stress and the compliance at its ball joint.

The part is built with the OpenCASCADE kernel that gmsh carries: two hollow tubes from the bush pivots meet a
spherical boss at the ball joint, and the bores are cut with a slightly longer inner cylinder so the cut is clean at
the ends. Walls are real (a tube, not a rod) because wall thickness is what a wishbone designer changes. The same
build writes ``part.step``: that file is the CAD the mesh came from, kept next to the result.

What the arm is asked for:

* its **mass**, from the exact B-rep volume of the Boolean result (not from the mesh, so it does not move with the
  element size);
* the **compliance matrix** at the ball joint -- three unit loads in x, y, z, the mean displacement of the boss
  surface -- which is the FEM half of the elastokinematic coupling in :mod:`.elastokinematics`;
* the **peak von Mises stress** under the corner's worst combined load, on the nodes away from the load and support
  patches (a stress reported at the node a load is applied to measures the load, not the part).

Every one of those numbers is verified in :mod:`pinneapple_simulation.suspension.verification`: a hollow tube built by
the same code, run through the same deck and the same solver, must reproduce Timoshenko's cantilever deflection.
"""
from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..external_solvers.calculix.study import FEModel, Material, Static, solve
from .hardpoints import DENSITY, UPPER_WALL, YIELD_STRENGTH, Corner

__all__ = ["Arm", "arm", "mesh_model", "response", "mesh_convergence", "corner_loads", "tube_cantilever"]

STEEL = Material("steel", E=210e9, nu=0.3, density=DENSITY)

#: The mesh every campaign run uses, and why: linear tets at 14 mm mesh a wishbone in ~4 s and solve in <1 s, which
#: is what makes a 120-design DOE affordable; the closed-form check in :func:`tube_cantilever` measures what that
#: choice costs in deflection (see :mod:`pinneapple_simulation.suspension.verification`), and
#: :func:`mesh_convergence` measures it on the arm itself.
PRODUCTION_SIZE = 0.014
PRODUCTION_ORDER = 1

#: The corner load case: 1.2 g vertical, 0.9 g lateral and 0.25 g longitudinal on a 350 kg corner, in newtons.
#: The whole corner load is put on the lower wishbone at the ball joint, with the bush ends fixed: conservative (the
#: upper wishbone and the spring take their share in a real corner, and the damper's reaction on this arm is left out).
CORNER_MASS = 350.0
LOAD_CASE = dict(vertical=1.2, lateral=0.9, longitudinal=0.25)      # multiples of CORNER_MASS * 9.81
G = 9.81


@dataclass(frozen=True)
class Arm:
    """One wishbone: the two chassis pivots, the ball joint, the tubes, and the boss the ball joint sits in."""
    name: str
    p1: np.ndarray
    p2: np.ndarray
    ball: np.ndarray
    od: float
    wall: float
    boss_radius: float


def arm(c: Corner, which: str = "lower") -> Arm:
    """The lower or upper wishbone of a corner, with its boss sized to hold the ball joint."""
    if which == "lower":
        od, wall = c.lc_od, c.lc_wall
        p1, p2, ball = c.lca_front, c.lca_rear, c.lbj
    elif which == "upper":
        od, wall = c.uc_od, UPPER_WALL
        p1, p2, ball = c.uca_front, c.uca_rear, c.ubj
    else:                                                              # pragma: no cover - caller error
        raise ValueError(f"which must be 'lower' or 'upper', not {which!r}")
    return Arm(which, np.asarray(p1, float), np.asarray(p2, float), np.asarray(ball, float), float(od),
               float(wall), max(0.022, 0.85 * float(od)))


def _occ_part(a: Arm, *, boss: bool = True, bore: bool = True):
    """Add the part to the current gmsh OCC model and return (volume, tip volume tag) -- volume from the B-rep."""
    import gmsh

    occ = gmsh.model.occ

    def cyl(p0, p1, radius, ext=0.0):
        d = np.asarray(p1, float) - np.asarray(p0, float)
        n = d / np.linalg.norm(d)
        s = np.asarray(p0, float) - n * ext
        return occ.addCylinder(*map(float, s), *map(float, d + 2 * ext * n), float(radius))

    legs = [cyl(a.p1, a.ball, a.od / 2), cyl(a.p2, a.ball, a.od / 2)]
    if boss:
        fused, _ = occ.fuse([(3, legs[0])], [(3, legs[1]), (3, occ.addSphere(*map(float, a.ball), a.boss_radius))])
    else:
        fused, _ = occ.fuse([(3, legs[0])], [(3, legs[1])])
    tag = fused[0][1]
    if bore:
        bores = [cyl(a.p1, a.ball, a.od / 2 - a.wall, ext=0.01),
                 cyl(a.p2, a.ball, a.od / 2 - a.wall, ext=0.01)]
        cut, _ = occ.cut([(3, tag)], [(3, t) for t in bores])
        tag = cut[0][1]
    return gmsh.model.occ.getMass(3, tag), tag


def mesh_model(a: Arm, size: float = PRODUCTION_SIZE, order: int = PRODUCTION_ORDER, workdir: str | None = None,
               save_step: bool = False) -> tuple[FEModel, float]:
    """Mesh the arm and return (model, exact volume in m^3). ``size``: the gmsh target element size in metres.

    A high-order mesh of this B-rep occasionally fails in gmsh's OCC curving step (``PLC Error: a segment and a facet
    intersect``); the retry grows the element size by 15 % at a time, and the size that was really used comes back in
    the model's ``info`` through :func:`response`.
    """
    box: dict[str, float] = {}

    def build() -> None:
        box["volume"], _ = _occ_part(a)

    last: Exception | None = None
    for k in range(3):
        h = size * 1.15 ** k
        try:
            model = FEModel.from_gmsh_model(build, h, order=order, workdir=workdir,
                                            save=("part.step",) if (save_step and k == 0) else ())
            break
        except Exception as exc:                                        # pragma: no cover - mesher hiccup
            last = exc
    else:                                                              # pragma: no cover - mesher hiccup
        raise RuntimeError(f"gmsh could not mesh the {a.name} wishbone at {size:.4g} m: {last}")
    model.material = STEEL
    model.title = f"{a.name} wishbone"
    return model, box["volume"]


# ------------------------------------------------------------------------------------------------ loads
def corner_loads() -> dict[str, float]:
    """The three components of the worst combined corner load at the ball joint, in newtons."""
    w = CORNER_MASS * G
    return {"fx": LOAD_CASE["longitudinal"] * w, "fy": LOAD_CASE["lateral"] * w,
            "fz": LOAD_CASE["vertical"] * w}


def _nodesets(model: FEModel, a: Arm, patch: float = 0.03) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fixed, boss, far): the bush ends (fixed), the ball-joint boss (loaded), and the nodes neither of them."""
    fixed = np.unique(np.r_[model.nodes_where(lambda X: np.linalg.norm(X - a.p1, axis=1) < patch),
                            model.nodes_where(lambda X: np.linalg.norm(X - a.p2, axis=1) < patch)])
    boss = model.nodes_where(lambda X: np.linalg.norm(X - a.ball, axis=1) <= 1.02 * a.boss_radius)
    far = np.setdiff1d(np.arange(len(model.nodes)), np.union1d(fixed, boss))
    return fixed, boss, far


def _mean_u(res, idx: np.ndarray) -> np.ndarray:
    return np.asarray(res.u[idx], float).mean(0)


def response(a: Arm, workdir: str, *, size: float = PRODUCTION_SIZE, order: int = PRODUCTION_ORDER,
             unit: float = 1000.0, keep: bool = False) -> dict[str, object]:
    """Mass, peak stress and the 3 x 3 ball-joint compliance of one arm.

    Four runs on one mesh: three unit loads along x, y, z for the compliance columns, and the corner's combined load
    for the stress. Returns ``volume_m3``, ``mass_kg``, ``vm_max_pa``, ``vm_node``, ``compliance_m_per_n`` (3 x 3),
    ``dofs`` and the timings the study reports.
    """
    model, volume = mesh_model(a, size, order=order, workdir=workdir)
    fixed, boss, far = _nodesets(model, a)
    fix = {tuple(fixed): (1, 2, 3)}
    loads = corner_loads()

    C = np.zeros((3, 3))
    for k in range(3):
        F = np.zeros(3)
        F[k] = unit
        res = solve(model, Static(fix=fix, loads={tuple(boss): tuple(F)}), os.path.join(workdir, f"unit{k}"),
                    threads=1, keep=keep)
        C[:, k] = _mean_u(res, boss) / unit

    res = solve(model, Static(fix=fix, loads={tuple(boss): (loads["fx"], loads["fy"], loads["fz"])}),
                os.path.join(workdir, "combined"), threads=1, keep=keep)
    vm = res.von_mises
    vm_far = vm[far] if len(far) else vm
    return {
        "volume_m3": float(volume),
        "mass_kg": float(volume) * DENSITY,
        "vm_max_pa": float(vm_far.max()),
        "vm_max_all_pa": float(vm.max()),
        "vm_node": int(far[int(np.argmax(vm_far))]) if len(far) else -1,
        "compliance_m_per_n": 0.5 * (C + C.T),
        "dofs": int(3 * len(model.nodes)),
        "elements": int(len(model.elements)),
        "u_boss_mm": float(np.linalg.norm(_mean_u(res, boss))) * 1e3,
        "model": model,
        "result": res,
    }


def mesh_convergence(a: Arm, workdir: str, sizes=(0.024, 0.019, PRODUCTION_SIZE),
                     order: int = PRODUCTION_ORDER) -> dict[str, object]:
    """The peak stress and the ball-joint deflection on three meshes, with Richardson and the GCI on each.

    A stress peak at a tube-to-boss junction is not a smooth quantity: this is the measurement that says how far from
    converged it is, which is what a design check against yield has to know before it trusts a safety factor.
    """
    from pinneapple_autolab.study.mesh_study import grid_convergence
    rows = []
    for s in sorted(sizes, reverse=True):
        r = response(a, os.path.join(workdir, f"h{s:.3f}"), size=s, order=order, keep=False)
        rows.append({"name": f"{s * 1e3:.0f} mm", "cells": int(r["elements"]), "values": {
            "vm": float(r["vm_max_pa"]) * 1e-6, "u": float(r["u_boss_mm"])}, "cost_s": float(r.get("cost_s", 0.0))})
    out: dict[str, object] = {"meshes": rows}
    try:
        out["gci"] = grid_convergence(rows, ("vm", "u"))
    except Exception as exc:                                           # pragma: no cover - helper shape
        out["gci"] = {"error": str(exc)}
    return out


# ------------------------------------------------------------------------------------------------ verification
def tube_cantilever(p0=(0.0, 0.0, 0.0), p1=(0.40, 0.0, 0.0), od=0.030, wall=0.003, size: float = 0.010,
                    order: int = 1, workdir: str | None = None, load: float = 1000.0) -> dict[str, object]:
    """A hollow tube cantilever built and solved exactly like a wishbone leg, against Timoshenko's beam.

    The closed form, for a thin-walled or thick-walled circular tube of outer diameter ``od``, wall ``wall``,
    length ``L``, with the shear coefficient of a hollow circle
    ``k = 6(1+nu)(1+m^2)^2 / ((7+6nu)(1+m^2)^2 + (20+12nu) m^2)``, ``m = di/do``:

        delta = F L^3 / (3 E I) + F L / (k G A)

    The tube is fixed on its whole end face, which is *not* the beam's plane-section-at-the-support assumption (the
    clamp suppresses the shear warping there), so agreement is reported as measured, not assumed.
    """
    a = Arm("tube", np.asarray(p0, float), np.asarray(p0, float), np.asarray(p1, float), od, wall, 0.0)
    model = FEModel.from_gmsh_model(lambda: _occ_part(a, boss=False), size, order=order,
                                    workdir=workdir or os.path.join(os.getcwd(), "_tube"))
    model.material = STEEL
    x0, x1 = np.asarray(p0, float), np.asarray(p1, float)
    axis = (x1 - x0) / np.linalg.norm(x1 - x0)
    # the clamp is the end *face* (2 mm window), not a 20 mm ball around it: clamping 20 mm of the tube would test a
    # 0.38 m cantilever and call it 0.40 m, which is exactly the 14 % the length cubed would hide
    root = model.nodes_where(lambda X: np.abs((X - x0) @ axis) < 2e-3)
    tip = model.nodes_where(lambda X: np.abs((X - x1) @ axis) < 2e-3)
    res = solve(model, Static(fix={tuple(root): (1, 2, 3)}, loads={tuple(tip): (0.0, 0.0, -load)}),
                os.path.join(workdir or "_tube", "run"), threads=1, keep=False)
    E, nu = STEEL.E, STEEL.nu
    di = od - 2 * wall
    I = math.pi * (od ** 4 - di ** 4) / 64
    A = math.pi * (od ** 2 - di ** 2) / 4
    m = di / od
    k = 6 * (1 + nu) * (1 + m ** 2) ** 2 / ((7 + 6 * nu) * (1 + m ** 2) ** 2 + (20 + 12 * nu) * m ** 2)
    Gs = E / (2 * (1 + nu))
    L = float(np.linalg.norm(np.asarray(p1) - np.asarray(p0)))
    theory = load * L ** 3 / (3 * E * I) + load * L / (k * Gs * A)
    got = float(-np.mean(res.u[tip, 2]))
    return {"deflection_m": got, "timoshenko_m": theory, "error_rel": got / theory - 1.0,
            "bending_share": (load * L ** 3 / (3 * E * I)) / theory, "L": L, "I": I, "A": A, "k_shear": k,
            "order": order, "size": size, "elements": int(len(model.elements))}

"""Post-processing of a finished internal-flow case: what the pipeline reports and checks.

Everything is read from the case (latest time) with the finite-volume reader of :mod:`pinneapple_check`
(``load_openfoam``, :class:`~pinneapple_check.fv.FV`) and from the solver log:

* per inlet / outlet patch: area, area-averaged static pressure (Pa), volume flux (m³/s, from ``phi``), and the
  bulk (flux-weighted, mixing-cup) value of the passive scalar;
* pressure drop inlet -> outlet (static and total), mass imbalance |sum of boundary fluxes| / inflow;
* cross sections (:class:`Section`): discs sampled inside the fluid, area-averaged pressure, flux through them and
  bulk scalar (e.g. the pressure gradient inside a bend, a bulk temperature along a heated pipe);
* forces on the walls and y+ per wall patch (:func:`wall_loads`, from the fields), the largest Courant number of a
  transient run (solver log);
* figures: speed and pressure on a plane through the domain, the mesh cut by that plane.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["Section", "CaseResult", "read_case", "section_average", "wall_loads", "plane_figure", "mesh_figure",
           "geometry_figure"]


@dataclass
class Section:
    """A disc of ``radius`` (m) centred at ``origin`` (m) with unit ``normal`` (the positive flow direction)."""
    name: str
    origin: tuple[float, float, float]
    normal: tuple[float, float, float]
    radius: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CaseResult:
    """Fields of the latest time of a case on its mesh: cell centres, volumes, U, p and the scalar, the boundary
    patches and their face values, with nearest-cell sampling."""

    def __init__(self, case: str | Path, scalar: str | None = None):
        from pinneapple_check.fields import load_openfoam
        from pinneapple_check.fv import FV
        self.case = Path(case)
        only = ["U", "p"] + ([scalar] if scalar else [])
        self.fd = load_openfoam(self.case, time="latest", only=only)
        self.fv = FV(self.fd)
        self.scalar = scalar
        self.time = self.fd.time
        self.U = np.asarray(self.fd.fields["U"], float)
        self.p = np.asarray(self.fd.fields["p"], float).ravel()
        self.s = None if not scalar or scalar not in self.fd.fields else np.asarray(self.fd.fields[scalar], float).ravel()
        self._tree = None

    @property
    def n_cells(self) -> int:
        return int(self.fv.nc)

    def tree(self):
        if self._tree is None:
            from scipy.spatial import cKDTree
            self._tree = cKDTree(self.fv.C)
        return self._tree

    def sample(self, P: np.ndarray, k: int = 6) -> dict[str, np.ndarray]:
        """Inverse-distance values at points from the ``k`` nearest cell centres, and ``inside`` (the nearest
        centre is closer than the size of the largest of those cells)."""
        d, i = self.tree().query(P, k=k)
        w = 1.0 / (d + 1e-12) ** 2
        w /= w.sum(1, keepdims=True)
        h = np.cbrt(np.abs(self.fv.V[i])).max(1)
        out = {"U": (self.U[i] * w[..., None]).sum(1), "p": (self.p[i] * w).sum(1), "inside": d[:, 0] <= h}
        if self.s is not None:
            out["s"] = (self.s[i] * w).sum(1)
        return out

    def patch(self, name: str) -> dict[str, float]:
        """Area, area-averaged p, flux (outward positive) and bulk scalar of one boundary patch."""
        fv, fd = self.fv, self.fd
        sl = fv.patch_slice(name)
        if sl is None:
            raise KeyError(f"no patch {name}; patches are {[p['name'] for p in fd.patches]}")
        area = fv.Smag[fv.ni:][sl]
        pb = fv.bvals("p", self.p)[sl]
        F = fv.flux(self.U, fv.bvals("U", self.U), fd.phi)[fv.ni:][sl]
        out = {"area_m2": float(area.sum()), "p_mean": float((pb * area).sum() / area.sum()), "flux_m3_s": float(F.sum())}
        if self.s is not None:
            sb = fv.bvals(self.scalar, self.s)[sl]
            out["scalar_bulk"] = float((F * sb).sum() / F.sum()) if abs(F.sum()) > 0 else float("nan")
            out["scalar_mean"] = float((sb * area).sum() / area.sum())
        return out


def _disc(origin, normal, radius: float, n_r: int = 14) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(normal, float)
    n /= np.linalg.norm(n)
    a = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    pts, wts = [], []
    for i in range(n_r):                                  # equal-area rings, points spread along each ring
        r0, r1 = radius * math.sqrt(i / n_r), radius * math.sqrt((i + 1) / n_r)
        r = 0.5 * (r0 + r1)
        m = max(6, int(round(2 * math.pi * r / (radius / n_r))))
        for j in range(m):
            t = 2 * math.pi * (j + 0.5 * (i % 2)) / m
            pts.append(np.asarray(origin, float) + r * (math.cos(t) * a + math.sin(t) * b))
            wts.append(math.pi * (r1 * r1 - r0 * r0) / m)
    return np.asarray(pts), np.asarray(wts)


def section_average(res: CaseResult, sec: Section) -> dict[str, float]:
    """Area-averaged p (kinematic), flux and bulk scalar over the part of the disc inside the fluid."""
    P, w = _disc(sec.origin, sec.normal, sec.radius)
    v = res.sample(P)
    m = v["inside"]
    if not m.any():
        raise ValueError(f"section {sec.name}: no sample point inside the fluid")
    n = np.asarray(sec.normal, float) / np.linalg.norm(sec.normal)
    un = v["U"][m] @ n
    out = {"area_m2": float(w[m].sum()), "fraction_inside": float(w[m].sum() / w.sum()),
           "p_mean": float((v["p"][m] * w[m]).sum() / w[m].sum()), "flux_m3_s": float((un * w[m]).sum())}
    if "s" in v:
        out["scalar_bulk"] = float((un * v["s"][m] * w[m]).sum() / (un * w[m]).sum())
    return out


def wall_loads(case: str | Path, walls: list[str], nu: float, rho: float) -> dict[str, Any]:
    """Pressure and viscous force (N) on the wall patches and y+ of their first cells, from the written fields:
    :meth:`~pinneapple_simulation.numerical_solvers.external_flow.CaseFields.wall` (laminar sublayer, log law above
    y+ = 11.25), y+ = d u_tau / nu with d the wall distance of the first cell centre."""
    from pinneapple_simulation.numerical_solvers.external_flow import CaseFields
    cf = CaseFields(str(case))
    Fp, Fv = np.zeros(3), np.zeros(3)
    yp: dict[str, dict[str, float]] = {}
    by = {p["name"]: p for p in cf.patches()}
    for name in walls:
        if name not in by or int(by[name]["nFaces"]) == 0:
            continue
        w = cf.wall(by[name], nu, 1.0)                          # q = 1: cf is tau_w / 1 (kinematic)
        Fp += w["f_pressure"].sum(0)
        Fv += w["f_viscous"].sum(0)
        s, n = int(by[name]["startFace"]), int(by[name]["nFaces"])
        idx = np.arange(s, s + n)
        own = cf.mesh.poly.owner[idx]
        nh = cf.fg["areas"][idx] / np.linalg.norm(cf.fg["areas"][idx], axis=1, keepdims=True)
        d = np.abs(((cf.cg["centres"][own] - cf.fg["centres"][idx]) * nh).sum(1))
        y = d * np.sqrt(w["cf"]) / nu
        A = np.linalg.norm(cf.fg["areas"][idx], axis=1)
        yp[name] = {"min": float(y.min()), "max": float(y.max()), "average": float((y * A).sum() / A.sum())}
    return {"pressure": (Fp * rho).tolist(), "viscous": (Fv * rho).tolist(), "total": ((Fp + Fv) * rho).tolist(),
            "yplus": yp}


def read_case(case: str | Path, kinds: dict[str, str], rho: float, sections: list[Section] = (),
              scalar: str | None = None, nu: float = 1e-6) -> dict[str, Any]:
    """Every measured quantity of a finished case (see the module docstring); pressures in Pa."""
    case = Path(case)
    res = CaseResult(case, scalar)
    out: dict[str, Any] = {"time": res.time, "cells": res.n_cells, "patches": {}, "sections": {}}
    for p, k in kinds.items():
        if k in ("inlet", "outlet"):
            d = res.patch(p)
            d["p_mean_pa"] = d["p_mean"] * rho
            d["velocity_mean"] = abs(d["flux_m3_s"]) / d["area_m2"]
            out["patches"][p] = d
    inl = [p for p, k in kinds.items() if k == "inlet"]
    outl = [p for p, k in kinds.items() if k == "outlet"]
    q_in = -sum(out["patches"][p]["flux_m3_s"] for p in inl)
    net = sum(v["flux_m3_s"] for v in out["patches"].values())
    walls = [p for p, k in kinds.items() if k == "wall"]
    for p in walls:                                       # no-slip walls carry no flux; count it if they do
        net += res.patch(p)["flux_m3_s"]
    out["inflow_m3_s"] = q_in
    out["mass_imbalance"] = abs(net) / q_in if q_in > 0 else float("inf")
    if inl and outl:
        def mean_over(ps, key):
            A = sum(out["patches"][p]["area_m2"] for p in ps)
            return sum(out["patches"][p][key] * out["patches"][p]["area_m2"] for p in ps) / A
        pin, pout = mean_over(inl, "p_mean_pa"), mean_over(outl, "p_mean_pa")
        uin, uout = mean_over(inl, "velocity_mean"), mean_over(outl, "velocity_mean")
        out["dp_static_pa"] = pin - pout
        out["dp_total_pa"] = pin - pout + 0.5 * rho * (uin ** 2 - uout ** 2)
        out["inlet_velocity_mean"] = uin
        if scalar and res.s is not None:
            out["scalar_in"] = mean_over(inl, "scalar_bulk")
            out["scalar_out"] = mean_over(outl, "scalar_bulk")
    for s in sections:
        d = section_average(res, s)
        d["p_mean_pa"] = d["p_mean"] * rho
        out["sections"][s.name] = d
    loads = wall_loads(case, walls, nu, rho) if walls else {"yplus": {}}
    out["yplus"] = loads.pop("yplus")
    out["forces_n"] = loads or None
    out["_result"] = res
    return out


# ----------------------------------------------------------------------------------------------- figures
def _plane_axes(normal) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    a = a - (a @ n) * n
    a /= np.linalg.norm(a)
    return n, a, np.cross(n, a)


def plane_figure(res: CaseResult, path: str | Path, origin, normal, title: str, rho: float,
                 scalar_label: str | None = None, n: int = 360) -> Path:
    """Speed, pressure (and the scalar) sampled on a plane through the domain; outside the fluid is blank."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nrm, a, b = _plane_axes(normal)
    C = res.fv.C
    rel = C - np.asarray(origin, float)
    near = np.abs(rel @ nrm) < 2 * np.cbrt(np.abs(res.fv.V)).max()
    ua, ub = rel[near] @ a, rel[near] @ b
    lo_a, hi_a, lo_b, hi_b = ua.min(), ua.max(), ub.min(), ub.max()
    na = n
    nb = max(20, int(n * (hi_b - lo_b) / max(hi_a - lo_a, 1e-30)))
    A, B = np.meshgrid(np.linspace(lo_a, hi_a, na), np.linspace(lo_b, hi_b, nb))
    P = np.asarray(origin, float) + A.ravel()[:, None] * a + B.ravel()[:, None] * b
    v = res.sample(P)
    mask = ~v["inside"].reshape(nb, na)
    panels = [("speed |U| [m/s]", np.linalg.norm(v["U"], axis=1)), ("static pressure p [Pa]", v["p"] * rho)]
    if "s" in v and scalar_label:
        panels.append((scalar_label, v["s"]))
    stretch = (hi_a - lo_a) / max(hi_b - lo_b, 1e-30) > 6          # long thin domains: stretch the short side
    aspect = "auto" if stretch else "equal"
    if stretch:
        title += " (short axis stretched)"
    fig, axes = plt.subplots(len(panels), 1, figsize=(8.5, (1.9 if stretch else 2.6) * len(panels) + 0.6), dpi=120,
                             squeeze=False)
    ext = [lo_a * 1e3, hi_a * 1e3, lo_b * 1e3, hi_b * 1e3]
    for ax, (lab, val) in zip(axes[:, 0], panels, strict=True):
        G = np.ma.array(val.reshape(nb, na), mask=mask)
        im = ax.imshow(G, origin="lower", extent=ext, cmap="viridis" if "speed" in lab else "coolwarm"
                       if "pressure" in lab else "inferno", aspect=aspect, interpolation="nearest")
        fig.colorbar(im, ax=ax, label=lab, shrink=0.9)
        ax.set_xlabel(f"along ({a[0]:.2g}, {a[1]:.2g}, {a[2]:.2g}) [mm]")
        ax.set_ylabel("[mm]")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


def mesh_figure(res: CaseResult, path: str | Path, origin, normal, title: str) -> Path:
    """The cells cut by a plane, each drawn as a square of its size at its centre (snappyHexMesh cells are cubes
    away from the walls), coloured by size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    nrm, a, b = _plane_axes(normal)
    h = np.cbrt(np.abs(res.fv.V))
    rel = res.fv.C - np.asarray(origin, float)
    fv = res.fv
    fd = (fv.Cf - np.asarray(origin, float)) @ nrm                  # cut cells: faces on both sides of the plane
    fd = np.where(np.abs(fd) < 1e-9 * h.max(), 0.0, fd)             # (a plane on faces takes the cells below it)
    lo, hi = np.full(fv.nc, np.inf), np.full(fv.nc, -np.inf)
    for cells, vals in ((fv.own, fd), (fv.nei, fd[:fv.ni])):
        np.minimum.at(lo, cells, vals)
        np.maximum.at(hi, cells, vals)
    cut = (lo < 0) & (hi >= 0)
    ua, ub, hh = rel[cut] @ a * 1e3, rel[cut] @ b * 1e3, h[cut] * 1e3
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=120)
    rects = [Rectangle((x - s / 2, y - s / 2), s, s) for x, y, s in zip(ua, ub, hh, strict=True)]
    pc = PatchCollection(rects, cmap="viridis", edgecolor="k", linewidth=0.15)
    pc.set_array(hh)
    ax.add_collection(pc)
    ax.set_xlim(ua.min() - hh.max(), ua.max() + hh.max())
    ax.set_ylim(ub.min() - hh.max(), ub.max() + hh.max())
    ax.set_aspect("equal")
    fig.colorbar(pc, ax=ax, label="cell size [mm]", shrink=0.9)
    ax.set_title(f"{title}: {cut.sum()} cells on the plane, {res.n_cells} in the mesh", fontsize=9)
    ax.set_xlabel("[mm]")
    ax.set_ylabel("[mm]")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


def geometry_figure(geom, assignment: dict[str, list[int]], path: str | Path, title: str) -> Path:
    """The exported solid with its faces coloured by patch (3-D, matplotlib)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    colors = ["#1f77b4", "#d62728", "#7f7f7f", "#ff7f0e", "#2ca02c", "#9467bd"]
    fig = plt.figure(figsize=(7.5, 5.2), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    lo, hi = geom.bounds()
    for k, (name, idx) in enumerate(assignment.items()):
        tris = np.concatenate([geom.faces[i].points[geom.faces[i].triangles] for i in idx])
        step = max(1, len(tris) // 6000)
        pc = Poly3DCollection(tris[::step], facecolor=colors[k % len(colors)], edgecolor="none", alpha=0.75)
        ax.add_collection3d(pc)
        ax.plot([], [], color=colors[k % len(colors)], lw=6, label=f"{name} ({len(idx)} faces)")
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.view_init(elev=28, azim=-60)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return Path(path)

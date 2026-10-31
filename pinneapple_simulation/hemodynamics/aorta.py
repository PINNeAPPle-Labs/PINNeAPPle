"""Pulsatile flow in the library's thoracic aorta (``aortic_arch``) with three-element Windkessel outlets, in
OpenFOAM.

* :func:`find_caps` finds the flat caps of a closed vessel surface (connected sets of exactly coplanar triangles
  that are discs), the inlet and outlets; :func:`aorta_stl` writes the aorta as a multi-solid STL with the patches
  ``inlet``, ``outlet_desc`` (descending aorta), ``outlet_rsa``, ``outlet_rcca``, ``outlet_lcca``, ``outlet_lsa``
  (right subclavian, right / left common carotid, left subclavian) and ``wall``;
* :func:`setup_case` meshes it (blockMesh + snappyHexMesh, through
  :mod:`pinneapple_simulation.pipelines.freecad_openfoam`) and writes a laminar ``pimpleFoam`` case: the inflow
  ``Q(t)`` imposed with ``flowRateInletVelocity`` (plug profile), fixed pressure on each outlet;
* :class:`CoupledRun` couples the outlets to :class:`~.windkessel.Windkessel` models from Python: the solver runs
  one window ``dt_couple`` at a time, then each outlet's flux (from ``phi``) advances its Windkessel exactly
  (piecewise-constant flow) and the new outlet pressure is written into the latest time before the next window.
  OpenFOAM v1912 from Ubuntu has no Windkessel boundary condition and no ``wmake`` for ``codedFixedValue``, hence
  this explicit, staggered coupling; with the exponential update it is stable for the window lengths used here, and
  :meth:`CoupledRun.check` reports how far the outlet pressures are from the Windkessels driven by the 3-D flows.

Blood is Newtonian by default (``rho`` 1060 kg/m^3, ``mu`` 3.5 mPa s, round values); ``viscosity="carreau"`` uses
OpenFOAM's BirdCarreau model with the Cho and Kensey (Biorheology 28:241-262, 1991) blood coefficients.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .windkessel import Windkessel

__all__ = ["find_caps", "aorta_stl", "setup_case", "CoupledRun", "BLOOD", "CARREAU_CHO_KENSEY"]

BLOOD = {"rho": 1060.0, "mu": 3.5e-3}
# Cho and Kensey 1991, Carreau model: mu0 0.056 Pa s, mu_inf 0.00345 Pa s, lambda 3.313 s, n 0.3568
CARREAU_CHO_KENSEY = {"mu0": 0.056, "mu_inf": 0.00345, "lambda": 3.313, "n": 0.3568}


def find_caps(V: np.ndarray, F: np.ndarray, *, compactness: float = 0.7, min_area: float = 1e-6):
    """Flat disc-shaped caps of a closed triangle surface: ``[{"tris", "normal" (outward), "centroid",
    "diameter", "area"}]``, largest first."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    P = V[F]
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    A = 0.5 * np.linalg.norm(n, axis=1)
    n = n / (2 * A[:, None])
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    tri = np.tile(np.arange(len(F)), 3)
    o = np.lexsort(E.T[::-1])
    E, tri = E[o], tri[o]
    same = np.all(E[1:] == E[:-1], axis=1)
    a, b = tri[:-1][same], tri[1:][same]
    ok = np.einsum("ij,ij->i", n[a], n[b]) > 0.99999
    _, lab = connected_components(coo_matrix((np.ones(ok.sum()), (a[ok], b[ok])), shape=(len(F), len(F))),
                                  directed=False)
    areas = np.bincount(lab, A)
    caps = []
    for c in np.flatnonzero(areas > min_area):
        m = lab == c
        cen = (P[m].mean(1) * A[m, None]).sum(0) / A[m].sum()
        r = np.linalg.norm(P[m].reshape(-1, 3) - cen, axis=1).max()
        if areas[c] / (math.pi * r * r) > compactness:
            caps.append({"tris": np.flatnonzero(m), "normal": n[m][0], "centroid": cen, "diameter": 2 * r,
                         "area": float(areas[c])})
    return sorted(caps, key=lambda c: -c["area"])


def _label_aorta_caps(caps, params) -> dict[str, dict]:
    """inlet: the cap at the origin (aortic annulus); outlet_desc: the lowest; branches by nearest diameter."""
    i_in = min(range(len(caps)), key=lambda i: np.linalg.norm(caps[i]["centroid"]))
    rest = [i for i in range(len(caps)) if i != i_in]
    i_desc = min(rest, key=lambda i: caps[i]["centroid"][2])
    branches = [i for i in rest if i != i_desc]
    want = {"outlet_rsa": params["rsa_diameter"], "outlet_rcca": params["rcca_diameter"],
            "outlet_lcca": params["lcca_diameter"], "outlet_lsa": params["lsa_diameter"]}
    out = {"inlet": caps[i_in], "outlet_desc": caps[i_desc]}
    for name, d in sorted(want.items(), key=lambda kv: kv[1]):
        best = min(branches, key=lambda i: abs(caps[i]["diameter"] - d))
        out[name] = caps[best]
        branches.remove(best)
    return out


def aorta_stl(path: str | Path, lod: str = "light", **params) -> dict[str, Any]:
    """Build ``aortic_arch`` (defaults or ``params``) and write the multi-solid STL; returns the caps (name ->
    centroid, outward normal, diameter, area) and a point inside the lumen."""
    from pinneapple_design import geometry_library as gl

    g = gl.build("aortic_arch", lod=lod, **params)
    body = g.bodies[0]
    V, F = np.asarray(body.vertices, float), np.asarray(body.faces, int)
    caps = find_caps(V, F)
    p = g.params
    if not p.get("bct_bifurcation", True):
        raise ValueError("aorta_stl expects the bifurcated brachiocephalic trunk (bct_bifurcation=True)")
    named = _label_aorta_caps(caps, p)
    used = np.zeros(len(F), bool)
    solids = {}
    for k, c in named.items():
        solids[k] = c["tris"]
        used[c["tris"]] = True
    solids["wall"] = np.flatnonzero(~used)
    lines = []
    for name, idx in solids.items():
        lines.append(f"solid {name}")
        for t in idx:
            a, b, c = V[F[t]]
            nn = np.cross(b - a, c - a)
            nn /= np.linalg.norm(nn) + 1e-300
            lines.append(f" facet normal {nn[0]:.9e} {nn[1]:.9e} {nn[2]:.9e}\n  outer loop")
            for v in (a, b, c):
                lines.append(f"   vertex {v[0]:.9e} {v[1]:.9e} {v[2]:.9e}")
            lines.append("  endloop\n endfacet")
        lines.append(f"endsolid {name}")
    Path(path).write_text("\n".join(lines) + "\n")
    inlet = named["inlet"]
    inside = inlet["centroid"] - 0.01 * inlet["normal"]          # 1 cm into the aorta from the annulus
    return {"caps": {k: {"centroid": c["centroid"].tolist(), "normal": c["normal"].tolist(),
                         "diameter": float(c["diameter"]), "area": c["area"]} for k, c in named.items()},
            "inside": inside.tolist(), "n_triangles": int(len(F)), "params": {k: v for k, v in p.items()}}


def _bashrc() -> str | None:
    from pinneapple_tools.automation.toolchain import Toolchain

    return Toolchain.discover().foam_bashrc


def _run(cmd: str, cwd: Path, log: str, timeout: float = 36000) -> None:
    rc = _bashrc()
    full = f"source {rc} >/dev/null 2>&1; {cmd}" if rc else cmd
    with open(cwd / log, "a") as fh:
        r = subprocess.run(["bash", "-c", full], cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd} failed in {cwd} (see {log})")


def setup_case(case: str | Path, *, cell_size: float = 1.5e-3, level: int = 1, inflow: Callable,
               period: float = 1.0, cycles: float = 2.0, dt: float = 5e-4, max_courant: float = 0.8,
               viscosity: str = "newtonian", blood: Mapping[str, float] = BLOOD, procs: int = 1,
               geometry: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Write and mesh the case; returns the geometry and mesh information (also saved to ``case/setup.json``).
    ``inflow(t)`` is the volumetric inflow (m^3/s), tabulated every 1 ms over ``cycles`` periods."""
    from pinneapple_simulation.pipelines.freecad_openfoam.case import Physics, write_internal_case
    from pinneapple_simulation.pipelines.freecad_openfoam.mesh import MeshSpec, write_internal_mesh

    case = Path(case)
    case.mkdir(parents=True, exist_ok=True)
    geo = aorta_stl(case / "aorta.stl", **(geometry or {}))
    kinds = {k: ("inlet" if k == "inlet" else "outlet") for k in geo["caps"]}
    kinds["wall"] = "wall"
    patches = {k: ("wall" if k == "wall" else "patch") for k in kinds}
    mesh = write_internal_mesh(case, case / "aorta.stl", patches, np.asarray(geo["inside"]), cell_size,
                               MeshSpec(level=level), procs=procs)
    rho = float(blood["rho"])
    nu = float(blood["mu"]) / rho
    phys = Physics(velocity=0.1, nu=nu, rho=rho, turbulence="laminar", solver="pimpleFoam",
                   end_time=period * cycles, dt=dt, max_courant=max_courant)
    normals = {k: np.asarray(v["normal"]) for k, v in geo["caps"].items() if k == "inlet"}
    write_internal_case(case, kinds, normals, phys, procs=procs)
    # inflow: volumetric flow rate table (plug profile)
    tt = np.arange(0.0, period * cycles + 1e-3, 1e-3)
    table = " ".join(f"({t:.4f} {float(inflow(t)):.9e})" for t in tt)
    Ufile = case / "0.orig" / "U"
    txt = Ufile.read_text()
    txt = re.sub(r"(\n    inlet\n    \{\n)(.*?)(    \})",
                 lambda m: m.group(1) + "        type flowRateInletVelocity;\n"
                 f"        volumetricFlowRate table ({table});\n        value uniform (0 0 0);\n" + m.group(3),
                 txt, count=1, flags=re.S)
    Ufile.write_text(txt)
    if viscosity == "carreau":
        c = CARREAU_CHO_KENSEY
        (case / "constant" / "transportProperties").write_text(
            (case / "constant" / "transportProperties").read_text().split("transportModel")[0]
            + "transportModel BirdCarreau;\nBirdCarreauCoeffs\n{\n"
            f"    nu0 [0 2 -1 0 0 0 0] {c['mu0'] / rho:.8g};\n    nuInf [0 2 -1 0 0 0 0] {c['mu_inf'] / rho:.8g};\n"
            f"    k [0 0 1 0 0 0 0] {c['lambda']:.8g};\n    n [0 0 0 0 0 0 0] {c['n']:.8g};\n}}\n")
    t0 = time.time()
    _run("./Allmesh", case, "log.Allmesh.out")
    _run("checkMesh -latestTime", case, "log.checkMesh")
    log = (case / "log.checkMesh").read_text()
    cells = int(re.search(r"cells:\s+(\d+)", log).group(1))
    ok = "Mesh OK" in log
    shutil.rmtree(case / "0", ignore_errors=True)
    shutil.copytree(case / "0.orig", case / "0")
    info = {"geometry": geo, "mesh": {**mesh, "cells": cells, "checkMesh_ok": ok, "seconds": time.time() - t0},
            "physics": {"rho": rho, "nu": nu, "viscosity": viscosity, "dt": dt, "max_courant": max_courant,
                        "period": period, "cycles": cycles}}
    (case / "setup.json").write_text(json.dumps(info, indent=1, default=float))
    return info


@dataclass
class CoupledRun:
    """Windkessel-coupled pimpleFoam run of a case written by :func:`setup_case`."""
    case: Path
    windkessels: dict[str, Windkessel]
    rho: float
    dt_couple: float = 0.01
    keep_every: float = 0.05
    history: list[dict[str, Any]] = field(default_factory=list)
    history_path: Path | None = None        # when set: written after every window, and read back to resume

    def __post_init__(self):
        self.case = Path(self.case)
        if self.history_path is not None:
            self.history_path = Path(self.history_path)
            if self.history_path.exists() and not self.history:
                self.history = json.loads(self.history_path.read_text())
                last = self.history[-1]
                for k, w in self.windkessels.items():   # restore each Windkessel from the last window
                    w.Pc = last[f"P_wk_{k}"] - w.Rp * last[f"Q_{k}"]
                # drop time folders written after the last recorded window (a run killed mid-window)
                for tt in self._times():
                    if tt > last["t"] + 1e-9:
                        shutil.rmtree(self.case / self._name(tt), ignore_errors=True)
            elif not self.history:
                for tt in self._times():               # a fresh start: only the initial time
                    if tt > 0:
                        shutil.rmtree(self.case / self._name(tt), ignore_errors=True)

    def _times(self) -> list[float]:
        out = []
        for d in self.case.iterdir():
            try:
                out.append(float(d.name))
            except ValueError:
                pass
        return sorted(out)

    @staticmethod
    def _name(t: float) -> str:
        return f"{t:.8g}"

    def _set_outlet_pressures(self, tdir: Path, P: Mapping[str, float]) -> None:
        f = tdir / "p"
        txt = f.read_text()
        for name, val in P.items():
            kin = val / self.rho
            block = (f"\n    {name}\n    {{\n        type            fixedValue;\n"
                     f"        value           uniform {kin:.10g};\n    }}")
            txt, n = re.subn(rf"\n    {name}\n    \{{.*?\n    \}}", lambda _m, b=block: b, txt, count=1, flags=re.S)
            if n != 1:
                raise RuntimeError(f"outlet {name} not found in {f}")
        f.write_text(txt)

    def _control(self, end: float) -> None:
        f = self.case / "system" / "controlDict"
        txt = f.read_text()
        txt = re.sub(r"^startFrom .*;", "startFrom latestTime;", txt, flags=re.M)
        txt = re.sub(r"^endTime .*;", f"endTime {end:.10g};", txt, flags=re.M)
        txt = re.sub(r"^writeInterval .*;", f"writeInterval {self.dt_couple:.10g};", txt, flags=re.M)
        f.write_text(txt)

    def run(self, t_end: float, verbose: bool = False) -> list[dict[str, Any]]:
        from pinneapple_simulation.pipelines.freecad_openfoam.post import CaseResult

        times = self._times()
        t = times[-1]
        if t == 0 and not self.history:
            self._set_outlet_pressures(self.case / "0", {k: w.pressure(0.0) for k, w in self.windkessels.items()})
        start = time.time()
        while t < t_end - 1e-9:
            t_next = round(t + self.dt_couple, 10)
            self._control(t_next)
            _run("pimpleFoam", self.case, "log.pimpleFoam")
            last = self._times()[-1]
            if abs(last - t_next) > 1e-6:
                raise RuntimeError(f"pimpleFoam stopped at {last}, expected {t_next}")
            res = CaseResult(self.case)
            rec = {"t": t_next, "seconds": time.time() - start, "cells": res.n_cells}
            P_new = {}
            for k, w in self.windkessels.items():
                pat = res.patch(k)
                Q = pat["flux_m3_s"]
                rec[f"Q_{k}"] = Q
                rec[f"p3d_{k}"] = pat["p_mean"] * self.rho
                P_new[k] = w.step(Q, self.dt_couple)
            inl = res.patch("inlet")
            rec["Q_inlet"] = -inl["flux_m3_s"]
            rec["p_inlet"] = inl["p_mean"] * self.rho
            rec["mass_imbalance"] = abs(sum(rec[f"Q_{k}"] for k in self.windkessels) - rec["Q_inlet"]) \
                / max(abs(rec["Q_inlet"]), 1e-12)
            self._set_outlet_pressures(self.case / self._name(last), P_new)
            for k, v in P_new.items():
                rec[f"P_wk_{k}"] = v
            self.history.append(rec)
            if self.history_path is not None:
                tmp = self.history_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(self.history))
                tmp.replace(self.history_path)
            if verbose:
                print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in rec.items()
                       if k in ("t", "Q_inlet", "p_inlet", "mass_imbalance", "seconds")}, flush=True)
            # keep disk use bounded: drop time folders not on the keep_every grid (and never the latest)
            for tt in self._times()[:-1]:
                if tt > 0 and abs(tt / self.keep_every - round(tt / self.keep_every)) > 1e-6:
                    shutil.rmtree(self.case / self._name(tt), ignore_errors=True)
            t = t_next
        return self.history

    def check(self, period: float) -> dict[str, Any]:
        """Over the last full period: mass imbalance, cycle-mean flows and pressures, and how far each outlet's 3-D
        pressure is from its Windkessel (staggering of the explicit coupling)."""
        h = [r for r in self.history if r["t"] > self.history[-1]["t"] - period + 1e-9]
        out = {"window": [h[0]["t"], h[-1]["t"]], "max_mass_imbalance": max(r["mass_imbalance"] for r in h),
               "Q_inlet_mean": float(np.mean([r["Q_inlet"] for r in h])),
               "p_inlet_mean": float(np.mean([r["p_inlet"] for r in h])), "outlets": {}}
        for k in self.windkessels:
            Q = np.array([r[f"Q_{k}"] for r in h])
            p3 = np.array([r[f"p3d_{k}"] for r in h])
            pw = np.array([r[f"P_wk_{k}"] for r in h])
            out["outlets"][k] = {"Q_mean": float(Q.mean()), "flow_fraction": float(Q.mean() / out["Q_inlet_mean"]),
                                 "p_mean": float(p3.mean()), "p_min": float(p3.min()), "p_max": float(p3.max()),
                                 "max_rel_diff_3d_vs_wk_next": float(np.max(np.abs(p3[1:] - pw[:-1]))
                                                                     / max(np.abs(pw).max(), 1e-12))}
        return out

"""Two industrial parts given only as STL, solved by the immersed method and by conforming FEM from the same file:
a finned heat sink under a chip (steady heat with convection) and an L-bracket bolted at two holes and loaded at the
end of its arm (linear elasticity, von Mises). The STL is written by gmsh from an OpenCASCADE model, so the
facets are the only geometry either method sees; the conforming path has to turn them into a volume mesh
(``tet_fem.occ_stl``) and the immersed path reads them directly (``STLGeometry``).

    from pinneapple_simulation.immersed import parts
    path = parts.write_stl("heat_sink", "hs.stl")
    case = parts.heat_sink_case(path)       # geometry, conductivity, load, conditions
"""
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .bcs import Convection, Displacement, Force, HeatInput

# heat sink: aluminium base 40 x 40 x 4 mm, 7 fins 1.5 mm thick and 20 mm high along y; a 15 x 15 mm chip
# dissipating 20 W under the centre; h = 25 W/(m^2 K) to air at 25 C on every other surface
HS = dict(L=0.04, W=0.04, base=0.004, fins=7, t=0.0015, H=0.02, chip=0.015, P=20.0, k=205.0, h=25.0, T_inf=25.0)
# bracket: steel L, two flanges 60 x 40 x 6 mm, inner fillet r = 4 mm, two 6 mm bolt holes in the upright flange
# (clamped), 500 N pulling down on the end face of the arm
BR = dict(L=0.06, W=0.04, t=0.006, fillet=0.004, hole=0.003, hz=0.04, hy=(0.012, 0.028), F=500.0, E=210e9, nu=0.3)


def occ_heat_sink(p: dict = HS) -> Callable:
    def build(g):
        occ = g.model.occ
        base = occ.addBox(-p["L"] / 2, -p["W"] / 2, 0, p["L"], p["W"], p["base"])
        pitch = (p["L"] - p["t"]) / (p["fins"] - 1)
        fins = [occ.addBox(-p["L"] / 2 + i * pitch, -p["W"] / 2, p["base"], p["t"], p["W"], p["H"])
                for i in range(p["fins"])]
        occ.fuse([(3, base)], [(3, f) for f in fins])
        occ.synchronize()
    return build


def occ_bracket(p: dict = BR) -> Callable:
    def build(g):
        occ = g.model.occ
        up = occ.addBox(0, 0, 0, p["t"], p["W"], p["L"])
        arm = occ.addBox(0, 0, 0, p["L"], p["W"], p["t"])
        (body,), _ = occ.fuse([(3, up)], [(3, arm)])
        occ.synchronize()
        # the inner edge along y at x = z = t
        edges = [e for _, e in g.model.getEntities(1)
                 if np.allclose(occ.getCenterOfMass(1, e), (p["t"], p["W"] / 2, p["t"]), atol=1e-9)]
        out = occ.fillet([body[1]], edges, [p["fillet"]])
        occ.synchronize()
        holes = [occ.addCylinder(-0.001, y, p["hz"], p["t"] + 0.002, 0, 0, p["hole"]) for y in p["hy"]]
        occ.cut(out, [(3, h) for h in holes])
        occ.synchronize()
    return build


def write_stl(name: str, path: str, size: float | None = None) -> str:
    """Triangulate the part's CAD with gmsh and write the STL (the input of both methods)."""
    import gmsh
    build = {"heat_sink": occ_heat_sink(), "bracket": occ_bracket()}[name]
    size = size or {"heat_sink": 0.002, "bracket": 0.0015}[name]
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(name)
        build(gmsh)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.model.mesh.generate(2)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        gmsh.write(path)
    finally:
        gmsh.finalize()
    return path


@dataclass
class HeatPart:
    name: str
    stl: str
    conductivity: float
    bcs: list
    source: float = 0.0


@dataclass
class ElasticPart:
    name: str
    stl: str
    E: float
    nu: float
    bcs: list


def heat_sink_case(stl: str, p: dict = HS) -> HeatPart:
    c = p["chip"] / 2
    chip = lambda X, n: (X[:, 2] < 1e-4) & (n[:, 2] < -0.5) & (np.abs(X[:, 0]) < c) & (np.abs(X[:, 1]) < c)  # noqa: E731
    return HeatPart("heat_sink", stl, p["k"], [HeatInput(p["P"], where=chip), Convection(p["h"], p["T_inf"])])


def bracket_case(stl: str, p: dict = BR) -> ElasticPart:
    def bolt(X, n):
        r = np.min([np.hypot(X[:, 1] - y, X[:, 2] - p["hz"]) for y in p["hy"]], axis=0)
        return (r < p["hole"] * 1.05) & (X[:, 0] < p["t"] + 1e-5) & (np.abs(n[:, 0]) < 0.5)
    end = lambda X, n: (X[:, 0] > p["L"] - 0.25 * p["t"]) & (n[:, 0] > 0.5)               # noqa: E731
    return ElasticPart("bracket", stl, p["E"], p["nu"], [Displacement(0.0, where=bolt),
                                                         Force((0.0, 0.0, -p["F"]), where=end)])


def sample_points(stl_geometry, n: int = 4000, seed: int = 0, inset: float = 0.0) -> np.ndarray:
    """Points inside the part (at least ``inset`` from its surface) to compare fields between methods."""
    rng = np.random.default_rng(seed)
    lo, hi = stl_geometry.bounds
    out = []
    while sum(len(o) for o in out) < n:
        X = rng.uniform(lo, hi, (4 * n, 3))
        out.append(X[stl_geometry.phi(X) < -inset])
    return np.concatenate(out)[:n]


# ---------------------------------------------------------------- the comparison (checkpointed)

PROBES = {"heat_sink": np.array([[0.0, 0.0, 2e-4]]),                                # chip centre, base underside
          "bracket": np.array([[BR["L"] - 5e-4, BR["W"] / 2, BR["t"] / 2]])}       # end of the arm


def _solve(name: str, method: str, level: float, stl: str):
    import time

    from pinneapple_simulation.numerical_solvers import tet_fem as T

    from . import STLGeometry
    from .assembly import Discretization
    from .elasticity import solve_elasticity
    from .heat import solve_heat
    case = heat_sink_case(stl) if name == "heat_sink" else bracket_case(stl)
    order = int(method[-1])
    t0 = time.perf_counter()
    if method.startswith("immersed"):
        geo = STLGeometry.from_file(stl)
        disc = Discretization(geo, level, order=order)
        if name == "heat_sink":
            sol = solve_heat(disc, case.conductivity, 0.0, case.bcs)
        else:
            sol = solve_elasticity(disc, case.E, case.nu, None, case.bcs)
        geo_s = disc.timings["quadrature_s"]
        dofs = disc.n_dofs * (1 if name == "heat_sink" else 3)
    else:
        mesh = T.gmsh_mesh(T.occ_stl(stl), level, order, high_order_optimize=False)
        if name == "heat_sink":
            sol = T.solve_heat(mesh, case.conductivity, 0.0, case.bcs)
        else:
            sol = T.solve_elasticity(mesh, case.E, case.nu, None, case.bcs)
        geo_s = mesh.meshing_s
        dofs = sol.info["dofs"]
    total = time.perf_counter() - t0
    row = {"part": name, "method": method, "level": level, "dofs": int(dofs), "geometry_s": geo_s,
           "assembly_s": sol.info["assembly_s"], "solve_s": sol.info["solve_seconds"],
           "compute_s": geo_s + sol.info["assembly_s"] + sol.info["solve_seconds"], "total_s": total,
           "balance": float(sol.balance["imbalance"])}
    return row, sol


def _fields(name: str, sol, X: np.ndarray) -> dict:
    if name == "heat_sink":
        return {"T": np.asarray(sol(X)).ravel()}
    return {"u": np.asarray(sol(X)).reshape(len(X), 3), "vm": np.asarray(sol.von_mises(X)).ravel()}


def study(name: str, plan: dict, reference: tuple[str, float], out_dir: str, n_samples: int = 3000,
          inset: float = 3e-4, log=print) -> list[dict]:
    """Run ``plan`` ({method: [levels]}) on the part, compare every run with the reference run at ``n_samples``
    interior points (at least ``inset`` inside), and append each row to ``out_dir/<name>.jsonl`` (already
    finished rows are skipped, so the study resumes after an interruption)."""
    import json

    from . import STLGeometry
    os.makedirs(out_dir, exist_ok=True)
    stl = os.path.join(out_dir, f"{name}.stl")
    if not os.path.exists(stl):
        write_stl(name, stl)
    rows_path = os.path.join(out_dir, f"{name}.jsonl")
    done = {}
    if os.path.exists(rows_path):
        for line in open(rows_path):
            r = json.loads(line)
            done[(r["method"], r["level"])] = r
    X = sample_points(STLGeometry.from_file(stl), n_samples, inset=inset)
    Xp = PROBES[name]
    ref_npz = os.path.join(out_dir, f"{name}_reference.npz")
    if os.path.exists(ref_npz):
        ref = dict(np.load(ref_npz))
    else:
        row, sol = _solve(name, reference[0], reference[1], stl)
        ref = {k: v for k, v in _fields(name, sol, X).items()}
        ref.update({"probe_" + k: v for k, v in _fields(name, sol, Xp).items()})
        np.savez(ref_npz, **ref)
        row["reference"] = True
        with open(rows_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        log(row)
    rows = []
    for method, levels in plan.items():
        for lv in levels:
            if (method, lv) in done:
                rows.append(done[(method, lv)])
                continue
            row, sol = _solve(name, method, lv, stl)
            fv = _fields(name, sol, X)
            pv = _fields(name, sol, Xp)
            if name == "heat_sink":
                T_inf = HS["T_inf"]
                row["T_L2_rel"] = float(np.sqrt(np.nanmean((fv["T"] - ref["T"]) ** 2) / np.mean((ref["T"] - T_inf) ** 2)))
                row["T_chip"] = float(pv["T"][0])
                row["T_chip_err"] = float((pv["T"][0] - ref["probe_T"][0]) / (ref["probe_T"][0] - T_inf))
            else:
                du = fv["u"] - ref["u"]
                row["u_L2_rel"] = float(np.sqrt(np.nanmean((du ** 2).sum(1)) / np.mean((ref["u"] ** 2).sum(1))))
                row["vm_max"] = float(np.nanmax(fv["vm"]))
                row["vm_max_err"] = float(np.nanmax(fv["vm"]) / np.max(ref["vm"]) - 1)
                row["vm_L2_rel"] = float(np.sqrt(np.nanmean((fv["vm"] - ref["vm"]) ** 2) / np.mean(ref["vm"] ** 2)))
                row["tip_uz"] = float(pv["u"][0, 2])
                row["tip_err"] = float(pv["u"][0, 2] / ref["probe_u"][0, 2] - 1)
            row["nan_samples"] = int(np.isnan(next(iter(fv.values()))).sum())
            with open(rows_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            log(row)
            rows.append(row)
    return rows

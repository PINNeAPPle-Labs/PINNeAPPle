"""The three pipeline actions that are not plain OpenFOAM commands, as functions of their parameters and a step
folder. The automation adapters (``freecadcmd.export``, ``openfoam.prepare``, ``openfoam.post`` in
:mod:`pinneapple_tools.automation.adapters.pipeline`) wrap them; each returns ``{"outputs", "artifacts", "checks",
"actions", "figures"}`` with checks as ``(name, passed, detail, value, threshold, source)``.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .cad import CadGeometry, PatchSpec, assign_patches, get_backend, write_stl
from .case import Physics, write_internal_case
from .mesh import MeshSpec, allmesh_script, allrun_script, write_internal_mesh
from .pipeline import evaluate, resolve_physics
from .post import CaseResult, Section, geometry_figure, mesh_figure, plane_figure, read_case

__all__ = ["cad_step", "prepare_step", "post_step", "point_inside", "auto_procs"]


def _check(name, passed, detail, value=None, threshold=None, source=""):
    return (name, bool(passed), detail, value, threshold, source)


def point_inside(V: np.ndarray, F: np.ndarray, p: np.ndarray) -> bool:
    """Whether ``p`` is inside the closed triangle mesh (parity of crossings along three skew rays, majority)."""
    votes = 0
    for d in (np.array([0.5773, 0.5774, 0.5775]), np.array([-0.8017, 0.2673, 0.5345]), np.array([0.1, -0.9, 0.42])):
        d = d / np.linalg.norm(d)
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        e1, e2 = b - a, c - a
        h = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, h)
        ok = np.abs(det) > 1e-300
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = p - a
        u = inv * np.einsum("ij,ij->i", s, h)
        q = np.cross(s, e1)
        v = inv * (q @ d)
        t = inv * np.einsum("ij,ij->i", e2, q)
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
        votes += int(hit.sum() % 2 == 1)
    return votes >= 2


def auto_procs(cells: float, cores: int | None = None, per_proc: float = 20000.0) -> int:
    """Processes for a mesh of about ``cells`` cells: one per ``per_proc`` cells, at most the cores here."""
    cores = cores or os.cpu_count() or 1
    return int(max(1, min(cores, round(cells / per_proc))))


# ----------------------------------------------------------------------------------------------- cad
def cad_step(params: dict[str, Any], step_dir: Path) -> dict[str, Any]:
    """Set the parameters, recompute, export, assign patches, write ``geometry.stl`` (m) and ``geometry.png``."""
    from .models import MODELS, build_document
    step_dir = Path(step_dir)
    step_dir.mkdir(parents=True, exist_ok=True)
    backend = get_backend(params.get("backend", "auto"))
    checks, actions = [], []
    doc = params.get("document") or ""
    tpl = params.get("template") or ""
    if not doc:
        if not tpl:
            raise ValueError("give a document or a template")
        ext = ".FCStd" if backend.name == "freecad" else ".fake.json"
        doc = str(step_dir / f"{tpl}{ext}")
        build_document(tpl, doc, backend=backend.name)
        actions.append(f"built the parametric document of template {tpl!r} with {backend.name} ({Path(doc).name})")
    parameters = {k: float(v) for k, v in (params.get("parameters") or {}).items()}
    solid = params.get("solid") or (MODELS[tpl].solid if tpl else "")
    geom: CadGeometry = backend.export(doc, parameters, solid, step_dir, float(params.get("tolerance_mm", 0.05)))
    info = geom.info
    actions.append(f"{backend.name}: set {', '.join(f'{k} = {v:g}' for k, v in parameters.items())}, recomputed, "
                   f"exported {solid} ({info.get('faces')} faces)")
    checks.append(_check("document recomputes without invalid objects", not info.get("invalid_objects"),
                         "invalid: " + ", ".join(info["invalid_objects"]) if info.get("invalid_objects") else
                         f"{info.get('backend')} {info.get('version')}", source="FreeCAD Document.recompute"))
    checks.append(_check(f"solid {solid} is valid and closed", info.get("valid") and info.get("closed") and
                         info.get("solids") == 1, f"valid {info.get('valid')}, closed {info.get('closed')}, "
                         f"{info.get('solids')} solid(s)", source="Part.Shape.isValid / Shell.isClosed"))
    open_edges = geom.open_edges()
    checks.append(_check("triangulation is watertight", open_edges == 0, f"{open_edges} open edges",
                         value=open_edges, threshold=0, source="pinneapple_simulation.pipelines.freecad_openfoam.cad"))
    vt, vc = geom.volume(), float(info.get("volume_mm3") or 0)
    rel = abs(vt - vc) / vc if vc else float("inf")
    checks.append(_check("triangulated volume matches the CAD volume", rel <= 0.02,
                         f"{vt:.6g} mm³ triangulated, {vc:.6g} mm³ CAD ({100 * rel:.2f} %)", value=rel, threshold=0.02,
                         source="divergence theorem on the exported triangles"))
    specs = PatchSpec.parse(params.get("patches") or (MODELS[tpl].patches if tpl else {}))
    assignment = assign_patches(geom, specs, info.get("parameters"))
    kinds = {p.name: p.kind for p in specs}
    stl = write_stl(step_dir / "geometry.stl", geom, assignment, scale=1e-3)
    areas = {k: float(sum(geom.faces[i].area for i in v)) for k, v in assignment.items()}
    normals = {}
    for p in specs:
        if p.kind == "inlet":
            n = sum(geom.faces[i].normal * geom.faces[i].area for i in assignment[p.name])
            normals[p.name] = (n / np.linalg.norm(n)).tolist()
    actions.append("patches by rule: " + "; ".join(f"{p.name} ({p.kind}) = faces {assignment[p.name]} "
                                                  f"[{p.where}]" for p in specs))
    out: dict[str, Any] = {"stl": str(stl), "kinds": kinds, "inlet_normals": normals, "patch_areas_mm2": areas,
                           "volume_mm3": vc or vt, "parameters": info.get("parameters"), "backend": info.get("backend"),
                           "freecad_version": info.get("version"), "document": info.get("document"),
                           "step": info.get("step"), "faces": info.get("faces"),
                           "bound_box_mm": [*map(float, geom.bounds()[0]), *map(float, geom.bounds()[1])]}
    inl = [k for k, v in kinds.items() if v == "inlet"]
    outl = [k for k, v in kinds.items() if v == "outlet"]
    out["inlet"], out["outlet"] = (inl[0] if inl else ""), (outl[0] if outl else "")
    if params.get("flow", "internal") == "internal":
        if not inl or not outl:
            raise ValueError(f"an internal flow needs an inlet and an outlet patch; kinds are {kinds}")
        f0 = max((geom.faces[i] for i in assignment[inl[0]]), key=lambda f: f.area)
        V, F = geom.mesh()
        n = np.asarray(normals[inl[0]])
        depth = 0.35 * math.sqrt(f0.area)
        a = np.cross(n, [0.3, 0.5, 0.81])
        a /= np.linalg.norm(a)
        p = f0.centre - depth * n + 0.0371 * math.sqrt(f0.area) * a
        ins = point_inside(V, F, p)
        checks.append(_check("location in the mesh is inside the fluid", ins,
                             f"{np.round(p, 4).tolist()} mm, {depth:.3g} mm in from {inl[0]}",
                             source="ray parity on the exported triangles"))
        out["inside_m"] = (p * 1e-3).tolist()
    else:
        out["inside_m"] = None
    png = geometry_figure(geom, assignment, step_dir / "geometry.png",
                          f"{tpl or Path(doc).stem}: " + ", ".join(f"{k} = {v:g}" for k, v in
                                                                   (info.get("parameters") or {}).items()))
    arts = [stl, png] + [Path(x) for x in (info.get("step"), info.get("document")) if x]
    if backend.name == "fake":
        actions.append("FakeFreeCAD: NumPy stand-in, no STEP file written")
    return {"outputs": out, "artifacts": arts, "checks": checks, "actions": actions,
            "figures": [(png, "render", "exported solid, faces coloured by patch")]}


# ----------------------------------------------------------------------------------------------- case
_EXTERNAL_KEYS = ("alpha", "beta", "half_model", "ground", "resolution", "surface_level", "iterations", "n_lines",
                  "wake_planes", "turbulence", "domain", "ref_area", "ref_length", "cells_per_length")


def prepare_step(params: dict[str, Any], step_dir: Path) -> dict[str, Any]:
    """Mesh dictionaries + case + ``Allmesh`` / ``Allrun`` in ``step_dir/case``."""
    step_dir = Path(step_dir)
    case = step_dir / "case"
    if case.exists():
        shutil.rmtree(case)
    inputs = {k: float(v) for k, v in (params.get("inputs") or {}).items()}
    phys = resolve_physics(params.get("physics") or {}, inputs)
    actions, checks = [], []
    procs_req = params.get("procs", "auto")
    stl = Path(params["stl"])
    if params["flow"] == "internal":
        ms = MeshSpec(**(params.get("mesh") or {}))
        h = ms.cell_size_m(inputs)
        from .mesh import _stl_points
        pts = _stl_points(stl)
        size = pts.max(0) - pts.min(0)
        V, F = _stl_mesh(stl)
        vol = abs(float(np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6))
        cells = vol / h ** 3
        procs = auto_procs(cells) if procs_req == "auto" else int(procs_req)
        kinds = dict(params["kinds"])
        types = {k: {"inlet": "patch", "outlet": "patch", "wall": "wall", "symmetry": "symmetry"}[v]
                 for k, v in kinds.items()}
        mi = write_internal_mesh(case, stl, types, np.asarray(params["inside"], float), h, ms, procs)
        physics = Physics(**phys)
        if physics.solver == "pimpleFoam":                   # first step within the Courant limit at twice the
            dt0 = 0.5 * physics.max_courant * h / (2.0 * physics.velocity)   # inlet speed; adjustTimeStep then
            if dt0 < physics.dt:
                actions.append(f"initial time step {physics.dt:g} s -> {dt0:.3g} s (Courant {physics.max_courant:g})")
                physics.dt = dt0
        write_internal_case(case, kinds, {k: np.asarray(v) for k, v in params["inlet_normals"].items()},
                                 physics, procs, size)
        actions.append(f"{mi['mesher']} mesh: wall cells {1e3 * h:.3g} mm (background {1e3 * mi['background_cell_m']:.3g}"
                       f" mm, {mi['background_cells']}), about {cells:.3g} cells, {procs} process(es)")
        actions.append(f"{physics.solver}, {physics.turbulence}, inlet {physics.velocity:g} m/s, nu {physics.nu:g} m²/s"
                       + (f", scalar {physics.scalar.name} (D {physics.scalar.D:g} m²/s)" if physics.scalar else ""))
        out = {"case": str(case), "procs": procs, "physics": physics.to_dict(), "estimated_cells": cells,
               "wall_cell_m": h, "mesher": mi["mesher"], "residual_tol": max(physics.residual_tol, 1e-12),
               "max_courant_allowed": max(1.0, 1.05 * physics.max_courant)}   # adjustTimeStep lags: the
        #                       target is max_courant, the check is the PIMPLE (one outer corrector) limit Co <= 1
    else:
        from pinneapple_simulation.numerical_solvers.external_flow import ExternalFlow
        opts = {k: phys[k] for k in _EXTERNAL_KEYS if k in phys}
        if "domain" in opts:
            opts["domain"] = tuple(opts["domain"])
        if "surface_level" in opts:
            opts["surface_level"] = tuple(int(v) for v in opts["surface_level"])
        flow = ExternalFlow(str(stl), speed=float(phys["speed"]), nu=float(phys["nu"]), rho=float(phys.get("rho", 1.2)),
                            title=params.get("title", "External flow"), **opts)
        procs = (os.cpu_count() or 1) if procs_req == "auto" else int(procs_req)
        info = flow.write(str(case), procs)
        (case / "0").rename(case / "0.orig")
        for name, text in (("Allmesh", allmesh_script(procs)), ("Allrun", allrun_script("simpleFoam", procs))):
            (case / name).write_text(text)
            (case / name).chmod(0o755)
        actions.append(f"ExternalFlow case: {info['bodies']}, speed {flow.speed:g} m/s, Re {info['Re']:.4g}, "
                       f"{opts.get('turbulence', 'kOmegaSST')}, {procs} process(es)")
        out = {"case": str(case), "procs": procs, "physics": {**phys, **{k: info[k] for k in ("Re", "ref_area",
                                                                                         "ref_length", "bodies")}},
               "residual_tol": float(phys.get("residual_tol", 1e-5)), "max_courant_allowed": 1e9, "Re": info["Re"]}
        _set_residual_control(case, float(phys.get("residual_tol", 1e-5)), opts.get("turbulence", "kOmegaSST"))
    return {"outputs": out, "artifacts": [case], "checks": checks, "actions": actions, "figures": []}


def _stl_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    from .mesh import _stl_points
    P = _stl_points(path)
    return P, np.arange(len(P)).reshape(-1, 3)


def _set_residual_control(case: Path, tol: float, turbulence: str) -> None:
    fs = case / "system" / "fvSolution"
    fields = ["U", "p"] + ([] if turbulence == "laminar" else ["k", "omega"])
    rc = "    residualControl\n    {\n" + "".join(f"        {f} {tol:g};\n" for f in fields) + "    }\n"
    text = fs.read_text()
    fs.write_text(re.sub(r"(SIMPLE\s*\{)", lambda m: m.group(1) + "\n" + rc, text, count=1))


# ----------------------------------------------------------------------------------------------- post
def _names(inputs: dict[str, float], physics: dict[str, Any]) -> dict[str, Any]:
    names = dict(inputs)
    for k, v in physics.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            names[k] = float(v)
    if isinstance(physics.get("scalar"), dict):
        names["D_scalar"] = float(physics["scalar"]["D"])
    return names


def post_step(params: dict[str, Any], step_dir: Path) -> dict[str, Any]:
    """Measured quantities, figures, derived quantities and reference errors of a finished case."""
    from pinneapple_data.simulation_metadata.openfoam import parse_log
    step_dir = Path(step_dir)
    case = Path(params["case"])
    phys = params.get("physics") or {}
    inputs = {k: float(v) for k, v in (params.get("inputs") or {}).items()}
    names = _names(inputs, phys)
    checks, actions, figures = [], [], []
    rho = float(phys.get("rho", 1.0))
    measured: dict[str, Any] = {}
    scalar = (phys.get("scalar") or {}).get("name") if isinstance(phys.get("scalar"), dict) else None
    plane_o = np.asarray(evaluate(params.get("plane", {}).get("origin", [0, 0, 0]), names), float) * 1e-3
    plane_n = np.asarray(evaluate(params.get("plane", {}).get("normal", [0, 0, 1]), names), float)
    if params["flow"] == "internal":
        secs = [Section(s["name"], tuple(np.asarray(evaluate(s["origin"], names), float) * 1e-3),
                        tuple(np.asarray(evaluate(s["normal"], names), float)), float(evaluate(s["radius"], names)) * 1e-3)
                for s in params.get("sections") or []]
        r = read_case(case, dict(params["kinds"]), rho, secs, scalar, float(phys.get("nu", 1e-6)))
        res: CaseResult = r.pop("_result")
        for k in ("cells", "dp_static_pa", "dp_total_pa", "inflow_m3_s", "mass_imbalance", "inlet_velocity_mean",
                  "scalar_in", "scalar_out"):
            if k in r:
                measured[k] = r[k]
        for p, d in r["patches"].items():
            measured[f"area_{p}"] = d["area_m2"]
            measured[f"p_{p}"] = d["p_mean_pa"]
            measured[f"q_{p}"] = d["flux_m3_s"]
            if "scalar_bulk" in d:
                measured[f"s_{p}"] = d["scalar_bulk"]
        for s, d in r["sections"].items():
            measured[f"p_{s}"] = d["p_mean_pa"]
            measured[f"q_{s}"] = d["flux_m3_s"]
            measured[f"area_{s}"] = d["area_m2"]
            measured[f"inside_{s}"] = d["fraction_inside"]
            if "scalar_bulk" in d:
                measured[f"s_{s}"] = d["scalar_bulk"]
        if r.get("forces_n"):
            for j, ax in enumerate("xyz"):
                measured[f"F{ax}"] = r["forces_n"]["total"][j]
        for p, y in r["yplus"].items():
            measured[f"yplus_max_{p}"] = y["max"]
            measured[f"yplus_mean_{p}"] = y["average"]
        checks.append(_check("mass balance", r["mass_imbalance"] <= 0.01,
                             f"|net boundary flux| / inflow = {r['mass_imbalance']:.3g} (inflow {r['inflow_m3_s']:.6g} m³/s)",
                             value=r["mass_imbalance"], threshold=0.01, source="pinneapple_check.fv (phi)"))
        for s, d in r["sections"].items():
            checks.append(_check(f"section {s} lies in the fluid", d["fraction_inside"] >= 0.95,
                                 f"{100 * d['fraction_inside']:.1f} % of its disc inside, flux {d['flux_m3_s']:.4g} m³/s "
                                 f"({100 * d['flux_m3_s'] / r['inflow_m3_s']:.1f} % of the inflow)",
                                 value=d["fraction_inside"], threshold=0.95, source="post.section_average"))
    else:
        from pinneapple_simulation.numerical_solvers.external_flow import ExternalFlow
        info = json.loads((case / "pinneapple_flow.json").read_text())
        flow = ExternalFlow(str(params["stl"]), speed=info["speed"], nu=info["nu"], rho=info["rho"],
                            half_model=info["half_model"], ref_area=info["ref_area"], ref_length=info["ref_length"],
                            n_lines=12, wake_planes=(0.5,))
        fr = flow.read(str(case))
        c = fr.coefficients
        for k in ("CD", "CL", "CD_pressure", "CD_friction", "drag_N", "lift_N", "cells"):
            measured[k] = c[k]
        from .post import wall_loads
        for p, y in wall_loads(case, info["bodies"], info["nu"], info["rho"])["yplus"].items():
            measured[f"yplus_max_{p}"] = y["max"]
        measured["Re"] = info["Re"]
        res = CaseResult(case)
    log = next((p for p in sorted(case.glob("log.*Foam"))), None)
    if log is not None:
        pl = parse_log(log.read_text(errors="replace"))
        if pl.get("courant_max"):
            measured["courant_max"] = float(max(pl["courant_max"]))
        measured["iterations"] = len(pl.get("time_steps") or [])
    names.update({k: v for k, v in measured.items() if isinstance(v, (int, float))})
    for k, e in (params.get("derived") or {}).items():
        try:
            names[k] = measured[k] = float(evaluate(e, names))
        except Exception as exc:                               # a derived quantity that cannot be computed fails
            checks.append(_check(f"derived {k}", False, f"{e}: {type(exc).__name__}: {exc}"))
    refs = {}
    for k, ref in (params.get("references") or {}).items():
        try:
            m, rv = float(evaluate(ref["measured"], names)), float(evaluate(ref["reference"], names))
        except Exception as exc:
            checks.append(_check(f"reference {k}", False, f"could not be evaluated: {type(exc).__name__}: {exc}"))
            continue
        err = abs(m - rv) / abs(rv) if rv else float("inf")
        measured[f"ref_{k}_measured"], measured[f"ref_{k}_reference"], measured[f"ref_{k}_error"] = m, rv, err
        refs[k] = {"measured": m, "reference": rv, "error": err, "tol": float(ref.get("tol", 0.05)),
                   "source": ref.get("source", ""), "expressions": {"measured": ref["measured"],
                                                                     "reference": ref["reference"]}}
        actions.append(f"{k}: measured {m:.5g}, reference {rv:.5g} ({ref.get('source', '')}), error {100 * err:.2f} %")
    title = params.get("title", "")
    lab = f"{scalar} [-]" if scalar else None
    figures.append((plane_figure(res, step_dir / "fields.png", plane_o, plane_n, f"{title}: fields on the plane", rho,
                                 lab), "plot", "speed, pressure (and scalar) sampled on the figure plane"))
    figures.append((mesh_figure(res, step_dir / "mesh.png", plane_o, plane_n, "Mesh"), "plot",
                    "cells cut by the figure plane"))
    (step_dir / "measured.json").write_text(json.dumps({"measured": measured, "references": refs,
                                                        "inputs": inputs, "physics": phys}, indent=1, default=float))
    outputs = {k: v for k, v in measured.items() if isinstance(v, (int, float))}
    outputs["references"] = refs
    return {"outputs": outputs, "artifacts": [step_dir / "measured.json"] + [f[0] for f in figures], "checks": checks,
            "actions": actions, "figures": figures}

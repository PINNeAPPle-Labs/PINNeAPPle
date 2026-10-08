"""Pre-flight rules for OpenFOAM cases: mesh, transport/thermo properties, boundary and initial conditions, schemes,
linear solvers and time stepping, checked before the solver runs.

Every rule cites what OpenFOAM does when it is violated; the FAIL rules were confirmed by running OpenFOAM v1912 on
broken copies of tutorial cases (the error text is in ``evidence``).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pinneapple_data.simulation_metadata.foam_dict import parse

from .common import Report, line_of, num, vec

SOLVERS = {
    # name: (kind, algorithm, base fields, uses turbulence, properties file)
    "simpleFoam": ("incompressible", "SIMPLE", ["U", "p"], True, "transportProperties"),
    "pimpleFoam": ("incompressible", "PIMPLE", ["U", "p"], True, "transportProperties"),
    "pisoFoam": ("incompressible", "PISO", ["U", "p"], True, "transportProperties"),
    "icoFoam": ("incompressible", "PISO", ["U", "p"], False, "transportProperties"),
    "nonNewtonianIcoFoam": ("incompressible", "PISO", ["U", "p"], False, "transportProperties"),
    "SRFSimpleFoam": ("incompressible", "SIMPLE", ["Urel", "p"], True, "transportProperties"),
    "porousSimpleFoam": ("incompressible", "SIMPLE", ["U", "p"], True, "transportProperties"),
    "potentialFoam": ("potential", None, ["U", "p"], False, None),
    "scalarTransportFoam": ("scalar", "SIMPLE", ["T", "U"], False, "transportProperties"),
    "laplacianFoam": ("scalar", "SIMPLE", ["T"], False, "transportProperties"),
    "buoyantSimpleFoam": ("compressible", "SIMPLE", ["U", "p", "p_rgh", "T"], True, "thermophysicalProperties"),
    "buoyantPimpleFoam": ("compressible", "PIMPLE", ["U", "p", "p_rgh", "T"], True, "thermophysicalProperties"),
    "rhoSimpleFoam": ("compressible", "SIMPLE", ["U", "p", "T"], True, "thermophysicalProperties"),
    "rhoPimpleFoam": ("compressible", "PIMPLE", ["U", "p", "T"], True, "thermophysicalProperties"),
}
TURB_FIELDS = {
    "kEpsilon": ["k", "epsilon", "nut"], "RNGkEpsilon": ["k", "epsilon", "nut"], "realizableKE": ["k", "epsilon", "nut"],
    "LaunderSharmaKE": ["k", "epsilon", "nut"], "kOmega": ["k", "omega", "nut"], "kOmegaSST": ["k", "omega", "nut"],
    "kOmegaSSTSAS": ["k", "omega", "nut"], "kOmegaSSTLM": ["k", "omega", "nut", "ReThetat", "gammaInt"],
    "SpalartAllmaras": ["nuTilda", "nut"], "v2f": ["k", "epsilon", "v2", "f", "nut"], "LienCubicKE": ["k", "epsilon", "nut"],
    "Smagorinsky": ["nut"], "WALE": ["nut"], "kEqn": ["k", "nut"], "dynamicKEqn": ["k", "nut"],
    "SpalartAllmarasDES": ["nuTilda", "nut"], "SpalartAllmarasDDES": ["nuTilda", "nut"], "kOmegaSSTDES": ["k", "omega", "nut"],
}
TRANSPORTED = {"U", "k", "epsilon", "omega", "nuTilda", "v2", "T", "ReThetat", "gammaInt", "h", "e", "K"}
CONSTRAINT = {"empty", "wedge", "symmetryPlane", "symmetry", "cyclic", "cyclicAMI", "cyclicACMI", "processor"}
P_FIXED = {"fixedValue", "totalPressure", "uniformFixedValue", "fixedMean", "uniformTotalPressure", "prghPressure",
           "prghTotalPressure", "fanPressure", "timeVaryingTotalPressure", "fixedMeanOutletInlet", "outletInlet",
           "prghTotalHydrostaticPressure", "fixedProfile", "codedFixedValue", "mappedField", "waveTransmissive"}
NEEDS = {"fixedValue": ["value"], "inletOutlet": ["inletValue"], "outletInlet": ["outletValue"],
         "uniformFixedValue": ["uniformValue"], "totalPressure": ["p0"], "uniformTotalPressure": ["p0"],
         "fixedMean": ["meanValue"], "flowRateInletVelocity": [("volumetricFlowRate", "massFlowRate")],
         "turbulentIntensityKineticEnergyInlet": ["intensity"], "turbulentMixingLengthDissipationRateInlet": ["mixingLength"],
         "turbulentMixingLengthFrequencyInlet": ["mixingLength"], "movingWallVelocity": ["value"],
         "pressureInletOutletVelocity": ["value"], "fixedGradient": ["gradient"], "rotatingWallVelocity": ["origin", "axis", "omega"]}
WALL_FUNCTIONS = {"nut": r"nut\w*WallFunction|nutLowRe\w*", "epsilon": r"epsilonWallFunction", "omega": r"omegaWallFunction",
                  "k": r"kqRWallFunction|kLowReWallFunction"}

EVIDENCE = {
    "missing_patch": "simpleFoam stops at start-up: 'Cannot find patchField entry for outlet'",
    "pref": "simpleFoam stops: 'Unable to set reference cell for field p. Please supply either pRefCell or pRefPoint'",
    "turb_zero": "simpleFoam with epsilon = 0: 'bounding epsilon, min: 0' then a floating-point exception in the first iteration",
    "no_relax": "simpleFoam (plain SIMPLE, no relaxation factors): floating-point exception in the first iteration",
    "relax_099": "simpleFoam (plain SIMPLE, relaxation 0.99): floating-point exception after 5 iterations",
    "no_nu": "icoFoam stops at start-up: \"Entry 'nu' not found in dictionary ... transportProperties\"",
    "courant": "icoFoam on the cavity ran to the end at Courant 3.4 and even 50, but the transient is wrong: "
               "above 1 PISO loses time accuracy, and convection-dominated flows diverge",
}


def _txt(fs: Dict[str, bytes], path: str) -> Optional[str]:
    b = fs.get(path)
    if b is None:
        b = fs.get(path + ".gz")
        if b is not None:
            import gzip
            b = gzip.decompress(b)
    return b.decode("latin-1") if b is not None else None


def _dict(fs, path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    t = _txt(fs, path)
    if t is None:
        return None, None
    try:
        return parse(t), t
    except Exception:                                                 # unparsable dictionary: report, never crash
        return {"__error__": True}, t


def _match(d: Dict[str, Any], name: str, groups: List[str] = ()) -> Optional[Tuple[str, Any]]:
    """Dictionary entry for a name: exact key, then patch groups, then regex keys (OpenFOAM's precedence)."""
    if name in d:
        return name, d[name]
    for g in groups:
        if g in d:
            return g, d[g]
    for k, v in d.items():
        pat = k.strip('"')
        if pat == k and not any(c in k for c in "()|.*[]"):
            continue
        try:
            if re.fullmatch(pat, name):
                return k, v
        except re.error:
            continue
    return None


def _patches(fs, root) -> Tuple[List[Dict[str, Any]], str]:
    from pinneapple_data.cae.foam import read_boundary
    for p in (root + "constant/polyMesh/boundary", root + "constant/polyMesh/boundary.gz"):
        if p in fs:
            out = read_boundary(fs[p])
            for x in out:
                x["groups"] = re.findall(r"\w+", x.get("inGroups", "").split("(", 1)[-1]) if x.get("inGroups") else []
                if x["type"] in ("wall",) and "wall" not in x["groups"]:
                    x["groups"].append("wall")
            return out, "constant/polyMesh/boundary"
    t = _txt(fs, root + "system/blockMeshDict")
    if t:
        m = re.search(r"\bboundary\s*\((.*)\)\s*;", re.sub(r"//[^\n]*|/\*.*?\*/", "", t, flags=re.S), re.S)
        if m:
            out = [{"name": n, "type": ty, "groups": ["wall"] if ty == "wall" else []}
                   for n, ty in re.findall(r"(\w+)\s*\{\s*type\s+(\w+)\s*;", m.group(1))]
            if out:
                return out, "system/blockMeshDict"
    return [], ""


def check(fs: Dict[str, bytes], root: str = "") -> Report:
    ctrl, ctrl_t = _dict(fs, root + "system/controlDict")
    app = (ctrl or {}).get("application", "?")
    rep = Report(solver=f"OpenFOAM {app}")
    known = SOLVERS.get(app)
    rep.info.update(application=app, root=root.rstrip("/") or ".")
    if known is None:
        rep.add("solver", "info", f"Solver '{app}' is not in the rule set", "Generic checks only (mesh, boundary "
                "coverage, schemes and linear solvers for the fields found in 0/).", rule="unknown_solver",
                file="system/controlDict", line=line_of(ctrl_t, r"^\s*application"))
        kind, algo, base, turb_used, props = "incompressible", None, [], False, None
    else:
        kind, algo, base, turb_used, props = known
    # ---------------------------------------------------------------- turbulence and required fields
    turb_model, sim_type = None, "laminar"
    tp, tp_t = _dict(fs, root + "constant/turbulenceProperties")
    if tp is None:
        tp, tp_t = _dict(fs, root + "constant/momentumTransport")
    if tp:
        sim_type = tp.get("simulationType", "laminar")
        if sim_type in ("RAS", "LES"):
            sub = tp.get(sim_type) or {}
            turb_model = sub.get("RASModel" if sim_type == "RAS" else "LESModel") or sub.get("model")
    turb_fields = TURB_FIELDS.get(turb_model, []) if turb_used and sim_type != "laminar" else []
    if turb_used and sim_type != "laminar" and turb_model and turb_model not in TURB_FIELDS:
        rep.add("solver", "info", f"Turbulence model {turb_model}: its fields are not in the rule set",
                rule="turb_unknown", file="constant/turbulenceProperties")
    if kind == "compressible" and turb_fields:
        turb_fields = turb_fields + ["alphat"]
    required = base + turb_fields
    rep.info.update(turbulence=f"{sim_type}{' ' + turb_model if turb_model else ''}", required_fields=required)
    # ---------------------------------------------------------------- mesh
    patches, psrc = _patches(fs, root)
    mesh_rep = None
    mesh = None
    if psrc == "constant/polyMesh/boundary":
        try:
            from pinneapple_data.cae import foam as F
            from pinneapple_data.cae import mesh_report
            mesh = F.read_polymesh(fs, root)
            mesh_rep = mesh_report(mesh, "cfd")
            rep.info["mesh"] = {"cells": mesh.n_cells, "faces": mesh.poly.n_faces, "points": mesh.n_points,
                                "types": mesh.element_counts(), "patches": len(patches)}
            rep.add("mesh", "pass", f"Mesh found: {mesh.n_cells:,} cells, {len(patches)} patches", rule="mesh_found",
                    file="constant/polyMesh")
        except Exception as e:
            rep.add("mesh", "fail", "The mesh cannot be read", f"{type(e).__name__}: {e}", "Regenerate the mesh "
                    "(blockMesh / snappyHexMesh) and include constant/polyMesh complete.", rule="mesh_unreadable",
                    file="constant/polyMesh")
    elif psrc:
        cells = None
        t = _txt(fs, root + "system/blockMeshDict") or ""
        for m in re.finditer(r"hex\s*\([^)]*\)\s*\((\d+)\s+(\d+)\s+(\d+)\)", re.sub(r"//[^\n]*", "", t)):
            cells = (cells or 0) + int(m.group(1)) * int(m.group(2)) * int(m.group(3))
        rep.add("mesh", "warn", "Mesh not generated yet", f"constant/polyMesh is missing; blockMeshDict describes "
                f"{cells:,} cells and {len(patches)} patches, used here for the boundary checks." if cells else
                "constant/polyMesh is missing.", "Run blockMesh before the solver (and include constant/polyMesh "
                "to get the mesh quality checks).", rule="mesh_missing", file="system/blockMeshDict")
    else:
        rep.add("mesh", "fail", "No mesh", "Neither constant/polyMesh nor system/blockMeshDict was uploaded.",
                "Include constant/polyMesh (or system/blockMeshDict).", rule="no_mesh")
    nonorth_max = None
    if mesh_rep:
        mt = mesh_rep["metrics"]
        nonorth_max = mt.get("non_orthogonality", {}).get("worst")
        for c in mesh_rep["checks"]:
            if c["key"] in ("regions", "volume", "non_orthogonality", "skewness", "aspect_ratio", "openness",
                            "face_pyramids") and (c["status"] != "pass" or c["key"] in ("volume", "regions")):
                st = c["status"]
                rep.add("mesh", st, c["message"], c.get("why", ""), c.get("action", ""), rule=f"mesh_{c['key']}",
                        file="constant/polyMesh")
        regs = [g for g in mesh_rep.get("regions", []) if g.get("centre")]
        if regs:
            rep.info["mesh_problem_regions"] = [{"name": g["name"], "cells": g["cells"], "near": g.get("nearest_patch"),
                                                 "centre": g["centre"]} for g in regs[:6]]
    # ---------------------------------------------------------------- properties
    _materials(rep, fs, root, kind, props, app)
    # ---------------------------------------------------------------- fields: presence, BC coverage, values
    zero = root + "0/"
    has0 = any(k.startswith(zero) for k in fs)
    if not has0 and any(k.startswith(root + "0.orig/") for k in fs):
        rep.add("initial_conditions", "fail", "Initial fields are in 0.orig, not 0", "The solver reads the 0 "
                "directory; tutorials copy 0.orig to 0 in their Allrun script.", "cp -r 0.orig 0 (or restore0Dir) "
                "before running.", rule="zero_orig", file="0.orig")
        zero = root + "0.orig/"
    fields_found = sorted({k[len(zero):].removesuffix(".gz") for k in fs if k.startswith(zero) and "/" not in k[len(zero):]})
    for f in required:
        if f not in fields_found:
            rep.add("initial_conditions", "fail", f"Missing field file 0/{f}",
                    f"{app} with {rep.info['turbulence']} solves for {f}; without 0/{f} it stops at start-up "
                    f"('cannot find file').", f"Add 0/{f} with an internalField and a boundary condition on every patch.",
                    rule="missing_field", file=f"0/{f}", entity=f)
    U_ref = None
    p_fixed_somewhere = False
    for f in [x for x in required if x in fields_found] + [x for x in fields_found if x not in required and x in ("T", "nut", "alphat")]:
        d, t = _dict(fs, zero + f)
        rel = zero[len(root):] + f
        if not d or d.get("__error__"):
            rep.add("boundary_conditions", "fail", f"{rel} cannot be parsed", "Unbalanced braces or a syntax error.",
                    "Fix the syntax (OpenFOAM reports the same line).", rule="parse_error", file=rel)
            continue
        bf = d.get("boundaryField") or {}
        directives = " ".join(bf.get("__directives__", []) + d.get("__directives__", []))
        constraint_auto = "setConstraintTypes" in directives
        missing, bad_constraint, need_value = [], [], []
        for p in patches:
            m = _match(bf, p["name"], p.get("groups", []))
            if m is None:
                if not (constraint_auto and p["type"] in CONSTRAINT):
                    missing.append(p["name"])
                continue
            key, spec = m
            if not isinstance(spec, dict):
                continue
            bct = spec.get("type", "")
            if p["type"] in CONSTRAINT and bct != p["type"] and not (p["type"] == "symmetry" and bct == "symmetryPlane"):
                bad_constraint.append((p["name"], p["type"], bct))
            if p["type"] not in CONSTRAINT and bct in CONSTRAINT:
                bad_constraint.append((p["name"], p["type"], bct))
            for req in NEEDS.get(bct, []):
                opts = req if isinstance(req, tuple) else (req,)
                if not any(o in spec for o in opts):
                    need_value.append((p["name"], bct, "/".join(opts)))
            if f == "U" and p["type"] == "wall" and bct in ("zeroGradient",):
                rep.add("boundary_conditions", "warn", f"U on wall patch {p['name']} is zeroGradient",
                        "A zero-gradient velocity on a wall lets fluid slide through it tangentially and normally.",
                        "Use noSlip (or slip for a frictionless wall).", rule="wall_u", file=rel,
                        line=line_of(t, rf"^\s*\"?{re.escape(key)}\"?\s*$|^\s*{re.escape(key)}\s*\{{"), entity=p["name"])
            if f in WALL_FUNCTIONS and sim_type == "RAS" and p["type"] == "wall" and not re.fullmatch(WALL_FUNCTIONS[f], bct):
                if not (f == "k" and bct == "fixedValue" and num(spec.get("value")) == 0):
                    rep.add("boundary_conditions", "warn", f"{f} on wall {p['name']}: {bct}, not a wall function",
                            "With a high-Reynolds RAS model the near-wall cells need wall functions unless the mesh "
                            "resolves y+ ≈ 1 and the model is a low-Re variant.",
                            f"Use {WALL_FUNCTIONS[f].split('|')[0].replace(chr(92) + 'w*', 'k')} (or check y+ ≈ 1).",
                            rule="wall_function", file=rel, entity=p["name"])
            if f in ("p", "p_rgh") and bct in P_FIXED:
                p_fixed_somewhere = True
            if f == "U" and bct in ("fixedValue", "movingWallVelocity", "inletOutlet"):
                v = vec(spec.get("value") or spec.get("inletValue"))
                if v and (U_ref is None or np.linalg.norm(v) > np.linalg.norm(U_ref)):
                    U_ref = v
        if missing:
            rep.add("boundary_conditions", "fail", f"{rel}: no boundary condition for {', '.join(missing)}",
                    f"Every mesh patch needs an entry in boundaryField (or a matching regex/group entry).",
                    f"Add an entry for {', '.join(missing)} in {rel}.", rule="missing_patch", file=rel,
                    line=line_of(t, r"^\s*boundaryField"), entity=", ".join(missing), evidence=EVIDENCE["missing_patch"])
        for pn, pt, bt in bad_constraint:
            rep.add("boundary_conditions", "fail", f"{rel}: patch {pn} is '{pt}' but its condition is '{bt}'",
                    "Constraint patches (empty, wedge, symmetry, cyclic) need the matching condition type and "
                    "ordinary patches cannot use a constraint type; OpenFOAM stops with \"patch type ... not "
                    "constraint type ...\".", f"Set type {pt if pt in CONSTRAINT else 'an ordinary condition'} for "
                    f"{pn}.", rule="constraint_type", file=rel, entity=pn)
        for pn, bt, key in need_value:
            rep.add("boundary_conditions", "fail", f"{rel}: {bt} on {pn} has no '{key}'",
                    f"{bt} needs '{key}'; OpenFOAM stops with \"Entry '{key}' not found\".",
                    f"Add {key} to the {pn} entry.", rule="bc_value", file=rel, entity=pn)
        unknown = [k for k in bf if k != "__directives__" and not any(_match({k: 1}, p["name"], p.get("groups", [])) for p in patches)]
        if patches and unknown:
            rep.add("boundary_conditions", "info", f"{rel}: entries for patches not in the mesh: {', '.join(unknown[:6])}",
                    "They are ignored by OpenFOAM (often left over from another mesh).", rule="unused_entry", file=rel)
        # initial values
        ifield = str(d.get("internalField", ""))
        if f in ("k", "epsilon", "omega", "nuTilda") and "nonuniform" not in ifield:
            v = num(ifield)
            if v is not None and (v <= 0 if f != "nuTilda" else v < 0):
                rep.add("initial_conditions", "fail", f"{f} starts at {v:g}", f"The eddy viscosity is built from {f}; "
                        f"{'zero or negative' if f != 'nuTilda' else 'negative'} values give a division by zero or an "
                        "unphysical viscosity in the first iteration.",
                        f"Start {f} from a small positive value (e.g. the inlet value: k = 1.5 (I U)², "
                        f"ε = Cμ^0.75 k^1.5 / l, ω = k^0.5 / (Cμ^0.25 l)).", rule="turb_init", file=rel,
                        line=line_of(t, r"^\s*internalField"), entity=f, evidence=EVIDENCE["turb_zero"])
        if f == "T" and "nonuniform" not in ifield and kind == "compressible":
            v = num(ifield)
            if v is not None and v <= 0:
                rep.add("initial_conditions", "fail", f"T starts at {v:g} K", "Absolute temperature must be positive.",
                        "Set T in kelvin.", rule="t_init", file=rel, entity="T")
        if "nonuniform" in ifield and mesh is not None:
            mm = re.search(r"nonuniform\s+List<\w+>\s*(\d+)", t or "")
            if mm and int(mm.group(1)) != mesh.n_cells:
                rep.add("initial_conditions", "fail", f"{rel}: {mm.group(1)} values for {mesh.n_cells} cells",
                        "The field was written for another mesh; OpenFOAM stops with 'size ... is not equal to'.",
                        "Map it to this mesh (mapFields) or start from a uniform value.", rule="field_size", file=rel)
    if required and all(f in fields_found for f in required):
        rep.add("initial_conditions", "pass", f"Initial fields present: {', '.join(required)}", rule="fields_present")
    if patches and not [x for x in rep.findings if x.rule in ("missing_patch", "constraint_type", "bc_value")]:
        rep.add("boundary_conditions", "pass", f"Every patch has a boundary condition in every field ({len(patches)} "
                "patches)", rule="bc_coverage")
    # ---------------------------------------------------------------- solver settings
    _solver(rep, fs, root, ctrl, ctrl_t, app, kind, algo, required, turb_fields, p_fixed_somewhere, U_ref, mesh,
            nonorth_max)
    return rep


def _materials(rep: Report, fs, root, kind, props, app):
    if props == "transportProperties":
        d, t = _dict(fs, root + "constant/transportProperties")
        if d is None:
            rep.add("materials", "fail", "constant/transportProperties is missing", f"{app} reads the kinematic "
                    "viscosity from it.", "Add constant/transportProperties with transportModel and nu.",
                    rule="no_transport", file="constant/transportProperties")
            return
        key = "DT" if app in ("scalarTransportFoam", "laplacianFoam") else "nu"
        model = d.get("transportModel", "Newtonian")
        if key == "nu" and model != "Newtonian":
            sub = d.get(f"{model}Coeffs") or {}
            rep.add("materials", "pass" if sub else "fail", f"Non-Newtonian model {model}",
                    "" if sub else f"{model}Coeffs is missing.", "" if sub else f"Add {model}Coeffs.",
                    rule="nonnewtonian", file="constant/transportProperties")
            return
        if key not in d:
            rep.add("materials", "fail", f"Missing {'kinematic viscosity nu' if key == 'nu' else 'diffusivity DT'}",
                    f"{app} cannot start without it.", f"Add '{key} [0 2 -1 0 0 0 0] <value>;' "
                    f"({'water ≈ 1e-6, air ≈ 1.5e-5 m²/s' if key == 'nu' else 'in m²/s'}).", rule="no_nu",
                    file="constant/transportProperties", evidence=EVIDENCE["no_nu"] if key == "nu" else None)
            return
        v = num(d[key])
        line = line_of(t, rf"^\s*{key}\b")
        if v is None or v <= 0:
            rep.add("materials", "fail", f"{key} = {d[key]}", "Must be a positive number.", f"Set {key} > 0.",
                    rule="nu_value", file="constant/transportProperties", line=line)
        elif key == "nu" and not (1e-7 <= v <= 1e-2):
            rep.add("materials", "warn", f"nu = {v:g} m²/s is unusual", "Liquids are around 1e-7 to 1e-4 m²/s, gases "
                    "around 1e-5; values outside 1e-7 to 1e-2 usually mean a unit error (dynamic viscosity in Pa·s, "
                    "or mm²/s).", "Check that nu is kinematic viscosity in m²/s (μ / ρ).", rule="nu_range",
                    file="constant/transportProperties", line=line)
        else:
            rep.add("materials", "pass", f"{'Kinematic viscosity' if key == 'nu' else 'Diffusivity'} {key} = {v:g} m²/s",
                    rule="nu_ok", file="constant/transportProperties", line=line)
    elif props == "thermophysicalProperties":
        d, t = _dict(fs, root + "constant/thermophysicalProperties")
        if d is None:
            rep.add("materials", "fail", "constant/thermophysicalProperties is missing", f"{app} needs the "
                    "thermophysical model and the fluid properties.", "Add constant/thermophysicalProperties.",
                    rule="no_thermo", file="constant/thermophysicalProperties")
            return
        flat = str(d)
        missing = [k for k in ("thermoType", "molWeight", "Cp|Cv|CpCoeffs", "mu|As|muCoeffs", "Pr|kappa")
                   if not re.search(rf"'({k})'", flat)]
        if missing:
            rep.add("materials", "fail", f"thermophysicalProperties lacks {', '.join(m.split('|')[0] for m in missing)}",
                    "The thermo model cannot be constructed without these entries.", "Complete the mixture "
                    "(specie, thermodynamics, transport) sub-dictionaries.", rule="thermo_entries",
                    file="constant/thermophysicalProperties")
        else:
            rep.add("materials", "pass", "Thermophysical properties complete (molWeight, heat capacity, viscosity, "
                    "Prandtl/conductivity)", rule="thermo_ok", file="constant/thermophysicalProperties")
        if app.startswith("buoyant") and root + "constant/g" not in fs:
            rep.add("materials", "fail", "constant/g is missing", "Buoyant solvers need the gravity vector.",
                    "Add constant/g (e.g. value (0 -9.81 0)).", rule="no_g", file="constant/g")


def _algo_dict(fv: Dict[str, Any], algo: Optional[str]) -> Dict[str, Any]:
    if not fv:
        return {}
    for a in ([algo] if algo else []) + ["SIMPLE", "PIMPLE", "PISO"]:
        if a and isinstance(fv.get(a), dict):
            return fv[a]
    return {}


def _solver(rep, fs, root, ctrl, ctrl_t, app, kind, algo, required, turb_fields, p_fixed, U_ref, mesh, nonorth_max):
    fv, fv_t = _dict(fs, root + "system/fvSolution")
    sc, sc_t = _dict(fs, root + "system/fvSchemes")
    for name, d in (("system/controlDict", ctrl), ("system/fvSolution", fv), ("system/fvSchemes", sc)):
        if d is None:
            rep.add("solver", "fail", f"{name} is missing", "Every OpenFOAM solver needs it.", f"Add {name}.",
                    rule="missing_dict", file=name)
    ad = _algo_dict(fv, algo)
    # pressure reference
    if kind == "incompressible" and "p" in required and not p_fixed:
        if not any(k in ad for k in ("pRefCell", "pRefPoint")):
            rep.add("boundary_conditions", "fail", "Pressure is not fixed anywhere and there is no reference cell",
                    "With only zero-gradient/flux pressure conditions the pressure level is undetermined.",
                    f"Fix p on an outlet (fixedValue 0 or totalPressure), or add pRefCell 0; pRefValue 0; to {algo}.",
                    rule="pref", file="system/fvSolution", line=line_of(fv_t, rf"^\s*{algo}\b") if algo else None,
                    evidence=EVIDENCE["pref"])
    if ctrl:
        st, et, dt = num(ctrl.get("startTime")), num(ctrl.get("endTime")), num(ctrl.get("deltaT"))
        if dt is not None and dt <= 0:
            rep.add("solver", "fail", f"deltaT = {dt:g}", "The time step must be positive.", "Set deltaT > 0.",
                    rule="dt", file="system/controlDict", line=line_of(ctrl_t, r"^\s*deltaT"))
        if st is not None and et is not None and et <= st and ctrl.get("stopAt", "endTime") == "endTime":
            rep.add("solver", "fail", f"endTime {et:g} is not after startTime {st:g}", "The run would stop at once.",
                    "Increase endTime.", rule="endtime", file="system/controlDict", line=line_of(ctrl_t, r"^\s*endTime"))
        elif dt and et is not None and st is not None:
            n = (et - st) / dt
            rep.info["steps"] = int(n)
            if n > 5e6:
                rep.add("solver", "warn", f"{n:,.0f} time steps", "endTime / deltaT is very large; check the units "
                        "of both.", "Increase deltaT or reduce endTime.", rule="nsteps", file="system/controlDict")
    # steady SIMPLE: relaxation
    if algo == "SIMPLE" and kind != "potential" and fv:
        consistent = str(ad.get("consistent", "no")).lower() in ("yes", "on", "true")
        rf = fv.get("relaxationFactors")
        if not rf:
            if not consistent:
                rep.add("solver", "fail", "No relaxation factors with plain SIMPLE",
                        "Without under-relaxation the SIMPLE pressure-velocity iteration overshoots and diverges.",
                        "Add relaxationFactors { fields { p 0.3; } equations { U 0.7; \".*\" 0.7; } } "
                        "(or use consistent yes; with U 0.9).", rule="no_relax", file="system/fvSolution",
                        evidence=EVIDENCE["no_relax"])
        else:
            fields_rf = rf.get("fields", {}) if isinstance(rf.get("fields"), dict) else {}
            eq_rf = rf.get("equations", {}) if isinstance(rf.get("equations"), dict) else {}
            if not fields_rf and not eq_rf:
                fields_rf = {k: v for k, v in rf.items() if k in ("p", "p_rgh")}
                eq_rf = {k: v for k, v in rf.items() if k not in ("p", "p_rgh")}
            pr = _match(fields_rf, "p") or _match(fields_rf, "p_rgh")
            ur = _match(eq_rf, "U")
            pv = num(pr[1]) if pr else (None if consistent else 1.0)
            uv = num(ur[1]) if ur else 1.0
            line = line_of(fv_t, r"^\s*relaxationFactors")
            if not consistent and pv is not None and uv is not None and pv >= 0.95 and uv >= 0.95:
                rep.add("solver", "fail", f"Relaxation p {pv:g}, U {uv:g} with plain SIMPLE",
                        "Factors close to 1 remove the under-relaxation SIMPLE needs.",
                        "Use p 0.3, U 0.7 (plain SIMPLE) or consistent yes with U 0.9.", rule="relax_high",
                        file="system/fvSolution", line=line, evidence=EVIDENCE["relax_099"])
            elif not consistent and pv is not None and uv is not None and pv + uv > 1.1:
                rep.add("solver", "warn", f"Relaxation p {pv:g} + U {uv:g} > 1.1 with plain SIMPLE",
                        "The usual rule for plain SIMPLE is p + U ≈ 1 (e.g. 0.3 and 0.7); more aggressive factors "
                        "often oscillate or diverge.", "Reduce p (0.3) or U (0.7), or switch to consistent yes.",
                        rule="relax_aggressive", file="system/fvSolution", line=line)
            else:
                rep.add("solver", "pass", f"Under-relaxation: p {pv if pv is not None else '—'}, U {uv:g}"
                        f"{' (SIMPLEC)' if consistent else ''}", rule="relax_ok", file="system/fvSolution", line=line)
    # schemes: convection terms
    if sc and not sc.get("__error__"):
        for sect in ("ddtSchemes", "gradSchemes", "divSchemes", "laplacianSchemes"):
            if not isinstance(sc.get(sect), dict):
                rep.add("solver", "fail", f"fvSchemes has no {sect}", "Required by every solver.", f"Add {sect}.",
                        rule="schemes_section", file="system/fvSchemes")
        div = sc.get("divSchemes") or {}
        if isinstance(div, dict) and str(div.get("default", "none")).strip() == "none":
            need = [f for f in required if f in TRANSPORTED and not (f == "U" and app in ("potentialFoam",))]
            if kind == "scalar":
                need = ["T"] if app == "scalarTransportFoam" else []
            if kind == "compressible":
                need = [f for f in need if f not in ("T",)] + ["h|e", "K"]
            miss = []
            for f in need:
                if not any(_match(div, f"div(phi,{x})") for x in f.split("|")):
                    miss.append(f"div(phi,{f.split('|')[0]})")
            if kind == "incompressible" and app not in ("icoFoam", "nonNewtonianIcoFoam") and \
                    not _match(div, "div((nuEff*dev2(T(grad(U)))))") and not _match(div, "div((nuEff*dev(T(grad(U)))))"):
                miss.append("div((nuEff*dev2(T(grad(U)))))")
            if miss:
                rep.add("solver", "fail", f"divSchemes has no entry for {', '.join(miss)}",
                        "With 'default none' every convection term must be listed; the solver stops with "
                        "\"keyword div(...) is undefined\" at the first iteration.", f"Add {miss[0]} (e.g. bounded "
                        "Gauss linearUpwind grad(U) for U, Gauss limitedLinear 1 for scalars).", rule="div_missing",
                        file="system/fvSchemes", line=line_of(sc_t, r"^\s*divSchemes"))
            else:
                rep.add("solver", "pass", "Convection schemes listed for every transported field", rule="div_ok",
                        file="system/fvSchemes")
    # linear solvers
    if fv and isinstance(fv.get("solvers"), dict):
        sv = fv["solvers"]
        solved = [f for f in required if f not in ("nut", "alphat", "p") or f == "p"]
        if kind == "compressible":
            solved = [f if f != "T" else "h|e" for f in solved if f != "p"] + ["p_rgh"] if "p_rgh" in required else solved
        if app in ("potentialFoam",):
            solved = ["Phi"]
        need = []
        for f in solved:
            names = f.split("|")
            if not any(_match(sv, n) for n in names):
                need.append(names[0])
            if algo in ("PISO", "PIMPLE") and f in ("p", "p_rgh") and not _match(sv, f"{f}Final"):
                need.append(f"{f}Final")
            if algo == "PIMPLE" and f not in ("p", "p_rgh") and not any(_match(sv, f"{n}Final") for n in names):
                need.append(f"{names[0]}Final")
        need = list(dict.fromkeys(need))
        if need:
            rep.add("solver", "fail", f"No linear solver for {', '.join(need)}", "fvSolution/solvers needs an entry "
                    "(or a matching regex) for every solved field; the solver stops when it first solves it.",
                    f"Add entries for {', '.join(need)} (e.g. \"(U|k|epsilon)Final\" with relTol 0).",
                    rule="linear_solver", file="system/fvSolution", line=line_of(fv_t, r"^\s*solvers"))
        else:
            rep.add("solver", "pass", "A linear solver for every solved field", rule="solvers_ok", file="system/fvSolution")
    # non-orthogonal correctors
    if nonorth_max is not None and algo:
        nno = num(ad.get("nNonOrthogonalCorrectors", 0)) or 0
        if nonorth_max > 70 and nno < 1:
            rep.add("solver", "warn", f"Mesh non-orthogonality {nonorth_max:.0f}° with nNonOrthogonalCorrectors {nno:g}",
                    "Above 70° the explicit non-orthogonal part of the Laplacian is large; without correctors the "
                    "pressure equation is inaccurate and can diverge.", "Set nNonOrthogonalCorrectors 2 and use "
                    "laplacian/snGrad 'corrected' limited 0.33 or 0.5.", rule="nonorth_correctors", file="system/fvSolution")
    # Courant number (transient)
    if algo in ("PISO", "PIMPLE") and ctrl and mesh is not None and U_ref is not None:
        dt = num(ctrl.get("deltaT"))
        adjust = str(ctrl.get("adjustTimeStep", "no")).lower() in ("yes", "on", "true")
        maxco = num(ctrl.get("maxCo"))
        nouter = num(_algo_dict(fv, algo).get("nOuterCorrectors", 1)) or 1
        limit = 1.0 if algo == "PISO" or nouter <= 1 else 5.0
        if adjust and maxco is not None:
            st = "warn" if maxco > limit else "pass"
            rep.add("solver", st, f"Adaptive time step, maxCo {maxco:g}", "" if st == "pass" else
                    f"maxCo above {limit:g} with {algo} (nOuterCorrectors {nouter:g}) loses time accuracy.",
                    "" if st == "pass" else f"Use maxCo ≤ {limit:g}.", rule="courant_adaptive", file="system/controlDict")
        elif dt:
            from pinneapple_data.cae import geometry as G
            fm = G.face_mesh(mesh)
            fg = G.face_geometry(fm)
            cg = G.cell_geometry(fm, fg)
            u = np.asarray(U_ref, float)
            flux = np.abs(fg["areas"] @ u)
            sumphi = np.bincount(fm.owner, flux, fm.n_cells) + np.bincount(fm.neighbour, flux[: fm.n_internal], fm.n_cells)
            co = 0.5 * sumphi / np.maximum(cg["volumes"], 1e-300) * dt
            rep.info["courant_estimate"] = {"max": float(co.max()), "mean": float(co.mean()), "U_ref": u.tolist(), "deltaT": dt}
            msg = f"Courant number ≲ {co.max():.3g} (max), {co.mean():.3g} (mean)"
            how = (f"Upper-bound estimate: the largest boundary velocity |U| = {np.linalg.norm(u):g} m/s applied in every "
                   "cell (OpenFOAM's own Courant number is usually somewhat lower)")
            if co.max() > 1.05 * limit:                                  # the estimate is an upper bound
                need_dt = dt * limit / co.max()
                rep.add("solver", "warn", msg + " may be excessive",
                        f"{how}. {algo} needs Co ≲ {limit:g} for time accuracy and, in convection-dominated flow, "
                        "for stability.", f"Reduce deltaT to about {need_dt:.3g} s, or use adjustTimeStep yes; "
                        f"maxCo {limit:g}.", rule="courant", file="system/controlDict",
                        line=line_of(ctrl_t, r"^\s*deltaT"), evidence=EVIDENCE["courant"])
            else:
                rep.add("solver", "pass", msg, how + ".", rule="courant_ok", file="system/controlDict")

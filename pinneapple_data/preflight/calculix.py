"""Pre-flight rules for CalculiX (and flat Abaqus) input decks: mesh, sections and materials, the properties each
procedure needs, unit consistency, supports and loads, initial conditions and step settings.

The FAIL rules were confirmed by running CalculiX 2.21 on broken copies of a cantilever deck (``evidence``).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.sparse.csgraph import connected_components

from pinneapple_data.cae import inp as I

from .common import Report

PROCEDURES = {"STATIC", "FREQUENCY", "HEAT TRANSFER", "DYNAMIC", "BUCKLE", "COUPLED TEMPERATURE-DISPLACEMENT",
              "UNCOUPLED TEMPERATURE-DISPLACEMENT", "MODAL DYNAMIC", "STEADY STATE DYNAMICS", "VISCO", "GREEN",
              "SENSITIVITY", "ELECTROMAGNETICS", "COMPLEX FREQUENCY", "THERMO-MECHANICAL", "CFD", "MASS FLOW"}
MECHANICAL = {"STATIC", "FREQUENCY", "DYNAMIC", "BUCKLE", "COUPLED TEMPERATURE-DISPLACEMENT", "MODAL DYNAMIC",
              "STEADY STATE DYNAMICS", "VISCO", "UNCOUPLED TEMPERATURE-DISPLACEMENT", "COMPLEX FREQUENCY", "GREEN"}
THERMAL = {"HEAT TRANSFER", "COUPLED TEMPERATURE-DISPLACEMENT", "UNCOUPLED TEMPERATURE-DISPLACEMENT"}
OUTPUT = {"NODE FILE", "EL FILE", "NODE PRINT", "EL PRINT", "NODE OUTPUT", "ELEMENT OUTPUT", "CONTACT FILE",
          "CONTACT PRINT", "SECTION PRINT", "FACE PRINT"}
EVIDENCE = {
    "undefined_set": "ccx stops: '*ERROR reading *BOUNDARY: node set CLAMP ... has not yet been defined'",
    "no_support": "ccx reports 'Job finished' without any error, and the tip displacement is 1.8e11 mm: "
                  "the unsupported beam moves as a rigid body",
    "no_conductivity": "ccx: '*WARNING in calinput: no conductivity', then aborts while factoring the system "
                       "(exit code 255, empty .frd and .dat)",
    "inverted": "ccx stops: '*ERROR in e_c3d: nonpositive jacobian determinant in element 1'",
    "no_section": "ccx stops: '*ERROR in calinput: no material was assigned to element 638' (and the others)",
    "units": "ccx computes the first bending frequency of the cantilever as 0.16 Hz instead of 209 Hz "
             "(analytical 208.9 Hz), without a warning",
}


def _num(s: str) -> Optional[float]:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def check(fs: Dict[str, bytes], main: str) -> Report:
    import os
    texts = {os.path.basename(n): b.decode("latin-1") for n, b in fs.items() if n.lower().endswith((".inp", ".msh", ".nam", ".txt"))}
    name = os.path.basename(main)
    deck = I.read_deck(fs[main].decode("latin-1"), name, texts)
    rep = Report(solver="CalculiX / Abaqus deck")
    rep.info["deck"] = name
    if deck.find("PART") or deck.find("INSTANCE"):
        rep.add("mesh", "info", "Abaqus part/assembly deck", "Checks run on the flattened keywords; instance-"
                "qualified set names are matched by name only.", rule="parts", file=name)
    for fn in deck.missing_includes:
        rep.add("mesh", "fail", f"*INCLUDE file {fn} not uploaded", "The solver stops when it cannot open it, and the "
                "checks below cannot see its content.", f"Upload {fn} with the deck.", rule="include", file=name)
    # ------------------------------------------------------------------ mesh
    mesh = None
    try:
        mesh = I.deck_mesh(deck)
    except Exception as e:
        rep.add("mesh", "fail", "Mesh cannot be built", str(e), "Check *NODE and *ELEMENT data.", rule="mesh_error",
                file=name)
    nsets = I.resolve_sets(deck, "NSET")
    elsets = I.resolve_sets(deck, "ELSET")
    surfaces = {b.params.get("NAME", "").upper(): b for b in deck.find("SURFACE")}
    steps = _steps(deck)
    procs = [s["procedure"] for s in steps]
    rep.info["steps"] = [{"procedure": s["procedure"], "line": s["line"], "params": s["params"]} for s in steps]
    node_ids = set()
    if mesh is not None:
        node_ids = set(mesh.source["node_ids"].tolist())
        skipped = mesh.source.get("skipped_elements") or {}
        rep.info["mesh"] = {"nodes": mesh.n_points, "elements": mesh.n_cells, "types": mesh.element_counts()}
        rep.add("mesh", "pass", f"Mesh found: {mesh.n_points:,} nodes, {mesh.n_cells:,} elements "
                f"({', '.join(f'{v:,} {k}' for k, v in mesh.element_counts().items())})", rule="mesh_found", file=name)
        if skipped:
            rep.add("mesh", "info", f"Element types not checked: {', '.join(f'{k} ({v})' for k, v in skipped.items())}",
                    "Springs, masses, gaps and other special elements are passed through.", rule="skipped_elements",
                    file=name)
        if mesh.dim >= 2:
            from pinneapple_data.cae import mesh_report
            mr = mesh_report(mesh, "fea")
            for c in mr["checks"]:
                if c["key"] == "scaled_jacobian" and c["status"] == "fail":
                    worst = mr["regions"][0]["worst"].get("scaled_jacobian", {}) if mr["regions"] else {}
                    rep.add("mesh", "fail", c["message"].replace("Scaled Jacobian (min)", "Inverted elements: scaled Jacobian"),
                            "Elements with a non-positive Jacobian cannot be integrated; ccx stops with 'nonpositive "
                            "jacobian determinant'.", c["action"], rule="inverted", file=name, entity=worst.get("at"),
                            evidence=EVIDENCE["inverted"])
                elif c["key"] in ("scaled_jacobian", "element_skewness", "edge_ratio") and c["status"] == "warn":
                    rep.add("mesh", "warn", c["message"], c["why"], c["action"], rule=f"quality_{c['key']}", file=name)
    # ------------------------------------------------------------------ sections and materials
    mats = _materials(deck)
    sections = [b for b in deck.blocks if b.keyword.endswith("SECTION") and b.keyword != "SECTION PRINT"]
    covered: Set[str] = set()
    for b in sections:
        es = b.params.get("ELSET", "").upper()
        mat = b.params.get("MATERIAL", "").upper()
        if es and es not in elsets:
            rep.add("materials", "fail", f"*{b.keyword} refers to element set {es}, which is not defined",
                    "ccx stops while reading the section.", f"Define *ELSET, ELSET={es} or correct the name.",
                    rule="section_elset", file=b.file, line=b.line, entity=es)
        if mat and mat not in mats:
            rep.add("materials", "fail", f"*{b.keyword} uses material {mat}, which is not defined",
                    "ccx stops: the material does not exist.", f"Add *MATERIAL, NAME={mat} or correct the name.",
                    rule="section_material", file=b.file, line=b.line, entity=mat)
        covered.update(elsets.get(es, []))
    if mesh is not None and sections:
        all_e = {str(int(e)) for blk in mesh.blocks for e in blk.ids} | {str(int(e)) for _, blk in mesh.boundary_blocks for e in blk.ids}
        un = sorted(all_e - covered, key=int)
        if un:
            rep.add("materials", "fail", f"{len(un):,} elements have no section (e.g. {', '.join(un[:5])})",
                    "Elements without a section have no material; ccx stops ('no material/section assigned').",
                    "Add a *SOLID/*SHELL/*BEAM SECTION for their element set, or delete them.", rule="no_section",
                    file=name, entity=", ".join(un[:5]), evidence=EVIDENCE["no_section"])
        elif not [f for f in rep.findings if f.rule.startswith("section_")]:
            rep.add("materials", "pass", f"Every element has a section ({len(sections)} section{'s' if len(sections) > 1 else ''})",
                    rule="sections_ok", file=name)
    elif mesh is not None and not sections:
        rep.add("materials", "fail", "No section definitions", "No element has a material.",
                "Add *SOLID SECTION, ELSET=..., MATERIAL=...", rule="no_sections", file=name)
    used = {b.params.get("MATERIAL", "").upper() for b in sections}
    need = _needed_properties(deck, steps)
    for m in sorted(used & set(mats)):
        props = mats[m]["props"]
        for prop, why, ev in need:
            if prop not in props and not (prop == "ELASTIC" and props & {"HYPERELASTIC", "USER MATERIAL", "DEFORMATION PLASTICITY"}):
                st = "warn" if prop == "EXPANSION" else "fail"
                label = {"CONDUCTIVITY": "thermal conductivity", "SPECIFIC HEAT": "specific heat", "DENSITY": "density",
                         "ELASTIC": "elastic constants", "EXPANSION": "thermal expansion"}[prop]
                rep.add("materials", st, f"Missing {label} in material {m}", why, f"Add *{prop} under *MATERIAL, NAME={m}.",
                        rule=f"missing_{prop.lower().replace(' ', '_')}", file=mats[m]["file"], line=mats[m]["line"],
                        entity=m, evidence=EVIDENCE["no_conductivity"] if prop == "CONDUCTIVITY" else None)
        el = mats[m].get("ELASTIC")
        if el:
            E, nu = el
            if E is not None and E <= 0:
                rep.add("materials", "fail", f"Young's modulus {E:g} in {m}", "Must be positive.", "Correct *ELASTIC.",
                        rule="E", file=mats[m]["file"], line=mats[m]["eline"], entity=m)
            if nu is not None and not (-1 < nu < 0.5):
                rep.add("materials", "fail", f"Poisson's ratio {nu:g} in {m}", "Must be between -1 and 0.5.",
                        "Correct *ELASTIC.", rule="nu", file=mats[m]["file"], line=mats[m]["eline"], entity=m)
            elif nu is not None and nu >= 0.49:
                rep.add("materials", "warn", f"Poisson's ratio {nu:g} in {m}: nearly incompressible",
                        "Linear and fully integrated elements lock (far too stiff) near 0.5.",
                        "Use hybrid/reduced-integration or C3D8I/C3D10 elements, or a hyperelastic model.",
                        rule="nu_locking", file=mats[m]["file"], line=mats[m]["eline"], entity=m)
        _units(rep, m, mats[m], mesh, procs)
    if used & set(mats) and not [f for f in rep.findings if f.section == "materials" and f.status in ("fail", "warn")]:
        rep.add("materials", "pass", f"Material properties complete for {', '.join(procs) or 'the steps'}",
                rule="materials_ok", file=name)
    # ------------------------------------------------------------------ boundary conditions and loads
    _bcs_loads(rep, deck, steps, nsets, elsets, surfaces, node_ids, mesh, name)
    # ------------------------------------------------------------------ initial conditions
    ic_temp = [b for b in deck.find("INITIAL CONDITIONS") if b.params.get("TYPE", "").upper() == "TEMPERATURE"]
    for s in steps:
        if s["procedure"] == "HEAT TRANSFER" and "STEADY STATE" not in s["proc_params"] and not ic_temp:
            rep.add("initial_conditions", "warn", "Transient heat transfer without initial temperatures",
                    "Nodes start at 0 (in the deck's temperature unit), which is rarely intended.",
                    "Add *INITIAL CONDITIONS, TYPE=TEMPERATURE (e.g. NALL, 20.).", rule="ic_heat", file=name,
                    line=s["line"])
        if s["procedure"] in MECHANICAL and s["has_temperature"] and not ic_temp:
            rep.add("initial_conditions", "warn", "Temperatures applied without an initial (reference) temperature",
                    "Thermal strain is computed from 0: every degree counts as expansion.",
                    "Add *INITIAL CONDITIONS, TYPE=TEMPERATURE with the stress-free temperature.", rule="ic_thermal_stress",
                    file=name, line=s["line"])
    if ic_temp:
        rep.add("initial_conditions", "pass", "Initial temperatures defined", rule="ic_ok", file=name, line=ic_temp[0].line)
    elif not any(s["procedure"] in THERMAL or s["has_temperature"] for s in steps):
        rep.add("initial_conditions", "pass", "No initial conditions needed (mechanical analysis from rest)", rule="ic_none")
    # ------------------------------------------------------------------ steps
    _steps_rules(rep, deck, steps, name)
    return rep


def _steps(deck: I.Deck) -> List[Dict[str, Any]]:
    steps, cur = [], None
    for b in deck.blocks:
        if b.keyword == "STEP":
            cur = {"line": b.line, "params": dict(b.params), "procedure": None, "proc_params": {}, "proc_data": [],
                   "blocks": [], "ended": False, "has_temperature": False}
            steps.append(cur)
            continue
        if cur is None:
            continue
        if b.keyword == "END STEP":
            cur["ended"] = True
            cur = None
            continue
        cur["blocks"].append(b)
        if b.keyword in PROCEDURES and cur["procedure"] is None:
            cur["procedure"] = b.keyword
            cur["proc_params"] = b.params
            cur["proc_data"] = b.data
            cur["proc_line"] = b.line
        if b.keyword == "TEMPERATURE":
            cur["has_temperature"] = True
    return steps


def _materials(deck: I.Deck) -> Dict[str, Dict[str, Any]]:
    mats: Dict[str, Dict[str, Any]] = {}
    cur = None
    for b in deck.blocks:
        if b.keyword == "MATERIAL":
            cur = b.params.get("NAME", "").upper()
            mats[cur] = {"props": set(), "file": b.file, "line": b.line}
            continue
        if cur is None:
            continue
        if b.keyword in ("ELASTIC", "DENSITY", "CONDUCTIVITY", "SPECIFIC HEAT", "EXPANSION", "PLASTIC", "HYPERELASTIC",
                         "USER MATERIAL", "DEFORMATION PLASTICITY", "CREEP", "DAMPING", "ELECTRICAL CONDUCTIVITY"):
            mats[cur]["props"].add(b.keyword)
            vals = [_num(x) for x in (b.data[0] if b.data else [])]
            if b.keyword == "ELASTIC" and b.params.get("TYPE", "ISO").upper().startswith("ISO"):
                mats[cur]["ELASTIC"] = (vals[0] if vals else None, vals[1] if len(vals) > 1 else None)
                mats[cur]["eline"] = b.line
            if b.keyword == "DENSITY":
                mats[cur]["DENSITY"] = vals[0] if vals else None
                mats[cur]["dline"] = b.line
        elif b.keyword not in ("MATERIAL",) and not b.keyword.startswith(("ELASTIC", "PLASTIC")):
            cur = None
    return mats


def _needed_properties(deck: I.Deck, steps) -> List[Tuple[str, str, Any]]:
    procs = {s["procedure"] for s in steps}
    need = []
    if procs & MECHANICAL:
        need.append(("ELASTIC", "Every mechanical step needs the elastic constants; ccx stops without them.", None))
    dyn = procs & {"FREQUENCY", "DYNAMIC", "MODAL DYNAMIC", "STEADY STATE DYNAMICS", "COMPLEX FREQUENCY"}
    grav = any(b.keyword == "DLOAD" and any(r and len(r) > 1 and r[1].upper() in ("GRAV", "CENTRIF") for r in b.data)
               for b in deck.blocks)
    transient_heat = any(s["procedure"] == "HEAT TRANSFER" and "STEADY STATE" not in s["proc_params"] for s in steps) or \
        "COUPLED TEMPERATURE-DISPLACEMENT" in procs
    if dyn or grav or transient_heat:
        why = ("mass for the " + ", ".join(sorted(dyn)).lower()) if dyn else "gravity / centrifugal loads" if grav else "heat capacity"
        need.append(("DENSITY", f"Density is needed for {why}; without it ccx stops or the mass matrix is zero.", None))
    if procs & THERMAL:
        need.append(("CONDUCTIVITY", "Heat conduction needs the thermal conductivity.", "no_conductivity"))
    if transient_heat:
        need.append(("SPECIFIC HEAT", "Transient heat transfer needs the specific heat.", None))
    if any(s["has_temperature"] for s in steps if s["procedure"] in MECHANICAL) or "COUPLED TEMPERATURE-DISPLACEMENT" in procs:
        need.append(("EXPANSION", "Temperatures are applied but without *EXPANSION they cause no thermal strain.", None))
    return need


def _units(rep: Report, m: str, mat: Dict[str, Any], mesh, procs) -> None:
    """E and density must belong to one consistent unit system (N-m-kg-s: E ~ 1e9-1e12, rho ~ 1e2-2e4;
    N-mm-t-s: E ~ 1e2-1e6, rho ~ 1e-10-2e-8)."""
    el, rho = mat.get("ELASTIC"), mat.get("DENSITY")
    if not el or el[0] is None:
        return
    E = el[0]
    sys_E = "SI (Pa)" if E > 1e7 else "N-mm (MPa)" if 1 < E < 1e7 else None
    size = None
    if mesh is not None:
        size = float(np.max(mesh.points.max(0) - mesh.points.min(0)))
    if rho is not None and sys_E:
        sys_r = "SI (kg/m³)" if 10 < rho < 3e4 else "N-mm (t/mm³)" if 1e-11 < rho < 1e-7 else "other"
        if (sys_E.startswith("SI") and not sys_r.startswith("SI")) or (sys_E.startswith("N-mm") and not sys_r.startswith("N-mm")):
            dyn = bool(set(procs) & {"FREQUENCY", "DYNAMIC", "MODAL DYNAMIC", "STEADY STATE DYNAMICS"})
            right = f"{rho * 1e-12:.3g} t/mm³" if sys_E.startswith("N-mm") and sys_r.startswith("SI") else \
                f"{rho * 1e12:.3g} kg/m³" if sys_E.startswith("SI") else "a value in the same system"
            rep.add("materials", "fail" if dyn else "warn", f"Inconsistent units in material {m}: E = {E:g} looks like "
                    f"{sys_E}, density = {rho:g} like {sys_r}",
                    "CalculiX has no units: E, density, lengths and loads must come from one consistent system. "
                    + ("Mass and frequencies are wrong by orders of magnitude, silently." if dyn else
                       "Static stresses are unaffected, but gravity loads and any dynamic step are wrong."),
                    f"In N-mm-t-s use density {right}; in SI use E in Pa.", rule="units", file=mat["file"],
                    line=mat.get("dline"), entity=m, evidence=EVIDENCE["units"] if dyn else None)
            return
    if sys_E and size is not None:
        if sys_E.startswith("N-mm") and size < 2:
            rep.add("materials", "warn", f"E = {E:g} (MPa) but the model is {size:g} long",
                    "A model under 2 length units with E in MPa usually means geometry in metres and E in MPa.",
                    "Use one system: geometry in mm with E in MPa, or metres with E in Pa.", rule="units_size",
                    file=mat["file"], line=mat.get("eline"), entity=m)
        if sys_E.startswith("SI") and size > 500:
            rep.add("materials", "warn", f"E = {E:g} (Pa) but the model is {size:g} long",
                    "A model over 500 length units with E in Pa usually means geometry in mm and E in Pa.",
                    "Use one system: geometry in mm with E in MPa, or metres with E in Pa.", rule="units_size",
                    file=mat["file"], line=mat.get("eline"), entity=m)


def _target_nodes(row: List[str], nsets, node_ids) -> Tuple[Optional[List[int]], Optional[str]]:
    t = row[0].strip()
    if re.fullmatch(r"-?\d+", t):
        n = int(t)
        return ([n], None) if n in node_ids else (None, f"node {n}")
    key = t.upper()
    if key in nsets:
        return [int(x) for x in nsets[key]], None
    return None, t


def _bcs_loads(rep: Report, deck, steps, nsets, elsets, surfaces, node_ids, mesh, name) -> None:
    constrained: Dict[int, Set[int]] = {}
    nonzero_disp = False
    temp_bc = False
    undefined: List[Tuple[Any, str]] = []
    for b in deck.find("BOUNDARY"):
        for r in b.data:
            if not r or not r[0]:
                continue
            nodes, bad = _target_nodes(r, nsets, node_ids)
            if bad:
                undefined.append((b, bad))
                continue
            d1 = int(_num(r[1]) or 0) if len(r) > 1 else 0
            d2 = int(_num(r[2]) or d1) if len(r) > 2 and r[2] else d1
            val = _num(r[3]) if len(r) > 3 else 0.0
            for dof in range(d1, d2 + 1):
                if dof == 11:
                    temp_bc = True
                for n in nodes:
                    constrained.setdefault(n, set()).add(dof)
            if val not in (None, 0.0) and d1 <= 6:
                nonzero_disp = True
    for b, bad in undefined:
        rep.add("boundary_conditions", "fail", f"*BOUNDARY refers to {bad}, which is not defined",
                "ccx stops while reading the deck.", f"Define *NSET, NSET={bad} (or correct the name/number).",
                rule="undefined_set", file=b.file, line=b.line, entity=bad, evidence=EVIDENCE["undefined_set"])
    loads = {"CLOAD": [], "DLOAD": [], "CFLUX": [], "DFLUX": [], "FILM": [], "RADIATE": [], "TEMPERATURE": []}
    for b in deck.blocks:
        if b.keyword in loads:
            loads[b.keyword].append(b)
            for r in b.data:
                if not r or not r[0] or re.fullmatch(r"-?\d+", r[0]):
                    continue
                key = r[0].upper()
                ok = key in (nsets if b.keyword in ("CLOAD", "CFLUX", "TEMPERATURE") else elsets) or key in surfaces \
                    or key in nsets or key in elsets
                if not ok:
                    rep.add("boundary_conditions", "fail", f"*{b.keyword} refers to {r[0]}, which is not defined",
                            "ccx stops while reading the load.", f"Define the set/surface {r[0]} or correct the name.",
                            rule="undefined_load_set", file=b.file, line=b.line, entity=r[0])
    procs = {s["procedure"] for s in steps}
    springs = any(b.keyword in ("SPRING", "EQUATION", "TIE", "CONTACT PAIR", "RIGID BODY", "COUPLING", "MPC")
                  for b in deck.blocks)
    if procs & {"STATIC", "BUCKLE", "VISCO", "COUPLED TEMPERATURE-DISPLACEMENT", "UNCOUPLED TEMPERATURE-DISPLACEMENT"} and mesh is not None:
        _rigid_modes(rep, mesh, constrained, springs, name)
    elif procs & {"FREQUENCY", "MODAL DYNAMIC"} and not any(d & {1, 2, 3} for d in constrained.values()):
        rep.add("boundary_conditions", "info", "No supports: free-free modal analysis",
                "The first six modes will be rigid-body modes at ~0 Hz.", rule="free_free", file=name)
    if "HEAT TRANSFER" in procs and not temp_bc and not loads["FILM"] and not loads["RADIATE"]:
        rep.add("boundary_conditions", "fail", "Heat transfer without any temperature condition, film or radiation",
                "With only heat fluxes the temperature level is undetermined (singular system) in steady state.",
                "Fix the temperature somewhere (*BOUNDARY, node set, 11, 11, T) or add *FILM convection.",
                rule="heat_no_sink", file=name)
    mech = procs & {"STATIC", "DYNAMIC", "VISCO", "BUCKLE"}
    if mech and not (loads["CLOAD"] or loads["DLOAD"] or loads["TEMPERATURE"] or nonzero_disp):
        rep.add("boundary_conditions", "warn", "No loads in the mechanical steps",
                "No *CLOAD, *DLOAD, temperature or prescribed displacement: the result will be zero everywhere.",
                "Add the load (*CLOAD / *DLOAD) inside the step.", rule="no_loads", file=name)
    n_loads = sum(len(v) for v in loads.values())
    if constrained and not undefined:
        rep.add("boundary_conditions", "pass", f"Supports on {len(constrained):,} nodes"
                f"{f', {n_loads} load block' + ('s' if n_loads != 1 else '') if n_loads else ''}; every set referenced "
                "is defined", rule="bcs_ok", file=name)


def _rigid_modes(rep: Report, mesh, constrained, springs, name) -> None:
    """Every connected part must be held: translations 1-3 constrained, at three non-collinear points (or more)."""
    from pinneapple_data.cae import geometry as G
    adj = G.element_adjacency_by_nodes(mesh)
    n_reg, lab = connected_components(adj, directed=False)
    nid = mesh.source["node_ids"]
    pos = {int(n): i for i, n in enumerate(nid)}
    node_region = np.full(len(nid), -1)
    start = 0
    for b in mesh.blocks:
        if b.dim != mesh.dim:
            start += len(b.conn)
            continue
        node_region[b.conn.ravel()] = np.repeat(lab[start:start + len(b.conn)], b.conn.shape[1])
        start += len(b.conn)
    for r in range(n_reg):
        cn = [n for n, d in constrained.items() if n in pos and node_region[pos[n]] == r]
        dofs = set().union(*[constrained[n] for n in cn]) if cn else set()
        size = int(np.sum(lab == r))
        where = f"part {r + 1} of {n_reg} ({size:,} elements)" if n_reg > 1 else "the model"
        if not cn or not (dofs & {1, 2, 3}):
            rep.add("boundary_conditions", "warn" if springs else "fail", f"No supports on {where}",
                    "Nothing stops it from moving as a rigid body: the stiffness matrix is singular. CalculiX may "
                    "still finish, with meaningless displacements." + (" A spring, tie or contact may hold it: check."
                                                                    if springs else ""),
                    "Fix it with *BOUNDARY (e.g. NFIX, 1, 3) or connect it to a supported part.",
                    rule="no_support", file=name, entity=where, evidence=EVIDENCE["no_support"])
            continue
        missing = {1, 2, 3} - dofs
        pts = mesh.points[[pos[n] for n in cn]]
        rank = np.linalg.matrix_rank(pts - pts.mean(0), tol=1e-9 * max(1.0, float(np.abs(pts).max()))) if len(pts) > 1 else 0
        if missing:
            rep.add("boundary_conditions", "warn" if springs else "fail", f"{where.capitalize()} is free in direction "
                    f"{', '.join(map(str, sorted(missing)))}", "Supports never constrain that translation: rigid-body "
                    "motion along it.", "Add a support in that direction.", rule="free_direction", file=name, entity=where)
        elif rank < 1 or (rank < 2 and not (dofs & {4, 5, 6})):
            rep.add("boundary_conditions", "warn", f"{where.capitalize()} is supported at a single point or line",
                    "It can rotate about that point/line (singular for solid elements without rotational DOFs).",
                    "Support at least three non-collinear nodes.", rule="rotation_free", file=name, entity=where)


def _steps_rules(rep: Report, deck, steps, name) -> None:
    if not steps:
        rep.add("solver", "fail", "No *STEP", "The deck defines a model but no analysis.", "Add *STEP ... *END STEP.",
                rule="no_step", file=name)
        return
    for i, s in enumerate(steps, 1):
        if s["procedure"] is None:
            rep.add("solver", "fail", f"Step {i} has no procedure", "Each step needs *STATIC, *FREQUENCY, *HEAT TRANSFER, "
                    "*DYNAMIC, ...", "Add the procedure keyword after *STEP.", rule="no_procedure", file=name, line=s["line"])
        if not s["ended"]:
            rep.add("solver", "fail", f"Step {i} has no *END STEP", "ccx stops at the end of the file.", "Add *END STEP.",
                    rule="end_step", file=name, line=s["line"])
        if s["procedure"] == "STATIC" and s["proc_data"]:
            v = [_num(x) for x in s["proc_data"][0]]
            if len(v) >= 2 and v[0] and v[1] and v[0] > v[1]:
                rep.add("solver", "warn", f"Step {i}: initial increment {v[0]:g} larger than the step time {v[1]:g}",
                        "ccx uses the step time instead.", "Set the initial increment ≤ the step time.",
                        rule="increment", file=name, line=s.get("proc_line"))
            if len(v) >= 3 and v[0] and v[2] and v[2] > v[0]:
                rep.add("solver", "fail", f"Step {i}: minimum increment {v[2]:g} larger than the initial {v[0]:g}",
                        "Inconsistent increment control.", "Reduce the minimum increment.", rule="min_increment",
                        file=name, line=s.get("proc_line"))
        if s["procedure"] in ("FREQUENCY", "BUCKLE") and s["proc_data"]:
            n = _num(s["proc_data"][0][0]) if s["proc_data"][0] else None
            if not n or n < 1:
                rep.add("solver", "fail", f"Step {i}: number of modes not set", "", "Give the number of eigenvalues on "
                        "the line after the procedure keyword.", rule="nmodes", file=name, line=s.get("proc_line"))
        outs = [b for b in s["blocks"] if b.keyword in OUTPUT]
        if not outs and not any(b.keyword in OUTPUT for st in steps[:i - 1] for b in st["blocks"]):
            rep.add("solver", "warn", f"Step {i} requests no output", "Without *NODE FILE / *EL FILE / *NODE PRINT "
                    "nothing is written to the .frd or .dat file.", "Add *NODE FILE (U, NT) and *EL FILE (S).",
                    rule="no_output", file=name, line=s["line"])
    cp = deck.find("CONTACT PAIR")
    inter = {b.params.get("NAME", "").upper() for b in deck.find("SURFACE INTERACTION")}
    for b in cp:
        it = b.params.get("INTERACTION", "").upper()
        if it and it not in inter:
            rep.add("solver", "fail", f"*CONTACT PAIR uses interaction {it}, which is not defined", "",
                    f"Add *SURFACE INTERACTION, NAME={it}.", rule="contact_interaction", file=b.file, line=b.line)
    good = [s for s in steps if s["procedure"] and s["ended"]]
    if good and not [f for f in rep.findings if f.section == "solver" and f.status in ("fail", "warn")]:
        rep.add("solver", "pass", f"{len(steps)} step{'s' if len(steps) > 1 else ''}: "
                f"{', '.join(s['procedure'].lower() for s in good)}; output requested", rule="steps_ok", file=name)

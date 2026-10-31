"""Problem nodes: domain (with the 1-D / 2-D / 3-D switch), mesh (structured grid, or unstructured triangles /
tetrahedra through gmsh), equations (editable, validated text)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node
from ..runtime import artifact
from ..templates import FAMILIES, template
from ._common import need_problem, system_of

SPATIAL = ("x", "y", "z")


def geometry_doc(params: dict[str, Any]) -> dict[str, Any]:
    d = int(params["dimension"])
    rng = {"x": params["x_range"], "y": params.get("y_range") or params["x_range"],
           "z": params.get("z_range") or params["x_range"]}
    bounds = {}
    for c in SPATIAL[:d]:
        lo, hi = (float(v) for v in rng[c])
        if not hi > lo:
            raise StepFailed(f"{c}_range must be [low, high] with high > low")
        bounds[c] = [lo, hi]
    shape = params.get("shape") or "box"
    names = {1: "interval", 2: "rectangle", 3: "box"}
    if shape in ("box", "auto"):
        shape = names[d]
    time = None
    if params.get("time_dependent"):
        t0, t1 = (float(v) for v in params["t_range"])
        if not t1 > t0:
            raise StepFailed("t_range must be [start, end] with end > start")
        time = [t0, t1]
    coords = list(SPATIAL[:d]) + (["t"] if time else [])
    return {"dimension": d, "shape": shape, "bounds": bounds, "time": time, "coords": coords}


@node("geometry.domain", "Domain", category="Problem",
      outputs=[Port("geometry", "geometry")],
      params=[Param("dimension", "enum", 2, choices=[1, 2, 3], help="spatial dimension (switch 2-D <-> 3-D here)"),
              Param("shape", "enum", "box", choices=["box", "interval", "rectangle"],
                    help="tensor-product box (classical solvers need a box)"),
              Param("x_range", "list", [0.0, 1.0]), Param("y_range", "list", [0.0, 1.0]),
              Param("z_range", "list", [0.0, 1.0]),
              Param("time_dependent", "bool", True), Param("t_range", "list", [0.0, 1.0])],
      calls="pinneapple_flow.nodes.problem.geometry_doc")
def domain(ctx, inputs, params):
    """The domain: dimension, bounds per axis and the time interval (if the problem evolves in time)."""
    g = geometry_doc(params)
    ctx.metric("dimension", g["dimension"])
    ctx.summary(f"{g['shape']} in {g['dimension']}-D, coordinates {g['coords']}")
    ctx.view("geometry", g)
    return {"geometry": g}


@node("geometry.mesh", "Mesh / grid", category="Problem",
      inputs=[Port("geometry", "geometry")], outputs=[Port("mesh", "mesh")],
      params=[Param("n", "int", 64, minimum=5, maximum=2048, help="nodes per spatial axis"),
              Param("n_t", "int", 41, minimum=2, maximum=2001, help="saved time levels")],
      calls="structured grid for pinneapple_simulation.numerical_solvers.symbolic_fd")
def mesh(ctx, inputs, params):
    """Structured grid over the domain: nodes per axis and the saved time levels."""
    g = inputs["geometry"]
    g = g.get("geometry", g)
    n = {c: int(params["n"]) for c in g["bounds"]}
    spacing = {c: (b[1] - b[0]) / (n[c] - 1) for c, b in g["bounds"].items()}
    total = 1
    for v in n.values():
        total *= v
    ctx.metric("nodes", total)
    ctx.table("spacing", ["axis", "nodes", "h"], [[c, n[c], spacing[c]] for c in n])
    return {"mesh": {"geometry": g, "n": n, "n_t": int(params["n_t"]) if g.get("time") else None}}


@node("geometry.unstructured_mesh", "Unstructured mesh", category="Problem",
      inputs=[Port("geometry", "geometry", required=False)], outputs=[Port("mesh", "unstructured_mesh")],
      params=[Param("shape", "enum", "box", choices=["domain", "box", "cylinder", "sphere", "rectangle", "disk", "file"],
                    help="domain: the box / rectangle of the connected geometry; file: an STL, STEP or IGES file"),
              Param("dimensions", "list", [], help="box [lx, ly, lz], cylinder [radius, height], sphere [radius], "
                                                    "rectangle [lx, ly], disk [radius] in m (empty: defaults)"),
              Param("file", "str", "", help="geometry file in the data folder (shape = file)"),
              Param("scale", "float", 1.0, minimum=1e-9, help="file length unit to metres (1e-3 for mm)"),
              Param("h", "float", 0.1, minimum=1e-6, help="target element size [m]"),
              Param("refine", "list", [], advanced=True,
                    help="balls of smaller elements: {center: [x, y, z], radius, h} (metres)"),
              Param("boundary_layer", "dict", {}, advanced=True,
                    help="wall layers: {first_layer: m, n_layers: 4, growth: 1.2} (empty: none)")],
      calls="pinneapple_design.geometry.mesh.generate.generate_mesh (gmsh; quality: gmsh minSICN / gamma, "
            "pinneapple_data.cae.quality.element_metrics; view: pp.viz.field_view)")
def unstructured_mesh(ctx, inputs, params):
    """Triangles (2-D) or tetrahedra (3-D, with prism layers at the walls on request) from a primitive, a geometry
    file or the connected domain, with the quality of every element, the .msh / .vtu files and a 3-D view."""
    from pinneapple_design.geometry.mesh.generate import SICN_WARN, generate_mesh

    from ..config import safe_data_path
    shape, dims, origin = params["shape"], list(params["dimensions"] or []), None
    g = (inputs.get("geometry") or {})
    g = g.get("geometry", g) if isinstance(g, dict) else {}
    if shape == "domain":
        b = g.get("bounds") or {}
        if len(b) not in (2, 3):
            raise StepFailed("shape = domain needs a connected 2-D or 3-D geometry")
        axes = [c for c in SPATIAL if c in b]
        dims = [float(b[c][1]) - float(b[c][0]) for c in axes]
        origin = [float(b[c][0]) for c in axes]
        shape = "rectangle" if len(axes) == 2 else "box"
    elif shape == "file":
        shape = str(safe_data_path(params["file"]))
    try:
        out = generate_mesh(shape, dims or None, h=float(params["h"]), scale=float(params["scale"]),
                            refine=params["refine"] or None, boundary_layer=params["boundary_layer"] or None,
                            out_dir=str(ctx.path("mesh")), origin=origin)
    except ValueError as exc:
        raise StepFailed(str(exc)) from None
    for k in ("n_nodes", "n_elements", "sicn_min", "gamma_min", "aspect_ratio_max", "h_min", "h_mean", "h_max"):
        ctx.metric(k, out[k])
    ctx.metric("volume" if out["dim"] == 3 else "area", out["volume"] if out["dim"] == 3 else out["area"])
    ctx.table("quality", ["metric", "min", "mean", "max", "threshold"], [
        ["SICN (1 ideal, <= 0 inverted)", out["sicn_min"], out["sicn_mean"], None, f">= {SICN_WARN}"],
        ["gamma (1 ideal)", out["gamma_min"], out["gamma_mean"], None, ""],
        ["edge ratio (1 ideal)", None, out["aspect_ratio_mean"], out["aspect_ratio_max"], ""],
        ["edge length [m]", out["h_min"], out["h_mean"], out["h_max"], f"target {float(params['h']):g}"]])
    ctx.table("elements", ["type", "count"], [[t, n] for t, n in out["element_counts"].items()])
    if out["sicn_min"] < SICN_WARN:
        ctx.log(f"worst element SICN {out['sicn_min']:.3g} < {SICN_WARN}: refine or smooth before solving")
    meta = json.loads(Path(out["files"]["view"]).read_text())
    ctx.view("scene", {"glb": out["files"]["glb"], "meta": meta})
    ctx.summary(f"{out['n_elements']} elements ({', '.join(f'{n} {t}' for t, n in out['element_counts'].items())}), "
                f"{out['n_nodes']} nodes, worst SICN {out['sicn_min']:.3f}")
    files = {k: artifact(v, k) for k, v in out["files"].items()}
    return {"mesh": {"geometry": g or None, "summary": {k: v for k, v in out.items() if k != "files"},
                     "files": files}}


@node("equations.system", "Equations", category="Problem",
      inputs=[Port("geometry", "geometry"), Port("mesh", "mesh", required=False)],
      outputs=[Port("problem", "problem"), Port("equations", "equations")],
      params=[Param("template", "enum", "custom", choices=["custom", *FAMILIES],
                    help="PDE family proposed by the planner"),
              Param("follow_template", "bool", False,
                    help="regenerate the equations from the template for the domain's dimension (the 2-D <-> 3-D "
                         "switch); edited coefficients are kept"),
              Param("fields", "list", ["u"], help="unknown fields"),
              Param("pdes", "equations", [], help="one PDE per line, e.g. u_t = alpha*lap(u)"),
              Param("conditions", "list", [], help="{kind: dirichlet|neumann|robin|initial|periodic, on: x_min|...|"
                                                   "boundary, equation: 'u = 0'}"),
              Param("coefficients", "dict", {}, help="coefficient values, e.g. {alpha: 0.1}"),
              Param("unknowns", "list", [], help="coefficients to identify from data (inverse problems)"),
              Param("exact", "expression", "", help="exact solution, if known (for validation)")],
      calls="pinneapple_physics.symbolic_pde.text.parse_system (safe AST parser, no eval)")
def equations(ctx, inputs, params):
    """The equation system: PDEs, boundary and initial conditions, coefficients. The text is parsed by a safe
    parser and validated (unknown names, missing values, conditions on unknown faces); the inferred family is shown."""
    geo = inputs["geometry"]
    m = inputs.get("mesh")
    eq = {k: params[k] for k in ("fields", "pdes", "conditions", "coefficients", "unknowns", "exact")}
    eq["family"] = params["template"]
    if params["template"] != "custom" and (params["follow_template"] or not params["pdes"]):
        try:
            t = template(params["template"], geo["dimension"])
        except ValueError as exc:
            raise StepFailed(str(exc)) from None
        coeffs = {**t["coefficients"], **{k: v for k, v in params["coefficients"].items()}}
        eq.update(fields=t["fields"], pdes=t["pdes"], conditions=t["conditions"], coefficients=coeffs,
                  exact=t["exact"] or "")
        ctx.log(f"equations from the {params['template']} template in {geo['dimension']}-D")
    if not eq["pdes"]:
        raise StepFailed("no PDE: write at least one, or pick a template")
    problem = {"geometry": geo, "equations": eq}
    if m:
        problem["mesh"] = {"n": m["n"], "n_t": m.get("n_t")}
    sys_ = system_of(problem)
    doc = sys_.to_dict()
    issues = [i for i in doc["issues"] if "without a value" not in i or not eq["unknowns"]]
    if issues:
        raise StepFailed("; ".join(issues))
    cls = doc["classification"]
    ctx.view("equations", doc)
    ctx.metric("family", cls.get("family"))
    ctx.metric("linear", cls.get("linear"))
    ctx.table("equations", ["kind", "on", "equation", "derivatives"],
              [[e["kind"], e["on"], e["text"], ", ".join(e["derivatives"])] for e in doc["pdes"] + doc["conditions"]])
    ctx.summary(f"{cls.get('family')} ({'linear' if cls.get('linear') else 'nonlinear'}), "
                f"{len(doc['pdes'])} PDE(s), {len(doc['conditions'])} condition(s)")
    eq["classification"] = cls
    return {"problem": problem, "equations": eq}


def require_problem(v):
    return need_problem(v)


@node("problem.generic", "Generic problem (identify the equations)", category="Problem",
      inputs=[Port("data", "solution", required=False, help="measured / simulated field on a space-time grid: the "
                                                            "equations are also discovered from it")],
      outputs=[Port("problem", "problem"), Port("geometry", "geometry"), Port("equations", "equations")],
      params=[Param("description", "text", "", help="the problem in your words: physics, geometry, conditions, "
                                                   "values with units"),
              Param("units", "dict", {}, help="quantity -> unit, e.g. {alpha: m^2/s, x: m}"),
              Param("dimension", "enum", 0, choices=[0, 1, 2, 3], help="0: as identified"),
              Param("candidate", "int", 0, minimum=0, maximum=5, help="which proposed system to accept"),
              Param("use_llm", "bool", True, help="also ask the configured LLM (validated; offline without one)"),
              Param("discover", "bool", True, help="sparse regression on the data input (SINDy / PDE-FIND style)"),
              Param("fields", "list", [], help="edit: fields (empty = as identified)"),
              Param("pdes", "equations", [], help="edit: PDEs (empty = the accepted candidate's)"),
              Param("conditions", "list", [], help="edit: conditions (empty = the candidate's)"),
              Param("coefficients", "dict", {}, help="edit: coefficient values (override the candidate's)"),
              Param("time_dependent", "enum", "auto", choices=["auto", "yes", "no"]),
              Param("x_range", "list", [], help="edit: domain [low, high] (empty = the candidate's)"),
              Param("t_range", "list", [])],
      calls="pinneapple_flow.identify.identify (+ discover_pde: SINDyIdentifier STRidge) -> parse_system")
def generic(ctx, inputs, params):
    """Describe any problem (and optionally give data): the governing equations are identified with a confidence,
    the reasoning and their references (conservation law, constitutive relation, dimension, conditions,
    coefficients). Accept a candidate or edit it; with data the equations are also discovered from the data and
    compared with the candidates."""
    from ..config import Settings
    from ..identify import confirm, discover_pde, identify
    from ._common import load_solution
    s = Settings.from_env()
    llm = None
    if params["use_llm"] and s.llm_provider not in ("", "none"):
        llm = {"provider": s.llm_provider, "model": s.llm_model, "base_url": s.llm_base_url, "api_key": s.llm_api_key}
    text = params["description"] or ""
    res = identify(text, dimension=int(params["dimension"]) or None, units=params["units"], llm=llm)
    for w in res["warnings"]:
        ctx.warn(w)
    cands = res["candidates"]
    found = None
    if inputs.get("data") and params["discover"]:
        sol = load_solution(inputs["data"])
        try:
            found = discover_pde(sol["axes"], sol["values"], sol["coords"], field=(sol["fields"] or ["u"])[0])
        except ValueError as exc:
            ctx.warn(f"discovery: {exc}")
        if found:
            ctx.metric("discovered_equation", found["pretty"])
            ctx.metric("discovery_r2", found["r2"], "fit of the sparse regression on interior grid points")
            ctx.report["decisions"].append(found["decision"])
            if found["warning"]:
                ctx.warn(found["warning"])
            for c in cands:
                c["data_check"] = confirm(c, found)
                if c["data_check"]["family_agrees"]:
                    c["confidence"] = round(min(0.95, c["confidence"] + 0.2), 2)
                    c["reasoning"].append(f"confirmed by the data: {found['pretty']} (R^2 {found['r2']:.4f})")
            cands.sort(key=lambda c: -c["confidence"])
            if not any(c["data_check"]["family_agrees"] for c in cands):
                ctx.warn(f"the data do not match any candidate; discovered {found['pretty']}")
    if not cands:
        raise StepFailed("no candidate system")
    k = min(int(params["candidate"]), len(cands) - 1)
    c = cands[k]
    ctx.view("candidates", {"candidates": cands, "accepted": k, "discovered": found, "detected": res["detected"]})
    ctx.table("candidate systems", ["#", "family", "PDEs", "confidence", "source"],
              [[i, x["family"], "; ".join(x["pdes"]), x["confidence"], x["source"]] for i, x in enumerate(cands)])
    for d in c["decisions"]:
        ctx.report["decisions"].append({**d, "node": ctx.node_id})
    edited = {k2: params[k2] for k2 in ("fields", "pdes", "conditions", "coefficients", "x_range", "t_range")
              if params[k2]}
    for k2, v in edited.items():
        ctx.user_choice({"pdes": "equations", "conditions": "bcs", "coefficients": "coefficients"}.get(k2, "other"),
                        f"{k2} edited by the user", v)
    dim = int(params["dimension"]) or int(c["dimension"])
    td = {"auto": c["time_dependent"], "yes": True, "no": False}[params["time_dependent"]]
    xr = params["x_range"] or list((c.get("bounds") or {}).get("x", [0.0, 1.0]))
    geo = geometry_doc({"dimension": dim, "shape": "box", "x_range": xr, "y_range": xr, "z_range": xr,
                        "time_dependent": td, "t_range": params["t_range"] or list(c.get("time") or [0.0, 1.0])})
    eq = {"fields": params["fields"] or c["fields"], "pdes": params["pdes"] or c["pdes"],
          "conditions": params["conditions"] or c["conditions"],
          "coefficients": {**c["coefficients"], **params["coefficients"]}, "unknowns": [],
          "exact": "" if edited else c.get("exact", ""), "family": c["family"]}
    problem = {"geometry": geo, "equations": eq}
    sys_ = system_of(problem)
    doc = sys_.to_dict()
    if doc["issues"]:
        raise StepFailed("; ".join(doc["issues"]))
    eq["classification"] = doc["classification"]
    ctx.view("equations", doc)
    ctx.metric("accepted", f"#{k} {c['family']} (confidence {c['confidence']})")
    ctx.metric("conservation_law", c.get("conservation_law", ""))
    ctx.metric("constitutive_relation", c.get("constitutive_relation", ""))
    ctx.summary(f"identified {c['family']} ({c['confidence']}), {dim}-D" + (f"; data: {found['pretty']}" if found
                                                                             else ""))
    return {"problem": problem, "geometry": geo, "equations": eq}

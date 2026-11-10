"""Steady compressible flow through an internal passage — valve, orifice, nozzle — solved and checked.

The work is in :func:`pinneapple_simulation.pipelines.compressible_internal.study.run_study`; this
node only picks the passage, the gas and the conditions from the parameters and turns the study into
the metrics, the ``checks`` table and the figures the app reads.

It is one capability for every throttling passage on purpose: the mesher, the solver, the post and
the checks know a *passage*, not a valve. ``preset`` chooses the span list.
"""
from __future__ import annotations

from ..registry import Param, Port, node

PRESETS = ("globe_valve", "orifice", "nozzle")
MESHES = {"grossa": dict(n_theta=8, radial_cells=(6, 5, 4), h_pipe=3.0, h_bowl=2.5, h_gap=1.2),
          "media": dict(n_theta=12, radial_cells=(8, 6, 5), h_pipe=2.5, h_bowl=2.0, h_gap=0.9),
          "fina": dict(n_theta=18, radial_cells=(9, 8, 6), h_pipe=1.8, h_bowl=1.5, h_gap=0.6)}


def _passage(preset: str, parameter: float, malha: str):
    from pinneapple_simulation.pipelines.compressible_internal import (
        converging_nozzle,
        globe_valve,
        orifice_plate,
    )
    m = MESHES.get(malha, MESHES["media"])
    if preset == "orifice":
        return orifice_plate(hole_ratio=parameter, n_theta=m["n_theta"],
                             n_radial=m["radial_cells"][0] + m["radial_cells"][1], h=m["h_pipe"])
    if preset == "nozzle":
        return converging_nozzle(area_ratio=max(parameter, 1.05), n_theta=m["n_theta"],
                                 n_radial=m["radial_cells"][0] + m["radial_cells"][1], h=m["h_pipe"])
    return globe_valve(lift=parameter, **m)


@node("cfd.compressible_passage", "Steady compressible flow in an internal passage, checked",
      category="Design", outputs=[Port("metrics", "metrics")],
      params=[Param("preset", "text", "globe_valve", help="globe_valve | orifice | nozzle"),
              Param("parameter", "float", 4.0,
                    help="opening in mm (valve lift), diameter ratio (orifice) or area ratio (nozzle)"),
              Param("p_in_bar", "float", 70.0, minimum=1.0, maximum=400.0,
                    help="inlet pressure the outer loop drives to"),
              Param("p_out_bar", "float", 60.0, minimum=0.5, maximum=400.0),
              Param("T_in_K", "float", 300.0, minimum=150.0, maximum=900.0),
              Param("gas", "text", "methane", help="methane | natural_gas | air | co2"),
              Param("malha", "text", "media", help="grossa | media | fina"),
              Param("iterations", "int", 900, minimum=100, maximum=8000),
              Param("solves", "int", 3, minimum=1, maximum=5,
                    help="pressure-drop loop passes (fewer is faster, more is closer to the target)"),
              Param("varredura", "bool", False, help="also sweep the opening (3 points)"),
              Param("render", "bool", False, help="Blender render + video (needs Blender; slow)"),
              Param("request", "text", "", max_length=2000)],
      calls="pinneapple_simulation.pipelines.compressible_internal.study.run_study "
            "(Peng-Robinson gas, rhoSimpleFoam, IEC 60534-2-1 and the Joule-Thomson reference)")
def compressible_passage(ctx, inputs, params):
    """A passage, its gas and its conditions in; the flow at the asked-for pressure drop, checked, out."""
    import os

    from pinneapple_simulation.pipelines.compressible_internal import (
        Conditions,
        StudySpec,
        run_study,
    )

    preset = str(params.get("preset") or "globe_valve").strip().lower()
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {', '.join(PRESETS)}; got {preset!r}")
    parameter = float(params.get("parameter") or 4.0)
    cond = Conditions(p_out_bar=float(params.get("p_out_bar") or 60.0),
                      T_in_K=float(params.get("T_in_K") or 300.0),
                      p_in_target_bar=float(params.get("p_in_bar") or 70.0),
                      iterations=int(params.get("iterations") or 900),
                      write_interval=max(int(params.get("iterations") or 900) // 3, 100))
    passage = _passage(preset, parameter, str(params.get("malha") or "media"))
    render = bool(params.get("render"))
    spec = StudySpec(passage, gas=str(params.get("gas") or "methane"), conditions=cond,
                     title=passage.title, render=render, video=render,
                     notes=("the passage is built from its span list; the geometry of the customer's CAD "
                            "is read off section by section in the library's studies module",
                            "adiabatic walls, steady RANS with wall functions"))

    sweep = None
    if params.get("varredura"):
        sweep = ([parameter * 0.25, parameter * 0.5, parameter] if preset == "globe_valve"
                 else [parameter * 0.5, parameter * 0.75, parameter])

    out = run_study(spec, os.path.join(str(ctx.workdir), "passage"), sweep_values=sweep,
                    builder=lambda v: _passage(preset, v, str(params.get("malha") or "media")),
                    max_solves=int(params.get("solves") or 3),
                    log=lambda m: ctx.log(m) if hasattr(ctx, "log") else None)

    n = out["nominal"]
    for key in ("mdot_kg_s", "mdot_kg_h", "p_in_bar", "dp_bar", "T_out_K", "dT_K", "Ma_max", "Kv_m3_h",
                "Q_m3_h", "u_max_m_s", "rho_in_kg_m3"):
        ctx.metric(f"resultado.{key}", n[key], "computed in this run by rhoSimpleFoam")
    ctx.metric("resultado.gas", out["gas"]["name"], "given")
    ctx.metric("resultado.rho_gas_ideal", out["gas"]["rho_ideal"], "for comparison: the ideal gas")
    ctx.metric("dominio.celulas", out["passage"].get("spans"), "spans of the passage")
    for c in out["checks"]:
        ctx.metric(f"check.{c['code']}", c["status"])
    ctx.table("checks", ["check", "status", "detail"],
              [[c["code"], c["status"], c["detail"]] for c in out["checks"]])
    ctx.table("assumptions", ["assumption"], [[a] for a in out["assumptions"]])
    if out.get("curve"):
        ctx.table("curva", list(out["curve"][0].keys()), [list(r.values()) for r in out["curve"]])
    for f in out["files"]:
        if str(f).endswith(".png"):
            ctx.report["figures"].append({"path": str(ctx.path(f)), "name": os.path.basename(f)[:-4],
                                          "caption": os.path.basename(f)})
    ctx.summary(out["summary"])
    return {"metrics": ctx.report["metrics"]}


def build_flow(text: str = "", **params):
    from ..planner.build import FlowBuilder
    title = "Compressible flow in an internal passage: solved, then checked against the physics"
    fb = FlowBuilder(title, "A globe valve, an orifice or a nozzle at a stated pressure drop: the flow "
                            "from rhoSimpleFoam with a Peng-Robinson gas, checked for mass conservation, "
                            "stagnation enthalpy, choking regime, the Joule-Thomson drop and IEC 60534-2-1 "
                            "(pinneapple_simulation.pipelines.compressible_internal).")
    fb.add("passage", "cfd.compressible_passage", dict(params, request=text), label="Internal passage")
    fb.add("report", "export.report", {"title": title}, label="Report")
    fb.connect("passage", "metrics", "report", "items")
    return fb


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="cfd.compressible_passage", title="Compressible flow in a valve, orifice or nozzle, checked",
        domain="cfd",
        keywords={"valve": 2.0, "válvula": 2.0, "valvula": 2.0, "globe valve": 2.5, "válvula globo": 2.5,
                  "orifice": 2.5, "orifício": 2.5, "orificio": 2.5, "nozzle": 2.0, "bocal": 2.0,
                  "choked": 1.5, "choking": 1.5, "afogamento": 1.5, "compressible": 2.0,
                  "compressível": 2.0, "pressure drop": 1.5, "perda de carga": 1.5, "methane": 1.5,
                  "metano": 1.5, "natural gas": 1.5, "gás natural": 1.5, "inlet pressure": 1.0,
                  "pressão de entrada": 1.0, "cfd": 1.0},
        requires_all=(("valve", "válvula", "valvula", "orifice", "orifício", "orificio", "nozzle", "bocal"),
                      ("compressible", "compressível", "pressure drop", "perda de carga", "choked",
                       "choking", "afogamento", "cfd", "gas", "gás")),
        min_score=4, nodes=("cfd.compressible_passage",), build=lambda text, det: build_flow(text),
        inputs=("the passage (globe valve, orifice or nozzle) and its size",
                "the inlet and outlet pressures and the inlet temperature",
                "the gas"),
        uncited="Peng-Robinson (1976), IEC 60534-2-1:2011 and the ISO 20765-2 reference behind the "
                "Joule-Thomson check are cited in the library modules; the flow reference file does not "
                "list them yet",
        description="pinneapple_simulation.pipelines.compressible_internal.study.run_study"))


_register_capabilities()

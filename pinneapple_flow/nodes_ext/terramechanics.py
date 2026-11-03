"""The standard terramechanics workflow (:mod:`pinneapple_simulation.terramechanics.workflow`, #1199): soil from the
database or identified from plate and shear tests through the inverse workflow, single-wheel performance with the
Wong-Reece stresses, the wheel designed for a drawbar pull within slip and sinkage limits at minimum mass,
re-verified with the full model, and the model checked against published single-wheel tests (Ding et al. 2011)."""
from __future__ import annotations

from ..registry import Param, Port, node

WHEEL = {"radius": 0.25, "width": 0.20, "grouser_count": 24, "grouser_height": 0.01}
BOUNDS = {"radius": [0.10, 0.40], "width": [0.08, 0.40], "grouser_height": [0.0, 0.03]}


def _soils() -> list[str]:
    try:
        from pinneapple_simulation.terramechanics import list_soils
        return list(list_soils())
    except Exception:  # noqa: BLE001 - the catalog must load without the simulation package
        return ["Ding2011-TYII"]


@node("terramechanics.workflow", "Terramechanics workflow (soil -> wheel performance -> wheel design -> re-verify)",
      category="Solvers", tags=("section:solvers", "terramechanics", "wheel", "bekker", "wong-reece"),
      outputs=[Port("metrics", "metrics")],
      params=[Param("soil", "enum", "Ding2011-TYII", choices=_soils(), help="database soil (parameters with sources)"),
              Param("identify", "enum", "synthetic", choices=["none", "synthetic", "data"],
                    help="none: the database soil; data: identify from the plate / shear tests below; synthetic: "
                         "identify from tests generated from the database soil (checks the identification)"),
              Param("plates", "dict", {}, help="{width, sinkage, pressure, sigma}: plate tests (m, m, Pa, Pa)",
                    advanced=True),
              Param("shear", "dict", {}, help="{normal_stress, displacement, shear_stress, sigma} (Pa, m, Pa, Pa)",
                    advanced=True),
              Param("load", "float", 80.0, minimum=0.1, help="vertical wheel load (N)"),
              Param("gravity", "float", 9.81, minimum=0.1, help="m/s^2 (bulldozing term; the load is prescribed)"),
              Param("wheel", "dict", WHEEL, help="the wheel the slip curves are computed for (m)"),
              Param("required_pull", "float", 0.0, minimum=0.0,
                    help="drawbar pull the designed wheel must give at the slip limit (N); 0: the given wheel's at "
                         "20 % slip"),
              Param("max_slip", "float", 0.3, minimum=0.01, maximum=0.9),
              Param("max_sinkage", "float", 0.03, minimum=1e-4, help="m"),
              Param("max_torque", "float", 0.0, minimum=0.0, help="N m; 0: no limit"),
              Param("bounds", "dict", BOUNDS, help="design variable -> [low, high] (m)"),
              Param("objective", "enum", "mass", choices=["mass", "torque"]),
              Param("grid", "int", 7, minimum=2, maximum=15, help="grid points per design variable"),
              Param("seed", "int", 0)],
      calls="pinneapple_simulation.terramechanics.workflow.terramechanics_workflow (identify_soil -> "
            "inverse_problems.workflow.inverse_workflow, WongReeceModel.steady_state, design_wheel, testbed.validate)")
def terramechanics_workflow(ctx, inputs, params):
    """Run the workflow and report the soil, the slip curves, the design, the re-verification and the checks."""
    from pinneapple_simulation.terramechanics.workflow import terramechanics_workflow as run
    r = run(soil=str(params["soil"]), identify=str(params["identify"]), plates=dict(params.get("plates") or {}) or None,
            shear=dict(params.get("shear") or {}) or None, load=float(params["load"]), g=float(params["gravity"]),
            wheel=dict(params.get("wheel") or {}) or None, required_pull=float(params["required_pull"]) or None,
            max_slip=float(params["max_slip"]), max_sinkage=float(params["max_sinkage"]),
            max_torque=float(params["max_torque"]) or None, bounds=dict(params.get("bounds") or {}) or None,
            objective=str(params["objective"]), grid=int(params["grid"]), seed=int(params["seed"]))
    ctx.metric("soil", r["soil"]["name"], r["soil_source"])
    for k in ("c", "phi_deg", "K", "k_c", "k_phi", "n0"):
        ctx.metric(f"soil.{k}", r["soil"][k])
    cur = r["curves"]
    ctx.table("single wheel against slip", ["slip", "sinkage [m]", "drawbar pull [N]", "torque [N m]", "DP/W"],
              [[s, round(z, 5), round(d, 3), round(t, 3), round(q, 4)] for s, z, d, t, q in
               zip(cur["slip"], cur["sinkage_m"], cur["drawbar_pull_N"], cur["torque_Nm"], cur["dp_over_w"],
                   strict=True)])
    ctx.metric("requirement.drawbar_pull_N", r["requirement"]["drawbar_pull_N"], "at the slip limit")
    d = r["design"]
    if d.get("feasible"):
        for k, v in d["design"].items():
            ctx.metric(f"design.{k}", v, "minimum-" + r["requirement"]["objective"] + " design")
        ctx.metric("design.mass_kg", d["mass_kg"], "wheel_mass (aluminium rim, sides, lugs, hub)")
        for k, v in r["verified"].items():
            ctx.metric(f"verified.{k}", v, "full model: 96 quadrature nodes, 80 bisections")
    for row in r.get("published", []):
        for k, v in row["rmse_rel"].items():
            ctx.metric(f"published.{row['dataset'].replace(' ', '_')}.{k}_rmse_rel", v, row["source"])
    ctx.table("checks", ["check", "status", "detail"], [[c["code"], c["status"], c["detail"]] for c in r["checks"]])
    ctx.metric("verdict", r["verdict"], ", ".join(r["failed"]) or "every check passes")
    ctx.summary((f"wheel R {d['design']['radius']:.3f} m, b {d['design']['width']:.3f} m, lugs "
                 f"{1e3 * d['design']['grouser_height']:.1f} mm, {d['mass_kg']:.2f} kg for "
                 f"{r['requirement']['drawbar_pull_N']:.1f} N at slip {r['requirement']['max_slip']:g}"
                 if d.get("feasible") else "no feasible wheel") +
                ("; fail: " + ", ".join(r["failed"]) if r["failed"] else "; verified"))
    return {"metrics": ctx.report["metrics"]}


def build_flow(**params):
    from ..planner.build import FlowBuilder
    title = "Terramechanics: soil, wheel performance, wheel design, re-verification"
    fb = FlowBuilder(title, "Soil parameters from the database or identified from plate and shear tests (inverse "
                            "workflow), single-wheel sinkage, drawbar pull and torque against slip (Wong-Reece), "
                            "the wheel radius, width and grouser height for a drawbar pull within slip and sinkage "
                            "limits at minimum mass, re-verified with the full model, and the model checked against "
                            "the published single-wheel tests of Ding et al. (2011) "
                            "(pinneapple_simulation.terramechanics.workflow).")
    fb.add("terra", "terramechanics.workflow", dict(params), label="Terramechanics workflow")
    fb.add("report", "export.report", {"title": title}, label="Report")
    fb.connect("terra", "metrics", "report", "items")
    return fb


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="terramechanics.workflow", title="Terramechanics: wheel-soil performance and wheel design (standard "
                                            "workflow)", domain="vehicles",
        keywords={"terramechanics": 4, "terramecânica": 4, "wheel-soil": 3, "roda-solo": 3, "drawbar pull": 3,
                  "tração na barra": 3, "força de tração": 2, "bekker": 3, "wong-reece": 3, "wong reece": 3,
                  "bevameter": 3, "bevâmetro": 3, "sinkage": 2, "afundamento": 2, "wheel design": 2,
                  "projeto da roda": 2, "dimensionar a roda": 2, "grouser": 2, "garra da roda": 2, "slip": 1,
                  "patinagem": 1, "deslizamento da roda": 2, "regolith": 1, "regolito": 1, "solo deformável": 2,
                  "deformable soil": 2},
        min_score=3, nodes=("terramechanics.workflow",), build=lambda text, det: build_flow(),
        inputs=("a soil (database name) or plate and shear-box tests", "the wheel load and gravity",
                "the drawbar pull, slip and sinkage limits of the wheel"),
        uncited="the workflow cites Bekker (1969), Wong & Reece (1967), Wong (2008), Janosi & Hanamoto (1961) and "
                "Ding et al. (2011) in the library modules; they are not in the flow reference file yet",
        description="pinneapple_simulation.terramechanics.workflow.terramechanics_workflow"))


_register_capabilities()

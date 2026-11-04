"""The standard chip-design workflow (:mod:`pinneapple_design.chip.workflow`, #1206): floorplan with power densities,
die and package thermal (3-D conduction network to the heat sink) with the leakage-temperature loop, power-grid IR
drop with electromigration against Black's equation, Elmore delay of the critical nets, block placement and strap
width designed for the least peak temperature within the IR-drop, lifetime, delay and area limits, and a finer
thermal model that re-verifies it. The bundled flow adds the lithography node of #1180 for the critical layers."""
from __future__ import annotations

from ..registry import Param, Port, node


@node("chip.design", "Chip design: floorplan thermal, IR drop and EM, Elmore delay, placement, re-verified",
      category="Design", outputs=[Port("metrics", "metrics")],
      params=[Param("floorplan", "dict", {}, help="die_mm, slots, blocks (name, dynamic_w, leakage_w), placement, "
                                                    "nets (empty: the default six-block SoC)"),
              Param("t_ambient_c", "float", 45.0), Param("r_sink_k_w", "float", 0.25, minimum=0.0,
                                                           help="heat sink, lid to ambient"),
              Param("t_max_c", "float", 105.0), Param("vdd", "float", 0.8, minimum=0.1),
              Param("ir_drop_max", "float", 0.05, minimum=0.001, maximum=0.5, help="fraction of Vdd"),
              Param("delay_budget_ps", "float", 200.0, minimum=1.0),
              Param("metal_fraction_max", "float", 0.30, minimum=0.01, maximum=1.0),
              Param("mttf_target_years", "float", 10.0, minimum=0.1),
              Param("nx", "int", 24, minimum=4, maximum=96, advanced=True),
              Param("refine", "int", 2, minimum=2, maximum=4, advanced=True)],
      calls="pinneapple_design.chip.workflow.chip_design_workflow")
def chip_design_node(ctx, inputs, params):
    from dataclasses import replace

    from pinneapple_design.chip import GridSpec, chip_design_workflow, default_stack
    stack = replace(default_stack(), t_ambient_c=float(params["t_ambient_c"]), r_sink_k_w=float(params["r_sink_k_w"]))
    r = chip_design_workflow(floorplan=dict(params.get("floorplan") or {}) or None, stack=stack,
                             grid=GridSpec(vdd=float(params["vdd"])), nx=int(params["nx"]),
                             t_max_c=float(params["t_max_c"]), ir_drop_max=float(params["ir_drop_max"]),
                             delay_budget_ps=float(params["delay_budget_ps"]),
                             metal_fraction_max=float(params["metal_fraction_max"]),
                             mttf_target_years=float(params["mttf_target_years"]), refine=int(params["refine"]))
    th = r["thermal"]
    ctx.metric("t_max_c", th["T_max_c"], f"hot spot in {th['hotspot_block']}; initial placement "
                                         f"{r['thermal_initial']['T_max_c']:.2f} C")
    ctx.metric("t_max_refined_c", r["thermal_refined"]["T_max_c"], f"{r['thermal_refined']['cells']} cells")
    ctx.metric("theta_ja_k_w", th["theta_ja_k_w"])
    ctx.metric("leakage_w", th["leakage_w"], f"loop gain {th['loop_gain']:.3g}")
    ctx.metric("strap_width_um", r["strap_width_um"])
    ctx.metric("ir_drop_fraction", r["power_grid"]["max_drop_fraction"])
    ctx.metric("em_mttf_min_years", r["power_grid"]["mttf_min_years"])
    ctx.table("placement", ["block", "slot", "power density W/mm^2", "T max C"],
              [[b["name"], b["slot"], b["power_density_w_mm2"], th["block_t_max_c"][b["name"]]]
               for b in r["floorplan"]["blocks"]])
    ctx.table("critical nets", ["net", "length mm", "wire C", "Elmore ps"],
              [[k, v["length_mm"], v["t_wire_c"], v["elmore_s"] * 1e12] for k, v in r["net_delays"].items()])
    ctx.table("checks", ["check", "status", "detail"], [[c["code"], c["status"], c["detail"]] for c in r["checks"]])
    ctx.metric("verdict", r["verdict"], ", ".join(r["failed"]) or "every check passes")
    ctx.summary(r["summary"])
    return {"metrics": ctx.report["metrics"]}


def build_flow(**params):
    from ..planner.build import FlowBuilder
    title = "Chip design: floorplan thermal, IR drop and electromigration, Elmore delay, lithography"
    fb = FlowBuilder(title, "Floorplan blocks with power densities; a 3-D die and package conduction network to the "
                            "heat sink with the leakage-temperature loop; power-grid IR drop with electromigration "
                            "(Black); Elmore delay of the critical nets checked by a trapezoidal transient; block "
                            "placement and strap width designed and re-verified on a finer thermal model "
                            "(pinneapple_design.chip.workflow); the critical layer's mask corrected and verified "
                            "over the process window (litho.opc, #1180).")
    fb.add("chip", "chip.design", dict(params), label="Chip design")
    fb.add("litho", "litho.opc", {}, label="Lithography of the critical layer")
    fb.add("report", "export.report", {"title": title}, label="Report")
    fb.connect("chip", "metrics", "report", "items")
    fb.connect("litho", "metrics", "report", "items")
    return fb


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="chip.design", title="Chip design (standard workflow)", domain="electronics",
        keywords={"chip": 2, "die": 1, "floorplan": 3, "planta baixa do chip": 3, "ir drop": 3,
                  "queda ir": 3, "queda de tensão": 2, "power grid": 3, "rede de alimentação": 3,
                  "electromigration": 3, "eletromigração": 3, "elmore": 3, "interconnect": 2, "interconexão": 2,
                  "hot spot": 2, "ponto quente": 2, "junction temperature": 2, "temperatura de junção": 2,
                  "leakage": 2, "fuga": 1, "soc": 2, "vlsi": 3, "semiconductor": 2, "semicondutor": 2,
                  "package thermal": 3, "térmica do encapsulamento": 3},
        min_score=3, nodes=("chip.design", "litho.opc"), build=lambda text, det: build_flow(),
        inputs=("the floorplan: blocks, powers, leakage, slots and critical nets", "the die stack and heat sink",
                "the limits: junction temperature, IR drop, delay budget, EM lifetime, metal area"),
        uncited="the method cites Black (1969), Sakurai & Tamaru (1983) and Gupta, Tutuianu & Pileggi (1997) in the "
                "library modules; they are not in the flow reference file yet",
        description="pinneapple_design.chip.workflow.chip_design_workflow"))


_register_capabilities()

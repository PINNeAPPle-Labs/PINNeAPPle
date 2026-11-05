"""Joule–Thomson cooling of natural gas across a globe valve, from a request in words
(:mod:`pinneapple_systems.process_components.joule_thomson`, #1463)."""
from __future__ import annotations

from ..registry import Param, Port, node


@node("process.valve_joule_thomson", "Joule–Thomson cooling across a valve (natural gas, real-gas EOS)",
      category="Design", outputs=[Port("metrics", "metrics")],
      params=[Param("request", "text", "", max_length=2000,
                    help="the case in words: inlet and outlet pressures, inlet temperature, Cv, pipe diameter, "
                         "composition, and optionally the outlet temperature a CFD run gave")],
      calls="pinneapple_systems.process_components.joule_thomson.valve_joule_thomson_study "
            "(real_gas_eos GERG-2008, control_valve IEC 60534-2-1, pinneapple_veriphysics.policy.decide)")
def valve_joule_thomson(ctx, inputs, params):
    """Outlet temperature by four equations of state, the isenthalpic path, valve flow, and the CFD check."""
    from pinneapple_systems.process_components.joule_thomson import valve_joule_thomson_study
    r = valve_joule_thomson_study(params.get("request") or "", out_dir=str(ctx.workdir))
    c = r["case"]
    for k, v in c["values"].items():
        ctx.metric(f"case.{k}", v, "default (not in the text)" if k in c["defaulted"] else "read from the text")
    ctx.metric("case.composition", ", ".join(f"{k} {100 * v:.1f} %" for k, v in c["composition"].items()),
               "illustrative default" if "composition" in c["defaulted"] else "read from the text")
    for k, v in r["outputs"].items():
        ctx.metric(k, v, "computed in this run")
    for f in r["files"]:
        ctx.report["figures"].append({"path": str(ctx.path(f)), "name": f[:-4], "caption": f})
    ctx.table("checks", ["check", "status", "detail"], [[x["code"], x["status"], x["detail"]] for x in r["checks"]])
    ctx.table("assumptions", ["assumption"], [[a] for a in r["assumptions"]])
    o = r["outputs"]
    ctx.summary(f"{c['values']['P1_bar']:g} → {c['values']['P2_bar']:g} bar from {c['values']['T1_C']:g} °C: "
                f"outlet {o['T2_C']:.1f} °C ({o['dT_K']:+.1f} K, GERG-2008); Peng–Robinson "
                f"{o['dT_K[Peng-Robinson]']:+.1f} K, ideal gas {o['dT_K[ideal gas]']:+.1f} K; "
                f"VeriPhysics {r['decision']['verdict']}")
    return {"metrics": ctx.report["metrics"]}


def build_flow(text: str = "", **params):
    from ..planner.build import FlowBuilder
    title = "Joule–Thomson cooling across a valve: real-gas outlet temperature, checked"
    fb = FlowBuilder(title, "The case is read from the text; the outlet temperature from GERG-2008, Peng–Robinson, "
                            "SRK and the ideal gas, the isenthalpic path, the valve flow by IEC 60534-2-1, and a "
                            "check of a CFD result (pinneapple_systems.process_components.joule_thomson).")
    fb.add("valve", "process.valve_joule_thomson", dict(params, request=text), label="Valve Joule–Thomson")
    fb.add("report", "export.report", {"title": title}, label="Report")
    fb.connect("valve", "metrics", "report", "items")
    return fb


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="process.valve_joule_thomson", title="Joule–Thomson cooling across a valve (natural gas)",
        domain="process",
        keywords={"joule-thomson": 3, "joule thomson": 3, "joule–thomson": 3, "valve": 1.5, "válvula": 1.5,
                  "valvula": 1.5, "globe valve": 1.5, "válvula globo": 1.5, "natural gas": 1.5, "gás natural": 1.5,
                  "throttling": 1.5, "estrangulamento": 1.5, "laminação": 1, "temperature drop": 1,
                  "queda de temperatura": 1, "real gas": 1, "gás real": 1, "cfd": 0.5},
        requires_all=(("joule", "throttl", "estrangul", "laminação"),
                      ("valve", "válvula", "valvula", "throttl", "estrangul")),
        min_score=4, nodes=("process.valve_joule_thomson",), build=lambda text, det: build_flow(text),
        inputs=("the inlet and outlet pressures and the inlet temperature", "the gas composition",
                "the valve Cv and the pipe diameter", "optionally, the outlet temperature your CFD gives"),
        uncited="GERG-2008 (Kunz & Wagner 2012) and IEC 60534-2-1 are cited in the library modules; the flow "
                "reference file does not list them yet",
        description="pinneapple_systems.process_components.joule_thomson.valve_joule_thomson_study"))


_register_capabilities()

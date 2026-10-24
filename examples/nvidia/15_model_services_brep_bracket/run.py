"""From a sentence to an exact, checked part: a model service reads the request, NVIDIA USD BRep builds the bracket,
PINNeAPPle sizes and verifies it.

Two requests go to ``pinneapple-intent`` on PINNeAPPle Model Services (OpenAI-format ``/v1/chat/completions`` with
``nvext.guided_json``, the same call NIM takes). The service is the deterministic extractor of
``pinneapple_design.intent``: every value comes with the quote it was read from, and its trust report says when a
number could not be placed. The client builds only what the service APPROVED; for an ambiguous request it shows the
reason instead. For the accepted request PINNeAPPle sizes the plate thickness from stock sizes with cantilever theory
and the safety factor, USD BRep builds the exact L bracket (inner fillet r = t, two M6 clearance holes), and
PINNeAPPle verifies it: exact mass and geometry, and its in-house hexahedral finite elements (C3D8I) against the beam
stress and deflection.

    PINNEAPPLE_BREP_PYTHON=/opt/brep/bin/python python examples/nvidia/15_model_services_brep_bracket/run.py    # ~30 s
"""
import argparse
import json
import os

from pinneapple_design.intent import materials
from pinneapple_design.intent.physics import bracket as B
from pinneapple_lab import experiment
from pinneapple_tools.model_services import ModelServiceClient, default_registry, serve_in_thread
from pinneapple_tools.nvidia.brep import evaluate_part, geometry_checks
from pinneapple_tools.nvidia.diagram import Block, architecture_diagram
from pinneapple_tools.nvidia.figures import checks_figure
from pinneapple_tools.nvidia.parts import l_bracket, l_bracket_volume
from pinneapple_tools.nvidia.runinfo import RunInfo

HERE = os.path.dirname(os.path.abspath(__file__))
REQUESTS = ["An aluminium shelf bracket that holds 1.5 kN at 120 mm from the wall, safety factor 2",
            "A bracket in aluminium holding 1.5 kN on a 120 mm arm, 50 mm wide, safety factor 2"]
SCHEMA = {"type": "object", "required": ["template", "values"],
          "properties": {"template": {"type": "string"}, "values": {"type": "object", "required": ["load", "arm_length"]}}}
STOCK_T = (0.003, 0.004, 0.005, 0.006, 0.008, 0.010, 0.012, 0.016)    # plate stock [m]
WIDTH, LEG, HOLE = 0.08, 0.06, 0.0066                          # chosen width and mounting leg; ISO 273 M6 clearance


def main(smoke=False, base=None):
    base = base or HERE
    OUT = os.path.join(base, "results")
    os.makedirs(OUT, exist_ok=True)
    info = RunInfo("15_model_services_brep_bracket")
    architecture_diagram(
        "15 From a sentence to an exact, checked part: PINNeAPPle + Model Services + USD BRep",
        [Block("txt", "Engineering request\n(plain language)", "data", 0, 0),
         Block("svc", "pinneapple-intent on Model Services\n/v1/chat/completions + guided_json", "services", 1, 0),
         Block("trust", "Trust of the extraction: quotes,\nunplaced numbers -> decision", "pinneapple", 1, 1),
         Block("size", "Sizing: beam theory, yield / SF,\nstock thickness", "pinneapple", 2, 1),
         Block("brep", "USD BRep: exact L bracket,\nfillet r = t, M6 holes", "brep", 2, 0),
         Block("ver", "Verify: exact mass, geometry,\nhexahedral FE vs beam", "pinneapple", 3, 1),
         Block("nim", "NVIDIA NIM LLM (same route,\nNVIDIA_API_KEY)", "nim", 3, 0, ran=False)],
        [("txt", "svc", "messages"), ("svc", "trust", "spec"), ("trust", "size", "APPROVED spec"),
         ("size", "brep", "recipe"), ("brep", "ver", "solid, mesh"), ("svc", "nim", "same client")],
        os.path.join(OUT, "architecture.png"),
        subtitle="Nothing is built from a number the service could not place")
    al = materials.get("al6061_t6")
    E, nu, fy, rho = (al.props[k].value for k in ("youngs_modulus", "poisson", "yield_strength", "density"))
    with experiment("examples/nvidia/15_model_services_brep_bracket", title="Request to exact bracket: model service, "
                    "USD BRep, PINNeAPPle sizing and verification", domain="structural mechanics",
                    methods=["intent extraction", "B-rep (SMLib)", "beam theory", "FEM (C3D8I)"], out_dir=base,
                    register=False, modules=["pinneapple_tools.model_services", "pinneapple_tools.nvidia.brep"]) as ex:
        srv, url = serve_in_thread(default_registry(zoo=False))
        c = ModelServiceClient(url)
        answers = []
        try:
            with info.step("extraction (Model Services)"):
                for text in REQUESTS:
                    r = c.chat("pinneapple-intent", [{"role": "user", "content": text}], guided_json=SCHEMA)
                    spec = ModelServiceClient.content_json(r)
                    answers.append({"text": text, "decision": r["pinneapple"]["trust"]["decision"],
                                    "schema_errors": r["pinneapple"]["guided_json_errors"],
                                    "values": {k: (v["value"], v["unit"], v["quote"]) for k, v in spec["values"].items()
                                               if v["value"] is not None}, "unsupported": spec.get("unsupported", []),
                                    "latency_ms": c.last_latency_ms})
        finally:
            srv.shutdown()
        acc = answers[0]
        P, arm, sf = acc["values"]["load"][0], acc["values"]["arm_length"][0], acc["values"]["safety_factor"][0]
        sig_allow = fy / sf
        sized = [(t, *B.beam(P, arm, WIDTH, t, E)) for t in STOCK_T]
        t, sig_b, dl_b = next(s for s in sized if s[1] <= sig_allow)
        dims = dict(base=LEG, height=arm, width=WIDTH, t=t, fillet=t, bolt_d=HOLE, web_hole_d=0.0,
                    bolt_y=(WIDTH / 4, 3 * WIDTH / 4), bolt_x=LEG / 2 + t / 2)
        with info.step("USD BRep: exact bracket"):
            rep = evaluate_part(l_bracket(**dims), chord_tolerances=[1e-4, 2e-5], max_edge_length=0.0012, density=rho)
        with info.step("hexahedral FE (C3D8I)"):
            d_fe, s_fe, _, mesh = B.hex_model(P, LEG, arm, WIDTH, t, E, nu, n_t=2 if smoke else 3)
        vol_cf = l_bracket_volume(**dims)
        checks = [{"name": "request 1: extraction APPROVED, schema satisfied", "value": None, "limit": None,
                   "passed": acc["decision"] == "APPROVED" and not acc["schema_errors"]},
                  {"name": "request 2 (ambiguous): not APPROVED, nothing built", "value": None, "limit": None,
                   "passed": answers[1]["decision"] != "APPROVED", "detail": "; ".join(answers[1]["unsupported"])},
                  {"name": "nominal root stress <= yield / SF (beam)", "value": sig_b / 1e6, "limit": sig_allow / 1e6,
                   "passed": sig_b <= sig_allow, "detail": f"t = {t * 1e3:g} mm from stock"},
                  {"name": "exact volume = closed form", "value": abs(rep.volume - vol_cf) / vol_cf, "limit": 1e-6,
                   "passed": abs(rep.volume - vol_cf) / vol_cf < 1e-6},
                  *geometry_checks(rep, element_size=0.0012),
                  {"name": "FE root stress vs beam", "value": abs(s_fe - sig_b) / sig_b, "limit": 0.10,
                   "passed": abs(s_fe - sig_b) / sig_b <= 0.10},
                  {"name": "FE tip deflection vs beam (FE includes the leg's rotation)", "value": d_fe / dl_b,
                   "limit": None, "passed": d_fe >= dl_b, "detail": f"{d_fe * 1e3:.3f} vs {dl_b * 1e3:.3f} mm"}]
        decision = "APPROVED" if all(c_["passed"] for c_ in checks) else "REVIEW"
        for c_ in checks:
            ex.check(c_["name"], c_["passed"], value=c_["value"], max=c_["limit"], detail=c_.get("detail", ""))
        ex.log_metric("thickness_mm", t * 1e3, "mm", "smallest stock plate with beam stress <= yield / SF")
        ex.log_metric("bracket_mass", rep.raw["mass"]["mass"] * 1e3, "g", "exact B-rep mass, 6061 density")
        info.mark("PINNeAPPle Model Services (pinneapple-intent)", ran=True)
        info.mark("NVIDIA USD BRep", ran=True)
        info.mark("NVIDIA NIM LLM", ran=False, why="needs NVIDIA_API_KEY")
        _figures(OUT, rep, answers, checks, decision, t, P, arm)
        summary = {"decision": decision, "answers": answers, "load_N": P, "arm_m": arm, "safety_factor": sf,
                   "thickness_mm": t * 1e3, "sigma_beam_MPa": sig_b / 1e6, "sigma_fe_MPa": s_fe / 1e6,
                   "deflection_beam_mm": dl_b * 1e3, "deflection_fe_mm": d_fe * 1e3, "fe_nodes": int(len(mesh.nodes)),
                   "mass_g": rep.raw["mass"]["mass"] * 1e3, "min_radius_mm": rep.min_radius * 1e3,
                   "tessellation": rep.tessellations[-1], "checks": checks}
        with open(os.path.join(OUT, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=float)
        info.save(os.path.join(OUT, "run_info.json"))
    print(json.dumps({k: summary[k] for k in ("decision", "thickness_mm", "sigma_beam_MPa", "sigma_fe_MPa", "mass_g")},
                     indent=1))
    return summary


def _figures(OUT, rep, answers, checks, decision, t, P, arm):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(12.5, 4.6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    V, F = rep.mesh(-1)
    ax.add_collection3d(Poly3DCollection(V[F] * 1e3, facecolor="#b8c4d6", edgecolor="#555555", linewidths=0.05))
    ax.set_xlim(-10, 70)
    ax.set_ylim(-10, 90)
    ax.set_zlim(0, 130)
    ax.set_box_aspect((80, 100, 130))
    ax.view_init(22, -55)
    ax.set_title(f"USD BRep: {t * 1e3:g} mm plate, fillet r = t, 2 x M6 holes ({rep.raw['mass']['mass'] * 1e3:.0f} g)",
                 fontsize=9)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis("off")
    y = 0.95
    for i, a in enumerate(answers):
        col = "#2e9d4f" if a["decision"] == "APPROVED" else "#e0a000"
        ax2.text(0, y, f"Request {i + 1}: \"{a['text']}\"", fontsize=8, wrap=True)
        y -= 0.09
        vals = ", ".join(f"{k} = {v[0]:g} {v[1]} ('{v[2]}')" for k, v in a["values"].items()
                         if k in ("load", "arm_length", "safety_factor"))
        ax2.text(0.03, y, vals, fontsize=7.5, color="#333333")
        y -= 0.07
        ax2.text(0.03, y, f"service decision: {a['decision']}" + (f"  ({'; '.join(a['unsupported'])})"
                                                                   if a["unsupported"] else ""),
                 fontsize=8, color=col, weight="bold", wrap=True)
        y -= 0.16
    ax2.text(0, y, f"Built from request 1: P = {P / 1e3:g} kN at {arm * 1e3:g} mm, design {decision}", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "request_to_part.png"), dpi=130)
    plt.close(fig)
    checks_figure(checks, os.path.join(OUT, "checks.png"), title="Request, geometry and stress checks",
                  decision=decision)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(ap.parse_args().smoke)

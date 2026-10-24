"""Exact CAD into Omniverse: an L bracket built in NVIDIA USD BRep goes to OpenUSD twice, as its native B-rep and as
a mesh carrying the exact curvature and the wall thickness, and PINNeAPPle checks both before anyone opens them.

The bracket (6 mm 6061-T6 plate, 5 mm inner fillet, two 9 mm bolt holes, a 12 mm hole in the web) is a recipe of
B-rep operations. USD BRep builds it, gives its exact volume and mass, exports the native B-rep USD
(``BrepArray`` prims) and tessellates it. PINNeAPPle checks the exact volume against the closed form, measures the
mesh's chord error and volume error, reads the exact curvature at every mesh vertex (so the radii in the scene are
the CAD's, not the mesh's), casts rays for the wall thickness, and writes an OpenUSD stage for Omniverse that
references the native B-rep and carries the mesh with those fields and the trust decision.

    PINNEAPPLE_BREP_PYTHON=/opt/brep/bin/python python examples/nvidia/14_brep_omniverse_bracket/run.py   # ~15 s
"""
import argparse
import json
import os
import re

import numpy as np

from pinneapple_design.intent import materials
from pinneapple_lab import experiment
from pinneapple_tools.nvidia.brep import evaluate_part, geometry_checks, mesh_thickness
from pinneapple_tools.nvidia.diagram import Block, architecture_diagram
from pinneapple_tools.nvidia.figures import checks_figure
from pinneapple_tools.nvidia.omniverse import (
    FieldStage,
    kit_render_script,
    render_stage_png,
    validate_stage,
)
from pinneapple_tools.nvidia.parts import l_bracket, l_bracket_volume
from pinneapple_tools.nvidia.runinfo import RunInfo

HERE = os.path.dirname(os.path.abspath(__file__))
DIMS = dict(base=0.08, height=0.06, width=0.05, t=0.006, fillet=0.005, bolt_d=0.009, web_hole_d=0.012)


def main(smoke=False, base=None):
    base = base or HERE
    OUT = os.path.join(base, "results")
    os.makedirs(OUT, exist_ok=True)
    info = RunInfo("14_brep_omniverse_bracket")
    architecture_diagram(
        "14 Exact CAD into Omniverse: PINNeAPPle + NVIDIA USD BRep + Omniverse",
        [Block("rec", "L-bracket recipe (data):\nunion, fillet, three holes", "pinneapple", 0, 0),
         Block("brep", "USD BRep: exact solid,\nmass, tessellation", "brep", 1, 0),
         Block("native", "Native B-rep USD\n(BrepArray prims)", "brep", 2, 0),
         Block("fields", "Exact curvature per vertex,\nwall thickness (ray casting)", "pinneapple", 1, 1),
         Block("chk", "Checks: volume vs closed form,\nchord error, stage metadata", "pinneapple", 2, 1),
         Block("stage", "OpenUSD stage: references the\nB-rep + mesh with fields + trust", "omniverse", 3, 0),
         Block("val", "UsdValidation + read-back\npreviews", "omniverse", 3, 1),
         Block("kit", "Omniverse Kit RTX", "omniverse", 4, 0, ran=False)],
        [("rec", "brep", "ops"), ("brep", "native", "export_brep"), ("brep", "fields", "mesh + surfaces"),
         ("fields", "chk", ""), ("native", "stage", "reference"), ("chk", "stage", "fields, decision"),
         ("stage", "val", ".usda"), ("stage", "kit", ".usda")],
        os.path.join(OUT, "architecture.png"),
        subtitle="The CAD stays exact in the scene; PINNeAPPle adds what the mesh alone cannot tell (radii, walls)")
    al = materials.get("al6061_t6")
    with experiment("examples/nvidia/14_brep_omniverse_bracket", title="L bracket: USD BRep geometry to Omniverse with "
                    "exact curvature and wall thickness", domain="geometry", methods=["B-rep (SMLib)", "OpenUSD"],
                    out_dir=base, register=False, modules=["pinneapple_tools.nvidia.brep", "pinneapple_tools.nvidia.parts",
                                                           "pinneapple_tools.nvidia.omniverse"], params=DIMS) as ex:
        native = os.path.join(OUT, "bracket_brep.usda")
        with info.step("USD BRep: solid, tessellation, curvature, native USD"):
            rep = evaluate_part(l_bracket(**DIMS), chord_tolerances=[1e-4, 2e-5], max_edge_length=0.004 if smoke else 0.0015,
                                density=al.props["density"].value, usd_brep_path=native, vertex_curvature=True)
        V, F = rep.mesh()
        with info.step("wall thickness (ray casting)"):
            th = mesh_thickness(V, F)
        k = np.asarray(rep.vertex_curvature)
        radius = np.where(k > 1.0, 1.0 / np.maximum(k, 1e-12), np.inf)
        vol_cf = l_bracket_volume(**DIMS)
        radii = sorted({round(float(r), 5) for r in radius[np.isfinite(radius)]})       # 0.01 mm bins
        expected = sorted({DIMS["bolt_d"] / 2, DIMS["fillet"], DIMS["web_hole_d"] / 2})
        wall = th[np.isfinite(th)]
        nat = validate_stage(native)
        with info.step("OpenUSD stage for Omniverse"):
            usd = os.path.join(OUT, "bracket.usda")
            st = FieldStage(usd, "L bracket: USD BRep geometry with PINNeAPPle fields")
            ref = st.stage.DefinePrim("/World/BRep")
            ref.GetReferences().AddReference(os.path.basename(native), "/World")
            th_show = np.minimum(th, 0.03)
            m = st.add_mesh("BracketMesh", V, F, fields={"curvature": k, "wall_thickness": th_show,
                                                         "brep_face": rep.vertex_face.astype(float)},
                            units={"curvature": "1/m", "wall_thickness": "m", "brep_face": "index"},
                            color_by="curvature", cmap="viridis", vrange=(0.0, float(k.max())))
            g_checks = geometry_checks(rep, element_size=0.004 if smoke else 0.0015)
            checks = [{"name": "exact volume = closed form (holes, fillet)", "value": abs(rep.volume - vol_cf) / vol_cf,
                       "limit": 1e-6, "passed": abs(rep.volume - vol_cf) / vol_cf < 1e-6},
                      {"name": "radii in the mesh are the CAD radii", "value": None, "limit": None,
                       "passed": np.allclose(radii, expected, rtol=1e-3) if len(radii) == 3 else False,
                       "detail": f"{[round(r * 1e3, 3) for r in radii]} mm"},
                      {"name": "thinnest wall = plate thickness", "value": float(wall.min()), "limit": DIMS["t"],
                       "passed": abs(wall.min() - DIMS["t"]) / DIMS["t"] < 1e-3},
                      *g_checks,
                      {"name": "native B-rep USD opens with OpenUSD", "value": None, "limit": None,
                       "passed": "does not open" not in " ".join(nat["problems"])}]
            st.set_trust(m, "APPROVED" if all(c["passed"] for c in checks) else "REJECT", checks,
                         source="NVIDIA USD BRep (SMLib)", mass_kg=rep.raw["mass"]["mass"])
            st.save()
            v = validate_stage(usd, required_primvars={m: ["curvature", "wall_thickness"]})
        checks.append({"name": "Omniverse stage valid (with the B-rep referenced)", "value": None, "limit": None,
                       "passed": v["ok"], "detail": "; ".join(v["problems"]) or "UsdValidation clean"})
        checks.append({"name": "native B-rep layer has stage metadata", "value": None, "limit": None,
                       "passed": nat["ok"], "detail": "; ".join(re.sub(r"<[^>]*>", "<layer>", p.split(":", 1)[-1]).strip()
                                                               for p in nat["problems"]) or "clean"})
        core = [c for c in checks if not c["name"].startswith("native B-rep layer")]
        decision = "APPROVED" if all(c["passed"] for c in core) else "REJECT"
        for c in checks:
            ex.check(c["name"], c["passed"], value=c["value"], max=c["limit"], detail=c.get("detail", ""))
        ex.log_metric("bracket_mass", rep.raw["mass"]["mass"] * 1e3, "g", "USD BRep mass properties, 6061-T6 density")
        ex.log_metric("chord_error_max", rep.tessellations[-1]["max_deviation"] * 1e6, "um",
                      "largest triangle-centroid distance to the exact surface")
        info.mark("NVIDIA USD BRep", ran=True, how="usd-brep 1.0.1 (CPython 3.12 worker)")
        info.mark("Omniverse Kit RTX render", ran=False, why="needs Omniverse Kit and an RTX GPU")
        render_stage_png(usd, os.path.join(OUT, "usd_curvature.png"), elev=30, azim=-50,
                         colorbar=("exact curvature [1/m]", 0.0, float(k.max()), "viridis"),
                         title="bracket.usda read back: exact curvature from the B-rep at each vertex")
        st2 = FieldStage(os.path.join(OUT, "bracket_thickness.usda"), "wall thickness view")
        st2.add_mesh("BracketMesh", V, F, fields={"wall_thickness": th_show}, color_by="wall_thickness", cmap="magma",
                     vrange=(0.0, 0.03))
        st2.save()
        render_stage_png(st2.path, os.path.join(OUT, "usd_thickness.png"), elev=30, azim=-50,
                         colorbar=("wall thickness [m] (capped at 30 mm)", 0.0, 0.03, "magma"),
                         title="wall thickness by ray casting on the B-rep mesh")
        with open(os.path.join(OUT, "kit_render.py"), "w") as f:
            f.write(kit_render_script(usd, os.path.join(OUT, "bracket_rtx.png")))
        checks_figure(checks, os.path.join(OUT, "checks.png"), title="Exact geometry in the scene", decision=decision)
        summary = {"decision": decision, "volume_m3": rep.volume, "volume_closed_form_m3": vol_cf,
                   "mass": rep.raw["mass"], "radii_mm": [r * 1e3 for r in radii], "thinnest_wall_mm": float(wall.min()) * 1e3,
                   "tessellations": rep.tessellations, "faces": rep.raw["faces"], "mesh_vertices": int(len(V)),
                   "native_brep_validation": {k2: nat[k2] for k2 in ("ok", "problems")},
                   "stage": {k2: v[k2] for k2 in ("ok", "problems", "meshes", "prims")}, "checks": checks}
        with open(os.path.join(OUT, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=float)
        info.save(os.path.join(OUT, "run_info.json"))
    print(json.dumps({k2: summary[k2] for k2 in ("decision", "radii_mm", "thinnest_wall_mm")}, indent=1))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(ap.parse_args().smoke)

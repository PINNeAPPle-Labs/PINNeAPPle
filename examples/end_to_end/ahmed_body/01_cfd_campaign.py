"""Stage 1 of the Ahmed-body study: geometry, mesh study, validation sweep and the design of experiments in OpenFOAM.

    python examples/end_to_end/ahmed_body/01_cfd_campaign.py [stage ...] [--procs 4]

Stages (default: all, in this order; each is resumable, finished runs are skipped):

* ``cad``        the body as OpenCASCADE CAD (STEP + STL) imported back through the STL path and checked;
* ``mesh``       the baseline (25 deg, 50 mm, 60 m/s) on the L0, L1 and L2 meshes (1000 iterations) for the GCI;
* ``validation`` the slant angles Ahmed et al. (1984) measured, at their 60 m/s and 50 mm, on the campaign mesh;
* ``doe``        48 maximin Latin-hypercube designs over slant 0-40 deg, clearance 30-80 mm, speed 20-60 m/s;
* ``cad_run``    the imported CAD body at the baseline, to compare with the parametric one;
* ``mesh_fine``  the L3 mesh (1.3 M cells, about an hour on 4 cores) for the asymptotic-range check.

Everything goes to ``_runs/cfd`` (records, fields, flow scenes; not committed) and the DOE plan to
``results/doe_plan.json``. Needs OpenFOAM (``FOAM_BASHRC``) and, for ``cad``, the gmsh binary.
"""
import json
import os
import sys

import numpy as np

from pinneapple_autolab.domains import ahmed as A
from pinneapple_autolab.study import campaign as C

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "_runs", "cfd")
RES = os.path.join(HERE, "results")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 4
stages = args or ["cad", "mesh", "validation", "doe", "cad_run", "mesh_fine"]
os.makedirs(RES, exist_ok=True)


def cad():
    files = A.ahmed_cad(A.BASELINE, os.path.join(RES, "cad"))
    body = A.import_body(files["stl"], A.BASELINE)
    V, F = A.make_body(A.BASELINE)
    from pinneapple_design.geometry_library import meshops as mo
    out = {"files": {k: os.path.relpath(v, HERE) for k, v in files.items()}, "issues": body["issues"],
           "triangles": len(body["F"]), "volume_m3": body["volume"], "parametric_volume_m3": mo.volume(V, F),
           "symmetry_mm": body["symmetry_mm"], "bounds": body["bounds"],
           "patches": {r: int((body["regions"] == r).sum()) for r in A.REGIONS}}
    json.dump(out, open(os.path.join(RES, "cad_import.json"), "w"), indent=1)
    print("CAD import:", out["issues"] or "no issues", f"{out['triangles']} triangles, patches {out['patches']}")
    return body


def run(sim, cases, purpose):
    return C.run_campaign(sim, cases, RUNS, purpose=purpose)


for st in stages:
    if st == "cad":
        cad()
    elif st in ("mesh", "mesh_fine"):
        for m in (["L0", "L1", "L2"] if st == "mesh" else ["L3"]):
            run(A.AhmedAero(mesh=m, iterations=1000, procs=procs), [{"id": f"mesh_{m}", **A.BASELINE}], "probe")
    elif st == "validation":
        cases = [{"id": f"val_{s:04.1f}", "slant_deg": s, "clearance": 0.05, "speed": 60.0} for s in A.AHMED_1984_CD]
        run(A.AhmedAero(procs=procs), cases, "baseline")
    elif st == "doe":
        plan = C.plan_doe(C.space_from_variables(A.VARIABLES), 48, "maximin", seed=0)
        json.dump({k: v for k, v in plan.items() if k != "unit"}, open(os.path.join(RES, "doe_plan.json"), "w"),
                  indent=1, default=float)
        cov, rnd = plan["coverage"], plan["random_reference"]
        print(f"DOE: discrepancy {cov['discrepancy']:.4f} (random {rnd['discrepancy']:.4f}), "
              f"min distance {cov['min_distance']:.3f} (random {rnd['min_distance']:.3f})")
        run(A.AhmedAero(procs=procs), plan["cases"], "doe")
    elif st == "cad_run":
        body = cad()
        sim = A.AhmedAero(procs=procs, geometry=lambda p, b=body: (b["V"], b["F"]))
        run(sim, [{"id": "cad_baseline", **A.BASELINE}], "probe")
    else:
        raise SystemExit(f"unknown stage {st!r}")
recs = C.load_records(RUNS)
print(f"{sum(r.ok for r in recs)} of {len(recs)} runs ok; cost {C.cost_table(recs)['total']}")
np.seterr(all="ignore")
os._exit(0)                                       # OpenMPI threads can keep the interpreter alive

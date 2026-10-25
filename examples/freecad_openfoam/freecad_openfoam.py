"""FreeCAD -> OpenFOAM pipelines: three validated worked studies, a parameter sweep and a surrogate.

    python examples/freecad_openfoam/freecad_openfoam.py            # about an hour on 4 cores; resumes if stopped
    python examples/freecad_openfoam/freecad_openfoam.py --quick    # coarse meshes and a small sweep

1. Pipe bend (laminar, Re 200, Rc / D = 4): the straight leg's friction factor against Hagen-Poiseuille and the
   developed curved/straight pressure-gradient ratio against White (1929), on 3 meshes + Richardson.
2. Heated pipe (laminar thermal entry, Re 100, Pr 2): mean Nusselt number against Hausen (1943), 3 meshes.
3. Sphere (Re 100, laminar, half model): drag coefficient against Clift, Grace & Weber (1978), 3 surface levels,
   then Re 50 and 200 on the middle mesh.
4. Sweep over the FreeCAD parameters of the bend (Rc, angle; Latin hypercube, one process per case, resumable) and a
   surrogate of the total pressure drop measured on held-out cases.

Every number written comes from the OpenFOAM runs this script makes (folders in ``_work/``, ``study_result.json``
and ``report.html`` per run); the figures and tables go to this folder.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import pinneapple_lab
from pinneapple_simulation.pipelines.freecad_openfoam import Sweep, doe, fit_surrogate, studies
from pinneapple_simulation.pipelines.freecad_openfoam.report import (
    mesh_study,
    mesh_study_figure,
    sweep_figure,
    validation_markdown,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--work", default=os.path.join(HERE, "_work"))
    ap.add_argument("--sweep", type=int, default=20)
    ap.add_argument("--parallel", type=int, default=2)
    a = ap.parse_args()
    q = a.quick
    fig = os.path.join(HERE, "figures")
    os.makedirs(fig, exist_ok=True)
    with pinneapple_lab.experiment(
            "examples/freecad_openfoam", title="FreeCAD -> OpenFOAM pipelines (bend, heated pipe, sphere)",
            description="Parametric FreeCAD documents run headless through snappyHexMesh and OpenFOAM by the "
                        "automation Runner; three studies checked against closed forms and published correlations "
                        "on three meshes each, a sweep over FreeCAD parameters and a surrogate measured on held-out "
                        "cases.", domain="computational fluid dynamics",
            methods=["FreeCAD 1.0 (freecadcmd)", "snappyHexMesh", "OpenFOAM simpleFoam (laminar)",
                     "scalarTransportFoam", "Richardson extrapolation / GCI", "Latin hypercube",
                     "surrogate study (equal budget, held-out test)"],
            modules=["pinneapple_simulation.pipelines.freecad_openfoam", "pinneapple_tools.automation",
                     "pinneapple_simulation.numerical_solvers.external_flow", "pinneapple_analysis.surrogate"],
            params={"quick": q, "sweep": a.sweep}, out_dir=HERE) as ex:
        w = a.work
        bend = mesh_study(studies.pipe_bend, [10, 14, 18] if q else [16, 24, 32], f"{w}/bend", "bend")
        straight = mesh_study(studies.pipe_bend, [10, 14, 18] if q else [16, 24, 32], f"{w}/bend", "straight")
        heat = mesh_study(studies.heated_pipe, [8, 10, 12] if q else [12, 16, 24], f"{w}/heated_pipe", "Nu")
        sph = mesh_study(studies.sphere, [2, 3, 4] if q else [3, 4, 5], f"{w}/sphere", "CD", h_of=lambda lv: 2.0 ** -lv)
        res = [straight, bend, heat, sph]
        for re in (50.0, 200.0):
            st = mesh_study(studies.sphere, [3 if q else 4], f"{w}/sphere_Re{re:g}", "CD", inputs={"Re": re})
            st["study"] = f"sphere_drag Re {re:g}"
            res.append(st)
        mesh_study_figure([straight, bend, heat, sph], os.path.join(fig, "mesh_studies.png"),
                          ["Straight leg: Darcy f vs 64/Re", "Bend: f_curved / f_straight vs White (1929)",
                           "Heated pipe: Nu_m vs Hausen (1943)", "Sphere Re 100: C_D vs Clift et al. (1978)"])
        for st, name in ((bend, "bend"), (heat, "heated_pipe"), (sph, "sphere")):
            for run, src, dst in ((st["rows"][-1]["folder"], "cad/geometry.png", "geometry"),
                                  (st["rows"][0]["folder"], "post/mesh.png", "mesh"),          # coarsest: legible
                                  (st["rows"][-1]["folder"], "post/fields.png", "fields")):
                if os.path.isfile(os.path.join(run, src)):
                    shutil.copy(os.path.join(run, src), os.path.join(fig, f"{name}_{dst}.png"))
        # sweep over the FreeCAD parameters of the bend and a surrogate of the total pressure drop
        design = doe({"Rc": (20.0, 80.0), "angle": (45.0, 180.0)}, 10 if q else a.sweep, "lhs", seed=0,
                     fixed={"Re": 200.0})
        sweep = studies.pipe_bend(10 if q else 16, procs=2, bend_window=False)
        rows = Sweep(sweep, design, f"{w}/sweep").run(parallel=a.parallel)
        sur = fit_surrogate(rows, ["Rc", "angle"], ["dp_total_pa"], test_frac=0.25, budget=5.0)
        sweep_figure(rows, ["Rc", "angle"], "dp_total_pa", sur, os.path.join(fig, "sweep_surrogate.png"), "[Pa]")
        shutil.copy(f"{w}/sweep/sweep.csv", os.path.join(HERE, "sweep.csv"))
        rank = sur.ranking("test")
        table = validation_markdown(res)
        with open(os.path.join(HERE, "validation.md"), "w") as f:
            f.write(table + "\n| surrogate (dp_total_pa from Rc, angle) | held-out nRMSE |\n|---|---|\n"
                    + "".join(f"| {r['title']}{' (chosen)' if r['method'] == sur.best else ''} | {r['error']:.4f} |\n"
                              for r in rank))
        with open(os.path.join(HERE, "validation.json"), "w") as f:
            json.dump({"mesh_studies": res, "surrogate": {"best": sur.best, "ranking": rank,
                                                          "trust": sur.trust.get("decision"),
                                                          "n_cases": len(rows)}}, f, indent=1, default=str)
        print(table)
        for st in res:
            r = st["rows"][-1]
            key = st["study"].replace(" ", "_")
            ex.log_metric(f"{key}_{st['quantity']}", r["measured"], "-",
                          f"finest mesh ({r['cells']} cells), vs {r['source']} = {r['reference']:.4g}")
            ex.check(f"{key}_{st['quantity']}_vs_reference", value=r["measured"], reference=r["reference"],
                     rtol=r["tol"], kind="reference", label=f"{st['quantity']} vs {r['source']}")
            if "extrapolated" in st:
                ex.log_metric(f"{key}_{st['quantity']}_richardson", st["extrapolated"], "-",
                              f"Richardson extrapolation of 3 meshes, p = {st['observed_order']:.2f}")
        best = next(r for r in rank if r["method"] == sur.best)
        ex.log_metric("surrogate_heldout_nrmse", best["error"], "-",
                      f"{sur.best} on {len(sur.splits.test)} held-out OpenFOAM cases (nRMSE of dp_total_pa)")
        ex.check("sweep_cases_finished", all(r["state"] in ("passed", "rejected") for r in rows),
                 label="every sweep case finished")


if __name__ == "__main__":
    main()

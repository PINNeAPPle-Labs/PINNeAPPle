"""End-to-end CFD surrogate study nodes (the Ahmed body in OpenFOAM): geometry, CFD campaign, grid convergence and
validation, dataset, surrogate benchmark, inverse problem, digital twin. Library:
:mod:`pinneapple_autolab.domains.ahmed_study` on :mod:`pinneapple_autolab.study`.

Every node works on a study folder (default: the committed ``examples/end_to_end/ahmed_body``): ``results`` holds
the JSON and figures, ``_runs/cfd`` the campaign records. ``source="recorded"`` reads what is there; nodes that can
run CFD take ``source="run"`` (needs OpenFOAM, minutes per run).
"""
from __future__ import annotations

import json
from pathlib import Path

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node
from ..runtime import artifact
from ..types import PORT_TYPES, register_port_type

if "cfd_runs" not in PORT_TYPES:
    register_port_type("cfd_runs", "a campaign of reference simulations: folder of run records (inputs, outputs, "
                                   "checks, cost) and their fields", converts_to=("dataset",))

_TAGS = ("section:surrogates", "cfd", "openfoam", "surrogate", "uncertainty", "inverse", "digital-twin")
_ROOT = Path(__file__).resolve().parents[2]
_STUDY = _ROOT / "examples" / "end_to_end" / "ahmed_body"


def _folders(params):
    base = Path(params.get("folder") or _STUDY)
    return base, base / "_runs" / "cfd", base / "results"


def _png(ctx, path, name):
    if Path(path).exists():
        import shutil
        out = Path(ctx.workdir) / name
        out.parent.mkdir(parents=True, exist_ok=True)
        return artifact(shutil.copy(path, out), "png")
    return None


_FOLDER = Param("folder", "str", "", help="study folder (empty: examples/end_to_end/ahmed_body)", advanced=True)


@node("study.ahmed_geometry", "Ahmed body: parametric and CAD geometry", category="Geometry",
      outputs=[Port("metrics", "metrics")],
      params=[Param("slant_deg", "float", 25.0, minimum=0, maximum=40), Param("clearance", "float", 0.05,
                                                                              minimum=0.03, maximum=0.08),
              Param("cad", "bool", True, help="also build the OpenCASCADE CAD (gmsh) and import its STL back")],
      calls="pinneapple_autolab.domains.ahmed.make_body / ahmed_cad / import_body", tags=_TAGS)
def ahmed_geometry(ctx, inputs, params):
    """The parametric body (the field template: 632 points, same for every variant) and, with ``cad``, the
    B-rep exported to STEP/STL and imported back with its checks and named patches."""
    from pinneapple_autolab.domains import ahmed as A
    from pinneapple_design.geometry_library import meshops as mo
    p = {"slant_deg": float(params["slant_deg"]), "clearance": float(params["clearance"]), "speed": 40.0}
    V, F = A.make_body(p)
    ctx.metric("volume_m3", mo.volume(V, F), "parametric body, divergence theorem on its triangles")
    m = {"parametric_volume_m3": mo.volume(V, F), "template_points": int(len(A.field_vertices(p)))}
    if params["cad"]:
        files = A.ahmed_cad(p, str(Path(ctx.workdir) / "cad"))
        b = A.import_body(files["stl"], p)
        if b["issues"]:
            raise StepFailed("imported STL: " + "; ".join(b["issues"]))
        m.update(cad_volume_m3=b["volume"], cad_triangles=len(b["F"]), symmetry_mm=b["symmetry_mm"],
                 patches={r: int((b["regions"] == r).sum()) for r in A.REGIONS})
        ctx.metric("cad_volume_m3", b["volume"], "imported STL of the OpenCASCADE body")
        ctx.table("named patches (CAD STL)", ["patch", "triangles"], [[k, v] for k, v in m["patches"].items()])
    return {"metrics": m}


@node("study.cfd_campaign", "CFD campaign (OpenFOAM, resumable)", category="Solvers",
      outputs=[Port("runs", "cfd_runs"), Port("metrics", "metrics")],
      params=[Param("source", "enum", "recorded", choices=["recorded", "run"],
                    help="recorded: the committed study's runs; run: plan and run the DOE now (OpenFOAM)"),
              Param("n", "int", 48, minimum=4, maximum=512, help="run only: DOE size (maximin LHS)"),
              Param("procs", "int", 4, minimum=1, maximum=64, advanced=True), _FOLDER],
      calls="pinneapple_autolab.study.campaign.plan_doe + run_campaign (pinneapple_autolab.domains.ahmed.AhmedAero)",
      tags=_TAGS)
def cfd_campaign(ctx, inputs, params):
    """Design of experiments over slant, clearance and speed run in OpenFOAM (k-omega SST, prism layers), each run
    checked (checkMesh, residuals, force convergence, mass balance, y+); returns the records folder."""
    from pinneapple_autolab.domains import ahmed as A
    from pinneapple_autolab.study import campaign as C
    base, runs, _ = _folders(params)
    if params["source"] == "run":
        from pinneapple_simulation.numerical_solvers.external_flow import openfoam_available
        if not openfoam_available():
            raise StepFailed("OpenFOAM is not installed here; use source=recorded")
        runs = Path(ctx.workdir) / "cfd"
        plan = C.plan_doe(C.space_from_variables(A.VARIABLES), int(params["n"]), "maximin", seed=0)
        C.run_campaign(A.AhmedAero(procs=int(params["procs"])), plan["cases"], str(runs), log=ctx.log)
    recs = C.load_records(str(runs))
    if not recs:
        raise StepFailed(f"no run records in {runs}")
    cost = C.cost_table(recs)
    ctx.metric("runs_ok", sum(r.ok for r in recs), "run records with ok = true")
    ctx.metric("cpu_hours", cost["total"]["cpu_h"], "wall time x processes, summed over the runs")
    ctx.table("cost per purpose", ["purpose", "runs", "failed", "CPU h"],
              [[k, v["runs"], v["failed"], round(v["cpu_h"], 2)] for k, v in cost["stages"].items()])
    return {"runs": {"folder": str(runs)}, "metrics": cost}


@node("study.grid_validation", "Grid convergence and validation against the wind tunnel", category="Analysis",
      inputs=[Port("runs", "cfd_runs")], outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[_FOLDER], calls="pinneapple_autolab.domains.ahmed_study.grid_and_validation", tags=_TAGS)
def grid_validation(ctx, inputs, params):
    """GCI (Celik et al. 2008) of the baseline on the mesh family; the drag against slant angle compared with
    Ahmed, Ramm & Faltin (1984)."""
    from pinneapple_autolab.domains import ahmed_study as S
    _, _, res = _folders(params)
    out = Path(ctx.workdir) / "results"
    r = S.grid_and_validation(S.load_records(inputs["runs"]["folder"]), str(out))
    for row in r["validation"]["rows"]:
        ctx.metric(f"CD_error_{row['slant_deg']:g}deg", row["rel_error"], "(CFD - measured) / measured, Ahmed 1984")
    g = (r["mesh_study"].get("CD") or {})
    if g.get("gci_fine") is not None:
        ctx.metric("gci_CD_fine", g["gci_fine"], "grid convergence index of the finest mesh, Celik et al. 2008")
    fig = _png(ctx, out / "figures" / "validation_cd.png", "validation_cd.png")
    return {"metrics": {"validation": r["validation"]["rows"], "gci": g}, **({"figure": fig} if fig else {})}


@node("study.dataset", "Study dataset with a held-out region", category="Data",
      inputs=[Port("runs", "cfd_runs")], outputs=[Port("dataset", "dataset"), Port("metrics", "metrics")],
      params=[_FOLDER], calls="pinneapple_autolab.domains.ahmed_study.study_dataset + StudyDataset.save", tags=_TAGS)
def dataset(ctx, inputs, params):
    """Scalars, the surface Cp on 632 fixed points and provenance as a DatasetSpec; slant > 34 deg is the
    extrapolation split."""
    from pinneapple_autolab.domains import ahmed_study as S
    ds = S.study_dataset(S.load_records(inputs["runs"]["folder"]))
    stem = Path(ctx.workdir) / "ahmed_body"
    spec = ds.save(str(stem), name="ahmed_body_rans_l1")
    sizes = {k: len(v) for k, v in ds.split.items()}
    ctx.table("splits", ["split", "cases"], [[k, v] for k, v in sizes.items()])
    return {"dataset": artifact(str(stem) + ".npz", "npz") | {"spec": spec}, "metrics": sizes}


@node("study.surrogates", "Surrogate families: equal budget, intervals, physics checks", category="Surrogates",
      inputs=[Port("dataset", "dataset")], outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("budget_s", "float", 30.0, minimum=1.0, help="training seconds per iterative fit"), _FOLDER],
      calls="pinneapple_autolab.domains.ahmed_study.benchmark_all (pinneapple_autolab.study.benchmark)", tags=_TAGS)
def surrogates(ctx, inputs, params):
    """GP, MLP ensemble and response surface for the forces; POD-GP, POD-MLP, POD-RBF and DeepONet for the Cp field;
    5-fold CV, cross-conformal 90 % intervals, physics checks on predictions, the written selection rule."""
    from pinneapple_autolab.domains import ahmed_study as S
    from pinneapple_autolab.study.dataset import StudyDataset
    stem = str(inputs["dataset"]["$artifact"])[:-4]
    ds = StudyDataset.load(stem)
    out = Path(ctx.workdir) / "results"
    b = S.benchmark_all(ds, str(out), budget_s=float(params["budget_s"]), log=ctx.log)
    rows = []
    for board in (b["scalar"], b["field"]):
        for r in board.table():
            rows.append([r["target"], r["family"], round(r.get("cv_rel_rmse", r.get("cv_rel_l2", float("nan"))), 4),
                         r.get("test_coverage90_conformal"), r["physics_passed"]])
    ctx.table("families", ["target", "family", "CV error", "test coverage 90 %", "physics"], rows)
    for t, f in b["selection"].items():
        ctx.decide("surrogate.selection", f"{t}: {f}", f, rationale=json.dumps(b["scalar"].select()["rule"])[:300])
    fig = _png(ctx, out / "figures" / "parity_CD.png", "parity_CD.png")
    return {"metrics": {"selection": b["selection"]}, **({"figure": fig} if fig else {})}


@node("study.inverse", "Slant angle from a drag measurement (amortised posterior)", category="Analysis",
      outputs=[Port("params", "params"), Port("figure", "figure")],
      params=[Param("cd", "float", 0.285, help="measured drag coefficient"),
              Param("clearance", "float", 0.05, minimum=0.03, maximum=0.08),
              Param("speed", "float", 60.0, minimum=20, maximum=60), _FOLDER],
      calls="pinneapple_autolab.domains.ahmed_study.fit_selected + train_slant_mdn", tags=_TAGS)
def inverse(ctx, inputs, params):
    """The posterior of the slant angle for one measured CD at a known clearance and speed (MDN on the selected
    surrogate, discrepancy from the validation sweep); several modes when the drag curve is not monotonic."""
    import numpy as np

    from pinneapple_autolab.domains import ahmed_study as S
    base, runs, res = _folders(params)
    sel = json.load(open(res / "benchmark.json"))["selection"]["CD"]
    val = json.load(open(res / "validation.json"))["rows"]
    sig = float(np.sqrt(np.mean([(r["CD_cfd"] - r["CD_measured"]) ** 2 for r in val])))
    dep = S.deployment_dataset(S.load_records(str(runs), ok_only=True))
    pred = S.fit_selected(dep, "CD", sel, np.arange(len(dep)))
    mdn = S.train_slant_mdn(pred, sig, budget_s=30.0)
    deg, iv = S._slant(mdn, float(params["cd"]), float(params["clearance"]), float(params["speed"]))
    ctx.metric("slant_median_deg", float(np.median(deg)), "median of 4000 MDN posterior samples")
    ctx.table("90 % highest-density set", ["from [deg]", "to [deg]"], [[round(a, 1), round(b, 1)] for a, b in iv])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(deg, bins=50, color="#2a78d6")
    ax.set_xlabel("slant angle [deg]")
    f = ctx.figure(fig, "posterior.png", "posterior of the slant angle")
    return {"params": {"median_deg": float(np.median(deg)), "hdi90": iv, "sigma_d": sig}, "figure": f}


@node("study.twin", "Surrogate digital twin (sliders, 3-D Cp, band, trust region)", category="Visualization",
      outputs=[Port("report", "report")], params=[_FOLDER],
      calls="pinneapple_autolab.domains.ahmed_study.twin (pinneapple_autolab.study.twin)", tags=_TAGS)
def twin(ctx, inputs, params):
    """The selected models refitted on every run, evaluated on a grid and written as a three.js page with sliders,
    the 90 % band and the out-of-trust-region warning."""
    from pinneapple_autolab.domains import ahmed_study as S
    base, runs, res = _folders(params)
    page = S.twin(S.load_records(str(runs), ok_only=True), str(res), str(Path(ctx.workdir) / "twin"))
    ctx.view("html", {"path": str(Path(page).resolve()), "title": "Ahmed body twin"})
    return {"report": artifact(page, "html")}


__all__ = ["ahmed_geometry", "cfd_campaign", "grid_validation", "dataset", "surrogates", "inverse", "twin"]

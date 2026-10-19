"""CFD nodes on OpenFOAM: the free-surface trough of ``examples/free_surface`` (a 2-D section with a liquid bath fed
by a jet, interFoam VOF with k-omega SST, :class:`pinneapple_simulation.numerical_solvers.free_surface.FreeSurfaceFlow`)
and the run campaign of ``examples/openfoam_campaign``
(:class:`pinneapple_simulation.external_solvers.openfoam.campaign.Campaign`). They finish ``not_run`` with the reason
when interFoam is not on the PATH."""
from __future__ import annotations

import json
import shutil
import time

import numpy as np

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node
from .examples import example_dir


def _foam_available():
    if shutil.which("interFoam") is None or shutil.which("blockMesh") is None:
        return False, "needs OpenFOAM v1912+ (interFoam, blockMesh on the PATH; apt install openfoam)"
    return True, ""


@node("cfd.trough_jet", "Free-surface trough with a jet (interFoam)", category="CFD",
      outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("slope_deg", "float", 0.0, minimum=-10.0, maximum=10.0,
                    help="floor slope; the floor pivots about the weir end (+: descending toward the weir)"),
              Param("length", "float", 8.0, minimum=1.0), Param("inlet_x", "float", 1.0),
              Param("inlet_width", "float", 0.08, minimum=0.005), Param("cell", "float", 0.04, minimum=0.002),
              Param("jet", "list", [1.0, 0.0, -2.5], help="jet velocity at the slot, world frame (x down, z up)"),
              Param("liquid", "dict", {"name": "metal", "rho": 7200.0, "nu": 8.472e-6}),
              Param("sigma", "float", 1.8, help="surface tension [N/m]"),
              Param("end_time", "float", 12.0, minimum=0.1), Param("average_from", "float", 4.0, minimum=0.0),
              Param("sample_every", "float", 0.1, minimum=0.001),
              Param("line_height", "float", 0.05, help="sample line above the floor [m]"),
              Param("zone", "list", [0.5, 3.0], help="impact zone along the floor [m]"),
              Param("procs", "int", 1, minimum=1, maximum=64), Param("label", "str", "")],
      calls="pinneapple_design.geometry.trough.TroughSection + pinneapple_simulation.numerical_solvers."
            "free_surface.FreeSurfaceFlow (write, run, profiles) + external_solvers.openfoam.campaign.parse_solver_log",
      available=_foam_available)
def trough_jet(ctx, inputs, params):
    """examples/free_surface/01_runner_slope_study.py for one slope: time-mean k a few cm above the floor in the
    impact zone, its peak, the depth under the jet and how settled the liquid volume is."""
    from pinneapple_design.geometry.trough import TroughSection
    from pinneapple_simulation.external_solvers.openfoam.campaign import parse_solver_log
    from pinneapple_simulation.numerical_solvers.free_surface import FreeSurfaceFlow, Phase, SampleLine
    sec = TroughSection(length=float(params["length"]), slope_deg=float(params["slope_deg"]),
                        inlet_x=float(params["inlet_x"]), inlet_width=float(params["inlet_width"]),
                        cell=float(params["cell"]))
    liq = params["liquid"]
    T_END, T_AVG = float(params["end_time"]), float(params["average_from"])
    h = float(params["line_height"])
    case = ctx.path("case")
    flow = FreeSurfaceFlow(
        mesh=sec.write_mesh(str(case / "mesh")), two_d=True,
        roles={"inlet": "inlet", "outlet": "outlet", "atmosphere": "atmosphere"},
        inlet_velocity={"inlet": sec.to_mesh(tuple(params["jet"]))}, level=sec.level(), gravity=sec.gravity(9.8),
        liquid=Phase(liq.get("name", "metal"), float(liq["rho"]), float(liq["nu"])), gas=Phase("air", 1.2, 1.5e-5),
        sigma=float(params["sigma"]), end_time=T_END, sample_every=float(params["sample_every"]), keep_every=T_AVG,
        lines=[SampleLine("floor", (0.0, sec.thickness / 2, h), (sec.length, sec.thickness / 2, h), 161)])
    run = case / "run"
    t0 = time.perf_counter()
    flow.write(str(run), procs=int(params["procs"]))
    flow.run(str(run), procs=int(params["procs"]), log=ctx.log)
    ctx.metric("seconds", round(time.perf_counter() - t0, 1))
    prof = flow.profiles(str(run), t_start=T_AVG)["floor"]
    x, k = np.asarray(prof["s"]), np.asarray(prof["mean_k"])
    zone = (x >= params["zone"][0]) & (x <= params["zone"][1])
    health = parse_solver_log((run / "log.interFoam").read_text(), every=1)
    vol = np.asarray(health["phase_volume"] or [(0.0, np.nan)], float)
    last = vol[:, 0] > T_END - 4
    name = params["label"] or f"{params['slope_deg']:+.0f} deg"
    ctx.metric(f"{name}.depth_at_jet_m", float(sec.depth_at(sec.inlet_x + 0.5 * sec.inlet_width)))
    ctx.metric(f"{name}.k_zone_mean", float(k[zone].mean()), f"time-mean k {h:g} m above the floor, "
                                                             f"{T_AVG:g}-{T_END:g} s, zone {params['zone']}")
    ctx.metric(f"{name}.k_peak", float(k[zone].max()))
    if last.any():
        ctx.metric(f"{name}.liquid_fraction_last_4s_change_pct",
                   float(100 * np.ptp(vol[last, 1]) / vol[-1, 1]), "settled when small")
    ctx.metric(f"{name}.courant_max", health["courant_max"])
    ctx.metric(f"{name}.ended", int(bool(health["ended"])))
    if health["fatal"]:
        raise StepFailed(f"interFoam failed: {health['fatal']}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    ax[0].plot(x, k)
    ax[0].axvspan(*params["zone"], color="0.92", zorder=0)
    ax[0].set(xlabel="distance from the upstream wall [m]", ylabel="mean k [m^2/s^2]", title=name)
    ax[1].plot(vol[:, 0], vol[:, 1] / vol[0, 1])
    ax[1].set(xlabel="time [s]", ylabel="liquid volume / initial")
    f = ctx.figure(fig, "trough_k", "k above the floor and the liquid volume")
    ctx.summary(f"{name}: k zone mean {k[zone].mean():.3g}, peak {k[zone].max():.3g}")
    return {"metrics": ctx.report["metrics"], "figure": f}


@node("cfd.ratio", "Ratios of a metric to a baseline case", category="CFD",
      inputs=[Port("metrics", "metrics", multiple=True)], outputs=[Port("metrics", "metrics")],
      params=[Param("metric", "str", "k_zone_mean", help="metric name after the case label"),
              Param("baseline", "str", "+0 deg", help="case label of the baseline")],
      calls="arithmetic on the upstream metrics (no library call)")
def ratio(ctx, inputs, params):
    """``<case>.<metric> / <baseline>.<metric>`` for every upstream case (e.g. k relative to the 0 degree floor)."""
    vals = {}
    for rep in ctx.upstream_reports.values():
        for k, v in (rep.get("metrics") or {}).items():
            case, _, m = k.rpartition(".")
            if m == params["metric"] and case:
                vals[case] = float(v["value"] if isinstance(v, dict) else v)
    if params["baseline"] not in vals:
        raise StepFailed(f"no {params['baseline']}.{params['metric']} upstream (found {sorted(vals)})")
    b = vals[params["baseline"]]
    for case, v in vals.items():
        ctx.metric(f"{case}.{params['metric']}_rel_to_baseline", v / b, f"relative to {params['baseline']}")
    return {"metrics": ctx.report["metrics"]}


@node("cfd.campaign", "OpenFOAM run campaign (plan file)", category="CFD",
      outputs=[Port("metrics", "metrics"), Port("figure", "figure", required=False)],
      params=[Param("plan", "str", "openfoam_campaign/plan.json", help="<example>/<plan>.json"),
              Param("runs", "list", [], help="run ids (empty: all)"),
              Param("end_time", "float", 0.0, minimum=0.0, help="override the plan's end time (0: keep it)"),
              Param("procs", "int", 0, minimum=0, maximum=64, help="override the plan's processes (0: keep)"),
              Param("base", "dict", {"length": 6.0, "inlet_x": 1.0, "slope_deg": 0.0, "cell": 0.04, "jet": [1.0, 0.0, -2.5],
                                     "liquid": {"name": "metal", "rho": 7200.0, "nu": 8.472e-6}, "sigma": 1.8,
                                     "end_time": 10.0, "sample_every": 0.1},
                    help="the base case of make_base.py (TroughSection + FreeSurfaceFlow)")],
      calls="pinneapple_simulation.external_solvers.openfoam.campaign.Campaign.from_json + run (parse_solver_log, "
            "LiveLineSampler); base case: TroughSection + FreeSurfaceFlow.write", available=_foam_available)
def campaign(ctx, inputs, params):
    """examples/openfoam_campaign: the base case written by the library, then each run of the plan (jet speed and
    angle) with its health (Courant, phase bounds, volume) and the sampled profiles."""
    from pinneapple_design.geometry.trough import TroughSection
    from pinneapple_simulation.external_solvers.openfoam.campaign import Campaign
    from pinneapple_simulation.numerical_solvers.free_surface import FreeSurfaceFlow, Phase
    ex, _, f = params["plan"].partition("/")
    src = (example_dir(ex) / f).resolve()
    if example_dir(ex) not in src.parents or not src.is_file():
        raise StepFailed(f"{params['plan']!r} is not a plan of the example")
    plan = json.loads(src.read_text())
    root = ctx.path("campaign")
    b = params["base"]
    sec = TroughSection(length=float(b["length"]), inlet_x=float(b["inlet_x"]), slope_deg=float(b["slope_deg"]),
                        cell=float(b["cell"]))
    procs = int(params["procs"]) or int(plan.get("procs", 1))
    flow = FreeSurfaceFlow(mesh=sec.write_mesh(str(root / "_out" / "mesh")), two_d=True,
                           roles={"inlet": "inlet", "outlet": "outlet", "atmosphere": "atmosphere"},
                           inlet_velocity={"inlet": tuple(b["jet"])}, level=sec.level(), gravity=sec.gravity(9.81),
                           liquid=Phase(b["liquid"]["name"], rho=float(b["liquid"]["rho"]), nu=float(b["liquid"]["nu"])),
                           sigma=float(b["sigma"]), end_time=float(b["end_time"]), sample_every=float(b["sample_every"]))
    flow.write(str(root / "_out" / "base"), procs=procs)
    (root / "plan.json").write_text(json.dumps(plan, indent=1))
    over = {"procs": procs, "workdir": str(root / "work")}
    if params["end_time"]:
        over["end_time"] = float(params["end_time"])
    camp = Campaign.from_json(root / "plan.json", **over)
    res = camp.run(ids=list(params["runs"]) or None, force=True, log=ctx.log)
    rows = []
    for r in res:
        h = r.get("health") or {}
        rid = r["id"]
        ctx.metric(f"{rid}.rc", int(r.get("rc", -1)))
        for k in ("courant_max", "interface_courant_max", "last_time"):
            if isinstance(h.get(k), (int, float)):
                ctx.metric(f"{rid}.{k}", float(h[k]))
        alpha = (h.get("bounds") or {}).get("alpha.metal")
        if alpha:
            ctx.metric(f"{rid}.alpha_min", float(alpha[0]))
            ctx.metric(f"{rid}.alpha_max", float(alpha[1]))
        line = (r.get("lines") or {}).get("mid_depth") or {}
        if line.get("mean_k"):
            ctx.metric(f"{rid}.k_mid_depth_mean", float(np.nanmean(line["mean_k"])))
        rows.append([rid, r.get("params"), r.get("rc"), h.get("courant_max"), h.get("last_time"), r.get("wall_s")])
    ctx.table("campaign runs", ["run", "params", "rc", "Courant max", "last time", "wall [s]"], rows)
    bad = [r["id"] for r in res if r.get("rc") != 0]
    if bad:
        ctx.warn(f"runs with a non-zero return code: {bad}")
    ctx.summary(f"{len(res)} runs, {len(bad)} failed")
    return {"metrics": ctx.report["metrics"]}


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="cfd.trough_jet", title="Free-surface trough: a jet into a liquid bath (interFoam VOF)", domain="CFD",
        example="free_surface",
        keywords={"free surface": 3, "free-surface": 3, "vof": 3, "interfoam": 3, "runner": 1.5, "trough": 2,
                  "jet": 1, "molten": 1.5, "melt": 1, "tapping": 1.5, "refractory": 1, "weir": 1.5},
        min_score=3, nodes=("cfd.trough_jet", "cfd.ratio"), references=("hirt1981vof", "menter1994sst"),
        inputs=("the section geometry, the jet and the liquid", "OpenFOAM v1912+"),
        template="free_surface/01_runner_slope_study", quick=True,
        description="examples/free_surface/01_runner_slope_study.py"))
    register_capability(Capability(
        id="cfd.campaign", title="OpenFOAM run campaign from a plan file", domain="CFD", example="openfoam_campaign",
        keywords={"campaign": 3, "parameter sweep": 2, "many runs": 1.5, "openfoam": 1.5, "plan file": 2,
                  "workers": 1}, requires_all=(("openfoam", "cfd", "interfoam", "foam"),), min_score=3,
        nodes=("cfd.campaign",), references=("hirt1981vof",), inputs=("a plan.json", "OpenFOAM v1912+"),
        template="openfoam_campaign/campaign", quick=True,
        description="examples/openfoam_campaign (make_base.py + plan.json)"))


_register_capabilities()

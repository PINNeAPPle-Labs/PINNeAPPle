"""Geomechanics, inspection and component-library nodes: the granular column collapse of ``examples/geomechanics``
(:class:`pinneapple_simulation.particle_dynamics.geo_mpm.GeoMPM`, MLS-MPM with Drucker-Prager sand, against the
runout law of Lube et al. 2005), the eddy-current inspection PINN of ``examples/inspection``
(:mod:`pinneapple_analysis.inspection`) and the component registry of ``examples/systems``
(:mod:`pinneapple_systems.component_library`)."""
from __future__ import annotations

import numpy as np

from ..registry import Param, Port, node


def lube_2005(a: float) -> float:
    """Lube et al. (2005) two-dimensional collapse runout (L_inf - L0) / L0: 1.2 a below a = 1.8, 1.9 a^(2/3) above
    2.8, the mean of both in the transition (as examples/geomechanics/01_granular_column_collapse.py)."""
    if a < 1.8:
        return 1.2 * a
    if a > 2.8:
        return 1.9 * a ** (2 / 3)
    return 0.5 * (1.2 * a + 1.9 * a ** (2 / 3))


@node("geo.column_collapse", "Granular column collapse (MPM)", category="Geomechanics",
      outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("aspects", "list", [0.5, 1.0, 2.0, 3.0], help="H0 / L0 of each column"),
              Param("L0", "float", 0.2, minimum=0.01), Param("friction_deg", "float", 30.0, minimum=5.0, maximum=60.0),
              Param("dx", "float", 0.01, minimum=0.001, maximum=0.1, help="grid cell [m]"),
              Param("save_every", "float", 0.05, minimum=0.001)],
      calls="pinneapple_simulation.particle_dynamics.geo_mpm.GeoMPM + Sand + box + runout; "
            "pinneapple_simulation.external_solvers.geotaichi.read_geotaichi + frames_to_samples")
def column_collapse(ctx, inputs, params):
    """Columns of dry cohesionless sand against a back wall on a rough floor, released at t = 0; the final runout
    against the law Lube et al. measured, the final height (examples/geomechanics/01_granular_column_collapse.py)."""
    import torch

    from pinneapple_simulation.external_solvers.geotaichi import frames_to_samples, read_geotaichi
    from pinneapple_simulation.particle_dynamics.geo_mpm import GeoMPM, Sand, box, runout
    torch.set_num_threads(1)
    L0, phi = float(params["L0"]), float(params["friction_deg"])
    rows, last_frames = [], {}
    for a in map(float, params["aspects"]):
        ctx.check_stop()
        sim = GeoMPM(domain=(L0 * (2 + 2.5 * a), a * L0 + 0.1), dx=float(params["dx"]), floor="sticky", walls="slip")
        sim.add_body(box((0, 0), (L0, a * L0)), Sand(friction_deg=phi))
        run_dir = ctx.path(f"a{a:g}")
        sim.run(t_end=0.6 + 0.4 * a, save_every=float(params["save_every"]), out=str(run_dir))
        frames = read_geotaichi(str(run_dir))
        last = frames[-1]
        dL = (runout(last["position"]) - L0) / L0
        H = float(np.quantile(last["position"][:, 1], 0.995))
        v = float(np.linalg.norm(last["velocity"], axis=1).max())
        tag = f"a={a:g}"
        ctx.metric(f"{tag}.runout_dL_over_L0", float(dL), "(L_inf - L0) / L0")
        ctx.metric(f"{tag}.lube_2005", lube_2005(a))
        ctx.metric(f"{tag}.ratio_to_lube", float(dL) / lube_2005(a))
        ctx.metric(f"{tag}.final_height_over_L0", H / L0)
        ctx.metric(f"{tag}.max_speed_at_end", v, "settled when small")
        rows.append([a, float(dL), lube_2005(a), H / L0, v])
        last_frames[a] = (frames[0], last)
        if a == 1.0:
            ctx.metric("samples_per_run", len(frames_to_samples(frames, source="pinneapple.geo_mpm")))
    ctx.table("runout against Lube et al. (2005)", ["a", "(L-L0)/L0", "Lube 2005", "H_end / L0", "max speed"], rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(last_frames), 1, figsize=(9, 1.8 * len(last_frames)), squeeze=False)
    for ax, (a, (f0, f1)) in zip(axs[:, 0], last_frames.items(), strict=True):
        ax.scatter(f0["position"][:, 0], f0["position"][:, 1], s=0.4, c="0.8")
        ax.scatter(f1["position"][:, 0], f1["position"][:, 1], s=0.4, c="#c8833a")
        ax.axvline(L0 * (1 + lube_2005(a)), color="k", ls="--", lw=1)
        ax.set_aspect("equal")
        ax.set_title(f"a = {a:g}: initial (grey), final (sand); dashed: Lube et al.", fontsize=8)
    f = ctx.figure(fig, "column_collapse", "initial and final columns against the runout law")
    return {"metrics": ctx.report["metrics"], "figure": f}


@node("inspection.eddy_current", "Eddy-current inspection PINN", category="Inspection",
      outputs=[Port("metrics", "metrics")],
      params=[Param("nr", "int", 24, minimum=4), Param("nz", "int", 40, minimum=4),
              Param("scan", "list", [-0.01, 0.01, 9], help="scan positions: start, stop, count [m]"),
              Param("hidden", "list", [64, 64, 64]), Param("epochs", "int", 60, minimum=1),
              Param("lr", "float", 1e-3), Param("batch_size", "int", 32, minimum=1)],
      calls="pinneapple_analysis.inspection.generate_eddy_current_synthetic (eddy_current_fdm) + train_eddy_current "
            "(InspectionPINN with a Helmholtz physics loss)")
def eddy_current(ctx, inputs, params):
    """A synthetic probe scan from the axisymmetric complex-Helmholtz FDM solver, then an InspectionPINN trained on
    it (examples/inspection/01_eddy_current_pinn.py)."""
    from pinneapple_analysis.inspection import generate_eddy_current_synthetic, train_eddy_current
    s0, s1, n = params["scan"]
    data = generate_eddy_current_synthetic(nr=int(params["nr"]), nz=int(params["nz"]),
                                           scan_positions=np.linspace(float(s0), float(s1), int(n)))
    for i, sig in enumerate(data["scan_signal"]):
        ctx.metric(f"scan{i}.abs_signal", float(abs(sig)))
        ctx.metric(f"scan{i}.phase_rad", float(np.angle(sig)))
    out = train_eddy_current(synthetic_data=data, hidden=[int(h) for h in params["hidden"]],
                             epochs=int(params["epochs"]), lr=float(params["lr"]), batch_size=int(params["batch_size"]),
                             log_dir=str(ctx.path("logs")), run_name="eddy_current")
    hist = out["train_out"]["history"]
    ctx.metric("val_total_epoch0", float(hist[0]["val_total"]))
    ctx.metric("val_total_final", float(hist[-1]["val_total"]))
    ctx.metric("best_val", float(out["train_out"]["best_val"]))
    ctx.summary(f"best validation loss {out['train_out']['best_val']:.3e}")
    return {"metrics": ctx.report["metrics"]}


@node("systems.component_fit", "Component library: build and fit a component", category="Systems",
      outputs=[Port("metrics", "metrics")],
      params=[Param("component", "str", "PipeVanillaPINN", help="pinneapple_systems.component_library name"),
              Param("toolbox", "str", "Fluid Mechanics Toolbox"), Param("in_dim", "int", 3), Param("out_dim", "int", 3),
              Param("hidden", "list", [64, 64, 64]), Param("epochs", "int", 5, minimum=1),
              Param("n", "int", 256, minimum=8), Param("seed", "int", 0)],
      calls="pinneapple_systems.component_library.ComponentRegistry.build / .fit / .evaluate + ToolboxRegistry.install")
def component_fit(ctx, inputs, params):
    """examples/systems/01_component_library_toolbox.py: the registered components and toolboxes, one component
    built and fitted. The example fits random targets (a demonstration of the API, not of the physics); this node
    seeds them."""
    import tempfile

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from pinneapple_systems.component_library import ComponentRegistry, ToolboxRegistry
    torch.manual_seed(int(params["seed"]))
    types = ComponentRegistry.component_types()
    ctx.metric("component_types", len(types))
    tb = ToolboxRegistry.install(params["toolbox"])
    ctx.table("toolbox components", ["component"], [[c.name] for c in tb.components()])
    comp = ComponentRegistry.build(params["component"], in_dim=int(params["in_dim"]), out_dim=int(params["out_dim"]),
                                   hidden=tuple(int(h) for h in params["hidden"]))
    n = int(params["n"])
    x, y = torch.rand(n, int(params["in_dim"])), torch.rand(n, int(params["out_dim"]))
    loader = DataLoader(TensorDataset(x, y), batch_size=32,
                        collate_fn=lambda b: {"x": torch.stack([i[0] for i in b]), "y": torch.stack([i[1] for i in b])})
    res = comp.fit(loader, loader, epochs=int(params["epochs"]), lr=1e-3, run_name="component",
                   log_dir=tempfile.mkdtemp(dir=str(ctx.workdir)))
    ctx.metric("best_val", float(res["best_val"]))
    for k, v in (comp.evaluate(torch.rand(64, int(params["in_dim"])), torch.rand(64, int(params["out_dim"]))) or {}).items():
        if isinstance(v, (int, float)):
            ctx.metric(f"eval.{k}", float(v))
    ctx.summary(f"{params['component']} ({comp.physics.name}): best_val {res['best_val']:.4f} on random targets")
    return {"metrics": ctx.report["metrics"]}


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="geo.column_collapse", title="Granular column collapse (MPM) against Lube et al.", domain="geomechanics",
        example="geomechanics",
        keywords={"granular": 3, "column collapse": 3, "sand": 2, "mpm": 3, "material point": 3, "runout": 2,
                  "landslide": 1.5, "drucker-prager": 2}, min_score=3, nodes=("geo.column_collapse",),
        uncited="Lube et al. (2005) is quoted by the example, not in the flow reference registry",
        template="geomechanics/01_granular_column_collapse", quick=True,
        description="examples/geomechanics/01_granular_column_collapse.py"))
    register_capability(Capability(
        id="inspection.eddy_current", title="Eddy-current non-destructive inspection with a PINN", domain="inspection",
        example="inspection", keywords={"eddy current": 3, "eddy-current": 3, "nde": 2, "ndt": 2, "inspection": 2,
                                        "probe scan": 2, "crack": 1}, min_score=3,
        nodes=("inspection.eddy_current",), references=("raissi2019pinn",), template="inspection/01_eddy_current_pinn",
        description="examples/inspection/01_eddy_current_pinn.py"))
    register_capability(Capability(
        id="systems.component", title="Component library: build and fit a physics component", domain="systems",
        example="systems", keywords={"component library": 3, "toolbox": 2, "component registry": 3,
                                     "pipe component": 2}, min_score=3, nodes=("systems.component_fit",),
        uncited="API demonstration", template="systems/01_component_library_toolbox",
        description="examples/systems/01_component_library_toolbox.py"))


_register_capabilities()

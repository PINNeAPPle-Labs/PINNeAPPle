"""Design nodes: topology optimization (``examples/topology_optimization``) on :mod:`pinneapple_design.topology`
(SIMP compliance of the MBB beam and the cantilever, thermal compliance, Stokes dissipation of the double pipe, the
iron of a reluctance rotor for torque, the sensitivity check) and design from intent (``examples/design_from_intent``,
``pp.design``). Metric names follow the keys of the examples' results JSON."""
from __future__ import annotations

import math
import time

import numpy as np

from ..registry import Param, Port, node

PROBLEMS = ["mbb", "cantilever", "thermal", "double_pipe"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _problem(params):
    from pinneapple_design.topology import (
        Compliance2D,
        DensityFilter,
        StokesDissipation2D,
        ThermalCompliance2D,
    )
    kind, nx, ny, r = params["problem"], int(params["nelx"]), int(params["nely"]), float(params["rmin"])
    if kind == "mbb":
        p = Compliance2D.mbb(nx, ny)
    elif kind == "cantilever":
        p = Compliance2D.cantilever(nx, ny)
    elif kind == "thermal":
        p = ThermalCompliance2D(nx, ny, sink_fraction=float(params["sink_fraction"]))
    else:
        p = StokesDissipation2D.double_pipe(nely=ny, length=float(params["length"]), q=float(params["q"][0]))
        return p, DensityFilter.grid(p.grid.nelx, p.grid.nely, r), (p.grid.nelx, p.grid.nely)
    return p, DensityFilter.grid(nx, ny, r), (nx, ny)


@node("design.topology2d", "Topology optimization (2-D SIMP)", category="Design",
      outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("problem", "enum", "mbb", choices=PROBLEMS,
                    help="mbb / cantilever: compliance; thermal: thermal compliance; double_pipe: Stokes dissipation"),
              Param("nelx", "int", 60, minimum=4, maximum=1000), Param("nely", "int", 20, minimum=4, maximum=1000),
              Param("volume", "float", 0.5, minimum=0.01, maximum=0.99, help="volume (or fluid) fraction"),
              Param("rmin", "float", 2.4, minimum=0.5, maximum=50.0, help="density filter radius (elements)"),
              Param("method", "enum", "oc", choices=["oc", "mma"]),
              Param("max_iter", "int", 300, minimum=1, maximum=5000),
              Param("projection", "bool", False, help="Heaviside projection with beta continuation"),
              Param("beta0", "float", 1.0), Param("beta_max", "float", 16.0), Param("beta_every", "int", 40),
              Param("move", "float", 0.2, minimum=0.01, maximum=1.0, advanced=True),
              Param("tol", "float", 0.01, minimum=0.0, advanced=True, help="stop tolerance of optimize()"),
              Param("sink_fraction", "float", 0.1, help="thermal: heat-sink length on the left edge"),
              Param("length", "float", 1.0, help="double pipe: length of the box (height 1)"),
              Param("q", "list", [0.01, 0.1], help="double pipe: Brinkman interpolation continuation"),
              Param("label", "str", "", help="name of this run in the metrics (default: the problem)")],
      calls="pinneapple_design.topology: Compliance2D.mbb / .cantilever, ThermalCompliance2D, "
            "StokesDissipation2D.double_pipe, DensityFilter.grid, optimize (oc / mma, beta continuation)")
def topology2d(ctx, inputs, params):
    """examples/topology_optimization/01_mbb_beam.py and 02_heat_and_stokes_flow.py."""
    from pinneapple_design.topology import DensityFilter, StokesDissipation2D, optimize
    t0 = time.perf_counter()
    p, filt, shape = _problem(params)
    kw = dict(max_iter=int(params["max_iter"]))
    if params["projection"]:
        kw.update(beta0=float(params["beta0"]), beta_max=float(params["beta_max"]),
                  beta_every=int(params["beta_every"]))
    vol = float(params["volume"])
    uniform = float(p.evaluate(np.full(p.n, vol))[0])
    if params["problem"] == "double_pipe":
        x = None
        for q in params["q"]:                         # continuation on the Brinkman parameter, as the example does
            p = StokesDissipation2D.double_pipe(nely=int(params["nely"]), length=float(params["length"]), q=float(q))
            filt = DensityFilter.grid(p.grid.nelx, p.grid.nely, float(params["rmin"]))
            r = optimize(p, vol, filt, params["method"], move=float(params["move"]), tol=1e-3, x0=x, **kw)
            x = r.x
        uniform = float(p.evaluate(np.full(p.n, vol))[0])
    elif params["problem"] == "thermal":
        r = optimize(p, vol, filt, params["method"], **kw)
    else:
        r = optimize(p, vol, filt, params["method"], **kw)
    name = params["label"] or params["problem"]
    ctx.metric(f"{name}.objective", float(r.objective), "compliance / thermal compliance / dissipated power")
    ctx.metric(f"{name}.uniform", uniform, "the same objective for a uniform design at the volume fraction")
    ctx.metric(f"{name}.iterations", len(r.history["objective"]))
    ctx.metric(f"{name}.volume", float(r.history["volume"][-1]) if r.history.get("volume") else float(np.mean(r.x_phys)))
    ctx.metric(f"{name}.gray_fraction", float(((r.x_phys > 0.05) & (r.x_phys < 0.95)).mean()))
    ctx.metric("seconds", round(time.perf_counter() - t0, 1))
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 8 * shape[1] / max(shape[0], 1) + 0.6))
    ax.imshow(r.x_phys.reshape(shape).T, origin="lower", cmap="Blues" if params["problem"] == "double_pipe"
              else "gray_r", vmin=0, vmax=1)
    ax.set_title(f"{name}: objective {r.objective:.4g}, {len(r.history['objective'])} iterations")
    ax.axis("off")
    f = ctx.figure(fig, f"topology_{name.replace(' ', '_').replace(',', '')}", "optimized layout")
    ctx.summary(f"{name}: {r.objective:.4g} (uniform {uniform:.4g})")
    return {"metrics": ctx.report["metrics"], "figure": f}


@node("design.topology_gradcheck", "Topology sensitivities against finite differences", category="Design",
      outputs=[Port("metrics", "metrics")],
      params=[Param("nelx", "int", 60, minimum=4), Param("nely", "int", 20, minimum=4),
              Param("rmin", "float", 2.4), Param("beta", "float", 4.0), Param("n_check", "int", 8, minimum=1),
              Param("step", "float", 1e-5), Param("seed", "int", 0)],
      calls="pinneapple_design.topology.DesignChain + finite_difference_check (Compliance2D.mbb)")
def topology_gradcheck(ctx, inputs, params):
    """The adjoint sensitivities of the filtered, projected MBB compliance against central differences on the
    largest components (examples/topology_optimization/01_mbb_beam.py)."""
    from pinneapple_design.topology import (
        Compliance2D,
        DensityFilter,
        DesignChain,
        finite_difference_check,
    )
    mbb = Compliance2D.mbb(int(params["nelx"]), int(params["nely"]))
    filt = DensityFilter.grid(int(params["nelx"]), int(params["nely"]), float(params["rmin"]))
    chain = DesignChain(mbb.n, filt, beta=float(params["beta"]))

    def fun(z):
        f, df = mbb.evaluate(chain.forward(z))
        return f, chain.backward(df)
    x = np.random.default_rng(int(params["seed"])).uniform(0.2, 0.9, mbb.n)
    _, g = fun(x)
    err = float(finite_difference_check(fun, x, np.argsort(-np.abs(g))[:int(params["n_check"])],
                                        float(params["step"])).max())
    ctx.metric("gradient_check_max_rel_err", err, "max relative error of the adjoint sensitivities")
    return {"metrics": ctx.report["metrics"]}


@node("design.reluctance_rotor", "Reluctance-rotor iron for torque (topology optimization)", category="Design",
      outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("I_peak", "float", 60.0), Param("gamma_deg", "float", 60.0),
              Param("positions_deg", "list", [0.0, 5.0, 10.0]), Param("volume", "float", 0.5),
              Param("max_iter", "int", 120, minimum=1, maximum=2000), Param("filter_radius", "float", 0.0018),
              Param("mu_r", "float", 1000.0, help="linear iron during the optimization")],
      calls="pinneapple_design.topology.magnetic.RotorTorqueProblem + SymmetryMap.rotational + DensityFilter + "
            "optimize (mma); pinneapple_simulation.electromagnetics.machine.SPMMachine (re-evaluation, linear and "
            "illustrative_steel, co-energy torque)")
def reluctance_rotor(ctx, inputs, params):
    """examples/topology_optimization/03_reluctance_rotor.py: 4-pole rotor core, 4-fold symmetry, adjoint
    sensitivities; the final design re-evaluated with nonlinear iron and with the co-energy torque."""
    from pinneapple_design.topology import (
        DensityFilter,
        DesignChain,
        SymmetryMap,
        finite_difference_check,
        optimize,
    )
    from pinneapple_design.topology.magnetic import RotorTorqueProblem
    from pinneapple_simulation.electromagnetics import LinearMaterial, illustrative_steel
    from pinneapple_simulation.electromagnetics.machine import SPMMachine
    geom = {"n_theta": 240, "slots": 24, "pole_pairs": 2, "magnet_thickness": 0.0, "rotor_core_inner": 0.01,
            "rotor_core_layers": 12, "airgap": 0.0006}
    m = SPMMachine(iron=LinearMaterial(float(params["mu_r"])), **geom)
    thetas = np.radians(np.asarray(params["positions_deg"], float))
    prob = RotorTorqueProblem(m, thetas, I_peak=float(params["I_peak"]), gamma=math.radians(float(params["gamma_deg"])),
                              objective="mean")
    sym = SymmetryMap.rotational(prob.centroids, 4, tol=1e-9)
    filt = DensityFilter(prob.centroids, float(params["filter_radius"]), prob.volumes)
    chain = DesignChain(prob.n, filt, sym, beta=2.0)

    def fun(z):
        f, df = prob.evaluate(chain.forward(z))
        return f, chain.backward(df)
    x_test = np.random.default_rng(0).uniform(0.3, 0.8, sym.n_var)
    _, g = fun(x_test)
    ctx.metric("gradient_check_max_rel_err", float(finite_difference_check(fun, x_test, np.argsort(-np.abs(g))[:5],
                                                                           1e-5).max()))
    ctx.metric("design_variables", int(sym.n_var))
    ctx.metric("solid_round_rotor_torque_Nm", float(-prob.evaluate(np.ones(prob.n))[0]))
    res = optimize(prob, float(params["volume"]), filt, "mma", max_iter=int(params["max_iter"]), symmetry=sym,
                   beta0=1.0, beta_max=8.0, beta_every=25, move=0.2, objective_scale=1.0)
    x = res.x_phys
    T_lin = np.array([m.torque(m.solve(t, prob.currents(t), design=x)) for t in thetas])
    m_nl = SPMMachine(iron=illustrative_steel(), **geom)
    T_nl = np.array([m_nl.torque(m_nl.solve(t, prob.currents(t), design=x)) for t in thetas])
    ctx.metric("optimized_mean_torque_linear_Nm", float(T_lin.mean()))
    for t, v in zip(params["positions_deg"], T_lin, strict=True):
        ctx.metric(f"torque_linear_{float(t):g}deg_Nm", float(v))
    if len(thetas) > 1:
        ctx.metric("coenergy_torque_at_5deg_Nm", float(m.torque_coenergy(thetas[1], prob.currents(thetas[1]),
                                                                         design=x)))
    ctx.metric("optimized_mean_torque_nonlinear_Nm", float(T_nl.mean()))
    ctx.metric("iterations", len(res.history["objective"]))
    ctx.metric("volume_fraction", float(np.sum(prob.volumes * x) / np.sum(prob.volumes)))
    plt = _plt()
    from matplotlib.collections import PolyCollection
    mesh = m.mesh(0.0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].add_collection(PolyCollection(mesh.points[mesh.tris[prob.elements]], array=x, cmap="gray_r",
                                        edgecolors="none", clim=(0, 1)))
    ax[0].set_aspect("equal")
    ax[0].autoscale()
    ax[0].set_title("optimized rotor iron (black)")
    ax[1].plot(-np.array(res.history["objective"]))
    ax[1].set(xlabel="iteration", ylabel="mean torque (N m), linear iron")
    f = ctx.figure(fig, "reluctance_rotor", "optimized rotor iron and the MMA history")
    return {"metrics": ctx.report["metrics"], "figure": f}


@node("design.from_intent", "Design from a plain-language goal", category="Design",
      outputs=[Port("metrics", "metrics")],
      params=[Param("goal", "text", "a glider wing for a 2 kg glider", max_length=2000),
              Param("answers", "dict", {}, help="answers to the questions the goal leaves open"),
              Param("verify", "bool", True, help="the full-fidelity check (needs the case's solver)")],
      calls="pinneapple_design.intent.api.design (pp.design)")
def from_intent(ctx, inputs, params):
    """A goal in plain language to a verified, buildable design, the questions it needs, or a refusal saying what
    is missing (examples/design_from_intent/design_from_intent.py)."""
    import pinneapple as pp
    answers = {k: tuple(v) if isinstance(v, list) else v for k, v in (params["answers"] or {}).items()}
    r = pp.design(params["goal"], answers=answers or None, out_dir=str(ctx.path("design")),
                  verify=bool(params["verify"]), progress=ctx.log)
    d = r.to_dict()
    if not hasattr(r, "summary"):                    # questions or out of scope
        status = d.get("status", type(r).__name__)
        ctx.metric("status", str(status))
        ctx.table("reply", ["key", "value"], [[k, str(v)[:200]] for k, v in d.items()])
        ctx.summary(f"{status}")
        return {"metrics": ctx.report["metrics"]}
    ctx.metric("verdict", r.verdict)
    ctx.metric("probability", float(r.assessment.probability), "probability the design works (assessment)")
    ctx.metric("trust", r.assessment.trust.decisao)
    for k, v in (d.get("design") or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            ctx.metric(f"design.{k}", float(v))
        elif isinstance(v, str):
            ctx.metric(f"design.{k}", v)
    ctx.metric("verification_ran", int(bool(r.verification.ran)), r.verification.label)
    for c in d["verification"].get("comparison") or []:
        for lvl in ("full", "reduced", "closed_form"):
            if isinstance(c.get(lvl), (int, float)):
                ctx.metric(f"{c['output']}.{lvl}", float(c[lvl]))
    ctx.table("full check against the reduced model and the closed form",
              ["output", "full", "reduced", "closed form", "relative error", "kind"],
              [[c.get("output"), c.get("full"), c.get("reduced"), c.get("closed_form"), c.get("rel_err"), c.get("kind")]
               for c in d["verification"].get("comparison") or []])
    ctx.summary(r.summary() if callable(getattr(r, "summary", None)) else r.verdict)
    return {"metrics": ctx.report["metrics"]}


# Ahmed, Ramm & Faltin (1984, SAE 840300), total drag coefficient against the slant angle, Re = 4.3e6
AHMED_1984_CD = {"0": 0.250, "12.5": 0.230, "25": 0.285, "30": 0.378, "35": 0.260, "40": 0.250}


@node("qualitative.ahmed_slant", "Qualitative drag preview of the Ahmed slant, checked against the wind tunnel",
      category="Design", outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("measured_cd", "dict", AHMED_1984_CD, help="slant angle (deg) -> measured CD"),
              Param("U", "float", 60.0), Param("lang", "enum", "en", choices=["en", "pt"])],
      calls="pinneapple_design.geometry.bodies.ahmed_body + pinneapple_design.qualitative.preview + "
            "QualitativePreview.quantify / figure")
def ahmed_slant(ctx, inputs, params):
    """examples/qualitative/03_ahmed_slant.py: the preview's expected drag change per slant against the measured
    one, its direction and the rank correlation; low confidence near the separation angle."""
    from pinneapple_design.geometry.bodies import ahmed_body
    from pinneapple_design.qualitative import preview
    meas = {float(k): float(v) for k, v in dict(params["measured_cd"]).items()}
    geoms = {f"slant {a:g}°": ahmed_body(a) for a in sorted(meas)}
    lang = str(params["lang"])
    p = preview(geoms, "minimizar coeficiente de arrasto" if lang == "pt" else "minimize drag coefficient",
                {"U": float(params["U"])}, lang=lang)
    png = ctx.path("ahmed_preview.png")
    p.figure(str(png))
    check = p.quantify(lambda name, g: meas[float(name.split()[1].rstrip("°"))], variants=list(geoms))
    rows = check.table
    for r in rows:
        if r["direction_ok"] is not None:
            ctx.metric(f"direction_ok.{r['variant'].split()[1].rstrip('°')}", int(r["direction_ok"]),
                       f"expected {r['expected_pct']:+.0f}%, measured {r['computed_pct']:+.0f}% ({r['confidence']})")
    ctx.metric("direction_agreement", float(check.direction_agreement))
    ctx.metric("rank_correlation", float(check.rank_correlation))
    ctx.table("expected against measured", ["variant", "expected %", "measured %", "direction", "confidence"],
              [[r["variant"], round(r["expected_pct"], 1), round(r["computed_pct"], 1),
                "-" if r["direction_ok"] is None else ("right" if r["direction_ok"] else "wrong"), r["confidence"]]
               for r in rows])
    from ..runtime import artifact
    ctx.report["figures"].append({"path": str(png), "name": "ahmed_preview", "caption": "qualitative preview"})
    return {"metrics": ctx.report["metrics"], "figure": artifact(str(png), "png")}


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="design.topology", title="Topology optimization (SIMP): MBB beam, cantilever, heat, Stokes pipes",
        domain="design", example="topology_optimization",
        keywords={"topology optimization": 4, "topology optimisation": 4, "otimização topológica": 4, "simp": 3,
                  "mbb": 3, "compliance": 2, "minimum compliance": 3, "density filter": 2, "volume fraction": 1.5},
        min_score=3, nodes=("design.topology2d", "design.topology_gradcheck", "examples.compare"),
        references=("brenner2008fem",), template="topology_optimization/01_mbb_beam",
        description="examples/topology_optimization/01_mbb_beam.py"))
    register_capability(Capability(
        id="design.reluctance_rotor", title="Reluctance-rotor iron for torque by topology optimization",
        domain="design", example="topology_optimization",
        keywords={"reluctance": 3, "rotor iron": 2.5, "rotor": 1, "torque": 1, "topology": 2},
        requires_all=(("reluctance", "rotor"),), min_score=3.5,
        nodes=("design.reluctance_rotor", "examples.compare"), references=("brenner2008fem",),
        template="topology_optimization/03_reluctance_rotor",
        description="examples/topology_optimization/03_reluctance_rotor.py"))

    register_capability(Capability(
        id="qualitative.ahmed_slant", title="Qualitative drag preview of the Ahmed body slant, against the tunnel",
        domain="design", example="qualitative",
        keywords={"ahmed": 3, "slant": 2, "qualitative": 2, "preview": 1, "drag": 1, "wind tunnel": 1},
        requires_all=(("ahmed", "slant"),), min_score=3, nodes=("qualitative.ahmed_slant",),
        references=("white2011fm",), template="qualitative/03_ahmed_slant",
        description="examples/qualitative/03_ahmed_slant.py"))


_register_capabilities()

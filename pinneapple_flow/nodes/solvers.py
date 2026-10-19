"""Solver nodes: classical (finite differences, finite elements, OpenFOAM), PINN, inverse PINN, calibration."""
from __future__ import annotations

import shutil
import time

import numpy as np

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node
from ..runtime import load_artifact
from ._common import (
    bounds_of,
    eval_axes,
    exact_function,
    grid_points,
    load_solution,
    need_problem,
    plot_solution,
    save_solution,
    system_of,
)


def _fem(problem, n):
    """Steady Poisson / Laplace with Dirichlet faces on a box through ``pp.solve(problem, "fem")``."""
    import sympy as sp

    import pinneapple as pp
    from pinneapple_physics.pde_environment.builder import ProblemBuilder
    from pinneapple_physics.symbolic_pde.text import compile_expr
    sys_ = system_of(problem)
    cls = sys_.classify()
    if cls["family"] not in ("poisson", "laplace") or sys_.time_dependent or len(sys_.fields) != 1:
        raise StepFailed(f"FEM here solves steady Poisson / Laplace (got {cls['family']}); use method 'fd'")
    u = sys_.ctx.functions[sys_.fields[0]]
    R = sys_.substituted()[0].doit()
    lap = sum(sp.Derivative(u, sys_.ctx.symbols[c], 2) for c in sys_.ctx.spatial)
    rest = sp.expand(R - lap)
    if rest.has(u) or rest.atoms(sp.Derivative):
        raise StepFailed("FEM needs the PDE in the form lap(u) = f(x)")
    f_expr = (-rest).xreplace({s: sp.Symbol(s.name) for s in rest.free_symbols})
    f = compile_expr(f_expr, "numpy")
    coords = sys_.coords
    b = ProblemBuilder("flow").domain(**{c: tuple(bounds_of(problem)[c]) for c in coords}).fields(sys_.fields[0])
    b = b.pde("poisson")
    for c in sys_.conditions:
        if c.kind != "dirichlet" or c.lhs != u:
            raise StepFailed(f"FEM here takes Dirichlet conditions 'u = g' only (got {c.kind}: {c.text!r})")
        g_expr = c.rhs.xreplace({s: sp.Symbol(s.name) for s in c.rhs.free_symbols})
        g = compile_expr(g_expr, "numpy")
        faces = [f"{a}_{s}" for a in coords for s in ("min", "max")] if c.on == "boundary" else [c.on]
        for face in faces:
            b = b.bc("dirichlet", field=sys_.fields[0],
                     value=lambda X, g=g: np.broadcast_to(np.asarray(g({k: X[:, i] for i, k in enumerate(coords)}),
                                                                     float), (len(X),)),
                     on=face)
    prob = pp.PhysicalProblem.from_pde_spec(b.build())
    src = lambda X, ctx=None: np.broadcast_to(np.asarray(f({k: X[:, i] for i, k in enumerate(coords)}),  # noqa
                                                         float), (len(X),))
    sol = pp.solve(prob, "fem", ctx={"source_fn": src}, n=int(n) - 1)
    return sol


@node("solver.classical", "Classical solver", category="Solvers",
      inputs=[Port("problem", "problem")],
      outputs=[Port("solution", "solution"), Port("metrics", "metrics")],
      params=[Param("method", "enum", "fd", choices=["fd", "fem"],
                    help="fd: finite differences / method of lines (any PDE system on a box, 1-3-D); "
                         "fem: P1 finite elements (steady Poisson/Laplace, Dirichlet)"),
              Param("n", "int", 0, minimum=0, maximum=2048, help="nodes per axis (0: from the mesh or default)"),
              Param("n_t", "int", 0, minimum=0, maximum=2001, help="saved time levels (0: mesh or 41)"),
              Param("rtol", "float", 1e-6, minimum=1e-12, maximum=1e-2, advanced=True)],
      calls="pinneapple_simulation.numerical_solvers.symbolic_fd.solve_fd | pp.solve(problem, 'fem')")
def classical(ctx, inputs, params):
    """Solve the edited equation system with a classical method and report its cost and residual."""
    problem = need_problem(inputs["problem"])
    sys_ = system_of(problem)
    if sys_.unknowns:
        raise StepFailed(f"unknown coefficients {sys_.unknowns}: use an inverse node, or give them values")
    axes = eval_axes(problem, params["n"] or None, params["n_t"] or None)
    coords = sys_.coords
    ctx.auto("method.fd" if params["method"] == "fd" else "method.fem", f"method: {params['method']}", "method",
             value=params["method"])
    t0 = time.perf_counter()
    if params["method"] == "fd":
        from pinneapple_simulation.numerical_solvers.symbolic_fd import FDError, solve_fd
        n = {c: len(axes[c]) for c in sys_.ctx.spatial}
        try:
            res = solve_fd(sys_, bounds_of(problem), n=n, n_save=len(axes["t"]) if "t" in axes else 41,
                           rtol=params["rtol"])
        except FDError as exc:
            raise StepFailed(f"finite differences: {exc}") from None
        vals = np.stack([res.fields[f] for f in sys_.fields], -1)
        if res.t is not None:
            vals = np.moveaxis(vals, 0, len(sys_.ctx.spatial))
            axes = {**res.axes, "t": res.t}
        else:
            axes = dict(res.axes)
        for k in ("method", "iterations", "nfev", "n_unknowns", "residual_inf"):
            if k in res.info:
                ctx.metric(k, res.info[k])
        if res.info.get("unconstrained_faces"):
            ctx.warn(f"faces without a condition: {res.info['unconstrained_faces']}")
        label = "FD"
    else:
        sol = _fem(problem, len(axes[coords[0]]))
        X = grid_points(axes, coords)
        vals = sol.predict(X).reshape(*[len(axes[c]) for c in coords], -1)
        ctx.metric("n_dof", sol.info.get("n_dof"))
        label = "FEM"
    secs = time.perf_counter() - t0
    ctx.metric("solve_seconds", round(secs, 4), "wall clock of the solve on this machine")
    out = save_solution(ctx, "solution", axes, vals, coords, sys_.fields, params["method"], label)
    ctx.figure(plot_solution(load_solution(out), f"{label} solution"), "solution")
    ctx.summary(f"{label} on {'x'.join(str(len(axes[c])) for c in coords)} points in {secs:.2f} s")
    return {"solution": out, "metrics": {"solve_seconds": secs, "method": params["method"]}}


def _openfoam_available():
    if shutil.which("blockMesh") and (shutil.which("icoFoam") or shutil.which("simpleFoam")):
        return True, ""
    return False, "OpenFOAM is not installed (blockMesh / icoFoam not on PATH)"


@node("solver.openfoam", "OpenFOAM case", category="Solvers",
      outputs=[Port("metrics", "metrics")],
      params=[Param("case", "str", "", help="path of an OpenFOAM case directory (under the data folder)"),
              Param("solver", "str", "icoFoam")],
      calls="pinneapple_core.workflows.OpenFOAMAdapter", available=_openfoam_available)
def openfoam(ctx, inputs, params):
    """Run an OpenFOAM case with the workflow engine's OpenFOAM adapter (blockMesh + solver) and read the solver
    log health. Needs OpenFOAM on PATH."""
    from pinneapple_core.workflows import LocalExecutor, OpenFOAMAdapter, Step, StepContext

    from ..config import safe_data_path
    case = safe_data_path(params["case"])
    step = Step(id=ctx.node_id, kind="openfoam", params={"case": str(case),
                                                          "commands": [["blockMesh"], [params["solver"]]]})
    out = OpenFOAMAdapter().run(StepContext("flow", step, ctx.workdir, LocalExecutor(), {}, ctx.log))
    for k, v in out.items():
        if isinstance(v, (int, float, str, bool)):
            ctx.metric(k, v)
    return {"metrics": out}


def _architecture_view(model):
    from ..learn import layer_graph
    return layer_graph(model)


def _pinn_common(ctx, problem, params, unknowns=None, data=None):
    from pinneapple_analysis.validation.pinn import train_pinn, validate_pinn
    from pinneapple_physics.symbolic_pde.pinn import pinn_problem
    sys_ = system_of(problem)
    b = bounds_of(problem)
    exact = exact_function(problem)
    try:
        P, model = pinn_problem(sys_, b, unknowns=unknowns, width=int(params["width"]), depth=int(params["depth"]),
                                exact=exact, name=ctx.node_id)
    except ValueError as exc:
        raise StepFailed(str(exc)) from None
    ctx.view("architecture", _architecture_view(model))
    ctx.auto("architecture.default_sizes", f"tanh MLP width {params['width']}, depth {params['depth']}", "width",
             "depth", value={"width": params["width"], "depth": params["depth"]})
    ctx.auto("method.pinn_training", f"{params['adam_steps']} Adam steps (lr {params['lr']}) then "
             f"{params['lbfgs_steps']} L-BFGS", "adam_steps", "lbfgs_steps", "lr",
             value={k: params[k] for k in ("adam_steps", "lbfgs_steps", "lr", "n_collocation")})
    ctx.decide("validation.threshold", "PINN checked at fresh points with validate_pinn's tolerances")
    steps = int(params["adam_steps"])
    every = max(1, steps // 20)

    def cb(step, loss):
        if step % every == 0:
            ctx.log(f"adam step {step}/{steps}: loss {loss:.3e}")
        ctx.check_stop()
    t0 = time.perf_counter()
    net, info = train_pinn(P, model, n_collocation=int(params["n_collocation"]), adam_steps=steps,
                           lbfgs_steps=int(params["lbfgs_steps"]), lr=float(params["lr"]), seed=int(params["seed"]),
                           data=data, callback=cb)
    secs = time.perf_counter() - t0
    ctx.metric("train_seconds", round(secs, 2), "wall clock of training on this machine (CPU)")
    ctx.metric("final_loss", info["loss"])
    ctx.metric("n_parameters", int(sum(p.numel() for p in net.net.parameters())))
    rep = validate_pinn(net, P, train_points=info["train_points"])
    ctx.table("PINN validation (fresh points)", ["check", "status", "summary"],
              [[c["id"], c["status"], c["summary"]] for c in rep.checks])
    for c in rep.checks:
        if c["id"] == "pinn.exact" and c["value"] is not None:
            ctx.metric("rel_l2_vs_exact", c["value"], "validate_pinn on a fresh grid")
        if c["id"] == "pinn.residual" and c["value"] is not None:
            ctx.metric("residual_rms_fresh", c["value"], "relative PDE residual at fresh points")
    ctx.metric("validation", "pass" if rep.passed else "fail")
    if rep.failure_modes:
        ctx.warn("failure modes: " + "; ".join(rep.failure_modes))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.semilogy(info["history"])
    ax.set_xlabel("Adam step")
    ax.set_ylabel("loss")
    ctx.figure(fig, "loss", "training loss")
    axes = eval_axes(problem)
    X = grid_points(axes, sys_.coords)
    import torch
    with torch.no_grad():
        U = net(torch.tensor(X, dtype=torch.float32)).numpy()
    own = [{"id": c["id"], "passed": c["status"] == "pass", "status": c["status"], "value": c["value"],
            "detail": c["summary"]} for c in rep.checks if c["status"] in ("pass", "fail")]
    sol = save_solution(ctx, "solution", axes, U, sys_.coords, sys_.fields, "pinn", "PINN",
                        extra={"own_checks": own})          # validate.compare passes them on to the Lab
    ctx.figure(plot_solution(load_solution(sol), "PINN solution"), "solution")
    mdl = ctx.save_torch("model", {"kind": "pinn", "state": net.state_dict(), "width": int(params["width"]),
                                   "depth": int(params["depth"]), "coords": sys_.coords, "fields": sys_.fields})
    return net, rep, sol, mdl, secs


_PINN_PARAMS = [Param("width", "int", 32, minimum=4, maximum=512), Param("depth", "int", 4, minimum=1, maximum=12),
                Param("adam_steps", "int", 2000, minimum=0, maximum=200000),
                Param("lbfgs_steps", "int", 200, minimum=0, maximum=20000),
                Param("lr", "float", 2e-3, minimum=1e-6, maximum=1.0),
                Param("n_collocation", "int", 2000, minimum=16, maximum=200000),
                Param("seed", "int", 0, minimum=0, advanced=True)]


@node("train.pinn", "PINN training", category="Learning",
      inputs=[Port("problem", "problem"), Port("data", "dataset", required=False,
                                               help="observations (physics + data training)")],
      outputs=[Port("model", "model"), Port("solution", "solution"), Port("metrics", "metrics")],
      params=list(_PINN_PARAMS),
      calls="pinneapple_physics.symbolic_pde.pinn.pinn_problem + pinneapple_analysis.validation.pinn.train_pinn / "
            "validate_pinn")
def pinn(ctx, inputs, params):
    """Train a physics-informed network on the edited equations (Adam then L-BFGS), optionally with observations,
    and validate it at fresh points (residual, boundary, initial, trivial solution, causality, spectrum, exact)."""
    problem = need_problem(inputs["problem"])
    data = None
    if inputs.get("data"):
        d = load_artifact(inputs["data"])
        if "X" not in d:
            raise StepFailed("physics + data training needs point observations (X, Y)")
        data = (d["X"], d["Y"])
        ctx.log(f"{len(d['X'])} observations in the loss")
    if system_of(problem).unknowns:
        raise StepFailed("the equations have unknown coefficients: use the parameter discovery node")
    net, rep, sol, mdl, secs = _pinn_common(ctx, problem, params, data=data)
    ctx.summary(f"PINN trained in {secs:.1f} s; validation {'passed' if rep.passed else 'failed'}")
    return {"model": mdl, "solution": sol, "metrics": ctx.report["metrics"]}


@node("inverse.discover", "Parameter discovery (inverse PINN)", category="Inverse",
      inputs=[Port("problem", "problem"), Port("data", "dataset")],
      outputs=[Port("params", "params"), Port("solution", "solution"), Port("model", "model"),
               Port("metrics", "metrics")],
      params=[Param("initial_guess", "dict", {}, help="starting values of the unknown coefficients"),
              *_PINN_PARAMS],
      calls="pinneapple_physics.symbolic_pde.pinn.pinn_problem(unknowns=...) + train_pinn(data=...)")
def discover(ctx, inputs, params):
    """Identify unknown coefficients of the PDE together with the field, from observations (inverse PINN)."""
    problem = need_problem(inputs["problem"])
    sys_ = system_of(problem)
    if not sys_.unknowns:
        raise StepFailed("no unknown coefficients: list them in the equations node's 'unknowns'")
    d = load_artifact(inputs["data"])
    guess = {k: float(params["initial_guess"].get(k, 1.0)) for k in sys_.unknowns}
    ctx.decide("inverse.pinn", f"identify {', '.join(sys_.unknowns)} with an inverse PINN", value=guess)
    net, rep, sol, mdl, secs = _pinn_common(ctx, problem, params, unknowns=guess, data=(d["X"], d["Y"]))
    est = net.estimates()
    truth = d.get("truth_json")
    rows = []
    for k, v in est.items():
        ctx.metric(f"{k}_estimate", v)
        tv = None
        if "truth_names" in d:
            names = [str(s) for s in d["truth_names"]]
            if k in names:
                tv = float(d["truth_values"][names.index(k)])
                ctx.metric(f"{k}_relative_error", abs(v - tv) / max(abs(tv), 1e-300), "against the value that "
                           "generated the synthetic data")
        rows.append([k, guess[k], v, tv])
    del truth
    ctx.table("identified coefficients", ["name", "initial guess", "estimate", "true (synthetic data)"], rows)
    ctx.summary(", ".join(f"{k} = {v:.5g}" for k, v in est.items()))
    return {"params": est, "solution": sol, "model": mdl, "metrics": ctx.report["metrics"]}


@node("inverse.calibrate", "Calibration (classical forward model)", category="Inverse",
      inputs=[Port("problem", "problem"), Port("data", "dataset")],
      outputs=[Port("params", "params"), Port("metrics", "metrics")],
      params=[Param("bounds", "dict", {}, help="{name: [low, high]} for each unknown"),
              Param("initial_guess", "dict", {}), Param("n", "int", 41, minimum=5, maximum=512),
              Param("bootstrap", "int", 0, minimum=0, maximum=200),
              Param("max_nfev", "int", 60, minimum=5, maximum=2000)],
      calls="pinneapple_analysis.inverse_problems.calibrate with a CallableModel around symbolic_fd.solve_fd")
def calibrate(ctx, inputs, params):
    """Least-squares calibration of the unknown coefficients with the finite-difference solver as forward model
    (the library's calibration: identifiability, intervals, fit metrics)."""
    from pinneapple_analysis import inverse_problems as inv
    from pinneapple_simulation.numerical_solvers.symbolic_fd import FDError, solve_fd

    from ._common import interpolate
    problem = need_problem(inputs["problem"])
    sys_ = system_of(problem)
    if not sys_.unknowns:
        raise StepFailed("no unknown coefficients: list them in the equations node's 'unknowns'")
    d = load_artifact(inputs["data"])
    X, Y = np.asarray(d["X"], float), np.asarray(d["Y"], float)
    coords = sys_.coords
    bnd = bounds_of(problem)
    n = {c: int(params["n"]) for c in sys_.ctx.spatial}
    calls = [0]

    def forward(theta, ins):
        calls[0] += 1
        ctx.check_stop()
        try:
            res = solve_fd(sys_, bnd, n=n, n_save=21, coefficients=dict(theta))
        except FDError as exc:
            raise StepFailed(str(exc)) from None
        vals = np.stack([res.fields[f] for f in sys_.fields], -1)
        axes = dict(res.axes)
        if res.t is not None:
            vals = np.moveaxis(vals, 0, len(sys_.ctx.spatial))
            axes["t"] = res.t
        P = np.stack([np.asarray(ins[c], float) for c in coords], 1)
        return interpolate({"axes": axes, "values": vals, "coords": coords}, P)[:, 0]

    pars = []
    for k in sys_.unknowns:
        lo, hi = params["bounds"].get(k, [1e-4, 10.0])
        pars.append(inv.Parameter(k, float(params["initial_guess"].get(k, (lo * hi) ** 0.5 if lo > 0 else
                                                                        (lo + hi) / 2)), lower=float(lo),
                                  upper=float(hi)))
    model = inv.CallableModel(forward, inputs=coords, parameters=pars)
    ctx.decide("inverse.calibration", f"least squares on {', '.join(sys_.unknowns)} with the FD solver",
               value={p.name: [p.lower, p.upper] for p in pars})
    sigma = float(np.std(Y[:, 0]) * 0.01 + 1e-9) if "sigma" not in d else float(d["sigma"])
    obs = inv.Observations({c: X[:, i] for i, c in enumerate(coords)}, sigma, values=Y[:, 0])
    t0 = time.perf_counter()
    res = inv.calibrate(model, obs, bootstrap=int(params["bootstrap"]), max_nfev=int(params["max_nfev"]))
    secs = time.perf_counter() - t0
    ctx.metric("forward_solves", calls[0])
    ctx.metric("calibration_seconds", round(secs, 2))
    rows = []
    for p in res.to_dict().get("parameters", []):
        rows.append([p.get("name"), p.get("calibrated"), p.get("lower"), p.get("upper"), p.get("identifiability")])
    ctx.table("calibrated parameters", ["name", "value", "interval low", "interval high", "identifiability"], rows)
    for k, v in res.theta.items():
        ctx.metric(f"{k}_estimate", v)
        if "truth_names" in d:
            names = [str(s) for s in d["truth_names"]]
            if k in names:
                tv = float(d["truth_values"][names.index(k)])
                ctx.metric(f"{k}_relative_error", abs(v - tv) / max(abs(tv), 1e-300))
    for k, v in (res.metrics or {}).items():
        if isinstance(v, (int, float)):
            ctx.metric(f"fit_{k}", v)
    ctx.summary(", ".join(f"{k} = {v:.5g}" for k, v in res.theta.items()))
    return {"params": dict(res.theta), "metrics": ctx.report["metrics"]}

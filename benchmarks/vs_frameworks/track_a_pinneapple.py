# Track A (PINN forward problems), PINNeAPPle runner.
# Follows the library's documented entry points at the commit under test:
#   A2: ``pp.solve(problem, "pinn", ...)`` with its defaults (AGENTS.md "Solve and compare" / "Equal-budget
#       comparison"; pinneapple_physics/solving.py::_pinn: modified_mlp 64 x 4, Adam lr 1e-3, 2048 fresh collocation
#       points per step, condition weights 10 x LossWeights(w_bc=10, w_ic=10)); Burgers from
#       ``pp.PhysicalProblem.from_preset("burgers_1d", nu=0.01 / pi)``; the others from ``ProblemBuilder``.
#       pp.solve has no time cap, so the number of epochs is set from a 100-epoch probe to fill the cap.
#   A1: ``pinneapple_physics.solve_pde(spec, model, x_col=..., x_bc=..., x_ic=..., weights=LossWeights(1, 1, 1, 1))``
#       with the registry's ``vanilla_pinn`` (tanh MLP) built 50 x 4 and loaded with the shared Xavier weights.
"""PINNeAPPle runner of track A (see the comment above and PROTOCOL.md)."""
from __future__ import annotations

import time

import numpy as np

from pinneapple_tools.benchmark_suite.crossframework import problems as P
from pinneapple_tools.benchmark_suite.crossframework import protocol as proto
from pinneapple_tools.benchmark_suite.crossframework import records as R
from pinneapple_tools.benchmark_suite.crossframework import weights as W

SOURCE = ("PINNeAPPle at the commit under test: pp.solve(problem, 'pinn') defaults (A2), "
          "pinneapple_physics.solve_pde with fixed points (A1)")


def builder_problem(prob: P.PINNProblem, weight: float, single_bc: bool = False):
    """ProblemSpec of ``prob`` through ProblemBuilder (the documented way to describe a problem). ``single_bc``: one
    Dirichlet condition on the whole boundary (A1: one boundary loss term, as in the other frameworks)."""
    from pinneapple_physics.pde_environment.builder import ProblemBuilder
    (a0, a1), (b0, b1) = prob.bounds
    if prob.name == "burgers":
        b = (ProblemBuilder("burgers_bench").domain(x=(a0, a1), t=(b0, b1)).fields("u")
             .pde("burgers", nu=prob.params["nu"])
             .bc("dirichlet", field="u", value=0.0, on="x_boundary", weight=weight, name="walls")
             .ic(field="u", fn=lambda X: -np.sin(np.pi * X[:, 0:1]), weight=weight, name="u0"))
    elif prob.name == "allen_cahn":
        # Allen-Cahn u_t = d u_xx + 5 u - 5 u^3 is the compiler's "opinion_dynamics_2d" kind
        # (u_t - D lap u - alpha u (1 - u^2)) with D = d and alpha = 5; its laplacian runs over the spatial coords only.
        # PINNeAPPle has no periodic condition: u(+-1, t) = -1 (what DeepXDE's official example imposes).
        b = (ProblemBuilder("allen_cahn_bench").domain(x=(a0, a1), t=(b0, b1)).fields("u")
             .pde("opinion_dynamics_2d", D=prob.params["d"], alpha=5.0)
             .bc("dirichlet", field="u", value=-1.0, on="x_boundary", weight=weight, name="walls")
             .ic(field="u", fn=lambda X: X[:, 0:1] ** 2 * np.cos(np.pi * X[:, 0:1]), weight=weight, name="u0"))
    else:
        kind = "poisson" if prob.name == "poisson" else "helmholtz"
        params = {} if kind == "poisson" else {"k": prob.params["k"]}
        b = ProblemBuilder(f"{prob.name}_bench").domain(x=(a0, a1), y=(b0, b1)).fields("u").pde(kind, **params)
        if single_bc:
            on_box = lambda X, ctx=None: (np.isclose(X[:, 0], a0) | np.isclose(X[:, 0], a1)  # noqa: E731
                                | np.isclose(X[:, 1], b0) | np.isclose(X[:, 1], b1))
            b = b.bc("dirichlet", field="u", value=0.0, on=on_box, weight=weight, name="walls")
        else:
            b = (b.bc("dirichlet", field="u", value=0.0, on="x_boundary", weight=weight, name="x_walls")
                 .bc("dirichlet", field="u", value=0.0, on="y_boundary", weight=weight, name="y_walls"))
    return b.build()


def source_ctx(prob: P.PINNProblem):
    if prob.time_dependent:
        return None
    return {"source_fn": lambda X, ctx=None: prob.source(np.asarray(X, dtype=np.float64))[:, None]}


def run_a1(args, prob, meta):
    import torch

    import pinneapple_neural.architectures  # noqa: F401  (registers the model zoo)
    from pinneapple_neural.architectures.registry import ModelRegistry
    from pinneapple_physics import solve_pde
    from pinneapple_physics.pinn_solver.compiler.loss import LossWeights

    pts = P.load_point_set(proto.ensure_point_set(prob, args.root))
    meta["config"]["point_set_sha256"] = P.point_set_digest(pts)
    spec = builder_problem(prob, weight=1.0, single_bc=True)
    torch.manual_seed(args.seed)
    model = ModelRegistry.build("vanilla_pinn", in_dim=2, out_dim=1, hidden=[proto.A1["width"]] * proto.A1["depth"],
                                activation="tanh")
    W.load_into_linears([m for m in model.net if isinstance(m, torch.nn.Linear)],
                        W.xavier_uniform_init(W.layer_sizes(), args.seed))
    t = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
    kw = {"x_bc": t(pts["X_b"]), "y_bc": t(pts["u_b"])}
    if prob.time_dependent:
        kw.update(x_ic=t(pts["X_0"]), y_ic=t(pts["u_0"]))
    ctx = source_ctx(prob)
    with R.Clock() as clk:
        out = solve_pde(spec, model, epochs=args.iters, device=args.device, lr=proto.A1["lr"], seed=args.seed,
                        weights=LossWeights(w_pde=1.0, w_bc=1.0, w_ic=1.0, w_data=1.0), ctx=ctx,
                        x_col=pts["X_f"], **kw)
    trained = out["model"].eval()

    def predict(X):
        with torch.no_grad():
            y = trained(torch.as_tensor(X, device=args.device))
            return getattr(y, "y", y).cpu().numpy()

    hist = out["history"]["loss"]
    meta.update(iterations=len(hist), loss_history=hist[:: max(1, len(hist) // 200)], final_loss=hist[-1],
                n_parameters=W.count_parameters(trained))
    return clk, predict


def run_a2(args, prob, meta):
    import pinneapple as pp

    if prob.name == "burgers":
        problem = pp.PhysicalProblem.from_preset("burgers_1d", nu=prob.params["nu"])
    else:
        problem = pp.PhysicalProblem.from_pde_spec(builder_problem(prob, weight=10.0))
    ctx = source_ctx(prob)
    opts = {"seed": args.seed, "device": args.device}
    if ctx:
        opts["ctx"] = ctx
    # Two probes (50 and 150 epochs): the slope is the cost per epoch, the intercept the fixed setup cost, so the
    # epochs fill the cap (a single probe counted the setup as per-epoch cost and used only ~80 % of the cap).
    times = {}
    for n in (50, 150):
        cap = R.TimeCap(1e9, args.cap_kind)
        pp.solve(problem, "pinn", epochs=n, **opts)
        times[n] = cap.used()
    per_epoch = max((times[150] - times[50]) / 100.0, 1e-6)
    setup = max(times[50] - 50 * per_epoch, 0.0)
    epochs = max(1, int((args.cap_s - setup) / per_epoch))
    meta["config"].update(probe_epochs=[50, 150], probe_s=times, probe_s_per_epoch=per_epoch, probe_setup_s=setup,
                          epochs=epochs)
    with R.Clock() as clk:
        sol = pp.solve(problem, "pinn", epochs=epochs, **opts)
    hist = sol.history.get("loss", [])
    meta.update(iterations=len(hist), loss_history=hist[:: max(1, len(hist) // 200)],
                final_loss=hist[-1] if hist else None, n_parameters=sol.info.get("n_parameters"),
                architecture=sol.info.get("architecture"))
    return clk, lambda X: sol.predict(X)


def main(argv=None):
    args = proto.runner_parser("PINNeAPPle, track A").parse_args(argv)
    R.setup_threads(args.threads)
    prob = P.get_problem(args.problem)
    cfg = dict(proto.A1) if args.mode == "a1" else {"recipe": "pp.solve defaults", **proto.A2, "cap_s": args.cap_s}
    meta = proto.base_meta(args, track="A", framework="pinneapple", source=SOURCE, config=cfg)
    t0 = time.perf_counter()
    try:
        clk, predict = (run_a1 if args.mode == "a1" else run_a2)(args, prob, meta)
        arrays = proto.export_predictions(prob, predict)
        meta.update(status="ok", wall_s=clk.wall_s, cpu_s=clk.cpu_s, loadavg_before=clk.loadavg_before,
                    loadavg_after=clk.loadavg_after)
    except Exception as exc:  # noqa: BLE001 - a failure is a result
        import traceback
        arrays = {}
        meta.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-4000:])
    meta["process_wall_s"] = time.perf_counter() - t0
    R.write_run(args.out, args.run_id, meta, arrays)


if __name__ == "__main__":
    main()

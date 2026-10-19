"""Method nodes: verification, data assimilation, uncertainty propagation, reduced-order models, MeshGraphNet,
meta-learning, the PINN benchmark arena, multi-objective optimisation and control. Each wraps the library's own
implementation; the node only adapts flow inputs (a problem, a solution, a series, a surrogate) to it."""
from __future__ import annotations

import time

import numpy as np

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node
from ..runtime import load_artifact
from ._common import bounds_of, exact_function, grid_points, load_solution, need_problem, system_of


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ------------------------------------------------------------------------------------------------ verification
@node("verify.convergence", "Grid convergence (Richardson, GCI)", category="Validation",
      inputs=[Port("problem", "problem")],
      outputs=[Port("metrics", "metrics"), Port("figure", "figure", required=False)],
      params=[Param("n", "int", 17, minimum=5, maximum=257, help="nodes per axis of the coarsest grid"),
              Param("quantity", "enum", "mean", choices=["mean", "max", "center"],
                    help="scalar of the (final-time) field compared across the grids"),
              Param("safety_factor", "float", 1.25, minimum=1.0, maximum=3.0)],
      calls="pinneapple_simulation.numerical_solvers.symbolic_fd.solve_fd + "
            "pinneapple_analysis.verification.convergence.richardson_extrapolate + pinneapple_lab.uq.gci (the check "
            "pinneapple_lab.uq.gci_check records)")
def convergence(ctx, inputs, params):
    """Solve the problem by finite differences on three nested grids (ratio 2), measure the observed order of a
    scalar of the solution, extrapolate it (Richardson) and give the grid convergence index; with an exact solution
    also the error on each grid and the order it implies."""
    from pinneapple_analysis.verification.convergence import richardson_extrapolate
    from pinneapple_simulation.numerical_solvers.symbolic_fd import FDError, solve_fd
    problem = need_problem(inputs["problem"])
    sys_ = system_of(problem)
    bounds = bounds_of(problem)
    exact = exact_function(problem)
    ctx.decide("verification.richardson", "three nested grids, ratio 2, Richardson extrapolation + GCI")
    n0 = int(params["n"])
    ns = [n0, 2 * n0 - 1, 4 * n0 - 3]
    q, errs, hs = [], [], []
    for n in ns:
        ctx.check_stop()
        try:
            res = solve_fd(sys_, bounds, n={c: n for c in sys_.ctx.spatial}, n_save=11)
        except FDError as exc:
            raise StepFailed(f"finite differences on {n} nodes: {exc}") from None
        f = res.final(sys_.fields[0])
        if params["quantity"] == "mean":                 # trapezoid average over the domain (a node average adds
            v = np.asarray(f, float)                     # an O(h) term of its own)
            for c in sys_.ctx.spatial:
                xs = np.asarray(res.axes[c], float)
                v = np.trapezoid(v, xs, axis=0) / (xs[-1] - xs[0])
            v = float(v)
        elif params["quantity"] == "max":
            v = float(np.max(np.abs(f)))
        else:
            v = float(f[tuple(s // 2 for s in f.shape)])
        q.append(v)
        hs.append((bounds[sys_.ctx.spatial[0]][1] - bounds[sys_.ctx.spatial[0]][0]) / (n - 1))
        if exact is not None:
            axes = dict(res.axes)
            X = grid_points(axes, list(sys_.ctx.spatial))
            if res.t is not None:
                X = np.concatenate([X, np.full((len(X), 1), float(res.t[-1]))], 1)
            ex = exact(X)[:, 0].reshape(f.shape)
            errs.append(float(np.linalg.norm(f - ex) / max(np.linalg.norm(ex), 1e-300)))
        ctx.log(f"{n} nodes per axis: quantity {v:.10g}")
    r = richardson_extrapolate(q[0], q[1], q[2], 2.0, safety_factor=float(params["safety_factor"]))
    rd = r.to_dict() if hasattr(r, "to_dict") else dict(vars(r))
    for k in ("observed_order", "extrapolated_value", "gci_fine", "gci_coarse", "asymptotic_ratio", "is_asymptotic"):
        if k in rd and isinstance(rd[k], (int, float, bool)):
            ctx.metric(k, rd[k], "pinneapple_analysis.verification.convergence.richardson_extrapolate")
    rows = [[n, h, v] + ([e] if errs else []) for n, h, v, e in zip(ns, hs, q, errs or [None] * 3, strict=True)]
    ctx.table("grids", ["nodes per axis", "h", f"quantity ({params['quantity']})"] + (["rel. L2 vs exact"]
                                                                                     if errs else []), rows)
    if errs and min(errs) > 0:
        orders = [float(np.log(errs[i] / errs[i + 1]) / np.log(2.0)) for i in range(2)]
        ctx.metric("order_from_exact_error", orders[-1], "log2 of the error ratio of the two finest grids")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 3.3))
    if errs:
        ax.loglog(hs, errs, "o-", label="rel. L2 vs exact")
        ax.set_ylabel("error")
    else:
        ax.semilogx(hs, q, "o-", label=params["quantity"])
        ax.axhline(rd.get("extrapolated_value", np.nan), ls=":", color="k", label="Richardson")
        ax.set_ylabel(params["quantity"])
    ax.set_xlabel("h")
    ax.legend(fontsize=8)
    fa = ctx.figure(fig, "convergence", "grid convergence")
    order = rd.get("observed_order")
    ctx.summary(f"observed order {order:.2f}, GCI (fine) {rd.get('gci_fine', float('nan')):.2e}"
                if isinstance(order, (int, float)) else "grid convergence computed")
    from pinneapple_lab.uq import gci
    g = gci(q, hs)                                       # the Lab's GCI, recorded as its gci_check does
    ctx.metric("lab_gci", g["gci"], "pinneapple_lab.uq.gci: relative band of the fine-grid value")
    checks = [{"name": "grid_convergence_asymptotic", "passed": bool(rd.get("is_asymptotic")), "kind": "physics",
               "value": rd.get("asymptotic_ratio"), "detail": "Richardson: the three grids are in the asymptotic range"},
              {"name": f"{params['quantity']}_discretisation_uncertainty", "value": g["gci"], "max": 0.05,
               "passed": bool(g["gci"] <= 0.05), "kind": "uncertainty",
               "detail": f"grid convergence index of the fine-mesh value (3 meshes, order {g['order']:.2f}, "
                         f"safety {g['safety']:g})"}]
    return {"metrics": {**{k: v for k, v in rd.items() if isinstance(v, (int, float, bool, str))}, "checks": checks,
                        "candidates": {}}, "figure": fa}


# ------------------------------------------------------------------------------------------------ assimilation
@node("assimilate.filter", "Kalman filter (EKF / EnKF)", category="Digital twin",
      inputs=[Port("series", "series", help="noisy measurements, one channel per column")],
      outputs=[Port("series", "series"), Port("metrics", "metrics"), Port("figure", "figure", required=False)],
      params=[Param("method", "enum", "ekf", choices=["ekf", "enkf"]),
              Param("model", "enum", "constant_velocity", choices=["random_walk", "constant_velocity"],
                    help="state model of each channel between samples"),
              Param("obs_std", "float", 0.0, minimum=0.0, help="measurement noise (0: estimated from the series)"),
              Param("process_std", "float", 0.0, minimum=0.0,
                    help="process noise (0: from the second differences of the series)"),
              Param("n_ens", "int", 100, minimum=10, maximum=2000)],
      calls="pinneapple_analysis.state_estimation.kalman.ExtendedKalmanFilter / EnsembleKalmanFilter")
def kalman(ctx, inputs, params):
    """Estimate the underlying states of noisy channels with the library's Kalman filters (each channel a random
    walk, or position + rate); reports the innovation statistics and the filtered series."""
    from pinneapple_analysis.state_estimation.kalman import (
        EnsembleKalmanFilter,
        ExtendedKalmanFilter,
    )
    s = load_artifact(inputs["series"])
    t = np.asarray(s["t"], float)
    Y = np.asarray(s["values"], float).reshape(len(t), -1)
    m = Y.shape[1]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    noise = ((inputs["series"] or {}).get("meta") or {}).get("noise")     # a live-data node states its noise
    r = float(params["obs_std"]) or float(noise or 0) or \
        float(np.median(np.std(np.diff(Y, axis=0), axis=0)) / np.sqrt(2) + 1e-12)
    qv = float(params["process_std"]) or float(np.median(np.std(np.diff(Y, 2, axis=0), axis=0)) / 2 + 1e-12)
    cv = params["model"] == "constant_velocity"
    n = 2 * m if cv else m
    A = np.eye(n)
    if cv:
        A[:m, m:] = dt * np.eye(m)

    def f(x):
        return A @ x

    def h(x):
        return x[:m]
    Q = np.eye(n) * qv ** 2
    R = np.eye(m) * r ** 2
    ctx.decide("assimilation.kalman", f"{params['method'].upper()} with a {params['model'].replace('_', ' ')} "
                                      "state model", value=params["method"])
    if params["method"] == "ekf":
        kf = ExtendedKalmanFilter(n, m, f, h, Q=Q, R=R, F_jac=lambda x: A, H_jac=lambda x: np.eye(m, n))
    else:
        kf = EnsembleKalmanFilter(n, m, f, h, Q=Q, R=R, n_ens=int(params["n_ens"]), seed=0)
    x0 = np.zeros(n)
    x0[:m] = Y[0]
    kf.initialize(x0, np.eye(n) * max(r ** 2, 1e-12) * 10)
    est, innov = [], []
    for y in Y:
        out = kf.step(y)
        est.append(np.asarray(out["x"], float)[:m])
        innov.append(np.asarray(out["innovation"], float))
    est, innov = np.asarray(est), np.asarray(innov)
    ctx.metric("obs_std", r, "given, stated by the stream, or the median channel std of first differences / "
                             "sqrt(2)")
    ctx.metric("innovation_rms", float(np.sqrt(np.mean(innov ** 2))), "measurement minus forecast, all channels")
    ctx.metric("innovation_rms_over_obs_std", float(np.sqrt(np.mean(innov ** 2)) / r),
               "about 1 or a bit above when the model and the noise levels fit the data")
    ctx.metric("residual_rms", float(np.sqrt(np.mean((Y - est) ** 2))), "measurement minus estimate")
    out = ctx.save_npz("filtered", t=t, values=est)
    out.update({"channels": list((inputs["series"] or {}).get("channels") or [f"y{i}" for i in range(m)])})
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 3))
    for i in range(min(m, 4)):
        ax.plot(t, Y[:, i], ".", ms=2, alpha=0.4)
        ax.plot(t, est[:, i], lw=1.2)
    ax.set_xlabel("t")
    fa = ctx.figure(fig, "filtered", "measurements (dots) and the filtered estimate")
    ctx.summary(f"{params['method'].upper()} on {m} channel(s), {len(t)} samples; innovation RMS / obs std "
                f"{np.sqrt(np.mean(innov ** 2)) / r:.2f}")
    return {"series": out, "metrics": ctx.report["metrics"], "figure": fa}


# ------------------------------------------------------------------------------------------------ uncertainty
@node("uq.propagate", "Uncertainty propagation (Monte Carlo / PCE)", category="Analysis",
      inputs=[Port("model", "model", help="a trained surrogate (its input ranges define the space)")],
      outputs=[Port("metrics", "metrics"), Port("figure", "figure", required=False)],
      params=[Param("method", "enum", "lhs", choices=["lhs", "mc", "sobol", "pce"]),
              Param("n", "int", 2048, minimum=16, maximum=200000),
              Param("degree", "int", 3, minimum=1, maximum=8, help="polynomial chaos degree")],
      calls="pinneapple_analysis.uncertainty.tolerance.propagate_space / pinneapple_analysis.uncertainty.pce."
            "polynomial_chaos")
def propagate(ctx, inputs, params):
    """Propagate uniform uncertainty over the surrogate's input ranges to its output (the mean of the predicted
    field): sampled (Latin hypercube, Monte Carlo, Sobol) or by polynomial chaos."""
    from ..libcalls import _surrogate_fn
    fn, space = _surrogate_fn(inputs["model"])
    if not space:
        raise StepFailed("the surrogate has no input ranges")
    ctx.decide("uq.propagation", f"{params['method']} over the surrogate's training ranges", value=params["method"])

    def f(P):
        return fn(np.asarray(P, float))
    if params["method"] == "pce":
        from pinneapple_analysis.uncertainty.pce import polynomial_chaos
        res = polynomial_chaos(f, space, degree=int(params["degree"]), seed=0)
        d = res.to_dict() if hasattr(res, "to_dict") else dict(vars(res))
        ctx.metric("mean", float(d["mean"]), "polynomial chaos")
        ctx.metric("std", float(d["std"]), "polynomial chaos")
        ctx.metric("n_evals", int(d["n_evals"]))
        ctx.summary(f"PCE degree {params['degree']}: mean {float(d['mean']):.4g}, std {float(d['std']):.3g}")
        return {"metrics": ctx.report["metrics"]}
    from pinneapple_analysis.uncertainty.tolerance import propagate_space
    res = propagate_space(f, space, n=int(params["n"]), method=params["method"], seed=0)
    d = res.to_dict() if hasattr(res, "to_dict") else dict(vars(res))
    ctx.metric("mean", float(np.mean(d["outputs"])), f"{params['method']} sample of {params['n']}")
    ctx.metric("std", float(np.std(d["outputs"])))
    for k, v in (d.get("percentiles") or {}).items():
        ctx.metric(f"p{k}", float(np.asarray(v).ravel()[0]))
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(np.asarray(d["outputs"]).ravel(), bins=40)
    ax.set_xlabel("output (mean of the predicted field)")
    fa = ctx.figure(fig, "distribution", "propagated output distribution")
    ctx.summary(f"{params['method']} ({params['n']}): mean {np.mean(d['outputs']):.4g}, "
                f"std {np.std(d['outputs']):.3g}")
    return {"metrics": ctx.report["metrics"], "figure": fa}


# ------------------------------------------------------------------------------------------------ ROMs
@node("rom.fit", "Reduced-order models (POD, DMD, OpInf)", category="Learning",
      inputs=[Port("solution", "solution", help="a time-dependent solution: its time steps are the snapshots")],
      outputs=[Port("metrics", "metrics"), Port("figure", "figure", required=False)],
      params=[Param("methods", "list", ["persistence", "pod_rbf", "dmd", "opinf"],
                    help="rom_study methods for time series: persistence, pod_rbf, dmd, opinf, neural"),
              Param("ranks", "list", [2, 4, 8]),
              Param("test_fraction", "float", 0.25, minimum=0.05, maximum=0.5)],
      calls="pinneapple_neural.architectures.rom.study.rom_study + pinneapple_data.snapshots.snapshots_from_arrays")
def rom_fit(ctx, inputs, params):
    """Fit reduced-order models to the solution's snapshots (the last ``test_fraction`` of the time steps held out)
    and compare their forecast errors at equal budget."""
    from pinneapple_data.snapshots import snapshots_from_arrays
    from pinneapple_neural.architectures.rom.study import rom_study
    sol = load_solution(inputs["solution"])
    if "t" not in sol["coords"]:
        raise StepFailed("a ROM needs a time-dependent solution (its time steps are the snapshots)")
    ti = sol["coords"].index("t")
    V = np.moveaxis(sol["values"][..., 0], ti, 0)
    X = V.reshape(V.shape[0], -1)
    snaps = snapshots_from_arrays({"u": X}, times=np.asarray(sol["axes"]["t"], float))
    ctx.decide("rom.methods", f"ROMs {', '.join(params['methods'])} at ranks {params['ranks']}",
               value=params["methods"])
    t0 = time.perf_counter()
    res = rom_study(snaps, methods=list(params["methods"]), ranks=[int(r) for r in params["ranks"]],
                    test_fraction=float(params["test_fraction"]), timing_repeats=2)
    rows = sorted(res.rows, key=lambda r: r["error_mean"])
    ctx.table("ROMs on the held-out time steps", ["method", "rank", "mean rel. error", "max rel. error", "fit [s]"],
              [[r["label"], r["rank"], r["error_mean"], r["error_max"], r["fit_s"]] for r in rows])
    best = rows[0]
    ctx.metric("best", f"{best['label']} (rank {best['rank']})")
    ctx.metric("best_error_mean", best["error_mean"], "mean relative error over the held-out time steps")
    ctx.metric("snapshots", int(X.shape[0]))
    ctx.metric("dofs", int(X.shape[1]))
    ctx.metric("study_seconds", round(time.perf_counter() - t0, 2))
    plt = _plt()
    sv = np.asarray(res.singular_values, float)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.semilogy(np.arange(1, len(sv) + 1), sv / sv[0], "o-", ms=3)
    ax.set_xlabel("mode")
    ax.set_ylabel("singular value / first")
    fa = ctx.figure(fig, "singular_values", "POD singular values of the snapshots")
    ctx.summary(f"best {best['label']} rank {best['rank']}: mean error {best['error_mean']:.2e} on held-out steps")
    return {"metrics": ctx.report["metrics"], "figure": fa}


# ------------------------------------------------------------------------------------------------ GNN
@node("gnn.meshgraphnet", "MeshGraphNet (mesh dynamics)", category="Learning",
      outputs=[Port("metrics", "metrics"), Port("architecture", "architecture", required=False)],
      params=[Param("n_train", "int", 12, minimum=2, maximum=200, help="random Delaunay meshes for training"),
              Param("n_test", "int", 3, minimum=1, maximum=50, help="unseen meshes for the rollout test"),
              Param("n_points", "int", 100, minimum=30, maximum=2000),
              Param("steps", "int", 1200, minimum=10, maximum=50000),
              Param("rollout", "int", 10, minimum=1, maximum=100),
              Param("hidden", "int", 32, minimum=8, maximum=256),
              Param("message_passing", "int", 4, minimum=1, maximum=15)],
      calls="pinneapple_neural.architectures.graphnn.mgn_dynamics.MeshDynamicsMGN + train_dynamics + eval_rollout")
def meshgraphnet(ctx, inputs, params):
    """Train a MeshGraphNet to step a diffusion field on random unstructured meshes and roll it out on unseen
    meshes, against the frozen-initial-state baseline (the library's synthetic mesh-dynamics benchmark)."""
    import torch
    from scipy.spatial import Delaunay

    from pinneapple_neural.architectures.graphnn.mgn_dynamics import (
        MeshDynamicsMGN,
        MeshGraph,
        eval_rollout,
        train_dynamics,
    )
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    def sample(n_pts, T=30, alpha=0.2):
        pts = rng.random((n_pts, 2))
        cells = Delaunay(pts).simplices
        pos = torch.from_numpy(pts).float()
        nt = torch.zeros(n_pts, dtype=torch.long)
        bnd = (pos[:, 0] < 0.05) | (pos[:, 0] > 0.95)
        nt[bnd] = 3
        g = MeshGraph.from_mesh(pos, torch.from_numpy(cells), nt)
        src, dst = g.edge_index
        deg = torch.zeros(n_pts).index_add_(0, dst, torch.ones(dst.numel()))
        u = torch.zeros(n_pts, 1)
        u[bnd] = float(rng.uniform(-1, 1))
        u = u + torch.exp(-((pos - torch.from_numpy(rng.random(2)).float()) ** 2).sum(-1, keepdim=True) / 0.02)
        traj = [u]
        for _ in range(T):
            agg = torch.zeros_like(u).index_add_(0, dst, u[src])
            un = u + alpha * (agg / deg.unsqueeze(-1) - u)
            un[bnd] = u[bnd]
            u = un
            traj.append(u)
        v = torch.stack(traj)
        return g, v, torch.zeros_like(v)
    train = [sample(int(params["n_points"])) for _ in range(int(params["n_train"]))]
    test = [sample(int(params["n_points"])) for _ in range(int(params["n_test"]))]
    ctx.decide("gnn.meshgraphnet", f"MeshGraphNet, {params['message_passing']} message-passing steps, width "
                                   f"{params['hidden']}")
    model = MeshDynamicsMGN(vel_dim=1, num_node_types=4, edge_dim=3, out_p=False, hidden_dim=int(params["hidden"]),
                            n_layers=2, n_message_passing=int(params["message_passing"]), activation="relu")
    model.fit_stats([v for _, v, _ in train], None, [g for g, _, _ in train])
    t0 = time.perf_counter()
    losses = train_dynamics(model, train, steps=int(params["steps"]), lr=2e-3, lr_final=1e-5, noise_std=0.0,
                            device="cpu", log_every=max(1, int(params["steps"]) // 5))
    secs = time.perf_counter() - t0
    m = eval_rollout(model, test, n_steps=int(params["rollout"]), device="cpu")
    k = int(params["rollout"])
    base = float(np.mean([((v[0:1] - v[1:k + 1])[:, g.node_type == 0] ** 2).mean().sqrt().item()
                          for g, v, _ in test]))
    ctx.metric("rollout_rmse", float(m["rollout_rmse"]), f"{k}-step rollout on {len(test)} unseen meshes")
    ctx.metric("one_step_rmse", float(m["one_step_rmse"]))
    ctx.metric("frozen_ic_baseline_rmse", base, "the initial state held fixed")
    ctx.metric("train_seconds", round(secs, 2))
    ctx.metric("n_parameters", int(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    if losses:
        ctx.metric("final_train_loss", float(losses[-1]))
    from .. import learn
    g = learn.layer_graph(model)
    ctx.view("architecture", g)
    ctx.summary(f"rollout RMSE {m['rollout_rmse']:.3g} vs {base:.3g} for the frozen initial state "
                f"({len(test)} unseen meshes, {k} steps)")
    return {"metrics": ctx.report["metrics"], "architecture": g}


# ------------------------------------------------------------------------------------------------ meta / transfer
def _pinn_loss(P, n_extra: int = 64):
    """``physics(model, X) -> loss``: mean squared PDE residual at ``X`` plus the boundary / initial misfit at fixed
    points of :class:`PINNProblem` ``P``."""
    import torch
    rng = np.random.default_rng(1)
    extra = [fn(n_extra, rng) for fn in (P.boundary, P.initial) if fn is not None]

    def physics(model, X):
        X = X.detach().requires_grad_(True)
        loss = (P.residual(model(X), X) ** 2).mean()
        for Xb, Yb in extra:
            Xt = torch.tensor(Xb, dtype=torch.float32)
            Yt = torch.tensor(Yb, dtype=torch.float32)
            mask = ~torch.isnan(Yt)
            loss = loss + ((model(Xt)[mask] - Yt[mask]) ** 2).mean()
        return loss
    return physics


@node("transfer.pinn", "Transfer a PINN to a new problem", category="Learning",
      inputs=[Port("model", "model", help="a PINN trained on the source problem (train.pinn)"),
              Port("problem", "problem", help="the target problem (same coordinates and fields)")],
      outputs=[Port("metrics", "metrics"), Port("figure", "figure", required=False)],
      params=[Param("strategy", "enum", "finetune", choices=["finetune", "feature_extract", "progressive"]),
              Param("epochs", "int", 300, minimum=1, maximum=100000),
              Param("lr", "float", 1e-3, minimum=1e-7, maximum=1.0),
              Param("n_collocation", "int", 1024, minimum=16, maximum=100000)],
      calls="pinneapple_adaptation.fine_tune (pinneapple_adaptation.transfer_learning.TransferTrainer) + "
            "pinneapple_physics.symbolic_pde.pinn.pinn_problem")
def transfer(ctx, inputs, params):
    """Fine-tune a trained PINN on the target problem's physics and compare with training the same network from a
    random initialisation for the same epochs (loss, and the error against the target's exact solution if any)."""
    import torch

    from pinneapple_adaptation import fine_tune
    from pinneapple_physics.symbolic_pde.pinn import pinn_problem
    src = load_artifact(inputs["model"])
    if not isinstance(src, dict) or src.get("kind") != "pinn":
        raise StepFailed("connect the model of a PINN node (train.pinn)")
    problem = need_problem(inputs["problem"])
    sys_ = system_of(problem)
    if list(sys_.coords) != list(src["coords"]) or list(sys_.fields) != list(src["fields"]):
        raise StepFailed(f"source PINN maps {src['coords']} -> {src['fields']}, target is {sys_.coords} -> "
                         f"{sys_.fields}")
    bounds = bounds_of(problem)
    exact = exact_function(problem)
    P, warm = pinn_problem(sys_, bounds, width=int(src["width"]), depth=int(src["depth"]), exact=exact)
    warm.load_state_dict(src["state"], strict=False)
    physics = _pinn_loss(P)
    lows = np.array([bounds[c][0] for c in sys_.coords])
    span = np.array([bounds[c][1] - bounds[c][0] for c in sys_.coords])
    rng = np.random.default_rng(0)
    n_col = int(params["n_collocation"])

    def fn(model, batch):
        X = torch.tensor(lows + rng.random((n_col, len(lows))) * span, dtype=torch.float32)
        return physics(model, X)
    ctx.decide("transfer.finetune", f"fine-tune ({params['strategy']}) from the source PINN vs from scratch",
               value=params["strategy"])
    t0 = time.perf_counter()
    rw = fine_tune(warm, fn, strategy=params["strategy"], epochs=int(params["epochs"]), lr=float(params["lr"]))
    t_w = time.perf_counter() - t0
    torch.manual_seed(0)
    _, cold = pinn_problem(sys_, bounds, width=int(src["width"]), depth=int(src["depth"]), exact=exact)
    rc = fine_tune(cold, fn, strategy="finetune", epochs=int(params["epochs"]), lr=float(params["lr"]))
    hw = [h["loss_total"] for h in rw["history"]]
    hc = [h["loss_total"] for h in rc["history"]]
    ctx.metric("loss_transferred", hw[-1], f"physics loss after {params['epochs']} epochs from the source PINN")
    ctx.metric("loss_from_scratch", hc[-1], "same epochs from a random initialisation")
    ctx.metric("transfer_seconds", round(t_w, 2))
    if exact is not None:
        g = [np.linspace(bounds[c][0], bounds[c][1], 41) for c in sys_.coords]
        X = np.stack([a.ravel() for a in np.meshgrid(*g, indexing="ij")], 1)
        ex = exact(X)
        for name, m in (("transferred", rw["model"]), ("from_scratch", rc["model"])):
            with torch.no_grad():
                u = m(torch.tensor(X, dtype=torch.float32)).numpy()
            ctx.metric(f"rel_l2_{name}", float(np.linalg.norm(u - ex) / max(np.linalg.norm(ex), 1e-300)),
                       "against the target's exact solution on a 41-point-per-axis grid")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.semilogy(hw, label="from the source PINN")
    ax.semilogy(hc, label="from scratch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("physics loss")
    ax.legend(fontsize=8)
    fa = ctx.figure(fig, "transfer", "fine-tuning on the target problem")
    ctx.summary(f"after {params['epochs']} epochs: loss {hw[-1]:.3g} transferred vs {hc[-1]:.3g} from scratch")
    return {"metrics": ctx.report["metrics"], "figure": fa}


# ------------------------------------------------------------------------------------------------ meta-learning
@node("meta.learn", "Meta-learning over a coefficient (Reptile / MAML)", category="Learning",
      inputs=[Port("problem", "problem")],
      outputs=[Port("metrics", "metrics"), Port("figure", "figure", required=False)],
      params=[Param("coefficient", "str", "", help="the coefficient that varies across tasks (default: the first)"),
              Param("range", "list", [], help="[low, high] (default: 0.5x to 2x its value)"),
              Param("algorithm", "enum", "reptile", choices=["reptile", "maml"]),
              Param("meta_epochs", "int", 40, minimum=1, maximum=5000),
              Param("inner_steps", "int", 5, minimum=1, maximum=100),
              Param("adapt_steps", "int", 20, minimum=1, maximum=2000),
              Param("width", "int", 32, minimum=4, maximum=256)],
      calls="pinneapple_adaptation.meta_learning.meta_train + PDETaskSampler (tasks: the problem with one coefficient "
            "varied)")
def meta_learn(ctx, inputs, params):
    """Meta-learn a PINN initialisation over a family of problems (one coefficient varied), then adapt it to an
    unseen coefficient and compare the physics loss with a random initialisation after the same number of steps."""
    import torch

    from pinneapple_adaptation import meta_train
    from pinneapple_adaptation.meta_learning import PDETaskSampler
    from pinneapple_physics.symbolic_pde.pinn import pinn_problem
    problem = need_problem(inputs["problem"])
    sys_ = system_of(problem)
    bounds = bounds_of(problem)
    coords = list(sys_.coords)
    coefs = dict(sys_.coefficients)
    name = params["coefficient"] or next(iter(coefs), "")
    if name not in coefs:
        raise StepFailed(f"no coefficient {name!r} to vary (the equations have {list(coefs)})")
    lo, hi = (float(params["range"][0]), float(params["range"][1])) if len(params["range"]) == 2 else \
        (0.5 * coefs[name], 2.0 * coefs[name])
    lows = np.array([bounds[c][0] for c in coords])
    span = np.array([bounds[c][1] - bounds[c][0] for c in coords])
    ctx.decide("meta.reptile", f"{params['algorithm']} over {name} in [{lo:g}, {hi:g}]", value=params["algorithm"])

    def loss_for(value):
        P, _ = pinn_problem(sys_, bounds, coefficients={**coefs, name: float(value)})
        return _pinn_loss(P)

    def factory(p):
        phys = loss_for(p[name])
        if params["algorithm"] == "maml":
            return lambda fwd, x: phys(fwd, x)
        return lambda model, batch: (phys(model, batch["x_col"]), {})

    pts_rng = np.random.default_rng(2)

    def data(p):
        X = lows + pts_rng.random((128, len(coords))) * span
        return {"x_col": torch.tensor(X, dtype=torch.float32)}
    sampler = PDETaskSampler(param_ranges={name: (lo, hi)}, physics_fn_factory=factory, data_factory=data,
                             input_dim=len(coords), seed=0)

    def net():
        w = int(params["width"])
        return torch.nn.Sequential(torch.nn.Linear(len(coords), w), torch.nn.Tanh(), torch.nn.Linear(w, w),
                                   torch.nn.Tanh(), torch.nn.Linear(w, len(sys_.fields)))
    torch.manual_seed(0)
    t0 = time.perf_counter()
    trainer = meta_train(net(), sampler, algorithm=params["algorithm"], n_meta_epochs=int(params["meta_epochs"]),
                         n_tasks_per_batch=2, n_inner_steps=int(params["inner_steps"]), seed=0,
                         checkpoint_every=0)          # no checkpoint files in the working directory
    secs = time.perf_counter() - t0
    held = 0.5 * (lo + hi) * 1.07                      # a coefficient not drawn exactly during meta-training
    phys = loss_for(held)
    Xq = torch.tensor(lows + np.random.default_rng(3).random((512, len(coords))) * span, dtype=torch.float32)

    def adapt(model, steps):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        curve = []
        for _ in range(steps):
            opt.zero_grad()
            ls = phys(model, Xq)
            ls.backward()
            opt.step()
            curve.append(float(ls.detach()))
        return curve
    import copy
    meta_model = copy.deepcopy(trainer.model)
    torch.manual_seed(1)
    c_meta = adapt(meta_model, int(params["adapt_steps"]))
    torch.manual_seed(1)
    c_rand = adapt(net(), int(params["adapt_steps"]))
    ctx.metric("held_out_coefficient", held, f"{name} for the adaptation test")
    ctx.metric("loss_after_adapt_meta", c_meta[-1], f"physics loss after {params['adapt_steps']} Adam steps")
    ctx.metric("loss_after_adapt_random", c_rand[-1], "same steps from a random initialisation")
    ctx.metric("meta_train_seconds", round(secs, 2))
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.semilogy(c_meta, label=f"{params['algorithm']} initialisation")
    ax.semilogy(c_rand, label="random initialisation")
    ax.set_xlabel("adaptation step")
    ax.set_ylabel("physics loss")
    ax.legend(fontsize=8)
    fa = ctx.figure(fig, "adaptation", f"adapting to {name} = {held:.4g}")
    ctx.summary(f"after {params['adapt_steps']} steps at {name} = {held:.4g}: loss {c_meta[-1]:.3g} from the "
                f"{params['algorithm']} initialisation vs {c_rand[-1]:.3g} from a random one")
    return {"metrics": ctx.report["metrics"], "figure": fa}


# ------------------------------------------------------------------------------------------------ benchmark
_BENCH_PROBLEMS = ["heat_1d", "burgers_1d", "poisson_2d", "wave_1d", "allen_cahn_1d", "ns_tgv_2d"]


@node("benchmark.arena", "PINN benchmark arena", category="Validation",
      outputs=[Port("metrics", "metrics")],
      params=[Param("problems", "list", ["heat_1d", "burgers_1d"], help=f"from {_BENCH_PROBLEMS}"),
              Param("models", "list", ["bench_mlp", "bench_siren"],
                    help="names from PINNArenaBenchmark.list_models()"),
              Param("epochs", "int", 200, minimum=1, maximum=50000),
              Param("n_col", "int", 512, minimum=16, maximum=100000)],
      calls="pinneapple_tools.benchmark_suite.benchmark.PINNArenaBenchmark")
def arena(ctx, inputs, params):
    """The same problems, collocation budget and epochs for every model; relative L2 against each problem's
    reference and the rank per problem."""
    from pinneapple_tools.benchmark_suite.benchmark import PINNArenaBenchmark
    bad = [p for p in params["problems"] if p not in _BENCH_PROBLEMS]
    if bad:
        raise StepFailed(f"unknown problems {bad}; known: {_BENCH_PROBLEMS}")
    known = set(PINNArenaBenchmark.list_models())
    badm = [m for m in params["models"] if m not in known]
    if badm:
        raise StepFailed(f"unknown models {badm}")
    ctx.decide("benchmark.arena", f"{len(params['problems'])} problems x {len(params['models'])} models, "
                                  f"{params['epochs']} epochs each")
    b = PINNArenaBenchmark.default(problems=list(params["problems"]), models=list(params["models"]),
                                   epochs=int(params["epochs"]), device="cpu")
    b.config.n_col = int(params["n_col"])
    res = b.run(verbose=False)
    rows = [[r.problem_id, r.model_id, r.metrics.get("rel_l2"), r.rank, r.n_params, round(r.elapsed_s, 2)]
            for r in res]
    ctx.table("arena", ["problem", "model", "rel. L2", "rank", "parameters", "seconds"], rows)
    for p in params["problems"]:
        rs = [r for r in res if r.problem_id == p and r.metrics.get("rel_l2") is not None]
        if rs:
            best = min(rs, key=lambda r: r.metrics["rel_l2"])
            ctx.metric(f"{p}_best", f"{best.model_id} ({best.metrics['rel_l2']:.3g})")
    ctx.summary(f"{len(res)} runs; " + ", ".join(f"{k}: {v}" for k, v in ctx.report["metrics"].items()))
    return {"metrics": {"runs": [dict(zip(["problem", "model", "rel_l2", "rank", "n_params", "seconds"], r,
                                          strict=True)) for r in rows]}}


# ------------------------------------------------------------------------------------------------ optimisation
@node("optimize.nsga2", "Multi-objective design (NSGA-II)", category="Design",
      outputs=[Port("metrics", "metrics"), Port("figure", "figure", required=False)],
      params=[Param("problem", "enum", "heatsink", choices=["heatsink", "airfoil"],
                    help="a design problem preset of pinneapple_design.design_optimizer"),
              Param("method", "enum", "nsga2", choices=["nsga2", "random", "scipy-de"]),
              Param("budget", "int", 400, minimum=10, maximum=100000),
              Param("seed", "int", 0, minimum=0)],
      calls="pinneapple_design.design_optimizer.methods.run + presets")
def nsga2(ctx, inputs, params):
    """Search a design problem with NSGA-II (or a baseline) and report the Pareto front."""
    from pinneapple_design.design_optimizer import presets
    from pinneapple_design.design_optimizer.methods import run
    prob = getattr(presets, f"{params['problem']}_problem")()
    ctx.decide("optimization.nsga2", f"{params['method']} on the {params['problem']} problem, budget "
                                     f"{params['budget']}", value=params["method"])
    res = run(prob, params["method"], budget=int(params["budget"]), seed=int(params["seed"]))
    summ = res.summary()
    for k in ("n_evals", "n_feasible", "n_pareto", "wall_s", "status"):
        if k in summ:
            ctx.metric(k, summ[k])
    F = np.asarray(res.evaluations.F, float)
    idx = list(res.pareto())
    names = [getattr(o, "name", str(o)) for o in getattr(prob, "objectives", [])] or [f"f{i}" for i in
                                                                                    range(F.shape[1])]
    ctx.table("Pareto front", names, [list(map(float, F[i])) for i in idx[:50]])
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    if F.shape[1] >= 2:
        ax.plot(F[:, 0], F[:, 1], ".", ms=3, alpha=0.4, label="evaluated")
        ax.plot(F[idx, 0], F[idx, 1], "o", label="Pareto")
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
        ax.legend(fontsize=8)
    fa = ctx.figure(fig, "pareto", f"{params['method']} on {params['problem']}")
    ctx.summary(f"{params['method']}: {summ.get('n_evals')} evaluations, {summ.get('n_feasible')} feasible, "
                f"{len(idx)} on the Pareto front")
    return {"metrics": ctx.report["metrics"], "figure": fa}


@node("control.compare", "Controllers: PID, rules, MPC", category="Systems",
      outputs=[Port("metrics", "metrics")],
      params=[Param("controllers", "list", ["pid", "rules", "lmpc"], help="pid, rules, lmpc, nmpc (needs casadi)")],
      calls="pinneapple_systems.operations.control.compare_controllers")
def control(ctx, inputs, params):
    """Closed-loop comparison of controllers on the library's heated-tank plant (simulation only: no command path
    to equipment)."""
    from pinneapple_systems.operations.control import compare_controllers
    ctx.decide("control.mpc", f"compare {', '.join(params['controllers'])} on the same plant and scenario")
    r = compare_controllers(controllers=tuple(params["controllers"]))
    keys = ["iae_h_m_s", "rmse_h_m", "rmse_T_K", "max_violation_h_m", "heater_energy_kWh", "mean_solve_ms"]
    rows = []
    for name, run_ in r["runs"].items():
        m = run_.get("metrics", run_) if isinstance(run_, dict) else {}
        rows.append([name] + [m.get(k) for k in keys])
    ctx.table("closed loop", ["controller"] + keys, rows)
    best = min((row for row in rows if isinstance(row[1], (int, float))), key=lambda row: row[1], default=None)
    if best:
        ctx.metric("best_by_iae_h", best[0], "lowest integral absolute level error")
    ctx.metric("mode", r.get("mode", ""))
    ctx.summary(f"{len(rows)} controllers; lowest level IAE: {best[0] if best else '-'}")
    return {"metrics": ctx.report["metrics"]}

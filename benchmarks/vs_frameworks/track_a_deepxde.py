# Track A (PINN forward problems), DeepXDE runner. DeepXDE 1.15.0, PyTorch backend (DDE_BACKEND=pytorch).
# Follows the official examples of DeepXDE v1.15.0 (github.com/lululxvi/deepxde/tree/v1.15.0/examples):
#   burgers    -> examples/pinn_forward/Burgers.py: FNN [2] + [20] * 3 + [1], tanh, Glorot normal; TimePDE with
#                 num_domain=2540, num_boundary=80, num_initial=160 on t in [0, 0.99]; Adam lr 1e-3 for 15000
#                 iterations, then L-BFGS (DeepXDE defaults).
#   poisson    -> examples/pinn_forward/Poisson_Lshape.py (the 2-D Poisson example; here on the unit square with the
#                 manufactured source): FNN [2] + [50] * 4 + [1], tanh, Glorot uniform, num_domain=1200,
#                 num_boundary=120, Adam lr 1e-3 for 50000 iterations, then L-BFGS.
#   helmholtz  -> examples/pinn_forward/Helmholtz_Dirichlet_2d.py: FNN [2] + [150] * 3 + [1], sin, Glorot uniform,
#                 hard Dirichlet constraint by output transform, 10 points per wavelength (here 40 x 40 interior),
#                 Adam lr 1e-3 for 5000 iterations.
#   allen_cahn -> examples/pinn_forward/Allen_Cahn.py: FNN [2] + [20] * 3 + [1], tanh, Glorot normal, hard IC/BC
#                 output transform x^2 cos(pi x) + t (1 - x^2) N, num_domain=8000, num_boundary=400,
#                 num_initial=800, Adam lr 1e-3 for 40000 iterations, then L-BFGS (the example uses d = 0.001 from
#                 arXiv:2111.02801; the protocol's d = 1e-4 is used here).
# A2 runs stop at the time cap through a dde.callbacks.Callback (model.stop_training); L-BFGS is checked every
# 100 inner iterations (LBFGS_options["iter_per_step"] = 100 instead of 1000, so the cap is not overshot).
# A1 uses the shared point sets (anchors + PointSetBC), FNN [2] + [50] * 4 + [1] tanh with the shared Xavier weights,
# Adam lr 1e-3, loss weights 1, a fixed number of iterations.
"""DeepXDE runner of track A (see the comment above and PROTOCOL.md)."""
from __future__ import annotations

import os
import time

import numpy as np

from pinneapple_tools.benchmark_suite.crossframework import problems as P
from pinneapple_tools.benchmark_suite.crossframework import protocol as proto
from pinneapple_tools.benchmark_suite.crossframework import records as R
from pinneapple_tools.benchmark_suite.crossframework import weights as W

SOURCE = "DeepXDE 1.15.0 examples/pinn_forward/{Burgers,Poisson_Lshape,Helmholtz_Dirichlet_2d,Allen_Cahn}.py"

RECIPES = {
    "burgers": {"example": "examples/pinn_forward/Burgers.py", "layers": [20] * 3, "act": "tanh",
                "init": "Glorot normal", "num_domain": 2540, "num_boundary": 80, "num_initial": 160,
                "adam_iters": 15000, "lbfgs": True},
    "poisson": {"example": "examples/pinn_forward/Poisson_Lshape.py", "layers": [50] * 4, "act": "tanh",
                "init": "Glorot uniform", "num_domain": 1200, "num_boundary": 120, "adam_iters": 50000,
                "lbfgs": True},
    "helmholtz": {"example": "examples/pinn_forward/Helmholtz_Dirichlet_2d.py", "layers": [150] * 3, "act": "sin",
                  "init": "Glorot uniform", "num_domain": 40 ** 2, "num_boundary": 4 * 40, "adam_iters": 5000,
                  "lbfgs": False, "hard_constraint": True},
    "allen_cahn": {"example": "examples/pinn_forward/Allen_Cahn.py", "layers": [20] * 3, "act": "tanh",
                   "init": "Glorot normal", "num_domain": 8000, "num_boundary": 400, "num_initial": 800,
                   "adam_iters": 40000, "lbfgs": True, "hard_constraint": True},
}


def pde_fn(prob: P.PINNProblem):
    import deepxde as dde
    import torch

    if prob.name == "burgers":
        nu = prob.params["nu"]

        def pde(x, y):
            return (dde.grad.jacobian(y, x, i=0, j=1) + y * dde.grad.jacobian(y, x, i=0, j=0)
                    - nu * dde.grad.hessian(y, x, i=0, j=0))
    elif prob.name == "allen_cahn":
        d = prob.params["d"]

        def pde(x, y):
            return dde.grad.jacobian(y, x, i=0, j=1) - d * dde.grad.hessian(y, x, i=0, j=0) - 5 * (y - y ** 3)
    elif prob.name == "poisson":
        def pde(x, y):
            f = -2 * np.pi ** 2 * torch.sin(np.pi * x[:, 0:1]) * torch.sin(np.pi * x[:, 1:2])
            return dde.grad.hessian(y, x, i=0, j=0) + dde.grad.hessian(y, x, i=1, j=1) - f
    else:
        a1, a2, k = prob.params["a1"], prob.params["a2"], prob.params["k"]

        def pde(x, y):
            q = (k ** 2 - (a1 * np.pi) ** 2 - (a2 * np.pi) ** 2) * torch.sin(a1 * np.pi * x[:, 0:1]) \
                * torch.sin(a2 * np.pi * x[:, 1:2])
            return dde.grad.hessian(y, x, i=0, j=0) + dde.grad.hessian(y, x, i=1, j=1) + k ** 2 * y - q
    return pde


def geometry(prob: P.PINNProblem, t_end=None):
    import deepxde as dde
    (a0, a1), (b0, b1) = prob.bounds
    if prob.time_dependent:
        return dde.geometry.GeometryXTime(dde.geometry.Interval(a0, a1), dde.geometry.TimeDomain(b0, t_end or b1))
    return dde.geometry.Rectangle([a0, b0], [a1, b1])


class _Cap:
    """Stop training when the time cap is used up (A2)."""

    def __new__(cls, cap):
        import deepxde as dde

        class CapCallback(dde.callbacks.Callback):
            def on_epoch_end(self):
                if cap.expired():
                    self.model.stop_training = True
        return CapCallback()


def run_a1(args, prob, meta):
    import deepxde as dde
    import torch

    pts = P.load_point_set(proto.ensure_point_set(prob, args.root))
    meta["config"]["point_set_sha256"] = P.point_set_digest(pts)
    bcs = [dde.icbc.PointSetBC(pts["X_b"], pts["u_b"], component=0)]
    if prob.time_dependent:
        bcs.append(dde.icbc.PointSetBC(pts["X_0"], pts["u_0"], component=0))
        data = dde.data.TimePDE(geometry(prob), pde_fn(prob), bcs, num_domain=0, num_boundary=0, num_initial=0,
                                anchors=pts["X_f"])
    else:
        data = dde.data.PDE(geometry(prob), pde_fn(prob), bcs, num_domain=0, num_boundary=0, anchors=pts["X_f"])
    torch.manual_seed(args.seed)
    net = dde.nn.FNN(W.layer_sizes(), "tanh", "Glorot uniform")
    W.load_into_linears(net.linears, W.xavier_uniform_init(W.layer_sizes(), args.seed))
    model = dde.Model(data, net)
    model.compile("adam", lr=proto.A1["lr"], loss_weights=[1.0] * (1 + len(bcs)))
    with R.Clock() as clk:
        losshistory, _ = model.train(iterations=args.iters, display_every=1000, verbose=0)
    loss = [float(np.sum(v)) for v in losshistory.loss_train]
    meta.update(iterations=int(model.train_state.iteration), loss_history=loss, final_loss=loss[-1],
                n_parameters=W.count_parameters(net), n_train_points=int(len(data.train_x)))
    return clk, model


def run_a2(args, prob, meta):
    import deepxde as dde
    import torch

    rc = RECIPES[prob.name]
    meta["config"].update(rc)
    (a0, a1), (b0, b1) = prob.bounds
    if prob.name == "burgers":
        geom = geometry(prob, t_end=0.99)
        bc = dde.icbc.DirichletBC(geom, lambda x: 0, lambda _, on_boundary: on_boundary)
        ic = dde.icbc.IC(geom, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial)
        data = dde.data.TimePDE(geom, pde_fn(prob), [bc, ic], num_domain=rc["num_domain"],
                                num_boundary=rc["num_boundary"], num_initial=rc["num_initial"])
    elif prob.name == "allen_cahn":
        geom = geometry(prob)
        data = dde.data.TimePDE(geom, pde_fn(prob), [], num_domain=rc["num_domain"], num_boundary=rc["num_boundary"],
                                num_initial=rc["num_initial"])
    elif prob.name == "poisson":
        geom = geometry(prob)
        bc = dde.icbc.DirichletBC(geom, lambda x: 0, lambda _, on_boundary: on_boundary)
        data = dde.data.PDE(geom, pde_fn(prob), bc, num_domain=rc["num_domain"], num_boundary=rc["num_boundary"])
    else:
        geom = geometry(prob)
        data = dde.data.PDE(geom, pde_fn(prob), [], num_domain=rc["num_domain"], num_boundary=rc["num_boundary"])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dde.config.set_random_seed(args.seed)
    net = dde.nn.FNN([2] + rc["layers"] + [1], rc["act"], rc["init"])
    if prob.name == "helmholtz":
        net.apply_output_transform(lambda x, y: (x[:, 0:1] - a0) * (a1 - x[:, 0:1]) * (x[:, 1:2] - b0)
                                   * (b1 - x[:, 1:2]) * y)
    elif prob.name == "allen_cahn":
        net.apply_output_transform(lambda x, y: x[:, 0:1] ** 2 * torch.cos(np.pi * x[:, 0:1])
                                   + x[:, 1:2] * (1 - x[:, 0:1] ** 2) * y)
    model = dde.Model(data, net)
    cap = R.TimeCap(args.cap_s, args.cap_kind)
    cb = _Cap(cap)
    iters, phases = 0, []
    with R.Clock() as clk:
        model.compile("adam", lr=1e-3)
        lh, _ = model.train(iterations=rc["adam_iters"], callbacks=[cb], display_every=1000, verbose=0)
        phases.append({"optimizer": "adam", "iterations": int(model.train_state.iteration)})
        if rc["lbfgs"] and not cap.expired():
            dde.optimizers.set_LBFGS_options()
            dde.optimizers.config.LBFGS_options["iter_per_step"] = 100
            model.compile("L-BFGS")
            before = int(model.train_state.iteration)
            lh, _ = model.train(callbacks=[cb], display_every=1000, verbose=0)
            phases.append({"optimizer": "L-BFGS", "iterations": int(model.train_state.iteration) - before})
    iters = int(model.train_state.iteration)
    loss = [float(np.sum(v)) for v in lh.loss_train]
    meta.update(iterations=iters, phases=phases, loss_history=loss, final_loss=loss[-1] if loss else None,
                n_parameters=W.count_parameters(net), cap_used_s=cap.used())
    return clk, model


def main(argv=None):
    os.environ.setdefault("DDE_BACKEND", "pytorch")
    args = proto.runner_parser("DeepXDE, track A").parse_args(argv)
    R.setup_threads(args.threads)
    prob = P.get_problem(args.problem)
    cfg = dict(proto.A1) if args.mode == "a1" else {"recipe": "official example", **proto.A2, "cap_s": args.cap_s}
    meta = proto.base_meta(args, track="A", framework="deepxde", source=SOURCE, config=cfg)
    t0 = time.perf_counter()
    try:
        import deepxde as dde
        if dde.backend.backend_name != "pytorch":
            raise RuntimeError(f"DeepXDE backend is {dde.backend.backend_name}, expected pytorch")
        clk, model = (run_a1 if args.mode == "a1" else run_a2)(args, prob, meta)
        arrays = proto.export_predictions(prob, lambda X: model.predict(X))
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

# Track C (inverse Burgers: nu from 2000 noisy samples), DeepXDE 1.15.0 runner (PyTorch backend).
# Follows DeepXDE v1.15.0 examples/pinn_inverse/diffusion_1d_inverse.py: the unknown coefficient is a dde.Variable
# (here started at 0.05) used inside the PDE, the observations are a dde.icbc.PointSetBC passed as anchors, and
# model.compile("adam", lr=1e-3, external_trainable_variables=nu); dde.callbacks.VariableValue records nu.
# As in Raissi et al. (2019) and in the PINNeAPPle runner, no initial / boundary condition is imposed: data + PDE.
# Network: the A1 network (tanh 2 -> 50 x 4 -> 1) with the shared Xavier weights; collocation: the 10000 shared
# interior points of the A1 Burgers point set (fixed).
"""DeepXDE runner of track C."""
from __future__ import annotations

import os
import time

import numpy as np

from pinneapple_tools.benchmark_suite.crossframework import inverse as INV
from pinneapple_tools.benchmark_suite.crossframework import problems as P
from pinneapple_tools.benchmark_suite.crossframework import protocol as proto
from pinneapple_tools.benchmark_suite.crossframework import records as R
from pinneapple_tools.benchmark_suite.crossframework import weights as W

SOURCE = "DeepXDE 1.15.0 examples/pinn_inverse/diffusion_1d_inverse.py"


def main(argv=None):
    os.environ.setdefault("DDE_BACKEND", "pytorch")
    args = proto.runner_parser("DeepXDE, track C").parse_args(argv)
    R.setup_threads(args.threads)
    cfg = {**proto.C, "iters": args.iters, "nu_param": "dde.Variable (linear)",
           "collocation": "the 10000 shared A1 Burgers interior points (fixed)"}
    meta = proto.base_meta(args, track="C", framework="deepxde", source=SOURCE, config=cfg)
    t0 = time.perf_counter()
    arrays = {}
    try:
        import deepxde as dde
        import torch

        prob = P.get_problem("burgers")
        pts = P.load_point_set(proto.ensure_point_set(prob, args.root))
        obs = INV.observations()
        nu = dde.Variable(proto.C["nu_init"])

        def pde(x, y):
            return (dde.grad.jacobian(y, x, i=0, j=1) + y * dde.grad.jacobian(y, x, i=0, j=0)
                    - nu * dde.grad.hessian(y, x, i=0, j=0))

        geom = dde.geometry.GeometryXTime(dde.geometry.Interval(-1, 1), dde.geometry.TimeDomain(0, 1))
        observe = dde.icbc.PointSetBC(obs["X"], obs["u"], component=0)
        data = dde.data.TimePDE(geom, pde, [observe], num_domain=0, num_boundary=0, num_initial=0,
                                anchors=np.concatenate([pts["X_f"], obs["X"]]))
        torch.manual_seed(args.seed)
        net = dde.nn.FNN(W.layer_sizes(), "tanh", "Glorot uniform")
        W.load_into_linears(net.linears, W.xavier_uniform_init(W.layer_sizes(), args.seed))
        model = dde.Model(data, net)
        model.compile("adam", lr=proto.C["lr"], external_trainable_variables=nu)
        with R.Clock() as clk:
            lh, _ = model.train(iterations=args.iters, display_every=1000, verbose=0)
        nu_hat = float(nu.detach().cpu().numpy())
        arrays = {"nu_hat": np.float64(nu_hat),
                  "u_test": proto.export_predictions(prob, lambda X: model.predict(X))["u_test"]}
        meta.update(status="ok", wall_s=clk.wall_s, cpu_s=clk.cpu_s, loadavg_before=clk.loadavg_before,
                    loadavg_after=clk.loadavg_after, iterations=int(model.train_state.iteration), nu_hat=nu_hat,
                    loss_history=[float(np.sum(v)) for v in lh.loss_train])
    except Exception as exc:  # noqa: BLE001 - a failure is a result
        import traceback
        meta.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-4000:])
    meta["process_wall_s"] = time.perf_counter() - t0
    R.write_run(args.out, args.run_id, meta, arrays)


if __name__ == "__main__":
    main()

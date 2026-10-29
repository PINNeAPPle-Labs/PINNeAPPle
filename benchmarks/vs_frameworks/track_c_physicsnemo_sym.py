# Track C (inverse Burgers: nu from 2000 noisy samples), PhysicsNeMo-Sym 2.4.0 runner.
# Follows the documented inverse approach of NVIDIA/physicsnemo-sym v2.4.0, docs/user_guide/foundational/
# inverse_problem.rst and examples/three_fin_2d/heat_sink_inverse.py: the unknown coefficient is the output of its own
# network ("invert network", inputs (x, t), output nu); the PDE node is built with detach_names for the field and its
# derivatives, so the residual trains only the invert network and the data train the field network; the data
# constraint is PointwiseConstraint.from_numpy with outvar {u: samples, residual: 0}; the identified value is the
# mean of nu over the data points (the example's PointwiseMonitor "mean_nu").
# Protocol choices on top of the example: the field network is the A1 network (tanh 50 x 4, shared Xavier weights,
# no weight norm); the invert network is the same size (Sym's default init) with its last layer set to output 0.05
# everywhere at the start (the protocol's initial nu); an extra residual constraint on the 10000 shared A1 Burgers
# interior points; per-point weight 1/N (each term a mean); Adam lr 1e-3 constant, no gradient clipping; no IC / BC.
"""PhysicsNeMo-Sym runner of track C."""
from __future__ import annotations

import tempfile
import time

import numpy as np

from benchmarks.vs_frameworks.track_a_physicsnemo_sym import _StepCounter, compose_cfg
from pinneapple_tools.benchmark_suite.crossframework import inverse as INV
from pinneapple_tools.benchmark_suite.crossframework import problems as P
from pinneapple_tools.benchmark_suite.crossframework import protocol as proto
from pinneapple_tools.benchmark_suite.crossframework import records as R
from pinneapple_tools.benchmark_suite.crossframework import weights as W

SOURCE = ("PhysicsNeMo-Sym 2.4.0 docs/user_guide/foundational/inverse_problem.rst, "
          "examples/three_fin_2d/heat_sink_inverse.py")


def main(argv=None):
    args = proto.runner_parser("PhysicsNeMo-Sym, track C").parse_args(argv)
    R.setup_threads(args.threads)
    cfg = {**proto.C, "iters": args.iters, "nu_param": "invert network output, mean over the data points",
           "collocation": "the 2000 data points (as in the example) + the 10000 shared A1 Burgers interior points"}
    meta = proto.base_meta(args, track="C", framework="physicsnemo_sym", source=SOURCE, config=cfg)
    t0 = time.perf_counter()
    arrays = {}
    try:
        import torch
        from physicsnemo.sym.domain import Domain
        from physicsnemo.sym.domain.constraint import PointwiseConstraint
        from physicsnemo.sym.eq.pde import PDE
        from physicsnemo.sym.key import Key
        from physicsnemo.sym.models.activation import Activation
        from physicsnemo.sym.models.fully_connected import FullyConnectedArch
        from physicsnemo.sym.solver import Solver
        from sympy import Function, Symbol

        x, t = Symbol("x"), Symbol("t")

        class BurgersInverse(PDE):
            name = "burgers_inverse"

            def __init__(self):
                u = Function("u")(x, t)
                nu = Function("nu")(x, t)
                self.equations = {"res": u.diff(t) + u * u.diff(x) - nu * u.diff(x, 2)}

        prob = P.get_problem("burgers")
        pts = P.load_point_set(proto.ensure_point_set(prob, args.root))
        obs = INV.observations()
        torch.manual_seed(args.seed)
        u_net = FullyConnectedArch(input_keys=[Key("x"), Key("t")], output_keys=[Key("u")], layer_size=50,
                                   nr_layers=4, activation_fn=Activation.TANH, weight_norm=False)
        W.load_into_linears([m for m in u_net.modules() if isinstance(m, torch.nn.Linear)],
                            W.xavier_uniform_init(W.layer_sizes(), args.seed))
        nu_net = FullyConnectedArch(input_keys=[Key("x"), Key("t")], output_keys=[Key("nu")], layer_size=50,
                                    nr_layers=4, activation_fn=Activation.TANH, weight_norm=False)
        last = [m for m in nu_net.modules() if isinstance(m, torch.nn.Linear)][-1]
        with torch.no_grad():
            last.weight.zero_()
            last.bias.fill_(proto.C["nu_init"])
        nodes = (BurgersInverse().make_nodes(detach_names=["u", "u__x", "u__t", "u__x__x"])
                 + [u_net.make_node(name="u_network"), nu_net.make_node(name="invert_nu_network")])
        domain = Domain()
        n_obs, n_f = len(obs["X"]), len(pts["X_f"])
        domain.add_constraint(PointwiseConstraint.from_numpy(
            nodes=nodes, invar={"x": obs["X"][:, 0:1], "t": obs["X"][:, 1:2]},
            outvar={"u": obs["u"], "res": np.zeros((n_obs, 1), np.float32)}, batch_size=n_obs,
            lambda_weighting={k: np.full((n_obs, 1), 1.0 / n_obs, np.float32) for k in ("u", "res")}), "data")
        domain.add_constraint(PointwiseConstraint.from_numpy(
            nodes=nodes, invar={"x": pts["X_f"][:, 0:1], "t": pts["X_f"][:, 1:2]},
            outvar={"res": np.zeros((n_f, 1), np.float32)}, batch_size=n_f,
            lambda_weighting={"res": np.full((n_f, 1), 1.0 / n_f, np.float32)}), "interior")
        with tempfile.TemporaryDirectory(prefix="sym_inv_") as wd:
            scfg = compose_cfg([f"training.max_steps={args.iters - 1}", "training.grad_clip_max_norm=1e30",
                                f"optimizer.lr={proto.C['lr']}", f"network_dir={wd}", f"device={args.device}"],
                               name="config_a1", device=args.device)
            slv = Solver(scfg, domain)
            steps = _StepCounter()
            with R.Clock() as clk:
                slv.solve(sigterm_handler=steps)
        u_net.eval()
        nu_net.eval()
        with torch.no_grad():
            Xo = torch.from_numpy(obs["X"])
            nu_field = nu_net({"x": Xo[:, 0:1], "t": Xo[:, 1:2]})["nu"].numpy().ravel()
        nu_hat = float(nu_field.mean())

        def predict(X):
            with torch.no_grad():
                Xt = torch.as_tensor(X, dtype=torch.float32)
                return u_net({"x": Xt[:, 0:1], "t": Xt[:, 1:2]})["u"].numpy()
        arrays = {"nu_hat": np.float64(nu_hat), "u_test": proto.export_predictions(prob, predict)["u_test"]}
        meta.update(status="ok", wall_s=clk.wall_s, cpu_s=clk.cpu_s, loadavg_before=clk.loadavg_before,
                    loadavg_after=clk.loadavg_after, iterations=steps.calls, nu_hat=nu_hat,
                    nu_field_std=float(nu_field.std()), nu_field_min=float(nu_field.min()),
                    nu_field_max=float(nu_field.max()))
    except Exception as exc:  # noqa: BLE001 - a failure is a result
        import traceback
        meta.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-4000:])
    meta["process_wall_s"] = time.perf_counter() - t0
    R.write_run(args.out, args.run_id, meta, arrays)


if __name__ == "__main__":
    main()

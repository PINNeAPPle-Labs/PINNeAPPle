# Track C (inverse Burgers: nu from 2000 noisy samples), PINNeAPPle runner.
# Follows pinneapple_neural/trainer/inverse_pinn.py (module docstring, "Typical workflow"): the user supplies
# residual_fn(model, x_col, params) and the trainer handles the loop. nu = InverseParam(0.05, log_space=True) (the
# documented option for a positive coefficient), InversePINN(model, [nu], residual_fn, sensors, values,
# domain_bounds, n_col=10000, lr=1e-3) with its defaults otherwise: lr_params 1e-2, fresh uniform collocation points
# every step, EMA loss balancing, gradient clipping and early stopping (patience 300) built in.
# Network: the A1 network (tanh 2 -> 50 x 4 -> 1) with the shared Xavier weights of the seed.
"""PINNeAPPle runner of track C."""
from __future__ import annotations

import time

import numpy as np

from pinneapple_tools.benchmark_suite.crossframework import inverse as INV
from pinneapple_tools.benchmark_suite.crossframework import problems as P
from pinneapple_tools.benchmark_suite.crossframework import protocol as proto
from pinneapple_tools.benchmark_suite.crossframework import records as R
from pinneapple_tools.benchmark_suite.crossframework import weights as W

SOURCE = "PINNeAPPle pinneapple_neural.trainer.inverse_pinn.InversePINN at the commit under test"


def main(argv=None):
    parser = proto.runner_parser("PINNeAPPle, track C")
    parser.add_argument("--patience", type=int, default=None,
                        help="diagnostic only: InversePINN early-stopping patience (default: the library's 300)")
    parser.add_argument("--framework", default=None,
                        help="label of the result row (default: pinneapple, or pinneapple_nostop with --patience)")
    args = parser.parse_args(argv)
    R.setup_threads(args.threads)
    # with --patience the run is the equal-budget variant (early stopping off), kept apart from the default recipe
    fw = args.framework or ("pinneapple_nostop" if args.patience else "pinneapple")
    cfg = {**proto.C, "iters": args.iters, "patience": args.patience or "library default (300)",
           "early_stopping_on": ("unweighted L_pde + L_data" if fw in ("pinneapple_fixed", "pinneapple_equal")
                                 else "weighted total"), "nu_param": "InverseParam(log_space=True)", "lr_params": 1e-2,
           "collocation": "10000 fresh uniform points per step (InversePINN's only mode)"}
    meta = proto.base_meta(args, track="C", framework=fw, source=SOURCE, config=cfg)
    t0 = time.perf_counter()
    arrays = {}
    try:
        import torch

        from pinneapple_neural.trainer.inverse_pinn import InverseParam, InversePINN

        obs = INV.observations()
        torch.manual_seed(args.seed)
        sizes = W.layer_sizes()
        layers = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:], strict=False)):
            layers.append(torch.nn.Linear(a, b))
            if i < len(sizes) - 2:
                layers.append(torch.nn.Tanh())
        model = torch.nn.Sequential(*layers)
        W.load_into_linears([m for m in model if isinstance(m, torch.nn.Linear)],
                            W.xavier_uniform_init(sizes, args.seed))

        def residual_fn(net, x, params):
            x.requires_grad_(True)
            u = net(x)
            g = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
            u_x, u_t = g[:, 0:1], g[:, 1:2]
            u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
            return u_t + u * u_x - params["nu"] * u_xx

        nu = InverseParam(initial_value=proto.C["nu_init"], log_space=True, name="nu")
        trainer = InversePINN(model, [nu], residual_fn, torch.from_numpy(obs["X"]), torch.from_numpy(obs["u"]),
                              domain_bounds=[(-1.0, 1.0), (0.0, 1.0)], n_col=proto.C["n_interior"], lr=proto.C["lr"],
                              device=args.device, **({"patience": args.patience} if args.patience else {}))
        meta["config"]["weighting"] = getattr(trainer, "weighting", "adaptive (no option at that commit)")
        with R.Clock() as clk:
            hist = trainer.train(n_epochs=args.iters)
        nu_hat = trainer.get_identified_params()["nu"]
        model.eval()

        def predict(X):
            with torch.no_grad():
                return model(torch.as_tensor(X, dtype=torch.float32)).numpy()
        prob = P.get_problem("burgers")
        arrays = {"nu_hat": np.float64(nu_hat), "u_test": proto.export_predictions(prob, predict)["u_test"]}
        n_it = len(hist["total"])
        meta.update(status="ok", wall_s=clk.wall_s, cpu_s=clk.cpu_s, loadavg_before=clk.loadavg_before,
                    loadavg_after=clk.loadavg_after, iterations=n_it, early_stopped=n_it < args.iters,
                    nu_hat=nu_hat, nu_history=hist["params"]["nu"][:: max(1, n_it // 200)],
                    loss_history=hist["total"][:: max(1, n_it // 200)],
                    pde_history=hist["pde"][:: max(1, n_it // 200)], data_history=hist["data"][:: max(1, n_it // 200)])
    except Exception as exc:  # noqa: BLE001 - a failure is a result
        import traceback
        meta.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-4000:])
    meta["process_wall_s"] = time.perf_counter() - t0
    R.write_run(args.out, args.run_id, meta, arrays)


if __name__ == "__main__":
    main()

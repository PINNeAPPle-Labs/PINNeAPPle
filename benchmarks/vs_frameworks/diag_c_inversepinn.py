"""Diagnostic of track C for InversePINN, outside the protocol rows: one knob changed at a time from the recipe of
``track_c_pinneapple.py`` (same network, weights, observations, n_col, lr), to find why nu stays near 0.04.

    python -m benchmarks.vs_frameworks.diag_c_inversepinn run <variant> <seed> <out.json> [--iters 3000]
    python -m benchmarks.vs_frameworks.diag_c_inversepinn collect <dir>     # -> results/C_diagnostics.json

The baseline is the recipe the benchmark ran first, with ``weighting="adaptive"`` (InversePINN's default then).
Variants (early stopping off in all, ``patience=None``, so each has the full budget of the other frameworks):
* ``equalw``: the adaptive PDE weight (``100 * clamp(ema_data / ema_pde, 0.01, 100)``) replaced by a fixed weight
  equal to the data weight (100), i.e. equal fixed weights as in DeepXDE's and Sym's runs (``weighting="equal"``,
  InversePINN's default since; the recorded runs replaced the module's ``min`` by ``lambda *a: 1.0``, the same
  arithmetic).
* ``linear``: nu as a linear parameter instead of log space.
* ``lrp3``: lr_params 1e-3 (the network's lr, as DeepXDE uses for its variable) instead of 1e-2.
* ``equalw_linear_lrp3``: all three.
This is a diagnosis, not tuning: none of these is reported as a PINNeAPPle result.
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"


def run(variant: str, seed: int, iters: int, out: str) -> dict:
    import torch

    from pinneapple_neural.trainer.inverse_pinn import InverseParam, InversePINN
    from pinneapple_tools.benchmark_suite.crossframework import inverse as INV
    from pinneapple_tools.benchmark_suite.crossframework import problems as P
    from pinneapple_tools.benchmark_suite.crossframework import protocol as proto
    from pinneapple_tools.benchmark_suite.crossframework import weights as W

    torch.set_num_threads(1)
    obs = INV.observations()
    torch.manual_seed(seed)
    sizes = W.layer_sizes()
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:], strict=False)):
        layers.append(torch.nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(torch.nn.Tanh())
    model = torch.nn.Sequential(*layers)
    W.load_into_linears([m for m in model if isinstance(m, torch.nn.Linear)], W.xavier_uniform_init(sizes, seed))

    def residual_fn(net, x, params):
        x.requires_grad_(True)
        u = net(x)
        g = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_x, u_t = g[:, 0:1], g[:, 1:2]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        return u_t + u * u_x - params["nu"] * u_xx

    nu = InverseParam(initial_value=proto.C["nu_init"], log_space="linear" not in variant, name="nu")
    tr = InversePINN(model, [nu], residual_fn, torch.from_numpy(obs["X"]), torch.from_numpy(obs["u"]),
                     domain_bounds=[(-1.0, 1.0), (0.0, 1.0)], n_col=proto.C["n_interior"], lr=proto.C["lr"],
                     lr_params=1e-3 if "lrp3" in variant else 1e-2, patience=None,
                     weighting="equal" if "equalw" in variant else "adaptive")
    t0 = time.process_time()
    h = tr.train(n_epochs=iters)
    nu_hat = tr.get_identified_params()["nu"]
    prob = P.get_problem("burgers")
    Xt = P.test_points(prob)
    with torch.no_grad():
        u = model(torch.as_tensor(Xt, dtype=torch.float32)).numpy().ravel()
    ref = prob.exact(Xt).ravel()
    true = prob.params["nu"]
    rec = {"variant": variant, "seed": seed, "iterations": len(h["total"]), "nu_hat": nu_hat,
           "nu_rel_err": abs(nu_hat - true) / true, "rel_l2": float(np.linalg.norm(u - ref) / np.linalg.norm(ref)),
           "cpu_s": time.process_time() - t0, "final_data_loss": h["data"][-1], "final_pde_loss": h["pde"][-1],
           "nu_history": h["params"]["nu"][::150]}
    Path(out).write_text(json.dumps(rec, indent=1))
    return rec


def collect(d: str) -> None:
    rows = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(Path(d) / "*.json")))]
    (RESULTS / "C_diagnostics.json").write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} diagnostic runs -> {RESULTS / 'C_diagnostics.json'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("variant", choices=("equalw", "linear", "lrp3", "equalw_linear_lrp3"))
    r.add_argument("seed", type=int)
    r.add_argument("out")
    r.add_argument("--iters", type=int, default=3000)
    c = sub.add_parser("collect")
    c.add_argument("dir")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        print(json.dumps(run(a.variant, a.seed, a.iters, a.out)))
    else:
        collect(a.dir)


if __name__ == "__main__":
    main()

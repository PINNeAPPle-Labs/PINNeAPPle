# Track A (PINN forward problems), PhysicsNeMo-Sym runner. nvidia-physicsnemo.sym 2.4.0 on nvidia-physicsnemo 2.2.2.
# Follows the official examples of NVIDIA/physicsnemo-sym v2.4.0 (examples/):
#   helmholtz  -> examples/helmholtz/helmholtz.py + conf/config.yaml: exactly this problem (a1 = 1, a2 = 4, k = 1 on
#                 [-1, 1]^2): HelmholtzEquation(u, k=1, dim=2), the default fully_connected arch (6 x 512, SiLU,
#                 weight norm), Rectangle geometry, wall batch 800, interior batch 4000 with lambda = sdf, Adam,
#                 tf_exponential_lr (0.95 every 200 steps), loss "sum", 20000 steps, gradient clipping 0.5 (default).
#   poisson    -> the same example structure (no Poisson example in Sym 2.4.0) with the Poisson residual.
#   burgers    -> examples/wave_equation/wave_1d.py + conf/config.yaml (the 1-D time-dependent example; Sym 2.4.0 has
#                 no Burgers PINN example): Line1D geometry with a time parameterization, IC as an interior
#                 constraint at t = 0, BC and interior constraints, batches IC 315 / BC 200 / interior 3150,
#                 tf_exponential_lr 0.95 every 100 steps, 10000 steps, the default arch. Burgers as a custom PDE
#                 (physicsnemo.sym.eq.pde.PDE, the documented way to add an equation).
#   allen_cahn -> the same wave_1d structure, Allen-Cahn as a custom PDE, u(+-1, t) = -1.
# A2 runs stop at the time cap through Solver.solve(sigterm_handler=...) (checked once per step by the trainer).
# A1: PointwiseConstraint.from_numpy on the shared point sets (full batch), FullyConnectedArch 50 x 4 tanh without
# weight norm, loaded with the shared Xavier weights, constant lr 1e-3 (exponential_lr, gamma = 1), lambda = 1 / N
# per point so that each loss term is a mean (Sym's PointwiseLossNorm sums over points), no gradient clipping
# (grad_clip_max_norm = 1e30; Sym clips to 0.5 by default), training.max_steps = iters - 1 (Sym runs steps 0..max).
"""PhysicsNeMo-Sym runner of track A (see the comment above and PROTOCOL.md)."""
from __future__ import annotations

import logging
import re
import tempfile
import time

import numpy as np

from pinneapple_tools.benchmark_suite.crossframework import problems as P
from pinneapple_tools.benchmark_suite.crossframework import protocol as proto
from pinneapple_tools.benchmark_suite.crossframework import records as R
from pinneapple_tools.benchmark_suite.crossframework import weights as W

SOURCE = ("PhysicsNeMo-Sym 2.4.0 examples/helmholtz/helmholtz.py (helmholtz, poisson) and "
          "examples/wave_equation/wave_1d.py (burgers, allen_cahn)")

RECIPES = {
    "helmholtz": {"example": "examples/helmholtz/helmholtz.py", "max_steps": 20000, "decay_steps": 200,
                  "batch": {"wall": 800, "interior": 4000}},
    "poisson": {"example": "examples/helmholtz/helmholtz.py (structure; no Poisson example)", "max_steps": 20000,
                "decay_steps": 200, "batch": {"wall": 800, "interior": 4000}},
    "burgers": {"example": "examples/wave_equation/wave_1d.py (structure; no Burgers example)", "max_steps": 10000,
                "decay_steps": 100, "batch": {"IC": 315, "BC": 200, "interior": 3150}},
    "allen_cahn": {"example": "examples/wave_equation/wave_1d.py (structure; no Allen-Cahn example)",
                   "max_steps": 10000, "decay_steps": 100, "batch": {"IC": 315, "BC": 200, "interior": 3150}},
}


def make_pde(prob: P.PINNProblem):
    """The residual as a Sym ``PDE`` (sympy), named ``"res"`` (``outvar`` 0, or the source for the steady ones)."""
    from physicsnemo.sym.eq.pde import PDE
    from sympy import Function, Symbol, pi, sin

    a, b = Symbol(prob.coords[0]), Symbol(prob.coords[1])

    class Residual(PDE):
        name = f"{prob.name}_residual"

        def __init__(self):
            u = Function("u")(a, b)
            self.equations = {}
            if prob.name == "burgers":
                self.equations["res"] = u.diff(b) + u * u.diff(a) - prob.params["nu"] * u.diff(a, 2)
            elif prob.name == "allen_cahn":
                self.equations["res"] = u.diff(b) - prob.params["d"] * u.diff(a, 2) - 5 * u + 5 * u ** 3
            elif prob.name == "poisson":
                self.equations["res"] = u.diff(a, 2) + u.diff(b, 2) + 2 * pi ** 2 * sin(pi * a) * sin(pi * b)
            else:
                a1, a2, k = prob.params["a1"], prob.params["a2"], prob.params["k"]
                q = (k ** 2 - (a1 * pi) ** 2 - (a2 * pi) ** 2) * sin(a1 * pi * a) * sin(a2 * pi * b)
                self.equations["res"] = u.diff(a, 2) + u.diff(b, 2) + k ** 2 * u - q
    return Residual()


class _StepCounter:
    """Passed as ``sigterm_handler``: the trainer calls it once at the top of every step; it counts the steps and
    asks to stop once the time cap (if any) is used up."""

    def __init__(self, cap=None):
        self.calls, self.cap = 0, cap

    def __call__(self) -> bool:
        if self.cap is not None and self.cap.expired():
            return True
        self.calls += 1
        return False


class _LossLog(logging.Handler):
    """Collects the loss the trainer prints every ``print_stats_freq`` steps."""
    pattern = re.compile(r"\[step:\s*(\d+)\] loss:\s*([-+0-9.eEna]+)")

    def __init__(self):
        super().__init__()
        self.steps, self.loss = [], []

    def emit(self, record):
        m = self.pattern.search(record.getMessage())
        if m:
            self.steps.append(int(m.group(1)))
            self.loss.append(float(m.group(2)))


def compose_cfg(overrides, name="config", device="cpu"):
    """Compose the Sym config. On a CPU-only machine the default ``cuda_graphs: true`` crashes in the first step
    (torch.cuda.Stream on a machine without a driver), so it is switched off there (the documented config option)."""
    from physicsnemo.sym.hydra.utils import compose
    extra = [] if str(device).startswith("cuda") else ["cuda_graphs=false"]
    return compose(config_name=name, config_path="sym_conf", overrides=list(overrides) + extra)


def _arch_linears(net):
    """The torch Linear layers of a FullyConnectedArch, input to output."""
    import torch
    lins = [m for m in net.modules() if isinstance(m, torch.nn.Linear)]
    return lins


def run_a1(args, prob, meta, workdir):
    import torch
    from physicsnemo.sym.domain import Domain
    from physicsnemo.sym.domain.constraint import PointwiseConstraint
    from physicsnemo.sym.key import Key
    from physicsnemo.sym.models.activation import Activation
    from physicsnemo.sym.models.fully_connected import FullyConnectedArch
    from physicsnemo.sym.solver import Solver

    pts = P.load_point_set(proto.ensure_point_set(prob, args.root))
    meta["config"]["point_set_sha256"] = P.point_set_digest(pts)
    ca, cb = prob.coords
    torch.manual_seed(args.seed)
    net = FullyConnectedArch(input_keys=[Key(ca), Key(cb)], output_keys=[Key("u")], layer_size=proto.A1["width"],
                             nr_layers=proto.A1["depth"], activation_fn=Activation.TANH, weight_norm=False)
    W.load_into_linears(_arch_linears(net), W.xavier_uniform_init(W.layer_sizes(), args.seed))
    nodes = make_pde(prob).make_nodes() + [net.make_node(name="net")]
    domain = Domain()

    def add(name, X, outvar):
        n = len(X)
        domain.add_constraint(PointwiseConstraint.from_numpy(
            nodes=nodes, invar={ca: X[:, 0:1], cb: X[:, 1:2]}, outvar=outvar, batch_size=n,
            lambda_weighting={k: np.full((n, 1), 1.0 / n, dtype=np.float32) for k in outvar}), name)

    add("interior", pts["X_f"], {"res": np.zeros((len(pts["X_f"]), 1), dtype=np.float32)})
    add("boundary", pts["X_b"], {"u": pts["u_b"]})
    if prob.time_dependent:
        add("initial", pts["X_0"], {"u": pts["u_0"]})
    cfg = compose_cfg([f"training.max_steps={args.iters - 1}", "training.grad_clip_max_norm=1e30",
                       f"optimizer.lr={proto.A1['lr']}", f"network_dir={workdir}", f"device={args.device}"],
                      name="config_a1", device=args.device)
    log = _LossLog()
    logging.getLogger().addHandler(log)
    slv = Solver(cfg, domain)
    steps = _StepCounter()
    with R.Clock() as clk:
        slv.solve(sigterm_handler=steps)
    meta.update(iterations=steps.calls, loss_history=log.loss,
                loss_steps=log.steps, final_loss=log.loss[-1] if log.loss else None,
                n_parameters=W.count_parameters(net), cfg_resolved={"jit": bool(cfg.jit),
                                                                    "cuda_graphs": bool(cfg.cuda_graphs)})
    return clk, net


def run_a2(args, prob, meta, workdir):
    import torch
    from physicsnemo.sym.domain import Domain
    from physicsnemo.sym.domain.constraint import (
        PointwiseBoundaryConstraint,
        PointwiseInteriorConstraint,
    )
    from physicsnemo.sym.eq.pdes.wave_equation import HelmholtzEquation
    from physicsnemo.sym.geometry.primitives_1d import Line1D
    from physicsnemo.sym.geometry.primitives_2d import Rectangle
    from physicsnemo.sym.hydra import instantiate_arch
    from physicsnemo.sym.key import Key
    from physicsnemo.sym.solver import Solver
    from sympy import Symbol

    rc = RECIPES[prob.name]
    meta["config"].update(rc)
    cfg = compose_cfg([f"training.max_steps={rc['max_steps']}", f"scheduler.decay_steps={rc['decay_steps']}",
                       f"network_dir={workdir}", f"device={args.device}"], device=args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ca, cb = prob.coords
    net = instantiate_arch(input_keys=[Key(ca), Key(cb)], output_keys=[Key("u")], cfg=cfg.arch.fully_connected)
    (a0, a1), (b0, b1) = prob.bounds
    domain = Domain()
    x, y = Symbol(ca), Symbol(cb)
    if prob.name in ("helmholtz", "poisson"):
        if prob.name == "helmholtz":
            pde = HelmholtzEquation(u="u", k=prob.params["k"], dim=2)
            key = "helmholtz"
            from sympy import pi, sin
            # the example's outvar: helmholtz = -(u_xx + u_yy + k^2 u) = -q
            q = (prob.params["k"] ** 2 - (prob.params["a1"] * pi) ** 2 - (prob.params["a2"] * pi) ** 2) \
                * sin(prob.params["a1"] * pi * x) * sin(prob.params["a2"] * pi * y)
            target = -q
        else:
            pde, key, target = make_pde(prob), "res", 0
        nodes = pde.make_nodes() + [net.make_node(name="net")]
        rec = Rectangle((a0, b0), (a1, b1))
        domain.add_constraint(PointwiseBoundaryConstraint(nodes=nodes, geometry=rec, outvar={"u": 0},
                                                          batch_size=rc["batch"]["wall"]), "wall")
        domain.add_constraint(PointwiseInteriorConstraint(
            nodes=nodes, geometry=rec, outvar={key: target}, batch_size=rc["batch"]["interior"],
            bounds={x: (a0, a1), y: (b0, b1)}, lambda_weighting={key: Symbol("sdf")}), "interior")
    else:
        nodes = make_pde(prob).make_nodes() + [net.make_node(name="net")]
        geo = Line1D(a0, a1)
        time_range = {y: (b0, b1)}
        u0 = prob.initial
        from sympy import cos, pi, sin
        ic = -sin(pi * x) if prob.name == "burgers" else x ** 2 * cos(pi * x)
        bc_val = 0.0 if prob.name == "burgers" else -1.0
        _ = u0
        domain.add_constraint(PointwiseInteriorConstraint(
            nodes=nodes, geometry=geo, outvar={"u": ic}, batch_size=rc["batch"]["IC"], lambda_weighting={"u": 1.0},
            parameterization={y: 0.0}), "IC")
        domain.add_constraint(PointwiseBoundaryConstraint(
            nodes=nodes, geometry=geo, outvar={"u": bc_val}, batch_size=rc["batch"]["BC"],
            parameterization=time_range), "BC")
        domain.add_constraint(PointwiseInteriorConstraint(
            nodes=nodes, geometry=geo, outvar={"res": 0}, batch_size=rc["batch"]["interior"],
            parameterization=time_range), "interior")
    cap = R.TimeCap(args.cap_s, args.cap_kind)
    log = _LossLog()
    logging.getLogger().addHandler(log)
    slv = Solver(cfg, domain)
    steps = _StepCounter(cap)
    with R.Clock() as clk:
        cap.restart()
        slv.solve(sigterm_handler=steps)
    meta.update(iterations=steps.calls, loss_history=log.loss,
                loss_steps=log.steps, final_loss=log.loss[-1] if log.loss else None,
                n_parameters=W.count_parameters(net), cap_used_s=cap.used(),
                arch={k: cfg.arch.fully_connected[k] for k in ("layer_size", "nr_layers", "activation_fn",
                                                               "weight_norm")})
    return clk, net


def main(argv=None):
    args = proto.runner_parser("PhysicsNeMo-Sym, track A").parse_args(argv)
    R.setup_threads(args.threads)
    prob = P.get_problem(args.problem)
    cfg = dict(proto.A1) if args.mode == "a1" else {"recipe": "official example", **proto.A2, "cap_s": args.cap_s}
    meta = proto.base_meta(args, track="A", framework="physicsnemo_sym", source=SOURCE, config=cfg)
    t0 = time.perf_counter()
    try:
        import torch
        with tempfile.TemporaryDirectory(prefix="sym_run_") as workdir:
            clk, net = (run_a1 if args.mode == "a1" else run_a2)(args, prob, meta, workdir)
            net.eval()
            ca, cb = prob.coords

            def predict(X):
                with torch.no_grad():
                    Xt = torch.as_tensor(X, dtype=torch.float32, device=args.device)
                    return net({ca: Xt[:, 0:1], cb: Xt[:, 1:2]})["u"].cpu().numpy()
            arrays = proto.export_predictions(prob, predict)
        it = meta.get("iterations")
        meta.update(status="ok", wall_s=clk.wall_s, cpu_s=clk.cpu_s, loadavg_before=clk.loadavg_before,
                    loadavg_after=clk.loadavg_after, iterations=it)
    except Exception as exc:  # noqa: BLE001 - a failure is a result
        import traceback
        arrays = {}
        meta.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-4000:])
    meta["process_wall_s"] = time.perf_counter() - t0
    R.write_run(args.out, args.run_id, meta, arrays)


if __name__ == "__main__":
    main()

"""Second-kind integral equations, solved in Nyström form by the PDE compiler's integral kinds.

PDE
---
    Fredholm   u(x) = f(x) + lambda * int_a^b K(x, s) u(s) ds
    Volterra   u(x) = f(x) + lambda * int_a^x K(x, s) u(s) ds

The integral is replaced by a fixed quadrature on the collocation points, so the residual is an algebraic equation
in the values the network produces: ``r_i = u_i - f_i - lambda (sum_j w_j K_ij u_j + D_i u_i)``, with ``D_i`` the
analytic self-cell integral of the singular part of the kernel.

What this shows
---------------
- Three cases with closed-form solutions and the measured order of the rule: second order for a bounded kernel,
  first order for a logarithmic one (the two cells next to the singularity carry an ``O(h)`` error, which is a
  property of the rule and not a bug).
- The residual the compiler hands to the training loop, evaluated on the exact solution and on a wrong one. The
  exact one falls with the grid, the wrong one does not.
- The direct Nyström solve of the same discretisation. This is the baseline any neural solution has to match, and
  it is why the claim "it solves integral equations" needs a case where the direct solve is expensive, which is the
  parametric and inverse setting and not the single linear solve.
- That ``lambda = 1`` is the eigenvalue of the unit kernel: the exact solution still has a vanishing residual while
  the linear system is numerically singular. The two facts are different and the Fredholm alternative is why.

Run
---
python examples/numerical_solvers/14_integral_equations_nystrom.py
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from pinneapple_physics.pde_environment.spec import PDETermSpec, ProblemSpec
from pinneapple_physics.pinn_solver.compiler.compile import compile_problem
from pinneapple_physics.pinn_solver.compiler.integral_kinds import (
    constant_kernel,
    log_diag,
    log_kernel,
    nystrom_matrix,
)

E = math.e


# -------------------------------------------------------------------------------------------------- plumbing
def spec_of(kind: str, field: str, params: dict) -> ProblemSpec:
    return ProblemSpec(name=kind, dim=1, coords=("x",), fields=(field,),
                       pde=PDETermSpec(kind=kind, fields=(field,), coords=("x",), params=params),
                       conditions=(), domain_bounds={"x": (0.0, 1.0)})


class Field(torch.nn.Module):
    """A stand-in model returning the closed-form field: the compiler only ever calls it."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, X):
        return self.fn(X)


def residual_rms(spec: ProblemSpec, fn, x: np.ndarray, ctx: dict) -> float:
    """Root-mean-square residual of the compiled PDE at the nodes ``x``."""
    loss = compile_problem(spec)(Field(fn), None, {
        "x_col": torch.as_tensor(x, dtype=torch.float64).reshape(-1, 1),
        "ctx": ctx,
    })
    return float(loss["pde"].detach()) ** 0.5


def order(values: list[float]) -> float:
    return float(np.mean(np.log(np.array(values[:-1]) / np.array(values[1:])) / np.log(2.0)))


def direct_solve(x: np.ndarray, kernel_fn, f: np.ndarray, lam: float, *, diag_fn=None, volterra=False):
    """Classical Nyström solve of ``u = f + lambda K u`` on the same nodes -- the baseline."""
    a, d = nystrom_matrix(x, kernel_fn, {}, diag_fn=diag_fn, volterra=volterra)
    return np.linalg.solve(np.eye(x.size) - lam * (a + np.diag(d)), f)


GRID = (21, 41, 81, 161)


# -------------------------------------------------------------------------------------------------- case 1
def case_bounded_kernel() -> None:
    """u = exp(x) solves u(x) = f(x) + int_0^1 1 * u(s) ds with f = exp(x) - (e - 1)."""
    print("\n[1] Fredholm, bounded kernel K = 1, f = exp(x) - (e - 1), u = exp(x)")
    spec = spec_of("fredholm2", "u", {"lambda": 1.0})
    ctx = {"kernel_fn": constant_kernel, "f_fn": lambda X, c: np.exp(X[:, 0]) - (E - 1.0)}
    exact, wrong, solve_err = [], [], []
    for n in GRID:
        x = np.linspace(0.0, 1.0, n)
        exact.append(residual_rms(spec, lambda X: torch.exp(X[:, 0:1]), x, ctx))
        wrong.append(residual_rms(spec, lambda X: torch.exp(0.9 * X[:, 0:1]), x, ctx))
        # the direct solve has to stay away from lambda = 1, see case 4
        u = direct_solve(x, constant_kernel, np.exp(x) - 0.5 * (E - 1.0), 0.5)
        solve_err.append(float(np.max(np.abs(u - np.exp(x)))))
    for n, a, b, c in zip(GRID, exact, wrong, solve_err, strict=True):
        print(f"    n={n:4d}  residual(exact)={a:.3e}  residual(wrong)={b:.3e}  solve err (lambda=0.5)={c:.3e}")
    print(f"    order: residual(exact)={order(exact):.2f}  solve err={order(solve_err):.2f}")


# -------------------------------------------------------------------------------------------------- case 2
def case_log_kernel() -> None:
    """u = 1 solves u(x) = f(x) + int_0^1 ln|x - s| u(s) ds with f = 1 - (x ln x + (1-x) ln(1-x) - 1)."""
    print("\n[2] Fredholm, logarithmic kernel, u = 1")
    with_diag = spec_of("fredholm2", "u", {"lambda": 1.0, "singular": True})
    without = spec_of("fredholm2", "u", {"lambda": 1.0})
    good = {"kernel_fn": log_kernel, "diag_fn": log_diag, "f_fn": _free_term_log}
    bad = {"kernel_fn": log_kernel, "f_fn": _free_term_log}
    ones = lambda X: torch.ones_like(X[:, 0:1])                                        # noqa: E731
    kept, dropped = [], []
    for n in GRID:
        x = np.linspace(0.0, 1.0, n)
        kept.append(residual_rms(with_diag, ones, x, good))
        dropped.append(residual_rms(without, ones, x, bad))
    for n, a, b in zip(GRID, kept, dropped, strict=True):
        print(f"    n={n:4d}  analytic diagonal={a:.3e}   no diagonal={b:.3e}")
    x = np.linspace(0.0, 1.0, 161)
    u = direct_solve(x, log_kernel, _free_term_log(x.reshape(-1, 1), {}), 1.0, diag_fn=log_diag)
    print(f"    order with the diagonal={order(kept):.2f} (first order: the two cells beside the singularity)")
    print(f"    direct solve at n=161: max |u - 1| = {np.max(np.abs(u - 1.0)):.3e}")


# -------------------------------------------------------------------------------------------------- case 3
def case_volterra() -> None:
    """u = exp(x) solves u(x) = 1 + int_0^x u(s) ds. The mask is not a detail: letting the future contribute turns
    the same equation into a different one, and the difference is not subtle."""
    print("\n[3] Volterra, K = 1, f = 1, u = exp(x)")
    spec = spec_of("volterra2", "u", {"lambda": 1.0})
    ctx = {"kernel_fn": constant_kernel, "f_fn": lambda X, c: np.ones(len(X))}
    exact, solve_err = [], []
    for n in GRID:
        x = np.linspace(0.0, 1.0, n)
        exact.append(residual_rms(spec, lambda X: torch.exp(X[:, 0:1]), x, ctx))
        solve_err.append(float(np.max(np.abs(direct_solve(x, constant_kernel, np.ones(n), 1.0, volterra=True)
                                               - np.exp(x)))))
    for n, a, b in zip(GRID, exact, solve_err, strict=True):
        print(f"    n={n:4d}  residual(exact)={a:.3e}  solve err={b:.3e}")
    print(f"    order: residual={order(exact):.2f}  solve err={order(solve_err):.2f}")

    x = np.linspace(0.0, 1.0, 81)
    a_ok, d = nystrom_matrix(x, constant_kernel, {}, volterra=True)
    a_bad, _ = nystrom_matrix(x, constant_kernel)            # every j != i: past and future
    u_ok = np.linalg.solve(np.eye(x.size) - (a_ok + np.diag(d)), np.ones(x.size))
    u_bad = np.linalg.solve(np.eye(x.size) - (a_bad + np.diag(d)), np.ones(x.size))
    print(f"    with the future allowed to contribute (n=81): max |u_wrong - u_right| = "
          f"{np.max(np.abs(u_bad - u_ok)):.1f}, against an exact solution bounded by {math.e:.2f}")


# -------------------------------------------------------------------------------------------------- case 4
def case_the_eigenvalue() -> None:
    """lambda = 1 is the eigenvalue of the unit kernel: residual zero, system singular."""
    print("\n[4] lambda = 1 is the eigenvalue of K = 1 (the Fredholm alternative)")
    x = np.linspace(0.0, 1.0, 81)
    a, d = nystrom_matrix(x, constant_kernel)
    cond = float(np.linalg.cond(np.eye(x.size) - (a + np.diag(d))))
    spec = spec_of("fredholm2", "u", {"lambda": 1.0})
    ctx = {"kernel_fn": constant_kernel, "f_fn": lambda X, c: np.exp(X[:, 0]) - (E - 1.0)}
    r = residual_rms(spec, lambda X: torch.exp(X[:, 0:1]), x, ctx)
    print(f"    cond(I - K) at n=81 = {cond:.3e}")
    print(f"    residual of the exact solution = {r:.3e}   (small, although the system is singular)")


def _free_term_log(X, ctx):
    x = X[:, 0]
    left = np.where(x > 0.0, x * np.log(np.clip(x, 1e-300, None)), 0.0)
    right = np.where(x < 1.0, (1.0 - x) * np.log(np.clip(1.0 - x, 1e-300, None)), 0.0)
    return 1.0 - (left + right - 1.0)


# -------------------------------------------------------------------------------------------------- case 5
def _mlp(in_dim: int, width: int = 64, depth: int = 3, seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    layers, d = [], in_dim
    for _ in range(depth):
        layers += [torch.nn.Linear(d, width), torch.nn.Tanh()]
        d = width
    return torch.nn.Sequential(*layers, torch.nn.Linear(d, 1)).double()


def case_neural_arm() -> None:
    """The trained arm on the same nodes as the direct solve: same discretisation, different search.

    The loss comes from ``compile_problem`` -- the same residual the compiler hands to any training loop -- so this
    is also the check that the compiled integral kind is differentiable and trainable, not a second implementation of
    the equation written next to it.
    """
    print("\n[5] Trained network against the direct Nystrom solve, n=81, lambda=0.5")
    x = np.linspace(0.0, 1.0, 81)
    nodes = torch.as_tensor(x).reshape(-1, 1)
    lam = 0.5
    ctx = {"kernel_fn": constant_kernel, "f_fn": lambda X, c: np.exp(X[:, 0]) - lam * (E - 1.0)}
    loss_fn = compile_problem(spec_of("fredholm2", "u", {"lambda": lam}))

    net = _mlp(1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 2500)
    start = time.time()
    for _ in range(2500):
        opt.zero_grad()
        loss = loss_fn(net, None, {"x_col": nodes, "ctx": ctx})["pde"]
        loss.backward()
        opt.step()
        schedule.step()
    trained_seconds = time.time() - start
    with torch.no_grad():
        u = net(nodes).numpy().ravel()
    residual = float(loss_fn(net, None, {"x_col": nodes, "ctx": ctx})["pde"].detach()) ** 0.5

    start = time.time()
    ud = direct_solve(x, constant_kernel, np.exp(x) - lam * (E - 1.0), lam)
    direct_seconds = time.time() - start
    print(f"    trained   max err = {np.max(np.abs(u - np.exp(x))):.3e}   residual={residual:.2e}   "
          f"({trained_seconds:.1f} s)")
    print(f"    direct    max err = {np.max(np.abs(ud - np.exp(x))):.3e}   ({direct_seconds * 1e3:.2f} ms)")
    print(f"    the direct solve is {np.max(np.abs(u - np.exp(x))) / np.max(np.abs(ud - np.exp(x))):.0f}x more "
          f"accurate, and it answers in milliseconds where the training took seconds")


# -------------------------------------------------------------------------------------------------- case 6
class _Parametrised(torch.nn.Module):
    """The network with the parameter appended as an extra input column: the model sees ``(x, lambda)``."""

    def __init__(self, net: torch.nn.Module, lam: float):
        super().__init__()
        self.net = net
        self.lam = lam

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([X, torch.full_like(X[:, :1], self.lam)], 1))


def case_parametric() -> None:
    """One model for a family, where the direct solve has to be repeated for every parameter.

    ``u(x) = exp(x) + lambda int_0^1 u ds`` has the closed form ``u = exp(x) + lambda (e - 1) / (1 - lambda)``: the
    free term is fixed and the parameter moves the solution, so the model cannot ignore it. ``lambda = 1`` is the
    eigenvalue, which is why the band stops before it.
    """
    print("\n[6] Parametric arm: one model over lambda in [0.2, 0.8] (the direct solve is repeated per lambda)")
    x = np.linspace(0.0, 1.0, 81)
    nodes = torch.as_tensor(x).reshape(-1, 1)
    spec = spec_of("fredholm2", "u", {"lambda": 0.5})
    loss_fn = compile_problem(spec)
    ctx = {"kernel_fn": constant_kernel, "f_fn": lambda X, c: np.exp(X[:, 0])}

    net = _mlp(2, seed=1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 2500)
    start = time.time()
    for _ in range(2500):
        opt.zero_grad()
        lam = float(torch.rand(1)) * 0.6 + 0.2
        spec.pde.params["lambda"] = lam          # the compiler reads the params at every call
        loss = loss_fn(_Parametrised(net, lam), None, {"x_col": nodes, "ctx": ctx})["pde"]
        loss.backward()
        opt.step()
        schedule.step()
    seconds = time.time() - start

    print(f"    trained once in {seconds:.1f} s; the direct solve would run once per parameter")
    worst = 0.0
    for lam in (0.25, 0.35, 0.5, 0.65, 0.75):
        with torch.no_grad():
            u = net(torch.cat([nodes, torch.full_like(nodes, lam)], 1)).numpy().ravel()
        closed = np.exp(x) + lam * (E - 1.0) / (1.0 - lam)
        err = float(np.max(np.abs(u - closed)) / np.max(np.abs(closed)))
        worst = max(worst, err)
        ud = direct_solve(x, constant_kernel, np.exp(x), lam)
        direct_err = float(np.max(np.abs(ud - closed)) / np.max(np.abs(closed)))
        print(f"    lambda={lam:.2f}  trained rel.err={err:.2e}   direct rel.err={direct_err:.2e}")
    print(f"    worst trained relative error over the held-out band: {worst:.2e}")

    start = time.time()
    hundred = [direct_solve(x, constant_kernel, np.exp(x), lam) for lam in np.linspace(0.2, 0.8, 100)]
    hundred_ms = (time.time() - start) * 1e3
    del hundred
    print(f"    one model replaces 100 direct solves of the band, which take {hundred_ms:.0f} ms and are exact. "
          f"On the clock the model does not win at this size: what it adds is a single differentiable artefact, "
          f"continuous in lambda, which is what a gradient-based design loop or an inverse problem consumes.")


def main() -> None:
    print(__doc__.split("Run\n---")[0].strip().splitlines()[0])
    case_bounded_kernel()
    case_log_kernel()
    case_volterra()
    case_the_eigenvalue()
    case_neural_arm()
    case_parametric()
    print("\nThe direct Nystrom solve is the baseline, not the competitor: a second-kind linear equation on a modest "
          "grid is cheap and accurate, and it wins the single solve by more than an order of magnitude in both error "
          "and time. The neural arm does not win on the clock at this size either; what it brings is one differentiable "
          "artefact for a whole family of parameters, which is what a design loop or an inverse problem consumes.")


if __name__ == "__main__":
    main()

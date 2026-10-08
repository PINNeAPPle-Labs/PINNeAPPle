"""``pp.compile``: optimise a physics problem once, solve it faster, with the same results.

>>> import pinneapple as pp
>>> compiled = pp.compile("burgers_1d", optimize="physics")           # doctest: +SKIP
>>> sol = compiled.solve("pinn", epochs=2000)                          # doctest: +SKIP
>>> print(compiled.benchmark(epochs=100))                              # doctest: +SKIP

What ``optimize="physics"`` does today: the PDE residual is built from several derivative calls on the same
field (the time derivative and the convective term of Burgers both differentiate ``u``; Navier-Stokes and the
elasticity and Euler kernels do it more). The compiled problem memoizes first derivatives within one loss
evaluation, so each field is differentiated once per step instead of once per term. The losses are the same;
the weights after training agree to floating-point summation order. The gain is the share of derivative graphs
saved, so it depends on the PDE (measured numbers in ``docs/core_concepts/solver.md``).

Not done: operator fusion beyond derivative reuse, kernel-level batching and memory planning. Forward-mode and
batched second derivatives were measured and were not faster on CPU (see the docs), so they are not used.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .physical_problem import PhysicalProblem
from .solving import Solution, as_problem, solve

__all__ = ["compile", "CompiledProblem", "Benchmark"]

_OPTIMIZE = ("physics", "none")


@dataclass
class Benchmark:
    problem: str
    epochs: int
    repeats: int
    baseline_s: float
    compiled_s: float
    speedup: float
    max_param_diff: float
    max_param_abs: float
    derivative_calls_saved: int
    derivative_calls_total: int

    def identical(self, rtol: float = 1e-4) -> bool:
        """True when trained weights of the two runs agree within ``rtol`` of the largest weight."""
        return self.max_param_diff <= rtol * max(self.max_param_abs, 1e-12)

    def __str__(self) -> str:
        return (f"{self.problem}: baseline {self.baseline_s:.2f}s  compiled {self.compiled_s:.2f}s  "
                f"speedup {self.speedup:.2f}x  ({self.derivative_calls_total - self.derivative_calls_saved} of "
                f"{self.derivative_calls_total} first-derivative graphs built per loss evaluation, max weight diff "
                f"{self.max_param_diff:.1e}, best of {self.repeats})")


class CompiledProblem:
    """A problem with its optimisations decided. ``solve`` and ``loss_fn`` use them."""

    def __init__(self, problem: PhysicalProblem, optimize: str = "physics") -> None:
        if optimize not in _OPTIMIZE:
            raise ValueError(f"optimize must be one of {_OPTIMIZE}, got {optimize!r}")
        self.problem = problem
        self.optimize = optimize
        self.cache_derivatives = optimize == "physics"

    @property
    def options(self) -> Dict[str, Any]:
        return {"cache_derivatives": self.cache_derivatives}

    def loss_fn(self, weights=None):
        """The PINN loss ``loss_fn(model, y_hat, batch)`` of this problem, compiled."""
        from .pinn_solver.compiler.compile import compile_problem
        return compile_problem(self.problem.to_pde_spec(), weights=weights, cache_derivatives=self.cache_derivatives)

    def solve(self, method: str = "pinn", **options: Any) -> Solution:
        """``pp.solve(problem, method)`` with the optimisations on (only the ``pinn`` method has any)."""
        if method == "pinn":
            options = {**self.options, **options}
        return solve(self.problem, method, **options)

    def benchmark(self, epochs: int = 100, repeats: int = 3, *, n_collocation: int = 2048, hidden_dim: int = 64,
                  n_layers: int = 4, num_threads: Optional[int] = 1, seed: int = 0, **pinn_options: Any) -> Benchmark:
        """Train the same PINN with and without the optimisations, alternating runs, and report the best CPU time
        of each plus how far the trained weights differ. ``num_threads=1`` keeps the timing stable."""
        import torch
        from .pinn_solver.compiler import autograd_ops as ao
        from . import solve_pde
        import pinneapple_neural.architectures  # noqa: F401
        from pinneapple_neural.architectures.registry import ModelRegistry

        spec = self.problem.to_pde_spec()
        old_threads = torch.get_num_threads()
        if num_threads:
            torch.set_num_threads(num_threads)

        def run(cache: bool):
            torch.manual_seed(seed)
            model = ModelRegistry.build("modified_mlp", in_dim=len(spec.coords), out_dim=len(spec.fields),
                                        hidden_dim=hidden_dim, n_layers=n_layers)
            t0 = time.process_time()
            out = solve_pde(spec, model, epochs=epochs, n_collocation=n_collocation, seed=seed,
                            cache_derivatives=cache, **pinn_options)
            dt = time.process_time() - t0
            return dt, [p.detach().clone() for p in out["model"].parameters()], dict(ao._STATS)

        try:
            base: List[float] = []
            fast: List[float] = []
            for _ in range(max(1, repeats)):
                d0, w0, _ = run(False)
                d1, w1, stats = run(True)
                base.append(d0)
                fast.append(d1)
        finally:
            torch.set_num_threads(old_threads)
        diff = max(float((a - b).abs().max()) for a, b in zip(w0, w1))
        scale = max(float(a.abs().max()) for a in w0)
        return Benchmark(self.problem.name, epochs, max(1, repeats), min(base), min(fast), min(base) / min(fast),
                         diff, scale, stats["hits"], stats["hits"] + stats["misses"])

    def __repr__(self) -> str:
        return f"CompiledProblem({self.problem.name!r}, optimize={self.optimize!r})"


def compile(problem: Union[PhysicalProblem, Any, str], optimize: str = "physics", **preset_kwargs: Any) -> CompiledProblem:
    """Prepare a problem (a ``PhysicalProblem``, a ``ProblemSpec`` or a preset name) for faster solving.

    ``optimize="physics"`` turns on the physics-aware optimisations (derivative reuse); ``"none"`` is the
    uncompiled baseline with the same interface, which is what ``benchmark`` compares against.
    """
    if hasattr(problem, "parameters") and hasattr(problem, "forward") and not isinstance(problem, (PhysicalProblem, str)):
        raise TypeError("pp.compile takes a problem, a ProblemSpec or a preset name. Compiling a bare torch model is not "
                        "supported: the optimisations act on the PDE residual, which a model alone does not define.")
    return CompiledProblem(as_problem(problem, **preset_kwargs), optimize)

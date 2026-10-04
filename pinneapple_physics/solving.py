"""``pp.solve`` and ``pp.compare``: one entry point to solve a problem by any method and compare methods.

>>> import pinneapple as pp
>>> sol = pp.solve("burgers_1d", method="pinn", epochs=2000)          # a PINN
>>> exact = pp.solve("burgers_1d", method="analytic")                 # Cole-Hopf closed form
>>> print(pp.compare("burgers_1d", methods=["pinn"], reference="analytic", options={"pinn": {"epochs": 2000}}))

``problem`` may be a ``PhysicalProblem``, a PDE ``ProblemSpec`` or a preset name. Built-in methods:

* ``"reference"`` -- the classical solver named in the problem's ``reference_solver`` (a preset's ``solver_spec``).
  Its grid output is interpolated linearly in between grid points. If that solver does not produce a field on a
  grid over **every** coordinate of the problem, ``solve`` raises ``MethodNotAvailable`` instead of returning a
  partial answer (for example a transient solver that keeps only the last time step).
* ``"pinn"`` -- a physics-informed network from the model registry trained with ``solve_pde``.
* ``"analytic"`` -- a closed-form solution the library knows for this problem (``list_analytic()``), for example the
  Cole-Hopf solution of the ``burgers_1d`` preset. Raises ``MethodNotAvailable`` when none is registered.
* ``"exact"`` -- a closed-form or manufactured solution passed as ``fn=lambda X: ...``.

Add your own with ``@register_method("name")``; the function receives ``(problem, **options)`` and returns a
``Solution``. Every ``Solution`` has ``predict(X)`` with ``X`` of shape ``(N, len(problem.coords))`` returning
``(N, len(problem.fields))``, plus the wall time of the solve.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from . import metrics as _metrics
from .physical_problem import PhysicalProblem

__all__ = ["Solution", "Comparison", "MethodNotAvailable", "solve", "compare", "register_method", "list_methods",
           "register_analytic", "list_analytic", "as_problem"]


class MethodNotAvailable(RuntimeError):
    """The requested method cannot solve this problem (missing solver output, missing option, ...)."""


@dataclass
class Solution:
    problem: PhysicalProblem
    method: str
    predict_fn: Callable[[np.ndarray], np.ndarray]
    wall_time_s: float
    history: Dict[str, Any] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)
    model: Any = None
    grid: Optional[Dict[str, np.ndarray]] = None  # coordinate axes of a grid solution, in problem.coords order

    def predict(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != len(self.problem.coords):
            raise ValueError(f"x must have shape (N, {len(self.problem.coords)}) for coords {self.problem.coords}, "
                             f"got {x.shape}")
        y = np.asarray(self.predict_fn(x), dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        if y.shape != (x.shape[0], len(self.problem.fields)):
            raise RuntimeError(f"method '{self.method}' returned shape {y.shape}, expected "
                               f"({x.shape[0]}, {len(self.problem.fields)}) for fields {self.problem.fields}")
        return y

    def grid_points(self) -> Optional[np.ndarray]:
        """All grid points ``(N, dim)`` of a grid solution, else ``None``."""
        if not self.grid:
            return None
        axes = [self.grid[c] for c in self.problem.coords]
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.stack([m.ravel() for m in mesh], axis=1)

    def __repr__(self) -> str:
        return f"Solution(problem={self.problem.name!r}, method={self.method!r}, wall_time_s={self.wall_time_s:.3g})"


# --------------------------------------------------------------------------- problems
def as_problem(problem: Union[PhysicalProblem, Any, str], **preset_kwargs: Any) -> PhysicalProblem:
    """``PhysicalProblem`` from a ``PhysicalProblem``, a PDE ``ProblemSpec`` or a preset name."""
    if isinstance(problem, PhysicalProblem):
        if preset_kwargs:
            raise TypeError("preset keyword arguments only apply when problem is a preset name")
        return problem
    if isinstance(problem, str):
        return PhysicalProblem.from_preset(problem, **preset_kwargs)
    if hasattr(problem, "pde") and hasattr(problem, "conditions") and hasattr(problem, "coords"):
        return PhysicalProblem.from_pde_spec(problem)
    raise TypeError(f"cannot make a PhysicalProblem from {type(problem).__name__}; pass a PhysicalProblem, "
                    "a ProblemSpec or a preset name (pp.list_presets())")


# --------------------------------------------------------------------------- methods
_METHODS: Dict[str, Callable[..., Solution]] = {}


def register_method(name: str):
    """Decorator registering ``fn(problem, **options) -> Solution`` under ``name`` for ``solve``/``compare``."""
    def deco(fn):
        if name in _METHODS:
            raise ValueError(f"method '{name}' is already registered")
        _METHODS[name] = fn
        return fn
    return deco


def list_methods() -> List[str]:
    return sorted(_METHODS)


@register_method("exact")
def _exact(problem: PhysicalProblem, *, fn: Optional[Callable[[np.ndarray], np.ndarray]] = None, **_: Any) -> Solution:
    if fn is None:
        raise MethodNotAvailable("method 'exact' needs fn=callable(X) -> values of shape (N, n_fields)")
    return Solution(problem, "exact", fn, 0.0, info={"source": getattr(fn, "__qualname__", "callable")})


# Closed-form solutions: (name, matcher(problem) -> bool, factory(problem) -> fn(X) -> (N, F)).
_ANALYTIC: List[tuple] = []


def register_analytic(name: str, matcher: Callable[[PhysicalProblem], bool]):
    """Decorator: ``factory(problem) -> fn(X)`` is the exact solution for every problem where ``matcher`` is True."""
    def deco(factory):
        _ANALYTIC.append((name, matcher, factory))
        return factory
    return deco


def list_analytic() -> List[str]:
    return [n for n, _, _ in _ANALYTIC]


def _is_burgers_sine(p: PhysicalProblem) -> bool:
    ic = (p.reference_solver.get("params") or {}).get("ic_type")
    return (p.pde is not None and p.pde.kind == "burgers" and tuple(p.coords) == ("x", "t")
            and tuple(p.fields) == ("u",) and tuple(p.domain_bounds.get("x", ())) == (-1.0, 1.0)
            and ic == "neg_sin_pi_x" and "nu" in p.parameter_values())


@register_analytic("burgers_sine_cole_hopf", _is_burgers_sine)
def _burgers_exact(p: PhysicalProblem):
    from .closed_form.burgers import burgers_sine_exact
    nu = float(p.parameter_values()["nu"])
    return lambda X: burgers_sine_exact(X[:, 0], X[:, 1], nu)[:, None]


@register_method("analytic")
def _analytic(problem: PhysicalProblem, **_: Any) -> Solution:
    for name, matcher, factory in _ANALYTIC:
        if matcher(problem):
            return Solution(problem, "analytic", factory(problem), 0.0, info={"solution": name})
    raise MethodNotAvailable(f"no closed-form solution is registered for problem '{problem.name}' "
                             f"(known: {list_analytic()})")


@register_method("reference")
def _reference(problem: PhysicalProblem, *, solver_config: Optional[Mapping[str, Any]] = None, seed: int = 0,
               **_: Any) -> Solution:
    from scipy.interpolate import RegularGridInterpolator
    from pinneapple_simulation.numerical_solvers.problem_runner import _run_reference_solver

    if not problem.reference_solver:
        raise MethodNotAvailable(f"problem '{problem.name}' names no reference solver")
    spec = problem.to_pde_spec()
    t0 = time.perf_counter()
    ref = _run_reference_solver(spec, dict(solver_config) if solver_config else None, np.random.default_rng(seed))
    elapsed = time.perf_counter() - t0
    name = problem.reference_solver.get("name", "?")
    if ref is None:
        raise MethodNotAvailable(f"reference solver '{name}' returned nothing for problem '{problem.name}'")
    coords = {k: np.asarray(v, dtype=np.float64) for k, v in ref.get("coords", {}).items()}
    if set(coords) != set(problem.coords):
        raise MethodNotAvailable(f"reference solver '{name}' returned a grid over {tuple(coords)}, but the problem "
                                 f"has coordinates {problem.coords}; a partial grid would be compared at the wrong points")
    missing = [f for f in problem.fields if f not in ref]
    if missing:
        raise MethodNotAvailable(f"reference solver '{name}' did not return field(s) {missing} of {problem.fields}")
    ref_order = list(coords)                      # axis order of the arrays as returned
    shape = tuple(len(coords[c]) for c in ref_order)
    perm = [ref_order.index(c) for c in problem.coords]
    interps = []
    for f in problem.fields:
        arr = np.asarray(ref[f], dtype=np.float64)
        if arr.shape != shape:
            raise MethodNotAvailable(f"reference field '{f}' has shape {arr.shape}, grid is {shape}")
        arr = np.transpose(arr, perm)
        interps.append(RegularGridInterpolator([coords[c] for c in problem.coords], arr, method="linear",
                                               bounds_error=False, fill_value=None))

    def predict(x: np.ndarray) -> np.ndarray:
        return np.stack([ip(x) for ip in interps], axis=1)

    return Solution(problem, "reference", predict, elapsed, info={"solver": name},
                    grid={c: coords[c] for c in problem.coords})


@register_method("pinn")
def _pinn(problem: PhysicalProblem, *, architecture: str = "modified_mlp", hidden_dim: int = 64, n_layers: int = 4,
          epochs: int = 2000, lr: float = 1e-3, n_collocation: int = 2048, seed: int = 0, device: str = "cpu",
          weights=None, **_: Any) -> Solution:
    import torch
    import pinneapple_neural.architectures  # noqa: F401  (registers the model zoo)
    from pinneapple_neural.architectures.registry import ModelRegistry
    from . import solve_pde

    spec = problem.to_pde_spec()
    torch.manual_seed(seed)
    model = ModelRegistry.build(architecture, in_dim=len(problem.coords), out_dim=len(problem.fields),
                                hidden_dim=hidden_dim, n_layers=n_layers)
    t0 = time.perf_counter()
    out = solve_pde(spec, model, epochs=epochs, device=device, lr=lr, n_collocation=n_collocation, seed=seed,
                    weights=weights)
    elapsed = time.perf_counter() - t0
    trained = out["model"]
    trained.eval()

    def predict(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            p = next(trained.parameters())
            y = trained(torch.as_tensor(x, dtype=p.dtype, device=p.device))
            y = getattr(y, "y", y)
            return y.detach().cpu().numpy()

    return Solution(problem, "pinn", predict, elapsed, history=out.get("history", {}), model=trained,
                    info={"architecture": architecture, "epochs": epochs, "n_collocation": n_collocation, "seed": seed})


# --------------------------------------------------------------------------- solve
def solve(problem: Union[PhysicalProblem, Any, str], method: Union[str, Callable[..., Solution]] = "pinn", *,
          preset_kwargs: Optional[Mapping[str, Any]] = None, **options: Any) -> Solution:
    """Solve ``problem`` with ``method`` (a registered name or a callable ``(problem, **options) -> Solution``).

    Raises ``MethodNotAvailable`` when the method cannot handle the problem, and ``ValueError`` when
    ``problem.validate()`` finds inconsistencies (a broken problem is never solved silently).
    """
    prob = as_problem(problem, **dict(preset_kwargs or {}))
    issues = prob.validate()
    if issues:
        raise ValueError(f"problem '{prob.name}' is not consistent:\n  - " + "\n  - ".join(issues))
    if callable(method):
        fn, name = method, getattr(method, "__name__", "custom")
    else:
        if method not in _METHODS:
            raise KeyError(f"unknown method '{method}'; available: {list_methods()}")
        fn, name = _METHODS[method], method
    sol = fn(prob, **options)
    if not isinstance(sol, Solution):
        raise TypeError(f"method '{name}' returned {type(sol).__name__}, expected a Solution")
    return sol


# --------------------------------------------------------------------------- compare
@dataclass
class Comparison:
    problem: PhysicalProblem
    reference: str
    n_points: int
    rows: List[Dict[str, Any]]

    def best(self, metric: str = "relative_l2") -> Optional[str]:
        """Method with the lowest mean (over fields) value of ``metric`` among the methods that ran."""
        scored = [(np.nanmean(list(r["metrics"][metric].values())), r["method"]) for r in self.rows
                  if r["status"] == "ok" and metric in r["metrics"]]
        scored = [s for s in scored if np.isfinite(s[0])]
        return min(scored)[1] if scored else None

    def to_dict(self) -> Dict[str, Any]:
        return {"problem": self.problem.name, "fingerprint": self.problem.fingerprint(), "reference": self.reference,
                "n_points": self.n_points, "rows": self.rows}

    def __str__(self) -> str:
        fields = self.problem.fields
        head = f"{'method':<16}{'status':<10}{'time [s]':>10}  " + "  ".join(f"relL2({f})" for f in fields)
        lines = [f"Comparison on '{self.problem.name}' against '{self.reference}' at {self.n_points} points", head]
        for r in self.rows:
            if r["status"] == "ok":
                rel = r["metrics"]["relative_l2"]
                cells = "  ".join(f"{rel[f]:>{len('relL2()') + len(f)}.3e}" for f in fields)
                lines.append(f"{r['method']:<16}{'ok':<10}{r['wall_time_s']:>10.3g}  {cells}")
            else:
                lines.append(f"{r['method']:<16}{'failed':<10}{'':>10}  {r['error']}")
        return "\n".join(lines)


def compare(problem: Union[PhysicalProblem, Any, str], methods: Sequence[Union[str, Callable[..., Solution]]], *,
            reference: Union[str, Solution] = "reference", options: Optional[Mapping[str, Mapping[str, Any]]] = None,
            points: Optional[np.ndarray] = None, n_points: int = 4096, seed: int = 0,
            metric_names: Sequence[str] = ("relative_l2", "rmse", "max_abs"),
            preset_kwargs: Optional[Mapping[str, Any]] = None) -> Comparison:
    """Solve ``problem`` with each of ``methods`` and score them against ``reference`` at the same points.

    Points: ``points`` if given; else up to ``n_points`` points of the reference grid (when the reference is a grid
    solution); else ``n_points`` uniform random points in ``domain_bounds``. A method that fails is reported with
    its error and does not stop the others.
    """
    prob = as_problem(problem, **dict(preset_kwargs or {}))
    opts = {str(k): dict(v) for k, v in (options or {}).items()}
    ref_sol = reference if isinstance(reference, Solution) else solve(prob, reference, **opts.get(reference, {}))
    rng = np.random.default_rng(seed)
    if points is None:
        grid = ref_sol.grid_points()
        if grid is not None:
            points = grid if len(grid) <= n_points else grid[rng.choice(len(grid), n_points, replace=False)]
        else:
            if set(prob.domain_bounds) != set(prob.coords):
                raise ValueError("give points=...: the problem has no box bounds for every coordinate and the "
                                 "reference is not a grid solution")
            lo = np.array([prob.domain_bounds[c][0] for c in prob.coords])
            hi = np.array([prob.domain_bounds[c][1] for c in prob.coords])
            points = lo + (hi - lo) * rng.random((n_points, len(prob.coords)))
    points = np.asarray(points, dtype=np.float64)
    truth = ref_sol.predict(points)
    rows: List[Dict[str, Any]] = []
    for m in methods:
        name = m if isinstance(m, str) else getattr(m, "__name__", "custom")
        try:
            sol = solve(prob, m, **opts.get(name, {}))
            pred = sol.predict(points)
            rows.append({"method": name, "status": "ok", "wall_time_s": sol.wall_time_s,
                         "metrics": _metrics.summary(pred, truth, prob.fields, metric_names), "info": sol.info})
        except Exception as exc:  # noqa: BLE001 - one failing method must not hide the others
            rows.append({"method": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    ref_name = reference if isinstance(reference, str) else reference.method
    return Comparison(prob, ref_name, len(points), rows)

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

* ``"fem"`` -- a classical P1 finite-element solve of steady ``poisson`` / ``laplace`` problems on a box with
  Dirichlet conditions (1D, 2D or 3D). The source ``f`` of ``laplace(u) = f`` comes from ``ctx={"source_fn": fn}``,
  the same ``ctx`` the PINN reads, so both backends solve the same problem through the same call.
* ``"external"`` -- any code that returns a field on a grid: ``runner=callable(problem) -> {"coords": {...}, field: array}``.

Every method has a ``kind`` (``classical``, ``neural``, ``analytic``, ``external``, ``reference`` or ``custom``) and every
``Solution`` has ``metadata()``, one dictionary with the same keys for all of them.

Add your own with ``@register_method("name", kind="classical")``; the function receives ``(problem, **options)`` and returns a
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
           "register_analytic", "list_analytic", "as_problem", "list_kinds"]


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
    kind: str = "custom"  # classical | neural | analytic | external | reference | custom

    def metadata(self) -> Dict[str, Any]:
        """The same keys for every backend: what was solved, by which method, how long it took, and the
        backend's own details under ``info``."""
        return {"problem": self.problem.name, "fingerprint": self.problem.fingerprint(), "method": self.method,
                "kind": self.kind, "wall_time_s": self.wall_time_s, "fields": list(self.problem.fields),
                "coords": list(self.problem.coords), "on_grid": self.grid is not None, "info": dict(self.info)}

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
        return (f"Solution(problem={self.problem.name!r}, method={self.method!r}, kind={self.kind!r}, "
                f"wall_time_s={self.wall_time_s:.3g})")


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
_KINDS: Dict[str, str] = {}


def register_method(name: str, kind: str = "custom"):
    """Decorator registering ``fn(problem, **options) -> Solution`` under ``name`` for ``solve``/``compare``.
    ``kind`` labels the family of the method (``classical``, ``neural``, ...) in every Solution it returns."""
    def deco(fn):
        if name in _METHODS:
            raise ValueError(f"method '{name}' is already registered")
        _METHODS[name] = fn
        _KINDS[name] = kind
        return fn
    return deco


def list_methods(kind: Optional[str] = None) -> List[str]:
    """Registered method names, optionally only those of one ``kind``."""
    return sorted(n for n in _METHODS if kind is None or _KINDS[n] == kind)


def list_kinds() -> Dict[str, List[str]]:
    """Methods grouped by kind."""
    out: Dict[str, List[str]] = {}
    for n in sorted(_METHODS):
        out.setdefault(_KINDS[n], []).append(n)
    return out


@register_method("exact", kind="analytic")
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


@register_method("analytic", kind="analytic")
def _analytic(problem: PhysicalProblem, **_: Any) -> Solution:
    for name, matcher, factory in _ANALYTIC:
        if matcher(problem):
            return Solution(problem, "analytic", factory(problem), 0.0, info={"solution": name})
    raise MethodNotAvailable(f"no closed-form solution is registered for problem '{problem.name}' "
                             f"(known: {list_analytic()})")


@register_method("reference", kind="classical")
def _reference(problem: PhysicalProblem, *, solver_config: Optional[Mapping[str, Any]] = None, seed: int = 0,
               **_: Any) -> Solution:
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
    return _grid_solution(problem, ref, name, "reference", elapsed, {"solver": name}, "classical")


def _grid_solution(problem: PhysicalProblem, ref: Mapping[str, Any], solver_name: str, method: str, elapsed: float,
                   info: Dict[str, Any], kind: str) -> Solution:
    """``Solution`` from a dict ``{"coords": {name: axis}, field: array}`` of a solver that returns grid fields."""
    from scipy.interpolate import RegularGridInterpolator

    name = solver_name
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

    return Solution(problem, method, predict, elapsed, info=info, grid={c: coords[c] for c in problem.coords}, kind=kind)




@register_method("pinn", kind="neural")
def _pinn(problem: PhysicalProblem, *, architecture: str = "modified_mlp", hidden_dim: int = 64, n_layers: int = 4,
          epochs: int = 2000, lr: float = 1e-3, n_collocation: int = 2048, seed: int = 0, device: str = "cpu",
          weights=None, ctx: Optional[Mapping[str, Any]] = None, **_: Any) -> Solution:
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
                    weights=weights, ctx=dict(ctx) if ctx else None)
    elapsed = time.perf_counter() - t0
    trained = out["model"]
    trained.eval()

    def predict(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            p = next(trained.parameters())
            y = trained(torch.as_tensor(x, dtype=p.dtype, device=p.device))
            y = getattr(y, "y", y)
            return y.detach().cpu().numpy()

    n_par = int(sum(p.numel() for p in trained.parameters()))
    return Solution(problem, "pinn", predict, elapsed, history=out.get("history", {}), model=trained,
                    info={"backend": "pinneapple.pinn", "architecture": architecture, "epochs": epochs,
                          "n_collocation": n_collocation, "seed": seed, "n_parameters": n_par}, kind="neural")


@register_method("fem", kind="classical")
def _fem(problem: PhysicalProblem, *, n: Union[int, Sequence[int], None] = None, ctx: Optional[Mapping[str, Any]] = None,
         max_dofs: int = 6000, **_: Any) -> Solution:
    """P1 finite elements for ``laplace(u) = f`` on a box with Dirichlet conditions (``poisson`` / ``laplace``).

    ``n`` is the number of cells per axis (default 32 in 1D/2D, 12 in 3D). Dense linear algebra: refuses more than
    ``max_dofs`` unknowns. Conditions must be ``dirichlet`` with callable selectors (``on="x_min"``, ``("x", "max")``,
    a function of the points, ...); ``tag`` selectors name regions of a real mesh and are refused, as in the PINN.
    """
    import torch
    from pinneapple_core import Mesh
    from pinneapple_core.fem import solve_poisson

    if problem.pde is None or problem.pde.kind not in ("poisson", "laplace"):
        raise MethodNotAvailable(f"method 'fem' handles steady poisson/laplace problems, not "
                                 f"{getattr(problem.pde, 'kind', None)!r}")
    if len(problem.fields) != 1:
        raise MethodNotAvailable("method 'fem' solves one scalar field")
    d = len(problem.coords)
    if d not in (1, 2, 3) or set(problem.domain_bounds) != set(problem.coords):
        raise MethodNotAvailable("method 'fem' needs a box: domain_bounds for every coordinate, in 1, 2 or 3 dimensions")
    ctx_d = dict(ctx or {})
    lo = np.array([problem.domain_bounds[c][0] for c in problem.coords], dtype=np.float64)
    hi = np.array([problem.domain_bounds[c][1] for c in problem.coords], dtype=np.float64)
    cells = np.full(d, 12 if d == 3 else 32) if n is None else np.broadcast_to(np.asarray(n, dtype=int), (d,)).copy()
    if int(np.prod(cells + 1)) > max_dofs:
        raise MethodNotAvailable(f"method 'fem' uses dense matrices: {int(np.prod(cells + 1))} unknowns exceed "
                                 f"max_dofs={max_dofs}; lower n or raise max_dofs")
    dirichlet = [c for c in problem.conditions if c.kind == "dirichlet"]
    others = [c.name for c in problem.conditions if c.kind != "dirichlet"]
    if others:
        raise MethodNotAvailable(f"method 'fem' supports Dirichlet conditions only; got {others}")
    bad = [c.name for c in dirichlet if c.selector_type not in ("callable", "all")]
    if bad:
        raise MethodNotAvailable(f"conditions {bad} use tag selectors, which need a real mesh; select the faces "
                                 f"with on='x_min', ('x', 'max'), or a function of the points")
    if not dirichlet:
        raise MethodNotAvailable("method 'fem' needs at least one Dirichlet condition")

    mesh = Mesh.structured(lo, hi, tuple(int(c) for c in cells))
    pts = mesh.points
    vals = np.full(mesh.n_points, np.nan)
    for c in dirichlet:
        mask = np.ones(len(pts), dtype=bool) if c.selector_type == "all" else np.asarray(c.selector(pts, ctx_d), dtype=bool)
        if mask.any():
            v = np.asarray(c.value_fn(pts[mask], ctx_d), dtype=np.float64)
            vals[mask] = v.reshape(v.shape[0], -1)[:, 0]
    nodes = np.flatnonzero(~np.isnan(vals))
    if nodes.size == 0:
        raise MethodNotAvailable("no mesh node lies on any Dirichlet selector; check the selectors and domain_bounds")

    f_fn = ctx_d.get("source_fn") or ctx_d.get("f_fn")
    f = np.zeros(mesh.n_points) if f_fn is None else np.asarray(f_fn(pts, ctx_d), dtype=np.float64).reshape(-1)
    t0 = time.perf_counter()
    # the problem convention is laplace(u) = f, the FEM form is -laplace(u) = -f
    u = solve_poisson(torch.as_tensor(pts), mesh.cells, torch.as_tensor(-f), nodes,
                      torch.as_tensor(vals[nodes])).numpy()
    elapsed = time.perf_counter() - t0
    axes = [np.linspace(lo[i], hi[i], cells[i] + 1) for i in range(d)]

    def predict(x: np.ndarray) -> np.ndarray:
        return mesh.interpolate(u, x, fill="nearest")

    return Solution(problem, "fem", predict, elapsed, kind="classical",
                    grid={c: axes[i] for i, c in enumerate(problem.coords)},
                    info={"backend": "pinneapple_core.fem", "discretization": "P1 finite elements",
                          "n_cells_per_axis": [int(c) for c in cells], "n_dof": int(mesh.n_points),
                          "n_dirichlet_nodes": int(nodes.size), "source": "ctx" if f_fn else "none"})


@register_method("external", kind="external")
def _external(problem: PhysicalProblem, *, runner: Optional[Callable[[PhysicalProblem], Mapping[str, Any]]] = None,
              name: str = "external", **_: Any) -> Solution:
    """Any code that solves the problem and returns grid fields: ``runner(problem) -> {"coords": {coord: axis},
    field: array}``, arrays indexed in the order of ``coords``. Use it to put OpenFOAM, FEniCS, a legacy code or a
    script behind the same ``solve``/``compare`` calls."""
    if runner is None:
        raise MethodNotAvailable("method 'external' needs runner=callable(problem) -> {'coords': {...}, field: array}")
    t0 = time.perf_counter()
    ref = runner(problem)
    elapsed = time.perf_counter() - t0
    if ref is None:
        raise MethodNotAvailable(f"external solver '{name}' returned nothing for problem '{problem.name}'")
    return _grid_solution(problem, ref, name, "external", elapsed, {"backend": name}, "external")


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
    if sol.kind == "custom":
        sol.kind = _KINDS.get(name, "custom")
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
        head = f"{'method':<16}{'kind':<11}{'status':<10}{'time [s]':>10}  " + "  ".join(f"relL2({f})" for f in fields)
        lines = [f"Comparison on '{self.problem.name}' against '{self.reference}' at {self.n_points} points", head]
        for r in self.rows:
            if r["status"] == "ok":
                rel = r["metrics"]["relative_l2"]
                cells = "  ".join(f"{rel[f]:>{len('relL2()') + len(f)}.3e}" for f in fields)
                lines.append(f"{r['method']:<16}{r['kind']:<11}{'ok':<10}{r['wall_time_s']:>10.3g}  {cells}")
            else:
                lines.append(f"{r['method']:<16}{'':<11}{'failed':<10}{'':>10}  {r['error']}")
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
            rows.append({"method": name, "status": "ok", "kind": sol.kind, "wall_time_s": sol.wall_time_s,
                         "metrics": _metrics.summary(pred, truth, prob.fields, metric_names), "info": sol.info})
        except Exception as exc:  # noqa: BLE001 - one failing method must not hide the others
            rows.append({"method": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    ref_name = reference if isinstance(reference, str) else reference.method
    return Comparison(prob, ref_name, len(points), rows)

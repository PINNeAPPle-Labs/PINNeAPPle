"""``PhysicsOptimizer(parameters, objective, constraints)``: one interface for several optimizer families.

>>> import torch
>>> from pinneapple_core.optim import PhysicsOptimizer, ineq
>>> opt = PhysicsOptimizer({"x": (-3, 3), "y": (-3, 3)},
...                        objective=lambda v: (v[0] - 2) ** 2 + (v[1] - 1) ** 2,
...                        constraints=[ineq(lambda v: v[0] + v[1] - 2)])       # x + y <= 2
>>> r = opt.minimize("slsqp")
>>> [round(float(t), 3) for t in r.x]
[1.5, 0.5]

The problem (parameters with bounds, an objective, inequality ``g(x) <= 0`` and equality ``h(x) = 0``
constraints) is written once. ``minimize(method)`` picks the family:

========================  =================================================================  ==========================
method                    what it is                                                          needs
========================  =================================================================  ==========================
``slsqp``                 sequential quadratic programming, autograd gradients               a torch objective
``adjoint``               the same SQP with a gradient you supply (adjoint, implicit solve)   ``grad=``
``penalty``               projected Adam on the objective plus a growing quadratic penalty    a torch objective
``augmented_lagrangian``  projected Adam with multiplier updates (exact constraints)           a torch objective
``differential_evolution`` population search, no gradients                                      nothing
``bayesian``              Gaussian-process expected improvement, for expensive objectives     nothing
========================  =================================================================  ==========================

The objective and constraint functions receive the parameter vector (a float64 torch tensor, or a numpy array
with ``backend="numpy"``; with ``argument="dict"`` a dict of scalars) and return a scalar (constraints may return
a vector). Every method returns the same :class:`OptimizeResult`; ``compare`` runs several on the same problem.
Optimal control and MPC are a separate roadmap item.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["PhysicsOptimizer", "OptimizeResult", "Constraint", "ineq", "eq", "list_methods"]


@dataclass(frozen=True)
class Constraint:
    """``fn(x) <= 0`` (``kind="ineq"``) or ``fn(x) == 0`` (``kind="eq"``)."""
    fn: Callable
    kind: str = "ineq"
    name: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("ineq", "eq"):
            raise ValueError("kind must be 'ineq' or 'eq'")


def ineq(fn: Callable, name: str = "") -> Constraint:
    """Inequality constraint ``fn(x) <= 0``."""
    return Constraint(fn, "ineq", name)


def eq(fn: Callable, name: str = "") -> Constraint:
    """Equality constraint ``fn(x) == 0``."""
    return Constraint(fn, "eq", name)


@dataclass
class OptimizeResult:
    x: np.ndarray
    params: Dict[str, float]
    fun: float
    violation: float
    feasible: bool
    n_evals: int
    method: str
    family: str
    success: bool
    message: str
    wall_time_s: float
    history: List[float] = field(default_factory=list)

    def __str__(self) -> str:
        xs = ", ".join(f"{k}={v:.4g}" for k, v in self.params.items())
        return (f"{self.method:<22} [{self.family}] f={self.fun:.6g}  {xs}  violation={self.violation:.1e}  "
                f"{'feasible' if self.feasible else 'INFEASIBLE'}  evals={self.n_evals}  {self.wall_time_s:.2f}s")


_FAMILY = {"slsqp": "gradient (SQP)", "adjoint": "adjoint (supplied gradient)", "penalty": "gradient (penalty)",
           "augmented_lagrangian": "gradient (augmented Lagrangian)", "differential_evolution": "evolutionary",
           "bayesian": "bayesian"}


def list_methods() -> List[str]:
    return list(_FAMILY)


class PhysicsOptimizer:
    """Parameters, an objective and constraints, minimised by any of :func:`list_methods`.

    Args:
        parameters: ``{name: (lo, hi)}``, ``{name: (lo, hi, x0)}`` or ``{name: {"bounds": (lo, hi), "x0": v}}``.
        objective: ``f(x) -> scalar`` to minimise.
        constraints: :func:`ineq` / :func:`eq` objects (or ``(fn, "ineq"|"eq")`` tuples).
        backend: ``"torch"`` (objective is differentiable by autograd) or ``"numpy"`` (plain function; gradient
            methods then use finite differences).
        argument: ``"vector"`` or ``"dict"``, what the functions receive.
        tol: a point is feasible when every violation is at most ``tol``.
    """

    def __init__(self, parameters: Mapping[str, Any], objective: Callable, constraints: Sequence[Any] = (), *,
                 backend: str = "torch", argument: str = "vector", tol: float = 1e-3) -> None:
        if backend not in ("torch", "numpy"):
            raise ValueError("backend must be 'torch' or 'numpy'")
        if argument not in ("vector", "dict"):
            raise ValueError("argument must be 'vector' or 'dict'")
        if not parameters:
            raise ValueError("parameters must name at least one variable")
        self.names: List[str] = list(parameters)
        lo, hi, x0 = [], [], []
        for n, spec in parameters.items():
            if isinstance(spec, Mapping):
                b, start = spec["bounds"], spec.get("x0")
            else:
                b, start = tuple(spec[:2]), (spec[2] if len(spec) > 2 else None)
            if not b[0] < b[1]:
                raise ValueError(f"parameter {n!r}: lower bound must be below the upper bound, got {b}")
            lo.append(float(b[0])); hi.append(float(b[1]))
            x0.append(float(start) if start is not None else 0.5 * (b[0] + b[1]))
        self.lo, self.hi, self.x0 = np.array(lo), np.array(hi), np.array(x0)
        if np.any(self.x0 < self.lo) or np.any(self.x0 > self.hi):
            raise ValueError("x0 must lie inside the bounds")
        self.objective = objective
        self.constraints: List[Constraint] = [c if isinstance(c, Constraint) else Constraint(*c) for c in constraints]
        self.backend, self.argument, self.tol = backend, argument, tol
        self._n_evals = 0

    # -- evaluation ----------------------------------------------------------
    def unpack(self, x: np.ndarray) -> Dict[str, float]:
        return {n: float(v) for n, v in zip(self.names, np.asarray(x, dtype=float))}

    def _call(self, fn: Callable, x, grad: bool):
        """Call a user function with the representation they asked for."""
        if self.backend == "numpy":
            arg = np.asarray(x, dtype=float)
            if self.argument == "dict":
                arg = dict(zip(self.names, arg))
            return fn(arg)
        import torch

        t = torch.as_tensor(np.asarray(x, dtype=float), dtype=torch.float64)
        if grad:
            t = t.clone().requires_grad_(True)
        arg = dict(zip(self.names, t.unbind())) if self.argument == "dict" else t
        return fn(arg), t

    def _value(self, fn: Callable, x) -> np.ndarray:
        if self.backend == "numpy":
            return np.atleast_1d(np.asarray(self._call(fn, x, False), dtype=float))
        import torch

        with torch.no_grad():
            out, _ = self._call(fn, x, False)
        return np.atleast_1d(np.asarray(torch.as_tensor(out).detach().numpy(), dtype=float))

    def _value_and_grad(self, fn: Callable, x) -> Tuple[np.ndarray, np.ndarray]:
        """Value (vector) and Jacobian (m, n) of ``fn`` at ``x``."""
        if self.backend == "numpy":
            from scipy.optimize._numdiff import approx_derivative

            f = lambda z: np.atleast_1d(np.asarray(self._call(fn, z, False), dtype=float))
            val = f(x)
            return val, np.atleast_2d(approx_derivative(f, np.asarray(x, dtype=float), method="2-point",
                                                       bounds=(self.lo, self.hi)))
        import torch

        out, t = self._call(fn, x, True)
        out = torch.as_tensor(out, dtype=torch.float64).reshape(-1)
        rows = []
        for i in range(out.numel()):
            (g,) = torch.autograd.grad(out[i], t, retain_graph=True, allow_unused=True)
            rows.append(np.zeros(len(self.names)) if g is None else g.detach().numpy())
        return out.detach().numpy(), np.array(rows)

    def f(self, x) -> float:
        """Objective value at ``x`` (counted as an evaluation)."""
        self._n_evals += 1
        return float(self._value(self.objective, x)[0])

    def violation(self, x) -> float:
        """Largest constraint violation at ``x`` (0 when feasible; bounds are always respected by the methods)."""
        worst = 0.0
        for c in self.constraints:
            v = self._value(c.fn, x)
            worst = max(worst, float(np.max(np.maximum(v, 0.0) if c.kind == "ineq" else np.abs(v))))
        return worst

    def _penalised(self, x, mu: float) -> float:
        total = self.f(x)
        for c in self.constraints:
            v = self._value(c.fn, x)
            total += mu * float(np.sum(np.maximum(v, 0.0) ** 2 if c.kind == "ineq" else v ** 2))
        return total

    # -- driver --------------------------------------------------------------
    def minimize(self, method: str = "slsqp", *, grad: Optional[Callable] = None, x0: Optional[Sequence[float]] = None,
                 seed: int = 0, **options: Any) -> OptimizeResult:
        """Run ``method`` (see the module table). Options per method: ``max_iter`` (all), ``lr`` / ``rounds`` /
        ``mu`` (penalty and augmented Lagrangian), ``popsize`` (evolutionary), ``n_initial`` / ``n_candidates``
        (bayesian). ``grad(x_numpy) -> gradient`` is required by ``adjoint``."""
        if method not in _FAMILY:
            raise ValueError(f"unknown method {method!r}; choose from {list_methods()}")
        start = np.array(self.x0 if x0 is None else x0, dtype=float)
        self._n_evals = 0
        t0 = time.perf_counter()
        runner = getattr(self, f"_run_{method}")
        if method == "adjoint":
            if grad is None:
                raise ValueError("method 'adjoint' needs grad=callable(x) -> gradient of the objective")
            x, ok, msg, hist = runner(start, grad, **options)
        elif method in ("differential_evolution", "bayesian"):
            x, ok, msg, hist = runner(start, seed=seed, **options)
        else:
            x, ok, msg, hist = runner(start, **options)
        x = np.clip(x, self.lo, self.hi)
        viol = self.violation(x)
        return OptimizeResult(x, self.unpack(x), self.f(x), viol, viol <= self.tol, self._n_evals, method,
                              _FAMILY[method], bool(ok), msg, time.perf_counter() - t0, hist)

    def compare(self, methods: Optional[Sequence[str]] = None, **options: Any) -> Dict[str, OptimizeResult]:
        """Run several methods on this problem. ``adjoint`` is skipped unless ``grad=`` is given."""
        out: Dict[str, OptimizeResult] = {}
        for m in methods or [m for m in list_methods() if m != "adjoint" or options.get("grad")]:
            out[m] = self.minimize(m, **options)
        return out

    # -- gradient family: SQP -------------------------------------------------
    def _sqp(self, start, max_iter, obj_grad=None):
        from scipy.optimize import minimize

        hist: List[float] = []

        def fun(z):
            self._n_evals += 1
            v, j = self._value_and_grad(self.objective, z)
            hist.append(float(v[0]))
            return float(v[0]), j[0] if obj_grad is None else np.asarray(obj_grad(z), dtype=float)

        cons = []
        for c in self.constraints:
            sign = -1.0 if c.kind == "ineq" else 1.0                  # scipy wants fun >= 0 for 'ineq'
            cons.append({"type": c.kind,
                         "fun": (lambda z, c=c, s=sign: s * self._value(c.fn, z)),
                         "jac": (lambda z, c=c, s=sign: s * self._value_and_grad(c.fn, z)[1])})
        res = minimize(fun, start, jac=True, method="SLSQP", bounds=list(zip(self.lo, self.hi)), constraints=cons,
                       options={"maxiter": max_iter, "ftol": 1e-10})
        return res.x, res.success, str(res.message), hist

    def _run_slsqp(self, start, max_iter: int = 200):
        return self._sqp(start, max_iter)

    def _run_adjoint(self, start, grad, max_iter: int = 200):
        return self._sqp(start, max_iter, obj_grad=grad)

    # -- gradient family: projected Adam with penalties / multipliers ------------
    def _adam_rounds(self, start, rounds, max_iter, lr, mu0, mu_growth, multipliers, lr_decay=0.5):
        import torch

        x = torch.as_tensor(start, dtype=torch.float64).clone()
        lo, hi = torch.as_tensor(self.lo), torch.as_tensor(self.hi)
        span = hi - lo
        mu = mu0
        lam = [torch.zeros(max(int(np.size(self._value(c.fn, start))), 1), dtype=torch.float64) for c in self.constraints]
        hist: List[float] = []
        inner = max(max_iter // rounds, 1)
        for _ in range(rounds):
            p = x.clone().requires_grad_(True)
            opt = torch.optim.Adam([p], lr=lr)
            lr *= lr_decay
            for it in range(inner):
                opt.zero_grad()
                arg = dict(zip(self.names, p.unbind())) if self.argument == "dict" else p
                self._n_evals += 1
                loss = torch.as_tensor(self.objective(arg), dtype=torch.float64).reshape(())
                for c, l in zip(self.constraints, lam):
                    v = torch.as_tensor(c.fn(arg), dtype=torch.float64).reshape(-1)
                    if c.kind == "ineq":
                        loss = loss + (torch.relu(l / mu + v) ** 2 - (l / mu) ** 2).sum() * (mu / 2.0) if multipliers \
                            else loss + mu * (torch.relu(v) ** 2).sum()
                    else:
                        loss = loss + ((l * v).sum() + mu / 2.0 * (v ** 2).sum() if multipliers else mu * (v ** 2).sum())
                loss.backward()
                opt.step()
                with torch.no_grad():
                    p.copy_(torch.minimum(torch.maximum(p, lo), hi))     # projection onto the bounds
                hist.append(float(loss.detach()))
            x = p.detach()
            if multipliers:                                              # first-order multiplier update
                for i, c in enumerate(self.constraints):
                    v = torch.as_tensor(self._value(c.fn, x.numpy()), dtype=torch.float64)
                    lam[i] = torch.relu(lam[i] + mu * v) if c.kind == "ineq" else lam[i] + mu * v
            mu *= mu_growth
        return x.numpy(), True, f"{rounds} rounds of {inner} Adam steps", hist

    def _run_penalty(self, start, max_iter: int = 3000, rounds: int = 6, lr: Optional[float] = None, mu: float = 10.0):
        if self.backend != "torch":
            raise ValueError("method 'penalty' needs backend='torch'")
        return self._adam_rounds(start, rounds, max_iter, lr or 0.05 * float(np.mean(self.hi - self.lo)), mu, 4.0, False)

    def _run_augmented_lagrangian(self, start, max_iter: int = 2000, rounds: int = 8, lr: Optional[float] = None,
                                  mu: float = 10.0):
        if self.backend != "torch":
            raise ValueError("method 'augmented_lagrangian' needs backend='torch'")
        return self._adam_rounds(start, rounds, max_iter, lr or 0.05 * float(np.mean(self.hi - self.lo)), mu, 1.5, True, 0.7)

    # -- derivative-free families --------------------------------------------------
    def _run_differential_evolution(self, start, seed: int = 0, max_iter: int = 60, popsize: int = 15):
        from scipy.optimize import NonlinearConstraint, differential_evolution

        cons = []
        for c in self.constraints:
            lo_b, hi_b = (-np.inf, 0.0) if c.kind == "ineq" else (-0.5 * self.tol, 0.5 * self.tol)    # equality as a thin band
            cons.append(NonlinearConstraint(lambda z, c=c: self._value(c.fn, z), lo_b, hi_b))
        hist: List[float] = []
        res = differential_evolution(lambda z: (hist.append(self.f(z)) or hist[-1]), list(zip(self.lo, self.hi)),
                                     seed=seed, maxiter=max_iter, popsize=popsize, constraints=cons, polish=False,
                                     tol=1e-8, x0=start, updating="immediate")
        return res.x, res.success, str(res.message), hist

    def _run_bayesian(self, start, seed: int = 0, max_iter: int = 60, n_initial: int = 10, n_candidates: int = 4000,
                      xi: float = 0.0):
        """Constrained expected improvement: a Gaussian process for the objective and one per constraint component;
        the acquisition is EI over the best feasible value times the probability that every constraint holds."""
        from math import pi, sqrt
        from scipy.special import ndtr
        from pinneapple_design.design_optimizer.optimizer import _NumpyGP

        rng = np.random.default_rng(seed)
        d = len(self.names)
        span = self.hi - self.lo
        to_x = lambda z: self.lo + z * span
        Z: List[np.ndarray] = [(start - self.lo) / span]
        Z += [rng.random(d) for _ in range(max(n_initial - 1, 0))]
        F: List[float] = []
        G: List[np.ndarray] = []            # constraint components, signed so that <= 0 means satisfied
        hist: List[float] = []

        def evaluate(z):
            x = to_x(z)
            F.append(self.f(x))
            parts = []
            for c in self.constraints:
                v = self._value(c.fn, x)
                parts.append(v if c.kind == "ineq" else np.concatenate([v, -v]))
            G.append(np.concatenate(parts) if parts else np.zeros(0))
            feas = [f for f, g in zip(F, G) if g.size == 0 or np.all(g <= self.tol)]
            hist.append(min(feas) if feas else float("nan"))

        for z in Z:
            evaluate(z)
        for _ in range(max(max_iter - len(Z), 0)):
            Zn = np.array(Z)
            Fa, Ga = np.array(F), np.array(G)
            feasible = np.all(Ga <= self.tol, axis=1) if Ga.shape[1] else np.ones(len(F), dtype=bool)
            cand = rng.random((n_candidates, d))
            anchor = Zn[int(np.argmin(np.where(feasible, Fa, np.inf)))] if feasible.any() else Zn[int(np.argmin(Ga.max(axis=1)))]
            cand = np.vstack([cand, np.clip(anchor + 0.05 * rng.standard_normal((n_candidates // 2, d)), 0.0, 1.0),
                              np.clip(anchor + 0.01 * rng.standard_normal((n_candidates // 4, d)), 0.0, 1.0)])
            p_feas = np.ones(len(cand))
            for j in range(Ga.shape[1]):
                col = Ga[:, j]
                sd = col.std() + 1e-12
                gp = _NumpyGP(noise=1e-6)
                gp.fit(Zn, (col - col.mean()) / sd)
                m, s_ = gp.predict(cand)
                p_feas *= ndtr((self.tol - (m * sd + col.mean())) / np.maximum(s_ * sd, 1e-12))
            if feasible.any():
                mu_f, sd_f = Fa.mean(), Fa.std() + 1e-12
                gp = _NumpyGP(noise=1e-6)
                gp.fit(Zn, (Fa - mu_f) / sd_f)
                m, s_ = gp.predict(cand)
                s_ = np.maximum(s_, 1e-12)
                best = (Fa[feasible].min() - mu_f) / sd_f
                u = (best - m - xi) / s_
                ei = (best - m - xi) * ndtr(u) + s_ * np.exp(-0.5 * u ** 2) / sqrt(2 * pi)
                score = ei * p_feas
            else:
                score = p_feas                     # nothing feasible yet: look for feasibility first
            z_new = cand[int(np.argmax(score))]
            Z.append(z_new)
            evaluate(z_new)
        Ga = np.array(G)
        feasible = np.all(Ga <= self.tol, axis=1) if Ga.shape[1] else np.ones(len(F), dtype=bool)
        pick = int(np.argmin(np.where(feasible, np.array(F), np.inf))) if feasible.any() else int(np.argmin(Ga.max(axis=1)))
        return to_x(Z[pick]), bool(feasible.any()), "GP constrained expected improvement", hist

    def __repr__(self) -> str:
        return f"PhysicsOptimizer(parameters={self.names}, constraints={len(self.constraints)}, backend={self.backend!r})"

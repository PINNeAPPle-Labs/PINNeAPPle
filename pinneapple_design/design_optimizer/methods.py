"""Optimisation methods on a :class:`~.study.DesignProblem`, all under the same evaluation budget.

``run(problem, method, budget=..., seed=...)`` returns an :class:`OptimizationResult` that holds every evaluated
design (:class:`~.study.Evaluations`), so the Pareto front, the convergence history and the cost are computed the
same way for every method.

=================  ==========================================================================================
method             what it does
=================  ==========================================================================================
``"random"``       Latin hypercube of the whole budget (the baseline every method has to beat)
``"nsga2"``        NSGA-II: :class:`~.optimizer.EvolutionaryDesignOptimizer` (non-dominated sorting, crowding,
                   Deb's constrained domination), uniform crossover, per-variable mutation scaled to the bounds
``"pymoo-nsga2"``  NSGA-II from pymoo (optional dependency; ``status="skipped"`` when it is not installed)
``"scipy-de"``     SciPy differential evolution on augmented Tchebycheff scalarisations, one per weight vector
``"scipy-slsqp"``  SciPy SLSQP with finite-difference gradients, weighted sums, constraints as inequalities
``"bayesian"``     Gaussian-process Bayesian optimisation, ParEGO scalarisation
                   (:class:`~.optimizer.BayesianDesignOptimizer`)
``"gradient"``     Adam through a differentiable evaluator (autograd: neural surrogates, PINNs, neural operators,
                   differentiable physics), weighted sums with a growing quadratic constraint penalty
=================  ==========================================================================================

Budget accounting: one evaluation is one design passed to the evaluator. A finite-difference gradient costs
``d + 1`` evaluations; an autograd step costs one (forward and backward pass, reported in wall time). Scalarising
methods spend a Latin hypercube of ``max(2 d, budget / 10)`` designs first to normalise the objectives; it counts.

Integer variables: population methods and the random baseline round them. Gradient-based methods relax them when the
evaluator accepts fractional values (``Evaluator.relaxable``), mark those iterates ``relaxed`` (never Pareto
members) and evaluate the rounded end point as a realisable design.
"""
from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .optimizer import BayesianDesignOptimizer, DesignOptimizerConfig, EvolutionaryDesignOptimizer
from .study import VALID, DesignProblem, Evaluations, feasibility_report, hypervolume

__all__ = ["METHODS", "OptimizationResult", "run", "available_methods", "normalise_front", "weight_vectors"]

BIG = 1e30


class _Spent(Exception):
    """The evaluation budget is spent."""


class _Abort(Exception):
    """The progress callback raised (e.g. a job was cancelled): stop the run and re-raise the original."""

    def __init__(self, original: BaseException):
        super().__init__(str(original))
        self.original = original


@dataclass
class OptimizationResult:
    """Every evaluation of one run of one method, with its cost."""
    problem: DesignProblem
    method: str
    seed: int
    budget: int
    evaluations: Evaluations | None
    wall_s: float
    status: str = "ok"                       # ok | failed | skipped
    message: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def n_evals(self) -> int:
        return 0 if self.evaluations is None else len(self.evaluations)

    def pareto(self) -> np.ndarray:
        return np.array([], int) if self.evaluations is None else self.evaluations.pareto()

    def best(self, objective: int = 0) -> int | None:
        """Index of the best feasible, realisable design on one objective."""
        ev = self.evaluations
        if ev is None:
            return None
        ok = np.where(ev.feasible & ~ev.relaxed)[0]
        return None if not len(ok) else int(ok[np.argmin(ev.F[ok, objective])])

    def hv_curve(self, ideal: np.ndarray, nadir: np.ndarray, n_points: int = 25,
                 ref: float = 1.1) -> list[tuple[int, float]]:
        """Hypervolume of the feasible realisable designs found after ``k`` evaluations, objectives normalised to
        ``[0, 1]`` between ``ideal`` and ``nadir``, reference point ``ref`` per axis."""
        ev = self.evaluations
        if ev is None or not len(ev):
            return []
        ks = np.unique(np.linspace(1, len(ev), min(n_points, len(ev))).astype(int))
        ok = ev.feasible & ~ev.relaxed
        Fn = normalise_front(ev.F, ideal, nadir)
        return [(int(k), hypervolume(Fn[:k][ok[:k]], np.full(Fn.shape[1], ref))) for k in ks]

    def summary(self) -> dict[str, Any]:
        ev = self.evaluations
        n_feas = 0 if ev is None else int((ev.feasible & ~ev.relaxed).sum())
        return {"method": self.method, "seed": self.seed, "budget": self.budget, "n_evals": self.n_evals,
                "n_feasible": n_feas, "n_pareto": int(len(self.pareto())), "wall_s": round(self.wall_s, 4),
                "status": self.status, "message": self.message, "notes": list(self.notes)}

    def to_dict(self, max_rows: int | None = None) -> dict[str, Any]:
        """JSON-ready: summary, Pareto designs with their feasibility report, and (up to ``max_rows``) all rows."""
        out = self.summary()
        out["problem"] = self.problem.to_dict()
        ev = self.evaluations
        if ev is None:
            return out
        p = self.pareto()
        out["pareto"] = ev.rows(self.problem.space, p)
        out["feasibility"] = feasibility_report(self.problem, ev, p)
        idx = range(len(ev)) if max_rows is None else np.unique(
            np.r_[p, np.linspace(0, len(ev) - 1, min(max_rows, len(ev))).astype(int)])
        out["evaluations"] = ev.rows(self.problem.space, idx)
        return out


def normalise_front(F: np.ndarray, ideal: np.ndarray, nadir: np.ndarray) -> np.ndarray:
    span = np.where(np.asarray(nadir) - np.asarray(ideal) > 0, np.asarray(nadir) - np.asarray(ideal), 1.0)
    return (np.asarray(F, dtype=float) - ideal) / span


def weight_vectors(n_obj: int, n: int, seed: int = 0) -> np.ndarray:
    """``n`` weight vectors on the simplex: evenly spaced for two objectives, seeded Dirichlet draws (plus the
    centre) above; one vector for a single objective."""
    if n_obj == 1:
        return np.ones((1, 1))
    if n_obj == 2:
        a = np.linspace(0.05, 0.95, n)
        return np.c_[a, 1 - a]
    W = np.random.default_rng(seed).dirichlet(np.ones(n_obj), size=n - 1)
    return np.vstack([np.full(n_obj, 1 / n_obj), W])


# --------------------------------------------------------------------------- budget bookkeeping
class _Run:
    def __init__(self, problem: DesignProblem, budget: int, progress: Callable[[int, int], None] | None = None):
        self.problem, self.budget, self.progress = problem, int(budget), progress
        self.ev: Evaluations | None = None

    @property
    def used(self) -> int:
        return 0 if self.ev is None else len(self.ev)

    @property
    def remaining(self) -> int:
        return self.budget - self.used

    def add(self, ev: Evaluations) -> Evaluations:
        self.ev = ev if self.ev is None else self.ev.extend(ev)
        if self.progress is not None:
            try:
                self.progress(self.used, self.budget)
            except Exception as e:  # noqa: BLE001 - re-raised by run() after the method unwinds
                raise _Abort(e) from e
        return ev

    def eval(self, X: np.ndarray, *, relaxed: bool = False, tag: str = "") -> Evaluations:
        """Evaluate at most the remaining budget's worth of rows of ``X``; raises :class:`_Spent` at zero."""
        X = np.atleast_2d(X)
        if self.remaining <= 0:
            raise _Spent
        X = X[: self.remaining]
        if relaxed:
            b = self.problem.space.bounds
            ev = self.problem.evaluate(np.clip(X, b[:, 0], b[:, 1]), repair=False, tag=tag)
            ev.relaxed[:] = _fractional(self.problem, ev.X)
        else:
            ev = self.problem.evaluate(X, tag=tag)
        return self.add(ev)


def _fractional(problem: DesignProblem, X: np.ndarray) -> np.ndarray:
    iv = problem.space.integer
    if not iv.any():
        return np.zeros(len(X), bool)
    return np.any(np.abs(X[:, iv] - np.round(X[:, iv])) > 1e-9, 1)


def _penalised(ev: Evaluations) -> tuple[np.ndarray, np.ndarray]:
    """Finite objectives and a penalty (0 when feasible) for survivor selection."""
    F = np.where(np.isfinite(ev.F), ev.F, BIG)
    pen = np.where(ev.feasible, 0.0, np.minimum(np.nan_to_num(ev.violation, posinf=BIG), BIG) + 1.0)
    return F, pen


def _normaliser(run: _Run, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Spend the initial Latin hypercube and return (ideal, nadir) of its feasible (else all finite) designs."""
    p = run.problem
    n0 = max(2 * p.space.dim, run.budget // 10)
    ev = run.eval(p.space.sample(n0, seed=seed), tag="initial sample")
    F = ev.F[np.all(np.isfinite(ev.F), 1)]
    Ff = ev.F[ev.feasible]
    F = Ff if len(Ff) >= 2 else F
    if not len(F):
        return np.zeros(p.n_obj), np.ones(p.n_obj)
    return F.min(0), F.max(0)


def _scalar(F: np.ndarray, w: np.ndarray, ideal: np.ndarray, nadir: np.ndarray, rho: float = 0.05) -> np.ndarray:
    """Augmented Tchebycheff scalarisation of normalised objectives."""
    Fn = normalise_front(F, ideal, nadir)
    return np.max(w * Fn, 1) + rho * np.sum(w * Fn, 1)


# --------------------------------------------------------------------------- methods
def _random(run: _Run, seed: int, **_):
    run.eval(run.problem.space.sample(run.budget, seed=seed), tag="lhs")


def _nsga2(run: _Run, seed: int, population: int | None = None, mutation: float = 0.08, **_):
    p = run.problem
    d = p.space.dim
    pop = population or int(np.clip(run.budget // 10, 12, 60))
    pop += pop % 2
    cfg = DesignOptimizerConfig(population_size=pop, mutation_std=0.0, crossover_rate=0.9)
    opt = EvolutionaryDesignOptimizer(cfg, seed=seed, multi_objective=True)
    unit = np.c_[np.zeros(d), np.ones(d)]
    rng = np.random.default_rng(seed + 1)
    gen = 0
    while run.remaining > 0:
        if gen == 0:
            U = p.space.to_unit(p.space.sample(pop, seed=seed))
        else:
            U = np.array(opt.ask(unit, pop))
            hit = rng.random(U.shape) < max(1.0 / d, 0.2)
            U = np.clip(U + hit * rng.normal(0, mutation, U.shape), 0, 1)
        ev = run.eval(p.space.from_unit(U), tag=f"generation {gen}")
        U = U[: len(ev)]
        F, pen = _penalised(ev)
        opt.tell(list(U), list(F[:, 0] + pen * BIG ** 0.5), obj_vecs=F.tolist() if p.n_obj > 1 else None,
                 penalties=pen.tolist())
        gen += 1


def _pymoo_nsga2(run: _Run, seed: int, population: int | None = None, **_):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize

    p = run.problem
    pop = population or int(np.clip(run.budget // 10, 12, 60))

    class _P(Problem):
        def __init__(self):
            super().__init__(n_var=p.space.dim, n_obj=p.n_obj, n_ieq_constr=len(p.constraints) + 1, xl=0.0, xu=1.0)

        def _evaluate(self, U, out, *args, **kwargs):
            n = len(U)
            F = np.full((n, p.n_obj), BIG)
            G = np.full((n, len(p.constraints) + 1), BIG)
            if run.remaining > 0:
                ev = run.eval(p.space.from_unit(U), tag="pymoo")
                k = len(ev)
                F[:k] = np.where(np.isfinite(ev.F), ev.F, BIG)
                G[:k, :-1] = np.minimum(ev.G, BIG)
                valid = np.asarray(ev.Y.get(VALID, np.ones(k)), dtype=bool)
                G[:k, -1] = np.where(valid, -1.0, 1.0)
            out["F"], out["G"] = F, G

    minimize(_P(), NSGA2(pop_size=pop), ("n_eval", run.budget), seed=seed, verbose=False)


def _scipy_de(run: _Run, seed: int, n_weights: int | None = None, **_):
    from scipy.optimize import differential_evolution

    p = run.problem
    ideal, nadir = _normaliser(run, seed)
    W = weight_vectors(p.n_obj, n_weights or (1 if p.n_obj == 1 else 5), seed)
    for j, w in enumerate(W):
        sub = run.remaining // (len(W) - j)
        stop = run.used + sub

        def f(U, w=w, stop=stop):
            U = np.atleast_2d(U.T)
            n = len(U)
            out = np.full(n, BIG)
            k = min(n, stop - run.used)
            if k <= 0:
                raise _Spent
            ev = run.eval(p.space.from_unit(U[:k]), tag=f"weights {np.round(w, 2).tolist()}")
            s = _scalar(np.where(np.isfinite(ev.F), ev.F, BIG), w, ideal, nadir)
            out[: len(ev)] = s + np.where(ev.feasible, 0.0, 10.0 + 100 * np.minimum(ev.violation, 1e6))
            return out

        try:
            differential_evolution(f, [(0.0, 1.0)] * p.space.dim, seed=seed + j, popsize=max(5, 40 // p.space.dim),
                                   maxiter=10_000, tol=0, polish=False, vectorized=True, updating="deferred")
        except _Spent:
            pass
        if run.remaining <= 0:
            break


def _slsqp(run: _Run, seed: int, n_weights: int | None = None, **_):
    from scipy.optimize import minimize

    p = run.problem
    relax = bool(p.space.integer.any() and p.evaluator.relaxable)
    ideal, nadir = _normaliser(run, seed)
    span = np.where(nadir - ideal > 0, nadir - ideal, 1.0)
    W = weight_vectors(p.n_obj, n_weights or (1 if p.n_obj == 1 else 5), seed)
    rng = np.random.default_rng(seed)
    cache: dict[bytes, Evaluations] = {}

    def at(u):
        key = np.round(u, 12).tobytes()
        if key not in cache:
            cache[key] = run.eval(p.space.from_unit(u), relaxed=relax, tag="slsqp")
        return cache[key]

    def fobj(u, w):
        ev = at(u)
        return float(np.sum(w * (np.minimum(ev.F[0], BIG) - ideal) / span))

    def fcon(u):
        ev = at(u)
        return np.array([float(c.slack(ev.Y[c.name][0])) for c in p.constraints]) if p.constraints else np.zeros(1)

    for j, w in enumerate(W):
        if run.remaining <= p.space.dim + 2:
            break
        sub_stop = run.used + run.remaining // (len(W) - j)
        ok = np.where(run.ev.feasible & ~run.ev.relaxed)[0] if run.ev is not None else []
        if len(ok):
            s = np.sum(w * (run.ev.F[ok] - ideal) / span, 1)
            u0 = p.space.to_unit(run.ev.X[ok[int(np.argmin(s))]])[0]
        else:
            u0 = rng.random(p.space.dim)
        u0 = np.clip(u0 + rng.normal(0, 0.02, len(u0)), 0, 1)
        try:
            res = minimize(fobj, u0, args=(w,), method="SLSQP", bounds=[(0.0, 1.0)] * p.space.dim,
                           constraints=[{"type": "ineq", "fun": fcon}] if p.constraints else [],
                           options={"maxiter": 200, "eps": 1e-3, "ftol": 1e-9},
                           callback=lambda _u, stop=sub_stop: _stop_at(run, stop))
            u_end = res.x
        except _Spent:
            u_end = None
        if u_end is not None and relax and run.remaining > 0:
            run.eval(p.space.from_unit(u_end), tag="slsqp rounded end point")


def _stop_at(run: _Run, stop: int):
    if run.used >= stop:
        raise _Spent


def _bayesian(run: _Run, seed: int, **_):
    p = run.problem
    d = p.space.dim
    cfg = DesignOptimizerConfig(n_initial_random=int(np.clip(run.budget // 4, 2 * d + 1, 4 * d + 4)))
    opt = BayesianDesignOptimizer(cfg, seed=seed)
    unit = np.c_[np.zeros(d), np.ones(d)]
    while run.remaining > 0:
        u = opt.propose(unit)
        ev = run.eval(p.space.from_unit(u), tag="bayesian")
        F, pen = _penalised(ev)
        opt.update(np.asarray(u, float), np.minimum(F[0], 1e12).tolist(), penalty=float(min(pen[0], 1e3)))


def _gradient(run: _Run, seed: int, n_weights: int | None = None, lr: float = 0.05, **_):
    import torch

    p = run.problem
    ev_ = p.evaluator
    if not ev_.differentiable:
        raise _NotApplicable(f"the evaluator {ev_.name!r} is not differentiable")
    has_int = bool(p.space.integer.any())
    if has_int and not ev_.relaxable:
        raise _NotApplicable("integer variables and an evaluator that does not accept fractional values")
    ideal, nadir = _normaliser(run, seed)
    span = torch.as_tensor(np.where(nadir - ideal > 0, nadir - ideal, 1.0), dtype=torch.float64)
    ideal_t = torch.as_tensor(ideal, dtype=torch.float64)
    b = torch.as_tensor(p.space.bounds, dtype=torch.float64)
    sign = torch.tensor([1.0 if o.sense == "min" else -1.0 for o in p.objectives], dtype=torch.float64)
    W = weight_vectors(p.n_obj, n_weights or (3 if p.n_obj == 1 else 5), seed)
    rng = np.random.default_rng(seed)
    for j, w in enumerate(W):
        if run.remaining < 3:
            break
        stop = run.used + run.remaining // (len(W) - j)
        wt = torch.as_tensor(w, dtype=torch.float64)
        ok = np.where(run.ev.feasible & ~run.ev.relaxed)[0]
        if len(ok) and p.n_obj > 1:
            s = np.sum(w * (run.ev.F[ok] - ideal) / span.numpy(), 1)
            u0 = p.space.to_unit(run.ev.X[ok[int(np.argmin(s))]])[0]
        else:
            u0 = rng.random(p.space.dim)
        u0 = np.clip(u0 + rng.normal(0, 0.05, len(u0)), 0.02, 0.98)
        z = torch.tensor(np.log(u0 / (1 - u0)), dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([z], lr=lr)
        mu = 10.0
        x_last = None
        while run.used < stop - 1:
            opt.zero_grad()
            x = b[:, 0] + torch.sigmoid(z) * (b[:, 1] - b[:, 0])
            t0 = time.perf_counter()
            out = ev_.torch(x[None, :])
            Fo = torch.stack([out[o.name].reshape(-1)[0] for o in p.objectives]) * sign
            loss = torch.sum(wt * (Fo - ideal_t) / span)
            for c in p.constraints:
                if c.name in out:
                    loss = loss + mu * torch.relu(-c.slack(out[c.name].reshape(-1)[0])) ** 2
            loss.backward()
            opt.step()
            Y = {k: v.detach().reshape(-1).numpy().astype(float) for k, v in out.items()}
            xr = x.detach().numpy()[None, :]
            ev = p.record(xr, Y, wall_s=time.perf_counter() - t0, tag=f"gradient weights {np.round(w, 2).tolist()}")
            ev.relaxed[:] = _fractional(p, xr)
            run.add(ev)
            mu = min(mu * 1.15, 1e6)
            x_last = xr
        if x_last is not None and (has_int or run.remaining > 0) and run.remaining > 0:
            run.eval(x_last, tag="gradient end point (rounded)")


class _NotApplicable(Exception):
    pass


METHODS: dict[str, Callable[..., None]] = {
    "random": _random, "nsga2": _nsga2, "pymoo-nsga2": _pymoo_nsga2, "scipy-de": _scipy_de,
    "scipy-slsqp": _slsqp, "bayesian": _bayesian, "gradient": _gradient,
}

FAMILIES = {"random": "baseline", "nsga2": "evolutionary", "pymoo-nsga2": "evolutionary", "scipy-de": "evolutionary",
            "scipy-slsqp": "gradient (finite differences)", "bayesian": "Bayesian", "gradient": "gradient (autograd)"}


def available_methods(problem: DesignProblem | None = None) -> dict[str, dict[str, Any]]:
    """Each method with its family and whether it can run here (``pymoo`` installed, evaluator differentiable)."""
    out = {}
    for m in METHODS:
        ok, why = True, ""
        if m == "pymoo-nsga2":
            try:
                import pymoo  # noqa: F401
            except ImportError:
                ok, why = False, "pymoo is not installed (pip install pymoo)"
        if m == "gradient" and problem is not None:
            e = problem.evaluator
            if not e.differentiable:
                ok, why = False, f"the evaluator {e.name!r} is not differentiable"
            elif problem.space.integer.any() and not e.relaxable:
                ok, why = False, "integer variables need an evaluator that accepts fractional values"
        out[m] = {"family": FAMILIES[m], "available": ok, "reason": why}
    return out


def run(problem: DesignProblem, method: str = "nsga2", *, budget: int = 400, seed: int = 0,
        initial: np.ndarray | None = None, progress: Callable[[int, int], None] | None = None,
        **options: Any) -> OptimizationResult:
    """Run one method on ``problem`` with at most ``budget`` evaluations.

    ``initial``: designs evaluated first (counted in the budget), e.g. a baseline design to beat. Methods that
    cannot run (missing optional dependency, non-differentiable evaluator) return ``status="skipped"``; a method
    that raises returns ``status="failed"`` with the message, keeping the evaluations it made. ``progress(used,
    budget)`` is called after every evaluated batch; an exception it raises (a cancelled job) stops the run and is
    re-raised.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; use one of {sorted(METHODS)}")
    if budget < 2:
        raise ValueError("budget must be at least 2 evaluations")
    r = _Run(problem, budget, progress)
    if method == "gradient":
        import torch.optim  # noqa: F401  (first import takes seconds: not charged to the method)
    t0 = time.perf_counter()
    status, msg = "ok", ""
    notes: list[str] = []
    avail = available_methods(problem)[method]
    if not avail["available"]:
        return OptimizationResult(problem, method, seed, budget, None, 0.0, "skipped", avail["reason"])
    try:
        if initial is not None and len(initial):
            r.eval(np.atleast_2d(initial), tag="initial designs")
        with warnings.catch_warnings():
            # GP length-scale bounds hit during Bayesian optimisation are reported by scikit-learn as warnings;
            # they do not change the evaluations recorded here
            warnings.filterwarnings("ignore", message=".*close to the specified (lower|upper) bound.*")
            METHODS[method](r, seed, **options)
    except _Spent:
        pass
    except _Abort as e:
        raise e.original from None
    except _NotApplicable as e:
        status, msg = "skipped", str(e)
    except Exception as e:  # noqa: BLE001 - the comparison reports a failed method instead of aborting
        status, msg = "failed", f"{type(e).__name__}: {e}"
    if method in ("gradient", "scipy-slsqp") and problem.space.integer.any() and problem.evaluator.relaxable:
        notes.append("integer variables relaxed during the gradient iterations; only rounded designs are candidates")
    if r.used < budget and status == "ok":
        notes.append(f"stopped after {r.used} of {budget} evaluations (the method converged or ran out of starts)")
    return OptimizationResult(problem, method, seed, budget, r.ev, time.perf_counter() - t0, status, msg, notes)


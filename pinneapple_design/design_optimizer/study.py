"""Design studies: a design space, objectives, constraints and an evaluator, and the record of every evaluation.

This is the problem description every optimisation method of :mod:`.methods` works on, whatever produces the
numbers: a closed-form model, a trained surrogate (a zoo model, any ``torch`` module), a high-fidelity solver or an
external program (:mod:`.evaluators`).

* :class:`Variable`, :class:`DesignSpace`: box-bounded continuous and integer design variables.
* :class:`Objective`, :class:`Constraint`: what to minimise / maximise and the limits (physical, manufacturing, model
  validity), each on one named output of the evaluator.
* :class:`Evaluator`: ``X (n, d) -> {output: (n,)}``; it says its fidelity (``"surrogate"``, ``"model"``,
  ``"reference"``) and whether it is differentiable (``torch`` path for gradient methods).
* :class:`DesignProblem`: the above together; :meth:`DesignProblem.evaluate` returns an :class:`Evaluations`
  record (objective matrix in minimisation form, normalised constraint violations, feasibility).
* :func:`feasibility_report`: active constraints and margins of chosen designs.
* :func:`hypervolume`: the dominated hypervolume of a front (exact in 2-D, seeded Monte Carlo above).

An evaluator may return the boolean output ``"_valid"``: ``False`` marks a design outside the evaluator's validity
range (e.g. a surrogate outside its training range); such a design is infeasible, with that reason.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .pareto import compute_pareto_front

__all__ = [
    "Variable", "DesignSpace", "Objective", "Constraint", "Evaluator", "FunctionEvaluator", "DesignProblem",
    "Evaluations", "feasibility_report", "hypervolume", "VALID", "constants_of",
]

VALID = "_valid"
FIDELITIES = ("surrogate", "model", "reference")


# --------------------------------------------------------------------------- space
@dataclass(frozen=True)
class Variable:
    """One design variable in ``[lower, upper]``; ``kind="integer"`` rounds it to whole numbers."""
    name: str
    lower: float
    upper: float
    kind: str = "continuous"
    unit: str = ""
    description: str = ""

    def __post_init__(self):
        if not self.upper > self.lower:
            raise ValueError(f"variable {self.name!r}: upper ({self.upper}) must exceed lower ({self.lower})")
        if self.kind not in ("continuous", "integer"):
            raise ValueError(f"variable {self.name!r}: kind must be 'continuous' or 'integer'")


class DesignSpace:
    """An ordered list of :class:`Variable`; design vectors are rows of ``X`` in this order."""

    def __init__(self, variables: Sequence[Variable]):
        if not variables:
            raise ValueError("a design space needs at least one variable")
        names = [v.name for v in variables]
        if len(set(names)) != len(names):
            raise ValueError("variable names must be unique")
        self.variables = list(variables)

    @property
    def names(self) -> list[str]:
        return [v.name for v in self.variables]

    @property
    def dim(self) -> int:
        return len(self.variables)

    @property
    def bounds(self) -> np.ndarray:
        return np.array([[v.lower, v.upper] for v in self.variables], dtype=float)

    @property
    def integer(self) -> np.ndarray:
        return np.array([v.kind == "integer" for v in self.variables])

    def repair(self, X: np.ndarray) -> np.ndarray:
        """Clip to the bounds and round the integer variables."""
        X = np.clip(np.atleast_2d(np.asarray(X, dtype=float)), self.bounds[:, 0], self.bounds[:, 1])
        X[:, self.integer] = np.round(X[:, self.integer])
        return X

    def to_unit(self, X: np.ndarray) -> np.ndarray:
        b = self.bounds
        return (np.atleast_2d(X) - b[:, 0]) / (b[:, 1] - b[:, 0])

    def from_unit(self, U: np.ndarray) -> np.ndarray:
        b = self.bounds
        return b[:, 0] + np.atleast_2d(U) * (b[:, 1] - b[:, 0])

    def sample(self, n: int, method: str = "lhs", seed: int = 0) -> np.ndarray:
        """``n`` repaired designs from :func:`pinneapple_data.parameter_sampling.sample_parameters`."""
        from pinneapple_data.parameter_sampling import sample_parameters
        rows = sample_parameters({v.name: (v.lower, v.upper) for v in self.variables}, int(n), method=method,
                                 seed=seed)
        return self.repair(np.array([[r[k] for k in self.names] for r in rows], dtype=float))

    def as_dicts(self, X: np.ndarray) -> list[dict[str, float]]:
        return [{v.name: (int(x[i]) if v.kind == "integer" else float(x[i])) for i, v in enumerate(self.variables)}
                for x in np.atleast_2d(X)]

    def from_dicts(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        missing = [k for k in self.names if any(k not in r for r in rows)]
        if missing:
            raise ValueError(f"designs miss the variables {sorted(set(missing))}")
        return self.repair(np.array([[float(r[k]) for k in self.names] for r in rows], dtype=float))

    def to_dict(self) -> dict[str, Any]:
        return {"variables": [asdict(v) for v in self.variables]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> DesignSpace:
        return cls([Variable(**v) for v in d["variables"]])

    @classmethod
    def from_parameter_space(cls, ps: Any) -> DesignSpace:
        """From a ``ParameterSpace``-like object or dict (``pinneapple_data.contracts.ParameterSpace``):
        ``parameters`` / ``variables`` entries with ``name`` and ``low``/``high`` (or ``lower``/``upper``, or
        ``bounds``), and optionally ``kind``/``type`` and ``unit``. Constant parameters are not design variables and
        are skipped (:func:`constants_of` returns them); categorical ones are refused."""
        d = ps.to_dict() if hasattr(ps, "to_dict") else ps
        entries = d.get("parameters", d.get("variables")) if isinstance(d, Mapping) else None
        if entries is None:
            raise ValueError("expected a parameter space with a 'parameters' list")
        out = []
        for e in entries:
            e = e if isinstance(e, Mapping) else vars(e)
            if e.get("kind") == "constant":
                continue
            if e.get("kind") == "categorical":
                raise ValueError(f"parameter {e.get('name')!r} is categorical: optimise each category separately "
                                 "(fix it as a constant)")
            lo = e.get("low", e.get("lower", (e.get("bounds") or [None, None])[0]))
            hi = e.get("high", e.get("upper", (e.get("bounds") or [None, None])[1]))
            if lo is None or hi is None:
                raise ValueError(f"parameter {e.get('name')!r} has no bounds")
            kind = str(e.get("kind", e.get("type", "continuous")))
            kind = "integer" if kind in ("integer", "int", "discrete") else "continuous"
            out.append(Variable(str(e["name"]), float(lo), float(hi), kind, str(e.get("unit", "") or ""),
                                str(e.get("description", "") or "")))
        return cls(out)


def constants_of(ps: Any) -> dict[str, Any]:
    """The constant parameters of a ``ParameterSpace``-like object or dict: ``{name: value}``."""
    d = ps.to_dict() if hasattr(ps, "to_dict") else ps
    out = {}
    for e in d.get("parameters", []) if isinstance(d, Mapping) else []:
        e = e if isinstance(e, Mapping) else vars(e)
        if e.get("kind") == "constant":
            out[str(e["name"])] = e.get("value")
    return out


# --------------------------------------------------------------------------- objectives and constraints
@dataclass(frozen=True)
class Objective:
    """Minimise (``sense="min"``) or maximise (``"max"``) the evaluator output ``name``."""
    name: str
    sense: str = "min"
    unit: str = ""
    label: str = ""

    def __post_init__(self):
        if self.sense not in ("min", "max"):
            raise ValueError("objective sense must be 'min' or 'max'")


@dataclass(frozen=True)
class Constraint:
    """``lower <= output <= upper`` on the evaluator output ``name``.

    ``kind`` is what the limit stands for (``"physical"``, ``"manufacturing"``, ``"validity"``); it only labels the
    feasibility report. The violation is normalised by ``|bound|`` (1 when the bound is 0), so constraints in
    different units add up in a penalty.
    """
    name: str
    upper: float | None = None
    lower: float | None = None
    unit: str = ""
    kind: str = "physical"
    label: str = ""

    def __post_init__(self):
        if self.upper is None and self.lower is None:
            raise ValueError(f"constraint {self.name!r} must have an upper or a lower bound")

    def violation(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        out = np.zeros_like(v)
        if self.upper is not None:
            out = np.maximum(out, (v - self.upper) / (abs(self.upper) or 1.0))
        if self.lower is not None:
            out = np.maximum(out, (self.lower - v) / (abs(self.lower) or 1.0))
        return np.where(np.isfinite(v), out, np.inf)

    def margin(self, v: np.ndarray) -> np.ndarray:
        """Distance to the nearest bound in the output's units (positive = inside)."""
        v = np.asarray(v, dtype=float)
        m = np.full_like(v, np.inf)
        if self.upper is not None:
            m = np.minimum(m, self.upper - v)
        if self.lower is not None:
            m = np.minimum(m, v - self.lower)
        return m

    def slack(self, v):
        """Signed normalised margin (>= 0 inside); works on NumPy arrays and on torch tensors."""
        terms = []
        if self.upper is not None:
            terms.append((self.upper - v) / (abs(self.upper) or 1.0))
        if self.lower is not None:
            terms.append((v - self.lower) / (abs(self.lower) or 1.0))
        if len(terms) == 1:
            return terms[0]
        try:
            import torch
            if isinstance(v, torch.Tensor):
                return torch.minimum(terms[0], terms[1])
        except ImportError:  # pragma: no cover
            pass
        return np.minimum(terms[0], terms[1])

    def describe(self) -> str:
        lab = self.label or self.name
        u = f" {self.unit}" if self.unit else ""
        if self.lower is not None and self.upper is not None:
            return f"{self.lower:g} <= {lab} <= {self.upper:g}{u}"
        return f"{lab} <= {self.upper:g}{u}" if self.upper is not None else f"{lab} >= {self.lower:g}{u}"


# --------------------------------------------------------------------------- evaluators
class Evaluator:
    """Maps designs ``X (n, d)`` (columns in the order of :attr:`DesignSpace.names`) to named outputs.

    Subclasses set ``name``, ``fidelity`` (``"surrogate"``: a learned model; ``"model"``: a classical engineering
    model; ``"reference"``: the high-fidelity solver used to verify) and ``outputs``, and implement :meth:`evaluate`.
    A differentiable evaluator sets ``differentiable = True`` and implements :meth:`torch` on a ``float64`` tensor.
    """
    name: str = "evaluator"
    fidelity: str = "model"
    outputs: Sequence[str] = ()
    differentiable: bool = False
    relaxable: bool = False          # accepts fractional values of integer variables (gradient methods relax them)
    description: str = ""

    def evaluate(self, X: np.ndarray) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def torch(self, X):  # pragma: no cover - only differentiable evaluators implement it
        raise NotImplementedError(f"{self.name} is not differentiable")

    def __call__(self, X: np.ndarray) -> dict[str, np.ndarray]:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = self.evaluate(X)
        return {k: np.asarray(v).reshape(len(X)) for k, v in out.items()}

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "fidelity": self.fidelity, "outputs": list(self.outputs),
                "differentiable": self.differentiable, "relaxable": self.relaxable, "description": self.description}


class FunctionEvaluator(Evaluator):
    """An evaluator from a vectorised function ``fn(X) -> {output: (n,)}`` (and optionally ``torch_fn``)."""

    def __init__(self, fn: Callable[[np.ndarray], Mapping[str, Any]], outputs: Sequence[str], *, name: str = "function",
                 fidelity: str = "model", torch_fn: Callable[[Any], Mapping[str, Any]] | None = None,
                 description: str = "", relaxable: bool = False):
        if fidelity not in FIDELITIES:
            raise ValueError(f"fidelity must be one of {FIDELITIES}")
        self.fn, self.torch_fn = fn, torch_fn
        self.outputs, self.name, self.fidelity, self.description = list(outputs), name, fidelity, description
        self.differentiable = torch_fn is not None
        self.relaxable = relaxable

    def evaluate(self, X):
        return dict(self.fn(X))

    def torch(self, X):
        if self.torch_fn is None:
            raise NotImplementedError(f"{self.name} is not differentiable")
        return dict(self.torch_fn(X))


# --------------------------------------------------------------------------- evaluation record
@dataclass
class Evaluations:
    """Every evaluated design of a study, in evaluation order.

    ``F`` is the objective matrix in minimisation form (maximised objectives negated), ``G`` the normalised
    violation per constraint (0 = satisfied), ``feasible`` all constraints satisfied and the design valid for its
    evaluator, ``relaxed`` marks designs that are not realisable (an integer variable evaluated at a fractional
    value by a gradient method); they never enter a Pareto set.
    """
    X: np.ndarray
    Y: dict[str, np.ndarray]
    F: np.ndarray
    G: np.ndarray
    feasible: np.ndarray
    relaxed: np.ndarray
    fidelity: list[str]
    wall_s: np.ndarray
    tag: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.X)

    @property
    def violation(self) -> np.ndarray:
        return self.G.sum(1) if self.G.size else np.zeros(len(self.X))

    def extend(self, other: Evaluations) -> Evaluations:
        keys = set(self.Y) | set(other.Y)
        nan = lambda n: np.full(n, np.nan)  # noqa: E731
        Y = {k: np.concatenate([self.Y.get(k, nan(len(self))), other.Y.get(k, nan(len(other)))]) for k in keys}
        return Evaluations(np.vstack([self.X, other.X]), Y, np.vstack([self.F, other.F]),
                           np.vstack([self.G, other.G]), np.r_[self.feasible, other.feasible],
                           np.r_[self.relaxed, other.relaxed], self.fidelity + other.fidelity,
                           np.r_[self.wall_s, other.wall_s], self.tag + other.tag)

    def pareto(self) -> np.ndarray:
        """Indices of the feasible, realisable, non-dominated designs (sorted by the first objective)."""
        ok = np.where(self.feasible & ~self.relaxed & np.all(np.isfinite(self.F), 1))[0]
        if not len(ok):
            return ok
        F = self.F[ok]
        # identical designs evaluated twice would both be non-dominated: keep the first
        _, first = np.unique(np.round(self.X[ok], 12), axis=0, return_index=True)
        keep = np.zeros(len(ok), bool)
        keep[first] = True
        mask = compute_pareto_front(F) & keep
        idx = ok[mask]
        return idx[np.argsort(self.F[idx, 0], kind="stable")]

    def rows(self, space: DesignSpace, idx: Sequence[int] | None = None) -> list[dict[str, Any]]:
        idx = range(len(self)) if idx is None else idx
        out = []
        for i in idx:
            r = {"index": int(i), **space.as_dicts(self.X[i])[0]}
            r.update({k: _num(v[i]) for k, v in self.Y.items() if k != VALID})
            r.update(feasible=bool(self.feasible[i]), violation=_num(self.violation[i]), fidelity=self.fidelity[i],
                     relaxed=bool(self.relaxed[i]))
            if self.tag:
                r["tag"] = self.tag[i]
            out.append(r)
        return out


def _num(v: Any) -> Any:
    v = float(v)
    return v if math.isfinite(v) else None


# --------------------------------------------------------------------------- problem
@dataclass
class DesignProblem:
    """A design optimisation problem: variables, objectives, constraints and the evaluator that computes them."""
    name: str
    space: DesignSpace
    objectives: list[Objective]
    constraints: list[Constraint]
    evaluator: Evaluator
    description: str = ""
    notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.objectives:
            raise ValueError("a design problem needs at least one objective")
        outs = set(self.evaluator.outputs)
        if outs:
            missing = [o.name for o in self.objectives if o.name not in outs]
            missing += [c.name for c in self.constraints if c.name not in outs]
            if missing:
                raise ValueError(f"the evaluator {self.evaluator.name!r} has no outputs {missing}")

    @property
    def n_obj(self) -> int:
        return len(self.objectives)

    def with_evaluator(self, evaluator: Evaluator, constraints: list[Constraint] | None = None) -> DesignProblem:
        """The same problem on another evaluator (e.g. the high-fidelity reference)."""
        cons = self.constraints if constraints is None else constraints
        outs = set(evaluator.outputs)
        if outs:
            cons = [c for c in cons if c.name in outs]
        return DesignProblem(self.name, self.space, self.objectives, cons, evaluator, self.description,
                             list(self.notes))

    def objective_matrix(self, Y: Mapping[str, np.ndarray]) -> np.ndarray:
        cols = [np.asarray(Y[o.name], dtype=float) * (1 if o.sense == "min" else -1) for o in self.objectives]
        return np.stack(cols, 1)

    def violations(self, Y: Mapping[str, np.ndarray], n: int) -> np.ndarray:
        if not self.constraints:
            return np.zeros((n, 0))
        return np.stack([c.violation(Y[c.name]) for c in self.constraints], 1)

    def record(self, X: np.ndarray, Y: Mapping[str, np.ndarray], *, wall_s: float = 0.0, relaxed: bool = False,
               fidelity: str | None = None, tag: str = "") -> Evaluations:
        X = np.atleast_2d(X)
        n = len(X)
        F = self.objective_matrix(Y)
        F = np.where(np.isfinite(F), F, np.inf)
        G = self.violations(Y, n)
        valid = np.asarray(Y.get(VALID, np.ones(n, bool)), dtype=bool)
        feasible = valid & np.all(G <= 0, 1) & np.all(np.isfinite(F), 1)
        return Evaluations(X.copy(), {k: np.asarray(v, dtype=float) for k, v in Y.items()}, F, G, feasible,
                           np.full(n, bool(relaxed)), [fidelity or self.evaluator.fidelity] * n,
                           np.full(n, wall_s / max(n, 1)), [tag] * n)

    def evaluate(self, X: np.ndarray, *, repair: bool = True, tag: str = "") -> Evaluations:
        X = self.space.repair(X) if repair else np.atleast_2d(np.asarray(X, dtype=float))
        t0 = time.perf_counter()
        Y = self.evaluator(X)
        return self.record(X, Y, wall_s=time.perf_counter() - t0, relaxed=not repair, tag=tag)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "space": self.space.to_dict(),
                "objectives": [asdict(o) for o in self.objectives],
                "constraints": [asdict(c) for c in self.constraints], "evaluator": self.evaluator.info(),
                "notes": list(self.notes)}


# --------------------------------------------------------------------------- reports
def feasibility_report(problem: DesignProblem, ev: Evaluations, idx: Sequence[int],
                       active_tol: float = 0.05) -> list[dict[str, Any]]:
    """Per design: each constraint's value, margin (in its units and relative to the bound) and status.

    ``status`` is ``"violated"``, ``"active"`` (inside, but within ``active_tol`` of the bound, relative) or
    ``"inactive"``. A design flagged invalid by its evaluator gets ``valid: False``.
    """
    out = []
    for i in idx:
        rows = []
        for c in problem.constraints:
            v = float(ev.Y[c.name][i])
            m = float(c.margin(np.array([v]))[0])
            bound = c.upper if (c.upper is not None and (c.lower is None or c.upper - v <= v - c.lower)) else c.lower
            rel = m / (abs(bound) or 1.0)
            status = "violated" if m < 0 or not math.isfinite(v) else "active" if rel <= active_tol else "inactive"
            rows.append({"constraint": c.name, "label": c.label or c.name, "kind": c.kind, "unit": c.unit,
                         "value": _num(v), "bound": bound, "margin": _num(m), "margin_rel": _num(rel),
                         "status": status, "rule": c.describe()})
        valid = bool(ev.Y.get(VALID, np.ones(len(ev), bool))[i])
        out.append({"index": int(i), "valid": valid, "feasible": bool(ev.feasible[i]),
                    "active": [r["constraint"] for r in rows if r["status"] == "active"],
                    "violated": [r["constraint"] for r in rows if r["status"] == "violated"]
                    + ([] if valid else ["outside the evaluator's validity range"]),
                    "constraints": rows})
    return out


def hypervolume(F: np.ndarray, ref: np.ndarray, n_mc: int = 20000, seed: int = 0) -> float:
    """Hypervolume dominated by the rows of ``F`` (minimisation) and bounded by ``ref``.

    Exact for one and two objectives; for more, a Monte Carlo estimate with ``n_mc`` seeded points (relative
    standard error about ``sqrt((1 - p) / (p n_mc))`` for a dominated fraction ``p``).
    """
    F = np.atleast_2d(np.asarray(F, dtype=float))
    ref = np.asarray(ref, dtype=float)
    if F.size == 0:
        return 0.0
    F = F[np.all(F < ref, 1)]
    if not len(F):
        return 0.0
    m = F.shape[1]
    if m == 1:
        return float(ref[0] - F[:, 0].min())
    if m == 2:
        P = F[compute_pareto_front(F)]
        P = P[np.argsort(P[:, 0])]
        hv, prev = 0.0, ref[1]
        for x, y in P:
            if y < prev:
                hv += (ref[0] - x) * (prev - y)
                prev = y
        return float(hv)
    lo = F.min(0)
    rng = np.random.default_rng(seed)
    S = lo + rng.random((n_mc, m)) * (ref - lo)
    dom = np.zeros(n_mc, bool)
    for f in F:
        dom |= np.all(S >= f, 1)
    return float(dom.mean() * np.prod(ref - lo))

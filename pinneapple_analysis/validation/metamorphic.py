"""Metamorphic tests for surrogates: relations between inputs and outputs that physics guarantees whatever the true
answer is, so a model can be tested where no reference solution exists (NumPy only; any callable model).

A relation transforms a sampled input, runs the model on both, and measures how far the outputs are from what the
relation requires, relative to the output scale. The worst sample is kept as the counterexample.

Relations
---------
``Symmetry``
    ``f(T x) = S(x, f(x))``: reflect or rotate the input field or geometry, and the output transforms accordingly
    (``reflect_field``, ``rotate_field`` build ``T``/``S`` for gridded fields).
``Similarity``
    Dimensionless groups unchanged, so the dimensionless output is unchanged: ``g(T x, f(T x)) = g(x, f(x))`` (same
    Reynolds number at another size or speed, same Biot or Fourier number, Darcy with ``a`` and ``f`` scaled
    together).
``Invariance``
    A change that must not move the output at all (``f(T x) = f(x)``), e.g. the thermal resistance of a linear
    conduction-convection model under a change of power.
``Monotonicity``
    More of an input gives no less (or no more) of an output: more heat input, no lower peak temperature; more air
    speed, no higher thermal resistance; larger permeability, no larger pressure drop.
``Superposition``
    Linear problems (Darcy with a fixed coefficient, linear elasticity, heat conduction):
    ``f(a + b) = f(a) + f(b)`` (with ``f(0)`` subtracted for affine maps) and optionally ``f(c a) = c f(a)``.
``ResolutionConsistency``
    For operators: a conserved or integral quantity of the output is the same when the input is given at another
    resolution.

Search
------
:func:`check_relations` draws inputs from a sampler (random search, vectorised) and, with ``search="hypothesis"``
when the ``hypothesis`` package is installed, also asks hypothesis for a violating sample (property-based search over
the sampler's seed or the input box, with shrinking). The worst violation found is returned as evidence.

Inputs can be anything the model takes: a batch array ``(n, ...)`` (fields) or a dict of ``(n,)`` columns (tabular
models such as the zoo predictors, whose outputs are dicts). ``output=`` selects an output column when the model
returns a dict.
"""
from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "Relation", "Symmetry", "Similarity", "Invariance", "Monotonicity", "Superposition", "ResolutionConsistency",
    "RelationResult", "MetamorphicReport", "check_relations", "box_sampler", "field_sampler", "reflect_field",
    "rotate_field", "relations_from_spec",
]

Batch = Any                                  # np.ndarray (n, ...) or dict[str, np.ndarray (n,)]
Sampler = Callable[[np.random.Generator, int], Batch]


# ---------------------------------------------------------------------------------------------- batch helpers
def _n(x: Batch) -> int:
    return len(next(iter(x.values()))) if isinstance(x, Mapping) else len(x)


def _take(x: Batch, i: int | np.ndarray) -> Batch:
    idx = np.atleast_1d(i)
    return {k: np.asarray(v)[idx] for k, v in x.items()} if isinstance(x, Mapping) else np.asarray(x)[idx]


def _jsonable(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {k: _jsonable(v) for k, v in x.items()}
    a = np.asarray(x)
    if a.size > 64:
        return {"shape": list(a.shape), "min": float(a.min()), "max": float(a.max()), "mean": float(a.mean())}
    return a.tolist()


def _out(y: Any, output: str | Sequence[str] | None) -> np.ndarray:
    """Model output as ``(n, m)`` float array (flattened per sample)."""
    if isinstance(y, Mapping):
        names = [output] if isinstance(output, str) else list(output) if output is not None else list(y)
        cols = [np.asarray(y[k], float).reshape(len(np.asarray(y[k])), -1) for k in names]
        return np.concatenate(cols, axis=1)
    if hasattr(y, "detach"):
        y = y.detach().cpu().numpy()
    a = np.asarray(y, float)
    return a.reshape(len(a), -1)


def box_sampler(bounds: Mapping[str, tuple[float, float]], fixed: Mapping[str, float] | None = None,
                log: Sequence[str] = ()) -> Sampler:
    """Uniform (or log-uniform for names in ``log``) samples in a box of named inputs, as a dict of columns."""
    names = list(bounds)

    def sample(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
        out = {}
        for k in names:
            lo, hi = map(float, bounds[k])
            u = rng.random(n)
            out[k] = np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))) if k in log else lo + u * (hi - lo)
        for k, v in (fixed or {}).items():
            out[k] = np.full(n, float(v))
        return out
    sample.bounds = dict(bounds)              # used by Monotonicity to stay inside the box
    return sample


def field_sampler(make: Callable[[np.random.Generator], np.ndarray]) -> Sampler:
    """Batch of fields from a one-sample generator ``make(rng) -> array``."""
    def sample(rng: np.random.Generator, n: int) -> np.ndarray:
        return np.stack([make(rng) for _ in range(n)])
    return sample


def reflect_field(axis: int) -> Callable[[np.ndarray], np.ndarray]:
    """Reflect a batch of fields ``(n, *space)`` along spatial ``axis`` (0-based, after the batch axis)."""
    return lambda x: np.flip(np.asarray(x), axis=1 + axis)


def rotate_field(k: int = 1, axes: tuple[int, int] = (0, 1)) -> Callable[[np.ndarray], np.ndarray]:
    """Rotate a batch of fields ``(n, *space)`` by ``k`` quarter turns in the plane of two spatial axes."""
    return lambda x: np.rot90(np.asarray(x), k=k, axes=(1 + axes[0], 1 + axes[1]))


# ---------------------------------------------------------------------------------------------- relations
@dataclass
class Relation:
    """Base class. ``violations(model, x)`` returns ``(per-sample relative violation, evidence per sample)``."""
    name: str
    tol: float = 1e-2
    output: str | Sequence[str] | None = None
    atol: float = 1e-12                       # floor of the output scale
    scale: float | None = None                # fixed output scale (e.g. the model's accepted error); default: RMS
    per_sample: bool = False                  # relative to each sample's own output (positive quantities)
    kind: str = field(default="relation", init=False)

    def violations(self, model: Callable, x: Batch) -> tuple[np.ndarray, dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError

    def _den(self, ref: np.ndarray) -> np.ndarray | float:
        """Denominator of the violation: per sample (``per_sample``) or one scale for the batch."""
        if self.per_sample and self.scale is None:
            return np.maximum(_rowrms(ref), self.atol)
        return self._scale(ref)

    def _scale(self, *ys: np.ndarray) -> float:
        if self.scale is not None:
            return float(self.scale)
        return max(float(np.sqrt(np.mean(np.concatenate([np.ravel(y) for y in ys]) ** 2))), self.atol)


def _rowrms(a: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.asarray(a, float).reshape(len(a), -1) ** 2, axis=1))


@dataclass
class Symmetry(Relation):
    """``f(T x) = S(x, f(x))``: ``transform_in(x)``; ``transform_out(x, y)`` acts on the output in the model's own
    form (array or dict) and defaults to the identity."""
    transform_in: Callable[[Batch], Batch] = None
    transform_out: Callable[[Batch, Any], Any] | None = None

    def __post_init__(self):
        self.kind = "symmetry"

    def violations(self, model, x):
        y = model(x)
        expected = _out(self.transform_out(x, y) if self.transform_out else y, self.output)
        got = _out(model(self.transform_in(x)), self.output)
        return _rowrms(got - expected) / self._den(expected), {"expected": expected, "got": got}


@dataclass
class Similarity(Relation):
    """``g(T x, f(T x)) = g(x, f(x))`` with ``dimensionless(x, y) -> array`` (default: the output itself)."""
    transform_in: Callable[[Batch], Batch] = None
    dimensionless: Callable[[Batch, Any], Any] | None = None

    def __post_init__(self):
        self.kind = "similarity"

    def violations(self, model, x):
        g = self.dimensionless or (lambda xx, yy: yy)
        x2 = self.transform_in(x)
        a = _out(g(x, model(x)), self.output)
        b = _out(g(x2, model(x2)), self.output)
        return _rowrms(b - a) / self._den(a), {"original": a, "scaled": b}


@dataclass
class Invariance(Relation):
    """``f(T x) = f(x)``."""
    transform_in: Callable[[Batch], Batch] = None

    def __post_init__(self):
        self.kind = "invariance"

    def violations(self, model, x):
        a = _out(model(x), self.output)
        b = _out(model(self.transform_in(x)), self.output)
        return _rowrms(b - a) / self._den(a), {"original": a, "changed": b}


@dataclass
class Monotonicity(Relation):
    """``output`` does not decrease (``direction=+1``) or increase (``-1``) when ``input`` grows.

    For dict inputs, ``input`` names a column and the step is ``step`` times the width of the sampler's box (the
    increased point is clipped into the box). For other inputs pass ``increase(x) -> x'`` instead. The violation is
    the size of the move in the wrong direction relative to the spread (standard deviation) of the output over the
    samples, so an offset such as an ambient temperature does not hide it (0 when the relation holds)."""
    input: str | None = None
    direction: int = 1
    step: float = 0.1
    increase: Callable[[Batch], Batch] | None = None
    bounds: Mapping[str, tuple[float, float]] | None = None

    def __post_init__(self):
        self.kind = "monotonicity"
        if self.direction not in (1, -1):
            raise ValueError("direction must be +1 (non-decreasing) or -1 (non-increasing)")

    def _up(self, x):
        if self.increase is not None:
            return self.increase(x)
        lo, hi = self.bounds[self.input] if self.bounds and self.input in self.bounds else (-np.inf, np.inf)
        width = (hi - lo) if np.isfinite(hi - lo) else max(float(np.ptp(x[self.input])), 1.0)
        x2 = dict(x)
        x2[self.input] = np.minimum(np.asarray(x[self.input], float) + self.step * width, hi)
        return x2

    def violations(self, model, x):
        a = _out(model(x), self.output)
        b = _out(model(self._up(x)), self.output)
        wrong = np.maximum(0.0, -self.direction * (b - a))
        den = float(self.scale) if self.scale is not None else max(float(np.std(a)), self.atol)
        return wrong.max(axis=1) / den, {"before": a, "after": b}


@dataclass
class Superposition(Relation):
    """``f(a + b) - f0 = (f(a) - f0) + (f(b) - f0)`` with ``f0 = f(0)`` when ``affine`` (else 0); with
    ``homogeneity`` also ``f(c a) - f0 = c (f(a) - f0)``. ``combine(a, b, ca, cb)`` forms ``ca a + cb b`` (default:
    arrays, or the ``linear_inputs`` columns of a dict, other columns taken from ``a``)."""
    affine: bool = False
    homogeneity: float | None = 2.0
    linear_inputs: Sequence[str] | None = None
    combine: Callable[[Batch, Batch, float, float], Batch] | None = None

    def __post_init__(self):
        self.kind = "superposition"

    def _comb(self, a, b, ca, cb):
        if self.combine is not None:
            return self.combine(a, b, ca, cb)
        if isinstance(a, Mapping):
            keys = self.linear_inputs or list(a)
            return {k: (ca * np.asarray(a[k]) + cb * np.asarray(b[k])) if k in keys else a[k] for k in a}
        return ca * np.asarray(a) + cb * np.asarray(b)

    def violations(self, model, x):
        n = _n(x)
        h = n // 2
        if h == 0:
            raise ValueError("superposition needs at least two samples")
        a, b = _take(x, np.arange(h)), _take(x, np.arange(h, 2 * h))
        f0 = _out(model(self._comb(a, a, 0.0, 0.0)), self.output) if self.affine else 0.0
        fa, fb = _out(model(a), self.output) - f0, _out(model(b), self.output) - f0
        fab = _out(model(self._comb(a, b, 1.0, 1.0)), self.output) - f0
        s = self._scale(fa, fb)
        v = _rowrms(fab - fa - fb) / s
        ev = {"f(a+b)": fab, "f(a)+f(b)": fa + fb}
        if self.homogeneity is not None:
            c = float(self.homogeneity)
            fca = _out(model(self._comb(a, a, c, 0.0)), self.output) - f0
            v = np.maximum(v, _rowrms(fca - c * fa) / s)
        return np.concatenate([v, v[: n - h]])[:n] if n > h else v, ev


@dataclass
class ResolutionConsistency(Relation):
    """``q(f(x)) = q(f(restrict(x)))``: ``restrict`` maps the input batch to another resolution, ``quantity`` maps an
    output batch to ``(n, k)`` integral quantities (default: the mean over all non-batch axes)."""
    restrict: Callable[[Batch], Batch] = None
    quantity: Callable[[Any], np.ndarray] | None = None

    def __post_init__(self):
        self.kind = "resolution"

    def violations(self, model, x):
        q = self.quantity or (lambda y: _out(y, self.output).mean(axis=1, keepdims=True))
        a = np.asarray(q(model(x)), float).reshape(_n(x), -1)
        b = np.asarray(q(model(self.restrict(x))), float).reshape(_n(x), -1)
        return _rowrms(b - a) / self._scale(a), {"fine": a, "coarse": b}


# ---------------------------------------------------------------------------------------------- running
@dataclass
class RelationResult:
    name: str
    kind: str
    passed: bool
    worst: float
    tol: float
    n_tested: int
    counterexample: dict[str, Any] | None = None
    search: str = "random"
    note: str = ""

    @property
    def check_id(self) -> str:
        return f"metamorphic.{self.kind}.{self.name}"

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.check_id, "name": self.name, "kind": self.kind, "passed": self.passed, "worst": self.worst,
                "tol": self.tol, "n_tested": self.n_tested, "counterexample": self.counterexample,
                "search": self.search, "note": self.note}


@dataclass
class MetamorphicReport:
    results: list[RelationResult]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def __getitem__(self, name: str) -> RelationResult:
        return next(r for r in self.results if r.name == name)

    def failures(self) -> list[RelationResult]:
        return [r for r in self.results if not r.passed]

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "results": [r.to_dict() for r in self.results]}

    def summary(self) -> str:
        lines = []
        for r in self.results:
            lines.append(f"{'PASS' if r.passed else 'FAIL'}  {r.kind:13s} {r.name}: worst violation {r.worst:.3g} "
                         f"(tolerance {r.tol:g}, {r.n_tested} samples, {r.search})" + (f" {r.note}" if r.note else ""))
        return "\n".join(lines)


def _evaluate(rel: Relation, model: Callable, x: Batch) -> tuple[np.ndarray, dict]:
    v, ev = rel.violations(model, x)
    v = np.asarray(v, float)
    v = np.where(np.isfinite(v), v, np.inf)
    return v, ev


def _counterexample(rel: Relation, x: Batch, v: np.ndarray, ev: dict) -> dict[str, Any]:
    i = int(np.argmax(v))
    j = min(i, _n(x) - 1)
    out = {"violation": float(v[i]), "input": _jsonable(_take(x, j))}
    for k, a in ev.items():
        a = np.asarray(a)
        if a.ndim and len(a) > min(i, len(a) - 1):
            out[k] = _jsonable(a[min(i, len(a) - 1)])
    return out


def _hypothesis_search(rel: Relation, model: Callable, sampler: Sampler, batch: int, max_examples: int,
                       seed: int) -> tuple[float, Batch | None, np.ndarray | None, dict | None]:
    from hypothesis import HealthCheck, Phase, find, settings
    from hypothesis import strategies as st

    best: dict[str, Any] = {"v": -1.0}

    def bad(s: int) -> bool:
        x = sampler(np.random.default_rng(s), batch)
        v, ev = _evaluate(rel, model, x)
        if v.max() > best["v"]:
            best.update(v=float(v.max()), x=x, vv=v, ev=ev)
        return bool(v.max() > rel.tol)

    cfg = settings(max_examples=max_examples, database=None, deadline=None, derandomize=True,
                   phases=[Phase.generate, Phase.shrink], suppress_health_check=list(HealthCheck))
    try:
        find(st.integers(min_value=seed, max_value=seed + 2 ** 31), bad, settings=cfg)
    except Exception:                                     # NoSuchExample: no violation found
        pass
    return best["v"], best.get("x"), best.get("vv"), best.get("ev")


def check_relations(model: Callable, relations: Sequence[Relation], sampler: Sampler, *, n: int = 64,
                    seed: int = 0, search: str = "random", max_examples: int = 50) -> MetamorphicReport:
    """Run every relation on ``n`` samples (``search="random"``) and, with ``search="hypothesis"``, add a
    property-based search for a violating sample (falls back to random search, noted, when hypothesis is missing)."""
    results = []
    for k, rel in enumerate(relations):
        rng = np.random.default_rng(seed + 7919 * k)
        x = sampler(rng, n)
        if isinstance(rel, Monotonicity) and rel.bounds is None and getattr(sampler, "bounds", None):
            rel.bounds = sampler.bounds
        v, ev = _evaluate(rel, model, x)
        worst, ce, used, note = float(v.max()), _counterexample(rel, x, v, ev), "random", ""
        if search == "hypothesis":
            try:
                hv, hx, hvv, hev = _hypothesis_search(rel, model, sampler, max(2, min(n, 8)), max_examples,
                                                      seed + 104729 * k)
                used = "random+hypothesis"
                if hx is not None and hv > worst:
                    worst, ce = hv, _counterexample(rel, hx, hvv, hev)
            except ImportError:
                note = "(hypothesis not installed: random search only)"
        results.append(RelationResult(rel.name, rel.kind, bool(worst <= rel.tol), worst, rel.tol, len(v),
                                      ce if worst > rel.tol else {"worst": ce}, used, note))
    return MetamorphicReport(results)


# ---------------------------------------------------------------------------------------------- declarative spec
_SAFE = {k: getattr(np, k) for k in ("sin", "cos", "tan", "exp", "log", "sqrt", "tanh", "abs", "minimum", "maximum",
                                     "where")}
_SAFE.update(pi=np.pi)

_BINOPS = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power,
           ast.Mod: np.mod}
_UNARY = {ast.USub: np.negative, ast.UAdd: np.positive}


def _compile_expr(expr: str) -> Callable[[dict], Any]:
    """A NumPy expression of the columns (``"4 * q"``, ``"-sin(alpha)"``), parsed into a whitelist of nodes:
    numbers, column names, the functions and constants of ``_SAFE``, arithmetic. No attribute access, subscripts,
    keywords or other calls, so a configuration file cannot run code (``eval`` without builtins is no sandbox)."""
    tree = ast.parse(str(expr), mode="eval").body

    def ev(node, env):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise ValueError(f"relation expression {expr!r}: unknown name {node.id!r}")
            return env[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](ev(node.left, env), ev(node.right, env))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand, env))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _SAFE \
                and callable(_SAFE[node.func.id]) and not node.keywords:
            return _SAFE[node.func.id](*(ev(a, env) for a in node.args))
        raise ValueError(f"relation expression {expr!r}: {type(node).__name__} is not allowed")
    return lambda env: ev(tree, env)


def _expr_map(mapping: Mapping[str, str]) -> Callable[[dict], dict]:
    """``{"M0": "-M0", "alpha_deg": "-alpha_deg"}``: new columns as NumPy expressions of the old ones."""
    compiled = {k: _compile_expr(e) for k, e in mapping.items()}

    def f(x):
        env = {**_SAFE, **{k: np.asarray(v) for k, v in x.items()}}
        out = dict(x)
        for k, c in compiled.items():
            out[k] = np.broadcast_to(np.asarray(c(env), float), np.shape(x[k])).copy()
        return out
    return f


def _output_map(m: Callable[[dict], dict]) -> Callable[[Batch, Any], dict]:
    return lambda x, y: m({k: np.asarray(v, float) for k, v in y.items()})


def relations_from_spec(spec: Sequence[Mapping[str, Any]]) -> list[Relation]:
    """Relations of a tabular model from a declarative list (TOML/JSON), e.g.::

        {kind = "monotonic", input = "air_velocity_m_s", output = "r_base_k_w", direction = "decreasing"}
        {kind = "invariant", inputs = {power_w = "2 * power_w"}, output = "r_base_k_w", tol = 0.02}
        {kind = "symmetric", inputs = {M0 = "-M0", alpha_deg = "-alpha_deg"}, outputs = {Cl = "-Cl"}}
    """
    out: list[Relation] = []
    for i, s in enumerate(spec):
        kind = str(s.get("kind", "")).lower()
        name = s.get("name") or f"{kind}-{i}"
        monotone = kind in ("monotonic", "monotonicity")
        tol = float(s.get("tol", 1e-3 if monotone else 1e-2))     # any measurable move the wrong way counts
        output = s.get("output")
        extra = {"scale": float(s["scale"])} if s.get("scale") is not None else {}
        if s.get("per_sample"):
            extra["per_sample"] = True
        if kind in ("monotonic", "monotonicity"):
            d = {"increasing": 1, "decreasing": -1, 1: 1, -1: -1}[s.get("direction", "increasing")]
            out.append(Monotonicity(name, tol, output, input=s["input"], direction=d, step=float(s.get("step", 0.1)),
                                    **extra))
        elif kind in ("invariant", "invariance"):
            out.append(Invariance(name, tol, output, transform_in=_expr_map(s["inputs"]), **extra))
        elif kind in ("symmetric", "symmetry"):
            outs = s.get("outputs", {})
            names = list(outs) if outs else output

            out.append(Symmetry(name, tol, names, transform_in=_expr_map(s["inputs"]),
                                transform_out=_output_map(_expr_map(outs)) if outs else None, **extra))
        elif kind in ("superposition", "linear"):
            out.append(Superposition(name, tol, output, affine=bool(s.get("affine", False)),
                                     linear_inputs=s.get("inputs"), **extra))
        else:
            raise ValueError(f"relation {name}: unknown kind {kind!r} (monotonic, invariant, symmetric, superposition)")
    return out

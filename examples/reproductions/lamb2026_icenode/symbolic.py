"""Equation discovery: a small genetic-programming symbolic regression, the paper's Figure 1c step.

Lamb et al. (2026) distil the trained neural ODE into a closed form with PySR: a genetic search over expression
trees, scored on accuracy and penalised for complexity, which returns a Pareto front of candidate equations. PySR
needs a Julia toolchain, so this module implements the same idea in numpy -- trees of ``+ - * / square sqrt`` over
the model's inputs and fitted constants, evolved with tournament selection, subtree crossover and mutation, with the
Pareto front of (complexity, error) kept and every constant then polished by nonlinear least squares
(:func:`refine`, a Levenberg-Marquardt-style fit on the tree's constants).

    front = discover({"m": m, "Si": Si}, y, generations=40, seed=0)
    best = front.best_by_mse()          # front.pareto holds one expression per complexity level
    print(best.render(), best.rmse(y, {"m": m, "Si": Si}))

The tool reports what it finds; it does not know the paper's equation, which is scored separately in the example.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Expr", "ParetoFront", "discover", "refine", "VAR", "CONST"]

VAR, CONST = "var", "const"
_BINARY = ("+", "-", "*", "/")
_UNARY = ("sq", "sqrt", "inv")


# ----------------------------------------------------------------------------------------------- expressions
@dataclass(frozen=True)
class Expr:
    """An expression tree: ``('var', name)``, ``('const', value)``, ``(op, a, b)`` or ``(op, a)``."""

    tree: tuple

    # ---- evaluation
    def eval(self, x: dict) -> np.ndarray:
        """Evaluate on the feature arrays ``x`` (values broadcast against each other)."""
        return _eval(self.tree, x)

    def rmse(self, y, x: dict) -> float:
        v = _eval(self.tree, x)
        return float(np.sqrt(np.mean((v - y) ** 2))) if np.all(np.isfinite(v)) else float("inf")

    # ---- structure
    @property
    def nodes(self) -> int:
        return _nodes(self.tree)

    @property
    def depth(self) -> int:
        return _depth(self.tree)

    def constants(self) -> list:
        return _constants(self.tree)

    def with_constants(self, values) -> Expr:
        return Expr(_set_constants(self.tree, list(values), [0]))

    def render(self, digits: int = 5) -> str:
        """The equation as a readable string."""
        return _render(self.tree, digits)

    def to_dict(self) -> dict:
        return {"expression": _render(self.tree, 6, sympy_like=True), "plain": self.render(), "nodes": self.nodes,
                "depth": self.depth, "tree": _to_json(self.tree)}


def _eval(t, x):
    op = t[0]
    if op == VAR:
        return np.asarray(x[t[1]], dtype=np.float64)
    if op == CONST:
        return np.float64(t[1])
    if op in _BINARY:
        a, b = _eval(t[1], x), _eval(t[2], x)
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        return a / np.where(np.abs(b) < 1e-12, np.copysign(1e-12, b), b)      # protected division
    a = _eval(t[1], x)
    if op == "sq":
        return a * a
    if op == "sqrt":
        return np.sqrt(np.abs(a))
    return 1.0 / np.where(np.abs(a) < 1e-12, np.copysign(1e-12, a), a)        # protected inverse


def _nodes(t) -> int:
    if t[0] in (VAR, CONST):
        return 1
    return 1 + sum(_nodes(c) for c in t[1:])


def _depth(t) -> int:
    if t[0] in (VAR, CONST):
        return 1
    return 1 + max(_depth(c) for c in t[1:])


def _constants(t) -> list:
    if t[0] == CONST:
        return [t[1]]
    if t[0] == VAR:
        return []
    return sum((_constants(c) for c in t[1:]), [])


def _set_constants(t, values, i):
    if t[0] == CONST:
        v = values[i[0]]
        i[0] += 1
        return (CONST, float(v))
    if t[0] == VAR:
        return t
    return (t[0],) + tuple(_set_constants(c, values, i) for c in t[1:])


def _render(t, digits=5, sympy_like=False) -> str:
    op = t[0]
    if op == VAR:
        return t[1]
    if op == CONST:
        return f"{t[1]:.{digits}g}"
    if op in _BINARY:
        a, b = _render(t[1], digits), _render(t[2], digits)
        sep = " * " if op == "*" else f" {op} "
        return f"({a}{sep}{b})"
    a = _render(t[1], digits)
    if op == "sq":
        return f"({a})^2"
    if op == "sqrt":
        return f"sqrt({a})"
    return f"1/({a})"


def _to_json(t):
    if t[0] in (VAR, CONST):
        return {"op": t[0], "value": t[1]}
    return {"op": t[0], "args": [_to_json(c) for c in t[1:]]}


# ----------------------------------------------------------------------------------------------- evolution
@dataclass
class ParetoFront:
    """One best expression per complexity level, the way a symbolic-regression Pareto front is reported."""

    pareto: list = None
    history: list = None
    seconds: float = 0.0

    def __post_init__(self):
        self.pareto = self.pareto or []
        self.history = self.history or []

    def best_by_mse(self) -> Expr:
        return min(self.pareto, key=lambda e: e[1])[0]

    def best_within(self, factor: float) -> Expr:
        """The simplest expression whose error is within ``factor`` of the best one (default: 1.0 = the best)."""
        best = min(e[1] for e in self.pareto)
        return min((e for e in self.pareto if e[1] <= factor * best), key=lambda e: e[0].nodes)[0]

    def to_dict(self) -> dict:
        return {"seconds": round(self.seconds, 1),
                "front": [dict(e[0].to_dict(), rmse=e[1]) for e in self.pareto],
                "history": [{"generation": g, "best_rmse": r, "complexity": n} for g, r, n in self.history]}


def _random_tree(rng, names, max_depth=3, allow_const=True):
    if max_depth <= 0 or (rng.random() < 0.3 and allow_const):
        if allow_const and rng.random() < 0.25:
            return (CONST, float(rng.choice([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])))
        return (VAR, str(rng.choice(names)))
    if rng.random() < 0.3:
        return (str(rng.choice(_UNARY)), _random_tree(rng, names, max_depth - 1, allow_const))
    op = str(rng.choice(_BINARY))
    return (op, _random_tree(rng, names, max_depth - 1, allow_const), _random_tree(rng, names, max_depth - 1,
                                                                                   allow_const))


def _subtree(t, rng):
    """A random subtree of ``t`` (uniform over nodes)."""
    n = _nodes(t)
    k = rng.integers(n)

    def walk(node):
        nonlocal k
        if k == 0:
            raise _Found(node)
        k -= 1
        if node[0] in (VAR, CONST):
            return node
        for c in node[1:]:
            walk(c)
        return node

    try:
        walk(t)
    except _Found as f:
        return f.node
    return t


class _Found(Exception):
    def __init__(self, node):
        self.node = node


def _replace(t, target, new):
    """Replace the first subtree equal to ``target`` by ``new``."""
    if t is target or t == target:
        return new
    if t[0] in (VAR, CONST):
        return t
    return (t[0],) + tuple(_replace(c, target, new) for c in t[1:])


def _mutate(t, rng, names):
    """One of four moves: regraft a random subtree, wrap a subtree in a unary op, grow a leaf into a binary op,
    or nudge a constant. The mix is what lets ``1/(a + b/m)``-shaped structures appear from simpler ancestors."""
    r = rng.random()
    if r < 0.35:
        return _replace(t, _subtree(t, rng), _random_tree(rng, names, 2))
    if r < 0.55:
        op = str(rng.choice(_UNARY))
        return _replace(t, _subtree(t, rng), (op, _subtree(t, rng)))
    if r < 0.85:
        leaf = _subtree(t, rng)
        op = str(rng.choice(_BINARY))
        other = _random_tree(rng, names, 1)
        return _replace(t, leaf, (op, leaf, other))
    if t[0] == CONST:
        return (CONST, float(t[1] * rng.choice([0.5, 0.8, 1.25, 2.0])))
    return _replace(t, _subtree(t, rng), _random_tree(rng, names, 2))


def _crossover(a, b, rng):
    sa = _subtree(a, rng)
    sb = _subtree(b, rng)
    return _replace(a, sa, sb)


def discover(x: dict, y, *, names=None, generations: int = 40, population: int = 240, max_nodes: int = 25,
             parsimony: float = 0.12, seed: int = 0, polish_every: int = 10, verbose: bool = False) -> ParetoFront:
    """Evolve expressions for ``y`` from the features ``x`` and return the Pareto front.

    The fitness is ``log10(rmse / std(y)) + parsimony * log10(nodes)``: accuracy relative to the target's spread,
    with a complexity price, which is what keeps the front from collapsing onto one huge expression.

    Every ``polish_every`` generations the best few individuals have their constants refitted by least squares
    (:func:`refine`) and re-enter the population, the optimisation step that makes a genetic search converge on the
    right *structure* instead of on lucky constants (``polish_every=0`` disables it).
    """
    import time

    t0 = time.time()
    rng = np.random.default_rng(seed)
    names = list(names or x.keys())
    y = np.asarray(y, dtype=np.float64)
    scale = float(np.std(y)) or 1.0
    oned = {k: np.asarray(v, dtype=np.float64).ravel() for k, v in x.items()}

    def fitness(expr):
        v = _eval(expr.tree, oned)
        if not np.all(np.isfinite(v)) or float(np.std(v)) == 0.0:
            return np.inf, np.inf
        r = float(np.sqrt(np.mean((v - y) ** 2)))
        return np.log10(r / scale + 1e-12) + parsimony * np.log10(expr.nodes), r

    pop = [_random_tree(rng, names, 3) for _ in range(population)]
    front: dict = {}
    hist = []
    for gen in range(generations):
        scored = [(fit, r, t) for t, (fit, r) in ((t, fitness(Expr(t))) for t in pop)]
        scored = [s for s in scored if np.isfinite(s[0])]
        if not scored:
            pop = [_random_tree(rng, names, 3) for _ in range(population)]
            continue
        scored.sort(key=lambda s: s[0])
        for _fit, r, t in scored[:20]:                                # keep one entry per complexity
            e = Expr(t)
            if e.nodes <= max_nodes and (e.nodes not in front or r < front[e.nodes][1]):
                front[e.nodes] = (e, r)
        hist.append((gen, scored[0][1], Expr(scored[0][2]).nodes))
        if verbose:
            print(f"  [gp {gen:3d}] rmse {scored[0][1]:.4g} nodes {Expr(scored[0][2]).nodes}", flush=True)
        # (mu + lambda): keep the best third, breed the rest
        survivors = [s[2] for s in scored[:max(8, population // 3)]]
        if polish_every and gen % polish_every == polish_every - 1:
            polished = []
            for t in survivors[:3]:
                try:
                    e = refine(Expr(t), oned, y)
                except Exception:                                    # pragma: no cover - a fit that will not run
                    continue
                v = _eval(e.tree, oned)
                if np.all(np.isfinite(v)):
                    polished.append((Expr(e.tree), float(np.sqrt(np.mean((v - y) ** 2)))))
            for e, r in polished:
                survivors.append(e.tree)
                if e.nodes <= max_nodes and (e.nodes not in front or r < front[e.nodes][1]):
                    front[e.nodes] = (e, r)
        children = list(survivors)
        while len(children) < population:
            def pick(survivors=survivors, scored=scored):
                a, b = rng.integers(len(survivors), size=2)
                return survivors[a] if scored[a][0] < scored[b][0] else survivors[b]
            if rng.random() < 0.35:
                child = _crossover(pick(), pick(), rng)
            else:
                child = _mutate(pick(), rng, names)
            if _nodes(child) <= max_nodes:
                children.append(child)
        pop = children
    pareto = sorted(front.items())
    return ParetoFront([(e, r) for _n, (e, r) in pareto], hist, time.time() - t0)


def refine(expr: Expr, x: dict, y, *, loss: str = "rmse") -> Expr:
    """Polish the constants of ``expr`` by nonlinear least squares, keeping its structure fixed."""
    from scipy.optimize import least_squares

    y = np.asarray(y, dtype=np.float64).ravel()
    oned = {k: np.asarray(v, dtype=np.float64).ravel() for k, v in x.items()}
    x0 = np.array(expr.constants(), dtype=float)
    if x0.size == 0:
        return expr

    def res(c):
        v = _eval(expr.with_constants(c).tree, oned)
        v = np.where(np.isfinite(v), v, 1e12)
        return v - y

    out = least_squares(res, x0, x_scale="jac", max_nfev=4000)
    return expr.with_constants(out.x)

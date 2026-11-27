"""Genetic-programming symbolic regression for closures and missing terms, checked against physics.

SINDy (:class:`~pinneapple_analysis.inverse_problems.missing_term.SINDyIdentifier`,
:func:`~pinneapple_analysis.inverse_problems.discovery.discover_ode`) searches a **library of candidate terms chosen
in advance**: coefficients are fitted, the terms are not invented. A third-order reaction ``k u^3``, a wall law with a
logarithm or a drag law with a square root are invisible to it — not badly fitted, invisible. This module searches the
expressions themselves, with products, ratios, powers and special functions, and then does the part that matters:
**it checks what it found against the physics, and against SINDy on the same data.**

Three pieces, usable separately:

* :class:`Expression` -- the tree (a frozen dataclass), what it prints, what it evaluates to, and what dimension it
  has (units from :mod:`pinneapple_data.physical_units`).
* :class:`PhysicsConstraints` -- the filter: a dimensional signature, a sign, a limit the term must respect, a
  symmetry and a balance (an integral the term has to add up to). :meth:`Expression.check` turns an expression into a
  verdict, with the reason it was refused.
* :class:`GeneticProgramming` -- the engine: tree genomes, subtree crossover and mutation, the constants fitted by
  least squares (linear scaling, Keijzer 2003), and NSGA-II (Deb et al. 2002) survival over the Pareto front of
  error against complexity. Dimensional inconsistency is pruned while the trees are bred, so the search never spends
  a generation on an expression with the wrong units.

:func:`discover_expression` is the one call that does all of it for a column of data; :func:`compare_gp_and_sindy`
runs the same problem through this engine and through SINDy's sparse regression over a library, and returns both arms'
numbers side by side -- recovery error, number of terms, seconds -- because a discovery that was not compared is a
curve with a story attached.

References
----------
Koza, J. R. (1992). *Genetic Programming: On the Programming of Computers by Means of Natural Selection*. MIT Press.
Schmidt, M. & Lipson, H. (2009). Distilling free-form natural laws from experimental data. *Science* 324, 81-85.
Keijzer, M. (2003). Improving symbolic regression with interval arithmetic and linear scaling. *EuroGP* 2003.
Deb, K., Pratap, A., Agarwal, S. & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm:
NSGA-II. *IEEE Trans. Evol. Comput.* 6(2), 182-197.
Brunton, S. L., Proctor, J. L. & Kutz, J. N. (2016). Discovering governing equations from data by sparse
identification of nonlinear dynamical systems. *PNAS* 113(15), 3932-3937.
"""
from __future__ import annotations

import ast
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np

__all__ = [
    "DimensionError",
    "Expression",
    "ExpressionVerdict",
    "PhysicsConstraints",
    "PolicyViolation",
    "GeneticProgramming",
    "SymbolicRegressionResult",
    "discover_expression",
    "compare_gp_and_sindy",
    "parse_expression",
    "advection_diffusion_reaction_case",
    "noise_sweep",
]


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

class DimensionError(ValueError):
    """An expression whose terms do not share a dimension, or a function of a dimensioned argument."""


class PolicyViolation(ValueError):
    """A constraint this engine will not accept silently (a mistyped unit, a balance given in the wrong unit)."""


# ---------------------------------------------------------------------------
# the tree
# ---------------------------------------------------------------------------

_BINARY = ("+", "-", "*", "/")
_UNARY = ("sin", "cos", "exp", "log", "sqrt")
_TRANSCENDENTAL = ("sin", "cos", "exp", "log")          # the argument has to be dimensionless
#: the whole-number powers a node may raise to (a negative or fractional power of a quantity with units is not a
#: quantity: ``sqrt`` carries its own rule, and ``u**0.5`` written as a power would hide it)
_POWERS = (2, 3)


def _protected_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a)
    np.divide(a, b, out=out, where=np.abs(b) > 1e-12)
    return out


_EVAL = {
    "+": np.add, "-": np.subtract, "*": np.multiply, "/": _protected_div,
    "sin": np.sin, "cos": np.cos, "exp": lambda x: np.exp(np.clip(x, -50.0, 50.0)),
    "log": lambda x: np.log(np.abs(x) + 1e-12), "sqrt": lambda x: np.sqrt(np.abs(x)),
    "tanh": np.tanh,
}


@dataclass(frozen=True)
class Expression:
    """A closed-form expression as a tree.

    ``op`` is ``"var"`` (a named column), ``"const"`` (a number, dimensionless), one of ``{"+", "-", "*", "/"}``
    with two children, one of ``{"sin", "cos", "exp", "log", "sqrt"}`` with one, or ``"^"`` with one child and
    ``power`` set. Evaluation is protected: division by zero gives zero, ``log`` and ``sqrt`` take the absolute value,
    ``exp`` is clipped at ``e^{+-50}``. Those choices affect the value of a broken expression, never the value of a
    correct one, and keep a bad individual from turning a whole population into ``nan``."""

    op: str
    children: tuple[Expression, ...] = ()
    name: str = ""
    value: float = 0.0
    power: int = 0

    # -- construction ---------------------------------------------------------
    @staticmethod
    def variable(name: str) -> Expression:
        return Expression("var", (), name=name)

    @staticmethod
    def constant(value: float) -> Expression:
        return Expression("const", (), value=float(value))

    def with_children(self, *children: Expression) -> Expression:
        return Expression(self.op, tuple(children), name=self.name, value=self.value, power=self.power)

    # -- shape ----------------------------------------------------------------
    def nodes(self) -> int:
        return 1 + sum(c.nodes() for c in self.children)

    def depth(self) -> int:
        return 1 + max((c.depth() for c in self.children), default=0)

    def variables(self) -> set[str]:
        if self.op == "var":
            return {self.name}
        out: set[str] = set()
        for c in self.children:
            out |= c.variables()
        return out

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()

    def replace_at(self, index: int, new: Expression) -> tuple[Expression, bool]:
        """``self`` with subtree ``index`` (pre-order) replaced by ``new``; ``(tree, replaced)``."""
        return self._replace([index], new)

    def _replace(self, path: list[int], new: Expression) -> tuple[Expression, bool]:
        if not path:
            return new, True
        if not self.children:
            return self, False
        i = path[0]
        if i >= len(self.children):
            return self, False
        child, ok = self.children[i]._replace(path[1:], new)
        if not ok:
            return self, False
        kids = list(self.children)
        kids[i] = child
        return self.with_children(*kids), True

    def subtree_at(self, index: int) -> Expression:
        """The ``index``-th subtree in pre-order."""
        for i, node in enumerate(self.walk()):
            if i == index:
                return node
        return self

    # -- value ----------------------------------------------------------------
    def evaluate(self, columns: Mapping[str, np.ndarray]) -> np.ndarray:
        """Evaluate over columns (name -> array); returns a float array of the columns' length."""
        if self.op == "var":
            if self.name not in columns:
                raise KeyError(f"unknown variable {self.name!r}: the data has {sorted(columns)}")
            return np.asarray(columns[self.name], dtype=float)
        if self.op == "const":
            length = len(next(iter(columns.values()))) if columns else 1
            return np.full(length, self.value)
        args = [c.evaluate(columns) for c in self.children]
        if self.op == "^":
            return np.power(args[0], self.power)
        with np.errstate(all="ignore"):
            return _EVAL[self.op](*args)

    def __call__(self, columns: Mapping[str, np.ndarray]) -> np.ndarray:
        return self.evaluate(columns)

    # -- text -----------------------------------------------------------------
    def to_string(self) -> str:
        """The expression as Python source, readable and parseable by :func:`pinneapple_data.expressions.check`."""
        if self.op == "var":
            return self.name
        if self.op == "const":
            return f"{self.value:.6g}"
        if self.op == "^":
            return f"({self.children[0].to_string()})**{self.power}"
        if self.op in _BINARY:
            left, right = self.children
            return f"({left.to_string()} {self.op} {right.to_string()})"
        return f"{self.op}({self.children[0].to_string()})"

    def simplified(self) -> Expression:
        """The same expression with constant subexpressions folded and arithmetic canonicalised, read back from
        :mod:`sympy` (``(u * u) * k`` becomes ``k*u**2``). Falls back to ``self`` when sympy cannot read it back."""
        import sympy as sp
        try:
            local = {n: sp.Symbol(n, real=True) for n in sorted(self.variables())}
            symbolic = self.to_sympy(local)
            compact = sp.N(sp.simplify(symbolic), 8)      # the numeric factors folded (0.5657*sqrt(2) is 0.8)
            if compact.free_symbols - set(local.values()):
                return self
            return _from_sympy(compact, local)
        except (sp.SympifyError, TypeError, ValueError, RecursionError):
            return self

    def to_sympy(self, local: Mapping[str, Any] | None = None):
        """A sympy expression (the names are symbols unless ``local`` says otherwise)."""
        import sympy as sp
        table = dict(local or {})
        if self.op == "var":
            return table.get(self.name) or sp.Symbol(self.name, real=True)
        if self.op == "const":
            return sp.Float(self.value) if self.value != int(self.value) else sp.Integer(int(self.value))
        if self.op == "^":
            return sp.Pow(self.children[0].to_sympy(table), self.power)
        fn = {"+": sp.Add, "-": sp.Add, "*": sp.Mul, "/": sp.Mul,
              "sin": sp.sin, "cos": sp.cos, "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "tanh": sp.tanh}[self.op]
        args = [c.to_sympy(table) for c in self.children]
        if self.op == "-":
            return sp.Add(*args[:1], sp.Mul(-1, *args[1:]))
        if self.op == "/":
            return sp.Mul(args[0], sp.Pow(args[1], -1))
        return fn(*args)

    def latex(self) -> str:
        import sympy as sp
        try:
            return sp.latex(self.to_sympy())
        except (sp.SympifyError, TypeError):
            return self.to_string()

    def __str__(self) -> str:
        return self.to_string()

    # -- units ----------------------------------------------------------------
    def dimension(self, units: Mapping[str, str]) -> tuple[Fraction, ...]:
        """The dimension of the expression, from the unit of every column (``{"u": "mol/m^3"}``).

        The exponents are fractions, not integers, because ``sqrt(u)`` of a concentration is a real closure shape
        (a surface-limited rate) whose prefactor carries the half units. Raises :class:`DimensionError` for the two
        mistakes a symbolic search makes over and over: two terms that are added but do not share a dimension
        (``u + x`` with ``u`` in mol/m^3 and ``x`` in m), and a function of a quantity that has units (``log(u)``,
        ``sin(t)``, ``exp(y)``)."""
        from pinneapple_data.physical_units import parse_unit

        dims: dict[str, tuple[int, ...]] = {}
        for name, unit in units.items():
            try:
                dims[name] = _doubled(parse_unit(str(unit)).dim)
            except Exception as exc:                       # noqa: BLE001 - any unit this library cannot read
                raise PolicyViolation(f"the unit of {name!r} ({unit!r}) is not one this library parses: {exc}") \
                    from None
        return tuple(Fraction(p, 2) for p in self._dimension(dims))

    def _dimension(self, dims: Mapping[str, tuple[int, ...]]) -> tuple[int, ...]:
        """The exponents **doubled**, so that ``sqrt`` stays exact: a half power is an odd integer, not a float."""
        zero: tuple[int, ...] = (0,) * 6
        if self.op == "const":
            return zero
        if self.op == "var":
            if self.name not in dims:
                raise DimensionError(f"the unit of {self.name!r} is not declared; give units={{{self.name!r}: '...'}}")
            return dims[self.name]
        child_dims = [c._dimension(dims) for c in self.children]
        if self.op in ("+", "-"):
            if child_dims[0] != child_dims[1]:
                raise DimensionError(f"'{self.children[0]}' is {_format_dim(child_dims[0])} but "
                                     f"'{self.children[1]}' is {_format_dim(child_dims[1])}: they cannot be "
                                     f"{'added' if self.op == '+' else 'subtracted'}")
            return child_dims[0]
        if self.op == "*":
            return tuple(a + b for a, b in zip(child_dims[0], child_dims[1], strict=True))
        if self.op == "/":
            return tuple(a - b for a, b in zip(child_dims[0], child_dims[1], strict=True))
        if self.op == "^":
            return tuple(p * self.power for p in child_dims[0])
        if self.op == "sqrt":
            if any(p % 2 for p in child_dims[0]):
                raise DimensionError(f"'{self.children[0]}' is {_format_dim(child_dims[0])}: its square root is not "
                                     f"a quantity with whole-number exponents (a quarter power)")
            return tuple(p // 2 for p in child_dims[0])
        if self.op in _TRANSCENDENTAL and child_dims[0] != zero:
            raise DimensionError(f"'{self.op}()' takes a dimensionless argument; '{self.children[0]}' is "
                                 f"{_format_dim(child_dims[0])}")
        return zero

    # -- the filter -----------------------------------------------------------
    def check(self, constraints: PhysicsConstraints, columns: Mapping[str, np.ndarray],
              scale: float = 1.0) -> ExpressionVerdict:
        """Run the physics filter over this expression and the data it was fitted to (see
        :class:`PhysicsConstraints`). ``scale`` is the fitted prefactor: the constraints apply to ``scale * self``,
        because that is the term that would be reported."""
        return constraints.check(self, columns, scale=scale)


def _doubled(dim: Sequence[int]) -> tuple[int, ...]:
    """The integer exponents from :mod:`pinneapple_data.physical_units`, in the engine's doubled representation."""
    return tuple(2 * int(p) for p in dim)


def _format_dim(dim: Sequence[int]) -> str:
    """A dimension (exponents doubled) in the reader's units: ``mol/m^3``, ``mol^0.5/m^1.5`` after a ``sqrt``."""
    from pinneapple_data.physical_units import dim_str
    if all(int(p) % 2 == 0 for p in dim):
        return dim_str(tuple(int(p) // 2 for p in dim))
    names = ("m", "kg", "s", "K", "A", "mol")
    top = [f"{n}^{p / 2:g}" if p != 2 else n for n, p in zip(names, dim, strict=True) if p > 0]
    bottom = [f"{n}^{-p / 2:g}" if p != -2 else n for n, p in zip(names, dim, strict=True) if p < 0]
    return "·".join(top or ["1"]) + ("/" + "·".join(bottom) if bottom else "")


def _from_sympy(node, local: Mapping[str, Any]) -> Expression:
    """A sympy expression back into an :class:`Expression` (the subset the engine writes)."""
    import sympy as sp
    reverse = {v: k for k, v in local.items()}
    if node in reverse:
        return Expression.variable(reverse[node])
    if node.is_Number:
        return Expression.constant(float(node))
    if isinstance(node, sp.Pow):
        base, exponent = node.as_base_exp()
        if exponent.is_Integer and int(exponent) in _POWERS:
            return Expression("^", (_from_sympy(base, local),), power=int(exponent))
        if exponent.is_Rational and exponent == sp.Rational(1, 2):
            return Expression("sqrt", (_from_sympy(base, local),))
        raise ValueError(f"{node} is not a form the engine writes")
    if isinstance(node, sp.Add):
        terms = list(node.args)
        if not terms:
            return Expression.constant(0.0)
        out = _from_sympy(terms[0], local)
        for term in terms[1:]:
            out = Expression("+", (out, _from_sympy(term, local)))
        return out
    if isinstance(node, sp.Mul):
        factors = list(node.args)
        out = _from_sympy(factors[0], local)
        for factor in factors[1:]:
            child = _from_sympy(factor, local)
            if factor.is_Pow and factor.exp.is_Integer and int(factor.exp) < 0:
                child = Expression("/", (Expression.constant(1.0), child))
            out = Expression("*", (out, child))
        return out
    if isinstance(node, sp.sin):
        return Expression("sin", (_from_sympy(node.args[0], local),))
    if isinstance(node, sp.cos):
        return Expression("cos", (_from_sympy(node.args[0], local),))
    if isinstance(node, sp.exp):
        return Expression("exp", (_from_sympy(node.args[0], local),))
    if isinstance(node, sp.log):
        return Expression("log", (_from_sympy(node.args[0], local),))
    if isinstance(node, sp.tanh):
        return Expression("tanh", (_from_sympy(node.args[0], local),))
    raise ValueError(f"{node} is not a form the engine writes")


# ---------------------------------------------------------------------------
# the physics filter
# ---------------------------------------------------------------------------

@dataclass
class ExpressionVerdict:
    """What the physics filter said about one expression, check by check.

    Every entry of :attr:`checks` is ``{"check": ..., "ok": bool, "detail": ...}``; :attr:`reasons` lists the
    failures in the words they are reported in."""

    expression: str
    checks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c["ok"] for c in self.checks)

    @property
    def reasons(self) -> list[str]:
        return [c["detail"] for c in self.checks if not c["ok"]]

    @property
    def refused(self) -> bool:
        return not self.ok

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        head = f"{self.expression}: {'accepted' if self.ok else 'refused'}"
        body = "; ".join(f"{c['check']} {'ok' if c['ok'] else 'FAILED'} ({c['detail']})" for c in self.checks)
        return f"{head} -- {body}" if body else head

    def to_dict(self) -> dict[str, Any]:
        return {"expression": self.expression, "ok": self.ok, "checks": self.checks, "reasons": self.reasons}


@dataclass
class PhysicsConstraints:
    """The physics an expression has to satisfy to be reported, checked on the data it was fitted to.

    ``units`` / ``target``
        Dimensional consistency (units from :mod:`pinneapple_data.physical_units`): every column's unit, and the
        unit of the term. The constants the engine fits are dimensionless, so ``target`` is what the expression
        itself has to work out to. When the prefactor carries a unit (an unknown rate constant, say), leave
        ``target=None`` and pass ``prefactor`` instead: the check then reports which unit the prefactor must have
        and refuses an expression whose terms do not all share one dimension, which is the mistake that matters.

    ``sign``
        ``">=0"`` (a source, a viscosity, a positive rate) or ``"<=0"`` (a sink) or ``None``. Checked on
        ``scale * expression`` over the data.

    ``limits``
        ``{"u": 0.0}``: the term has to vanish where every named column takes the given value -- a closure that does
        not vanish with the quantity it closes. Checked on the rows where the columns are within ``limit_tol`` of the
        values.

    ``symmetry``
        ``{"u": "even"}`` / ``{"u": "odd"}``: the term must not change sign (``even``) or must change sign (``odd``)
        when that column is negated. Checked on the data.

    ``balance``
        ``(value, tolerance)``: the integral of the term over the data (``sum(term)``, or ``sum(term * weights)``
        when ``weights`` is given) has to equal ``value`` within ``tolerance`` -- the conservation statement a
        control volume gives. A term that fits the points but adds up to the wrong total creates or destroys mass,
        momentum or energy and is refused.
    """

    units: Mapping[str, str] | None = None
    target: str | None = None
    prefactor: str | None = None
    sign: str | None = None
    limits: Mapping[str, float] | None = None
    symmetry: Mapping[str, str] | None = None
    balance: tuple[float, float] | None = None
    weights: np.ndarray | None = None
    limit_tol: float = 1e-9

    def __post_init__(self) -> None:
        if self.sign not in (None, ">=0", "<=0"):
            raise PolicyViolation("sign is '>=0', '<=0' or None")
        for name, kind in dict(self.symmetry or {}).items():
            if kind not in ("even", "odd"):
                raise PolicyViolation(f"symmetry of {name!r} is 'even' or 'odd', not {kind!r}")
        if self.target is not None and self.units is None:
            raise PolicyViolation("a target unit needs the units of the columns: pass units={...} too")
        from pinneapple_data.physical_units import parse_unit
        self._dims: dict[str, tuple[int, ...]] = {}          # parsed once, exponents doubled: the engine asks on
        for name, unit in dict(self.units or {}).items():     # every child it breeds
            try:
                self._dims[str(name)] = _doubled(parse_unit(str(unit)).dim)
            except Exception as exc:                       # noqa: BLE001 - any unit this library cannot read
                raise PolicyViolation(f"the unit of {name!r} ({unit!r}) is not one this library parses: {exc}") \
                    from None
        self._target_dim: tuple[int, ...] | None = None
        if self.target is not None:
            try:
                self._target_dim = _doubled(parse_unit(self.target).dim)
            except Exception as exc:                       # noqa: BLE001
                raise PolicyViolation(f"the target unit {self.target!r} is not one this library parses: {exc}") \
                    from None
            if self.prefactor is not None:
                try:
                    pre = _doubled(parse_unit(self.prefactor).dim)
                except Exception as exc:                   # noqa: BLE001
                    raise PolicyViolation(f"the prefactor unit {self.prefactor!r} is not one this library parses: "
                                          f"{exc}") from None
                self._target_dim = tuple(t - p for t, p in zip(self._target_dim, pre, strict=True))

    @property
    def active(self) -> bool:
        return bool(self.units or self.sign or self.limits or self.symmetry or self.balance)

    def dimension_ok(self, expression: Expression) -> tuple[bool, str]:
        """The dimensional check alone (cheap; the engine calls it on every child it breeds)."""
        if self.units is None:
            return True, "no units declared"
        try:
            dim = expression._dimension(self._dims)
        except DimensionError as exc:
            return False, str(exc)
        if self.target is None:
            return True, f"{_format_dim(dim)} (prefactor carries the rest)"
        if dim != self._target_dim:
            return False, (f"the expression is {_format_dim(dim)}, the term is {_format_dim(self._target_dim)}"
                           + (f" (prefactor in {self.prefactor})" if self.prefactor else ""))
        return True, f"every term is {_format_dim(dim)}"

    def check(self, expression: Expression, columns: Mapping[str, np.ndarray],
              scale: float = 1.0) -> ExpressionVerdict:
        """The full filter, one entry per declared constraint."""
        verdict = ExpressionVerdict(str(expression))
        if self.units is not None:
            ok, detail = self.dimension_ok(expression)
            verdict.checks.append({"check": "dimension", "ok": ok, "detail": detail})
        with np.errstate(all="ignore"):
            term = scale * expression.evaluate(columns)
        if self.sign is not None:
            got = _sign_violation(term, self.sign)
            verdict.checks.append({"check": "sign", "ok": got is None,
                                   "detail": got or f"the term is {self.sign} over all {term.size} samples"})
        if self.limits:
            mask = np.ones(term.size, dtype=bool)
            for name, value in self.limits.items():
                column = np.asarray(columns[name], dtype=float)
                tol = self.limit_tol * max(1.0, float(np.max(np.abs(column))))
                mask &= np.abs(column - float(value)) <= tol
            if mask.any():
                worst = float(np.max(np.abs(term[mask])))
                size = float(np.max(np.abs(term))) or 1.0
                ok = worst <= 1e-6 * size
                at = f"{dict(self.limits)} ({int(mask.sum())} samples)"
                detail = (f"the term vanishes where {at}, largest {worst:.3g}" if ok else
                          f"the term is {worst:.3g} where {at}: it has to vanish there")
                verdict.checks.append({"check": "limit", "ok": ok, "detail": detail})
            else:
                verdict.checks.append({"check": "limit", "ok": True,
                                       "detail": f"no sample has {dict(self.limits)}: the limit was not tested"})
        if self.symmetry:
            for name, kind in self.symmetry.items():
                other = dict(columns)
                other[name] = -np.asarray(columns[name], dtype=float)
                try:
                    mirrored = scale * expression.evaluate(other)
                except KeyError:
                    verdict.checks.append({"check": f"symmetry:{name}", "ok": False,
                                           "detail": f"{name!r} is not a column of the data"})
                    continue
                expected = term if kind == "even" else -term
                size = float(np.max(np.abs(term))) or 1.0
                worst = float(np.max(np.abs(mirrored - expected))) / size
                verdict.checks.append({"check": f"symmetry:{name}", "ok": worst <= 1e-6,
                                       "detail": (f"the term is {kind} in {name} (largest departure {worst:.2e} of "
                                                  f"its size)") if worst <= 1e-6 else
                                                 (f"the term is not {kind} in {name}: negating it changes the result by "
                                                  f"{worst:.2%} of its size")})
        if self.balance is not None:
            want, tol = float(self.balance[0]), float(self.balance[1])
            w = np.ones(term.size) if self.weights is None else np.asarray(self.weights, dtype=float)
            if w.shape != term.shape:
                raise PolicyViolation(f"the balance weights are {w.shape}, the data is {term.shape}")
            got = float(np.sum(term * w))
            ok = abs(got - want) <= tol
            verdict.checks.append({"check": "balance", "ok": ok,
                                   "detail": (f"the term integrates to {got:.6g} against {want:.6g} "
                                              f"(allowed +-{tol:.3g})")})
        return verdict


def _sign_violation(term: np.ndarray, sign: str) -> str | None:
    """``None`` when ``term`` has the requested sign everywhere, else the problem in words."""
    if sign == ">=0":
        worst = float(np.min(term))
        if worst < -1e-9 * max(1.0, float(np.max(np.abs(term)))):
            return f"the term is {worst:.3g} at its most negative: it has to be >= 0 everywhere"
        return None
    worst = float(np.max(term))
    if worst > 1e-9 * max(1.0, float(np.max(np.abs(term)))):
        return f"the term is {worst:.3g} at its most positive: it has to be <= 0 everywhere"
    return None


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------

@dataclass
class SymbolicRegressionResult:
    """One run: the Pareto front of error against complexity, the chosen expression and how long it took.

    ``expressions`` is the front, each entry ``{"expression", "scale", "relative_error", "nodes", "depth",
    "verdict", "feasible"}``, ordered by error. :attr:`best` is the lowest-error **feasible** member (``None`` when
    the filter refused every expression it found -- which is a result, and is reported rather than hidden).
    """

    expressions: list[dict[str, Any]]
    population: int
    generations: int
    seconds: float
    seed: int
    checks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def feasible(self) -> list[dict[str, Any]]:
        return [e for e in self.expressions if e["feasible"]]

    @property
    def best(self) -> dict[str, Any] | None:
        return self.feasible[0] if self.feasible else None

    @property
    def refused(self) -> list[dict[str, Any]]:
        return [e for e in self.expressions if not e["feasible"]]

    @property
    def expression(self) -> str:
        best = self.best
        return "0" if best is None else best["expression"]

    def equation(self, lhs: str = "term") -> str:
        best = self.best
        if best is None:
            return f"{lhs} = <refused by the physics filter>"
        return f"{lhs} = {_scaled_text(best)}"

    def to_dict(self) -> dict[str, Any]:
        return {"expressions": self.expressions, "best": self.best, "population": self.population,
                "generations": self.generations, "seconds": self.seconds, "seed": self.seed,
                "checks": self.checks, "equation": self.equation()}

    def __str__(self) -> str:
        lines = [self.equation(), f"{len(self.expressions)} on the Pareto front, "
                                  f"{self.population} x {self.generations} in {self.seconds:.2f} s"]
        for entry in self.expressions[:5]:
            lines.append(f"  err {entry['relative_error']:.3e}  nodes {entry['nodes']:2d}  "
                         f"{'ok' if entry['feasible'] else 'refused'}  {_scaled_text(entry)}")
        return "\n".join(lines)


def _scaled_text(entry: Mapping[str, Any]) -> str:
    scale = float(entry.get("scale", 1.0))
    return entry["expression"] if abs(scale - 1.0) <= 1e-9 else f"{scale:.6g} * {entry['expression']}"


class GeneticProgramming:
    """Symbolic regression by genetic programming, with the constants fitted and the units enforced.

    Parameters
    ----------
    population, generations
        the size and the number of generations of every run.
    operators
        the unary functions a tree may use (``"sin"``, ``"cos"``, ``"exp"``, ``"log"``, ``"sqrt"``, ``"tanh"``) on
        top of ``+ - * /`` and the whole-number powers 2 and 3.
    max_depth, max_nodes
        the bloat guard: a child deeper or larger than this is replaced by its parent.
    crossover, mutation
        the probabilities of subtree crossover, subtree mutation and constant jitter (which sum to at most 1; the
        rest of the population is copied).
    tournament
        the tournament size of the binary-tournament mating selection.
    constraints
        a :class:`PhysicsConstraints`; a child that breaks the dimensional check is not bred, and the soft checks
        (sign, limit, balance) are added to the selection error of every individual, so the search itself is steered
        to physically admissible expressions.
    max_front
        how many expressions of the final Pareto front are simplified and reported (the front itself is kept whole
        for the choice; a hundred spellings of the same formula are one entry).
    seed
        the seed of the generator, so a run is reproducible.
    """

    def __init__(self, *, population: int = 200, generations: int = 30, operators: Sequence[str] = ("sin", "cos",
                 "exp", "log", "sqrt", "tanh"), max_depth: int = 6, max_nodes: int = 40, crossover: float = 0.7,
                 mutation: float = 0.2, point: float = 0.1, tournament: int = 3, constraints: PhysicsConstraints
                 | None = None, seed: int = 0, max_front: int = 25):
        unknown = set(operators) - set(_EVAL)
        if unknown:
            raise PolicyViolation(f"unknown operator(s) {sorted(unknown)}: the engine knows "
                                  f"{sorted(set(_EVAL) - {'+' , '-', '*', '/'})}")
        self.population = int(population)
        self.generations = int(generations)
        self.operators = tuple(operators)
        self.max_depth = int(max_depth)
        self.max_nodes = int(max_nodes)
        self.crossover = float(crossover)
        self.mutation = float(mutation)
        self.point = float(point)
        self.tournament = int(tournament)
        self.constraints = constraints or PhysicsConstraints()
        self.seed = int(seed)
        self.max_front = int(max_front)

    # -- random trees ---------------------------------------------------------
    def _random_tree(self, rng: np.random.Generator, names: Sequence[str], depth: int) -> Expression:
        if depth <= 1 or rng.random() < 0.25:
            if rng.random() < 0.85:
                return Expression.variable(names[rng.integers(len(names))])
            return Expression.constant(float(np.round(rng.uniform(0.2, 2.0), 3)))
        kind = rng.random()
        if kind < 0.45:
            op = _BINARY[rng.integers(len(_BINARY))]
            return Expression(op, (self._random_tree(rng, names, depth - 1),
                                   self._random_tree(rng, names, depth - 1)))
        if kind < 0.55:
            return Expression("^", (self._random_tree(rng, names, depth - 1),), power=int(rng.choice(_POWERS)))
        op = self.operators[rng.integers(len(self.operators))] if self.operators else "*"
        return Expression(op, (self._random_tree(rng, names, depth - 1),))

    def _bred(self, tree: Expression, rng: np.random.Generator, names: Sequence[str]) -> Expression | None:
        """``tree`` if its units are admissible, else ``None`` (the caller keeps the parent)."""
        if tree.nodes() > self.max_nodes or tree.depth() > self.max_depth:
            return None
        if not self.constraints.dimension_ok(tree)[0]:
            return None
        return tree

    # -- variation ------------------------------------------------------------
    def _cross(self, a: Expression, b: Expression, rng: np.random.Generator, names: Sequence[str]) -> Expression:
        na, nb = a.nodes(), b.nodes()
        donor = b.subtree_at(int(rng.integers(nb)))
        if donor.nodes() >= na + self.max_nodes // 2:
            return a
        child, _ = a.replace_at(int(rng.integers(na)), donor)
        return self._bred(child, rng, names) or a

    def _mutate(self, a: Expression, rng: np.random.Generator, names: Sequence[str]) -> Expression:
        fresh = self._random_tree(rng, names, int(rng.integers(1, max(2, self.max_depth - 1))))
        child, _ = a.replace_at(int(rng.integers(a.nodes())), fresh)
        return self._bred(child, rng, names) or a

    def _jitter(self, a: Expression, rng: np.random.Generator) -> Expression:
        consts = [i for i, node in enumerate(a.walk()) if node.op == "const"]
        if not consts:
            return a
        index = consts[int(rng.integers(len(consts)))]
        node = a.subtree_at(index)
        value = float(np.clip(node.value * math.exp(rng.normal(0.0, 0.4)), 1e-3, 1e3))
        child, _ = a.replace_at(index, Expression.constant(value))
        return child

    # -- scoring --------------------------------------------------------------
    @staticmethod
    def _scale(f: np.ndarray, y: np.ndarray) -> float:
        """The least-squares prefactor of ``f`` for ``y`` (linear scaling, Keijzer 2003)."""
        denominator = float(np.dot(f, f))
        if not np.isfinite(denominator) or denominator < 1e-300:
            return 0.0
        return float(np.dot(f, y) / denominator)

    def _error(self, tree: Expression, columns: Mapping[str, np.ndarray],
               y: np.ndarray) -> tuple[float, float, np.ndarray | None]:
        """``(relative RMS error of scale*tree, scale, scale*tree)``; a non-finite prediction scores worst."""
        try:
            f = tree.evaluate(columns)
        except (KeyError, ValueError, FloatingPointError):
            return float("inf"), 0.0, None
        if not np.all(np.isfinite(f)):
            return float("inf"), 0.0, None
        scale = self._scale(f, y)
        if not np.isfinite(scale):
            return float("inf"), 0.0, None
        term = scale * f
        residual = np.sqrt(np.mean((term - y) ** 2))
        return float(residual / (np.sqrt(np.mean(y ** 2)) or 1.0)), scale, term

    # -- NSGA-II --------------------------------------------------------------
    @staticmethod
    def _dominance(objectives: np.ndarray) -> np.ndarray:
        """``dom[i, j]``: member ``i`` dominates member ``j`` (no worse on any objective, better on one)."""
        at_most = (objectives[:, None, :] <= objectives[None, :, :]).all(axis=2)
        better = (objectives[:, None, :] < objectives[None, :, :]).any(axis=2)
        return at_most & better

    @classmethod
    def _non_dominated(cls, objectives: np.ndarray) -> list[int]:
        """The indices of the members no other member dominates."""
        if len(objectives) == 0:
            return []
        return [int(i) for i in np.flatnonzero(~cls._dominance(objectives).any(axis=0))]

    @classmethod
    def _nsga2_survive(cls, objectives: np.ndarray, size: int) -> np.ndarray:
        """NSGA-II: fill the next generation front by front, the last one cut by crowding distance."""
        n = len(objectives)
        if n <= size:
            return np.arange(n)
        dom = cls._dominance(objectives)
        counts = dom.sum(axis=0)                          # how many members dominate each one
        out: list[int] = []
        alive = np.ones(n, dtype=bool)
        while len(out) < size:
            front = np.flatnonzero(alive & (counts == 0))
            if front.size == 0:                           # cannot happen with a strict partial order
                break
            if len(out) + front.size <= size:
                out.extend(int(i) for i in front)
            else:
                crowd = _crowding(objectives[front])
                order = np.argsort(-crowd, kind="stable")
                out.extend(int(front[i]) for i in order[: size - len(out)])
                break
            alive[front] = False
            counts = counts - dom[front].sum(axis=0)
        return np.asarray(out, dtype=int)

    # -- fit ------------------------------------------------------------------
    def fit(self, X: Any, y: Any, *, names: Sequence[str] | None = None) -> SymbolicRegressionResult:
        """Search expressions ``f`` with ``y ~ scale * f(X)``.

        ``X`` is an array ``(N, d)`` (or a mapping name -> column); ``names`` labels its columns when it is an array.
        Returns a :class:`SymbolicRegressionResult`; the run is deterministic in ``seed``.
        """
        columns = _as_columns(X, names)
        names = list(columns)
        target = np.asarray(y, dtype=float).ravel()
        if target.size != len(next(iter(columns.values()))):
            raise ValueError(f"X has {len(next(iter(columns.values())))} rows and y has {target.size}")
        if not np.all(np.isfinite(target)):
            raise ValueError("y has non-finite values; drop or replace them before the search")
        rng = np.random.default_rng(self.seed)
        self._cached_limit_mask = None
        start = time.perf_counter()
        members = [self._random_tree(rng, names, int(rng.integers(2, self.max_depth))) for _ in
                   range(self.population)]
        members = [m for m in members if self.constraints.dimension_ok(m)[0]] or members
        while len(members) < self.population:            # a dimensionally admissible initial population
            extra = self._random_tree(rng, names, int(rng.integers(2, self.max_depth)))
            if self.constraints.dimension_ok(extra)[0]:
                members.append(extra)
        scored = [self._error(m, columns, target) for m in members]
        for _ in range(self.generations):
            errors = [s[0] for s in scored]
            children: list[Expression] = []
            for _ in range(self.population):
                parent = self._tournament(members, errors, rng)
                draw = rng.random()
                if draw < self.crossover:
                    other = self._tournament(members, errors, rng)
                    children.append(self._cross(parent, other, rng, names))
                elif draw < self.crossover + self.mutation:
                    children.append(self._mutate(parent, rng, names))
                elif draw < self.crossover + self.mutation + self.point:
                    children.append(self._jitter(parent, rng))
                else:
                    children.append(parent)
            scored_children = [self._error(c, columns, target) for c in children]
            pool = members + children
            pool_scored = scored + scored_children
            fitness = np.array([[s[0] + self._penalty(columns, s[1], s[2]), float(c.nodes())]
                                for c, s in zip(pool, pool_scored, strict=True)])
            keep = self._nsga2_survive(_rankable(fitness), self.population)
            members = [pool[i] for i in keep]
            scored = [pool_scored[i] for i in keep]
        return self._result(members, scored, columns, target, start)

    def _tournament(self, members: Sequence[Expression], errors: Sequence[float],
                    rng: np.random.Generator) -> Expression:
        picks = rng.integers(0, len(members), size=max(2, self.tournament))
        best = min(picks, key=lambda i: (errors[i], members[i].nodes()))
        return members[int(best)]

    def _penalty(self, columns: Mapping[str, np.ndarray], scale: float, term: np.ndarray | None) -> float:
        """How far the soft constraints are from being satisfied, as error the selection can see.

        Only the constraints that read the prediction are used here (sign, limit, balance); the dimensional check is
        already excluded from the search, and the symmetry check needs a second evaluation, so it is applied to the
        front only."""
        constraints = self.constraints
        if not constraints.active or term is None or scale == 0.0:
            return 0.0
        penalty = 0.0
        if constraints.sign and _sign_violation(term, constraints.sign):
            penalty += 1.0
        if constraints.limits:
            mask = self._limit_mask(columns, term.size)
            if mask is not None and mask.any():
                size = float(np.max(np.abs(term))) or 1.0
                if float(np.max(np.abs(term[mask]))) > 1e-6 * size:
                    penalty += 1.0
        if constraints.balance is not None:
            want, tol = constraints.balance
            w = np.ones(term.size) if constraints.weights is None else np.asarray(constraints.weights, dtype=float)
            if abs(float(np.sum(term * w)) - float(want)) > float(tol):
                penalty += 1.0
        return penalty

    def _limit_mask(self, columns: Mapping[str, np.ndarray], size: int) -> np.ndarray | None:
        """The rows where every column named in ``limits`` takes its value (cached: it does not depend on the tree)."""
        if getattr(self, "_cached_limit_mask", None) is not None:
            return self._cached_limit_mask
        mask = np.ones(size, dtype=bool)
        for name, value in dict(self.constraints.limits or {}).items():
            column = np.asarray(columns[name], dtype=float)
            tol = self.constraints.limit_tol * max(1.0, float(np.max(np.abs(column))))
            mask &= np.abs(column - float(value)) <= tol
        self._cached_limit_mask: np.ndarray | None = mask
        return mask

    def _result(self, members: Sequence[Expression], scored: Sequence[tuple],
                columns: Mapping[str, np.ndarray], target: np.ndarray, start: float) -> SymbolicRegressionResult:
        objectives = np.array([[s[0], float(m.nodes())] for m, s in zip(members, scored, strict=True)])
        front = self._non_dominated(_rankable(objectives))
        entries = []
        seen: dict[str, int] = {}
        for i in sorted(front, key=lambda k: (objectives[k][0], objectives[k][1])):
            folded = _fold_scale(members[i], scored[i][1])
            tree = folded.simplified()
            error, scale, _term = self._error(tree, columns, target)
            key = f"{tree.to_string()}|{scale:.9g}"
            if key in seen:                                # the same formula written twice is one entry
                continue
            verdict = tree.check(self.constraints, columns, scale=scale)
            entry = {"expression": tree.to_string(), "latex": tree.latex(), "scale": scale,
                     "relative_error": error, "nodes": tree.nodes(), "depth": tree.depth(),
                     "verdict": verdict.to_dict(), "feasible": verdict.ok, "reasons": verdict.reasons}
            seen[key] = len(entries)
            entries.append(entry)
            if len(entries) >= self.max_front:
                break
        if len(entries) > 1:                               # simplifying changes the node count, so the front is
            objectives_out = np.array([[e["relative_error"], float(e["nodes"])] for e in entries])   # re-taken
            entries = [entries[i] for i in self._non_dominated(_rankable(objectives_out))]
        seconds = time.perf_counter() - start
        checks = _regression_checks(entries, target)
        return SymbolicRegressionResult(expressions=entries, population=self.population,
                                        generations=self.generations, seconds=seconds, seed=self.seed,
                                        checks=checks)


def _fold_scale(tree: Expression, scale: float) -> Expression:
    """``scale * tree`` as one tree, so the simplification folds the prefactor into the constant it multiplies."""
    if not np.isfinite(scale) or scale == 0.0:
        return tree
    if tree.op == "*" and tree.children and tree.children[0].op == "const":
        return tree.with_children(Expression.constant(tree.children[0].value * scale), *tree.children[1:])
    return Expression("*", (Expression.constant(scale), tree))


def _rankable(objectives: np.ndarray) -> np.ndarray:
    """Objectives with ``inf`` mapped to a large finite number, so dominance and crowding are well defined."""
    out = np.array(objectives, dtype=float)
    finite = np.isfinite(out)
    if not finite.all():
        worst = float(np.max(out[finite])) if finite.any() else 1.0
        out[~finite] = worst * 10.0 + 1.0
    return out


def _crowding(objectives: np.ndarray) -> np.ndarray:
    """NSGA-II crowding distance of one front."""
    n = len(objectives)
    distance = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for column in range(objectives.shape[1]):
        order = np.argsort(objectives[:, column])
        values = objectives[order, column]
        span = float(values[-1] - values[0])
        distance[order[0]] = distance[order[-1]] = np.inf
        if span <= 0:
            continue
        for k in range(1, n - 1):
            distance[order[k]] += float(values[k + 1] - values[k - 1]) / span
    return distance


def _regression_checks(entries: Sequence[Mapping[str, Any]], target: np.ndarray) -> list[dict[str, Any]]:
    """The two checks a discovered expression owes the reader: it fits, and the filter did not empty the front."""
    if not entries:
        return [{"code": "gp.front", "status": "fail", "detail": "the search found no expression", "value": None}]
    feasible = [e for e in entries if e["feasible"]]
    best = feasible[0] if feasible else entries[0]
    fits = best["relative_error"] <= 0.1
    return [
        {"code": "gp.fit", "status": "pass" if fits else "fail",
         "detail": f"the reported expression predicts the data to {best['relative_error']:.2%} of its RMS",
         "value": best["relative_error"]},
        {"code": "gp.physics", "status": "pass" if feasible else "fail",
         "detail": (f"{len(feasible)} of {len(entries)} expressions on the Pareto front pass the physics filter"
                    if feasible else "the physics filter refused every expression on the Pareto front: "
                    + "; ".join(entries[0]["reasons"])),
         "value": len(feasible)},
        {"code": "gp.pareto", "status": "pass" if len(entries) >= 1 else "fail",
         "detail": f"{len(entries)} expressions on the front, {min(e['nodes'] for e in entries)} to "
                   f"{max(e['nodes'] for e in entries)} nodes", "value": len(entries)},
    ]


def _as_columns(X: Any, names: Sequence[str] | None) -> dict[str, np.ndarray]:
    if isinstance(X, Mapping):
        return {str(k): np.asarray(v, dtype=float).ravel() for k, v in X.items()}
    array = np.asarray(X, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    labels = list(names) if names is not None else [f"x{i}" for i in range(array.shape[1])]
    if len(labels) != array.shape[1]:
        raise ValueError(f"{array.shape[1]} columns but {len(labels)} names")
    return {label: array[:, i] for i, label in enumerate(labels)}


# ---------------------------------------------------------------------------
# the two public workflows
# ---------------------------------------------------------------------------

def discover_expression(X: Any, y: Any, *, names: Sequence[str] | None = None,
                        constraints: PhysicsConstraints | None = None, population: int = 150,
                        generations: int = 20, seed: int = 0, **kwargs: Any) -> SymbolicRegressionResult:
    """Discover ``y = scale * f(X)`` by genetic programming, filtered by the physics.

    ``X`` is ``(N, d)`` or a mapping ``name -> column``; ``y`` is the term to explain (a residual, a closure, a
    drag coefficient). ``constraints`` is the :class:`PhysicsConstraints` the answer has to respect; the dimensional
    check is enforced while the trees are bred, the rest decide which expression is reported. The remaining keyword
    arguments go to :class:`GeneticProgramming`. Every number the run produced is on the result: the Pareto front,
    the time, and the checks each expression passed or failed.

    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> u = rng.uniform(0.1, 2.0, 400)                                    # a concentration, mol/m^3
    >>> y = -0.7 * u ** 2                                                 # a second-order sink
    >>> result = discover_expression({"u": u}, y, constraints=PhysicsConstraints(
    ...     units={"u": "mol/m^3"}, target="mol/(m^3*s)", prefactor="m^3/(mol*s)", sign="<=0"),
    ...     population=120, generations=12, seed=0)
    >>> result.best["relative_error"] < 1e-6                              # doctest: +SKIP
    True
    """
    engine = GeneticProgramming(population=population, generations=generations, seed=seed,
                                constraints=constraints, **kwargs)
    return engine.fit(X, y, names=names)


def compare_gp_and_sindy(X: Any, y: Any, *, names: Sequence[str] | None = None,
                         library: Any = None, constraints: PhysicsConstraints | None = None,
                         truth: Expression | str | None = None, population: int = 150,
                         generations: int = 20, seed: int = 0, **kwargs: Any) -> dict[str, Any]:
    """Run the same problem through genetic programming and through SINDy, and put the numbers side by side.

    SINDy gets the library built by :class:`~pinneapple_analysis.inverse_problems.missing_term.CandidateLibrary`
    (``library`` may be one) over the same columns and the same target, and is fitted by
    :class:`~pinneapple_analysis.inverse_problems.missing_term.SINDyIdentifier` -- the sparse regression of the
    missing-term workflow, not a second implementation of it. Genetic programming gets no library: it searches.

    ``truth`` (an :class:`Expression` or its text) is what both arms are scored against when it is known:
    ``recovered`` says whether the discovered form is a constant multiple of it, and ``coefficient`` the factor
    against the true one. Returns a dict with ``gp`` and ``sindy`` arms -- relative error on the data, number of
    terms, seconds -- and the ``comparison`` line and checks that say which one won and on what.
    """
    from .missing_term import CandidateLibrary, SINDyIdentifier

    columns = _as_columns(X, names)
    labels = list(columns)
    target = np.asarray(y, dtype=float).ravel()
    matrix = np.column_stack([columns[k] for k in labels])
    reference = None if truth is None else (truth if isinstance(truth, Expression) else parse_expression(truth))

    start = time.perf_counter()
    engine = GeneticProgramming(population=population, generations=generations, seed=seed, constraints=constraints,
                                **kwargs)
    gp_result = engine.fit(columns, target)
    gp_seconds = time.perf_counter() - start

    builder = library if library is not None else CandidateLibrary(poly_order=2)
    start = time.perf_counter()
    theta, term_names = builder.build(matrix)
    sindy = SINDyIdentifier().fit(theta, target, term_names=term_names)
    sindy_seconds = time.perf_counter() - start
    sindy_prediction = theta @ sindy.coefficients
    sindy_error = float(np.sqrt(np.mean((sindy_prediction - target) ** 2))
                        / (np.sqrt(np.mean(target ** 2)) or 1.0))
    gp_best = gp_result.best
    gp_error = float("inf") if gp_best is None else float(gp_best["relative_error"])

    arms = {
        "gp": {"relative_error": gp_error, "n_terms": 0 if gp_best is None else gp_best["nodes"],
               "expression": gp_result.equation(), "seconds": gp_seconds,
               "recovered": None if reference is None else _recovers(gp_best, reference, columns)},
        "sindy": {"relative_error": sindy_error, "n_terms": int(sindy.n_terms_selected),
                  "expression": sindy.equation(), "seconds": sindy_seconds,
                  "recovered": None if reference is None else _sindy_recovers(builder, sindy.coefficients, labels,
                                                                            reference, columns)},
    }
    winner = "gp" if arms["gp"]["relative_error"] <= arms["sindy"]["relative_error"] else "sindy"
    if gp_best is None:                                    # nothing feasible: SINDy is the only arm with an answer
        winner = "sindy"
    cost_ratio = arms["gp"]["seconds"] / max(arms["sindy"]["seconds"], 1e-9)
    checks = [
        {"code": "compare.error", "status": "pass" if min(arms["gp"]["relative_error"],
                                                         arms["sindy"]["relative_error"]) <= 0.1 else "fail",
         "detail": f"best of the two arms predicts the term to {min(arms['gp']['relative_error'], arms['sindy']['relative_error']):.2%}",
         "value": min(arms["gp"]["relative_error"], arms["sindy"]["relative_error"])},
        {"code": "compare.recovery", "status": "pass" if reference is None
         or arms["gp"]["recovered"] or arms["sindy"]["recovered"] else "fail",
         "detail": ("no truth given" if reference is None else
                    f"the true expression was recovered by "
                    f"{', '.join(a for a in ('gp', 'sindy') if arms[a]['recovered']) or 'neither arm'}"),
         "value": None},
    ]
    return {"gp": arms["gp"], "sindy": arms["sindy"], "winner": winner, "cost_ratio": cost_ratio,
            "comparison": (f"GP {arms['gp']['relative_error']:.2%} / {arms['gp']['n_terms']} nodes / "
                           f"{arms['gp']['seconds']:.2f} s  vs  SINDy {arms['sindy']['relative_error']:.2%} / "
                           f"{arms['sindy']['n_terms']} terms / {arms['sindy']['seconds']:.2f} s"),
            "checks": checks, "gp_result": gp_result, "sindy_result": sindy}


def parse_expression(text: str) -> Expression:
    """Read an expression written the way :meth:`Expression.to_string` writes it into an :class:`Expression`.

    The text has to pass :func:`pinneapple_data.expressions.check` first, so a caller can hand it a formula from a
    file or a request without giving it a way into ``eval``."""
    from pinneapple_data import expressions
    names = sorted(expressions.names_in(text))
    expressions.check(text, names=names, functions=[f for f in _EVAL if f.isalpha()])
    tree = ast.parse(text.strip(), mode="eval").body
    return _from_ast(tree)


_AST_BINOPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
_AST_UNARY = {"sin": "sin", "cos": "cos", "exp": "exp", "log": "log", "sqrt": "sqrt", "tanh": "tanh"}


def _from_ast(node) -> Expression:
    if isinstance(node, ast.Expression):
        return _from_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Expression.constant(float(node.value))
    if isinstance(node, ast.Name):
        return Expression.variable(node.id)
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Pow):
            if not (isinstance(node.right, ast.Constant) and node.right.value in _POWERS):
                raise PolicyViolation("only the whole-number powers 2 and 3 are part of the expression language")
            return Expression("^", (_from_ast(node.left),), power=int(node.right.value))
        return Expression(_AST_BINOPS[type(node.op)], (_from_ast(node.left), _from_ast(node.right)))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return Expression("*", (Expression.constant(-1.0), _from_ast(node.operand)))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _AST_UNARY:
        return Expression(_AST_UNARY[node.func.id], (_from_ast(node.args[0]),))
    raise PolicyViolation(f"{type(node).__name__} is not part of the expression language")


def _recovers(entry: Mapping[str, Any] | None, truth: Expression, columns: Mapping[str, np.ndarray]) -> bool:
    """Did the genetic engine report a constant multiple of the truth, with that constant right?"""
    if entry is None:
        return False
    try:
        candidate = parse_expression(entry["expression"])
    except (PolicyViolation, SyntaxError, ValueError):
        return False
    scale = float(entry["scale"])

    def predict(target: Mapping[str, np.ndarray]) -> np.ndarray:
        return scale * candidate.evaluate(target)

    return _same_form(predict, truth, columns)


def _sindy_recovers(builder: Any, coefficients: np.ndarray, labels: Sequence[str], truth: Expression,
                    columns: Mapping[str, np.ndarray]) -> bool:
    """Did SINDy's selected library reproduce the truth -- by the test the genetic arm is held to?"""
    def predict(target: Mapping[str, np.ndarray]) -> np.ndarray:
        matrix = np.column_stack([np.asarray(target[k], dtype=float) for k in labels])
        theta, _names = builder.build(matrix)
        return theta @ coefficients

    return _same_form(predict, truth, columns)


def _same_form(predict: Any, truth: Expression, columns: Mapping[str, np.ndarray]) -> bool:
    """Is ``predict`` a constant multiple of the truth -- on the data **and** outside it -- with the factor 1?

    The ratio of the two models over the data has to be flat and one to within 5 %, which rejects a library fit
    whose terms are wrong even when its curve passes near the truth (three polynomial terms trace a third-order
    sink over the sampled range without being it). And the same has to hold past the data's range, from a quarter
    of the smallest value to four times the largest: a polynomial that imitates ``sqrt(u)`` between 0.3 and
    1.5 mol/m^3 is not ``sqrt(u)``, and the extrapolation is what says so. Recovery means the form was found, not
    that the curve was followed.

    The points where the truth is smaller than 1 % of its own peak are left out of the ratio: a spurious constant
    of 1e-4 next to a term of order one is a wrong term, but it does not make the *form* wrong, and at the far end
    of the extension the truth itself is that small."""
    for target in (columns, _extended(columns)):
        try:
            exact = truth.evaluate(target)
            predicted = np.asarray(predict(target), dtype=float)
        except (KeyError, ValueError, FloatingPointError):
            return False
        if predicted.shape != exact.shape:
            return False
        if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(exact)):
            return False
        size = float(np.max(np.abs(exact)))
        if size < 1e-300:
            return False
        mask = np.abs(exact) > 1e-2 * size
        if int(mask.sum()) < 10:
            return False
        ratio = predicted[mask] / exact[mask]
        if float(np.max(np.abs(ratio - 1.0))) > 5e-2 or float(np.max(ratio) - np.min(ratio)) > 5e-2:
            return False
        if float(np.sqrt(np.mean((predicted - exact) ** 2)) / np.sqrt(np.mean(exact ** 2))) > 5e-2:
            return False
    return True


def _extended(columns: Mapping[str, np.ndarray], n: int = 41) -> dict[str, np.ndarray]:
    """The data's columns with, appended, a grid that runs past them on both sides.

    Each column is extended on its own: the test is of the *form*, so the rows the extension makes need not be
    physically consistent (the same expression is compared at the same points for both models)."""
    out: dict[str, np.ndarray] = {}
    for name, values in columns.items():
        values = np.asarray(values, dtype=float).ravel()
        low, high = float(np.min(values)), float(np.max(values))
        span = high - low
        if span <= 0:
            out[name] = values
            continue
        below = max(low - span, 0.25 * low) if low > 0 else low - span
        out[name] = np.concatenate([values, np.linspace(below, high + 3.0 * span, n)])
    return out


# ---------------------------------------------------------------------------
# the benchmark case: a known reaction term in a field the library already solves
# ---------------------------------------------------------------------------

def advection_diffusion_reaction_case(*, order: int = 2, rate: float = 0.7, diffusivity: float = 0.05,
                                      advection: float = 1.0, nx: int = 96, nt: int = 41,
                                      t_end: float = 1.0) -> dict[str, Any]:
    """A periodic 1-D field obeying ``u_t + a u_x = D u_xx + R(u)`` with ``R(u) = -rate * u**order``, and the
    column every discovery method has to explain.

    The field is integrated with spectral derivatives and an adaptive Runge-Kutta method (DOP853, ``rtol`` 1e-10),
    so the truth is exact to solver tolerance and the expression is known by construction. ``u`` is a concentration
    (mol/m^3) and the time is in seconds.

    What comes back:

    ``u``, ``x``, ``t``
        the field ``(nt, nx)`` and its coordinates.
    ``columns``
        ``{"u": ..., "u_t": ..., "u_x": ..., "u_xx": ...}`` on the interior of the grid, finite differences
        (``np.gradient``, second order -- the same operator PDE-FIND uses in
        :func:`~pinneapple_analysis.inverse_problems.discovery.discover_pde`), and ``"residual"``: the part of
        ``u_t + a u_x - D u_xx`` the reaction has to explain. That residual carries the finite-difference error,
        which is the floor any discovery on this data works against -- it is measured, not assumed, and returned in
        ``residual_floor``.
    ``truth``
        :class:`Expression` ``-rate * u**order``, and ``expression`` its text.
    ``prefactor``, ``constraints``
        the unit the rate constant of that order carries (mol/m^3 over a second, so ``m^3/(mol*s)`` for a
        second-order sink and ``m^6/(mol^2*s)`` for a third-order one), and the
        :class:`PhysicsConstraints` that goes with it -- the rate must vanish with the concentration and be a
        sink. A half-integer order has no such unit (``sqrt(mol/m^3)/s`` is not a unit this library writes), so
        ``prefactor`` is ``None`` there and the filter checks only that the terms of the expression agree.
    ``checks``
        whether the solver reproduces the library's exact linear solution when ``rate = 0``, whether the field
        stays positive, and the size of the finite-difference floor.
    """
    from scipy.integrate import solve_ivp

    from pinneapple_physics.advection_diffusion_1d import AdvectionDiffusion1D

    x = 2 * np.pi * np.arange(nx) / nx
    t = np.linspace(0.0, t_end, nt)
    k = np.fft.fftfreq(nx, d=1.0 / nx)
    # fixed and smooth: this case is about the term, not about a random initial field
    u0 = 1.0 + 0.5 * np.sin(x + 0.3) + 0.2 * np.cos(2 * x + 0.1) + 0.1 * np.sin(3 * x)

    def rhs(_t, v):
        vhat = np.fft.fft(v)
        dvdx = np.real(np.fft.ifft(1j * k * vhat))
        d2vdx2 = np.real(np.fft.ifft(-(k ** 2) * vhat))
        return diffusivity * d2vdx2 - advection * dvdx - rate * np.power(np.abs(v), order - 1) * v

    solution = solve_ivp(rhs, (t[0], t[-1]), u0, t_eval=t, method="DOP853", rtol=1e-10, atol=1e-12)
    if not solution.success:
        raise RuntimeError(f"the field could not be integrated: {solution.message}")
    u = solution.y.T                                        # (nt, nx)

    # the same integration without the reaction, against the library's exact Fourier solution
    linear = AdvectionDiffusion1D(nx=nx, nt=nt, t_end=t_end)
    exact = linear.exact({"u0": u0, "nu": diffusivity, "c": advection})
    no_reaction = solve_ivp(lambda tt, v: rhs(tt, v) + rate * np.power(np.abs(v), order - 1) * v,
                            (t[0], t[-1]), u0, t_eval=t, method="DOP853", rtol=1e-10, atol=1e-12)
    linear_error = float(np.max(np.abs(no_reaction.y.T - exact)) / np.max(np.abs(exact)))

    dx, dt = float(x[1] - x[0]), float(t[1] - t[0])
    ut = np.gradient(u, dt, axis=0, edge_order=2)
    ux = np.gradient(u, dx, axis=1, edge_order=2)
    uxx = np.gradient(ux, dx, axis=1, edge_order=2)
    trim = 3
    sl = (slice(trim, -trim), slice(trim, -trim))
    residual = ut[sl] + advection * ux[sl] - diffusivity * uxx[sl]
    columns = {"u": u[sl].T.ravel(), "u_t": ut[sl].T.ravel(), "u_x": ux[sl].T.ravel(),
               "u_xx": uxx[sl].T.ravel(), "residual": residual.T.ravel()}
    exact_term = -rate * np.power(u[sl].T.ravel(), order)
    floor = float(np.sqrt(np.mean((residual.T.ravel() - exact_term) ** 2))
                  / (np.sqrt(np.mean(exact_term ** 2)) or 1.0))
    checks = [
        {"code": "adr.linear_solver", "status": "pass" if linear_error <= 1e-6 else "fail",
         "detail": f"with rate=0 the integrator reproduces the library's exact advection-diffusion solution to "
                   f"{linear_error:.2e} of its size", "value": linear_error},
        {"code": "adr.positive", "status": "pass" if float(u.min()) > 0 else "fail",
         "detail": f"the concentration stays positive (minimum {float(u.min()):.3g} mol/m^3)",
         "value": float(u.min())},
        {"code": "adr.residual_floor", "status": "pass" if floor <= 0.05 else "fail",
         "detail": f"the finite-difference residual differs from the true reaction term by {floor:.2%} of its RMS: "
                   f"that is the floor any discovery works against on this grid", "value": floor},
    ]
    try:
        truth = Expression("*", (Expression.constant(-rate),
                                 Expression("^", (Expression.variable("u"),), power=order)))
    except (TypeError, ValueError):
        raise PolicyViolation(f"order {order} is not one the expression tree holds") from None
    prefactor = None
    if isinstance(order, int) and order >= 1:
        power = 1 - order                                  # (mol/m^3)^(1-order) / s
        prefactor = "1/s" if power == 0 else (
            f"m^{3 * -power}/(mol^{-power}*s)" if power < 0 else f"m^{3 * power}/(mol^{power}*s)")
    constraints = PhysicsConstraints(
        units={"u": "mol/m^3"}, prefactor=prefactor,
        target="mol/(m^3*s)" if prefactor else None,
        sign="<=0", limits={"u": 0.0})
    return {"x": x, "t": t, "u": u, "columns": columns, "truth": truth, "expression": truth.to_string(),
            "residual_floor": floor, "checks": checks, "order": order, "rate": rate, "diffusivity": diffusivity,
            "advection": advection, "prefactor": prefactor, "constraints": constraints}


def noise_sweep(case: Mapping[str, Any], *, levels: Sequence[float] = (0.0, 0.01, 0.05, 0.10), seeds: int = 5,
                library_orders: Sequence[int] = (2, 3), population: int = 120, generations: int = 15,
                constraints: PhysicsConstraints | None = None) -> dict[str, Any]:
    """Recovery rate of the true expression against noise, for genetic programming and for SINDy's libraries.

    The target column of ``case`` (the part of ``u_t + a u_x - D u_xx`` the reaction has to explain) is the same for
    both arms; noise of ``level`` (a fraction of the target's RMS, Gaussian) is added to it, ``seeds`` times. The
    same ``constraints`` the filter is given are used in the search, so the rate that comes back is the rate of
    expressions the filter accepts.

    Returns ``{"levels": [...], "gp": {...}, "sindy": {...}}`` with, per level, the fraction of runs whose reported
    expression is a constant multiple of the truth with the constant right (:func:`_recovers`) and the median
    relative error on the clean signal. SINDy is reported per polynomial order of its library, because its recovery
    rate depends on whether the library happens to contain the term -- which is the whole reason this engine exists.
    """
    from .missing_term import CandidateLibrary, SINDyIdentifier

    columns = case["columns"]
    truth: Expression = case["truth"]
    X = {"u": columns["u"]}
    matrix = np.column_stack([X["u"]])
    clean = truth.evaluate(X)
    size = float(np.sqrt(np.mean(clean ** 2))) or 1.0
    rng = np.random.default_rng(11)
    gp_rate: dict[float, dict[str, Any]] = {}
    sindy_rate: dict[float, dict[int, dict[str, float]]] = {}
    for level in levels:
        gp_hits, gp_errors = 0, []
        sindy_hits = {order: 0 for order in library_orders}
        for _ in range(seeds):
            noisy = clean + level * size * rng.normal(size=clean.size)
            run = discover_expression(X, noisy, constraints=constraints, population=population,
                                      generations=generations, seed=int(rng.integers(1_000_000)))
            gp_hits += bool(_recovers(run.best, truth, X))
            gp_errors.append(float("inf") if run.best is None else run.best["relative_error"])
            for order in library_orders:
                one_order = CandidateLibrary(poly_order=order)
                theta, term_names = one_order.build(matrix)
                fitted = SINDyIdentifier().fit(theta, noisy, term_names=term_names)
                sindy_hits[order] += bool(_sindy_recovers(one_order, fitted.coefficients, ["u"], truth, X))
        gp_rate[level] = {"recovered": gp_hits / seeds, "median_error": float(np.median(gp_errors))}
        sindy_rate[level] = {order: {"recovered": sindy_hits[order] / seeds} for order in library_orders}
    return {"levels": list(levels), "gp": gp_rate, "sindy": sindy_rate, "residual_floor": case["residual_floor"]}

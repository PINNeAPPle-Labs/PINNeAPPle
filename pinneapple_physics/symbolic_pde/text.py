"""Equations as text, parsed safely into SymPy: ``"u_t = alpha*lap(u) + sin(pi*x)"``.

The text is read with :mod:`ast` and translated node by node into SymPy objects; nothing is handed to ``eval`` or
``sympify``. Allowed: numbers, ``+ - * / ** ^``, parentheses, coordinates, fields, coefficient names and the
functions in :data:`FUNCTIONS`, plus three operators:

==================  ===========================================================================================
``u_t``, ``u_xx``   derivative shorthand: a field, an underscore, then one coordinate letter per derivative
``d(f, x, ...)``    derivative of any expression (chain rule applied), e.g. ``d(u**2, x)`` or ``d(u, x, x)``
``lap(f)``          Laplacian over the *spatial* coordinates (so the same text works in 1-D, 2-D and 3-D)
==================  ===========================================================================================

Any other name is a coefficient: it needs a value (``coefficients={"alpha": 0.1}``) before the system is solved, or
it is declared unknown (an inverse problem). Attribute access, subscripts, keywords, strings, lambdas, comprehensions
and calls of anything not listed are refused with :class:`EquationSyntaxError`.

    sys_ = parse_system(["u_t = alpha*lap(u)"], [{"kind": "dirichlet", "on": "x_min", "equation": "u = 0"}],
                        coords=["x", "t"], fields=["u"], coefficients={"alpha": 0.1})
    sys_.classify()["family"]       # "heat"
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any

import sympy as sp
from sympy.core.function import AppliedUndef

MAX_LENGTH = 600
MAX_NODES = 400
MAX_LITERAL = 1e15
MAX_POWER = 12

FUNCTIONS: dict[str, Any] = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "abs": sp.Abs,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh, "atan": sp.atan, "asin": sp.asin, "acos": sp.acos,
    "sign": sp.sign, "heaviside": sp.Heaviside,
}
OPERATORS = ("d", "diff", "lap", "laplacian")
CONSTANTS = {"pi": sp.pi}
CONDITION_KINDS = ("dirichlet", "neumann", "robin", "initial", "periodic")
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9]*(_[A-Za-z0-9]+)*$")
_FIELD = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")


class EquationSyntaxError(ValueError):
    """The text is not an equation this parser accepts (the message says what and where)."""


@dataclass
class Context:
    coords: list[str]                       # every independent variable, e.g. ["x", "y", "t"]
    fields: list[str]                       # dependent variables, e.g. ["u"] or ["u", "v", "p"]
    time: str = "t"

    def __post_init__(self):
        for c in self.coords:
            if not re.fullmatch(r"[a-z]", c):
                raise EquationSyntaxError(f"coordinates are single lower-case letters, got {c!r}")
        for f in self.fields:
            if not _FIELD.match(f) or f in self.coords or f in FUNCTIONS or f in OPERATORS or f in CONSTANTS:
                raise EquationSyntaxError(f"invalid field name {f!r}: letters and digits, not a coordinate, function "
                                          f"or constant")
        self.symbols = {c: sp.Symbol(c, real=True) for c in self.coords}
        self.functions = {f: sp.Function(f)(*[self.symbols[c] for c in self.coords]) for f in self.fields}

    @property
    def spatial(self) -> list[str]:
        return [c for c in self.coords if c != self.time]


def _derivative_name(name: str, ctx: Context):
    """``u_xx`` -> (u, ["x", "x"]) when ``u`` is a field and every letter after ``_`` a coordinate."""
    if "_" not in name:
        return None
    head, _, tail = name.rpartition("_")
    if head in ctx.fields and tail and all(ch in ctx.coords for ch in tail):
        return head, list(tail)
    return None


class _Translator:
    def __init__(self, ctx: Context, text: str):
        self.ctx, self.text = ctx, text
        self.coefficients: set[str] = set()

    def fail(self, node, why: str):
        col = getattr(node, "col_offset", None)
        where = f" at column {col + 1}" if col is not None else ""
        raise EquationSyntaxError(f"{why}{where} in {self.text!r}")

    def __call__(self, node):  # noqa: C901 - one branch per allowed node type
        if isinstance(node, ast.Expression):
            return self(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                self.fail(node, "only numbers are allowed as literals")
            if abs(node.value) > MAX_LITERAL:
                self.fail(node, f"number larger than {MAX_LITERAL:g}")
            return sp.Integer(node.value) if isinstance(node.value, int) else sp.Float(node.value)
        if isinstance(node, ast.Name):
            return self.name(node)
        if isinstance(node, ast.UnaryOp):
            v = self(node.operand)
            if isinstance(node.op, ast.USub):
                return -v
            if isinstance(node.op, ast.UAdd):
                return v
            self.fail(node, f"operator {type(node.op).__name__} is not allowed")
        if isinstance(node, ast.BinOp):
            a, b = self(node.left), self(node.right)
            op = node.op
            if isinstance(op, ast.Add):
                return a + b
            if isinstance(op, ast.Sub):
                return a - b
            if isinstance(op, ast.Mult):
                return a * b
            if isinstance(op, ast.Div):
                return a / b
            if isinstance(op, (ast.Pow, ast.BitXor)):
                if b.is_Number and abs(float(b)) > MAX_POWER:
                    self.fail(node, f"exponents are limited to {MAX_POWER}")
                return a ** b
            self.fail(node, f"operator {type(op).__name__} is not allowed")
        if isinstance(node, ast.Call):
            return self.call(node)
        self.fail(node, f"{type(node).__name__} is not allowed in an equation")

    def name(self, node: ast.Name):
        n, ctx = node.id, self.ctx
        if n in ctx.symbols:
            return ctx.symbols[n]
        if n in ctx.functions:
            return ctx.functions[n]
        if n in CONSTANTS:
            return CONSTANTS[n]
        der = _derivative_name(n, ctx)
        if der is not None:
            f, wrt = der
            return sp.Derivative(ctx.functions[f], *[ctx.symbols[c] for c in wrt])
        if n in FUNCTIONS or n in OPERATORS:
            self.fail(node, f"{n!r} is a function: call it, e.g. {n}(x)")
        if not _NAME.match(n) or n.startswith("_"):
            self.fail(node, f"invalid name {n!r}")
        self.coefficients.add(n)
        return sp.Symbol(n, real=True)

    def call(self, node: ast.Call):
        if not isinstance(node.func, ast.Name) or node.keywords:
            self.fail(node, "only plain calls of the listed functions are allowed")
        fn = node.func.id
        args = [self(a) for a in node.args]
        if fn in FUNCTIONS:
            if len(args) != 1:
                self.fail(node, f"{fn} takes one argument")
            return FUNCTIONS[fn](args[0])
        if fn in ("lap", "laplacian"):
            if len(args) != 1:
                self.fail(node, f"{fn} takes one argument")
            if not self.ctx.spatial:
                self.fail(node, "lap() needs spatial coordinates")
            return sum(sp.diff(args[0], self.ctx.symbols[c], 2) for c in self.ctx.spatial)
        if fn in ("d", "diff"):
            if len(args) < 2:
                self.fail(node, f"{fn}(f, x[, x ...]) needs an expression and at least one coordinate")
            wrt = args[1:]
            if not all(w in self.ctx.symbols.values() for w in wrt):
                self.fail(node, f"{fn}() differentiates with respect to coordinates {self.ctx.coords} only")
            return sp.diff(args[0], *wrt)
        self.fail(node, f"unknown function {fn!r} (allowed: {', '.join(sorted(FUNCTIONS) + list(OPERATORS))})")


def _parse_tree(text: str) -> ast.Expression:
    if not isinstance(text, str) or not text.strip():
        raise EquationSyntaxError("an equation is a non-empty string")
    if len(text) > MAX_LENGTH:
        raise EquationSyntaxError(f"an equation has at most {MAX_LENGTH} characters")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as exc:
        raise EquationSyntaxError(f"cannot parse {text!r}: {exc.msg} (column {exc.offset})") from None
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise EquationSyntaxError(f"equation too long (more than {MAX_NODES} syntax nodes)")
    return tree


def parse_expression(text: str, ctx: Context) -> tuple[sp.Expr, set[str]]:
    """``text`` (no ``=``) as a SymPy expression, and the coefficient names it uses."""
    tr = _Translator(ctx, text)
    expr = tr(_parse_tree(text))
    return sp.sympify(expr), tr.coefficients     # sympify of a SymPy object: a no-op type normalisation


def _split(text: str) -> tuple[str, str]:
    t = text.strip()
    parts = re.split(r"(?<![<>=!])=(?!=)", t)
    if len(parts) == 1:
        return t, "0"
    if len(parts) != 2:
        raise EquationSyntaxError(f"an equation has at most one '=': {text!r}")
    if not parts[0].strip() or not parts[1].strip():
        raise EquationSyntaxError(f"empty side of '=' in {text!r}")
    return parts[0], parts[1]


@dataclass
class Equation:
    text: str
    lhs: sp.Expr
    rhs: sp.Expr
    coefficients: set[str]
    kind: str = "pde"                      # pde | dirichlet | neumann | robin | initial | periodic
    on: str = ""                           # boundary for a condition: "x_min", "y_max", "boundary", "t_min"

    @property
    def residual(self) -> sp.Expr:
        return sp.expand(self.lhs - self.rhs)

    def derivatives(self) -> list[tuple[str, tuple[str, ...]]]:
        out = set()
        for dv in self.residual.atoms(sp.Derivative):
            if isinstance(dv.expr, AppliedUndef):
                wrt = tuple(str(s) for s, k in dv.variable_count for _ in range(int(k)))
                out.add((dv.expr.func.__name__, tuple(sorted(wrt))))
        return sorted(out)

    def fields_used(self) -> list[str]:
        return sorted({a.func.__name__ for a in self.residual.atoms(AppliedUndef)})

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "kind": self.kind, "on": self.on, "residual": str(self.residual),
                "latex": sp.latex(sp.Eq(self.lhs, self.rhs)), "coefficients": sorted(self.coefficients),
                "derivatives": ["{}_{}".format(f, "".join(w)) for f, w in self.derivatives()],
                "fields": self.fields_used()}


def parse_equation(text: str, ctx: Context, *, kind: str = "pde", on: str = "") -> Equation:
    a, b = _split(text.replace("^", "**") if "^" in text else text)
    lhs, c1 = parse_expression(a, ctx)
    rhs, c2 = parse_expression(b, ctx)
    eq = Equation(text=text.strip(), lhs=lhs, rhs=rhs, coefficients=c1 | c2, kind=kind, on=on)
    if kind == "pde" and not eq.fields_used():
        raise EquationSyntaxError(f"a PDE must involve a field ({ctx.fields}): {text!r}")
    return eq


@dataclass
class EquationSystem:
    ctx: Context
    pdes: list[Equation]
    conditions: list[Equation] = field(default_factory=list)
    coefficients: dict[str, float] = field(default_factory=dict)
    unknowns: list[str] = field(default_factory=list)        # coefficients to identify from data

    @property
    def coords(self) -> list[str]:
        return self.ctx.coords

    @property
    def fields(self) -> list[str]:
        return self.ctx.fields

    @property
    def time_dependent(self) -> bool:
        return self.ctx.time in self.ctx.coords

    def used_coefficients(self) -> set[str]:
        out: set[str] = set()
        for e in self.pdes + self.conditions:
            out |= e.coefficients
        return out

    def issues(self) -> list[str]:
        """What keeps the system from being solved (empty when it is complete)."""
        out = []
        missing = sorted(self.used_coefficients() - set(self.coefficients) - set(self.unknowns))
        if missing:
            out.append(f"coefficient(s) without a value: {missing} (give a value or mark them unknown)")
        used = set()
        for e in self.pdes:
            used |= set(e.fields_used())
        if set(self.fields) - used:
            out.append(f"field(s) {sorted(set(self.fields) - used)} appear in no PDE")
        if len(self.pdes) < len(self.fields):
            out.append(f"{len(self.fields)} fields need at least as many PDEs (got {len(self.pdes)})")
        faces = {f"{c}_{s}" for c in self.ctx.spatial for s in ("min", "max")}
        for c in self.conditions:
            if c.kind == "initial" and not self.time_dependent:
                out.append(f"initial condition {c.text!r} in a steady problem")
            if c.kind not in ("initial",) and c.on not in faces | {"boundary"} | set(self.ctx.spatial):
                out.append(f"condition {c.text!r}: 'on' must be one of {sorted(faces)} or 'boundary'")
        if self.time_dependent and not any(c.kind == "initial" for c in self.conditions):
            out.append("a time-dependent problem needs an initial condition")
        return out

    def substituted(self, values: dict[str, float] | None = None) -> list[sp.Expr]:
        """PDE residuals with the coefficient values substituted (``values`` override the system's)."""
        vals = {**self.coefficients, **(values or {})}
        rep = {sp.Symbol(k, real=True): sp.Float(v) for k, v in vals.items()}
        return [e.residual.xreplace(rep) for e in self.pdes]

    def classify(self) -> dict[str, Any]:
        return classify(self)

    def to_dict(self) -> dict[str, Any]:
        return {"coords": self.coords, "fields": self.fields,
                "pdes": [e.to_dict() for e in self.pdes], "conditions": [e.to_dict() for e in self.conditions],
                "coefficients": dict(self.coefficients), "unknowns": list(self.unknowns),
                "issues": self.issues(), "classification": classify(self)}


def parse_system(pdes: list[str], conditions: list[dict] | None = None, *, coords: list[str], fields: list[str],
                 coefficients: dict[str, float] | None = None, unknowns: list[str] | None = None,
                 time: str = "t") -> EquationSystem:
    """Parse PDE texts and conditions (``{"kind", "on", "equation"}``) into an :class:`EquationSystem`.
    EquationSyntaxError names the first text that fails."""
    ctx = Context(list(coords), list(fields), time)
    eqs = [parse_equation(p, ctx) for p in pdes]
    conds = []
    for c in conditions or []:
        if not isinstance(c, dict) or "equation" not in c:
            raise EquationSyntaxError("a condition is {'kind': ..., 'on': ..., 'equation': ...}")
        kind = str(c.get("kind", "dirichlet")).lower()
        if kind not in CONDITION_KINDS:
            raise EquationSyntaxError(f"condition kind {kind!r} is not one of {CONDITION_KINDS}")
        on = str(c.get("on", "t_min" if kind == "initial" else "boundary"))
        conds.append(parse_equation(str(c["equation"]), ctx, kind=kind, on=on))
    coeffs = {}
    for k, v in (coefficients or {}).items():
        if not _NAME.match(str(k)):
            raise EquationSyntaxError(f"invalid coefficient name {k!r}")
        try:
            coeffs[str(k)] = float(v)
        except (TypeError, ValueError):
            raise EquationSyntaxError(f"coefficient {k!r} must be a number, got {v!r}") from None
    return EquationSystem(ctx, eqs, conds, coeffs, list(unknowns or []))


# ------------------------------------------------------------------------------------------------- classification
def _order(expr: sp.Expr, coord: str) -> int:
    best = 0
    for dv in expr.atoms(sp.Derivative):
        for s, k in dv.variable_count:
            if str(s) == coord:
                best = max(best, int(k))
    return best


def classify(system: EquationSystem) -> dict[str, Any]:
    """The PDE family of a single-field system (heat, wave, poisson, laplace, helmholtz, burgers,
    advection_diffusion, advection, reaction_diffusion, or ``custom``), its orders and linearity."""
    if not system.pdes:
        return {"family": "none"}
    t = system.ctx.time
    spatial = system.ctx.spatial
    out: dict[str, Any] = {"n_fields": len(system.fields), "dimension": len(spatial),
                           "time_dependent": system.time_dependent}
    R = system.pdes[0].residual
    out["time_order"] = _order(R, t) if system.time_dependent else 0
    out["space_order"] = max([_order(R, c) for c in spatial] or [0])
    funcs = list(system.ctx.functions.values())
    derivs = [a for a in R.atoms(sp.Derivative) if isinstance(a.expr, AppliedUndef)]
    gens = funcs + derivs
    try:
        poly = sp.Poly(R, *gens)
        degree = poly.total_degree()
    except sp.PolynomialError:
        degree = 99
    out["linear"] = degree <= 1
    fam = "custom"
    if len(system.fields) == 1 and len(system.pdes) == 1:
        u = funcs[0]
        has_lap = out["space_order"] == 2
        first_space = any(all(str(s) != t for s, _ in d.variable_count)
                          and sum(int(k) for _, k in d.variable_count) == 1 for d in derivs)
        rest = R
        for d in derivs:
            rest = rest.subs(d, 0)
        has_reaction = rest.has(u)
        if out["time_order"] == 2 and has_lap:
            fam = "wave"
        elif out["time_order"] == 1:
            if not out["linear"] and first_space and has_lap:
                fam = "burgers"
            elif has_lap and first_space:
                fam = "advection_diffusion"
            elif has_lap and has_reaction:
                fam = "reaction_diffusion"
            elif has_lap:
                fam = "heat"
            elif first_space:
                fam = "advection"
        elif out["time_order"] == 0 and has_lap:
            if has_reaction:
                fam = "helmholtz" if out["linear"] else "custom"
            else:
                fam = "laplace" if sp.simplify(rest) == 0 else "poisson"
    out["family"] = fam
    return out


# ------------------------------------------------------------------------------------------------- numeric compilers
def derivative_key(dv: sp.Derivative) -> str:
    """Environment key of a field derivative: ``u__xx`` for ``u_xx`` (letters sorted)."""
    wrt = "".join(str(s) * int(k) for s, k in dv.variable_count)
    return f"{dv.expr.func.__name__}__{''.join(sorted(wrt))}"


def derivative_keys(expr: sp.Expr) -> dict[str, tuple[str, str]]:
    """``{key: (field, wrt letters)}`` for every field derivative in ``expr``."""
    out = {}
    for dv in expr.atoms(sp.Derivative):
        if isinstance(dv.expr, AppliedUndef):
            out[derivative_key(dv)] = (dv.expr.func.__name__, "".join(str(s) * int(k) for s, k in dv.variable_count))
    return out


def _backend(name: str):
    if name == "numpy":
        import numpy as np
        return {sp.sin: np.sin, sp.cos: np.cos, sp.tan: np.tan, sp.exp: np.exp, sp.log: np.log, sp.sinh: np.sinh,
                sp.cosh: np.cosh, sp.tanh: np.tanh, sp.atan: np.arctan, sp.asin: np.arcsin, sp.acos: np.arccos,
                sp.Abs: np.abs, sp.sign: np.sign,
                sp.Heaviside: lambda a: np.heaviside(a, 0.5)}
    if name == "torch":
        import torch
        return {sp.sin: torch.sin, sp.cos: torch.cos, sp.tan: torch.tan, sp.exp: torch.exp, sp.log: torch.log,
                sp.sinh: torch.sinh, sp.cosh: torch.cosh, sp.tanh: torch.tanh, sp.atan: torch.atan,
                sp.asin: torch.asin, sp.acos: torch.acos, sp.Abs: torch.abs, sp.sign: torch.sign,
                sp.Heaviside: lambda a: torch.heaviside(a, torch.tensor(0.5, dtype=a.dtype))}
    raise ValueError(f"unknown backend {name!r}")


def compile_expr(expr: sp.Expr, backend: str = "numpy"):
    """A function ``env -> value`` for ``expr`` built as a closure tree (no ``eval``, no ``lambdify``).

    ``env`` maps coordinate and coefficient names, field names (their values) and :func:`derivative_key` keys (their
    derivative values) to numbers, arrays or tensors."""
    funcs = _backend(backend)

    def build(e):
        if e.is_Number or isinstance(e, sp.NumberSymbol):
            v = float(e)
            return lambda env: v
        if isinstance(e, sp.Symbol):
            name = e.name
            return lambda env: env[name]
        if isinstance(e, AppliedUndef):
            name = e.func.__name__
            return lambda env: env[name]
        if isinstance(e, sp.Derivative):
            if not isinstance(e.expr, AppliedUndef):
                return build(e.doit())
            key = derivative_key(e)
            return lambda env: env[key]
        if isinstance(e, sp.Add):
            parts = [build(a) for a in e.args]

            def add(env):
                out = parts[0](env)
                for p in parts[1:]:
                    out = out + p(env)
                return out
            return add
        if isinstance(e, sp.Mul):
            parts = [build(a) for a in e.args]

            def mul(env):
                out = parts[0](env)
                for p in parts[1:]:
                    out = out * p(env)
                return out
            return mul
        if isinstance(e, sp.Pow):
            b, x = build(e.base), build(e.exp)
            return lambda env: b(env) ** x(env)
        if e.func in funcs:
            f, a = funcs[e.func], build(e.args[0])
            return lambda env: f(a(env))
        raise EquationSyntaxError(f"cannot evaluate {type(e).__name__} ({e}) numerically")

    return build(expr)

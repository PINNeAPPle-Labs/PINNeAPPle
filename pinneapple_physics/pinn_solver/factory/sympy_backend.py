"""SymPy-based compiler for symbolic equations to torch-callable lambdas.

Design note — two SymPy→PyTorch compilation paths exist in this codebase:

* **This module** (``SympyTorchCompiler``): uses ``sympy.lambdify`` to emit a
  fast Python lambda.  Gradients do *not* flow through the symbolic evaluation
  itself; the lambdified call is opaque to autograd.  Suitable when PDE losses
  are evaluated at fixed collocation points and runtime speed is the priority.

* ``pinneapple_symbolic.SymbolicPDE``: walks the expression tree with
  ``torch.autograd.grad``, so gradients *do* flow through the evaluation.
  Use that path when you need the symbolic expression to participate in the
  autograd graph (e.g. differentiating the residual w.r.t. model weights for
  meta-learning, or computing higher-order cross derivatives)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

import sympy as sp
from sympy.core.function import AppliedUndef

# Names the equation strings may use as math, not as user variables. Everything else that
# is not declared is created fresh (Symbol, or Function when called), so user names such as
# S, E, I, N, O, Q, beta or gamma never resolve to SymPy built-ins: before this, "S(t)"
# silently compiled to "t" (sympy.S is SymPy's singleton registry).
_MATH_NAMES = {
    "Derivative", "sin", "cos", "tan", "exp", "log", "sqrt", "Abs", "pi", "tanh", "sinh", "cosh",
    "asin", "acos", "atan", "atan2", "sign", "Heaviside", "Max", "Min", "Rational", "Integer", "Float",
}
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


# -----------------------------
# Compiled equation container
# -----------------------------
@dataclass(frozen=True)
class CompiledEquation:
    """
    A compiled symbolic equation.

    - expr: SymPy expression
    - derivatives: set of SymPy Derivative atoms appearing in expr
    - call: torch-compatible callable from sympy.lambdify
    - args_order: stable argument order for call(...)
    """
    expr: sp.Expr
    derivatives: Set[sp.Derivative]
    call: Callable[..., Any]
    args_order: List[sp.Basic]
    # Known input signals used by the equation (e.g. S(t), Tamb(t)), in call order.
    exogenous: List[str] = field(default_factory=list)
    # Undeclared bare symbols (e.g. T0) whose value is given at call time, in call order.
    runtime_constants: List[str] = field(default_factory=list)


# -----------------------------
# Low-level compiler
# -----------------------------
class SympyTorchCompiler:
    """
    Compiles equation strings into torch-callable lambdas using SymPy.
    """

    def __init__(
        self,
        independent_vars: List[str],
        dependent_vars: List[str],
        inverse_params: Optional[List[str]] = None,
        extra_namespace: Optional[Dict[str, Any]] = None,
    ):
        self.ind_vars_str = list(independent_vars)
        self.dep_vars_str = list(dependent_vars)
        self.inv_vars_str = list(inverse_params or [])

        self.ind_symbols = self._create_symbols(self.ind_vars_str)
        self.inv_symbols = self._create_symbols(self.inv_vars_str)

        self.dep_func_classes = {v: sp.Function(v) for v in self.dep_vars_str}
        self.dep_symbols = {v: self.dep_func_classes[v](*self.ind_symbols) for v in self.dep_vars_str}

        # Locals namespace for sympify
        self.namespace: Dict[str, Any] = {}
        self.namespace.update({s.name: s for s in self.ind_symbols + self.inv_symbols})
        self.namespace.update(self.dep_func_classes)
        self.namespace.update(
            {
                "sin": sp.sin,
                "cos": sp.cos,
                "exp": sp.exp,
                "pi": sp.pi,
                "sqrt": sp.sqrt,
                "log": sp.log,
                "Abs": sp.Abs,
            }
        )
        if extra_namespace:
            self.namespace.update(extra_namespace)

    @staticmethod
    def _create_symbols(names: List[str]) -> List[sp.Symbol]:
        """Create SymPy symbols from variable names; returns list of symbols."""
        if not names:
            return []
        syms = sp.symbols(" ".join(names))
        if isinstance(syms, tuple):
            return list(syms)
        return [syms]

    def _locals_for(self, eq_str: str) -> Dict[str, Any]:
        """Namespace for one equation: declared names + fresh objects for every other identifier."""
        ns = dict(self.namespace)
        ns["Derivative"] = sp.Derivative
        for m in _IDENT.finditer(eq_str):
            name = m.group(0)
            if name in ns or name in _MATH_NAMES:
                continue
            called = eq_str[m.end():].lstrip().startswith("(")
            ns[name] = sp.Function(name) if called else sp.Symbol(name)
        for name in ("tan", "tanh", "sinh", "cosh", "asin", "acos", "atan", "atan2", "sign",
                     "Heaviside", "Max", "Min"):
            ns.setdefault(name, getattr(sp, name))
        return ns

    def compile(self, eq_str: str, *, constants: Optional[Dict[str, float]] = None,
                strict: bool = True) -> CompiledEquation:
        """Compile an equation string into a torch-callable residual.

        Besides the declared independent/dependent variables and inverse parameters, an
        equation may use:

        - **exogenous signals**: any undeclared function of the independent variables, e.g.
          ``S(t)`` or ``Tamb(t)``: a known input (sensor data, forcing), not a network output.
          Its values are passed at call time, after the inverse parameters.
        - **constants**: undeclared bare symbols (e.g. ``T0``) given in ``constants``; they
          are substituted by their numeric value.

        Any other undeclared bare symbol raises a ``ValueError`` that names it when
        ``strict`` (default); with ``strict=False`` it becomes a *runtime constant*: an extra
        call argument (after the exogenous signals) whose value is supplied when the residual
        is evaluated.
        """
        constants = dict(constants or {})
        expr = sp.sympify(eq_str, locals=self._locals_for(eq_str), evaluate=False)
        def _expand_numeric_pow(e):
            if isinstance(e, sp.Pow) and e.base.is_Number and e.exp.is_Integer and int(e.exp) >= 0:
                n = int(e.exp)
                if n == 0:
                    return sp.Integer(1)
                return sp.Mul(*([e.base] * n), evaluate=False)
            return e

        expr = expr.replace(lambda e: isinstance(e, sp.Pow) and e.base.is_Number and e.exp.is_Integer, _expand_numeric_pow)

        # Exogenous signals: applied functions that are not dependent variables.
        exo_apps = [a for a in expr.atoms(AppliedUndef) if a.func.__name__ not in self.dep_func_classes]
        exo_names = sorted({a.func.__name__ for a in exo_apps})
        ind_set = set(self.ind_symbols)
        for a in exo_apps:
            if not a.args or not set(a.args) <= ind_set or len(set(a.args)) != len(a.args):
                raise ValueError(
                    f"'{a}' in '{eq_str}': an exogenous signal must be a function of independent "
                    f"variables {self.ind_vars_str} only (e.g. {a.func.__name__}({', '.join(self.ind_vars_str)}))."
                )
        for d in expr.atoms(sp.Derivative):
            if isinstance(d.expr, AppliedUndef) and d.expr.func.__name__ in exo_names:
                raise ValueError(
                    f"'{d}' in '{eq_str}': derivatives of exogenous signals are not supported; "
                    f"pass the derivative as its own signal (e.g. d{d.expr.func.__name__}(t))."
                )
        exo_symbols = {n: sp.Symbol(f"__exo_{n}") for n in exo_names}
        if exo_apps:
            expr = expr.xreplace({a: exo_symbols[a.func.__name__] for a in exo_apps})

        if constants:
            expr = expr.xreplace({sp.Symbol(k): sp.Float(v) for k, v in constants.items()})

        known = ind_set | set(self.inv_symbols) | set(exo_symbols.values())
        unknown = sorted(str(s_) for s_ in expr.free_symbols - known)
        if unknown and strict:
            raise ValueError(
                f"Undeclared name(s) {unknown} in '{eq_str}'. Declare each one as a constant "
                f"(PINNProblemSpec.constants={{'{unknown[0]}': value}} or the condition's 'constants'), "
                f"as an inverse parameter (inverse_params=[...]), or write it as a function of the "
                f"independent variables to feed it as an exogenous signal (e.g. {unknown[0]}"
                f"({', '.join(self.ind_vars_str)}))."
            )

        runtime_symbols = [sp.Symbol(n) for n in unknown] if not strict else []

        derivatives = set(expr.atoms(sp.Derivative))

        # Stable ordering:
        # 1) indep symbols
        # 2) dependent symbols (u(t,x), v(t,x), ...)
        # 3) inverse params
        # 4) exogenous signals (sorted by name)
        # 5) runtime constants (sorted by name; only with strict=False)
        # 6) derivatives (sorted)
        deriv_sorted = sorted(list(derivatives), key=str)
        args_order: List[sp.Basic] = [
            *self.ind_symbols,
            *self.dep_symbols.values(),
            *self.inv_symbols,
            *[exo_symbols[n] for n in exo_names],
            *runtime_symbols,
            *deriv_sorted,
        ]

        call = sp.lambdify(args_order, expr, "torch")
        return CompiledEquation(expr=expr, derivatives=derivatives, call=call, args_order=args_order,
                                exogenous=exo_names, runtime_constants=[str(r) for r in runtime_symbols])


class SympyBackend:
    """
    Compatibility wrapper.

    Your current PINNFactory uses SympyTorchCompiler directly.
    Some examples may import SympyBackend; this class simply exposes a
    make_compiler(...) method returning SympyTorchCompiler.
    """

    def __init__(self, *, extra_namespace: Optional[Dict[str, Any]] = None) -> None:
        self.extra_namespace = extra_namespace or {}

    def make_compiler(
        self,
        *,
        independent_vars: List[str],
        dependent_vars: List[str],
        inverse_params: Optional[List[str]] = None,
    ) -> SympyTorchCompiler:
        """Create SympyTorchCompiler with given vars and optional inverse params."""
        return SympyTorchCompiler(
            independent_vars=independent_vars,
            dependent_vars=dependent_vars,
            inverse_params=inverse_params or [],
            extra_namespace=self.extra_namespace,
        )

"""Safe evaluation of constraint expressions such as ``"chord - 2 * thickness >= 0.01"``.

Only arithmetic, comparisons, ``and`` / ``or`` / ``not``, parameter names, numbers, strings and a few math functions
are allowed; anything else (attribute access, calls to other functions, subscripts, lambdas) is refused before the
expression runs. Every name is a column, so one call checks a whole table of cases at once.

The expression is interpreted node by node, never handed to ``eval``: this is the evaluator for any formula a user
sends to a server (constraints, boundary values, source terms). ``functions=`` replaces the callable table and
:func:`check` validates an expression for another consumer (e.g. before ``sympy.sympify``).
"""
from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

FUNCTIONS = {
    "abs": np.abs, "sqrt": np.sqrt, "exp": np.exp, "log": np.log, "log10": np.log10,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "min": np.minimum, "max": np.maximum,
}
_BINOPS = {
    ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide,
    ast.Pow: np.power, ast.Mod: np.mod, ast.FloorDiv: np.floor_divide,
}
_CMPOPS = {
    ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater, ast.GtE: np.greater_equal,
    ast.Eq: np.equal, ast.NotEq: np.not_equal,
}


MAX_LENGTH = 2000


class ExpressionError(ValueError):
    """The expression uses something that is not allowed, or a name that is not a parameter."""


def names_in(expression: str) -> set[str]:
    """Parameter names an expression refers to (function names excluded)."""
    return names_in_tree(_parse(expression))


def evaluate(expression: str, values: Mapping[str, Any], functions: Mapping[str, Any] | None = None) -> np.ndarray:
    """Evaluate ``expression`` with ``values`` (name -> scalar or column) and return a boolean (or numeric) array.
    ``functions``: the names it may call (default :data:`FUNCTIONS`)."""
    fns = FUNCTIONS if functions is None else functions
    return np.asarray(_eval(_parse(expression, fns).body, values, fns))


def check(expression: str, names: Iterable[str], functions: Iterable[str] = (), strings: bool = False) -> None:
    """Raise :class:`ExpressionError` unless ``expression`` uses only the grammar above, the variables ``names`` and
    calls of ``functions`` (and no string literals unless ``strings``)."""
    fns = dict.fromkeys(functions)
    tree = _parse(expression, fns, strings=strings)
    unknown = names_in_tree(tree) - set(names)
    if unknown:
        raise ExpressionError(f"unknown name(s) {sorted(unknown)} in {expression!r}")


def names_in_tree(tree: ast.AST) -> set[str]:
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - called


def _parse(expression: str, functions: Mapping[str, Any] = FUNCTIONS, strings: bool = True) -> ast.Expression:
    if not isinstance(expression, str) or len(expression) > MAX_LENGTH:
        raise ExpressionError(f"an expression is a string of at most {MAX_LENGTH} characters")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"cannot parse constraint {expression!r}: {exc.msg}") from None
    for node in ast.walk(tree):
        ok = isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Name,
                               ast.Constant, ast.Load, ast.Call, ast.operator, ast.unaryop, ast.boolop, ast.cmpop))
        if not ok:
            raise ExpressionError(f"{type(node).__name__} is not allowed in a constraint: {expression!r}")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in functions
                                               and not node.keywords
                                               and not any(isinstance(a, ast.Starred) for a in node.args)):
            raise ExpressionError(f"only {sorted(functions)} may be called in a constraint: {expression!r}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, bool) + ((str,) if strings else ())):
            raise ExpressionError(f"constant {node.value!r} is not allowed: {expression!r}")
    return tree


def _eval(node: ast.AST, values: Mapping[str, Any], functions: Mapping[str, Any] = FUNCTIONS):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in values:
            raise ExpressionError(f"unknown name {node.id!r} (not a parameter)")
        return np.asarray(values[node.id])
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ExpressionError(f"operator {type(node.op).__name__} is not allowed")
        return op(_eval(node.left, values, functions), _eval(node.right, values, functions))
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand, values, functions)
        if isinstance(node.op, ast.USub):
            return np.negative(v)
        if isinstance(node.op, ast.UAdd):
            return v
        if isinstance(node.op, ast.Not):
            return np.logical_not(v)
        raise ExpressionError(f"operator {type(node.op).__name__} is not allowed")
    if isinstance(node, ast.BoolOp):
        parts = [_eval(v, values, functions) for v in node.values]
        combine = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
        out = parts[0]
        for p in parts[1:]:
            out = combine(out, p)
        return out
    if isinstance(node, ast.Compare):
        left = _eval(node.left, values, functions)
        out = None
        for op, comp in zip(node.ops, node.comparators, strict=True):
            fn = _CMPOPS.get(type(op))
            if fn is None:
                raise ExpressionError(f"comparison {type(op).__name__} is not allowed")
            right = _eval(comp, values, functions)
            res = fn(left, right)
            out = res if out is None else np.logical_and(out, res)
            left = right
        return out
    if isinstance(node, ast.Call):
        return functions[node.func.id](*[_eval(a, values, functions) for a in node.args])
    raise ExpressionError(f"{type(node).__name__} is not allowed")  # pragma: no cover - refused in _parse

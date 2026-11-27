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


def _torch_backend():
    """The same operator table over ``torch`` tensors, so an expression can be differentiated.

    Imported lazily: ``pinneapple_data`` is used by modules that never need torch, and this is only asked for when a
    caller passes ``ops="torch"``.
    """
    import torch

    return ({ast.Add: torch.add, ast.Sub: torch.subtract, ast.Mult: torch.multiply, ast.Div: torch.divide,
             ast.Pow: torch.pow, ast.Mod: torch.remainder, ast.FloorDiv: torch.floor_divide},
            {ast.Lt: torch.lt, ast.LtE: torch.le, ast.Gt: torch.gt, ast.GtE: torch.ge,
             ast.Eq: torch.eq, ast.NotEq: torch.ne},
            {"abs": torch.abs, "sqrt": torch.sqrt, "exp": torch.exp, "log": torch.log, "log10": torch.log10,
             "sin": torch.sin, "cos": torch.cos, "tan": torch.tan, "min": torch.minimum, "max": torch.maximum})


def _tables(ops: Any, functions: Mapping[str, Any] | None):
    """``(binops, cmpops, functions)`` for ``ops`` -- ``None``/``"numpy"`` is the default table, ``"torch"`` the
    differentiable one."""
    if ops is None or ops == "numpy":
        return _BINOPS, _CMPOPS, (FUNCTIONS if functions is None else functions)
    if ops == "torch":
        binops, cmpops, fns = _torch_backend()
        return binops, cmpops, ({**fns, **(functions or {})})
    raise ValueError(f"ops must be None, 'numpy' or 'torch', got {ops!r}")


class ExpressionError(ValueError):
    """The expression uses something that is not allowed, or a name that is not a parameter."""


def names_in(expression: str) -> set[str]:
    """Parameter names an expression refers to (function names excluded)."""
    return names_in_tree(_parse(expression))


def evaluate(expression: str, values: Mapping[str, Any], functions: Mapping[str, Any] | None = None,
             *, ops: Any = None) -> Any:
    """Evaluate ``expression`` with ``values`` (name -> scalar or column) and return a boolean (or numeric) array.
    ``functions``: the names it may call (default :data:`FUNCTIONS`). ``ops="torch"`` evaluates over torch tensors
    (the result is a tensor and carries its graph) instead of converting to numpy."""
    binops, cmpops, fns = _tables(ops, functions)
    out = _eval(_parse(expression, fns).body, values, fns, binops, cmpops)
    return out if ops == "torch" else np.asarray(out)


def residual(expression: str, values: Mapping[str, Any], functions: Mapping[str, Any] | None = None,
             *, ops: Any = None) -> Any:
    """The **signed margin** of a comparison expression: ``<= 0`` when satisfied, ``> 0`` by how much it is violated.

    ``evaluate`` answers "does this hold" with a boolean, which is enough to reject a design and not enough to
    penalise it in training, rank it in a report or differentiate it inside an optimizer. This returns
    ``left - right`` with the sign of the comparison folded in -- ``"t <= 800"`` gives ``t - 800``, ``"mass >= 2"``
    gives ``2 - mass``, ``"x == 1"`` gives ``abs(x - 1)`` -- so the same expression object is a penalty, a margin and
    a constraint in every one of those places.

    A strict comparison is treated as its non-strict form (a margin of exactly zero is a tie, not a pass), and the
    expression has to be a single comparison: an ``and`` of two inequalities is two constraints, and saying so is
    better than returning a margin for a boolean.
    """
    binops, cmpops, fns = _tables(ops, functions)
    tree = _parse(expression, fns)
    body = tree.body
    if not isinstance(body, ast.Compare) or len(body.ops) != 1:
        raise ExpressionError(f"a constraint expression must be a single comparison (a <= b, a >= b, ...); "
                              f"{expression!r} is not -- write it as two constraints")
    left = _eval(body.left, values, fns, binops, cmpops)
    right = _eval(body.comparators[0], values, fns, binops, cmpops)
    op = type(body.ops[0])
    if op in (ast.Lt, ast.LtE):
        out = binops[ast.Sub](left, right)
    elif op in (ast.Gt, ast.GtE):
        out = binops[ast.Sub](right, left)
    elif op is ast.Eq:
        out = fns["abs"](binops[ast.Sub](left, right))
    else:                                              # ast.NotEq: satisfied away from the value
        out = binops[ast.Sub](0, fns["abs"](binops[ast.Sub](left, right)))
    return out if ops == "torch" else np.asarray(out)


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


def _eval(node: ast.AST, values: Mapping[str, Any], functions: Mapping[str, Any] = FUNCTIONS,
          binops: Mapping[type, Any] = None, cmpops: Mapping[type, Any] = None):
    binops = _BINOPS if binops is None else binops
    cmpops = _CMPOPS if cmpops is None else cmpops
    is_torch = binops is not _BINOPS                 # a torch run keeps tensors (and their graph) as they are
    wrap = (lambda v: v) if is_torch else np.asarray
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in values:
            raise ExpressionError(f"unknown name {node.id!r} (not a parameter)")
        return wrap(values[node.id])
    if isinstance(node, ast.BinOp):
        op = binops.get(type(node.op))
        if op is None:
            raise ExpressionError(f"operator {type(node.op).__name__} is not allowed")
        return op(_eval(node.left, values, functions, binops, cmpops),
                  _eval(node.right, values, functions, binops, cmpops))
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand, values, functions, binops, cmpops)
        if isinstance(node.op, ast.USub):
            return binops[ast.Sub](0, v)
        if isinstance(node.op, ast.UAdd):
            return v
        if isinstance(node.op, ast.Not):
            return _logical(v, "not") if is_torch else np.logical_not(v)
        raise ExpressionError(f"operator {type(node.op).__name__} is not allowed")
    if isinstance(node, ast.BoolOp):
        parts = [_eval(v, values, functions, binops, cmpops) for v in node.values]
        combine = (lambda a, b: _logical(a, "and", b)) if is_torch else \
            (np.logical_and if isinstance(node.op, ast.And) else np.logical_or)
        if is_torch and isinstance(node.op, ast.Or):
            combine = lambda a, b: _logical(a, "or", b)  # noqa: E731
        out = parts[0]
        for p in parts[1:]:
            out = combine(out, p)
        return out
    if isinstance(node, ast.Compare):
        left = _eval(node.left, values, functions, binops, cmpops)
        out = None
        for op, comp in zip(node.ops, node.comparators, strict=True):
            fn = cmpops.get(type(op))
            if fn is None:
                raise ExpressionError(f"comparison {type(op).__name__} is not allowed")
            right = _eval(comp, values, functions, binops, cmpops)
            res = fn(left, right)
            out = res if out is None else (_logical(out, "and", res) if is_torch else np.logical_and(out, res))
            left = right
        return out
    if isinstance(node, ast.Call):
        return functions[node.func.id](*[_eval(a, values, functions, binops, cmpops) for a in node.args])
    raise ExpressionError(f"{type(node).__name__} is not allowed")  # pragma: no cover - refused in _parse


def _logical(a: Any, op: str, b: Any = None) -> Any:
    """Boolean combination over torch tensors (numpy has the same thing as a ufunc)."""
    import torch

    if op == "and":
        return torch.logical_and(a, b)
    if op == "or":
        return torch.logical_or(a, b)
    return torch.logical_not(a)

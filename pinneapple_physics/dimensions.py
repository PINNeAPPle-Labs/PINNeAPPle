"""Dimensional consistency of a PDE, checked before anything is trained (#205).

A PDE whose terms do not share a dimension is not a modelling subtlety: it is a unit mistake, and a network trained
on it fits the residual of a statement that cannot be true. The failure shows up later as an error floor nobody can
explain. This reads the equation the way a reader does — ``dT/dt = alpha * laplacian(T) + q / (rho * cp)`` — assigns
a dimension to every term from the units the problem already declares, and says which term broke it:

    >>> from pinneapple_physics.dimensions import check_dimensions
    >>> report = check_dimensions("dT/dt = alpha*laplacian(T) + q/(rho*cp)", units={
    ...     "T": "K", "t": "s", "x": "m", "alpha": "m^2/s", "q": "W/m^3", "rho": "kg/m^3", "cp": "J/(kg*K)"})
    >>> print(report)                                    # doctest: +SKIP
    dT/dt -> K/s, alpha*laplacian(T) -> K/s, q/(rho*cp) -> m^2/s^2/K: inconsistent
    >>> [str(p) for p in report.problems]                # doctest: +SKIP
    ["'q/(rho*cp)' is m^2/s^2/K, the other terms of the equation are K/s"]

What it will not do: guess a unit. A name it does not know is refused by name, because a checker that assumed the
missing unit would pass exactly the equation it was written to catch.

Notation it understands, so the equation can be written the way the paper writes it:

======================  ============================================================================
``dT/dt``, ``∂T/∂t``    time derivative: the field's unit over the time coordinate's unit
``d2T/dx2``, ``∂²T/∂x²`` second spatial derivative
``laplacian(T)``, ``lap(T)``, ``∇²T``   field over a length squared
``grad(T)``, ``div(u)``  field over a length
``d_dt(T)``, ``dt(T)``, ``d_dx(T)``, ``dx(T)``, ``d2_dx2(T)``   the same three, in call form
======================  ============================================================================

Units come from the problem when one is given (``problem.units`` and each parameter's ``unit``), from ``units=`` when
not, and ``units=`` wins where both say something.
"""
from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pinneapple_data.physical_units import UnitError, parse_unit

__all__ = ["DimensionReport", "Term", "check_dimensions", "format_dim", "unit_dim"]

Dim = tuple[int, int, int, int, int, int]
_DIMLESS: Dim = (0, 0, 0, 0, 0, 0)
_BASE = ("m", "kg", "s", "K", "A", "mol")
_LENGTH: Dim = (1, 0, 0, 0, 0, 0)
_TIME: Dim = (0, 0, 1, 0, 0, 0)

#: derived units worth naming in a report, checked before falling back to base units
_DERIVED: dict[Dim, str] = {}
for _symbol in ("W", "J", "N", "Pa", "V", "T", "Hz", "coulomb", "ohm", "farad", "henry", "siemens", "weber"):
    _DERIVED.setdefault(parse_unit(_symbol).dim, _symbol)


def unit_dim(text: str) -> Dim:
    """The dimension vector of a unit string; ``"m^2/s"`` -> m²·s⁻¹. Raises ``ValueError`` for a unit it cannot read."""
    try:
        return parse_unit(str(text)).dim
    except UnitError as exc:
        raise ValueError(f"{text!r} is not a unit this library parses: {exc}") from None


def format_dim(dim: Dim) -> str:
    """A readable unit for a dimension: ``K/s``, ``m^2/s``, ``W/m^3``, ``dimensionless``.

    A short base product wins (``K/s`` reads better than ``Hz·K``). A longer one is named after the derived unit it
    contains, because that is the unit the term is quoted in: ``W/m^3``, not ``kg/(m·s^3)``."""
    if dim == _DIMLESS:
        return "dimensionless"
    if dim in _DERIVED:
        return _DERIVED[dim]
    numerator, denominator = [], []
    for name, power in zip(_BASE, dim, strict=True):
        if power > 0:
            numerator.append(name if power == 1 else f"{name}^{power}")
        elif power < 0:
            denominator.append(name if power == -1 else f"{name}^{-power}")

    def join(parts: list[str], bottom: list[str]) -> str:
        top = "·".join(parts) or "1"
        if not bottom:
            return top
        return f"{top}/{bottom[0] if len(bottom) == 1 else '(' + '·'.join(bottom) + ')'}"

    if len(numerator) + len(denominator) <= 2:
        return join(numerator, denominator)
    best: tuple[int, str] | None = None
    for derived_dim, symbol in _DERIVED.items():
        rest = _add(dim, derived_dim, -1)                 # what is left over once the derived unit is taken out
        for position, power in enumerate(rest):
            if power and rest == tuple(0 if i != position else power for i in range(6)):
                tail = _BASE[position] if abs(power) == 1 else f"{_BASE[position]}^{abs(power)}"
                text = f"{symbol}/{tail}" if power < 0 else f"{symbol}·{tail}"
                if best is None:                          # the first derived unit that fits is the conventional one
                    best = (abs(power), text)
    return best[1] if best else join(numerator, denominator)


def _add(a: Dim, b: Dim, sign: int = 1) -> Dim:
    return tuple(x + sign * y for x, y in zip(a, b, strict=True))  # type: ignore[return-value]


def _scale(a: Dim, n: int) -> Dim:
    return tuple(x * n for x in a)  # type: ignore[return-value]


# ── notation ────────────────────────────────────────────────────────────────
_SUPERSCRIPTS = str.maketrans({"²": "^2", "³": "^3", "·": "*", "⋅": "*", "×": "*", "−": "-"})
_GRADIENT = [(re.compile(r"∇²\s*([A-Za-z_]\w*)"), r"laplacian(\1)"),
             (re.compile(r"∇·\s*([A-Za-z_]\w*)"), r"div(\1)"),
             (re.compile(r"∇\s*([A-Za-z_]\w*)"), r"grad(\1)")]
_DERIVATIVE = re.compile(r"d\^?(?P<order>\d)?\s*(?P<field>[A-Za-z_]\w*)\s*/\s*d\s*(?P<axis>[A-Za-z_]+)"
                         r"(?:\s*\^?\s*(?P<axis_order>\d))?")
_SPLIT = re.compile(r"(?<![<>!=])=(?!=)")


def _normalise(text: str) -> str:
    """``∇²T`` and ``∂²T/∂x²`` become things the expression parser reads; the original text is kept for the report."""
    out = text
    for pattern, replacement in _GRADIENT:
        out = pattern.sub(replacement, out)
    return out.translate(_SUPERSCRIPTS).replace("∂", "d")


@dataclass(frozen=True)
class Term:
    """One additive term of the equation, with the unit it works out to."""

    text: str
    dim: Dim

    @property
    def unit(self) -> str:
        return format_dim(self.dim)

    def __str__(self) -> str:
        return f"{self.text} -> {self.unit}"


@dataclass
class DimensionReport:
    """What the checker found: the terms and their units, and every reason the equation cannot hold."""

    equation: str
    terms: list[Term] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_markdown(self) -> str:
        verdict = "consistent" if self.ok else "inconsistent"
        rows = ["| term | unit |", "|---|---|"]
        rows += [f"| `{t.text}` | {t.unit} |" for t in self.terms]
        body = "\n".join(rows)
        head = f"**`{self.equation}`** — {verdict}\n\n{body}"
        if self.problems:
            head += "\n\n**Problems:**\n" + "\n".join(f"- {p}" for p in self.problems)
        return head

    def raise_if_inconsistent(self) -> DimensionReport:
        """Refuse to go on: the message is the report, so the failing term is the first thing read."""
        if not self.ok:
            raise ValueError("dimensionally inconsistent equation:\n" + self.to_markdown())
        return self

    def __str__(self) -> str:
        """``dT/dt -> K/s, alpha*laplacian(T) -> K/s: inconsistent`` — the one-line form of the report."""
        if not self.terms:
            return f"{self.equation}: {self.problems[0] if self.problems else 'nothing to check'}"
        terms = ", ".join(str(t) for t in self.terms)
        return f"{terms}: {'consistent' if self.ok else 'inconsistent'}"

    def __bool__(self) -> bool:
        return self.ok

    def to_dict(self) -> dict[str, Any]:
        return {"equation": self.equation, "ok": self.ok, "consistent": self.ok,
                "terms": [{"term": t.text, "unit": t.unit, "dim": list(t.dim)} for t in self.terms],
                "problems": list(self.problems)}


# ── evaluation ──────────────────────────────────────────────────────────────
class _Unknown(Exception):
    """A name or function the checker cannot give a dimension to; never guessed."""


def _flatten_terms(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        return _flatten_terms(node.left) + _flatten_terms(node.right)
    return [node]


def _eval(node: ast.AST, units: Mapping[str, Dim], space: Dim, time: Dim) -> Dim:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return _DIMLESS
        raise _Unknown(f"{node.value!r} is not a number")
    if isinstance(node, ast.Name):
        if node.id not in units:
            raise _Unknown(f"unknown name '{node.id}': give its unit, e.g. units={{'{node.id}': 'K'}}")
        return units[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _eval(node.operand, units, space, time)
    if isinstance(node, ast.BinOp):
        left = _eval(node.left, units, space, time)
        right = _eval(node.right, units, space, time)
        if isinstance(node.op, ast.Add | ast.Sub):
            if left != right:
                raise _Mismatch(f"'{_text(node.left)}' is {format_dim(left)} but "
                                f"'{_text(node.right)}' is {format_dim(right)}")
            return left
        if isinstance(node.op, ast.Mult):
            return _add(left, right)
        if isinstance(node.op, ast.Div):
            return _add(left, right, -1)
        if isinstance(node.op, ast.Pow):
            exponent = node.right
            if not (isinstance(exponent, ast.Constant) and isinstance(exponent.value, int)):
                raise _Unknown(f"the exponent of '{_text(node.left)}' has to be a whole number")
            return _scale(left, exponent.value)
        raise _Unknown(f"operator {type(node.op).__name__} is not something a PDE term uses")
    if isinstance(node, ast.Call):
        return _call(node, units, space, time)
    raise _Unknown(f"{type(node).__name__} is not something a PDE term uses")


class _Mismatch(Exception):
    """Two terms that have to share a dimension do not."""


#: name in the equation -> (argument position, how the dimension changes)
_OPERATORS = {
    "laplacian": (0, lambda d, space, time: _add(d, _scale(space, 2), -1)),
    "lap": (0, lambda d, space, time: _add(d, _scale(space, 2), -1)),
    "grad": (0, lambda d, space, time: _add(d, space, -1)),
    "div": (0, lambda d, space, time: _add(d, space, -1)),
    "dt": (0, lambda d, space, time: _add(d, time, -1)),
    "d_dt": (0, lambda d, space, time: _add(d, time, -1)),
    "dx": (0, lambda d, space, time: _add(d, space, -1)),
    "d_dx": (0, lambda d, space, time: _add(d, space, -1)),
    "d2_dx2": (0, lambda d, space, time: _add(d, _scale(space, 2), -1)),
}


def _call(node: ast.Call, units: Mapping[str, Dim], space: Dim, time: Dim) -> Dim:
    name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
    if name not in _OPERATORS:
        raise _Unknown(f"unknown function '{name}()': the checker knows {sorted(_OPERATORS)}")
    position, change = _OPERATORS[name]
    if len(node.args) <= position:
        raise _Unknown(f"{name}() needs an argument")
    return change(_eval(node.args[position], units, space, time), space, time)


def _text(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:                                       # pragma: no cover - ast.unparse is stdlib since 3.9
        return "?"


# ── the checker ─────────────────────────────────────────────────────────────
def _units_from_problem(problem: Any) -> dict[str, str]:
    out: dict[str, str] = {k: v for k, v in dict(getattr(problem, "units", {}) or {}).items() if v}
    for name, parameter in dict(getattr(problem, "parameters", {}) or {}).items():
        unit = getattr(parameter, "unit", None)
        if unit and name not in out:
            out[name] = unit
    return out


def _coordinate_dim(name: str, dims: Mapping[str, Dim], space: Dim, time: Dim) -> Dim | None:
    """The dimension a coordinate stands for: its own if it declares one, else the usual name (x/y/z, t)."""
    if name in dims:
        return dims[name]
    if name in ("x", "y", "z", "r"):
        return space
    if name in ("t", "time"):
        return time
    return None


def _expand_derivatives(side: str, dims: Mapping[str, Dim], space: Dim, time: Dim
                        ) -> tuple[str, dict[str, Dim], dict[str, str], list[str]]:
    """``d2T/dx2`` becomes one symbol whose dimension is already worked out, so the expression parser sees a product.

    The original spelling is kept per symbol: the report has to quote the equation the reader wrote, not the form the
    parser wanted."""
    synthetic: dict[str, Dim] = {}
    spelling: dict[str, str] = {}
    problems: list[str] = []

    def replace(match: re.Match[str]) -> str:
        field, axis = match.group("field"), match.group("axis")
        order = int(match.group("order") or match.group("axis_order") or 1)
        if field not in dims:
            problems.append(f"unknown name '{field}': give its unit, e.g. units={{'{field}': 'K'}}")
            return match.group(0)
        axis_dim = _coordinate_dim(axis, dims, space, time)
        if axis_dim is None:
            problems.append(f"unknown coordinate '{axis}' in '{match.group(0).strip()}': declare its unit")
            return match.group(0)
        name = f"_d{len(synthetic)}"
        synthetic[name] = _add(dims[field], _scale(axis_dim, order), -1)
        spelling[name] = match.group(0).strip()
        return name

    return _DERIVATIVE.sub(replace, side), synthetic, spelling, problems


def _with_spelling(node: ast.AST, spelling: Mapping[str, str]) -> str:
    text = _text(node)
    for name, original in spelling.items():
        text = re.sub(rf"\b{name}\b", original, text)
    return text


def check_dimensions(equation: str, *, problem: Any = None, units: Mapping[str, str] | None = None,
                     space: str | None = None, time: str | None = None) -> DimensionReport:
    """Check that every term of ``equation`` shares a dimension.

    ``equation`` is one statement, ``"dT/dt = alpha*laplacian(T) + q/(rho*cp)"``. Units come from ``problem`` when one
    is given and from ``units`` in either case, with ``units`` winning. ``space`` and ``time`` name the units behind
    ``laplacian``, ``grad``, ``div`` and the derivatives; without them they are taken from a coordinate whose unit is a
    length or a time, and the checker refuses rather than assuming metres and seconds.
    """
    declared = _units_from_problem(problem) if problem is not None else {}
    declared.update({k: v for k, v in dict(units or {}).items() if v})
    dims: dict[str, Dim] = {}
    report = DimensionReport(equation=equation)
    for name, unit in declared.items():
        try:
            dims[name] = unit_dim(unit)
        except ValueError as exc:
            report.problems.append(f"'{name}': {exc}")

    space_dim = unit_dim(space) if space else (_coordinate_dim("x", dims, _LENGTH, _TIME) or _LENGTH)
    time_dim = unit_dim(time) if time else (_coordinate_dim("t", dims, space_dim, _TIME) or _TIME)
    space_given = bool(space) or any(d == _LENGTH for name, d in dims.items() if name not in "t")
    time_given = bool(time) or any(d == _TIME for d in dims.values())

    normalised = _normalise(equation)
    if not space_given and re.search(r"\b(laplacian|lap|grad|div|d_dx|dx|d2_dx2)\b|\bd\^?\d?\w+\s*/\s*d[xyzr]",
                                     normalised):
        report.problems.append("a spatial derivative needs a length: give the unit of a coordinate "
                               "(units={'x': 'm'}) or pass space='m'")
        return report
    if not time_given and re.search(r"\b(dt|d_dt)\b|\bd\^?\d?\w+\s*/\s*dt?\b", normalised):
        report.problems.append("a time derivative needs a time: give the unit of a coordinate (units={'t': 's'}) "
                               "or pass time='s'")
        return report

    sides = _SPLIT.split(normalised)
    if len(sides) != 2:
        report.problems.append(f"the equation has to have exactly one '=', found {len(sides) - 1}")
        return report

    terms: list[Term] = []
    for side in sides:
        expanded, synthetic, spelling, problems = _expand_derivatives(side, dims, space_dim, time_dim)
        report.problems.extend(problems)
        if problems:
            return report
        try:
            tree = ast.parse(expanded.strip(), mode="eval").body
        except SyntaxError as exc:
            report.problems.append(f"cannot read '{side.strip()}': {exc.msg}")
            return report
        for node in _flatten_terms(tree):
            if isinstance(node, ast.Constant) and node.value == 0:
                continue                                  # "… = 0" says the expression vanishes; it is not a term
            try:
                terms.append(Term(_with_spelling(node, spelling), _eval(node, {**dims, **synthetic},
                                                                        space_dim, time_dim)))
            except (_Unknown, _Mismatch) as exc:
                report.problems.append(str(exc))
                return report
    report.terms = terms

    wanted = terms[0].dim if terms else _DIMLESS
    for term in terms[1:]:
        if term.dim != wanted:
            report.problems.append(f"'{term.text}' is {term.unit}, the other terms of the equation are "
                                   f"{format_dim(wanted)}")
    return report

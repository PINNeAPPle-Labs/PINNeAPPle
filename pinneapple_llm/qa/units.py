"""Dimensional analysis for the quality pipeline -- dependency-free, deliberately small.

Why not ``pint``: this repo keeps ``pinneapple_llm`` importable with no
heavy dependency (see ``pyproject.toml``'s optional extras), and adds
``pint`` -- a large package with a plugin system -- for the handful of
quantities a physics artifact actually declares would be a bad trade. The
cost is real and is stated rather than hidden: this parser understands a
FIXED grammar over a FIXED unit table. An unknown unit raises
:class:`UnitError`; it is never silently treated as dimensionless, which
would turn "I don't recognise this unit" into "everything is compatible".

Grammar accepted::

    unit      := factor (('*' | '/' | '.') factor)*
    factor    := name ('^' | '**')? integer?
    name      := a key of ``_UNITS``, optionally SI-prefixed

Examples: ``"m"``, ``"mm"``, ``"m/s"``, ``"m^2"``, ``"kg/m^3"``,
``"mm/year"``, ``"W/(m*K)"``, ``"1/s"``.

Base dimensions are the SI seven, in this order: length, time, mass,
temperature, electric current, amount, luminous intensity.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Tuple

Dim = Tuple[int, int, int, int, int, int, int]

_DIM_NAMES = ("L", "T", "M", "Theta", "I", "N", "J")
_DIMENSIONLESS: Dim = (0, 0, 0, 0, 0, 0, 0)


class UnitError(ValueError):
    """Raised for an unparsable unit string or an incompatible conversion."""


def _d(*exponents: int) -> Dim:
    if len(exponents) != 7:
        raise ValueError("a dimension needs exactly 7 exponents (L,T,M,Theta,I,N,J)")
    return tuple(exponents)  # type: ignore[return-value]


@dataclass(frozen=True)
class Unit:
    """An SI-scaled unit: ``si_value = value * scale + offset``."""

    name: str
    dim: Dim
    scale: float = 1.0
    offset: float = 0.0

    def __str__(self) -> str:
        return self.name

    @property
    def dimensionless(self) -> bool:
        return self.dim == _DIMENSIONLESS

    def to_si(self, value: float) -> float:
        return float(value) * self.scale + self.offset

    def from_si(self, si_value: float) -> float:
        return (float(si_value) - self.offset) / self.scale

    def compatible_with(self, other: "Unit") -> bool:
        return self.dim == other.dim

    def dimension_str(self) -> str:
        parts = [f"{n}^{e}" for n, e in zip(_DIM_NAMES, self.dim) if e]
        return "dimensionless" if not parts else "*".join(parts)


# --- base units -----------------------------------------------------------
_BASE: Dict[str, Unit] = {
    "m": Unit("m", _d(1, 0, 0, 0, 0, 0, 0)),
    "s": Unit("s", _d(0, 1, 0, 0, 0, 0, 0)),
    "kg": Unit("kg", _d(0, 0, 1, 0, 0, 0, 0)),
    "K": Unit("K", _d(0, 0, 0, 1, 0, 0, 0)),
    "A": Unit("A", _d(0, 0, 0, 0, 1, 0, 0)),
    "mol": Unit("mol", _d(0, 0, 0, 0, 0, 1, 0)),
    "cd": Unit("cd", _d(0, 0, 0, 0, 0, 0, 1)),
    # dimensionless
    "": Unit("", _DIMENSIONLESS),
    "1": Unit("1", _DIMENSIONLESS),
    "-": Unit("-", _DIMENSIONLESS),
    "%": Unit("%", _DIMENSIONLESS, scale=0.01),
    "rad": Unit("rad", _DIMENSIONLESS),
    "deg": Unit("deg", _DIMENSIONLESS, scale=math.pi / 180.0),
}

# --- named units, all expressed in SI base terms --------------------------
_UNITS: Dict[str, Unit] = dict(_BASE)
_UNITS.update({
    # length
    "in": Unit("in", _d(1, 0, 0, 0, 0, 0, 0), scale=0.0254),
    "ft": Unit("ft", _d(1, 0, 0, 0, 0, 0, 0), scale=0.3048),
    "mi": Unit("mi", _d(1, 0, 0, 0, 0, 0, 0), scale=1609.344),
    # time
    "min": Unit("min", _d(0, 1, 0, 0, 0, 0, 0), scale=60.0),
    "h": Unit("h", _d(0, 1, 0, 0, 0, 0, 0), scale=3600.0),
    "hr": Unit("hr", _d(0, 1, 0, 0, 0, 0, 0), scale=3600.0),
    "day": Unit("day", _d(0, 1, 0, 0, 0, 0, 0), scale=86400.0),
    "d": Unit("d", _d(0, 1, 0, 0, 0, 0, 0), scale=86400.0),
    "year": Unit("year", _d(0, 1, 0, 0, 0, 0, 0), scale=365.25 * 86400.0),
    "yr": Unit("yr", _d(0, 1, 0, 0, 0, 0, 0), scale=365.25 * 86400.0),
    "a": Unit("a", _d(0, 1, 0, 0, 0, 0, 0), scale=365.25 * 86400.0),
    # mass
    "g": Unit("g", _d(0, 0, 1, 0, 0, 0, 0), scale=1e-3),
    "t": Unit("t", _d(0, 0, 1, 0, 0, 0, 0), scale=1e3),
    "lb": Unit("lb", _d(0, 0, 1, 0, 0, 0, 0), scale=0.45359237),
    # temperature
    "degC": Unit("degC", _d(0, 0, 0, 1, 0, 0, 0), offset=273.15),
    "°C": Unit("°C", _d(0, 0, 0, 1, 0, 0, 0), offset=273.15),
    "degF": Unit("degF", _d(0, 0, 0, 1, 0, 0, 0), scale=5.0 / 9.0, offset=255.3722222222),
    "°F": Unit("°F", _d(0, 0, 0, 1, 0, 0, 0), scale=5.0 / 9.0, offset=255.3722222222),
    # force / pressure / energy / power
    "N": Unit("N", _d(1, -2, 1, 0, 0, 0, 0)),
    "kPa": Unit("kPa", _d(-1, -2, 1, 0, 0, 0, 0), scale=1e3),
    "MPa": Unit("MPa", _d(-1, -2, 1, 0, 0, 0, 0), scale=1e6),
    "GPa": Unit("GPa", _d(-1, -2, 1, 0, 0, 0, 0), scale=1e9),
    "Pa": Unit("Pa", _d(-1, -2, 1, 0, 0, 0, 0)),
    "bar": Unit("bar", _d(-1, -2, 1, 0, 0, 0, 0), scale=1e5),
    "mbar": Unit("mbar", _d(-1, -2, 1, 0, 0, 0, 0), scale=1e2),
    "J": Unit("J", _d(2, -2, 1, 0, 0, 0, 0)),
    "kJ": Unit("kJ", _d(2, -2, 1, 0, 0, 0, 0), scale=1e3),
    "W": Unit("W", _d(2, -3, 1, 0, 0, 0, 0)),
    "kW": Unit("kW", _d(2, -3, 1, 0, 0, 0, 0), scale=1e3),
    "MW": Unit("MW", _d(2, -3, 1, 0, 0, 0, 0), scale=1e6),
    # common derived
    "m2": Unit("m2", _d(2, 0, 0, 0, 0, 0, 0)),
    "m3": Unit("m3", _d(3, 0, 0, 0, 0, 0, 0)),
    "L": Unit("L", _d(3, 0, 0, 0, 0, 0, 0), scale=1e-3),
    "Hz": Unit("Hz", _d(0, -1, 0, 0, 0, 0, 0)),
})

# SI prefixes, longest first so "da" is tried before "d".
_PREFIXES = (
    ("da", 1e1), ("k", 1e3), ("M", 1e6), ("G", 1e9), ("T", 1e12),
    ("u", 1e-6), ("µ", 1e-6), ("μ", 1e-6), ("n", 1e-9), ("p", 1e-12),
    ("c", 1e-2), ("d", 1e-1), ("m", 1e-3),
)

_TOKEN = re.compile(r"^([A-Za-z°µμ%][A-Za-z0-9°µμ%]*)(?:[\^]{1,2}(-?\d+))?$")
_TRAILING_POWER = re.compile(r"^([A-Za-z°µμ%][A-Za-z°µμ%]*?)(\d+)$")


def _apply_power(unit: Unit, exponent: int, token: str) -> Unit:
    if exponent == 0:
        return Unit(token, _DIMENSIONLESS)
    dim = tuple(e * exponent for e in unit.dim)
    scale = unit.scale ** exponent
    if unit.offset and exponent != 1:
        raise UnitError(
            f"unit '{token}': a temperature-offset unit cannot be raised to a power"
        )
    return Unit(token, dim, scale=scale, offset=unit.offset)


def _lookup(name: str) -> Unit:
    """Resolve one *name* (no operators, no exponent) to a :class:`Unit`."""
    if name in _UNITS:
        return _UNITS[name]
    for prefix, factor in _PREFIXES:
        if name.startswith(prefix) and len(name) > len(prefix):
            rest = name[len(prefix):]
            if rest in _UNITS:
                base = _UNITS[rest]
                # Only prefix BASE units. Prefixing a derived table entry
                # ("m2" -> milli-*) would multiply a squared scale by a
                # linear factor: 1 mm^2 is 1e-6 m^2, not 1e-3 m^2.
                if base.offset:
                    raise UnitError(
                        f"unit '{name}': SI-prefixing a temperature-offset unit ('{rest}') is ambiguous"
                    )
                if sum(abs(e) for e in base.dim) != 1:
                    raise UnitError(
                        f"unit '{name}': '{rest}' is a derived unit and cannot be SI-prefixed here "
                        f"-- write the base units explicitly (e.g. 'mm^2' instead of 'mm2')"
                    )
                return Unit(name, base.dim, scale=base.scale * factor)
    raise UnitError(
        f"unknown unit '{name}' -- this parser only accepts units in its own table "
        f"(add it to pinneapple_llm/qa/units.py if it is genuinely needed); "
        "an unrecognised unit is an error, never silently dimensionless"
    )


def _split_top_level(text: str) -> Tuple[str, list]:
    """Split ``"W/(m*K)"`` into numerator ``"W"`` and denominator factors."""
    # strip one level of redundant parentheses around the whole expression
    t = text.strip()
    if t.startswith("(") and t.endswith(")") and _balanced(t[1:-1]):
        t = t[1:-1].strip()
    if not t:
        raise UnitError("empty unit string")
    without_outer = _remove_group_parens(t)
    parts = re.split(r"(?<![\^*])([/.])(?!$)", without_outer)
    # re-join: re.split with a capturing group alternates operand/operator
    numerator = parts[0]
    denominator = parts[2::2]
    return numerator, denominator


def _balanced(text: str) -> bool:
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _remove_group_parens(text: str) -> str:
    """``"a/(b*c)"`` -> ``"a/b*c"``; only single-level groups are supported."""
    out = []
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
            if depth > 1:
                raise UnitError(f"unit '{text}': nested parentheses are not supported")
            continue
        if ch == ")":
            depth -= 1
            continue
        if ch == "/" and depth == 1:
            # a '/' inside a denominator group means the whole group divides:
            # a/(b/c) is unsupported rather than silently mis-parsed
            raise UnitError(f"unit '{text}': a '/' inside parentheses is not supported")
        out.append(ch)
    if depth != 0:
        raise UnitError(f"unit '{text}': unbalanced parentheses")
    return "".join(out)


def _parse_factor(token: str) -> Unit:
    token = token.strip()
    if not token:
        raise UnitError("empty factor in unit expression")
    if token in ("1", "-"):
        return _UNITS[token]
    m = _TOKEN.match(token)
    if not m:
        raise UnitError(f"cannot parse unit factor '{token}'")
    name, exp = m.group(1), m.group(2)
    try:
        base = _lookup(name)
    except UnitError:
        # ``"m2"``/``"s-1"``-style suffix powers: only reached when the name
        # is not itself a table entry, so ``"m2"`` (table) still wins there.
        suffix = _TRAILING_POWER.match(name)
        if suffix is None:
            raise
        base = _lookup(suffix.group(1))
        return _apply_power(base, int(suffix.group(2)), token)
    return _apply_power(base, int(exp) if exp is not None else 1, token)


def parse_unit(text: str) -> Unit:
    """Parse a unit string into a :class:`Unit`. Raises :class:`UnitError`."""
    if text is None:
        raise UnitError("unit is None")
    text = str(text).strip()
    if text == "":
        return _UNITS[""]
    numerator, denominator = _split_top_level(text)
    unit = Unit("", _DIMENSIONLESS, scale=1.0)
    for factor_text in re.split(r"[*·]", numerator):
        unit = _multiply(unit, _parse_factor(factor_text), text)
    for factor_text in denominator:
        for piece in re.split(r"[*·]", factor_text):
            unit = _divide(unit, _parse_factor(piece), text)
    # A single factor keeps its offset (``degC``), so parse -> to_si -> parse
    # round-trips as an affine conversion; a compound expression never has one.
    offset = 0.0
    if not denominator:
        factors = [f for f in re.split(r"[*·]", numerator) if f.strip()]
        if len(factors) == 1:
            offset = _parse_factor(factors[0]).offset
    return Unit(text, unit.dim, scale=unit.scale, offset=offset)


def _multiply(a: Unit, b: Unit, context: str) -> Unit:
    if (a.offset or b.offset) and not (a.dimensionless and b.dimensionless):
        if a.offset and b.offset:
            raise UnitError(f"unit '{context}': cannot multiply two temperature-offset units")
    dim = tuple(x + y for x, y in zip(a.dim, b.dim))
    return Unit(context, dim, scale=a.scale * b.scale)


def _divide(a: Unit, b: Unit, context: str) -> Unit:
    if b.offset:
        raise UnitError(f"unit '{context}': cannot divide by a temperature-offset unit")
    dim = tuple(x - y for x, y in zip(a.dim, b.dim))
    return Unit(context, dim, scale=a.scale / b.scale)


def are_compatible(a: str, b: str) -> bool:
    """True if two unit strings have the same physical dimension."""
    return parse_unit(a).dim == parse_unit(b).dim


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert ``value`` between two compatible units. Raises on incompatibility."""
    src, dst = parse_unit(from_unit), parse_unit(to_unit)
    if src.dim != dst.dim:
        raise UnitError(
            f"cannot convert {value} from '{from_unit}' ({src.dimension_str()}) to "
            f"'{to_unit}' ({dst.dimension_str()}): different physical dimensions"
        )
    return dst.from_si(src.to_si(value))


def dimension_of(unit: str) -> Dim:
    return parse_unit(unit).dim


def dimension_str(unit: str) -> str:
    return parse_unit(unit).dimension_str()

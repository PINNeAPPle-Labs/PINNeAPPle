"""Physical units and quantities for engineering data (PINNeAPPle Physical Data Layer).

Dependency-free (standard library only) so that data tools can use it without the
training stack. It answers three questions about a column of engineering data:

1. **What unit is it in?** ``split_header("T_supply [°C]") -> ("T_supply", "°C")`` and
   ``parse_unit("kg/(m·s)")`` turn header text into a :class:`Unit` with an SI factor,
   an offset (temperatures only) and a dimension vector.
2. **What physical quantity is it?** ``infer_quantity(name, unit)`` combines the unit's
   dimension with keywords in the name (pressure and stress share a dimension; the name
   decides) and returns a :class:`QuantityGuess` with a confidence and the reason.
3. **Is a value physically possible?** Every quantity in :data:`QUANTITIES` carries hard
   limits (impossible values, e.g. an absolute temperature below 0 K) and a typical
   engineering range (suspicious values), both in SI.

Conversions are exact definitions where they exist (NIST SP 811, Appendix B).
Temperature *differences* are not distinguished from temperatures: a column of
"ΔT [°F]" converts with the absolute-temperature offset. :func:`convert` therefore takes
``difference=True`` for those columns, and :func:`infer_quantity` flags names with
"delta"/"diff"/"Δ".
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# Dimension vector: (m, kg, s, K, A, mol)
Dim = Tuple[int, int, int, int, int, int]
DIMLESS: Dim = (0, 0, 0, 0, 0, 0)
_L, _M, _T, _K, _A, _N = (1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 1)


def _d(*parts: Tuple[Dim, int]) -> Dim:
    out = [0] * 6
    for dim, p in parts:
        for i in range(6):
            out[i] += dim[i] * p
    return tuple(out)  # type: ignore[return-value]


@dataclass(frozen=True)
class Unit:
    """A unit: ``SI value = value * factor + offset``."""
    symbol: str
    factor: float
    dim: Dim
    offset: float = 0.0
    gauge: bool = False          # pressure relative to atmosphere (psig, barg)

    def to_si(self, x, difference: bool = False):
        return x * self.factor + (0.0 if difference else self.offset)

    def from_si(self, x, difference: bool = False):
        return (x - (0.0 if difference else self.offset)) / self.factor


# ── unit table ──────────────────────────────────────────────────────────────
# symbol -> (factor to SI, dimension, offset)
_PA = _d((_M, 1), (_L, -1), (_T, -2))
_W = _d((_M, 1), (_L, 2), (_T, -3))
_J = _d((_M, 1), (_L, 2), (_T, -2))
_N_ = _d((_M, 1), (_L, 1), (_T, -2))
_V = _d((_M, 1), (_L, 2), (_T, -3), (_A, -1))
_INCH, _FT, _LB = 0.0254, 0.3048, 0.45359237
_LBF = _LB * 9.80665
_BTU = 1055.05585262          # International Table BTU
_GAL = 3.785411784e-3         # US gallon

_BASE: Dict[str, Tuple[float, Dim, float]] = {
    # length
    "m": (1, _L, 0), "mm": (1e-3, _L, 0), "cm": (1e-2, _L, 0), "km": (1e3, _L, 0), "um": (1e-6, _L, 0),
    "µm": (1e-6, _L, 0), "in": (_INCH, _L, 0), "ft": (_FT, _L, 0), "mi": (1609.344, _L, 0),
    # time
    "s": (1, _T, 0), "ms": (1e-3, _T, 0), "us": (1e-6, _T, 0), "min": (60, _T, 0), "h": (3600, _T, 0),
    "hr": (3600, _T, 0), "d": (86400, _T, 0), "day": (86400, _T, 0),
    # mass
    "kg": (1, _M, 0), "g": (1e-3, _M, 0), "mg": (1e-6, _M, 0), "t": (1e3, _M, 0), "lb": (_LB, _M, 0),
    "lbm": (_LB, _M, 0),
    # temperature
    "K": (1, _K, 0), "°C": (1, _K, 273.15), "°F": (5 / 9, _K, 273.15 - 32 * 5 / 9), "°R": (5 / 9, _K, 0),
    # amount, current
    "mol": (1, _N, 0), "kmol": (1e3, _N, 0), "A": (1, _A, 0), "mA": (1e-3, _A, 0), "kA": (1e3, _A, 0),
    # pressure / stress
    "Pa": (1, _PA, 0), "mPa": (1e-3, _PA, 0), "hPa": (1e2, _PA, 0), "kPa": (1e3, _PA, 0), "MPa": (1e6, _PA, 0), "GPa": (1e9, _PA, 0),
    "bar": (1e5, _PA, 0), "mbar": (1e2, _PA, 0), "psi": (_LBF / _INCH ** 2, _PA, 0), "ksi": (1e3 * _LBF / _INCH ** 2, _PA, 0),
    "atm": (101325, _PA, 0), "torr": (101325 / 760, _PA, 0), "mmHg": (133.322387415, _PA, 0),
    "inH2O": (249.08891, _PA, 0), "mmH2O": (9.80665, _PA, 0), "inHg": (3386.389, _PA, 0),
    # energy, power
    "J": (1, _J, 0), "kJ": (1e3, _J, 0), "MJ": (1e6, _J, 0), "Wh": (3600, _J, 0), "kWh": (3.6e6, _J, 0),
    "MWh": (3.6e9, _J, 0), "BTU": (_BTU, _J, 0), "cal": (4.184, _J, 0), "kcal": (4184, _J, 0),
    "W": (1, _W, 0), "mW": (1e-3, _W, 0), "kW": (1e3, _W, 0), "MW": (1e6, _W, 0), "hp": (745.69987158227022, _W, 0),
    "TR": (12000 * _BTU / 3600, _W, 0), "BTU/h": (_BTU / 3600, _W, 0), "MBH": (1000 * _BTU / 3600, _W, 0),
    # force, torque handled as N*m
    "N": (1, _N_, 0), "kN": (1e3, _N_, 0), "lbf": (_LBF, _N_, 0), "kgf": (9.80665, _N_, 0),
    # electrical
    "V": (1, _V, 0), "mV": (1e-3, _V, 0), "kV": (1e3, _V, 0), "ohm": (1, _d((_V, 1), (_A, -1)), 0),
    "Ω": (1, _d((_V, 1), (_A, -1)), 0), "VA": (1, _W, 0), "kVA": (1e3, _W, 0), "var": (1, _W, 0), "kvar": (1e3, _W, 0),
    # frequency / rotation (rpm is revolutions per minute -> 1/s)
    "Hz": (1, _d((_T, -1)), 0), "kHz": (1e3, _d((_T, -1)), 0), "rpm": (1 / 60, _d((_T, -1)), 0),
    # volume
    "L": (1e-3, _d((_L, 3)), 0), "l": (1e-3, _d((_L, 3)), 0), "mL": (1e-6, _d((_L, 3)), 0), "gal": (_GAL, _d((_L, 3)), 0),
    # common flow shorthands
    "gpm": (_GAL / 60, _d((_L, 3), (_T, -1)), 0), "lpm": (1e-3 / 60, _d((_L, 3), (_T, -1)), 0),
    "cfm": (_FT ** 3 / 60, _d((_L, 3), (_T, -1)), 0),
    # viscosity
    "P": (0.1, _d((_PA, 1), (_T, 1)), 0), "cP": (1e-3, _d((_PA, 1), (_T, 1)), 0),
    "St": (1e-4, _d((_L, 2), (_T, -1)), 0), "cSt": (1e-6, _d((_L, 2), (_T, -1)), 0),
    # angle (dimensionless)
    "rad": (1, DIMLESS, 0), "deg": (math.pi / 180, DIMLESS, 0), "°": (math.pi / 180, DIMLESS, 0),
    # ratios
    "%": (1e-2, DIMLESS, 0), "ppm": (1e-6, DIMLESS, 0), "-": (1, DIMLESS, 0), "1": (1, DIMLESS, 0),
    "fraction": (1, DIMLESS, 0), "µε": (1e-6, DIMLESS, 0), "ustrain": (1e-6, DIMLESS, 0),
}

# alias -> canonical symbol (case-sensitive first, then lower-case lookup)
_ALIASES = {
    "degC": "°C", "deg C": "°C", "deg_c": "°C", "degc": "°C", "C": "°C", "ºC": "°C", "celsius": "°C", "Celsius": "°C",
    "degF": "°F", "deg F": "°F", "degf": "°F", "F": "°F", "ºF": "°F", "fahrenheit": "°F", "Fahrenheit": "°F",
    "kelvin": "K", "Kelvin": "K", "degK": "K", "R": "°R", "degR": "°R", "rankine": "°R",
    "sec": "s", "secs": "s", "seconds": "s", "mins": "min", "minute": "min", "minutes": "min",
    "hrs": "h", "hour": "h", "hours": "h", "days": "d",
    "Bar": "bar", "BAR": "bar", "PSI": "psi", "Psi": "psi", "kpa": "kPa", "KPa": "kPa", "mpa": "MPa", "pa": "Pa",
    "KW": "kW", "kw": "kW", "Kw": "kW", "MW": "MW", "mw": "MW", "kwh": "kWh", "KWh": "kWh", "KWH": "kWh", "mwh": "MWh",
    "Btu": "BTU", "btu": "BTU", "Btu/h": "BTU/h", "btu/h": "BTU/h", "BTUH": "BTU/h", "Btuh": "BTU/h", "btuh": "BTU/h",
    "BTU/hr": "BTU/h", "Btu/hr": "BTU/h", "tons": "TR", "ton": "TR", "RT": "TR", "tonR": "TR",
    "GPM": "gpm", "LPM": "lpm", "l/min": "lpm", "L/min": "lpm", "CFM": "cfm",
    "RPM": "rpm", "rev/min": "rpm", "1/min": "rpm", "hz": "Hz", "HZ": "Hz",
    "v": "V", "volt": "V", "volts": "V", "amp": "A", "amps": "A", "Amps": "A", "ampere": "A",
    "liter": "L", "liters": "L", "litre": "L", "litres": "L", "gallon": "gal", "gallons": "gal", "usgal": "gal",
    "lbs": "lb", "pct": "%", "percent": "%", "perc": "%", "%RH": "%", "RH%": "%", "%rh": "%",
    "micron": "µm", "microns": "µm", "μm": "µm", "με": "µε", "inch": "in", "inches": "in", "feet": "ft", "foot": "ft",
    "mmhg": "mmHg", "inwc": "inH2O", "inWC": "inH2O", "in.w.c.": "inH2O", "mmwc": "mmH2O", "mmH₂O": "mmH2O",
    "cps": "cP", "cp": "cP", "cst": "cSt", "degree": "deg", "degrees": "deg", "Ohm": "ohm", "ohms": "ohm",
    "kgf/cm2": "kgf/cm^2",
}

# Gauge pressure: same scale, relative to local atmosphere
_GAUGE = {"psig": "psi", "barg": "bar", "kPag": "kPa", "kPa(g)": "kPa", "bar(g)": "bar", "psi(g)": "psi"}
_ABS = {"psia": "psi", "bara": "bar", "kPaa": "kPa", "kPa(a)": "kPa", "bar(a)": "bar", "psi(a)": "psi"}

_SUPERSCRIPT = str.maketrans({"²": "^2", "³": "^3", "¹": "^1", "⁻": "^-", "·": "*", "⋅": "*", "×": "*"})


class UnitError(ValueError):
    pass


def _lookup(tok: str) -> Optional[Tuple[float, Dim, float]]:
    if tok in _BASE:
        return _BASE[tok]
    if tok in _ALIASES:
        return _lookup(_ALIASES[tok]) if _ALIASES[tok] != tok else None
    low = tok.lower()
    for k in (low,):
        if k in _ALIASES:
            return _lookup(_ALIASES[k])
    for k, v in _BASE.items():
        # case-insensitive fallback, but never across the milli/mega prefixes (mPa is not MPa, mW is not MW)
        if k.lower() == low and k not in ("mm", "Mm") and not (tok[:1] in "mM" and k[:1] != tok[:1] and len(tok) > 1):
            return v
    return None


def parse_unit(text: str) -> Unit:
    """Parse a unit expression: ``°C``, ``kg/m^3``, ``W/(m·K)``, ``m³/h``, ``N*m``, ``psig``.

    Raises :class:`UnitError` if any token is unknown."""
    if text is None:
        raise UnitError("no unit")
    raw = str(text).strip()
    s = raw.translate(_SUPERSCRIPT).strip()
    if not s:
        raise UnitError("empty unit")
    for table, is_gauge in ((_GAUGE, True), (_ABS, False)):
        for k, base in table.items():
            if s.lower() == k.lower():
                u = parse_unit(base)
                return Unit(raw, u.factor, u.dim, u.offset, gauge=is_gauge)
    whole = _lookup(s)
    if whole is not None:                       # e.g. "BTU/h", "%RH", "deg C"
        f, dim, off = whole
        return Unit(raw, f, dim, off)
    if s.lower() in ("m3h", "m3/hr", "m³h", "cmh"):
        s = "m^3/h"
    # tokenise into numerator / denominator (one level of parentheses after the "/")
    s = re.sub(r"(?<=[A-Za-z])\.(?=[A-Za-z])", "*", s.replace(" per ", "/").replace(" ", "*"))
    num, _, den = s.partition("/")
    den = den.strip("()").replace("/", "*")
    factor, dim = 1.0, [0] * 6

    def apply(part: str, sign: int) -> None:
        nonlocal factor
        for tok in filter(None, part.strip("()").split("*")):
            m = re.fullmatch(r"(.+?)(?:\^(-?\d+))?", tok)
            name, power = m.group(1), int(m.group(2) or 1)
            if _lookup(name) is None:                     # "m3", "s2" without a caret
                mm = re.fullmatch(r"([A-Za-zµμ]+)([234])", name)
                if mm:
                    name, power = mm.group(1), int(mm.group(2)) * power
            hit = _lookup(name)
            if hit is None:
                raise UnitError(f"unknown unit '{name}' in '{raw}'")
            f, d, _off = hit                              # °C inside a compound is an interval
            factor *= f ** (power * sign)
            for i in range(6):
                dim[i] += d[i] * power * sign

    apply(num, +1)
    if den:
        apply(den, -1)
    return Unit(raw, factor, tuple(dim), 0.0)  # type: ignore[arg-type]


def try_parse_unit(text: Optional[str]) -> Optional[Unit]:
    try:
        return parse_unit(text) if text else None
    except UnitError:
        return None


def convert(values, src: str, dst: str, difference: bool = False):
    """Convert values (scalar or numpy array) from ``src`` to ``dst``."""
    a, b = parse_unit(src), parse_unit(dst)
    if a.dim != b.dim:
        raise UnitError(f"cannot convert {src} ({dim_str(a.dim)}) to {dst} ({dim_str(b.dim)})")
    return b.from_si(a.to_si(values, difference), difference)


def dim_str(dim: Dim) -> str:
    names = ("m", "kg", "s", "K", "A", "mol")
    parts = [f"{n}^{p}" if p != 1 else n for n, p in zip(names, dim) if p]
    return "·".join(parts) or "dimensionless"


# ── header parsing ──────────────────────────────────────────────────────────
_BRACKETS = [r"\[([^\]]+)\]\s*$", r"\(([^)]+)\)\s*$", r"\{([^}]+)\}\s*$", r",\s*([^,]+)$", r"\s+in\s+(\S+)$"]
_SUFFIX_SEP = re.compile(r"[_\-. ]+")
# single-letter / very short suffixes that are units only if the name says the quantity
_WEAK_SUFFIX = {"c", "f", "k", "r", "a", "v", "w", "t", "d", "g", "h", "s", "l", "m", "n", "p", "in", "st", "min", "mi", "1", "-"}


def split_header(header: str) -> Tuple[str, Optional[str]]:
    """Split a column header into (name, unit text). ``"T_supply [°C]"`` -> ``("T_supply", "°C")``;
    also understands ``Pressure (bar)``, ``flow, m3/h``, ``power in kW`` and suffixes like
    ``T_inlet_degC``, ``pressure_bar`` or ``vibration_mm_s``. Returns ``(header, None)`` when no
    unit is found. A short suffix (``_C``, ``_A``) counts as a unit only when the rest of the name
    names a matching quantity (``T_out_C`` yes, ``Line_A`` no)."""
    h = str(header).strip()
    for pat in _BRACKETS:
        m = re.search(pat, h)
        if m and try_parse_unit(m.group(1).strip()) is not None:
            return h[:m.start()].strip(" _-,"), m.group(1).strip()
    parts = [p for p in _SUFFIX_SEP.split(h) if p]
    if len(parts) < 2:
        return h, None
    cands = []
    if len(parts) >= 3:
        a, b = parts[-2], parts[-1]
        cands.append((2, ("deg" + b) if a.lower() == "deg" else f"{a}/{b}"))
    cands.append((1, parts[-1]))
    for n, txt in cands:
        u = try_parse_unit(txt)
        if u is None:
            continue
        rest = parts[:-n]
        if n == 2 and not _quantities_for_dim(u.dim) and not txt.lower().startswith("deg"):
            continue
        if (txt.lower() in _WEAK_SUFFIX or n == 2) and not _name_hints(rest, u):
            continue
        return "_".join(rest), txt
    return h, None


def _name_hints(rest: Sequence[str], u: "Unit") -> bool:
    toks = _tokens(" ".join(rest))
    return any(_kw_score(QUANTITIES[qn], toks) for qn in _quantities_for_dim(u.dim))


# ── quantities ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Quantity:
    name: str
    si_unit: str
    dim: Dim
    keywords: Tuple[str, ...]
    hard: Tuple[Optional[float], Optional[float]] = (None, None)      # impossible outside (SI)
    typical: Tuple[Optional[float], Optional[float]] = (None, None)   # unusual outside (SI)
    display: str = ""                                                 # preferred engineering unit


QUANTITIES: Dict[str, Quantity] = {q.name: q for q in [
    Quantity("temperature", "K", _K, ("temp", "tmp", "t", "degc", "ambient", "oat", "lwt", "ewt", "sat", "rat", "mat", "dat", "tc",
             "tin", "tout", "thermo", "cht", "egt"),
             (0.0, None), (173.15, 2273.15), "°C"),
    Quantity("pressure", "Pa", _PA, ("press", "pres", "p", "dp", "head", "suct", "disch", "static", "vacuum", "psi"), (None, None), (-1.1e5, 1e9), "bar"),
    Quantity("stress", "Pa", _PA, ("stress", "sigma", "von", "mises", "tensile", "yield"), (None, None), (-5e9, 5e9), "MPa"),
    Quantity("mass_flow_rate", "kg/s", _d((_M, 1), (_T, -1)), ("mass", "mdot", "flow", "rate", "massflow"), (None, None),
             (-1e5, 1e5), "kg/s"),
    Quantity("volumetric_flow_rate", "m^3/s", _d((_L, 3), (_T, -1)), ("flow", "vol", "q", "gpm", "cfm", "rate", "fr", "airflow"),
             (None, None), (-1e3, 1e3), "m^3/h"),
    Quantity("velocity", "m/s", _d((_L, 1), (_T, -1)), ("vel", "velo", "speed", "wind", "u", "v", "w", "airspeed", "vib", "vibration"),
             (None, None), (-1e3, 1e3), "m/s"),
    Quantity("power", "W", _W, ("power", "kw", "load", "demand", "capa", "heat", "duty", "q", "consumption", "pwr", "cooling"), (None, None), (-1e10, 1e10), "kW"),
    Quantity("energy", "J", _J, ("energy", "kwh", "consumption", "meter", "work", "heat"), (None, None), (None, None), "kWh"),
    Quantity("torque", "N*m", _J, ("torque", "moment", "tq"), (None, None), (-1e8, 1e8), "N*m"),
    Quantity("force", "N", _N_, ("force", "load", "thrust", "weight", "lbf"), (None, None), (-1e9, 1e9), "kN"),
    Quantity("voltage", "V", _V, ("volt", "voltage", "vac", "vdc", "u", "v"), (None, None), (-1e6, 1e6), "V"),
    Quantity("current", "A", _A, ("current", "amp", "i", "amps"), (None, None), (-1e5, 1e5), "A"),
    Quantity("frequency", "Hz", _d((_T, -1)), ("freq", "hz", "rpm", "speed", "rotation", "rev"), (0.0, None), (0.0, 1e6), "Hz"),
    Quantity("length", "m", _L, ("length", "level", "height", "depth", "posi", "displ", "pos", "x", "y", "z", "dist", "gap",
             "thick", "elev"), (None, None), (-1e5, 1e5), "mm"),
    Quantity("mass", "kg", _M, ("mass", "weight", "wt"), (0.0, None), (0.0, 1e7), "kg"),
    Quantity("time", "s", _T, ("time", "duration", "elapsed", "runtime"), (None, None), (None, None), "s"),
    Quantity("density", "kg/m^3", _d((_M, 1), (_L, -3)), ("dens", "rho"), (0.0, None), (1e-3, 2.3e4), "kg/m^3"),
    Quantity("dynamic_viscosity", "Pa*s", _d((_PA, 1), (_T, 1)), ("visc", "mu"), (0.0, None), (1e-6, 1e5), "cP"),
    Quantity("kinematic_viscosity", "m^2/s", _d((_L, 2), (_T, -1)), ("visc", "nu"), (0.0, None), (1e-8, 1e-1), "cSt"),
    Quantity("heat_flux", "W/m^2", _d((_M, 1), (_T, -3)), ("flux", "irradiance", "radiation", "solar", "ghi"),
             (None, None), (-1e8, 1e8), "W/m^2"),
    Quantity("thermal_conductivity", "W/(m*K)", _d((_W, 1), (_L, -1), (_K, -1)), ("conduct", "k"), (0.0, None),
             (1e-3, 3e3), "W/(m*K)"),
    Quantity("acceleration", "m/s^2", _d((_L, 1), (_T, -2)), ("acc", "accel", "vib", "vibration"), (None, None), (-1e4, 1e4), "m/s^2"),
    Quantity("relative_humidity", "1", DIMLESS, ("rh", "humid", "hum"), (0.0, 1.0), (0.0, 1.0), "%"),
    Quantity("fraction", "1", DIMLESS, ("eff", "effi", "ratio", "frac", "pct", "perc", "open", "valve", "posi", "duty", "cop", "pf",
             "util", "%", "load", "speed", "cmd"), (None, None), (None, None), "%"),
    Quantity("angle", "rad", DIMLESS, ("angle", "deg", "direction", "dir", "heading", "phase", "theta"), (None, None),
             (None, None), "deg"),
    Quantity("strain", "1", DIMLESS, ("strain", "eps", "microstrain"), (None, None), (-0.5, 0.5), "µε"),
    Quantity("electrical_resistance", "ohm", _d((_V, 1), (_A, -1)), ("resist", "ohm", "insulation"), (0.0, None),
             (0.0, 1e12), "ohm"),
]}


def _quantities_for_dim(dim: Dim) -> List[str]:
    return [q.name for q in QUANTITIES.values() if q.dim == dim]


@dataclass
class QuantityGuess:
    quantity: Optional[str]
    confidence: float                  # 0..1
    reason: str
    alternatives: List[str] = field(default_factory=list)
    is_difference: bool = False


_DIFF = re.compile(r"(^|[_\s\-])(delta|diff|dt|d_t|Δ)([_\s\-]|$)|Δ|delta", re.I)


def _tokens(name: str) -> List[str]:
    n = re.sub(r"([a-z])([A-Z])", r"\1_\2", str(name))
    return [t for t in re.split(r"[^a-z0-9%°]+", n.lower()) if t]


def _kw_score(q: "Quantity", toks: Sequence[str]) -> int:
    """Keywords of 1-3 characters must equal a token; longer ones may prefix a token
    (``temp`` matches ``temperature``)."""
    hits = 0
    for k in q.keywords:
        for t in toks:
            if t == k or (len(k) >= 4 and t.startswith(k)):
                hits += 1
                break
    return hits


_TIME_NAMES = {"time", "timestamp", "date", "datetime", "ts", "utc", "localtime", "epoch"}


def infer_quantity(name: str, unit: Optional[Unit] = None) -> QuantityGuess:
    """Infer the physical quantity of a column from its unit dimension and name keywords."""
    tok = _tokens(name)
    is_diff = bool(_DIFF.search(str(name)))

    def score(q: Quantity) -> int:
        return _kw_score(q, tok)

    if unit is None and tok and set(tok) <= _TIME_NAMES | {"local", "stamp"}:
        return QuantityGuess("time", 0.5, "time-axis name", is_difference=False)

    if unit is not None:
        cands = _quantities_for_dim(unit.dim)
        if not cands:
            return QuantityGuess(None, 0.3, f"unit {unit.symbol} ({dim_str(unit.dim)}) has no catalogued quantity",
                                 is_difference=is_diff)
        if unit.symbol.strip().lower() in ("%rh", "rh%") or (unit.dim == DIMLESS and score(QUANTITIES["relative_humidity"])):
            return QuantityGuess("relative_humidity", 0.9, "humidity keyword with a ratio unit", is_difference=is_diff)
        if unit.dim == DIMLESS:
            sym = unit.symbol.strip().lower()
            if sym in ("deg", "°", "rad", "degree", "degrees"):
                return QuantityGuess("angle", 0.85, f"unit {unit.symbol} is an angle", is_difference=is_diff)
            if sym in ("µε", "ustrain"):
                return QuantityGuess("strain", 0.9, "strain unit", is_difference=is_diff)
            ranked = sorted(cands, key=lambda c: -score(QUANTITIES[c]))
            best = ranked[0] if score(QUANTITIES[ranked[0]]) else "fraction"
            return QuantityGuess(best, 0.6, f"dimensionless unit {unit.symbol}; name suggests {best}",
                                 [c for c in ranked if c != best][:3], is_diff)
        if len(cands) == 1:
            return QuantityGuess(cands[0], 0.95, f"unit {unit.symbol} is a {cands[0].replace('_', ' ')} unit",
                                 is_difference=is_diff)
        ranked = sorted(cands, key=lambda c: -score(QUANTITIES[c]))
        top = score(QUANTITIES[ranked[0]])
        if top and (len(ranked) < 2 or top > score(QUANTITIES[ranked[1]])):
            return QuantityGuess(ranked[0], 0.9, f"unit {unit.symbol} + name keyword", ranked[1:], is_diff)
        default = {_PA: "pressure", _J: "energy", _d((_T, -1)): "frequency", DIMLESS: "fraction"}.get(unit.dim, ranked[0])
        return QuantityGuess(default, 0.6, f"unit {unit.symbol} fits {', '.join(cands)}; assumed {default}",
                             [c for c in cands if c != default], is_diff)
    # no unit: keywords only (weak)
    ranked = sorted(QUANTITIES.values(), key=lambda q: -score(q))
    if score(ranked[0]) == 0:
        return QuantityGuess(None, 0.0, "no unit and no recognisable keyword", is_difference=is_diff)
    alts = [q.name for q in ranked[1:4] if score(q)]
    return QuantityGuess(ranked[0].name, 0.35, "name keyword only, no unit given", alts, is_diff)


def describe_unit(u: Unit) -> Dict[str, object]:
    return {"symbol": u.symbol, "si_factor": u.factor, "si_offset": u.offset, "dimension": dim_str(u.dim),
            "gauge": u.gauge}


def si_unit_for(quantity: str) -> str:
    return QUANTITIES[quantity].si_unit


__all__ = ["Unit", "UnitError", "Quantity", "QuantityGuess", "QUANTITIES", "parse_unit", "try_parse_unit",
           "convert", "split_header", "infer_quantity", "dim_str", "describe_unit", "si_unit_for", "DIMLESS"]

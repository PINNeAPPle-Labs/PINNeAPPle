"""Physical units on a ``PhysicalSample``, both ways, through astropy (#522).

``PhysicalSample`` already carries units, as *text*: ``schema["units"]["T"] = "K"``, or the ``units`` attribute of an
xarray variable, or ``schema["variables"]["T"]["unit"]``. ``pinneapple_data.physical_units`` reads and converts that
text without a dependency (that is the path a CSV takes, and it stays that way). What it cannot do is talk to the rest
of astronomy: ``astropy.units.Quantity`` is what a FITS file, a catalogue or a VizieR table arrives in, and a quantity
carries its dimension *and* the operation that produced it.

This module is the bridge, and it is deliberately strict:

    from pinneapple_data.astro import units
    q = units.quantity_of(sample, "T")                  # the field, as a Quantity in the unit it declares
    units.set_quantity(sample, "T", 300 * u.K)          # back in, converting to the declared unit
    units.set_quantity(sample, "T", 3 * u.m)            # raises UnitMismatch: no number is written

* units that are **compatible but different** are converted (``300 K`` into a field declared in ``degC``), and the
  conversion is recorded in ``provenance["unit_conversions"]`` -- a conversion that leaves no trace is how a factor of
  1000 gets lost in a plot nobody questions;
* units that are **not compatible** are refused with :class:`UnitMismatch`, which says which dimension each side has
  and what to do; the field is left exactly as it was. Nothing is dropped to a bare float, and no field is filled with
  a number in the wrong unit;
* a field with **no declared unit** accepts the quantity's unit and writes it down, so the next reader knows.

The same two-way conversion is offered for the lattice-unit helper of the LBM solver
(:class:`pinneapple_simulation.numerical_solvers.lbm_tunnel.Units`), which is the other place in the library where a
number's unit lives in a formula instead of in the data: :func:`to_lattice` and :func:`from_lattice` convert a Quantity
through it and refuse a quantity of the wrong dimension.

astropy is optional. Nothing here imports it until a function is called, so ``import pinneapple_data`` keeps working on
a machine without the extra; a caller that does need it gets an ImportError carrying the install line.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from pinneapple_data.physical_units import UnitError

__all__ = [
    "ASTROPY_INSTALL_HINT",
    "LATTICE_KINDS",
    "UnitMismatch",
    "available",
    "declared_unit",
    "from_lattice",
    "quantity_of",
    "sample_quantities",
    "set_quantity",
    "to_lattice",
]

ASTROPY_INSTALL_HINT = 'units on a sample need astropy: pip install astropy (or pip install "pinneapple[astro]")'

#: The lattice quantities the LBM helper converts, with the SI unit each one is expressed in and the helper's method
#: for the way back. ``None`` marks a direction the helper only offers one way: lattice density, kinematic viscosity
#: and force density have no ``si_*`` method, so :func:`from_lattice` refuses them with that reason instead of
#: inventing a factor.
LATTICE_KINDS: dict[str, tuple[str, str | None, str]] = {
    "x": ("m", "si_x", "length"),
    "u": ("m/s", "si_u", "velocity"),
    "t": ("s", "si_t", "time"),
    "F": ("N", "si_F", "force"),
    "p": ("Pa", "si_p", "pressure"),
    "rho": ("kg/m**3", None, "density"),
    "nu": ("m**2/s", None, "kinematic viscosity"),
    "f": ("N/m**3", None, "force density"),
}


def _units():
    """``astropy.units``, or an ImportError that says how to get it."""
    try:
        from astropy import units as u
    except Exception as exc:                                     # ImportError, or a broken install of its own deps
        raise ImportError(ASTROPY_INSTALL_HINT) from exc
    return u


def available() -> tuple[bool, str]:
    """Whether astropy can be imported, and the reason when it cannot (the sentence a caller can print)."""
    try:
        import astropy.units  # noqa: F401
    except Exception as exc:
        return False, f"{ASTROPY_INSTALL_HINT} ({type(exc).__name__}: {exc})"
    return True, ""


class UnitMismatch(UnitError):
    """A quantity whose unit cannot become the unit of the field, refused instead of converted.

    Inherits :class:`pinneapple_data.physical_units.UnitError`, so it renders the same way (the fact, then a ``Fix:``
    line) and is still a ``ValueError`` for callers that catch that.
    """


# ---------------------------------------------------------------------------------------- reading the declared unit
def _is_dataset(sample: Any) -> bool:
    import xarray as xr

    return isinstance(getattr(sample, "state", None), xr.Dataset)


def declared_unit(sample: Any, name: str) -> str:
    """The unit a sample declares for ``name``, as text, or ``""`` when it declares none.

    Three places are consulted, in the order the house writes them: the xarray variable's ``units`` (what a dataset
    read from a netCDF file carries, and what the yt reader sets), then ``schema["variables"][name]["unit"]`` (the UPD
    field description), then ``schema["units"][name]`` (the mapping the FITS and yt readers fill). A field found in none
    of them has no declared unit, and saying so is the point of the function: a missing unit is not "dimensionless".
    """
    state = getattr(sample, "state", None)
    if _is_dataset(sample) and name in state.variables:
        for attr in ("units", "unit"):
            value = state[name].attrs.get(attr)
            if value:
                return str(value)
    schema = getattr(sample, "schema", None) or {}
    variables = schema.get("variables") or {}
    entry = variables.get(name) if isinstance(variables, dict) else None
    if isinstance(entry, dict):
        for key in ("unit", "units"):
            if entry.get(key):
                return str(entry[key])
    units = schema.get("units") or {}
    if isinstance(units, dict) and units.get(name):
        return str(units[name])
    return ""


#: How this library's own unit spellings map onto astropy's. ``physical_units`` writes ``°C`` / ``degC`` (what a CSV
#: header carries) and names a dimensionless column ``1``, ``-`` or ``ustrain``; astropy spells those differently and
#: would refuse the text. The mapping is one way and only for names that mean exactly the same thing -- a unit that is
#: genuinely ambiguous (``C`` for coulomb or for Celsius) is still refused rather than guessed at.
_HOUSE_ALIASES = {
    "°C": "deg_C", "degC": "deg_C", "degc": "deg_C", "celsius": "deg_C", "Celsius": "deg_C",
    "°F": "deg_F", "degF": "deg_F", "degf": "deg_F", "fahrenheit": "deg_F",
    "°R": "deg_R", "degR": "deg_R", "rankine": "deg_R",
    "K": "K", "kelvin": "K",
    "1": "", "-": "", "fraction": "", "ustrain": "", "µε": "", "%": "percent",
}


def _unit(text: str, field: str):
    """``astropy.units.Unit`` for a declared unit string, refused with the field's name when astropy cannot parse it.

    Two parsers are tried, both astropy's: the general one, then the FITS one (``format="fits"``), because a header
    writes ``erg / (cm2 s Angstrom)`` in the FITS spelling. A 1995-era header writes ``ERGS/CM**2/S/ANGSTRM``, which
    neither accepts -- astropy's own FITS unit table has no ``ANGSTRM`` -- and that is refused rather than guessed at:
    the text stays in the sample, and :func:`quantity_of` takes ``unit=`` when the caller knows what the file meant.
    """
    u = _units()
    spelled = _HOUSE_ALIASES.get(str(text).strip(), str(text).strip())
    if not spelled:
        return u.dimensionless_unscaled
    for parse in (lambda: u.Unit(spelled), lambda: u.Unit(spelled, format="fits")):
        try:
            return parse()
        except Exception:                                        # the next parser gets a chance; both may refuse
            continue
    raise UnitMismatch(
        f"{text!r} (the unit of field {field!r}) is not a unit astropy can parse, in its general or its FITS spelling",
        fix=(f'read the values with the unit the file meant: quantity_of(sample, "{field}", unit="..."); or declare it '
             'the way astropy spells it ("K", "km/s", "erg / (cm2 s Angstrom)")'))


def _values(sample: Any, name: str):
    """The values behind ``name``: a data variable or a coordinate of the dataset, or a key of a dict state."""
    state = getattr(sample, "state", None)
    if _is_dataset(sample):
        if name in state.variables:
            return state[name].values
        raise KeyError(f"this sample has no field or coordinate {name!r} (it has: {list(state.variables)[:20]})")
    if isinstance(state, dict):
        if name in state:
            return state[name]
        raise KeyError(f"this sample has no field {name!r} (it has: {list(state)[:20]})")
    raise TypeError(f"this sample's state is a {type(state).__name__}; expected an xarray.Dataset or a dict")


def quantity_of(sample: Any, name: str, *, unit: str | None = None):
    """The field ``name`` of ``sample`` as an ``astropy.units.Quantity``, in the unit it declares.

    ``unit`` reads the values in the unit given, converting when the field declares a compatible one and refusing with
    :class:`UnitMismatch` when the two are a different dimension. It is also the escape hatch for a declared unit
    astropy cannot parse -- an archival ``ERGS/CM**2/S/ANGSTRM`` -- where the caller reads the header and says what
    the numbers are; the text in the sample is left exactly as it was.

    The values are not modified: the Quantity wraps the same array. Raises :class:`UnitMismatch` when the field
    declares no unit and none is given, because a quantity invented out of a bare array is exactly the silent mistake
    this module exists to stop.
    """
    u = _units()
    declared = declared_unit(sample, name)
    values = _values(sample, name)
    if not declared and not unit:
        raise UnitMismatch(f"field {name!r} declares no unit, so it cannot become a Quantity",
                           fix=(f'write it down first: schema["units"]["{name}"] = "K"; or read it in a unit you know '
                                f'with quantity_of(sample, "{name}", unit="K")'))
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number):
        raise UnitMismatch(f"field {name!r} holds {array.dtype} values, which carry no magnitude to convert",
                           fix="ask for a numeric field: a table's string column has no unit")
    if not declared:
        return u.Quantity(array, _unit(unit, name))
    try:
        source_unit = _unit(declared, name)
    except UnitMismatch:
        if not unit:
            raise
        return u.Quantity(array, _unit(unit, name))                    # the caller's reading of an unparseable text
    if unit:
        target_unit = _unit(unit, name)
        return u.Quantity(_convert(array, source_unit, target_unit, name), target_unit)
    return u.Quantity(array, source_unit)


def sample_quantities(sample: Any, *, names: list[str] | None = None) -> dict[str, Any]:
    """Every field of the sample that declares a unit, as ``{name: Quantity}``.

    The convenience the round trip starts from. A field with no declared unit is simply absent -- it is not guessed at,
    and it is not silently given a unit it never had.
    """
    state = getattr(sample, "state", None)
    if names is None:
        if _is_dataset(sample):
            names = list(state.variables)
        else:
            schema = getattr(sample, "schema", None) or {}
            names = list(schema.get("units") or {}) or list(state or {})
    return {name: quantity_of(sample, name) for name in names if declared_unit(sample, name)}


# ---------------------------------------------------------------------------------------- writing a quantity back
def _linear(source: Any, target: Any, field: str) -> tuple[float, float]:
    """``(scale, offset)`` of ``source -> target``, refused by :class:`UnitMismatch` when the dimensions differ.

    Every conversion astropy performs is affine, so two evaluations determine it: what one unit becomes and what zero
    becomes. The offset is what a temperature scale adds (``deg_C -> K`` is ``+273.15``); reading both off astropy with
    its own temperature equivalencies keeps every constant the standard's, never a number written here.
    """
    u = _units()
    equiv = u.temperature()                       # deg_C and deg_F are the same dimension only through an offset
    if not source.is_equivalent(target, equivalencies=equiv):
        raise UnitMismatch(
            f"field {field!r}: {source} ({source.physical_type}) and {target} ({target.physical_type}) are different "
            "dimensions, so there is no conversion -- nothing was converted and nothing was written",
            fix=f"use {source}-compatible units for this field (or convert the value where the mistake is)")
    one = float((1.0 * source).to_value(target, equivalencies=equiv))
    zero = float((0.0 * source).to_value(target, equivalencies=equiv))
    return one - zero, zero


def _convert(values: Any, source: Any, target: Any, field: str, *, difference: bool = False):
    """``values`` (in ``source``) as floats in ``target``; the offset is dropped for a difference."""
    scale, offset = _linear(source, target, field)
    return np.asarray(values) * scale + (0.0 if difference else offset)


def set_quantity(sample: Any, name: str, quantity: Any, *, dims: Any | None = None,
                 difference: bool = False) -> dict[str, Any]:
    """Write ``quantity`` into field ``name``, converting it to the unit the field declares.

    The four cases, all of them stated rather than silent:

    1. the field declares the same unit -- the values are written as they are;
    2. the field declares a compatible unit -- the values are converted (``300 K`` into ``degC`` becomes ``26.85``) and
       the step is appended to ``provenance["unit_conversions"]`` with both units, the scale and the offset;
    3. the field declares an incompatible unit -- :class:`UnitMismatch` is raised and **nothing is written**;
    4. the field declares no unit -- the quantity's unit is adopted and written to ``schema["units"]`` and to the
       variable's ``units`` attribute, so the sample now says what it is.

    ``dims`` names the dimensions when ``name`` is a new field of a dataset state (an existing variable keeps its own).
    ``difference=True`` treats the quantity as a temperature difference: 5 K -> 5 degC, not -268.15 degC.

    Returns the record of what happened -- ``{"field", "unit", "action", "from_unit", "scale", "offset"}`` -- so a
    caller (a UI, a log, a test) sees it without reading the sample back.
    """
    u = _units()
    if not isinstance(quantity, u.Quantity):
        raise UnitMismatch(f"field {name!r} would be filled with a {type(quantity).__name__}, a bare number whose unit "
                           "only the caller knows",
                           fix='attach the unit at the call: set_quantity(sample, "T", 300 * astropy.units.K)')
    declared = declared_unit(sample, name)
    scale, offset, action = 1.0, 0.0, "declared"
    if declared:
        target_unit = _unit(declared, name)
        scale, offset = _linear(quantity.unit, target_unit, name)          # refuses before anything is written
        action = "same" if quantity.unit == target_unit else "converted"
        values = (_convert(quantity.value, quantity.unit, target_unit, name, difference=difference)
                  if action == "converted" else quantity.value)
    else:
        target_unit, values = u.Unit(quantity.unit), quantity.value

    state = getattr(sample, "state", None)
    if _is_dataset(sample):
        array = np.asarray(values)
        if name in state.variables:
            shape = state[name].shape
            if array.shape != shape:
                try:                                             # a constant written into a field of known shape
                    array = np.broadcast_to(array, shape)
                except ValueError as exc:
                    raise ValueError(f"field {name!r} has shape {shape}; the quantity given has shape {array.shape}, "
                                     "which does not broadcast onto it -- nothing was written") from exc
        attrs = dict(state[name].attrs) if name in state.variables else {}
        attrs["units"] = str(target_unit)
        if name in state.variables:
            new_dims = state[name].dims
        else:
            new_dims = tuple(dims if dims is not None else tuple(f"dim{i}" for i in range(array.ndim)))
            known = tuple(state.sizes.get(d) for d in new_dims)
            if None not in known and array.shape != known:
                try:                                             # a constant written onto named dimensions
                    array = np.broadcast_to(array, known)
                except ValueError as exc:
                    raise ValueError(f"new field {name!r} was given dims {new_dims} of shape {known} but a value of "
                                     f"shape {array.shape}; nothing was written") from exc
        state[name] = (new_dims, array, attrs)
    elif isinstance(state, dict):
        state[name] = np.asarray(values)
    else:
        raise TypeError(f"this sample's state is a {type(state).__name__}; expected an xarray.Dataset or a dict")

    if not isinstance(sample.schema, dict):
        sample.schema = {}
    sample.schema.setdefault("units", {})[name] = str(target_unit)
    variables = sample.schema.setdefault("variables", {})
    entry = variables.get(name) if isinstance(variables.get(name), dict) else {}
    variables[name] = {**entry, "unit": str(target_unit)}

    record = {"field": name, "unit": str(target_unit), "action": action, "from_unit": str(quantity.unit),
              "scale": float(scale), "offset": 0.0 if difference else float(offset)}
    if action == "converted":
        if not isinstance(sample.provenance, dict):
            sample.provenance = {}
        sample.provenance.setdefault("unit_conversions", []).append(
            {**record, "difference": bool(difference), "by": "pinneapple_data.astro.units.set_quantity"})
    return record


# ---------------------------------------------------------------------------------------- the LBM lattice bridge
def _lattice_kind(kind: str) -> tuple[Any, str | None, str]:
    u = _units()
    if kind not in LATTICE_KINDS:
        raise UnitMismatch(f"unknown lattice quantity {kind!r}", fix=f"use one of {sorted(LATTICE_KINDS)}")
    si_unit, back, label = LATTICE_KINDS[kind]
    return u.Unit(si_unit), back, label


def to_lattice(units: Any, quantity: Any, kind: str = "x") -> float:
    """A Quantity in lattice units, through the LBM solver's own :class:`Units` helper.

    ``kind`` is one of :data:`LATTICE_KINDS` (``"x"`` length, ``"u"`` velocity, ``"t"`` time, ``"nu"`` kinematic
    viscosity, ``"rho"`` density, ``"F"`` force, ``"f"`` force density, ``"p"`` pressure). The quantity's dimension is
    checked against the SI unit that kind is defined in, and a mismatch is refused by name: ``to_lattice(units,
    300 * u.K, "x")`` does not become a length because a length was asked for.

    >>> from pinneapple_simulation.numerical_solvers.lbm_tunnel import Units
    >>> lu = Units.from_si(lbm_length=1.0, lbm_velocity=0.1, lbm_rho=1.0, si_length=1.0, si_velocity=1.0, si_rho=1.0)
    >>> to_lattice(lu, 0.5 * u.m, "x")
    0.5
    """
    u = _units()
    si_unit, _back, label = _lattice_kind(kind)
    if not isinstance(quantity, u.Quantity):
        raise UnitMismatch(f"a {type(quantity).__name__} is not a Quantity, so it cannot become a {label} in lattice "
                           "units",
                           fix="attach the unit first: 0.5 * astropy.units.m")
    if not quantity.unit.is_equivalent(si_unit):
        raise UnitMismatch(f"a {label} is measured in {si_unit} ({si_unit.physical_type}); the quantity is in "
                           f"{quantity.unit} ({quantity.unit.physical_type})",
                           fix=f"pass it in {si_unit}, or ask for the lattice kind that matches its dimension")
    method = getattr(units, kind, None)
    if method is None:
        raise UnitMismatch(f"this Units helper has no {kind}() conversion",
                           fix=f"pass an object that has {kind}(), or convert in SI yourself")
    return float(method(float(quantity.to_value(si_unit))))


def from_lattice(units: Any, value: float, kind: str = "x") -> Any:
    """A lattice number as a Quantity in SI, through the LBM solver's :class:`Units` helper.

    The inverse of :func:`to_lattice`, limited to the kinds the helper converts both ways (``x``, ``u``, ``t``, ``F``,
    ``p``): lattice density, kinematic viscosity and force density are defined by a formula rather than by the helper's
    three scales, and are refused with that reason instead of a made-up conversion.
    """
    si_unit, back, label = _lattice_kind(kind)
    if back is None:
        raise UnitMismatch(f"the Units helper converts a {label} from SI (its {kind}() method) but has no inverse, so "
                           f"a lattice {label} cannot be read back",
                           fix="keep the SI value alongside the lattice one, or extend Units with the inverse")
    method = getattr(units, back, None)
    if method is None:
        raise UnitMismatch(f"this Units helper has no {back}() conversion",
                           fix=f"pass an object that has {back}(), or read the SI value yourself")
    return float(method(float(value))) * si_unit

"""Check a surrogate's value at one point against the reference model, by running the model there.

    out = verify_against_model({"k": 0.02, "tau": 120.0}, 0.95, "pinneapple_physics.closed_form.ideal_reactors:"
                               "pfr_conversion_ktau", rtol=0.01, quantity="X")
    out["verifications"]       # [{name, status, value, limit, detail}], status pass | fail | inconclusive | not_run

The answer is ``pass`` only when the model ran, gave a finite value for ``quantity``, the point lies inside the box
the model is declared for, the units agree, and ``|surrogate - model| <= rtol * |model|``. A model that raises or
returns no value, a point outside the model's declared ranges (its ``x_ranges``) or a unit mismatch makes the answer
``inconclusive``, never ``pass``: the check did not happen.
"""
from __future__ import annotations

import importlib
import math
import sys
from collections.abc import Callable, Mapping
from typing import Any, TypedDict

__all__ = ["verify_against_model", "resolve_model", "model_id", "model_ranges", "output_unit", "VerifyResult", "X_OUTPUTS"]


# what each tool returns, with units (read by the Studio catalog)
X_OUTPUTS = {
    "verify_against_model": [
        {"name": "model_value", "type": "float", "unit": "same unit as the quantity",
         "description": "the reference model at the point (null when it did not run)"},
        {"name": "rel_error", "type": "float", "unit": "-", "description": "|surrogate - model| / |model|"},
        {"name": "status", "type": "str", "unit": "", "description": "pass, fail or inconclusive"},
        {"name": "verifications", "type": "list", "unit": "",
         "description": "records {name, status pass|fail|inconclusive|not_run, value, limit, detail}"}],
}


class VerifyResult(TypedDict):
    model_value: float | None
    rel_error: float | None
    status: str
    verifications: list[dict[str, Any]]


LIBRARY_PREFIX = "pinneapple"


def _in_library(module: str) -> bool:
    return (module or "").split(".")[0].startswith(LIBRARY_PREFIX)


def model_id(model: str | Callable[..., Any]) -> str:
    """The tool id of a model: the string itself, or ``"module:qualname"`` of a callable."""
    if callable(model):
        return f"{getattr(model, '__module__', '?')}:{getattr(model, '__qualname__', repr(model))}"
    return str(model)


def resolve_model(model: str | Callable[..., Any]) -> Callable[..., Any]:
    """A callable from a tool id ``"package.module:name"`` (or ``"package.module.name"``), or the callable itself.

    A tool id is resolved only inside the PINNeAPPle library: the module's top-level package must start with
    ``pinneapple`` (checked before anything is imported), no part of the path may start with ``_``, and the object
    found must itself be defined in the library (so a library module's import of ``os`` or ``subprocess`` cannot be
    reached through it). Anything else raises ``PermissionError`` without being called: a served tool id cannot run
    an arbitrary function such as ``shutil:rmtree`` or ``builtins:eval``. A callable passed directly (Python API) is
    used as given."""
    if callable(model):
        return model
    text = str(model)
    mod, _, name = text.partition(":")
    if not name:
        mod, _, name = mod.rpartition(".")
    parts = [*mod.split("."), *name.split(".")]
    if not _in_library(mod) or not name or any(not p or p.startswith("_") for p in parts):
        raise PermissionError(f"model {text!r} is not a public PINNeAPPle tool id ('pinneapple...module:function')")
    obj: Any = importlib.import_module(mod)
    for part in name.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f"{text} is not callable")
    if not _in_library(getattr(obj, "__module__", "") or ""):
        raise PermissionError(f"model {text!r} resolves to {getattr(obj, '__module__', '?')}."
                              f"{getattr(obj, '__qualname__', '?')}, which is not defined in PINNeAPPle")
    return obj


def model_ranges(fn: Callable[..., Any]) -> dict[str, tuple[float, float]]:
    """The input box a model is declared for: its ``x_ranges`` attribute, ``{input: (min, max)}``, ``None`` for an
    open side (empty if none). Write it as a literal after the function so the Studio can read it without importing:
    ``f.x_ranges = {"k": (0.0, 10.0), "tau": (0.0, None)}``."""
    def side(v, default):
        return default if v is None else float(v)
    return {k: (side(a, -math.inf), side(b, math.inf)) for k, (a, b) in (getattr(fn, "x_ranges", None) or {}).items()}


def output_unit(fn: Callable[..., Any], quantity: str) -> str | None:
    """The unit the model declares for ``quantity`` in its module's ``X_OUTPUTS`` (the Studio's outputs
    declaration; ``None`` when it declares none)."""
    mod = sys.modules.get(getattr(fn, "__module__", "") or "")
    declared = (getattr(mod, "X_OUTPUTS", None) or {}).get(getattr(fn, "__name__", ""))
    for o in declared or []:
        if o.get("name") == quantity:
            return o.get("unit")
    return None


def _value(result: Any, quantity: str) -> float | None:
    if isinstance(result, Mapping):
        result = result.get(quantity)
    elif hasattr(result, quantity):
        result = getattr(result, quantity)
    try:
        v = float(result)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _same_unit(a: str, b: str) -> bool:
    norm = lambda u: u.replace(" ", "").replace("1", "-") if u in ("1", "-", "") else u.replace(" ", "")  # noqa: E731
    return norm(a) == norm(b)


def verify_against_model(point: dict[str, float], surrogate_value: float | dict[str, Any], model: str, rtol: float,
                         quantity: str = "X") -> VerifyResult:
    """Run ``model`` at ``point`` and compare its ``quantity`` with the surrogate's value.

    Parameters
    ----------
    point : dict[str, float]
        the model's inputs at the point, in the model's units (e.g. D [m], L [m], k [1/s], Q [m3/s])
    surrogate_value : float or dict
        the surrogate's value of ``quantity`` there, a number or ``{"value": v, "unit": "..."}`` (the unit is checked
        against the model's declared one)
    model : str
        the executable tool id of the reference model, ``"package.module:function"``
    rtol : float
        relative tolerance of the agreement [-]; required, there is no default
    quantity : str
        the output compared (a key of the model's result)

    Returns
    -------
    model_value : float [same unit as the quantity]
        the model's value at the point (null when it did not run)
    rel_error : float [-]
        |surrogate - model| / |model| (null when there is nothing to compare)
    status : str
        pass, fail or inconclusive: the agreement verification's status
    verifications : list
        one record per check: model_runs, inside_model_ranges, units_agree, agreement
    """
    if not (isinstance(rtol, (int, float)) and math.isfinite(rtol) and rtol >= 0):
        raise ValueError("rtol must be a finite number >= 0")
    if isinstance(surrogate_value, Mapping):
        s_val, s_unit = float(surrogate_value["value"]), surrogate_value.get("unit")
    else:
        s_val, s_unit = float(surrogate_value), None
    checks: list[dict[str, Any]] = []

    def add(name, status, value=None, limit=None, detail=""):
        checks.append({"name": name, "status": status, "value": value, "limit": limit, "detail": detail})

    try:
        fn = resolve_model(model)
    except Exception as e:  # noqa: BLE001 - an unknown model is an inconclusive check, reported
        add("model_runs", "inconclusive", detail=f"model {model!r} could not be loaded: {type(e).__name__}: {e}")
        for n in ("inside_model_ranges", "units_agree", "agreement"):
            add(n, "not_run", detail="the model did not load")
        return {"model_value": None, "rel_error": None, "status": "inconclusive", "verifications": checks}

    ranges = model_ranges(fn)
    outside = [f"{k} = {point[k]:g} outside [{a:g}, {b:g}]" for k, (a, b) in ranges.items()
               if k in point and not a <= float(point[k]) <= b]
    if not ranges:
        add("inside_model_ranges", "not_run", detail="the model declares no input ranges (x_ranges)")
    elif outside:
        add("inside_model_ranges", "inconclusive", detail="; ".join(outside))
    else:
        add("inside_model_ranges", "pass", detail="every input inside the model's declared ranges")

    m_val, err = None, ""
    try:
        m_val = _value(fn(**{k: float(v) for k, v in point.items()}), quantity)
        if m_val is None:
            err = f"the model returned no finite value for {quantity!r}"
    except Exception as e:  # noqa: BLE001 - a failing model is an inconclusive check, reported
        err = f"{type(e).__name__}: {e}"
    add("model_runs", "pass" if m_val is not None else "inconclusive", m_val, detail=err or "the model ran at the point")

    m_unit = output_unit(fn, quantity)
    if s_unit is None or m_unit is None:
        add("units_agree", "not_run", detail="no unit given" + (" by the surrogate" if s_unit is None else
                                                                " by the model"))
        units_ok = True
    else:
        units_ok = _same_unit(s_unit, m_unit)
        add("units_agree", "pass" if units_ok else "inconclusive", detail=f"surrogate [{s_unit}], model [{m_unit}]")

    rel = None
    if m_val is None or outside or not units_ok:
        why = ("the model gave no value" if m_val is None else "the point is outside the model's ranges" if outside
               else "the units differ")
        if m_val is not None:
            rel = abs(s_val - m_val) / max(abs(m_val), 1e-300)
        add("agreement", "inconclusive", rel, rtol, detail=f"not decided: {why}")
        status = "inconclusive"
    else:
        rel = abs(s_val - m_val) / max(abs(m_val), 1e-300)
        status = "pass" if rel <= rtol else "fail"
        add("agreement", status, rel, rtol, detail=f"surrogate {s_val:.6g}, model {m_val:.6g}")
    return {"model_value": m_val, "rel_error": rel, "status": status, "verifications": checks}



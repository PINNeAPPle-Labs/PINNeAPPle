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
from collections.abc import Callable, Mapping
from typing import Any, TypedDict

__all__ = ["verify_against_model", "resolve_model", "model_ranges", "output_unit", "VerifyResult"]


class VerifyResult(TypedDict):
    model_value: float | None
    rel_error: float | None
    status: str
    verifications: list[dict[str, Any]]


def resolve_model(model: str | Callable[..., Any]) -> Callable[..., Any]:
    """A callable from a tool id ``"package.module:name"`` (or ``"package.module.name"``), or the callable itself."""
    if callable(model):
        return model
    mod, _, name = str(model).partition(":")
    if not name:
        mod, _, name = mod.rpartition(".")
    obj: Any = importlib.import_module(mod)
    for part in name.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f"{model} is not callable")
    return obj


def model_ranges(fn: Callable[..., Any]) -> dict[str, tuple[float, float]]:
    """The input box a model is declared for: its ``x_ranges`` attribute, ``{input: (min, max)}`` (empty if none)."""
    return {k: (float(a), float(b)) for k, (a, b) in (getattr(fn, "x_ranges", None) or {}).items()}


def output_unit(fn: Callable[..., Any], quantity: str) -> str | None:
    """The unit the model declares for ``quantity`` in its ``x_outputs`` (``None`` when it declares none)."""
    for o in getattr(fn, "x_outputs", None) or []:
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


verify_against_model.x_role = "verify"
verify_against_model.x_outputs = [
    {"name": "model_value", "type": "float", "unit": "same as the quantity", "description": "the model at the point"},
    {"name": "rel_error", "type": "float", "unit": "-", "description": "|surrogate - model| / |model|"},
    {"name": "status", "type": "str", "unit": "", "description": "pass, fail or inconclusive"},
    {"name": "verifications", "type": "list", "unit": "",
     "description": "records {name, status pass|fail|inconclusive|not_run, value, limit, detail}"}]

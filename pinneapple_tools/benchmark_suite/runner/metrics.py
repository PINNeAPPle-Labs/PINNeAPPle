from __future__ import annotations

from typing import Any


def ensure_float_dict(d: dict[str, Any]) -> dict[str, float]:
    """
    Convert dictionary values to floats when possible.

    Parameters
    ----------
    d : Dict[str, Any]
        Input dictionary with arbitrary value types.

    Returns
    -------
    Dict[str, float]
        New dictionary containing only the keys whose values
        could be successfully converted to float. Keys are
        coerced to strings. Non-numeric values are silently skipped.
    """
    out: dict[str, float] = {}
    for k, v in d.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            # skip non-numeric
            continue
    return out
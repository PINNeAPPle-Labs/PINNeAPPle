"""Shared helpers of the built-in failure modes."""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

from ..chain import Validity, Verification, tr
from ..library import LIBRARY, Infeasible, self_check  # noqa: F401  (re-exported for the modes)

DESIGN_MARGIN = 0.0     # countermeasures size to margin >= this; a case may set "design_margin" (premise)


def verdict(*, demand: float, capacity: float, unit: str, scale: float | None = None, demand_label, capacity_label,
            mechanism, method: str, quantities: dict[str, list] | None = None, assumptions: Iterable = (),
            sources: Iterable[str] = (), validity: Iterable[Validity] = (), self_checks: Iterable = (),
            series: dict[str, Any] | None = None, passed: bool | None = None) -> Verification:
    """A verification that passes when ``demand <= capacity``; margin = (capacity - demand) / scale."""
    sc = abs(scale) if scale else (abs(capacity) if capacity and math.isfinite(capacity) else 1.0)
    if math.isfinite(demand) and math.isfinite(capacity):
        margin = (capacity - demand) / sc
    else:
        margin = -math.inf if demand == math.inf or capacity == -math.inf else math.inf
    ok = (demand <= capacity) if passed is None else passed
    return Verification(mode="", requirement="", passed=bool(ok), demand=float(demand), capacity=float(capacity),
                        unit=unit, margin=float(margin), demand_label=demand_label, capacity_label=capacity_label,
                        mechanism=mechanism, method=method, quantities=dict(quantities or {}),
                        assumptions=[a if isinstance(a, dict) else tr(a, a) for a in assumptions],
                        sources=list(sources), validity=list(validity), self_checks=list(self_checks),
                        series=dict(series or {}))


def q(value: float, unit: str, label: str) -> list:
    return [float(value), unit, label]


def target_margin(p: dict[str, Any]) -> float:
    return float(p.get("design_margin", DESIGN_MARGIN))


def size_to_pass(mode: str, p: dict[str, Any], values: Iterable[Any], make: Callable[[Any], dict[str, Any]],
                 limits: dict[str, float] | None = None, add: tuple[str, ...] = ()) -> tuple[dict[str, Any], Any]:
    """The first candidate (in order) whose changes make ``mode`` pass with the case's design margin.

    ``make(value) -> changes``; ``add``: elements the trial concept has. Raises :class:`Infeasible` if none does.
    """
    m = LIBRARY.mode(mode)
    limits = p.get("_limits") if limits is None else limits
    need = target_margin(p)
    last = None
    for val in values:
        ch = make(val)
        trial = {**p, **ch, "_elements": tuple(p.get("_elements", ())) + tuple(add)}
        v = m.evaluate(trial, limits=limits)
        last = v
        if v.passed and v.margin >= need:
            return ch, val
    raise Infeasible(
        f"nenhum valor na faixa testada faz '{mode}' passar (melhor margem {last.margin:.3g})" if last else
        f"sem candidatos para '{mode}'",
        f"no value in the tested range makes '{mode}' pass (best margin {last.margin:.3g})" if last else
        f"no candidates for '{mode}'")


def fmt(x: float, nd: int = 3) -> str:
    if not math.isfinite(x):
        return "∞" if x > 0 else "-∞"
    return f"{x:.{nd}g}"

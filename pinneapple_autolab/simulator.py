"""The reference simulator the lab calls: a solver run on one design, its checks before it runs, its cost after.

Any physics plugs in by implementing :class:`Simulator`:

* ``check(params) -> list[str]``: why this design cannot be simulated as asked (invalid geometry, boundary
  conditions that do not match the mission); empty when it can. The lab never runs a design with issues.
* ``run(params, workdir) -> SimOutcome``: the outputs the solver produced, whether it succeeded, its wall time and,
  optionally, fields on a fixed set of points (for field surrogates). A failed run returns ``ok=False`` with the
  reason; it never returns made-up outputs.

:class:`FunctionSimulator` wraps a Python function (tests, analytic toy problems).
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["SimOutcome", "Simulator", "FunctionSimulator"]


@dataclass
class SimOutcome:
    outputs: dict[str, float]
    ok: bool = True
    message: str = ""
    cost_s: float = 0.0
    fields: dict[str, np.ndarray] | None = None       # name -> (n_points,) on the simulator's fixed points
    meta: dict[str, Any] = field(default_factory=dict)


class Simulator:
    """Base class: set ``name``, ``fidelity``, ``outputs`` (and ``field_names`` / ``field_points`` / ``regions``
    if it returns fields), implement :meth:`check` and :meth:`run`."""
    name: str = "simulator"
    fidelity: str = "reference"
    outputs: Sequence[str] = ()
    field_names: Sequence[str] = ()
    regions: Mapping[str, np.ndarray] = {}     # region name -> boolean mask (or index array) over the field points

    def check(self, params: Mapping[str, float]) -> list[str]:
        return []

    def run(self, params: Mapping[str, float], workdir: str) -> SimOutcome:
        raise NotImplementedError

    def field_points(self, params: Mapping[str, float]) -> np.ndarray | None:
        """Coordinates of the field points for a design (for plots); None if the simulator returns no fields."""
        return None

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "fidelity": self.fidelity, "outputs": list(self.outputs),
                "fields": list(self.field_names)}


class FunctionSimulator(Simulator):
    """``fn(params) -> {output: value}`` (optionally also ``"__fields__": {name: array}``); ``checker(params) ->
    list[str]``; ``fail(params) -> str | None`` makes a run fail with that message (to test failure paths)."""

    def __init__(self, fn: Callable[[Mapping[str, float]], Mapping[str, Any]], outputs: Sequence[str], *,
                 name: str = "function", checker: Callable[[Mapping[str, float]], list[str]] | None = None,
                 fail: Callable[[Mapping[str, float]], str | None] | None = None, field_names: Sequence[str] = (),
                 regions: Mapping[str, np.ndarray] | None = None, points: np.ndarray | None = None):
        self.fn, self.outputs, self.name = fn, list(outputs), name
        self.checker, self.fail = checker, fail
        self.field_names, self.regions, self._points = list(field_names), dict(regions or {}), points

    def check(self, params):
        return list(self.checker(params)) if self.checker else []

    def field_points(self, params):
        return self._points

    def run(self, params, workdir):
        t0 = time.perf_counter()
        if self.fail is not None:
            msg = self.fail(params)
            if msg:
                return SimOutcome({}, False, msg, time.perf_counter() - t0)
        out = dict(self.fn(params))
        flds = out.pop("__fields__", None)
        vals = {k: float(out[k]) for k in self.outputs}
        bad = [k for k, v in vals.items() if not np.isfinite(v)]
        if bad:
            return SimOutcome({}, False, f"non-finite outputs {bad}", time.perf_counter() - t0)
        return SimOutcome(vals, True, "", time.perf_counter() - t0,
                          {k: np.asarray(v, float) for k, v in flds.items()} if flds else None)

"""The physics query engine: ask for one quantity, at one physical place, under given conditions, to a stated
accuracy, and get the value with its uncertainty, validity, provenance and the method that produced it.

    engine.query("T", "die@0.31,0.72", conditions={"power": P}, accuracy=0.05)

Two kinds of answer are kept apart (:attr:`Answer.kind`):

* ``"read"``: the conditions were already solved; the value is read from that state (interpolation error only).
* ``"predicted"``: nothing was solved for these conditions; the value is computed from other states or a model
  (superposition of solved states, a bound, a condensed or learned model) or by a new solve (``"solved"``).

A :class:`Planner` asks every registered :class:`Method` for an *offer* (can it answer, at what expected cost, with
what error bound, inside its validity domain) and takes the cheapest offer whose bound meets the accuracy target; if
none does, the last-resort method (a full solve) answers. Every answer records the offers it chose from.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .refs import PhysicalRef, parse_ref


@dataclass
class Query:
    quantity: str
    where: PhysicalRef
    time: float | None = None
    conditions: dict[str, Any] = field(default_factory=dict)
    accuracy: float | None = None             # absolute, in the quantity's unit; None: the engine's default
    rigorous: bool = False                    # only methods whose bound is guaranteed (an interval)

    @property
    def key(self) -> str:
        return condition_key(self.conditions)


@dataclass
class Offer:
    method: str
    cost_s: float                             # expected wall time
    bound: float                              # expected absolute error (or rigorous bound if ``rigorous``)
    valid: bool = True
    rigorous: bool = False
    why: str = ""


@dataclass
class Answer:
    value: float
    unit: str
    uncertainty: float
    kind: str                                 # "read" | "predicted" | "solved"
    method: str
    validity: dict[str, Any]
    provenance: dict[str, Any]
    seconds: float
    offers: list[dict] = field(default_factory=list)
    rigorous: bool = False
    interval: tuple[float, float] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def condition_key(conditions: dict[str, Any]) -> str:
    """A stable hash of the conditions (arrays by their bytes)."""
    h = hashlib.sha256()
    for k in sorted(conditions):
        v = conditions[k]
        h.update(k.encode())
        if isinstance(v, np.ndarray):
            h.update(np.ascontiguousarray(v, float).tobytes())
        else:
            h.update(json.dumps(v, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


class Method:
    """A way to answer queries. ``offer`` must be cheap (it decides, it does not compute the answer)."""
    name = "method"

    def offer(self, q: Query) -> Offer | None:  # pragma: no cover - interface
        raise NotImplementedError

    def answer(self, q: Query) -> Answer:  # pragma: no cover - interface
        raise NotImplementedError


class Planner:
    """Cheapest valid offer that meets the accuracy target. When none meets it (or no target is given), the valid
    offer with the smallest error bound (then the cheapest); ``fallback`` only when no method offers at all."""

    def __init__(self, methods: list[Method], fallback: Method | None = None):
        self.methods, self.fallback = methods, fallback

    def choose(self, q: Query) -> tuple[Method, list[Offer]]:
        offers = []
        for m in self.methods:
            o = m.offer(q)
            if o is not None:
                offers.append(o)
        valid = [o for o in offers if o.valid and (o.rigorous or not q.rigorous)]
        ok = [o for o in valid if q.accuracy is not None and o.bound <= q.accuracy]
        if ok:
            ok.sort(key=lambda o: (o.cost_s, o.bound))
        else:                                                # best available: smallest bound, then cost
            ok = sorted(valid, key=lambda o: (o.bound, o.cost_s))
        if ok:
            name = ok[0].method
            return next(m for m in self.methods if m.name == name), offers
        if self.fallback is None:
            raise LookupError(f"no method can answer {q.quantity} at {q.where}")
        return self.fallback, offers


class Engine:
    """``query(quantity, where, time=, conditions=, accuracy=, rigorous=)`` through a :class:`Planner`; ``log`` keeps
    every answer (method, kind, latency) for the audit. ``default_accuracy``: the target used when a query gives
    none (per quantity; a quantity without one gets the most accurate available answer)."""

    def __init__(self, planner: Planner, units: dict[str, str] | None = None,
                 default_accuracy: dict[str, float] | None = None):
        self.planner = planner
        self.units = units or {}
        self.default_accuracy = default_accuracy or {}
        self.log: list[dict] = []
        self.hooks: list[Callable[[Query, Answer], None]] = []

    def query(self, quantity: str, where, time: float | None = None, conditions: dict | None = None,
              accuracy: float | None = None, rigorous: bool = False) -> Answer:
        t0 = _now()
        if accuracy is None:
            accuracy = self.default_accuracy.get(quantity)
        q = Query(quantity, parse_ref(where), time, dict(conditions or {}), accuracy, rigorous)
        method, offers = self.planner.choose(q)
        ans = method.answer(q)
        ans.seconds = _now() - t0
        ans.offers = [asdict(o) for o in offers]
        for h in self.hooks:
            h(q, ans)
        self.log.append({"quantity": quantity, "where": str(q.where), "method": ans.method, "kind": ans.kind,
                         "seconds": ans.seconds, "uncertainty": ans.uncertainty})
        return ans


def _now() -> float:
    return time.perf_counter()

"""The library of failure modes and countermeasures, by physical domain.

A :class:`FailureMode` is a quantitative check: given the parameters of a concept it computes the demand and the
capacity that decide whether the mode occurs, every intermediate number, the mechanism in words built from those
numbers, the applicability range of the models it used and its own internal checks. A :class:`Countermeasure` is a
change to the concept (a new element, a resized parameter) that acts on the mechanism of one or more modes; it
declares its expected physical effect and computes its cost from the change it made.

New domains plug in with :func:`register_mode` / :func:`register_countermeasure` (or the ``@mode`` /
``@countermeasure`` decorators); the built-in ones live in :mod:`pinneapple_reason.modes`.

Costs are ``quantity x unit cost``: the quantity (extra area, steel mass, fan power, ...) is computed from the change;
the unit cost is a relative index declared by the countermeasure as a premise (adjustable per case through the
concept parameter ``cost.<countermeasure>``). The engine only compares costs of options for the same finding.
"""
from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from pinneapple_analysis.trust import Check

from .chain import Concept, Validity, Verification, text_of, tr

__all__ = ["FailureMode", "Countermeasure", "Infeasible", "Library", "LIBRARY", "register_mode",
           "register_countermeasure", "param", "Need", "self_check"]


class Infeasible(Exception):
    """A countermeasure cannot be applied to this concept (the message says why, in both languages)."""

    def __init__(self, pt: str, en: str):
        super().__init__(en)
        self.text = tr(pt, en)


@dataclass(frozen=True)
class Need:
    """A parameter a failure mode reads: SI unit and meaning."""
    name: str
    unit: str
    label: str


def param(p: dict[str, Any], name: str, default: Any = None) -> Any:
    if name in p:
        return p[name]
    if default is None:
        raise KeyError(f"parameter {name!r} is required")
    return default


def self_check(name: str, passed: bool, value: float | None = None, limit: float | None = None, source: str = "",
               reason: str = "") -> Check:
    """An internal check of a computation (residual, closure, agreement with an independent solution)."""
    v = None if value is None or not math.isfinite(value) else float(value)
    return Check(name, "supports" if passed else "contradicts", critical=False, value=v, limit=limit, source=source,
                 reason=reason)


@dataclass
class FailureMode:
    """A failure mode and its quantitative check.

    ``evaluate(p, limits) -> Verification``: ``p`` are the concept's parameters plus its elements under
    ``p["_elements"]``; ``limits`` the requirement's overrides. ``known_answers`` are cases with a published or
    hand-checked answer (used by the tests and by the trust card's model stage). ``uncertain`` names the premises
    whose spread the uncertainty check propagates: ``{param: relative half-width}``.
    """
    name: str
    domain: str
    title: dict[str, str]
    summary: dict[str, str]
    evaluate_fn: Callable[[dict[str, Any], dict[str, float]], Verification]
    needs: tuple[Need, ...] = ()
    sources: tuple[str, ...] = ()
    countermeasures: tuple[str, ...] = ()
    uncertain: dict[str, float] = field(default_factory=dict)
    known_answers: Callable[[], list[dict[str, Any]]] | None = None

    def evaluate(self, concept: Concept | dict[str, Any], requirement: str = "", limits: dict[str, float] | None = None
                 ) -> Verification:
        p = _params(concept)
        missing = [n.name for n in self.needs if n.name not in p]
        if missing:
            raise KeyError(f"failure mode {self.name!r} needs parameters {missing}")
        v = self.evaluate_fn(p, dict(limits or {}))
        v.mode, v.requirement = self.name, requirement or v.requirement
        return v

    def describe(self, lang: str = "pt") -> dict[str, Any]:
        return {"name": self.name, "domain": self.domain, "title": text_of(self.title, lang),
                "summary": text_of(self.summary, lang), "needs": [n.__dict__ for n in self.needs],
                "sources": list(self.sources), "countermeasures": list(self.countermeasures),
                "uncertain": dict(self.uncertain)}


def _params(concept: Concept | dict[str, Any]) -> dict[str, Any]:
    if isinstance(concept, Concept):
        return {**concept.params, "_elements": tuple(concept.elements)}
    p = dict(concept)
    p.setdefault("_elements", ())
    return p


@dataclass
class Countermeasure:
    """A change that acts on the mechanism of a failure mode.

    ``apply(p, v) -> (changes, add, remove)``: ``p`` the concept's parameters (with ``_elements``), ``v`` the failing
    verification (a countermeasure may size itself from it: the slope that makes the lap hold, the closure time
    that keeps the surge below the rating). Raises :class:`Infeasible` when it cannot work here.
    ``cost(p_before, p_after) -> (quantity, unit, what)``: the physical quantity the change costs; the cost index is
    that quantity times ``unit_cost`` (a relative premise).
    """
    name: str
    element: str
    title: dict[str, str]
    effect: dict[str, str]
    addresses: tuple[str, ...]
    apply_fn: Callable[[dict[str, Any], Verification], tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]]
    cost_fn: Callable[[dict[str, Any], dict[str, Any]], tuple[float, str, str]]
    unit_cost: float = 1.0
    unit_cost_note: str = "relative index (premise)"
    requires: tuple[str, ...] = ()          # elements the concept must already have
    excludes: tuple[str, ...] = ()          # elements that make it pointless
    repeatable: bool = False

    def applicable(self, concept: Concept) -> tuple[bool, dict[str, str]]:
        miss = [e for e in self.requires if not concept.has(e)]
        if miss:
            return False, tr(f"precisa antes de: {', '.join(miss)}", f"needs first: {', '.join(miss)}")
        bad = [e for e in self.excludes if concept.has(e)]
        if bad:
            return False, tr(f"incompatível com: {', '.join(bad)}", f"incompatible with: {', '.join(bad)}")
        if not self.repeatable and concept.has(self.element):
            return False, tr(f"o conceito já tem '{self.element}'", f"the concept already has '{self.element}'")
        return True, {}

    def apply(self, concept: Concept, v: Verification, limits: dict[str, float] | None = None) -> Concept:
        p = _params(concept)
        p["_limits"] = dict(limits or {})
        changes, add, remove = self.apply_fn(p, v)
        changes = {k: val for k, val in changes.items() if not k.startswith("_")}
        add = tuple(add) + ((self.element,) if self.element and self.element not in add else ())
        return concept.with_changes(changes, add=add, remove=tuple(remove), source=f"countermeasure:{self.name}")

    def cost(self, before: Concept, after: Concept) -> dict[str, Any]:
        qty, unit, what = self.cost_fn(_params(before), _params(after))
        uc = float(before.params.get(f"cost.{self.name}", self.unit_cost))
        return {"quantity": float(qty), "unit": unit, "what": what, "unit_cost": uc,
                "index": float(qty) * uc, "note": self.unit_cost_note}

    def describe(self, lang: str = "pt") -> dict[str, Any]:
        return {"name": self.name, "element": self.element, "title": text_of(self.title, lang),
                "effect": text_of(self.effect, lang), "addresses": list(self.addresses),
                "requires": list(self.requires), "excludes": list(self.excludes), "unit_cost": self.unit_cost,
                "unit_cost_note": self.unit_cost_note}


class Library:
    def __init__(self):
        self.modes: dict[str, FailureMode] = {}
        self.countermeasures: dict[str, Countermeasure] = {}

    def add_mode(self, m: FailureMode, replace: bool = False) -> FailureMode:
        if m.name in self.modes and not replace:
            raise ValueError(f"failure mode {m.name!r} is already registered")
        self.modes[m.name] = m
        return m

    def add_countermeasure(self, c: Countermeasure, replace: bool = False) -> Countermeasure:
        if c.name in self.countermeasures and not replace:
            raise ValueError(f"countermeasure {c.name!r} is already registered")
        self.countermeasures[c.name] = c
        return c

    def mode(self, name: str) -> FailureMode:
        _load()
        if name not in self.modes:
            raise KeyError(f"unknown failure mode {name!r}; known: {sorted(self.modes)}")
        return self.modes[name]

    def countermeasure(self, name: str) -> Countermeasure:
        _load()
        if name not in self.countermeasures:
            raise KeyError(f"unknown countermeasure {name!r}; known: {sorted(self.countermeasures)}")
        return self.countermeasures[name]

    def for_mode(self, mode: str) -> list[Countermeasure]:
        """Countermeasures that act on ``mode`` (declared by the mode or by the countermeasure)."""
        _load()
        m = self.mode(mode)
        names = list(m.countermeasures) + [c.name for c in self.countermeasures.values()
                                           if mode in c.addresses and c.name not in m.countermeasures]
        return [self.countermeasures[n] for n in names if n in self.countermeasures]

    def domains(self) -> dict[str, list[str]]:
        _load()
        out: dict[str, list[str]] = {}
        for m in self.modes.values():
            out.setdefault(m.domain, []).append(m.name)
        return {k: sorted(v) for k, v in sorted(out.items())}

    def describe(self, lang: str = "pt") -> dict[str, Any]:
        _load()
        return {"modes": [m.describe(lang) for m in self.modes.values()],
                "countermeasures": [c.describe(lang) for c in self.countermeasures.values()],
                "domains": self.domains()}


LIBRARY = Library()
_LOADED = False


def _load() -> None:
    global _LOADED
    if not _LOADED:
        _LOADED = True
        importlib.import_module("pinneapple_reason.modes")       # registers the built-in modes


def register_mode(m: FailureMode, replace: bool = False) -> FailureMode:
    return LIBRARY.add_mode(m, replace)


def register_countermeasure(c: Countermeasure, replace: bool = False) -> Countermeasure:
    return LIBRARY.add_countermeasure(c, replace)


def validity(name: str, value: float, lo: float | None = None, hi: float | None = None, note: str = "") -> Validity:
    return Validity(name, float(value), lo, hi, note)


def iter_known_answers(names: Iterable[str] | None = None):
    """``(mode, case)`` for every known-answer case of the library (or of ``names``)."""
    _load()
    for n in names or sorted(LIBRARY.modes):
        m = LIBRARY.modes[n]
        if m.known_answers is not None:
            for case in m.known_answers():
                yield m, case

"""The chain of whys: requirement -> naive concept -> physical check -> failure found -> countermeasure -> check ...

Every object here is plain data (``to_dict`` / ``from_dict`` round-trip through JSON) so a chain can be saved, sent to
an agent, drawn in the Flow Studio and re-read later. Nothing in this module computes physics: the numbers in a
:class:`Verification` come from a failure mode of :mod:`pinneapple_reason.library`, and the engine
(:mod:`pinneapple_reason.engine`) is the only thing that builds a :class:`Chain`.

Text meant for people is stored in both languages as ``{"pt": ..., "en": ...}`` (see :func:`tr`).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pinneapple_analysis.trust import Check

__all__ = ["Requirement", "Concept", "Verification", "Validity", "Finding", "Option", "Step", "Chain", "tr",
           "text_of", "RESOLVED", "UNRESOLVED", "MAX_STEPS"]

RESOLVED, UNRESOLVED, MAX_STEPS = "resolved", "unresolved", "max_steps"


def tr(pt: str, en: str) -> dict[str, str]:
    """A text in Portuguese and English."""
    return {"pt": pt, "en": en}


def text_of(t: Any, lang: str = "pt") -> str:
    """The text in ``lang`` (falls back to the other language, or to ``str(t)``)."""
    if isinstance(t, dict):
        return t.get(lang) or t.get("en") or t.get("pt") or ""
    return "" if t is None else str(t)


def _num(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return "inf" if x > 0 else "-inf" if x < 0 else "nan"
    return x


def _unnum(x: Any) -> Any:
    return float(x) if x in ("inf", "-inf", "nan") else x


@dataclass
class Requirement:
    """What the design must do (``function``) and the failure modes that would stop it (``modes``).

    ``limits`` overrides a mode's acceptance limit for this requirement (e.g. ``{"t_junction_max": 398.15}``), with
    the reason in ``source``.
    """
    id: str
    text: dict[str, str]
    modes: list[str]
    limits: dict[str, float] = field(default_factory=dict)
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Requirement:
        text = d["text"] if isinstance(d["text"], dict) else tr(d["text"], d["text"])
        return cls(d["id"], text, list(d["modes"]), dict(d.get("limits", {})), d.get("source", ""))


@dataclass
class Concept:
    """One version of the design: its parameters (SI units), the elements it has and where each value came from.

    ``sources[param]`` says whether a value is a premise of the case, a cited datasheet / standard value, or was set
    by a countermeasure (and which one).
    """
    name: str
    params: dict[str, Any]
    elements: list[str] = field(default_factory=list)
    description: dict[str, str] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    version: int = 0

    def with_changes(self, changes: dict[str, Any], *, add: tuple[str, ...] = (), remove: tuple[str, ...] = (),
                     source: str = "") -> Concept:
        params = {**self.params, **changes}
        elements = [e for e in self.elements if e not in remove] + [e for e in add if e not in self.elements]
        sources = {**self.sources, **{k: source for k in changes}} if source else dict(self.sources)
        return Concept(self.name, params, elements, dict(self.description), sources, self.version + 1)

    def has(self, element: str) -> bool:
        return element in self.elements

    def key(self) -> str:
        """Identity of the physical state (parameters and elements), for caching and cycle detection."""
        return json.dumps({"p": {k: _num(v) for k, v in sorted(self.params.items())},
                           "e": sorted(self.elements)}, sort_keys=True, default=str)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["params"] = {k: _num(v) for k, v in self.params.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Concept:
        return cls(d["name"], {k: _unnum(v) for k, v in d["params"].items()}, list(d.get("elements", [])),
                   dict(d.get("description", {})), dict(d.get("sources", {})), int(d.get("version", 0)))


@dataclass
class Validity:
    """One applicability condition of the model a check used: ``lo <= value <= hi`` (either bound may be None)."""
    name: str
    value: float
    lo: float | None = None
    hi: float | None = None
    note: str = ""

    @property
    def ok(self) -> bool:
        return ((self.lo is None or self.value >= self.lo) and (self.hi is None or self.value <= self.hi)
                and math.isfinite(self.value))


@dataclass
class Verification:
    """The result of one failure mode checked on one concept.

    ``demand`` and ``capacity`` are the two numbers that decide it (the mode passes when ``demand <= capacity``),
    in ``unit``; ``margin`` is the mode's normalised margin (>= 0 passes). ``quantities`` holds every number the
    check measured, ``mechanism`` explains the physics with those numbers, ``method`` says how they were computed and
    ``sources`` where the models and constants come from. ``self_checks`` are the internal checks of the computation
    (energy balance closure, agreement of a solver with the closed form, ...).
    """
    mode: str
    requirement: str
    passed: bool
    demand: float
    capacity: float
    unit: str
    margin: float
    demand_label: dict[str, str]
    capacity_label: dict[str, str]
    mechanism: dict[str, str]
    method: str
    quantities: dict[str, list] = field(default_factory=dict)          # name -> [value, unit, label]
    assumptions: list[dict[str, str]] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    validity: list[Validity] = field(default_factory=list)
    self_checks: list[Check] = field(default_factory=list)
    series: dict[str, Any] = field(default_factory=dict)              # curves for figures (lists of floats)

    @property
    def in_range(self) -> bool:
        return all(v.ok for v in self.validity)

    def check(self) -> Check:
        """The trust-layer check of this verification (critical: a failure mode that occurs fails the design)."""
        return Check(f"{self.requirement}/{self.mode}", "supports" if self.passed else "contradicts", critical=True,
                     value=_finite(self.demand), limit=_finite(self.capacity), source=self.method,
                     reason="" if self.passed else text_of(self.mechanism, "en")[:300])

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("demand", "capacity", "margin"):
            d[k] = _num(d[k])
        d["quantities"] = {k: [_num(v[0]), *v[1:]] for k, v in self.quantities.items()}
        d["validity"] = [{**asdict(v), "value": _num(v.value), "ok": v.ok} for v in self.validity]
        d["self_checks"] = [c.__dict__ for c in self.self_checks]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Verification:
        d = dict(d)
        d["validity"] = [Validity(**{k: _unnum(v) if k == "value" else v for k, v in x.items() if k != "ok"})
                         for x in d.get("validity", [])]
        d["self_checks"] = [Check(**c) for c in d.get("self_checks", [])]
        for k in ("demand", "capacity", "margin"):
            d[k] = _unnum(d[k])
        d["quantities"] = {k: [_unnum(v[0]), *v[1:]] for k, v in d.get("quantities", {}).items()}
        return cls(**d)


def _finite(x: float) -> float | None:
    return float(x) if isinstance(x, (int, float)) and math.isfinite(x) else None


@dataclass
class Finding:
    """The first failure mode that occurs: which requirement it breaks and the number that proves it."""
    requirement: str
    mode: str
    title: dict[str, str]
    verification: Verification

    @property
    def proof(self) -> str:
        v = self.verification
        return (f"{text_of(v.demand_label, 'en')} = {_fmt(v.demand)} {v.unit} > "
                f"{text_of(v.capacity_label, 'en')} = {_fmt(v.capacity)} {v.unit}")

    def to_dict(self) -> dict[str, Any]:
        return {"requirement": self.requirement, "mode": self.mode, "title": self.title,
                "verification": self.verification.to_dict(), "proof": self.proof}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Finding:
        return cls(d["requirement"], d["mode"], d["title"], Verification.from_dict(d["verification"]))


def _fmt(x: float) -> str:
    if not isinstance(x, (int, float)) or not math.isfinite(x):
        return str(x)
    a = abs(x)
    return f"{x:.3g}" if a and (a < 1e-2 or a >= 1e4) else f"{x:.3f}".rstrip("0").rstrip(".") if a < 10 else f"{x:.1f}"


@dataclass
class Option:
    """A countermeasure the engine tried for a finding: what it changed, what it costs and what it did to every
    requirement (margins before / after), or why it could not be applied."""
    countermeasure: str
    element: str
    title: dict[str, str]
    feasible: bool
    reason: dict[str, str] = field(default_factory=dict)
    changes: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)        # {"value", "unit", "quantity", "index", "note"}
    effect: dict[str, str] = field(default_factory=dict)      # expected physical effect (why it should work)
    target_margin_before: float | None = None
    target_margin_after: float | None = None
    fixes_target: bool = False
    passing_after: int = 0
    regressions: list[str] = field(default_factory=list)
    proposed_by: str = "library"                               # "library" or "llm"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["changes"] = {k: _num(v) for k, v in self.changes.items()}
        for k in ("target_margin_before", "target_margin_after"):
            d[k] = None if d[k] is None else _num(d[k])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Option:
        d = dict(d)
        for k in ("target_margin_before", "target_margin_after"):
            d[k] = None if d.get(k) is None else _unnum(d[k])
        d["changes"] = {k: _unnum(v) for k, v in d.get("changes", {}).items()}
        return cls(**d)


@dataclass
class Step:
    """Version ``version`` of the design, every check run on it, the first failure (if any), the countermeasures
    tried for it and the one chosen (``chosen``: index into ``options``)."""
    version: int
    concept: Concept
    verifications: list[Verification]
    finding: Finding | None = None
    options: list[Option] = field(default_factory=list)
    chosen: int | None = None
    why: dict[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(v.passed for v in self.verifications)

    @property
    def countermeasure(self) -> Option | None:
        return None if self.chosen is None else self.options[self.chosen]

    def verification(self, mode: str) -> Verification | None:
        return next((v for v in self.verifications if v.mode == mode), None)

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "concept": self.concept.to_dict(),
                "verifications": [v.to_dict() for v in self.verifications],
                "finding": None if self.finding is None else self.finding.to_dict(),
                "options": [o.to_dict() for o in self.options], "chosen": self.chosen, "why": self.why,
                "passed": self.passed}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Step:
        return cls(d["version"], Concept.from_dict(d["concept"]),
                   [Verification.from_dict(v) for v in d["verifications"]],
                   None if d.get("finding") is None else Finding.from_dict(d["finding"]),
                   [Option.from_dict(o) for o in d.get("options", [])], d.get("chosen"), d.get("why", {}))


@dataclass
class Chain:
    """The whole reasoning: requirements, every version of the design and why each one was made."""
    title: dict[str, str]
    requirements: list[Requirement]
    steps: list[Step]
    status: str
    notes: list[dict[str, str]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def final(self) -> Step:
        return self.steps[-1]

    @property
    def resolved(self) -> bool:
        return self.status == RESOLVED

    def path(self) -> list[tuple[int, str, str]]:
        """``[(version, failure mode found, countermeasure chosen)]`` for each step that changed the design."""
        return [(s.version, s.finding.mode, s.countermeasure.countermeasure) for s in self.steps
                if s.finding is not None and s.countermeasure is not None]

    def explain(self, lang: str = "pt") -> str:
        """The chain as plain text: version 0 -> version N, and why."""
        from .narrative import explain
        return explain(self, lang)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "pinneapple.reason.chain/1", "title": self.title, "status": self.status,
                "requirements": [r.to_dict() for r in self.requirements], "steps": [s.to_dict() for s in self.steps],
                "notes": self.notes, "meta": self.meta, "path": [list(p) for p in self.path()]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chain:
        return cls(d["title"], [Requirement.from_dict(r) for r in d["requirements"]],
                   [Step.from_dict(s) for s in d["steps"]], d["status"], list(d.get("notes", [])),
                   dict(d.get("meta", {})))

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=1, default=str))
        return p

    @classmethod
    def load(cls, path: str | Path) -> Chain:
        return cls.from_dict(json.loads(Path(path).read_text()))

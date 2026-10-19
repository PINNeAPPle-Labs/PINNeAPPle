"""Typed requirement objects: what the user asked for, with units, bounds and the provenance of every value.

A :class:`Quantity` is one number (always stored in SI) that remembers where it came from:

``stated``     the user wrote it (``quote`` is the text it was read from);
``answered``   the user gave it in reply to a follow-up question;
``defaulted``  the template filled it in (``source`` says from where: a standard, a datasheet, a scale preset);
``missing``    nobody gave it and it has no default: a critical one becomes a :class:`Question`.

A :class:`Slot` is a value a problem template can use (a load, a power, a temperature limit, ...), declared with its
SI unit, its admissible range and the words that point to it in a sentence. A :class:`RequirementSpec` is the filled
form: the template it was matched to, the values of its slots, the objectives and the open questions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

from pinneapple_data.physical_units import UnitError, dim_str, parse_unit

STATED, ANSWERED, DEFAULTED, MISSING, DERIVED = "stated", "answered", "defaulted", "missing", "derived"
PROVENANCES = (STATED, ANSWERED, DEFAULTED, MISSING, DERIVED)


class SpecError(ValueError):
    """A value that cannot be accepted: unknown unit, wrong dimension, or outside the template's range."""


def to_si(value: float, unit: str) -> float:
    """``value`` in ``unit`` converted to SI (absolute temperatures keep their offset)."""
    return float(parse_unit(unit).to_si(float(value)))


def from_si(value: float, unit: str) -> float:
    return float(parse_unit(unit).from_si(float(value)))


def same_dimension(a: str, b: str) -> bool:
    try:
        return parse_unit(a).dim == parse_unit(b).dim
    except UnitError:
        return False


@dataclass(frozen=True)
class Quantity:
    name: str
    value: Any                      # SI number, or a string for a choice (material, process)
    unit: str = ""                  # SI unit of ``value`` ("" for choices and pure numbers)
    provenance: str = MISSING
    source: str = ""                # citation / reason for a default, or "user" for stated values
    quote: str = ""                 # the words it was read from (stated values)

    def __post_init__(self):
        if self.provenance not in PROVENANCES:
            raise SpecError(f"unknown provenance {self.provenance!r}")

    @property
    def known(self) -> bool:
        return self.provenance != MISSING and self.value is not None

    def display(self, unit: str | None = None, digits: int = 4) -> str:
        if not self.known:
            return "?"
        if isinstance(self.value, str):
            return self.value
        if unit and unit != self.unit:
            return f"{from_si(self.value, unit):.{digits}g} {unit}"
        return f"{self.value:.{digits}g} {self.unit}".strip()

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value, "unit": self.unit, "provenance": self.provenance,
                "source": self.source, "quote": self.quote}


@dataclass(frozen=True)
class Slot:
    """A value a template needs or can use.

    ``unit`` is the SI unit the value is stored in (``""`` for a dimensionless number, ``"choice"`` for a name picked
    from ``choices``). ``display_unit`` is how reports show it. ``keywords`` are the words near a number that point to
    this slot ("radius", "span", "load"). ``cues`` map phrases to values ("still air" -> 0 m/s). ``critical`` slots
    without a value and without a default stop the pipeline with a question. ``bounds`` is the template's range of
    validity in SI (a value outside it is refused, not clipped)."""
    name: str
    label: str
    unit: str = ""
    display_unit: str = ""
    keywords: tuple[str, ...] = ()
    cues: tuple[tuple[str, Any], ...] = ()
    critical: bool = False
    default: Any = None
    default_source: str = ""
    bounds: tuple[float, float] | None = None
    question: str = ""
    choices: tuple[str, ...] = ()
    role: str = "environment"       # environment | load | limit | scale | choice | budget

    @property
    def is_choice(self) -> bool:
        return self.unit == "choice"

    def shown_unit(self) -> str:
        return self.display_unit or self.unit

    def accepts(self, unit: str) -> bool:
        if self.is_choice:
            return False
        if not self.unit:
            return unit in ("", "1", "-", "%", "x", "times")
        return same_dimension(unit, self.unit)

    def check(self, value_si: Any) -> None:
        if self.is_choice:
            if self.choices and value_si not in self.choices:
                raise SpecError(f"{self.label}: {value_si!r} is not one of {list(self.choices)}")
            return
        v = float(value_si)
        if not math.isfinite(v):
            raise SpecError(f"{self.label}: not a finite number")
        if self.bounds is not None:
            lo, hi = self.bounds
            if not lo <= v <= hi:
                shown = self.shown_unit()
                f = (lambda x: from_si(x, shown)) if shown and shown != self.unit else (lambda x: x)
                raise SpecError(f"{self.label} = {f(v):.4g} {shown} is outside the range this template covers "
                                f"({f(lo):.4g} to {f(hi):.4g} {shown})")

    def quantity(self, value: Any, unit: str, provenance: str, *, source: str = "", quote: str = "") -> Quantity:
        """Convert ``value`` given in ``unit`` to this slot's SI unit, check it, and wrap it."""
        if self.is_choice:
            self.check(value)
            return Quantity(self.name, value, "", provenance, source, quote)
        if self.unit:
            try:
                u = parse_unit(unit) if unit else None
            except UnitError as e:
                raise SpecError(f"{self.label}: {e}") from None
            if u is None:
                raise SpecError(f"{self.label}: a unit is needed (e.g. {self.shown_unit()})")
            if u.dim != parse_unit(self.unit).dim:
                raise SpecError(f"{self.label}: '{unit}' is {dim_str(u.dim)}, expected {dim_str(parse_unit(self.unit).dim)}"
                                f" (e.g. {self.shown_unit()})")
            v = float(u.to_si(float(value)))
            v = float(parse_unit(self.unit).from_si(v))
        else:
            v = float(value) / (100.0 if unit == "%" else 1.0)
        self.check(v)
        return Quantity(self.name, v, self.unit, provenance, source, quote)

    def default_quantity(self) -> Quantity:
        if self.default is None:
            return Quantity(self.name, None, self.unit if not self.is_choice else "", MISSING)
        return Quantity(self.name, self.default, "" if self.is_choice else self.unit, DEFAULTED,
                        self.default_source)


@dataclass(frozen=True)
class Question:
    slot: str
    text: str
    unit: str = ""
    choices: tuple[str, ...] = ()
    why: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"slot": self.slot, "question": self.text, "unit": self.unit, "choices": list(self.choices),
                "why": self.why}


@dataclass(frozen=True)
class ObjectiveChoice:
    """An objective a template can optimise: ``name`` is an output of its models, ``sense`` min / max."""
    name: str
    sense: str
    label: str
    keywords: tuple[str, ...] = ()


@dataclass
class RequirementSpec:
    text: str
    template: str
    values: dict[str, Quantity] = field(default_factory=dict)
    objectives: list[ObjectiveChoice] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)          # rejected values (unit, range) with the reason
    unsupported: list[str] = field(default_factory=list)     # parts of the request the template cannot address
    parser: str = "template"                                 # "template" | "llm+template"
    match_score: float = 0.0

    # ------------------------------------------------------------------ access
    def __getitem__(self, name: str) -> Any:
        return self.values[name].value

    def get(self, name: str, default: Any = None) -> Any:
        q = self.values.get(name)
        return q.value if q is not None and q.known else default

    @property
    def complete(self) -> bool:
        return not self.questions and not self.issues

    def by_provenance(self, prov: str) -> list[Quantity]:
        return [q for q in self.values.values() if q.provenance == prov]

    def answer(self, template_slots: dict[str, Slot], **answers) -> RequirementSpec:
        """A new spec with the given answers (``name=(value, unit)`` or ``name=value`` in the slot's SI unit)."""
        vals = dict(self.values)
        issues = [i for i in self.issues if not any(i.startswith(template_slots[k].label) for k in answers
                                                    if k in template_slots)]
        for k, a in answers.items():
            if k not in template_slots:
                raise SpecError(f"unknown value {k!r}; this template takes {sorted(template_slots)}")
            sl = template_slots[k]
            v, u = a if isinstance(a, tuple) else (a, "" if sl.is_choice else sl.unit)
            vals[k] = sl.quantity(v, u, ANSWERED, source="user (follow-up)")
        qs = [q for q in self.questions if q.slot not in answers]
        return replace(self, values=vals, questions=qs, issues=issues)

    def table(self, slots: dict[str, Slot] | None = None) -> list[dict[str, Any]]:
        rows = []
        for q in self.values.values():
            sl = (slots or {}).get(q.name)
            shown = sl.shown_unit() if sl and not sl.is_choice else None
            prov = "not given (optional)" if q.provenance == MISSING and sl is not None and not sl.critical \
                else q.provenance
            rows.append({"name": q.name, "label": sl.label if sl else q.name,
                         "value": q.display(shown), "provenance": prov,
                         "source": q.quote or q.source})
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "template": self.template, "parser": self.parser,
                "match_score": self.match_score,
                "values": {k: v.to_dict() for k, v in self.values.items()},
                "objectives": [{"name": o.name, "sense": o.sense, "label": o.label} for o in self.objectives],
                "questions": [q.to_dict() for q in self.questions], "issues": list(self.issues),
                "unsupported": list(self.unsupported)}

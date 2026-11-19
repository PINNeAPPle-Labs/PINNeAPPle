"""The envelope an analysis returns: a status, a severity, one sentence, the numbers behind it and the source.

This is the shape ``pinneapple_core.analysis`` puts around every calculation, so that a result is read the same
way wherever it came from (a closed-form model, a solver bridge, an app, the Studio):

* **status** — ``ok``, ``warning``, ``bad``, ``not_applicable``, ``insufficient_data`` or ``error``. A call that
  cannot answer says so; it never raises at the caller and it never returns a confident number it cannot stand
  behind.
* **severity** — how much the finding matters (``none`` to ``critical``), which is what a person acts on; a
  calculation that only reports a number leaves it ``none``.
* **the sentence** — one line in the language in force, with the numbers in it, saying what was found. This is
  the text a report prints; the numbers alone are for the next tool.
* **evidence** — the intermediate values the sentence rests on, so a reader can check the claim.
* **source** — the maturity of the model and its references, carried on the result so a finding is never
  separated from where it came from.

    >>> from pinneapple_core.analysis import InputSpec, Severity, Status
    >>> InputSpec(unit="m", min=0.0).check("L", -2.0)
    ['length (L) must be at least 0 m, got -2']
"""
from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from pinneapple_core.jsonable import to_jsonable

from .text import fmt_number, say, text_of, tr

__all__ = ["Status", "Severity", "Maturity", "Reference", "InputSpec", "OutputSpec", "Finding", "Result", "tr"]


class Status(str, Enum):
    """How much of an answer the analysis could actually give."""

    OK = "ok"
    WARNING = "warning"                      # the calculation ran and the result is worth a look
    BAD = "bad"                              # the calculation ran and the result is outside what is acceptable
    NOT_APPLICABLE = "not_applicable"        # the model does not cover this input, and says why
    INSUFFICIENT_DATA = "insufficient_data"  # something the answer depends on is missing or unusable
    ERROR = "error"                          # the analysis itself failed; the message says where

    def __str__(self) -> str:
        return self.value


class Severity(str, Enum):
    """How much a finding matters to the person who reads it."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def __str__(self) -> str:
        return self.value


class Maturity(str, Enum):
    """What stands behind the model: where a number earns the right to be quoted."""

    LITERATURE_BASED = "literature_based"      # a published model, cited in ``references``
    ENGINEERING_HEURISTIC = "engineering_heuristic"  # a rule of practice, valid in a stated range
    EXPERIMENTAL = "experimental"              # new code: runs, but has no reference comparison yet

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Reference:
    """A source a model comes from. ``year`` and ``venue`` are optional; nothing here is invented."""

    title: str
    year: int | None = None
    authors: Sequence[str] = ()
    venue: str = ""
    doi: str = ""
    url: str = ""

    def __str__(self) -> str:
        head = f"{', '.join(self.authors)}. " if self.authors else ""
        tail = f" ({self.year})" if self.year else ""
        return f"{head}{self.title}{tail}"

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "year": self.year, "authors": list(self.authors), "venue": self.venue,
                "doi": self.doi, "url": self.url}


#: messages an input check can produce, in both languages; ``{who}`` is the input's label and name
MESSAGES = {
    "missing": tr("falta {who}", "{who} is missing"),
    "not_number": tr("{who} precisa ser um número, veio {got}", "{who} must be a number, got {got}"),
    "not_integer": tr("{who} precisa ser inteiro, veio {value}", "{who} must be an integer, got {value}"),
    "not_finite": tr("{who} precisa ser finito, veio {value}", "{who} must be finite, got {value}"),
    "not_text": tr("{who} precisa ser texto, veio {got}", "{who} must be text, got {got}"),
    "empty": tr("{who} está vazio", "{who} is empty"),
    "array_not_numbers": tr("{who} tem valores que não são números", "{who} holds values that are not numbers"),
    "array_not_finite": tr("{who} tem {count} valores não finitos (NaN ou infinito)",
                           "{who} holds {count} non-finite values (NaN or infinity)"),
    "below": tr("{who} precisa ser pelo menos {limit}, veio {value}", "{who} must be at least {limit}, got {value}"),
    "above": tr("{who} precisa ser no máximo {limit}, veio {value}", "{who} must be at most {limit}, got {value}"),
    "above_exclusive": tr("{who} precisa ser maior que {limit}, veio {value}",
                          "{who} must be greater than {limit}, got {value}"),
    "below_exclusive": tr("{who} precisa ser menor que {limit}, veio {value}",
                          "{who} must be less than {limit}, got {value}"),
    "not_in_choices": tr("{who} precisa ser um de {choices}, veio {value}",
                         "{who} must be one of {choices}, got {value}"),
}


def _say(template: str, lang: str | None = None, **values: Any) -> str:
    return say(template, lang, **values)


@dataclass(frozen=True)
class InputSpec:
    """What one input of an analysis takes: its name in a sentence, its unit and the range the model holds in.

    The range is the validity domain, not a preference: an analysis refuses a value outside it with a reason a
    person can act on (:meth:`check`), because a number computed outside the model's range is a wrong answer
    wearing the right shape.
    """

    unit: str = ""
    kind: str = "number"                   # number | integer | array | text | any
    min: float | None = None               # inclusive
    max: float | None = None               # inclusive
    gt: float | None = None                # exclusive
    lt: float | None = None                # exclusive
    choices: Sequence[Any] | None = None
    required: bool = True
    label: Any = ""                        # a tr() text, or a plain string
    doc: Any = ""                          # a tr() text, or a plain string

    def __post_init__(self) -> None:
        if self.kind not in {"number", "integer", "array", "text", "any"}:
            raise ValueError(f"unknown input kind {self.kind!r}; use number, integer, array, text or any")

    def label_of(self, name: str = "", lang: str | None = None) -> str:
        """The input as a sentence names it: its label with the parameter name, or the name alone without one."""
        text = text_of(self.label, lang)
        if text and name:
            return f"{text} ({name})"
        return text or name or "value"

    def check(self, name: str, value: Any, lang: str | None = None) -> list[str]:
        """Every reason ``value`` cannot be used for this input — empty when it can."""
        who = self.label_of(name, lang)
        if value is None:
            return [] if not self.required else [_say(MESSAGES["missing"], lang, who=who)]
        if self.kind in {"number", "integer"}:
            return self._check_number(who, value, lang)
        if self.kind == "array":
            return self._check_array(who, value, lang)
        if self.kind == "text" and not isinstance(value, str):
            return [_say(MESSAGES["not_text"], lang, who=who, got=type(value).__name__)]
        return []

    def _check_number(self, who: str, value: Any, lang: str | None) -> list[str]:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            return [_say(MESSAGES["not_number"], lang, who=who, got=type(value).__name__)]
        number = float(value)
        if not math.isfinite(number):
            return [_say(MESSAGES["not_finite"], lang, who=who, value=fmt_number(number, 3, lang))]
        if self.kind == "integer" and number != int(number):
            return [_say(MESSAGES["not_integer"], lang, who=who, value=fmt_number(number, 3, lang))]
        unit = f" {self.unit}" if self.unit else ""
        shown = fmt_number(number, 4, lang)
        if self.gt is not None and not number > float(self.gt):
            return [_say(MESSAGES["above_exclusive"], lang, who=who, value=shown,
                         limit=f"{fmt_number(self.gt, 4, lang)}{unit}")]
        if self.lt is not None and not number < float(self.lt):
            return [_say(MESSAGES["below_exclusive"], lang, who=who, value=shown,
                         limit=f"{fmt_number(self.lt, 4, lang)}{unit}")]
        if self.min is not None and number < float(self.min):
            return [_say(MESSAGES["below"], lang, who=who, value=shown,
                         limit=f"{fmt_number(self.min, 4, lang)}{unit}")]
        if self.max is not None and number > float(self.max):
            return [_say(MESSAGES["above"], lang, who=who, value=shown,
                         limit=f"{fmt_number(self.max, 4, lang)}{unit}")]
        if self.choices is not None and value not in list(self.choices):
            return [_say(MESSAGES["not_in_choices"], lang, who=who, value=shown,
                         choices=", ".join(str(c) for c in self.choices))]
        return []

    def _check_array(self, who: str, value: Any, lang: str | None) -> list[str]:
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return [_say(MESSAGES["array_not_numbers"], lang, who=who)]
        if array.size == 0:
            return [_say(MESSAGES["empty"], lang, who=who)]
        bad = int(np.count_nonzero(~np.isfinite(array)))
        if bad:
            return [_say(MESSAGES["array_not_finite"], lang, who=who, count=fmt_number(bad, 0, lang))]
        return []

    def to_dict(self, lang: str | None = None) -> dict[str, Any]:
        shown: dict[str, Any] = {"unit": self.unit, "kind": self.kind, "required": self.required,
                                 "label": text_of(self.label, lang)}
        for key in ("min", "max", "gt", "lt"):
            if getattr(self, key) is not None:
                shown[key] = getattr(self, key)
        if self.choices is not None:
            shown["choices"] = list(self.choices)
        if self.doc:
            shown["doc"] = text_of(self.doc, lang)
        return shown


@dataclass(frozen=True)
class OutputSpec:
    """What one output of an analysis is: its unit and what it means, so a result card needs no other source."""

    unit: str = ""
    label: Any = ""
    doc: Any = ""

    def to_dict(self, lang: str | None = None) -> dict[str, Any]:
        shown = {"unit": self.unit, "label": text_of(self.label, lang)}
        if self.doc:
            shown["doc"] = text_of(self.doc, lang)
        return shown


@dataclass
class Finding:
    """What an analysis returns when it has something to say beyond the numbers.

    A calculation returns a mapping of metrics; an analysis that judges the numbers (a limit, a comparison, a
    regime the model does not hold in) returns a ``Finding`` whose ``status``, ``severity`` and ``message`` the
    registry puts into the :class:`Result`.
    """

    metrics: dict[str, Any] = field(default_factory=dict)
    status: Status = Status.OK
    severity: Severity = Severity.NONE
    message: Any = ""                        # a tr() text, worded once, when ``describe`` is not enough
    evidence: dict[str, Any] = field(default_factory=dict)
    limitations: Sequence[Any] = ()


@dataclass
class Result:
    """One call of one analysis: never an exception, always a status and a sentence."""

    id: str
    status: Status = Status.OK
    severity: Severity = Severity.NONE
    sentence: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)
    problems: Sequence[str] = ()             # why the inputs could not be used, when they could not
    evidence: dict[str, Any] = field(default_factory=dict)
    references: Sequence[Reference] = ()
    limitations: Sequence[str] = ()
    maturity: Maturity = Maturity.EXPERIMENTAL
    language: str = "en"

    @property
    def ok(self) -> bool:
        """Whether the analysis produced an answer (``ok`` or ``warning``/``bad``: a verdict is still an answer)."""
        return self.status in {Status.OK, Status.WARNING, Status.BAD}

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable({
            "id": self.id,
            "status": self.status.value,
            "severity": self.severity.value,
            "sentence": self.sentence,
            "metrics": self.metrics,
            "inputs": self.inputs,
            "problems": list(self.problems),
            "evidence": self.evidence,
            "references": [r.to_dict() for r in self.references],
            "limitations": list(self.limitations),
            "maturity": self.maturity.value,
            "language": self.language,
        })

    def to_text(self) -> str:
        """A short block a report can print: the sentence, then the numbers it was built from."""
        lines = [f"{self.id}: {self.status.value}" + (f" ({self.severity.value})" if self.severity != Severity.NONE else "")]
        if self.sentence:
            lines.append(self.sentence)
        for reason in self.problems:
            lines.append(f"  - {reason}")
        for name, value in self.metrics.items():
            lines.append(f"  {name} = {fmt_number(value, 6, self.language) if isinstance(value, numbers.Real) else value}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.sentence or f"{self.id}: {self.status.value}"


def as_status(value: Any, default: Status = Status.OK) -> Status:
    """A ``Status`` from an enum, its value or its name; ``default`` when it is not one."""
    if isinstance(value, Status):
        return value
    try:
        return Status(str(value).strip().lower())
    except ValueError:
        return default


def as_severity(value: Any, default: Severity = Severity.NONE) -> Severity:
    """A ``Severity`` from an enum, its value or its name; ``default`` when it is not one."""
    if isinstance(value, Severity):
        return value
    try:
        return Severity(str(value).strip().lower())
    except ValueError:
        return default


def as_mapping(value: Any) -> dict[str, Any] | None:
    """``value`` as a plain dict when it is one (a ``Finding`` counts as one), else None."""
    if isinstance(value, Finding):
        return dict(value.metrics)
    if isinstance(value, Mapping):
        return dict(value)
    return None

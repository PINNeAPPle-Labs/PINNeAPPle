"""An engineer's correction, stored as a rule that every later design is checked against (#173).

An engineer looks at a generated design, says "the trailing edge is too thin, it will not hold a fastener", and the
next run generates the same design again: the correction lived in a chat, and a chat is not a check. Here a
correction becomes a **record** -- the case it applies to, the condition, the reviewer and the date -- and every
later design is checked against it, with the outcome in the same vocabulary as the rest of the trust report
(``supports`` / ``contradicts`` / ``not_run``), naming the rule that produced it.

The condition is an expression over the design's own numbers, the same grammar the dataset audit checks
constraints with (:func:`pinneapple_data.audit.constraint_mask`), so it is data and not code: it can be stored, read
back, diffed and reviewed.

>>> from pinneapple_analysis.trust.corrections import Correction, CorrectionStore, correction_checks
>>> rule = Correction(rule_id="R-001", condition="thickness_mm >= 2.0", case="bracket*",
...                   statement="a 1.5 mm web cracked in test 42; nothing under 2 mm", reviewer="A. Ribeiro",
...                   date="2026-11-20", source="design D-17")            # doctest: +SKIP
>>> store = CorrectionStore("/tmp/corrections.jsonl")                     # doctest: +SKIP
>>> store.save(rule)                                                      # doctest: +SKIP
>>> [c.status for c in correction_checks(store.rules(), {"thickness_mm": 1.5}, case="bracket-A")]   # doctest: +SKIP
['contradicts']

A rule whose condition cannot be checked on a given design is ``not_run`` with the reason -- never ``supports``: the
design did not pass the review, the review could not be asked.
"""
from __future__ import annotations

import fnmatch
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import date as _date
from pathlib import Path
from typing import Any

from pinneapple_data.review import digest_of

from .trust_report import CONTRADICTS, NOT_RUN, SUPPORTS, Check, TrustReport

__all__ = ["Correction", "CorrectionError", "CorrectionStore", "correction_checks", "with_corrections"]

SEVERITIES = ("critical", "major")


class CorrectionError(ValueError):
    """The rule cannot be recorded or read back: a missing reviewer, a date that is not a date, a bad expression."""


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class Correction:
    """One correction, as a rule.

    ``condition`` is what the design has to satisfy (``"thickness_mm >= 2"``, ``"cd <= 0.012 * cd_reference"``);
    ``case`` is what it applies to (``"*"`` for every design, else a glob over the case name); ``source`` is the
    design or test the correction came from, so the reason survives with the rule. A ``critical`` rule that a design
    contradicts rejects the trust report on its own (rule 1 of :class:`~pinneapple_analysis.trust.TrustReport`).
    """

    rule_id: str
    condition: str
    statement: str
    reviewer: str
    date: str
    case: str = "*"
    severity: str = "critical"
    source: str = ""
    units: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.rule_id).strip():
            raise CorrectionError("a correction needs a rule id: a rule nobody can name cannot be cited in a report")
        if not str(self.reviewer).strip():
            raise CorrectionError("a correction needs the reviewer who made it")
        if not str(self.statement).strip():
            raise CorrectionError("a correction needs the sentence that was corrected: the condition alone does not "
                                  "say why")
        try:
            _date.fromisoformat(str(self.date))
        except ValueError as exc:
            raise CorrectionError(f"date must be ISO 8601 (YYYY-MM-DD), got {self.date!r}") from exc
        if self.severity not in SEVERITIES:
            raise CorrectionError(f"severity must be one of {SEVERITIES}, got {self.severity!r}")
        if not str(self.case).strip():
            raise CorrectionError("case must name what the rule applies to ('*' for every design)")
        try:
            _ = self.names                                     # parse the condition now: a typo fails at recording
        except Exception as exc:  # noqa: BLE001 -- any parse failure is a rule that could never be checked
            raise CorrectionError(f"condition {self.condition!r} is not a checkable expression: {exc}") from exc

    # -- the rule as a checkable thing --------------------------------------------------------------------
    @property
    def names(self) -> set[str]:
        """The columns of a design the condition reads (function names excluded)."""
        from pinneapple_data.expressions import names_in
        return names_in(self.condition)

    def applies_to(self, case: str) -> bool:
        """``"*"`` applies to every design; otherwise a glob over the case name (``"airfoil*"``)."""
        return self.case == "*" or fnmatch.fnmatch(str(case), self.case)

    def holds(self, columns: Mapping[str, Any]) -> bool:
        """Whether the design satisfies the condition. Raises ``KeyError`` when a column it reads is absent."""
        from pinneapple_data.audit import constraint_mask
        vals = {name: [columns[name]] for name in self.names}
        return bool(constraint_mask(self.condition, vals)[0])

    def check(self, columns: Mapping[str, Any]) -> Check:
        """The rule as a :class:`Check`: ``supports`` when the design satisfies it, ``contradicts`` when it does not,
        ``not_run`` when the design does not carry the number the rule reads."""
        name = f"correction {self.rule_id}"
        missing = sorted(n for n in self.names if n not in columns)
        if missing:
            return Check(name, NOT_RUN, critical=self.severity == "critical", source=self.source,
                         reason=f"the design has no {', '.join(missing)}: the rule reads "
                                f"{self.condition!r} and cannot be asked")
        blank = sorted(n for n in self.names if _isnan(columns[n]))
        if blank:
            return Check(name, NOT_RUN, critical=self.severity == "critical", source=self.source,
                         reason=f"the design has no value for {', '.join(blank)}: a missing number is not a pass")
        try:
            ok = self.holds(columns)
        except Exception as exc:  # noqa: BLE001 -- a rule that cannot be evaluated is not a pass either
            return Check(name, NOT_RUN, critical=self.severity == "critical", source=self.source,
                         reason=f"the condition {self.condition!r} could not be evaluated: {exc}")
        return Check(name, SUPPORTS if ok else CONTRADICTS, critical=self.severity == "critical",
                     source=f"{self.reviewer}, {self.date}" + (f" ({self.source})" if self.source else ""),
                     reason="" if ok else f"{self.condition} does not hold -- {self.statement}")

    # -- serialisation ------------------------------------------------------------------------------------
    @property
    def id(self) -> str:
        return digest_of(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, **asdict(self)}

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> Correction:
        fields = {k: v for k, v in row.items() if k in cls.__dataclass_fields__ and k != "units"}
        return cls(**fields, units=dict(row.get("units") or {}))


def _isnan(value: Any) -> bool:
    try:
        return value != value                     # NaN, for a number
    except Exception:  # noqa: BLE001
        return False


class CorrectionStore:
    """Append-only, chained store of corrections on disk (one JSON object per line).

    Append-only for the same reason the review records are: a correction is evidence of what an engineer said and
    when, so it is superseded by a new rule, never edited. Every line holds the digest of the line before it, and
    :meth:`verify` recomputes the chain -- an edited or removed rule is visible instead of silent.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def records(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        out: list[dict[str, Any]] = []
        for i, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise CorrectionError(f"{self.path.name}: line {i} is not JSON ({exc})") from exc
        return out

    def save(self, correction: Correction) -> Correction:
        """Append the rule; saving the same rule twice is idempotent, and it is never an edit."""
        existing = self.records()
        for row in existing:
            if row.get("id") == correction.id:
                return correction
        payload = {**correction.to_dict(), "prev": existing[-1]["id"] if existing else ""}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(_canonical(payload) + "\n")
        return correction

    def rules(self, case: str | None = None) -> list[Correction]:
        """Every stored rule, or the ones that apply to ``case``."""
        out = [Correction.from_dict(row) for row in self.records()]
        return [c for c in out if case is None or c.applies_to(case)]

    def verify(self) -> list[str]:
        """Every problem with the chain; an empty list means the rules are exactly the ones that were written."""
        problems: list[str] = []
        prev = ""
        for i, row in enumerate(self.records(), start=1):
            body = {k: v for k, v in row.items() if k not in ("id", "prev")}
            if row.get("id") != digest_of(body):
                problems.append(f"rule {i} does not match its own id: it was altered after saving")
            if row.get("prev", "") != prev:
                problems.append(f"rule {i} points at {row.get('prev')!r}, the previous rule is {prev!r}")
            prev = row.get("id", "")
        return problems

    def __len__(self) -> int:
        return len(self.records())


def correction_checks(rules: Iterable[Correction], columns: Mapping[str, Any], *, case: str = "") -> list[Check]:
    """The checks of every rule that applies to ``case``, in the order they were recorded."""
    return [c.check(columns) for c in rules if c.applies_to(case)]


def with_corrections(report: TrustReport, rules: Iterable[Correction], columns: Mapping[str, Any], *,
                     case: str = "") -> TrustReport:
    """``report`` with the stored corrections for ``case`` appended: recording a rule once is enough for it to be
    applied to the next design, and the report names the rule that failed."""
    rules = list(rules)
    new = correction_checks(rules, columns, case=case)
    details = dict(report.details)
    details["corrections"] = {"case": case, "applied": [c.rule_id for c in rules if c.applies_to(case)],
                             "failed": [k.name for k in new if k.status == CONTRADICTS],
                             "not_run": [k.name for k in new if k.status == NOT_RUN]}
    return replace(report, checks=list(report.checks) + new, details=details)

"""Scope of use: which uses a result can serve, where it fails, and why -- beyond one approve / reject verdict.

A trust score says whether a result can be used; a reviewing engineer also needs to know *for what*: a surrogate can
be fine for field values away from the walls and wrong for the boundary fluxes of the same prediction. A
:class:`ScopeOfUse` lists each use with its verdict, the measured value against the limit, what the use needs, and
where the evidence comes from:

- ``certified``: a guaranteed bound (e.g. a residual-based error bound);
- ``estimated``: an estimate without a guarantee (e.g. an error map from the residual), shown with its validation;
- ``input check``: the inputs against the validated range -- a precondition of every use, not a use of its own;
- ``reference``: a comparison with a reference solution.

Like the rest of this package it computes nothing: it carries verdicts that the analysis code already produced
(for instance ``pinneapple_physics.darcy_checks.use_verdicts``) into the Decision Record and the Evidence Report.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["EVIDENCE_KINDS", "UseScope", "ScopeOfUse", "scope_from_verdicts", "scope_from_batch"]

#: Where the evidence of a use verdict comes from, strongest first.
EVIDENCE_KINDS = ("certified", "reference", "estimated", "input check")


@dataclass
class UseScope:
    """One use of a result. ``ok``: True (usable), False (not), None (not assessed)."""
    use: str
    ok: bool | None
    reason: str = ""                       # the measured value against the limit
    why: str = ""                          # what the use needs
    evidence: str = "estimated"            # one of EVIDENCE_KINDS
    values: dict[str, float] = field(default_factory=dict)
    limits: dict[str, float] = field(default_factory=dict)
    fraction_ok: float | None = None       # over a batch, the share of samples for which the use is OK
    precondition: bool = False             # a condition of every use (inputs in range), not a use of its own

    def __post_init__(self):
        if self.evidence not in EVIDENCE_KINDS:
            raise ValueError(f"evidence must be one of {EVIDENCE_KINDS}, got {self.evidence!r}")

    @property
    def status(self) -> str:
        if self.ok is None:
            return "not assessed"
        if self.precondition:
            return "MET" if self.ok else "NOT MET"
        return "USABLE" if self.ok else "NOT USABLE"

    def line(self) -> str:
        share = "" if self.fraction_ok is None else f" ({100 * self.fraction_ok:.0f} % of samples)"
        why = f" -- needs: {self.why}" if self.why else ""
        return f"[{self.status}] {self.use}{share}: {self.reason} [{self.evidence}]{why}"


@dataclass
class ScopeOfUse:
    """The uses a result can and cannot serve, with an optional figure (a path to a PNG) and notes (for instance
    how an estimate was validated)."""
    subject: str
    uses: list[UseScope] = field(default_factory=list)
    figure: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def usable(self) -> list[str]:
        return [u.use for u in self.uses if u.ok and not u.precondition]

    @property
    def not_usable(self) -> list[str]:
        return [u.use for u in self.uses if u.ok is False and not u.precondition]

    @property
    def preconditions_not_met(self) -> list[str]:
        return [u.use for u in self.uses if u.precondition and u.ok is False]

    def summary(self) -> str:
        """One sentence: ``"use for: a, b; not for: c; precondition not met: inputs"``."""
        ok, bad, pre = self.usable, self.not_usable, self.preconditions_not_met
        parts = []
        if ok:
            parts.append("use for: " + ", ".join(ok))
        if bad:
            parts.append("not for: " + ", ".join(bad))
        if pre:
            parts.append("precondition not met: " + ", ".join(pre))
        elif any(u.precondition for u in self.uses):
            parts.append("preconditions met")
        return "; ".join(parts) or "no use assessed"

    def render(self) -> str:
        lines = [f"Scope of use -- {self.subject}", f"  {self.summary()}"]
        lines += [f"  - {u.line()}" for u in self.uses]
        lines += [f"  note: {n}" for n in self.notes]
        if self.figure:
            lines.append(f"  figure: {self.figure}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ScopeOfUse:
        return cls(subject=d["subject"], uses=[UseScope(**u) for u in d.get("uses", [])], figure=d.get("figure"),
                   notes=list(d.get("notes", [])))


def scope_from_verdicts(subject: str, verdicts: Mapping[str, Mapping[str, Any]], *,
                        evidence: str | Mapping[str, str] = "estimated", figure: str | None = None,
                        notes: Sequence[str] = ()) -> ScopeOfUse:
    """A :class:`ScopeOfUse` from per-use verdicts ``{use: {"ok", "reason", "why", "values", "limits"}}`` (the form
    of ``darcy_checks.use_verdicts``). ``evidence``: one kind for every use, or ``{use: kind}``."""
    uses = []
    for name, v in verdicts.items():
        kind = evidence if isinstance(evidence, str) else evidence.get(name, "estimated")
        uses.append(UseScope(name, v.get("ok"), str(v.get("reason", "")), str(v.get("why", "")), kind,
                             {k: float(x) for k, x in (v.get("values") or {}).items()},
                             {k: float(x) for k, x in (v.get("limits") or {}).items()},
                             precondition=kind == "input check"))
    return ScopeOfUse(subject, uses, figure, list(notes))


def scope_from_batch(subject: str, verdicts: Sequence[Mapping[str, Mapping[str, Any]]], *,
                     min_fraction: float = 0.9, evidence: str | Mapping[str, str] = "estimated",
                     figure: str | None = None, notes: Sequence[str] = ()) -> ScopeOfUse:
    """A :class:`ScopeOfUse` over a batch of per-sample verdicts: a use is OK when at least ``min_fraction`` of the
    samples serve it (0.9: nine in ten, the batch convention of ``pinneapple_physics.darcy_checks``); the share is
    kept in ``fraction_ok`` and the reason quotes the median measured values."""
    if not verdicts:
        raise ValueError("an empty batch has no scope of use")
    import statistics
    uses = []
    for name in verdicts[0]:
        rows = [v[name] for v in verdicts]
        frac = sum(bool(r.get("ok")) for r in rows) / len(rows)
        first = rows[0]
        med = {k: statistics.median(float(r["values"][k]) for r in rows) for k in (first.get("values") or {})}
        lim = {k: float(x) for k, x in (first.get("limits") or {}).items()}
        reason = ", ".join(f"median {k} {100 * med[k]:.1f} % (limit {100 * lim[k]:g} %)" for k in med) or \
            f"{100 * frac:.0f} % of samples OK"
        kind = evidence if isinstance(evidence, str) else evidence.get(name, "estimated")
        uses.append(UseScope(name, frac >= min_fraction, reason, str(first.get("why", "")), kind, med, lim, frac,
                             precondition=kind == "input check"))
    return ScopeOfUse(subject, uses, figure, list(notes))

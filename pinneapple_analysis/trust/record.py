"""pinneapple_analysis.trust.record — the trust layer as one object (#345).

Trust is produced in several places of the library and read in several others: :class:`pinneapple_physics.ensemble.
PhysicsEnsemble` has residual lookahead and a measurement check, :mod:`pinneapple_physics.weather.evaluate` measures
how far a forecast stays useful, :mod:`pinneapple_analysis.uncertainty` measures whether an interval is calibrated
and :mod:`pinneapple_decision.verifiers` turns an execution result into checks. This module is the object all of them
publish into, and the one a consumer reads:

.. code-block:: python

    t = pp.trust(ensemble, query)          # or Trust.from_* built by one of the adapters
    t.in_domain, t.domain_score            # is the input inside the training distribution, and by how much
    t.horizon                              # useful / no-skill lead
    t.residuals                            # per law: the value, its in-domain level, where it was evaluated
    t.measurement                          # checks on the readings the prediction was built from
    t.uncertainty                          # the interval, its width and its measured coverage
    t.provenance                           # model, data, versions, fingerprints
    t.to_json(); t.badge()                 # one versioned schema; good / fading / no_skill

The rule of the object is that **every field is either a number with the source it came from or an explicit
``not_measured`` with the reason**. There is no default value, no zero standing in for "unknown", and no combined
"trust score": :meth:`Trust.badge` is a decision over the measured checks, written down in its docstring, not a
weighted average of them.

Where the fields come from (the adapters live in :mod:`pinneapple_analysis.trust.adapters`):

===============  ==========================================================================
field            measured by
===============  ==========================================================================
``residuals``    ``PhysicsEnsemble.predict`` (``residual_fn`` / ``residual_lookahead``),
                 and the in-domain levels of :func:`pinneapple_physics.ensemble.residual_baseline`
``measurement``  ``PhysicsEnsemble``'s ``measurement_check`` (a suspect reading)
``horizon``      :func:`pinneapple_physics.weather.evaluate.horizons` (ACC 0.6 / climatology RMSE),
                 :func:`pinneapple_analysis.validation.rollout.divergence_time` (trust horizon)
``uncertainty``  :class:`pinneapple_analysis.uncertainty.calibration.CalibrationMetrics` (coverage, ECE)
                 and the interval ``PhysicsEnsemble.predict`` returns
``in_domain``    :class:`pinneapple_analysis.trust.trust_gate.TrustGate` (chi-squared p-value per point)
``provenance``   the model, the data and the library version, hashed where a hash is possible
===============  ==========================================================================
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

__all__ = [
    "SCHEMA", "MEASURED", "NOT_MEASURED", "GOOD", "FADING", "NO_SKILL",
    "GATE_PASS", "GATE_FAIL", "GATE_NO_SKILL", "GATE_UNMEASURED",
    "TrustValue", "Horizon", "Residual", "Measurement", "Uncertainty", "Provenance", "Gate", "Trust",
]

#: The schema this module writes. A reader compares the major number and refuses what it does not know.
SCHEMA = "pinneapple.trust/1"

MEASURED, NOT_MEASURED = "measured", "not_measured"

#: The three badges, and the four states a gate can be in.
GOOD, FADING, NO_SKILL = "good", "fading", "no_skill"
GATE_PASS, GATE_FAIL, GATE_NO_SKILL, GATE_UNMEASURED = "pass", "fail", "no_skill", "unmeasured"

_BADGE_TEXT = {GOOD: "good", FADING: "fading", NO_SKILL: "no skill"}


def _finite(value: Any, what: str) -> float:
    """A measured value has to be a finite real: NaN and inf are the two ways a number stops being one."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must be a real number, got {value!r}") from exc
    if not math.isfinite(v):
        raise ValueError(f"{what} must be finite, got {v!r}: a non-finite value is not a measurement "
                         "(mark the field not_measured and say why instead)")
    return v


def _num(value: float) -> float:
    """JSON leaves -0.0 and 1e-324 alone; normalising keeps a round trip exact."""
    return 0.0 if value == 0 else float(value)


@dataclass(frozen=True)
class TrustValue:
    """One field of a :class:`Trust`: a number with the source it came from, or the reason there is none.

    ``source`` names the function that produced the value and ``method`` says how; a measured value without a source
    is refused in ``__post_init__``, which is how "no number without provenance" is enforced instead of documented.

    ``reference`` is the level the value is judged against, when the source measures one (the residual of the same
    model inside its own domain, the nominal coverage of an interval, the fraction of points inside the distribution
    that counts as in domain). It carries its own ``reference_source``: a threshold nobody measured is not a
    reference.
    """

    status: str
    value: float | None = None
    unit: str = ""
    source: str = ""
    method: str = ""
    reference: float | None = None
    reference_source: str = ""
    where: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in (MEASURED, NOT_MEASURED):
            raise ValueError(f"status must be {MEASURED!r} or {NOT_MEASURED!r}, got {self.status!r}")
        if self.status == MEASURED:
            if self.value is None:
                raise ValueError("a measured field needs a value; use TrustValue.not_measured(reason) for the "
                                 "absence of one")
            object.__setattr__(self, "value", _finite(self.value, "a measured value"))
            if not self.source:
                raise ValueError("a measured field needs the source it came from: a number without provenance is "
                                 "not a measurement")
            if self.reference is not None:
                object.__setattr__(self, "reference", _finite(self.reference, "a reference level"))
                if not self.reference_source:
                    raise ValueError(f"the reference {self.reference} of a measured field needs its own source "
                                     "(reference_source): where does that level come from?")
            if self.reason:
                raise ValueError("a measured field carries a value, not a reason")
        else:
            if self.value is not None:
                raise ValueError(f"a {NOT_MEASURED} field cannot carry a value ({self.value!r}): that is how a "
                                 "missing measurement turns into a zero")
            if not self.reason:
                raise ValueError(f"a {NOT_MEASURED} field has to say why: a field with no measurement and no reason "
                                 "is not honest, it is empty")
            if self.reference is not None or self.reference_source:
                raise ValueError("a field that was not measured has nothing to judge against either")

    # -- construction ---------------------------------------------------------------------------------------
    @classmethod
    def measured(cls, value: Any, *, source: str, method: str = "", unit: str = "", reference: float | None = None,
                 reference_source: str = "", where: str = "") -> TrustValue:
        """A number the library computed, with where it came from."""
        return cls(MEASURED, float(value), unit, source, method, reference, reference_source, where)

    @classmethod
    def not_measured(cls, reason: str, *, source: str = "", method: str = "") -> TrustValue:
        """The absence of a measurement, with the reason it is absent — never a zero."""
        return cls(NOT_MEASURED, None, "", source, method, None, "", "", reason)

    @property
    def is_measured(self) -> bool:
        return self.status == MEASURED

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status}
        if self.status == MEASURED:
            out.update(value=_num(self.value), unit=self.unit, source=self.source, method=self.method)
            if self.reference is not None:
                out["reference"] = _num(self.reference)
                out["reference_source"] = self.reference_source
            if self.where:
                out["where"] = self.where
        else:
            out.update(source=self.source, method=self.method, reason=self.reason)
        return out

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> TrustValue:
        if document.get("status") == MEASURED:
            return cls.measured(document["value"], source=document.get("source", ""),
                                method=document.get("method", ""), unit=document.get("unit", ""),
                                reference=document.get("reference"),
                                reference_source=document.get("reference_source", ""),
                                where=document.get("where", ""))
        return cls.not_measured(document.get("reason", ""), source=document.get("source", ""),
                                method=document.get("method", ""))


@dataclass(frozen=True)
class Horizon:
    """How far a prediction stays useful: the lead at which it stops being useful and the lead at which it adds
    nothing over the reference (climatology, persistence, a constant).

    ``useful`` and ``no_skill`` are measured by whoever scores the predictions against truth — for a weather
    forecast, :func:`pinneapple_physics.weather.evaluate.horizons`; for an autoregressive rollout,
    :func:`pinneapple_analysis.validation.rollout.divergence_time`. ``requested`` is the lead this prediction is
    asked to cover, so :meth:`covers` can compare the two without inventing a third number.
    """

    useful: TrustValue
    no_skill: TrustValue
    requested: TrustValue
    unit: str = ""
    source: str = ""

    @classmethod
    def not_measured(cls, reason: str, *, source: str = "", unit: str = "") -> Horizon:
        empty = TrustValue.not_measured(reason, source=source)
        return cls(empty, replace(empty), replace(empty), unit, source)

    def covers(self) -> TrustValue:
        """Whether the requested lead is inside the no-skill lead. Measured only when both are; the comparison is
        derived from the two measurements and says so in ``method``."""
        if not self.requested.is_measured or not self.no_skill.is_measured:
            missing = "requested lead" if not self.requested.is_measured else "no-skill lead"
            return TrustValue.not_measured(
                f"the horizon test needs both leads and the {missing} is not measured",
                source=self.source or "pinneapple_analysis.trust.record.Horizon.covers",
                method="requested <= no_skill")
        ok = float(self.requested.value <= self.no_skill.value)  # type: ignore[operator]
        return TrustValue.measured(
            ok, unit="[-]", source=self.source or "pinneapple_analysis.trust.record.Horizon.covers",
            method=f"requested lead {self.requested.value:g} {self.unit} <= no-skill lead "
                   f"{self.no_skill.value:g} {self.unit}")

    def to_dict(self) -> dict[str, Any]:
        return {"unit": self.unit, "source": self.source, "useful": self.useful.to_dict(),
                "no_skill": self.no_skill.to_dict(), "requested": self.requested.to_dict(),
                "covers": self.covers().to_dict()}

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Horizon:
        return cls(TrustValue.from_dict(document["useful"]), TrustValue.from_dict(document["no_skill"]),
                   TrustValue.from_dict(document["requested"]), document.get("unit", ""),
                   document.get("source", ""))


@dataclass(frozen=True)
class Residual:
    """The residual of one law, for one model, on the case this prediction answers.

    ``law`` names what was conserved or balanced (``"mass"``, ``"momentum"``, ``"energy"``, or ``"pde"`` for the
    aggregate a source computes without splitting it). ``expert`` names the model when the source evaluates one
    residual per model. ``value`` carries its own in-domain ``reference`` — the residual of the same model inside
    its own domain, which is the only honest scale for it (:func:`pinneapple_physics.ensemble.residual_baseline`);
    without that level a residual has no pass mark and :meth:`Trust.gates` says so.

    ``scope`` is where the residual was evaluated ("whole field", "inlet patch"). A source that reports only a
    scalar cannot say where inside the field the law fails; that map comes from
    :func:`pinneapple_analysis.validation.error_map.error_map` and is not measured here.
    """

    law: str
    value: TrustValue
    scope: str = ""
    expert: str = ""

    @property
    def name(self) -> str:
        return f"{self.expert}/{self.law}" if self.expert else self.law

    def to_dict(self) -> dict[str, Any]:
        return {"law": self.law, "expert": self.expert, "scope": self.scope, "value": self.value.to_dict()}

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Residual:
        return cls(document["law"], TrustValue.from_dict(document["value"]), document.get("scope", ""),
                   document.get("expert", ""))


@dataclass(frozen=True)
class Measurement:
    """Checks on the readings the prediction was built from — not on the prediction.

    ``suspect`` is 1 when the reading failed the model's own check (:class:`pinneapple_physics.ensemble.
    PhysicsEnsemble`'s ``measurement_check``: a failed balance between redundant sensors, an impossible jump) and
    0 when it passed. ``checks`` keeps every named check the source reported, each with its own source.
    """

    suspect: TrustValue
    checks: Mapping[str, TrustValue] = field(default_factory=dict)
    n_readings: TrustValue | None = None
    n_suspect: TrustValue | None = None
    source: str = ""

    @classmethod
    def not_measured(cls, reason: str, *, source: str = "") -> Measurement:
        return cls(TrustValue.not_measured(reason, source=source), {}, None, None, source)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"source": self.source, "suspect": self.suspect.to_dict(),
                               "checks": {k: v.to_dict() for k, v in self.checks.items()}}
        if self.n_readings is not None:
            out["n_readings"] = self.n_readings.to_dict()
        if self.n_suspect is not None:
            out["n_suspect"] = self.n_suspect.to_dict()
        return out

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Measurement:
        return cls(TrustValue.from_dict(document["suspect"]),
                   {k: TrustValue.from_dict(v) for k, v in (document.get("checks") or {}).items()},
                   TrustValue.from_dict(document["n_readings"]) if document.get("n_readings") else None,
                   TrustValue.from_dict(document["n_suspect"]) if document.get("n_suspect") else None,
                   document.get("source", ""))


@dataclass(frozen=True)
class Uncertainty:
    """The interval a prediction carries, and how calibrated it actually is.

    ``level`` is the nominal level the model claims for its interval (0.9 for a 90 % interval), when it states one.
    ``coverage`` is the fraction of held-out points that actually fell inside it, measured by
    :meth:`pinneapple_analysis.uncertainty.calibration.CalibrationMetrics.coverage_at_level`; ``ece`` is the
    expected calibration error of the same predictions. An interval whose measured coverage is below its nominal
    level is narrow, whatever it claims — :meth:`calibrated` compares exactly those two numbers.
    """

    level: float | None
    coverage: TrustValue
    width: TrustValue
    ece: TrustValue
    method: str = ""
    source: str = ""

    @classmethod
    def not_measured(cls, reason: str, *, source: str = "") -> Uncertainty:
        empty = TrustValue.not_measured(reason, source=source)
        return cls(None, replace(empty), replace(empty), replace(empty), "", source)

    def calibrated(self) -> TrustValue:
        """Measured coverage against the nominal level: 1 when the interval covered at least what it claims."""
        if not self.coverage.is_measured or self.level is None:
            why = "the coverage" if not self.coverage.is_measured else "the nominal level"
            return TrustValue.not_measured(f"{why} is not measured, so calibration cannot be judged",
                                           source=self.source or "pinneapple_analysis.trust.record."
                                                                 "Uncertainty.calibrated",
                                           method="coverage >= level")
        ok = float(self.coverage.value >= self.level)  # type: ignore[operator]
        return TrustValue.measured(
            ok, unit="[-]", source=self.source or "pinneapple_analysis.trust.record.Uncertainty.calibrated",
            method=f"measured coverage {self.coverage.value:.4g} >= nominal level {self.level:.4g}",
            reference=float(self.level),
            reference_source="the nominal level of the interval, stated by the model")

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level, "method": self.method, "source": self.source,
                "coverage": self.coverage.to_dict(), "width": self.width.to_dict(),
                "ece": self.ece.to_dict(), "calibrated": self.calibrated().to_dict()}

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Uncertainty:
        return cls(document.get("level"), TrustValue.from_dict(document["coverage"]),
                   TrustValue.from_dict(document["width"]), TrustValue.from_dict(document["ece"]),
                   document.get("method", ""), document.get("source", ""))


@dataclass(frozen=True)
class Provenance:
    """Where the prediction came from: the model, the data, the versions, and a fingerprint of each where one can be
    taken. A field that could not be recorded is ``None`` in the JSON and named by :meth:`not_recorded`, never
    filled with an invented value.
    """

    kind: str = ""
    model: str = ""
    model_fingerprint: str = ""
    model_fingerprint_method: str = ""
    data_fingerprint: str = ""
    data_fingerprint_method: str = ""
    library_version: str = ""
    created: str = ""
    inputs: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def not_recorded(self) -> tuple[str, ...]:
        names = ("kind", "model", "model_fingerprint", "data_fingerprint", "library_version", "created")
        return tuple(n for n in names if not getattr(self, n))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind or None, "model": self.model or None,
                "model_fingerprint": self.model_fingerprint or None,
                "model_fingerprint_method": self.model_fingerprint_method or None,
                "data_fingerprint": self.data_fingerprint or None,
                "data_fingerprint_method": self.data_fingerprint_method or None,
                "library_version": self.library_version or None, "created": self.created or None,
                "inputs": dict(self.inputs), "notes": list(self.notes),
                "not_recorded": list(self.not_recorded())}

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Provenance:
        return cls(*(document.get(k) or "" for k in ("kind", "model", "model_fingerprint",
                                                     "model_fingerprint_method", "data_fingerprint",
                                                     "data_fingerprint_method", "library_version", "created")),
                   dict(document.get("inputs") or {}), tuple(document.get("notes") or ()))


@dataclass(frozen=True)
class Gate:
    """One check of a :class:`Trust`, decided, with the sentence that decided it."""

    name: str
    status: str
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "evidence": self.evidence}


def _fingerprint_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fingerprint_array(array: Any) -> tuple[str, str]:
    """SHA-256 of an array's float64 values in C order, with the method that produced it. Anything that is not an
    array of finite numbers gets ``("", "")``: a hash of ``repr`` is not a fingerprint of the data."""
    import numpy as np

    try:
        a = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    except (TypeError, ValueError):
        return "", ""
    if a.size == 0 or not np.all(np.isfinite(a)):
        return "", ""
    return _fingerprint_bytes(a.tobytes()), f"sha256 of the array ({a.dtype}, C order, shape {a.shape})"


def fingerprint_text(text: str) -> tuple[str, str]:
    """SHA-256 of a text (a model name, a serialised state), with the method that produced it."""
    if not text:
        return "", ""
    return _fingerprint_bytes(text.encode("utf-8")), "sha256 of the utf-8 text"


def library_version() -> str:
    """``"pinneapple x.y.z"``, from the installed distribution or, in a source checkout, from the package itself;
    empty when neither can be read (and then the provenance says it was not recorded)."""
    try:
        from importlib.metadata import version
        return f"pinneapple {version('pinneapple')}"
    except Exception:                                    # not installed as a distribution (source checkout)
        try:
            from pinneapple import __version__
            return f"pinneapple {__version__}"
        except Exception:
            return ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Trust:
    """Everything the library knows about one prediction, in one object a consumer can read (#345).

    The seven fields are the ones issue #345 names. Each is a :class:`TrustValue` (or a section of them), so every
    number carries its source and every absence carries its reason.
    """

    kind: str
    in_domain: TrustValue
    domain_score: TrustValue
    horizon: Horizon
    residuals: tuple[Residual, ...] = ()
    measurement: Measurement = field(
        default_factory=lambda: Measurement.not_measured("this Trust was built without a measurement adapter"))
    uncertainty: Uncertainty = field(
        default_factory=lambda: Uncertainty.not_measured("this Trust was built without an uncertainty adapter"))
    provenance: Provenance = field(default_factory=Provenance)
    schema: str = SCHEMA
    notes: tuple[str, ...] = ()

    # -- construction ---------------------------------------------------------------------------------------
    @classmethod
    def empty(cls, *, kind: str, model: str = "", reason: str = "no adapter was given for this prediction kind",
              library: str | None = None, created: str | None = None, notes: Iterable[str] = ()) -> Trust:
        """A Trust in which every field says ``not_measured`` with ``reason``: the shape a consumer can always read,
        before any adapter has filled it."""
        missing = TrustValue.not_measured(reason)
        return cls(
            kind=kind,
            in_domain=replace(missing),
            domain_score=replace(missing),
            horizon=Horizon.not_measured(reason),
            residuals=(),
            measurement=Measurement.not_measured(reason),
            uncertainty=Uncertainty.not_measured(reason),
            provenance=Provenance(kind=kind, model=model, library_version=library if library is not None
                                  else library_version(), created=created if created is not None else utc_now(),
                                  notes=tuple(notes)),
            notes=tuple(notes),
        )

    def with_(self, **changes: Any) -> Trust:
        """A copy with some sections replaced — how the adapters compose without mutating a shared object."""
        return replace(self, **changes)

    # -- reading --------------------------------------------------------------------------------------------
    def residual(self, law: str, expert: str = "") -> Residual | None:
        """The residual recorded for ``law`` (and, when several models report it, for ``expert``)."""
        for r in self.residuals:
            if r.law == law and (not expert or r.expert == expert):
                return r
        return None

    def residuals_by_law(self) -> dict[str, tuple[Residual, ...]]:
        out: dict[str, list[Residual]] = {}
        for r in self.residuals:
            out.setdefault(r.law, []).append(r)
        return {k: tuple(v) for k, v in out.items()}

    def gates(self) -> tuple[Gate, ...]:
        """Every check this object can decide, with the number and the source behind each decision.

        A gate is ``unmeasured`` when the number it needs is missing **or** when the number is there but nothing was
        measured to compare it against: a residual with no in-domain level has no pass mark, and saying "pass"
        because there is nothing to compare it with is how a check stops meaning anything.
        """
        out: list[Gate] = []

        if self.in_domain.is_measured:
            ok = self.in_domain.value == 1.0
            score = (f" (domain_score {self.domain_score.value:g})" if self.domain_score.is_measured else "")
            out.append(Gate("domain", GATE_PASS if ok else GATE_NO_SKILL,
                            f"in_domain = {self.in_domain.value:g}{score}, from {self.in_domain.source}"))
        else:
            out.append(Gate("domain", GATE_UNMEASURED, self.in_domain.reason))

        covers = self.horizon.covers()
        if covers.is_measured:
            out.append(Gate("horizon", GATE_PASS if covers.value == 1.0 else GATE_NO_SKILL, covers.method))
        else:
            out.append(Gate("horizon", GATE_UNMEASURED, covers.reason))

        for r in self.residuals:
            v = r.value
            if not v.is_measured:
                out.append(Gate(f"residual:{r.name}", GATE_UNMEASURED, v.reason))
            elif v.reference is None:
                out.append(Gate(f"residual:{r.name}", GATE_UNMEASURED,
                                f"the residual is measured ({v.value:g}) but its in-domain level is not, so nothing "
                                f"says whether that value is good: {v.source}"))
            else:
                ok = v.value <= v.reference
                out.append(Gate(f"residual:{r.name}", GATE_PASS if ok else GATE_FAIL,
                                f"{v.value:g} against its in-domain level {v.reference:g} ({v.reference_source})"))

        if self.measurement.suspect.is_measured:
            s = self.measurement.suspect.value or 0.0
            out.append(Gate("measurement", GATE_PASS if s == 0.0 else GATE_FAIL,
                            f"suspect = {s:g} on the readings the prediction used ({self.measurement.suspect.source})"))
        else:
            out.append(Gate("measurement", GATE_UNMEASURED, self.measurement.suspect.reason))
        for name, check in self.measurement.checks.items():
            if check.is_measured:
                ok = check.value == 1.0
                out.append(Gate(f"check:{name}", GATE_PASS if ok else GATE_FAIL,
                                f"{check.value:g} = {'passed' if ok else 'failed'} ({check.source})"))
            else:
                out.append(Gate(f"check:{name}", GATE_UNMEASURED, check.reason))

        cal = self.uncertainty.calibrated()
        out.append(Gate("calibration", GATE_PASS if cal.value == 1.0 else GATE_FAIL, cal.method)
                   if cal.is_measured else Gate("calibration", GATE_UNMEASURED, cal.reason))
        return tuple(out)

    def badge(self) -> str:
        """``good`` / ``fading`` / ``no_skill`` — a decision over :meth:`gates`, not a score.

        The rule, in order and with no weights:

        1. **no_skill** — a measured gate proves the prediction carries no information: its inputs are outside the
           training distribution (``in_domain`` = 0), or the lead it is asked to cover is past the measured
           no-skill lead.
        2. **fading** — nothing proves no skill, but at least one measured gate fails (a residual above its own
           in-domain level, a suspect reading, a coverage below the nominal level) or at least one gate was not
           measured. A missing check never counts as a pass: an object that cannot vouch for a field says so.
        3. **good** — every gate this object carries is measured and passes.

        There is no fourth value and no averaged score: two models can both be "fading" for opposite reasons, and
        :meth:`badge_reason` gives the sentence (and the sources) that decided it. In particular, "fading" because a
        field was not measured is not a failure of the prediction — it is the object refusing to claim more than it
        measured.
        """
        gates = self.gates()
        if any(g.status == GATE_NO_SKILL for g in gates):
            return NO_SKILL
        if any(g.status in (GATE_FAIL, GATE_UNMEASURED) for g in gates):
            return FADING
        return GOOD

    def badge_text(self) -> str:
        """The badge as it is written on a figure: ``good`` / ``fading`` / ``no skill``."""
        return _BADGE_TEXT[self.badge()]

    def badge_reason(self) -> str:
        """One sentence per gate that decided the badge, each naming its number and its source."""
        badge = self.badge()
        deciding = [g for g in self.gates() if g.status in (GATE_NO_SKILL, GATE_FAIL, GATE_UNMEASURED)]
        if badge == GOOD:
            return f"good: all {len(self.gates())} gates measured and passing"
        return _BADGE_TEXT[badge] + ": " + "; ".join(f"{g.name} {g.status} — {g.evidence}" for g in deciding)

    # -- serialisation ----------------------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "kind": self.kind,
            "badge": self.badge(),
            "badge_text": self.badge_text(),
            "badge_reason": self.badge_reason(),
            "in_domain": self.in_domain.to_dict(),
            "domain_score": self.domain_score.to_dict(),
            "horizon": self.horizon.to_dict(),
            "residuals": [r.to_dict() for r in self.residuals],
            "measurement": self.measurement.to_dict(),
            "uncertainty": self.uncertainty.to_dict(),
            "provenance": self.provenance.to_dict(),
            "gates": [g.to_dict() for g in self.gates()],
            "notes": list(self.notes),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False, ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Trust:
        schema = str(document.get("schema", ""))
        if schema.split("/")[0] != SCHEMA.split("/")[0]:
            raise ValueError(f"this is a {SCHEMA.split('/')[0]} document, not a {schema or 'versioned'} one")
        return cls(
            kind=document.get("kind", ""),
            in_domain=TrustValue.from_dict(document["in_domain"]),
            domain_score=TrustValue.from_dict(document["domain_score"]),
            horizon=Horizon.from_dict(document["horizon"]),
            residuals=tuple(Residual.from_dict(r) for r in document.get("residuals") or ()),
            measurement=Measurement.from_dict(document["measurement"]),
            uncertainty=Uncertainty.from_dict(document["uncertainty"]),
            provenance=Provenance.from_dict(document.get("provenance") or {}),
            schema=schema or SCHEMA,
            notes=tuple(document.get("notes") or ()),
        )

    @classmethod
    def from_json(cls, text: str) -> Trust:
        return cls.from_dict(json.loads(text))

    def write(self, path: Any, *, indent: int | None = 2) -> Any:
        """Write the JSON to ``path`` and return it."""
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(indent=indent), encoding="utf-8")
        return p

    def __str__(self) -> str:
        return f"Trust({self.kind}): {self.badge_text()} — {self.badge_reason()}"

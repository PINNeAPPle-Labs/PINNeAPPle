"""The contract every system in the Digital Twin Library honours.

A twin in this library is a *reference model*: a deterministic, physically interpretable
reduced model of an engineering system, written so that three things can be checked without
anyone's opinion — its conservation laws, the invariants that follow from its own equations,
and the domain where it stops being valid.

Why a reference model and not a surrogate
-----------------------------------------
A trained surrogate is only as good as the solver output behind it, and the solver is only
as good as the case it was given. A reference model closes the loop the other way: it is
cheap enough to run in a test suite, it states its own assumptions, and it produces the
numbers a surrogate would later be trained on. Every twin here is the *first rung* of the
fidelity ladder the Digital Twin Library issues describe — 0D now, 1D/2D/3D and a trained
surrogate later, on top of data this model generates.

What is deliberately not here
-----------------------------
No telemetry, no state estimation, no streaming. Those live in
:mod:`pinneapple_systems.digital_twin` (the runtime) and
:mod:`pinneapple_systems.digital_twin.product` (the commissioning layer, whose
:class:`~pinneapple_systems.digital_twin.product.spec.TwinSpec` can wrap a model written
here). This package only answers: *given these inputs, what does the physics say, and does
it balance?*

The three checks a result carries
---------------------------------
``balances``
    Conservation laws with both sides computed independently — carbon in versus carbon out,
    energy in versus energy stored plus lost. A residual above the tolerance fails the run.
``checks``
    Invariants that follow from the equations themselves and must hold for *any* parameter
    set (a recovery that cannot exceed 100 %, an equilibrium conversion that rises with
    pressure, a landing that cannot use more propellant than it carries).
``not_verified``
    The honesty list: what this model cannot say, written by the author of the model rather
    than discovered by the user. It is printed by the examples and stored in the JSON.
"""
from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "LibraryError",
    "ParameterError",
    "BalanceFailed",
    "Parameter",
    "Quantity",
    "Balance",
    "Check",
    "TwinRun",
    "ReferenceModel",
    "TINY",
]

#: Relative residual below which a conservation law is called closed. Chosen for a model
#: whose terms are computed from the same inputs, not propagated through a mesh: 1e-9 is
#: what double precision leaves after a handful of operations, and anything larger means the
#: model itself lost or invented mass.
TINY = 1e-12
BALANCE_TOLERANCE = 1e-9


class LibraryError(ValueError):
    """The twin cannot be built or run as asked."""


class ParameterError(LibraryError):
    """A parameter is missing, out of its declared range, or not a finite number."""


class BalanceFailed(LibraryError):
    """A conservation law did not close. Raised only when a caller asks for it."""


# ---------------------------------------------------------------------------
# Parameters and quantities
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Parameter:
    """One input of a reference model: its unit, its range, and what it means.

    ``lo``/``hi`` are the range in which the model's own equations are the right physics —
    not merely where the arithmetic survives. A hydrostatic line model is valid at 50 bar and
    arithmetic at 5000 bar; the difference is a phase change the model does not carry, so the
    range stops first. Values outside the range raise unless the caller passes
    ``strict=False`` to :meth:`ReferenceModel.simulate`, which is how an extrapolation study
    is run deliberately and recorded as such.
    """

    name: str
    unit: str
    default: float
    lo: float
    hi: float
    meaning: str

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ParameterError(f"parameter {self.name!r}: use a Python identifier")
        if not self.unit:
            raise ParameterError(f"parameter {self.name!r}: a unit is required (use '-' if none)")
        if not self.lo < self.hi:
            raise ParameterError(f"parameter {self.name!r}: lo ({self.lo}) must be below hi ({self.hi})")
        if not self.lo <= self.default <= self.hi:
            raise ParameterError(
                f"parameter {self.name!r}: default {self.default} is outside [{self.lo}, {self.hi}]"
            )

    def check(self, value: Any, *, strict: bool = True) -> float:
        """Return ``value`` as a float, refusing what the model cannot mean."""
        try:
            v = float(value)
        except (TypeError, ValueError) as exc:
            raise ParameterError(f"parameter {self.name!r}: {value!r} is not a number") from exc
        if not math.isfinite(v):
            raise ParameterError(f"parameter {self.name!r}: {v!r} is not finite")
        if strict and not (self.lo <= v <= self.hi):
            raise ParameterError(
                f"parameter {self.name!r} = {v} {self.unit} is outside the model's range "
                f"[{self.lo}, {self.hi}] — pass strict=False to extrapolate on purpose"
            )
        return v

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "unit": self.unit, "default": self.default,
            "lo": self.lo, "hi": self.hi, "meaning": self.meaning,
        }


@dataclass(frozen=True)
class Quantity:
    """A number, its unit, and the sentence that says how it was obtained."""

    name: str
    value: float
    unit: str
    how: str = ""

    def __post_init__(self) -> None:
        # A quantity is a scalar with a unit. Anything array-like arriving here is a bug in the
        # model, and it is better to hear about it here than in a formatted report.
        object.__setattr__(self, "value", float(self.value))
        if not self.unit:
            raise ParameterError(f"quantity {self.name!r}: a unit is required (use '-' if none)")

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value, "unit": self.unit, "how": self.how}

    def __str__(self) -> str:
        return f"{self.name} = {self.value:.6g} {self.unit}"


@dataclass(frozen=True)
class Balance:
    """One conservation law, with both sides computed from independent input routes.

    ``inflow`` and ``outflow`` are not two halves of the same sum — that would close by
    construction and prove nothing. For the blast furnace, for instance, the carbon entering
    is weight times assay, while the carbon leaving is read back out of the top-gas
    composition and the metal's carbon content; they agree only if the stoichiometry is right.
    """

    name: str
    inflow: float
    outflow: float
    unit: str
    tolerance: float = BALANCE_TOLERANCE
    detail: str = ""
    scale: float | None = None

    @property
    def residual(self) -> float:
        """Relative residual, against ``scale`` when the two sides are near zero.

        A flight that starts and ends at rest has a net energy change of almost nothing, while
        the energy exchanged along the way is enormous: judging that balance against its own
        endpoints would flag a 3e-5 quadrature error as a 116 % failure. ``scale`` names the
        magnitude the residual belongs to — the energy traded during the flight, the peak speed
        — and the balance is then read the way an engineer reads it.
        """
        reference = self.scale if self.scale is not None else max(abs(self.inflow), abs(self.outflow))
        return (self.inflow - self.outflow) / max(abs(reference), 1e-30)

    @property
    def ok(self) -> bool:
        return abs(self.residual) <= self.tolerance

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "inflow": self.inflow, "outflow": self.outflow, "unit": self.unit,
            "residual": self.residual, "tolerance": self.tolerance, "ok": self.ok,
            "scale": self.scale, "detail": self.detail,
        }

    def __str__(self) -> str:
        return f"{'PASS' if self.ok else 'FAIL'} {self.name}: {self.residual:+.2e} rel. ({self.unit})"


@dataclass(frozen=True)
class Check:
    """An invariant of the model, asserted rather than measured."""

    name: str
    passed: bool
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}

    def __str__(self) -> str:
        return f"{'PASS' if self.passed else 'FAIL'} {self.name}" + (f" — {self.detail}" if self.detail else "")


def relative_error(computed: float, reference: float) -> float:
    """``|computed - reference| / |reference|``, safe at a zero reference."""
    return abs(computed - reference) / max(abs(reference), TINY)


def close(name: str, inflow: float, outflow: float, unit: str, *, tolerance: float = BALANCE_TOLERANCE,
          detail: str = "", scale: float | None = None) -> Balance:
    """Build a :class:`Balance` from two independently computed totals."""
    return Balance(name=name, inflow=float(inflow), outflow=float(outflow), unit=unit,
                   tolerance=tolerance, detail=detail, scale=scale)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

@dataclass
class TwinRun:
    """Everything one run of a reference model produces, in a form that outlives the session."""

    system: str
    title: str
    parameters: dict[str, float]
    outputs: dict[str, Quantity]
    balances: list[Balance] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    series: dict[str, list[float]] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    limits: list[str] = field(default_factory=list)
    not_verified: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    extrapolated: list[str] = field(default_factory=list)
    created: float = field(default_factory=time.time)

    # -- verdicts ----------------------------------------------------------
    @property
    def balanced(self) -> bool:
        """Every conservation law closed within its own tolerance."""
        return all(b.ok for b in self.balances)

    @property
    def invariants_hold(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def ok(self) -> bool:
        """The run is usable: every balance closed and every invariant held."""
        return self.balanced and self.invariants_hold

    def value(self, name: str) -> float:
        """The numeric value of one output, raising a clear error if it was not produced."""
        if name not in self.outputs:
            raise LibraryError(f"{self.system} did not produce output {name!r}; it has {sorted(self.outputs)}")
        return self.outputs[name].value

    def require(self) -> TwinRun:
        """Raise :class:`BalanceFailed` unless the run passed. The chainable form of ``assert run.ok``."""
        if not self.ok:
            bad = [str(b) for b in self.balances if not b.ok] + [str(c) for c in self.checks if not c.passed]
            raise BalanceFailed(f"{self.system}: " + "; ".join(bad))
        return self

    # -- serialisation -----------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "title": self.title,
            "created": self.created,
            "parameters": dict(self.parameters),
            "outputs": {k: v.as_dict() for k, v in self.outputs.items()},
            "balances": [b.as_dict() for b in self.balances],
            "checks": [c.as_dict() for c in self.checks],
            "series_length": {k: len(v) for k, v in self.series.items()},
            "assumptions": list(self.assumptions),
            "limits": list(self.limits),
            "not_verified": list(self.not_verified),
            "sources": list(self.sources),
            "extrapolated": list(self.extrapolated),
            "ok": self.ok,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=False)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(self.to_json(), encoding="utf-8")
        return p

    def series_json(self, *, indent: int = 2) -> str:
        """The time series, kept apart from the summary so a run's JSON stays readable."""
        return json.dumps({"system": self.system, "series": self.series}, indent=indent)

    def report(self, *, width: int = 78) -> str:
        """One screen: the outputs, the balances, the invariants and what was not checked."""
        lines = [f"{self.title}", "=" * min(width, max(len(self.title), 20)), ""]
        lines.append("outputs")
        for q in self.outputs.values():
            lines.append(f"  {q.name:<28s} {q.value:>14.6g} {q.unit:<12s} {q.how}")
        if self.balances:
            lines += ["", "conservation"]
            lines += ["  " + str(b) for b in self.balances]
        if self.checks:
            lines += ["", "invariants"]
            lines += ["  " + str(c) for c in self.checks]
        if self.limits:
            lines += ["", "model limits"]
            lines += [f"  - {s}" for s in self.limits]
        if self.not_verified:
            lines += ["", "not verified"]
            lines += [f"  - {s}" for s in self.not_verified]
        if self.extrapolated:
            lines += ["", "extrapolated (outside the declared range)"]
            lines += [f"  ! {s}" for s in self.extrapolated]
        lines += ["", f"verdict: {'PASS' if self.ok else 'FAIL'}"]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class ReferenceModel:
    """Base class for a system in the library.

    A subclass declares ``slug``, ``title``, ``physics``, ``parameters`` (a mapping of
    :class:`Parameter`), its ``assumptions``, ``limits`` and ``sources``, and implements
    :meth:`_run`, returning a :class:`TwinRun`. It must not implement its own validation of
    parameters or its own JSON: that is this class's job, so that every twin in the library
    behaves the same way.

    The dependencies between parameters are the one thing a declaration cannot express
    (``pressure_outlet`` must stay below ``pressure_reservoir``), so :meth:`_validate` exists
    for those and raises :class:`ParameterError` with the offending pair named.
    """

    slug: str = ""
    title: str = ""
    physics: str = ""
    parameters: Mapping[str, Parameter] = {}
    assumptions: Sequence[str] = ()
    limits: Sequence[str] = ()
    sources: Sequence[str] = ()

    #: Set by :meth:`simulate` when the caller asked for ``strict=False`` and actually left
    #: the declared range; recorded in the run so an extrapolation can never pass for a result.
    extrapolated: list[str] = []

    def defaults(self) -> dict[str, float]:
        """Every parameter at its declared default — the reproducible scenario."""
        return {name: p.default for name, p in self.parameters.items()}

    def describe(self) -> str:
        lines = [f"{self.title} ({self.slug})", f"physics: {self.physics}", "", "parameters"]
        for p in self.parameters.values():
            lines.append(f"  {p.name:<26s} {p.default:>12.6g} {p.unit:<10s} [{p.lo}, {p.hi}]  {p.meaning}")
        if self.assumptions:
            lines += ["", "assumptions"] + [f"  - {a}" for a in self.assumptions]
        if self.limits:
            lines += ["", "limits"] + [f"  - {a}" for a in self.limits]
        if self.sources:
            lines += ["", "sources"] + [f"  - {a}" for a in self.sources]
        return "\n".join(lines)

    def simulate(self, values: Mapping[str, Any] | None = None, *, strict: bool = True, **kwargs: Any) -> TwinRun:
        """Validate the inputs, run the model, and stamp the result with its provenance."""
        given: dict[str, Any] = dict(values or {})
        overlap = set(given) & set(kwargs)
        if overlap:
            raise ParameterError(f"parameter(s) {sorted(overlap)} given twice")
        given.update(kwargs)
        unknown = set(given) - set(self.parameters)
        if unknown:
            raise ParameterError(
                f"{self.slug}: unknown parameter(s) {sorted(unknown)}; it takes {sorted(self.parameters)}"
            )

        params: dict[str, float] = {}
        extrapolated: list[str] = []
        for name, spec in self.parameters.items():
            raw = given.get(name, spec.default)
            value = spec.check(raw, strict=False)
            params[name] = value
            if not (spec.lo <= value <= spec.hi):
                if strict:
                    spec.check(raw, strict=True)  # raises with the canonical message
                extrapolated.append(f"{name} = {value} {spec.unit} outside [{spec.lo}, {spec.hi}]")

        self._validate(params)
        run = self._run(params)
        run.extrapolated = extrapolated
        if not run.assumptions:
            run.assumptions = list(self.assumptions)
        if not run.limits:
            run.limits = list(self.limits)
        if not run.sources:
            run.sources = list(self.sources)
        return run

    # -- to be provided by the system --------------------------------------
    def _validate(self, params: Mapping[str, float]) -> None:
        """Cross-parameter checks. Raise :class:`ParameterError` when the combination is wrong."""

    def _run(self, params: Mapping[str, float]) -> TwinRun:  # pragma: no cover - abstract
        raise NotImplementedError(f"{type(self).__name__} does not implement _run")

    # -- helpers for subclasses --------------------------------------------
    @staticmethod
    def _series(length: int, **columns: Iterable[float]) -> dict[str, list[float]]:
        cols = {k: [float(x) for x in v] for k, v in columns.items()}
        bad = {k: len(v) for k, v in cols.items() if len(v) != length}
        if bad:
            raise LibraryError(f"series columns of the wrong length: {bad}, expected {length}")
        return cols

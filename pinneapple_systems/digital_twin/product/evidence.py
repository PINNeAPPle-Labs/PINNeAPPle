"""The evidence a twin has to produce to be trusted, and the shapes it is filed in.

Two rules run through this module.

**A number carries its source.** Every metric knows how many comparisons produced it and
over which window, because "RMSE 0.4 °C" means nothing without "on 1184 held-out
observations over 19.7 h".

**What was not checked is listed.** :class:`TwinEvidence` carries an ``unverified`` list
next to its metrics. A report that only shows what passed is a brochure, not evidence.

Nothing here computes physics: these are containers plus honest formatting. The numbers
come from :mod:`~pinneapple_systems.digital_twin.product.commission`.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

__all__ = [
    "PASS", "WARN", "FAIL", "INFO",
    "CheckItem", "CheckReport",
    "FieldMetrics", "TrackingReport",
    "TwinEvidence",
]

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
INFO = "INFO"

_SEVERITY = {PASS: 0, INFO: 1, WARN: 2, FAIL: 3}


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckItem:
    """One line of a preflight checklist.

    ``why`` says what goes wrong if it is ignored; ``fix`` says what to change. A checklist
    that only says "invalid" pushes the diagnosis back to the reader.
    """

    check_id: str
    status: str
    title: str
    detail: str = ""
    why: str = ""
    fix: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        if self.status not in _SEVERITY:
            raise ValueError(f"status must be one of {tuple(_SEVERITY)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def line(self) -> str:
        detail = f" — {self.detail}" if self.detail else ""
        return f"[{self.status:4}] {self.check_id} {self.title}{detail}"


@dataclass
class CheckReport:
    """An ordered checklist with a verdict: any ``FAIL`` blocks commissioning."""

    title: str
    items: list[CheckItem] = field(default_factory=list)

    def add(self, check_id: str, status: str, title: str, **kw: Any) -> CheckItem:
        item = CheckItem(check_id=check_id, status=status, title=title, **kw)
        self.items.append(item)
        return item

    def extend(self, items: Iterable[CheckItem]) -> None:
        self.items.extend(items)

    @property
    def failed(self) -> list[CheckItem]:
        return [i for i in self.items if i.status == FAIL]

    @property
    def warnings(self) -> list[CheckItem]:
        return [i for i in self.items if i.status == WARN]

    @property
    def passed(self) -> bool:
        return not self.failed

    def counts(self) -> dict[str, int]:
        out = {k: 0 for k in _SEVERITY}
        for i in self.items:
            out[i.status] += 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "passed": self.passed,
            "counts": self.counts(),
            "items": [i.to_dict() for i in self.items],
        }

    def render(self) -> str:
        head = f"{self.title}: {'PASS' if self.passed else 'FAIL'}  {self.counts()}"
        return "\n".join([head] + [f"  {i.line()}" for i in self.items])


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldMetrics:
    """Agreement between the twin and reality, on observations it never assimilated.

    ``rmse_frac`` and ``bias_frac`` are divided by the field's declared span so the gate
    means the same thing across quantities. ``coverage`` is the fraction of held-out
    observations inside the twin's own 95 % band: the honesty check. A twin can have a low
    RMSE and a poor coverage at the same time — right on average, wrong where it counts.
    """

    field_name: str
    unit: str
    n: int
    rmse: float
    mae: float
    bias: float
    p95_abs: float
    coverage: float
    span: float | None = None
    window_s: float | None = None
    coverage_raw: float | None = None
    band_scale: float = 1.0
    calibration_s: float | None = None

    @property
    def rmse_frac(self) -> float | None:
        return None if not self.span else self.rmse / self.span

    @property
    def bias_frac(self) -> float | None:
        return None if not self.span else self.bias / self.span

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["rmse_frac"] = self.rmse_frac
        d["bias_frac"] = self.bias_frac
        return d

    def render(self) -> str:
        frac = f" ({100 * self.rmse_frac:.2f}% of span)" if self.rmse_frac is not None else ""
        win = f", {self.window_s / 3600:.1f} h" if self.window_s else ""
        cov = f"coverage {100 * self.coverage:.1f}%"
        if self.coverage_raw is not None and abs(self.band_scale - 1.0) > 1e-9:
            cov += f" (band ×{self.band_scale:.2f}, raw {100 * self.coverage_raw:.1f}%)"
        return (
            f"{self.field_name}: RMSE {self.rmse:.4g} {self.unit}{frac}, "
            f"bias {self.bias:+.4g}, {cov}, n={self.n}{win}"
        )


@dataclass
class TrackingReport:
    """Per-field tracking metrics plus the gate verdict and the alarm lead times."""

    metrics: list[FieldMetrics] = field(default_factory=list)
    gates: dict[str, Any] = field(default_factory=dict)
    breaches: list[str] = field(default_factory=list)
    alarm_lead_s: dict[str, float] = field(default_factory=dict)
    forecast_error_s: dict[str, float] = field(default_factory=dict)
    final_latent: dict[str, dict[str, Any]] = field(default_factory=dict)
    n_steps: int = 0

    def field(self, name: str) -> FieldMetrics:
        for m in self.metrics:
            if m.field_name == name:
                return m
        raise KeyError(f"no metrics for field {name!r}")

    @property
    def passed(self) -> bool:
        return not self.breaches

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "n_steps": self.n_steps,
            "gates": self.gates,
            "breaches": list(self.breaches),
            "alarm_lead_s": dict(self.alarm_lead_s),
            "forecast_error_s": dict(self.forecast_error_s),
            "final_latent": dict(self.final_latent),
            "metrics": [m.to_dict() for m in self.metrics],
        }

    def render(self) -> str:
        lines = [f"Tracking: {'PASS' if self.passed else 'FAIL'} over {self.n_steps} steps"]
        lines += [f"  {m.render()}" for m in self.metrics]
        if self.alarm_lead_s:
            for name, lead in self.alarm_lead_s.items():
                fe = self.forecast_error_s.get(name)
                extra = f", forecast landed {fe:+.0f} s from the crossing" if fe is not None else ""
                lines.append(f"  alarm {name}: {lead:+.0f} s ahead of the measured crossing{extra}")
        for b in self.breaches:
            lines.append(f"  GATE BREACH: {b}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The whole evidence bundle
# ---------------------------------------------------------------------------

@dataclass
class TwinEvidence:
    """Everything a commissioned twin produces, in one file, keyed by the spec digest."""

    asset_id: str
    spec_digest: str
    model_id: str
    created_unix: float = field(default_factory=time.time)
    preflight: CheckReport | None = None
    tracking: TrackingReport | None = None
    decisions: list[dict[str, Any]] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)
    initial_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: str = ""

    @property
    def commissioned(self) -> bool:
        """A twin is commissioned only if preflight passed *and* tracking cleared its gates."""
        return bool(
            self.preflight is not None
            and self.preflight.passed
            and self.tracking is not None
            and self.tracking.passed
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "spec_digest": self.spec_digest,
            "model_id": self.model_id,
            "created_unix": self.created_unix,
            "commissioned": self.commissioned,
            "preflight": self.preflight.to_dict() if self.preflight else None,
            "tracking": self.tracking.to_dict() if self.tracking else None,
            "decisions": list(self.decisions),
            "unverified": list(self.unverified),
            "initial_state": dict(self.initial_state),
            "notes": self.notes,
        }

    def save(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path

    def render(self) -> str:
        """A plain-text report: what passed, what the tracking error is, what was skipped."""
        head = [
            f"PINNeAPPle Twin — {self.asset_id}",
            f"  spec {self.spec_digest} · model {self.model_id}",
            f"  commissioned: {'YES' if self.commissioned else 'NO'}",
        ]
        parts = ["\n".join(head)]
        if self.preflight:
            parts.append(self.preflight.render())
        if self.tracking:
            parts.append(self.tracking.render())
        if self.decisions:
            parts.append("Decisions:\n" + "\n".join(f"  {d}" for d in self.decisions))
        if self.unverified:
            parts.append("Not verified:\n" + "\n".join(f"  - {u}" for u in self.unverified))
        if self.notes:
            parts.append(self.notes)
        return "\n\n".join(parts)

"""Core data structures for pinneapple_validate.

Provides ``CheckResult`` and ``ValidationReport`` — the fundamental containers
used by every check in the validation pipeline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List


@dataclass
class CheckResult:
    """Result of a single physics validation check.

    Attributes
    ----------
    name:
        Short identifier for the check (e.g. ``"mass_conservation"``).
    passed:
        Whether the check is considered passing given the threshold. ``None`` means the check **did not run** — the
        model does not return what it would take to ask (a plain tensor where a unit is needed, say). A check that
        could not be asked never claims a pass: the same rule the data gates follow.
    value:
        Numerical value computed by the check (error, residual, integral, …).
    threshold:
        Acceptance threshold.  The check passes when ``value <= threshold``
        (or as defined by the check implementation for signed quantities).
    description:
        Human-readable explanation of what was checked.
    unit:
        Optional unit string for the reported ``value`` (e.g. ``"kg/s"``).
    direction:
        Which side of the threshold passes: ``"lower"`` (the default: an error below the threshold is good) or
        ``"higher"`` (a fraction that has to reach the threshold, as in the out-of-distribution check).
    """

    name: str
    passed: bool | None
    value: float
    threshold: float
    description: str
    unit: str = ""
    direction: str = "lower"

    @property
    def status(self) -> str:
        """``"passed"``, ``"failed"`` or ``"not run"`` — one word a report can be read by."""
        if self.passed is None:
            return "not run"
        return "passed" if self.passed else "failed"

    @property
    def ran(self) -> bool:
        return self.passed is not None

    @classmethod
    def not_run(cls, name: str, description: str, *, threshold: float = 0.0, unit: str = "") -> CheckResult:
        """A check that could not be asked, with the reason in ``description``."""
        return cls(name=name, passed=None, value=float("nan"), threshold=threshold, description=description,
                   unit=unit)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        status = {"passed": "PASS", "failed": "FAIL", "not run": "SKIP"}[self.status]
        unit_str = f" {self.unit}" if self.unit else ""
        return (
            f"[{status}] {self.name}: {self.value:.4e}{unit_str}"
            f" (threshold={self.threshold:.4e}) — {self.description}"
        )


@dataclass
class ValidationReport:
    """Aggregated result of a full physics validation run.

    Attributes
    ----------
    model_name:
        Identifier of the model that was validated.
    timestamp:
        ISO-8601 timestamp of when the report was created (UTC).
    checks:
        Ordered list of individual :class:`CheckResult` objects.
    """

    model_name: str
    timestamp: str
    checks: List[CheckResult] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, model_name: str) -> ValidationReport:
        """Return a new report stamped with the current UTC time."""
        ts = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        return cls(model_name=model_name, timestamp=ts)

    # ------------------------------------------------------------------
    # Aggregate properties
    # ------------------------------------------------------------------

    @property
    def passed(self) -> bool:
        """``True`` when at least one check ran and every check that ran passed.

        A report whose checks all declined to run is **not** a pass: nothing was verified."""
        ran = [c for c in self.checks if c.ran]
        return bool(ran) and all(c.passed for c in ran)

    @property
    def n_passed(self) -> int:
        """Number of checks that passed."""
        return sum(1 for c in self.checks if c.passed is True)

    @property
    def n_failed(self) -> int:
        """Number of checks that failed."""
        return sum(1 for c in self.checks if c.passed is False)

    @property
    def n_not_run(self) -> int:
        """Number of checks that could not be asked."""
        return sum(1 for c in self.checks if c.passed is None)

    def failures(self) -> list[CheckResult]:
        """The checks that failed (not the ones that did not run)."""
        return [c for c in self.checks if c.passed is False]

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a pretty-printed table of all check results."""
        lines: List[str] = []
        sep = "-" * 72
        lines.append(sep)
        lines.append(
            f"Validation Report  |  model: {self.model_name}"
            f"  |  {self.timestamp}"
        )
        lines.append(sep)

        # Column widths
        name_w = max((len(c.name) for c in self.checks), default=4) + 2
        val_w = 14
        thr_w = 14

        header = (
            f"{'Check':<{name_w}}  {'Status':<6}  "
            f"{'Value':>{val_w}}  {'Threshold':>{thr_w}}  Description"
        )
        lines.append(header)
        lines.append(sep)

        for c in self.checks:
            status = {"passed": "PASS", "failed": "FAIL", "not run": "SKIP"}[c.status]
            unit_str = f" {c.unit}" if c.unit else ""
            val_str = "—" if not c.ran else f"{c.value:.4e}{unit_str}"
            thr_str = f"{c.threshold:.4e}"
            lines.append(
                f"{c.name:<{name_w}}  {status:<6}  "
                f"{val_str:>{val_w + len(unit_str)}}  {thr_str:>{thr_w}}  "
                f"{c.description}"
            )

        lines.append(sep)
        overall = "PASSED" if self.passed else "FAILED"
        skipped = f", {self.n_not_run} not run" if self.n_not_run else ""
        lines.append(
            f"Overall: {overall}  ({self.n_passed}/{len(self.checks)} checks passed{skipped})"
        )
        lines.append(sep)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict representation."""
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "passed": self.passed,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "n_not_run": self.n_not_run,
            "checks": [asdict(c) for c in self.checks],
        }

    def save(self, path: str) -> None:
        """Persist the report to *path* as a JSON file.

        Parameters
        ----------
        path:
            File-system path (with or without ``.json`` extension).
        """
        if not path.endswith(".json"):
            path = path + ".json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

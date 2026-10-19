"""The detection matrix: every check against every fault and magnitude, and against clean cases.

``run_fault_injection(problem)`` draws correct cases from the problem's verified solver (with the variation a
threshold must tolerate: coefficient contrast, feature size, amplitude, solver resolution), injects every fault at
every magnitude and records each check's value. From the values:

* the detection rate of each check (and of all checks together) per fault and magnitude;
* the false-alarm rate of each check on the clean cases;
* the minimal magnitude each check detects (detection rate >= ``min_rate``, default 90 %);
* the ROC AUC of each check per fault and magnitude (clean values against faulty values);
* the true error each fault causes (relative L2 against the correct field).

The report renders as Markdown (the "check power" section of a validation report) and as JSON.
"""
from __future__ import annotations

import json
import math
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .checks import dataset_checks, score
from .faults import DATA_FAULTS, MODEL_FAULTS, Fault
from .problems import BenchmarkProblem, Dataset, get_problem

__all__ = ["DetectionMatrix", "FaultInjectionReport", "run_fault_injection", "auc", "fault_rng"]


def auc(clean: Sequence[float], faulty: Sequence[float]) -> float:
    """ROC AUC of a score separating ``faulty`` (positives) from ``clean``: P(faulty > clean) + P(tie) / 2
    (Mann-Whitney). 0.5 = no better than chance, 1 = perfect separation."""
    c = np.asarray(clean, float)
    f = np.asarray(faulty, float)
    if not c.size or not f.size:
        return math.nan
    c = np.where(np.isfinite(c), c, np.finfo(float).max)
    f = np.where(np.isfinite(f), f, np.finfo(float).max)
    gt = (f[:, None] > c[None, :]).mean()
    eq = (f[:, None] == c[None, :]).mean()
    return float(gt + 0.5 * eq)


@dataclass
class DetectionMatrix:
    """Check values on clean inputs and on every (fault, magnitude); ``level`` is ``"model"`` (one case per
    trial) or ``"data"`` (one dataset per trial)."""
    problem: str
    level: str
    tolerances: dict[str, float]
    clean: dict[str, np.ndarray]
    scores: dict[tuple[str, float], dict[str, np.ndarray]]
    errors: dict[tuple[str, float], np.ndarray]
    faults: dict[str, Fault]
    min_rate: float = 0.9
    trials: list[tuple[int, dict[str, float]]] = field(default_factory=list)   # (seed, variation) of each clean case

    def with_tolerances(self, tolerances: dict[str, float]) -> DetectionMatrix:
        """The same values judged with other tolerances (a subset of the checks, e.g. a calibrated defence)."""
        return DetectionMatrix(self.problem, self.level, dict(tolerances),
                               {c: self.clean[c] for c in tolerances},
                               {k: {c: v[c] for c in tolerances} for k, v in self.scores.items()}, self.errors,
                               self.faults, self.min_rate, self.trials)

    @property
    def checks(self) -> list[str]:
        return list(self.tolerances)

    def _flags(self, values: dict[str, np.ndarray], check: str | None) -> np.ndarray:
        names = self.checks if check is None else [check]
        return np.any([values[n] > self.tolerances[n] for n in names], axis=0)

    def detection_rate(self, fault: str, magnitude: float, check: str | None = None) -> float:
        """Fraction of trials flagged (``check=None``: by any check)."""
        return float(self._flags(self.scores[(fault, magnitude)], check).mean())

    def false_alarm_rate(self, check: str | None = None) -> float:
        """Fraction of clean trials flagged (``check=None``: by any check)."""
        return float(self._flags(self.clean, check).mean())

    def minimal_magnitude(self, fault: str, check: str | None = None) -> float | None:
        """Smallest magnitude detected in at least ``min_rate`` of the trials (None: not at any magnitude)."""
        for m in self.faults[fault].magnitudes:
            if self.detection_rate(fault, m, check) >= self.min_rate:
                return m
        return None

    def auc(self, fault: str, magnitude: float, check: str) -> float:
        return auc(self.clean[check], self.scores[(fault, magnitude)][check])

    def caught_at_documented(self, fault: str) -> list[str]:
        """Checks that detect ``fault`` at its documented magnitude in at least ``min_rate`` of the trials."""
        m = self.faults[fault].documented_for(self.problem)
        return [c for c in self.checks if self.detection_rate(fault, m, c) >= self.min_rate]

    def true_error(self, fault: str, magnitude: float) -> float:
        """Median true relative error of the faulty trials."""
        return float(np.median(self.errors[(fault, magnitude)]))

    def failures(self) -> list[str]:
        """Acceptance (#697): faults not caught at their documented magnitude, checks with > 1 % false alarms."""
        out = [f"{self.level} fault {f!r} not caught by any check at magnitude "
               f"{self.faults[f].documented_for(self.problem):g}"
               for f in self.faults if not self.caught_at_documented(f)]
        out += [f"check {c!r}: false-alarm rate {self.false_alarm_rate(c):.1%} > 1 %" for c in self.checks
                if self.false_alarm_rate(c) > 0.01]
        return out

    def to_dict(self) -> dict[str, Any]:
        rows = []
        for f, m in self.scores:
            rows.append({"fault": f, "magnitude": m, "true_error": self.true_error(f, m),
                         "detection": {c: self.detection_rate(f, m, c) for c in self.checks},
                         "detection_any": self.detection_rate(f, m),
                         "auc": {c: self.auc(f, m, c) for c in self.checks}})
        return {
            "problem": self.problem, "level": self.level, "min_rate": self.min_rate,
            "n_clean": int(len(next(iter(self.clean.values())))) if self.clean else 0,
            "tolerances": self.tolerances,
            "false_alarm_rate": {c: self.false_alarm_rate(c) for c in self.checks},
            "false_alarm_any": self.false_alarm_rate(),
            "faults": {f: {"documented": F.documented_for(self.problem), "unit": F.unit, "description": F.description,
                           "caught_at_documented_by": self.caught_at_documented(f),
                           "minimal_magnitude": {c: self.minimal_magnitude(f, c) for c in self.checks},
                           "minimal_magnitude_any": self.minimal_magnitude(f)} for f, F in self.faults.items()},
            "rows": rows,
        }

    def to_markdown(self) -> str:
        short = {c: f"C{i + 1}" for i, c in enumerate(self.checks)}
        n_clean = len(next(iter(self.clean.values())))
        lines = [f"**Checks ({self.level} level, {self.problem}):** "
                 + "; ".join(f"{short[c]} = {c} (tol {self.tolerances[c]:.3g})" for c in self.checks), "",
                 "| fault | documented | caught at documented by | min. detectable (any check) | "
                 + " | ".join(f"min. {short[c]}" for c in self.checks) + " |",
                 "|---|---|---|---|" + "---|" * len(self.checks)]
        for f, F in self.faults.items():
            caught = ", ".join(short[c] for c in self.caught_at_documented(f)) or "**none**"
            mm = self.minimal_magnitude(f)
            cells = [("—" if (v := self.minimal_magnitude(f, c)) is None else f"{v:g}") for c in self.checks]
            lines.append(f"| {f} | {F.documented_for(self.problem):g} ({F.unit}) | {caught} | {'—' if mm is None else f'{mm:g}'} | "
                         + " | ".join(cells) + " |")
        lines += ["", f"False-alarm rate on {n_clean} clean {'cases' if self.level == 'model' else 'datasets'}: "
                  + "; ".join(f"{short[c]} {self.false_alarm_rate(c):.1%}" for c in self.checks)
                  + f"; any check {self.false_alarm_rate():.1%}.", "",
                  "| fault | magnitude | true error (median) | any | "
                  + " | ".join(f"{short[c]} rate / AUC" for c in self.checks) + " |",
                  "|---|---|---|---|" + "---|" * len(self.checks)]
        for (f, m) in self.scores:
            cells = [f"{self.detection_rate(f, m, c):.0%} / {self.auc(f, m, c):.2f}" for c in self.checks]
            lines.append(f"| {f} | {m:g} | {self.true_error(f, m):.3g} | {self.detection_rate(f, m):.0%} | "
                         + " | ".join(cells) + " |")
        return "\n".join(lines)


@dataclass
class FaultInjectionReport:
    problem: str
    model: DetectionMatrix
    data: DetectionMatrix | None
    settings: dict[str, Any] = field(default_factory=dict)

    def failures(self) -> list[str]:
        return self.model.failures() + (self.data.failures() if self.data is not None else [])

    @property
    def passed(self) -> bool:
        return not self.failures()

    def check_power(self) -> dict[str, Any]:
        """The "check power" section of a validation report: per check, its false-alarm rate and the minimal
        magnitude of each fault it detects."""
        out: dict[str, Any] = {"problem": self.problem, "min_rate": self.model.min_rate, "checks": {}}
        for mat in (self.model, self.data):
            if mat is None:
                continue
            for c in mat.checks:
                out["checks"][c] = {"level": mat.level, "tolerance": mat.tolerances[c],
                                    "false_alarm_rate": mat.false_alarm_rate(c),
                                    "detects": {f: mat.minimal_magnitude(f, c) for f in mat.faults}}
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"problem": self.problem, "settings": self.settings, "model": self.model.to_dict(),
                "data": None if self.data is None else self.data.to_dict(), "failures": self.failures()}

    def to_markdown(self) -> str:
        s = self.settings
        head = (f"### Check power: {self.problem}\n\n{s.get('n_cases')} clean cases (seed {s.get('seed')}), "
                f"{s.get('n_datasets')} datasets of {s.get('dataset_size')} samples; detected = flagged in at "
                f"least {self.model.min_rate:.0%} of the trials.\n\n#### Model faults\n\n")
        body = self.model.to_markdown()
        if self.data is not None:
            body += "\n\n#### Data faults\n\n" + self.data.to_markdown()
        fails = self.failures()
        tail = "\n\n**Acceptance:** " + ("every fault caught at its documented magnitude, false alarms <= 1 % per "
                                         "check." if not fails else "; ".join(fails))
        return head + body + tail + "\n"

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".md":
            path.write_text(self.to_markdown(), encoding="utf-8")
        else:
            path.write_text(json.dumps(self.to_dict(), indent=1, default=float), encoding="utf-8")
        return path


def _values(checks, items) -> dict[str, np.ndarray]:
    return {c.name: np.array([score(c, it) for it in items]) for c in checks}


def _ds(check, ds: Dataset) -> float:
    try:
        v = float(check.fn(None, ds))
    except Exception:  # noqa: BLE001 - a crash on a dataset is a flag
        return float("inf")
    return v if np.isfinite(v) else float("inf")


def fault_rng(seed: int, fault: str, magnitude: float, case_seed: int) -> np.random.Generator:
    """The random stream of one fault injection (so a single trial can be replayed)."""
    return np.random.default_rng([int(seed), zlib.crc32(fault.encode()), int(round(magnitude * 1e6)),
                                  int(case_seed) % (2 ** 32)])


def run_fault_injection(problem: str | BenchmarkProblem = "darcy_2d", *, checks: Sequence | None = None,
                        data_checks: Sequence | None = None, model_faults: Sequence[Fault] | None = None,
                        data_faults: Sequence[Fault] | None = None, n_cases: int = 50, n_datasets: int = 20,
                        dataset_size: tuple[int, int] = (16, 8), seed: int = 0, min_rate: float = 0.9,
                        include_data: bool = True, **problem_options: Any) -> FaultInjectionReport:
    """Build the detection matrix of ``checks`` (default: the problem's ``check_library``) against the fault
    library on ``problem`` (``"darcy_2d"``, ``"burgers_1d"`` or a :class:`BenchmarkProblem`).

    Clean cases are drawn with ``sample_varied`` (correct but unusual); every model fault is injected into each of
    them at every magnitude (seeded). Data faults are injected into ``n_datasets`` datasets of ``dataset_size``
    (train, test) correct samples and scored with ``data_checks`` (default: ``dataset_checks``).
    """
    from .checks import check_library
    p = get_problem(problem, **problem_options)
    checks = list(checks) if checks is not None else list(check_library(p).values())
    rng = np.random.default_rng(seed)
    clean = p.sample_varied(n_cases, seed=seed)
    faults = list(model_faults) if model_faults is not None else list(MODEL_FAULTS)
    scores, errors = {}, {}
    for F in faults:
        for m in F.magnitudes:
            pairs = [F.apply(p, c, m, fault_rng(seed, F.name, m, c.seed)) for c in clean]
            scores[(F.name, m)] = _values(checks, [fc for fc, _ in pairs])
            errors[(F.name, m)] = np.array([p.true_error(fc, ref) for fc, ref in pairs])
    model = DetectionMatrix(p.name, "model", {c.name: float(c.tolerance) for c in checks}, _values(checks, clean),
                            scores, errors, {F.name: F for F in faults}, min_rate,
                            trials=[(c.seed, dict(c.variation)) for c in clean])
    data = None
    if include_data:
        dchecks = list(data_checks) if data_checks is not None else dataset_checks(p)
        n_tr, n_te = dataset_size
        pool = p.sample_varied(max(n_tr + n_te, 2 * (n_tr + n_te)), seed=seed + 1)
        sets = []
        for _ in range(n_datasets):
            idx = rng.permutation(len(pool))[:n_tr + n_te]
            sets.append(Dataset([pool[i] for i in idx[:n_tr]], [pool[i] for i in idx[n_tr:]]))
        dfaults = list(data_faults) if data_faults is not None else list(DATA_FAULTS)
        dscores, derrors = {}, {}
        for F in dfaults:
            for m in F.magnitudes:
                frng = np.random.default_rng([seed, zlib.crc32(F.name.encode()), int(m * 1e6), 1])
                pairs = [F.apply(p, ds, m, frng) for ds in sets]
                dscores[(F.name, m)] = {c.name: np.array([_ds(c, d) for d, _ in pairs]) for c in dchecks}
                derrors[(F.name, m)] = np.array([e for _, e in pairs])
        data = DetectionMatrix(p.name, "data", {c.name: float(c.tolerance) for c in dchecks},
                               {c.name: np.array([_ds(c, d) for d in sets]) for c in dchecks}, dscores, derrors,
                               {F.name: F for F in dfaults}, min_rate)
    settings = {"n_cases": n_cases, "n_datasets": n_datasets if include_data else 0,
                "dataset_size": list(dataset_size), "seed": seed, "options": dict(p.options)}
    return FaultInjectionReport(p.name, model, data, settings)

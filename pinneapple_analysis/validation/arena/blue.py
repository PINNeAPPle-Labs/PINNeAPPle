"""Blue team: calibrated thresholds with a stated false-alarm rate, one combined test, and hardening.

**Defence.** A set of checks with a threshold each; a case is rejected when any check exceeds its threshold. Round
0 of the arena is the hand-set tolerances of the checks (``Defence.from_checks``).

**Calibration** (``calibrate``). The checks are combined into one statistic so that false alarms do not grow with
the number of checks. With the clean calibration cases split in two halves A and B:

1. each check's scale ``s_j`` is the split-conformal quantile of its values on A
   (``pinneapple_analysis.uncertainty.conformal.conformal_quantile``), at least 1 % of the check's own tolerance;
2. the combined statistic of a case is ``T = max_j value_j / s_j``;
3. its threshold ``q`` is the split-conformal quantile of ``T`` on B at level ``alpha_cal``.

``T`` is a fixed function of a case once A is used, so for a new clean case exchangeable with B,
``P(T > q) <= alpha_cal`` (Vovk, Gammerman & Shafer 2005; Angelopoulos & Bates 2022, arXiv:2107.07511) for the
whole family of checks at once: the family-wise false-alarm rate. The guarantee needs
``ceil((n_B + 1)(1 - alpha_cal)) <= n_B``; ``Defence.alpha_guaranteed`` says whether that holds. ``alpha_cal`` is
``safety * alpha`` (default ``safety = 0.5``) so that the rate measured on a finite held-out set stays below the stated
``alpha``. The combined test is the same as per-check thresholds ``q s_j``, which the reports list.

**Hardening** (``harden``). Stored false alarms are correct cases: ``q`` is raised just above their statistic (this
only lowers the false-alarm rate). Then, while a stored escape passes, the check of the library that catches the most
remaining escapes (after recalibration) is added; the log says which checks were added and what each one caught.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..fault_injection.checks import score
from ..fault_injection.problems import Case

__all__ = ["Defence", "calibrate", "calibrate_scores", "harden", "score_matrix"]


@dataclass
class Defence:
    """Checks with a threshold each (reject when any value exceeds its threshold)."""
    checks: list
    thresholds: dict[str, float]
    kind: str = "hand-set tolerances"
    alpha: float | None = None
    alpha_cal: float | None = None
    scales: dict[str, float] | None = None
    statistic_threshold: float | None = None
    n_calibration: int = 0
    alpha_guaranteed: bool | None = None
    log: list[str] = field(default_factory=list)

    @classmethod
    def from_checks(cls, checks: Sequence) -> Defence:
        return cls(list(checks), {c.name: float(c.tolerance) for c in checks})

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.checks]

    def scores(self, case: Case) -> dict[str, float]:
        return {c.name: score(c, case) for c in self.checks}

    def margins(self, values: Mapping[str, float]) -> dict[str, float]:
        """Per check ``value / threshold - 1`` (> 0 rejects); a zero threshold gives ``1 + value`` above it."""
        out = {}
        for n, t in self.thresholds.items():
            v = float(values[n])
            if not math.isfinite(v):
                out[n] = math.inf
            elif t > 0:
                out[n] = v / t - 1.0
            else:
                out[n] = 1.0 + v if v > t else -1.0
        return out

    def rejects_values(self, values: Mapping[str, float]) -> bool:
        return any(float(values[n]) > t or not math.isfinite(float(values[n])) for n, t in self.thresholds.items())

    def rejects(self, case: Case) -> bool:
        return self.rejects_values(self.scores(case))

    def statistic(self, values: Mapping[str, float]) -> float:
        """The combined statistic ``max_j value_j / scale_j`` (calibrated defences only)."""
        if not self.scales:
            raise ValueError("not a calibrated defence")
        return _stat(np.array([[values[n] for n in self.names]]), [self.scales[n] for n in self.names])[0]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "checks": self.names, "thresholds": self.thresholds, "alpha": self.alpha,
                "alpha_cal": self.alpha_cal, "scales": self.scales, "statistic_threshold": self.statistic_threshold,
                "n_calibration": self.n_calibration, "alpha_guaranteed": self.alpha_guaranteed, "log": self.log}

    def describe(self) -> str:
        head = (f"{self.kind}: " if self.alpha is None else
                f"{self.kind} (alpha = {self.alpha:g}, calibrated at {self.alpha_cal:g} on {self.n_calibration} clean "
                f"cases{'' if self.alpha_guaranteed else ', finite-sample guarantee NOT met'}): ")
        return head + "; ".join(f"{n} <= {t:.3g}" for n, t in self.thresholds.items())


def score_matrix(checks: Sequence, cases: Sequence[Case]) -> dict[str, np.ndarray]:
    """``{check name: values on cases}``."""
    return {c.name: np.array([score(c, x) for x in cases]) for c in checks}


def _stat(V: np.ndarray, scales: Sequence[float]) -> np.ndarray:
    s = np.asarray(scales, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(s > 0, V / np.where(s > 0, s, 1.0), np.where(V > 0, np.inf, 0.0))
    R = np.where(np.isfinite(V), R, np.inf)
    return R.max(axis=1)


def calibrate_scores(checks: Sequence, values: Mapping[str, np.ndarray], alpha: float = 0.01, *,
                     safety: float = 0.5, seed: int = 0, false_alarms: Mapping[str, np.ndarray] | None = None,
                     kind: str = "split-conformal combined test", floor: float = 0.01) -> Defence:
    """``calibrate`` from precomputed values (``{check: values on the clean calibration cases}``);
    ``false_alarms``: values on stored false alarms, which must stay below the threshold."""
    from pinneapple_analysis.uncertainty.conformal import (
        conformal_quantile,
        conformal_quantile_valid,
    )
    names = [c.name for c in checks]
    V = np.stack([np.asarray(values[n], float) for n in names], axis=1)
    n = len(V)
    if n < 4:
        raise ValueError("calibration needs at least 4 clean cases")
    perm = np.random.default_rng(seed).permutation(n)
    A, B = V[perm[: n // 2]], V[perm[n // 2:]]
    alpha_cal = float(alpha) * float(safety)
    scales = []
    for j in range(len(names)):
        col = A[:, j][np.isfinite(A[:, j])]
        q_j = float(conformal_quantile(col, alpha_cal)) if col.size else math.inf
        # floor: 1 % of the check's own tolerance, so a check that is exactly 0 on most clean cases (an envelope,
        # a bound) is not judged on its rare round-off values; a fixed function of A, the guarantee is unchanged
        scales.append(max(q_j, floor * float(getattr(checks[j], "tolerance", 0.0) or 0.0)))
    T = _stat(B, scales)
    q = float(conformal_quantile(np.where(np.isfinite(T), T, np.finfo(float).max), alpha_cal))
    log = []
    if false_alarms is not None and len(next(iter(false_alarms.values()), [])):
        F = np.stack([np.asarray(false_alarms[n], float) for n in names], axis=1)
        tf = float(_stat(F, scales).max())
        if math.isfinite(tf) and tf >= q:
            log.append(f"threshold raised from {q:.4g} to {tf:.4g} so that {len(F)} stored false alarm(s) pass")
            q = float(np.nextafter(tf, math.inf)) * (1 + 1e-9)
    thresholds = {nm: (q * s if math.isfinite(s) else math.inf) for nm, s in zip(names, scales, strict=True)}
    return Defence(list(checks), thresholds, kind, float(alpha), alpha_cal, dict(zip(names, scales, strict=True)),
                   q, n, conformal_quantile_valid(len(B), alpha_cal), log)


def calibrate(checks: Sequence, clean_cases: Sequence[Case], alpha: float = 0.01, *, safety: float = 0.5,
              seed: int = 0, false_alarms: Sequence[Case] = ()) -> Defence:
    """Split-conformal combined test of ``checks`` on ``clean_cases`` (see the module docstring)."""
    fa = score_matrix(checks, false_alarms) if false_alarms else None
    return calibrate_scores(checks, score_matrix(checks, clean_cases), alpha, safety=safety, seed=seed,
                            false_alarms=fa)


def harden(current: Sequence, library: Mapping[str, Any], clean_cases: Sequence[Case], *, escapes: Sequence[Case] = (),
           false_alarms: Sequence[Case] = (), alpha: float = 0.01, safety: float = 0.5, seed: int = 0,
           clean_values: Mapping[str, np.ndarray] | None = None) -> Defence:
    """Calibrate ``current`` and add checks from ``library`` until every stored escape is rejected (or no check
    helps); stored false alarms always pass. ``clean_values`` caches the library's values on ``clean_cases``."""
    allc = {**{c.name: c for c in current}, **dict(library)}
    cv = dict(clean_values or {})
    for nm, c in allc.items():
        if nm not in cv:
            cv[nm] = np.array([score(c, x) for x in clean_cases])
    ev = {nm: np.array([score(c, x) for x in escapes]) for nm, c in allc.items()}
    fv = {nm: np.array([score(c, x) for x in false_alarms]) for nm, c in allc.items()} if false_alarms else None

    def build(names: list[str]) -> tuple[Defence, int]:
        chk = [allc[n] for n in names]
        d = calibrate_scores(chk, {n: cv[n] for n in names}, alpha, safety=safety, seed=seed,
                             false_alarms=None if fv is None else {n: fv[n] for n in names})
        caught = sum(d.rejects_values({n: ev[n][i] for n in names}) for i in range(len(escapes)))
        return d, caught

    names = [c.name for c in current]
    d, caught = build(names)
    log = [f"calibrated {', '.join(names)}: {caught}/{len(escapes)} stored escape(s) rejected"] + d.log
    while caught < len(escapes):
        best = None
        for nm in allc:
            if nm in names:
                continue
            d2, c2 = build([*names, nm])
            if c2 > caught and (best is None or c2 > best[2]):
                best = (nm, d2, c2)
        if best is None:
            log.append(f"no check of the library rejects the {len(escapes) - caught} remaining escape(s)")
            break
        names.append(best[0])
        d, caught = best[1], best[2]
        log.append(f"added '{best[0]}': {caught}/{len(escapes)} stored escape(s) rejected")
    d.kind = "split-conformal combined test (hardened)"
    d.log = log + [x for x in d.log if x not in log]
    return d

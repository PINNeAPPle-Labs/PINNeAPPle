"""What the data can recover, with an interval whose coverage is measured (#77).

A fitted parameter without an interval is a number with no claim attached, and an interval nobody has checked is a
claim with no evidence attached. This does both halves:

    >>> from pinneapple_analysis.inverse_problems.recovery import recover, coverage_study
    >>> result = recover(model, params, observations, level=0.95)     # doctest: +SKIP
    >>> result.interval("k").estimate, result.interval("k").lower, result.interval("k").upper   # doctest: +SKIP
    (0.81, 0.74, 0.89)

The interval is the linearised (Gauss-Newton) one from the Fisher information at the optimum, which is the interval the
data supports: ``J^T J`` with the noise the observations declare, plus the prior precision when a prior is given. The
same matrix answers the other question — **is this parameter identifiable at all?** — through
:class:`~pinneapple_analysis.inverse_problems.sensitivity.IdentifiabilityAnalyzer`, and a parameter the data cannot
separate from another is reported by name instead of being given a confident-looking interval.

``coverage_study`` is how the interval is checked: repeat a synthetic experiment with a known truth, count how often the
interval contains it, and compare with the level that was claimed. An interval that contains the truth 70 % of the time
while claiming 95 % is not a wide interval, it is a wrong one.

Both halves reuse what is already here: the standard deviation and the null space come from the same Fisher information
and the same reader that :mod:`~pinneapple_analysis.inverse_problems.identifiability` uses (:func:`_std_from_F`,
:func:`_combination`), so the two cannot describe the same direction differently. ``interval="profile"`` swaps the
linearised interval for the profile-likelihood one, which is the honest choice when the model is visibly nonlinear.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .estimation import FitResult, Prior, least_squares_fit
from .forward import ForwardModel, Observations, ParameterSet
from .identifiability import _combination, _std_from_F, profile_likelihood
from .sensitivity import IdentifiabilityAnalyzer, IdentifiabilityResult

__all__ = ["CoverageReport", "ParameterInterval", "RecoveryResult", "coverage_study", "recover"]


def _z_quantile(level: float) -> float:
    from scipy.stats import norm

    if not 0.0 < level < 1.0:
        raise ValueError(f"level has to be between 0 and 1, got {level}")
    return float(norm.ppf(0.5 + level / 2.0))


def _dead_directions(analysis: IdentifiabilityResult, names: Sequence[str], log: Sequence[bool]) -> list[dict]:
    """The parameter combinations the data does not constrain, each named by the parameters that carry it.

    ``IdentifiabilityResult.unidentifiable_indices`` indexes **eigen-directions**, not parameters: the direction is a
    combination, and reporting it as one parameter would blame whichever parameter happens to share its index. So the
    eigenvector is read instead, and the combination is described the same way the identifiability report describes it.
    """
    out: list[dict] = []
    for i in analysis.unidentifiable_indices:
        v = analysis.eigenvectors[:, i]
        if not np.any(v):
            continue
        info = _combination(list(names), list(log), v)
        if info["parameters"] and info["parameters"] not in [o["parameters"] for o in out]:
            out.append(info)
    return out


@dataclass(frozen=True)
class ParameterInterval:
    """One recovered parameter with the interval the data supports at ``level``."""

    name: str
    estimate: float
    lower: float
    upper: float
    relative_std: float
    level: float

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def contains(self, value: float) -> bool:
        return self.lower <= float(value) <= self.upper

    def __str__(self) -> str:
        return f"{self.name} = {self.estimate:.6g} [{self.lower:.6g}, {self.upper:.6g}] ({self.level:.0%})"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "estimate": self.estimate, "lower": self.lower, "upper": self.upper,
                "relative_std": self.relative_std, "level": self.level}


@dataclass
class RecoveryResult:
    """The fit, the intervals, and what could not be recovered — the three things a discovery report needs."""

    fit: FitResult
    intervals: list[ParameterInterval]
    identifiability: IdentifiabilityResult
    interval_kind: str = "linear"
    dead: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings

    @property
    def unidentifiable(self) -> list[str]:
        return [name for group in self.dead for name in group["parameters"]]

    def interval(self, name: str) -> ParameterInterval:
        return next(i for i in self.intervals if i.name == name)

    def contains(self, truth: Mapping[str, float]) -> dict[str, bool]:
        return {i.name: i.contains(truth[i.name]) for i in self.intervals if i.name in truth}

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "chi2": self.fit.chi2, "theta": dict(self.fit.theta), "interval": self.interval_kind,
                "intervals": [i.to_dict() for i in self.intervals], "unidentifiable": self.dead,
                "condition_number": self.identifiability.condition_number, "warnings": list(self.warnings)}

    def to_markdown(self) -> str:
        rows = ["| parameter | estimate | interval | relative std |", "|---|---|---|---|"]
        for i in self.intervals:
            std = "unbounded" if not np.isfinite(i.relative_std) else f"{i.relative_std:.1%}"
            span = f"[{i.lower:.6g}, {i.upper:.6g}]" if np.isfinite(i.width) else "&minus;&infin; to &plus;&infin;"
            rows.append(f"| `{i.name}` | {i.estimate:.6g} | {span} | {std} |")
        head = f"**recovered at {self.intervals[0].level:.0%}** — chi² = {self.fit.chi2:.4g}" if self.intervals else ""
        body = "\n".join(rows)
        if self.warnings:
            body += "\n\n**Problems:**\n" + "\n".join(f"- {w}" for w in self.warnings)
        return f"{head}\n\n{body}"

    def __str__(self) -> str:
        lines = [str(i) for i in self.intervals]
        lines += [f"warning: {w}" for w in self.warnings]
        return "\n".join(lines)


def recover(model: ForwardModel, params: ParameterSet, obs: Observations, *, level: float = 0.95,
            prior: Prior | None = None, free: Sequence[int] | None = None, interval: str = "linear",
            analyzer: IdentifiabilityAnalyzer | None = None, **fit_kwargs: Any) -> RecoveryResult:
    """Fit the free parameters and say, for each one, the interval the data supports and whether it is identifiable.

    ``level`` is the two-sided coverage of the interval. ``interval="linear"`` is the Gauss-Newton one — fast, and the
    right shape when the model is close to linear over a standard deviation; ``"profile"`` re-fits the others over a
    grid and takes the ``chi2_min + chi2_1(level)`` crossing, which stays honest when it is not. A parameter the data
    cannot separate from another gets an infinite interval rather than a confident-looking one.
    """
    q = _z_quantile(level)
    if interval not in ("linear", "profile"):
        raise ValueError(f"interval is 'linear' or 'profile', got {interval!r}")
    if interval == "profile" and abs(level - 0.95) > 1e-9:
        raise ValueError("the profile interval is the 95 % chi-square crossing (chi2_1 = 3.84): "
                         "use level=0.95 with interval='profile', or interval='linear' for another level")
    fit = least_squares_fit(model, params, obs, prior=prior, free=free, **fit_kwargs)
    weighted = fit.jac / obs.sigma.reshape(-1, 1)                     # d(pred/sigma)/dz at the optimum
    precision = weighted.T @ weighted
    if prior is not None:
        precision = precision + prior.precision(len(fit.z))
    analyzer = analyzer or IdentifiabilityAnalyzer()
    identifiability = analyzer.analyze(precision, list(params.names))
    std = _std_from_F(precision, analyzer.tol)
    dead = _dead_directions(identifiability, params.names, [p.log for p in params.free])

    profiles = profile_likelihood(model, params, obs, fit.z, fit.chi2, std, fisher=precision) \
        if interval == "profile" else None

    intervals: list[ParameterInterval] = []
    warnings: list[str] = []
    for k, parameter in enumerate(params.free):
        z = float(fit.z[k])
        if profiles is not None:
            lo95, hi95 = profiles[parameter.name]["ci95"]
            low = -math.inf if lo95 is None else float(lo95)
            high = math.inf if hi95 is None else float(hi95)
        else:
            low = parameter.from_z(z - q * float(std[k]))
            high = parameter.from_z(z + q * float(std[k]))
        relative = abs(float(std[k])) if not parameter.log else abs(math.expm1(float(std[k])))
        intervals.append(ParameterInterval(parameter.name, parameter.from_z(z), min(low, high), max(low, high),
                                           relative, level))
    for group in dead:
        warnings.append(group["text"] + "; the intervals below are the data's way of saying so")
    for name in fit.at_bound:
        warnings.append(f"'{name}' ended on its bound: the data wants a value outside the bounds it was given")
    if len(params) > 1 and identifiability.condition_number > 1e8 and not dead:
        warnings.append(f"the parameters are nearly degenerate (condition number "
                        f"{identifiability.condition_number:.1e}): their intervals are wide and correlated")
    return RecoveryResult(fit=fit, intervals=intervals, identifiability=identifiability, interval_kind=interval,
                          dead=dead, warnings=warnings)


@dataclass
class CoverageReport:
    """How often the interval contained the truth, per parameter, against the level that was claimed."""

    level: float
    n: int
    covered: dict[str, int]
    width: dict[str, float]
    tolerance: float = 0.1

    def rate(self, name: str) -> float:
        return self.covered[name] / self.n if self.n else 0.0

    def ci(self, name: str) -> tuple[float, float]:
        """Wilson interval of the coverage itself, so a rate of 0.93 from 20 runs is not read as 0.93."""
        k, n = self.covered[name], self.n
        if not n:
            return (0.0, 1.0)
        z = 1.96
        p = min(max(k / n, 0.0), 1.0)
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return (max(0.0, centre - half), min(1.0, centre + half))

    @property
    def calibrations(self) -> dict[str, str]:
        """``under`` when the interval covers less than claimed, ``over`` when it is needlessly wide, else ``ok``.

        A parameter whose interval is infinite is ``unbounded``: it covers the truth every time, and that is not
        calibration, it is the width saying the data constrain nothing.
        """
        out = {}
        for name in self.covered:
            if not np.isfinite(self.width[name]):
                out[name] = "unbounded"
                continue
            low, high = self.ci(name)
            if high < self.level - self.tolerance:
                out[name] = "under"
            elif low > self.level + self.tolerance:
                out[name] = "over"
            else:
                out[name] = "ok"
        return out

    @property
    def ok(self) -> bool:
        return all(v == "ok" for v in self.calibrations.values())

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level, "n": self.n, "tolerance": self.tolerance,
                "parameters": {name: {"coverage": self.rate(name), "ci": list(self.ci(name)),
                                      "median_width": self.width[name], "calibration": self.calibrations[name]}
                               for name in self.covered},
                "ok": self.ok}

    def to_markdown(self) -> str:
        rows = ["| parameter | coverage | 95 % CI of the coverage | median width | reading |", "|---|---|---|---|---|"]
        for name in self.covered:
            low, high = self.ci(name)
            width = f"{self.width[name]:.4g}" if np.isfinite(self.width[name]) else "unbounded"
            rows.append(f"| `{name}` | {self.rate(name):.1%} | [{low:.0%}, {high:.0%}] | {width} | "
                        f"{self.calibrations[name]} |")
        return (f"**coverage of the {self.level:.0%} interval over {self.n} synthetic experiments**\n\n"
                + "\n".join(rows))

    def __str__(self) -> str:
        parts = [f"{name} {self.rate(name):.0%}" for name in self.covered]
        return f"coverage of the {self.level:.0%} interval over {self.n} runs: " + ", ".join(parts)


def coverage_study(case: Callable[[np.random.Generator], tuple[Any, ParameterSet, Observations]], *,
                   n: int = 200, level: float = 0.95, seed: int = 0, tolerance: float = 0.1,
                   **recover_kwargs: Any) -> CoverageReport:
    """Repeat a synthetic experiment with a known truth and count how often the interval contains it.

    ``case(rng)`` builds one experiment: it returns the model, the parameters (with ``truth`` set) and the
    observations, whose values are already drawn at the truth with their declared noise. A case that draws its noise
    from ``rng`` is what makes the repeats independent.
    """
    rng = np.random.default_rng(seed)
    covered: dict[str, int] = {}
    widths: dict[str, list[float]] = {}
    for _ in range(int(n)):
        model, params, obs = case(rng)
        truth = params.truth()
        if not truth:
            raise ValueError("the case has to set Parameter.truth on every free parameter: a coverage study needs "
                             "the answer it is checking against")
        result = recover(model, params, obs, level=level, **recover_kwargs)
        for interval in result.intervals:
            covered[interval.name] = covered.get(interval.name, 0) + int(interval.contains(truth[interval.name]))
            widths.setdefault(interval.name, []).append(interval.width)
    return CoverageReport(level=level, n=int(n), covered=covered,
                          width={name: float(np.median(w)) for name, w in widths.items()}, tolerance=tolerance)
